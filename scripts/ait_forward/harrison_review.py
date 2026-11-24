"""Review whole Harrison connections against all potentially associated service requests."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

from .harrison_evidence import (
    ACCESS_MEMBER,
    APACHE_ROW,
    checked_source_paths,
    consume,
    get_clock,
    json_lines,
    write_new_report,
)

RAW8 = {
    "ip_protocol", "orig_port", "resp_port", "duration_seconds",
    "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
}


def read_linked_inputs(projection_path, source_path, manifest_path):
    projection_bytes = projection_path.read_bytes()
    source_bytes = source_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    projection, source, manifest = map(json.loads, (projection_bytes, source_bytes, manifest_bytes))
    if (projection["schema"] != "ait-harrison-observation-candidates-v1"
            or source["schema"] != "ait-harrison-source-qualification-v1"
            or hashlib.sha256(source_bytes).hexdigest() != projection["source_qualification_sha256"]
            or hashlib.sha256(manifest_bytes).hexdigest() != source["source_manifest"]["sha256"]):
        raise ValueError("Projection and linked source manifest do not agree")
    return projection, checked_source_paths(manifest), hashlib.sha256(projection_bytes).hexdigest()


def projection_refs(projection):
    flow_anchors = collections.defaultdict(list)
    reviewed_requests = collections.defaultdict(list)
    for event in projection["events"]:
        control = event["kind"] == "author_labeled_control_request"
        for request in event["request_anchors"]:
            ids = [ref["flow_id"] for ref in request["candidate_flow_ids"]]
            if len(ids) != len(set(ids)):
                raise ValueError("Repeated flow in a source request candidate set")
            source_key = (request["request_member"], request["request_line"])
            for flow in ids:
                evidence = {
                    "event_id": event["event_id"], "control_annotation": control,
                    "request_member": source_key[0], "request_line": source_key[1],
                    "http_response_status": request["http_response_status"],
                    "unique_flow": len(ids) == 1 and not request["unbounded_candidate_flow_ids"],
                    "flow_id": flow,
                }
                flow_anchors[flow].append(evidence)
                reviewed_requests[source_key].append(evidence)
    return flow_anchors, reviewed_requests


def load_candidate_observations(inputs, flow_anchors):
    observations = {}
    by_endpoint = collections.defaultdict(list)
    for record in inputs:
        path = Path(record["path"])
        with path.open("rb") as stream:
            _, _, sha256 = consume(stream)
        if record["sha256"] != sha256:
            raise ValueError("Projection's packet observation export changed")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                flow = row["flow_id"]
                if flow not in flow_anchors:
                    continue
                if row.get("label") is not None:
                    raise ValueError("Input includes a pre-existing flow label")
                if flow in observations:
                    raise ValueError("Repeated projected flow identity")
                observations[flow] = row
                key = (row["capture_host"], row["source_address"], row["destination_address"])
                by_endpoint[key].append(row)
    if observations.keys() != flow_anchors.keys():
        raise ValueError("Projection references observations not provided")
    return observations, by_endpoint


def actual_service_refs(paths, servers, by_endpoint, reviewed_requests, request_payload_seconds):
    by_flow = collections.defaultdict(list)
    for member, path in sorted(paths.items()):
        if not ACCESS_MEMBER.fullmatch(member):
            continue
        host = member.split("/")[1]
        server_ip = servers[host]["default_ipv4_address"]
        label_member = member.replace("gather/", "labels/", 1)
        author_annotations = collections.defaultdict(list)
        if label_member in paths:
            for label_line, row in json_lines(paths[label_member]):
                if row["labels"]:
                    author_annotations[row["line"]].append(label_line)
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                row = APACHE_ROW.match(line)
                if row is None:
                    raise ValueError("Cannot assert whole-connection accounting with malformed Apache rows")
                same_peer = by_endpoint.get((host, row["peer"], server_ip))
                if not same_peer:
                    continue
                second = datetime.strptime(row["timestamp"], "%d/%b/%Y:%H:%M:%S %z").timestamp()
                refs = reviewed_requests.get((member, lineno), [])
                for flow in same_peer:
                    if flow["connection_start"] >= second + 1 or flow["observed_end"] < second:
                        continue
                    flow_id = flow["flow_id"]
                    if flow_id in request_payload_seconds and second not in request_payload_seconds[flow_id]:
                        continue
                    positive = author_annotations.get(lineno, [])
                    normal = [
                        ref["event_id"] for ref in refs
                        if not ref["control_annotation"] and ref["unique_flow"]
                        and ref["flow_id"] == flow_id
                    ]
                    by_flow[flow_id].append({
                        "request_member": member, "request_line": lineno,
                        "request_unix": second, "author_annotation_lines": positive,
                        "affirmative_normal_event_ids": sorted(set(normal)),
                    })
    return by_flow


def complete_observation_reasons(row, start, end):
    reasons = []
    raw8 = row["raw8"]
    if (raw8.keys() != RAW8 or row.get("raw8_complete") is not True
            or any(not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
                   for x in raw8.values())):
        reasons.append("raw8_missing_or_nonfinite")
    if not (start <= row["connection_start"] <= row["observed_end"] < end):
        reasons.append("connection_not_wholly_in_simulation")
    export = row.get("replay_export_time")
    if not (isinstance(export, (float, int)) and math.isfinite(export)
            and row["observed_end"] <= export < end):
        reasons.append("missing_or_out_of_order_replay_export_clock")
    if row.get("after_input_eof") is not False or row.get("during_shutdown") is not False:
        reasons.append("eof_shutdown_or_uninstrumented")
    if row.get("conn_state") != "SF":
        reasons.append("connection_not_sf")
    if row.get("flow_key_unique") is not True:
        reasons.append("nonunique_physical_flow_identity")
    return reasons


def evaluate_connection(row, anchors, requests, start, end):
    reasons = complete_observation_reasons(row, start, end)
    control = [a for a in anchors if a["control_annotation"]]
    normal = [a for a in anchors if not a["control_annotation"]]
    if not requests:
        reasons.append("no_full_connection_service_requests")
    if control and normal:
        reasons.append("conflicting_control_and_normal_source_anchors")
    if control:
        if not any(a["unique_flow"] and 200 <= a["http_response_status"] < 300 for a in control):
            reasons.append("no_unique_successful_control_request_binding")
        if any(not request["author_annotation_lines"] for request in requests):
            reasons.append("unverified_or_mixed_full_connection_service_requests")
        candidate_kind = "control"
    else:
        if not any(a["unique_flow"] for a in normal):
            reasons.append("no_unique_normal_request_binding")
        if any(request["author_annotation_lines"] for request in requests):
            reasons.append("author_labeled_request_in_normal_candidate_flow")
        if any(not request["affirmative_normal_event_ids"] for request in requests):
            reasons.append("full_connection_has_some_requests_without_positive_normal_evidence")
        candidate_kind = "normal"
    return {
        "flow_id": row["flow_id"], "stable_flow_id": row["stable_flow_id"],
        "candidate_kind": candidate_kind,
        "review_status": "unresolved" if reasons else f"source_bound_{candidate_kind}_candidate",
        "role": row["capture_host"], "exclusion_reasons": reasons,
        "source_events": sorted({a["event_id"] for a in anchors}),
        "source_requests": requests,
    }


def review(projection_path, source_path, manifest_path):
    projection, paths, projection_sha256 = read_linked_inputs(
        projection_path, source_path, manifest_path
    )
    flow_anchors, reviewed_requests = projection_refs(projection)
    observations, by_endpoint = load_candidate_observations(projection["inputs"], flow_anchors)
    roster = [member for member in paths if member.endswith("/logs/sm.log")]
    servers, start, end = get_clock(paths, roster)
    payload_seconds = {
        flow: set(seconds) for receipt in projection["packet_request_payload_refinements"].values()
        for flow, seconds in receipt["flow_captured_request_payload_seconds"].items()
        if flow in observations
    }
    requests = actual_service_refs(paths, servers, by_endpoint, reviewed_requests, payload_seconds)
    rows = [
        evaluate_connection(row, flow_anchors[flow], requests.get(flow, []), start, end)
        for flow, row in sorted(observations.items())
    ]
    summary = collections.Counter(row["review_status"] for row in rows)
    reason_counts = collections.Counter(reason for row in rows for reason in row["exclusion_reasons"])
    return {
        "schema": "ait-harrison-whole-flow-source-review-v1", "admission_status": "not_admitted",
        "projection_sha256": projection_sha256, "counts": dict(summary),
        "reason_counts": dict(reason_counts), "flow_reviews": rows,
        "scope": "Source-bound candidate only; every associated Apache request must have "
                 "affirmative actor evidence for a normal candidate, or an author annotation "
                 "for a control candidate. Request-second associations refined by verified TCP "
                 "headers are conditional. No reviewed label or score is emitted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--source-qualification", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    result = review(args.projection, args.source_qualification, args.manifest)
    write_new_report(args.output, result)
    print(f"Reviewed {len(result['flow_reviews'])} candidate identities without scoring")


if __name__ == "__main__":
    main()
