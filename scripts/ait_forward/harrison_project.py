"""Project reviewed Harrison HTTP request anchors onto completed-flow candidates."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from bisect import bisect_left
from pathlib import Path

from scripts.packet_evidence.extract_headers import iter_tcp_headers

from .harrison_evidence import consume, write_new_report

HOSTS = {"intranet_server", "cloud_share"}
ARTIFACTS = ("control_request_annotations", "normal_action_candidates")


def read_verified_artifact(review, artifact_name):
    artifact = review["artifacts"][artifact_name]
    path = Path(artifact["path"])
    with path.open("rb") as stream:
        _, _, digest = consume(stream)
    if digest != artifact["sha256"]:
        raise ValueError(f"Source qualification artifact changed: {artifact_name}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != artifact["rows"]:
        raise ValueError(f"Source qualification artifact row count changed: {artifact_name}")
    return rows


def build_request_index(review):
    anchors = {}
    by_endpoint = collections.defaultdict(list)
    for artifact in ARTIFACTS:
        for row in read_verified_artifact(review, artifact):
            event_id = row["event_id"]
            if event_id in anchors or row["service"] not in HOSTS:
                raise ValueError("Duplicate source anchor or unexpected source service")
            anchor = {
                "event_id": event_id, "kind": row["kind"], "service": row["service"],
                "request_anchors": [],
            }
            anchors[event_id] = anchor
            requests = row["service_candidates"] if artifact == "normal_action_candidates" else [row]
            for request in requests:
                timestamp = request["request_unix"]
                if not math.isfinite(timestamp) or int(timestamp) != timestamp:
                    raise ValueError("Reviewed Apache request lacks one-second clock precision")
                candidate = {
                    "request_member": request["request_member"],
                    "request_line": request["request_line"],
                    "request_unix": timestamp,
                    "http_response_status": (
                        request["http_response_status"] if artifact == "control_request_annotations" else 200
                    ),
                    "candidate_flow_ids": [],
                    "unbounded_candidate_flow_ids": [],
                }
                key = (row["service"], request["peer_ip"], request["server_ip"])
                by_endpoint[key].append(candidate)
                anchor["request_anchors"].append(candidate)
    for candidates in by_endpoint.values():
        candidates.sort(key=lambda item: item["request_unix"])
    return anchors, {
        endpoint: ([r["request_unix"] for r in candidates], candidates)
        for endpoint, candidates in by_endpoint.items()
    }


def add_observation(row, requests):
    host = row["capture_host"]
    if row["scenario"] != "harrison" or host not in HOSTS:
        raise ValueError("Input is not a designated Harrison observation")
    raw8 = row["raw8"]
    if raw8["ip_protocol"] != 6 or raw8["resp_port"] != 443:
        return
    key = (host, row["source_address"], row["destination_address"])
    if key not in requests:
        return
    start, end = row["connection_start"], row["observed_end"]
    times, candidates = requests[key]
    identity = {"flow_id": row["flow_id"], "stable_flow_id": row["stable_flow_id"]}
    if not math.isfinite(start):
        raise ValueError("Harrison observation lacks a finite connection start")
    if end is None or not math.isfinite(end) or end < start:
        first = bisect_left(times, math.floor(start))
        for request in candidates[first:]:
            request["unbounded_candidate_flow_ids"].append(identity)
        return False
    # Apache reports start to one second. Preserve every matching flow observed during that second.
    first = bisect_left(times, math.floor(start))
    stop = bisect_left(times, math.floor(end) + 1)
    for request in candidates[first:stop]:
        request["candidate_flow_ids"].append(identity)
    if first < stop:
        return True


def refine_request_payload(capture_path, host, anchors, observations, source_sha256):
    with capture_path.open("rb") as stream:
        size, _, capture_sha256 = consume(stream)
    if source_sha256 != {capture_sha256}:
        raise ValueError(f"Packet headers do not come from the original {host} export capture")
    checks = collections.defaultdict(list)
    requests = [
        request for anchor in anchors.values() if anchor["service"] == host
        for request in anchor["request_anchors"]
    ]
    used_ids = {
        item["flow_id"] for request in requests
        for key in ("candidate_flow_ids", "unbounded_candidate_flow_ids")
        for item in request[key]
    }
    for flow_id in used_ids:
        row = observations[flow_id]
        key = (row["source_address"], row["raw8"]["orig_port"],
               row["destination_address"], row["raw8"]["resp_port"])
        checks[key].append((row["connection_start"], row["observed_end"], flow_id))
    packet_counts = collections.Counter()
    active = collections.defaultdict(set)
    for header in iter_tcp_headers(capture_path, packet_counts):
        if header["captured_tcp_payload_bytes"] <= 0:
            continue
        key = (header["source"], header["source_port"], header["destination"], header["destination_port"])
        timestamp = header["timestamp_ns"] / 1_000_000_000
        for start, end, flow in checks.get(key, ()):
            if start <= timestamp and (end is None or timestamp <= end):
                active[flow].add(header["timestamp_ns"] // 1_000_000_000)
    for request in requests:
        request["envelope_candidate_flow_ids"] = request["candidate_flow_ids"]
        request["unbounded_envelope_flow_ids"] = request["unbounded_candidate_flow_ids"]
        request["candidate_flow_ids"] = [
            item for item in request["envelope_candidate_flow_ids"]
            if request["request_unix"] in active.get(item["flow_id"], ())
        ]
        request["unbounded_candidate_flow_ids"] = [
            item for item in request["unbounded_envelope_flow_ids"]
            if request["request_unix"] in active.get(item["flow_id"], ())
        ]
    return {
        "path": str(capture_path), "bytes": size, "sha256": capture_sha256,
        "counts": dict(packet_counts),
        "flow_captured_request_payload_seconds": {
            flow: sorted(active.get(flow, ())) for flow in sorted(used_ids)
        },
        "method": "Client-to-service, captured TCP payload length >0, exact request second and directed "
                  "four-tuple within connection time. No payload bytes retained.",
    }


def project(review_path, flow_paths, captures=None):
    raw = review_path.read_bytes()
    review = json.loads(raw)
    if review["schema"] != "ait-harrison-source-qualification-v1":
        raise ValueError("Unexpected source qualification schema")
    anchors, requests = build_request_index(review)
    inputs = []
    flow_ids = set()
    capture_hosts = set()
    captures = captures or {}
    source_sha256 = {host: set() for host in captures}
    observations = {host: {} for host in captures}
    endpoint_rows_without_observed_interval = 0
    for path in flow_paths:
        with path.open("rb") as stream:
            _, _, sha256 = consume(stream)
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                rows += 1
                if row["flow_id"] in flow_ids:
                    raise ValueError("Repeated network flow identity")
                flow_ids.add(row["flow_id"])
                capture_hosts.add(row["capture_host"])
                result = add_observation(row, requests)
                endpoint_rows_without_observed_interval += result is False
                host = row["capture_host"]
                if host in captures:
                    source_sha256[host].add(row["capture_sha256_original"])
                    if result is not None:
                        observations[host][row["flow_id"]] = row
        inputs.append({"path": str(path), "sha256": sha256, "rows": rows})
    header_filters = {
        host: refine_request_payload(path, host, anchors, observations[host], source_sha256[host])
        for host, path in captures.items()
    }
    summary = collections.Counter()
    for anchor in anchors.values():
        candidates = set()
        for request in anchor["request_anchors"]:
            request["candidate_flow_ids"].sort(key=lambda row: row["flow_id"])
            request["unbounded_candidate_flow_ids"].sort(key=lambda row: row["flow_id"])
            candidates.update(row["flow_id"] for row in request["candidate_flow_ids"])
        if anchor["service"] not in capture_hosts:
            status = "capture_not_supplied"
        elif any(request["unbounded_candidate_flow_ids"] for request in anchor["request_anchors"]):
            status = "observation_set_incomplete"
        elif not candidates:
            status = "no_candidate"
        else:
            status = "unique_candidate" if len(candidates) == 1 else "multiple_candidates"
        anchor["mapping_status"] = status
        summary[anchor["kind"], anchor["mapping_status"]] += 1
    return {
        "schema": "ait-harrison-observation-candidates-v1",
        "admission_status": "not_admitted",
        "scope": "Reviewed HTTP sources only. Candidates require same capture host, directed "
                 "peer and service IP, TCP/443 and overlapping observed request second.",
        "source_qualification_sha256": hashlib.sha256(raw).hexdigest(),
        "inputs": inputs,
        "capture_hosts": sorted(capture_hosts),
        "packet_request_payload_refinements": header_filters,
        "matching_endpoint_rows_without_observed_interval": endpoint_rows_without_observed_interval,
        "counts": [{"kind": kind, "mapping_status": status, "count": count}
                   for (kind, status), count in sorted(summary.items())],
        "events": list(anchors.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-qualification", type=Path, required=True)
    parser.add_argument("--connections", type=Path, nargs="+", required=True)
    parser.add_argument("--cloud-capture", type=Path, help="Original cloud PCAP for packet-header activity only")
    parser.add_argument("--intranet-capture", type=Path,
                        help="Original intranet PCAP for packet-header activity only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    captures = {host: path for host, path in (
        ("cloud_share", args.cloud_capture), ("intranet_server", args.intranet_capture),
    ) if path is not None}
    report = project(args.source_qualification, args.connections, captures)
    write_new_report(args.output, report)
    print(f"Projected {len(report['events'])} source anchors without assigning labels")


if __name__ == "__main__":
    main()
