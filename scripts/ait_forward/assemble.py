"""Bind source-adjudicated Harrison flows into a score-free review manifest."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
from hashlib import file_digest
import json
import math
from pathlib import Path

from scripts.ait_forward import prepare, run


HOST_UNITS = {
    "cloud_share": "ait:harrison:owncloud_file_transfer",
    "intranet_server": "ait:harrison:intranet_page_retrieval",
}
_NORMAL = "source_bound_normal_candidate"
_CONTROL = "source_bound_control_candidate"
_DOWNLOAD = {
    "unique_eligible_flow_with_author_completion_and_observed_http_response",
    "unique_eligible_flow_with_author_completion_and_observed_http_request",
}


def _sha(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(path.read_bytes())


def _qualified_rows(projection, expected_ids, captures, start, end):
    rows = {}
    for input_file in projection["inputs"]:
        path = Path(input_file["path"])
        extraction = path.parent / "extraction_receipt.json"
        metadata = _json(extraction)
        host = metadata["capture_host"]
        digest = _sha(path)
        if (digest != input_file["sha256"] or digest != metadata["outputs"][path.name]["sha256"]
                or metadata["scenario"] != "harrison" or host not in captures
                or captures[host]["sha256"] != metadata["source_sha256"]
                or metadata["zeek_version"] != "8.0.11" or metadata["zeek_mode"] != "ignore_checksums"
                or metadata["summary"]["connections"] != input_file["rows"]):
            raise ValueError(f"Unverified full-capture export: {host}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                identity = row["flow_id"]
                if identity not in expected_ids:
                    continue
                if identity in rows:
                    raise ValueError("Duplicate candidate identity across capture exports")
                observed, export = row["observed_end"], row["replay_export_time"]
                if (observed is None or export is None or not start <= observed <= export < end
                        or row["after_input_eof"] is not False or row["during_shutdown"] is not False
                        or not row["flow_key_unique"]):
                    continue
                raw8 = {key: value if value is not None and math.isfinite(value) and value >= 0 else None
                        for key, value in row["raw8"].items()}
                rows[identity] = {
                    "flow_id": identity, "stable_flow_id": row["stable_flow_id"], "label": None,
                    "label_evidence": None, "target_kind": None, "raw8": raw8,
                    "observed_end": observed, "replay_export_time": export, "actual_available_at": None,
                }
    return rows


def _source_events(projection, download, protocol):
    events = {}
    for event in projection["events"]:
        if event["kind"] != "author_labeled_control_request":
            continue
        all_candidates = sorted({ref["flow_id"] for request in event["request_anchors"]
                                 for ref in request["candidate_flow_ids"]})
        events[event["event_id"]] = ("control", all_candidates, event["mapping_status"])
    for event in download["events"]:
        candidates = sorted({row["flow_id"] for row in event["network_request_records"] if "flow_id" in row})
        events[event["event_id"]] = ("download", candidates, event["status"])
    registered = {row["event_id"]: row["kind"] for row in protocol["required_events"]}
    if {eid: detail[0] for eid, detail in events.items()} != registered:
        raise ValueError("Source audit must account for each independently registered event")
    return events


def assemble(root, projection_path, review_path, download_path, source_path, protocol_path):
    projection, review, download, source, protocol = map(
        _json, (projection_path, review_path, download_path, source_path, protocol_path),
    )
    captures = projection["packet_request_payload_refinements"]
    if (captures.keys() != HOST_UNITS.keys()
            or review["projection_sha256"] != _sha(projection_path)
            or projection["source_qualification_sha256"] != _sha(source_path)
            or protocol["schema"] != run.EVENT_PROTOCOL_SCHEMA):
        raise ValueError("Source review requires the two exact captures and bound source evidence")
    connection_paths = {item["sha256"]: Path(item["path"]) for item in projection["inputs"]}
    if download["connections_sha256"] not in connection_paths:
        raise ValueError("Download join does not share the verified connection export")
    extractor_paths = [path.parent / "extraction_receipt.json" for path in connection_paths.values()]
    intranet_receipts = {_sha(path) for path in extractor_paths if _json(path)["capture_host"] == "intranet_server"}
    if intranet_receipts != {download["extractor_receipt_sha256"]}:
        raise ValueError("The source review and download join use different intranet captures")

    start = source["time"]["simulation_start_unix"]
    end = source["time"]["simulation_end_exclusive_unix"]
    source_events = _source_events(projection, download, protocol)
    flow_reviews = {row["flow_id"]: row for row in review["flow_reviews"]}
    expected_ids = set(flow_reviews) | {fid for _, ids, _ in source_events.values() for fid in ids}
    rows = _qualified_rows(projection, expected_ids, captures, start, end)
    normal_units = {unit: [] for unit in HOST_UNITS.values()}
    for identity, evidence in flow_reviews.items():
        verdict = evidence["review_status"]
        if verdict not in {_NORMAL, _CONTROL}:
            continue
        row = rows[identity]
        if (evidence["stable_flow_id"] != row["stable_flow_id"] or evidence["exclusion_reasons"]
                or None in row["raw8"].values()):
            raise ValueError("Source-approved observation does not match its review")
        reason = f"{review_path.name}#{identity}; events {', '.join(evidence['source_events'])}"
        if verdict == _NORMAL:
            row.update(label=0, label_evidence=reason)
            normal_units[HOST_UNITS[evidence["role"]]].append(identity)
        else:
            row.update(label=1, target_kind="control", label_evidence=reason)

    for eid, (kind, candidates, status) in source_events.items():
        if kind != "download" or status not in _DOWNLOAD:
            continue
        if len(candidates) != 1:
            raise ValueError("Exact download review did not identify a single flow")
        row = rows[candidates[0]]
        if row["label"] is not None or None in row["raw8"].values():
            raise ValueError("Conflicting or incomplete exact download observation")
        row.update(label=1, target_kind=kind, label_evidence=f"{download_path.name}#{eid}; {status}")

    positive = Counter(row["target_kind"] for row in rows.values() if row["label"] == 1)
    targets = {kind: {"status": "qualified" if positive[kind] else "unavailable",
                      "evidence": f"Source-audited {kind} flow count {positive[kind]}; see registered event reviews."}
               for kind in run.TARGETS}
    events = []
    for eid, (kind, candidates, status) in source_events.items():
        if not positive[kind]:
            continue
        modeled = [fid for fid in candidates if fid in rows and (
            rows[fid]["label"] is None or rows[fid]["target_kind"] == kind
        )]
        closed = (len(candidates) == len(modeled) == 1 and rows[modeled[0]]["label"] == 1
                  and status in _DOWNLOAD | {"unique_candidate"})
        events.append({"event_id": eid, "kind": kind, "candidate_flow_ids": modeled,
                       "association_mode": "all_confirmed" if closed else "one_of",
                       "candidate_set_complete": closed,
                       "label_evidence": f"Frozen source event {eid}; source association {status}"})

    def binding(path):
        return {"path": path.resolve().relative_to(root).as_posix(), "sha256": _sha(path)}

    material = root / "reviewed"
    material.mkdir()
    units = [{"unit_id": unit, "flow_ids": sorted(ids)} for unit, ids in normal_units.items() if ids]
    admission = all(normal_units.values()) and bool(positive)
    inputs = {}
    for name, records in (("rows", list(rows.values())), ("normal_units", units), ("events", events)):
        path = material / f"{name}.jsonl.gz"
        prepare._write_rows(path, records)
        inputs[name] = binding(path)
    window = {key: datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")
              for key, value in (("start_utc", start), ("end_utc", end))}
    denominators = {unit: len(ids) for unit, ids in normal_units.items()}
    manifest = {
        "schema": prepare.INPUT_SCHEMA, "source_group": run.GROUP, "clock_basis": run.CLOCK,
        "confirmation_window": window, "targets": targets,
        "normal_evidence": f"Whole-flow affirmative source review; denominators {denominators}",
        "clock_evidence": "Bound source UTC window and complete Zeek 8.0.11 network-time export; excludes EOF and shutdown.",
        "inputs": inputs, "evidence_files": [binding(path) for path in (
            projection_path, review_path, download_path, source_path, protocol_path, *extractor_paths,
        )],
    }
    path = root / "reviewed-inputs.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return prepare.prepare(path, root / ("qualified" if admission else "diagnostic"), diagnostic=not admission)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("root", "projection", "review", "download-join", "source-qualification", "event-protocol"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    print(assemble(args.root.resolve(), args.projection, args.review, args.download_join,
                   args.source_qualification, args.event_protocol))


if __name__ == "__main__":
    main()
