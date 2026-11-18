"""Add original conn observations for normal-event UIDs UNION object-parent UIDs.

No downloads and no normal-label filtering. Existing artifacts are read-only.
python assemble_normal_observations.py
"""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from fetch_normal_sources import BASE, sha256
from assemble_normal_business import bro_rows, utc, write_jsonl_gz

RAW8 = ("ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")


def main():
    event_path = BASE / "normal_flow_events.jsonl.gz"
    object_path = BASE / "normal_http_file_objects.jsonl.gz"
    before = {str(p): sha256(p) for p in (event_path, object_path)}
    manifest = json.loads((BASE / "normal_business_manifest.json").read_text(encoding="utf-8"))
    source = manifest["sources"]["conn"]
    if sha256(source["path"]) != source["sha256"]:
        raise ValueError("Original conn SHA mismatch")
    with gzip.open(event_path, "rt", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]
    with gzip.open(object_path, "rt", encoding="utf-8") as f:
        objects = [json.loads(line) for line in f]
    normal_uids = {e["uid"] for e in events}
    object_uids = {o["uid"] for o in objects}
    object_counts = {}
    for obj in objects:
        object_counts[obj["uid"]] = object_counts.get(obj["uid"], 0) + 1
    wanted = normal_uids | object_uids
    if len(events) != 41630 or len(normal_uids) != 41630 or len(objects) != 5026:
        raise ValueError("Expected the unchanged 41,630 unique normal-event UIDs")
    observations = {}
    stats = {}
    for c in bro_rows(source["path"], stats):
        uid = c["uid"]
        if uid not in wanted:
            continue
        if uid in observations:
            raise ValueError(f"Duplicate source UID: {uid}")
        if c["proto"] != "tcp":
            raise ValueError(f"Unexpected protocol in fixed cohort: {c['proto']}")
        start = float(c["ts"])
        duration = float(c["duration"])
        end = start + duration
        row = {
            "flow_id": "optc:20190919:" + uid,
            "uid": uid,
            "normal_flow_label_eligible": uid in normal_uids,
            "normal_http_file_object_count": object_counts.get(uid, 0),
            "source": c["id.orig_h"], "destination": c["id.resp_h"],
            "native_protocol": c["proto"], "ip_protocol": 6,
            "orig_port": int(c["id.orig_p"]), "resp_port": int(c["id.resp_p"]),
            "duration_seconds": duration,
            "orig_pkts": int(c["orig_pkts"]), "resp_pkts": int(c["resp_pkts"]),
            "orig_ip_bytes": int(c["orig_ip_bytes"]), "resp_ip_bytes": int(c["resp_ip_bytes"]),
            "orig_bytes": int(c["orig_bytes"]), "resp_bytes": int(c["resp_bytes"]),
            "missed_bytes": int(c["missed_bytes"]),
            "connection_start": utc(start),
            "zeek_reported_end": utc(end),
            "available_at_lower_bound": utc(end),
            "actual_available_at": None,
            "source_line": c["_source_line"], "source_path": source["path"],
        }
        if any(row[k] is None for k in RAW8):
            raise ValueError(f"Missing raw8 field: {uid}")
        observations[uid] = row
    if set(observations) != wanted:
        raise ValueError(f"Unmatched requested UIDs: {len(wanted - set(observations))}")
    rows = sorted(observations.values(), key=lambda r: (r["connection_start"], r["flow_id"]))
    if len({r["flow_id"] for r in rows}) != len(wanted):
        raise ValueError("Non-unique flow_id")
    for e in events:
        row = observations[e["uid"]]
        if row["connection_start"] != e["start_utc"] or row["zeek_reported_end"] != e["flow_end_utc"]:
            raise ValueError(f"Timestamp mismatch against existing event: {e['uid']}")
    for obj in objects:
        if obj["uid"] not in observations:
            raise ValueError(f"Object parent connection missing: {obj['event_id']}")
    output = BASE / "normal_flow_observations.jsonl.gz"
    write_jsonl_gz(output, rows)
    after = {path: sha256(path) for path in before}
    if before != after:
        raise ValueError("Existing event artifacts unexpectedly changed")
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "output": {"path": str(output), "rows": len(rows), "bytes": output.stat().st_size, "sha256": sha256(output)},
        "source_conn": source, "source_conn_rows_scanned": stats["data_rows"],
        "raw8_fields": list(RAW8), "raw8_missing_values": {key: sum(r[key] is None for r in rows) for key in RAW8},
        "unique_flow_ids": len({r["flow_id"] for r in rows}),
        "unique_uids": len(observations), "unmatched_event_uids": 0,
        "normal_flow_uid_count": len(normal_uids),
        "http_file_object_count": len(objects), "http_object_parent_unique_uid_count": len(object_uids),
        "normal_flow_object_parent_uid_intersection_count": len(normal_uids & object_uids),
        "normal_flow_only_uid_count": len(normal_uids - object_uids),
        "object_parent_only_uid_count": len(object_uids - normal_uids),
        "union_uid_count": len(wanted), "unmatched_object_uids": 0,
        "normal_flow_label_eligible_true": sum(r["normal_flow_label_eligible"] for r in rows),
        "normal_flow_label_eligible_false": sum(not r["normal_flow_label_eligible"] for r in rows),
        "existing_artifact_sha256_before": before, "existing_artifact_sha256_after": after,
        "time_semantics": "zeek_reported_end = conn.ts + conn.duration; available_at_lower_bound equals this end, actual log emission/ingestion time remains null.",
        "label_semantics": "Exact UID union of existing normal-flow events and existing HTTP objects. A parent connection is normal_flow_label_eligible only if already in the normal-flow table; an object does not promote its whole connection to normal.",
    }
    (BASE / "normal_flow_observations_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["output"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
