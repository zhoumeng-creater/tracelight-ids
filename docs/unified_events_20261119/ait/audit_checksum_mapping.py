"""Quantify both fixed checksum exports for every request-active packet candidate."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
MODES = ["strict", "ignore_checksums"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    doc = json.loads((ROOT / "normal_event_connections.json").read_text(encoding="utf-8"))
    events = [e for e in doc["events"] if e["business_label"] == "normal_file_download"]
    assert len(events) == 31
    active = defaultdict(set)
    request_flow_ids = defaultdict(lambda: {mode: set() for mode in MODES})
    packets = {}
    for e in events:
        for p in e["packet_connection_candidates"]:
            packets[p["packet_connection_id"]] = p["packet_envelope"]
            for link in p["zeek_links"]:
                if link["packet_payload_active_in_request_second"]:
                    request_flow_ids[p["packet_connection_id"]][link["variant"]].add(link["flow_id"])
        for raw_id in e["request_second_proxy_payload_packet_ids"]:
            active[raw_id].add(e["event_id"])
    with (ROOT / "event_flow_links.csv").open(encoding="utf-8", newline="") as f:
        referenced = {r["flow_id"] for r in csv.DictReader(f)}
    groups = {key: {mode: [] for mode in MODES} for key in active}
    selected_flows = []
    with (ROOT / "connections.jsonl").open(encoding="utf-8") as f:
        for line in f:
            flow = json.loads(line)
            if flow["flow_id"] in referenced:
                selected_flows.append(flow)
            for raw_id in flow["raw_packet_candidate_ids"]:
                if raw_id in groups:
                    groups[raw_id][flow["variant"]].append(flow)
    with (ROOT / "referenced_connections.jsonl").open("w", encoding="utf-8") as f:
        for flow in selected_flows:
            f.write(json.dumps(flow, ensure_ascii=False) + "\n")
    results = []
    for raw_id, arms in sorted(groups.items()):
        vectors = {mode: sorted(tuple(row[k] for k in FIELDS) for row in rows) for mode, rows in arms.items()}
        one_each = all(len(rows) == 1 for rows in arms.values())
        request_arms = {mode: [row for row in rows if row["flow_id"] in request_flow_ids[raw_id][mode]] for mode, rows in arms.items()}
        request_one_each = all(len(rows) == 1 for rows in request_arms.values())
        results.append({
            "packet_connection_id": raw_id, "event_ids_with_request_second_proxy_payload": sorted(active[raw_id]),
            "packet_envelope": packets[raw_id],
            "whole_packet_episode_zeek_record_counts": {mode: len(rows) for mode, rows in arms.items()},
            "raw8_record_multisets_equal": vectors["strict"] == vectors["ignore_checksums"],
            "one_record_in_each_arm": one_each,
            "single_record_field_equality": {field: arms["strict"][0][field] == arms["ignore_checksums"][0][field] for field in FIELDS} if one_each else None,
            "request_second_zeek_record_counts": {mode: len(rows) for mode, rows in request_arms.items()},
            "request_second_one_record_in_each_arm": request_one_each,
            "request_second_field_equality": {field: request_arms["strict"][0][field] == request_arms["ignore_checksums"][0][field] for field in FIELDS} if request_one_each else None,
            "request_second_flow_ids": {mode: [row["flow_id"] for row in rows] for mode, rows in request_arms.items()},
            "arms": {mode: [{"flow_id": row["flow_id"], "raw8": {field: row[field] for field in FIELDS}, "connection_start": row["connection_start"], "zeek_reported_end": row["zeek_reported_end"], "replay_export_time": row["replay_export_time"], "after_input_eof": row["after_input_eof"], "during_shutdown": row["during_shutdown"], "raw_packet_candidate_ids": row["raw_packet_candidate_ids"]} for row in rows] for mode, rows in arms.items()},
        })
    event_counts = [{"event_id": e["event_id"], "active_packet_ids": e["request_second_proxy_payload_packet_ids"], "request_second_zeek_records": {mode: len(e["request_second_payload_zeek_ids_by_mode"][mode]) for mode in MODES}} for e in events]
    all_rows = [row for arms in groups.values() for rows in arms.values() for row in rows]
    request_ids = {fid for arms in request_flow_ids.values() for ids in arms.values() for fid in ids}
    protocol_observation = {}
    for mode in MODES:
        rows = [row for row in selected_flows if row["variant"] == mode and row["flow_id"] in request_ids]
        protocol_observation[mode] = {
            "records": len(rows), "records_with_ssl_log": sum(bool(row["ssl"]) for row in rows),
            "http_records": sum(len(row["http"]) for row in rows),
            "tls_versions": dict(Counter(str(s["version"]) for row in rows for s in row["ssl"])),
            "tls_server_names": dict(Counter(str(s["server_name"]) for row in rows for s in row["ssl"])),
            "established_ssl_records": sum(s["established"] is True for row in rows for s in row["ssl"]),
        }
    summary = {
        "normal_events": 31, "unique_request_active_packet_candidates": len(active),
        "raw8_equal_record_multisets": sum(r["raw8_record_multisets_equal"] for r in results),
        "one_record_in_each_arm": sum(r["one_record_in_each_arm"] for r in results),
        "request_second_one_record_in_each_arm": sum(r["request_second_one_record_in_each_arm"] for r in results),
        "request_second_all_raw8_equal": sum(r["request_second_field_equality"] is not None and all(r["request_second_field_equality"].values()) for r in results),
        "request_second_equal_counts_by_raw8_field": {field: sum(r["request_second_field_equality"] is not None and r["request_second_field_equality"][field] for r in results) for field in FIELDS},
        "record_count_pairs_strict_then_ignore": dict(Counter(f'{r["whole_packet_episode_zeek_record_counts"]["strict"]}:{r["whole_packet_episode_zeek_record_counts"]["ignore_checksums"]}' for r in results)),
        "after_eof_records": sum(row["after_input_eof"] is True for row in all_rows),
        "shutdown_records": sum(row["during_shutdown"] is True for row in all_rows),
        "records_with_multiple_packet_candidates": len({row["flow_id"] for row in all_rows if len(row["raw_packet_candidate_ids"]) > 1}),
        "referenced_zeek_records_all_event_candidates": len(selected_flows),
        "request_second_protocol_observation": protocol_observation,
    }
    output = {
        "summary": summary, "event_request_second_counts": event_counts,
        "comparison_scope": "All actual packet episodes with proxy payload in a confirmed event's request second; whole-episode Zeek record counts are separate from request-second record counts. No arm or candidate selected by features, bytes, or future model scores.",
        "raw8_fields": FIELDS, "packet_candidates": results,
        "inputs": {name: digest(ROOT / name) for name in ["normal_event_connections.json", "connections.jsonl", "event_flow_links.csv"]},
        "script_sha256": digest(Path(__file__)),
    }
    (ROOT / "checksum_mapping_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
