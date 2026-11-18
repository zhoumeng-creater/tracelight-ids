"""Assemble all opened AIT normal-download events with actual packet/Zeek candidates.

No model calls or threshold work. The old packet parser supplies observed packet
envelopes, not Zeek features; pinned Zeek exports supply the separate raw8 fields.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
THESIS = WORK.parent
DEFAULT_REPO = THESIS / "代码/tracelight-ids/.worktrees/ait-flow-contract-20261118"
MANIFEST = WORK / "独立来源与算法主线_20261118/normal_event_manifest.json"
OLD_CATALOG = THESIS / "实验归档/20261116_第二十七轮独立事件取证/回传/packet/tcp-connection-candidates.csv"
PROXY, CLOUD = "172.19.130.68", "172.19.130.106"
MODES = ("strict", "ignore_checksums")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cardinality(items):
    return "none" if not items else "unique" if len(items) == 1 else "multiple"


def endpoint_key(a, ap, b, bp):
    return tuple(sorted(((a, int(ap)), (b, int(bp)))))


def packet_key(row):
    return endpoint_key(row["origin_ip"], row["origin_port"], row["responder_ip"], row["responder_port"])


def zeek_key(row):
    return endpoint_key(row["id.orig_h"], row["id.orig_p"], row["id.resp_h"], row["id.resp_p"])


def backend_pair(row):
    return (row["origin_ip"] == PROXY and row["responder_ip"] == CLOUD and int(row["responder_port"]) == 443) or (row["responder_ip"] == PROXY and row["origin_ip"] == CLOUD and int(row["origin_port"]) == 443)


def audit_old(manifest):
    rows = csv_read(OLD_CATALOG)
    proxies = [r for r in rows if PROXY in (r["origin_ip"], r["responder_ip"])]
    pairs = [r for r in rows if backend_pair(r)]
    result = {
        "catalog": str(OLD_CATALOG), "catalog_sha256": sha(OLD_CATALOG),
        "source_summary": str(OLD_CATALOG.parent / "packet-evidence-summary.json"),
        "source_summary_sha256": sha(OLD_CATALOG.parent / "packet-evidence-summary.json"),
        "connection_candidates": len(rows), "any_proxy_endpoint": len(proxies),
        "proxy_cloud_connections": len(pairs),
        "events": [{"event_id": e["event_id"], "backend_candidate_connection_ids": [r["connection_id"] for r in pairs if float(r["first"]) < e["backend"]["timestamp"] + 1 and float(r["last"]) >= e["backend"]["timestamp"]]} for e in manifest["events"]],
        "conclusion": "The existing IPv4 TCP catalog contains no proxy endpoint and cannot associate these proxy-to-cloud downloads. This is a catalog coverage result, not a claim about unknown mirror configurations or unsupported packets.",
    }
    save(ROOT / "intranet_coverage.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    manifest = load(MANIFEST)
    confirmed_ids = {e["event_id"] for e in manifest["events"] if e["business_label"] == "normal_file_download"}
    assert len(confirmed_ids) == 31 and len(manifest["events"]) == 36
    old_audit = audit_old(manifest)
    if args.catalog_only:
        print(json.dumps({"catalog_candidates": old_audit["connection_candidates"], "proxy_endpoint_candidates": old_audit["any_proxy_endpoint"]}))
        return

    code = args.code_repo.resolve()
    sys.path.insert(0, str(code / "scripts/ait_contract"))
    import zeek_runtime
    import analyze
    spec = importlib.util.spec_from_file_location("ait_source_packet_reader", code / "scripts/prepare_ait_packet_evidence.py")
    packet_reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packet_reader)
    packet_reader.SERVER = CLOUD
    receipt = load(ROOT / "source/cloud-source-receipt.json")
    pcap = Path(receipt["pcap"])
    assert pcap.stat().st_size == receipt["bytes"] and sha(pcap) == receipt["sha256"]
    runtime = load(ROOT / "runtime-receipt.json")
    assert runtime["version"] == "8.0.11"

    def export(mode):
        destination = ROOT / "zeek" / mode
        previous = destination / "export-receipt.json"
        if previous.exists():
            saved = load(previous)
            assert saved["pcap_sha256"] == receipt["sha256"] and saved["mode"] == mode
            for name, info in saved["logs"].items():
                if info["exists"]:
                    assert sha(destination / f"{name}.log") == info["sha256"]
            return saved
        return zeek_runtime.export_pcap(pcap, destination, mode, runtime)

    with ThreadPoolExecutor(max_workers=2) as pool:
        exports = dict(zip(MODES, pool.map(export, MODES)))
    packet_connections, packet_counts = packet_reader.parse_connections(pcap, start=float("-inf"), stop=float("inf"))
    csv_write(ROOT / "cloud_packet_connections.csv", packet_connections)
    save(ROOT / "cloud_packet_counts.json", packet_counts)
    packet_by_key = defaultdict(list)
    for p in packet_connections:
        packet_by_key[packet_key(p)].append(p)
    # Access-log seconds give a narrower observable anchor than a whole TCP lifetime.
    # Keep envelope candidates even when no payload was captured in that second.
    request_seconds = {int(e["backend"]["timestamp"]) for e in manifest["events"]}
    packet_activity = defaultdict(Counter)
    for timestamp, linktype, payload, _wire in packet_reader.pcap_records(pcap):
        second = math.floor(timestamp)
        if second not in request_seconds:
            continue
        parsed = packet_reader.ipv4_tcp(payload, linktype, Counter())
        if parsed is None:
            continue
        source, target, _seq, _flags, captured_payload_bytes = parsed
        key = tuple(sorted((source, target)))
        for p in packet_by_key[key]:
            if p["first"] <= timestamp <= p["last"]:
                counts = packet_activity[(p["connection_id"], second)]
                counts["packets"] += 1
                if captured_payload_bytes > 0:
                    direction = "proxy_to_cloud" if source[0] == PROXY and target[0] == CLOUD else "cloud_to_proxy" if source[0] == CLOUD and target[0] == PROXY else "other"
                    counts[direction + "_payload_packets"] += 1
                    counts[direction + "_captured_payload_bytes"] += captured_payload_bytes

    all_flows, mode_flows = [], {}
    for mode in MODES:
        conn_rows = json_rows(ROOT / "zeek" / mode / "conn.log")
        ssl = defaultdict(list)
        http = defaultdict(list)
        for row in json_rows(ROOT / "zeek" / mode / "ssl.log"):
            ssl[row["uid"]].append({k: row.get(k) for k in ["ts", "server_name", "version", "cipher", "established", "resumed"]})
        for row in json_rows(ROOT / "zeek" / mode / "http.log"):
            http[row["uid"]].append({k: row.get(k) for k in ["ts", "method", "host", "uri", "status_code", "trans_depth", "response_body_len", "request_body_len"]})
        current = []
        for row in conn_rows:
            normalized = analyze.normalize_connection(row, (1642723200, 1643068800))
            identifier = f'ait-cloud:{mode}:{row["uid"]}'
            packet_candidates = []
            if row["proto"] == "tcp":
                for p in packet_by_key[zeek_key(row)]:
                    z_end = normalized["zeek_reported_end"]
                    overlaps = (p["first"] <= row["ts"] <= p["last"]) if z_end is None else (row["ts"] <= p["last"] and z_end >= p["first"])
                    if overlaps:
                        packet_candidates.append(p["connection_id"])
            flow = {"flow_id": identifier, "capture_member": receipt["member"], "capture_sha256": receipt["sha256"], "variant": mode, **normalized, "raw_packet_candidate_ids": packet_candidates, "ssl": ssl[row["uid"]], "http": http[row["uid"]]}
            current.append(flow)
            all_flows.append(flow)
        mode_flows[mode] = current
    with (ROOT / "connections.jsonl").open("w", encoding="utf-8") as f:
        for flow in all_flows:
            f.write(json.dumps(flow, ensure_ascii=False, allow_nan=False) + "\n")
    flow_by_packet = defaultdict(list)
    for flow in all_flows:
        for raw_id in flow["raw_packet_candidate_ids"]:
            flow_by_packet[raw_id].append(flow)

    events, links = [], []
    packet_to_events = defaultdict(set)
    for event in manifest["events"]:
        ts = event["backend"]["timestamp"]
        action = event["action_candidates"][0] if len(event["action_candidates"]) == 1 else None
        start, end = (math.floor(action["start"]), action["success"]) if action else (ts, ts + 1)
        candidates, request_ids, active_request_ids = [], [], []
        linked_modes = {mode: set() for mode in MODES}
        active_linked_modes = {mode: set() for mode in MODES}
        for packet in packet_connections:
            if not backend_pair(packet):
                continue
            request_overlap = packet["first"] < ts + 1 and packet["last"] >= ts
            action_overlap = packet["first"] <= end and packet["last"] >= start
            if not (request_overlap or action_overlap):
                continue
            raw_id = packet["connection_id"]
            if request_overlap:
                request_ids.append(raw_id)
            activity = dict(packet_activity[(raw_id, int(ts))])
            if activity.get("proxy_to_cloud_payload_packets", 0) > 0:
                active_request_ids.append(raw_id)
            packet_to_events[raw_id].add(event["event_id"])
            flow_links = []
            for flow in flow_by_packet[raw_id]:
                z_start, z_end = flow["connection_start"], flow["zeek_reported_end"]
                zeek_overlap = (start <= z_start <= end) if z_end is None else (z_start <= end and z_end >= start)
                zeek_request_overlap = (ts <= z_start < ts + 1) if z_end is None else (z_start < ts + 1 and z_end >= ts)
                active_request_flow = activity.get("proxy_to_cloud_payload_packets", 0) > 0 and zeek_request_overlap
                if zeek_overlap:
                    linked_modes[flow["variant"]].add(flow["flow_id"])
                if active_request_flow:
                    active_linked_modes[flow["variant"]].add(flow["flow_id"])
                flow_links.append({"flow_id": flow["flow_id"], "variant": flow["variant"], "zeek_interval_overlaps_event": zeek_overlap, "zeek_interval_overlaps_request_second": zeek_request_overlap, "packet_payload_active_in_request_second": active_request_flow, "zeek_start": z_start, "zeek_reported_end": z_end, "replay_export_time": flow["replay_export_time"], "after_input_eof": flow["after_input_eof"], "during_shutdown": flow["during_shutdown"]})
                links.append({"event_id": event["event_id"], "business_label": event["business_label"], "leg": "proxy_to_cloud", "packet_connection_id": raw_id, "request_second_overlap": request_overlap, "action_window_overlap": action_overlap, "zeek_interval_overlaps_event": zeek_overlap, "zeek_interval_overlaps_request_second": zeek_request_overlap, "packet_payload_active_in_request_second": active_request_flow, "variant": flow["variant"], "flow_id": flow["flow_id"]})
            candidates.append({"packet_connection_id": raw_id, "request_second_overlap": request_overlap, "action_window_overlap": action_overlap, "activity_in_request_second": activity, "packet_envelope": packet, "zeek_links": flow_links})
        events.append({
            "event_id": event["event_id"], "business_label": event["business_label"], "source_event": event,
            "request_second": [ts, ts + 1], "action_window_used": [start, end],
            "packet_candidate_class": cardinality(candidates), "request_second_candidate_class": cardinality(request_ids),
            "request_second_packet_ids": request_ids, "packet_connection_candidates": candidates,
            "request_second_proxy_payload_candidate_class": cardinality(active_request_ids),
            "request_second_proxy_payload_packet_ids": active_request_ids,
            "zeek_candidate_class_by_mode": {mode: cardinality(ids) for mode, ids in linked_modes.items()},
            "zeek_candidate_ids_by_mode": {mode: sorted(ids) for mode, ids in linked_modes.items()},
            "request_second_payload_zeek_class_by_mode": {mode: cardinality(ids) for mode, ids in active_linked_modes.items()},
            "request_second_payload_zeek_ids_by_mode": {mode: sorted(ids) for mode, ids in active_linked_modes.items()},
            "association_basis": "Observed IPv4 TCP proxy/cloud:443 endpoints and packet-envelope overlap; action/proxy/backend logs establish business provenance; encrypted/request-sharing flows are not pure file-labelled samples.",
            "front_end_leg": {"status": "application_logs_only", "packet_candidates": [], "counts_as_independent_event": False},
            "availability": {"action_start": action["start"] if action else None, "action_success": action["success"] if action else None, "backend_request_second_start": ts, "flow_features_available": "per candidate Zeek replay export clock; not actual ingestion", "actual_detector_received_time": None, "normal_feedback_available_time": None},
        })
    assert len(events) == 36 and sum(e["business_label"] == "normal_file_download" for e in events) == 31
    for event in events:
        for candidate in event["packet_connection_candidates"]:
            candidate["shared_with_event_ids"] = sorted(packet_to_events[candidate["packet_connection_id"]])
    csv_write(ROOT / "event_flow_links.csv", links)
    confirmed = [e for e in events if e["business_label"] == "normal_file_download"]
    summary = {
        "capture": receipt, "source_manifest": str(MANIFEST), "source_manifest_sha256": sha(MANIFEST),
        "intranet_catalog_proxy_connections": old_audit["any_proxy_endpoint"],
        "normal_events": len(confirmed), "retained_unresolved_events": len(events) - len(confirmed),
        "packet_connections_whole_capture": len(packet_connections),
        "normal_packet_candidate_distribution": dict(Counter(e["packet_candidate_class"] for e in confirmed)),
        "normal_request_second_distribution": dict(Counter(e["request_second_candidate_class"] for e in confirmed)),
        "normal_request_second_proxy_payload_distribution": dict(Counter(e["request_second_proxy_payload_candidate_class"] for e in confirmed)),
        "normal_zeek_distribution_by_mode": {mode: dict(Counter(e["zeek_candidate_class_by_mode"][mode] for e in confirmed)) for mode in MODES},
        "normal_request_second_payload_zeek_distribution_by_mode": {mode: dict(Counter(e["request_second_payload_zeek_class_by_mode"][mode] for e in confirmed)) for mode in MODES},
        "unique_packet_candidates_linked_to_normal_events": len({p["packet_connection_id"] for e in confirmed for p in e["packet_connection_candidates"]}),
        "packet_candidates_shared_between_events": {str(k): sorted(v) for k, v in packet_to_events.items() if len(v) > 1},
        "zeek_connection_rows": {mode: len(rows) for mode, rows in mode_flows.items()},
        "export_receipts": exports, "new_model_fits": 0, "model_forward_calls": 0,
        "source_code": {str(path): sha(path) for path in [Path(__file__), code / "scripts/prepare_ait_packet_evidence.py", code / "scripts/ait_contract/zeek_runtime.py", code / "scripts/ait_contract/export.zeek", code / "scripts/ait_contract/analyze.py"]},
    }
    save(ROOT / "normal_event_connections.json", {"schema": "ait-normal-event-connections-v1", "summary": summary, "events": events})
    save(ROOT / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k in ["normal_events", "retained_unresolved_events", "packet_connections_whole_capture", "normal_packet_candidate_distribution", "normal_request_second_distribution", "normal_request_second_proxy_payload_distribution", "normal_zeek_distribution_by_mode", "normal_request_second_payload_zeek_distribution_by_mode", "unique_packet_candidates_linked_to_normal_events", "zeek_connection_rows"]}), flush=True)


if __name__ == "__main__":
    main()
