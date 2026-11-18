"""Export all selected-host observations from two already-downloaded conn rolls.

No label filtering, downloads, model calls or prefix-context computation.
python export_context_inputs.py
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path

from fetch_normal_sources import BASE, sha256
from assemble_normal_business import bro_rows, utc

NORMAL_GROUP = "optc:20190919:declared_normal_business_subset"
EMPIRE_GROUP = "optc:empire_custom:20190924_continues_20190925"
RAW8 = ("ip_protocol", "orig_port", "resp_port", "duration", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")


def number(value, floating=False):
    if value in (None, "-", "(empty)", ""):
        return None
    parsed = float(value) if floating else int(value)
    return parsed if math.isfinite(parsed) else None


def main():
    normal_event_path = BASE / "normal_flow_events.jsonl.gz"
    normal_obs_path = BASE / "normal_flow_observations.jsonl.gz"
    with gzip.open(normal_event_path, "rt", encoding="utf-8") as f:
        normal_by_uid = {row["uid"]: row for row in map(json.loads, f)}
    host_addresses = defaultdict(set)
    with gzip.open(normal_obs_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["uid"] in normal_by_uid:
                # Existing normal eCAR joins prove the monitored browser host is
                # the original source endpoint. This selects hosts, not labels
                # of the background records exported below.
                host_addresses[normal_by_uid[row["uid"]]["hostname"]].add(row["source"])
    normal_addresses = set().union(*host_addresses.values())
    if len(normal_by_uid) != 41630 or len(host_addresses) != 17:
        raise ValueError("Declared normal-host registry changed")
    normal_receipt = json.loads((BASE / "source_receipt.json").read_text(encoding="utf-8"))
    previous_dir = BASE.parents[1] / "独立来源与算法主线_20261118" / "evidence" / "optc"
    empire_receipt = json.loads((previous_dir / "optc_event_slice.json").read_text(encoding="utf-8"))
    selections = [
        {"date": "20190919", "group": NORMAL_GROUP, "hosts": normal_addresses, "source": normal_receipt["sources"]["conn"]},
        {"date": "20190924", "group": EMPIRE_GROUP, "hosts": {"142.20.57.246"}, "source": empire_receipt["source_manifest"]["conn"]},
    ]
    for selection in selections:
        source = selection["source"]
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Conn source SHA mismatch: {source['path']}")
    output = BASE / "context_inputs.jsonl.gz"
    temp = output.with_suffix(output.suffix + ".partial")
    group_receipts = []
    all_view_ids = set()
    all_physical_ids = set()
    with temp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for selection in selections:
                counts = Counter(); host_counts = Counter(); protocol_counts = Counter()
                missing = Counter({key: 0 for key in RAW8})
                source_stats = {}; physical_ids = set()
                starts, ends = [], []
                for c in bro_rows(selection["source"]["path"], source_stats):
                    origin, response = c["id.orig_h"], c["id.resp_h"]
                    selected_hosts = sorted({origin, response} & selection["hosts"])
                    if not selected_hosts:
                        continue
                    counts["selected_physical_records"] += 1
                    if len(selected_hosts) == 2:
                        counts["both_endpoints_selected_records"] += 1
                    if origin == response:
                        counts["self_endpoint_records"] += 1
                    physical_id = f"optc:{selection['date']}:{c['uid']}"
                    if physical_id in physical_ids:
                        raise ValueError(f"Duplicate source UID within selected physical records: {physical_id}")
                    physical_ids.add(physical_id); all_physical_ids.add(physical_id)
                    native_protocol = c["proto"]
                    protocol = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}.get(native_protocol)
                    duration = number(c["duration"], floating=True)
                    start = float(c["ts"])
                    end = None if duration is None else start + duration
                    starts.append(start)
                    if end is not None:
                        ends.append(end)
                    vector = [protocol, number(c["id.orig_p"]), number(c["id.resp_p"]), duration,
                              number(c["orig_pkts"]), number(c["resp_pkts"]),
                              number(c["orig_ip_bytes"]), number(c["resp_ip_bytes"])]
                    for host in selected_hosts:
                        flow_id = physical_id + ":" + host
                        if flow_id in all_view_ids:
                            raise ValueError(f"Duplicate host-view ID: {flow_id}")
                        all_view_ids.add(flow_id)
                        peer = response if host == origin else origin
                        row = {
                            "flow_id": flow_id, "view_id": c["uid"] + ":" + host,
                            "physical_flow_id": physical_id, "uid": c["uid"],
                            "group": selection["group"], "host": host, "peer": peer,
                            "available_at": None, "flow_end_at": utc(end) if end is not None else None,
                            "connection_start": utc(start), "raw8": vector,
                            "source": origin, "destination": response, "native_protocol": native_protocol,
                            "host_is_originator": host == origin,
                            "source_line": c["_source_line"], "source_path": selection["source"]["path"],
                        }
                        gz.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                        counts["host_views"] += 1
                        host_counts[host] += 1
                        protocol_counts[native_protocol] += 1
                        if end is None:
                            counts["views_with_missing_flow_end_at"] += 1
                        if any(value is None for value in vector):
                            counts["views_with_any_missing_raw8"] += 1
                        if host == origin:
                            counts["originator_views"] += 1
                        else:
                            counts["responder_views"] += 1
                        for key, value in zip(RAW8, vector):
                            missing[key] += value is None
                group_receipts.append({
                    "group": selection["group"], "date": selection["date"],
                    "declared_host_ips": sorted(selection["hosts"]), "source": selection["source"],
                    "source_scan": source_stats, "counts": dict(counts),
                    "host_view_counts": dict(sorted(host_counts.items())),
                    "protocol_view_counts": dict(protocol_counts), "missing_raw8_view_counts": dict(missing),
                    "unique_physical_flows": len(physical_ids),
                    "selected_connection_start_min": utc(min(starts)) if starts else None,
                    "selected_connection_start_max": utc(max(starts)) if starts else None,
                    "selected_observed_end_min": utc(min(ends)) if ends else None,
                    "selected_observed_end_max": utc(max(ends)) if ends else None,
                })
    temp.replace(output)
    receipt = {
        "schema_version": "optc_prefix_context_inputs_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                   "unique_host_views": len(all_view_ids), "unique_physical_flows": len(all_physical_ids)},
        "raw8_fields_in_order": list(RAW8), "groups": group_receipts,
        "normal_host_selection_provenance": {
            "event_table": str(normal_event_path), "event_table_sha256": sha256(normal_event_path),
            "observation_table": str(normal_obs_path), "observation_table_sha256": sha256(normal_obs_path),
            "hostnames_and_all_observed_addresses": {host: sorted(ips) for host, ips in sorted(host_addresses.items())},
        },
        "declaration": {
            "complete": "All original conn records involving at least one declared host, within each named source rotation file; no label/protocol/duration/quality filtering.",
            "not_complete": "Not all enterprise hosts, not full-day/full-exercise history, and not a guarantee of complete wall-clock-hour packet arrival history. conn files are output rotations.",
            "directions": "One view per selected host. Both-target-host records yield two views sharing physical_flow_id. raw8 retains original conn direction in both views.",
            "available_at": "Always null: actual log emission/receipt time is unavailable. Use only explicitly declared offline_flow_end_replay for non-null flow_end_at; never replace a missing end with connection_start.",
            "missing": "Unavailable raw8 values and missing flow_end_at remain null; receipt reports counts. No zeros substituted and no incomplete observations silently removed.",
            "cold_start": "Context starts empty at each group boundary. Earlier omitted rotations may contain prior flows; this export does not reconstruct them.",
            "late_output": "A conn rotation can report connections that started before its named hour, and later rotations may contain earlier-starting connections. Sorting by observed end is an offline convention, not actual emission order.",
            "labels": "The word normal in the normal group's name describes host selection only; every exported background observation remains unlabeled here. Object and event labels never filter context input.",
            "reserved_branch": "No 2019-09-25 Meterpreter source read or included.",
            "execution": "Exporter only; no prefix module, fitting, threshold selection or model scoring executed.",
        },
    }
    (BASE / "context_inputs_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": receipt["output"], "groups": [{"group": r["group"], "counts": r["counts"], "protocols": r["protocol_view_counts"], "missing": r["missing_raw8_view_counts"]} for r in group_receipts]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
