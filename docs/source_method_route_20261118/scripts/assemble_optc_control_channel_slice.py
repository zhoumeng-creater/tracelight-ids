"""Join known Empire channels from already-opened OpTC originals; no downloads.

python assemble_optc_control_channel_slice.py
Uses the PDF agent table and lifecycle, exact eCAR bro_uid/five-tuple, and conn.
This is source qualification, not method evaluation or per-command ground truth.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

from assemble_optc_event_slice import BASE, bro_rows, sha256, utc, value_set


OUT = BASE / "evidence" / "optc"
HOST = "sysclient0501.systemia.com"
SOURCE_IP = "142.20.57.246"
SERVER_IP = "202.6.172.98"
AGENTS = {
    648: {"agent": "4BW2MKUF", "start": "2019-09-24T10:46:02-04:00", "stop": "2019-09-24T11:35:53-04:00", "stop_event": "Killed agent"},
    5076: {"agent": "9HUGDCRL", "start": "2019-09-24T11:26:34-04:00", "stop": "2019-09-24T11:35:53-04:00", "stop_event": "Killed agent"},
    1748: {"agent": "6H8SZPCW", "start": "2019-09-24T11:33:15-04:00", "stop": "2019-09-24T13:11:23-04:00", "stop_event": "Started reverse SSH and lost contact"},
}


def epoch(value):
    return datetime.fromisoformat(value).timestamp()


def exact_tuple(ecar, conn):
    prop = ecar.get("properties", {})
    return (
        ecar.get("hostname", "").casefold() == HOST
        and prop.get("src_ip") == conn.get("id.orig_h") == SOURCE_IP
        and prop.get("src_port") == conn.get("id.orig_p")
        and prop.get("dest_ip") == conn.get("id.resp_h") == SERVER_IP
        and prop.get("dest_port") == conn.get("id.resp_p") == "443"
        and prop.get("l4protocol") == "6" and conn.get("proto") == "tcp"
    )


def main():
    download_event_path = OUT / "optc_event_slice.json"
    unchanged_sha = sha256(download_event_path)
    previous = json.loads(download_event_path.read_text(encoding="utf-8"))
    manifest = previous["source_manifest"]
    for key in ("ecarbro", "conn", "http", "files", "ground_truth"):
        if sha256(Path(manifest[key]["path"])) != manifest[key]["sha256"]:
            raise ValueError(f"Source SHA mismatch: {key}")
    counts = Counter({
        "joined_label_known_control_channel": 0, "joined_label_unconfirmed": 0,
        "joined_label_ambiguous_multiple_agent_matches": 0,
        "conn_endpoint_without_ecar_uid": 0, "conn_endpoint_with_tuple_mismatch": 0,
        "confirmed_pid_648": 0, "confirmed_pid_5076": 0, "confirmed_pid_1748": 0,
    })
    ecar_candidates = []
    ecar_by_uid = defaultdict(list)
    with gzip.open(manifest["ecarbro"]["path"], "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            counts["ecar_all_records"] += 1
            if row.get("hostname", "").casefold() != HOST:
                continue
            counts["ecar_target_hostname"] += 1
            prop = row.get("properties", {})
            if prop.get("dest_ip") != SERVER_IP or prop.get("dest_port") != "443":
                continue
            row["_source_line"] = lineno
            counts["ecar_target_hostname_server_port"] += 1
            counts[f"ecar_server_port_pid_{row.get('pid')}"] += 1
            ecar_candidates.append(row)
            ecar_by_uid[prop.get("bro_uid")].append(row)
    conn_stats = {}
    conn_candidates = []
    for row in bro_rows(Path(manifest["conn"]["path"]), conn_stats):
        if row.get("id.orig_h") == SOURCE_IP and row.get("id.resp_h") == SERVER_IP and row.get("id.resp_p") == "443":
            conn_candidates.append(row)
    counts["conn_all_records"] = conn_stats["data_rows"]
    counts["conn_target_source_server_port"] = len(conn_candidates)
    joins = []
    confirmed_uids = set()
    seen_conn_uids = set()
    for conn in conn_candidates:
        uid = conn["uid"]
        seen_conn_uids.add(uid)
        rows = ecar_by_uid.get(uid, [])
        if not rows:
            counts["conn_endpoint_without_ecar_uid"] += 1
        start = float(conn["ts"])
        end = None if conn["duration"] == "-" else start + float(conn["duration"])
        reasons = []
        matched = []
        for ecar in rows:
            pid = ecar.get("pid")
            agent = AGENTS.get(pid)
            host_ts = epoch(ecar["timestamp"])
            tuple_ok = exact_tuple(ecar, conn)
            # The report gives second precision. Include that displayed final second,
            # while retaining both raw clocks and their difference without correction.
            conn_active = bool(agent and epoch(agent["start"]) <= start < epoch(agent["stop"]) + 1)
            host_active = bool(agent and epoch(agent["start"]) <= host_ts < epoch(agent["stop"]) + 1)
            flags = {
                "exact_host_uid_five_tuple": tuple_ok,
                "pid_has_official_agent_mapping": bool(agent),
                "conn_start_inside_documented_agent_window": conn_active,
                "ecar_time_inside_documented_agent_window": host_active,
                "bidirectional_established_observation": conn.get("conn_state") == "SF" and conn.get("resp_bytes") not in (None, "-", "0"),
            }
            ok = all(flags.values())
            matched.append({
                "ecar_record": ecar, "official_agent": agent, "checks": flags,
                "qualifies_as_known_control_channel": ok,
                "conn_start_minus_ecar_seconds": start - host_ts,
            })
            if not ok:
                reasons.extend(k for k, v in flags.items() if not v)
        valid_matches = [r for r in matched if r["qualifies_as_known_control_channel"]]
        if len(valid_matches) == 1:
            label = "known_control_channel"
            confirmed_uids.add(uid)
            counts[f"confirmed_pid_{valid_matches[0]['ecar_record']['pid']}"] += 1
        elif len(valid_matches) > 1:
            label = "ambiguous_multiple_agent_matches"
            reasons.append(label)
        else:
            label = "unconfirmed"
            if not rows:
                reasons.append("no_ecar_bro_uid_match")
        counts[f"joined_label_{label}"] += 1
        if rows and not any(r["checks"]["exact_host_uid_five_tuple"] for r in matched):
            counts["conn_endpoint_with_tuple_mismatch"] += 1
        for reason in set(reasons):
            counts[f"unconfirmed_reason_{reason}"] += 1
        joins.append({
            "uid": uid, "label": label, "conn_record": conn,
            "ecar_join_candidates": matched, "unconfirmed_reasons": sorted(set(reasons)),
            "conn_start_utc": utc(start), "flow_observed_end_utc": utc(end) if end is not None else None,
            "available_at_lower_bound_utc": utc(max([end] + [epoch(x['ecar_record']['timestamp']) for x in matched])) if end is not None else None,
            "available_at_actual_utc": None,
        })
    absent = [r for r in ecar_candidates if r.get("properties", {}).get("bro_uid") not in seen_conn_uids]
    counts["ecar_server_port_records_without_conn_in_supplied_hour"] = len(absent)
    http_stats, file_stats = {}, {}
    http_hits, file_hits = [], []
    for row in bro_rows(Path(manifest["http"]["path"]), http_stats):
        if row.get("uid") in confirmed_uids:
            http_hits.append(row)
    for row in bro_rows(Path(manifest["files"]["path"]), file_stats):
        if value_set(row.get("conn_uids")) & confirmed_uids:
            file_hits.append(row)
    counts["http_all_records"] = http_stats["data_rows"]
    counts["files_all_records"] = file_stats["data_rows"]
    counts["http_confirmed_control_uid_matches"] = len(http_hits)
    counts["files_confirmed_control_uid_matches"] = len(file_hits)
    counts["files_control_tls_certificates"] = sum(r.get("source") == "SSL" and r.get("mime_type") == "application/pkix-cert" for r in file_hits)
    confirmed = [r for r in joins if r["label"] == "known_control_channel"]
    import fitz
    anchors = []
    with fitz.open(manifest["ground_truth"]["path"]) as pdf:
        for number in (3, 4):
            lines = pdf[number].get_text().splitlines()
            for i, line in enumerate(lines):
                if any(needle in line for needle in ("10:46:02", "11:26:34", "11:33:15", "11:35:53", "13:05:03", "13:11:23", "13:19:38")):
                    anchors.append({"page_one_based": number + 1, "text": "\n".join(lines[i:i + 3])})
    output = {
        "schema_version": "optc_control_channel_slice_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "helper_script_path": previous["script_path"], "helper_script_sha256": sha256(Path(previous["script_path"])),
        "status": "known_control_channels_linked" if confirmed else "no_control_channel_confirmed",
        "scope": "Query of already-opened source data, not model/method evaluation. No downloads, training or scores.",
        "query": {"host": HOST, "source_ip": SOURCE_IP, "server_domain_from_ground_truth": "sports.com", "server_ip": SERVER_IP, "server_port": 443, "agents_from_official_pdf_page4": AGENTS},
        "official_ground_truth_anchors": anchors,
        "source_manifest": manifest,
        "download_event_json": {"path": str(download_event_path), "sha256_before": unchanged_sha, "sha256_after": sha256(download_event_path)},
        "counts": dict(sorted(counts.items())),
        "conn_file_schema_and_time_range": conn_stats,
        "all_ecar_host_server_port_candidates": ecar_candidates,
        "all_conn_endpoint_candidates_and_joins": joins,
        "confirmed_control_uids": sorted(confirmed_uids),
        "http_records_for_confirmed_uids": http_hits,
        "file_records_for_confirmed_uids": file_hits,
        "evidence_interpretation": {
            "label": "Known Empire control transport for a documented host, agent PID, server and active time; does not identify an executed command in each session.",
            "agent_count_in_confirmed_rows": len({r["ecar_join_candidates"][0]["ecar_record"]["pid"] for r in confirmed}),
            "independence": "All confirmed rows belong to the same documented attack episode; multiple beacons are not independent attack chains.",
            "time": "PDF uses second precision and local -04:00 is corroborated by the separate exact-second download. eCAR and Bro raw clocks are both retained; no global clock correction applied.",
            "unmatched_ecar": "The 25-host eCAR file spans more than the supplied conn hour. Missing hour-local conn matches are not automatically linkage errors or benign traffic.",
            "availability": "Full conn statistics require at least ts+duration. Actual log emission/timeout/ingestion timestamp is unknown. HTTP/file zero matches do not refute encrypted control channels.",
            "files": "Matched SSL application/pkix-cert records are TLS certificates, not payload/file-download attack events.",
            "hour_boundary": "The supplied 17-18 log can contain connections that started just before 17:00; retained raw timestamps define the actual interval.",
            "nonmatches": "Unmatched records remain unconfirmed; no residual normal labels assigned.",
        },
    }
    if output["download_event_json"]["sha256_before"] != output["download_event_json"]["sha256_after"]:
        raise ValueError("Download event evidence unexpectedly changed")
    dest = OUT / "optc_control_channel_slice.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "counts": output["counts"], "first_control_utc": min((r["conn_start_utc"] for r in confirmed), default=None), "last_control_utc": max((r["conn_start_utc"] for r in confirmed), default=None), "output": str(dest)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
