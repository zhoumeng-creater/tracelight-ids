"""Fetch only the approved 2019-09-24 OpTC slice and join one documented action.

Usage: python assemble_optc_event_slice.py [--with-conn] [--offline]
No model, score, or inferred normal labels are produced. Re-runs reuse size-checked
originals and recompute SHA256. All candidate records matching the stated queries
are retained; ambiguity is reported rather than resolved by choosing the first row.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

import requests


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "evidence" / "optc"
ACTION_LOCAL = "2019-09-24T13:26:57-04:00"
ACTION_EPOCH = datetime.fromisoformat(ACTION_LOCAL).timestamp()
TARGET_IP = "142.20.57.246"
TARGET_HOST = "sysclient0501.systemia.com"
TARGET_FILENAME = "filetransfer1000.exe"
SOURCES = {
    "ecarbro": ("optc_2019-09-24_AIA-501-525_ecarbro.json.gz", "1-45syji9pX-_QMxR1E_xIfjxcZ2eAPLB", 19992689),
    "http": ("optc_2019-09-24_http_17-18.log.gz", "14zNUSYWap4CiCKdTv4HZ9v56P6bFIPiQ", 17159303),
    "files": ("optc_2019-09-24_files_17-18.log.gz", "1AkEqZRG7Ox-PwsnywhG3vxY_iDnDoz1e", 12607089),
    "conn": ("optc_2019-09-24_conn_17-18.log.gz", "1EYNYdkVSIDw78XfDcUGD9cRFHzjsSmMH", 58407076),
    "ground_truth": ("OpTCRedTeamGroundTruth.pdf", "https://raw.githubusercontent.com/FiveDirections/OpTC-data/master/OpTCRedTeamGroundTruth.pdf", 453244),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def obtain(key: str, offline: bool) -> tuple[str, dict]:
    name, identifier, size = SOURCES[key]
    path = OUT / name
    url = identifier if identifier.startswith("https://") else (
        "https://drive.usercontent.google.com/download?id=" + identifier
        + "&export=download&confirm=t"
    )
    if not path.exists() or path.stat().st_size != size:
        if offline:
            raise FileNotFoundError(f"Missing exact-size source: {path}")
        partial = path.with_suffix(path.suffix + ".partial")
        print(f"DOWNLOAD {key}: {size} bytes", flush=True)
        received = 0
        with requests.get(url, stream=True, timeout=(30, 120), headers={"Cache-Control": "no-cache"}) as r:
            r.raise_for_status()
            with partial.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        received += len(chunk)
                        if received > size:
                            raise ValueError(f"Source larger than approved manifest: {key}")
                        f.write(chunk)
        if received != size:
            raise ValueError(f"Source-size mismatch {key}: {received} != {size}")
        partial.replace(path)
    result = {
        "path": str(path), "url": url, "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "view_url": identifier if identifier.startswith("https://") else f"https://drive.google.com/file/d/{identifier}/view",
    }
    print(f"READY {key}: {result['sha256']}", flush=True)
    return key, result


def bro_rows(path: Path, stats: dict):
    fields = None
    stats.update({"data_rows": 0, "fields": [], "first_ts": None, "last_ts": None})
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if line.startswith("#fields\t"):
                fields = line.split("\t")[1:]
                stats["fields"] = fields
            elif line.startswith("#") or not line:
                continue
            else:
                if fields is None:
                    raise ValueError(f"No Bro header: {path}:{lineno}")
                values = line.split("\t")
                if len(values) != len(fields):
                    raise ValueError(f"Invalid column count: {path}:{lineno}")
                row = dict(zip(fields, values))
                row["_source_line"] = lineno
                stats["data_rows"] += 1
                ts = float(row["ts"])
                stats["first_ts"] = ts if stats["first_ts"] is None else min(ts, stats["first_ts"])
                stats["last_ts"] = ts if stats["last_ts"] is None else max(ts, stats["last_ts"])
                yield row


def value_set(value):
    return set() if value in (None, "-", "(empty)", "") else set(value.split(","))


def utc(epoch):
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat()


def decorate_bro(row):
    return {
        **row,
        "_timestamp_utc": utc(row["ts"]),
        "_seconds_from_action_assuming_minus04": float(row["ts"]) - ACTION_EPOCH,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-conn", action="store_true")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = ["ecarbro", "http", "files", "ground_truth"] + (["conn"] if args.with_conn else [])
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(lambda key: obtain(key, args.offline), keys))
    stats = {}
    http_filename, http_host_port = [], []
    stats["http"] = {}
    for row in bro_rows(Path(manifest["http"]["path"]), stats["http"]):
        row = decorate_bro(row)
        if TARGET_FILENAME in unquote(row.get("uri", "")).casefold():
            http_filename.append(row)
        if row.get("id.orig_h") == TARGET_IP and row.get("id.resp_p") == "8080":
            http_host_port.append(row)
    uids = {row["uid"] for row in http_filename}
    fuids = set().union(*(value_set(r.get("resp_fuids")) | value_set(r.get("orig_fuids")) for r in http_filename))
    print(f"HTTP full scan: {stats['http']['data_rows']} rows, filename={len(http_filename)}, target_host_port={len(http_host_port)}", flush=True)
    file_matches = []
    stats["files"] = {}
    for row in bro_rows(Path(manifest["files"]["path"]), stats["files"]):
        if row.get("fuid") in fuids or value_set(row.get("conn_uids")) & uids:
            file_matches.append(decorate_bro(row))
    flow_matches, flow_host_port = [], []
    stats["ecarbro"] = {"data_rows": 0, "target_hostname_rows": 0, "first_timestamp": None, "last_timestamp": None}
    with gzip.open(manifest["ecarbro"]["path"], "rt", encoding="utf-8", errors="strict") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = lineno
            stats["ecarbro"]["data_rows"] += 1
            if stats["ecarbro"]["first_timestamp"] is None:
                stats["ecarbro"]["first_timestamp"] = row.get("timestamp")
            stats["ecarbro"]["last_timestamp"] = row.get("timestamp")
            prop = row.get("properties", {})
            if prop.get("bro_uid") in uids:
                flow_matches.append(row)
            if row.get("hostname", "").casefold() == TARGET_HOST:
                stats["ecarbro"]["target_hostname_rows"] += 1
                if prop.get("dest_port") == "8080":
                    flow_host_port.append(row)
    conn_matches = []
    if args.with_conn:
        stats["conn"] = {}
        for row in bro_rows(Path(manifest["conn"]["path"]), stats["conn"]):
            if row.get("uid") in uids:
                conn_matches.append(decorate_bro(row))
    ground_truth_hits = []
    import fitz
    with fitz.open(manifest["ground_truth"]["path"]) as doc:
        for page_idx, page in enumerate(doc):
            lines = page.get_text().splitlines()
            for idx, line in enumerate(lines):
                if TARGET_FILENAME in line.casefold():
                    ground_truth_hits.append({"page_one_based": page_idx + 1, "text": "\n".join(lines[max(0, idx - 2):idx + 4])})
    primary = [r for r in http_filename if r.get("id.orig_h") == TARGET_IP and r.get("id.resp_p") == "8080"]
    primary_uids = {r["uid"] for r in primary}
    primary_files = [r for r in file_matches if value_set(r.get("conn_uids")) & primary_uids]
    primary_flows = [r for r in flow_matches if r.get("properties", {}).get("bro_uid") in primary_uids]
    primary_conn = [r for r in conn_matches if r["uid"] in primary_uids]
    linked = bool(primary and primary_files and primary_flows)
    status = "linked_single_session" if linked and len(primary_uids) == 1 else ("ambiguous_multiple_sessions" if linked else "incomplete_or_no_match")
    numeric_fields = ("duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes")
    numerics = []
    for row in primary_conn:
        numerics.append({"uid": row["uid"], **{k: None if row[k] == "-" else float(row[k]) if k == "duration" else int(row[k]) for k in numeric_fields}})
    sessions = []
    for uid in sorted(primary_uids):
        hr = [r for r in primary if r["uid"] == uid]
        er = [r for r in primary_flows if r.get("properties", {}).get("bro_uid") == uid]
        fr = [r for r in primary_files if uid in value_set(r.get("conn_uids"))]
        cr = [r for r in primary_conn if r["uid"] == uid]
        flow_ends = [float(r["ts"]) + float(r["duration"]) for r in cr if r["duration"] != "-"]
        file_ends = [float(r["ts"]) + float(r["duration"]) for r in fr if r["duration"] != "-"]
        host_times = [datetime.fromisoformat(r["timestamp"]).timestamp() for r in er]
        tuple_checks = []
        for e in er:
            prop = e["properties"]
            for h in hr:
                tuple_checks.append({
                    "ecar_line": e["_source_line"], "http_line": h["_source_line"],
                    "src_ip": prop.get("src_ip") == h.get("id.orig_h"),
                    "src_port": prop.get("src_port") == h.get("id.orig_p"),
                    "dest_ip": prop.get("dest_ip") == h.get("id.resp_h"),
                    "dest_port": prop.get("dest_port") == h.get("id.resp_p"),
                    "protocol_tcp": prop.get("l4protocol") == "6",
                    "http_minus_ecar_seconds": float(h["ts"]) - datetime.fromisoformat(e["timestamp"]).timestamp(),
                })
        lower_bound = max(flow_ends + file_ends + host_times) if flow_ends else None
        sessions.append({
            "uid": uid, "http_source_lines": [r["_source_line"] for r in hr],
            "ecar_source_lines": [r["_source_line"] for r in er],
            "files_source_lines": [r["_source_line"] for r in fr],
            "conn_source_lines": [r["_source_line"] for r in cr],
            "http_request_utc": [utc(r["ts"]) for r in hr],
            "request_matches_reported_second_under_minus04": any(int(float(r["ts"])) == int(ACTION_EPOCH) for r in hr),
            "ecar_http_tuple_checks": tuple_checks,
            "http_response_status_codes": [r.get("status_code") for r in hr],
            "http_response_bytes": [int(r["response_body_len"]) for r in hr],
            "file_completeness": [{"fuid": r["fuid"], "seen_bytes": int(r["seen_bytes"]), "total_bytes": None if r["total_bytes"] == "-" else int(r["total_bytes"]), "missing_bytes": int(r["missing_bytes"]), "overflow_bytes": int(r["overflow_bytes"]), "timedout": r["timedout"]} for r in fr],
            "flow_observed_end_utc": [utc(t) for t in flow_ends],
            "file_observed_end_utc": [utc(t) for t in file_ends],
            "http_response_completion_timestamp_utc": None,
            "http_completion_evidence": "HTTP contains completed response length/status, and files contains seen/total bytes; http.log has no separate completion timestamp.",
            "available_at_lower_bound_utc": utc(lower_bound) if lower_bound is not None else None,
            "available_at_actual_utc": None,
            "availability_semantics": "Lower bound is max(conn ts+duration, files ts+duration, matched eCAR timestamp). Actual Bro output/timeout/ingestion time is not recorded here. Do not substitute action_start or request_start.",
            "file_content_hash": None,
            "file_identity_limit": "Author-described nc.exe identified by URI, response size/type and process context; file SHA fields are unset, so exact executable bytes are not hashed here.",
        })
    same_second_uids = [s["uid"] for s in sessions if s["request_matches_reported_second_under_minus04"]]
    if linked and len(primary_uids) > 1 and len(same_second_uids) == 1:
        status = "linked_multiple_downloads_one_reported_second_match"
    output = {
        "schema_version": "optc_event_slice_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "status": status,
        "scope": "One red-team download action; full scans of explicitly listed originals; no model scoring or residual-normal labeling.",
        "action": {
            "event_id": "optc_20190924_sysclient0501_fileTransfer1000",
            "reported_local_time": "2019-09-24 13:26:57",
            "assumed_offset_for_comparison": "-04:00 (consistent with eCAR timestamps; PDF clock must be corroborated by event match)",
            "expected_utc_under_assumption": utc(ACTION_EPOCH),
            "host": "Sysclient0501", "ip": TARGET_IP,
            "description": "Chrome download of fileTransfer1000.exe (nc.exe) from news.com:8080, documented after RDP access.",
            "ground_truth_excerpt": ground_truth_hits,
        },
        "source_manifest": manifest,
        "full_scan_counts_and_schema": stats,
        "selection_rules": {
            "http_filename": "case-insensitive filetransfer1000.exe substring in URL-decoded uri, any host, all records in supplied hour",
            "http_target_host_port": "id.orig_h=142.20.57.246 AND id.resp_p=8080, all records in supplied hour",
            "ecarbro_target_host_port": "hostname=sysclient0501.systemia.com case-insensitive AND dest_port=8080, all records in supplied 25-host file",
            "join": "ecarbro.properties.bro_uid=HTTP.uid=conn.uid; HTTP resp/orig_fuids=files.fuid or HTTP.uid in files.conn_uids",
            "uniqueness": "based on full selected-file scans only; repeated requests/sessions remain explicit",
        },
        "candidates": {
            "http_filename_all": http_filename, "http_target_host_port_all": http_host_port,
            "ecarbro_uid_matches_all": flow_matches, "ecarbro_target_host_port_all": flow_host_port,
            "files_fuid_or_uid_matches_all": file_matches, "conn_uid_matches_all": conn_matches,
        },
        "joined_event": {
            "primary_http_rows": len(primary), "primary_unique_uids": sorted(primary_uids),
            "primary_files_rows": len(primary_files), "primary_ecarbro_rows": len(primary_flows),
            "primary_conn_rows": len(primary_conn), "conn_raw_numeric_fields": numerics,
            "feature_contract": "Raw conn numeric values only. Not yet a declared model feature vector.",
            "same_reported_second_candidate_uids": same_second_uids,
            "sessions": sessions,
            "action_assignment": "All matching downloads are retained. A same-second request is separately identified as strongest support for the documented action; other requests are repeated-download candidates, not new independent attack chains.",
        },
        "normal_background_status": "unknown; no residual records labeled normal",
    }
    dest = OUT / "optc_event_slice.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(dest), "status": status, "http_filename_count": len(http_filename), "primary_uids": sorted(primary_uids), "flow_matches": len(primary_flows), "files_matches": len(primary_files), "conn_matches": len(primary_conn)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
