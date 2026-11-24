"""Corroborate source download anchors with exact observed HTTP request digests."""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from .harrison_download_audit import http_request_digest
from .harrison_evidence import checked_source_paths, consume, get_clock, write_new_report
from .harrison_review import complete_observation_reasons


def load_json(path):
    content = path.read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def checked_receipt_file(path, record):
    with path.open("rb") as source:
        _, _, digest = consume(source)
    if digest != record["sha256"]:
        raise ValueError("Capture export evidence does not match its recorded digest")
    return digest


def get_http_digest(row):
    authority = row.get("host")
    uri = row.get("uri")
    port = row.get("id.resp_p")
    if not isinstance(authority, str) or not isinstance(uri, str) or not isinstance(port, int):
        return None
    try:
        observed = urlsplit(uri)
        if observed.netloc:
            parsed = observed
        else:
            parsed = urlsplit("//" + authority)
        if not parsed.hostname or parsed.username:
            return None
        return http_request_digest(parsed.hostname, parsed.port or port, observed.path, observed.query)
    except ValueError:
        return None


def join_http(extraction_dir, audit_path, manifest_path):
    archive_audit, audit_sha = load_json(audit_path)
    manifest, manifest_sha = load_json(manifest_path)
    if (archive_audit["schema"] != "ait-harrison-transfer-source-audit-v1"
            or archive_audit["source_manifest_sha256"] != manifest_sha):
        raise ValueError("Download audit does not bind the source manifest")
    raw_paths = checked_source_paths(manifest)
    roster = [name for name in raw_paths if name.endswith("/logs/sm.log")]
    servers, start, end = get_clock(raw_paths, roster)
    victim = servers["intranet_server"]["default_ipv4_address"]
    extraction, extraction_sha = load_json(extraction_dir / "extraction_receipt.json")
    if extraction["scenario"] != "harrison" or extraction["capture_host"] != "intranet_server":
        raise ValueError("Expected a full Harrison intranet export")
    export_receipt_path = extraction_dir / "zeek" / "export-receipt.json"
    export_receipt_sha = checked_receipt_file(
        export_receipt_path, extraction["outputs"]["zeek/export-receipt.json"]
    )
    export, _ = load_json(export_receipt_path)
    if export["pcap_sha256"] != extraction["declared_capture_sha256"]:
        raise ValueError("Network HTTP receipt is not from the same normalized capture")
    events = archive_audit["author_completion_anchors"]
    matches = collections.defaultdict(list)
    http_log = extraction_dir / "zeek" / "http.log"
    http_sha = None
    if export["logs"]["http"]["exists"]:
        http_sha = checked_receipt_file(http_log, export["logs"]["http"])
        with http_log.open(encoding="utf-8") as stream:
            for line, record in enumerate(stream, 1):
                row = json.loads(record)
                if row.get("id.orig_h") != victim or row.get("method") != "GET":
                    continue
                digest = get_http_digest(row)
                time = row.get("ts")
                if digest is None or not isinstance(time, (float, int)):
                    continue
                for event in events:
                    anchor = event["intent"]
                    if (digest in anchor["request_key_sha256"]
                            and anchor["time_unix"] <= time <= event["completion"]["time_unix"]):
                        body = row.get("response_body_len")
                        matches[event["event_sha256"]].append({
                            "request_line": line, "request_record_sha256": hashlib.sha256(record.encode()).hexdigest(),
                            "request_unix": time, "uid": row["uid"],
                            "http_response_success_and_positive_body": (
                                isinstance(row.get("status_code"), int) and 200 <= row["status_code"] < 300
                                and isinstance(body, (float, int)) and body > 0
                            ),
                        })
    conn_path = extraction_dir / "connections.jsonl.gz"
    conn_sha = checked_receipt_file(conn_path, extraction["outputs"]["connections.jsonl.gz"])
    wanted = {request["uid"] for requests in matches.values() for request in requests}
    by_uid = {}
    if wanted:
        with gzip.open(conn_path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["uid"] in wanted:
                    if row["uid"] in by_uid:
                        raise ValueError("Export contains multiple rows for a matched HTTP UID")
                    by_uid[row["uid"]] = row
    results = []
    for event in events:
        sha = event["event_sha256"]
        request_records = []
        for request in matches[sha]:
            uid = request.pop("uid")
            row = by_uid.get(uid)
            if row is None:
                request["qualification_reasons"] = ["network_http_uid_without_flow_observation"]
            else:
                request["flow_id"] = row["flow_id"]
                request["stable_flow_id"] = row["stable_flow_id"]
                request["qualification_reasons"] = complete_observation_reasons(row, start, end)
                if not row["connection_start"] <= request["request_unix"] <= row["observed_end"]:
                    request["qualification_reasons"].append("request_outside_network_observation")
                response_bytes = row.get("resp_bytes")
                request["network_transport_response_positive"] = (
                    isinstance(response_bytes, (float, int)) and response_bytes > 0
                )
            request_records.append(request)
        valid = [
            r for r in request_records if not r["qualification_reasons"]
            and r["network_transport_response_positive"]
        ]
        exact_ids = {r.get("flow_id") for r in request_records}
        if not request_records:
            status = "no_exact_network_http_request_observed"
        elif len(exact_ids) != 1 or None in exact_ids or not valid:
            status = "network_request_or_transport_not_uniquely_qualified"
        elif any(r["http_response_success_and_positive_body"] for r in valid):
            status = "unique_eligible_flow_with_author_completion_and_observed_http_response"
        else:
            status = "unique_eligible_flow_with_author_completion_and_observed_http_request"
        results.append({
            "event_id": f"ait:harrison:download:{sha}", "source_event_sha256": sha,
            "status": status, "network_request_records": request_records,
        })
    return {
        "schema": "ait-harrison-download-http-source-join-v1", "admission_status": "not_admitted",
        "download_source_audit_sha256": audit_sha, "extractor_receipt_sha256": extraction_sha,
        "zeek_export_receipt_sha256": export_receipt_sha, "http_log_sha256": http_sha,
        "connections_sha256": conn_sha,
        "scope": "Source and exact network request digest only. Author completion, captured transport "
                 "and observed response body are distinguished. No response content or model scores.",
        "events": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-export", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = join_http(args.capture_export, args.audit, args.manifest)
    write_new_report(args.output, report)
    print(f"Checked {len(report['events'])} source anchors against recorded HTTP evidence")


if __name__ == "__main__":
    main()
