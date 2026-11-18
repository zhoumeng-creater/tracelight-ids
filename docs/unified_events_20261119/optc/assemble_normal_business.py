"""Assemble positively supported normal business from fixed, already-opened sources.

python assemble_normal_business.py
No downloads/training/scoring. Outputs complete enumerations of declared business
cohorts, not a normal label for all remaining traffic. Source metadata, process,
URL, MIME, SNI and parent-page fields establish labels and are NOT model inputs.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import bisect
import gzip
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit

from fetch_normal_sources import BASE, sha256

START = datetime.fromisoformat("2019-09-19T17:00:00+00:00").timestamp()
END = START + 3600
PARENT_MAX_AGE = 120.0
EXCLUDED_SERVERS = {"132.197.158.98", "202.6.172.98", "53.192.68.50"}
EXCLUDED_NAMES = {"news.com", "sports.com", "notepadplus-sourceforge.net"}
RAW_FIELDS = ("duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes")


def bro_rows(path, stats):
    fields = None
    stats.update({"data_rows": 0, "fields": [], "min_ts": None, "max_ts": None})
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
        for lineno, text in enumerate(f, 1):
            line = text.rstrip("\r\n")
            if line.startswith("#fields\t"):
                fields = line.split("\t")[1:]
                stats["fields"] = fields
            elif not line or line.startswith("#"):
                continue
            else:
                values = line.split("\t")
                if fields is None or len(fields) != len(values):
                    raise ValueError(f"Invalid record {path}:{lineno}")
                row = dict(zip(fields, values)); row["_source_line"] = lineno
                stats["data_rows"] += 1
                ts = float(row["ts"])
                stats["min_ts"] = ts if stats["min_ts"] is None else min(ts, stats["min_ts"])
                stats["max_ts"] = ts if stats["max_ts"] is None else max(ts, stats["max_ts"])
                yield row


def values(s):
    return [] if s in (None, "-", "(empty)", "") else s.split(",")


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def hostname(s):
    return (s or "").lower().split(":", 1)[0].rstrip(".")


def path_and_query(uri):
    p = urlsplit(unquote(uri))
    return (p.path or "/") + ("?" + p.query if p.query else "")


def tuple_match(e, r):
    p = e.get("properties", {})
    return (p.get("src_ip") == r.get("id.orig_h") and p.get("src_port") == r.get("id.orig_p")
            and p.get("dest_ip") == r.get("id.resp_h") and p.get("dest_port") == r.get("id.resp_p")
            and p.get("l4protocol") == "6")


def business_type(h):
    mime = h.get("resp_mime_types")
    path = urlsplit(unquote(h.get("uri", ""))).path.lower()
    if mime in ("image/jpeg", "image/png", "image/gif", "image/x-icon", "image/x-ms-bmp"):
        return "webpage_image_download"
    if mime == "application/font-woff":
        return "webpage_font_download"
    if mime in ("text/plain", "text/css") and path.endswith(".css"):
        return "webpage_stylesheet_download"
    return None


def write_jsonl_gz(path, rows):
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
            for row in rows:
                f.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def main():
    receipt = json.loads((BASE / "source_receipt.json").read_text(encoding="utf-8"))
    sources = receipt["sources"]
    for key, source in sources.items():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Source SHA mismatch: {key}")
    stats = {}; counts = Counter(); ecar_by_uid = defaultdict(list)
    process_counts = Counter()
    with gzip.open(sources["ecarbro"]["path"], "rt", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            r = json.loads(line); counts["ecar_all_records"] += 1
            p = r.get("properties", {}); image = p.get("image_path", "").lower()
            process_counts[image] += 1
            if image.endswith("firefox.exe") or image.endswith("chrome.exe"):
                r["_source_line"] = n
                ecar_by_uid[p.get("bro_uid")].append(r)
                counts["ecar_browser_records"] += 1
    http_by_uid = defaultdict(list); http_stats = {}; wanted_fuids = set(); all_mimes = Counter()
    for h in bro_rows(sources["http"]["path"], http_stats):
        all_mimes[h.get("resp_mime_types")] += 1
        if h["uid"] in ecar_by_uid:
            http_by_uid[h["uid"]].append(h)
            wanted_fuids.update(values(h.get("resp_fuids")))
    stats["http"] = http_stats
    counts["http_browser_uid_rows"] = sum(map(len, http_by_uid.values()))
    files = {}; file_stats = {}
    for f in bro_rows(sources["files"]["path"], file_stats):
        if f.get("fuid") in wanted_fuids:
            files[f["fuid"]] = f
    stats["files"] = file_stats
    ssl_by_uid = defaultdict(list); ssl_stats = {}
    for s in bro_rows(sources["ssl"]["path"], ssl_stats):
        if s["uid"] in ecar_by_uid:
            ssl_by_uid[s["uid"]].append(s)
    stats["ssl"] = ssl_stats
    counts["ssl_browser_uid_rows"] = sum(map(len, ssl_by_uid.values()))
    wanted_uids = set(http_by_uid) | set(ssl_by_uid)
    conn = {}; conn_stats = {}
    for c in bro_rows(sources["conn"]["path"], conn_stats):
        if c["uid"] in wanted_uids:
            if c["uid"] in conn:
                raise ValueError("Duplicate conn UID")
            conn[c["uid"]] = c
    stats["conn"] = conn_stats
    counts["conn_browser_http_or_ssl_uid_matches"] = len(conn)

    def context(uid):
        c = conn.get(uid)
        if c is None:
            return None, None, "no_conn"
        candidates = [e for e in ecar_by_uid[uid] if tuple_match(e, c)]
        if len(candidates) != 1:
            return c, None, "ecar_tuple_absent_or_ambiguous"
        e = candidates[0]
        if c.get("proto") != "tcp" or c.get("conn_state") != "SF" or c.get("missed_bytes") != "0":
            return c, e, "conn_not_complete_tcp_SF"
        if c.get("duration") == "-":
            return c, e, "no_duration"
        start = float(c["ts"]); end = start + float(c["duration"])
        if not (START <= start <= end < END):
            return c, e, "crosses_capture_hour_boundary"
        if c.get("id.resp_h") in EXCLUDED_SERVERS:
            return c, e, "known_attack_infrastructure_unknown"
        return c, e, None

    def complete_http_file(h):
        if h.get("method") != "GET" or h.get("status_code") != "200" or h.get("request_body_len") != "0":
            return None, "not_successful_get"
        fids = values(h.get("resp_fuids"))
        if len(fids) != 1 or fids[0] not in files:
            return None, "file_fuid_missing_or_multiple"
        f = files[fids[0]]
        if (h["uid"] not in values(f.get("conn_uids")) or f.get("source") != "HTTP"
                or int(f["seen_bytes"]) != int(h["response_body_len"])
                or int(f["seen_bytes"]) <= 0 or f.get("missing_bytes") != "0"
                or f.get("overflow_bytes") != "0" or f.get("timedout") != "F"):
            return f, "file_response_not_complete"
        if f.get("total_bytes") != "-" and int(f["total_bytes"]) != int(f["seen_bytes"]):
            return f, "file_total_mismatch"
        return f, None

    documents = defaultdict(list)
    for uid, hs in http_by_uid.items():
        c, e, err = context(uid)
        if err:
            continue
        for h in hs:
            if h.get("resp_mime_types") != "text/html" or not tuple_match(e, h):
                continue
            f, error = complete_http_file(h)
            host = hostname(h.get("host"))
            if error or host in EXCLUDED_NAMES:
                continue
            documents[(e["actorID"], host, path_and_query(h["uri"]))].append((float(h["ts"]), h, f, e))
    for ds in documents.values():
        ds.sort(key=lambda r: r[0])
    counts["complete_browser_html_documents"] = sum(map(len, documents.values()))
    objects = []; raw_object_examples = []; objects_by_uid = defaultdict(list)
    qualified_http_lines = set(); object_reasons = Counter()
    for uid, hs in http_by_uid.items():
        c, e, error = context(uid)
        if error:
            object_reasons[error] += len(hs); continue
        for h in hs:
            kind = business_type(h)
            if kind is None:
                object_reasons["not_declared_static_object_type"] += 1; continue
            f, err = complete_http_file(h)
            if err or not tuple_match(e, h):
                object_reasons[err or "http_ecar_tuple_mismatch"] += 1; continue
            ref = urlsplit(unquote(h.get("referrer", "")))
            host = hostname(h.get("host"))
            if ref.scheme not in ("http", "https") or hostname(ref.netloc) != host or host in EXCLUDED_NAMES:
                object_reasons["no_same_site_parent_url"] += 1; continue
            possible = documents.get((e["actorID"], host, path_and_query(h["referrer"])), [])
            at = float(h["ts"])
            idx = bisect.bisect_right([r[0] for r in possible], at) - 1
            if idx < 0 or at - possible[idx][0] > PARENT_MAX_AGE:
                object_reasons["no_observed_parent_document_within_120s"] += 1; continue
            parent = possible[idx]
            obj_id = "optc_20190919_httpfile_" + f["fuid"]
            item = {
                "event_id": obj_id, "label": "normal", "evidence_grade": "B1_author_background_plus_observed_page_object_process",
                "business_type": kind, "uid": uid, "fuid": f["fuid"], "hostname": e["hostname"],
                "actor_id": e["actorID"], "pid": e["pid"], "process_image": e["properties"]["image_path"],
                "url": "http://" + h["host"] + h["uri"], "referrer": h["referrer"],
                "mime_type": h["resp_mime_types"], "object_bytes": int(f["seen_bytes"]),
                "request_utc": utc(at), "file_observed_end_utc": utc(float(f["ts"]) + float(f["duration"])),
                "conn_end_utc": utc(float(c["ts"]) + float(c["duration"])),
                "available_at_lower_bound_utc": utc(max(float(c["ts"]) + float(c["duration"]), float(f["ts"]) + float(f["duration"]))),
                "available_at_actual_utc": None,
                "source_rows": {"http": h["_source_line"], "files": f["_source_line"], "conn": c["_source_line"], "ecarbro": e["_source_line"], "parent_http": parent[1]["_source_line"], "parent_files": parent[2]["_source_line"]},
                "parent_document_uid": parent[1]["uid"], "parent_age_seconds": at - parent[0],
                "scope": "Known normal webpage object; whole-flow label requires all HTTP transactions to pass the same declared rule.",
            }
            objects.append(item); objects_by_uid[uid].append(item); qualified_http_lines.add(h["_source_line"])
            if len(raw_object_examples) < 12:
                raw_object_examples.append({"event": item, "http": h, "file": f, "conn": c, "ecar": e, "parent_http": parent[1], "parent_file": parent[2]})
    flows = []; raw_flow_examples = []; flow_reasons = Counter(); tls_names = Counter()

    def add_flow(uid, business, grade, proof):
        c, e, error = context(uid)
        assert error is None
        features = {k: float(c[k]) if k == "duration" else int(c[k]) for k in RAW_FIELDS}
        assert all(math.isfinite(v) and v >= 0 for v in features.values())
        at = float(c["ts"]); end = at + float(c["duration"])
        domain = proof.get("server_name") or hostname(urlsplit(objects_by_uid[uid][0]["url"]).netloc)
        group = hashlib.sha256((e["hostname"] + "|" + e["actorID"] + "|" + domain).encode()).hexdigest()[:16]
        item = {
            "event_id": "optc_20190919_normalflow_" + uid,
            "label": "normal", "evidence_grade": grade, "business_type": business,
            "source": "OpTC", "role": "opened_development_source_qualification", "split": "development",
            "acquisition_group": "optc_benign_20190919_AIA501-525_hour17",
            "business_episode_group": "optc_browser_20190919_" + group,
            "uid": uid, "hostname": e["hostname"], "actor_id": e["actorID"], "pid": e["pid"],
            "process_image": e["properties"]["image_path"], "start_utc": utc(at), "flow_end_utc": utc(end),
            "available_at_lower_bound_utc": utc(end), "available_at_actual_utc": None,
            "conn_minus_ecar_seconds": at - datetime.fromisoformat(e["timestamp"]).timestamp(),
            "source_rows": {"conn": c["_source_line"], "ecarbro": e["_source_line"]},
            "protocol": c["proto"], "service": c["service"], "destination_port_for_audit": int(c["id.resp_p"]),
            "conn_state": c["conn_state"], "raw_numeric_features": features,
            "label_provenance_only": proof,
        }
        flows.append(item)
        if sum(x["event"]["business_type"] == business for x in raw_flow_examples) < 4:
            raw_flow_examples.append({"event": item, "conn": c, "ecar": e, "ssl": ssl_by_uid.get(uid, [])})

    for uid, os in objects_by_uid.items():
        hs = http_by_uid[uid]
        if not all(h["_source_line"] in qualified_http_lines for h in hs):
            flow_reasons["http_connection_has_other_unqualified_transactions"] += 1; continue
        c, e, error = context(uid)
        if error or c["service"] != "http":
            flow_reasons[error or "not_http_service"] += 1; continue
        add_flow(uid, "normal_http_webpage_file_transfer", "B1_author_background_plus_observed_page_object_process", {
            "object_event_ids": [r["event_id"] for r in os], "http_transaction_count": len(hs),
            "all_http_transactions_qualify": True, "object_bytes_sum": sum(r["object_bytes"] for r in os),
            "object_business_types": sorted({r["business_type"] for r in os}),
            "http_rows": [h["_source_line"] for h in hs],
        })
    tls_reasons = Counter()
    for uid, ss in ssl_by_uid.items():
        c, e, error = context(uid)
        if error:
            tls_reasons[error] += 1; continue
        if c.get("id.resp_p") != "443" or c.get("service") not in ("ssl", "tls"):
            tls_reasons["not_tls_443"] += 1; continue
        if len(ss) != 1:
            tls_reasons["multiple_ssl_records"] += 1; continue
        s = ss[0]; name = hostname(s.get("server_name"))
        if not name or name == "-" or "." not in name or name in EXCLUDED_NAMES:
            tls_reasons["no_usable_site_sni"] += 1; continue
        if s.get("established") != "T" or not tuple_match(e, s):
            tls_reasons["tls_not_established_or_tuple_mismatch"] += 1; continue
        tls_names[name] += 1
        add_flow(uid, "normal_tls_browser_background", "B2_author_background_plus_browser_uid_tuple_and_established_site_TLS", {
            "server_name": name, "ssl_source_line": s["_source_line"], "tls_version": s["version"],
            "tls_established": True, "resumed": s["resumed"], "cert_chain_fuids": values(s.get("cert_chain_fuids")),
            "certificate_subject": s.get("subject"), "certificate_issuer": s.get("issuer"),
            "object_content_observed": False,
            "basis": "Official benign collection + concrete browser process + exact network tuple + established named-site TLS. No claim of an observed HTTPS URL, file payload, or certificate trust validation.",
        })
    flows.sort(key=lambda r: (r["start_utc"], r["event_id"]))
    objects.sort(key=lambda r: (r["request_utc"], r["event_id"]))
    assert len({r["event_id"] for r in flows}) == len(flows)
    assert len({r["event_id"] for r in objects}) == len(objects)
    write_jsonl_gz(BASE / "normal_flow_events.jsonl.gz", flows)
    write_jsonl_gz(BASE / "normal_http_file_objects.jsonl.gz", objects)
    http_flows = [r for r in flows if r["business_type"] == "normal_http_webpage_file_transfer"]
    tls_flows = [r for r in flows if r["business_type"] == "normal_tls_browser_background"]
    counts.update({"normal_http_file_objects": len(objects), "normal_http_flow_events": len(http_flows), "normal_tls_browser_flow_events": len(tls_flows), "normal_flow_events": len(flows)})
    prior_unknown = BASE.parents[1] / "独立来源与算法主线_20261118" / "evidence" / "optc" / "benign_file_business.json"
    manifest = {
        "schema_version": "optc_normal_business_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "sources": sources, "official_collection_evidence": receipt["official_directory_evidence"],
        "additional_data_bytes": receipt["additional_data_bytes"], "scan_stats": stats,
        "counts": dict(counts), "all_http_mime_counts": dict(all_mimes),
        "normal_object_types": dict(Counter(r["business_type"] for r in objects)),
        "normal_flow_host_count": len({r["hostname"] for r in flows}),
        "normal_flow_episode_count": len({r["business_episode_group"] for r in flows}),
        "normal_tls_site_counts": dict(tls_names),
        "rejections": {"http_object": dict(object_reasons), "http_flow": dict(flow_reasons), "tls_flow": dict(tls_reasons)},
        "comparability": {
            "attack_download_object_bytes": 59392, "attack_download_conn_resp_bytes": 59598,
            "normal_objects_in_half_to_twice_attack_object_bytes": sum(29696 <= r["object_bytes"] <= 118784 for r in objects),
            "normal_http_flows_in_half_to_twice_attack_conn_resp_bytes": sum(29799 <= r["raw_numeric_features"]["resp_bytes"] <= 119196 for r in http_flows),
            "http_business": "Ordinary webpage image/font/CSS transfers, not normal executable/software-update downloads. All are real complete HTTP objects; non-EXE content does not invalidate them as HTTP transfer counterexamples.",
            "port_difference": "Known malicious download uses HTTP 8080; normal HTTP transfer cohort may use port 80. Destination port is audit-only and excluded from the shared numeric input contract.",
            "tls_business": "Known Empire control and this browser cohort both use established TLS on TCP 443. HTTPS payload/command identity is not observable here.",
            "permitted_shared_numeric_inputs": list(RAW_FIELDS),
            "provenance_not_model_inputs": ["host/IP", "PID/process", "actorID", "uid/fuid", "URL/referrer", "SNI", "MIME", "source date/line", "label", "destination port"],
        },
        "denominator": {
            "complete_within_declared_business_filters": True,
            "http_file_transfer_flows": len(http_flows), "tls_browser_flows": len(tls_flows),
            "whole_network_normal_denominator_established": False,
            "meaning": "Every qualifying record in the specified source hour and 25-host eCAR shard was enumerated. This is a conditional normal-business denominator, not all normal enterprise traffic or evidence for remote administration/software updates.",
            "unit": "Unique connection UID. Multiple file objects share connections and multiple connections share browser episodes; neither is counted as an independent attack/event repetition.",
            "unknown_news_com_PE_events": 6, "unknown_evidence_path": str(prior_unknown), "unknown_evidence_sha256": sha256(prior_unknown),
        },
        "rules": {
            "http": "Official benign shard; unique browser eCAR uid/five-tuple; complete SF conn within hour; GET 200 zero request body; image/font/CSS MIME and extension where appropriate; response byte count equals complete files object; same-site parent HTML+file observed by same actor within prior 120s. Whole-flow normal requires ALL observed HTTP transactions to meet that rule.",
            "tls": "Official benign shard; unique browser eCAR uid/five-tuple; complete SF TLS/443 conn within hour; one established SSL record with named-site SNI. Not a payload/action-level normal label.",
            "excluded_known_infrastructure": sorted(EXCLUDED_SERVERS),
            "not_used": "No model predictions/scores. No assumption that all non-red-team-IP traffic is normal. No relabeling of six PE anomalies.",
        },
        "artifacts": {},
    }
    for name in ("normal_flow_events.jsonl.gz", "normal_http_file_objects.jsonl.gz"):
        file = BASE / name
        manifest["artifacts"][name] = {"path": str(file), "bytes": file.stat().st_size, "sha256": sha256(file)}
    examples = {"http_object_examples": raw_object_examples, "flow_examples": raw_flow_examples}
    (BASE / "normal_event_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (BASE / "normal_business_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": manifest["counts"], "object_types": manifest["normal_object_types"], "size_overlap_objects": manifest["comparability"]["normal_objects_in_half_to_twice_attack_object_bytes"], "size_overlap_http_flows": manifest["comparability"]["normal_http_flows_in_half_to_twice_attack_conn_resp_bytes"], "hosts": manifest["normal_flow_host_count"], "episodes": manifest["normal_flow_episode_count"], "artifacts": manifest["artifacts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
