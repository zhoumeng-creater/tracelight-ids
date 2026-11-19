"""Join AIT event evidence to Zeek records without fitting, scoring, or assigning truth."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


RAW8 = ("ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts",
        "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
PROTOCOL = {"icmp": 1, "tcp": 6, "udp": 17, "icmp6": 58, "icmpv6": 58, "sctp": 132}
SERVER = "10.143.2.4"
MISSING = (None, "", "-", "(empty)")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected Zeek JSON object at {path.name}:{number}")
                result.append(value)
    return result


def write_csv(path, rows, fields=()):
    columns = list(dict.fromkeys([*fields, *(key for row in rows for key in row)])) or ["no_rows"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                             else value for key, value in row.items()})


def number(value, *, integer=False):
    if value in MISSING:
        return -1
    result = float(value)
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise ValueError(f"Invalid nonnegative Zeek numeric value: {value!r}")
    return int(result) if integer else result


def epoch(value):
    if isinstance(value, (float, int)):
        return float(value)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)  # Author dataset.yaml UTC.
    return parsed.timestamp()


def optional_time(value):
    return None if value in MISSING else number(value)


def normalize_connection(row, interval):
    for field in ("uid", "ts", "id.orig_h", "id.resp_h", "proto"):
        if row.get(field) in MISSING:
            raise ValueError(f"Connection lacks required field {field}")
    ts, duration = number(row["ts"]), number(row.get("duration"))
    last = ts + duration if duration >= 0 else None
    start, stop = interval
    if last is None:
        relation = "unknown_end"
    elif last < start:
        relation = "before"
    elif ts >= stop:
        relation = "after"
    elif ts < start and last >= stop:
        relation = "crosses_both_boundaries"
    elif ts < start:
        relation = "crosses_start"
    elif last >= stop:
        relation = "crosses_end"
    else:
        relation = "inside"
    features = {"ip_protocol": PROTOCOL.get(str(row["proto"]).lower(), -1),
                "orig_port": number(row.get("id.orig_p"), integer=True),
                "resp_port": number(row.get("id.resp_p"), integer=True),
                "duration_seconds": duration}
    for field in RAW8[4:]:
        features[field] = number(row.get(field), integer=True)
    if any(features[name] > 65535 for name in ("orig_port", "resp_port")):
        raise ValueError("Invalid Zeek port")
    export = optional_time(row.get("contract_export_network_time"))
    after_eof, shutdown = row.get("contract_after_input_eof"), row.get("contract_during_shutdown")
    for value in (after_eof, shutdown):
        if value is not None and not isinstance(value, bool):
            raise ValueError("EOF/shutdown instrumentation must be boolean or absent")
    return {"uid": str(row["uid"]), **features, "source": row["id.orig_h"],
            "destination": row["id.resp_h"], "native_protocol": row["proto"],
            "connection_start": ts, "zeek_reported_end": last, "interval_relation": relation,
            "duration_missing": duration < 0, "service": row.get("service"),
            "conn_state": row.get("conn_state"), "history": row.get("history"),
            "missed_bytes": number(row.get("missed_bytes"), integer=True),
            "replay_export_time": export, "after_input_eof": after_eof,
            "during_shutdown": shutdown, "detector_received_time": None,
            "feedback_available_time": None, "label": "unknown",
            "export_before_zeek_reported_end": bool(export is not None and last is not None and export < last)}


def load_actions(evidence, anchors):
    actions = []
    for role in evidence.get("controlled_normal_business", []):
        for action in role["actions"]:
            actions.append({"action_id": action["transition_id"], "kind": "normal_business",
                            "event_group": action["transition_id"], "actor_role": role["role"],
                            "client_ip": role["apache_observed_ip"],
                            "start": float(action["intent_timestamp"]), "end": float(action["success_timestamp"]),
                            "success_time": float(action["success_timestamp"]), "url": action["target_url"],
                            "evidence_member": role["member"], "evidence_line": action["intent_line"]})
    clients = list(evidence.get("apache", {}).get("command_client_addresses", {}))
    chain = anchors.get("independent_attack_chain_id", "russellmitchell-webshell-20220124")
    for command in evidence.get("apache", {}).get("commands", []):
        client = command.get("client") or command.get("client_ip")
        if not client:
            if len(clients) != 1:
                raise ValueError("Command client is not specified uniquely")
            client = clients[0]
        begin = epoch(command["time"])
        action = {"action_id": f'{command["member"]}:{command["line"]}', "kind": "control_request",
                  "event_group": chain, "client_ip": client, "start": begin, "end": begin + 1,
                  "success_time": None, "url": "", "evidence_member": command["member"],
                  "evidence_line": command["line"], "time_precision": "one-second Apache interval"}
        disambiguation = anchors.get("control_disambiguation", {})
        if command["line"] == disambiguation.get("apache_line"):
            precise = disambiguation["whoami_interval_from_author_config"]
            action.update(author_stage_start=epoch(precise["start"]), author_stage_end=epoch(precise["stop"]))
        actions.append(action)
    for download in anchors.get("downloads", []):
        actions.append({"action_id": "download:" + download["name"], "kind": "download_action",
                        "event_group": chain, "client_ip": anchors.get("server_ip", SERVER),
                        "start": float(download["start"]), "end": float(download["end"]),
                        "success_time": float(download["end"]), **download})
    return actions


def by_uid(rows, *, files=False):
    result = defaultdict(list)
    for row in rows:
        uids = row.get("conn_uids", [row.get("uid")]) if files else [row.get("uid")]
        if isinstance(uids, str):
            uids = [uids]
        for uid in uids:
            if uid:
                result[str(uid)].append(row)
    return result


def overlap(connection, action):
    return connection["connection_start"] < action["end"] and (
        connection["zeek_reported_end"] is None or connection["zeek_reported_end"] >= action["start"])


def http_matches_action(row, action):
    parsed = urlsplit(action.get("url", ""))
    if not parsed.hostname:
        return False
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    host = str(row.get("host", "")).lower().split(":")[0]
    request_time = optional_time(row.get("ts"))
    return (host == parsed.hostname.lower() and row.get("uri") == path
            and request_time is not None and action["start"] <= request_time <= action["end"])


def github_domain(host):
    host = str(host).lower().rstrip(".")
    return host in {"github.com", "githubusercontent.com"} or host.endswith((".github.com", ".githubusercontent.com"))


def association(connection, action, context, anchors):
    if connection["native_protocol"] != "tcp" or not overlap(connection, action):
        return None
    uid = connection["uid"]
    http, tls, file_rows = (context[k].get(uid, []) for k in ("http", "ssl", "files"))
    server = anchors.get("server_ip", SERVER)
    if action["kind"] != "download_action":
        port = 443 if urlsplit(action.get("url", "")).scheme == "https" else None
        allowed = {port} if port else {80, 443}
        if not (connection["source"] == action["client_ip"] and connection["destination"] == server
                and connection["resp_port"] in allowed):
            return None
        support = "time_endpoint_service_association"
    else:
        if connection["source"] != server:
            return None
        if (connection["destination"] == anchors.get("monitoring_ip", "10.143.0.35")
                and connection["resp_port"] == anchors.get("monitoring_port", 5044)):
            return None
        if action.get("target_ip"):
            if not (connection["destination"] == action["target_ip"]
                    and connection["resp_port"] == action.get("port", 80)):
                return None
            support = "explicit_actor_target_and_transfer_association"
        else:
            hostnames = [str(x.get("server_name", "")).lower() for x in tls]
            github_host = any(github_domain(h) for h in hostnames)
            dns_support = any(connection["destination"] in d.get("answers", [])
                              and github_domain(d.get("query", ""))
                              and optional_time(d.get("ts")) is not None
                              and action["start"] <= float(d["ts"]) <= action["end"] for d in context["dns_all"])
            if not (connection["resp_port"] == 443 and
                    (connection["destination"] in action.get("historical_candidate_ips", []) or github_host or dns_support)):
                return None
            support = "tls_download_candidate"
    matched_http = [h for h in http if http_matches_action(h, action)]
    if matched_http:
        support = "http_host_path_time_observed"
    file_hashes = [{key: row[key] for key in ("fuid", "md5", "sha1", "sha256") if row.get(key)}
                   for row in file_rows if any(row.get(k) for k in ("md5", "sha1", "sha256"))]
    prior_upload = anchors.get("control_disambiguation", {}).get("candidate_3544", {})
    known_command = anchors.get("control_disambiguation", {}).get("candidate_3545", {})
    same_upload_tuple = (action.get("author_stage_start") is not None and
                         connection["source"] == prior_upload.get("origin_ip") and
                         connection["orig_port"] == prior_upload.get("origin_port") and
                         connection["destination"] == prior_upload.get("responder_ip") and
                         connection["resp_port"] == prior_upload.get("responder_port"))
    same_command_tuple = (action.get("author_stage_start") is not None and
                          connection["source"] == known_command.get("origin_ip") and
                          connection["orig_port"] == known_command.get("origin_port") and
                          connection["destination"] == known_command.get("responder_ip") and
                          connection["resp_port"] == known_command.get("responder_port"))
    return {"uid": uid, "support": support, "duration_missing": connection["duration_missing"],
            "exact_http_request_count": len(matched_http), "http_status_codes": [h.get("status_code") for h in matched_http],
            "tls_server_names": sorted({str(t["server_name"]) for t in tls if t.get("server_name")}),
            "file_uids": sorted({str(f["fuid"]) for f in file_rows if f.get("fuid")}),
            "file_hash_observations": file_hashes, "file_hash_matches_expected": None,
            "tls_sni_is_file_identity": False, "association_is_pure_flow_truth": False,
            "matches_prior_upload_tuple": same_upload_tuple,
            "matches_known_command_tuple": same_command_tuple,
            "starts_in_author_command_stage": (action["author_stage_start"] <= connection["connection_start"]
                                                <= action["author_stage_end"])
            if action.get("author_stage_start") is not None else None}


def analyze(log_dir, evidence_path, output, variant, anchors_path=None):
    log_dir, evidence_path, output = Path(log_dir), Path(evidence_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if variant not in {"strict", "ignore_checksums"}:
        raise ValueError("Unknown extraction variant")
    evidence = read_json(evidence_path)
    anchors = read_json(anchors_path) if anchors_path else {}
    interval = (epoch(evidence["simulation_start"]), epoch(evidence["simulation_end_exclusive"]))
    logs = {name: records(log_dir / (name + ".log")) for name in ("conn", "http", "ssl", "dns", "files", "weird")}
    connections = [normalize_connection(row, interval) for row in logs["conn"]]
    if len({c["uid"] for c in connections}) != len(connections):
        raise ValueError("Duplicate conn UID cannot be joined without ambiguity")
    context = {k: by_uid(logs[k], files=k == "files") for k in ("http", "ssl", "files")}
    context["dns_all"] = logs["dns"]
    actions = load_actions(evidence, anchors)
    pairs, action_summary = [], []
    memberships = defaultdict(list)
    for action in actions:
        selected = []
        for connection in connections:
            match = association(connection, action, context, anchors)
            if match is not None:
                row = {**action, **match, "connection_start": connection["connection_start"],
                       "zeek_reported_end": connection["zeek_reported_end"],
                       "replay_export_time": connection["replay_export_time"],
                       "after_input_eof": connection["after_input_eof"], "during_shutdown": connection["during_shutdown"],
                       "detector_received_time": None, "feedback_available_time": None}
                comparable = action.get("success_time") is not None and connection["replay_export_time"] is not None
                row["success_at_or_before_replay_export"] = (
                    action["success_time"] <= connection["replay_export_time"] if comparable else None)
                row["success_minus_replay_export_seconds"] = (
                    action["success_time"] - connection["replay_export_time"] if comparable else None)
                selected.append(row)
                memberships[connection["uid"]].append(action)
        category = "zero" if not selected else "one" if len(selected) == 1 else "multiple"
        if not selected:
            pairs.append({**action, "uid": None, "match_class": category, "candidate_count": 0,
                          "support": "no_connection_candidate", "feedback_available_time": None})
        for row in selected:
            row.update(match_class=category, candidate_count=len(selected))
            pairs.append(row)
        action_summary.append({"action_id": action["action_id"], "kind": action["kind"],
                               "event_group": action["event_group"], "match_class": category,
                               "candidate_count": len(selected)})
    conflicts = set()
    for uid, members in memberships.items():
        kinds = {a["kind"] for a in members}
        if "normal_business" in kinds and len(kinds) > 1:
            conflicts.add(uid)
    for row in pairs:
        row["normal_attack_association_conflict"] = row.get("uid") in conflicts
    details, background, times = [], [], []
    for connection in connections:
        uid, members = connection["uid"], memberships.get(connection["uid"], [])
        http, tls, file_rows = (context[k].get(uid, []) for k in ("http", "ssl", "files"))
        detail = {"uid": uid, "association_status": "mixed_associations" if uid in conflicts else
                  "event_associated_unadjudicated" if members else "background_unknown",
                  "action_ids": [a["action_id"] for a in members],
                  "event_groups": sorted({a["event_group"] for a in members}),
                  "http_requests": [{k: h.get(k) for k in ("ts", "method", "host", "uri", "status_code", "response_body_len")}
                                    for h in http],
                  "tls_server_names": sorted({str(t["server_name"]) for t in tls if t.get("server_name")}),
                  "files": [{k: f.get(k) for k in ("fuid", "mime_type", "filename", "seen_bytes", "total_bytes", "missing_bytes", "md5", "sha1", "sha256")}
                            for f in file_rows], "normal_or_attack_truth": "unknown"}
        details.append(detail)
        if not members:
            background.append({**connection, **detail, "normal_file_download_confirmed": False})
        times.append({k: connection[k] for k in ("uid", "connection_start", "zeek_reported_end", "replay_export_time",
                     "after_input_eof", "during_shutdown", "detector_received_time", "feedback_available_time", "export_before_zeek_reported_end")})
    distribution = Counter((c["native_protocol"], c["destination"], c["resp_port"], c["interval_relation"]) for c in connections)
    services = [{"protocol": key[0], "destination": key[1], "destination_port": key[2],
                 "interval_relation": key[3], "connections": value, "truth": "unknown"}
                for key, value in sorted(distribution.items())]
    write_csv(output / "raw8.csv", connections, ("uid", *RAW8))
    write_csv(output / "action_connection_pairs.csv", pairs)
    write_csv(output / "connection_evidence.csv", details)
    write_csv(output / "service_distribution.csv", services)
    write_csv(output / "background_transfer_candidates.csv", sorted(background, key=lambda r:r["resp_ip_bytes"], reverse=True))
    write_csv(output / "time_availability.csv", times)
    dump(output / "protocol_evidence.json", {name: logs[name] for name in ("http", "ssl", "dns", "files", "weird")})
    instrumented = sum(c["replay_export_time"] is not None and c["after_input_eof"] is not None
                       and c["during_shutdown"] is not None for c in connections)
    counts = Counter(a["kind"] for a in actions)
    summary = {"status": "evidence_analysis_complete", "variant": variant, "connections": len(connections),
               "raw8": list(RAW8), "missing_value": -1,
               "log_rows": {name: len(rows) for name, rows in logs.items()},
               "log_files_present": {name: (log_dir/(name+".log")).exists() for name in logs},
               "input_sha256": {name: hashlib.sha256((log_dir/(name+".log")).read_bytes()).hexdigest()
                                for name in logs if (log_dir/(name+".log")).exists()},
               "action_counts": dict(counts), "action_results": action_summary,
               "conflicting_normal_attack_associations": sorted(conflicts),
               "interval_relations": dict(Counter(c["interval_relation"] for c in connections)),
               "missing_fields": {f: sum(c[f] < 0 for c in connections) for f in RAW8},
               "clock": {"instrumented_connections": instrumented,
                         "eof_connections": sum(c["after_input_eof"] is True for c in connections),
                         "shutdown_connections": sum(c["during_shutdown"] is True for c in connections),
                         "export_before_zeek_reported_end": sum(c["export_before_zeek_reported_end"] for c in connections),
                         "interpretation": "offline network-time export instrumentation, not actual detector receipt"},
               "conditions": {
                   "connection_records": "met" if connections else "unmet",
                   "export_clock_instrumentation": "met" if connections and instrumented == len(connections) else "unresolved",
                   "same_service_normal_event_candidates": "observed" if any(p["kind"] == "normal_business" and p.get("uid") for p in pairs) else "unresolved",
                   "pure_normal_evaluation_denominator": "unresolved_requires_business_adjudication",
                   "normal_file_download_business_provenance": "unresolved_not_supplied",
                   "online_feedback_arrival": "unresolved_not_supplied",
                   "http_download_request_observed": "observed" if any(p["kind"] == "download_action" and p.get("exact_http_request_count", 0) for p in pairs) else "unresolved",
               }, "new_fits": 0, "model_forward_calls": 0, "fpr_computed": False,
               "truth_policy": "All connections remain unknown; event associations and visible protocol evidence do not automatically create pure labels."}
    dump(output / "evidence_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["strict", "ignore_checksums"], required=True)
    parser.add_argument("--anchors", type=Path)
    args = parser.parse_args()
    summary = analyze(args.log_dir, args.evidence, args.output, args.variant, args.anchors)
    print(json.dumps({k: summary[k] for k in ("status", "variant", "connections", "action_counts", "conditions")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
