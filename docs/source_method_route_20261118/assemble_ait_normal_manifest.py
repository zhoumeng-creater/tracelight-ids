"""Rebuild AIT action/server/proxy evidence from the six downloaded small logs.

Offline only: no PCAP reads, network requests, model calls, or threshold selection.
Requires Python 3 and PyYAML. Paths can be overridden for a relocated checkout.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import parse_qsl, unquote, urlsplit

import yaml


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent / "代码" / "tracelight-ids"
HASHES = {
    "gather/internal_employee_2/logs/sm.log": "18d5d3e7425bb7a2f92fa353960ab7487a9e1b09ed77fc35e0e3f294e9e0cc2c",
    "gather/internal_employee_3/logs/sm.log": "5c2ff0fc8f338490a375f19319c9c1bc0e75b7bbe6c9e3db4665fc9eb7045800",
    "gather/cloud_share/logs/apache2/other_vhosts_access.log.1": "dd84f0feb1700360023d7d821aa5b2dd2602b0890f18b536ea80f9e3a414b134",
    "gather/cloud_share/logs/apache2/other_vhosts_access.log.4": "0c140f45b8367d20f876ed853252fa41e8908f8c8505b7339c1b2ee5e8d9b987",
    "gather/webserver/logs/apache2/cloud.smith.russellmitchell.com-access.log.1": "b87709fc6b7385ae12324688892f5a7ded0f9967951e4297862c9b8b0d008294",
    "gather/webserver/logs/apache2/cloud.smith.russellmitchell.com-access.log.4": "34fa3568b528da7b8387c48b3fcfadddb38b2c1b6ea2505e18a6b9f92208fef9",
}
ACCESS = re.compile(r'\[([^\]]+)\] "(\w+) ([^ ]+) ([^\"]+)" (\d+) (\d+)')
START, END = 1642723200, 1643068800


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_access(line, member, number):
    m = ACCESS.search(line)
    if m is None:
        return None
    date, method, target, version, status, byte_count = m.groups()
    url = urlsplit(target)
    key = (method, unquote(url.path), tuple(sorted(parse_qsl(url.query, keep_blank_values=True))), version, int(status))
    prefix = line[:m.start()].split()
    return {
        "key": key, "member": member, "line": number,
        "line_sha256": sha256(line.encode("utf-8")),
        "timestamp": datetime.strptime(date, "%d/%b/%Y:%H:%M:%S %z").timestamp(),
        "client_ip": prefix[1] if member.startswith("gather/cloud_share/") else prefix[0],
        "virtual_host": prefix[0] if member.startswith("gather/cloud_share/") else None,
        "method": method, "path": unquote(url.path), "http_version": version,
        "status": int(status), "logged_bytes": int(byte_count),
        "request_identity_sha256": sha256(json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
    }


def without_key(record):
    return {k: v for k, v in record.items() if k != "key"}


def cardinality(n):
    return "missing" if n == 0 else "unique" if n == 1 else "ambiguous"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "ait_small_sources")
    parser.add_argument("--servers", type=Path, default=PROJECT / "data/raw/ait-lds/source-audit-20261115/processing/config/servers.yaml")
    parser.add_argument("--contract", type=Path, default=PROJECT / ".worktrees/ait-flow-contract-20261118/configs/ait-flow-contract-20261118/contract.json")
    parser.add_argument("--output", type=Path, default=ROOT / "normal_event_manifest.json")
    args = parser.parse_args()
    inputs, lines = [], {}
    for member, expected in HASHES.items():
        path = args.sources / member
        raw = path.read_bytes()
        actual = sha256(raw)
        assert actual == expected, (member, actual)
        inputs.append({"path": str(path.resolve()), "member": member, "sha256": actual, "bytes": len(raw)})
        lines[member] = raw.decode("utf-8").splitlines()

    prior_path = args.sources / "normal_download_action_server_matches.json"
    for path in [prior_path, args.servers, args.contract, Path(__file__)]:
        raw = path.read_bytes()
        inputs.append({"path": str(path.resolve()), "sha256": sha256(raw), "bytes": len(raw)})
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    servers = yaml.safe_load(args.servers.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    proxy_ip = servers["webserver"]["default_ipv4_address"]
    cloud_ip = servers["cloud_share"]["default_ipv4_address"]
    intranet_ip = servers["intranet_server"]["default_ipv4_address"]

    proxy_index = defaultdict(list)
    for member, contents in lines.items():
        if member.startswith("gather/webserver/"):
            for number, line in enumerate(contents, 1):
                record = parse_access(line, member, number)
                if record is not None and START <= record["timestamp"] < END:
                    proxy_index[record["key"]].append(record)

    events, used_actions = [], []
    for old in prior["matches"]:
        server = old["server"]
        member = "gather/cloud_share/logs/apache2/" + server["file"]
        backend = parse_access(lines[member][server["line"] - 1], member, server["line"])
        assert backend is not None
        assert backend["timestamp"] == server["ts"] and backend["path"] == server["path"]
        assert backend["status"] == server["status"] and backend["logged_bytes"] == server["logged_bytes"]
        assert backend["client_ip"] == proxy_ip
        assert backend["virtual_host"] == "cloud.dmz.smith.russellmitchell.com:443"
        query = defaultdict(list)
        for key, value in backend["key"][2]:
            query[key].append(value)
        assert dict(query) == server["query"]
        candidates = []
        for proxy in proxy_index[backend["key"]]:
            candidates.append({
                **without_key(proxy),
                "proxy_minus_backend_seconds": proxy["timestamp"] - backend["timestamp"],
                "proxy_minus_backend_logged_bytes": proxy["logged_bytes"] - backend["logged_bytes"],
            })
        actions = []
        for old_action in old["matches"]:
            action = dict(old_action)
            action_member = f'gather/{action["host"]}/logs/sm.log'
            start = json.loads(lines[action_member][action["start_line"] - 1])
            success = json.loads(lines[action_member][action["success_line"] - 1])
            assert start["transition_id"] == success["transition_id"] == action["transition_id"]
            assert success["message"] == "Downloaded file" and success["file"] == action["file"]
            assert start["timestamp"] == action["start"] and success["timestamp"] == action["success"]
            assert math.floor(action["start"]) <= backend["timestamp"] <= action["success"]
            artifact = action["file"]
            if artifact["file_type"] == "dir":
                assert backend["path"] == "/index.php/apps/files/ajax/download.php"
                assert query["files"] == [artifact["name"]] and query["dir"] == [artifact["path"]]
            else:
                expected = "/remote.php/webdav" + artifact["path"].rstrip("/") + "/" + artifact["name"]
                assert backend["path"] == expected
            action["member"] = action_member
            action["configured_client_ip"] = servers[action["host"]]["default_ipv4_address"]
            action["configured_username"] = servers[action["host"]]["username"]
            actions.append(action)
        identity_ok = len(actions) == len(candidates) == 1 and candidates[0]["client_ip"] == actions[0]["configured_client_ip"]
        aligned = identity_ok and candidates[0]["proxy_minus_backend_seconds"] == 0
        if aligned:
            used_actions.append(actions[0]["transition_id"])
        events.append({
            "event_id": f'russellmitchell:{server["file"]}:{server["line"]}',
            "business_label": "normal_file_download" if aligned else None,
            "status": "action_server_proxy_confirmed" if aligned else "unresolved",
            "prior_layer": old,
            "action_candidates": actions, "action_match_status": cardinality(len(actions)),
            "backend": without_key(backend), "proxy_candidates": candidates,
            "proxy_match_status": cardinality(len(candidates)),
            "action_proxy_client_identity_agrees": identity_ok,
            "packet_flow_association_status": "not_associated",
            "flow_label": None,
            "network_legs": {
                "client_to_proxy": {
                    "orig_ip_candidates": sorted({c["client_ip"] for c in candidates}),
                    "orig_port": None, "resp_host": "webserver",
                    "configured_resp_ip": proxy_ip, "wire_resp_ip": None, "resp_port": None,
                    "endpoint_basis": "client IP observed by proxy; proxy address from host config; DNS/DNAT and listener port not established by these access logs",
                },
                "proxy_to_cloud": {
                    "orig_ip": backend["client_ip"], "orig_port": None,
                    "configured_resp_ip": cloud_ip, "resp_port": 443,
                    "endpoint_basis": "backend access client IP plus cloud host config and :443 virtual host; packet tuple still unverified",
                },
            },
            "time_contract": {
                "business_success_timestamp": actions[0]["success"] if len(actions) == 1 else None,
                "flow_completion_timestamp": None, "feedback_available_timestamp": None,
            },
        })
    assert len(events) == 36 and len(used_actions) == len(set(used_actions)) == 31
    confirmed = [e for e in events if e["status"] == "action_server_proxy_confirmed"]
    summary = {
        "server_events": len(events), "prior_successful_actions_all_four_days": len(prior["actions"]),
        "confirmed_normal_events": len(confirmed), "unresolved_events": len(events) - len(confirmed),
        "proxy_match_counts": dict(Counter(e["proxy_match_status"] for e in events)),
        "action_match_counts": dict(Counter(e["action_match_status"] for e in events)),
        "confirmed_by_client_ip": dict(Counter(e["proxy_candidates"][0]["client_ip"] for e in confirmed)),
        "unresolved_proxy_client_ips": sorted({c["client_ip"] for e in events if e["status"] == "unresolved" for c in e["proxy_candidates"]}),
        "confirmed_same_second_proxy_backend": sum(e["proxy_candidates"][0]["proxy_minus_backend_seconds"] == 0 for e in confirmed),
        "packet_associated_events": 0,
    }
    output = {
        "version": "ait-normal-events-v1", "scenario": "russellmitchell",
        "role": "opened_development_source_evidence", "interval": {"start": START, "end_exclusive": END},
        "matching_rule": "all exact canonical method/path/query/HTTP-version/status matches across both proxy files; no candidate removed by client, time, or bytes; unique candidates then checked for same second and expected actor IP",
        "query_policy": "query token values participate in exact matching; original prior-layer events are preserved; additional proxy records store request identity hashes and original line references",
        "log_byte_policy": "backend and proxy counters are retained separately and are not file size, packet bytes, or an identity criterion",
        "inputs": inputs, "summary": summary,
        "current_pcap": {
            **contract["pcap"], "configured_host_ip": intranet_ip,
            "membership_status": "not_established", "global_mirror_evidence": None,
            "normal_chain_contains_configured_intranet_ip": False,
            "basis": "confirmed client -> proxy -> cloud host addresses exclude the intranet server; this does not prove that an interface/mirror capture cannot observe them",
            "next_join": "use already extracted intranet connection endpoints and times to retain every matching candidate; do not assign flow truth or choose a PCAP arm from detection scores",
        },
        "prior_successful_actions_all_four_days": prior["actions"], "events": events,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
