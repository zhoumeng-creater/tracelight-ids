"""Synthetic JSON-line contracts and real frozen metadata; no PCAP/model/network work."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ait_contract_analysis", REPO / "scripts/ait_contract/analyze.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def conn(uid="c1", ts=10., duration=5., source="10.143.2.91", destination=mod.SERVER, port=443, **extra):
    return {"uid": uid, "ts": ts, "duration": duration, "id.orig_h": source, "id.resp_h": destination,
            "id.orig_p": 50000, "id.resp_p": port, "proto": "tcp", "orig_pkts": 3, "resp_pkts": 8,
            "orig_ip_bytes": 160, "resp_ip_bytes": 900, "orig_bytes": 7, "resp_bytes": 21,
            "history": "ShADadFf", "contract_export_network_time": ts + (duration or 0) + 60,
            "contract_after_input_eof": False, "contract_during_shutdown": False, **extra}


def normal(action_id="normal1", begin=10., end=12., client="10.143.2.91"):
    return {"role": "remote_employee_1", "apache_observed_ip": client, "member": "normal/sm.log",
            "actions": [{"transition_id": action_id, "intent_timestamp": begin, "success_timestamp": end,
                         "intent_line": 1, "target_url": "https://intranet.example/?p=5"}]}


def fixture(tmp_path, connections=(), normals=(), commands=(), downloads=(), extra_logs=None):
    evidence = {"simulation_start": "1970-01-01T00:00:00", "simulation_end_exclusive": "1970-01-01T00:01:40",
                "controlled_normal_business": list(normals),
                "apache": {"command_client_addresses": {"10.143.2.91": len(commands)}, "commands": list(commands)}}
    anchors = {"server_ip": mod.SERVER, "monitoring_ip": "10.143.0.35", "monitoring_port": 5044,
               "independent_attack_chain_id": "one-attack-chain", "downloads": list(downloads)}
    ep, ap, logs, output = tmp_path / "evidence.json", tmp_path / "anchors.json", tmp_path / "logs", tmp_path / "out"
    logs.mkdir()
    ep.write_text(json.dumps(evidence), encoding="utf-8")
    ap.write_text(json.dumps(anchors), encoding="utf-8")
    for name, rows in {"conn": connections, **(extra_logs or {})}.items():
        (logs / (name + ".log")).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = mod.analyze(logs, ep, output, "strict", ap)
    return result, output


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_raw8_preserves_missingness_uses_ip_bytes_and_separates_export_clock(tmp_path):
    row = conn(duration=None, resp_pkts=None, resp_ip_bytes=None, contract_export_network_time=99.,
               contract_after_input_eof=True, contract_during_shutdown=True)
    result, output = fixture(tmp_path, [row])
    parsed = csv_rows(output / "raw8.csv")[0]
    assert parsed["duration_seconds"] == parsed["resp_pkts"] == parsed["resp_ip_bytes"] == "-1"
    assert parsed["orig_ip_bytes"] == "160"  # Never the orig_bytes payload field 7.
    assert parsed["zeek_reported_end"] == ""
    assert parsed["replay_export_time"] == "99.0"
    assert parsed["detector_received_time"] == parsed["feedback_available_time"] == ""
    assert result["clock"]["eof_connections"] == result["clock"]["shutdown_connections"] == 1
    assert result["conditions"]["export_clock_instrumentation"] == "met"


def test_missing_instrumentation_stays_unknown_not_ts_plus_duration(tmp_path):
    row = conn()
    for key in list(row):
        if key.startswith("contract_"):
            del row[key]
    result, output = fixture(tmp_path, [row])
    parsed = csv_rows(output / "raw8.csv")[0]
    assert parsed["zeek_reported_end"] == "15.0"
    assert parsed["replay_export_time"] == "" and parsed["after_input_eof"] == ""
    assert result["conditions"]["export_clock_instrumentation"] == "unresolved"


def test_multi_connection_normal_groups_are_all_preserved_and_unmatched_not_dropped(tmp_path):
    rows = [conn(uid="n" + str(i), ts=10 + i * .01) for i in range(6)]
    rows.append(conn(uid="single", ts=20.))
    result, output = fixture(tmp_path, rows, [normal(), normal("single", 20, 21), normal("zero", 30, 31)])
    assert [r["candidate_count"] for r in result["action_results"]] == [6, 1, 0]
    pairs = csv_rows(output / "action_connection_pairs.csv")
    assert len(pairs) == 8 and sum(p["match_class"] == "multiple" for p in pairs) == 6
    assert all(p["success_at_or_before_replay_export"] == "True" for p in pairs if p["uid"])
    assert all(p["feedback_available_time"] == "" for p in pairs)
    assert all(r["label"] == "unknown" for r in csv_rows(output / "raw8.csv"))
    assert len({p["event_group"] for p in pairs}) == 3


def test_normal_control_shared_connection_is_reported_not_overwritten(tmp_path):
    command = {"member": "apache/access.log", "line": 1, "time": "1970-01-01T00:00:10+00:00"}
    result, output = fixture(tmp_path, [conn()], [normal()], [command])
    assert result["conflicting_normal_attack_associations"] == ["c1"]
    assert all(p["normal_attack_association_conflict"] == "True"
               for p in csv_rows(output / "action_connection_pairs.csv"))
    detail = csv_rows(output / "connection_evidence.csv")[0]
    assert detail["association_status"] == "mixed_associations" and detail["normal_or_attack_truth"] == "unknown"


def test_explicit_http_download_is_stronger_than_time_overlap_but_not_file_hash_proof(tmp_path):
    download = {"name": "password list", "start": 10., "end": 12.,
                "url": "http://192.168.230.122/rockyou.txt", "target_ip": "192.168.230.122", "port": 80}
    rows = [conn(uid="download", source=mod.SERVER, destination="192.168.230.122", port=80),
            conn(uid="monitoring", source=mod.SERVER, destination="10.143.0.35", port=5044)]
    http = {"uid": "download", "ts": 10.4, "host": "192.168.230.122", "uri": "/rockyou.txt", "status_code": 200}
    files = [{"uid": "download", "fuid": "Fone", "mime_type": "text/plain", "seen_bytes": 42}]
    result, output = fixture(tmp_path, rows, downloads=[download], extra_logs={"http": [http], "files": files})
    pairs = csv_rows(output / "action_connection_pairs.csv")
    assert len(pairs) == 1 and pairs[0]["uid"] == "download"
    assert pairs[0]["support"] == "http_host_path_time_observed"
    assert pairs[0]["association_is_pure_flow_truth"] == "False"
    assert pairs[0]["file_hash_matches_expected"] == ""
    assert json.loads(pairs[0]["file_uids"]) == ["Fone"]  # Zeek 8.0.11 uses singular uid.
    assert result["conditions"]["http_download_request_observed"] == "observed"
    assert csv_rows(output / "background_transfer_candidates.csv")[0]["uid"] == "monitoring"


def test_tls_sni_and_dns_only_support_candidates_not_file_identity(tmp_path):
    download = {"name": "WPHashCrack", "start": 10., "end": 12.,
                "url": "https://github.com/ait-aecid/wphashcrack/archive/refs/tags/v0.1.tar.gz",
                "historical_candidate_ips": []}
    rows = [conn(uid="valid", source=mod.SERVER, destination="1.2.3.4"),
            conn(uid="lookalike", source=mod.SERVER, destination="1.2.3.5")]
    tls = [{"uid": "valid", "server_name": "codeload.github.com"}]
    dns = [{"ts": 10., "query": "evilgithub.com", "answers": ["1.2.3.5"]}]
    _, output = fixture(tmp_path, rows, downloads=[download], extra_logs={"ssl": tls, "dns": dns})
    pairs = csv_rows(output / "action_connection_pairs.csv")
    assert [p["uid"] for p in pairs] == ["valid"]
    assert pairs[0]["support"] == "tls_download_candidate" and pairs[0]["tls_sni_is_file_identity"] == "False"


def test_eof_and_boundary_crossing_connections_remain_in_output(tmp_path):
    rows = [conn(uid="before", ts=0, duration=1), conn(uid="cross", ts=99, duration=9,
            contract_after_input_eof=True, contract_during_shutdown=True), conn(uid="after", ts=105)]
    result, output = fixture(tmp_path, rows)
    assert result["connections"] == 3 and result["interval_relations"] == {"inside": 1, "crosses_end": 1, "after": 1}
    assert len(csv_rows(output / "raw8.csv")) == 3
    assert result["clock"]["eof_connections"] == 1


def test_duplicate_uids_and_invalid_numeric_fields_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate conn UID"):
        fixture(tmp_path, [conn(), conn()])
    with pytest.raises(ValueError, match="Invalid nonnegative"):
        mod.normalize_connection(conn(resp_ip_bytes=-5), (0, 100))
    with pytest.raises(ValueError, match="Invalid nonnegative"):
        mod.normalize_connection(conn(orig_pkts=1.5), (0, 100))


def test_empty_strict_extraction_is_reported_as_unmet_not_effect_failure(tmp_path):
    result, output = fixture(tmp_path)
    assert result["connections"] == 0 and result["conditions"]["connection_records"] == "unmet"
    assert result["new_fits"] == result["model_forward_calls"] == 0 and not result["fpr_computed"]
    assert (output / "raw8.csv").read_text(encoding="utf-8-sig").startswith("uid,ip_protocol")


def test_real_registered_metadata_schema_yields_all_102_actions_without_network(tmp_path):
    evidence = REPO / "docs/research_alignment_20261115/risk_transfer/source/ait-small-evidence.json"
    anchors = REPO / "configs/ait-flow-contract-20261118/anchors.json"
    normal_action = json.loads(evidence.read_text(encoding="utf-8"))["controlled_normal_business"][0]["actions"][0]
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "conn.log").write_text(json.dumps(conn(ts=normal_action["intent_timestamp"] + .01)) + "\n", encoding="utf-8")
    result = mod.analyze(logs, evidence, tmp_path / "out", "ignore_checksums", anchors)
    assert result["action_counts"] == {"normal_business": 68, "control_request": 32, "download_action": 2}
    assert len(result["action_results"]) == 102
    assert result["action_results"][0]["candidate_count"] == 1
    attack_groups = {a["event_group"] for a in result["action_results"] if a["kind"] != "normal_business"}
    assert attack_groups == {"russellmitchell-webshell-20220124"}
