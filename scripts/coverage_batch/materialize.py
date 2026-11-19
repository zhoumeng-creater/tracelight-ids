"""Freeze the single-flow source-coverage x port-input batch; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("coverage_batch_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
RAW8_FIELDS = (
    "ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
)
CONTROL_KINDS = {"webshell_command", "reverse_shell_channel", "known_control_transport"}
DOWNLOAD_KINDS = {"attack_download", "malicious_tool_download"}
TARGET_HOST = "10.1.3.8"


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_rows(path, rows):
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")


def materialize(work_root, output):
    now = work_root / "覆盖与历史依据落实_20261119"
    old = work_root / "拟合来源与划分落实_20261119/dataset"
    unified = now / "unified"
    normal = now / "normal_calibration/v2"
    management = now / "normal_management"
    attack = now / "attack_coverage/full_capture"
    used = set()

    def load(path):
        used.add(path)
        return read_rows(path)

    def load_json(path):
        used.add(path)
        return json.loads(path.read_text(encoding="utf-8"))

    unified_summary = load_json(unified / "summary.json")
    if unified_summary["attack_intake"]["state"] != "full_capture_ingested":
        raise ValueError("Merge the frozen complete Unraveled capture into unified tables first")
    full_receipt = load_json(attack / "assembly_receipt.json")
    current = {r["flow_id"]: r for r in load(unified / "flow_rows.jsonl.gz")}
    old_inference = load(old / "inference_rows.jsonl.gz")
    old_models = load(old / "model_rows.jsonl.gz")
    old_events = load(old / "event_metadata.jsonl.gz")
    old_units = load(old / "normal_units.jsonl.gz")
    c_candidates = load(normal / "C_candidates.jsonl.gz")
    c_service_candidates = load(normal / "connection_candidates.jsonl.gz")
    c_downloads = load_json(normal / "file_business_evidence.json")["requests"]
    ssh_candidates = load(management / "complete_control_candidates.jsonl.gz")
    ssh_events = {r["event_id"]: r for r in load(management / "events.jsonl.gz")}
    full_rows = load(attack / "connections.jsonl.gz")
    full_events = load(attack / "events.jsonl")
    full_normal_strata = load_json(attack / "target_host_normal_business_strata.json")

    base_fit = {r["flow_id"]: r for r in old_models if r["dataset_role"] == "F"}
    if len(base_fit) != 471 or any(r["scenario"] != "santos" for r in base_fit.values()):
        raise ValueError("F_base must be the frozen 471 santos model_rows identities")
    c_ids = {r["flow_id"] for r in c_candidates}
    if len(c_ids) != 3475 or any(r["label"] != 0 or not r["calibration_candidate"] for r in c_candidates):
        raise ValueError("Expected the complete 3475-row russell normal candidate cohort")
    if len(ssh_candidates) != 1:
        raise ValueError("Expected the one evidenced complete fox SSH normal candidate")

    flow_event_kinds = defaultdict(set)
    for event in old_events:
        for fid in event["candidate_flow_ids"]:
            if event["label"] == 1:
                flow_event_kinds[fid].add(event["kind"])

    def old_target_kind(fid, label):
        if label == 0:
            return "normal"
        if label is None:
            return "unknown"
        kinds = flow_event_kinds[fid]
        targets = ({"control"} if kinds & CONTROL_KINDS else set()) | ({"download"} if kinds & DOWNLOAD_KINDS else set())
        if len(targets) != 1:
            raise ValueError(f"Positive old flow lacks an unambiguous control/download kind: {fid}")
        return next(iter(targets))

    result = {}

    def make_row(fid, role, label, target_kind, features=None):
        observation = current[fid]
        if observation["current_role"] == "reserved_historical_confirmation":
            raise ValueError("Reserved material cannot enter the scoring universe")
        row = {key: observation.get(key) for key in (
            "flow_id", "stable_flow_id", "raw8", "source_family", "source_run", "source_group",
            "campaign_id", "connection_start", "observed_end", "replay_export_time", "actual_available_at", "clock_basis",
        )}
        row.update(dataset_role=role, label=label, target_label=label, target_kind=target_kind,
                   features8=observation["features8"] if features is None else features,
                   allocation_group=observation["source_group"],
                   evidence_role_before_batch=observation["current_role"], historical_roles=observation["historical_roles"])
        if not row["allocation_group"]:
            raise ValueError(f"Missing complete-source allocation group: {fid}")
        if fid in result:
            if result[fid]["stable_flow_id"] != row["stable_flow_id"]:
                raise ValueError("Conflicting stable identity")
            raise ValueError(f"Duplicate score identity: {fid}")
        result[fid] = row
        return row

    role_map = {"F": "F_base", "V": "V", "external_development": "external_development"}
    for original in old_inference:
        if original["dataset_role"] == "C":
            continue  # Reinsert once from the complete current C cohort below.
        role = role_map[original["dataset_role"]]
        fid = original["flow_id"]
        row = make_row(fid, role, original["label"], old_target_kind(fid, original["label"]), original["features8"])
        row["source_label_basis"] = original.get("label_basis")
        row["event_ids"] = original.get("event_ids", [])

    for original in c_candidates:
        fid = original["flow_id"]
        row = make_row(fid, "C", 0, "normal")
        row["source_label_basis"] = original["label_basis"]
        row["business_stratum"] = original["business_stratum"]
        row["capture_host"] = original["capture_host"]
        row["input_quality"] = original["input_quality"]

    # Preserve all registered request candidates, including the 12 unknown connections.
    request_candidates = defaultdict(set)
    for candidate in c_service_candidates:
        for request_id in candidate["file_transfer_request_ids"]:
            request_candidates[request_id].add(candidate["flow_id"])
    c_background = set()
    for request in c_downloads:
        for fid in request_candidates[request["service_request_id"]] - c_ids:
            c_background.add(fid)
    for fid in sorted(c_background):
        if current[fid]["label"] is not None:
            raise ValueError("C event background must remain unknown, not a hidden normal denominator")
        make_row(fid, "C_background", None, "unknown")

    ssh = ssh_candidates[0]
    ssh_id = ssh["flow_id"]
    if not ssh_id.startswith("ait:fox:") or ssh["normal_management_candidate_label"] != 0:
        raise ValueError("The additional SSH normal must remain in the fox source")
    ssh_row = make_row(ssh_id, "V", 0, "normal")
    ssh_row["source_label_basis"] = "original normal actor direct complete SSH session"
    ssh_row["business_stratum"] = "normal_SSH_management"

    full_target_rows = []
    expansion_ids = set()
    for original in full_rows:
        endpoints = {original.get("source_address", original.get("source")), original.get("destination_address", original.get("destination"))}
        if TARGET_HOST not in endpoints:
            continue
        full_target_rows.append(original)
        kind = original["target_kind"]
        target_label = original["target_label"]
        kind = "control" if kind == "c2" else kind
        if kind not in {"normal", "control", "other_attack", "unknown"}:
            raise ValueError(f"Unsupported full-source target kind: {kind}")
        expected = 0 if kind == "normal" else 1 if kind == "control" else None
        if target_label != expected:
            raise ValueError("Other attacks or unknown rows cannot become C2 or negatives")
        fid = original["flow_id"]
        row = make_row(fid, "F_expansion", target_label, kind)
        row.update({k: original.get(k) for k in (
            "author_attack_label", "author_activity", "author_stage", "author_signature",
            "author_annotation", "author_flow_candidate_ids", "label_basis", "service", "native_protocol",
        )})
        row["source_input_quality"] = {k: original[k] for k in (
            "partial_capture", "full_connection_observed", "input_quality", "quality_flags",
            "raw8_missing_fields", "after_input_eof", "during_shutdown", "measurement_contract",
            "label_provenance_limits", "completeness_evidence", "raw8_observation_scope", "partial_capture_basis",
        ) if k in original}
        row["source_fact_label"] = original.get("label")
        if row["allocation_group"] != "unraveled:apt_2021_single_campaign":
            raise ValueError("Full Unraveled campaign must have one allocation identity")
        if target_label in (0, 1):
            expansion_ids.add(fid)

    cohorts = {
        "F_base": sorted(base_fit),
        "F_expanded": sorted(set(base_fit) | expansion_ids),
        "C": sorted(c_ids),
        "V": sorted(fid for fid, r in result.items() if r["dataset_role"] == "V"),
        "external_development": sorted(fid for fid, r in result.items() if r["dataset_role"] == "external_development"),
    }
    for fid, original in base_fit.items():
        if result[fid]["label"] != original["label"] or result[fid]["features8"] != original["features8"]:
            raise ValueError("Frozen base fit features/labels changed")
    for fid, row in result.items():
        row["fit_base_member"] = fid in base_fit
        row["fit_expanded_member"] = fid in base_fit or fid in expansion_ids

    events = []
    for original in old_events:
        event = copy.deepcopy(original)
        event["historical_role"] = event["role"]
        event["role"] = role_map[event["role"]]
        event["candidate_metadata_unchanged"] = True
        events.append(event)

    for request in c_downloads:
        actions = request["exact_object_and_time_actor_completion_candidates"]
        if len(actions) != 1:
            raise ValueError("C download event requires its source-verified unique actual action")
        action = actions[0]
        ids = sorted(request_candidates[request["service_request_id"]])
        events.append({
            "event_id": action["action_id"], "label": 0, "kind": "normal_file_download", "role": "C",
            "chain_id": "ait:russellmitchell:complete_run", "unit": "actual_actor_download_action",
            "candidate_flow_ids": ids, "association_mode": "one_of", "candidate_set_complete": False,
            "candidate_completeness_basis": "Source retains all registered interval/request candidates but does not assert an exhaustive physical assignment for the action.",
            "service_request_id": request["service_request_id"], "actor_completion_evidence": action,
            "normal_candidate_ids": [f for f in ids if f in c_ids],
            "unknown_background_candidate_ids": [f for f in ids if f in c_background],
            "in_calibration_source": True,
        })

    for sid in ssh["complete_normal_control_candidate_session_ids"]:
        source = ssh_events[sid]
        events.append({"event_id": sid, "label": 0, "kind": "normal_ssh_session", "role": "V",
            "chain_id": "ait:fox:complete_run", "unit": "actual_actor_SSH_session",
            "candidate_flow_ids": source["candidate_flow_ids"], "association_mode": "one_of",
            "candidate_set_complete": False,
            "candidate_completeness_basis": "Unique source association is preserved; no new event-wide completeness claim is added.",
            "source_event_reference": {"path": str(management / "events.jsonl.gz"), "event_id": sid}})

    outside_target_events = []
    for original in full_events:
        ids = original.get("candidate_flow_ids", [])
        if not any(fid in result for fid in ids):
            outside_target_events.append(original)
            continue
        if any(fid not in result for fid in ids):
            raise ValueError("Full-source event crosses the fixed host scoring universe; do not truncate it")
        event = copy.deepcopy(original)
        event["role"] = "F_expansion"
        event["label"] = original.get("target_label")
        event["in_sample_for_expanded_arms"] = True
        event.setdefault("association_mode", None)
        event.setdefault("candidate_set_complete", False)
        events.append(event)

    units = []
    for original in old_units:
        if original["role"] == "C":
            continue
        unit = copy.deepcopy(original)
        unit["historical_role"] = unit["role"]
        unit["role"] = role_map[unit["role"]]
        unit["in_sample_for_arms"] = list(runner.ARMS) if unit["role"] == "F_base" else []
        units.append(unit)

    def unit(uid, role, ids, **metadata):
        ids = sorted(set(ids))
        if ids:
            units.append({"unit_id": uid, "role": role, "flow_ids": ids,
                "normal_count": len(ids), "one_percent_integer_budget": len(ids) // 100, **metadata})

    for host in sorted({r["capture_host"] for r in c_candidates}):
        unit(f"C:host:{host}", "C", [r["flow_id"] for r in c_candidates if r["capture_host"] == host],
             view="host", conditional_normal_denominator=True, calibration_source=True)
    for stratum in sorted({r["business_stratum"] for r in c_candidates}):
        unit(f"C:business:{stratum}", "C", [r["flow_id"] for r in c_candidates if r["business_stratum"] == stratum],
             view="business", conditional_normal_denominator=True, calibration_source=True,
             overlaps_host_view=True)
    unit("V:intranet_server:normal_SSH_management", "V", [ssh_id],
         denominator_scope="one verified complete normal management session", in_sample_for_arms=[])
    for stratum, source_unit in sorted(full_normal_strata["strata"].items()):
        unit(f"Unraveled:normal:{stratum}", "F_expansion", source_unit["accepted_normal_flow_ids"],
             business_protocol_stratum=stratum,
             author_explicit_normal_rows=source_unit["author_explicit_normal_rows"],
             source_grade=full_normal_strata["source_grade"],
             denominator_scope="Source-assembled explicit Normal application/protocol stratum at host 10.1.3.8; observations are not independent actor actions or a label for unlabelled background.",
             strata_can_share_connections=full_normal_strata["strata_can_share_connections"],
             in_sample_for_arms=[name for name, (cohort, _) in runner.ARMS.items() if cohort == "F_expanded"],
             independent_confirmation=False)

    output.mkdir(parents=True, exist_ok=True)
    sorted_rows = [result[fid] for fid in sorted(result)]
    save_rows(output / "rows.jsonl.gz", sorted_rows)
    save_json(output / "cohorts.json", cohorts)
    save_rows(output / "events.jsonl.gz", events)
    save_rows(output / "normal_units.jsonl.gz", units)
    save_rows(output / "source_events_outside_target_host.jsonl.gz", outside_target_events)
    inputs = {key: {"path": filename, "sha256": digest(output / filename)} for key, filename in (
        ("rows", "rows.jsonl.gz"), ("cohorts", "cohorts.json"), ("events", "events.jsonl.gz"), ("normal_units", "normal_units.jsonl.gz"),
    )}
    registration = {
        "schema": "source-coverage-port-batch-20261119-v1", "execution_enabled": False,
        "threads": 4, "parameters": runner.PARAMETERS,
        "arms": {name: list(value) for name, value in runner.ARMS.items()},
        "primary_arm": "mixed_ports_zero",
        "primary_arm_basis": "沿用 R26/R30 对端口来源捷径的处理，检验扩充监督后的行为统计基线；在读取本批次结果前固定。",
        "comparison_policy": "四臂全部报告；其余三臂为固定对照，不根据 fox/OpTC 选择赢家或重训。",
        "negative_result_scope": "阴性结果只否定这套 HGB、输入与监督设定的收益，不推出 raw8 或所有历史信息普遍无效。",
        "code_sha256": {"runner": digest(HERE / "run.py"), "report_helper": digest(runner.LEGACY)},
        "inputs": inputs, "allocation_unit": "complete source run/campaign; no cross-role split",
        "calibration": {"cohort": "C", "alpha": 0.01, "rule": "score > sorted_C[-(floor(.01*n)+1)]", "population_guarantee": False},
        "model_selection": False, "seed_search": False, "refits": 0,
        "target": "Control and malicious download; other author attacks retain truth but target_label=null",
        "development_only": True, "algorithm_innovation_claim": False,
    }
    save_json(output / "registration.json", registration)
    summary = {
        "unique_score_rows": len(result), "cohorts": {k: len(v) for k, v in cohorts.items()},
        "score_role_counts": dict(Counter(r["dataset_role"] for r in sorted_rows)),
        "target_kind_counts_by_role": [{"role": role, "target_kind": kind, "count": count}
            for (role, kind), count in sorted(Counter((r["dataset_role"], r["target_kind"]) for r in sorted_rows).items())],
        "old_event_metadata_rows_preserved": len(old_events), "normal_download_actions_added": len(c_downloads),
        "normal_download_candidate_relations": sum(len(request_candidates[r["service_request_id"]]) for r in c_downloads),
        "C_background_unknown_rows": len(c_background), "events": len(events), "normal_units": len(units),
        "Unraveled_target_host_observations": len(full_target_rows),
        "Unraveled_training_expansion": len(expansion_ids),
        "Unraveled_other_attack_target_null": sum(r["target_kind"] == "other_attack" for r in full_target_rows),
        "full_source_receipt": full_receipt, "training_calls": 0, "score_calls": 0, "threshold_calls": 0,
        "execution_enabled": False, "inputs_only_pending": True,
    }
    save_json(output / "materialization_summary.json", summary)
    save_json(output / "source_receipt.json", {
        "materializer_sha256": digest(Path(__file__)),
        "source_files": [{"path": str(path), "sha256": digest(path)} for path in sorted(used)],
        "old_registrations_modified": False,
    })
    print(json.dumps({k: v for k, v in summary.items() if k != "full_source_receipt"}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.work_root / "覆盖与历史依据落实_20261119/batch"
    materialize(args.work_root, destination)
