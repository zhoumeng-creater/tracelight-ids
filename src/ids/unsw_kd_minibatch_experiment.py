"""Bound two-arm mini-batch execution, with fixed endpoint and no matrix expansion."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from .budget_transfer import _no_links, _scores, _state_hash
from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .models import save_bundle
from .risk_research import _fixed_metrics
from .runner import verify_development_inputs
from .transfer_diagnostics import binary_rank_metrics
from .unsw_diagnostic_experiment import _bound_file, _bound_inputs
from .unsw_kd import array_sha256
from .unsw_kd_experiment import preprocessing_state
from .unsw_kd_minibatch import fit_minibatch_arm
from .unsw_kd_minibatch_evaluation import evaluate_minibatch_endpoints

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/unsw-kd-minibatch-minimal-20261106.json"
PROTOCOL_SEMANTIC_SHA256 = "e856ba68a9758f99fdc20c8d746cd599052770bf994ccfec9c2ee20f28155f3d"
COMPLETE = "UNSW_KD_MINIBATCH_COMPLETE.json"
ARMS = ("H", "K")
ROLES = ("reference", "constraint")


def _protocol():
    protocol = read_json(PROTOCOL)
    if canonical_hash(protocol) != PROTOCOL_SEMANTIC_SHA256:
        raise ValueError("Frozen mini-batch protocol changed; no parameter override")
    return protocol


def _permutations(protocol):
    spec = protocol["permutations"]
    if np.__version__ != spec["numpy_version"]:
        raise ValueError("Frozen permutation generation requires numpy==1.26.4")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(spec["seed_sequence"])))
    epochs, n = spec["shape"]
    orders = np.stack([rng.permutation(n) for _ in range(epochs)]).astype("<i8")
    if (array_sha256(orders, "<i8") != spec["sha256"]
            or [array_sha256(row, "<i8") for row in orders] != spec["epoch_sha256"]):
        raise ValueError("Frozen common F permutations changed")
    orders.setflags(write=False)
    return orders


def _qualification(protocol):
    spec = protocol["qualification"]
    evidence = read_json(_no_links(REPO / spec["path"]))
    if (canonical_hash(evidence) != spec["semantic_sha256"]
            or evidence["independent_validity_passed"] is not True
            or not evidence["checks"] or not all(v is True for v in evidence["checks"].values())
            or evidence["source"]["commit"] != spec["diagnostic_commit"]):
        raise ValueError("Prior zero-fit qualification evidence changed or failed")
    return evidence


def _historical(context):
    """Recount old endpoint caches, without calling any model or choosing a threshold."""
    old = context["roots"]["round10"]
    endpoints = {}
    for arm in ARMS:
        threshold = read_json(old / f"arms/{arm}/run.json")["endpoint"]["working_point"]["threshold"]
        if not np.isfinite(threshold):
            raise ValueError("Historical full-batch threshold is nonfinite")
        roles = {}
        with np.load(old / f"arms/{arm}/scores.npz", allow_pickle=False) as cached:
            for role in ROLES:
                frame = context["frames"][role]
                values = np.asarray(cached[role], dtype=np.float64)
                if values.shape != (len(frame),) or not np.isfinite(values).all():
                    raise ValueError("Historical endpoint cache shape or values changed")
                roles[role] = {"metrics": _fixed_metrics(frame, values, threshold),
                    "rank": binary_rank_metrics(frame.label.to_numpy(), values, .01)}
        endpoints[arm] = {"working_point": {"threshold": threshold}, "roles": roles}
    delta = (endpoints["K"]["roles"]["constraint"]["metrics"]["true_positives"]
             - endpoints["H"]["roles"]["constraint"]["metrics"]["true_positives"])
    expected = context["protocol"]["acceptance"]["historical_full_C_net_TP"]
    if delta != expected:
        raise ValueError("Historical full-batch K-H difference must equal the frozen +5")
    return {"source": "SHA-bound original endpoint scores and thresholds; no new model query",
            "C_net_TP_K_H": delta, "endpoints": endpoints}


def _context(screen_dir, round10_dir, data_dir):
    protocol = _protocol()
    qualification = _qualification(protocol)
    orders = _permutations(protocol)
    context = _bound_inputs(screen_dir, round10_dir, data_dir)
    context["input_receipt"] = context["receipt"]
    context["protocol"] = protocol
    anchor = context["bundles"]["anchor"]
    for role, expected in protocol["data"]["roles"].items():
        frame = context["frames"][role]
        if (len(frame) != expected["rows"] or int((frame.label == 0).sum()) != expected["normal"]
                or int((frame.label == 1).sum()) != expected["attack"]):
            raise ValueError(f"Frozen mini-batch {role} cohort changed")
    student = protocol["student"]
    if (anchor.encoded_dim != student["input_features"] or list(anchor.hidden_dims) != student["hidden"]
            or sum(p.numel() for p in anchor.network_state.values()) != student["parameters"]
            or _state_hash(anchor.network_state) != student["anchor_state_sha256"]
            or canonical_hash(preprocessing_state(anchor.preprocessor))
            != protocol["data"]["preprocessing_semantic_sha256"]):
        raise ValueError("Mini-batch anchor or preprocessing changed")
    historical = _historical(context)
    context.update(permutations=orders, qualification=qualification, historical=historical)
    context["receipt"] = {
        "status": "minibatch_preflight_complete", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL), "protocol_semantic_sha256": canonical_hash(protocol),
        "qualification_semantic_sha256": canonical_hash(qualification),
        "permutations_sha256": array_sha256(orders, "<i8"),
        "epoch_permutation_sha256": protocol["permutations"]["epoch_sha256"],
        "historical_full_C_net_TP": historical["C_net_TP_K_H"],
        "input_preflight": context["input_receipt"], "planned_budget": protocol["budget"],
        "student_fits": 0, "teacher_fits": 0, "optimizer_updates": 0,
        "new_inference_rows": 0, "reserved_partitions_read": [],
    }
    return context


def preflight_minibatch(screen_dir, round10_dir, data_dir):
    return _context(screen_dir, round10_dir, data_dir)["receipt"]


def _verify_immutable(context, source):
    original = context["input_receipt"]
    for key, digest in original["files_sha256"].items():
        alias, relative = key.split("/", 1)
        _bound_file(context["roots"][alias], relative, digest)
    if verify_development_inputs(context["data"], splits=("train",)) != original["data_manifest"]:
        raise ValueError("Prepared inputs changed during mini-batch execution")
    if (sha256_file(context["binding_path"]) != original["binding_sha256"]
            or sha256_file(PROTOCOL) != context["receipt"]["protocol_sha256"]
            or canonical_hash(_qualification(context["protocol"]))
            != context["receipt"]["qualification_semantic_sha256"]):
        raise ValueError("Frozen protocol, qualification or binding changed during execution")
    if source.get("files") and code_fingerprint() != source:
        raise ValueError("Source changed during mini-batch execution")


def _arm_checks(context, bundle, history, report, arm):
    p, t = context["protocol"], context["protocol"]["training"]
    count = t["epochs"] * t["batches_per_epoch"]
    anchor_hash = p["student"]["anchor_state_sha256"]
    expected_history = len(history) == t["epochs"]
    for i, row in enumerate(history, 1):
        expected_history &= (row["epoch"] == i and row["optimizer_updates"] == t["batches_per_epoch"]
            and row["backward_calls"] == t["batches_per_epoch"]
            and row["gradient_forward_rows"] == p["data"]["roles"]["fit"]["rows"]
            and row["last_batch_rows"] == t["last_batch_rows"]
            and row["permutation_sha256"] == p["permutations"]["epoch_sha256"][i-1]
            and row["cumulative_optimizer_updates"] == i*t["batches_per_epoch"])
    return {
        "common_anchor_unchanged": report["common_initial_state_sha256"] == anchor_hash
            == report["anchor_state_after_sha256"] == _state_hash(context["bundles"]["anchor"].network_state),
        "fixed_endpoint_and_exact_updates": report["endpoint_epoch"] == t["endpoint_epoch"]
            and report["checkpoint_selection"] is False and report["optimizer_updates"] == count
            and report["forward_calls"] == count == report["backward_calls"]
            and report["gradient_forward_rows"] == t["epochs"]*p["data"]["roles"]["fit"]["rows"],
        "same_frozen_orders_and_targets": report["permutations_sha256"] == p["permutations"]["sha256"]
            and report["epoch_permutation_sha256"] == p["permutations"]["epoch_sha256"]
            and report["target_sha256"] == array_sha256(context["targets"][f"target_{arm}"])
            and report["weights_sha256"] == array_sha256(context["targets"]["weights"]),
        "unchanged_structure_and_preprocessing": list(bundle.hidden_dims) == p["student"]["hidden"]
            and bundle.encoded_dim == p["student"]["input_features"]
            and report["parameters"] == p["student"]["parameters"]
            and canonical_hash(preprocessing_state(bundle.preprocessor)) == p["data"]["preprocessing_semantic_sha256"]
            and _state_hash(bundle.network_state) == report["final_state_sha256"],
        "frozen_optimizer_and_precision": report["optimizer"] == t["optimizer"]
            and all(report[key] == t[key] for key in ("seed", "batch_size", "network_dtype", "loss_dtype",
                    "amp", "tf32", "threads", "deterministic_algorithms", "weighting")),
        "complete_epoch_logs": bool(expected_history),
        "no_extra_queries": report["teacher_queries"] == report["extra_F_evaluation_rows"]
            == report["R_C_evaluation_rows_inside_training"] == 0,
    }


def _report(root, result):
    ev = result["evaluation"]
    lines = ["# H/K mini-batch 最小受控验证", "",
        f"执行有效性：{'通过' if result['validity_passed'] else '未通过'}。",
        f"质量门槛：{'通过' if result['quality_acceptance_passed'] else '未通过'}；"
        f"质量与交互门槛：{'通过' if result['acceptance_passed'] else '未通过'}。", "",
        f"C 上 K−H 净 TP：{ev['interaction']['mini_net_TP']}；原 full 差值："
        f"{ev['interaction']['historical_full_net_TP']}；交互量：{ev['interaction']['net_TP_interaction']}。",
        "", "| 验收项 | 结果 |", "|---|---|"]
    lines.extend(f"| {key} | {'通过' if value else '未通过'} |" for key, value in ev["checks"].items())
    lines += ["", "history 的损失来自一遍内不断更新的模型，不是该遍终点的 F 损失。",
        "本轮只检验整体优化方案与 KD 的交互；没有新 S/M、成本计时或独立确认。",
        "未达门槛时保留完整负结果并结束本分支，不自动加 seed、遍数或超参数组合。"]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _execute(context, output, *, device, source):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    p = context["protocol"]
    for name, value in (("protocol", p), ("binding", context["binding"]),
                        ("preflight", context["receipt"]), ("qualification", context["qualification"]),
                        ("historical-full", context["historical"])):
        write_json(root / f"{name}.json", value)
    reports, scores, arm_checks = {}, {}, {}
    scoring_rows = scoring_calls = 0
    try:
        write_json(root / "plan.json", {"source": source, "environment": environment_snapshot(),
            "device": device, "budget": p["budget"], "automatic_retry": False, "checkpoint_selection": False})
        for name, digest in source.get("files", {}).items():
            origin = _bound_file(REPO, name, digest)
            target = root / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
        np.savez_compressed(root / "permutations.npz", orders=context["permutations"])
        np.savez_compressed(root / "targets.npz", **{key: context["targets"][key]
            for key in ("labels", "weights", "teacher_q", "target_H", "target_K")})
        for role, frame in context["frames"].items():
            frame[["sample_id", "feature_hash", "label", "attack_cat"]].to_parquet(
                root / f"{role}-identity.parquet", index=False)
        anchor = context["bundles"]["anchor"]
        encoded = anchor.preprocessor.transform(context["frames"]["fit"])
        query = {role: context["frames"][role] for role in ROLES}
        for arm in ARMS:
            folder = root / "arms" / arm
            folder.mkdir(parents=True)
            def on_epoch(row):
                with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                print(f"[mini-batch {arm}] epoch {row['epoch']}/{p['training']['epochs']}; "
                      f"updates={row['cumulative_optimizer_updates']}", flush=True)
            bundle, history, report = fit_minibatch_arm(anchor, encoded, context["targets"]["labels"],
                context["targets"][f"target_{arm}"], context["permutations"], arm=arm, device=device,
                teacher_probabilities=context["targets"]["teacher_q"] if arm == "K" else None,
                on_epoch=on_epoch)
            save_bundle(bundle, folder / "model.joblib")
            write_json(folder / "history.json", history)
            reports[arm] = report
            arm_checks[arm] = _arm_checks(context, bundle, history, report, arm)
            write_json(folder / "run.json", {**report, "validity_checks": arm_checks[arm]})
            if not all(arm_checks[arm].values()):
                raise ValueError(f"{arm} violated the fixed execution contract; preserve partial results")
            chunk = p["training"]["endpoint_score_chunk_size"]
            with threadpool_limits(limits=4):
                scores[arm] = _scores(bundle, query, chunk)
            scoring_rows += sum(len(v) for v in scores[arm].values())
            scoring_calls += sum(math.ceil(len(v)/chunk) for v in scores[arm].values())
            np.savez_compressed(folder / "scores.npz", **scores[arm])
        evaluation, tables = evaluate_minibatch_endpoints(query, scores,
            historical_full_delta=context["historical"]["C_net_TP_K_H"])
        for name, table in tables.items():
            table.to_parquet(root / f"{name}.parquet", index=False)
        for arm in ARMS:
            write_json(root / f"arms/{arm}/run.json", {**reports[arm], "validity_checks": arm_checks[arm],
                "endpoint": evaluation["endpoints"][arm]})
        budget = {"student_fits": len(reports), "teacher_fits": 0, "warmup_fits": 0,
            **{key: sum(r[key] for r in reports.values()) for key in ("teacher_queries", "gradient_forward_rows",
                "forward_calls", "backward_calls", "optimizer_updates", "extra_F_evaluation_rows")},
            "endpoint_R_C_scoring_rows": scoring_rows, "endpoint_R_C_scoring_calls": scoring_calls,
            "endpoint_models": len(scores), "intermediate_R_C_scoring_rows": sum(
                r["R_C_evaluation_rows_inside_training"] for r in reports.values()),
            "CPU_cost_calls": 0, "reserved_score_rows": 0}
        _verify_immutable(context, source)
        checks = {"arms_valid": all(all(v.values()) for v in arm_checks.values()),
                  "exact_frozen_budget": budget == p["budget"], "immutable_inputs_and_source": True,
                  "permutations_unchanged": array_sha256(context["permutations"], "<i8") == p["permutations"]["sha256"]}
        # The existing full-batch caches give the other two cells of the 2x2 table.
        vs_full = {}
        for arm in ARMS:
            vs_full[arm] = {}
            for role in ROLES:
                old = context["historical"]["endpoints"][arm]["roles"][role]
                new = evaluation["endpoints"][arm]["roles"][role]
                vs_full[arm][role] = {"net_true_positives": new["metrics"]["true_positives"]-old["metrics"]["true_positives"],
                    "net_false_positives": new["metrics"]["false_positives"]-old["metrics"]["false_positives"],
                    "delta_pauc_standardized": new["rank"]["partial_auc_standardized"]-old["rank"]["partial_auc_standardized"]}
        valid = all(checks.values())
        result = {"status": "unsw_kd_minibatch_complete", "protocol_id": p["protocol_id"],
            "student_fits": budget["student_fits"], "teacher_fits": 0, "optimizer_updates": budget["optimizer_updates"],
            "budget": budget, "budget_limits": p["budget"], "checks": checks, "arm_checks": arm_checks,
            "validity_passed": valid, "evaluation": evaluation, "mini_minus_full": vs_full,
            "quality_acceptance_passed": valid and evaluation["metric_acceptance_passed"],
            "acceptance_passed": valid and evaluation["interaction_acceptance_passed"],
            "automatic_expansion": False, "reserved_partitions_read": [], "independent_confirmation": False}
        write_json(root / "summary.json", result)
        _report(root, result)
        write_json(root / COMPLETE, {"status": result["status"], "created_at": utc_now(),
            "student_fits": budget["student_fits"], "teacher_fits": 0, "optimizer_updates": budget["optimizer_updates"],
            "validity_passed": valid, "acceptance_passed": result["acceptance_passed"],
            "files": {path.relative_to(root).as_posix(): sha256_file(path)
                      for path in root.rglob("*") if path.is_file()}})
        return result
    except BaseException as exc:
        write_json(root / "FAILURE.json", {"status": "technical_failure", "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "completed_arms": list(reports),
            "action": "preserve partial results; no automatic retry or continuation"})
        raise


def run_minibatch(screen_dir, round10_dir, data_dir, output, *, device="cuda"):
    import torch

    root = _no_links(output)
    inputs = [_no_links(p) for p in (screen_dir, round10_dir, data_dir)]
    if root.exists():
        raise FileExistsError("Preserve previous output; no overwrite or automatic resume")
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in [REPO, *inputs]):
        raise ValueError("Output must be separate from source and immutable inputs")
    if (device not in {"cuda", "cuda:0"} or not torch.cuda.is_available()
            or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper()):
        raise ValueError("Real mini-batch execution requires laboratory CUDA0 RTX4090D")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("Use the lab entry with CUBLAS_WORKSPACE_CONFIG=:4096:8")
    source = code_fingerprint()
    if source.get("dirty") is not False or source.get("commit") is None:
        raise ValueError("Laboratory execution requires a clean committed checkout")
    context = _context(*inputs)
    return _execute(context, root, device=device, source=source)
