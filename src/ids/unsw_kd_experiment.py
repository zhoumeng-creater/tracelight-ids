"""Bound UNSW four-arm laboratory run. Never reads V/calibration/test."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .budget_transfer import _no_links, _scores, _state_hash, _training_frames, _working_point
from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .input_extrapolation import benchmark_pair
from .models import FeaturePreprocessor, load_bundle, save_bundle
from .risk_research import _fixed_metrics
from .runner import verify_development_inputs
from .transfer_diagnostics import binary_rank_metrics
from .transfer_screen import verify_transfer_screen
from .unsw_kd import ARMS, array_sha256, build_targets, fit_arm
from .unsw_kd_evaluation import evaluate_endpoints

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = REPO / "configs/unsw-small-kd-minimal-20261105.json"
COMPLETE = "UNSW_KD_COMPLETE.json"
ROLES = ("fit", "reference", "constraint")


def preprocessing_state(prep):
    """The same saved-state representation used in the frozen artifact audit."""
    if type(prep) is not FeaturePreprocessor:
        raise ValueError("Original, unwrapped FeaturePreprocessor is required")
    return {
        "selected_features": prep.selected_features,
        "numeric_columns": prep.numeric_columns,
        "categorical_columns": prep.categorical_columns,
        "encoded_dim": prep.encoded_dim,
        "imputer_statistics": prep.imputer.statistics_.tolist(),
        "imputer_strategy": prep.imputer.strategy,
        "scaler_mean": prep.scaler.mean_.tolist(),
        "scaler_var": prep.scaler.var_.tolist(),
        "scaler_scale": prep.scaler.scale_.tolist(),
        "scaler_n_samples_seen": np.asarray(prep.scaler.n_samples_seen_).tolist(),
        "encoder_categories": [x.tolist() for x in prep.encoder.categories_],
        "encoder_handle_unknown": prep.encoder.handle_unknown,
        "encoder_dtype": np.dtype(prep.encoder.dtype).str,
        "output_dtype": "<f4",
        "numeric_then_categorical_order": True,
    }


def _protocol(path=None):
    selected = _no_links(PROTOCOL if path is None else path)
    result = read_json(selected)
    if result != read_json(PROTOCOL):
        raise ValueError("Only the checked-in frozen protocol is executable; no matrix overrides")
    return result, selected


def _bound_inputs(screen_dir, data_dir, *, protocol_path=None, frozen_targets_path=None):
    """Verify immutable inputs, decode only F/R/C and cached F teacher targets."""
    p, protocol_path = _protocol(protocol_path)
    screen, data = _no_links(screen_dir), _no_links(data_dir)
    complete_file = _no_links(screen / "TRANSFER_SCREEN_COMPLETE.json")
    if sha256_file(complete_file) != p["data"]["screen_complete_sha256"]:
        raise ValueError("This protocol requires the exact frozen UNSW screen archive")
    done = verify_transfer_screen(screen)
    binding_path = _no_links(REPO / p["data"]["binding"])
    binding = read_json(binding_path)
    if done["binding_sha256"] != binding["screen_binding_sha256"]:
        raise ValueError("Frozen screen binding changed")
    for name, digest in binding["files_sha256"].items():
        candidate = _no_links(screen / name)
        if not candidate.is_relative_to(screen) or sha256_file(candidate) != digest:
            raise ValueError(f"Frozen input artifact changed: {name}")
    plan = read_json(screen / "plan.json")
    if plan["binding"]["source"]["commit"] != p["data"]["source_screen_commit"]:
        raise ValueError("Frozen screen source commit changed")
    manifest = verify_development_inputs(data, splits=("train",))
    if manifest != plan["binding"]["data"]:
        raise ValueError("Prepared manifest differs from the sealed screen")
    if manifest.get("data_kind") == "synthetic":
        raise ValueError("Production entry requires the frozen real UNSW source, not synthetic replacement")
    frames = _training_frames(screen, data)
    for role in ROLES:
        expected = p["data"]["roles"][role]
        f = frames[role]
        if (len(f) != expected["rows"] or int((f.label == 0).sum()) != expected["normal"]
                or int((f.label == 1).sum()) != expected["attack"]):
            raise ValueError(f"Frozen {role} counts changed")
        if array_sha256(f.label, "<i8") != binding["identities"][role]["ordered_label_i8_sha256"]:
            raise ValueError(f"Frozen {role} label order changed")
    anchor = load_bundle(screen / p["student"]["anchor"])
    if (anchor.model_name != "mlp" or anchor.estimator is not None
            or list(anchor.hidden_dims) != p["student"]["hidden"]
            or anchor.encoded_dim != p["student"]["input_features"]
            or len(anchor.selected_features) != p["data"]["raw_fields"]
            or sum(v.numel() for v in anchor.network_state.values()) != p["student"]["parameters"]):
        raise ValueError("Frozen small anchor architecture changed")
    if _state_hash(anchor.network_state) != p["student"]["anchor_state_sha256"]:
        raise ValueError("Frozen anchor weights changed")
    prep = preprocessing_state(anchor.preprocessor)
    if canonical_hash(prep) != p["data"]["preprocessing_semantic_sha256"]:
        raise ValueError("Frozen preprocessing changed")
    with np.load(screen / p["teacher"]["scores"], allow_pickle=False) as cached:
        logits = np.asarray(cached["fit"], dtype=np.float64)
    if array_sha256(logits) != binding["score_sha256_little_endian_f8"]["fit"]:
        raise ValueError("Frozen F teacher native margins changed")
    frozen_q = None
    if frozen_targets_path is not None:
        frozen = _no_links(frozen_targets_path)
        if sha256_file(frozen) != p["teacher"]["frozen_target_npz_sha256"]:
            raise ValueError("Supplied frozen target archive differs from the preregistered file")
        with np.load(frozen, allow_pickle=False) as saved:
            frozen_q = saved["teacher_q"].copy()
    targets, audit, arrays = build_targets(frames["fit"], logits, frozen_q=frozen_q)
    expected_hashes = {
        "teacher_q_sha256": p["teacher"]["q_le_f8_sha256"],
        "q_perm_sha256": p["permutation"]["q_perm_le_f8_sha256"],
        "donor_i8_sha256": p["permutation"]["donor_le_i8_sha256"],
    }
    for key, digest in expected_hashes.items():
        if audit[key] != digest:
            raise ValueError(f"Frozen {key} mismatch; do not relax hashes or change the teacher")
    for value in audit["by_class"].values():
        if (value["eligible_row_fraction"] < p["permutation"]["min_eligible_row_fraction_each_class"]
                or value["eligible_abs_q_minus_y_mass_fraction"] is None
                or value["eligible_abs_q_minus_y_mass_fraction"]
                < p["permutation"]["min_eligible_abs_q_minus_y_mass_fraction_each_class"]):
            raise ValueError("Frozen permutation control coverage failed")
    if audit["class_balanced_mean_abs_target_K_minus_S"] <= 0:
        raise ValueError("Permutation control leaves all effective targets unchanged")
    receipt = {
        "status": "unsw_kd_preflight_passed", "protocol_id": p["protocol_id"],
        "protocol_sha256": sha256_file(protocol_path), "protocol_semantic_sha256": canonical_hash(p),
        "binding_sha256": sha256_file(binding_path),
        "screen_dir": str(screen), "data_dir": str(data),
        "screen_complete_sha256": sha256_file(complete_file), "data_manifest": manifest,
        "source_screen_binding_sha256": done["binding_sha256"],
        "anchor_state_sha256": _state_hash(anchor.network_state),
        "preprocessing_semantic_sha256": canonical_hash(prep), "target_audit": audit,
        "roles": copy.deepcopy(p["data"]["roles"]),
        "new_teacher_fits": 0, "new_warmup_fits": 0, "new_student_fits": 0,
        "new_inference_rows": 0, "validation_loaded": False, "reserved_partitions_read": [],
    }
    return {"protocol": p, "protocol_path": protocol_path, "frames": frames, "anchor": anchor,
            "targets": targets, "arrays": arrays, "receipt": receipt, "screen": screen, "data": data}


def preflight_unsw_kd(screen_dir, data_dir, *, protocol_path=None, frozen_targets_path=None):
    """Zero-fit, zero-model-query validation; creates no output directories."""
    return _bound_inputs(screen_dir, data_dir, protocol_path=protocol_path,
                         frozen_targets_path=frozen_targets_path)["receipt"]


def _record_endpoint(bundle, query, chunk):
    scores = _scores(bundle, query, chunk)
    wp = _working_point(scores, query, .01)
    wp["source"] = "fixed endpoint max(R-normal,C-normal); no checkpoint selection"
    record = {
        "working_point": wp,
        "roles": {r: {"metrics": _fixed_metrics(f, scores[r], wp["threshold"]),
                       "rank": binary_rank_metrics(f.label.to_numpy(), scores[r], .01)}
                  for r, f in query.items()},
    }
    for role in record["roles"].values():
        role["metrics"]["decision_rule"] = "float64(raw_score)>=registered_max_R_C_normal_threshold"
    return scores, record


def _deployment(bundle, path):
    return {
        "raw_fields": len(bundle.selected_features), "encoded_dim": bundle.encoded_dim,
        "hidden": list(bundle.hidden_dims),
        "parameters": sum(t.numel() for t in bundle.network_state.values()),
        "preprocessing_semantic_sha256": canonical_hash(preprocessing_state(bundle.preprocessor)),
        "student_only": bundle.estimator is None and bundle.model_name == "mlp",
        "package_bytes": path.stat().st_size,
    }


def _cost(hard, candidate, frame, *, measure_cpu):
    if not measure_cpu:
        return {"measured": False, "passed": False, "reason": "formal cost not measured"}
    timing = benchmark_pair(hard, candidate, frame, repeats=25, blocks=3, seed=20261105)
    ratios = {}
    for size, batch in timing["batches"].items():
        batch["H"], batch["K"] = batch.pop("original"), batch.pop("clipped")
        batch["order_labels"] = ["H", "K"]
        ratios[size] = [b["p95_ms"] / a["p95_ms"]
                        for a, b in zip(batch["H"]["blocks"], batch["K"]["blocks"])]
    return {"measured": True, "passed": all(r <= 1.25 for values in ratios.values() for r in values),
            "threshold": 1.25, "block_p95_ratios_K_over_H": ratios,
            "percentile_method": "numpy.percentile(values,95,method='linear')", "timing": timing}


def _write_report(root, result):
    ev = result["evaluation"]
    lines = [
        "# UNSW 普通 KD 四臂固定终点结果",
        "",
        "状态：四臂均已完成；以下是开发验收，不是独立确认。",
        f"完整推进条件通过：{result['acceptance_passed']}。",
        f"指标通过：{ev['metric_acceptance_passed']}；结构通过：{result['structure_passed']}；"
        f"成本通过：{result['cost']['passed']}。",
        "",
        "| 臂 | R TP | R FP | C TP | C FP | C 标准化 pAUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        roles = ev["endpoints"][arm]["roles"]
        r, c = roles["reference"]["metrics"], roles["constraint"]["metrics"]
        lines.append(f"| {arm} | {r['true_positives']} | {r['false_positives']} | "
                     f"{c['true_positives']} | {c['false_positives']} | "
                     f"{roles['constraint']['rank']['partial_auc_standardized']:.9f} |")
    lines += ["", "所有条件与逐项布尔值见 summary.json；六份配对事件表位于 paired/。",
              "验收未通过时停止该冻结候选；不自动追加温度、权重、教师、训练步数或 seed。"]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute(context, output, *, device, measure_cpu, source):
    """Internal executor; tests inject a fully synthetic pre-bound context."""
    import torch

    root = Path(output)
    p, frames, anchor = context["protocol"], context["frames"], context["anchor"]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "protocol.json", p)
    write_json(root / "preflight.json", context["receipt"])
    write_json(root / "plan.json", {"protocol_id": p["protocol_id"], "source": source,
        "environment": environment_snapshot(), "device": device,
        "student_fits_planned": 4, "teacher_fits_planned": 0, "warmup_fits_planned": 0,
        "reserved_partitions_read": [], "validation_loaded": False, "automatic_expansion": False,
        "checkpoint_selection": False, "endpoint_step": p["training"]["endpoint"]})
    for name, digest in source.get("files", {}).items():
        origin = _no_links(REPO / name)
        if not origin.is_relative_to(REPO) or sha256_file(origin) != digest:
            raise ValueError("Execution source changed before snapshot")
        destination = root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
    for role, frame in frames.items():
        frame.loc[:, ["sample_id", "feature_hash", "label", "attack_cat"]].to_parquet(
            root / f"{role}-identity.parquet", index=False)
    np.savez_compressed(root / "targets.npz", **context["arrays"])
    write_json(root / "target-audit.json", context["receipt"]["target_audit"])
    query = {r: frames[r] for r in ("reference", "constraint")}
    bundles, reports, endpoint_scores = {}, {}, {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        with threadpool_limits(limits=4):
            initial_scores, initial_report = _record_endpoint(anchor, query, p["training"]["chunk_size"])
            (root / "step0").mkdir()
            save_bundle(anchor, root / "step0/model.joblib")
            np.savez_compressed(root / "step0/scores.npz", **initial_scores)
            write_json(root / "step0/report.json", initial_report)
            encoded = anchor.preprocessor.transform(frames["fit"])
            y = frames["fit"].label.to_numpy()
            qnames = {"K": "teacher_q", "S": "permuted_teacher_q", "M": "class_mean_q"}
            for arm in ARMS:
                folder = root / "arms" / arm
                folder.mkdir(parents=True)
                print(f"[unsw-kd] {arm}: fixed {p['training']['macro_steps']} full-F updates", flush=True)
                with (folder / "history.jsonl").open("x", encoding="utf-8") as log:
                    def record(row):
                        log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        log.flush()
                        if row["step"] % 10 == 0:
                            print(f"[unsw-kd] {arm} step {row['step']}: "
                                  f"loss={row['objective_bce']:.8f}", flush=True)
                    bundle, history, report = fit_arm(
                        anchor, encoded, y, context["targets"][arm], arm=arm, device=device,
                        teacher_probabilities=None if arm == "H" else context["arrays"][qnames[arm]],
                        macro_steps=p["training"]["macro_steps"], chunk_size=p["training"]["chunk_size"],
                        on_step=record)
                model_path = folder / "model.joblib"
                save_bundle(bundle, model_path)
                write_json(folder / "history.json", history)
                scores, endpoint = _record_endpoint(bundle, query, p["training"]["chunk_size"])
                np.savez_compressed(folder / "scores.npz", **scores)
                report["endpoint"] = endpoint
                report["deployment"] = _deployment(bundle, model_path)
                write_json(folder / "run.json", report)
                bundles[arm], reports[arm], endpoint_scores[arm] = bundle, report, scores
            evaluation, tables = evaluate_endpoints(
                query, endpoint_scores,
                minimum_net_tp=p["acceptance"]["C"]["K_vs_each_H_S_M"]["minimum_net_true_positives"],
                max_fpr=p["evaluation"]["max_fpr"], pauc_tolerance=p["evaluation"]["pauc_numeric_tolerance"])
            (root / "paired").mkdir()
            for name, table in tables.items():
                table.to_csv(root / "paired" / f"{name}.csv", index=False, float_format="%.17g")
            pd.DataFrame(evaluation["concentration"]["groups"]).to_csv(
                root / "paired/K-H_constraint_feature_groups.csv", index=False)
            budget = {
                "student_fits": len(reports), "teacher_fits": 0, "warmup_fits": 0,
                "optimizer_updates": sum(r["optimizer_updates"] for r in reports.values()),
                "gradient_forward_rows": sum(r["gradient_forward_rows"] for r in reports.values()),
                "backward_calls": sum(r["backward_calls"] for r in reports.values()),
                "R_C_scoring_rows": 5 * sum(len(f) for f in query.values()),
            }
            budget_passed = (
                budget["student_fits"] == 4
                and budget["optimizer_updates"] == p["training"]["total_optimizer_updates"]
                and budget["gradient_forward_rows"] == p["training"]["total_training_row_forwards"]
                and budget["backward_calls"] == p["training"]["total_backward_calls"]
                and budget["R_C_scoring_rows"] == p["training"]["total_shared_step0_plus_endpoints_R_C_score_rows"]
                and all(r["endpoint_step"] == p["training"]["endpoint"]
                        and r["common_initial_state_sha256"] == p["student"]["anchor_state_sha256"]
                        and r["checkpoint_selection"] is False for r in reports.values()))
            if not budget_passed:
                write_json(root / "budget-failure.json", {"budget": budget, "runs": reports})
                raise ValueError("Executed training budget or common-anchor/endpoint contract changed")
            structure_checks = {
                arm: (r["deployment"]["raw_fields"] == p["data"]["raw_fields"]
                      and r["deployment"]["encoded_dim"] == p["student"]["input_features"]
                      and r["deployment"]["hidden"] == p["student"]["hidden"]
                      and r["deployment"]["parameters"] == p["student"]["parameters"]
                      and r["deployment"]["student_only"]
                      and r["deployment"]["preprocessing_semantic_sha256"]
                      == p["data"]["preprocessing_semantic_sha256"])
                for arm, r in reports.items()}
            package_ratio = reports["K"]["deployment"]["package_bytes"] / reports["H"]["deployment"]["package_bytes"]
            structure_passed = all(structure_checks.values()) and package_ratio <= 1.05
            cost = _cost(bundles["H"], bundles["K"], query["constraint"], measure_cpu=measure_cpu)
            if source.get("files") and code_fingerprint() != source:
                raise ValueError("Execution source changed during the run")
            result = {
                "status": "unsw_kd_complete", "protocol_id": p["protocol_id"],
                "student_fits": 4, "teacher_fits": 0, "warmup_fits": 0,
                "budget": budget, "budget_passed": budget_passed, "runs": reports,
                "evaluation": evaluation, "structure_checks": structure_checks,
                "K_to_H_package_bytes_ratio": package_ratio, "structure_passed": structure_passed,
                "cost": cost,
                "acceptance_passed": bool(evaluation["metric_acceptance_passed"] and structure_passed
                                          and budget_passed and cost["passed"]),
                "validation_loaded": False, "reserved_partitions_read": [],
                "automatic_expansion": False, "independent_confirmation": False,
            }
            write_json(root / "summary.json", result)
            _write_report(root, result)
            write_json(root / COMPLETE, {
                "schema_version": 1, "status": result["status"], "created_at": utc_now(),
                "student_fits": 4, "teacher_fits": 0, "warmup_fits": 0,
                "acceptance_passed": result["acceptance_passed"],
                "validation_loaded": False, "reserved_partitions_read": [],
                "files": {f.relative_to(root).as_posix(): sha256_file(f)
                          for f in root.rglob("*") if f.is_file()},
            })
            return result
    except BaseException as exc:
        write_json(root / "FAILURE.json", {
            "status": "technical_failure", "exception": type(exc).__name__, "message": str(exc),
            "completed_arms": list(reports), "traceback": traceback.format_exc(),
            "validation_loaded": False, "reserved_partitions_read": [],
            "action": "preserve partial artifacts; no automatic retry or additional arms"})
        raise
    finally:
        torch.set_num_threads(previous_threads)


def run_unsw_kd(screen_dir, data_dir, output, *, device="cuda", measure_cpu=True,
                protocol_path=None, frozen_targets_path=None):
    """Run the exact preregistered four arms on the lab GPU; no resume/search."""
    import torch

    root, screen, data = map(_no_links, (output, screen_dir, data_dir))
    if root.exists():
        raise FileExistsError("Preserve previous output; choose a new UNSW KD directory")
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (screen, data, REPO)):
        raise ValueError("Output must be separate from repository and immutable inputs")
    if (device not in {"cuda", "cuda:0"} or not torch.cuda.is_available()
            or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper()):
        raise ValueError("Real four-arm training requires laboratory CUDA0 RTX4090D")
    if not measure_cpu:
        raise ValueError("Production run includes the frozen H/K formal CPU cost check")
    source = code_fingerprint()
    if source.get("dirty") is not False or source.get("commit") is None:
        raise ValueError("Laboratory execution requires a clean committed checkout")
    context = _bound_inputs(screen, data, protocol_path=protocol_path,
                            frozen_targets_path=frozen_targets_path)
    return _execute(context, root, device=device, measure_cpu=measure_cpu, source=source)
