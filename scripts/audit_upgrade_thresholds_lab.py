"""No-fit laboratory audit of exact threshold feasibility for saved V3/V4 models.

Both environments are required. Threshold selection uses R and training C;
validation is scored only AFTER thresholds are fixed. This is not calibration
on an independent certificate set, model retraining, or a new efficacy claim.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ids.common import canonical_hash, code_fingerprint, read_json, sha256_file, utc_now, write_json  # noqa: E402
from ids.models import load_bundle  # noqa: E402
from ids.risk_distillation import build_risk_reference, freeze_margin_threshold  # noqa: E402
from ids.risk_research import _constraints, _fixed_metrics, _infer, verify_risk_research  # noqa: E402
from ids.runner import verify_development_inputs  # noqa: E402
from ids.upgrade_research import _cohorts, _network_from, empirical_constraints, verify_upgrade_research  # noqa: E402
from predict_risk import raw_cpu_logits  # noqa: E402
from run_risk_lab import (  # noqa: E402
    _activate_cuda_zero, _assert_no_links, _dataset_record, _inventory, _verify_zip,
    check_environment, require_lab_gpu,
)

COMPLETE = "THRESHOLD_AUDIT_COMPLETE.json"
PREFLIGHT = "THRESHOLD_AUDIT_LAB_PREFLIGHT.json"
PACKAGE_MANIFEST = "THRESHOLD_AUDIT_PACKAGE_MANIFEST.sha256"


def solve_threshold_interval(*args, **kwargs):
    from ids.threshold_feasibility import solve_threshold_interval as solve
    return solve(*args, **kwargs)


def _metadata_record(v3_dir, v4_dir, data_dir):
    """Metadata only: no parquet/model reads and no full payload hashing."""
    v3, v4, data = (_assert_no_links(p) for p in (v3_dir, v4_dir, data_dir))
    c3, c4 = read_json(v3 / "RISK_RESEARCH_COMPLETE.json"), read_json(v4 / "UPGRADE_COMPLETE.json")
    p3, p4 = read_json(v3 / "plan.json"), read_json(v4 / "plan.json")
    manifest = read_json(data / "manifest.json")
    for root, complete in ((v3, c3), (v4, c4)):
        if (complete.get("status") != "development_complete" or complete.get("reserved_partitions_read") != []
                or complete.get("files", {}).get("plan.json") != sha256_file(root / "plan.json")):
            raise ValueError("saved development completion/plan metadata is invalid")
    if (p3.get("reserved_partitions_read") != [] or p4.get("reserved_partitions_read") != []
            or p3.get("plan_id") != canonical_hash({k: v for k, v in p3.items() if k != "plan_id"})
            or c3.get("plan_id") != p3.get("plan_id")):
        raise ValueError("V3 plan identity or reserved-partition scope differs")
    b4 = p4.get("binding", {})
    if p4.get("binding_sha256") != canonical_hash(b4) or c4.get("binding_sha256") != p4.get("binding_sha256"):
        raise ValueError("V4 binding identity differs")
    if (b4.get("v3_complete_sha256") != sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json")
            or b4.get("v3_plan_id") != p3.get("plan_id") or b4.get("data") != manifest or p3.get("data") != manifest):
        raise ValueError("V3/V4/prepared identity mismatch; do not substitute new data")
    cfg3, cfg4 = p3.get("config", {}), b4.get("config", {})
    if (cfg3.get("purpose") != "risk_v3_development_only" or cfg3.get("cohort_policy") != "complete_development"
            or cfg4.get("purpose") != "upgrade_v4_development_only"
            or not set(cfg4.get("seeds", [])).issubset(cfg3.get("seeds", []))
            or not cfg4.get("seeds") or cfg4.get("max_fpr") != cfg3.get("max_fpr")
            or any(cfg4.get("optimization", {}).get(k) != cfg3.get("optimization", {}).get(k)
                   for k in ("attack_loss_limit", "min_family_support"))):
        raise ValueError("saved studies do not share the frozen cohort/seeds/risk budgets")
    return {"v3_dir": str(v3), "v4_dir": str(v4), "data_dir": str(data),
            "v3_complete_sha256": sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json"),
            "v4_complete_sha256": sha256_file(v4 / "UPGRADE_COMPLETE.json"),
            "v3_plan_sha256": sha256_file(v3 / "plan.json"), "v4_plan_sha256": sha256_file(v4 / "plan.json"),
            "manifest_sha256": sha256_file(data / "manifest.json"), "v3_plan_id": p3["plan_id"],
            "v4_binding_sha256": p4["binding_sha256"], "seeds": cfg4["seeds"], "max_fpr": cfg4["max_fpr"]}


def _full_source_check(record):
    verify_risk_research(record["v3_dir"])
    verify_upgrade_research(record["v4_dir"])
    verify_development_inputs(record["data_dir"])
    if _metadata_record(record["v3_dir"], record["v4_dir"], record["data_dir"]) != record:
        raise ValueError("source identity changed since preflight")


def model_catalog(v3, v4, seed):
    """Only models declared in verified source completion inventories are loaded."""
    done3, done4 = read_json(v3 / "RISK_RESEARCH_COMPLETE.json"), read_json(v4 / "UPGRADE_COMPLETE.json")
    source3, source4 = done3["files"], done4["files"]
    catalog = []
    for key, relative in (("v3_anchor", f"anchors/s{seed}.joblib"), ("v3_hard", f"students/hard-s{seed}/model.joblib")):
        if relative not in source3:
            raise ValueError("required original anchor/hard model is missing")
        catalog.append({"name": f"{key}-s{seed}", "seed": seed, "source_kind": key, "path": str(v3 / relative),
                        "source_relative_path": relative, "model_sha256": source3[relative]})
    for row in read_json(v4 / "summary.json")["runs"]:
        if row["seed"] != seed or row["status"] == "no_feasible_start":
            continue
        relative = f"students/{row['condition']}-s{seed}/model.joblib"
        if relative not in source4:
            raise ValueError("trained V4 result has no sealed model")
        catalog.append({"name": f"v4_{row['condition']}-s{seed}", "seed": seed, "source_kind": "v4_selected",
                        "condition": row["condition"], "source_candidate_status": row["status"],
                        "selected_step": row.get("selected_step"), "path": str(v4 / relative),
                        "source_relative_path": relative, "model_sha256": source4[relative]})
    phase = read_json(v4 / f"phase1-s{seed}.json")
    checkpoint = phase.get("checkpoint_file")
    expected = f"phase1-s{seed}.joblib"
    if expected in source4:
        if checkpoint != expected or phase.get("checkpoint_sha256") != source4[expected]:
            raise ValueError("Phase-I checkpoint metadata/inventory mismatch")
        catalog.append({"name": f"v4_phase1-s{seed}", "seed": seed, "source_kind": "v4_phase1_checkpoint",
                        "path": str(v4 / expected), "source_relative_path": expected, "model_sha256": source4[expected],
                        "source_candidate_status": phase["status"]})
        missing = None
    else:
        if checkpoint:
            raise ValueError("Phase-I metadata claims a checkpoint absent from source inventory")
        missing = {"seed": seed, "phase1_status": phase["status"], "updates": phase["updates"],
                   "status": "missing_phase1_snapshot", "reason": "original package has no saved weights; not reconstructed or retrained"}
    return catalog, missing


def _score_model(bundle, frames, threads):
    # Only an in-memory setting is changed; saved source packages remain sealed.
    bundle.num_threads = threads
    return {role: raw_cpu_logits(bundle, frames[role]) for role in ("reference", "constraint")}


def _threshold_metrics(scores, frames, threshold, reference):
    value = float(threshold)
    normal = frames["reference"].label.to_numpy() == 0
    reference_fp = int((scores["reference"][normal] >= value).sum())
    return {"threshold": value, "reference_normal_count": int(normal.sum()), "reference_false_positives": reference_fp,
            "reference_fpr": reference_fp / int(normal.sum()),
            "constraint": empirical_constraints(scores["constraint"], value, reference),
            "validation": _fixed_metrics(frames["validation"], scores["validation"], value)}


def audit_model(catalog_row, bundle, frames, reference, anchor_threshold, alpha, threads):
    """Fix all thresholds from R/C before scoring V; V never enters the solver."""
    scores = _score_model(bundle, frames, threads)
    rid = tuple(frames["reference"].sample_id.astype(str))
    cid = tuple(frames["constraint"].sample_id.astype(str))
    r_threshold = freeze_margin_threshold(torch.tensor(scores["reference"]), torch.tensor(frames["reference"].label.to_numpy()), alpha)["threshold"]
    interval = solve_threshold_interval(scores["reference"], frames["reference"].label.to_numpy(), scores["constraint"], reference,
                                        reference_ids=rid, constraint_ids=cid, max_fpr=alpha)
    selected = interval["threshold"] if interval["feasible"] else None
    # Only after both threshold choices are fixed do we score validation.
    scores["validation"] = raw_cpu_logits(bundle, frames["validation"])
    exact = {"solution": interval,
             "evaluation": _threshold_metrics(scores, frames, selected, reference) if selected is not None else None,
             "lower_bound_diagnostic": None}
    if selected is None and interval.get("lower_bound") is not None:
        exact["lower_bound_diagnostic"] = _threshold_metrics(scores, frames, interval["lower_bound"], reference)
    result = {"model": catalog_row, "regimes": {
        "R_only": _threshold_metrics(scores, frames, r_threshold, reference),
        "frozen_anchor": _threshold_metrics(scores, frames, anchor_threshold, reference),
        "exact_R_C_interval": exact},
        "threshold_selection_roles": ["reference", "constraint"], "validation_role": "post_selection_description_only",
        "fits": 0, "scope": "fixed-model finite training-pool diagnostic, not independent certification or efficacy ranking"}
    return result, scores


def teacher_score_variants(bundle, frame):
    """Native HGB margins and the EXACT old float32 probability/clip path.

    Both query the same fitted preprocessing and public HGB APIs. No fitting,
    private raw-prediction API, class-prior correction or alarm centering.
    """
    if bundle.model_name != "hgb" or bundle.estimator.classes_.tolist() != [0, 1]:
        raise ValueError("teacher-value audit requires the saved binary [0,1] HGB teacher")
    if frame.empty:
        raise ValueError("teacher-value scoring requires a nonempty population")
    matrix = bundle.preprocessor.transform(frame)
    native = np.asarray(bundle.estimator.decision_function(matrix), dtype=np.float64)
    probability = np.asarray(bundle.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)
    if (native.shape != (len(frame),) or probability.shape != native.shape or not np.isfinite(native).all()
            or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any()):
        raise ValueError("teacher returned invalid raw margins/probabilities")
    difference = float(np.max(np.abs(expit(native) - probability)))
    if difference > 1e-12:
        raise ValueError("native binary HGB margin/link disagrees with predict_proba")
    # Do not promote before clipping: the original ModelBundle returned f32,
    # np.clip and scipy.special.logit preserved f32, then the OOF array was f64.
    old_probability = probability.astype(np.float32)
    clipped = np.clip(old_probability, 1e-6, 1 - 1e-6)
    legacy = logit(clipped).astype(np.float64)
    if not np.isfinite(legacy).all():
        raise ValueError("legacy teacher representation is nonfinite")
    return {"native": native, "legacy_clipped": legacy}, {
        "native_link_max_abs_probability_difference": difference,
        "legacy_probability_dtype": "float32", "legacy_logit_dtype_before_saved_cast": "float32",
        "legacy_lower_clip_fraction": float((old_probability <= np.float32(1e-6)).mean()),
        "legacy_upper_clip_fraction": float((old_probability >= np.float32(1 - 1e-6)).mean())}


def _score_profile(scores, labels):
    values, y = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if values.shape != y.shape or values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("teacher profiles need aligned finite scores and binary labels")
    result = {}
    for label in (0, 1):
        sample = values[y == label]
        if not len(sample):
            result[str(label)] = {"n": 0, "quantiles": None, "unique_scores": 0, "largest_tie_fraction": None}
            continue
        _, counts = np.unique(sample, return_counts=True)
        result[str(label)] = {"n": len(sample), "quantiles": np.quantile(sample, [0, .01, .25, .5, .75, .99, 1]).tolist(),
            "unique_scores": len(counts), "largest_tie_fraction": float(counts.max() / len(sample))}
    return result


def teacher_student_complementarity(labels, families, teacher_scores, teacher_threshold, student_scores, student_threshold):
    """Descriptive paired events at separately R-chosen operating points.

    This is not an oracle ensemble, a transfer gain or a matched-V-FPR ranking.
    """
    y = np.asarray(labels)
    t, s = np.asarray(teacher_scores, dtype=np.float64), np.asarray(student_scores, dtype=np.float64)
    family = np.asarray(families).astype(str)
    if (y.ndim != 1 or not len(y) or t.shape != y.shape or s.shape != y.shape or family.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(t).all() or not np.isfinite(s).all()
            or not np.isfinite([teacher_threshold, student_threshold]).all()):
        raise ValueError("paired teacher/student decisions must be finite aligned binary populations")
    td, sd = t >= float(teacher_threshold), s >= float(student_threshold)
    corrected = (y == 1) & ~sd & td
    harmed = (y == 1) & sd & ~td
    repaired_fp = (y == 0) & sd & ~td
    added_fp = (y == 0) & ~sd & td
    return {"teacher_corrects_student_fn": int(corrected.sum()), "teacher_loses_student_tp": int(harmed.sum()),
        "teacher_repairs_student_fp": int(repaired_fp.sum()), "teacher_adds_student_fp": int(added_fp.sum()),
        "teacher_minus_student_tp": int(corrected.sum()) - int(harmed.sum()),
        "teacher_minus_student_fp": int(added_fp.sum()) - int(repaired_fp.sum()),
        "student_threshold": float(student_threshold), "teacher_threshold": float(teacher_threshold),
        "per_attack_family": {name: {"support": int(((y == 1) & (family == name)).sum()),
            "teacher_corrects_student_fn": int((corrected & (family == name)).sum()),
            "teacher_loses_student_tp": int((harmed & (family == name)).sum())}
            for name in sorted(set(family[y == 1]))},
        "scope": "error-complementarity diagnostic; no selector learned; V FPRs need not be equal"}


def _oof_family_support(targets, fit_frame, metadata):
    """Which actual OOF teachers saw each diagnostic attack family?

    Families are diagnostics, not the binary fitting labels. Missing same-family
    support does not imply missing attack-class support or data leakage.
    """
    if len(targets) != len(fit_frame) or not np.array_equal(targets.label.to_numpy(), fit_frame.label.to_numpy()):
        raise ValueError("OOF family audit requires aligned fitting labels")
    y = targets.label.to_numpy()
    fold = targets.fold.to_numpy()
    raw = targets.teacher_logit.to_numpy(dtype=np.float64)
    margin = targets.teacher_margin.to_numpy(dtype=np.float64)
    if (not np.isin(y, [0, 1]).all() or not np.isfinite(raw).all() or not np.isfinite(margin).all()
            or not np.issubdtype(fold.dtype, np.integer)):
        raise ValueError("OOF family audit needs finite logits and integer fold assignments")
    fitted = {int(row.get("fold", index)): row["fit"] for index, row in enumerate(metadata["folds"])}
    if len(fitted) != len(metadata["folds"]) or not set(fold).issubset(fitted):
        raise ValueError("OOF family audit is missing a unique source fold")
    if any(not isinstance(row.get("families"), dict) for row in fitted.values()):
        raise ValueError("OOF fold metadata is missing its family-support map; absence is unknown")
    family = fit_frame.attack_cat.astype(str).to_numpy()
    result = {}
    for name in sorted(set(family[y == 1])):
        mask = (y == 1) & (family == name)
        used = sorted(set(fold[mask]))
        support = {str(k): int(fitted[k]["families"].get(name, 0)) for k in used}
        absent = sum(int((mask & (fold == k)).sum()) for k in used if support[str(k)] == 0)
        result[name] = {"n": int(mask.sum()), "teacher_fit_family_counts_by_target_fold": support,
            "teacher_fit_binary_counts_by_target_fold": {str(k): fitted[k].get("labels") for k in used},
            "n_rows_with_family_absent_in_teacher_fit": absent,
            "raw_probability_median": float(np.median(expit(raw[mask]))),
            "raw0_5_tpr": float((raw[mask] >= 0).mean()), "workingpoint_tpr": float((margin[mask] >= 0).mean()),
            "family_is_binary_training_target": False}
    return result


def audit_oof_teacher_value(v3, seed, frames, targets, metadata, inventory, alpha, threads):
    """Query the actual archived fold teachers, never fit or pool raw margins.

    Each teacher scores only its original F holdout and common R/C. Its R
    normals register its own two score representations. Pooled OOF reporting
    combines fixed decisions, not incomparable scores from different models.
    """
    fit = frames["fit"]
    if (targets.sample_id.tolist() != fit.sample_id.tolist()
            or not np.array_equal(targets.label.to_numpy(), fit.label.to_numpy())):
        raise ValueError("OOF query requires the original aligned F membership")
    fold_ids = targets.fold.to_numpy()
    fold_metadata = {int(row.get("fold", i)): row for i, row in enumerate(metadata["folds"])}
    if (not np.issubdtype(fold_ids.dtype, np.integer) or len(fold_metadata) != len(metadata["folds"])
            or set(fold_ids) != set(fold_metadata)):
        raise ValueError("OOF query requires every unique archived fold")
    arrays = {"fold": fold_ids.astype(np.int64)}
    for variant in ("native", "legacy_clipped"):
        arrays[f"{variant}_fit_oof"] = np.full(len(fit), np.nan, dtype=np.float64)
        arrays[f"{variant}_fit_oof_decision"] = np.zeros(len(fit), dtype=np.uint8)
    reports = []
    for fold in sorted(fold_metadata):
        relative = f"teachers/s{seed}/fold-{fold}.joblib"
        path, expected = v3 / relative, inventory.get(relative)
        if not expected or sha256_file(path) != expected:
            raise ValueError("actual OOF teacher is not bound to the source completion")
        heldout = np.flatnonzero(fold_ids == fold)
        model = load_bundle(path, device="cpu")
        model.num_threads = threads
        r_scores, r_checks = teacher_score_variants(model, frames["reference"])
        working = {variant: freeze_margin_threshold(torch.tensor(values),
            torch.tensor(frames["reference"].label.to_numpy()), alpha)
            for variant, values in r_scores.items()}
        subsets = {"heldout": fit.iloc[heldout], "constraint": frames["constraint"]}
        scores, checks = {"reference": r_scores}, {"reference": r_checks}
        for role, frame in subsets.items():
            scores[role], checks[role] = teacher_score_variants(model, frame)
        legacy = scores["heldout"]["legacy_clipped"]
        cached = targets.teacher_logit.to_numpy(dtype=np.float64)[heldout]
        if not np.array_equal(legacy, cached):
            raise ValueError("archived OOF model no longer reproduces its exact legacy targets")
        evaluations = {}
        for variant in ("native", "legacy_clipped"):
            threshold = working[variant]["threshold"]
            arrays[f"{variant}_fit_oof"][heldout] = scores["heldout"][variant]
            arrays[f"{variant}_fit_oof_decision"][heldout] = scores["heldout"][variant] >= threshold
            evaluations[variant] = {"working_point": working[variant], "profiles": {}, "roles": {}}
            for role, frame in {"reference": frames["reference"], **subsets}.items():
                z, labels = scores[role][variant], frame.label.to_numpy()
                evaluations[variant]["profiles"][role] = _score_profile(z, labels)
                evaluations[variant]["roles"][role] = {
                    "metrics": _fixed_metrics(frame, z, threshold),
                    "partial_auc_standardized_mcclish": float(roc_auc_score(labels, z, max_fpr=alpha))
                    if len(np.unique(labels)) == 2 else None}
                if role != "heldout":
                    arrays[f"fold_{fold}_{variant}_{role}"] = z
        reports.append({"fold": fold, "source_model_relative_path": relative, "source_model_sha256": expected,
            "fit_support": fold_metadata[fold]["fit"], "heldout_rows": len(heldout),
            "legacy_targets_exactly_reproduced": True, "link_checks": checks, "evaluations": evaluations})
    aggregate = {}
    labels = fit.label.to_numpy()
    for variant in ("native", "legacy_clipped"):
        decisions = arrays[f"{variant}_fit_oof_decision"]
        aggregate[variant] = _fixed_metrics(fit, decisions.astype(np.float64), .5)
        aggregate[variant]["threshold"] = None
        aggregate[variant]["decision_rule"] = "each row uses its archived fold teacher and that fold's R threshold"
    return {"native_fold_predictions_queried": True, "query_roles": ["F_original_fold_holdout", "R", "C"],
        "fits": 0, "folds": reports, "pooled_oof_fixed_decisions": aggregate,
        "pooled_score_auc": None, "pooled_score_auc_reason": "raw coordinates from distinct fold models are not pooled",
        "scope": "actual OOF target diagnostics; F is a student fitting pool, not a fair student comparison or transfer proof"}, arrays


def audit_teacher_value(v3, seed, frames, model_results, output, alpha, threads):
    """Audit full-F teacher capability; actual OOF targets remain separate."""
    relative = f"teachers/s{seed}/reference.joblib"
    complete = read_json(v3 / "RISK_RESEARCH_COMPLETE.json")
    source = v3 / relative
    expected = complete["files"].get(relative)
    if not expected or sha256_file(source) != expected:
        raise ValueError("full-F teacher is not bound to the original completion")
    began = time.perf_counter()
    bundle = load_bundle(source, device="cpu")
    bundle.num_threads = threads
    scores, profiles, link_checks, working = {}, {}, {}, {}
    # R/C are scored first. No teacher V score is queried until all R choices
    # for BOTH representations are frozen; no native/clip winner is selected.
    for role in ("reference", "constraint"):
        variants, link_checks[role] = teacher_score_variants(bundle, frames[role])
        for variant, values in variants.items():
            scores[f"{variant}_{role}"] = values
    for variant in ("native", "legacy_clipped"):
        working[variant] = freeze_margin_threshold(torch.tensor(scores[f"{variant}_reference"]),
            torch.tensor(frames["reference"].label.to_numpy()), alpha)
    variants, link_checks["validation"] = teacher_score_variants(bundle, frames["validation"])
    for variant, values in variants.items():
        scores[f"{variant}_validation"] = values
    vlabels = frames["validation"].label.to_numpy()
    evaluations, paired = {}, {}
    for variant in ("native", "legacy_clipped"):
        profiles[variant] = {role: _score_profile(scores[f"{variant}_{role}"], frames[role].label.to_numpy())
                             for role in ("reference", "constraint", "validation")}
        threshold = working[variant]["threshold"]
        z = scores[f"{variant}_validation"]
        evaluations[variant] = {"working_point": working[variant],
            "validation": _fixed_metrics(frames["validation"], z, threshold),
            "roc_auc": float(roc_auc_score(vlabels, z)) if len(np.unique(vlabels)) == 2 else None,
            "partial_auc_standardized_mcclish": float(roc_auc_score(vlabels, z, max_fpr=alpha)) if len(np.unique(vlabels)) == 2 else None,
            "partial_auc_max_fpr": alpha,
            "partial_auc_units": "sklearn McClish standardized, not raw interval area"}
        paired[variant] = {}
        for row in model_results:
            name = row["model"]["name"]
            # Reuse saved MLP arrays instead of doing another forward pass.
            with np.load(output / "models" / name / "scores.npz", allow_pickle=False) as stored:
                student = stored["validation"]
            comparison = teacher_student_complementarity(vlabels,
                frames["validation"].attack_cat.to_numpy(), z, threshold, student,
                row["regimes"]["R_only"]["threshold"])
            comparison["student_source_kind"] = row["model"].get("source_kind")
            comparison["student_budget_note"] = ("original anchor fitted on F; unlike the updated F+C students"
                if row["model"].get("source_kind") == "v3_anchor" else
                "updated-study F+C hard-label budget; selected step0/Phase-I state may have used fewer updates")
            paired[variant][name] = comparison

    target_relative, oof_relative = f"teachers/s{seed}/targets.parquet", f"teachers/s{seed}/oof.json"
    for rel in (target_relative, oof_relative):
        if rel not in complete["files"] or sha256_file(v3 / rel) != complete["files"][rel]:
            raise ValueError("actual OOF metadata/targets changed")
    targets = pd.read_parquet(v3 / target_relative, columns=["sample_id", "label", "teacher_logit", "teacher_margin", "fold"])
    if (targets.sample_id.tolist() != frames["fit"].sample_id.tolist()
            or not np.array_equal(targets.label.to_numpy(), frames["fit"].label.to_numpy())):
        raise ValueError("actual OOF targets do not match source F identities")
    oof = read_json(v3 / oof_relative)
    oof_profile = {"targets_sha256": complete["files"][target_relative], "metadata_sha256": complete["files"][oof_relative],
        "row_count": len(targets), "fold_fit_rows": [row["fit"]["n_rows"] for row in oof["folds"]],
        "legacy_logit_profile": _score_profile(targets.teacher_logit.to_numpy(), targets.label.to_numpy()),
        "boundary_target_profile": _score_profile(expit(targets.teacher_margin.to_numpy()), targets.label.to_numpy()),
        "per_attack_family": _oof_family_support(targets, frames["fit"], oof)}
    native_oof, oof_scores = audit_oof_teacher_value(v3, seed, frames, targets, oof, complete["files"], alpha, threads)
    oof_profile.update(native_oof)
    count0, count1 = int((frames["fit"].label == 0).sum()), int((frames["fit"].label == 1).sum())
    result = {"seed": seed, "source_model_relative_path": relative, "source_model_sha256": expected,
        "source_model_bytes": source.stat().st_size, "fits": 0, "score_representations": list(evaluations),
        "teacher_training_roles": ["fit"], "teacher_F_rows": len(frames["fit"]),
        "teacher_uses_balanced_sample_weights": True, "teacher_relative_positive_weight": count0 / count1,
        "student_hard_training_roles": ["fit", "constraint"],
        "student_budget_note": "main updated-student budget; original F-only anchor is identified separately in each pair",
        "student_hard_rows": len(frames["fit"]) + len(frames["constraint"]),
        "teacher_identity": "full-F reference teacher, not the three fold models actually supplying KD targets",
        "actual_oof_targets": oof_profile, "profiles": profiles, "link_checks": link_checks,
        "evaluations": evaluations, "paired_with_mlp_R_only": paired,
        "query_seconds": time.perf_counter() - began,
        "query_cost_scope": "audit queries, not repeated deployment benchmarking",
        "scope": "teacher capability and error-complementarity diagnostic, not fair-budget leaderboard or proven transfer gain",
        "validation_role": "post-R-threshold description only; no representation selected by V"}
    teacher_dir = output / f"teacher-value-s{seed}"
    teacher_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(teacher_dir / "scores.npz", **scores)
    np.savez_compressed(teacher_dir / "oof-scores.npz", **oof_scores)
    write_json(teacher_dir / "audit.json", result)
    return result


def _save_reference(output, seed, reference):
    identities = output / "constraint-identity.parquet"
    arrays = {"anchor_margins_C": reference.anchor_margins.cpu().numpy(), "labels_C": reference.labels.cpu().numpy()}
    arrays.update({f"member_{i}": mask.cpu().numpy() for i, mask in enumerate(reference.members)})
    np.savez_compressed(output / f"frozen-reference-s{seed}.npz", **arrays)
    write_json(output / f"frozen-reference-s{seed}.json", {
        "constraints": [asdict(x) for x in reference.constraints], "margin_floor": reference.margin_floor,
        "majorizer": reference.majorizer, "normal_repair_credit": reference.normal_repair_credit,
        "normal_loss_form": reference.normal_loss_form, "signature": reference.signature,
        "constraint_identity_file": identities.name, "constraint_identity_sha256": sha256_file(identities),
        "baseline_source": "original V3 training-device anchor margins, not CPU candidate scores"})


def _audit_environment(name, record, output, binding, threads):
    _full_source_check(record)
    v3, v4, data = (Path(record[k]) for k in ("v3_dir", "v4_dir", "data_dir"))
    frames = _cohorts(v3, data)
    p3 = read_json(v3 / "plan.json")
    original_device = p3["device"]
    if original_device not in {"cuda", "cuda:0"} and p3["data"].get("data_kind") != "synthetic":
        raise ValueError("real source V3 must have used laboratory CUDA0")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", {"binding": binding, "binding_sha256": canonical_hash(binding), "input": record,
                                      "environment_name": name, "reserved_partitions_read": []})
    evidence = output / "source-evidence"
    evidence.mkdir(exist_ok=False)
    shutil.copyfile(v3 / "RISK_RESEARCH_COMPLETE.json", evidence / "v3-complete.json")
    for role in ("fit", "reference", "constraint", "validation"):
        shutil.copyfile(v3 / f"{role}-identity.parquet", output / f"{role}-identity.parquet")
    for relative in binding["source"]["files"]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    results, missing, teacher_audits = [], [], []
    with threadpool_limits(limits=threads):
        for seed in record["seeds"]:
            print(f"[threshold audit] {name} seed={seed} original anchor events", flush=True)
            anchor = load_bundle(v3 / f"anchors/s{seed}.joblib", device="cpu")
            x = anchor.preprocessor.transform(frames["constraint"])
            network = _network_from(anchor, original_device)
            raw_original = _infer(network, x, original_device, p3["config"]["optimization"]["chunk_size"])
            anchor_threshold = read_json(v3 / f"students/hard-s{seed}/run.json")["working_point"]["threshold"]
            families = set().union(*(set(frames[r].loc[frames[r].label == 1, "attack_cat"]) for r in ("fit", "reference", "constraint")))
            specs, family_vector, support = _constraints(frames["constraint"], p3["config"], families)
            reference = build_risk_reference(torch.tensor(raw_original - anchor_threshold), torch.tensor(frames["constraint"].label.to_numpy()),
                specs, sample_ids=tuple(frames["constraint"].sample_id.astype(str)), attack_families=torch.tensor(family_vector),
                margin_floor=p3["config"]["optimization"]["margin_floor"])
            expected_reference = read_json(v3 / f"students/hard-s{seed}/run.json")["reference_signature"]
            if reference.signature != expected_reference:
                raise ValueError("original GPU risk-reference identity differs; do not redefine the V3 baseline")
            _save_reference(output, seed, reference)
            write_json(output / f"family-support-s{seed}.json", support)
            catalog, absent = model_catalog(v3, v4, seed)
            if absent:
                missing.append(absent)
            seed_results = []
            for model in catalog:
                print(f"[threshold audit] {name} {model['name']}", flush=True)
                path = _assert_no_links(model["path"])
                if sha256_file(path) != model["model_sha256"]:
                    raise ValueError("source model changed before scoring")
                bundle = load_bundle(path, device="cpu")
                result, scores = audit_model(model, bundle, frames, reference, anchor_threshold, record["max_fpr"], threads)
                model_dir = output / "models" / model["name"]
                model_dir.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(model_dir / "scores.npz", **scores)
                write_json(model_dir / "audit.json", result)
                results.append(result)
                seed_results.append(result)
            print(f"[teacher-value audit] {name} seed={seed} full-F reference teacher", flush=True)
            teacher_audits.append(audit_teacher_value(v3, seed, frames, seed_results, output, record["max_fpr"], threads))
    _full_source_check(record)
    if code_fingerprint()["source_sha256"] != binding["source"]["source_sha256"]:
        raise ValueError("source changed during audit; preserve incomplete results")
    write_json(output / "summary.json", {"environment": name, "results": results, "missing_phase1_snapshots": missing,
        "teacher_value_audits": teacher_audits,
        "scope": "no-fit diagnostic; R+C thresholds are training-pool choices, not independent certificates", "fits": 0})
    write_json(output / COMPLETE, {"status": "threshold_audit_complete", "artifact_schema_version": 3, "created_at": utc_now(), "fits": 0,
        "n_models": len(results), "n_teacher_value_audits": len(teacher_audits), "reserved_partitions_read": [], "binding_sha256": canonical_hash(binding),
        "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return verify_audit(output)


def verify_audit(output):
    root = _assert_no_links(output)
    complete = read_json(root / COMPLETE)
    artifact_schema = complete.get("artifact_schema_version", 1)
    if type(artifact_schema) is not int or artifact_schema not in {1, 2, 3}:
        raise ValueError("unsupported threshold-audit artifact schema")
    files = complete.get("files", {})
    if (complete.get("status") != "threshold_audit_complete" or complete.get("fits") != 0
            or complete.get("reserved_partitions_read") != [] or not files):
        raise ValueError("not a completed no-fit threshold audit")
    inventory = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / COMPLETE}
    if set(files) != inventory:
        raise ValueError("audit inventory mismatch")
    for name, digest in files.items():
        target = _assert_no_links(root / name)
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != digest:
            raise ValueError("audit artifact mismatch: " + name)
    plan, summary = read_json(root / "plan.json"), read_json(root / "summary.json")
    if (plan.get("reserved_partitions_read") != [] or plan.get("binding_sha256") != canonical_hash(plan["binding"])
            or complete.get("binding_sha256") != plan.get("binding_sha256") or summary.get("fits") != 0
            or complete.get("n_models") != len(summary.get("results", []))):
        raise ValueError("audit plan/coverage mismatch")
    rows = summary.get("results", [])
    names = [row["model"]["name"] for row in rows]
    if len(set(names)) != len(names) or any(not {"v3_anchor", "v3_hard"}.issubset(
            {row["model"].get("source_kind") for row in rows if row["model"].get("seed") == seed}) for seed in plan["input"]["seeds"]):
        raise ValueError("audit is missing source anchor/hard coverage")
    required = {"plan.json", "summary.json", "reference-identity.parquet", "constraint-identity.parquet", "validation-identity.parquet"}
    for seed in plan["input"]["seeds"]:
        required.update((f"frozen-reference-s{seed}.npz", f"frozen-reference-s{seed}.json", f"family-support-s{seed}.json"))
    if artifact_schema >= 2:
        required.add("source-evidence/v3-complete.json")
        if files.get("source-evidence/v3-complete.json") != plan["input"]["v3_complete_sha256"]:
            raise ValueError("teacher source completion evidence differs")
        original_inventory = read_json(root / "source-evidence/v3-complete.json")["files"]
        teachers = summary.get("teacher_value_audits", [])
        if (len(teachers) != len(plan["input"]["seeds"]) or {row["seed"] for row in teachers} != set(plan["input"]["seeds"])
                or complete.get("n_teacher_value_audits") != len(teachers)):
            raise ValueError("teacher-value audit coverage mismatch")
        for row in teachers:
            prefix = f"teacher-value-s{row['seed']}/"
            required.update((prefix + "audit.json", prefix + "scores.npz"))
            relative = f"teachers/s{row['seed']}/reference.joblib"
            if (row.get("fits") != 0 or read_json(root / (prefix + "audit.json")) != row
                    or row.get("source_model_relative_path") != relative
                    or row.get("source_model_sha256") != original_inventory.get(relative)
                    or row.get("actual_oof_targets", {}).get("targets_sha256") != original_inventory.get(f"teachers/s{row['seed']}/targets.parquet")
                    or row.get("actual_oof_targets", {}).get("metadata_sha256") != original_inventory.get(f"teachers/s{row['seed']}/oof.json")):
                raise ValueError("teacher-value per-seed summary differs")
            if artifact_schema >= 3:
                required.update(("fit-identity.parquet", prefix + "oof-scores.npz"))
                oof = row.get("actual_oof_targets", {})
                if oof.get("native_fold_predictions_queried") is not True or oof.get("fits") != 0 or not oof.get("folds"):
                    raise ValueError("actual OOF teacher audit is incomplete")
                for fold in oof["folds"]:
                    relative = f"teachers/s{row['seed']}/fold-{fold['fold']}.joblib"
                    if (fold.get("source_model_relative_path") != relative
                            or fold.get("source_model_sha256") != original_inventory.get(relative)
                            or fold.get("legacy_targets_exactly_reproduced") is not True):
                        raise ValueError("actual OOF teacher source binding differs")
    for row in summary["results"]:
        prefix = "models/" + row["model"]["name"] + "/"
        required.update((prefix + "audit.json", prefix + "scores.npz"))
        if read_json(root / (prefix + "audit.json")) != row or row.get("fits") != 0:
            raise ValueError("audit per-model summary differs")
    for name, digest in plan["binding"]["source"]["files"].items():
        required.add("source/" + name)
        if files.get("source/" + name) != digest:
            raise ValueError("audit source snapshot differs")
    if not required.issubset(files):
        raise ValueError("audit required payloads missing")
    return complete


def package_audit(output, target, *, resume=False):
    root, target = _assert_no_links(output), _assert_no_links(target)
    if target.is_relative_to(root):
        raise ValueError("package must be outside sealed audit")
    verify_audit(root)
    inventory = _inventory(root)
    payload = {root.name + "/" + n: d for n, d in inventory.items()}
    content = "".join(f"{d}  {n}\n" for n, d in sorted(payload.items())).encode()
    expected = {**payload, PACKAGE_MANIFEST: hashlib.sha256(content).hexdigest()}
    sidecar = _assert_no_links(target.with_name(target.name + ".sha256"))
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file() or sidecar.read_text().strip() != sha256_file(target):
            raise FileExistsError("existing/partial audit package is preserved")
        _verify_zip(target, expected)
        return {"zip": str(target), "sha256": sha256_file(target), "status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in inventory:
            archive.write(root / name, root.name + "/" + name)
        archive.writestr(PACKAGE_MANIFEST, content)
    _verify_zip(target, expected)
    if _inventory(root) != inventory:
        raise ValueError("audit changed while packaging")
    digest = sha256_file(target)
    with sidecar.open("x", encoding="ascii") as stream:
        stream.write(digest + "\n")
    return {"zip": str(target), "sha256": digest, "status": "verified_new"}


def run_lab(*, work_dir, unsw_v3_dir=None, unsw_v4_dir=None, unsw_data_dir=None,
            cicids_v3_dir=None, cicids_v4_dir=None, cicids_data_dir=None, threads=4, dry_run=False, resume=False):
    triples = {"unsw": (unsw_v3_dir, unsw_v4_dir, unsw_data_dir), "cicids": (cicids_v3_dir, cicids_v4_dir, cicids_data_dir)}
    if any(any(p is None for p in paths) for paths in triples.values()):
        raise ValueError("this audit requires BOTH environments with V3, V4 and prepared data paths")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    environment = check_environment()
    gpu = require_lab_gpu(environment)
    _activate_cuda_zero()
    inputs = {}
    for name, (v3, v4, data) in triples.items():
        _dataset_record(name, data)
        inputs[name] = _metadata_record(v3, v4, data)
    work = _assert_no_links(work_dir)
    for record in inputs.values():
        for key in ("v3_dir", "v4_dir", "data_dir"):
            old = Path(record[key])
            if work.is_relative_to(old) or old.is_relative_to(work):
                raise ValueError("new audit work must be separate from every source/data directory")
    stable = copy.deepcopy(environment)
    stable.pop("recorded_at", None)
    binding = {"purpose": "threshold_feasibility_no_fit_audit", "inputs": inputs, "source": code_fingerprint(),
               "threads": threads, "environment": stable, "interpreter": str(Path(sys.executable).resolve()), "device": "cuda:0"}
    preflight = work / PREFLIGHT
    if preflight.exists():
        if not resume or read_json(preflight).get("binding") != binding:
            raise FileExistsError("existing preflight needs identical-input complete-only resume")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("new audit work must be empty")
        work.mkdir(parents=True, exist_ok=True)
        write_json(preflight, {"created_at": utc_now(), "binding": binding, "selected_gpu": gpu, "fits": 0,
                               "dry_run_scope": "JSON metadata and environment only; no models/parquet", "reserved_partitions_read": []})
    results = {}
    for name, record in inputs.items():
        output = work / ("threshold-audit-" + name)
        if dry_run:
            results[name] = {"status": "plan_only", "fits": 0}
            continue
        previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        try:
            if output.exists():
                if not resume or not (output / COMPLETE).is_file():
                    raise FileExistsError("incomplete/existing audit preserved; no partial resume")
                complete = verify_audit(output)
                if read_json(output / "plan.json")["binding"] != binding:
                    raise ValueError("complete audit binding changed")
                _full_source_check(record)
            else:
                complete = _audit_environment(name, record, output, binding, threads)
        finally:
            torch.set_num_threads(previous[0])
            torch.use_deterministic_algorithms(previous[1])
        if complete.get("fits") != 0:
            raise ValueError("a no-fit audit cannot contain fitted models")
        if code_fingerprint()["source_sha256"] != binding["source"]["source_sha256"]:
            raise ValueError("audit source changed; do not package")
        results[name] = {"status": "verified_complete", "models_scored": complete["n_models"],
                         "teacher_value_audits": complete.get("n_teacher_value_audits", 0),
                         "package": package_audit(output, work / "packages" / f"THRESHOLD_AUDIT_{name.upper()}.zip", resume=resume)}
    if not dry_run:
        packages = work / "packages"
        receipt = packages / PREFLIGHT
        raw = preflight.read_bytes()
        if receipt.exists():
            if not resume or receipt.read_bytes() != raw:
                raise FileExistsError("preflight receipt preserved; bytes differ")
        else:
            with receipt.open("xb") as stream:
                stream.write(raw)
        sidecar = receipt.with_name(receipt.name + ".sha256")
        digest = hashlib.sha256(raw).hexdigest()
        if sidecar.exists():
            if not resume or sidecar.read_text().strip() != digest:
                raise ValueError("preflight receipt hash differs")
        else:
            sidecar.write_text(digest + "\n", encoding="ascii")
    return {"status": "threshold_audit_plan_only" if dry_run else "threshold_audit_complete", "fits": 0,
            "environments": results, "work_dir": str(work), "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("unsw", "cicids"):
        for role in ("v3", "v4", "data"):
            parser.add_argument(f"--{name}-{role}-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_lab(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"Threshold audit rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
