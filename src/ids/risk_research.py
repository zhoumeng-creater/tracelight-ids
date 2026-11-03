"""V3 controlled development study. Never reads calibration or historical test.

All students start from one anchor, receive F+C hard labels and take the same
number of full-population AdamW steps. OOF soft targets are used on F only.
R freezes working points; C supplies explicit empirical training constraints.
This is an independently implemented binary adaptation, not an author repro.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import yaml

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import load_partition
from .evaluation import calibrate_threshold, evaluate_binary
from .models import ModelBundle, _network, save_bundle, train_model
from .research import _group_column, _identity, _sample_record, group_budget, train_research_mlp
from .risk_baselines import adapted_ekd_rank_count, mpt_style_focal_kl, pct_focal_kl
from .risk_distillation import (BoundaryKDConfig, ProjectedDual, RiskConstraint, boundary_kd_objective,
                                build_risk_reference, constraint_risks, freeze_margin_threshold,
                                risk_lagrangian, working_margins)
from .risk_roles import make_risk_roles
from .runner import verify_development_inputs
from .selection import normalize_feature_schema

CONDITIONS = {"hard", "boundary_kd", "risk_hard", "risk_matched", "risk_kd", "normal_hinge_kd",
              "pct", "dual_source_fd", "ekd"}
COMPLETE = "RISK_RESEARCH_COMPLETE.json"
PROVENANCE = {
    "pct": {"paper": "https://arxiv.org/html/2011.09161v3", "kind": "paper-correct binary focal KL adaptation",
            "author_commit": "amazon-science/regression-constraint-model-upgrade@fb2268c67706834632ec1427945fcf30a7218227",
            "difference": "per-row weighting; does not reproduce author-code NxN broadcasting"},
    "dual_source_fd": {"paper": "https://arxiv.org/html/2511.08322v2", "kind": "MPT-style double-source FD adaptation",
                       "author_commit": "miccunifi/negative_flip_MPT@2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e",
                       "difference": "fixed binary labels, no class-incremental bias; explicit T squared normalization"},
    "ekd": {"paper": "https://arxiv.org/html/2206.02325", "kind": "binary rank-count adaptation",
            "author_commit": "Tencent/TFace@171ec1434aca0972e111663ac646c9c1e3e44b5b",
            "difference": "row scores, frozen R thresholds, no face-pair mining or momentum thresholds"},
    "risk": {"kind": "candidate only; anchor-conditioned capped hinge is not a new loss family",
             "control": "normal_hinge_kd replaces capped normal envelope by ordinary uncapped hinge"},
}


def validate_risk_config(config):
    keys = {"schema_version", "purpose", "cohort_policy", "seeds", "max_fpr", "threads", "folds", "fold_seed",
            "role_seed", "reference_fraction", "constraint_fraction", "max_train_rows", "max_validation_rows",
            "sampling_seed", "anchor", "teacher", "optimization", "conditions"}
    if not isinstance(config, dict) or set(config) != keys or config["schema_version"] != 1:
        raise ValueError("risk config keys/version do not match the fixed contract")
    if config["purpose"] != "risk_v3_development_only":
        raise ValueError("V3 is development only")
    if config["cohort_policy"] not in {"complete_development", "bounded_pilot"}:
        raise ValueError("explicit cohort_policy required")
    for key in ("threads", "folds", "fold_seed", "role_seed", "sampling_seed"):
        if type(config[key]) is not int or config[key] < (2 if key == "folds" else 1):
            raise ValueError(f"invalid {key}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    for key in ("max_fpr", "reference_fraction", "constraint_fraction"):
        if isinstance(config[key], bool) or not np.isfinite(config[key]) or not 0 < config[key] < 1:
            raise ValueError(f"invalid {key}")
    if config["reference_fraction"] + config["constraint_fraction"] >= 1:
        raise ValueError("roles leave no fitting population")
    for key in ("max_train_rows", "max_validation_rows"):
        value = config[key]
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"invalid {key}")
        if config["cohort_policy"] == "complete_development" and value is not None:
            raise ValueError("complete development forbids row caps")
    conditions = config["conditions"]
    if not isinstance(conditions, list) or not conditions or any(x not in CONDITIONS for x in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError("invalid or duplicate conditions")
    teacher = config["teacher"]
    if teacher.get("model") != "hgb" or teacher.get("features") != {"method": "all"} or teacher.get("hyperparameters", {}).get("early_stopping") is not False:
        raise ValueError("teacher must be full-field HGB with early_stopping false")
    if set(config["anchor"]) != {"hyperparameters"}:
        raise ValueError("anchor requires explicit hyperparameters")
    op = config["optimization"]
    required = {"macro_steps", "chunk_size", "learning_rate", "weight_decay", "temperature", "kd_weight", "fd_alpha",
                "fd_beta", "fd_temperature", "ekd_temperature", "margin_floor", "dual_step_size", "dual_maximum",
                "attack_loss_limit", "gradient_clip", "min_family_support"}
    if set(op) != required:
        raise ValueError("optimization keys do not match contract")
    for key, value in op.items():
        if key in {"macro_steps", "chunk_size", "min_family_support"}:
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid {key}")
        elif isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
    for key in ("learning_rate", "temperature", "fd_temperature", "ekd_temperature", "margin_floor", "dual_step_size", "dual_maximum", "gradient_clip"):
        if op[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if op["attack_loss_limit"] > 1:
        raise ValueError("attack_loss_limit must be a mass in [0,1]")
    BoundaryKDConfig(temperature=op["temperature"], kd_weight=op["kd_weight"])


def _state(network):
    return {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}


def _state_hash(state):
    return canonical_hash({k: v.tolist() for k, v in state.items()})


def _infer(network, x, device, chunk):
    import torch
    network.eval()
    with torch.inference_mode():
        return np.concatenate([network(torch.from_numpy(x[i:i + chunk]).to(device)).squeeze(1).cpu().numpy()
                               for i in range(0, len(x), chunk)]).astype(np.float64)


def _fixed_metrics(frame, logits, threshold, anchor_logits=None):
    """Primary decisions use raw float64 logits, preserving nextafter boundaries."""
    y = frame.label.to_numpy()
    predicted = logits >= threshold
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    result = {"false_positives": int((predicted & (y == 0)).sum()), "n_normal": n0,
              "true_positives": int((predicted & (y == 1)).sum()), "n_attack": n1,
              "fpr": float(predicted[y == 0].mean()) if n0 else None,
              "tpr": float(predicted[y == 1].mean()) if n1 else None,
              "threshold": threshold, "decision_rule": "float64(raw_logit)>=frozen_R_threshold",
              "families": {str(f): {"support": int(((frame.attack_cat == f) & (y == 1)).sum()),
                  "tpr": float(predicted[(frame.attack_cat == f) & (y == 1)].mean())}
                  for f in sorted(frame.loc[y == 1, "attack_cat"].unique())}}
    if anchor_logits is not None:
        b = anchor_logits >= threshold
        result["attack_lost_tp_mass"] = float((b & ~predicted)[y == 1].mean())
        result["normal_repaired_fp_mass"] = float((b & ~predicted)[y == 0].mean())
    return result


def _teacher_targets(fit, reference, validation, config, seed, schema, output):
    """Precheck ALL group folds before the first expensive teacher fit."""
    import torch
    output.mkdir(parents=True)
    groups, labels = fit[_group_column(fit)].to_numpy(), fit.label.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=config["folds"], shuffle=True, random_state=config["fold_seed"])
    splits = list(splitter.split(np.zeros(len(fit)), labels, groups))
    for left, right in splits:
        if set(labels[left]) != {0, 1} or set(labels[right]) != {0, 1}:
            raise ValueError("OOF fold lacks binary support; do not retry role/fold seeds")
        for key in {_group_column(fit), "feature_hash"}:
            if set(fit.iloc[left][key]) & set(fit.iloc[right][key]):
                raise ValueError("OOF group isolation failed")
    scores, folds = np.full(len(fit), np.nan), np.full(len(fit), -1, dtype=int)
    records, grids = [], []
    tc = copy.deepcopy(config["teacher"])
    tc["hyperparameters"]["num_threads"] = config["threads"]
    rates = [config["max_fpr"] / 2, config["max_fpr"], min(config["max_fpr"] * 2, .99)]
    begin = time.perf_counter()
    for fold, (left, right) in enumerate(splits):
        print(f"[v3] teacher seed={seed} fold={fold + 1}/{len(splits)}", flush=True)
        bundle = train_model(fit.iloc[left], fit.iloc[left], tc, seed, feature_schema=schema)
        # Explicit score->logit clipping is common to all external-teacher losses.
        raw = logit(np.clip(bundle.predict_scores(reference), 1e-6, 1 - 1e-6))
        grid = [freeze_margin_threshold(torch.tensor(raw), torch.tensor(reference.label.to_numpy()), rate)["threshold"] for rate in rates]
        scores[right] = logit(np.clip(bundle.predict_scores(fit.iloc[right]), 1e-6, 1 - 1e-6))
        folds[right] = fold
        grids.append(grid)
        save_bundle(bundle, output / f"fold-{fold}.joblib")
        records.append({"fold": fold, "fit": _sample_record(fit.iloc[left]), "heldout": _sample_record(fit.iloc[right]),
                        "reference_threshold_grid": grid})
    if not np.isfinite(scores).all() or (folds < 0).any():
        raise ValueError("OOF coverage incomplete")
    full = train_model(fit, fit, tc, seed, feature_schema=schema)
    save_bundle(full, output / "reference.joblib")
    r = logit(np.clip(full.predict_scores(reference), 1e-6, 1 - 1e-6))
    threshold = freeze_margin_threshold(torch.tensor(r), torch.tensor(reference.label.to_numpy()), config["max_fpr"])["threshold"]
    v = logit(np.clip(full.predict_scores(validation), 1e-6, 1 - 1e-6))
    record = {"folds": records, "rates": rates, "logit_clip_probability": [1e-6, 1 - 1e-6],
              "full_teacher_validation": _fixed_metrics(validation, v, threshold), "generation_seconds": time.perf_counter() - begin,
              "full_teacher_usage": "diagnostic only; never supplies in-sample fitting targets"}
    write_json(output / "oof.json", record)
    _identity(fit).assign(fold=folds, teacher_logit=scores,
                          teacher_margin=scores - np.asarray(grids)[folds, 1]).to_parquet(output / "targets.parquet", index=False)
    return scores, folds, np.asarray(grids), record


def _constraints(frame, config, all_families):
    op = config["optimization"]
    specs = [RiskConstraint("normal", "normal_fpr", config["max_fpr"]),
             RiskConstraint("attack", "attack_loss", op["attack_loss_limit"])]
    mapping = {name: i for i, name in enumerate(sorted(all_families))}
    support = {}
    for name, number in mapping.items():
        count = int(((frame.attack_cat == name) & (frame.label == 1)).sum())
        supported = count >= op["min_family_support"]
        support[name] = {"support": count, "protected": supported,
                         "reason": "predeclared support rule" if supported else "insufficient C support; no family protection claim"}
        if supported:
            specs.append(RiskConstraint(f"family_{number}", "attack_loss", op["attack_loss_limit"], group=number))
    families = frame.attack_cat.map(mapping).fillna(-1).to_numpy(dtype=np.int64)
    return specs, families, support


def _student(fit, constraint, reference, validation, anchor, teacher, config, seed, condition, output, device):
    import torch
    import torch.nn.functional as F
    op, chunk = config["optimization"], config["optimization"]["chunk_size"]
    output.mkdir(parents=True)
    frames = {"fit": fit, "constraint": constraint, "reference": reference, "validation": validation}
    xs = {key: anchor.preprocessor.transform(value) for key, value in frames.items()}
    ys = {key: torch.tensor(value.label.to_numpy(), dtype=torch.float32, device=device) for key, value in frames.items()}
    network = _network(anchor.encoded_dim, anchor.hidden_dims).to(device)
    network.load_state_dict(anchor.network_state)
    anchor_z = {key: _infer(network, x, device, chunk) for key, x in xs.items()}
    threshold_record = freeze_margin_threshold(torch.tensor(anchor_z["reference"]), ys["reference"].cpu(), config["max_fpr"])
    threshold = threshold_record["threshold"]
    rates = [config["max_fpr"] / 2, config["max_fpr"], min(config["max_fpr"] * 2, .99)]
    anchor_grid = [freeze_margin_threshold(torch.tensor(anchor_z["reference"]), ys["reference"].cpu(), rate)["threshold"] for rate in rates]
    all_families = set().union(*(set(frame.loc[frame.label == 1, "attack_cat"]) for frame in (fit, constraint, reference)))
    specs, families, support = _constraints(constraint, config, all_families)
    ids = tuple(constraint.sample_id.astype(str))
    risk_reference = build_risk_reference(torch.tensor(anchor_z["constraint"] - threshold), ys["constraint"].cpu(), specs,
        sample_ids=ids, attack_families=torch.tensor(families), margin_floor=op["margin_floor"],
        normal_loss_form="hinge" if condition == "normal_hinge_kd" else "capped")
    dual = ProjectedDual(specs, step_size=op["dual_step_size"], maximum=op["dual_maximum"], reference_signature=risk_reference.signature)
    use_risk = condition.startswith("risk_") or condition == "normal_hinge_kd"
    soft_mode = "matched_hard" if condition == "risk_matched" else "kd"
    kd_config = BoundaryKDConfig(mode=soft_mode, kd_weight=op["kd_weight"], temperature=op["temperature"])
    total = len(fit) + len(constraint)
    n0 = int((fit.label == 0).sum() + (constraint.label == 0).sum())
    n1 = total - n0
    pos_weight = torch.tensor(n0 / n1, device=device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=op["learning_rate"], weight_decay=op["weight_decay"])
    tz, fold, teacher_grids, _ = teacher
    teacher_margin = tz - teacher_grids[fold, 1]
    fit_counts = {label: int((fit.label == label).sum()) for label in (0, 1)}
    history, best_state, best_rank, selected_step = [], None, None, None
    begin = time.perf_counter()
    for step in range(1, op["macro_steps"] + 1):
        network.train()
        optimizer.zero_grad(set_to_none=True)
        hard_total, soft_total = 0., 0.
        # Fold-local chunking retains each teacher's own frozen reference grid.
        for k in range(config["folds"]):
            indices = np.flatnonzero(fold == k)
            for start in range(0, len(indices), chunk):
                ix = indices[start:start + chunk]
                z = network(torch.from_numpy(xs["fit"][ix]).to(device)).squeeze(1)
                y = ys["fit"][ix]
                hard = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight, reduction="sum") / total
                soft = z.sum() * 0
                if condition in {"boundary_kd", "risk_kd", "risk_matched", "normal_hinge_kd"}:
                    soft = op["kd_weight"] * boundary_kd_objective(z, y, kd_config, student_threshold=threshold,
                        teacher_margins=torch.tensor(teacher_margin[ix], device=device), partition="fit")["soft"] * len(ix) / len(fit)
                elif condition == "pct":
                    soft = pct_focal_kl(z, y, anchor_logits=torch.tensor(anchor_z["fit"][ix], device=device), anchor_threshold=threshold,
                        alpha=op["fd_alpha"], beta=op["fd_beta"], temperature=op["fd_temperature"], coefficient=op["kd_weight"])["soft"] * len(ix) / len(fit)
                elif condition == "dual_source_fd":
                    soft = mpt_style_focal_kl(z, y, anchor_logits=torch.tensor(anchor_z["fit"][ix], device=device),
                        teacher_logits=torch.tensor(tz[ix], device=device), anchor_threshold=threshold, teacher_threshold=teacher_grids[k, 1],
                        alpha=op["fd_alpha"], beta=op["fd_beta"], temperature=op["fd_temperature"],
                        anchor_coefficient=op["kd_weight"] / 2, teacher_coefficient=op["kd_weight"] / 2)["soft"] * len(ix) / len(fit)
                elif condition == "ekd":
                    soft = adapted_ekd_rank_count(z, y, teacher_logits=torch.tensor(tz[ix], device=device),
                        student_thresholds=anchor_grid, teacher_thresholds=teacher_grids[k].tolist(), temperature=op["ekd_temperature"],
                        normal_coefficient=int((y == 0).sum()) / fit_counts[0],
                        attack_coefficient=int((y == 1).sum()) / fit_counts[1], coefficient=op["kd_weight"])["soft"]
                loss = hard + soft
                if not bool(torch.isfinite(loss)):
                    raise ValueError("non-finite fitting loss")
                loss.backward()
                hard_total += float(hard.detach())
                soft_total += float(soft.detach())
        # Complete C graph is retained for ONE global-denominator risk reduction.
        # Chunking bounds forward batch size, not the total C autograd footprint.
        zc = torch.cat([network(torch.from_numpy(xs["constraint"][i:i + chunk]).to(device)).squeeze(1)
                        for i in range(0, len(constraint), chunk)])
        hard_c = F.binary_cross_entropy_with_logits(zc, ys["constraint"], pos_weight=pos_weight, reduction="sum") / total
        report = constraint_risks(working_margins(zc, threshold=threshold), risk_reference, sample_ids=ids)
        risk = risk_lagrangian(report, dual.weights) if use_risk else zc.sum() * 0
        if not bool(torch.isfinite(hard_c + risk)):
            raise ValueError("non-finite constraint loss")
        (hard_c + risk).backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), op["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        zc_after = _infer(network, xs["constraint"], device, chunk)
        after = constraint_risks(torch.tensor(zc_after - threshold), risk_reference, sample_ids=ids)
        if use_risk:
            dual.update(after)
        zv = _infer(network, xs["validation"], device, chunk)
        vm = _fixed_metrics(validation, zv, threshold, anchor_z["validation"])
        actual_feasible = all(float(after.actual_values[i]) <= spec.limit for i, spec in enumerate(specs))
        eligible = actual_feasible and vm["fpr"] <= config["max_fpr"] and (not use_risk or bool((after.violations <= 0).all()))
        rank = (vm["tpr"], -vm["fpr"], -step)
        if eligible and (best_rank is None or rank > best_rank):
            best_state, best_rank, selected_step = _state(network), rank, step
            selected_dual = dual.state_dict()
        history.append({"step": step, "hard": hard_total + float(hard_c.detach()), "soft": soft_total,
                        "risk_lagrangian_before_step": float(risk.detach()), "gradient_norm_before_clip": float(norm),
                        "constraints": after.detached_metrics(), "dual": dual.weights.tolist(),
                        "surrogate_feasible": bool((after.violations <= 0).all()), "actual_C_feasible": actual_feasible,
                        "validation": vm, "eligible": eligible})
        write_json(output / "history.json", history)
        if step == 1 or step % 10 == 0 or step == op["macro_steps"]:
            print(f"[v3] {condition} seed={seed} step={step}/{op['macro_steps']} V_fpr={vm['fpr']:.5f} V_tpr={vm['tpr']:.5f} eligible={eligible}", flush=True)
    diagnostic_only = best_state is None
    if diagnostic_only:
        best_state = _state(network)
        selected_dual = dual.state_dict()
    network.load_state_dict(best_state)
    # The saved working point is a raw-logit contract; probability conversion is
    # independently audited and must not silently change boundary decisions.
    cpu = _network(anchor.encoded_dim, anchor.hidden_dims)
    cpu.load_state_dict(best_state)
    cpu_z = {key: _infer(cpu, x, "cpu", chunk) for key, x in xs.items()}
    gpu_z = {key: _infer(network, x, device, chunk) for key, x in xs.items()}
    metrics = _fixed_metrics(validation, cpu_z["validation"], threshold, anchor_z["validation"])
    cpu_report = constraint_risks(torch.tensor(cpu_z["constraint"] - threshold), risk_reference, sample_ids=ids)
    cpu_eligible = (metrics["fpr"] <= config["max_fpr"]
                    and all(float(cpu_report.actual_values[i]) <= spec.limit for i, spec in enumerate(specs))
                    and (not use_risk or bool((cpu_report.violations <= 0).all())))
    # Full training diagnostics stay external; deployable package size must not
    # depend on how many risk/dual fields a condition happens to log.
    bundle = ModelBundle("mlp", anchor.preprocessor, anchor.selected_features, anchor.encoded_dim, [],
        network_state=best_state, hidden_dims=anchor.hidden_dims, device="cpu", batch_size=chunk,
        num_threads=config["threads"], feature_schema=anchor.feature_schema, best_epoch=selected_step, seed=seed)
    save_bundle(bundle, output / "model.joblib")
    score = bundle.predict_scores(validation).astype(np.float64)
    converted_threshold = float(expit(threshold))
    probability_disagreements = int(((score >= converted_threshold) != (cpu_z["validation"] >= threshold)).sum())
    predictions = validation.loc[:, ["sample_id", "feature_hash", "label", "attack_cat"]].copy()
    predictions["cpu_raw_logit"], predictions["score"] = cpu_z["validation"], score
    predictions["raw_logit_threshold"] = threshold
    predictions["predicted_label"] = (cpu_z["validation"] >= threshold).astype(int)
    predictions.to_parquet(output / "validation-predictions.parquet", index=False)
    descriptive = calibrate_threshold(validation.label, score, config["max_fpr"])
    record = {"condition": condition, "seed": seed, "status": "eligible_development_candidate" if not diagnostic_only and cpu_eligible else "diagnostic_no_eligible_checkpoint",
              "selected_step": selected_step, "macro_steps": len(history), "common_initial_state_sha256": _state_hash(anchor.network_state),
              "positive_weight_F_plus_C": n0 / n1, "hard_labels": {"fit": len(fit), "constraint": len(constraint)},
              "soft_targets": ({"hard": "none", "risk_hard": "none", "risk_matched": "F labels with matched soft-objective budget",
                                "pct": "frozen anchor predictions on F", "dual_source_fd": "anchor plus OOF teacher on F"}.get(condition, "OOF teacher on F only")),
              "working_point": threshold_record, "family_support": support,
              "reference_signature": risk_reference.signature, "normal_loss_form": risk_reference.normal_loss_form,
              "selection_rule": "all: actual C budgets and frozen-threshold V FPR; risk arms additionally surrogate C budgets; max V TPR, min V FPR, earliest step",
              "surrogate_feasible_cpu": bool((cpu_report.violations <= 0).all()),
              "constraint_metrics_cpu": cpu_report.detached_metrics(), "validation_cpu_raw_logit": metrics,
              "validation_curve_descriptive_only": evaluate_binary(validation.label, score, descriptive["threshold"], validation.attack_cat),
              "score_threshold_conversion_disagreements": probability_disagreements,
              "training_vs_cpu_decision_disagreements": {key: int(((cpu_z[key] >= threshold) != (gpu_z[key] >= threshold)).sum()) for key in frames},
              "deployment_contract": "model.joblib raw logits plus run.json working_point; probability threshold is NOT interchangeable",
              "training_seconds": time.perf_counter() - begin, "package_bytes": (output / "model.joblib").stat().st_size,
              "actual_training_device": device, "final_scoring_device": "cpu",
              "parameter_count": sum(p.numel() for p in network.parameters()),
              "provenance": PROVENANCE.get(condition, PROVENANCE["risk"] if use_risk else {"kind": "hard-label baseline" if condition == "hard" else "boundary-centered BCE distillation adaptation"})}
    write_json(output / "run.json", record)
    write_json(output / "history.json", history)
    write_json(output / "dual.json", {"selected_model": selected_dual, "final_trajectory_diagnostic": dual.state_dict()})
    return record


def _binding(data_dir, config, device):
    return {"schema_version": 1, "config": config, "source": code_fingerprint(),
            "data": verify_development_inputs(data_dir), "data_dir": str(Path(data_dir).resolve()), "device": device}


def run_risk_research(data_dir, config, output, *, device="cuda", threads=None, resume=False):
    import torch
    config = copy.deepcopy(config)
    if threads is not None:
        config["threads"] = threads
    validate_risk_config(config)
    root = Path(output).resolve()
    binding = _binding(data_dir, config, device)
    if root.exists():
        if not resume:
            raise FileExistsError("Output exists; use a new directory or verify complete --resume")
        complete = verify_risk_research(root)
        plan = read_json(root / "plan.json")
        if any(plan.get(k) != v for k, v in binding.items()):
            raise ValueError("resume config/source/data/device identity differs")
        return complete
    # Real-data fits are laboratory-only, even when bypassing the wrapper.
    if binding["data"].get("data_kind") != "synthetic":
        if device not in {"cuda", "cuda:0"} or not torch.cuda.is_available() or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper():
            raise ValueError("real research requires laboratory CUDA0 RTX4090D")
    train, validation = (load_partition(data_dir, split) for split in ("train", "validation"))
    schema = normalize_feature_schema(train.attrs.get("feature_schema"))
    if config["cohort_policy"] == "bounded_pilot" and "cic" in schema["name"].lower():
        raise ValueError("bounded CIC pilot would discard giant groups; use complete development")
    train = group_budget(train, config["max_train_rows"], config["sampling_seed"])
    validation = group_budget(validation, config["max_validation_rows"], config["sampling_seed"] + 1)
    roles, role_manifest = make_risk_roles(train, validation, seed=config["role_seed"],
        reference_fraction=config["reference_fraction"], constraint_fraction=config["constraint_fraction"])
    root.mkdir(parents=True, exist_ok=False)
    plan = {**binding, "role_manifest_hash": canonical_hash(role_manifest), "reserved_partitions_read": []}
    plan["plan_id"] = canonical_hash(plan)
    write_json(root / "plan.json", plan)
    write_json(root / "roles.json", role_manifest)
    write_json(root / "environment.json", environment_snapshot())
    write_json(root / "provenance.json", PROVENANCE)
    source_root = Path(__file__).resolve().parents[2]
    for name in binding["source"]["files"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    for role, frame in {**roles, "validation": validation}.items():
        columns = [c for c in ("sample_id", "feature_hash", "group_id", "group_hash", "session_hash", "label", "attack_cat") if c in frame]
        frame[columns].to_parquet(root / f"{role}-identity.parquet", index=False)
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    records = []
    try:
        with threadpool_limits(limits=config["threads"]):
            for seed in config["seeds"]:
                teacher = _teacher_targets(roles["fit"], roles["reference"], validation, config, seed, schema, root / "teachers" / f"s{seed}")
                anchor_config = {"model": "mlp", "features": {"method": "all"}, "hyperparameters": config["anchor"]["hyperparameters"], "loss": {"mode": "hard"}}
                print(f"[v3] anchor seed={seed}", flush=True)
                anchor = train_research_mlp(roles["fit"], validation, anchor_config, seed, schema=schema, device=device, threads=config["threads"])
                save_bundle(anchor, root / "anchors" / f"s{seed}.joblib")
                write_json(root / "anchors" / f"s{seed}.json", anchor.training_statistics)
                for condition in config["conditions"]:
                    records.append(_student(roles["fit"], roles["constraint"], roles["reference"], validation, anchor, teacher,
                                            config, seed, condition, root / "students" / f"{condition}-s{seed}", device))
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
    if code_fingerprint()["source_sha256"] != binding["source"]["source_sha256"]:
        raise ValueError("source changed during research; preserve incomplete artifacts and run a frozen checkout")
    if verify_development_inputs(data_dir) != binding["data"]:
        raise ValueError("development data changed during research")
    write_json(root / "summary.json", {"status": "development_only_not_independent_confirmation", "runs": records,
        "n_eligible": sum(x["status"] == "eligible_development_candidate" for x in records),
        "limitations": ["historical test/calibration not accessed", "no population FPR guarantee", "adaptations are not full author reproductions",
                        "pAUC and stronger independently tuned tabular baselines remain separate work", "no new innovation/effect claim from synthetic validation"]})
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    complete = {"status": "development_complete", "created_at": utc_now(), "reserved_partitions_read": [],
                "n_students": len(records), "n_teacher_fits": (config["folds"] + 1) * len(config["seeds"]),
                "n_anchor_fits": len(config["seeds"]), "files": files, "plan_id": plan["plan_id"]}
    write_json(root / COMPLETE, complete)
    return verify_risk_research(root)


def verify_risk_research(output):
    root = Path(output).resolve()
    result = read_json(root / COMPLETE)
    if result.get("status") != "development_complete" or result.get("reserved_partitions_read") != []:
        raise ValueError("not completed development-only research")
    files = result.get("files", {})
    inventory = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != COMPLETE}
    if not files or set(files) != inventory:
        raise ValueError("artifact inventory mismatch")
    for name, digest in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"artifact mismatch: {name}")
    plan = read_json(root / "plan.json")
    config = plan["config"]
    validate_risk_config(config)
    if plan["plan_id"] != canonical_hash({k: v for k, v in plan.items() if k != "plan_id"}) or result.get("plan_id") != plan["plan_id"]:
        raise ValueError("plan identity mismatch")
    if canonical_hash(read_json(root / "roles.json")) != plan["role_manifest_hash"]:
        raise ValueError("role manifest mismatch")
    expected = {(c, s) for c in config["conditions"] for s in config["seeds"]}
    runs = read_json(root / "summary.json")["runs"]
    if len(runs) != len(expected) or {(r["condition"], r["seed"]) for r in runs} != expected or result["n_students"] != len(expected) or result["n_teacher_fits"] != (config["folds"] + 1) * len(config["seeds"]) or result.get("n_anchor_fits") != len(config["seeds"]):
        raise ValueError("candidate/seed fit coverage mismatch")
    required = {"roles.json", "plan.json", "summary.json", "provenance.json", "environment.json"}
    required.update(f"{role}-identity.parquet" for role in ("fit", "reference", "constraint", "validation"))
    required.update("source/" + name for name in plan["source"]["files"])
    for condition, seed in expected:
        prefix = f"students/{condition}-s{seed}/"
        required.update(prefix + name for name in ("model.joblib", "run.json", "history.json", "dual.json", "validation-predictions.parquet"))
        if read_json(root / (prefix + "run.json")) != next(r for r in runs if r["condition"] == condition and r["seed"] == seed):
            raise ValueError("run/summary mismatch")
    for seed in config["seeds"]:
        required.update(f"anchors/s{seed}{suffix}" for suffix in (".joblib", ".json"))
        required.update(f"teachers/s{seed}/{name}" for name in ("reference.joblib", "oof.json", "targets.parquet"))
        required.update(f"teachers/s{seed}/fold-{k}.joblib" for k in range(config["folds"]))
    if not required.issubset(files):
        raise ValueError("required artifacts missing")
    for name, digest in plan["source"]["files"].items():
        if files["source/" + name] != digest:
            raise ValueError("source snapshot mismatch")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_risk_research(args.output)
    else:
        if not args.data_dir or not args.config:
            parser.error("--data-dir and --config are required")
        result = run_risk_research(args.data_dir, yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), args.output,
                                   device=args.device, resume=args.resume)
    print({k: v for k, v in result.items() if k != "files"})


if __name__ == "__main__":
    main()
