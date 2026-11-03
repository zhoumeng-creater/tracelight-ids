"""V4 mechanism study: correct KD coordinates, working points, finite-C updates.

Reuses VERIFIED v3 roles/teachers/anchors, never fits a new teacher or reads
historical calibration/test. R recalibration is a declared evaluation regime,
not post-hoc validation threshold tuning. No population risk certificate.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
import yaml

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import load_partition
from .models import ModelBundle, _network, load_bundle, save_bundle
from .research import _identity
from .risk_baselines import adapted_ekd_rank_count
from .risk_distillation import BoundaryKDConfig, build_risk_reference, boundary_kd_objective, constraint_risks, freeze_margin_threshold
from .risk_research import _constraints, _fixed_metrics, _infer, _state, _state_hash, verify_risk_research
from .runner import verify_development_inputs
from .upgrade_guard import guarded_adamw_step
from .upgrade_kd import raw_bernoulli_kl_rows

COMPLETE = "UPGRADE_COMPLETE.json"
CONDITIONS = ("hard", "legacy_boundary_kd", "legacy_ekd", "raw_kd", "balanced_raw_kd", "guarded_hard",
              "guarded_raw_kd", "guarded_balanced_raw_kd")


def validate_upgrade_config(config):
    keys = {"schema_version", "purpose", "seeds", "threads", "max_fpr", "primary_working_point", "conditions", "optimization"}
    if not isinstance(config, dict) or set(config) != keys or type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("upgrade configuration schema/keys mismatch")
    if config["purpose"] != "upgrade_v4_development_only" or config["primary_working_point"] != "recalibrated_reference":
        raise ValueError("V4 requires the declared R-recalibrated development regime")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("threads must be positive integer")
    if isinstance(config["max_fpr"], bool) or not np.isfinite(config["max_fpr"]) or not 0 < config["max_fpr"] < 1:
        raise ValueError("max_fpr must be in (0,1)")
    conditions = config["conditions"]
    if not isinstance(conditions, list) or not conditions or any(x not in CONDITIONS for x in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError("unknown/duplicate conditions")
    op = config["optimization"]
    if not isinstance(op, dict) or set(op) != {"macro_steps", "phase1_steps", "chunk_size", "learning_rate", "weight_decay", "temperature",
            "kd_weight", "gradient_clip", "max_backtracks", "shrink", "min_family_support", "attack_loss_limit"}:
        raise ValueError("optimization schema mismatch")
    for key in ("macro_steps", "phase1_steps", "chunk_size", "max_backtracks", "min_family_support"):
        if type(op[key]) is not int or op[key] < (0 if key in {"phase1_steps", "max_backtracks"} else 1):
            raise ValueError(f"invalid {key}")
    for key in ("learning_rate", "weight_decay", "temperature", "kd_weight", "gradient_clip", "shrink", "attack_loss_limit"):
        value = op[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid {key}")
    if min(op["learning_rate"], op["temperature"], op["gradient_clip"]) <= 0 or not 0 < op["shrink"] < 1 or op["attack_loss_limit"] > 1:
        raise ValueError("invalid positive scale/budget")


def _no_links(path):
    target = Path(path).absolute()
    for value in (target, *target.parents):
        if value.is_symlink() or (hasattr(value, "is_junction") and value.is_junction()):
            raise ValueError("linked artifact/data paths are not accepted")
    return target.resolve()


def _bind(v3_dir, data_dir, config, device):
    v3, data = _no_links(v3_dir), _no_links(data_dir)
    complete = verify_risk_research(v3)
    plan = read_json(v3 / "plan.json")
    manifest = verify_development_inputs(data)
    if manifest != plan["data"]:
        raise ValueError("prepared data differs from verified v3 source")
    if plan["config"]["cohort_policy"] != "complete_development":
        raise ValueError("V4 reuses complete v3 cohorts, not a capped pilot")
    if not set(config["seeds"]).issubset(plan["config"]["seeds"]):
        raise ValueError("missing source teacher/anchor seed")
    if (config["max_fpr"] != plan["config"]["max_fpr"] or any(config["optimization"][k] != plan["config"]["optimization"][k]
            for k in ("attack_loss_limit", "min_family_support"))):
        raise ValueError("do not change the frozen v3 empirical risk/support budgets")
    return {"schema_version": 1, "config": config, "source": code_fingerprint(), "device": device,
            "v3_dir": str(v3), "v3_complete_sha256": sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json"),
            "v3_plan_id": complete["plan_id"], "data_dir": str(data), "data": manifest}


def _cohorts(v3, data):
    train, validation = (load_partition(data, split) for split in ("train", "validation"))
    if train.sample_id.duplicated().any() or validation.sample_id.duplicated().any():
        raise ValueError("source IDs are not unique")
    result = {}
    for role in ("fit", "reference", "constraint", "validation"):
        identity = pd.read_parquet(v3 / f"{role}-identity.parquet")
        source = validation if role == "validation" else train
        indexed = source.set_index("sample_id", drop=False)
        try:
            selected = indexed.loc[identity.sample_id.tolist()].reset_index(drop=True)
        except KeyError as exc:
            raise ValueError("source role IDs missing from prepared data") from exc
        # Preserve saved order and ALL identity columns, including family labels.
        if not selected.loc[:, identity.columns].equals(identity.reset_index(drop=True)):
            raise ValueError("source role identity changed")
        result[role] = selected
    all_ids = [str(x) for role in ("fit", "reference", "constraint") for x in result[role].sample_id]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(train.sample_id.astype(str)):
        raise ValueError("source training roles are not an exhaustive disjoint partition")
    if set(all_ids) & set(result["validation"].sample_id.astype(str)):
        raise ValueError("validation role overlaps training")
    return result


def empirical_constraints(logits, threshold, reference):
    """Exact integer event budgets; surrogate is reported without selecting it."""
    values = np.asarray(logits, dtype=np.float64)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.number)) or not np.isfinite(threshold):
        raise ValueError("constraint threshold must be a finite numeric scalar")
    if values.shape != (len(reference.sample_ids),) or not np.isfinite(values).all():
        raise ValueError("invalid constraint logits")
    decision = values >= float(threshold)
    baseline = reference.anchor_margins.numpy() >= 0
    metrics = {}
    for spec, membership in zip(reference.constraints, reference.members):
        mask = membership.numpy()
        support = int(mask.sum())
        count = int(decision[mask].sum()) if spec.kind == "normal_fpr" else int((baseline & ~decision)[mask].sum())
        allowed = math.floor(spec.limit * support)
        metrics[spec.name] = {"kind": spec.kind, "count": count, "support": support, "allowed_count": allowed,
                              "actual": count / support, "limit": spec.limit, "feasible": count <= allowed}
        if spec.kind == "attack_loss":
            metrics[spec.name]["baseline_tp_count"] = int(baseline[mask].sum())
    return {"feasible": all(x["feasible"] for x in metrics.values()), "constraints": metrics}


def _cpu_copy(network, dimension):
    import torch
    # Loading an existing model must not consume the training RNG just because
    # constructing the temporary Sequential happens to initialize parameters.
    with torch.random.fork_rng(devices=[]):
        cpu = _network(dimension, network._upgrade_hidden)
    cpu.load_state_dict(_state(network))
    return cpu


def _oracle(network, xs, frames, anchor_threshold, risk_reference, alpha, chunk):
    """R and complete C ONLY. CPU deployment arithmetic is the acceptance oracle."""
    import torch
    cpu = _cpu_copy(network, xs["fit"].shape[1])
    r = _infer(cpu, xs["reference"], "cpu", chunk)
    c = _infer(cpu, xs["constraint"], "cpu", chunk)
    wp = freeze_margin_threshold(torch.tensor(r), torch.tensor(frames["reference"].label.to_numpy()), alpha)
    empirical = empirical_constraints(c, wp["threshold"], risk_reference)
    proxy = constraint_risks(torch.tensor(c - wp["threshold"]), risk_reference, sample_ids=risk_reference.sample_ids)
    frozen = empirical_constraints(c, anchor_threshold, risk_reference)
    return {"feasible": empirical["feasible"], "metrics": {"working_point": wp, "actual": empirical["constraints"],
            "surrogate": proxy.detached_metrics(), "surrogate_feasible": bool((proxy.violations <= 0).all()),
            "frozen_anchor_threshold_actual": frozen, "oracle": "CPU complete C with candidate R threshold; no V access"}}


def _validation(network, xs, frames, threshold, anchor_threshold, chunk):
    cpu = _cpu_copy(network, xs["fit"].shape[1])
    z = _infer(cpu, xs["validation"], "cpu", chunk)
    return {"primary": _fixed_metrics(frames["validation"], z, threshold),
            "frozen_anchor": _fixed_metrics(frames["validation"], z, anchor_threshold)}, z


def _network_from(anchor, device, state=None):
    network = _network(anchor.encoded_dim, anchor.hidden_dims).to(device)
    network._upgrade_hidden = anchor.hidden_dims
    network.load_state_dict(anchor.network_state if state is None else state)
    return network


def _grad_vector(loss, parameters):
    import torch
    gradient = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu().double() for g, p in zip(gradient, parameters)])


def _backward(network, xs, frames, teacher, config, mode, anchor_threshold, *, diagnostics=False):
    """Full F+C hard objective; global F/class denominators even across folds."""
    import torch
    import torch.nn.functional as F
    op, chunk = config["optimization"], config["optimization"]["chunk_size"]
    device = next(network.parameters()).device
    fit, constraint = frames["fit"], frames["constraint"]
    n0, n1 = int((fit.label == 0).sum()), int((fit.label == 1).sum())
    total = len(fit) + len(constraint)
    weight = (n0 + int((constraint.label == 0).sum())) / (n1 + int((constraint.label == 1).sum()))
    pos_weight = torch.tensor(weight, device=device)
    parameters = tuple(network.parameters())
    vectors = {"hard": torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64),
               "soft": torch.zeros(sum(p.numel() for p in parameters), dtype=torch.float64)} if diagnostics else None
    totals = {"hard": 0., "soft": 0.}
    mode = mode.removeprefix("guarded_")
    for role in ("fit", "constraint"):
        frame = frames[role]
        if role == "fit":
            partitions = [np.flatnonzero(teacher["fold"] == k) for k in range(len(teacher["grids"]))]
        else:
            partitions = [np.arange(len(frame))]
        for fold, indices in enumerate(partitions):
            for offset in range(0, len(indices), chunk):
                ix = indices[offset:offset + chunk]
                z = network(torch.from_numpy(xs[role][ix]).to(device)).squeeze(1)
                y = torch.tensor(frame.label.to_numpy()[ix], dtype=torch.float32, device=device)
                hard = F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight, reduction="sum") / total
                soft = z.sum() * 0
                if role == "fit" and mode in {"raw_kd", "balanced_raw_kd"}:
                    rows = raw_bernoulli_kl_rows(z, teacher_logits=torch.tensor(teacher["logits"][ix], device=device), temperature=op["temperature"])
                    if mode == "balanced_raw_kd":
                        soft = op["kd_weight"] * .5 * (rows[y == 0].sum() / n0 + rows[y == 1].sum() / n1)
                    else:
                        soft = op["kd_weight"] * rows.sum() / len(fit)
                elif role == "fit" and mode == "legacy_boundary_kd":
                    kd = BoundaryKDConfig(mode="kd", kd_weight=op["kd_weight"], temperature=op["temperature"])
                    soft = op["kd_weight"] * boundary_kd_objective(z, y, kd, student_threshold=anchor_threshold,
                        teacher_margins=torch.tensor(teacher["margins"][ix], device=device))["soft"] * len(ix) / len(fit)
                elif role == "fit" and mode == "legacy_ekd":
                    soft = adapted_ekd_rank_count(z, y, teacher_logits=torch.tensor(teacher["logits"][ix], device=device),
                        student_thresholds=teacher["anchor_grid"], teacher_thresholds=teacher["grids"][fold].tolist(),
                        temperature=teacher["ekd_temperature"], normal_coefficient=int((y == 0).sum()) / n0,
                        attack_coefficient=int((y == 1).sum()) / n1, coefficient=op["kd_weight"])["soft"]
                loss = hard + soft
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite upgrade loss")
                if diagnostics:
                    vectors["hard"] += _grad_vector(hard, parameters)
                    vectors["soft"] += _grad_vector(soft, parameters)
                loss.backward()
                totals["hard"] += float(hard.detach())
                totals["soft"] += float(soft.detach())
    if diagnostics:
        h, s = vectors["hard"], vectors["soft"]
        hn, sn = float(h.norm()), float(s.norm())
        totals["gradient_diagnostics"] = {"hard_norm": hn, "soft_norm": sn, "cosine": float(h.dot(s) / (hn * sn)) if hn and sn else None,
            "soft_to_hard_norm": sn / hn if hn else None, "hard_final_bias": float(h[-1]), "soft_final_bias": float(s[-1]),
            "scope": "full objectives, before clipping/AdamW; loss constants do not imply gradient dominance"}
    return totals


def _teacher(v3, seed, frames, anchor, xs, device, config):
    import torch
    targets = pd.read_parquet(v3 / f"teachers/s{seed}/targets.parquet")
    identity = _identity(frames["fit"])
    if not identity.equals(targets.loc[:, identity.columns]):
        raise ValueError("teacher target row alignment changed")
    metadata = read_json(v3 / f"teachers/s{seed}/oof.json")
    folds = targets.fold.to_numpy(dtype=int)
    grids = np.asarray([x["reference_threshold_grid"] for x in metadata["folds"]], dtype=np.float64)
    if set(folds) != set(range(len(grids))):
        raise ValueError("teacher fold coverage changed")
    net = _network_from(anchor, device)
    source_plan = read_json(v3 / "plan.json")
    source_chunk = source_plan["config"]["optimization"]["chunk_size"]
    original_c = _infer(net, xs["constraint"], device, source_chunk)
    original_r = _infer(net, xs["reference"], device, source_chunk)
    anchor_threshold = read_json(v3 / f"students/hard-s{seed}/run.json")["working_point"]["threshold"]
    rates = metadata["rates"]
    anchor_grid = [freeze_margin_threshold(torch.tensor(original_r), torch.tensor(frames["reference"].label.to_numpy()), rate)["threshold"] for rate in rates]
    teacher = {"logits": targets.teacher_logit.to_numpy(dtype=np.float64), "margins": targets.teacher_margin.to_numpy(dtype=np.float64),
               "fold": folds, "grids": grids, "anchor_grid": anchor_grid,
               "ekd_temperature": source_plan["config"]["optimization"]["ekd_temperature"]}
    diagnostic = {"cached_teacher_targets_sha256": sha256_file(v3 / f"teachers/s{seed}/targets.parquet"),
                  "representation": "cached clipped v3 raw logits; raw KD never subtracts alarm thresholds",
                  "teacher_is_class_weighted": True, "fold_positive_weights": [x["fit"]["labels"]["0"] / x["fit"]["labels"]["1"] for x in metadata["folds"]],
                  "threshold_grids": grids.tolist(), "unique_thresholds_per_fold": [len(set(x)) for x in grids.tolist()], "classes": {}}
    for label in (0, 1):
        mask = targets.label.to_numpy() == label
        q = expit(teacher["margins"][mask])
        diagnostic["classes"][str(label)] = {"n": int(mask.sum()), "boundary_target_mean": float(q.mean()),
            "boundary_target_half_fraction": float(np.isclose(q, .5, atol=1e-12, rtol=0).mean()),
            "raw_prediction_mean": float(expit(teacher["logits"][mask]).mean())}
    return teacher, anchor_threshold, original_c, diagnostic


def _fit_condition(anchor, warm_state, frames, xs, teacher, risk_reference, anchor_threshold, config, condition, seed, output, device):
    import torch
    op = config["optimization"]
    chunk = op["chunk_size"]
    network = _network_from(anchor, device, warm_state)
    optimizer = torch.optim.AdamW(network.parameters(), lr=op["learning_rate"], weight_decay=op["weight_decay"])
    output.mkdir(parents=True, exist_ok=False)
    def evaluate():
        return _oracle(network, xs, frames, anchor_threshold, risk_reference, config["max_fpr"], chunk)
    current = evaluate()
    if not current["feasible"]:
        raise ValueError("shared warm start is not C-feasible")
    history, best_state, selected, best_rank, best_oracle = [], None, None, None, None
    started = time.perf_counter()
    accepted, accepted_moving = 0, 0
    warm_validation = None
    for step in range(op["macro_steps"] + 1):
        update, losses = None, None
        if step:
            network.train()
            optimizer.zero_grad(set_to_none=True)
            losses = _backward(network, xs, frames, teacher, config, condition, anchor_threshold, diagnostics=step == 1)
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), op["gradient_clip"], error_if_nonfinite=True)
            losses["gradient_norm_before_clip"] = float(norm)
            if condition.startswith("guarded_"):
                update = guarded_adamw_step(network, optimizer, evaluate=evaluate, max_backtracks=op["max_backtracks"], shrink=op["shrink"])
                accepted += int(update["accepted"])
                accepted_moving += int(update["accepted"] and update["moved"])
                current = evaluate()
                if not current["feasible"]:
                    raise ValueError("guard committed a C-infeasible update")
            else:
                old = _state(network)
                optimizer.step()
                accepted += 1
                accepted_moving += int(any(not torch.equal(value, old[name]) for name, value in _state(network).items()))
                current = evaluate()
        v, _ = _validation(network, xs, frames, current["metrics"]["working_point"]["threshold"], anchor_threshold, chunk)
        if step == 0:
            warm_validation = v
        eligible = current["feasible"] and v["primary"]["fpr"] <= config["max_fpr"]
        rank = (v["primary"]["tpr"], -v["primary"]["fpr"], -step)
        if eligible and (best_rank is None or rank > best_rank):
            best_state, selected, best_rank, best_oracle = _state(network), step, rank, copy.deepcopy(current)
        history.append({"step": step, "losses": losses, "constraints": current, "validation": v,
                        "eligible": eligible, "update": update})
        write_json(output / "history.json", history)
        if step == 0 or step % 10 == 0 or step == op["macro_steps"]:
            print(f"[v4] {condition} seed={seed} step={step} C={current['feasible']} V_fpr={v['primary']['fpr']:.5f} frozenV_fpr={v['frozen_anchor']['fpr']:.5f}", flush=True)
    if best_state is None:
        best_state = _state(network)
        best_oracle = current
    network.load_state_dict(best_state)
    final = evaluate()
    if final != best_oracle:
        raise ValueError("selected model/threshold/constraint oracle is not reproducible")
    v, raw = _validation(network, xs, frames, final["metrics"]["working_point"]["threshold"], anchor_threshold, chunk)
    model = ModelBundle("mlp", anchor.preprocessor, anchor.selected_features, anchor.encoded_dim, [], network_state=best_state,
        hidden_dims=anchor.hidden_dims, device="cpu", batch_size=chunk, num_threads=config["threads"], feature_schema=anchor.feature_schema,
        best_epoch=selected, seed=seed)
    save_bundle(model, output / "model.joblib")
    prediction = frames["validation"][["sample_id", "feature_hash", "label", "attack_cat"]].copy()
    prediction["cpu_raw_logit"] = raw
    prediction["raw_logit_threshold"] = final["metrics"]["working_point"]["threshold"]
    prediction["predicted_label"] = (raw >= prediction.raw_logit_threshold.to_numpy()).astype(int)
    prediction.to_parquet(output / "validation-predictions.parquet", index=False)
    record = {"condition": condition, "seed": seed, "status": "eligible_development_candidate" if selected is not None else "diagnostic_no_eligible_checkpoint",
        "selected_step": selected, "accepted_updates": accepted, "attempted_updates": op["macro_steps"], "selected_is_step0": selected == 0,
        "accepted_updates_with_parameter_movement": accepted_moving,
        "common_initial_state_sha256": _state_hash(warm_state), "selected_state_sha256": _state_hash(best_state),
        "working_point": final["metrics"]["working_point"], "working_point_regime": config["primary_working_point"],
        "anchor_threshold": anchor_threshold, "constraints": final, "validation": v, "warm_validation": warm_validation,
        "validation_tpr_delta_from_warm": v["primary"]["tpr"] - warm_validation["primary"]["tpr"],
        "selection_rule": "same actual C budgets + V FPR for ALL conditions, highest primary V TPR, min FPR, earliest step including zero",
        "surrogate_is_diagnostic_not_v3_eligibility": True, "reserved_partitions_read": [], "teacher_refits": 0, "actual_training_device": device,
        "training_seconds": time.perf_counter() - started, "package_bytes": (output / "model.joblib").stat().st_size,
        "guard_scope": "CPU finite C only; candidate R recalibration is part of each proposal; no V in acceptance",
        "method_status": "standard controls and empirical step filter, not proven new KD mechanism"}
    write_json(output / "run.json", record)
    return record


def run_upgrade_research(v3_dir, data_dir, config, output, *, device="cuda", threads=None, resume=False):
    import torch
    config = copy.deepcopy(config)
    if threads is not None:
        config["threads"] = threads
    validate_upgrade_config(config)
    root, v3, data = _no_links(output), _no_links(v3_dir), _no_links(data_dir)
    if root.is_relative_to(v3) or root.is_relative_to(data) or v3.is_relative_to(root) or data.is_relative_to(root):
        raise ValueError("new output and source/data must be separate non-nested directories")
    binding = _bind(v3, data, config, device)
    if root.exists():
        if not resume:
            raise FileExistsError("new V4 output required; existing artifacts are preserved")
        complete = verify_upgrade_research(root)
        if read_json(root / "plan.json")["binding"] != binding:
            raise ValueError("resume source/config/data/v3/device identity differs")
        return complete
    if binding["data"].get("data_kind") != "synthetic":
        if device not in {"cuda", "cuda:0"} or not torch.cuda.is_available() or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper():
            raise ValueError("real upgrade fits require laboratory CUDA0 RTX4090D")
    frames = _cohorts(v3, data)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {"binding": binding, "binding_sha256": canonical_hash(binding), "reserved_partitions_read": []})
    write_json(root / "environment.json", environment_snapshot())
    for name in ("plan.json", "roles.json", "RISK_RESEARCH_COMPLETE.json"):
        (root / "v3-evidence").mkdir(exist_ok=True)
        shutil.copyfile(v3 / name, root / "v3-evidence" / name)
    source_root = Path(__file__).resolve().parents[2]
    for name in binding["source"]["files"]:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / name, target)
    # Identity only; no prepared features or old teacher/model packages copied.
    for role in frames:
        shutil.copyfile(v3 / f"{role}-identity.parquet", root / f"{role}-identity.parquet")
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    op, records, phase_records = config["optimization"], [], []
    try:
        with threadpool_limits(limits=config["threads"]):
            for seed in config["seeds"]:
                phase_started = time.perf_counter()
                anchor = load_bundle(v3 / f"anchors/s{seed}.joblib", device="cpu")
                xs = {k: anchor.preprocessor.transform(frame) for k, frame in frames.items()}
                teacher, anchor_threshold, original_c, teacher_diagnostic = _teacher(v3, seed, frames, anchor, xs, device, config)
                write_json(root / f"teacher-diagnostics-s{seed}.json", teacher_diagnostic)
                families = set().union(*(set(frames[r].loc[frames[r].label == 1, "attack_cat"]) for r in ("fit", "constraint", "reference")))
                specs, family_vector, support = _constraints(frames["constraint"], {"max_fpr": config["max_fpr"], "optimization": op}, families)
                reference = build_risk_reference(torch.tensor(original_c - anchor_threshold), torch.tensor(frames["constraint"].label.to_numpy()), specs,
                    sample_ids=tuple(frames["constraint"].sample_id.astype(str)), attack_families=torch.tensor(family_vector), margin_floor=.05)
                network = _network_from(anchor, device)
                optimizer = torch.optim.AdamW(network.parameters(), lr=op["learning_rate"], weight_decay=op["weight_decay"])
                phase = []
                current = _oracle(network, xs, frames, anchor_threshold, reference, config["max_fpr"], op["chunk_size"])
                phase.append({"step": 0, "constraints": current})
                print(f"[v4] shared phase-I seed={seed} step=0 C_feasible={current['feasible']}", flush=True)
                for step in range(1, op["phase1_steps"] + 1):
                    if current["feasible"]:
                        break
                    network.train()
                    optimizer.zero_grad(set_to_none=True)
                    _backward(network, xs, frames, teacher, config, "hard", anchor_threshold)
                    torch.nn.utils.clip_grad_norm_(network.parameters(), op["gradient_clip"], error_if_nonfinite=True)
                    optimizer.step()
                    current = _oracle(network, xs, frames, anchor_threshold, reference, config["max_fpr"], op["chunk_size"])
                    phase.append({"step": step, "constraints": current})
                    if current["feasible"] or step % 10 == 0 or step == op["phase1_steps"]:
                        print(f"[v4] shared phase-I seed={seed} step={step} C_feasible={current['feasible']}", flush=True)
                phase_record = {"seed": seed, "status": "feasible_start" if current["feasible"] else "no_feasible_start",
                    "updates": len(phase) - 1, "history": phase, "family_support": support,
                    "phase1_and_reused_artifact_setup_seconds": time.perf_counter() - phase_started,
                    "baseline_event_source": "original v3 anchor/threshold on original training device; not the warm-start predictions",
                    "warm_state_sha256": _state_hash(_state(network)), "no_validation_used": True}
                phase_records.append(phase_record)
                write_json(root / f"phase1-s{seed}.json", phase_record)
                for condition in config["conditions"]:
                    out = root / "students" / f"{condition}-s{seed}"
                    if not current["feasible"]:
                        out.mkdir(parents=True, exist_ok=False)
                        record = {"condition": condition, "seed": seed, "status": "no_feasible_start", "reserved_partitions_read": []}
                        write_json(out / "run.json", record)
                    else:
                        record = _fit_condition(anchor, _state(network), frames, xs, teacher, reference, anchor_threshold,
                                                config, condition, seed, out, device)
                    records.append(record)
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
    if code_fingerprint()["source_sha256"] != binding["source"]["source_sha256"] or verify_development_inputs(data) != binding["data"]:
        raise ValueError("source/development data changed during run; preserve incomplete artifacts")
    # Old artifact bytes must also remain identical, not just their metadata path.
    verify_risk_research(v3)
    if sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json") != binding["v3_complete_sha256"]:
        raise ValueError("v3 artifact changed during run")
    write_json(root / "summary.json", {"scope": "development_only_no_independent_confirmation", "runs": records,
        "n_eligible": sum(r["status"] == "eligible_development_candidate" for r in records),
        "limitations": ["working point regime changed explicitly; v3 conclusions unchanged", "only finite adaptive training C is protected",
                        "guard may retain step0 without improvement", "pAUC/stronger tuned baselines and new-source confirmation remain unfinished"]})
    skipped = sum(r["status"] == "no_feasible_start" for r in records)
    complete = {"status": "development_complete", "created_at": utc_now(), "reserved_partitions_read": [], "n_students_planned": len(records),
        "n_students": len(records) - skipped, "n_skipped_no_feasible_start": skipped, "n_teacher_fits": 0, "n_anchor_fits": 0,
        "binding_sha256": canonical_hash(binding), "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    write_json(root / COMPLETE, complete)
    return verify_upgrade_research(root)


def verify_upgrade_research(output):
    root = _no_links(output)
    complete = read_json(root / COMPLETE)
    if complete.get("status") != "development_complete" or complete.get("reserved_partitions_read") != []:
        raise ValueError("not a completed development upgrade artifact")
    files = complete.get("files", {})
    inventory = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / COMPLETE}
    if not files or set(files) != inventory:
        raise ValueError("upgrade inventory mismatch")
    for name, digest in files.items():
        target = _no_links(root / name)
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"upgrade artifact mismatch: {name}")
    plan = read_json(root / "plan.json")
    binding = plan["binding"]
    validate_upgrade_config(binding["config"])
    if canonical_hash(binding) != plan["binding_sha256"] or complete["binding_sha256"] != plan["binding_sha256"] or plan["reserved_partitions_read"] != []:
        raise ValueError("upgrade binding mismatch")
    config = binding["config"]
    expected = {(c, seed) for c in config["conditions"] for seed in config["seeds"]}
    runs = read_json(root / "summary.json")["runs"]
    if len(runs) != len(expected) or {(r["condition"], r["seed"]) for r in runs} != expected:
        raise ValueError("upgrade coverage mismatch")
    skipped = sum(r["status"] == "no_feasible_start" for r in runs)
    if (complete["n_students_planned"] != len(expected) or complete["n_students"] != len(expected) - skipped
            or complete["n_skipped_no_feasible_start"] != skipped or complete["n_teacher_fits"] != 0 or complete["n_anchor_fits"] != 0):
        raise ValueError("upgrade fitting count mismatch")
    required = {"plan.json", "summary.json", "environment.json", "v3-evidence/plan.json", "v3-evidence/roles.json", "v3-evidence/RISK_RESEARCH_COMPLETE.json"}
    required.update(f"{role}-identity.parquet" for role in ("fit", "reference", "constraint", "validation"))
    for seed in config["seeds"]:
        required.update((f"teacher-diagnostics-s{seed}.json", f"phase1-s{seed}.json"))
    for run in runs:
        prefix = f"students/{run['condition']}-s{run['seed']}/"
        required.add(prefix + "run.json")
        if read_json(root / (prefix + "run.json")) != run:
            raise ValueError("upgrade summary/run mismatch")
        if run["status"] != "no_feasible_start":
            required.update(prefix + name for name in ("model.joblib", "history.json", "validation-predictions.parquet"))
    for name, digest in binding["source"]["files"].items():
        required.add("source/" + name)
        if files.get("source/" + name) != digest:
            raise ValueError("upgrade source snapshot mismatch")
    if not required.issubset(files) or files["v3-evidence/RISK_RESEARCH_COMPLETE.json"] != binding["v3_complete_sha256"]:
        raise ValueError("upgrade required artifacts/source reference mismatch")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_upgrade_research(args.output)
    else:
        if not all((args.v3_dir, args.data_dir, args.config)):
            parser.error("--v3-dir, --data-dir, --config required")
        result = run_upgrade_research(args.v3_dir, args.data_dir, yaml.safe_load(Path(args.config).read_text(encoding="utf-8")), args.output,
                                     device=args.device, resume=args.resume)
    print({k: v for k, v in result.items() if k != "files"})


if __name__ == "__main__":
    main()
