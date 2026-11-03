"""Run a CPU-only synthetic engineering check of the V3 risk-training chain.

No dataset paths or reserved-data options exist. All four roles are generated
in memory. This checks finite arithmetic, empirical bounds and state identity;
it does not test detection performance, low-tail validity or a foundation model.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import pandas as pd
import torch
from torch import nn


# Support direct execution without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids import risk_data, risk_distillation  # noqa: E402
from ids.risk_distillation import (  # noqa: E402
    BoundaryKDConfig,
    ProjectedDual,
    RiskConstraint,
    boundary_kd_objective,
    build_risk_reference,
    constraint_risks,
    freeze_margin_threshold,
    risk_lagrangian,
    working_margins,
)


SCOPE = "synthetic_engineering_only"
FAMILIES = ("toy_attack_a", "toy_attack_b", "toy_attack_c")
MAX_FPR = .15  # Engineering scale only; this is NOT the formal 1% operating point.
MARGIN_FLOOR = .25


def _integer(value, *, minimum):
    number = int(value)
    if number < minimum:
        raise argparse.ArgumentTypeError(f"must be at least {minimum}")
    return number


def _sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_sha(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    # Every path is inside a newly-created output directory; exclusive writes
    # also prevent an accidental overwrite if another process touches it.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _model():
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1)).cpu()


def _logits(model, features):
    return model(features).squeeze(-1)


def _teacher_score(features):
    """An explicitly known synthetic function, not a trained/TFM teacher."""
    x = features
    return (1.3 * x[:, 0] + 1.0 * x[:, 1] + .7 * x[:, 2]
            + .4 * torch.sin(x[:, 3]) + .15 * x[:, 4] * x[:, 5] - 1.0)


def _generate_role(name, normals, per_family, generator):
    count = normals + len(FAMILIES) * per_family
    features = torch.randn(count, 6, generator=generator, device="cpu")
    labels = torch.zeros(count, dtype=torch.int64)
    family_ids = torch.full((count,), -1, dtype=torch.int64)
    categories = ["toy_normal"] * normals
    for family in range(len(FAMILIES)):
        start = normals + family * per_family
        stop = start + per_family
        features[start:stop, family] += 2.1
        features[start:stop, (family + 1) % 3] += .5
        labels[start:stop] = 1
        family_ids[start:stop] = family
        categories.extend([FAMILIES[family]] * per_family)
    hashes = [hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in features]
    frame = pd.DataFrame({
        "sample_id": [f"synthetic-{name}-{i:05d}" for i in range(count)],
        "feature_hash": hashes,
        "label": labels.numpy(),
        "attack_cat": categories,
        "group_id": [f"synthetic-isolation-{name}-{i:05d}" for i in range(count)],
    })
    return {"x": features, "y": labels, "families": family_ids, "identity": frame}


def _freeze_threshold(scores, labels):
    return freeze_margin_threshold(scores, labels, max_fpr=MAX_FPR, partition="reference")


def _pointwise(anchor, student, floor):
    a, m = anchor.detach().double(), student.detach().double()
    b, s = a >= 0, m >= 0
    divisor = a.abs().clamp_min(floor)
    normal_upper = (b.double() + (~b) * torch.relu(1 + m / divisor)
                    - b * (-m / floor).clamp(0, 1))
    attack_upper = b * torch.relu(1 - m / divisor)
    normal_gap = normal_upper - s.double()
    attack_gap = attack_upper - (b & ~s).double()
    finite = bool(torch.isfinite(normal_upper).all() and torch.isfinite(attack_upper).all())
    result = {"rows_checked_for_each_event": len(a), "finite": finite,
              "normal_bound_violations": int((normal_gap < -1e-12).sum()),
              "attack_bound_violations": int((attack_gap < -1e-12).sum()),
              "normal_min_gap": float(normal_gap.min()), "attack_min_gap": float(attack_gap.min())}
    if not finite or result["normal_bound_violations"] or result["attack_bound_violations"]:
        raise RuntimeError("synthetic pointwise event bound failed")
    return result


def _snapshot(step, model, pool, reference, threshold, dual, *, losses=None):
    with torch.no_grad():
        margins = working_margins(_logits(model, pool["x"]), threshold=threshold)
        report = constraint_risks(margins, reference, sample_ids=reference.sample_ids)
        pointwise = _pointwise(reference.anchor_margins, margins, reference.margin_floor)
    covered = bool((report.actual_values <= report.values + 1e-12).all())
    if not covered:
        raise RuntimeError("full-cohort empirical bound failed")
    finite = all(bool(torch.isfinite(value).all()) for value in model.state_dict().values())
    if not finite:
        raise RuntimeError("student parameters are non-finite")
    limits = report.values.new_tensor([spec.limit for spec in reference.constraints])
    return {"step": step, "parameters_finite": finite, "pointwise_bounds": pointwise,
            "margin_dtype": str(margins.dtype),
            "aggregate_bounds_hold": covered, "metrics": report.detached_metrics(),
            "surrogate_budget_satisfied": bool((report.violations <= 1e-12).all()),
            "actual_budget_satisfied": bool((report.actual_values <= limits + 1e-12).all()),
            "dual_weights": dual.weights.tolist(), "dual_updates": dual.updates,
            "dual_saturated": bool((dual.weights >= dual.maximum).any()),
            "losses_before_step": losses}


def _hard_counts(model, role, threshold):
    with torch.no_grad():
        return _score_counts(_logits(model, role["x"]), role["y"], threshold)


def _score_counts(scores, labels, threshold):
    decision = working_margins(scores, threshold=threshold) >= 0
    normal, attack = labels == 0, labels == 1
    return {"scope": "synthetic_counts_only_no_performance_claim",
            "normal_support": int(normal.sum()), "false_positive_count": int((decision & normal).sum()),
            "attack_support": int(attack.sum()), "true_positive_count": int((decision & attack).sum())}


def run(output, seed, steps):
    sources = {"script": Path(__file__), "risk_distillation": Path(risk_distillation.__file__),
               "risk_data": Path(risk_data.__file__)}
    source_hashes = {name: _sha_file(path) for name, path in sources.items()}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sizes = {"fit": (96, 32), "reference": (64, 16),
             "constraint": (96, 32), "validation": (64, 16)}
    roles = {name: _generate_role(name, *counts, generator) for name, counts in sizes.items()}
    original = pd.concat([roles[name]["identity"] for name in ("fit", "reference", "constraint")],
                         ignore_index=True)
    manifest = risk_data.validate_risk_roles(original, *(roles[name]["identity"]
                                           for name in ("fit", "reference", "constraint", "validation")))
    manifest["synthetic_group_semantics"] = "unique_isolation_units_not_risk_strata_or_real_sessions"
    manifest_before = copy.deepcopy(manifest)
    fit, calibration, pool = (roles[name] for name in ("fit", "reference", "constraint"))
    anchor = _model()
    anchor_optimizer = torch.optim.Adam(anchor.parameters(), lr=.03)
    hard_config = BoundaryKDConfig(mode="hard", kd_weight=0.)
    for _ in range(60):
        anchor_optimizer.zero_grad(set_to_none=True)
        objective = boundary_kd_objective(_logits(anchor, fit["x"]), fit["y"], hard_config,
                                          student_threshold=0.)["total"]
        objective.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in anchor.parameters()):
            raise RuntimeError("hard-anchor gradient is non-finite or missing")
        anchor_optimizer.step()
    anchor.eval().requires_grad_(False)
    for parameter in anchor.parameters():
        parameter.grad = None
    anchor_sha = _state_sha(anchor.state_dict())
    with torch.no_grad():
        anchor_threshold_state = _freeze_threshold(_logits(anchor, calibration["x"]), calibration["y"])
        teacher_threshold_state = _freeze_threshold(_teacher_score(calibration["x"]), calibration["y"])
        threshold = anchor_threshold_state["threshold"]
        teacher_threshold = teacher_threshold_state["threshold"]
        anchor_margins = working_margins(_logits(anchor, pool["x"]), threshold=threshold)
        teacher_margins = working_margins(_teacher_score(fit["x"]), threshold=teacher_threshold).detach()
        reference_counts = _hard_counts(anchor, calibration, threshold)
        teacher_reference_counts = _score_counts(_teacher_score(calibration["x"]),
                                                calibration["y"], teacher_threshold)
    reference_counts["allowed_false_positive_count"] = anchor_threshold_state["allowed_false_positives"]
    teacher_reference_counts["allowed_false_positive_count"] = teacher_threshold_state["allowed_false_positives"]
    reference_count_checks = all(
        state["allowed_false_positives"] == math.floor(MAX_FPR * counts["normal_support"])
        and state["n_normal"] == counts["normal_support"]
        and state["false_positives"] == counts["false_positive_count"]
        and counts["false_positive_count"] <= state["allowed_false_positives"]
        for state, counts in ((anchor_threshold_state, reference_counts),
                              (teacher_threshold_state, teacher_reference_counts)))
    if not reference_count_checks:
        raise RuntimeError("frozen reference threshold violates its floor-count budget")
    constraints = (RiskConstraint("normal_fpr", "normal_fpr", MAX_FPR, group=0),
                   *(RiskConstraint(f"attack_loss_{i}", "attack_loss", .05, group=i)
                     for i in range(len(FAMILIES))))
    reference = build_risk_reference(anchor_margins, pool["y"], constraints,
                                    sample_ids=pool["identity"].sample_id.tolist(),
                                    normal_strata=torch.zeros(len(pool["y"]), dtype=torch.int64),
                                    attack_families=pool["families"], margin_floor=MARGIN_FLOOR)
    student = copy.deepcopy(anchor).train().requires_grad_(True)
    warmstart_sha = _state_sha(student.state_dict())
    optimizer = torch.optim.Adam(student.parameters(), lr=.015)
    dual = ProjectedDual(constraints, step_size=.5, maximum=10., reference_signature=reference.signature)
    kd_config = BoundaryKDConfig(mode="kd", kd_weight=.3, positive_weight=1., temperature=1.)
    fit_weight = len(fit["y"]) / (len(fit["y"]) + len(pool["y"]))
    constraint_weight = 1 - fit_weight
    history = [_snapshot(0, student, pool, reference, threshold, dual)]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        objectives = boundary_kd_objective(_logits(student, fit["x"]), fit["y"], kd_config,
                                           student_threshold=threshold, teacher_margins=teacher_margins)
        constraint_logits = _logits(student, pool["x"])
        constraint_hard = nn.functional.binary_cross_entropy_with_logits(
            constraint_logits, pool["y"].to(constraint_logits.dtype),
            pos_weight=constraint_logits.new_tensor(kd_config.positive_weight))
        common_hard = fit_weight * objectives["hard"] + constraint_weight * constraint_hard
        base_objective = common_hard + kd_config.kd_weight * objectives["soft"]
        full_report = constraint_risks(working_margins(constraint_logits, threshold=threshold),
                                       reference, sample_ids=reference.sample_ids)
        penalty = risk_lagrangian(full_report, dual.weights)
        total = base_objective + penalty
        if not bool(torch.isfinite(total)):
            raise RuntimeError("training objective is non-finite")
        total.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in student.parameters()):
            raise RuntimeError("student gradient is non-finite or missing")
        optimizer.step()
        with torch.no_grad():
            post_report = constraint_risks(working_margins(_logits(student, pool["x"]), threshold=threshold),
                                          reference, sample_ids=reference.sample_ids)
        dual.update(post_report)
        losses = {name: float(value.detach()) for name, value in objectives.items()}
        losses.update(constraint_hard=float(constraint_hard.detach()),
                      common_hard=float(common_hard.detach()), base_objective=float(base_objective.detach()),
                      lagrangian=float(penalty.detach()), combined=float(total.detach()))
        history.append(_snapshot(step, student, pool, reference, threshold, dual, losses=losses))
    student.eval()
    restored_dual = ProjectedDual(constraints, step_size=.5, maximum=10., reference_signature=reference.signature)
    restored_dual.load_state_dict(dual.state_dict())
    manifest_after = risk_data.validate_risk_roles(original, *(roles[name]["identity"]
                                          for name in ("fit", "reference", "constraint", "validation")))
    manifest_after["synthetic_group_semantics"] = manifest["synthetic_group_semantics"]
    checks = {"role_identities_unchanged": manifest_before == manifest_after,
              "warmstart_exact_anchor_state": warmstart_sha == anchor_sha,
              "anchor_frozen_and_unchanged": anchor_sha == _state_sha(anchor.state_dict())
              and all(not p.requires_grad and p.grad is None for p in anchor.parameters()),
              "reference_detached": not reference.anchor_margins.requires_grad,
              "dual_state_roundtrip_exact": restored_dual.state_dict() == dual.state_dict(),
              "dual_update_count_matches_steps": dual.updates == steps,
              "reference_signature_bound": dual.reference_signature == reference.signature,
              "all_finite": all(item["parameters_finite"] for item in history),
              "all_empirical_bounds_hold": all(item["aggregate_bounds_hold"] for item in history),
              "reference_thresholds_within_floor_count_budget": reference_count_checks,
              "all_working_margins_float64": anchor_margins.dtype == torch.float64
              and teacher_margins.dtype == torch.float64
              and all(item["margin_dtype"] == "torch.float64" for item in history),
              "sources_unchanged_during_run": source_hashes == {name: _sha_file(path)
                                                                for name, path in sources.items()}}
    grid = torch.tensor([-2., -.25, -.01, 0., .01, .25, 2.], dtype=torch.float64)
    a_grid, m_grid = torch.meshgrid(grid, grid, indexing="ij")
    grid_check = _pointwise(a_grid.flatten(), m_grid.flatten(), MARGIN_FLOOR)
    if not all(checks.values()):
        raise RuntimeError(f"engineering identity check failed: {checks}")
    metadata = {"scope": SCOPE, "architecture": "Linear(6,8)-Tanh-Linear(8,1)",
                "reference_signature": reference.signature, "student_logit_threshold": threshold,
                "anchor_logit_threshold": threshold, "margin_floor_kappa": MARGIN_FLOOR,
                "working_margin_dtype": "float64", "working_margin_scale": 1.}
    anchor_path, student_path = output / "anchor.pt", output / "student.pt"
    with anchor_path.open("xb") as stream:
        torch.save({"metadata": metadata, "state_dict": anchor.state_dict()}, stream)
    with student_path.open("xb") as stream:
        torch.save({"metadata": metadata, "state_dict": student.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state()}, stream)
    # Load only files this process just created, on CPU; no input-file API exists.
    restored_student = _model()
    restored_student.load_state_dict(torch.load(student_path, map_location="cpu", weights_only=True)["state_dict"])
    checks["student_cpu_serialization_exact"] = (_state_sha(restored_student.state_dict())
                                                  == _state_sha(student.state_dict()))
    if not checks["student_cpu_serialization_exact"]:
        raise RuntimeError("student CPU serialization check failed")
    _write_json(output / "history.json", {"scope": SCOPE, "iterates": history})
    _write_json(output / "dual-state.json", dual.state_dict())
    feasible_steps = [item["step"] for item in history if item["surrogate_budget_satisfied"]]
    actual_steps = [item["step"] for item in history if item["actual_budget_satisfied"]]
    final = history[-1]
    report = {"schema_version": 1, "scope": SCOPE, "engineering_status": "passed",
              "claim": "synthetic engineering connectivity only; no measured research improvement",
              "limitations": ["not real data, TFM, lab hardware or a performance comparison",
                              "max_fpr=0.15 is not the formal 0.01 operating point",
                              "small synthetic cohorts cannot validate low-tail/generalization guarantees",
                              "constraint participates in training; it is not an independent certificate",
                              "last iterate saved, no seed/validation/checkpoint selection"],
              "device": "cpu", "torch_threads": torch.get_num_threads(),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "pandas": pd.__version__}, "seed": seed, "student_steps": steps,
              "hyperparameters": {"anchor_steps": 60, "anchor_optimizer": "Adam", "anchor_lr": .03,
                                    "student_optimizer": "Adam", "student_lr": .015,
                                    "dual_step_size": .5, "dual_maximum": 10.,
                                    "kd": asdict(kd_config), "constraints": [asdict(x) for x in constraints],
                                    "majorizer": "adaptive", "normal_repair_credit": True,
                                    "risk_pass": "full_constraint_cohort_before_and_after_each_step"},
              "supervision_contract": {"anchor_hard": "fit_only", "student_hard": "fit_union_constraint",
                                       "student_kd": "fit_only", "reference": "thresholds_only_not_hard_training",
                                       "validation": "post_freeze_counts_only_not_selection",
                                       "common_hard_fit_weight": fit_weight,
                                       "common_hard_constraint_weight": constraint_weight,
                                       "common_hard_positive_weight": kd_config.positive_weight,
                                       "hard_formula": "wF*hardF+wC*hardC; same C forward reused for risk",
                                       "fair_comparator_requirement": "all student controls must receive same C labels/forward budget",
                                       "teacher": "known_function_no_fit_or_OOF_claim"},
              "margin_definition": {**metadata, "tie": "attack_if_logit_greater_than_or_equal_threshold",
                                    "threshold_source": "reference_normal_order_statistic_floor_count_plus_float64_nextafter",
                                    "teacher_type": "known_synthetic_function_not_fitted",
                                    "teacher_logit_threshold": teacher_threshold,
                                    "teacher_formula": "1.3*x0+x1+0.7*x2+0.4*sin(x3)+0.15*x4*x5-1",
                                    "teacher_target": "sigmoid((score-threshold)/temperature), not posterior"},
              "role_manifest": manifest, "checks": checks, "pointwise_grid": grid_check,
              "budget_status": {"surrogate": "feasible_iterate_observed" if feasible_steps else "no_feasible_iterate",
                                "surrogate_feasible_steps": feasible_steps, "actual_feasible_steps": actual_steps,
                                "final_surrogate_satisfied": final["surrogate_budget_satisfied"],
                                "final_actual_satisfied": final["actual_budget_satisfied"],
                                "saved_checkpoint": "last_iterate_regardless_of_feasibility",
                                "initial": history[0]["metrics"], "final": final["metrics"]},
              "validation_after_freeze_only": {"anchor": _hard_counts(anchor, roles["validation"], threshold),
                                               "last_student": _hard_counts(student, roles["validation"], threshold)},
              "frozen_reference_thresholds": {"anchor": anchor_threshold_state, "teacher": teacher_threshold_state},
              "reference_normal_counts": reference_counts,
              "teacher_reference_normal_counts": teacher_reference_counts,
              "state_sha256": {"anchor": anchor_sha, "student": _state_sha(student.state_dict())},
              "source_sha256": source_hashes,
              "artifact_sha256": {name: _sha_file(output / name)
                                  for name in ("anchor.pt", "student.pt", "history.json", "dual-state.json")}}
    _write_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are rejected")
    parser.add_argument("--seed", type=lambda value: _integer(value, minimum=0), default=20261103)
    parser.add_argument("--steps", type=lambda value: _integer(value, minimum=1), default=20)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("--output must be a new directory; nothing was overwritten")
    try:
        report = run(output, args.seed, args.steps)
    except Exception as exc:
        _write_json(output / "failure.json", {"scope": SCOPE, "engineering_status": "failed",
                                             "error_type": type(exc).__name__, "error": str(exc)})
        print(f"FAILED {SCOPE}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"scope": SCOPE, "engineering_status": report["engineering_status"],
                      "output": str(output), "budget_status": report["budget_status"]["surrogate"],
                      "final_surrogate_satisfied": report["budget_status"]["final_surrogate_satisfied"],
                      "final_actual_satisfied": report["budget_status"]["final_actual_satisfied"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
