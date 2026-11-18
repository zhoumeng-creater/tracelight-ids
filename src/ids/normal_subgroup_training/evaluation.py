"""One frozen C/E evaluation, exact mechanism checks, and retained R30 identities."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids.model_training.evaluation import IDENTITY, _counts
from ids.model_training_redesign import data as old_data
from ids.model_training_redesign import evaluation as old_evaluation
from ids.model_training_redesign.sampling import canonical_frame

from .bindings import SEEDS, sha256
from .selection import QUANTILES, VALIDATION_STEPS, checkpoint_guards, metrics_to_json


ARMS = ("transport_target", "identity_target")
MODEL_IDS = tuple(f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(metrics_to_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _fractions(value):
    """Restore exact receipt arithmetic; the rounded display value is never compared."""
    if isinstance(value, dict):
        if {"numerator", "denominator"}.issubset(value):
            n, d = value["numerator"], value["denominator"]
            if type(n) is not int or type(d) is not int or d <= 0:
                raise ValueError("Invalid exact Fraction receipt")
            return Fraction(n, d)
        return {key: _fractions(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_fractions(item) for item in value]
    return value


def compare_mechanism(transport_metrics, control_metrics, baseline_metrics, *, selected_step):
    transport, control, baseline = map(_fractions, (transport_metrics, control_metrics, baseline_metrics))
    tg, cg = checkpoint_guards(transport, baseline), checkpoint_guards(control, baseline)
    h, c, b = (m["tail"]["H"] for m in (transport, control, baseline))
    if not all(isinstance(value, Fraction) for value in (h, c, b)):
        raise ValueError("Mechanism comparison requires exact Fraction H values")
    checks = {"nonfallback_transport": selected_step in VALIDATION_STEPS,
              "transport_guards_pass": tg["passed"], "control_guards_pass": cg["passed"],
              "H_strictly_below_R30": h < b, "H_strictly_below_control": h < c,
              "each_H_q_not_above_control": all(transport["tail"]["H_q"][q]
                                                <= control["tail"]["H_q"][q] for q, _ in QUANTILES),
              "each_source_H_q_not_above_control": all(
                  transport["source_tails"][p]["H_q"][q] <= control["source_tails"][p]["H_q"][q]
                  for p in ("A", "B") for q, _ in QUANTILES)}
    return {"passed": all(checks.values()), "checks": checks,
            "H_transport": h, "H_control": c, "H_R30": b,
            "transport_guards": tg, "control_guards": cg}


def _changes(candidate, reference):
    truth = candidate.label.eq(1)
    added = candidate.alert & ~reference.alert
    lost = ~candidate.alert & reference.alert
    columns = [*IDENTITY, "family", "ip_protocol", "duration_seconds"]
    return {name: candidate.loc[mask, columns].to_dict("records") for name, mask in {
        "tp_added": added & truth, "tp_lost": lost & truth,
        "fp_added": added & ~truth, "fp_removed": lost & ~truth,
        "alerts_added": added, "alerts_removed": lost}.items()}


def evaluate_scores(c_scores, evaluation_frame, scores, reference_decisions, *, design,
                    protocol_design, model_id, historical_decisions=None):
    """Pure once-per-model calibration/reporting; caller supplies bound arrays only."""
    calibration = old_evaluation.calibrate(
        c_scores, expected_normal_rows=int(design["calibration"]["normal_rows"]))
    units, days, decisions = old_evaluation.evaluate(evaluation_frame, scores, calibration, model_id=model_id)
    if reference_decisions.model_id.nunique() != 1:
        raise ValueError("R30 reference requires exactly one frozen model ID")
    reference_id = str(reference_decisions.model_id.iloc[0])
    if reference_id == model_id:
        raise ValueError("Candidate and R30 reference model IDs must differ")
    historical = historical_decisions or {}
    if model_id in historical or reference_id in historical:
        raise ValueError("Historical model IDs collide with candidate or R30 reference")
    frames = {model_id: decisions, reference_id: reference_decisions, **historical}
    roles = old_evaluation._roles(protocol_design)
    aligned = old_evaluation._align(frames, list(frames), {model_id}, roles, protocol_design)
    current, reference = aligned[model_id], aligned[reference_id]
    cells = [{"source_id": source, **_counts(part)}
             for source, part in current.groupby("source_id", sort=True)]
    role_counts = {role: _counts(current.loc[current.source_id.isin(sources)])
                   for role, sources in roles.items()}
    for role, target in (("E_B", "each_seed_B_TP"), ("E_D", "each_seed_D_TP")):
        previous = _counts(reference.loc[reference.source_id.isin(roles[role])])
        if previous["tp"] != design["adoption"][target] or not previous["ready"]:
            raise ValueError("Frozen R30 reference no longer has its registered complete TP set")
    changes = _changes(current, reference)
    checks = {"calibration_ready": bool(calibration["ready"]),
              "B_TP_preserved": role_counts["E_B"]["tp"] == design["adoption"]["each_seed_B_TP"],
              "D_TP_preserved": role_counts["E_D"]["tp"] == design["adoption"]["each_seed_D_TP"],
              "all_R30_TP_identities_preserved": not changes["tp_lost"],
              "all_15_cells_within_original_budget": len(cells) == 15 and all(c["qualified"] for c in cells)}
    comparison = {}
    for name in (reference_id, *historical):
        comparison[name] = {**old_evaluation._pair(current, aligned[name], set(roles["E_D"])),
                            "identity_changes": _changes(current, aligned[name])}
    return {"model_id": model_id, "calibration": calibration, "cells": cells,
            "roles": role_counts, "checks": checks, "development_pass": all(checks.values()),
            "excess_X": sum(max(c["fp"] - c["fp_budget"], 0) for c in cells),
            "identity_changes": changes, "reference_comparisons": comparison,
            "units": units, "utc_days": days, "decision_frame": current}


def compare_models(reports, selections, *, implementation_contract_pass):
    """No seed average, seed selection, threshold rescue, or mixed-arm adoption."""
    if type(implementation_contract_pass) is not bool:
        raise ValueError("Implementation contract result must be an explicit boolean")
    if set(reports) != set(MODEL_IDS) or set(selections) != set(MODEL_IDS):
        raise ValueError("Exactly the six registered model reports and selections are required")
    seeds = {}
    for seed in SEEDS:
        main_id, control_id = (f"{arm}_s{seed}" for arm in ARMS)
        main, control = reports[main_id], reports[control_id]
        a, c = selections[main_id], selections[control_id]
        if _fractions(a["baseline_metrics"]) != _fractions(c["baseline_metrics"]):
            raise ValueError("Paired arms must use identical baseline selection metrics")
        mechanism = compare_mechanism(a["metrics"], c["metrics"], a["baseline_metrics"],
                                      selected_step=a["selected_step"])
        advantage = main["excess_X"] == 0 and main["excess_X"] < control["excess_X"]
        adoption = bool(implementation_contract_pass and mechanism["passed"]
                        and main["development_pass"] and advantage)
        control_valid = bool(implementation_contract_pass and main["excess_X"] == control["excess_X"] == 0
                             and control["development_pass"] and mechanism["control_guards"]["passed"])
        seeds[str(seed)] = {"transport_id": main_id, "control_id": control_id,
                            "mechanism": mechanism, "transport_development_pass": main["development_pass"],
                            "X_transport": main["excess_X"], "X_control": control["excess_X"],
                            "strict_excess_advantage": advantage, "transport_adoption_pass": adoption,
                            "identity_control_valid_development_result": control_valid}
    adopted = all(row["transport_adoption_pass"] for row in seeds.values())
    control_valid = all(row["identity_control_valid_development_result"] for row in seeds.values())
    return {"status": "complete", "completed": True, "adoption_pass": adopted,
            "implementation_contract_pass": implementation_contract_pass,
            "all_seed_mechanism_pass": all(row["mechanism"]["passed"] for row in seeds.values()),
            "all_seed_identity_control_valid": control_valid,
            "retained_development_arm": "transport_target" if adopted else "identity_target" if control_valid else "R30",
            "new_blind_confirmation": False, "automatic_deployment": False, "seeds": seeds,
            "models": reports, "failure_action": "archive_result_without_retry_or_threshold_target_seed_changes"}


def _bound_path(root, item):
    root = Path(root).resolve()
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Frozen payload path escapes its root")
    if sha256(path) != item["sha256"] or ("bytes" in item and path.stat().st_size != item["bytes"]):
        raise ValueError(f"Frozen payload changed: {item['path']}")
    return path


def verify_frozen(preflight_dir, models_root, freeze_path, *, design_path, bindings_path):
    """This entire gate completes before any archive C/E payload is touched."""
    frozen = _read(freeze_path)
    if (frozen.get("status") != "all_six_artifacts_frozen" or set(frozen.get("models", {})) != set(MODEL_IDS)
            or frozen.get("implementation_contract_pass") is not True):
        raise ValueError("All six verified artifacts must be frozen before C/E access")
    if sha256(design_path) != frozen["design_sha256"] or sha256(bindings_path) != frozen["bindings_sha256"]:
        raise ValueError("Frozen design/input bindings changed")
    paired = frozen.get("paired_contracts", {})
    if set(paired) != set(map(str, SEEDS)) or any(paired[str(s)].get("passed") is not True for s in SEEDS):
        raise ValueError("All three paired contracts must pass before C/E access")
    preflight = Path(preflight_dir).resolve()
    status = _read(preflight / "status.json")
    manifest_path = preflight / "output_manifest.json"
    if (status.get("status") != "zero_update_preflight_complete"
            or sha256(manifest_path) != frozen["preflight_manifest_sha256"]
            or sha256(manifest_path) != status["output_manifest"]["sha256"]):
        raise ValueError("Frozen preflight output manifest changed or is incomplete")
    for relative, item in _read(manifest_path)["files"].items():
        _bound_path(preflight, {"path": relative, **item})
    selections, artifacts = {}, {}
    for name in MODEL_IDS:
        item = frozen["models"][name]
        if (item["arm"] not in ARMS or item["seed"] not in SEEDS
                or name != f"{item['arm']}_s{item['seed']}"
                or item["selected_step"] not in (0, *VALIDATION_STEPS)):
            raise ValueError("Frozen model identity or selected step differs from the contract")
        artifacts[name] = _bound_path(models_root, item["artifact"])
        metrics = _fractions(_read(_bound_path(models_root, item["selected_metrics"])))
        baseline = _fractions(_read(_bound_path(models_root, item["baseline_metrics"])))
        if not checkpoint_guards(metrics, baseline)["passed"]:
            raise ValueError("Selected artifact does not satisfy its frozen F/V protection guards")
        if item["selected_step"] == 0:
            if metrics != baseline:
                raise ValueError("Step0 fallback metrics must be identical to its baseline")
        elif metrics["tail"]["H"] >= baseline["tail"]["H"]:
            raise ValueError("Selected nonfallback checkpoint must strictly improve pooled H")
        selections[name] = {"metrics": metrics, "baseline_metrics": baseline,
                            "selected_step": item["selected_step"]}
    for seed in SEEDS:
        if selections[f"transport_target_s{seed}"]["baseline_metrics"] != selections[
                f"identity_target_s{seed}"]["baseline_metrics"]:
            raise ValueError("Paired baseline selection metrics differ")
    return frozen, artifacts, selections


def _load_predictor(path):
    from ids.model_training_redesign.training import load_predictor
    return load_predictor(path, threads=1, device="cpu")


def evaluate_frozen(archive, preflight_dir, models_root, freeze_path, output_dir, *,
                    design_path, bindings_path, loader=None):
    """Score each frozen artifact on C and E exactly once through the CPU endpoint."""
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Evaluation output must be empty; do not recalibrate or retry partial evaluation")
    frozen, artifacts, selections = verify_frozen(preflight_dir, models_root, freeze_path,
                                                design_path=design_path, bindings_path=bindings_path)
    design, bindings = _read(design_path), _read(bindings_path)
    if (design["seeds"] != list(SEEDS) or design["arms"] != list(ARMS)
            or design["calibration"]["normal_rows"] != 5511
            or design["calibration"]["one_based_sorted_rank"] != 5457
            or design["calibration"]["decision"] != "strict_greater"):
        raise ValueError("Formal evaluation requires the original seeds, arms, and C program")
    # Only after the six-model gate may even hashes of C/E payloads be read.
    required = ["protocol.json", "inputs/C.parquet", "inputs/E_B.parquet", "inputs/E_D.parquet",
                "inputs/reference_decisions.parquet"]
    required += [f"reports/protocol_blind_s{s}-decisions.parquet" for s in SEEDS]
    paths = {relative: _bound_path(archive, {"path": relative, **bindings["files"][relative]})
             for relative in required}
    protocol_design = _read(paths["protocol.json"])["design"]
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "status.json", {"status": "started", "completed": False, "adoption_pass": False,
                                  "freeze_sha256": sha256(freeze_path), "calibrations_per_model": 1})
    try:
        calibration_frame = canonical_frame(pd.read_parquet(paths["inputs/C.parquet"]))
        evaluation_frame = canonical_frame(pd.concat([pd.read_parquet(paths[f"inputs/{role}.parquet"])
                                                      for role in ("E_B", "E_D")], ignore_index=True))
        if (len(calibration_frame) != 5511 or not calibration_frame.label.eq(0).all()
                or not calibration_frame.profile.eq("B").all()):
            raise ValueError("C must contain the original 5511 B normal identities only")
        c_ids = set(map(tuple, calibration_frame[IDENTITY].to_numpy()))
        if c_ids & set(map(tuple, evaluation_frame[IDENTITY].to_numpy())):
            raise ValueError("C and E identities overlap")
        evaluation_frame.to_parquet(root / "E.parquet", index=False)
        historical_frame = pd.read_parquet(paths["inputs/reference_decisions.parquet"])
        historical = {name: frame.reset_index(drop=True)
                      for name, frame in historical_frame.groupby("model_id", sort=True)}
        references = {seed: pd.read_parquet(paths[f"reports/protocol_blind_s{seed}-decisions.parquet"])
                      for seed in SEEDS}
        reports = {}
        load = _load_predictor if loader is None else loader
        for name in MODEL_IDS:
            predictor = load(artifacts[name])
            c_scores = np.asarray(predictor.score(old_data.blind(calibration_frame)), dtype=np.float64)
            e_scores = np.asarray(predictor.score(old_data.blind(evaluation_frame)), dtype=np.float64)
            seed = frozen["models"][name]["seed"]
            report = evaluate_scores(c_scores, evaluation_frame, e_scores, references[seed], design=design,
                                     protocol_design=protocol_design, model_id=name,
                                     historical_decisions=historical)
            folder = root / name
            folder.mkdir()
            for role, frame, scores in (("C", calibration_frame, c_scores), ("E", evaluation_frame, e_scores)):
                saved = frame[IDENTITY].copy()
                saved["score"] = scores
                saved.to_parquet(folder / f"{role}-scores.parquet", index=False)
            report.pop("decision_frame").to_parquet(folder / "decisions.parquet", index=False)
            _write(folder / "calibration.json", report["calibration"])
            _write(folder / "report.json", report)
            reports[name] = report
            del predictor
        # Recheck frozen files before marking completion; no second forward/calibration.
        verify_frozen(preflight_dir, models_root, freeze_path, design_path=design_path, bindings_path=bindings_path)
        result = compare_models(reports, selections, implementation_contract_pass=True)
        result["freeze_sha256"] = sha256(freeze_path)
        result["calibration_calls"] = len(MODEL_IDS)
        result["C_E_forward_calls"] = 2 * len(MODEL_IDS)
        _write(root / "comparison.json", result)
        _write(root / "status.json", {key: result[key] for key in (
            "status", "completed", "adoption_pass", "retained_development_arm", "freeze_sha256")})
        return result
    except Exception as error:
        _write(root / "status.json", {"status": "failed", "completed": False, "adoption_pass": False,
                                      "error_type": type(error).__name__, "error": str(error),
                                      "automatic_retry": False})
        raise
