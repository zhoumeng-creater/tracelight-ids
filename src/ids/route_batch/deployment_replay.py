"""Complete cold C/E replay before a separate inference-cost subprocess.

Only named CPU foundation rebuilds are accepted. This worker records evidence;
it does not train weights, load TabM, bootstrap metrics, or benchmark latency.
"""
from __future__ import annotations

from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.preprocessing import CSVSchema, labels, read_csv
from .foundation import load_foundation
from .metrics import decisions, threshold
from .preprocessing import raw_numeric

ATOL = 3e-5
RTOL = 3e-5
FORMAT = "tracelight-foundation-full-cold-replay-v1"


def _identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _model_path(directory, descriptor):
    relative = descriptor.get("model_file")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("foundation recipe must use a relative deployment path")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError("foundation recipe is missing or escapes deployment directory")
    if sha256_file(path) != descriptor.get("model_sha256"):
        raise ValueError("foundation recipe SHA-256 mismatch")
    return path


def _registered_frame(config_path, spec, schema, role):
    item = spec["files"][role]
    path = Path(item["path"])
    if not path.is_absolute():
        path = config_path.parent / path
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"registered {role} SHA-256 mismatch")
    frame = read_csv(path, schema)
    if len(frame) != item["rows"]:
        raise ValueError(f"registered {role} row count differs")
    if schema.id_column is None or frame[schema.id_column].astype(str).duplicated().any():
        raise ValueError(f"registered {role} requires unique explicit sample IDs")
    if role == "calibration" and labels(frame, schema).any():
        raise ValueError("calibration must contain normal rows only")
    return frame, _identity(path)


def _saved_frame(directory, frame, schema, role, registration):
    path = directory / f"{role}.csv"
    saved = pd.read_csv(path, dtype={"row_id": str}, float_precision="round_trip")
    if (saved.row_id.tolist() != frame[schema.id_column].astype(str).tolist()
            or not np.array_equal(saved.label, labels(frame, schema))
            or saved.attack_family.astype(str).tolist() != frame[schema.family_column].astype(str).tolist()
            or not np.isfinite(saved.score).all()):
        raise ValueError(f"saved {role} score IDs/labels/families differ from registered input")
    if not np.array_equal(saved.predicted_attack, decisions(saved.score, registration).astype(int)):
        raise ValueError(f"saved {role} decisions differ from its registered threshold")
    return saved, _identity(path)


def _saved_probabilities(directory, role, saved):
    embedded = ["raw_probability_normal", "raw_probability_attack"]
    if set(embedded).issubset(saved.columns):
        values = saved[embedded].to_numpy(dtype=np.float64)
        source = "embedded saved score CSV"
        identity = None
    elif (directory / f"{role}_probabilities.csv").is_file():
        path = directory / f"{role}_probabilities.csv"
        frame = pd.read_csv(path, dtype={"row_id": str}, float_precision="round_trip")
        if frame.row_id.tolist() != saved.row_id.tolist():
            raise ValueError(f"saved {role} probability IDs differ from saved scores")
        values = frame[["probability_normal", "probability_attack"]].to_numpy(dtype=np.float64)
        source, identity = "saved probability sidecar", _identity(path)
    else:
        return None, {"available": False}
    _valid_probabilities(values, len(saved))
    return values, {"available": True, "source": source, "file": identity}


def _valid_probabilities(values, rows):
    if (values.shape != (rows, 2) or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1, atol=2e-6, rtol=0)):
        raise ValueError("replay probabilities must be finite aligned binary probability rows")


def _evidence(directory, role, saved, output, old_registration, new_registration, saved_probabilities):
    expected = saved.score.to_numpy(dtype=np.float64)
    actual = np.asarray(output["scores"], dtype=np.float64)
    probabilities = np.asarray(output["probabilities"], dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError(f"replayed {role} scores must be finite and aligned")
    _valid_probabilities(probabilities, len(saved))
    error = actual - expected
    allowed = ATOL + RTOL * np.abs(expected)
    close = np.abs(error) <= allowed
    original = decisions(expected, old_registration)
    fixed = decisions(actual, old_registration)
    recalibrated = decisions(actual, new_registration)
    rows = saved[["row_id", "label", "attack_family"]].copy()
    rows["saved_score"] = expected
    rows["replayed_score"] = actual
    rows["signed_error"] = error
    rows["absolute_error"] = np.abs(error)
    rows["allowed_error"] = allowed
    rows["within_tolerance"] = close
    rows["raw_probability_normal"] = probabilities[:, 0]
    rows["raw_probability_attack"] = probabilities[:, 1]
    rows["saved_threshold_decision"] = original
    rows["replayed_saved_threshold_decision"] = fixed
    rows["replayed_recalibrated_decision"] = recalibrated
    rows["saved_threshold_decision_changed"] = original != fixed
    rows["recalibrated_decision_changed"] = original != recalibrated
    probability_comparison = {"saved_raw_probabilities_available": saved_probabilities is not None}
    if saved_probabilities is not None:
        rows["saved_raw_probability_normal"] = saved_probabilities[:, 0]
        rows["saved_raw_probability_attack"] = saved_probabilities[:, 1]
        delta = probabilities[:, 1] - saved_probabilities[:, 1]
        rows["raw_attack_probability_signed_error"] = delta
        probability_comparison.update({"max_attack_probability_abs_error": float(np.abs(delta).max()),
                                       "different_attack_probability_rows": int((delta != 0).sum())})
    path = directory / f"cold_replay_{role}.csv"
    rows.to_csv(path, index=False)
    return {"rows": len(saved), "all_scores_within_tolerance": bool(close.all()),
            "scores_bitwise_equal": bool(np.array_equal(actual, expected)),
            "mismatched_score_rows": int((~close).sum()), "max_absolute_score_error": float(np.abs(error).max()),
            "mean_absolute_score_error": float(np.abs(error).mean()),
            "saved_threshold_decision_changed_rows": int((original != fixed).sum()),
            "recalibrated_decision_changed_rows": int((original != recalibrated).sum()),
            "saved_alerts": int(original.sum()), "replayed_saved_threshold_alerts": int(fixed.sum()),
            "replayed_recalibrated_alerts": int(recalibrated.sum()),
            "raw_probability_zero_rows": int((probabilities[:, 1] == 0).sum()),
            "raw_probability_one_rows": int((probabilities[:, 1] == 1).sum()),
            "probability_comparison": probability_comparison, "row_evidence": _identity(path)}


def run(config_path: Path, output_dir: Path) -> dict:
    """Cold-rebuild one CPU foundation deployment, then replay all C followed by E.

    Write a failure report and any completed row evidence before raising. A new
    cost process may run only after all three prescribed replay gates pass.
    """
    config_path, directory = Path(config_path).resolve(), Path(output_dir).resolve()
    if not directory.is_dir():
        raise ValueError("existing foundation deployment directory is required")
    report_path = directory / "cold_replay_report.json"
    report = {"format": FORMAT, "status": "failed", "all_checks_passed": False, "timing_allowed": False,
              "atol": ATOL, "rtol": RTOL, "reference_for_relative_tolerance": "saved_score",
              "score_device": "cpu", "rebuild_device": "cpu", "roles": {}, "files": {},
              "call_order": ["calibration", "evaluation"], "loaded_data_roles": ["calibration", "evaluation"],
              "scope": "isolated full cold replay; subsequent cost measurement requires another fresh process",
              "gradient_training_performed": False, "benchmark_performed": False, "bootstrap_performed": False}
    stage = "validate_deployment"
    started = time.perf_counter()
    try:
        spec, descriptor = read_json(config_path), read_json(directory / "deployment.json")
        settings = spec["settings"]
        if descriptor.get("kind") != "foundation" or descriptor.get("rebuild_device") != "cpu":
            raise ValueError("full replay accepts only an explicit CPU foundation deployment")
        if descriptor.get("metadata", {}).get("kind") not in {"tabicl", "tabpfn25"}:
            raise ValueError("full replay supports only registered TabICL or TabPFN foundation models")
        if settings.get("threads") != 1 or settings.get("query_batch_size") != 128:
            raise ValueError("full replay requires one-thread CPU scoring in registered batches of 128")
        recipe = _model_path(directory, descriptor)
        schema = CSVSchema.from_dict(spec["schema"])
        registration = read_json(directory / "threshold.json")
        if registration.get("model") != descriptor or registration.get("calibration_sha256") != spec["files"]["calibration"]["sha256"]:
            raise ValueError("saved threshold does not bind this deployment and registered calibration")
        report["files"].update({"config": _identity(config_path), "deployment": _identity(directory / "deployment.json"),
                                "recipe": _identity(recipe), "saved_threshold": _identity(directory / "threshold.json")})
        frames, saved, raw_saved = {}, {}, {}
        for role in ("calibration", "evaluation"):
            stage = f"validate_{role}"
            frames[role], report["files"][role + "_input"] = _registered_frame(config_path, spec, schema, role)
            saved[role], report["files"][role + "_saved_scores"] = _saved_frame(directory, frames[role], schema, role, registration)
            raw_saved[role], probability_source = _saved_probabilities(directory, role, saved[role])
            report[role + "_saved_probability_source"] = probability_source
        actual_registration = threshold(saved["calibration"].score, settings["alpha"])
        for key in ("alpha", "n_normal", "rank", "threshold", "comparison", "comparison_dtype", "calibration_fp"):
            if registration.get(key) != actual_registration[key]:
                raise ValueError("saved C scores do not reproduce saved threshold registration")
        report["saved_threshold"] = actual_registration
        stage = "cold_rebuild"
        load_started = time.perf_counter()
        model = load_foundation(recipe, device="cpu", diagnostics=True)
        report["cold_load_and_rebuild_seconds"] = time.perf_counter() - load_started
        if model.metadata.get("build_device") != "cpu" or model.metadata.get("score_device") != "cpu":
            raise ValueError("loaded foundation did not retain the named CPU build/score path")
        new_registration = None
        for role in ("calibration", "evaluation"):
            stage = f"replay_{role}"
            role_started = time.perf_counter()
            outputs = model.predict_outputs(raw_numeric(frames[role], schema), batch_size=128, diagnostics=True)
            elapsed = time.perf_counter() - role_started
            if role == "calibration":
                new_registration = threshold(outputs["scores"], settings["alpha"])
                report["replayed_threshold"] = new_registration
            report["roles"][role] = _evidence(directory, role, saved[role], outputs, registration,
                                               new_registration, raw_saved[role])
            report["roles"][role]["raw_transform_and_score_seconds"] = elapsed
            # Persist C evidence/report even if E prediction fails afterward.
            write_json(report_path, report)
        report["gates"] = {
            "full_C_and_E_scores_within_original_tolerance": all(
                value["all_scores_within_tolerance"] for value in report["roles"].values()),
            "E_decisions_unchanged_at_saved_threshold": report["roles"]["evaluation"]["saved_threshold_decision_changed_rows"] == 0,
            "E_decisions_unchanged_after_C_recalibration": report["roles"]["evaluation"]["recalibrated_decision_changed_rows"] == 0,
        }
        passed = all(report["gates"].values())
        report.update(status="matched" if passed else "mismatch", all_checks_passed=passed, timing_allowed=passed,
                      elapsed_seconds=time.perf_counter() - started,
                      calibration_decision_changes_are_reported_not_an_additional_gate=True)
        write_json(report_path, report)
        if not passed:
            stage = "replay_acceptance"
            raise ValueError("full cold replay failed fixed score/decision gates; evidence saved in cold_replay_report.json")
        return report
    except Exception as exc:
        report.update(error=str(exc), error_type=type(exc).__name__, failed_stage=stage,
                      traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started,
                      all_checks_passed=False, timing_allowed=False)
        write_json(report_path, report)
        raise
