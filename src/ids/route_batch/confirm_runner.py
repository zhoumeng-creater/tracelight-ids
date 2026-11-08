"""Freeze every available final model, then score one unseen-group holdout."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.context_cache.preprocessing import CSVSchema, labels, read_csv
from .confirmation import CONFIRMATION_SEED, prepare_confirmation
from .data import load_inputs
from .metrics import paired
from .protocol import REPO, source_snapshot
from . import runner


FORMAT = "tracelight-route-all-model-confirmation-freeze-v1"
CALIBRATION_ROWS = 10000
EVALUATION_ROWS = 50000


def _file(path, expected=None):
    path = Path(path).resolve()
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Frozen artifact SHA-256 differs: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def _verify_file(item):
    path = Path(item["path"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
        raise ValueError(f"Frozen artifact changed or missing: {path}")
    return path


def _inside(attempt, name):
    path = (attempt / name).resolve()
    if not path.is_relative_to(attempt):
        raise ValueError("Task model path escapes its frozen attempt")
    return path


def _model_entry(state, spec, tabpfn_python):
    task, attempt, result = state["task"], state["attempt"], state["result"]
    model_id = task["id"]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", model_id) is None:
        raise ValueError("Invalid task ID for confirmation")
    original_status = result.get("status", "unknown") if result else "pending_or_interrupted"
    entry = {"id": model_id, "task": task, "original_status": original_status,
             "eligible": False, "freeze_status": "unavailable",
             "attempt": str(attempt.resolve()) if attempt else None}
    if result:
        entry["original_reason"] = result.get("error", result.get("reason"))
        entry["original_result"] = _file(attempt / "result.json")
    descriptor = result.get("model") if result else None
    try:
        # A failed cost/endpoint diagnostic does not discard an already-complete refit.
        if task["kind"] == "tabm" and attempt and (attempt / "refit" / "summary.json").is_file():
            refit = read_json(attempt / "refit" / "summary.json")
            if refit.get("status") == "complete":
                from .tabm_training import SCORE_CONTRACT
                descriptor = descriptor or {"kind": "tabm", "model_file": "refit/model.pt",
                    "preprocessing_file": "preprocessing.joblib", "score_contract": SCORE_CONTRACT}
                entry["eligibility_evidence"] = "completed full-P refit; original task status retained"
                entry["refit_summary"] = _file(attempt / "refit" / "summary.json")
        if task["kind"] == "foundation" and descriptor is None and attempt:
            threshold_path = attempt / "threshold.json"
            if threshold_path.is_file():
                descriptor = read_json(threshold_path).get("model")
                entry["eligibility_evidence"] = "saved final foundation descriptor before later task failure"
        if task["kind"] == "foundation" and descriptor and original_status == "measurement_failed":
            entry["eligibility_evidence"] = "completed foundation fit and scoring; cost measurement failed"
        if task["kind"] == "reference":
            descriptor = descriptor or {"kind": "reference",
                "directory": str(Path(spec["round14_root"]) / "comparison" / task["name"]),
                "score_contract": "FP32 attack-normal logit"}
        if not descriptor or (original_status != "complete" and task["kind"] not in {"reference"}
                              and "eligibility_evidence" not in entry):
            entry["reason"] = "No completed final model; development performance was not used for inclusion"
            return entry
        if descriptor["kind"] != task["kind"]:
            raise ValueError("Final descriptor kind differs from the declared task")
        files = {}
        if task["kind"] == "tabm":
            files["model"] = _file(_inside(attempt, descriptor["model_file"]), descriptor.get("model_sha256"))
            files["preprocessing"] = _file(_inside(attempt, descriptor["preprocessing_file"]),
                                            descriptor.get("preprocessing_sha256"))
        elif task["kind"] == "foundation":
            files["model"] = _file(_inside(attempt, descriptor["model_file"]), descriptor.get("model_sha256"))
            metadata = descriptor["metadata"]
            files["checkpoint"] = _file(metadata["checkpoint_path"], metadata["checkpoint_sha256"])
        elif task["kind"] == "reference":
            directory = Path(descriptor["directory"]).resolve()
            manifest = read_json(directory / "deployment.json")
            if manifest.get("candidate") != task["name"]:
                raise ValueError("Reference deployment candidate differs from task")
            required = {"model.pt", "preprocessor.joblib", "threshold.json"}
            if set(manifest.get("files", {})) != required:
                raise ValueError("Reference deployment must bind model, preprocessing and threshold")
            if manifest.get("calibration_sha256") != spec["files"]["calibration"]["sha256"]:
                raise ValueError("Reference deployment does not belong to the registered round-14 C")
            for filename in sorted(required):
                files[filename] = _file(directory / filename, manifest["files"][filename])
            files["deployment.json"] = _file(directory / "deployment.json")
            if task["name"] == "full" and spec.get("baseline_model_sha256") is not None:
                if files["model.pt"]["sha256"] != spec["baseline_model_sha256"]:
                    raise ValueError("Full reference differs from the original registered model")
        else:
            raise ValueError("Unsupported final model kind")
        python = ((tabpfn_python or spec.get("tabpfn_python"))
                  if task.get("model_kind") == "tabpfn25" else None) or sys.executable
        entry.update(eligible=True, freeze_status="ready", descriptor=descriptor,
                     files=files, python=str(python))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        entry.update(freeze_status="artifact_missing" if isinstance(exc, FileNotFoundError) else "artifact_invalid",
                     reason=str(exc), error_type=type(exc).__name__)
    return entry


def _create_freeze(root, source_zip, previous_prepared_dir, tabpfn_python):
    current_source = source_snapshot()["source_sha256"]
    original_source = read_json(root / "source_snapshot.json")
    if current_source != original_source.get("source_sha256"):
        raise ValueError("Confirmation source code differs from the original run source snapshot")
    spec = read_json(root / "protocol.json")
    inputs = load_inputs(root / "protocol.json")
    previous = Path(previous_prepared_dir).resolve()
    previous_files = {name: _file(previous / name) for name in (
        "preparation.json", "support.csv", "normal_calibration.csv", "development_evaluation.csv")}
    settings = spec["settings"]
    if settings["alpha"] != .01:
        raise ValueError("Confirmation retains the registered 1% primary calibration")
    frozen = {"format": FORMAT, "protocol": _file(root / "protocol.json"),
              "source_code_sha256": current_source,
              "run_source_snapshot": _file(root / "source_snapshot.json"),
              "policy": "all available final models and every seed; no development-score ranking or selection",
              "schema": inputs["schema"].to_dict(),
              "settings": {"alpha": .01, "sensitivity_alphas": settings["sensitivity_alphas"],
                           "query_batch_size": settings["query_batch_size"], "reference": []},
              "source_archive": _file(source_zip), "previous_prepared_dir": str(previous),
              "previous_files": previous_files,
              "data_recipe": {"seed": CONFIRMATION_SEED, "calibration_rows": CALIBRATION_ROWS,
                              "evaluation_rows": EVALUATION_ROWS},
              "models": [_model_entry(state, spec, tabpfn_python)
                         for state in runner._task_states(root, spec["tasks"])]}
    frozen["freeze_id"] = canonical_hash(frozen)
    confirmation = root / "confirmation"
    confirmation.mkdir(parents=True, exist_ok=True)
    write_json(confirmation / "freeze.json", frozen)
    return frozen, inputs


def _read_freeze(root, *, verify_models=False):
    frozen = read_json(root / "confirmation" / "freeze.json")
    identity = {key: value for key, value in frozen.items() if key != "freeze_id"}
    if frozen.get("format") != FORMAT or frozen.get("freeze_id") != canonical_hash(identity):
        raise ValueError("Confirmation freeze identity differs")
    _verify_file(frozen["protocol"])
    _verify_file(frozen["run_source_snapshot"])
    if source_snapshot()["source_sha256"] != frozen["source_code_sha256"]:
        raise ValueError("Confirmation source code differs from the frozen implementation")
    if verify_models:
        for entry in frozen["models"]:
            if entry["eligible"]:
                for item in entry["files"].values():
                    _verify_file(item)
    return frozen


def _prepared_binding(root, frozen, inputs=None):
    confirmation = root / "confirmation"
    binding_path = confirmation / "data_binding.json"
    if binding_path.is_file():
        binding = read_json(binding_path)
        if binding.get("freeze_id") != frozen["freeze_id"]:
            raise ValueError("Prepared confirmation belongs to a different model freeze")
        _verify_file(binding["preparation"])
        for item in binding["files"].values():
            _verify_file(item)
        return binding
    prepared = confirmation / "prepared"
    # A completed receipt after interruption is reused; no second sample is drawn.
    report_path = prepared / "preparation.json"
    if report_path.is_file():
        report = read_json(report_path)
    else:
        if inputs is None:
            inputs = load_inputs(root / "protocol.json")
        report = prepare_confirmation(frozen["source_archive"]["path"], inputs, prepared,
            previous_prepared_dir=frozen["previous_prepared_dir"], **frozen["data_recipe"])
    recipe = frozen["data_recipe"]
    if (report.get("status") != "prepared" or report.get("seed") != recipe["seed"]
            or report.get("schema") != frozen["schema"]
            or report.get("source", {}).get("archive_sha256") != frozen["source_archive"]["sha256"]):
        raise ValueError("Prepared confirmation data differs from the frozen sampling recipe")
    from ids.context_cache.lab_data import FILENAMES
    protocol = read_json(frozen["protocol"]["path"])
    expected_seen = {("round13", role): frozen["previous_files"][filename]["sha256"]
                     for role, filename in FILENAMES.items()}
    expected_seen.update({("round14", role): item["sha256"] for role, item in protocol["files"].items()})
    seen_evidence = report.get("seen_evidence", [])
    observed_seen = {(item["round"], item["role"]): item["sha256"] for item in seen_evidence}
    if len(seen_evidence) != 6 or observed_seen != expected_seen:
        raise ValueError("Prepared confirmation excluded a different set of six seen CSVs")
    files = {}
    for role, number_key in (("calibration", "calibration_rows"), ("evaluation", "evaluation_rows")):
        item = report["files"][role]
        if item["rows"] != recipe[number_key]:
            raise ValueError("Prepared confirmation row count differs from frozen recipe")
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(prepared.resolve()):
            raise ValueError("Prepared confirmation CSV is outside its fixed directory")
        files[role] = {**_file(path, item["sha256"]), "rows": item["rows"]}
    binding = {"freeze_id": frozen["freeze_id"], "preparation": _file(report_path),
               "files": files, "scope": report["scope"]}
    write_json(binding_path, binding)
    return binding


def _confirmation_inputs(frozen, binding):
    schema = CSVSchema.from_dict(frozen["schema"])
    frames = {}
    for role, item in binding["files"].items():
        frame = read_csv(_verify_file(item), schema)
        if len(frame) != item["rows"]:
            raise ValueError("Confirmation CSV row count differs")
        frames[role] = frame
    if np.any(labels(frames["calibration"], schema) != 0):
        raise ValueError("Confirmation calibration must contain only normal rows")
    return {"schema": schema, "frames": frames,
            "spec": {"settings": copy.deepcopy(frozen["settings"]), "files": binding["files"]}}


def _latest(root, model_id):
    folder = root / "confirmation" / "models" / model_id
    attempts = sorted(path for path in folder.glob("attempt-*") if path.is_dir()) if folder.exists() else []
    current = attempts[-1] if attempts else None
    result = read_json(current / "result.json") if current and (current / "result.json").is_file() else None
    return folder, current, result


def _verify_result(result, frozen, binding):
    if result.get("freeze_id") != frozen["freeze_id"]:
        raise ValueError("Confirmation result belongs to another model freeze")
    if result.get("data_sha256") != {role: item["sha256"] for role, item in binding["files"].items()}:
        raise ValueError("Confirmation result belongs to different prepared data")
    if result["status"] == "complete":
        for item in result["outputs"].values():
            _verify_file(item)


def _predictor(entry, schema, batch_size):
    files, descriptor = entry["files"], entry["descriptor"]
    kind = descriptor["kind"]
    if kind == "tabm":
        from .preprocessing import Representation
        from .tabm_training import load_predictor
        representation = Representation.load(files["preprocessing"]["path"])
        model = load_predictor(files["model"]["path"], device="cpu")
        return lambda frame: model.predict_scores(representation.transform(frame), batch_size=batch_size)
    if kind == "foundation":
        from .foundation import load_foundation
        from .preprocessing import raw_numeric
        model = load_foundation(files["model"]["path"], checkpoint_path=files["checkpoint"]["path"], device="cpu")
        return lambda frame: model.predict_scores(raw_numeric(frame, schema), batch_size=batch_size)
    if kind == "reference":
        from ids.context_cache.csv_workflow import CSVDetector
        model = CSVDetector.load(descriptor["directory"])
        return lambda frame: model.predict_frame(frame)["score"].to_numpy(dtype=np.float64)
    raise ValueError("Unsupported frozen model kind")


def confirm_worker(run_root, model_id, *, retry_failed=False):
    """Score one frozen final model in a fresh process; never train or select it."""
    import torch
    from threadpoolctl import threadpool_limits
    from .foundation import FoundationUnavailable

    if not isinstance(retry_failed, bool):
        raise ValueError("retry_failed must be boolean")
    root = Path(run_root).resolve()
    frozen = _read_freeze(root)
    entry = next(item for item in frozen["models"] if item["id"] == model_id)
    if not entry["eligible"]:
        raise ValueError("Model was unavailable at confirmation freeze; it cannot be added later")
    for item in entry["files"].values():
        _verify_file(item)
    binding = _prepared_binding(root, frozen)
    folder, previous, old_result = _latest(root, model_id)
    if old_result is not None:
        _verify_result(old_result, frozen, binding)
        if not (retry_failed and old_result["status"] in {"failed", "unavailable"}):
            return old_result
    folder.mkdir(parents=True, exist_ok=True)
    number = sum(path.is_dir() for path in folder.glob("attempt-*")) + 1
    output = folder / f"attempt-{number:03d}"
    output.mkdir()
    write_json(output / "state.json", {"status": "running", "freeze_id": frozen["freeze_id"]})
    result = {"model_id": model_id, "freeze_id": frozen["freeze_id"],
              "data_sha256": {role: item["sha256"] for role, item in binding["files"].items()},
              "original_task_status": entry["original_status"], "training_updates": 0}
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        inputs = _confirmation_inputs(frozen, binding)
        with threadpool_limits(limits=1):
            predict = _predictor(entry, inputs["schema"], frozen["settings"]["query_batch_size"])
            arrays = {role: predict(frame) for role, frame in inputs["frames"].items()}
            descriptor = {**entry["descriptor"], "confirmation_freeze_id": frozen["freeze_id"],
                          "frozen_model_id": model_id}
            metrics = runner._write_scores(output, inputs, arrays, descriptor)
        metrics["threshold"]["scope"] = "normal calibration for same-source unseen communication groups"
        metrics["confirmation_scope"] = binding["scope"]
        write_json(output / "threshold.json", metrics["threshold"])
        write_json(output / "metrics.json", metrics)
        result.update(status="complete", metrics=metrics,
                      cache_rebuilt_without_weight_updates=entry["descriptor"]["kind"] == "foundation",
                      outputs={name: _file(output / name) for name in (
                          "threshold.json", "metrics.json", "calibration.csv", "evaluation.csv")})
    except FoundationUnavailable as exc:
        result.update(status="unavailable", reason=str(exc))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "result.json", result)
    write_json(output / "state.json", {"status": result["status"], "freeze_id": frozen["freeze_id"]})
    return result


def _summarize(root, frozen, binding=None):
    rows, completed = [], {}
    for entry in frozen["models"]:
        _, _, result = _latest(root, entry["id"])
        row = {"id": entry["id"], "kind": entry["task"]["kind"],
               "original_task_status": entry["original_status"], "freeze_status": entry["freeze_status"],
               "status": "not_available_at_freeze" if not entry["eligible"] else "pending_or_interrupted"}
        if result is not None:
            _verify_result(result, frozen, binding)
            row["status"] = result["status"]
            if result["status"] == "complete":
                completed[entry["id"]] = result
                metric = result["metrics"]
                row.update({name: metric[name] for name in ("tp", "fp", "fn", "tpr", "fpr", "auc", "normalized_pauc_0_01")})
                row["scanning_tp"] = metric.get("families", {}).get("Scanning", {}).get("tp")
        rows.append(row)
    pairs = {}
    if "reference_full" in completed:
        baseline = pd.read_csv(completed["reference_full"]["outputs"]["evaluation.csv"]["path"], dtype={"row_id": str})
        for model_id, result in completed.items():
            frame = pd.read_csv(result["outputs"]["evaluation.csv"]["path"], dtype={"row_id": str})
            if (frame["row_id"].tolist() != baseline["row_id"].tolist()
                    or not np.array_equal(frame["label"], baseline["label"])):
                raise ValueError("New-holdout paired scores have different row identities")
            pairs[model_id] = paired(baseline["label"].to_numpy(), baseline["predicted_attack"].to_numpy(),
                                     frame["predicted_attack"].to_numpy())
    eligible = sum(entry["eligible"] for entry in frozen["models"])
    all_available_scored = eligible > 0 and len(completed) == eligible
    status = ("not_run_no_available_models" if not eligible else "complete" if len(completed) == len(rows)
              else "complete_with_unavailable_models" if all_available_scored else "partial")
    summary = {"status": status, "freeze_id": frozen["freeze_id"], "selection_policy": frozen["policy"],
               "scope": binding["scope"] if binding else "confirmation data not prepared",
               "planned_models": len(rows), "available_at_freeze": eligible,
               "models_scored": len(completed), "all_available_models_scored": all_available_scored,
               "all_planned_models_scored": len(completed) == len(rows),
               "results": rows, "paired_vs_new_full": pairs,
               "no_post_confirmation_model_or_seed_selection": True,
               "data_binding": str(root / "confirmation" / "data_binding.json") if binding else None}
    write_json(root / "confirmation" / "summary.json", summary)
    pd.DataFrame(rows).to_csv(root / "confirmation" / "summary.csv", index=False)
    write_json(root / "confirmation" / "paired_vs_new_full.json", pairs)
    return summary


def run_confirmation(run_root, source_zip, previous_prepared_dir, *, tabpfn_python=None, retry_failed=False):
    """Freeze once, prepare once, and score every eligible model in isolated workers."""
    if not isinstance(retry_failed, bool):
        raise ValueError("retry_failed must be boolean")
    root = Path(run_root).resolve()
    freeze_path = root / "confirmation" / "freeze.json"
    if freeze_path.is_file():
        frozen, inputs = _read_freeze(root, verify_models=True), None
        if (Path(source_zip).resolve() != Path(frozen["source_archive"]["path"])
                or Path(previous_prepared_dir).resolve() != Path(frozen["previous_prepared_dir"])):
            raise ValueError("Confirmation resume requires the original source and previous prepared directory")
        _verify_file(frozen["source_archive"])
        for item in frozen["previous_files"].values():
            _verify_file(item)
    else:
        frozen, inputs = _create_freeze(root, source_zip, previous_prepared_dir, tabpfn_python)
    if not any(entry["eligible"] for entry in frozen["models"]):
        return _summarize(root, frozen)
    binding = _prepared_binding(root, frozen, inputs)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    for entry in frozen["models"]:
        if not entry["eligible"]:
            continue
        folder, previous_attempt, old = _latest(root, entry["id"])
        if old is not None:
            _verify_result(old, frozen, binding)
            if not (retry_failed and old["status"] in {"failed", "unavailable"}):
                continue
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"worker-{len(list(folder.glob('worker-*.log'))) + 1:03d}.log"
        print(f"Confirming frozen model {entry['id']}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                command = [entry["python"], "-m", "ids.route_batch", "_confirm-worker",
                           "--run-dir", str(root), "--model-id", entry["id"]]
                if retry_failed:
                    command.append("--retry-failed")
                child = subprocess.run(command, env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
                failure = {"status": "failed", "error": "confirmation worker exited without result",
                           "exit_code": child.returncode}
            except OSError as exc:
                failure = {"status": "unavailable", "error": str(exc), "error_type": type(exc).__name__}
        _, latest, result = _latest(root, entry["id"])
        if result is None or (old is not None and latest == previous_attempt):
            if latest is None or latest == previous_attempt:
                number = sum(path.is_dir() for path in folder.glob("attempt-*")) + 1
                latest = folder / f"attempt-{number:03d}"
                latest.mkdir()
            failure.update(model_id=entry["id"], freeze_id=frozen["freeze_id"],
                data_sha256={role: item["sha256"] for role, item in binding["files"].items()})
            write_json(latest / "result.json", failure)
        _summarize(root, frozen, binding)
    return _summarize(root, frozen, binding)
