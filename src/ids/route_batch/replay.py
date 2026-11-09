"""Read-only v1 TabICL replay diagnosis in isolated build-path workers.

The CPU rebuild is a distinct, named deployment path. Its recalibrated scores
must never be substituted into the original GPU-build result or its cost.
"""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.context_cache.preprocessing import labels
from .data import load_inputs
from .foundation import SCORE_EPSILON, load_foundation
from .metrics import decisions, evaluate, paired, threshold
from .preprocessing import raw_numeric
from .protocol import REPO, source_snapshot, validate

FORMAT = "tracelight-tabicl-build-path-replay-v1"
V1_SOURCE_SHA256 = "67f931d0da7ac59bbbdb6de8dc04fdeb29b800a4a814ece9c3eb55cee4c7b047"
REPLAY_ATOL = 3e-5
REPLAY_RTOL = 3e-5
TASKS = {"tabicl_official_1": 1, "tabicl_official_8": 8}
BUILD_PATHS = ("original", "cpu")


def _file(path, expected=None):
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f"original artifact SHA-256 mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("original recipe path escapes its attempt")
    return path


def _output_path(original, output):
    root, target = Path(original).resolve(), Path(output).resolve()
    if target.is_relative_to(root) or root.is_relative_to(target):
        raise ValueError("diagnostic output must be separate from the original run")
    if target.exists():
        raise FileExistsError("diagnostic output already exists; preserve the previous diagnosis")
    return target


def validate_replay_inputs(original_run_dir, task_id, *, checkpoint_path=None):
    """Verify original files before loading a trusted recipe; never build a model.

    Returns inputs/old scores plus a JSON-serializable ``binding``. Development
    runs require the actual round-17 v1 source fingerprint, not today's source.
    """
    if task_id not in TASKS:
        raise ValueError("replay is restricted to tabicl_official_1 and tabicl_official_8")
    root = Path(original_run_dir).resolve()
    spec = validate(read_json(root / "protocol.json"))
    snapshot = read_json(root / "source_snapshot.json")
    if canonical_hash(snapshot.get("files")) != snapshot.get("source_sha256"):
        raise ValueError("original source snapshot is internally inconsistent")
    if spec["data_kind"] == "development" and snapshot["source_sha256"] != V1_SOURCE_SHA256:
        raise ValueError("original source is not the frozen round-17 v1 source")
    preflight = read_json(root / "preflight.json")
    if (preflight.get("protocol_sha256") != canonical_hash(spec)
            or canonical_hash(preflight.get("source")) != canonical_hash(snapshot)):
        raise ValueError("original preflight does not bind this protocol and source")
    tasks = [task for task in spec["tasks"] if task["id"] == task_id]
    if len(tasks) != 1 or tasks[0].get("model_kind") != "tabicl" or tasks[0].get("n_estimators") != TASKS[task_id]:
        raise ValueError("original task does not match the registered TabICL recipe")
    attempts = sorted(p for p in (root / "tasks" / task_id).glob("attempt-*") if p.is_dir())
    if not attempts:
        raise ValueError("original task has no completed attempt")
    attempt = attempts[-1].resolve()
    result = read_json(attempt / "result.json")
    if result.get("status") not in {"complete", "measurement_failed"} or result.get("task") != tasks[0]:
        raise ValueError("original task lacks a completed fit and score result")
    descriptor = result.get("model", {})
    if descriptor.get("kind") != "foundation" or not descriptor.get("model_sha256"):
        raise ValueError("original foundation recipe has no registered SHA-256")
    recipe = _inside(attempt, descriptor["model_file"])
    recipe_identity = _file(recipe, descriptor["model_sha256"])
    metadata = descriptor["metadata"]
    settings = spec["settings"]
    if (metadata.get("kind") != "tabicl" or metadata.get("seed") != settings["foundation_seed"]
            or metadata.get("requested_estimators") != TASKS[task_id]
            or metadata.get("actual_estimators") != TASKS[task_id]
            or metadata.get("build_device") != settings["device"]
            or metadata.get("score_device") != "cpu" or metadata.get("threads") != 1
            or metadata.get("query_batch_size") != 128
            or metadata.get("score_clip_epsilon") != SCORE_EPSILON):
        raise ValueError("original recipe metadata differs from frozen build/score settings")
    checkpoint = Path(checkpoint_path or spec["checkpoint"]["path"]).resolve()
    checkpoint_identity = _file(checkpoint, spec["checkpoint"]["sha256"])
    if checkpoint_identity["sha256"] != metadata.get("checkpoint_sha256"):
        raise ValueError("original recipe checkpoint differs from the registered checkpoint")
    inputs = load_inputs(root / "protocol.json")
    # Hash verification above must precede joblib deserialization.
    payload = joblib.load(recipe)
    if (not isinstance(payload, dict) or payload.get("format") != "tracelight-foundation-rebuild-v1"
            or payload.get("kind") != "tabicl" or payload.get("metadata") != metadata):
        raise ValueError("original recipe payload and descriptor differ")
    support = inputs["frames"]["support"]
    x, y = raw_numeric(support, inputs["schema"]), labels(support, inputs["schema"])
    if (not np.array_equal(payload.get("support_x"), x, equal_nan=True)
            or not np.array_equal(payload.get("support_y"), y)):
        raise ValueError("original recipe support arrays differ from registered P")
    registration = read_json(attempt / "threshold.json")
    if registration.get("model") != descriptor or registration.get("calibration_sha256") != spec["files"]["calibration"]["sha256"]:
        raise ValueError("original threshold does not bind this model and calibration set")
    saved = {}
    identities = {"protocol": _file(root / "protocol.json"), "source_snapshot": _file(root / "source_snapshot.json"),
                  "preflight": _file(root / "preflight.json"), "result": _file(attempt / "result.json"),
                  "recipe": recipe_identity, "threshold": _file(attempt / "threshold.json"),
                  "checkpoint": checkpoint_identity}
    schema = inputs["schema"]
    for role in ("calibration", "evaluation"):
        path = attempt / f"{role}.csv"
        frame = pd.read_csv(path, dtype={"row_id": str}, float_precision="round_trip")
        registered = inputs["frames"][role]
        if (frame.row_id.tolist() != registered[schema.id_column].astype(str).tolist()
                or not np.array_equal(frame.label, labels(registered, schema))
                or frame.attack_family.astype(str).tolist() != registered[schema.family_column].astype(str).tolist()
                or not np.isfinite(frame.score).all()):
            raise ValueError(f"original {role} score IDs/labels/families differ from registered inputs")
        if not np.array_equal(frame.predicted_attack, decisions(frame.score, registration).astype(int)):
            raise ValueError(f"original {role} saved decisions differ from its threshold")
        saved[role] = frame
        identities[role + "_scores"] = _file(path)
    actual = threshold(saved["calibration"].score, settings["alpha"])
    for key in ("alpha", "n_normal", "rank", "threshold", "comparison", "comparison_dtype", "calibration_fp"):
        if registration.get(key) != actual[key]:
            raise ValueError("original calibration threshold cannot be reproduced from saved C scores")
    evaluation = evaluate(saved["evaluation"].label, saved["evaluation"].score, actual,
                          saved["evaluation"].attack_family)
    old_metrics = result.get("metrics", {})
    for key in ("tp", "fp", "fn", "tn", "attack_rows", "normal_rows", "families"):
        if evaluation[key] != old_metrics.get(key):
            raise ValueError("original saved score metrics differ from original result")
    for role in ("support", "calibration", "evaluation"):
        identities[role + "_input"] = _file(inputs["paths"][role], spec["files"][role]["sha256"])
    binding = {"format": FORMAT, "original_run": str(root), "task": tasks[0],
               "original_task_status": result["status"], "original_source_sha256": snapshot["source_sha256"],
               "original_protocol_sha256": canonical_hash(spec), "files": identities,
               "support_rows": len(x), "support_ids_sha256": canonical_hash(support[schema.id_column].astype(str).tolist()),
               "seed": metadata["seed"], "original_build_device": metadata["build_device"],
               "score_device": "cpu", "query_batch_size": 128,
               "tolerance": {"atol": REPLAY_ATOL, "rtol": REPLAY_RTOL,
                             "reference_side": "original/reference scores"}}
    return {"inputs": inputs, "saved": saved, "registration": registration,
            "descriptor": descriptor, "recipe": recipe, "checkpoint": checkpoint, "binding": binding}


def score_difference(reference, observed):
    reference, observed = np.asarray(reference, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    if reference.shape != observed.shape or not np.isfinite(reference).all() or not np.isfinite(observed).all():
        raise ValueError("replay scores must be aligned and finite")
    error = observed - reference
    tolerance = REPLAY_ATOL + REPLAY_RTOL * np.abs(reference)
    return {"rows": len(reference), "bitwise_equal": bool(np.array_equal(reference, observed)),
            "outside_original_tolerance": int((np.abs(error) > tolerance).sum()),
            "max_abs_difference": float(np.max(np.abs(error), initial=0)),
            "mean_abs_difference": float(np.abs(error).mean()) if len(error) else 0.,
            "atol": REPLAY_ATOL, "rtol": REPLAY_RTOL}


def _compare_old(context, role, outputs, new_registration):
    saved, old_registration = context["saved"][role], context["registration"]
    old = saved.score.to_numpy(dtype=np.float64)
    scores = outputs["scores"]
    y, family = saved.label.to_numpy(), saved.attack_family.to_numpy()
    old_pred = decisions(old, old_registration)
    fixed_pred = decisions(scores, old_registration)
    new_pred = decisions(scores, new_registration)
    return {"scores_vs_original": score_difference(old, scores),
            "raw_probability_diagnostics": outputs["diagnostics"],
            "old_threshold_decision_flips": int((old_pred != fixed_pred).sum()),
            "recalibrated_decision_flips": int((old_pred != new_pred).sum()),
            "old_threshold_metrics": evaluate(y, scores, old_registration, family),
            "recalibrated_metrics": evaluate(y, scores, new_registration, family),
            "old_threshold_paired_vs_original": paired(y, old_pred, fixed_pred),
            "recalibrated_paired_vs_original": paired(y, old_pred, new_pred)}


def replay_path_worker(original_run_dir, task_id, output_dir, *, build_path="original", checkpoint_path=None):
    """Run one named construction path; caller supplies a fresh isolated process."""
    if build_path not in BUILD_PATHS:
        raise ValueError("build_path must be original or cpu")
    output = _output_path(original_run_dir, output_dir)
    started = time.perf_counter()
    context = validate_replay_inputs(original_run_dir, task_id, checkpoint_path=checkpoint_path)
    validation_seconds = time.perf_counter() - started
    output.mkdir(parents=True)
    write_json(output / "binding.json", context["binding"])
    write_json(output / "source_snapshot.json", source_snapshot())
    try:
        build_start = time.perf_counter()
        model = load_foundation(context["recipe"], checkpoint_path=context["checkpoint"], device="cpu",
                                preserve_build_device=build_path == "original", diagnostics=True)
        build_seconds = time.perf_counter() - build_start
        inputs = context["inputs"]
        schema = inputs["schema"]
        # Match the original runner's C -> E query sequence before any probe.
        # Otherwise query-state changes could be mistaken for build-device effects.
        arrays, times = {}, {}
        for role in ("calibration", "evaluation"):
            role_start = time.perf_counter()
            arrays[role] = model.predict_outputs(raw_numeric(inputs["frames"][role], schema), diagnostics=True)
            times[role + "_raw_transform_and_score_seconds"] = time.perf_counter() - role_start
        query = raw_numeric(inputs["frames"]["evaluation"].iloc[:128], schema)
        query_start = time.perf_counter()
        first = model.predict_outputs(query, diagnostics=True)
        first_seconds = time.perf_counter() - query_start
        query_start = time.perf_counter()
        repeat = model.predict_outputs(query, diagnostics=True)
        repeat_seconds = time.perf_counter() - query_start
        old_probe = context["saved"]["evaluation"].iloc[:len(query)]
        old_score = old_probe.score.to_numpy(dtype=np.float64)
        old_threshold = context["registration"]
        pd.DataFrame({"row_id": old_probe.row_id, "label": old_probe.label,
                      "attack_family": old_probe.attack_family, "original_score": old_score,
                      "same_path_initial_E_score": arrays["evaluation"]["scores"][:len(query)],
                      "probe_minus_initial_E_score": first["scores"] - arrays["evaluation"]["scores"][:len(query)],
                      "score": first["scores"], "repeat_score": repeat["scores"],
                      "difference_vs_original": first["scores"] - old_score,
                      "allowed_abs_difference": REPLAY_ATOL + REPLAY_RTOL * np.abs(old_score),
                      "within_original_tolerance": np.isclose(first["scores"], old_score, atol=REPLAY_ATOL, rtol=REPLAY_RTOL),
                      "original_predicted_attack": decisions(old_score, old_threshold).astype(int),
                      "old_threshold_predicted_attack": decisions(first["scores"], old_threshold).astype(int),
                      "old_threshold_flip": decisions(old_score, old_threshold) != decisions(first["scores"], old_threshold),
                      "original_implied_clipped_attack_probability": np.exp(-np.logaddexp(0, -old_score)),
                      "raw_probability_normal": first["probabilities"][:, 0],
                      "raw_probability_attack": first["probabilities"][:, 1],
                      "repeat_raw_probability_normal": repeat["probabilities"][:, 0],
                      "repeat_raw_probability_attack": repeat["probabilities"][:, 1],
                      }).to_csv(output / "probe.csv", index=False)
        model.save(output / "model.joblib")
        descriptor = {"kind": "foundation", "model_file": "model.joblib",
                      "model_sha256": sha256_file(output / "model.joblib"), "metadata": model.metadata,
                      "score_contract": context["descriptor"]["score_contract"],
                      "rebuild_device": model.metadata["build_device"], "diagnostic_build_path": build_path,
                      "original_task_id": task_id, "original_model_sha256": context["descriptor"]["model_sha256"]}
        write_json(output / "deployment.json", descriptor)
        from .runner import _write_scores
        scoring_inputs = {**inputs, "spec": copy.deepcopy(inputs["spec"])}
        scoring_inputs["spec"]["settings"]["reference"] = []
        metrics = _write_scores(output, scoring_inputs, {r: a["scores"] for r, a in arrays.items()}, descriptor)
        registration = metrics["threshold"]
        for role, values in arrays.items():
            frame = pd.read_csv(output / f"{role}.csv", float_precision="round_trip")
            frame["raw_probability_normal"] = values["probabilities"][:, 0]
            frame["raw_probability_attack"] = values["probabilities"][:, 1]
            frame.to_csv(output / f"{role}.csv", index=False)
        report = {"format": FORMAT, "status": "complete", "build_path": build_path,
                  "build_device": model.metadata["build_device"], "score_device": "cpu",
                  "model": descriptor, "query_batch_size": 128,
                  "call_order": ["calibration", "evaluation", "evaluation_first128_probe1", "evaluation_first128_probe2"],
                  "probe": {"rows": len(query), "vs_original": score_difference(old_score, first["scores"]),
                            "scope": "same-object probes after primary C then E; not fresh-load first queries",
                            "same_object_repeat": score_difference(first["scores"], repeat["scores"]),
                            "raw_probabilities_repeat_bitwise_equal": bool(np.array_equal(first["probabilities"], repeat["probabilities"])),
                            "max_raw_probability_repeat_abs_difference": float(np.max(np.abs(first["probabilities"]-repeat["probabilities"]), initial=0)),
                            "full_E_vs_later_probe": score_difference(arrays["evaluation"]["scores"][:len(query)], first["scores"])},
                  "roles": {role: _compare_old(context, role, values, registration) for role, values in arrays.items()},
                  "timing": {"original_input_validation_seconds": validation_seconds,
                             "load_recipe_verify_checkpoint_rebuild_and_cpu_transfer_seconds": build_seconds,
                             "reported_build_and_cpu_transfer_seconds": model.metadata.get("build_and_cpu_transfer_seconds"),
                             "probe_first_score_seconds": first_seconds, "probe_repeat_score_seconds": repeat_seconds,
                             **times, "worker_elapsed_seconds": time.perf_counter() - started,
                             "scope": "diagnostic wall time; not deployment latency benchmark; includes stated rebuild/transform operations"},
                  "old_raw_probabilities_available": False,
                  "original_probability_column": "inverse of saved clipped log odds only; not original raw predict_proba",
                  "cost_status": "not_measured; same-path independent cold rebuild must first reproduce this path's saved scores"}
        write_json(output / "path_report.json", report)
        files = {name: _file(output / name) for name in ("model.joblib", "deployment.json", "calibration.csv", "evaluation.csv",
                                                        "threshold.json", "metrics.json", "probe.csv", "path_report.json")}
        scores_match = all(item["scores_vs_original"]["outside_original_tolerance"] == 0
                           for item in report["roles"].values())
        old_decisions_match = report["roles"]["evaluation"]["old_threshold_decision_flips"] == 0
        recalibrated_decisions_match = report["roles"]["evaluation"]["recalibrated_decision_flips"] == 0
        result = {"status": "complete", "build_path": build_path, "output": str(output),
                  "files": files, "descriptor": descriptor,
                  "scores_match_original_within_tolerance": scores_match,
                  "old_threshold_E_decisions_match": old_decisions_match,
                  "recalibrated_E_decisions_match": recalibrated_decisions_match,
                  "diagnostic_matches_original": scores_match and old_decisions_match and recalibrated_decisions_match}
        del model
        gc.collect()
    except Exception as exc:
        result = {"status": "failed", "build_path": build_path, "output": str(output),
                  "error": str(exc), "traceback": traceback.format_exc()}
    write_json(output / "result.json", result)
    return result


def run_replay_diagnostic(original_run_dir, task_id, output_dir, *, checkpoint_path=None):
    """Launch independent original-build and CPU-build workers, then compare."""
    output = _output_path(original_run_dir, output_dir)
    context = validate_replay_inputs(original_run_dir, task_id, checkpoint_path=checkpoint_path)
    output.mkdir(parents=True)
    write_json(output / "binding.json", context["binding"])
    write_json(output / "source_snapshot.json", source_snapshot())
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    paths = {}
    for build_path in BUILD_PATHS:
        folder = output / build_path
        command = [sys.executable, "-m", "ids.route_batch", "_replay-path-worker", "--original-run",
                   str(Path(original_run_dir).resolve()), "--task", task_id, "--output", str(folder),
                   "--build-path", build_path, "--checkpoint", str(context["checkpoint"])]
        with (output / f"{build_path}-worker.log").open("w", encoding="utf8") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        result_path = folder / "result.json"
        result = read_json(result_path) if result_path.is_file() else {
            "status": "failed", "build_path": build_path, "output": str(folder),
            "error": "isolated replay worker exited without result", "exit_code": completed.returncode}
        if result.get("status") == "complete":
            if completed.returncode:
                raise ValueError("replay worker returned success artifact with nonzero exit")
            for item in result["files"].values():
                _file(item["path"], item["sha256"])
            if canonical_hash(read_json(folder / "binding.json")) != canonical_hash(context["binding"]):
                raise ValueError("replay worker original inputs changed after parent validation")
            result["report"] = read_json(folder / "path_report.json")
        paths[build_path] = result
    cross = {}
    if all(p["status"] == "complete" for p in paths.values()):
        for role in ("calibration", "evaluation"):
            left = pd.read_csv(output / "original" / f"{role}.csv", float_precision="round_trip")
            right = pd.read_csv(output / "cpu" / f"{role}.csv", float_precision="round_trip")
            if not np.array_equal(left.row_id, right.row_id):
                raise ValueError("build paths scored different row identities")
            cross[role] = {"scores_cpu_minus_original_path": score_difference(left.score, right.score),
                           "max_raw_probability_abs_difference": float(np.max(np.abs(
                               right.raw_probability_attack.to_numpy() - left.raw_probability_attack.to_numpy()), initial=0)),
                           "raw_attack_probability_zero_original_path": int((left.raw_probability_attack == 0).sum()),
                           "raw_attack_probability_zero_cpu_path": int((right.raw_probability_attack == 0).sum()),
                           "raw_attack_probability_one_original_path": int((left.raw_probability_attack == 1).sum()),
                           "raw_attack_probability_one_cpu_path": int((right.raw_probability_attack == 1).sum())}
            raw_difference = np.abs(right.raw_probability_attack.to_numpy() - left.raw_probability_attack.to_numpy())
            cross[role].update({
                "raw_attack_probability_abs_difference_quantiles": dict(zip(
                    ("p50", "p95", "p99"), np.quantile(raw_difference, [.5, .95, .99]).tolist())),
                "raw_attack_probabilities_different_rows": int((raw_difference != 0).sum()),
                "scores_tied_despite_different_raw_attack_probabilities": int(((raw_difference != 0)
                    & (left.score.to_numpy() == right.score.to_numpy())).sum()),
            })
    report = {"format": FORMAT, "status": "complete" if all(p["status"] == "complete" for p in paths.values()) else "partial",
              "task_id": task_id, "paths": paths, "cross_build_path": cross,
              "original_binding": context["binding"], "old_run_read_only": True,
              "scope": "engineering replay on original development C/E; not a new confirmation cohort",
              "original_scores_replaced": False, "tolerance_relaxed": False,
              "cost_policy": "benchmark only a named build path after independent same-path cold rebuild reproduces its own saved C/E scores"}
    # Detect original-file writes or concurrent mutation even after workers have finished.
    for item in context["binding"]["files"].values():
        _file(item["path"], item["sha256"])
    write_json(output / "report.json", report)
    return report
