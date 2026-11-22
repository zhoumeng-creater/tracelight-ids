"""Registered laboratory batch with resumable trajectories and frozen evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
from scripts.calibration_repair import data
from scripts.calibration_repair.run import (
    environment, text_digest, load_calibration, load_evaluation,
    calibration_threshold, checked_scores,
)
from scripts.comprehensive_batch.matrix import build_plan

DIRECTORY = Path("docs/comprehensive_batch_20261122")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf8", newline="\n")
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save_scores(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, scores=np.asarray(values, dtype=np.float64))
    os.replace(temporary, path)


def prepare(repo=REPO):
    registration = read(repo / DIRECTORY / "registration.json")
    plan = read(repo / DIRECTORY / "plan.json")
    if plan != build_plan() or data.json_digest(plan) != registration["plan_sha256"]:
        raise ValueError("Registered experiment matrix changed")
    if not registration["execution_enabled"]:
        raise ValueError("This design is not released for execution")
    for name, expected in registration["code_sha256_lf"].items():
        if text_digest(repo / name) != expected:
            raise ValueError(f"Registered source changed: {name}")
    rows, processor, _, _, manifest = data.prepare_inputs(repo)
    if data.json_digest(manifest) != registration["F_manifest_sha256"]:
        raise ValueError("Frozen F inputs changed")
    return registration, plan, rows, processor


def _error(error, status):
    return {"status": status, "type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "automatic_refit": False}


def _resource_error(error):
    message = str(error).lower()
    device_failures = ("out of memory", "illegal memory access", "device-side assert",
                       "device side assert", "cublas_status_alloc_failed", "cuda error",
                       "cuda driver", "cuda initialization", "unspecified launch failure")
    return isinstance(error, (MemoryError, OSError)) or any(value in message for value in device_failures)


def fit_hgb(rows, directory, *, job, reuse=None):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    from scripts.coverage_batch.run import ARMS, PARAMETERS, features
    arm = job["job_id"].removeprefix("HGB_")
    role, zero = ARMS[arm]
    selected = [row for row in rows if role == "F_expanded" or row["dataset_role"] == "F_base"]
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "model.joblib"
    started = time.perf_counter()
    imported = None
    if reuse is not None:
        reuse = Path(reuse)
        source = read(reuse / "source_receipt.json")
        old_registration = REPO / "docs/coverage_basis_20261119/batch/registration.json"
        if source["registration_sha256"] != data.file_digest(old_registration):
            raise ValueError("Coverage reuse is not the registered original four-fit batch")
        receipt = read(reuse / f"{arm}.fit_receipt.json")
        original = reuse / f"{arm}.joblib"
        if receipt["fit_rows"] != len(selected):
            raise ValueError("Coverage model training cohort differs")
        if data.file_digest(original) != receipt["model_sha256"]:
            raise ValueError("Coverage model reuse hash differs")
        shutil.copyfile(original, artifact)
        imported = {"path": str(original), "original_receipt": receipt}
        fit_seconds = receipt["fit_seconds"]
    else:
        with threadpool_limits(limits=4):
            model = HistGradientBoostingClassifier(**PARAMETERS)
            model.fit(features(selected, zero), np.asarray([row["label"] for row in selected]))
            fit_seconds = time.perf_counter() - started
            joblib.dump(model, artifact)
    return {"status": "complete", "family": "hgb", "job_id": job["job_id"],
            "fit_calls": int(imported is None), "reused": imported,
            "fit_rows": len(selected), "fit_seconds": fit_seconds,
            "snapshots": [{"step": 200, "path": artifact.name,
                           "sha256": data.file_digest(artifact), "bytes": artifact.stat().st_size,
                           "cost": {"original_fit_seconds": fit_seconds,
                                    "fit_calls_in_this_batch": int(imported is None)}}]}


class HGBPredictor:
    def __init__(self, path, zero_ports):
        import joblib
        self.model = joblib.load(path)
        self.zero_ports = zero_ports

    def score(self, rows):
        from scripts.coverage_batch.run import features
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=4):
            return self.model.decision_function(features(rows, self.zero_ports))


def predictor_for(path, metadata, *, device="cuda:0"):
    if metadata["family"] == "hgb":
        return HGBPredictor(path, metadata["job_id"].endswith("ports_zero"))
    from scripts.comprehensive_batch.training import load_predictor
    return load_predictor(path, device=device, threads=4)


def _collect(directory, job, family, snapshots, job_status, output):
    result = {}
    for snapshot in snapshots:
        step = int(snapshot["step"])
        if family == "tabm" and step not in job["checkpoints"]:
            raise ValueError("Unregistered snapshot step")
        path = directory / snapshot["path"]
        if not path.is_file() or data.file_digest(path) != snapshot["sha256"]:
            raise ValueError(f"Checkpoint binding differs: {path}")
        identity = job["job_id"] if family == "hgb" else f"{job['job_id']}_step{step}"
        result[identity] = {**job, "family": family, "step": None if family == "hgb" else step, "status": "complete",
                            "job_status": job_status, "path": path.relative_to(output).as_posix(),
                            "sha256": snapshot["sha256"], "bytes": path.stat().st_size}
        if "cost" in snapshot:
            result[identity]["training_cost_at_snapshot"] = snapshot["cost"]
    return result


def train_phase(output, rows, processor, plan, *, resume=False, fit_fn=None,
                hgb_fn=None, coverage_results=None):
    if (output / "calibration_started.json").exists():
        raise RuntimeError("Calibration has started: training can never resume in this batch")
    if fit_fn is None:
        from scripts.comprehensive_batch.training import fit
        fit_fn = fit
    hgb_fn = hgb_fn or fit_hgb
    statuses, models, pairing = {}, {}, {}
    jobs = [(job, "hgb") for job in plan["hgb_jobs"]] + [(job, "tabm") for job in plan["tabm_jobs"]]
    for position, (job, family) in enumerate(jobs, 1):
        identity = job["job_id"]
        directory = output / "fits" / identity
        status_path = output / "job_statuses" / f"{identity}.json"
        print(json.dumps({"phase": "fit", "job": identity, "position": position,
                          "total_jobs": len(jobs)}, ensure_ascii=False), flush=True)
        cached = read(status_path) if status_path.exists() else None
        if cached and not resume:
            raise FileExistsError("Use --resume for the same registered batch")
        if cached and cached["status"] in {"complete", "failed_numerical", "failed_job"}:
            summary = cached
        else:
            write(status_path, {"status": "running", "job": job})
            started = time.perf_counter()
            attempt_status = "interrupted"
            try:
                if family == "hgb":
                    summary = hgb_fn(rows, directory, job=job, reuse=coverage_results)
                else:
                    summary = fit_fn(rows, processor, directory, job=job,
                                     device="cuda:0", threads=4, resume=bool(resume and directory.exists()))
                summary = {**summary, "status": "complete", "job": job}
                attempt_status = "complete"
            except KeyboardInterrupt:
                write(status_path, {"status": "interrupted", "job": job})
                raise
            except Exception as error:
                if _resource_error(error):
                    attempt_status = "interrupted_resource"
                    write(status_path, {**_error(error, "interrupted_resource"), "job": job})
                    raise
                status = "failed_numerical" if isinstance(error, FloatingPointError) else "failed_job"
                attempt_status = status
                snapshots = read(directory / "snapshots.json") if (directory / "snapshots.json").exists() else []
                summary = {**_error(error, status), "job": job, "snapshots": snapshots}
                if (directory / "cost.json").exists():
                    summary["last_persisted_cost"] = read(directory / "cost.json")
                print(json.dumps({"job": identity, "status": status, "error": str(error)}), flush=True)
            finally:
                elapsed = time.perf_counter() - started
                attempts_path = output / "job_invocations" / f"{identity}.json"
                attempts = read(attempts_path) if attempts_path.exists() else []
                attempts.append({"attempt": len(attempts) + 1, "status": attempt_status,
                                 "wall_seconds": elapsed, "resume_requested": resume,
                                 "includes_initialization_diagnostics_and_io": True})
                write(attempts_path, attempts)
            write(status_path, summary)
        if summary["status"] == "complete" and family == "tabm":
            # All configurations share their seed's initialization and BCE draws.
            signature = (summary["initial_state_sha256"], summary["sampler_draw_sha256"])
            if job["seed"] in pairing and pairing[job["seed"]] != signature:
                raise ValueError("Paired initialization or sampling changed across configurations")
            pairing[job["seed"]] = signature
        models.update(_collect(directory, job, family, summary.get("snapshots", []), summary["status"], output))
        statuses[identity] = summary
        write(output / "job_statuses.json", statuses)
    if not models:
        raise RuntimeError("No registered model snapshot completed; evaluation is unavailable")
    write(output / "models_frozen.json", {"models": models, "jobs": statuses,
                                          "all_registered_jobs_terminal": True,
                                          "no_training_after_this_barrier": True})
    return models


def evaluate_phase(output, plan, models, *, calibration_loader=None, evaluation_loader=None,
                   loader=None, writer=None, repo=REPO):
    calibration_loader = calibration_loader or load_calibration
    evaluation_loader = evaluation_loader or load_evaluation
    loader = loader or predictor_for
    if writer is None:
        from scripts.comprehensive_batch.evaluation import evaluate_saved
        writer = evaluate_saved
    if not (output / "models_frozen.json").is_file():
        raise RuntimeError("All model states must be frozen before opening C")
    barrier = {"models_manifest_sha256": data.file_digest(output / "models_frozen.json"),
               "training_closed": True}
    if (output / "calibration_started.json").exists() and read(output / "calibration_started.json") != barrier:
        raise ValueError("Frozen model manifest changed after calibration started")
    write(output / "calibration_started.json", barrier)
    calibration = calibration_loader(repo)
    c_ids = [row["flow_id"] for row in calibration]
    thresholds, costs = {}, {}
    c_directory, score_directory = output / "calibration_scores", output / "scores"
    c_directory.mkdir(exist_ok=True)
    score_directory.mkdir(exist_ok=True)
    for identity, meta in models.items():
        path = output / meta["path"]
        if data.file_digest(path) != meta["sha256"]:
            raise ValueError("Frozen model modified before calibration")
        receipt_path = c_directory / f"{identity}.json"
        score_path = c_directory / f"{identity}.npz"
        if receipt_path.exists():
            receipt = read(receipt_path)
            if (receipt["model_sha256"] != meta["sha256"] or receipt["C_ids_sha256"] != data.json_digest(c_ids)
                    or data.file_digest(score_path) != receipt["scores_sha256"]):
                raise ValueError("Frozen calibration cache changed")
            thresholds[identity], costs[identity] = receipt["threshold"], receipt["cost"]
            continue
        predictor = loader(path, meta)
        started = time.perf_counter()
        values = checked_scores(predictor, calibration)
        costs[identity] = {"C_score_seconds": time.perf_counter() - started}
        del predictor
        thresholds[identity] = calibration_threshold(values)
        save_scores(score_path, values)
        write(receipt_path, {"model_sha256": meta["sha256"], "C_ids_sha256": data.json_digest(c_ids),
                             "scores_sha256": data.file_digest(score_path), "threshold": thresholds[identity],
                             "cost": costs[identity]})
    write(output / "thresholds.json", thresholds)
    write(output / "thresholds_frozen.json", {"thresholds_sha256": data.file_digest(output / "thresholds.json"),
                                              "models": list(models), "threshold_searches": 0})
    all_rows, cohorts, events, units = evaluation_loader(repo)
    if set(c_ids) != set(cohorts["C"]):
        raise ValueError("C identities differ between phase loaders")
    identities = [row["flow_id"] for row in all_rows]
    c_set = set(c_ids)
    remaining = [row for row in all_rows if row["flow_id"] not in c_set]
    write(output / "score_identities.json", identities)
    identity_hash = data.json_digest(identities)
    for identity, meta in models.items():
        path = output / meta["path"]
        if data.file_digest(path) != meta["sha256"]:
            raise ValueError("Frozen model modified before development")
        score_path = score_directory / f"{identity}.npz"
        receipt_path = score_directory / f"{identity}.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            if (receipt["model_sha256"] != meta["sha256"] or receipt["identity_sha256"] != identity_hash
                    or data.file_digest(score_path) != receipt["scores_sha256"]):
                raise ValueError("Frozen development cache changed")
            costs[identity].update(receipt["cost"])
            continue
        predictor = loader(path, meta)
        started = time.perf_counter()
        values = checked_scores(predictor, remaining)
        cost = {"remaining_score_seconds": time.perf_counter() - started}
        del predictor
        with np.load(c_directory / f"{identity}.npz", allow_pickle=False) as saved:
            c_values = saved["scores"]
        mapping = dict(zip(c_ids, c_values)) | dict(zip((row["flow_id"] for row in remaining), values))
        save_scores(score_path, [mapping[fid] for fid in identities])
        costs[identity].update(cost)
        write(receipt_path, {"model_sha256": meta["sha256"], "identity_sha256": identity_hash,
                             "scores_sha256": data.file_digest(score_path), "cost": cost})
        print(json.dumps({"phase": "scoring", "snapshot": identity}), flush=True)
    write(output / "scoring_cost.json", costs)
    result = writer(all_rows, cohorts, events, units, models, thresholds, score_directory,
                    output / "evaluation", plan)
    write(output / "evaluation_complete.json", result)
    return result


def benchmark_phase(output, rows, models):
    # Measure every main-configuration arm/seed plus each HGB. No winner selection.
    selected = {identity: meta for identity, meta in models.items()
                if meta["family"] == "hgb" or (meta.get("primary_configuration") and meta["step"] == 15600)}
    write(output / "benchmark_rows.json", rows[:1024])
    for identity in selected:
        result = output / "benchmarks" / f"{identity}.json"
        if result.exists():
            saved = read(result)
            if saved["model_sha256"] != selected[identity]["sha256"]:
                raise ValueError("Cost record belongs to a different model")
            continue
        subprocess.run([sys.executable, str(REPO / "scripts/comprehensive_batch/benchmark.py"),
                        "--output", str(output), "--snapshot", identity], check=True)
    write(output / "benchmark_complete.json", {"registered_selection": "all default-LR weight-one arms at step15600 plus four HGB",
                                               "available_models": list(selected), "no_best_model_selection": True})


def package(output):
    # Resumption states stay in the laboratory; inference snapshots are all returned.
    files = [path for path in sorted(output.rglob("*")) if path.is_file()
             and path.name not in {"recovery.pt", "recovery_previous.pt", "output_receipt.json"}
             and not path.name.endswith((".tmp", ".part"))]
    write(output / "output_receipt.json", {path.relative_to(output).as_posix(): {
        "sha256": data.file_digest(path), "bytes": path.stat().st_size} for path in files})
    files.append(output / "output_receipt.json")
    archive = Path(str(output) + ".zip")
    temporary = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as stream:
        for path in files:
            stream.write(path, arcname=(Path(output.name) / path.relative_to(output)).as_posix())
    os.replace(temporary, archive)
    write(Path(str(archive) + ".sha256.json"), {"file": archive.name, "bytes": archive.stat().st_size,
                                               "sha256": data.file_digest(archive),
                                               "scope": "this batch only; recovery states retained at laboratory"})
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--coverage-results", type=Path)
    args = parser.parse_args()
    registration, plan, rows, processor = prepare()
    output = args.output.resolve()
    source = {"registration_sha256_lf": text_digest(REPO / DIRECTORY / "registration.json"),
              "registration": registration, "plan_sha256": data.json_digest(plan),
              "coverage_results": str(args.coverage_results.resolve()) if args.coverage_results else None}
    if output.exists() and any(output.iterdir()):
        if not args.resume or read(output / "source_receipt.json") != source:
            raise ValueError("Use an empty output, or --resume with the identical registered batch")
    output.mkdir(parents=True, exist_ok=True)
    write(output / "source_receipt.json", source)
    write(output / "plan.json", plan)
    if args.inputs_only:
        write(output / "inputs_check.json", {"F_rows": len(rows), "real_fits": 0, "real_scores": 0,
                                             "planned_tabm_trajectories": len(plan["tabm_jobs"]),
                                             "planned_HGB_jobs": len(plan["hgb_jobs"])})
        print("F-only inputs and fixed matrix verified; no models constructed.", flush=True)
        return
    write(output / "environment.json", environment())
    try:
        if (output / "models_frozen.json").exists():
            models = read(output / "models_frozen.json")["models"]
        else:
            models = train_phase(output, rows, processor, plan, resume=args.resume,
                                 coverage_results=args.coverage_results)
        result = evaluate_phase(output, plan, models)
        benchmark_phase(output, rows, models)
        jobs = read(output / "models_frozen.json")["jobs"]
        failures = [identity for identity, value in jobs.items() if value["status"] != "complete"]
        write(output / "summary.json", {"status": "complete" if not failures else "complete_with_failed_jobs",
                                        "failed_jobs": failures, "completed_snapshots": len(models),
                                        "automatic_model_selection": False, "threshold_searches": 0,
                                        "historical_development_only": True, "evaluation": result})
        archive = package(output)
        print(f"Return only this batch archive and its SHA receipt: {archive}", flush=True)
    except BaseException as error:
        write(output / "interruption.json", _error(error, "interrupted"))
        raise


if __name__ == "__main__":
    main()
