"""Development and final evaluation are deliberately separate operations."""
from __future__ import annotations

import copy
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import (canonical_hash, code_fingerprint, environment_snapshot, read_json,
                     sha256_file, utc_now, write_json)
from .data import FEATURES, load_partition
from .evaluation import calibrate_threshold, evaluate_binary
from .models import load_bundle, save_bundle, train_model


def validate_suite(config: dict) -> None:
    if not isinstance(config, dict) or not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("Suite must define a nonempty models list")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", config.get("suite_name", "")):
        raise ValueError("suite_name must be a short ASCII filename")
    seeds = config.get("seeds", [])
    if not seeds or any(type(s) is not int for s in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    if not 0 <= config.get("max_fpr", .01) < 1:
        raise ValueError("max_fpr must be in [0,1)")
    if type(config.get("threads", 1)) is not int or config.get("threads", 1) < 1:
        raise ValueError("threads must be a positive integer")


def _predictions(frame, scores, threshold, split, run_id, data_kind):
    out = pd.DataFrame({"sample_id": frame["sample_id"].to_numpy(),
                        "score": np.asarray(scores, dtype=float),
                        "threshold": float(threshold),
                        "predicted_label": (np.asarray(scores, dtype=np.float64) >= threshold).astype(int),
                        "true_label": frame["label"].to_numpy(),
                        "attack_category": frame["attack_cat"].to_numpy(),
                        "split": split, "run_id": run_id, "data_kind": data_kind})
    return out


def _examples(frame, predictions, limit=200):
    # First N records by stable ID; no truth-based cherry-picking.
    indexes = np.argsort(frame["sample_id"].astype(str).to_numpy())[:limit]
    rows = []
    for i in indexes:
        event = predictions.iloc[int(i)].to_dict()
        event["features"] = {
            k: (None if pd.isna(frame.iloc[int(i)][k]) else frame.iloc[int(i)][k]) for k in FEATURES
        }
        rows.append(event)
    return {"selection": "first_n_by_stable_sample_id", "limit": limit,
            "total_partition_rows": len(frame), "events": rows,
            "notice": "Selected validation records, not a representative population estimate."}


def verify_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    meta = read_json(root / "run.json")
    if meta.get("status") != "complete":
        raise ValueError("Run is not complete")
    for name, expected in meta["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid artifact filename")
        if not (root / name).is_file() or sha256_file(root / name) != expected:
            raise ValueError(f"Run artifact changed: {name}")
    return meta


def run_development(data_dir: str | Path, model_config: dict, output_dir: str | Path,
                    seed: int = 17, device: str = "cpu", max_fpr: float = .01,
                    threads: int = 1, purpose: str = "development") -> Path:
    data_dir, root = Path(data_dir), Path(output_dir)
    if threads < 1:
        raise ValueError("threads must be positive")
    model_config = copy.deepcopy(model_config)
    model_config.setdefault("hyperparameters", {})["num_threads"] = threads
    if model_config["model"] == "rf":
        model_config["hyperparameters"]["n_jobs"] = threads
    if root.exists():
        raise FileExistsError(f"Run already exists: {root.name}; choose a new run or use suite --resume")
    data_manifest = read_json(data_dir / "manifest.json")
    # Neither calibration nor test is loaded in this function.
    train = load_partition(data_dir, "train")
    validation = load_partition(data_dir, "validation")
    root.mkdir(parents=True)
    started = utc_now()
    config = {"model_config": model_config, "seed": seed, "requested_device": device,
              "max_fpr": max_fpr, "threads": threads, "purpose": purpose}
    write_json(root / "config.json", config)
    write_json(root / "environment.json", environment_snapshot())
    write_json(root / "code.json", code_fingerprint())
    write_json(root / "dataset.json", data_manifest)
    try:
        with threadpool_limits(limits=threads):
            if model_config["model"] == "mlp":
                import torch
                torch.set_num_threads(threads)
            begin = time.perf_counter()
            bundle = train_model(train, validation, model_config, seed=seed, device=device)
            training_seconds = time.perf_counter() - begin
            scores = bundle.predict_scores(validation)
        operating = calibrate_threshold(validation["label"], scores, max_fpr=max_fpr)
        operating["source_split"] = "validation"
        operating["usage"] = "development_only_not_final_threshold"
        metrics = evaluate_binary(validation["label"], scores, operating["threshold"],
                                  attack_categories=validation["attack_cat"])
        metrics.update({"split": "validation", "phase": "development",
                        "data_kind": data_manifest["data_kind"],
                        "threshold_source": "same_validation_partition_exploratory",
                        "training_seconds": training_seconds,
                        "n_raw_features": len(bundle.selected_features),
                        "encoded_dim": bundle.encoded_dim})
        pred = _predictions(validation, scores, operating["threshold"], "validation", root.name,
                            data_manifest["data_kind"])
        pred.to_parquet(root / "validation_predictions.parquet", index=False)
        save_bundle(bundle, root / "model.joblib")
        write_json(root / "validation_metrics.json", metrics)
        write_json(root / "validation_threshold.json", operating)
        write_json(root / "selected_features.json", list(bundle.selected_features))
        write_json(root / "training_history.json", bundle.history)
        write_json(root / "examples.json", _examples(validation, pred))
        meta = {"schema_version": 1, "run_id": root.name, "status": "complete", "phase": "development",
                "started_at": started, "finished_at": utc_now(), "config_sha256": canonical_hash(config),
                "dataset_manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "data_kind": data_manifest["data_kind"], "model_name": bundle.model_name,
                "files": {p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}}
        write_json(root / "run.json", meta)
    except Exception as exc:
        write_json(root / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                                          "message": str(exc), "time": utc_now()})
        raise
    return root


def run_suite(data_dir, suite: dict, output_root, device="cpu", resume=False) -> list[Path]:
    validate_suite(suite)
    root = Path(output_root) / suite["suite_name"]
    root.mkdir(parents=True, exist_ok=True)
    suite_path = root / "suite.json"
    spec = {"config": suite, "dataset_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
            "requested_device": device}
    if suite_path.exists():
        if not resume:
            raise FileExistsError("Suite exists. Use --resume only for the identical configuration.")
        if read_json(suite_path) != spec:
            raise ValueError("Cannot resume: suite, dataset, or device changed")
    else:
        write_json(suite_path, spec)
    runs = []
    for i, model in enumerate(suite["models"]):
        for seed in suite["seeds"]:
            run_dir = root / f"{i:02d}-{model['model']}-s{seed}"
            if run_dir.exists() and resume and not (run_dir / "run.json").exists():
                if (run_dir / "code.json").exists() and read_json(run_dir / "code.json")["source_sha256"] != code_fingerprint()["source_sha256"]:
                    raise ValueError("Cannot retry incomplete run after source changes; use a new suite name")
                archive = root / "failed_attempts" / f"{run_dir.name}-{time.time_ns()}"
                # Both the existing source and destination must stay inside this suite.
                if run_dir.is_symlink() or not run_dir.resolve().is_relative_to(root.resolve()) or not archive.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Invalid incomplete run archive path")
                archive.parent.mkdir(parents=True, exist_ok=True)
                run_dir.rename(archive)
                print(f"Preserved incomplete attempt: {archive.name}; restarting this model from its seed", flush=True)
            if run_dir.exists() and resume:
                meta = verify_run(run_dir)
                # Verify only development inputs; never open calibration/test on resume.
                current = read_json(Path(data_dir) / "manifest.json")
                if (Path(data_dir) / "manifest.sha256").read_text().strip() != sha256_file(Path(data_dir) / "manifest.json"):
                    raise ValueError("Manifest checksum mismatch")
                for split in ("train", "validation"):
                    record = current["partitions"][split]
                    if record["file"] != f"{split}.parquet" or sha256_file(Path(data_dir) / record["file"]) != record["sha256"]:
                        raise ValueError(f"Partition checksum mismatch: {split}")
                if sha256_file(Path(data_dir) / "audit.json") != current["audit"]["sha256"]:
                    raise ValueError("Audit checksum mismatch")
                if meta["dataset_manifest_sha256"] != spec["dataset_manifest_sha256"]:
                    raise ValueError("Cannot resume mismatched dataset")
                if read_json(run_dir / "code.json")["source_sha256"] != code_fingerprint()["source_sha256"]:
                    raise ValueError("Cannot resume after source code changes; use a new suite name")
                print(f"Verified completed run: {run_dir.name}", flush=True)
            else:
                print(f"Training {run_dir.name} ({device}); validation only", flush=True)
                run_development(data_dir, model, run_dir, seed, device, suite.get("max_fpr", .01),
                                suite.get("threads", 1), suite.get("purpose", "development"))
            runs.append(run_dir)
    write_json(root / "summary.json", [{"run_id": p.name,
                                        "metrics": read_json(p / "validation_metrics.json")} for p in runs])
    return runs


def freeze_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    meta = verify_run(root)
    if (root / "evaluation_plan.json").exists():
        raise FileExistsError("Evaluation plan already frozen")
    plan = {"schema_version": 1, "frozen_at": utc_now(), "run_manifest_sha256": sha256_file(root / "run.json"),
            "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
            "model_sha256": meta["files"]["model.joblib"],
            "max_fpr": read_json(root / "config.json")["max_fpr"],
            "protocol": "fit_train_select_validation_threshold_calibration_evaluate_test",
            "source_sha256": read_json(root / "code.json")["source_sha256"]}
    write_json(root / "evaluation_plan.json", plan)
    (root / "evaluation_plan.sha256").write_text(sha256_file(root / "evaluation_plan.json") + "\n", encoding="ascii")
    return plan


def verify_evaluation(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    verify_run(root)
    completion = read_json(root / "evaluation" / "completion.json")
    if sha256_file(root / "evaluation_plan.json") != completion["plan_sha256"]:
        raise ValueError("Evaluation plan changed")
    plan = read_json(root / "evaluation_plan.json")
    if (root / "evaluation_plan.sha256").read_text().strip() != sha256_file(root / "evaluation_plan.json"):
        raise ValueError("Frozen evaluation plan changed")
    if sha256_file(root / "run.json") != plan["run_manifest_sha256"]:
        raise ValueError("Frozen run manifest changed")
    for name, expected in completion["files"].items():
        if Path(name).name != name or sha256_file(root / "evaluation" / name) != expected:
            raise ValueError(f"Evaluation artifact changed: {name}")
    return completion


def evaluate_frozen(run_dir: str | Path, data_dir: str | Path) -> dict:
    root, data_dir = Path(run_dir), Path(data_dir)
    if not (root / "evaluation_plan.json").is_file():
        raise ValueError("Final evaluation requires `freeze` first; development must not read test")
    plan = read_json(root / "evaluation_plan.json")
    if (root / "evaluation_plan.sha256").read_text().strip() != sha256_file(root / "evaluation_plan.json"):
        raise ValueError("Frozen evaluation plan changed")
    if plan["max_fpr"] != read_json(root / "config.json")["max_fpr"]:
        raise ValueError("Frozen FPR budget differs from training plan")
    verify_run(root)
    if sha256_file(root / "run.json") != plan["run_manifest_sha256"]:
        raise ValueError("Run changed after freeze")
    if sha256_file(data_dir / "manifest.json") != plan["dataset_manifest_sha256"]:
        raise ValueError("Prepared dataset changed after freeze")
    if sha256_file(root / "model.joblib") != plan["model_sha256"]:
        raise ValueError("Model changed after freeze")
    if code_fingerprint()["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Source changed since training; use a new development run")
    output = root / "evaluation"
    if output.exists():
        raise FileExistsError("Evaluation already started or completed; results are never overwritten")
    bundle = load_bundle(root / "model.joblib", device="cpu")
    if bundle.model_name == "mlp":
        import torch
        torch.set_num_threads(bundle.num_threads)
    calibration = load_partition(data_dir, "calibration")
    with threadpool_limits(limits=bundle.num_threads):
        calibration_scores = bundle.predict_scores(calibration)
    operating = calibrate_threshold(calibration["label"], calibration_scores, plan["max_fpr"])
    operating["source_split"] = "calibration"
    output.mkdir()
    write_json(output / "threshold.json", operating)
    # Test is loaded only after an immutable model and calibration-only threshold exist.
    test = load_partition(data_dir, "test")
    with threadpool_limits(limits=bundle.num_threads):
        scores = bundle.predict_scores(test)
    metrics = evaluate_binary(test["label"], scores, operating["threshold"],
                              attack_categories=test["attack_cat"])
    metrics.update({"phase": "frozen_evaluation", "split": "test", "threshold_source": "calibration",
                    "target_fpr": plan["max_fpr"], "data_kind": read_json(root / "dataset.json")["data_kind"]})
    _predictions(test, scores, operating["threshold"], "test", root.name, metrics["data_kind"]).to_parquet(
        output / "test_predictions.parquet", index=False)
    write_json(output / "test_metrics.json", metrics)
    write_json(output / "completion.json", {"finished_at": utc_now(), "plan_sha256": sha256_file(root / "evaluation_plan.json"),
                                           "files": {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}})
    return metrics
