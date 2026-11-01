"""Bound, sequential CPU measurements of trusted development artifacts."""
from __future__ import annotations

from pathlib import Path
import random

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, write_json


def run_identity(run_dir: Path, data_dir: Path) -> dict:
    from .runner import verify_run
    meta = verify_run(run_dir)
    if meta["dataset_manifest_sha256"] != sha256_file(data_dir / "manifest.json"):
        raise ValueError("Dataset differs from training")
    if meta.get("phase") != "development":
        raise ValueError("Benchmark requires a development run")
    return {"run_id": run_dir.name, "run_manifest_sha256": sha256_file(run_dir / "run.json"),
            "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
            "model_sha256": meta["files"]["model.joblib"],
            "source_sha256": read_json(run_dir / "code.json")["source_sha256"],
            "split": "validation"}


def benchmark_run(run_dir: Path, data_dir: Path, output: Path, threads: int = 1,
                  measurement: dict | None = None) -> dict:
    from .benchmark import benchmark_predict
    from .data import load_partition
    from .models import load_bundle
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    identity = run_identity(run_dir, data_dir)
    if output.exists():
        raise FileExistsError("Benchmark output exists")
    bundle = load_bundle(run_dir / "model.joblib", device="cpu")
    bundle.num_threads = threads
    if bundle.model_name == "rf":
        bundle.estimator.n_jobs = threads
    if bundle.model_name == "mlp":
        import torch
        torch.set_num_threads(threads)
    result = benchmark_predict(bundle.predict_scores, load_partition(data_dir, "validation"), threads=threads)
    result.update(identity)
    result.update({"environment": environment_snapshot(),
                   "benchmark_code_sha256": code_fingerprint()["source_sha256"],
                   "model_file_bytes": (run_dir / "model.joblib").stat().st_size})
    if measurement:
        result.update(measurement)
    write_json(output, result)
    return result


def benchmark_suite(run_root: Path, data_dir: Path, output_dir: Path, threads: int = 1,
                    order_seed: int = 20261101, resume: bool = False) -> list[Path]:
    """Measure each expected run once in a saved shuffled order, after training.

    Resume checks the plan, artifacts and output checksums. A resume is a new
    timing session (timestamps are retained), not a claim of uninterrupted load.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    spec = read_json(run_root / "suite.json")["config"]
    runs = [run_root / f"{i:02d}-{model['model']}-s{seed}"
            for i, model in enumerate(spec["models"]) for seed in spec["seeds"]]
    if len({p.name for p in runs}) != len(runs):
        raise ValueError("Duplicate expected runs")
    random.Random(order_seed).shuffle(runs)
    identities = [run_identity(p, data_dir) for p in runs]
    environment = environment_snapshot()
    environment.pop("recorded_at", None)
    plan = {"kind": "benchmark_suite_plan", "schema_version": 1, "order_seed": order_seed,
            "threads": threads, "settings": {"warmup": 5, "repeats": 30, "single_rows": 100, "batch_size": 256},
            "benchmark_code_sha256": code_fingerprint()["source_sha256"],
            "environment_sha256": canonical_hash(environment), "runs": identities}
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        if not resume:
            raise FileExistsError("Benchmark suite exists; use --resume for the same plan")
        if read_json(plan_path) != plan:
            raise ValueError("Cannot resume: benchmark inputs, settings, source or environment changed")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("Benchmark directory must be empty for a new plan")
        write_json(plan_path, plan)
    plan_hash = sha256_file(plan_path)
    outputs = []
    for index, (run, identity) in enumerate(zip(runs, identities)):
        output = output_dir / f"{run.name}.json"
        checksum = output.with_suffix(".sha256")
        measurement = {"measurement_order_index": index, "plan_sha256": plan_hash}
        if output.exists():
            if not resume or not checksum.exists() or checksum.read_text().strip() != sha256_file(output):
                raise ValueError(f"Unverified existing benchmark: {output.name}")
            result = read_json(output)
            required = {**identity, **measurement, "threads": threads,
                        "benchmark_code_sha256": plan["benchmark_code_sha256"]}
            if any(result.get(key) != value for key, value in required.items()):
                raise ValueError(f"Benchmark binding changed: {output.name}")
            print(f"Verified benchmark {index + 1}/{len(runs)}: {run.name}", flush=True)
        else:
            print(f"CPU benchmark {index + 1}/{len(runs)}: {run.name}", flush=True)
            benchmark_run(run, data_dir, output, threads, measurement)
            checksum.write_text(sha256_file(output) + "\n", encoding="ascii")
        outputs.append(output)
    return outputs
