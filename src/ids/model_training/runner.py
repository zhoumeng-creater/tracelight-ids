"""Seven bounded trajectories/refits, global freeze, fixed-C scoring and audit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, run_lock, stage
from . import VERSION
from . import data, evaluation
from .protocol import REPO, code_snapshot, specifications, tasks, verify_live_code, verify_runtime


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Artifact path escapes run directory: {relative}")
    return path


def files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file())


def verify_receipt(root, receipt, binding=None):
    item = read_json(receipt)
    if binding is not None and item["binding"] != binding:
        raise ValueError(f"Stage binding changed: {receipt.name}")
    for rel, digest in item["outputs"].items():
        path = contained(root, rel)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Completed stage artifact changed: {rel}")
    return item


def checked_stage(root, name, binding, execute):
    receipt = root / "receipts" / f"{name}.json"
    if receipt.exists():
        verify_receipt(root, receipt, binding)
    stage(root, name, binding, execute)


def snapshot_sources(root, protocol):
    target = root / "source_snapshot"
    for rel in protocol["code"]:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / rel).read_bytes().replace(b"\r\n", b"\n"))
    return files(target)


def initialize(root, r26, recovered, raw_dir, threads, device):
    if threads < 2:
        raise ValueError("At least two CPU threads are required for the HGB/GPU worker split")
    if root.is_relative_to(r26) or r26.is_relative_to(root):
        raise ValueError("New output and the frozen R26 run must be separate directories")
    registration, design = specifications()
    roles, binding, references = data.load_inputs(r26, recovered, raw_dir, root, registration, design)
    protocol = {"version": VERSION, "registration": registration, "design": design,
                "tasks": tasks(design), "inputs": binding, "code": code_snapshot(),
                "threads": threads, "device": device, "score_device": "cpu", "score_threads": 1,
                "new_blind_confirmation": False}
    destination = root / "protocol.json"
    if destination.exists():
        if read_json(destination) != protocol:
            raise ValueError("Run input, code, design or execution identity changed; preserve this output")
    else:
        if any(p.name != "run.lock" and p.name != "recovered" for p in root.iterdir()):
            raise ValueError("Unregistered output directory is not empty")
        write_json(destination, protocol)
    key = canonical_hash(protocol)
    checked_stage(root, "inputs", key, lambda: data.persist_inputs(root, roles, binding, references))
    checked_stage(root, "sources", key, lambda: snapshot_sources(root, protocol))
    return protocol


def verify_inputs(root, protocol):
    verify_receipt(root, root / "receipts/inputs.json", canonical_hash(protocol))


def train_task(root, task_id, backend=None):
    """Worker API: accepts only F, V and P. C/ED are not loaded here."""
    if backend is None:
        from . import models as backend
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_inputs(root, protocol)
    task = next(t for t in protocol["tasks"] if t["id"] == task_id)
    base = root / "models" / task_id
    hgb_threads = max(1, protocol["threads"] // 2)
    threads = hgb_threads if task["arm"] == "hgb" else protocol["threads"] - hgb_threads
    arguments = {"arm": task["arm"], "seed": task["seed"], "threads": threads,
                 "device": "cpu" if task["arm"] == "hgb" else protocol["device"],
                 "design": protocol["design"]}
    key = {"protocol": canonical_hash(protocol), "task": task}

    def trajectory():
        f, v = (pd.read_parquet(root / f"inputs/{name}.parquet") for name in ("F", "V"))
        result = backend.fit_trajectory(f, v, base / "trajectory", **arguments)
        if result["status"] != "complete":
            raise RuntimeError(f"Failed trajectory: {task_id}; inspect its failure.json")
        verify_live_code(protocol)
        return files(base / "trajectory")

    checked_stage(root, task_id + "-trajectory", key, trajectory)
    selected = read_json(base / "trajectory/summary.json")["selected_stage"]

    def final():
        p = pd.read_parquet(root / "inputs/P.parquet")
        result = backend.fit_final(p, base / "final", stage=selected, **arguments)
        if result["status"] != "complete":
            raise RuntimeError(f"Failed final refit: {task_id}; inspect its failure.json")
        verify_live_code(protocol)
        return files(base / "final")

    checked_stage(root, task_id + "-final", {**key, "selected_stage": selected}, final)


def launch(root, mode, task_id):
    log = root / "logs" / f"{mode}-{task_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    count = "1" if mode == "cost-worker" else str(max(1, read_json(root / "protocol.json")["threads"] // 2))
    env.update({name: count for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    with log.open("a", encoding="utf-8") as output:
        subprocess.run([sys.executable, "-m", "ids.model_training", "--" + mode, task_id,
                        "--output-dir", str(root)], cwd=REPO, env=env,
                       stdout=output, stderr=subprocess.STDOUT, check=True)


def train_all(root, protocol):
    def queue(items):
        for task in items:
            launch(root, "worker", task["id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, [t for t in protocol["tasks"] if (t["arm"] == "hgb") == hgb])
                   for hgb in (True, False)]
        for future in futures:
            future.result()


def freeze(root, protocol):
    artifacts = {}
    for task in protocol["tasks"]:
        task_id = task["id"]
        for phase in ("trajectory", "final"):
            verify_receipt(root, root / f"receipts/{task_id}-{phase}.json")
        summary = read_json(root / f"models/{task_id}/final/summary.json")
        if summary["status"] != "complete":
            raise ValueError(f"Cannot freeze incomplete model: {task_id}")
        artifact = root / f"models/{task_id}/final" / Path(summary["artifact_path"]).name
        artifacts[task_id] = {"path": artifact.relative_to(root).as_posix(),
                              "sha256": sha256_file(artifact), "selected_stage": summary["selected_stage"]}
    write_json(root / "freeze.json", {"created": utc_now(), "protocol": canonical_hash(protocol),
        "all_final_models_complete": True, "actual_fit_calls": 14, "artifacts": artifacts,
        "calibration_and_evaluation_scoring_started": False})
    return [root / "freeze.json"]


def verify_freeze(root, protocol):
    verify_receipt(root, root / "receipts/freeze.json", canonical_hash(protocol))
    frozen = read_json(root / "freeze.json")
    if set(frozen["artifacts"]) != {t["id"] for t in protocol["tasks"]}:
        raise ValueError("All seven final models must be frozen before calibration/evaluation")
    for spec in frozen["artifacts"].values():
        if sha256_file(contained(root, spec["path"])) != spec["sha256"]:
            raise ValueError("A frozen final artifact changed")
    return frozen


def score_task(root, protocol, task, backend=None):
    if backend is None:
        from . import models as backend
    frozen = verify_freeze(root, protocol)
    task_id = task["id"]
    predictor = backend.load_predictor(contained(root, frozen["artifacts"][task_id]["path"]), device="cpu", threads=1)
    folder = root / "evidence" / task_id
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("C", "ED"):
        frame = pd.read_parquet(root / f"inputs/{name}.parquet")
        score = np.asarray(predictor.score(data.blind(frame)), dtype=np.float64)
        if score.shape != (len(frame),) or not np.isfinite(score).all():
            raise ValueError(f"Invalid {name} scores from {task_id}")
        saved = frame[data.IDENTITY].copy()
        saved["score"] = score
        saved.to_parquet(folder / f"{name}-scores.parquet", index=False)
        if name == "C":
            write_json(folder / "calibration.json", evaluation.calibrate(score))
    verify_live_code(protocol)
    return files(folder)


def aligned_scores(root, task_id, name, frame):
    saved = pd.read_parquet(root / f"evidence/{task_id}/{name}-scores.parquet")
    pd.testing.assert_frame_equal(saved[data.IDENTITY], frame[data.IDENTITY], check_dtype=False)
    return saved.score.to_numpy(dtype=float)


def report(root, protocol):
    frame = pd.read_parquet(root / "inputs/ED.parquet")
    references = read_json(root / "inputs/references.json")
    decisions, units, days = {}, [], []
    folder = root / "reports"
    folder.mkdir(parents=True, exist_ok=True)

    def add(model_id, result):
        unit, day, decision = result
        units.extend(unit)
        days.extend(day)
        decisions[model_id] = decision
        decision.to_parquet(folder / f"{model_id}-decisions.parquet", index=False)

    for task in protocol["tasks"]:
        name = task["id"]
        add(name, evaluation.evaluate(frame, aligned_scores(root, name, "ED", frame),
            read_json(root / f"evidence/{name}/calibration.json"), model_id=name))
    old_hgb = frame["source-raw8-hgb"].to_numpy() > references["legacy_hgb"]["threshold"]
    rule = evaluation.legacy_rule(frame)
    policies = {"legacy_hgb": old_hgb, "legacy_rule": rule, "legacy_hgb_or_rule": old_hgb | rule}
    for column, expected in (("reference-hgb-original-C", old_hgb), ("reference-udp443-30s", rule),
                             ("reference-hgb-original-C-or-udp443-30s", old_hgb | rule)):
        if not np.array_equal(frame[column].to_numpy(), expected):
            raise ValueError(f"Cached legacy policy disagrees with frozen rule: {column}")
    for name, alerts in policies.items():
        add(name, evaluation.evaluate_decisions(frame, alerts, model_id=name))
    for group in ("hgb", "mlp17", "mlp29", "mlp43"):
        name = "r26_target_" + group
        add(name, evaluation.evaluate(frame, frame[f"target-ports_zero8-{group}"].to_numpy(),
                                      references[group], model_id=name))
    write_json(folder / "units.json", units)
    write_json(folder / "utc_days.json", days)
    for name, rows in (("units", units), ("utc_days", days)):
        pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                      for row in rows]).to_csv(folder / f"{name}.csv", index=False)
    write_json(folder / "comparison.json", evaluation.compare(decisions, protocol["design"]))
    return files(folder)


def cost_worker(root, task_id, backend=None):
    """One isolated process per model: CPU first, then separate TabM GPU throughput."""
    from threadpoolctl import threadpool_limits
    from ids.route_batch.benchmark import benchmark_predictor
    if backend is None:
        from . import models as backend
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_inputs(root, protocol)
    frozen = verify_freeze(root, protocol)
    artifact = contained(root, frozen["artifacts"][task_id]["path"])
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        predictor = backend.load_predictor(artifact, device="cpu", threads=1)
        cold_load_seconds = time.perf_counter() - started
        frame = pd.read_parquet(root / "inputs/ED.parquet")
        raw = data.blind(frame)
        scores = predictor.score(raw)
        expected = aligned_scores(root, task_id, "ED", frame)
        if not np.array_equal(scores, expected):
            raise ValueError(f"Cold CPU replay does not exactly reproduce frozen scores: {task_id}")
        cost = benchmark_predictor(predictor.score, raw, **protocol["registration"]["benchmark"])
        transform = predictor.preprocessor.hgb_matrix if task_id == "hgb" else predictor.preprocessor.transform
        preprocessing_cost = benchmark_predictor(transform, raw, **protocol["registration"]["benchmark"])
        # Batch-dependent arithmetic is also checked for timed single queries.
        indices = [(i * len(raw)) // min(64, len(raw)) for i in range(min(64, len(raw)))]
        singles = np.array([predictor.score(raw.iloc[[i]])[0] for i in indices])
        boundary = read_json(root / f"evidence/{task_id}/calibration.json")["threshold"]
        changed = [] if boundary is None else [int(i) for i, a, b in zip(indices, singles, expected[indices])
                                              if (a > boundary) != (b > boundary)]
        result = {"model_id": task_id, "device": "cpu", "threads": 1,
                  "cold_artifact_load_seconds": cold_load_seconds,
                  "cold_scope": "artifact/preprocessor load including lazy framework imports; Python startup excluded; OS file cache uncontrolled",
                  "model_artifact_bytes": artifact.stat().st_size, "cold_scores_exact": True,
                  "single_query_max_abs_score_delta": float(np.max(np.abs(singles - expected[indices]))),
                  "single_query_changed_decision_indices": changed,
                  "preprocessing_included": True, "timing": cost,
                  "preprocessing_only_timing": preprocessing_cost,
                  "tail_interpretation": "retain raw samples; three batch repeats are descriptive, not a stable tail estimate"}
        if task_id != "hgb":
            import torch
            gpu_device = torch.device(protocol["device"])
            spec = protocol["registration"]["gpu_benchmark"]
            count = min(spec["batch_size"], len(raw))
            indices = [(i * len(raw)) // count for i in range(count)]
            batch = raw.iloc[indices].copy()
            torch.cuda.synchronize(gpu_device)
            torch.cuda.reset_peak_memory_stats(gpu_device)
            then = time.perf_counter()
            gpu = backend.load_predictor(artifact, device=protocol["device"], threads=1)
            torch.cuda.synchronize(gpu_device)
            load_seconds = time.perf_counter() - then
            for _ in range(spec["warmup"]):
                gpu.score(batch)
            torch.cuda.synchronize(gpu_device)
            timings = []
            for _ in range(spec["repeats"]):
                then = time.perf_counter()
                gpu_scores = gpu.score(batch)
                torch.cuda.synchronize(gpu_device)
                timings.append(time.perf_counter() - then)
            changed = [] if boundary is None else [int(i) for i, a, b in zip(indices, gpu_scores, expected[indices])
                                                  if (a > boundary) != (b > boundary)]
            result["gpu_throughput"] = {
                "device": protocol["device"], "batch_rows": count, "repeats": len(timings),
                "row_indices": indices, "seconds_per_repeat": timings,
                "rows_per_second": count * len(timings) / sum(timings),
                "artifact_load_seconds": load_seconds, "preprocessing_and_host_device_copies_included": True,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu_device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(gpu_device),
                "max_abs_delta_from_registered_cpu": float(np.max(np.abs(gpu_scores - expected[indices]))),
                "changed_decision_indices_at_cpu_threshold": changed,
                "scope": "descriptive batch throughput; qualification uses frozen CPU scores only"}
        else:
            result["gpu_throughput"] = {"applicable": False, "reason": "registered HGB has no GPU implementation"}
    write_json(root / f"cost/{task_id}.json", result)


def manifest(root):
    selected = [p for p in files(root) if p.name != "run.lock" and p.name != "archive_manifest.json"
                and not p.name.endswith((".zip", ".zip.sha256", ".tmp"))]
    write_json(root / "archive_manifest.json", {"version": VERSION, "created": utc_now(),
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in selected}})


def verify(root):
    record = read_json(root / "archive_manifest.json")
    for rel, digest in record["files"].items():
        path = contained(root, rel)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Run artifact changed: {rel}")
    for receipt in (root / "receipts").glob("*.json"):
        verify_receipt(root, receipt)
    protocol = read_json(root / "protocol.json")
    status = read_json(root / "status.json")
    if status["status"] == "complete":
        verify_freeze(root, protocol)
        expected = {"inputs", "sources", "freeze", "report"}
        expected |= {t["id"] + "-" + phase for t in protocol["tasks"]
                     for phase in ("trajectory", "final", "scores", "cost")}
        if not expected <= {p.stem for p in (root / "receipts").glob("*.json")}:
            raise ValueError("A complete run is missing required stage receipts")
    return {"status": status["status"], "verified_files": len(record["files"]), "model_forwards": 0,
            "verification": "offline hashes and required stage completeness"}


def pack(root):
    validation = verify(root)
    return write_archive(root, validation)


def write_archive(root, validation, suffix="results"):
    path = root / (VERSION + "-" + suffix + ".zip")
    record = read_json(root / "archive_manifest.json")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for rel in [*record["files"], "archive_manifest.json"]:
            archive.write(contained(root, rel), rel)
    path.with_suffix(".zip.sha256").write_text(sha256_file(path) + "  " + path.name + "\n", encoding="utf-8")
    return {**validation, "archive": str(path), "sha256": sha256_file(path)}


def archive_failure(root, error):
    """Preserve broken evidence without pretending its completed receipts pass."""
    broken = []
    for receipt in (root / "receipts").glob("*.json"):
        try:
            verify_receipt(root, receipt)
        except Exception as receipt_error:
            broken.append({"receipt": receipt.relative_to(root).as_posix(),
                           "error_type": type(receipt_error).__name__, "error": str(receipt_error)})
    report = {"status": "failed_evidence_only", "original_error_type": type(error).__name__,
              "original_error": str(error), "broken_receipts": broken,
              "completed_experiment_verified": False, "model_forwards": 0}
    write_json(root / "failure_integrity.json", report)
    manifest(root)
    return write_archive(root, report, suffix="failed-evidence")


def run(root, *, r26, recovered=None, raw_dir=None, threads=8, device="cuda:0", inputs_only=False):
    root, r26 = Path(root).resolve(), Path(r26).resolve()
    with run_lock(root):
        if (root / "archive_manifest.json").exists():
            verify(root)
        protocol = initialize(root, r26, recovered, raw_dir, threads, device)
        if inputs_only:
            if not (root / "status.json").exists():
                write_json(root / "status.json", {"status": "inputs_ready", "fits": 0, "model_forwards": 0})
            manifest(root)
            return verify(root)
        if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "complete":
            return pack(root)
        try:
            runtime = verify_runtime(protocol["registration"], device)
            if (root / "runtime.json").exists() and read_json(root / "runtime.json") != runtime:
                raise ValueError("Laboratory runtime changed across continuation")
            write_json(root / "runtime.json", runtime)
            own_windows_workers()
            with ResourceLog(root) as resource:
                resource.activity = "F/V selection and P final refit; HGB CPU plus one GPU worker"
                train_all(root, protocol)
                verify_live_code(protocol)
                key = canonical_hash(protocol)
                checked_stage(root, "freeze", key, lambda: freeze(root, protocol))
                resource.activity = "all seven frozen; CPU fixed-C and E/D scoring"
                for task in protocol["tasks"]:
                    checked_stage(root, task["id"] + "-scores", key,
                                  lambda task=task: score_task(root, protocol, task))
                checked_stage(root, "report", key, lambda: report(root, protocol))
                resource.activity = "serial isolated CPU costs after all training"
                for task in protocol["tasks"]:
                    def measure(task=task):
                        launch(root, "cost-worker", task["id"])
                        return [root / f"cost/{task['id']}.json"]
                    checked_stage(root, task["id"] + "-cost", key, measure)
            verify_live_code(protocol)
            write_json(root / "status.json", {"status": "complete", "fits": 14, "finished": utc_now(),
                "comparison": "reports/comparison.json", "new_blind_confirmation": False})
        except BaseException as error:
            write_json(root / "status.json", {"status": "failed", "finished": utc_now(),
                       "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "automatic_fit_retry": False})
            try:
                print(archive_failure(root, error), flush=True)
            except Exception as archive_error:
                print(f"Failure archive also failed: {archive_error}; original exception follows", file=sys.stderr)
            raise
        manifest(root)
        return pack(root)
