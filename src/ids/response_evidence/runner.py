"""Round 23. One resumable Windows laboratory batch with CPU and CUDA queues."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys
import traceback

import psutil
import pyarrow as pa

from ids.common import canonical_hash, environment_snapshot, read_json, utc_now, write_json
from ids.contact_evidence import experiment as previous
from ids.contact_evidence.runner import gpu_probe
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.route_batch.protocol import REPO, source_snapshot
from . import controls, data, evaluation, experiment
from .cost import cost
from .history import enrich_prepared
from .history_benchmark import benchmark_history

RELEASE = "response-observability-lab-v1"


def synthetic(root):
    tests = sorted((REPO / "tests").glob("test_response_*.py"))
    if not tests:
        raise RuntimeError("Missing observability checks")
    log, xml = root / "evidence/check.log", root / "evidence/check.xml"
    with log.open("w", encoding="utf-8") as out:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(root / "pytest-temp"),
                "--junitxml=" + str(xml),
                *map(str, tests),
            ],
            cwd=REPO,
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return [log, xml]


def make_features(root, name, workers):
    output = previous.enriched(root, name)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = enrich_prepared(data.prepared(root, name), output, workers=workers)
    audit = root / "evidence/features" / (name + ".json")
    write_json(audit, info)
    return [output, audit]


def probe_history(root, name, workers):
    report = benchmark_history(data.prepared(root, name), workers=workers)
    output = root / "evidence/history-benchmark.json"
    write_json(output, {"capture": name, **report})
    return [output]


def prepare(root, source_dir, spec, workers, binding):
    pa.set_cpu_count(workers)
    name = spec["name"]
    stage(root, "parse-" + name, binding, partial(data.prepare_one, root, source_dir, spec))
    stage(root, "history-" + name, binding, partial(make_features, root, name, workers))


def fit_one(root, group, arm, threads, binding):
    pa.set_cpu_count(threads)
    base = read_json(root / "protocol.json")["reuse_r22_controls"]
    job = (
        partial(controls.reuse, root, base, group, arm)
        if base and arm in ("raw", "contact")
        else partial(experiment.fit, root, group, arm, threads)
    )
    stage(root, "fit-" + group + "-" + arm, binding, job)


def fit_gpu(root, threads, binding):
    for group in previous.NEURAL_GROUPS:
        for arm in experiment.ARMS:
            fit_one(root, group, arm, threads, binding)


def calibrate(root, reg, groups, threads, binding):
    pa.set_cpu_count(threads)
    for g in groups:
        stage(root, "normal-cut-" + g, binding, partial(experiment.calibrate, root, reg, g, threads))


def score(root, name, groups, threads, binding):
    pa.set_cpu_count(threads)
    for g in groups:
        stage(
            root, "seal-" + name + "-" + g, binding, partial(experiment.blind_score, root, name, g, threads)
        )


def run(root, source_dir, threads, prep_workers):
    reg = data.read_registration()
    env = environment_snapshot()
    spec = {
        "release": RELEASE,
        "data": reg,
        "arms": experiment.ARMS,
        "groups": previous.GROUPS,
        "comparisons": experiment.COMPARISONS,
        "bit_order": experiment.ARM_NAMES,
        "counter_masked": "quality model with flagged directional zero counters mapped to -1; same weights, own normal cut",
        "primary": ["hgb", "contact", "quality"],
        "budget": 0.01,
        "threads": threads,
        "prep_workers": prep_workers,
        "src_sha256": source_snapshot()["source_sha256"],
        "runtime": {k: env[k] for k in ("python", "packages")},
    }
    protocol = root / "protocol.json"
    spec["reuse_r22_controls"] = (
        read_json(protocol)["reuse_r22_controls"]
        if protocol.exists()
        else controls.previous_output(reg, spec["runtime"])
    )
    binding = canonical_hash(spec)
    if protocol.exists() and canonical_hash(read_json(protocol)) != binding:
        raise ValueError("Run identity changed; use the original checkout, parameters and environment")
    write_json(protocol, spec)
    write_json(root / "environment.json", env)
    (root / "evidence").mkdir(exist_ok=True)
    train = [s for s in reg["sources"] if s["role"] in {"P", "C"}]
    tests = [s for s in reg["sources"] if s["role"] in {"E", "R", "H", "X"}]
    if len(train) + len(tests) != len(reg["sources"]) or not tests:
        raise ValueError("Unexpected roles in registration")
    gpu_threads = max(1, min(3, threads // 6))
    fit_jobs = min(4, max(1, (threads - gpu_threads) // 3))
    history_threads = max(1, (threads - gpu_threads) // 2)
    fit_threads = max(1, (threads - history_threads - gpu_threads) // fit_jobs)
    bench_capture = "CTU-IoT-Malware-Capture-52-1"
    with ResourceLog(root) as monitor:
        monitor.activity = "preflight"
        stage(root, "checks", binding, partial(synthetic, root))
        stage(root, "cuda", binding, partial(gpu_probe, root))
        monitor.activity = "early_capture_preparation"
        with ProcessPoolExecutor(max_workers=prep_workers) as early:
            jobs = [
                early.submit(prepare, root, source_dir, s, max(1, threads // prep_workers), binding)
                for s in train
            ]
            for j in jobs:
                j.result()
        stage(root, "train-selection", binding, partial(experiment.training, root, reg))
        with (
            ProcessPoolExecutor(max_workers=fit_jobs) as cpu,
            ProcessPoolExecutor(max_workers=1) as gpu,
            ProcessPoolExecutor(max_workers=1) as prep,
        ):
            # Large captures run one at a time to bound resident feature/state arrays.
            monitor.activity = "large_log_history_concurrent_cpu_and_cuda_fit"
            prep_jobs = [prep.submit(prepare, root, source_dir, s, history_threads, binding) for s in tests]
            neural = gpu.submit(fit_gpu, root, gpu_threads, binding)
            trees = [
                cpu.submit(fit_one, root, g, arm, fit_threads, binding)
                for g in previous.TREE_GROUPS
                for arm in experiment.ARMS
            ]
            for j in trees:
                j.result()
            cpu_cal = cpu.submit(calibrate, root, reg, previous.TREE_GROUPS, fit_threads, binding)
            neural.result()
            gpu_cal = gpu.submit(calibrate, root, reg, previous.NEURAL_GROUPS, gpu_threads, binding)
            cpu_cal.result()
            gpu_cal.result()
            stage(root, "r22-control-replay", binding, partial(controls.replay_calibration, root, reg))
            monitor.activity = "blind_scoring_while_next_capture_prepares"
            for s, ready in zip(tests, prep_jobs):
                ready.result()
                name = s["name"]
                scoring_budget = (
                    threads - gpu_threads - (0 if all(job.done() for job in prep_jobs) else history_threads)
                )
                neural = gpu.submit(score, root, name, previous.NEURAL_GROUPS, gpu_threads, binding)
                jobs = [
                    cpu.submit(score, root, name, [g], max(1, scoring_budget // 2), binding)
                    for g in previous.TREE_GROUPS
                ]
                for j in jobs:
                    j.result()
                neural.result()
        monitor.activity = "check_chronology_and_open_sealed_decisions"
        stage(root, "chronology", binding, partial(data.check_chronology, root, reg))
        for s in tests:
            stage(root, "judge-" + s["name"], binding, partial(evaluation.judge, root, s))
        stage(root, "outcome", binding, partial(evaluation.overview, root, tests))
        monitor.activity = "exclusive_history_benchmark"
        stage(
            root, "history-benchmark", binding, partial(probe_history, root, bench_capture, history_threads)
        )
        monitor.activity = "exclusive_single_row_cost"
        for role in ("R", "H", "E", "X"):
            name = next((s["name"] for s in tests if s["role"] == role), None)
            if name is not None:
                stage(root, "cost-" + name, binding, partial(cost, root, name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--threads", type=int, default=min(64, psutil.cpu_count(logical=False) or 4))
    parser.add_argument("--prepare-workers", type=int, default=2)
    args = parser.parse_args()
    if not 4 <= args.threads <= 64 or not 1 <= args.prepare_workers <= min(4, args.threads):
        parser.error("Expected 4 to 64 threads and 1 to 4 early capture workers")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "run.lock").is_file():
        parser.error("Use an empty directory or resume this batch at its original path")
    own_windows_workers()
    with run_lock(root):
        try:
            run(
                root,
                args.source_dir.resolve() if args.source_dir else None,
                args.threads,
                args.prepare_workers,
            )
        except Exception:
            write_json(
                root / "last_invocation.json",
                {"status": "failed", "utc": utc_now(), "trace": traceback.format_exc()},
            )
            pack(root, prefix="response-results")
            raise
        write_json(root / "last_invocation.json", {"status": "complete", "utc": utc_now()})
        pack(root, prefix="response-results")
