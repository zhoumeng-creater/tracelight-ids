"""Run the preregistered IoT-23 forward comparison, with source and per-stage receipts."""

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
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.route_batch.protocol import source_snapshot
from . import experiment
from .observations import prepare
from .sources import REGISTRY, resolve_log


def preflight(root):
    repository = REGISTRY.parents[2]
    tests = sorted((repository / "tests").glob("test_threshold_*.py"))
    if not tests:
        raise ValueError("Missing threshold replay tests")
    log = root / "evidence/preflight.log"
    junit = root / "evidence/preflight.xml"
    with log.open("w", encoding="utf-8") as out:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}", *map(str, tests)],
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=repository,
            check=True,
        )
    return [log, junit]


def prepare_one(root, source_dir, spec):
    path = resolve_log(source_dir, root / "source", spec)
    dest = experiment.prepared(root, spec["name"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit = prepare(path, dest)
    if audit["rows"] != spec["rows"] or audit["family_counts"] != spec["families"]:
        raise ValueError(f"Source population differs from registration: {spec['name']}")
    doc = root / "evidence/sources" / (spec["name"] + ".json")
    write_json(doc, {"source_sha256": spec["sha256"], **audit})
    return [dest, doc]


def before_scoring(root, reg):
    train, control, future = (experiment.inputs(reg, role) for role in ("P", "C", "E"))

    def bounds(name):
        audit = read_json(root / "evidence/sources" / (name + ".json"))
        return audit["first_flow_end_ms"], audit["last_flow_end_ms"]

    if not (
        max(bounds(n)[1] for n in train) < min(bounds(n)[0] for n in control)
        and max(bounds(n)[1] for n in control) < bounds(future[0])[0]
        and bounds(future[0])[1] < bounds(future[1])[0]
    ):
        raise ValueError("Preparation and untouched evaluation periods must be strictly chronological")
    path = root / "evidence/chronology.json"
    write_json(
        path,
        {
            "P": {n: bounds(n) for n in train},
            "C": {n: bounds(n) for n in control},
            "E": {n: bounds(n) for n in future},
            "separate_scenarios": True,
        },
    )
    return [path]


def perform(job):
    root, binding, label, func, args, worker_threads = job
    pa.set_cpu_count(worker_threads)
    stage(root, label, binding, partial(func, *args))


def parallel(jobs, workers):
    if workers == 1:
        for job in jobs:
            perform(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = [pool.submit(perform, job) for job in jobs]
        for result in results:
            result.result()


def run(root, source_dir, threads, prep_workers):
    own_windows_workers()
    reg = read_json(REGISTRY)
    env = environment_snapshot()
    spec = {
        "registration": reg,
        "source_sha256": source_snapshot()["source_sha256"],
        "threads": threads,
        "preparation_workers": prep_workers,
        "runtime": {k: env.get(k) for k in ("python", "packages")},
    }
    binding = canonical_hash(spec)
    protocol = root / "protocol.json"
    if protocol.is_file() and read_json(protocol) != spec:
        raise ValueError("Output is bound to another protocol or runtime")
    write_json(protocol, spec)
    write_json(root / "environment.json", env)
    (root / "evidence").mkdir(exist_ok=True)
    with ResourceLog(root) as monitor:
        monitor.activity = "synthetic"
        stage(root, "preflight", binding, partial(preflight, root))
        monitor.activity = "verify_and_prepare_sources"
        jobs = [
            (
                root,
                binding,
                "source-" + s["name"],
                prepare_one,
                (root, source_dir, s),
                max(1, threads // prep_workers),
            )
            for s in reg["sources"]
        ]
        parallel(jobs, prep_workers)
        stage(root, "chronology", binding, partial(before_scoring, root, reg))
        stage(root, "train-selection", binding, partial(experiment.draw_train, root, reg))
        half = max(1, threads // 2)
        monitor.activity = "paired_hgb_fit_and_normal_calibration"
        jobs = [
            (root, binding, arm + "-fit", experiment.train_arm, (root, arm, half, reg), half)
            for arm in experiment.ARMS
        ]
        parallel(jobs, 2)
        future = experiment.inputs(reg, "E")
        monitor.activity = "blind_batch_scoring"
        jobs = [
            (root, binding, n + "-" + a + "-scores", experiment.score_arm, (root, n, a, half), half)
            for n in future
            for a in experiment.ARMS
        ]
        parallel(jobs, 2)
        monitor.activity = "compare_fixed_and_past_only_decisions"
        for name in future:
            stage(root, name + "-evaluation", binding, partial(experiment.judge, root, name))
    path = root / "evidence/paired-outcome.json"
    per_scenario = {}
    for name in future:
        doc = read_json(root / "evidence" / name / "comparison.json")
        per_scenario[name] = {}
        for key, res in doc["comparisons"].items():
            by = res["groups"]
            per_scenario[name][key] = {
                "fp": by["normal"],
                "scans": by["Scanning"],
                "cc": by.get("C&C"),
                "max_fp": res["normal_false_alert_budget"],
                "within_budget": res["candidate_within_one_percent"],
                "baseline_within_budget": res["baseline_within_one_percent"],
                "within_budget_more_scan_and_no_previously_detected_cc_lost": res[
                    "within_budget_more_scan_and_no_previously_detected_cc_lost"
                ],
                "both_within_budget_more_scan_and_cc_preserved": res[
                    "both_within_budget_more_scan_and_cc_preserved"
                ],
                "any_cc_detected": res["any_cc_detected"],
                "cc_all_retained_nonvacuous": res["cc_all_retained_nonvacuous"],
            }
    write_json(
        path,
        {
            "primary": "raw_vs_history_unlabeled",
            "scenarios": per_scenario,
            "cross_scenario_claim_requires_both": True,
            "cc_means": "Exact author C&C label; CIC2018 Bot is a different endpoint.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="Folder with all six registered gzip logs; fully offline."
    )
    parser.add_argument("--threads", type=int, default=min(24, psutil.cpu_count(logical=False) or 2))
    parser.add_argument("--prepare-workers", type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.threads <= 64 or not 1 <= args.prepare_workers <= min(6, args.threads):
        parser.error("Expected 2-64 CPU threads and 1-6 preparation workers within that budget")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "run.lock").is_file():
        parser.error("Use an empty directory or a previous run of this experiment")
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
                {"status": "failed", "at": utc_now(), "error": traceback.format_exc()},
            )
            pack(root, prefix="forward-threshold-results")
            raise
        write_json(root / "last_invocation.json", {"status": "complete", "at": utc_now()})
        pack(root, prefix="forward-threshold-results")
