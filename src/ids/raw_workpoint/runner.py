"""Evaluate frozen low-FPR controls on IoT-23 capture 43-1."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys
import traceback

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import canonical_hash, environment_snapshot, read_json, utc_now, write_json
from ids.contact_evidence.data import prepared, truth
from ids.contact_evidence.experiment import GROUPS, NEURAL_GROUPS, TREE_GROUPS
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.response_evidence.history import enrich_prepared
from ids.route_batch.protocol import REPO, source_snapshot
from . import controls, source
from .cost import measure
from .evaluate import evaluate
from .examples import collect
from .outcome import summarize_outcome
from .ranking import diagnose
from .scoring import score_group
from .support import summarize

NAME = "CTU-IoT-Malware-Capture-43-1"
RELEASE = "raw-working-point-lab-v1"


def checks(root):
    tests = sorted((REPO / "tests").glob("test_workpoint_*.py"))
    if not tests:
        raise RuntimeError("Missing workpoint synthetic checks")
    log = root / "evidence/check.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as file:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, tests)],
            cwd=REPO,
            stdout=file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return [log]


def prepare_source(root, source_dir, workers):
    paths = source.prepare(root, source_dir=source_dir, workers=workers)
    with pq.ParquetFile(prepared(root, NAME)) as stream:
        first = next(stream.iter_batches(columns=["flow_end_ms"], batch_size=1))["flow_end_ms"][0].as_py()
    last = read_json(controls.CONFIG / "controls.json")["last_calibration_ms"]
    if first <= last:
        raise ValueError("Capture begins before the end of normal calibration")
    return paths


def history(root, threads):
    enriched = root / "local/features" / (NAME + ".parquet")
    stats = enrich_prepared(prepared(root, NAME), enriched, workers=threads)
    report = root / "evidence/history.json"
    write_json(report, stats)
    return [enriched, report]


def prediction_path(root, group):
    return root / "local/predictions" / (group + ".parquet")


def score(root, group, threads, batch_rows):
    pa.set_cpu_count(threads)
    base = root / "local/controls"
    output = prediction_path(root, group)
    output.parent.mkdir(parents=True, exist_ok=True)
    device = "cpu" if group in TREE_GROUPS else "cuda:0"
    stats = score_group(
        root / "local/features" / (NAME + ".parquet"),
        prepared(root, NAME),
        base / "models/frozen",
        base / "evidence/calibration",
        group,
        output,
        threads=threads,
        device=device,
        batch_rows=batch_rows,
    )
    report = root / "evidence/scoring" / (group + ".json")
    write_json(report, stats)
    return [output, report]


def score_queue(root, groups, threads, batch_rows, binding):
    for group in groups:
        stage(root, "score-" + group, binding, partial(score, root, group, threads, batch_rows))


def judge(root, group, batch_rows, binding):
    path = root / "evidence/paired" / (group + ".json")

    def run():
        evaluate(
            prediction_path(root, group), truth(root, NAME), prepared(root, NAME), path, batch_rows=batch_rows
        )
        return [path]

    stage(root, "judge-" + group, binding, run)


def run(root, source_dir, threads, batch_rows):
    environment = environment_snapshot()
    register = {
        "release": RELEASE,
        "source": source.read_spec(),
        "controls": read_json(controls.CONFIG / "controls.json"),
        "src_sha256": source_snapshot()["source_sha256"],
        "runtime": environment["packages"],
        "threads": threads,
        "batch_rows": batch_rows,
        "groups": list(GROUPS),
        "primary": "hgb:add_contact vs raw; whole capture and every UTC day",
        "secondary": ["add_web64k", "add_quality", "contact", "quality", "add_quic30"],
        "quic30_role": "duration-specificity negative control; no independent QUIC C2 claim",
        "fit": False,
        "cut_inputs": "Round 23 early normals only",
    }
    binding = canonical_hash(register)
    protocol = root / "protocol.json"
    if protocol.exists() and canonical_hash(read_json(protocol)) != binding:
        raise ValueError("Resuming requires the original code, controls, inputs and parameters")
    write_json(protocol, register)
    write_json(root / "environment.json", environment)
    gpu_threads = max(1, min(4, threads // 6))
    tree_threads = max(1, (threads - gpu_threads) // len(TREE_GROUPS))
    with ResourceLog(root) as sampler:
        sampler.activity = "checks_and_frozen_control_replay"
        stage(root, "checks", binding, partial(checks, root))
        stage(root, "controls", binding, partial(controls.unpack, root))
        for group in GROUPS:
            stage(
                root,
                "probe-" + group,
                binding,
                partial(
                    controls.probe,
                    root,
                    group,
                    tree_threads if group in TREE_GROUPS else gpu_threads,
                    batch_rows=batch_rows,
                ),
            )
        sampler.activity = "download_and_parallel_decode"
        stage(root, "prepare", binding, partial(prepare_source, root, source_dir, threads))
        sampler.activity = "parallel_60s_history"
        stage(root, "history", binding, partial(history, root, threads))
        sampler.activity = "concurrent_HGB_CPU_and_MLP_CUDA"
        with ProcessPoolExecutor(max_workers=3) as pool:
            jobs = [
                pool.submit(score_queue, root, [g], tree_threads, batch_rows, binding) for g in TREE_GROUPS
            ]
            jobs.append(pool.submit(score_queue, root, NEURAL_GROUPS, gpu_threads, batch_rows, binding))
            for job in jobs:
                job.result()
        sampler.activity = "open_labels_and_evaluate_sealed_predictions"
        workers = max(1, min(len(GROUPS), threads // 2))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(judge, root, g, batch_rows, binding) for g in GROUPS]
            for job in jobs:
                job.result()
        sampler.activity = "retrospective_ranking_ceiling"
        stage(root, "outcome", binding, partial(summarize_outcome, root))
        stage(
            root,
            "rank-hgb",
            binding,
            partial(
                diagnose,
                prediction_path(root, "hgb"),
                truth(root, NAME),
                prepared(root, NAME),
                root / "evidence/paired/hgb.json",
                root / "evidence/hgb-ranking.json",
                batch_rows,
            ),
        )
        sampler.activity = "retain_stratified_hgb_examples"
        stage(
            root,
            "services-hgb",
            binding,
            partial(
                summarize,
                prediction_path(root, "hgb"),
                truth(root, NAME),
                prepared(root, NAME),
                root / "evidence/hgb-service-support.csv.gz",
                batch_rows=batch_rows,
            ),
        )
        stage(
            root,
            "examples-hgb",
            binding,
            partial(
                collect,
                prediction_path(root, "hgb"),
                truth(root, NAME),
                root / "local/features" / (NAME + ".parquet"),
                root / "evidence/hgb-examples.parquet",
                batch_rows=batch_rows,
            ),
        )
        sampler.activity = "exclusive_single_flow_decision_cost"
        stage(root, "cost", binding, partial(measure, root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--threads", type=int, default=max(4, min(64, psutil.cpu_count(logical=False) or 4)))
    parser.add_argument("--batch-rows", type=int, default=65536)
    args = parser.parse_args()
    if not 4 <= args.threads <= 64 or not 1 <= args.batch_rows <= 262144:
        parser.error("Require 4..64 CPU threads and 1..262144 rows per batch")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "run.lock").is_file():
        parser.error("Use an empty output directory or resume at its original path")
    own_windows_workers()
    with run_lock(root):
        try:
            run(root, args.source_dir, args.threads, args.batch_rows)
        except Exception:
            write_json(
                root / "last_invocation.json",
                {"status": "failed", "utc": utc_now(), "trace": traceback.format_exc()},
            )
            pack(root, prefix="workpoint-incomplete")
            raise
        write_json(root / "last_invocation.json", {"status": "complete", "utc": utc_now()})
        pack(root, prefix="workpoint-results")
