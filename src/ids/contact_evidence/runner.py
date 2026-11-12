"""Run the frozen contact-context experiment with separate CPU and CUDA queues."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.route_batch.protocol import REPO, source_snapshot
from . import data, evaluation, experiment
from .features import ARMS, enrich_prepared
from .online import benchmark_online
from .policy import GATE_COLUMNS, POLICIES, SINGLE

RELEASE = "contact-evidence-lab-v1"


def synthetic(root):
    tests = sorted((REPO / "tests").glob("test_contact_*.py"))
    if not tests:
        raise RuntimeError("Missing contact experiment tests")
    log, junit = root / "evidence/check.log", root / "evidence/check.xml"
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                str(root / "pytest-temp"),
                "--junitxml=" + str(junit),
                *map(str, tests),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=REPO,
            check=True,
        )
    return [log, junit]


def gpu_probe(root):
    import torch
    from ids.flow_history.model import _network

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable in this interpreter; check the lab route-core environment and driver"
        )
    net = _network(len(ARMS["contact"]), seed=17, base_width=len(ARMS["raw"])).to("cuda:0").eval()
    x = torch.zeros((4096, len(ARMS["contact"])), device="cuda:0")
    with torch.inference_mode():
        net(x)
        torch.cuda.synchronize()
        start = time.perf_counter()
        z = net(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    if z.device.type != "cuda" or not torch.isfinite(z).all().item():
        raise RuntimeError("CUDA model forward did not return finite predictions on the requested device")
    result = root / "evidence/cuda-probe.json"
    write_json(
        result,
        {
            "gpu": torch.cuda.get_device_name(),
            "device": str(z.device),
            "torch": str(torch.__version__),
            "rows": len(x),
            "seconds": elapsed,
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "probe_input": "synthetic zeros",
        },
    )
    return [result]


def augment(root, name):
    dest = experiment.enriched(root, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    audit = enrich_prepared(data.prepared(root, name), dest)
    report = root / "evidence/features" / (name + ".json")
    write_json(report, audit)
    return [dest, report]


def prepare_capture(root, source_dir, spec, binding, threads):
    pa.set_cpu_count(threads)
    name = spec["name"]
    stage(root, "prepare-" + name, binding, partial(data.prepare_one, root, source_dir, spec))
    stage(root, "features-" + name, binding, partial(augment, root, name))


def fit_group(root, group, threads, binding):
    for arm in SINGLE:
        stage(root, "fit-" + group + "-" + arm, binding, partial(experiment.fit, root, group, arm, threads))


def gpu_queue(root, groups, threads, binding):
    pa.set_cpu_count(threads)
    for group in groups:
        fit_group(root, group, threads, binding)


def fit_cpu(root, group, arm, threads, binding):
    pa.set_cpu_count(threads)
    stage(root, "fit-" + group + "-" + arm, binding, partial(experiment.fit, root, group, arm, threads))


def calibrate_groups(root, reg, groups, threads, binding):
    pa.set_cpu_count(threads)
    for g in groups:
        stage(root, "calibrate-" + g, binding, partial(experiment.calibrate, root, reg, g, threads))


def score_groups(root, names, groups, threads, binding):
    pa.set_cpu_count(threads)
    for name in names:
        for g in groups:
            stage(
                root,
                "score-" + name + "-" + g,
                binding,
                partial(experiment.blind_score, root, name, g, threads),
            )


def cost(root, name, group):
    """Time HGB alone, and the deployed two-model alarm with the physical stream replay."""
    fitted = experiment.models(root, group, 1)
    thresholds = read_json(root / "evidence/calibration" / (group + ".json"))
    evidence = {}
    positions = [ARMS["contact"].index(col) for col in GATE_COLUMNS]
    blind = pq.read_table(root / "evidence/blind" / name / (group + ".parquet"), columns=["flags"])[
        "flags"
    ].to_numpy()
    with threadpool_limits(limits=1):
        for key in ("raw", "history", "contact", "contact_add"):
            arm = "contact" if key == "contact_add" else key

            def one(vector):
                row = np.asarray(vector).reshape(1, -1)
                if key != "contact_add":
                    return float(fitted[arm].estimator.decision_function(row)[0])
                # The anchor short-circuits. Only eligible new contacts require a second prediction.
                raw = fitted["raw"].estimator.decision_function(row[:, : len(ARMS["raw"])])
                anchor_cut = thresholds["anchor"]["threshold"]
                if anchor_cut is not None and raw[0] > anchor_cut:
                    return 1.0
                service, pair = row[0, positions]
                if not (0 <= service < 1 and pair <= 0):
                    return 0.0
                extra = fitted["contact"].estimator.decision_function(row)
                rule = thresholds["contact_add"]
                cut = rule["threshold"]
                if cut is None:
                    return 0.0
                return float(extra[0] >= cut if rule["included_at_cut"] else extra[0] > cut)

            cut = 0.5 if key == "contact_add" else thresholds[key]["threshold"]
            measured = benchmark_online(experiment.enriched(root, name), one, arm, rows=128, threshold=cut)
            rows = np.concatenate(
                [
                    np.arange(b["first_row_id"] - 1, b["first_row_id"] - 1 + b["rows"])
                    for b in measured["blocks"]
                ]
            )
            observed = measured.get("alerts", np.zeros(len(rows), dtype=bool))
            if not np.array_equal(observed, evaluation.unpack(blind[rows], key)):
                raise ValueError(f"Online decisions differ from the sealed {key} predictions")
            measured["matches_sealed_predictions"] = True
            components = ["raw", "contact"] if key == "contact_add" else [arm]
            measured["model_artifact_bytes"] = sum(
                experiment.artifact(root, group, c).stat().st_size for c in components
            )
            evidence[key] = measured
    target = root / "evidence/cost" / name / (group + ".json")
    write_json(target, evidence)
    return [target]


def run(root, source_dir, threads, prep_workers):
    own_windows_workers()
    reg = data.read_registration()
    env = environment_snapshot()
    previous = root / "protocol.json"
    spec = {
        "release": RELEASE,
        "data": reg,
        "groups": list(experiment.GROUPS),
        "arms": ARMS,
        "policies_in_bit_order": list(POLICIES),
        "gate_bit": 7,
        "cpu_threads": threads,
        "prepare_workers": prep_workers,
        "source": source_snapshot()["source_sha256"],
        "r21_reference_sha256": {
            f: sha256_file(data.REGISTRY.with_name(f))
            for f in ("r21-reference.json", "r21-normal-controls.parquet")
        },
        "runtime": {k: env[k] for k in ("python", "packages")},
    }
    # JSON round trip also converts immutable feature tuples to their persisted representation.
    binding = canonical_hash(spec)
    if previous.is_file() and canonical_hash(read_json(previous)) != binding:
        raise ValueError("Existing output belongs to a different release, environment, or resource plan")
    write_json(previous, spec)
    write_json(root / "environment.json", env)
    (root / "evidence").mkdir(exist_ok=True)
    trees = experiment.TREE_GROUPS
    mlps = experiment.NEURAL_GROUPS
    names = experiment.sources(reg, "E")
    gpu_threads = min(3, max(1, threads // 6))
    cpu_budget = threads - gpu_threads
    cpu_jobs = min(4, cpu_budget)
    with ResourceLog(root) as monitor:
        monitor.activity = "synthetic_preflight"
        stage(root, "cuda-probe", binding, partial(gpu_probe, root))
        stage(root, "synthetic", binding, partial(synthetic, root))
        monitor.activity = "prepare_earlier_and_independent_captures"
        with (
            ProcessPoolExecutor(max_workers=prep_workers) as pool,
            ProcessPoolExecutor(max_workers=cpu_jobs) as cpu,
            ProcessPoolExecutor(max_workers=1) as gpu,
        ):
            earlier = [s for s in reg["sources"] if s["role"] != "E"]
            future = [s for s in reg["sources"] if s["role"] == "E"]
            tasks = [
                pool.submit(prepare_capture, root, source_dir, s, binding, max(1, threads // prep_workers))
                for s in earlier
            ]
            future_jobs = [pool.submit(prepare_capture, root, source_dir, s, binding, 1) for s in future]
            for t in tasks:
                t.result()
            stage(root, "train-selection", binding, partial(experiment.training, root, reg))
            monitor.activity = "cpu_and_cuda_fit_while_future_logs_are_prepared"
            neural = gpu.submit(gpu_queue, root, mlps, gpu_threads, binding)
            fit_threads = max(1, (cpu_budget - prep_workers) // cpu_jobs)
            tasks = [cpu.submit(fit_cpu, root, g, arm, fit_threads, binding) for g in trees for arm in SINGLE]
            for t in tasks:
                t.result()
            neural.result()
            monitor.activity = "normal_only_registration"
            neural = gpu.submit(calibrate_groups, root, reg, mlps, gpu_threads, binding)
            cpu.submit(
                calibrate_groups, root, reg, trees, max(1, cpu_budget - prep_workers), binding
            ).result()
            neural.result()
            stage(root, "r21-control-replay", binding, partial(experiment.replay_previous_controls, root))
            for t in future_jobs:
                t.result()
            stage(root, "chronology", binding, partial(data.check_chronology, root, reg))
            monitor.activity = "blind_cpu_and_cuda_scoring"
            neural = gpu.submit(score_groups, root, names, mlps, gpu_threads, binding)
            workers = min(cpu_budget, len(trees))
            jobs = [
                cpu.submit(score_groups, root, names, [g], max(1, cpu_budget // workers), binding)
                for g in trees
            ]
            for job in jobs:
                job.result()
            neural.result()
        monitor.activity = "sealed_predictions_family_evaluation"
        for name in names:
            stage(root, "judge-" + name, binding, partial(evaluation.judge, root, name))
        stage(root, "outcome", binding, partial(evaluation.overview, root, names))
        monitor.activity = "exclusive_stream_and_two_model_cost"
        pa.set_cpu_count(1)
        for name in names:
            for group in trees:
                stage(root, "cost-" + name + "-" + group, binding, partial(cost, root, name, group))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--threads", type=int, default=min(64, psutil.cpu_count(logical=False) or 2))
    parser.add_argument("--prepare-workers", type=int, default=2)
    args = parser.parse_args()
    if not 2 <= args.threads <= 64 or not 1 <= args.prepare_workers <= min(4, args.threads):
        parser.error("Expected 2–64 threads and 1–4 source workers within that budget")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not (root / "run.lock").is_file():
        parser.error("Use an empty output or the original round-22 run directory")
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
            pack(root, prefix="contact-results")
            raise
        write_json(root / "last_invocation.json", {"status": "complete", "utc": utc_now()})
        pack(root, prefix="contact-results")
