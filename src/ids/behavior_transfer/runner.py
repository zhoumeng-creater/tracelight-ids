"""Run the registered R25 data, frozen-model, threshold and event comparisons."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import sys
import traceback

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import canonical_hash, environment_snapshot, read_json, utc_now, write_json
from ids.contact_evidence.experiment import GROUPS, NEURAL_GROUPS, TREE_GROUPS
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.raw_workpoint import controls
from ids.raw_workpoint.scoring import score_group
from ids.response_evidence.history import enrich_prepared
from ids.route_batch.protocol import REPO, source_snapshot
from . import dedale, sfc
from .decisions import calibration_frame, decide, COMPONENTS
from .evaluation import judge
from .scan_history import build_scan_history

RELEASE = "behavior-transfer-lab-v1"
CONFIG = REPO / "configs/behavior-transfer-20261114"


def checks(root):
    path = root / "evidence/check.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    tests = sorted((REPO / "tests").glob("test_behavior_*.py"))
    if not tests:
        raise ValueError("Missing R25 checks")
    with path.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, tests)],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return [path]


def prepare_provider(root, cache, provider, workers):
    module = {"sfc": sfc, "dedale": dedale}[provider]
    folder = root / "local/providers" / provider
    specs = module.prepare_sources(cache / provider, folder, workers=workers)
    if not specs:
        raise ValueError(f"{provider} produced no registered daily sources")
    serializable = [{key: str(value) if isinstance(value, Path) else value for key, value in s.items()}
                    for s in specs]
    path = root / "evidence/sources" / (provider + "-index.json")
    write_json(path, serializable)
    outputs = [path]
    for name in ("native_identity", "author_event_schedule"):
        source_audit = folder / "audit" / (name + ".json")
        if source_audit.is_file():
            audit = root / "evidence/sources" / (provider + "-" + name + ".json")
            write_json(audit, read_json(source_audit))
            outputs.append(audit)
    for spec in serializable:
        if spec.get("status") == "failed":
            continue
        for name in ("features", "truth", "audit"):
            member = Path(spec[name])
            if not member.is_file():
                raise ValueError(f"Prepared {provider} source missing {name}: {spec['name']}")
            outputs.append(member)
        audit = root / "evidence/sources" / (spec["name"] + ".json")
        write_json(audit, read_json(spec["audit"]))
        outputs.append(audit)
    if any(s.get("status") == "failed" for s in serializable):
        # Index and completed outputs stay available, but no success receipt freezes an incomplete provider.
        raise RuntimeError(f"{provider} has failed daily sources; see {path}")
    return outputs


def features(root, spec, threads):
    name = spec["name"]
    path = root / "local/features" / (name + ".parquet")
    stats = enrich_prepared(Path(spec["features"]), path, workers=threads)
    proxy_columns = ["row_id", "source", "destination", "native_protocol", "resp_port",
                     "flow_end_ms", "response_evidence_code"]
    raw = pd.read_parquet(spec["features"], columns=proxy_columns)
    proxy = build_scan_history(raw, workers=min(2, threads))
    report = root / "evidence/history" / (name + ".json")
    write_json(report, {"source": name, "main": stats, "scan_proxy": proxy.attrs["evidence"]})
    proxy.insert(0, "row_id", raw.row_id)
    output = root / "local/proxy" / (name + ".parquet")
    output.parent.mkdir(parents=True, exist_ok=True)
    proxy.to_parquet(output, index=False, compression="zstd")
    return [path, output, report]


def score(root, spec, group, threads, batch_rows):
    name = spec["name"]
    base = root / "local/controls"
    output = root / "local/scores" / name / (group + ".parquet")
    device = "cpu" if group in TREE_GROUPS else "cuda:0"
    neural = None
    if group in NEURAL_GROUPS:
        neural = read_json(root / "evidence/gpu-batch" / (group + ".json"))["selected"]
    stats = score_group(root / "local/features" / (name + ".parquet"), Path(spec["features"]),
                        base / "models/frozen", base / "evidence/calibration", group, output,
                        threads=threads, device=device, batch_rows=batch_rows, neural_batch_rows=neural)
    report = root / "evidence/scoring" / name / (group + ".json")
    write_json(report, stats)
    return [output, report]


def seal(root, spec, group):
    name = spec["name"]
    return decide(root / "local/scores" / name / (group + ".parquet"),
                  root / "local/features" / (name + ".parquet"),
                  root / "local/proxy" / (name + ".parquet"), calibration_frame(root, group),
                  root / "local/decisions" / name / (group + ".parquet"),
                  root / "evidence/thresholds" / name / (group + ".json"))


def evaluate(root, spec, group):
    return judge(root / "local/decisions" / spec["name"] / (group + ".parquet"),
                 Path(spec["truth"]), root / "local/features" / (spec["name"] + ".parquet"),
                 root / "evidence/evaluation" / spec["name"] / group, source=spec, group=group,
                 scores_file=root / "local/scores" / spec["name"] / (group + ".parquet"))


def attempt(root, name, binding, callback):
    try:
        stage(root, name, binding, callback)
        return None
    except Exception:
        failure = {"stage": name, "utc": utc_now(), "trace": traceback.format_exc()}
        write_json(root / "evidence/failures" / (name + ".json"), failure)
        print("[failed] " + name + "; independent work continues", flush=True)
        return failure


def score_queue(root, spec, groups, threads, batch_rows, binding):
    pa.set_cpu_count(threads)
    failures = []
    for group in groups:
        name = spec["name"] + "-" + group
        fail = attempt(root, "score-" + name, binding,
                       partial(score, root, spec, group, threads, batch_rows))
        if fail is None:
            fail = attempt(root, "seal-" + name, binding, partial(seal, root, spec, group))
        if fail is not None:
            failures.append(fail)
    return failures


def summarize(root, specs, failures):
    tables = []
    pairs = []
    for spec in specs:
        for group in GROUPS:
            folder = root / "evidence/evaluation" / spec["name"] / group
            if (folder / "budget.csv.gz").is_file():
                tables.append(pd.read_csv(folder / "budget.csv.gz"))
                pairs.append(pd.read_csv(folder / "paired.csv.gz"))
    outputs = []
    for name, frames in (("budgets", tables), ("paired", pairs)):
        if frames:
            path = root / "evidence" / ("all-" + name + ".csv.gz")
            pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")
            outputs.append(path)
    path = root / "outcome.json"
    write_json(path, {"status": "complete" if not failures else "incomplete", "release": RELEASE,
                      "successful_prepared_sources": len(specs), "expected_groups_per_source": len(GROUPS),
                      "evaluated_group_sources": len(tables),
                      "failures": failures, "fit_count": 0,
                      "primary_policy": "quality_counts_local; all registered learners reported",
                      "required_interpretation": [
                          "Each registered day-file/block is a cold start; each observed UTC day includes its startup rows.",
                          "Local first-hour rules use all traffic. Their reference alert budget is not a future FPR guarantee.",
                          "Unknown or author-related-only truth is separate from normal and verified attacks.",
                          "Count independent author events, not repeated flows or seeds.",
                          "Class gains and losses are paired against both frozen raw and frozen add_quality.",
                          "Earlier or already opened rows do not become new forward evidence.",
                          "No candidate is selected by best test day or seed.",
                      ]})
    return outputs + [path]


def run(root, cache, threads, batch_rows, providers):
    from .resources import gpu_batch_probe, history_benchmark
    environment = environment_snapshot()
    registered = {p: read_json(CONFIG / (p + ".json")) for p in providers}
    protocol = {"release": RELEASE, "sources": registered, "providers": providers,
                "controls": read_json(controls.CONFIG / "controls.json"),
                "source_snapshot": source_snapshot(), "threads": threads, "batch_rows": batch_rows,
                "runtime": environment["packages"], "groups": list(GROUPS),
                "components": {k: list(v) for k, v in COMPONENTS.items()},
                "local_prefix_ms": 3600000, "alpha": 0.01, "cold_start": "per registered source day file/block",
                "fit": False, "primary": "quality_counts_local, each day whole and startup/after"}
    binding = canonical_hash(protocol)
    if (root / "protocol.json").exists() and canonical_hash(read_json(root / "protocol.json")) != binding:
        raise ValueError("Resume requires the same source, settings, runtime and code")
    write_json(root / "protocol.json", protocol)
    write_json(root / "environment.json", environment)
    gpu_threads = max(1, min(4, threads // 6))
    tree_threads = max(1, (threads - gpu_threads) // 2)
    failures, specs = [], []
    with ResourceLog(root) as monitor:
        stage(root, "checks", binding, partial(checks, root))
        stage(root, "controls", binding, partial(controls.unpack, root))
        monitor.activity = "frozen_C_replay_and_GPU_throughput"
        for group in GROUPS:
            stage(root, "probe-" + group, binding,
                  partial(controls.probe, root, group,
                          tree_threads if group in TREE_GROUPS else gpu_threads, batch_rows=batch_rows))
        for group in NEURAL_GROUPS:
            stage(root, "gpu-batch-" + group, binding, partial(gpu_batch_probe, root, group, gpu_threads))
        monitor.activity = "parallel_source_download_decode_identity"
        # Each provider manages independent member/day workers and its own resumable inputs.
        for provider in providers:
            failure = attempt(root, "prepare-" + provider, binding,
                              partial(prepare_provider, root, cache, provider, min(8, threads)))
            if failure:
                failures.append(failure)
            index = root / "evidence/sources" / (provider + "-index.json")
            if index.is_file():
                items = read_json(index)
                specs.extend(s for s in items if s.get("status") != "failed")
        names = [s["name"] for s in specs]
        if len(names) != len(set(names)):
            raise ValueError("Daily source names collide")
        benchmark_sources = {}
        for spec in specs:
            provider = "sfc" if spec["name"].startswith("IoT-SFC") else "dedale"
            rows = pq.read_metadata(spec["features"]).num_rows
            if provider not in benchmark_sources or rows > benchmark_sources[provider][0]:
                benchmark_sources[provider] = (rows, spec["name"])
        for spec in specs:
            name = spec["name"]
            monitor.activity = "history-" + name
            failure = attempt(root, "history-" + name, binding, partial(features, root, spec, threads))
            if failure:
                failures.append(failure)
                continue
            provider = "sfc" if name.startswith("IoT-SFC") else "dedale"
            if name == benchmark_sources[provider][1]:
                failure = attempt(root, "history-benchmark-" + provider, binding,
                                  partial(history_benchmark, Path(spec["features"]),
                                          root / "evidence/history-benchmark" / provider, threads))
                if failure:
                    failures.append(failure)
            monitor.activity = "CPU_trees_and_CUDA_MLP-" + name
            with ProcessPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(score_queue, root, spec, [g], tree_threads, batch_rows, binding)
                        for g in TREE_GROUPS]
                jobs.append(pool.submit(score_queue, root, spec, NEURAL_GROUPS, gpu_threads, batch_rows, binding))
                for job in jobs:
                    failures.extend(job.result())
            monitor.activity = "sealed_label_evaluation-" + name
            for group in GROUPS:
                if not (root / "receipts" / ("seal-" + name + "-" + group + ".json")).is_file():
                    continue
                failure = attempt(root, "judge-" + name + "-" + group, binding,
                                  partial(evaluate, root, spec, group))
                if failure:
                    failures.append(failure)
        if not specs:
            failures.append({"stage": "source_coverage", "reason": "No prepared source available"})
        summarize(root, specs, failures)
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "local/behavior-transfer-r25")
    parser.add_argument("--cache", type=Path, default=REPO / "data/raw/behavior-transfer-20261114")
    parser.add_argument("--threads", type=int, default=max(4, min(64, psutil.cpu_count(logical=False) or 4)))
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument("--providers", nargs="+", choices=("sfc", "dedale"), default=["sfc", "dedale"])
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not 4 <= args.threads <= 64 or not 4096 <= args.batch_rows <= 262144:
        parser.error("Require 4..64 threads and 4096..262144 batch rows")
    if len(set(args.providers)) != len(args.providers):
        parser.error("Duplicate provider")
    if args.plan:
        for provider in args.providers:
            reg = read_json(CONFIG / (provider + ".json"))
            print(provider, reg.get("description", reg.get("format", "registered sources")))
        print("groups", ", ".join(GROUPS), "; new fits = 0; CPU and CUDA queues; daily 1h prefixes")
        return
    pa.set_cpu_count(args.threads)
    own_windows_workers()
    root, cache = args.output.resolve(), args.cache.resolve()
    with run_lock(root):
        complete = False
        try:
            complete = run(root, cache, args.threads, args.batch_rows, args.providers)
        except Exception:
            write_json(root / "fatal.json", {"utc": utc_now(), "trace": traceback.format_exc()})
            raise
        finally:
            write_json(root / "last_invocation.json",
                       {"status": "complete" if complete else "incomplete", "utc": utc_now()})
            pack(root, prefix="behavior-results" if complete else "behavior-incomplete")
        if not complete:
            raise SystemExit(2)
