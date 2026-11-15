"""Replay completed R25 caches: no downloads, histories, fitting or model forwards."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.route_batch.protocol import REPO, source_snapshot
from . import GROUPS, RELEASE
from .inputs import CONFIG, below, distinct_roots, inspect_previous
from .readiness import repair_decisions
from .evaluation import prepare_context, evaluate_frames, evaluation_scopes
from .ranking import rank_diagnostics


def checks(root):
    tests = sorted((REPO / "tests").glob("test_readiness_*.py"))
    if not tests:
        raise ValueError("Missing readiness-replay checks")
    path = root / "evidence/check.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *map(str, tests)],
                       cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return [path]


def completed(root, name, binding):
    receipt = root / "receipts" / (name + ".json")
    if not receipt.is_file():
        return False
    value = read_json(receipt)
    if value.get("binding") != binding or not value.get("outputs"):
        raise ValueError(f"Repair receipt settings changed: {name}")
    for member, digest in value["outputs"].items():
        path = below(root, member)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Repair receipt output changed: {name}/{member}")
    return True


def profile(specs, threads, requested_workers, memory_gib):
    available = psutil.virtual_memory().available
    budget = int(memory_gib * 1024**3) if memory_gib is not None else int(available * .6)
    if budget <= 0 or budget > available:
        raise ValueError("Worker memory budget must fit currently available physical memory")
    largest = max((s["estimated_worker_bytes"] for s in specs), default=512 * 1024**2)
    memory_workers = budget // largest
    if specs and memory_workers < 1:
        raise ValueError(f"At least {largest / 1024**3:.2f} GiB worker budget needed for the largest source")
    limit = max(1, min(8, threads // 2))
    if requested_workers is not None:
        limit = min(limit, requested_workers)
    workers = max(1, min(limit, max(1, len(specs)), max(1, memory_workers)))
    return {"workers": workers, "threads_per_worker": max(1, threads // workers),
            "requested_cpu_threads": threads, "memory_budget_bytes": budget,
            "available_memory_at_dispatch_bytes": available, "largest_estimated_worker_bytes": largest,
            "scheduling": "One persistent pool; largest source first; shared features/truth/context for pending groups; finished groups excluded before pool creation"}


def original_budget_parity(previous, source, group, budget_file):
    old = pd.read_csv(previous / "evidence/evaluation" / source / group / "budget.csv.gz")
    new = pd.read_csv(budget_file)
    new = new.loc[new.version.eq("original") & new.phase.isin(["all", "original_startup", "original_after"])].copy()
    new["phase"] = new.phase.replace({"original_startup": "startup", "original_after": "after"})
    keys = ["source", "group", "window", "phase", "policy"]
    values = ["normal", "fp", "cap", "unresolved_rows", "unresolved_alerts"]
    a = old.set_index(keys)[values].sort_index()
    b = new.set_index(keys)[values].sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_names=True)
    return {"status": "exact", "rows": len(a), "fields": values}


def source_worker(previous, output, spec, groups, threads, binding):
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(max(1, min(4, threads)))
    with threadpool_limits(limits=threads):
        return _source_worker(previous, output, spec, groups, threads, binding)


def _source_worker(previous, output, spec, groups, threads, binding):
    started, cpu = time.perf_counter(), time.process_time()
    name = spec["name"]
    feature_path = below(previous, spec["features"])
    schema = pq.read_schema(feature_path)
    columns = [field.name for field in schema
               if field.name == "uid" or pa.types.is_integer(field.type)
               or pa.types.is_floating(field.type) or pa.types.is_boolean(field.type)]
    features = pd.read_parquet(feature_path, columns=columns)
    common_read_seconds = time.perf_counter() - started
    truth, context = None, None
    context_seconds = 0.0
    done, failures = [], []

    def run_group(group):
        nonlocal truth, context, context_seconds
        tick = time.perf_counter()
        scores = pd.read_parquet(previous / "local/scores" / name / (group + ".parquet"),
                                 columns=["row_id", "score_raw", "score_contact", "score_quality"])
        old = pd.read_parquet(previous / "local/decisions" / name / (group + ".parquet"))
        registration = read_json(previous / "evidence/thresholds" / name / (group + ".json"))
        reads = time.perf_counter() - tick
        tick = time.perf_counter()
        repaired, report = repair_decisions(features, scores, old, registration)
        decision_path = output / "local/decisions" / name / (group + ".parquet")
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_parquet(decision_path, index=False, compression="zstd")
        threshold_path = output / "evidence/readiness" / name / (group + ".json")
        write_json(threshold_path, report)
        repair_seconds = time.perf_counter() - tick

        # The repair takes no truth argument. Seal its flags before loading truth for the first group.
        if truth is None:
            tick = time.perf_counter()
            truth = pd.read_parquet(below(previous, spec["truth"]))
            if "uid" in features and "uid" in truth and not np.array_equal(features.uid, truth.uid):
                raise ValueError("Cached feature and truth UID order differs")
            context = prepare_context(features, truth)
            context_seconds = time.perf_counter() - tick
        folder = output / "evidence/evaluation" / name / group
        tick = time.perf_counter()
        files = evaluate_frames(features, scores, truth, old, repaired,
                                source=spec, group=group, output=folder, context=context)
        parity = original_budget_parity(previous, name, group, folder / "budget.csv.gz")
        evaluation_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        files += rank_diagnostics(features, scores, truth, evaluation_scopes(old, repaired),
                                  source=spec, group=group, output=folder)
        ranking_seconds = time.perf_counter() - tick
        performance = output / "evidence/performance" / name / (group + ".json")
        write_json(performance, {"source": name, "group": group, "rows": len(features),
                                 "pid": os.getpid(), "threads": threads, "input_read_seconds": reads,
                                 "repair_and_write_seconds": repair_seconds,
                                 "evaluation_and_parity_seconds": evaluation_seconds,
                                 "ranking_seconds": ranking_seconds, "original_budget_parity": parity,
                                 "process_rss_after_group": psutil.Process().memory_info().rss,
                                 "fits": 0, "model_forward_calls": 0})
        return [decision_path, threshold_path, performance, *files]

    for group in groups:
        stage_name = "replay-" + name + "-" + group
        try:
            stage(output, stage_name, binding, partial(run_group, group))
            done.append(group)
        except Exception:
            failure = {"source": name, "group": group, "stage": stage_name,
                       "trace": traceback.format_exc()}
            write_json(output / "evidence/failures" / (stage_name + ".json"), failure)
            failures.append(failure)
    write_json(output / "evidence/performance" / (name + ".json"), {
        "source": name, "pid": os.getpid(), "rows": len(features), "pending_groups": list(groups),
        "completed_groups": done, "threads": threads, "feature_read_seconds": common_read_seconds,
        "truth_read_and_context_seconds": context_seconds, "seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu, "process_rss_at_end": psutil.Process().memory_info().rss,
        "estimate_bytes": spec["estimated_worker_bytes"], "context_reused_across_groups": True})
    return failures


def summarize(root, sources, binding, failures):
    complete = []
    tables = {name: [] for name in ("budget", "paired", "ranking_thresholds", "ranking_families")}
    for spec in sources:
        for group in GROUPS:
            if not completed(root, "replay-" + spec["name"] + "-" + group, binding):
                continue
            complete.append({"source": spec["name"], "group": group})
            folder = root / "evidence/evaluation" / spec["name"] / group
            for name, frames in tables.items():
                path = folder / (name + ".csv.gz")
                if path.is_file():
                    frames.append(pd.read_csv(path))
    for name, frames in tables.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(root / "evidence" / ("all-" + name + ".csv.gz"),
                                                        index=False, compression="gzip")
    expected = len(read_json(CONFIG)["sources"]) * len(GROUPS)
    success = len(complete) == expected and not failures
    write_json(root / "outcome.json", {"status": "complete" if success else "incomplete",
                                      "release": RELEASE, "expected_group_sources": expected,
                                      "completed_group_sources": len(complete), "completed": complete,
                                      "failures": failures, "fits": 0, "model_forward_calls": 0,
                                      "history_builds": 0, "downloads": 0,
                                      "interpretation": [
                                          "R25 data are already opened: repaired causal replay, not new prospective confirmation.",
                                          "Finite rank readiness does not establish representativeness or future FPR.",
                                          "All includes startup; original_startup/after and repaired startup/after use different partitions.",
                                          "Oracle uses normal evaluation labels, each scalar separately; it is not an online detector.",
                                          "Do not add repeated flows across policies, scopes or learners as new events.",
                                          "Record both new detections and previously detected attacks lost; zero alarms can be failure.",
                                      ]})
    return success


def run(previous, output, *, threads, workers=None, memory_gib=None, preflight_only=False):
    previous, output = distinct_roots(previous, output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    environment = environment_snapshot(include_torch=False)
    with ResourceLog(output) as monitor:
        monitor.activity = "read_only_cached_input_verification"
        sources, preflight = inspect_previous(previous, hash_threads=min(4, threads))
        write_json(output / "evidence/input_preflight.json", preflight)
        if preflight_only:
            write_json(output / "outcome.json", {"status": "preflight_only", "usable_sources": len(sources),
                                                "failures": preflight["failures"], "model_forward_calls": 0})
            return not preflight["failures"]
        protocol = {"release": RELEASE, "registration": read_json(CONFIG), "source_snapshot": source_snapshot(),
                    "input_sha256": preflight["input_sha256"], "threads": threads,
                    "requested_workers": workers, "requested_memory_gib": memory_gib,
                    "runtime": {"python": environment["python"], "packages": environment["packages"]},
                    "previous_root": str(previous), "minimum_elapsed_ms": 3600000,
                    "minimum_reference_rows": 399, "fits": 0, "model_forward_calls": 0}
        binding = canonical_hash(protocol)
        path = output / "protocol.json"
        if path.exists() and canonical_hash(read_json(path)) != binding:
            raise ValueError("Resume requires the same cached inputs, code, runtime and settings; use a new output root")
        write_json(path, protocol)
        write_json(output / "environment.json", environment)
        stage(output, "checks", binding, partial(checks, output))
        failures = list(preflight["failures"])
        pending = []
        reused = 0
        for spec in sources:
            groups = [g for g in GROUPS if not completed(output, "replay-" + spec["name"] + "-" + g, binding)]
            reused += len(GROUPS) - len(groups)
            if groups:
                pending.append((spec, groups))
        plan = profile([s for s, _ in pending], threads, workers, memory_gib)
        plan.update(pending_sources=len(pending), reused_group_sources=reused,
                    pending_group_sources=sum(len(groups) for _, groups in pending))
        write_json(output / "evidence/dispatch.json", plan)
        print(f"[dispatch] {len(pending)} sources, {plan['pending_group_sources']} groups; {reused} already complete", flush=True)
        if pending:
            monitor.activity = "parallel_cached_decision_repair_evaluation_and_rank_diagnostics"
            with ProcessPoolExecutor(max_workers=plan["workers"]) as pool:
                jobs = {pool.submit(source_worker, previous, output, spec, groups,
                                    plan["threads_per_worker"], binding): spec["name"]
                        for spec, groups in sorted(pending, key=lambda x: -x[0]["rows"])}
                for job in as_completed(jobs):
                    try:
                        failures.extend(job.result())
                        print("[finished] " + jobs[job], flush=True)
                    except Exception:
                        failure = {"source": jobs[job], "stage": "source_worker", "trace": traceback.format_exc()}
                        write_json(output / "evidence/failures" / (jobs[job] + ".json"), failure)
                        failures.append(failure)
        monitor.activity = "aggregate_completed_evidence"
        success = summarize(output, sources, binding, failures)
        write_json(output / "evidence/invocation.json", {"seconds": time.perf_counter() - started,
                                                        "completed": success, **plan})
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "local/readiness-replay-r25")
    parser.add_argument("--threads", type=int, default=max(2, min(64, psutil.cpu_count(logical=False) or 2)))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--memory-gib", type=float)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 64 or (args.workers is not None and not 1 <= args.workers <= 8):
        parser.error("Use 1..64 CPU threads and 1..8 workers")
    if args.plan:
        print(f"{RELEASE}: 42 cached source blocks x five frozen learners; 399 prior rows + one hour; zero model forwards/fits")
        print("Full R25 local features/truth/scores/decisions required; summary ZIP alone is insufficient.")
        return
    if args.check:
        check_output = args.output / "local/check-only"
        checks(check_output)
        print((check_output / "evidence/check.log").read_text(encoding="utf-8"))
        return
    if args.previous is None:
        parser.error("--previous is the completed R25 output folder, containing protocol.json and local/")
    previous, output = distinct_roots(args.previous, args.output)
    own_windows_workers()
    with run_lock(output):
        success = False
        try:
            success = run(previous, output, threads=args.threads, workers=args.workers,
                          memory_gib=args.memory_gib, preflight_only=args.preflight_only)
        except Exception:
            write_json(output / "fatal.json", {"utc": utc_now(), "trace": traceback.format_exc()})
            raise
        finally:
            status = "preflight_complete" if success and args.preflight_only else "complete" if success else "incomplete"
            write_json(output / "last_invocation.json", {"status": status, "utc": utc_now()})
            pack(output, prefix="readiness-preflight" if args.preflight_only else
                 "readiness-replay-results" if success else "readiness-replay-incomplete")
        if not success:
            raise SystemExit(2)
