"""Independent-process execution, preserving the registered scientific matrix."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from scripts.comprehensive_batch import run

TERMINAL = {"complete", "failed_numerical", "failed_job"}


def output_processes(output):
    """Find Python batch processes still using this exact output or a child directory."""
    root = Path(output).resolve()
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        arguments = process.info.get("cmdline") or []
        if not arguments or "python" not in Path(arguments[0]).name.lower():
            continue
        try:
            position = arguments.index("--output")
            target = Path(arguments[position + 1]).resolve()
            target.relative_to(root)
        except (ValueError, IndexError, OSError):
            continue
        found.append({"pid": process.pid, "command": arguments})
    return found


@contextmanager
def exclusive_output(output):
    """A coordinator owns the output until all its workers exit."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    live = output_processes(output)
    if live:
        raise RuntimeError(f"This output still has active workers: {[item['pid'] for item in live]}")
    path = output / ".execution.lock"
    stream = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def adopt_v1(output, source, plan, *, repo=run.REPO):
    """Explicitly adopt a stopped, exact v1 output, retaining its original receipt."""
    old_path = output / "source_receipt.json"
    original = old_path.read_bytes()
    old = json.loads(original)
    baseline_path = repo / run.DIRECTORY / "resource_v1_registration.json"
    if run.text_digest(baseline_path) != source["registration"]["legacy_registration_sha256_lf"]:
        raise ValueError("Archived v1 registration binding changed")
    baseline = run.read(baseline_path)
    expected = {"registration_sha256_lf": run.text_digest(repo / run.DIRECTORY / "resource_v1_registration.json"),
                "registration": baseline, "plan_sha256": run.data.json_digest(plan),
                "coverage_results": source["coverage_results"]}
    if old != expected or old["plan_sha256"] != source["plan_sha256"]:
        raise ValueError("Only the unchanged comprehensive-calibration-v1 output can be adopted")
    if output_processes(output):
        raise RuntimeError("Stop the old batch at a recovery point before adopting its output")
    for job in [*plan["hgb_jobs"], *plan["tabm_jobs"]]:
        directory = output / "fits" / job["job_id"]
        status_path = output / "job_statuses" / f"{job['job_id']}.json"
        status = run.read(status_path) if status_path.exists() else {}
        snapshots = status.get("snapshots", [])
        if not snapshots and (directory / "snapshots.json").exists():
            snapshots = run.read(directory / "snapshots.json")
        run._collect(directory, job, job["family"], snapshots, status.get("status"), output)
    backup = output / "source_receipt_v1.json"
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError("Existing v1 provenance differs")
    if not backup.exists():
        backup.write_bytes(original)
    run.write(output / "execution_adoption.json", {
        "from_release": "comprehensive-calibration-v1", "to_release": source["registration"]["release_tag"],
        "old_receipt_sha256": run.data.file_digest(backup), "plan_sha256": source["plan_sha256"],
        "scientific_job_definitions_changed": False, "completed_models_retrained": False,
        "recovery_fingerprint_checks_preserved": True,
        "old_job_times": "Earlier entries retain serial-v1 timing; new invocations label shared-resource execution."})
    run.write(old_path, source)


def _launch(output, job, *, resume, coverage_results, selection):
    logs = output / "worker_logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{job['job_id']}_{time.time_ns()}.log"
    stream = log.open("wb")
    command = [sys.executable, "-X", "utf8", "-u", str(run.REPO / "scripts/comprehensive_batch/worker.py"),
               "--output", str(output), "--job-id", job["job_id"],
               "--gpu-workers", str(selection["gpu_workers"]), "--cpu-workers", str(selection["cpu_workers"])]
    if resume:
        command.append("--resume")
    if coverage_results is not None:
        command += ["--coverage-results", str(coverage_results)]
    try:
        # Children must leave Ctrl+C handling to the coordinator, which requests
        # a stop at each job's next committed recovery boundary.
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=flags, start_new_session=os.name != "nt")
    except BaseException:
        stream.close()
        raise
    return {"process": process, "stream": stream, "job": job, "log": str(log)}


def _drain(active, stop_path, *, grace_seconds=180):
    stop_path.write_text("Stop after the next committed recovery checkpoint.\n", encoding="utf8")
    deadline = time.monotonic() + grace_seconds
    while active and time.monotonic() < deadline:
        for item in list(active):
            if item["process"].poll() is not None:
                item["stream"].close()
                active.remove(item)
        if active:
            time.sleep(.2)
    for item in active:
        item["process"].terminate()
    for item in active:
        try:
            item["process"].wait(timeout=10)
        except subprocess.TimeoutExpired:
            item["process"].kill()
            item["process"].wait()
        item["stream"].close()
    active.clear()


def train_parallel(output, plan, selection, *, resume=False, coverage_results=None,
                   launcher=None, poll_interval=.2):
    """Run independent CPU/GPU queues; freeze only after every job is terminal."""
    if (output / "calibration_started.json").exists():
        raise RuntimeError("Calibration has started: no more training")
    launcher = launcher or _launch
    queues = {"hgb": [], "tabm": []}
    limits = {"hgb": selection["cpu_workers"], "tabm": selection["gpu_workers"]}
    if any(not isinstance(value, int) or value < 1 for value in limits.values()):
        raise ValueError("Worker counts must be positive integers")
    for family, jobs in (("hgb", plan["hgb_jobs"]), ("tabm", plan["tabm_jobs"])):
        for job in jobs:
            path = output / "job_statuses" / f"{job['job_id']}.json"
            cached = run.read(path) if path.exists() else None
            if cached and not resume:
                raise FileExistsError("Use --resume for this existing batch")
            if cached and cached["status"] in TERMINAL:
                run._collect(output / "fits" / job["job_id"], job, family,
                             cached.get("snapshots", []), cached["status"], output)
            else:
                queues[family].append(job)
    stop_path = output / "execution_stop.request"
    if stop_path.exists():
        if not resume:
            raise RuntimeError("An interrupted batch requires --resume")
        stop_path.unlink()
    run.write(output / "execution_schedule.json", {
        "gpu_workers": limits["tabm"], "cpu_workers": limits["hgb"], "threads_per_worker": 4,
        "pending_jobs": {family: [job["job_id"] for job in jobs] for family, jobs in queues.items()},
        "timing_scope": "Job wall times include concurrent CPU/GPU contention; isolated CPU inference measurements run later.",
        "scientific_matrix_changed": False})
    active, completions = [], []
    started = time.perf_counter()
    try:
        while active or any(queues.values()):
            if stop_path.exists():
                raise InterruptedError("A worker or operator requested a checkpoint stop")
            for family in ("tabm", "hgb"):
                available = limits[family] - sum(item["job"]["family"] == family for item in active)
                for _ in range(min(available, len(queues[family]))):
                    job = queues[family].pop(0)
                    active.append(launcher(output, job, resume=resume, coverage_results=coverage_results,
                                           selection=selection))
                    print(json.dumps({"phase": "parallel_fit", "started": job["job_id"],
                                      "active_jobs": len(active)}, ensure_ascii=False), flush=True)
            for item in list(active):
                code = item["process"].poll()
                if code is None:
                    continue
                item["stream"].close()
                active.remove(item)
                job = item["job"]
                path = output / "job_statuses" / f"{job['job_id']}.json"
                state = run.read(path) if path.exists() else {}
                completions.append({"job_id": job["job_id"], "exit_code": code,
                                    "status": state.get("status"), "log": item["log"]})
                run.write(output / "execution_completions.json", completions)
                if code != 0 or state.get("status") not in TERMINAL:
                    raise RuntimeError(f"Worker interrupted: {job['job_id']}; see {item['log']}")
                print(json.dumps({"phase": "parallel_fit", "finished": job["job_id"],
                                  "status": state["status"]}), flush=True)
            if active:
                time.sleep(poll_interval)
    except BaseException:
        _drain(active, stop_path)
        raise
    finally:
        run.write(output / "execution_wall.json", {"invocation_wall_seconds": time.perf_counter() - started,
                   "finished_this_invocation": len(completions), "parallelism": limits,
                   "individual_job_times_in": "job_invocations"})
    return run.finalize_training(output, plan)
