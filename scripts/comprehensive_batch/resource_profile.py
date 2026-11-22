"""Choose execution concurrency from synchronized, generated-data GPU work.

This startup profile never opens F/C/V or changes a model/training parameter.
Separate spawn processes execute the fixed TabM+PLE S objective on one GPU.
Initialization and warmup are excluded from aggregate steady-step throughput.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

FORMAT = "comprehensive-generated-resource-profile-v1"
THREADS = 4
SYSTEM_CORES = 4
HEADROOM = .20
MINIMUM_GAIN = .10
WARMUP_STEPS = 3
TIMED_STEPS = 12
POLL_SECONDS = .5
PROFILE_TIMEOUT_SECONDS = 1800
GROUP_SIZES = (125, 314, 4388, 1645, 884, 27, 263, 20, 5081)
NORMAL_ROWS = 12747
POSITIVE_ROWS = 505
ROWS = NORMAL_ROWS + POSITIVE_ROWS
CARDINALITIES = (6, 2, 2, 2, 2, 2, 2, 2)
SOURCE_PATHS = (
    "scripts/comprehensive_batch/resource_profile.py", "scripts/comprehensive_batch/training.py",
    "scripts/comprehensive_batch/objective.py", "scripts/comprehensive_batch/matrix.py",
    "scripts/calibration_repair/training.py", "scripts/calibration_repair/objective.py",
    "scripts/calibration_transfer/objective.py", "scripts/normal_tail_mechanism.py",
    "src/ids/recent_baselines.py", "src/ids/route_batch/tabm_training.py",
    "src/ids/flow_history/model.py", "src/ids/model_training/preprocessing.py",
)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf8", newline="\n")
    os.replace(temporary, path)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def implementation_identity():
    """Source and installed package versions; never any research-data digest."""
    files = {name: hashlib.sha256((REPO / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
             for name in SOURCE_PATHS}
    versions = {}
    for name in ("torch", "tabm", "rtdl-num-embeddings", "numpy", "psutil", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_sha256_lf": files, "versions": versions,
            "python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "profile_settings": {"warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS,
                                 "threads": THREADS, "headroom": HEADROOM, "gain": MINIMUM_GAIN}}


def _gpu_sample():
    binary = shutil.which("nvidia-smi")
    if binary is None:
        fallback = Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/nvidia-smi.exe"
        binary = str(fallback) if fallback.is_file() else "nvidia-smi"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    identity = visible or "0"
    command = [binary, "-i", identity,
               "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("The profile requires one identified CUDA device")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError("Unrecognized nvidia-smi resource response")
    return {"index": int(fields[0]), "uuid": fields[1], "name": fields[2],
            "total_bytes": int(float(fields[3]) * 1024 ** 2),
            "used_bytes": int(float(fields[4]) * 1024 ** 2),
            "utilization_percent": float(fields[5]), "driver": fields[6]}


def hardware_info():
    memory = psutil.virtual_memory()
    return {"hostname": platform.node(), "platform": platform.platform(), "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1,
            "ram_total_bytes": int(memory.total), "ram_available_bytes": int(memory.available),
            "gpu": _gpu_sample(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def _machine_identity(hardware):
    result = {key: value for key, value in hardware.items() if key not in {"ram_available_bytes", "gpu"}}
    result["gpu"] = {key: value for key, value in hardware["gpu"].items()
                     if key not in {"used_bytes", "utilization_percent"}}
    return result


def synthetic_inputs(seed=20261122):
    """Fixed shape/counts, new generated values and anonymous memberships only."""
    rng = np.random.default_rng(seed)
    numeric = np.ascontiguousarray(rng.normal(size=(ROWS, 5)), dtype=np.float32)
    categorical = np.ascontiguousarray(np.column_stack(
        [rng.integers(0, width, ROWS) for width in CARDINALITIES]), dtype=np.int64)
    labels = np.r_[np.zeros(NORMAL_ROWS), np.ones(POSITIVE_ROWS)].astype(np.float32)
    groups = {"references": [{"id": "generated_source_0", "indices": list(range(439))},
                              {"id": "generated_source_1", "indices": list(range(439, NORMAL_ROWS))}],
              "targets": [], "attacks": list(range(NORMAL_ROWS, ROWS))}
    start = 0
    for number, count in enumerate(GROUP_SIZES):
        groups["targets"].append({"id": f"generated_business_{number}", "indices": list(range(start, start + count))})
        start += count
    return numeric, categorical, labels, groups


def _optimizer_step(network, optimizer, numeric, categorical, resident, x, cats, target,
                    sampler, groups, device):
    """Same BCE + resident exact S replay + clipping + AdamW as production."""
    import torch
    from ids.recent_baselines import tabm_head_loss
    from scripts.comprehensive_batch.objective import backward_objective

    network.train()
    indices = torch.from_numpy(sampler.next()).to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = network(x[indices], cats[indices])
    if logits.dtype != torch.float32:
        raise ValueError("Expected the fixed FP32 member logits")
    loss = tabm_head_loss(logits, target[indices])
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite generated member BCE")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    auxiliary = backward_objective(network, numeric, categorical, groups, arm="S_alignment_group",
                                   device=device, batch_size=1024, weight=1., resident=resident)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"member_BCE": float(loss.detach()), "auxiliary_penalty": auxiliary["penalty"],
            "preclip_gradient_norm": float(norm), "auxiliary_cost": auxiliary["cost"]}


def _worker_should_stop(stop):
    parent = multiprocessing.parent_process()
    return stop.is_set() or (parent is not None and not parent.is_alive())


def _wait_gate(gate, stop):
    """An orphaned spawn worker must not hold its CUDA context at a barrier."""
    while not _worker_should_stop(stop):
        if gate.wait(timeout=.25):
            return not _worker_should_stop(stop)
    return False


def _worker(rank, warmup_gate, timed_gate, stop, messages):
    """Spawn entrypoint. No production fit/data loader or model file is used."""
    process_started = time.perf_counter()
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(THREADS)
        import torch
        from threadpoolctl import threadpool_limits
        from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network
        from scripts.calibration_repair.training import ClassSampler, FIXED_MODEL, _seed, scoring_arithmetic
        from scripts.comprehensive_batch.objective import resident_inputs

        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(1)
        device = torch.device("cuda:0")
        with threadpool_limits(limits=THREADS), scoring_arithmetic():
            numeric, categorical, labels, groups = synthetic_inputs()
            parameters = _parameters(FIXED_MODEL)
            bins, _ = _numeric_bins(numeric, parameters)
            _seed(17 + rank, device)
            network = make_tabm_network(5, list(CARDINALITIES), parameters, numeric_bins=bins).to(device)
            x, cats, target = (torch.from_numpy(value).to(device) for value in (numeric, categorical, labels))
            resident = resident_inputs(numeric, categorical, device=device, numeric_tensor=x, categorical_tensor=cats)
            optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=.0003)
            sampler = ClassSampler(labels, seed=17 + rank, per_class=128)
            torch.cuda.synchronize(device)
            initialized = time.perf_counter()
            if _worker_should_stop(stop):
                return
            messages.put({"stage": "ready", "rank": rank, "initialization_seconds": initialized - process_started})
            if not _wait_gate(warmup_gate, stop):
                return
            started = time.perf_counter()
            for _ in range(WARMUP_STEPS):
                _optimizer_step(network, optimizer, numeric, categorical, resident, x, cats, target,
                                sampler, groups, device)
                if _worker_should_stop(stop):
                    return
            warmup_seconds = time.perf_counter() - started
            messages.put({"stage": "warmed", "rank": rank, "warmup_seconds": warmup_seconds})
            if not _wait_gate(timed_gate, stop):
                return
            torch.cuda.reset_peak_memory_stats(device)
            cpu_start = psutil.Process().cpu_times()
            started = time.perf_counter()
            steps = []
            for _ in range(TIMED_STEPS):
                at = time.perf_counter()
                detail = _optimizer_step(network, optimizer, numeric, categorical, resident, x, cats, target,
                                         sampler, groups, device)
                steps.append({"seconds": time.perf_counter() - at, **detail})
                if _worker_should_stop(stop):
                    return
            finished = time.perf_counter()
            cpu_end = psutil.Process().cpu_times()
            messages.put({"stage": "done", "rank": rank, "initialization_seconds": initialized - process_started,
                          "warmup_seconds": warmup_seconds, "timed_start": started, "timed_finish": finished,
                          "timed_steps": TIMED_STEPS, "timed_seconds": finished - started,
                          "cpu_seconds": cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system,
                          "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                          "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                          "final_rss_bytes": psutil.Process().memory_info().rss, "steps": steps})
    except BaseException as error:
        messages.put({"stage": "error", "rank": rank, "type": type(error).__name__, "error": str(error),
                      "traceback": traceback.format_exc()})
    finally:
        parent = multiprocessing.parent_process()
        if parent is not None and not parent.is_alive():
            # No coordinator remains to drain the pipe. Do not keep this process
            # (and its CUDA context) alive waiting for the Queue feeder to flush.
            messages.cancel_join_thread()


def _sample(processes):
    memory = psutil.virtual_memory()
    rss = 0
    cpu_seconds = 0.
    for child in processes:
        try:
            process = psutil.Process(child.pid)
            rss += process.memory_info().rss
            cpu = process.cpu_times()
            cpu_seconds += cpu.user + cpu.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    gpu = _gpu_sample()
    return {"at": time.perf_counter(), "ram_available_bytes": int(memory.available),
            "process_rss_bytes": rss, "process_cpu_seconds": cpu_seconds,
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "gpu_used_bytes": gpu["used_bytes"], "gpu_utilization_percent": gpu["utilization_percent"]}


def _unsafe(sample, hardware):
    reasons = []
    if sample["gpu_used_bytes"] > (1 - HEADROOM) * hardware["gpu"]["total_bytes"]:
        reasons.append("gpu_headroom_below_20_percent")
    if sample["ram_available_bytes"] < HEADROOM * hardware["ram_total_bytes"]:
        reasons.append("ram_headroom_below_20_percent")
    return reasons


def _run_candidate(workers, hardware, directory):
    """Synchronized phases and whole-device/whole-host memory observations."""
    context = multiprocessing.get_context("spawn")
    warmup_gate, timed_gate, stop = context.Event(), context.Event(), context.Event()
    messages = context.Queue()
    children = [context.Process(target=_worker, args=(rank, warmup_gate, timed_gate, stop, messages))
                for rank in range(workers)]
    ready, warmed, finished, failures, samples = {}, {}, {}, [], []
    begin = time.perf_counter()
    common_start = None
    try:
        for child in children:
            child.start()
        next_sample = begin
        while len(finished) < workers:
            now = time.perf_counter()
            if now - begin > PROFILE_TIMEOUT_SECONDS:
                failures.append({"error": "fixed_startup_profile_timeout", "seconds": PROFILE_TIMEOUT_SECONDS})
                break
            if now >= next_sample:
                sample = _sample(children)
                samples.append(sample)
                reasons = _unsafe(sample, hardware)
                if reasons:
                    failures.append({"error": "memory_headroom", "reasons": reasons})
                    break
                next_sample = time.perf_counter() + POLL_SECONDS
            try:
                message = messages.get(timeout=.1)
            except queue.Empty:
                # A process killed externally may never put an error in Queue.
                unexpected = [child.exitcode for rank, child in enumerate(children)
                              if child.exitcode is not None and rank not in finished]
                if unexpected:
                    failures.append({"error": "worker_exited_without_result", "exitcodes": unexpected})
                    break
                continue
            stage, rank = message["stage"], message["rank"]
            if stage == "error":
                failures.append(message)
                break
            if stage == "ready":
                ready[rank] = message
                if len(ready) == workers:
                    warmup_gate.set()
            elif stage == "warmed":
                warmed[rank] = message
                if len(warmed) == workers:
                    common_start = time.perf_counter()
                    timed_gate.set()
            elif stage == "done":
                finished[rank] = message
    except Exception as error:
        failures.append({"error": str(error), "type": type(error).__name__, "traceback": traceback.format_exc()})
    finally:
        stop.set()
        warmup_gate.set()
        timed_gate.set()
        for child in children:
            if child.pid is None:
                continue
            child.join(timeout=1.)
            if child.is_alive():
                child.terminate()
                child.join(timeout=5.)
        messages.close()
    observed_gpu = max((s["gpu_used_bytes"] for s in samples), default=hardware["gpu"]["used_bytes"])
    # Sampling can miss a short allocation peak. Per-process allocator maxima
    # add a conservative bound; the global samples also include CUDA contexts
    # and other applications, which the per-process allocator cannot see.
    reserved_bound = hardware["gpu"]["used_bytes"] + sum(
        v["peak_cuda_reserved_bytes"] for v in finished.values())
    peak_gpu = max(observed_gpu, reserved_bound)
    if peak_gpu > (1 - HEADROOM) * hardware["gpu"]["total_bytes"]:
        failures.append({"error": "allocator_peak_headroom", "peak_gpu_bound_bytes": peak_gpu})
    complete = not failures and len(finished) == workers and common_start is not None
    elapsed = max(value["timed_finish"] for value in finished.values()) - common_start if complete else None
    rate = sum(value["timed_steps"] for value in finished.values()) / elapsed if complete and elapsed > 0 else None
    result = {"workers": workers, "status": "complete" if complete else "failed", "safe": complete,
              "aggregate_steps_per_second": rate, "common_timed_seconds": elapsed,
              "initialization_and_warmup_excluded": True, "profile_wall_seconds": time.perf_counter() - begin,
              "ready": ready, "warmup": warmed, "worker_results": finished, "errors": failures,
              "resource_samples": samples,
              "peak_gpu_used_bytes": peak_gpu,
              "observed_peak_gpu_used_bytes": observed_gpu, "allocator_reserved_bound_bytes": reserved_bound,
              "minimum_ram_available_bytes": min((s["ram_available_bytes"] for s in samples), default=None),
              "peak_process_rss_bytes": max((s["process_rss_bytes"] for s in samples), default=None)}
    if complete:
        result["worker_start_skew_seconds"] = max(v["timed_start"] for v in finished.values()) - min(v["timed_start"] for v in finished.values())
        result["mean_cpu_cores_during_timed_work"] = sum(v["cpu_seconds"] for v in finished.values()) / elapsed
    _write(Path(directory) / "profile.json", result)
    return result


def _choose(profiles, hardware):
    selected = None
    for profile in sorted(profiles, key=lambda p: p["workers"]):
        rate = profile.get("aggregate_steps_per_second")
        if profile["status"] != "complete" or not profile.get("safe") or rate is None or not np.isfinite(rate) or rate <= 0:
            continue
        if selected is None or rate >= selected["aggregate_steps_per_second"] * (1 + MINIMUM_GAIN):
            selected = profile
    if selected is None:
        raise RuntimeError("No safely completed generated-data concurrency profile")
    available_cores = hardware["physical_cores"] - SYSTEM_CORES - selected["workers"] * THREADS
    cpu_workers = min(2, max(0, available_cores // THREADS))
    return selected["workers"], cpu_workers


def _started(output):
    if any((output / name).exists() for name in ("job_statuses.json", "models_frozen.json", "calibration_started.json")):
        return True
    fits = output / "fits"
    return fits.exists() and any(fits.iterdir())


def select_execution(output, allow_started_without_profile=False):
    """Select once per machine/source binding, or reuse its completed profile.

    Only the runner may authorize profiling an already-started legacy v1 batch,
    after it has verified the old source receipt and that no workers remain.
    A changed binding on an existing profile never triggers silent reprofiling.
    """
    output = Path(output)
    hardware = hardware_info()
    implementation = implementation_identity()
    binding = _hash({"machine": _machine_identity(hardware), "implementation": implementation})
    cache = output / "execution_profile.json"
    if cache.is_file():
        value = json.loads(cache.read_text(encoding="utf8"))
        if value.get("format") != FORMAT or value.get("binding_sha256") != binding or value.get("status") != "complete":
            raise ValueError("Existing execution profile belongs to another machine/source/environment; no automatic reprofile")
        return {**value, "cache_reused": True, "current_hardware": hardware}
    if _started(output) and not allow_started_without_profile:
        raise ValueError("Started batch has no execution profile; legacy adoption must be verified by the runner")
    root = output / "resource_profile"
    root.mkdir(parents=True, exist_ok=True)
    initial = {"gpu_used_bytes": hardware["gpu"]["used_bytes"], "ram_available_bytes": hardware["ram_available_bytes"]}
    if _unsafe(initial, hardware):
        raise RuntimeError("Insufficient initial GPU/RAM headroom for generated-data profiling")
    profiles = []
    for workers in (1, 2, 4):
        # A separate four-thread HGB slot and four physical system cores remain.
        if (workers + 1) * THREADS + SYSTEM_CORES > hardware["physical_cores"]:
            profile = {"workers": workers, "status": "skipped", "safe": False,
                       "aggregate_steps_per_second": None, "reason": "physical_cpu_budget_with_HGB_and_system_reserve"}
            profiles.append(profile)
            _write(root / f"workers_{workers}" / "profile.json", profile)
            continue
        memory_failure = any(p["status"] == "failed" and any(
            error.get("error") in {"memory_headroom", "allocator_peak_headroom"}
            or "out of memory" in error.get("error", "").lower()
            for error in p.get("errors", [])) for p in profiles)
        if memory_failure:
            profile = {"workers": workers, "status": "skipped", "safe": False,
                       "aggregate_steps_per_second": None, "reason": "smaller_concurrency_exceeded_memory_reserve"}
            profiles.append(profile)
            _write(root / f"workers_{workers}" / "profile.json", profile)
            continue
        previous = [p for p in profiles if p["status"] == "complete" and p.get("safe")]
        if previous:
            last = previous[-1]
            per_worker = max(0, last["peak_gpu_used_bytes"] - hardware["gpu"]["used_bytes"]) / last["workers"]
            estimated_gpu = hardware["gpu"]["used_bytes"] + workers * per_worker
            per_host_worker = last["peak_process_rss_bytes"] / last["workers"]
            estimated_available = hardware["ram_available_bytes"] - workers * per_host_worker
            if estimated_gpu > (1 - HEADROOM) * hardware["gpu"]["total_bytes"] or estimated_available < HEADROOM * hardware["ram_total_bytes"]:
                profile = {"workers": workers, "status": "skipped", "safe": False,
                           "aggregate_steps_per_second": None, "reason": "previous_profile_memory_projection",
                           "estimated_gpu_used_bytes": estimated_gpu, "estimated_ram_available_bytes": estimated_available}
                profiles.append(profile)
                _write(root / f"workers_{workers}" / "profile.json", profile)
                continue
        print(json.dumps({"phase": "generated_resource_profile", "workers": workers,
                          "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS}), flush=True)
        profiles.append(_run_candidate(workers, hardware, root / f"workers_{workers}"))
        _write(root / "candidates.json", profiles)
    try:
        gpu_workers, cpu_workers = _choose(profiles, hardware)
    except RuntimeError:
        _write(root / "failed.json", {"hardware": hardware, "implementation": implementation, "profiles": profiles})
        raise
    value = {"format": FORMAT, "status": "complete", "binding_sha256": binding,
             "created_at": datetime.now(timezone.utc).isoformat(), "gpu_workers": gpu_workers,
             "cpu_workers": cpu_workers, "threads_per_worker": THREADS, "hardware": hardware,
             "implementation": implementation, "profiles": profiles, "cache_reused": False,
             "legacy_started_batch_adoption": bool(allow_started_without_profile),
             "selection": "Ascending safe candidates: increase concurrency only for >=10% aggregate steady-step gain; never by GPU utilization.",
             "memory_headroom_fraction": HEADROOM, "reserved_physical_cores": SYSTEM_CORES,
             "generated_only": True, "real_rows_read": 0, "arm": "S_alignment_group",
             "rows_per_worker": ROWS, "normal_rows": NORMAL_ROWS, "positive_rows": POSITIVE_ROWS,
             "numeric_features": 5, "categorical_cardinalities": list(CARDINALITIES),
             "training_batch_size": 256, "score_batch_size": 1024,
             "includes_checkpoint_diagnostics_or_disk_IO": False,
             "interpretation": "Short startup execution profile; not experimental evidence, real-data convergence, or an end-to-end batch-time prediction."}
    _write(cache, value)
    return value
