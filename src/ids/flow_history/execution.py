"""Single-owner execution, verifiable stage receipts and lightweight resource sampling."""
import csv
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import threading
import time
from zipfile import ZIP_DEFLATED, ZipFile

import psutil

from ids.common import read_json, sha256_file, utc_now, write_json


def own_windows_workers():
    """Kill worker descendants on coordinator exit, including abrupt session termination."""
    if os.name != "nt":
        return
    import ctypes as ct
    from ctypes import wintypes as wt

    size = ct.c_size_t

    class BasicLimits(ct.Structure):
        _fields_ = [("process_time", ct.c_int64), ("job_time", ct.c_int64), ("flags", wt.DWORD),
                    ("minimum_ws", size), ("maximum_ws", size), ("active_processes", wt.DWORD),
                    ("affinity", size), ("priority", wt.DWORD), ("scheduling", wt.DWORD)]

    class ExtendedLimits(ct.Structure):
        _fields_ = [("basic", BasicLimits), ("io_counters", ct.c_uint64 * 6),
                    ("process_memory", size), ("job_memory", size), ("process_peak", size), ("job_peak", size)]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ct.c_void_p, wt.LPCWSTR]
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    kernel.GetCurrentProcess.restype = wt.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not job or not kernel.SetInformationJobObject(job, 9, ct.byref(limits), ct.sizeof(limits))
            or not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess())):
        raise ct.WinError(ct.get_last_error())
    # This non-inheritable handle belongs to the coordinator process until OS teardown.


@contextmanager
def run_lock(root):
    """The OS releases this lock even if the owning Python process is killed."""
    root.mkdir(parents=True, exist_ok=True)
    handle = (root / "run.lock").open("a+b")
    try:
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another process owns the experiment lock at {root}") from exc
    try:
        yield
    finally:
        handle.close()


def stage(root, name, binding, execute):
    receipt = root / "receipts" / (name + ".json")
    if receipt.exists():
        done = read_json(receipt)
        if binding != done["binding"] or any(
            not (root / key).is_file() or sha256_file(root / key) != value
            for key, value in done["outputs"].items()
        ):
            raise ValueError(f"Stage identity or completed outputs changed: {name}")
        print(f"[reuse] {name}", flush=True)
        return
    print(f"[run] {name}", flush=True)
    started = utc_now()
    then = time.perf_counter()
    files = list(execute())
    if not files:
        raise RuntimeError(f"Stage {name} produced no evidence")
    outputs = {Path(p).relative_to(root).as_posix(): sha256_file(p) for p in files}
    write_json(receipt, {"stage": name, "binding": binding, "started": started, "ended": utc_now(),
                         "seconds": time.perf_counter() - then, "outputs": outputs})


class ResourceLog:
    def __init__(self, root):
        self.activity = "startup"
        self.stop = threading.Event()
        self.path = root / ("resources-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv")
        self.worker = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        process = psutil.Process()
        smi = shutil.which("nvidia-smi")
        previous = {}
        last = time.monotonic()
        with self.path.open("w", newline="", encoding="utf-8") as file:
            rows = csv.writer(file)
            rows.writerow(["utc", "stage", "system_cpu_percent", "tree_observed_cpu_cores", "tree_processes",
                           "tree_rss_sum_bytes", "gpu0_util_percent", "gpu0_device_used_MiB", "smi_status"])
            while not self.stop.is_set():
                now = time.monotonic()
                current = {}
                rss = 0
                for proc in [process, *process.children(recursive=True)]:
                    try:
                        times = proc.cpu_times()
                        current[(proc.pid, proc.create_time())] = times.user + times.system
                        rss += proc.memory_info().rss
                    except psutil.Error:
                        continue
                cores = sum(current[k] - previous[k] for k in current.keys() & previous.keys()) / max(now-last, 1e-6)
                gpu, memory, status = "", "", "unavailable"
                if smi:
                    try:
                        raw = subprocess.check_output(
                            [smi, "-i", "0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                            text=True, timeout=4, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        gpu, memory = [int(x.strip()) for x in raw.strip().split(",")]
                        status = "ok"
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        status = type(exc).__name__
                rows.writerow([utc_now(), self.activity, psutil.cpu_percent(), round(cores, 3) if previous else "",
                               len(current), rss, gpu, memory, status])
                file.flush()
                previous, last = current, now
                self.stop.wait(2)

    def __enter__(self):
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.worker.join()


def pack(root, *, prefix="flow-history-results"):
    """Return the new small detectors and their evidence; leave the capture and raw IP fixtures in the lab."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"{prefix}-{stamp}.zip"
    files = list(root.glob("*.json")) + list(root.glob("resources-*.csv"))
    for folder in ("receipts", "evidence"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file() and not p.name.endswith(".tmp")
                     and not any(part.startswith("inputs-") for part in p.relative_to(root).parts))
    files.extend(p for p in (root / "models").glob("*/*") if p.suffix in {".pt", ".joblib"})
    with ZipFile(output.with_suffix(".tmp"), "w", compression=ZIP_DEFLATED, compresslevel=4) as archive:
        manifest = {}
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            archive.write(path, name)
            manifest[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        archive.writestr("archive_manifest.json", json.dumps(manifest, indent=2))
    output.with_suffix(".tmp").replace(output)
    write_json(Path(str(output) + ".sha256.json"), {"file": output.name, "sha256": sha256_file(output),
                                                    "bytes": output.stat().st_size})
    print(f"[return] {output}", flush=True)
    return output
