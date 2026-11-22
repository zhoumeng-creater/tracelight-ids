"""Low-frequency coordinator-tree and whole-device observations; never model selection."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import psutil


FIELDS = (
    "utc", "phase", "coordinator_pid", "sample_seconds", "system_cpu_percent",
    "tree_observed_cpu_cores", "tree_processes", "tree_rss_sum_bytes",
    "system_ram_total_bytes", "system_ram_available_bytes", "system_ram_used_bytes",
    "system_ram_percent", "gpu0_util_percent", "gpu0_device_used_MiB",
    "gpu0_device_total_MiB", "gpu0_power_watts", "smi_status", "monitor_errors",
)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _error(error):
    return f"{type(error).__name__}: {error}"


def _gpu_sample():
    """nvidia-smi reports the whole registered device, not this task's allocation."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"smi_status": "unavailable: nvidia-smi not found"}
    try:
        value = subprocess.run(
            [executable, "-i", "0", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        fields = next(csv.reader(value.stdout.strip().splitlines()))
        if len(fields) != 4:
            raise ValueError("Expected utilization, used/total memory and power for GPU 0")
        names = ("gpu0_util_percent", "gpu0_device_used_MiB", "gpu0_device_total_MiB", "gpu0_power_watts")
        result = {}
        for name, field in zip(names, fields):
            try:
                number = float(field.strip())
                result[name] = number if math.isfinite(number) else ""
            except ValueError:
                result[name] = ""
        result["smi_status"] = "ok" if all(v != "" for v in result.values()) else "partial: unsupported field"
        return result
    except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as error:
        return {"smi_status": _error(error)}


class ResourceMonitor:
    """Write one CSV per invocation, sampling every five seconds by default.

    CPU cores use CPU-time deltas for matching (PID, creation time) identities.
    New/exited short-lived children can be missed. RSS sums may double-count
    shared pages. Device utilization/memory/power include unrelated processes.
    Sampling errors are fields in the CSV; an unwritable log disables monitoring
    and records the error on stderr and ``errors`` without failing scientific work.
    The context must end before isolated performance measurements begin.
    """

    def __init__(self, output, *, phase="startup", interval=5.0):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Resource sampling interval must be positive")
        self.interval = float(interval)
        self.pid = os.getpid()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.path = Path(output) / f"resources-{stamp}-{self.pid}.csv"
        self.errors = []
        self._phase = str(phase)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._previous = {}
        self._last = None

    def set_phase(self, phase):
        with self._lock:
            self._phase = str(phase)

    def _sample(self):
        with self._lock:
            phase = self._phase
        started = time.monotonic()
        row = {"utc": _utc(), "phase": phase, "coordinator_pid": self.pid}
        errors = []
        current, rss = {}, 0
        try:
            parent = psutil.Process(self.pid)
            processes = [parent]
            try:
                processes += parent.children(recursive=True)
            except psutil.Error as error:
                errors.append(_error(error))
            for process in processes:
                try:
                    identity = (process.pid, process.create_time())
                    usage = process.cpu_times()
                    memory = process.memory_info().rss
                    current[identity] = usage.user + usage.system
                    rss += memory
                except psutil.Error as error:
                    errors.append(_error(error))
            if self._last is not None and started > self._last:
                row["tree_observed_cpu_cores"] = sum(
                    max(0., current[key] - self._previous[key])
                    for key in current.keys() & self._previous.keys()
                ) / (started - self._last)
            row.update(tree_processes=len(current), tree_rss_sum_bytes=rss)
        except Exception as error:
            errors.append(_error(error))
        self._previous, self._last = current, started
        try:
            memory = psutil.virtual_memory()
            row.update(system_cpu_percent=psutil.cpu_percent(interval=None),
                       system_ram_total_bytes=memory.total, system_ram_available_bytes=memory.available,
                       system_ram_used_bytes=memory.used, system_ram_percent=memory.percent)
        except Exception as error:
            errors.append(_error(error))
        try:
            row.update(_gpu_sample())
        except Exception as error:
            row["smi_status"] = _error(error)
        row["sample_seconds"] = time.monotonic() - started
        row["monitor_errors"] = "; ".join(errors)
        return row

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", newline="", encoding="utf8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                while not self._stop.is_set():
                    started = time.monotonic()
                    try:
                        row = self._sample()
                    except Exception as error:
                        with self._lock:
                            phase = self._phase
                        row = {"utc": _utc(), "phase": phase, "coordinator_pid": self.pid,
                               "monitor_errors": _error(error)}
                    writer.writerow(row)
                    handle.flush()
                    self._stop.wait(max(0., self.interval - (time.monotonic() - started)))
        except Exception as error:
            message = f"Resource monitor stopped at {_utc()}: {_error(error)}"
            self.errors.append(message)
            try:
                print(message, file=sys.stderr, flush=True)
            except Exception:
                pass

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, name="comprehensive-resource-monitor", daemon=True)
        try:
            self._thread.start()
        except Exception as error:
            self.errors.append(_error(error))
            self._thread = None
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return False
