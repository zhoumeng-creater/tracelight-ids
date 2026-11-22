"""Synthetic process counters and mocked nvidia-smi; no real GPU or model calls."""
import csv
from types import SimpleNamespace
import subprocess
import threading

import pytest

from scripts.comprehensive_batch import resources


def test_gpu_sample_records_whole_device_fields_and_bounds_query(monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "synthetic-smi")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="42, 4096, 24564, 123.5\n")

    monkeypatch.setattr(resources.subprocess, "run", run)
    value = resources._gpu_sample()
    assert value == {"gpu0_util_percent": 42., "gpu0_device_used_MiB": 4096.,
                     "gpu0_device_total_MiB": 24564., "gpu0_power_watts": 123.5, "smi_status": "ok"}
    assert calls[0][1]["timeout"] == 2 and calls[0][1]["check"] is True


@pytest.mark.parametrize("raw", ["7, 100, 24564, N/A\n", "7, 100, 24564, [Not Supported]\n"])
def test_unsupported_power_retains_other_fields(monkeypatch, raw):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "synthetic-smi")
    monkeypatch.setattr(resources.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=raw))
    value = resources._gpu_sample()
    assert value["gpu0_util_percent"] == 7 and value["gpu0_power_watts"] == ""
    assert value["smi_status"].startswith("partial")


def test_gpu_timeout_is_a_record_not_a_task_failure(monkeypatch):
    monkeypatch.setattr(resources.shutil, "which", lambda name: "synthetic-smi")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("synthetic-smi", 2)

    monkeypatch.setattr(resources.subprocess, "run", timeout)
    assert resources._gpu_sample()["smi_status"].startswith("TimeoutExpired")


def test_cpu_deltas_use_process_identity_and_system_memory(monkeypatch, tmp_path):
    state = {"tick": 0}

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            # Child PID is reused; its accumulated time must not become a delta.
            return 1 if self.pid == 1 else state["tick"] + 10

        def cpu_times(self):
            return SimpleNamespace(user=(1 + state["tick"] * 2) if self.pid == 1 else 999, system=0)

        def memory_info(self):
            return SimpleNamespace(rss=100 * self.pid)

        def children(self, **kwargs):
            return [Process(2)]

    monkeypatch.setattr(resources.psutil, "Process", Process)
    monkeypatch.setattr(resources.psutil, "cpu_percent", lambda **kwargs: 50)
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(
        total=1000, available=700, used=300, percent=30))
    monkeypatch.setattr(resources, "_gpu_sample", lambda: {"smi_status": "synthetic"})
    monkeypatch.setattr(resources.time, "monotonic", lambda: 1 + 5 * state["tick"])
    monitor = resources.ResourceMonitor(tmp_path)
    monitor.pid = 1
    first = monitor._sample()
    assert "tree_observed_cpu_cores" not in first
    state["tick"] = 1
    second = monitor._sample()
    assert second["tree_observed_cpu_cores"] == pytest.approx(.4)
    assert second["tree_rss_sum_bytes"] == 300 and second["tree_processes"] == 2
    assert second["system_ram_available_bytes"] == 700


def test_background_phase_errors_and_shutdown_preserve_body_exception(monkeypatch, tmp_path):
    sampled = threading.Event()
    count = {"n": 0}

    def sample(self):
        count["n"] += 1
        if count["n"] == 1:
            self.set_phase("training")
            raise RuntimeError("synthetic missing counter")
        sampled.set()
        return {"utc": "synthetic", "phase": self._phase, "coordinator_pid": self.pid}

    monkeypatch.setattr(resources.ResourceMonitor, "_sample", sample)
    with pytest.raises(ValueError, match="body failure"):
        with resources.ResourceMonitor(tmp_path, phase="profile", interval=.01) as monitor:
            assert sampled.wait(2)
            raise ValueError("body failure")
    assert not monitor._thread.is_alive()
    with monitor.path.open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "synthetic missing counter" in rows[0]["monitor_errors"]
    assert rows[-1]["phase"] == "training"


def test_unwritable_log_does_not_fail_scientific_body(tmp_path):
    output = tmp_path / "file-not-directory"
    output.write_text("keep", encoding="utf8")
    with resources.ResourceMonitor(output) as monitor:
        pass
    assert monitor.errors and output.read_text(encoding="utf8") == "keep"
