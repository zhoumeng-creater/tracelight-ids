"""Mocks and tiny generated arrays only; never start a lab/GPU profile."""
from copy import deepcopy
import queue
import time

import numpy as np
import pytest

from scripts.comprehensive_batch import resource_profile as profile

GIB = 1024 ** 3


def hardware():
    return {"hostname": "generated-host", "platform": "synthetic", "processor": "synthetic",
            "physical_cores": 24, "logical_cores": 32, "ram_total_bytes": 64 * GIB,
            "ram_available_bytes": 60 * GIB, "gpu": {"index": 0, "uuid": "generated-gpu",
            "name": "generated", "total_bytes": 24 * GIB, "used_bytes": GIB,
            "utilization_percent": 0., "driver": "synthetic"}, "cuda_visible_devices": None}


def candidate(workers, rate, *, safe=True, status="complete", gpu=None):
    return {"workers": workers, "aggregate_steps_per_second": rate, "status": status, "safe": safe,
            "peak_gpu_used_bytes": gpu if gpu is not None else (1 + 2 * workers) * GIB,
            "peak_process_rss_bytes": workers * GIB, "minimum_ram_available_bytes": (60 - workers) * GIB}


def fake_environment(monkeypatch, rates=(10., 18., 28.)):
    machine = hardware()
    monkeypatch.setattr(profile, "hardware_info", lambda: deepcopy(machine))
    monkeypatch.setattr(profile, "implementation_identity", lambda: {"generated_source_hash": "one"})
    calls = []

    def run(workers, info, directory):
        calls.append(workers)
        return candidate(workers, rates[(1, 2, 4).index(workers)])

    monkeypatch.setattr(profile, "_run_candidate", run)
    return machine, calls


def test_generated_dimensions_labels_partition_and_reproducibility():
    numeric, categorical, labels, groups = profile.synthetic_inputs()
    assert numeric.shape == (13252, 5) and numeric.dtype == np.float32
    assert categorical.shape == (13252, 8) and categorical.dtype == np.int64
    assert np.count_nonzero(labels == 0) == 12747
    assert np.count_nonzero(labels == 1) == 505
    assert [len(g["indices"]) for g in groups["references"]] == [439, 12308]
    assert [len(g["indices"]) for g in groups["targets"]] == list(profile.GROUP_SIZES)
    assert sorted(i for g in groups["targets"] for i in g["indices"]) == list(range(12747))
    assert set(groups["attacks"]) == set(range(12747, 13252))
    assert all(categorical[:, i].min() >= 0 and categorical[:, i].max() < width
               for i, width in enumerate(profile.CARDINALITIES))
    for left, right in zip((numeric, categorical, labels), profile.synthetic_inputs()[:3]):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("rates,expected", [((10., 10.9, 10.8), 1), ((10., 11., 11.9), 2),
                                           ((10., 18., 28.), 4), ((10., 9., 14.), 4)])
def test_selects_aggregate_throughput_with_ten_percent_increment(rates, expected):
    values = [candidate(workers, rate) for workers, rate in zip((1, 2, 4), rates)]
    gpu, cpu = profile._choose(values, hardware())
    assert gpu == expected
    assert cpu == (1 if expected == 4 else 2)
    assert (gpu + cpu) * 4 <= 24 - 4


def test_unsafe_failure_and_busy_gpu_percentage_cannot_win():
    values = [candidate(1, 10.), candidate(2, 100., safe=False), candidate(4, 200., status="failed")]
    values[0]["gpu_utilization_percent"] = 15.
    values[1]["gpu_utilization_percent"] = 100.
    assert profile._choose(values, hardware()) == (1, 2)
    with pytest.raises(RuntimeError, match="No safely completed"):
        profile._choose(values[1:], hardware())


def test_profile_cache_same_machine_source_does_not_repeat_work(tmp_path, monkeypatch):
    machine, calls = fake_environment(monkeypatch)
    first = profile.select_execution(tmp_path)
    cache = tmp_path / "execution_profile.json"
    old_bytes = cache.read_bytes()
    machine["ram_available_bytes"] -= GIB
    machine["gpu"]["used_bytes"] += GIB
    second = profile.select_execution(tmp_path)
    assert calls == [1, 2, 4]
    assert first["gpu_workers"] == second["gpu_workers"] == 4
    assert second["cache_reused"] and not first["cache_reused"]
    assert second["current_hardware"]["gpu"]["used_bytes"] == 2 * GIB
    assert cache.read_bytes() == old_bytes
    assert first["generated_only"] and first["real_rows_read"] == 0


@pytest.mark.parametrize("change", ["machine", "source"])
def test_changed_cached_binding_never_silently_reprofiles(tmp_path, monkeypatch, change):
    machine, calls = fake_environment(monkeypatch)
    profile.select_execution(tmp_path)
    if change == "machine":
        machine["gpu"]["uuid"] = "other-device"
    else:
        monkeypatch.setattr(profile, "implementation_identity", lambda: {"generated_source_hash": "changed"})
    with pytest.raises(ValueError, match="another machine/source/environment"):
        profile.select_execution(tmp_path, allow_started_without_profile=True)
    assert calls == [1, 2, 4]


def test_legacy_started_batch_requires_runner_authorization(tmp_path, monkeypatch):
    _, calls = fake_environment(monkeypatch)
    (tmp_path / "job_statuses.json").write_text("{}")
    with pytest.raises(ValueError, match="legacy adoption"):
        profile.select_execution(tmp_path)
    assert calls == []
    result = profile.select_execution(tmp_path, allow_started_without_profile=True)
    assert calls == [1, 2, 4]
    assert result["legacy_started_batch_adoption"]


def test_CPU_reserve_also_leaves_one_HGB_slot(tmp_path, monkeypatch):
    machine, calls = fake_environment(monkeypatch)
    machine["physical_cores"] = 16
    result = profile.select_execution(tmp_path)
    assert calls == [1, 2]
    assert result["gpu_workers"] == 2 and result["cpu_workers"] == 1
    assert result["profiles"][-1]["status"] == "skipped"
    assert (result["gpu_workers"] + result["cpu_workers"]) * 4 + 4 <= 16


def test_memory_projection_skips_unsafe_larger_concurrency(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    calls = []

    def measure(workers, info, directory):
        calls.append(workers)
        return candidate(workers, 10., gpu=12 * GIB)

    monkeypatch.setattr(profile, "_run_candidate", measure)
    result = profile.select_execution(tmp_path)
    assert calls == [1]
    assert result["gpu_workers"] == 1
    assert all(p["reason"] == "previous_profile_memory_projection" for p in result["profiles"][1:])


def test_smaller_memory_failure_does_not_try_more_workers(tmp_path, monkeypatch):
    fake_environment(monkeypatch)
    calls = []

    def measure(workers, info, directory):
        calls.append(workers)
        return {**candidate(workers, None, safe=False, status="failed"),
                "errors": [{"error": "CUDA out of memory"}]}

    monkeypatch.setattr(profile, "_run_candidate", measure)
    with pytest.raises(RuntimeError, match="No safely completed"):
        profile.select_execution(tmp_path)
    assert calls == [1]
    assert (tmp_path / "resource_profile/failed.json").is_file()
    assert not (tmp_path / "execution_profile.json").exists()


@pytest.mark.parametrize("resource", ["GPU", "RAM"])
def test_initial_memory_headroom_checked_before_any_worker(tmp_path, monkeypatch, resource):
    machine, calls = fake_environment(monkeypatch)
    if resource == "GPU":
        machine["gpu"]["used_bytes"] = 20 * GIB
    else:
        machine["ram_available_bytes"] = 10 * GIB
    with pytest.raises(RuntimeError, match="initial GPU/RAM headroom"):
        profile.select_execution(tmp_path)
    assert calls == []


def test_nvidia_query_records_selected_device_without_GPU_runtime(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-generated")
    monkeypatch.setattr(profile.shutil, "which", lambda name: "synthetic-nvidia-smi")
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return type("Result", (), {"stdout": "0, GPU-generated, NVIDIA RTX 4090 D, 24564, 512, 8, 560.94\n"})()

    monkeypatch.setattr(profile.subprocess, "run", run)
    result = profile._gpu_sample()
    assert seen[0][2] == "GPU-generated"
    assert result["total_bytes"] == 24564 * 1024 ** 2
    assert result["used_bytes"] == 512 * 1024 ** 2
    assert result["utilization_percent"] == 8.


def test_prepared_warmup_barriers_and_aggregate_window_with_mock_processes(tmp_path, monkeypatch):
    events = []
    workers = 2

    class Event:
        def __init__(self):
            self.at = None
            events.append(self)

        def set(self):
            if self.at is None:
                self.at = time.perf_counter()

    class Messages:
        def __init__(self):
            self.items = iter([(stage, rank) for stage in ("ready", "warmed", "done") for rank in range(workers)])

        def get(self, timeout):
            try:
                stage, rank = next(self.items)
            except StopIteration:
                raise queue.Empty
            if stage == "warmed":
                assert events[0].at is not None, "warmup cannot start until every worker is ready"
                assert events[1].at is None, "timing cannot start before every worker warmed"
            if stage == "done":
                assert events[1].at is not None
                return {"stage": stage, "rank": rank, "timed_start": events[1].at,
                        "timed_finish": events[1].at + 4., "timed_steps": 12,
                        "cpu_seconds": 1., "peak_cuda_reserved_bytes": GIB}
            return {"stage": stage, "rank": rank}

        def close(self):
            pass

    class Process:
        def __init__(self, **kwargs):
            self.pid = None
            self.exitcode = None

        def start(self):
            self.pid = 99

        def join(self, timeout):
            self.exitcode = 0

        def is_alive(self):
            return False

    class Context:
        def Event(self):
            return Event()

        def Queue(self):
            return Messages()

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(profile.multiprocessing, "get_context", lambda mode: Context())
    monkeypatch.setattr(profile, "_sample", lambda processes: {
        "at": time.perf_counter(), "gpu_used_bytes": 3 * GIB, "ram_available_bytes": 58 * GIB,
        "process_rss_bytes": 2 * GIB, "process_cpu_seconds": 0.,
        "system_cpu_percent": 0., "gpu_utilization_percent": 0.})
    result = profile._run_candidate(workers, hardware(), tmp_path)
    assert result["status"] == "complete"
    assert result["aggregate_steps_per_second"] == pytest.approx(6., rel=1e-4)
    assert result["initialization_and_warmup_excluded"]
    assert result["worker_start_skew_seconds"] == 0.
    assert len(result["ready"]) == len(result["warmup"]) == len(result["worker_results"]) == 2


def test_profile_step_matches_actual_short_training_arithmetic(tmp_path):
    import torch
    from scripts.comprehensive_batch import training, objective

    class TinyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(.1)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, numeric, categorical):
            return self.linear(self.dropout(numeric)).unsqueeze(-1)

    torch.set_num_threads(1)
    torch.manual_seed(19)
    network = TinyHeads()
    expected = deepcopy(network)
    numeric = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
    categorical = np.zeros((12, 1), dtype=np.int64)
    labels = np.array([0] * 8 + [1] * 4, dtype=np.float32)
    groups = {"references": [{"id": "r1", "indices": [0, 1, 2, 3]}, {"id": "r2", "indices": [4, 5, 6, 7]}],
              "targets": [{"id": "t1", "indices": [0, 1]}, {"id": "t2", "indices": [2, 3, 4, 5, 6, 7]}],
              "attacks": [8, 9, 10, 11]}
    job = {"job_id": "generated_resource_arithmetic", "arm": "S_alignment_group", "seed": 17,
           "learning_rate": .002, "auxiliary_weight": 1., "optimizer_steps": 1, "checkpoints": [1]}
    spec = dict(training.FIXED_TRAINING, batch_size=4, draws_per_class=2, report_every=1, recovery_every=1)
    result = training.train_loop(expected, numeric, categorical, labels, groups, job=job,
                                 training_spec=spec, device="cpu", output_dir=tmp_path)
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(network.parameters(), lr=.002, weight_decay=.0003)
    sampler = training.ClassSampler(labels, seed=17, per_class=2)
    x, cats, target = (torch.from_numpy(value) for value in (numeric, categorical, labels))
    resident = objective.resident_inputs(numeric, categorical, device=device, numeric_tensor=x, categorical_tensor=cats)
    training._seed(training._derived_seed(17, "dropout"), device)
    measured = profile._optimizer_step(network, optimizer, numeric, categorical, resident, x, cats, target, sampler, groups, device)
    assert training.state_digest(network.state_dict()) == training.state_digest(expected.state_dict())
    assert sampler.audit()["draw_sha256"] == result["sampler_draw_sha256"]
    assert measured["member_BCE"] == result["history"][0]["mean_path_member_BCE"]
    assert measured["auxiliary_cost"]["discovery_forward_rows_this_call"] == 12
    assert measured["auxiliary_penalty"] == result["history"][0]["mean_path_auxiliary"]


@pytest.mark.parametrize("exit_point", ["before_wait", "while_waiting", "as_gate_opens"])
def test_orphaned_profile_worker_leaves_barrier_without_GPU_work(monkeypatch, exit_point):
    class Parent:
        alive = exit_point != "before_wait"

        def is_alive(self):
            return self.alive

    class Stop:
        def is_set(self):
            return False

    parent = Parent()
    waits = []

    class Gate:
        def wait(self, timeout):
            waits.append(timeout)
            parent.alive = False
            return exit_point == "as_gate_opens"

    monkeypatch.setattr(profile.multiprocessing, "parent_process", lambda: parent)
    assert profile._wait_gate(Gate(), Stop()) is False
    assert waits == ([] if exit_point == "before_wait" else [.25])


def test_profile_step_boundary_observes_parent_exit_and_normal_release(monkeypatch):
    class Parent:
        alive = True

        def is_alive(self):
            return self.alive

    class Stop:
        requested = False

        def is_set(self):
            return self.requested

    class Gate:
        def wait(self, timeout):
            return True

    parent, stop = Parent(), Stop()
    monkeypatch.setattr(profile.multiprocessing, "parent_process", lambda: parent)
    assert profile._wait_gate(Gate(), stop) is True
    assert profile._worker_should_stop(stop) is False
    parent.alive = False
    assert profile._worker_should_stop(stop) is True
    parent.alive = True
    stop.requested = True
    assert profile._wait_gate(Gate(), stop) is False
