"""Cost-worker orchestration with fake predictors and a fake torch.cuda API."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.model_training import runner
from ids.route_batch import benchmark


class FakeCostBackend:
    def __init__(self, events, *, cpu_delta=0.0, gpu_delta=0.25):
        self.events = events
        self.cpu_delta, self.gpu_delta = cpu_delta, gpu_delta

    def load_predictor(self, path, *, device, threads):
        assert path.is_file() and threads == 1
        self.events.append(("load", device))

        def score(raw):
            assert list(raw.columns) == list(runner.data.RAW_FEATURES)
            self.events.append(("score", device, len(raw)))
            delta = self.cpu_delta if device == "cpu" else self.gpu_delta
            return raw.duration_seconds.to_numpy(dtype=np.float64) + delta

        def transform(raw):
            self.events.append(("preprocess", "tabm", len(raw)))
            return np.zeros((len(raw), 5)), np.zeros((len(raw), 6), dtype=int)

        def hgb_matrix(raw):
            self.events.append(("preprocess", "hgb", len(raw)))
            return np.zeros((len(raw), 17))

        score.cost_kind = "full_predictor"
        transform.cost_kind = "tabm_preprocessing"
        hgb_matrix.cost_kind = "hgb_preprocessing"
        return SimpleNamespace(score=score, preprocessor=SimpleNamespace(
            transform=transform, hgb_matrix=hgb_matrix))


@pytest.fixture
def cost_case(tmp_path, monkeypatch):
    def prepare(task_id="tabm_ple_s17", *, rows=1024, boundary=0.5, cpu_delta=0.0):
        root = tmp_path / task_id
        root.mkdir()
        artifact = root / "model.fake"
        artifact.write_bytes(b"synthetic model artifact, no real network")
        protocol = {"device": "cuda:0", "registration": {
            "benchmark": {"batch_size": 128, "single_rows": 64, "repeats": 3, "warmup": 8},
            "gpu_benchmark": {"batch_size": 512, "repeats": 3, "warmup": 8}}}
        write_json(root / "protocol.json", protocol)
        write_json(root / f"evidence/{task_id}/calibration.json", {"threshold": boundary})
        frame = pd.DataFrame({name: np.ones(rows) for name in runner.data.RAW_FEATURES})
        frame["duration_seconds"] = np.arange(rows, dtype=float) / 1000
        frame["source_id"], frame["row_id"] = "synthetic", np.arange(rows)
        frame["label"] = np.arange(rows) % 2
        (root / "inputs").mkdir()
        frame.to_parquet(root / "inputs/ED.parquet", index=False)
        expected = frame.duration_seconds.to_numpy().copy()
        frozen = {"artifacts": {task_id: {"path": "model.fake"}}}
        events = []

        def verify_code(value):
            assert value == protocol
            events.append(("verify", "code"))

        def verify_inputs(path, value):
            assert path == root and value == protocol
            events.append(("verify", "inputs"))

        def verify_freeze(path, value):
            assert path == root and value == protocol
            events.append(("verify", "freeze"))
            return frozen

        def aligned(path, name, role, actual):
            assert path == root and name == task_id and role == "ED"
            pd.testing.assert_frame_equal(actual, frame)
            return expected.copy()

        def fake_benchmark(predict, raw, **options):
            assert options == protocol["registration"]["benchmark"]
            events.append(("benchmark_start", predict.cost_kind))
            predict(raw.iloc[:4])
            events.append(("benchmark_end", predict.cost_kind))
            return {"synthetic": True, "kind": predict.cost_kind, "rows": len(raw)}

        def sync(device):
            assert device == "cuda:0"
            events.append(("sync", device))

        def reset(device):
            assert device == "cuda:0"
            events.append(("reset_peak", device))

        fake_torch = SimpleNamespace(device=lambda device: device, cuda=SimpleNamespace(
            synchronize=sync, reset_peak_memory_stats=reset,
            max_memory_allocated=lambda device: 1234, max_memory_reserved=lambda device: 5678))
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setattr(runner, "verify_live_code", verify_code)
        monkeypatch.setattr(runner, "verify_inputs", verify_inputs)
        monkeypatch.setattr(runner, "verify_freeze", verify_freeze)
        monkeypatch.setattr(runner, "aligned_scores", aligned)
        monkeypatch.setattr(benchmark, "benchmark_predictor", fake_benchmark)
        ticks = iter(range(1000))
        monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: float(next(ticks))))
        backend = FakeCostBackend(events, cpu_delta=cpu_delta)
        return SimpleNamespace(root=root, task_id=task_id, events=events, backend=backend,
                               expected=expected, boundary=boundary, rows=rows, protocol=protocol)
    return prepare


@pytest.mark.parametrize("rows,boundary", [(1024, 0.5), (17, None)])
def test_tabm_cpu_costs_precede_gpu_with_registered_warmups_and_synchronization(cost_case, rows, boundary):
    case = cost_case(rows=rows, boundary=boundary)
    runner.cost_worker(case.root, case.task_id, backend=case.backend)
    report = read_json(case.root / f"cost/{case.task_id}.json")
    assert case.events[:4] == [("verify", "code"), ("verify", "inputs"), ("verify", "freeze"),
                               ("load", "cpu")]
    gpu_start = case.events.index(("sync", "cuda:0"))
    cpu_events, gpu_events = case.events[:gpu_start], case.events[gpu_start:]
    assert ("benchmark_end", "full_predictor") in cpu_events
    assert ("benchmark_end", "tabm_preprocessing") in cpu_events
    assert sum(event == ("score", "cpu", 1) for event in cpu_events) == min(64, rows)
    assert all(not (event[0] == "preprocess" or (event[0] == "score" and event[1] == "cpu"))
               for event in gpu_events)
    count = min(512, rows)
    assert gpu_events == [
        ("sync", "cuda:0"), ("reset_peak", "cuda:0"), ("load", "cuda:0"), ("sync", "cuda:0"),
        *[("score", "cuda:0", count)] * 8, ("sync", "cuda:0"),
        *[("score", "cuda:0", count), ("sync", "cuda:0")] * 3,
    ]
    assert report["cold_scores_exact"] and report["preprocessing_included"]
    assert report["timing"]["kind"] == "full_predictor"
    assert report["preprocessing_only_timing"]["kind"] == "tabm_preprocessing"
    assert report["single_query_max_abs_score_delta"] == 0
    assert report["single_query_changed_decision_indices"] == []
    gpu = report["gpu_throughput"]
    assert gpu["device"] == "cuda:0" and gpu["batch_rows"] == count
    assert gpu["repeats"] == 3 and gpu["seconds_per_repeat"] == [1.0] * 3
    assert gpu["rows_per_second"] == count
    assert gpu["artifact_load_seconds"] == 1
    assert gpu["preprocessing_and_host_device_copies_included"]
    assert gpu["peak_allocated_bytes"] == 1234 and gpu["peak_reserved_bytes"] == 5678
    assert gpu["max_abs_delta_from_registered_cpu"] == pytest.approx(0.25)
    indices = [(i * rows) // count for i in range(count)]
    assert gpu["row_indices"] == indices
    wanted = [] if boundary is None else [i for i in indices
        if (case.expected[i] + 0.25 > boundary) != (case.expected[i] > boundary)]
    assert gpu["changed_decision_indices_at_cpu_threshold"] == wanted
    if boundary is not None:
        assert wanted  # A measurable difference is reported; no threshold is changed.
    assert "frozen CPU scores only" in gpu["scope"]
    assert read_json(case.root / f"evidence/{case.task_id}/calibration.json") == {"threshold": boundary}


def test_hgb_reports_gpu_not_applicable_and_uses_native_preprocessing(cost_case, monkeypatch):
    case = cost_case("hgb", rows=100)

    class ForbiddenTorch:
        def __getattr__(self, name):
            raise AssertionError("HGB cost must not access a CUDA runtime")

    monkeypatch.setitem(sys.modules, "torch", ForbiddenTorch())
    runner.cost_worker(case.root, case.task_id, backend=case.backend)
    report = read_json(case.root / "cost/hgb.json")
    assert [event for event in case.events if event[0] == "load"] == [("load", "cpu")]
    assert not any(event[0] in {"sync", "reset_peak"} for event in case.events)
    assert report["preprocessing_only_timing"]["kind"] == "hgb_preprocessing"
    assert report["gpu_throughput"] == {
        "applicable": False, "reason": "registered HGB has no GPU implementation"}


def test_cpu_replay_failure_stops_before_timing_or_gpu_and_writes_no_cost(cost_case):
    case = cost_case(cpu_delta=0.01)
    with pytest.raises(ValueError, match="Cold CPU replay does not exactly reproduce"):
        runner.cost_worker(case.root, case.task_id, backend=case.backend)
    assert not any(event[0] in {"benchmark_start", "sync", "reset_peak"} for event in case.events)
    assert [event for event in case.events if event[0] == "load"] == [("load", "cpu")]
    assert not (case.root / f"cost/{case.task_id}.json").exists()
