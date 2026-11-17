"""Cost-worker wiring with fake predictors/CUDA; no real model execution."""
from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.model_training_redesign import costs, runner


class FakePreprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, frame):
        self.calls += 1
        return (frame[["duration_seconds"]].to_numpy(dtype=np.float32),
                frame[["ip_protocol"]].to_numpy(dtype=np.int64))


class FakePredictor:
    def __init__(self, *, cold_offset=0.0, transformed_offset=0.0):
        self.preprocessor = FakePreprocessor()
        self.model_calls = 0
        self.cold_offset = cold_offset
        self.transformed_offset = transformed_offset

    def score(self, frame):
        numeric, categorical = self.preprocessor.transform(frame)
        self.model_calls += 1
        assert len(numeric) == len(categorical)
        return numeric[:, 0].astype(np.float64) + self.cold_offset

    def score_transformed(self, numeric, categorical):
        self.model_calls += 1
        assert len(numeric) == len(categorical)
        return numeric[:, 0].astype(np.float64) + self.transformed_offset


class FakeBackend:
    def __init__(self, **kwargs):
        self.cpu = FakePredictor(**kwargs)
        self.gpu = FakePredictor()
        self.loads = []

    def load_predictor(self, artifact, *, threads, device):
        self.loads.append({"artifact": artifact, "threads": threads, "device": device})
        return self.cpu if device == "cpu" else self.gpu


class FakeCUDA:
    def __init__(self):
        self.synchronizations = []
        self.resets = []

    def synchronize(self, device):
        self.synchronizations.append(device)

    def reset_peak_memory_stats(self, device):
        self.resets.append(device)

    @staticmethod
    def max_memory_allocated(device):
        assert device == "cuda:0"
        return 1000

    @staticmethod
    def max_memory_reserved(device):
        assert device == "cuda:0"
        return 2000


@pytest.fixture
def setup_worker(tmp_path, monkeypatch):
    root, task_id = tmp_path, "protocol_blind_s17"
    model = root / "models" / task_id / "trajectory/model.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fake artifact; never loaded by torch")
    frame = pd.DataFrame({
        "source_id": ["synthetic"] * 5, "row_id": np.arange(5),
        "duration_seconds": np.arange(5, dtype=float),
        "orig_pkts": np.ones(5), "resp_pkts": np.ones(5),
        "orig_ip_bytes": np.full(5, 40.0), "resp_ip_bytes": np.full(5, 76.0),
        "ip_protocol": np.full(5, 6),
    }, index=np.arange(100, 105))
    (root / "inputs").mkdir()
    frame.to_parquet(root / "inputs/ED.parquet", index=True)
    protocol = {
        "device": "cuda:0", "registration": {
            "benchmark": {"single_rows": 2, "repeats": 3, "warmup": 1, "batch_size": 3},
            "gpu_benchmark": {"batch_size": 4, "warmup": 1, "repeats": 2}}}
    write_json(root / "protocol.json", protocol)
    write_json(root / f"evidence/{task_id}/calibration.json", {"threshold": 1.5})
    checks = []
    monkeypatch.setattr(costs, "verify_live_code", lambda p: checks.append("code"))
    monkeypatch.setattr(runner, "verify_inputs", lambda r, p: checks.append("inputs"))

    def frozen(r, p):
        checks.append("freeze")
        return {"artifacts": {task_id: {"path": model.relative_to(root).as_posix()}}}

    monkeypatch.setattr(runner, "verify_freeze", frozen)
    monkeypatch.setattr(runner, "aligned_scores", lambda r, t, role, f:
                        f.duration_seconds.to_numpy(dtype=np.float64))
    limits = []

    def thread_limit(*, limits):
        assert limits == 1
        return nullcontext()

    monkeypatch.setattr(costs, "threadpool_limits", thread_limit)
    cuda = FakeCUDA()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda d: d, cuda=cuda))
    return root, task_id, protocol, checks, cuda, limits


def test_three_cpu_timings_are_independent_and_model_only_preserves_row_gather(setup_worker, monkeypatch):
    root, task_id, protocol, checks, cuda, _ = setup_worker
    backend = FakeBackend()
    calls = []

    def benchmark(fn, raw, **spec):
        assert spec == protocol["registration"]["benchmark"]
        before = backend.cpu.preprocessor.calls, backend.cpu.model_calls
        # Noncontiguous and reversed indexes must address the precomputed arrays.
        subset = raw.iloc[[3, 0]]
        value = fn(subset)
        after = backend.cpu.preprocessor.calls, backend.cpu.model_calls
        delta = tuple(b - a for a, b in zip(before, after))
        calls.append(delta)
        if isinstance(value, tuple):
            np.testing.assert_array_equal(value[0][:, 0], [3, 0])
        else:
            np.testing.assert_array_equal(value, [3, 0])
        return {"marker": len(calls), "scope": "fake benchmark", "rss": {"scope": "whole process"}}

    monkeypatch.setattr(costs, "benchmark_predictor", benchmark)
    result = costs.measure(root, task_id, backend=backend)
    assert calls == [(1, 1), (1, 0), (0, 1)]
    assert [result[key]["marker"] for key in
            ("timing", "preprocessing_only_timing", "model_only_timing")] == [1, 2, 3]
    assert "no preprocessing" in result["model_only_timing"]["scope"]
    assert "no model" in result["preprocessing_only_timing"]["scope"]
    assert result["cold_scores_exact"] and result["pretransformed_scores_exact"]
    assert result["single_query_max_abs_score_delta"] == 0
    assert result["single_query_changed_decision_indices"] == []
    assert [(item["threads"], item["device"]) for item in backend.loads] == [(1, "cpu"), (1, "cuda:0")]
    assert checks == ["code", "inputs", "freeze", "code"]
    assert cuda.resets == ["cuda:0"]
    assert len(cuda.synchronizations) == 5
    gpu = result["gpu_throughput"]
    assert gpu["batch_rows"] == 4 and gpu["repeats"] == 2
    assert gpu["peak_allocated_bytes"] == 1000 and gpu["peak_reserved_bytes"] == 2000
    assert gpu["max_abs_delta_from_registered_cpu"] == 0
    assert gpu["changed_decision_indices_at_cpu_threshold"] == []
    assert read_json(root / f"cost/{task_id}.json") == result


def test_cold_replay_mismatch_stops_before_timing_or_gpu(setup_worker, monkeypatch):
    root, task_id, _, _, cuda, _ = setup_worker
    backend = FakeBackend(cold_offset=0.01)

    def forbidden(*args, **kwargs):
        pytest.fail("No benchmark is allowed after a cold-score mismatch")

    monkeypatch.setattr(costs, "benchmark_predictor", forbidden)
    with pytest.raises(ValueError, match="Cold CPU replay differs"):
        costs.measure(root, task_id, backend=backend)
    assert len(backend.loads) == 1 and cuda.synchronizations == []
    assert not (root / f"cost/{task_id}.json").exists()


def test_pretransformed_mismatch_cannot_be_reported_as_model_only_cost(setup_worker, monkeypatch):
    root, task_id, _, _, cuda, _ = setup_worker
    backend = FakeBackend(transformed_offset=0.01)
    benchmark_calls = []

    def benchmark(fn, raw, **spec):
        benchmark_calls.append(fn)
        fn(raw.iloc[[0]])
        return {"scope": "fake benchmark"}

    monkeypatch.setattr(costs, "benchmark_predictor", benchmark)
    with pytest.raises(ValueError, match="Pretransformed model scoring changes"):
        costs.measure(root, task_id, backend=backend)
    assert len(benchmark_calls) == 2
    assert len(backend.loads) == 1 and cuda.synchronizations == []
    assert not (root / f"cost/{task_id}.json").exists()
