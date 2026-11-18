"""Pure stubs exercise cost wiring and deployment drift; no real benchmarks."""
from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.normal_subgroup_training import costs


class Preprocessor:
    def __init__(self):
        self.calls = 0

    def transform(self, frame):
        self.calls += 1
        return frame[["duration_seconds"]].to_numpy(np.float32), frame[["ip_protocol"]].to_numpy(np.int64)


class Predictor:
    def __init__(self, *, cold_offset=0., transformed_offset=0., single_offset=0., batch_size=256):
        self.preprocessor = Preprocessor()
        self.batch_size, self.calls = batch_size, 0
        self.cold_offset, self.transformed_offset, self.single_offset = cold_offset, transformed_offset, single_offset

    def score(self, frame):
        assert "label" not in frame and "source_id" not in frame
        numeric, _ = self.preprocessor.transform(frame)
        self.calls += 1
        offset = self.cold_offset + (self.single_offset if len(frame) == 1 else 0.)
        return numeric[:, 0].astype(np.float64) + offset

    def score_transformed(self, numeric, categorical):
        assert len(numeric) == len(categorical)
        self.calls += 1
        return numeric[:, 0].astype(np.float64) + self.transformed_offset


class CUDA:
    def __init__(self):
        self.syncs, self.resets = [], []

    def synchronize(self, device):
        self.syncs.append(device)

    def reset_peak_memory_stats(self, device):
        self.resets.append(device)

    def get_device_name(self, device):
        return "synthetic CUDA; not benchmarked"

    def max_memory_allocated(self, device):
        return 123

    def max_memory_reserved(self, device):
        return 456


@pytest.fixture
def case(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"synthetic artifact; never deserialized")
    frame = pd.DataFrame({"source_id": ["synthetic"] * 70, "row_id": np.arange(1, 71),
                          "duration_seconds": np.arange(70, dtype=float), "ip_protocol": np.full(70, 6),
                          "orig_pkts": np.ones(70), "resp_pkts": np.ones(70),
                          "orig_ip_bytes": np.full(70, 40.), "resp_ip_bytes": np.full(70, 80.),
                          "label": np.zeros(70, dtype=int)}, index=np.arange(100, 170))
    scores = frame[["source_id", "row_id"]].copy()
    scores["score"] = frame.duration_seconds
    scores = scores.iloc[::-1].reset_index(drop=True)
    cuda = CUDA()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda d: d, cuda=cuda))
    calls = []

    def benchmark(fn, raw, **spec):
        assert spec == {"batch_size": 128, "repeats": 3, "single_rows": 64, "warmup": 8}
        result = fn(raw.iloc[[3, 0]])
        if isinstance(result, tuple):
            np.testing.assert_array_equal(result[0][:, 0], [3, 0])
        else:
            # A deliberate score offset is tested separately with no timings.
            np.testing.assert_array_equal(result, [3, 0])
        calls.append(fn)
        return {"synthetic": True, "number": len(calls)}

    monkeypatch.setattr(costs, "benchmark_predictor", benchmark)
    kwargs = {"task_id": "transport_target_s17", "artifact": artifact,
              "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
              "frame": frame, "registered_scores": scores, "threshold": 10.,
              "output_path": tmp_path / "cost.json"}
    return kwargs, calls, cuda


def loader_for(cpu, gpu=None):
    gpu = Predictor() if gpu is None else gpu
    loads = []

    def loader(path, *, threads, device):
        assert threads == 1
        loads.append(device)
        return cpu if device == "cpu" else gpu

    return loader, loads, gpu


def test_costs_use_three_separate_scopes_fixed_rows_and_descriptive_gpu(case):
    kwargs, calls, cuda = case
    cpu = Predictor()
    loader, loads, gpu = loader_for(cpu)
    result = costs.measure_artifact(**kwargs, loader=loader)
    assert len(calls) == 3 and loads == ["cpu", "cuda:0"]
    assert result["cold_scores_exact"] and result["pretransformed_scores_exact"]
    assert result["deployment_contract_pass"]
    assert result["single_query_checked_rows"] == 64
    assert len(set(result["single_query_row_indices"])) == 64
    assert "no model" in result["preprocessing_only_timing"]["scope"]
    assert "no preprocessing" in result["model_only_timing"]["scope"]
    assert result["gpu_throughput"]["warmup"] == 8
    assert result["gpu_throughput"]["repeats"] == 3 and gpu.calls == 11
    assert result["gpu_throughput"]["internal_score_batch_size"] == 256
    assert result["gpu_throughput"]["requested_batch_rows"] == 512
    assert cuda.resets == ["cuda:0"] and len(cuda.syncs) == 6
    assert kwargs["output_path"].exists()


def test_cold_or_transformed_mismatch_stops_before_all_timing(case):
    kwargs, calls, cuda = case
    for setting, match in [({"cold_offset": .01}, "Cold CPU replay differs"),
                           ({"transformed_offset": .01}, "Pretransformed CPU scoring changes")]:
        loader, loads, _ = loader_for(Predictor(**setting))
        with pytest.raises(ValueError, match=match):
            costs.measure_artifact(**kwargs, loader=loader)
        assert loads == ["cpu"] and not calls and not cuda.syncs
        assert not kwargs["output_path"].exists()


def test_single_decision_drift_is_failure_even_when_cold_replay_is_exact(case):
    kwargs, _, _ = case
    loader, _, _ = loader_for(Predictor(single_offset=.1))
    result = costs.measure_artifact(**kwargs, loader=loader)
    assert result["cold_scores_exact"] and not result["deployment_contract_pass"]
    assert result["single_query_changed_decision_indices"] == [10]
    assert result["single_query_changed_decision_identities"] == [{"source_id": "synthetic", "row_id": 11}]
    assert result["single_query_max_abs_score_delta"] == pytest.approx(.1)


def test_gpu_decision_difference_is_separate_from_cpu_deployment_contract(case):
    kwargs, _, _ = case
    loader, _, _ = loader_for(Predictor(), Predictor(cold_offset=.1))
    result = costs.measure_artifact(**kwargs, loader=loader)
    assert result["deployment_contract_pass"]
    assert result["gpu_throughput"]["changed_decision_indices_at_cpu_threshold"] == [10]


def test_sha_identity_threshold_and_batch_contracts(case):
    kwargs, calls, _ = case
    loader, loads, _ = loader_for(Predictor())
    with pytest.raises(ValueError, match="hash mismatch"):
        costs.measure_artifact(**dict(kwargs, artifact_sha256="wrong"), loader=loader)
    with pytest.raises(ValueError, match="identities differ"):
        costs.measure_artifact(**dict(kwargs, registered_scores=kwargs["registered_scores"].iloc[1:]), loader=loader)
    with pytest.raises(ValueError, match="finite frozen"):
        costs.measure_artifact(**dict(kwargs, threshold=np.nan), loader=loader)
    assert loads == [] and calls == []
    wrong, loads, _ = loader_for(Predictor(batch_size=128))
    with pytest.raises(ValueError, match="internal batch size 256"):
        costs.measure_artifact(**kwargs, loader=wrong)
    assert loads == ["cpu"] and not calls


def test_existing_cost_output_is_not_overwritten(case):
    kwargs, calls, _ = case
    kwargs["output_path"].write_text("existing", encoding="utf-8")
    loader, loads, _ = loader_for(Predictor())
    with pytest.raises(ValueError, match="already exists"):
        costs.measure_artifact(**kwargs, loader=loader)
    assert not loads and not calls
    assert kwargs["output_path"].read_text(encoding="utf-8") == "existing"
