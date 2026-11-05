"""Synthetic/mock checks only: no archived model is loaded or timed."""
from collections import Counter
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids import unsw_diagnostic_cost as cost


@pytest.fixture
def inputs(tmp_path):
    frame = pd.DataFrame({"sample_id": [f"C-{i}" for i in range(300)],
                          "value": np.arange(300, dtype=np.float32)})
    batches = {"1": {"sample_ids": [[f"C-{299-i}"] for i in range(75)]},
               "256": {"sample_ids": [[f"C-{(i+j) % 300}" for j in range(255, -1, -1)]
                                       for i in range(75)]}}
    specs = {}
    for label, kind in cost.MODEL_KINDS.items():
        path = tmp_path / label
        if kind == "catboost":
            path.mkdir()
            (path / "model.cbm").write_bytes(b"synthetic-native-model")
            (path / "metadata.json").write_text("{}", encoding="utf-8")
        else:
            path.write_bytes(b"synthetic-model")
        specs[label] = {"kind": kind, "path": str(path), "threshold": 0.125}
    return frame, batches, specs


@pytest.fixture
def fake_runtime(monkeypatch):
    """Count the actual fake predictor invocations, including RSS worker calls."""
    state = SimpleNamespace(calls=[], loads=[], imports=[], launches=[], phase="parent",
                            fail_at=None, rss_calls=0, thread_scopes=[])

    def load(spec):
        label = next(label for label, kind in cost.MODEL_KINDS.items() if kind == spec["kind"])
        state.loads.append((state.phase, label))

        def predict(frame):
            state.calls.append({"phase": state.phase, "model": label,
                                "sample_ids": frame.sample_id.tolist(), "rows": len(frame)})
            if state.fail_at == len(state.calls):
                raise RuntimeError("synthetic predictor failure")
            return np.zeros(len(frame), dtype=bool)
        return predict

    def memory_info():
        state.rss_calls += 1
        return SimpleNamespace(rss=state.rss_calls * 100)

    def run(argv, **kwargs):
        state.launches.append((argv, kwargs))
        assert argv[1:4] == ["-m", "ids.unsw_diagnostic_cost", "--rss-worker"]
        request = cost.read_json(argv[4])
        previous = state.phase
        state.phase = request["model"] + "-worker"
        try:
            cost._rss_worker(request)
        finally:
            state.phase = previous
        return SimpleNamespace(returncode=0)

    def threads(use_torch):
        state.thread_scopes.append(use_torch)
        return nullcontext()

    ticks = itertools.count(0, 1_000_000)
    monkeypatch.setattr(cost, "_load_pipeline", load)
    monkeypatch.setattr(cost, "_import_backend", state.imports.append)
    monkeypatch.setattr(cost, "_cpu_threads", threads)
    monkeypatch.setattr(cost.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(cost.psutil, "Process", lambda: SimpleNamespace(memory_info=memory_info))
    monkeypatch.setattr(cost.subprocess, "run", run)
    return state


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_frozen_workload_actual_calls_ids_order_blocks_and_rss(inputs, fake_runtime, tmp_path):
    frame, batches, specs = inputs
    output = tmp_path / "cost"
    result = cost.measure_cost(specs, frame, batches, output)
    assert result["status"] == "complete"
    assert len(fake_runtime.calls) == 483
    assert sum(c["rows"] for c in fake_runtime.calls) == 62_448
    assert Counter(c["model"] for c in fake_runtime.calls) == dict.fromkeys(cost.MODEL_KINDS, 161)
    assert len(fake_runtime.loads) == 6  # three parent loads, one per separate worker
    assert len(fake_runtime.launches) == 3
    assert fake_runtime.thread_scopes == [True, True, False, False]
    assert result["budget"] == {
        "formal_calls": 450, "formal_rows": 57_825, "warmup_calls": 30, "warmup_rows": 3_855,
        "rss_processes": 3, "rss_forward_calls": 3, "rss_forward_rows": 768,
        "new_fits": 0, "optimizer_updates": 0,
    }
    assert cost.read_json(output / "cost-summary.json") == result
    formal = _jsonl(output / "timings.jsonl")
    warm = _jsonl(output / "warmup.jsonl")
    assert len(formal) == 450 and len(warm) == 30
    assert all(row["status"] == "complete" for row in formal + warm)
    assert all(row["duration_ms"] == 1.0 for row in formal)
    actual_parent = [c for c in fake_runtime.calls if c["phase"] == "parent"]
    cursor = 0
    rng = np.random.default_rng(np.random.SeedSequence([20261105, 3]))
    for size in (1, 256):
        size_key = str(size)
        expected_warm = [row for row in warm if row["size"] == size]
        assert len(expected_warm) == 15
        for recorded, actual in zip(expected_warm, actual_parent[cursor:cursor+15], strict=True):
            assert recorded["sample_ids"] == batches[size_key]["sample_ids"][0]
            assert actual["sample_ids"] == recorded["sample_ids"]
            assert actual["model"] == recorded["model"]
        cursor += 15
        size_rows = [row for row in formal if row["size"] == size]
        for group in range(75):
            order = [list(cost.MODEL_KINDS)[i] for i in rng.permutation(3)]
            for position, label in enumerate(order):
                row = size_rows[group*3+position]
                actual = actual_parent[cursor]
                assert row["execution_order"] == order
                assert row["model"] == actual["model"] == label
                assert row["sample_ids"] == actual["sample_ids"] == batches[size_key]["sample_ids"][group]
                assert (row["group"], row["block"], row["repeat_in_block"], row["position"]) == (
                    group, group // 25, group % 25, position,
                )
                cursor += 1
        summary = result["batches"][size_key]
        assert summary["sample_ids"] == batches[size_key]["sample_ids"]
        for label in cost.MODEL_KINDS:
            assert summary[label]["p50_ms"] == summary[label]["p95_ms"] == 1.0
            assert summary[label]["rows_per_second"] == size * 1000
            assert len(summary[label]["durations_ms"]) == 75
            assert len(summary[label]["blocks"]) == 3
            assert all(block["p95_ms"] == 1.0 for block in summary[label]["blocks"])
    assert cursor == 480
    rss_actual = [c for c in fake_runtime.calls if c["phase"] != "parent"]
    assert len(rss_actual) == 3
    assert all(c["sample_ids"] == batches["256"]["sample_ids"][0] for c in rss_actual)
    assert fake_runtime.rss_calls == 12
    for i, label in enumerate(cost.MODEL_KINDS):
        rss = result["rss"][label]
        assert [rss[key] for key in ("rss_after_import_bytes", "rss_after_input_bytes",
                                    "rss_after_load_bytes", "rss_after_probe_bytes")] == [
                                        (4*i+j)*100 for j in range(1, 5)]
        assert rss["model_load_ms"] == 1.0
        _, launch = fake_runtime.launches[i]
        assert launch["check"] is True
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert launch["env"][key] == "4"
        assert (output / f"rss-{label}-stderr.txt").exists()
    assert result["model_specs"]["CatBoost_reference"]["package_bytes"] == len(b"synthetic-native-model{}")


@pytest.mark.parametrize("change", ["duplicate", "unknown", "missing", "batch_count", "batch_size", "threshold"])
def test_input_failure_before_any_predictor_load(inputs, fake_runtime, tmp_path, change):
    frame, batches, specs = inputs
    batches = deepcopy(batches)
    if change == "duplicate":
        frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    elif change == "unknown":
        batches["1"]["sample_ids"][0] = ["absent"]
    elif change == "missing":
        frame.loc[0, "sample_id"] = None
    elif change == "batch_count":
        batches["1"]["sample_ids"].pop()
    elif change == "batch_size":
        batches["256"]["sample_ids"][0].pop()
    else:
        specs["H40"]["threshold"] = float("nan")
    with pytest.raises(ValueError):
        cost.measure_cost(specs, frame, batches, tmp_path / "invalid")
    assert fake_runtime.calls == fake_runtime.loads == []
    assert not (tmp_path / "invalid").exists()


def test_failed_formal_call_is_preserved_without_retry(inputs, fake_runtime, tmp_path):
    frame, batches, specs = inputs
    fake_runtime.fail_at = 17  # fifteen warmup + second formal call
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic predictor failure"):
        cost.measure_cost(specs, frame, batches, output)
    assert len(fake_runtime.calls) == 17
    assert fake_runtime.launches == []
    rows = _jsonl(output / "timings.jsonl")
    assert [r["status"] for r in rows] == ["complete", "failed"]
    failure = cost.read_json(output / "cost-failure.json")
    assert failure["status"] == "failed"
    assert failure["budget"]["formal_calls"] == 2
    assert failure["budget"]["warmup_calls"] == 15
    assert not (output / "cost-summary.json").exists()


def test_failed_warmup_is_preserved_without_retry(inputs, fake_runtime, tmp_path):
    frame, batches, specs = inputs
    fake_runtime.fail_at = 2
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic predictor failure"):
        cost.measure_cost(specs, frame, batches, output)
    assert len(fake_runtime.calls) == 2
    assert [r["status"] for r in _jsonl(output / "warmup.jsonl")] == ["complete", "failed"]
    assert _jsonl(output / "timings.jsonl") == []


def test_rss_launch_failure_retains_stderr_and_does_not_retry(inputs, fake_runtime, monkeypatch, tmp_path):
    frame, batches, specs = inputs
    launches = []

    def run(argv, **kwargs):
        launches.append(argv)
        kwargs["stderr"].write("synthetic child load failure")
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(cost.subprocess, "run", run)
    output = tmp_path / "failed-rss"
    with pytest.raises(subprocess.CalledProcessError):
        cost.measure_cost(specs, frame, batches, output)
    assert len(launches) == 1
    assert len(fake_runtime.calls) == 480
    assert (output / "rss-H40-stderr.txt").read_text() == "synthetic child load failure"
    assert cost.read_json(output / "cost-failure.json")["budget"]["rss_processes"] == 1


def test_rss_worker_rejects_wrong_input_size_before_load(inputs, fake_runtime, tmp_path):
    frame, _, specs = inputs
    path = tmp_path / "rss.parquet"
    frame.iloc[:255].to_parquet(path)
    with pytest.raises(ValueError, match="exactly 256"):
        cost._rss_worker({"model": "H40", "spec": specs["H40"], "input_path": str(path),
                          "output_path": str(tmp_path / "worker.json")})
    assert fake_runtime.loads == fake_runtime.calls == []


def test_hgb_adapter_uses_raw_margin_and_one_transform(monkeypatch):
    from ids import models
    calls = []
    margins = np.array([-20.0, 0.125 - 1e-10, 0.125, 20.0], dtype=np.float64)

    def transform(frame):
        calls.append("transform")
        return frame.value.to_numpy().reshape(-1, 1)

    def predict(data):
        calls.append("decision_function")
        assert data.shape == (4, 1)
        return margins

    bundle = SimpleNamespace(model_name="hgb", preprocessor=SimpleNamespace(transform=transform),
                             estimator=SimpleNamespace(decision_function=predict))
    monkeypatch.setattr(models, "load_bundle", lambda path, device: bundle)
    pipeline = cost._load_pipeline({"kind": "hgb", "path": "mock", "threshold": 0.125})
    assert calls == []
    assert pipeline(pd.DataFrame({"value": np.arange(4)})).tolist() == [False, False, True, True]
    assert calls == ["transform", "decision_function"]


def test_catboost_adapter_uses_native_raw_margin_once(monkeypatch):
    from ids import native_trees
    calls = []
    margins = np.array([-20.0, 0.125 - 1e-10, 0.125, 20.0], dtype=np.float64)

    def predict(frame, prediction_type, thread_count):
        calls.append(("predict", frame, prediction_type, thread_count))
        return margins

    def convert(frame, selected_features, categorical):
        calls.append(("conversion", frame, selected_features, categorical))
        return frame

    bundle = native_trees.NativeTreeBundle(SimpleNamespace(predict=predict), ["value"], [], {}, {}, 4)
    monkeypatch.setattr(native_trees, "native_frame", convert)
    monkeypatch.setattr(native_trees, "load_native_tree", lambda path: bundle)
    pipeline = cost._load_pipeline({"kind": "catboost", "path": "mock", "threshold": 0.125})
    assert calls == []
    frame = pd.DataFrame({"value": np.arange(4)})
    assert pipeline(frame).tolist() == [False, False, True, True]
    assert len(calls) == 2
    assert calls[0][0] == "conversion" and calls[0][1] is frame
    assert calls[1][0] == "predict" and calls[1][1] is frame
    assert calls[1][2:] == ("RawFormulaVal", 4)
    bundle.num_threads = 1
    with pytest.raises(ValueError, match="four threads"):
        cost._load_pipeline({"kind": "catboost", "path": "mock", "threshold": 0.125})


def test_mlp_adapter_does_not_query_during_load_and_uses_raw_logits(monkeypatch):
    from ids import models
    import torch
    calls = []
    logits = torch.tensor([[-20.0], [0.124], [0.125], [20.0]], dtype=torch.float32)

    class FakeNet:
        def load_state_dict(self, state):
            assert state == {"synthetic": True}

        def cpu(self):
            return self

        def eval(self):
            return self

        def __call__(self, data):
            assert data.device.type == "cpu" and not torch.is_grad_enabled()
            calls.append("network")
            return logits

    def transform(frame):
        calls.append("transform")
        return frame.value.to_numpy(dtype=np.float32).reshape(-1, 1)

    bundle = SimpleNamespace(model_name="mlp", encoded_dim=1, hidden_dims=(2,),
                             network_state={"synthetic": True}, preprocessor=SimpleNamespace(transform=transform))
    monkeypatch.setattr(models, "load_bundle", lambda path, device: bundle)
    monkeypatch.setattr(models, "_network", lambda *args: FakeNet())
    pipeline = cost._load_pipeline({"kind": "mlp", "path": "mock", "threshold": 0.125})
    assert calls == []
    assert pipeline(pd.DataFrame({"value": np.arange(4)})).tolist() == [False, False, True, True]
    assert calls == ["transform", "network"]


@pytest.mark.parametrize("kind", ["mlp", "hgb", "catboost"])
@pytest.mark.parametrize("invalid", ["nan", "inf", "wrong_rows"])
def test_each_adapter_rejects_invalid_raw_scores_after_exactly_one_query(monkeypatch, kind, invalid):
    from ids import models, native_trees
    import torch
    calls = []
    margins = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    if invalid == "wrong_rows":
        margins = margins[:2]
    else:
        margins[1] = float(invalid)

    def predict(*args, **kwargs):
        calls.append("query")
        return margins

    prep = SimpleNamespace(transform=lambda frame: frame.value.to_numpy(dtype=np.float32).reshape(-1, 1))
    bundle = SimpleNamespace(model_name=kind, preprocessor=prep,
                             estimator=SimpleNamespace(decision_function=predict),
                             num_threads=4, predict_raw=predict, encoded_dim=1, hidden_dims=(2,),
                             network_state={})
    monkeypatch.setattr(models, "load_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(native_trees, "load_native_tree", lambda *a, **k: bundle)

    class FakeNet:
        def load_state_dict(self, state):
            pass

        def cpu(self):
            return self

        def eval(self):
            return self

        def __call__(self, matrix):
            return torch.as_tensor(predict(), dtype=torch.float32).reshape(-1, 1)

    monkeypatch.setattr(models, "_network", lambda *a, **k: FakeNet())
    pipeline = cost._load_pipeline({"kind": kind, "path": "mock", "threshold": .125})
    assert calls == []
    with pytest.raises(ValueError, match="nonfinite or mis-shaped raw margins"):
        pipeline(pd.DataFrame({"value": np.arange(3)}))
    assert calls == ["query"]


def test_backend_imports_precede_threadpool_discovery(inputs, fake_runtime, tmp_path, monkeypatch):
    observations = []

    @contextmanager
    def threads(use_torch):
        observations.append(list(fake_runtime.imports))
        yield

    monkeypatch.setattr(cost, "_cpu_threads", threads)
    frame, batches, specs = inputs
    cost.measure_cost(specs, frame, batches, tmp_path / "ordered-imports")
    assert [len(imports) for imports in observations] == [3, 4, 5, 6]
    assert observations[0] == ["mlp", "hgb", "catboost"]
    assert [imports[-1] for imports in observations[1:]] == ["mlp", "hgb", "catboost"]


@pytest.mark.parametrize("old_pythonpath", [None, "old-checkout-src" + os.pathsep + "other-src"])
def test_rss_child_uses_current_source_before_existing_pythonpath(
        inputs, fake_runtime, tmp_path, monkeypatch, old_pythonpath):
    if old_pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", old_pythonpath)
    frame, batches, specs = inputs
    cost.measure_cost(specs, frame, batches, tmp_path / "source-bound-workers")
    current_source = str(Path(cost.__file__).resolve().parents[1])
    expected = current_source + (os.pathsep + old_pythonpath if old_pythonpath else "")
    assert len(fake_runtime.launches) == 3
    assert all(options["env"]["PYTHONPATH"] == expected for _, options in fake_runtime.launches)
    assert os.environ.get("PYTHONPATH") == old_pythonpath


def test_statistics_use_frozen_linear_percentiles():
    values = np.arange(1, 26, dtype=np.float64)
    stats = cost._statistics(values, 256)
    assert stats["p50_ms"] == 13.0
    assert stats["p95_ms"] == pytest.approx(23.8)
    assert stats["rows_per_second"] == pytest.approx(256_000 / 13)


def test_cpu_threads_sets_four_and_restores_even_on_failure(monkeypatch):
    import torch
    events = []

    @contextmanager
    def limit(limits):
        events.append(("libraries", limits))
        try:
            yield
        finally:
            events.append(("libraries", "restored"))

    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: events.append(("torch", value)))
    monkeypatch.setattr(cost, "threadpool_limits", limit)
    with pytest.raises(RuntimeError, match="synthetic"):
        with cost._cpu_threads(True):
            events.append(("body", 4))
            raise RuntimeError("synthetic")
    assert events == [("torch", 4), ("libraries", 4), ("body", 4),
                      ("libraries", "restored"), ("torch", 8)]


def test_cli_only_dispatches_explicit_worker_request(tmp_path, monkeypatch):
    request = {"model": "H40", "synthetic": True}
    path = tmp_path / "request.json"
    cost.write_json(path, request)
    calls = []
    monkeypatch.setattr(cost, "_rss_worker", calls.append)
    cost.main(["--rss-worker", str(path)])
    assert calls == [request]
