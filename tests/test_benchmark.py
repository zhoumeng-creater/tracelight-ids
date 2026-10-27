import itertools
import json

import numpy as np
import pandas as pd
import pytest

from ids.benchmark import benchmark_predict


def test_real_single_calls_and_batch_throughput_have_separate_denominators(monkeypatch):
    clock = itertools.count(start=0, step=1_000_000)
    monkeypatch.setattr("ids.benchmark.time.perf_counter_ns", lambda: next(clock))
    seen = []

    def predict(frame):
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["feature"]
        seen.append(len(frame))
        return np.zeros(len(frame))

    result = benchmark_predict(
        predict, pd.DataFrame({"feature": range(5)}),
        warmup=1, repeats=2, single_rows=2, batch_size=3,
    )
    assert seen == [1, 3, 1, 1, 1, 1, 3, 3]
    assert result["single"]["n_unique_rows"] == 2
    assert result["single"]["n_observations"] == 4
    assert result["single"]["row_indices"] == [0, 2]
    assert result["single"]["latencies_ms"] == [1.0] * 4
    assert result["single"]["p50_ms"] == 1
    assert result["single"]["p95_ms"] == 1
    assert result["batch"]["latencies_ms"] == [1.0] * 2
    assert result["batch"]["throughput_rows_per_second"] == 3000
    assert result["batch"]["total_rows_processed"] == 6
    assert result["rss"]["peak_bytes"] is None
    assert result["scope"] == "preprocessing_plus_predict_fn"
    json.dumps(result, allow_nan=False)


def test_small_input_caps_requested_sizes_and_supports_numpy():
    result = benchmark_predict(
        lambda rows: rows.sum(axis=1), np.ones((2, 3)),
        warmup=0, repeats=1, single_rows=100, batch_size=256,
    )
    assert result["single"]["n_unique_rows"] == 2
    assert result["single"]["n_observations"] == 2
    assert result["batch"]["batch_size"] == 2
    assert result["batch"]["requested_batch_size"] == 256
    assert result["batch"]["throughput_rows_per_second"] > 0


@pytest.mark.parametrize(
    "kwargs", [{"threads": 0}, {"warmup": -1}, {"repeats": 0},
               {"single_rows": 0}, {"batch_size": 0}, {"repeats": 1.5},
               {"threads": True}],
)
def test_invalid_measurement_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        benchmark_predict(lambda x: x, np.ones((2, 2)), **kwargs)


def test_empty_input_and_prediction_errors_are_not_silently_hidden():
    with pytest.raises(ValueError, match="at least one"):
        benchmark_predict(lambda x: x, np.empty((0, 2)))

    def broken(_):
        raise RuntimeError("prediction failed")

    with pytest.raises(RuntimeError, match="prediction failed"):
        benchmark_predict(broken, [[1]], warmup=0, repeats=1, single_rows=1)


def test_non_advancing_timer_is_rejected(monkeypatch):
    monkeypatch.setattr("ids.benchmark.time.perf_counter_ns", lambda: 1)
    with pytest.raises(RuntimeError, match="timer"):
        benchmark_predict(lambda x: x, [[1]], warmup=0, repeats=1, single_rows=1)
