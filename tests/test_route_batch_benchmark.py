"""Benchmark call accounting and scope; never assert machine speed."""

import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

from ids.route_batch.benchmark import benchmark_predictor


def test_default_measurement_uses_fixed_rows_and_all_three_modes():
    frame = pd.DataFrame({"original_row": np.arange(700), "raw_text": ["80"] * 700})
    calls = []

    def predict(values):
        assert list(values.columns) == ["original_row", "raw_text"]
        calls.append(values["original_row"].tolist())
        return np.zeros(len(values))

    result = benchmark_predictor(predict, frame)
    assert result["counts"] == {"warmup_calls": 24, "timed_calls": 198, "timed_rows": 2112}
    assert len(calls) == 222
    assert [len(row) for row in calls[:3]] == [1, 128, 512]
    singles = result["single"]
    assert singles["row_indices"] == [(i * 700) // 64 for i in range(64)]
    assert singles["n_unique_rows"] == 64
    assert singles["n_observations"] == 192
    assert len(singles["blocks_ms"]) == 3
    for repetition in range(3):
        measured = calls[24 + repetition * 64:24 + (repetition + 1) * 64]
        assert measured == [[index] for index in singles["row_indices"]]
    assert result["batch"]["total_rows_processed"] == 384
    assert result["batch_512"]["total_rows_processed"] == 1536
    assert result["batch"]["n_observations"] == result["batch_512"]["n_observations"] == 3
    assert result["rss"]["sampled_peak_bytes"] >= result["rss"]["before_bytes"]
    assert result["rss"]["sampled_peak_bytes"] >= result["rss"]["after_bytes"]
    assert result["rss"]["samples"] >= 2
    assert result["rss"]["sampling_interval_ms"] == 10
    assert result["file_io_included"] is result["cold_model_load_measured"] is False
    assert result["warmup_included_in_timing"] is result["warmup_included_in_rss"] is False
    json.dumps(result, allow_nan=False)


def test_short_frame_caps_all_modes_and_preserves_input():
    frame = pd.DataFrame({"value": [1, 2, 3]}, index=[19, 4, 71])
    before = frame.copy(deep=True)
    calls = []

    def predict(values):
        calls.append(values["value"].tolist())
        return values["value"].to_numpy()

    result = benchmark_predictor(predict, frame, repeats=2, single_rows=64, warmup=0)
    assert result["counts"] == {"warmup_calls": 0, "timed_calls": 10, "timed_rows": 18}
    assert calls[:6] == [[1], [2], [3], [1], [2], [3]]
    assert result["batch"]["batch_size"] == result["batch_512"]["batch_size"] == 3
    assert result["single"]["row_indices"] == [0, 1, 2]
    pd.testing.assert_frame_equal(frame, before)


def test_rss_sampling_thread_stops_when_prediction_raises():
    def predict(_):
        raise RuntimeError("synthetic predictor failure")

    with pytest.raises(RuntimeError, match="synthetic predictor failure"):
        benchmark_predictor(predict, pd.DataFrame({"value": [1, 2]}), warmup=0)
    assert not any(thread.name == "route-batch-rss" and thread.is_alive() for thread in threading.enumerate())


def test_rss_sampler_records_during_timed_inference(monkeypatch):
    class FakeProcess:
        def memory_info(self):
            return type("Memory", (), {"rss": 123456})()

    monkeypatch.setattr("ids.route_batch.benchmark.psutil.Process", FakeProcess)

    def predict(values):
        time.sleep(.025)
        return np.zeros(len(values))

    result = benchmark_predictor(predict, pd.DataFrame({"value": [1]}), repeats=1,
                                 single_rows=1, warmup=0)
    assert result["rss"]["samples"] > 2
    assert result["rss"]["sampled_peak_bytes"] == 123456
    assert result["rss"]["sampling_errors"] == []


@pytest.mark.parametrize("keyword,value", [("batch_size", 0), ("repeats", 0), ("single_rows", 0),
                                           ("warmup", -1), ("repeats", True), ("batch_size", 1.5)])
def test_invalid_counts_are_rejected(keyword, value):
    with pytest.raises(ValueError):
        benchmark_predictor(lambda frame: np.zeros(len(frame)), pd.DataFrame({"value": [1]}),
                            **{keyword: value})


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="nonempty raw DataFrame"):
        benchmark_predictor(lambda frame: frame, pd.DataFrame())
