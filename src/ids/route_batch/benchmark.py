"""Hot CPU inference timing and interval RSS sampling for one isolated worker."""

from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd
import psutil


def _positive_int(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _indices(length, requested):
    count = min(length, requested)
    return [(index * length) // count for index in range(count)]


def _elapsed_ms(predict_fn, frame):
    start = time.perf_counter_ns()
    predict_fn(frame)
    elapsed = time.perf_counter_ns() - start
    if elapsed <= 0:
        raise RuntimeError("inference timer did not advance")
    return elapsed / 1_000_000


def _batch_report(timings, indices, requested):
    values = np.asarray(timings, dtype=np.float64)
    rows = len(indices)
    return {
        "requested_batch_size": requested, "batch_size": rows,
        "row_indices": indices, "n_observations": len(timings),
        "total_rows_processed": rows * len(timings), "latencies_ms": timings,
        "p50_batch_ms": float(np.percentile(values, 50)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "throughput_rows_per_second": rows * len(timings) / (float(values.sum()) / 1000),
        "throughput_per_repeat": [rows / (value / 1000) for value in timings],
    }


def benchmark_predictor(predict_fn, frame, *, batch_size=128, repeats=3, single_rows=64, warmup=8):
    """Measure raw-frame preprocessing plus synchronous CPU scoring.

    The caller must provide CPU inference in its own worker, configure every
    estimator/BLAS/Torch worker count to one, and avoid competing benchmark
    jobs. This function cannot inspect an opaque callable's device or threads.
    Row selection and copying occur before timers start. Warmup runs once per
    mode before both timing and RSS sampling; external model loading is not
    measured. RSS includes the entire process, sampled every 10 ms during the
    timed section, and can miss allocation peaks between samples.
    """
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("benchmark requires a nonempty raw DataFrame")
    batch_size = _positive_int(batch_size, "batch_size")
    repeats = _positive_int(repeats, "repeats")
    single_rows = _positive_int(single_rows, "single_rows")
    warmup = _positive_int(warmup, "warmup", minimum=0)
    single_indices = _indices(len(frame), single_rows)
    batch_indices = _indices(len(frame), batch_size)
    full_indices = _indices(len(frame), 512)
    single_inputs = [frame.iloc[[index]].copy() for index in single_indices]
    batch = frame.iloc[batch_indices].copy()
    full_batch = frame.iloc[full_indices].copy()
    for index in range(warmup):
        predict_fn(single_inputs[index % len(single_inputs)])
        predict_fn(batch)
        predict_fn(full_batch)

    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    samples = [rss_before]
    sampling_errors = []
    stop = threading.Event()

    def sample_rss():
        while not stop.wait(.01):
            try:
                samples.append(int(process.memory_info().rss))
            except (psutil.Error, OSError) as exc:
                sampling_errors.append(type(exc).__name__)
                break

    sampler = threading.Thread(target=sample_rss, name="route-batch-rss", daemon=True)
    sampler.start()
    single_blocks, batch_ms, full_ms = [], [], []
    try:
        for _ in range(repeats):
            single_blocks.append([_elapsed_ms(predict_fn, row) for row in single_inputs])
        for _ in range(repeats):
            batch_ms.append(_elapsed_ms(predict_fn, batch))
        for _ in range(repeats):
            full_ms.append(_elapsed_ms(predict_fn, full_batch))
    finally:
        stop.set()
        sampler.join()
    rss_after = int(process.memory_info().rss)
    samples.append(rss_after)
    single_ms = [value for block in single_blocks for value in block]
    return {
        "scope": "hot_cpu_raw_dataframe_preprocessing_and_scoring",
        "device_contract": "caller supplies synchronous CPU inference in an isolated worker",
        "thread_contract": "caller sets estimator, BLAS and Torch threads to one; no competing benchmark jobs",
        "file_io_included": False, "input_slicing_included": False,
        "cold_model_load_measured": False, "ui_included": False,
        "warmup_calls_per_mode": warmup, "warmup_included_in_timing": False,
        "warmup_included_in_rss": False, "repeats": repeats,
        "single": {
            "batch_size": 1, "n_unique_rows": len(single_indices), "row_indices": single_indices,
            "n_observations": len(single_ms), "total_rows_processed": len(single_ms),
            "p50_ms": float(np.percentile(single_ms, 50)),
            "p95_ms": float(np.percentile(single_ms, 95)),
            "latencies_ms": single_ms, "blocks_ms": single_blocks,
        },
        "batch": _batch_report(batch_ms, batch_indices, batch_size),
        "batch_512": _batch_report(full_ms, full_indices, 512),
        "counts": {
            "warmup_calls": 3 * warmup,
            "timed_calls": repeats * (len(single_indices) + 2),
            "timed_rows": repeats * (len(single_indices) + len(batch_indices) + len(full_indices)),
        },
        "rss": {
            "before_bytes": rss_before, "after_bytes": rss_after,
            "sampled_peak_bytes": max(samples), "peak_bytes": max(samples),
            "sampling_interval_ms": 10, "samples": len(samples),
            "sampling_errors": sampling_errors,
            "scope": "whole isolated-worker process during timed hot inference, after warmup",
            "measurement": "sampled RSS including process baseline; not model-only memory or exact allocation peak",
        },
        "statistical_scope": "timing repetitions on fixed rows, not independent network samples",
    }
