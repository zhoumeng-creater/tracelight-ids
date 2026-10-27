"""CPU benchmarks for preprocessing plus an opaque prediction callable.

The caller must supply CPU inference, including its preprocessing, and configure
model-specific worker counts. These timings exclude file parsing, input slicing,
thread configuration and UI/network work. They are not pure-model timings.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits


def _integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _rows(X: Any, start: int, stop: int) -> Any:
    return X.iloc[start:stop] if hasattr(X, "iloc") else X[start:stop]


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _elapsed_ms(predict_fn: Callable[[Any], Any], values: Any) -> float:
    start = time.perf_counter_ns()
    predict_fn(values)
    elapsed = time.perf_counter_ns() - start
    if elapsed <= 0:
        raise RuntimeError("benchmark timer did not advance; increase timer resolution")
    return elapsed / 1_000_000


def benchmark_predict(
    predict_fn: Callable[[Any], Any],
    X: Any,
    threads: int = 1,
    warmup: int = 5,
    repeats: int = 30,
    single_rows: int = 100,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Measure real batch=1 latency separately from fixed-batch throughput.

    Each mode gets ``warmup`` calls. ``repeats`` complete blocks measure up to
    ``single_rows`` evenly spaced records individually, followed by repeated
    calls on one batch of min(batch_size, n_rows) records. Keep raw timings;
    repeated calls are timing observations, not independent network samples.
    """
    threads = _integer("threads", threads, 1)
    warmup = _integer("warmup", warmup, 0)
    repeats = _integer("repeats", repeats, 1)
    single_rows = _integer("single_rows", single_rows, 1)
    batch_size = _integer("batch_size", batch_size, 1)
    n_rows = int(X.shape[0]) if hasattr(X, "shape") else len(X)
    if n_rows == 0:
        raise ValueError("benchmark requires at least one input row")
    n_single = min(single_rows, n_rows)
    indices = [(i * n_rows) // n_single for i in range(n_single)]
    single_inputs = [_rows(X, index, index + 1) for index in indices]
    n_batch = min(batch_size, n_rows)
    batch = _rows(X, 0, n_batch)
    single_ms: list[float] = []
    single_blocks_ms: list[list[float]] = []
    batch_ms: list[float] = []
    rss_before = _rss_bytes()
    with threadpool_limits(limits=threads):
        for i in range(warmup):
            predict_fn(single_inputs[i % n_single])
            predict_fn(batch)
        for _ in range(repeats):
            block = [_elapsed_ms(predict_fn, row) for row in single_inputs]
            single_blocks_ms.append(block)
            single_ms.extend(block)
        for _ in range(repeats):
            batch_ms.append(_elapsed_ms(predict_fn, batch))
    rss_after = _rss_bytes()
    return {
        "scope": "preprocessing_plus_predict_fn",
        "device": "cpu",
        "device_contract": "caller must supply CPU-only inference; an opaque callable cannot be inspected for GPU use",
        "threads": threads,
        "thread_control": "threadpoolctl BLAS/OpenMP limits; caller controls estimator workers and torch threads",
        "warmup": warmup,
        "warmup_scope": "calls per mode; excluded from timings",
        "repeats": repeats,
        "input_slicing_included": False,
        "file_io_included": False,
        "single": {
            "batch_size": 1,
            "n_unique_rows": n_single,
            "row_indices": indices,
            "n_observations": len(single_ms),
            "p50_ms": float(np.percentile(single_ms, 50)),
            "p95_ms": float(np.percentile(single_ms, 95)),
            "latencies_ms": single_ms,
            "blocks_ms": single_blocks_ms,
            "block_p50_ms": [float(np.median(block)) for block in single_blocks_ms],
        },
        "batch": {
            "requested_batch_size": batch_size,
            "batch_size": n_batch,
            "n_rows": n_batch,
            "n_observations": len(batch_ms),
            "total_rows_processed": n_batch * repeats,
            "p50_batch_ms": float(np.percentile(batch_ms, 50)),
            "p95_batch_ms": float(np.percentile(batch_ms, 95)),
            "latencies_ms": batch_ms,
            "throughput_rows_per_second": n_batch * repeats / (sum(batch_ms) / 1000),
            "throughput_per_repeat": [n_batch / (elapsed / 1000) for elapsed in batch_ms],
        },
        "rss": {
            "before_bytes": rss_before,
            "after_bytes": rss_after,
            "peak_bytes": None,
            "measurement": "process RSS snapshots only; not peak RSS or model-attributable memory",
        },
        "statistical_scope": "timing repetitions on fixed inputs; not independent network samples or a confidence interval",
    }
