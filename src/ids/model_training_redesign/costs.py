"""Isolated end-to-end, preprocessing and pretransformed-model timings."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ids.common import read_json, write_json
from ids.route_batch.benchmark import benchmark_predictor
from . import data
from .protocol import verify_live_code


def measure(root, task_id, backend=None):
    from .runner import aligned_scores, contained, verify_inputs, verify_freeze
    if backend is None:
        from . import training as backend
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_inputs(root, protocol)
    frozen = verify_freeze(root, protocol)
    artifact = contained(root, frozen["artifacts"][task_id]["path"])
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        predictor = backend.load_predictor(artifact, threads=1, device="cpu")
        load_seconds = time.perf_counter() - started
        frame = pd.read_parquet(root / "inputs/ED.parquet")
        raw = data.blind(frame).reset_index(drop=True)
        expected = aligned_scores(root, task_id, "ED", frame)
        replay = predictor.score(raw)
        if not np.array_equal(replay, expected):
            raise ValueError(f"Cold CPU replay differs from frozen scores: {task_id}")
        spec = protocol["registration"]["benchmark"]
        full_cost = benchmark_predictor(predictor.score, raw, **spec)
        preprocessing_cost = benchmark_predictor(predictor.preprocessor.transform, raw, **spec)
        numeric, categorical = predictor.preprocessor.transform(raw)

        def transformed(batch):
            indices = batch.index.to_numpy(dtype=np.int64)
            return predictor.score_transformed(numeric[indices], categorical[indices])

        if not np.array_equal(transformed(raw), expected):
            raise ValueError("Pretransformed model scoring changes the frozen CPU score contract")
        model_cost = benchmark_predictor(transformed, raw, **spec)
        model_cost["scope"] = "pretransformed numeric/category row gather and CPU scoring; no preprocessing"
        preprocessing_cost["scope"] = "CPU raw-dataframe preprocessing only; no model"
        indices = [(i * len(raw)) // min(64, len(raw)) for i in range(min(64, len(raw)))]
        singles = np.array([predictor.score(raw.iloc[[i]])[0] for i in indices])
        boundary = read_json(root / f"evidence/{task_id}/calibration.json")["threshold"]
        changed = [int(i) for i, a, b in zip(indices, singles, expected[indices])
                   if (a > boundary) != (b > boundary)]
        result = {"model_id": task_id, "device": "cpu", "threads": 1,
            "cold_artifact_load_seconds": load_seconds,
            "cold_scope": "artifact/preprocessor load; Python startup excluded; OS file cache uncontrolled",
            "model_artifact_bytes": artifact.stat().st_size, "cold_scores_exact": True,
            "pretransformed_scores_exact": True, "preprocessing_included": True,
            "single_query_max_abs_score_delta": float(np.max(np.abs(singles - expected[indices]))),
            "single_query_changed_decision_indices": changed,
            "timing": full_cost, "preprocessing_only_timing": preprocessing_cost,
            "model_only_timing": model_cost,
            "tail_interpretation": "raw repeats retained; three batch repeats are descriptive"}

        import torch
        gpu_device = torch.device(protocol["device"])
        gpu_spec = protocol["registration"]["gpu_benchmark"]
        count = min(gpu_spec["batch_size"], len(raw))
        indices = [(i * len(raw)) // count for i in range(count)]
        batch = raw.iloc[indices].copy()
        torch.cuda.synchronize(gpu_device)
        torch.cuda.reset_peak_memory_stats(gpu_device)
        started = time.perf_counter()
        gpu = backend.load_predictor(artifact, threads=1, device=protocol["device"])
        torch.cuda.synchronize(gpu_device)
        gpu_load = time.perf_counter() - started
        for _ in range(gpu_spec["warmup"]):
            gpu.score(batch)
        torch.cuda.synchronize(gpu_device)
        times = []
        for _ in range(gpu_spec["repeats"]):
            started = time.perf_counter()
            scores = gpu.score(batch)
            torch.cuda.synchronize(gpu_device)
            times.append(time.perf_counter() - started)
        result["gpu_throughput"] = {
            "device": protocol["device"], "batch_rows": count, "row_indices": indices,
            "repeats": len(times), "seconds_per_repeat": times,
            "rows_per_second": count * len(times) / sum(times), "artifact_load_seconds": gpu_load,
            "preprocessing_and_host_device_copies_included": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(gpu_device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(gpu_device),
            "max_abs_delta_from_registered_cpu": float(np.max(np.abs(scores - expected[indices]))),
            "changed_decision_indices_at_cpu_threshold": [int(i) for i, a, b in
                zip(indices, scores, expected[indices]) if (a > boundary) != (b > boundary)],
            "scope": "descriptive GPU throughput; qualification uses frozen CPU scores"}
    verify_live_code(protocol)
    write_json(root / f"cost/{task_id}.json", result)
    return result
