"""Costs in one isolated worker; scores and threshold are already frozen.

The runner owns subprocess isolation, serial scheduling, source-code checks and
the final adoption AND. This module never recalibrates, trains or chooses a model.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ids.common import write_json
from ids.model_training_redesign.data import blind
from ids.route_batch.benchmark import benchmark_predictor


CPU_BENCHMARK = {"batch_size": 128, "repeats": 3, "single_rows": 64, "warmup": 8}
GPU_BENCHMARK = {"batch_size": 512, "repeats": 3, "warmup": 8}
IDENTITY = ["source_id", "row_id"]
SCORE_BATCH = 256


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scores(values, length):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("Expected aligned finite score vector")
    return result


def _expected_scores(frame, registered_scores):
    if frame.empty or registered_scores.empty:
        raise ValueError("Cost input and registered scores must be nonempty")
    for value in (frame, registered_scores):
        if (not set(IDENTITY).issubset(value.columns) or value[IDENTITY].isna().any().any()
                or value.duplicated(IDENTITY).any()):
            raise ValueError("Cost inputs require unique nonmissing physical identities")
    if "score" not in registered_scores:
        raise ValueError("Frozen evaluation score column is required")
    expected = registered_scores.set_index(IDENTITY)
    index = frame.set_index(IDENTITY).index
    if len(expected) != len(frame) or set(expected.index) != set(index):
        raise ValueError("Cost and frozen evaluation identities differ")
    return _scores(expected.reindex(index).score.to_numpy(), len(frame))


def _indices(length, requested):
    count = min(length, requested)
    return [(i * length) // count for i in range(count)]


def _identity_rows(frame, positions):
    return [{"source_id": str(frame.iloc[i].source_id), "row_id": int(frame.iloc[i].row_id)}
            for i in positions]


def measure_artifact(*, task_id, artifact, artifact_sha256, frame, registered_scores, threshold,
                     gpu_device="cuda:0", output_path=None, loader=None):
    """Measure one frozen model, using caller-bound evaluation frames and threshold.

    loader(artifact, threads=1, device=...) must return a predictor with score,
    score_transformed, preprocessor.transform and batch_size=256. The default
    loader is the new training package's artifact reader, imported only at run.
    """
    artifact = Path(artifact).resolve()
    if _sha(artifact) != artifact_sha256:
        raise ValueError("Frozen model artifact hash mismatch")
    if isinstance(threshold, bool) or not np.isfinite(float(threshold)):
        raise ValueError("A finite frozen calibration threshold is required")
    threshold = float(threshold)
    if not str(gpu_device).startswith("cuda"):
        raise ValueError("GPU throughput requires the registered CUDA device")
    if output_path is not None and Path(output_path).exists():
        raise ValueError("Cost output already exists; measurements are not automatically repeated")
    expected = _expected_scores(frame, registered_scores)
    # Preserve evaluation row order and strip labels, identities, and source metadata.
    raw = blind(frame).reset_index(drop=True)
    if loader is None:
        from .training import load_predictor
        loader = load_predictor
    with threadpool_limits(limits=1):
        started = time.perf_counter()
        predictor = loader(artifact, threads=1, device="cpu")
        load_seconds = time.perf_counter() - started
        if predictor.batch_size != SCORE_BATCH:
            raise ValueError("Formal CPU scoring must use internal batch size 256")
        replay = _scores(predictor.score(raw), len(raw))
        if replay.tobytes() != expected.tobytes():
            raise ValueError(f"Cold CPU replay differs from frozen evaluation scores: {task_id}")
        numeric, categorical = predictor.preprocessor.transform(raw)

        def transformed(batch):
            indices = batch.index.to_numpy(dtype=np.int64)
            return predictor.score_transformed(numeric[indices], categorical[indices])

        pretransformed = _scores(transformed(raw), len(raw))
        if pretransformed.tobytes() != expected.tobytes():
            raise ValueError("Pretransformed CPU scoring changes the frozen score contract")
        full_cost = benchmark_predictor(predictor.score, raw, **CPU_BENCHMARK)
        preprocessing_cost = benchmark_predictor(predictor.preprocessor.transform, raw, **CPU_BENCHMARK)
        model_cost = benchmark_predictor(transformed, raw, **CPU_BENCHMARK)
        preprocessing_cost["scope"] = "CPU raw-dataframe preprocessing only; no model"
        model_cost["scope"] = "pretransformed numeric/category row gather and CPU scoring; no preprocessing"

        indices = _indices(len(raw), 64)
        singles = _scores([predictor.score(raw.iloc[[i]])[0] for i in indices], len(indices))
        changed = [int(i) for i, score, original in zip(indices, singles, expected[indices])
                   if (score > threshold) != (original > threshold)]
        result = {
            "model_id": task_id, "artifact_sha256": artifact_sha256,
            "model_artifact_bytes": artifact.stat().st_size,
            "device": "cpu", "threads": 1, "internal_score_batch_size": SCORE_BATCH,
            "cold_artifact_load_seconds": load_seconds,
            "cold_scope": "artifact/preprocessor load; Python startup excluded; OS file cache uncontrolled",
            "cold_scores_exact": True, "pretransformed_scores_exact": True,
            "preprocessing_included": True,
            "frozen_threshold": threshold, "decision_rule": "score_strictly_greater_than_threshold",
            "single_query_checked_rows": len(indices), "single_query_row_indices": indices,
            "single_query_identities": _identity_rows(frame, indices),
            "single_query_max_abs_score_delta": float(np.max(np.abs(singles - expected[indices]))),
            "single_query_changed_decision_indices": changed,
            "single_query_changed_decision_identities": _identity_rows(frame, changed),
            "deployment_contract_pass": not changed,
            "deployment_contract_scope": "exact full CPU batch256 replay plus fixed sampled single-row decisions",
            "timing": full_cost, "preprocessing_only_timing": preprocessing_cost,
            "model_only_timing": model_cost,
            "tail_interpretation": "raw repeats retained; three batch repeats are descriptive",
            "worker_contract": "caller runs this function in its own isolated, serial cost subprocess",
        }

        import torch
        device = torch.device(gpu_device)
        gpu_indices = _indices(len(raw), GPU_BENCHMARK["batch_size"])
        batch = raw.iloc[gpu_indices].copy()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        gpu = loader(artifact, threads=1, device=gpu_device)
        torch.cuda.synchronize(device)
        gpu_load = time.perf_counter() - started
        if gpu.batch_size != SCORE_BATCH:
            raise ValueError("GPU descriptive scoring must retain internal batch size 256")
        for _ in range(GPU_BENCHMARK["warmup"]):
            gpu.score(batch)
        torch.cuda.synchronize(device)
        times, last_scores = [], None
        for _ in range(GPU_BENCHMARK["repeats"]):
            started = time.perf_counter()
            last_scores = _scores(gpu.score(batch), len(batch))
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if elapsed <= 0:
                raise RuntimeError("GPU inference timer did not advance")
            times.append(elapsed)
        gpu_changed = [int(i) for i, score, original in zip(gpu_indices, last_scores, expected[gpu_indices])
                       if (score > threshold) != (original > threshold)]
        result["gpu_throughput"] = {
            "device": str(device), "device_name": torch.cuda.get_device_name(device),
            "batch_rows": len(batch), "requested_batch_rows": GPU_BENCHMARK["batch_size"],
            "internal_score_batch_size": SCORE_BATCH, "row_indices": gpu_indices,
            "warmup": GPU_BENCHMARK["warmup"], "repeats": len(times), "seconds_per_repeat": times,
            "rows_per_second": len(batch) * len(times) / sum(times), "artifact_load_seconds": gpu_load,
            "preprocessing_and_host_device_copies_included": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "max_abs_delta_from_registered_cpu": float(np.max(np.abs(last_scores - expected[gpu_indices]))),
            "changed_decision_indices_at_cpu_threshold": gpu_changed,
            "changed_decision_identities_at_cpu_threshold": _identity_rows(frame, gpu_changed),
            "scope": "descriptive GPU throughput; deployment qualification uses frozen CPU scoring",
        }
    if _sha(artifact) != artifact_sha256:
        raise ValueError("Frozen model artifact changed during cost measurement")
    if output_path is not None:
        write_json(Path(output_path), result)
    return result
