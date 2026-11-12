"""One-core model and threshold cost on already materialized input features."""

from time import perf_counter_ns

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_info, threadpool_limits

from ids.common import read_json, sha256_file, write_json
from ids.contact_evidence import experiment as previous
from ids.flow_history import trees
from .experiment import ARMS, ARM_NAMES


def _sealed_flags(path, ids):
    output = np.empty(len(ids), dtype=np.uint16)
    pos = 0
    found = 0
    with pq.ParquetFile(path) as file:
        for batch in file.iter_batches(columns=["flags"], batch_size=65536):
            end = pos + len(batch)
            stop = ids.searchsorted(end, side="right")
            if stop != found:
                output[found:stop] = batch.column(0).to_numpy()[ids[found:stop] - pos - 1]
                found = stop
            pos = end
            if found == len(ids):
                break
    if found != len(ids):
        raise ValueError("Cost sample is absent from sealed flags")
    return output


def cost(root, name):
    """Measure contact/quality on identical sealed rows after all training/scoring jobs exit."""
    folder = root / "evidence/blind" / name
    group = "hgb"
    arms = ("contact", "quality")
    saved = pd.read_parquet(folder / (group + "-sample.parquet"), columns=["row_id", "flags", *arms])
    if saved.empty or not np.array_equal(saved.row_id, np.arange(len(saved)) * 4096 + 1):
        raise ValueError("Unexpected sealed cost sample identity")
    indices = np.linspace(0, len(saved) - 1, min(96, len(saved)), dtype=int)
    sample = saved.iloc[indices]
    ids = sample.row_id.to_numpy()
    flags = _sealed_flags(folder / (group + ".parquet"), ids)
    if not np.array_equal(sample["flags"], flags):
        raise ValueError("Sealed score sample disagrees with the full decision stream")
    frame = pd.read_parquet(
        previous.enriched(root, name),
        columns=["row_id", *ARMS["quality"]],
        filters=[("row_id", "in", ids.tolist())],
    ).sort_values("row_id")
    if not np.array_equal(frame.row_id, ids):
        raise ValueError("Missing or duplicate cost sample row")
    rules = read_json(root / "evidence/calibration/hgb.json")
    matrices, estimators, cuts, results = {}, {}, {}, {}
    for arm in arms:
        model_file = previous.artifact(root, group, arm)
        model = trees.load_arm(model_file, threads=1)
        if tuple(model.features) != tuple(ARMS[arm]):
            raise ValueError("Cost model feature order differs from registered arm")
        matrices[arm] = np.ascontiguousarray(frame[list(model.features)], dtype=np.float64)
        estimators[arm] = model.estimator
        cuts[arm] = rules[arm]["threshold"]
        results[arm] = {
            "model_artifact_bytes": model_file.stat().st_size,
            "model_sha256": sha256_file(model_file),
            "latencies_ns": [],
            "features": list(model.features),
        }

    actual_scores = {arm: np.empty(len(ids)) for arm in arms}
    actual_flags = {arm: np.empty(len(ids), dtype=bool) for arm in arms}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        for arm in arms:
            estimators[arm].decision_function(matrices[arm][:1])
        # Alternate order to avoid always timing one arm on a colder CPU.
        for i in range(len(ids)):
            for arm in arms if i % 2 == 0 else arms[::-1]:
                vector = matrices[arm][i : i + 1]
                estimator, cut = estimators[arm], cuts[arm]
                start = perf_counter_ns()
                score = float(estimator.decision_function(vector)[0])
                alert = cut is not None and score > cut
                elapsed = perf_counter_ns() - start
                actual_scores[arm][i], actual_flags[arm][i] = score, alert
                results[arm]["latencies_ns"].append(elapsed)
    for arm in arms:
        ref = sample[arm].to_numpy()
        expected_flags = ((flags >> ARM_NAMES.index(arm)) & 1).astype(bool)
        if not np.array_equal(actual_scores[arm], ref) or not np.array_equal(
            actual_flags[arm], expected_flags
        ):
            raise ValueError(f"Single-row {arm} score or decision differs from the sealed batch")
        latency = np.asarray(results[arm]["latencies_ns"]) / 1e6
        results[arm].update(
            {
                "p50_ms": float(np.median(latency)),
                "p95_ms": float(np.quantile(latency, 0.95)),
                "exact_sealed_scores": len(ids),
                "exact_sealed_decisions": len(ids),
            }
        )
    path = root / "evidence/cost" / name / "hgb.json"
    write_json(
        path,
        {
            "capture": name,
            "rows": len(ids),
            "row_ids": ids.tolist(),
            "device": "cpu",
            "threads": 1,
            "native_threadpools": pools,
            "arms": results,
            "timed_scope": "One precomputed float64 feature vector through fitted HGB decision_function and its frozen threshold. Warm model. Excludes parsing, response classification, history building, disk I/O and feature materialization.",
        },
    )
    return [path]
