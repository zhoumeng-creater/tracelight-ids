"""Single-native-thread HGB decision latency on materialized, sealed R24 examples."""

import json
from time import perf_counter_ns

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from ids.common import sha256_file, write_json
from ids.flow_history import trees
from ids.response_evidence.experiment import ARMS, FULL
from .scoring import POLICIES, SCORE_COLUMNS, VIEWS, make_decisions, register

MEASURED = ("raw", "add_web64k", "add_contact", "add_quality")
REPEATS = 3


def _sample(path):
    rows = pd.read_parquet(path, columns=["row_id", *POLICIES, *SCORE_COLUMNS, *FULL]).sort_values("row_id")
    if rows.empty or rows.row_id.isna().any() or rows.row_id.duplicated().any():
        raise ValueError("Missing or duplicate sealed cost sample identity")
    indices = np.linspace(0, len(rows) - 1, min(96, len(rows)), dtype=int)
    return rows.iloc[indices]


def measure(root):
    source = root / "evidence/hgb-examples.parquet"
    sample = _sample(source)
    base = root / "local/controls"
    rules, supplements = register(base / "evidence/calibration", "hgb")
    models, matrices, saved_models = {}, {}, {}
    for view in VIEWS:
        file = base / "models/frozen" / ("hgb-" + view + ".joblib")
        model = trees.load_arm(file, threads=1)
        if model.features != tuple(ARMS[view]):
            raise ValueError(f"Frozen HGB input order differs for {view}")
        models[view] = model.estimator
        matrices[view] = np.ascontiguousarray(sample[list(model.features)], dtype=np.float64)
        saved_models[view] = {"file": file.name, "bytes": file.stat().st_size, "sha256": sha256_file(file)}
    registered = {
        "raw": {"raw_strict_cut": rules["raw"]["threshold"]},
        "add_web64k": {
            "raw_strict_cut": rules["raw"]["threshold"],
            "ip_protocol": 6,
            "resp_port": 80,
            "min_resp_ip_bytes": 65536,
        },
        **{
            "add_" + v: {
                "raw_strict_cut": rules["raw"]["threshold"],
                v + "_residual_strict_cut": supplements[v]["threshold"],
            }
            for v in ("contact", "quality")
        },
    }
    protocols, ports, sizes = (sample[k].to_numpy() for k in ("ip_protocol", "resp_port", "resp_ip_bytes"))
    raw_cut = rules["raw"]["threshold"]

    def predict(policy, i):
        raw_score = models["raw"].decision_function(matrices["raw"][i : i + 1])[0]
        raw = raw_cut is not None and raw_score > raw_cut
        if policy == "raw":
            return raw
        if policy == "add_web64k":
            return raw or (protocols[i] == 6 and ports[i] == 80 and sizes[i] >= 65536)
        helper = "contact" if policy == "add_contact" else "quality"
        helper_score = models[helper].decision_function(matrices[helper][i : i + 1])[0]
        return raw or helper_score > supplements[helper]["threshold"]

    times = {p: [] for p in MEASURED}
    sealed = {p: sample[p].to_numpy() for p in POLICIES}
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        replay = {
            view: np.asarray([models[view].decision_function(x[i : i + 1])[0] for i in range(len(sample))])
            for view, x in matrices.items()
        }
        for view in VIEWS:
            if not np.array_equal(replay[view], sample["score_" + view]):
                raise ValueError(f"Single-row HGB {view} score differs from the sealed score")
        for policy, flags in make_decisions(replay, rules, supplements, sample).items():
            if not np.array_equal(sealed[policy], flags):
                raise ValueError(f"Sealed decision differs for {policy}")
        for policy in MEASURED:
            for i in range(min(4, len(sample))):
                predict(policy, i)
        for repeat in range(REPEATS):
            for i in range(len(sample)):
                # Rotate the starting position so each policy runs at different points in the measurement.
                shift = (i + repeat) % len(MEASURED)
                for policy in MEASURED[shift:] + MEASURED[:shift]:
                    start = perf_counter_ns()
                    alert = predict(policy, i)
                    elapsed = perf_counter_ns() - start
                    if alert != sealed[policy][i]:
                        raise ValueError(f"Timed {policy} decision differs at row {sample.row_id.iloc[i]}")
                    times[policy].append(elapsed)
    results = {}
    for policy, latencies in times.items():
        helper = policy.removeprefix("add_")
        views = ["raw", helper] if helper in ("contact", "quality") else ["raw"]
        model_bytes = sum(saved_models[v]["bytes"] for v in views)
        rule = registered[policy]
        descriptor = json.dumps(rule, sort_keys=True, separators=(",", ":")).encode("utf-8")
        latency = np.asarray(latencies) / 1e6
        results[policy] = {
            "p50_ms": float(np.median(latency)),
            "p95_ms": float(np.quantile(latency, 0.95)),
            "timed_single_record_decisions": len(latencies),
            "exact_sealed_decisions": len(sample),
            "models": views,
            "combined_model_artifact_bytes": model_bytes,
            "rule_descriptor": rule,
            "compact_json_descriptor_bytes": len(descriptor),
            "model_and_descriptor_bytes": model_bytes + len(descriptor),
            "needs_history": len(views) == 2,
        }
    path = root / "evidence/decision-cost.json"
    write_json(
        path,
        {
            "sample": "Evenly spaced by row-id order within the sealed, stratified examples",
            "rows": len(sample),
            "row_ids": sample.row_id.tolist(),
            "sample_file_sha256": sha256_file(source),
            "repeats": REPEATS,
            "warmup_per_policy": min(4, len(sample)),
            "device": "cpu",
            "native_threads": 1,
            "native_threadpools": pools,
            "models": saved_models,
            "policies": results,
            "verified": {
                "score_columns": list(SCORE_COLUMNS),
                "decision_columns": list(POLICIES),
                "exact_rows_per_column": len(sample),
            },
            "timed_scope": "One already-materialized numeric record, HGB decision_function(s), strict frozen cut(s) and Boolean rule. Counts both HGB evaluations for additive model policies.",
            "excluded": [
                "input parsing",
                "feature materialization",
                "response classification",
                "history construction",
                "disk I/O",
                "model load",
                "runtime installation",
            ],
            "observed_full_history_build_report": "evidence/history.json",
        },
    )
    return [path]
