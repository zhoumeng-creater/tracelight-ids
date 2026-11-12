"""Fixed multicore hist-gradient-boosting control for the two information sets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits

from ids.common import sha256_file, write_json
from .arm_cost import benchmark_arm, checked_features


FORMAT = "flow-history-hgb-v1"
# Same fixed HGB budget as configs/controlled-features.yaml and research-v2.yaml.
PARAMETERS = {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.1,
              "early_stopping": False, "random_state": 20261111}


@dataclass
class TreeArm:
    estimator: HistGradientBoostingClassifier
    features: tuple[str, ...]
    threads: int
    pools: ThreadpoolController


def _threads(value):
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
        raise ValueError("threads must be a positive integer")
    return int(value)


def _matrix(frame, names):
    values = np.ascontiguousarray(frame.loc[:, list(names)].to_numpy(dtype=np.float64))
    if values.shape != (len(frame), len(names)) or not np.isfinite(values).all():
        raise ValueError("HGB input must have one finite numeric column for each registered feature")
    return values


def fit_arm(p, features, path, *, threads, sample_weight=None):
    names = checked_features(features)
    threads = _threads(threads)
    labels = p["label"].to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("training P must contain both binary labels")
    values = _matrix(p, names)
    classifier = HistGradientBoostingClassifier(**PARAMETERS)
    start = time.perf_counter()
    with threadpool_limits(limits=threads):
        pools = threadpool_info()
        classifier.fit(values, labels, sample_weight=sample_weight)
        train_seconds = time.perf_counter() - start
        probe = values[:512]
        start = time.perf_counter()
        out = classifier.decision_function(probe)
        probe_seconds = time.perf_counter() - start
        if not np.isfinite(out).all():
            raise FloatingPointError("HGB probe returned nonfinite scores")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump({"format": FORMAT, "features": list(names), "estimator": classifier}, tmp)
    tmp.replace(path)
    write_json(path.with_suffix(".fit.json"), {
        "format": FORMAT, "features": list(names), "rows": len(p), "positive_rows": int(np.sum(labels)),
        "parameters": classifier.get_params(), "iterations_fitted": classifier.n_iter_, "device": "cpu",
        "threads": threads, "native_threadpools": pools, "train_seconds": train_seconds,
        "training_rows_per_second": len(p)/train_seconds, "model_probe_rows": len(probe),
        "model_probe_seconds": probe_seconds, "model_probe_rows_per_second": len(probe)/probe_seconds,
        "score_contract": "binary HGB decision_function; selected raw finite numeric fields; "
                          "all port/protocol codes are numeric; fitted bins and trees use only P",
        "model_sha256": sha256_file(path), "model_artifact_bytes": path.stat().st_size,
    })
    return path


def load_arm(path, *, threads):
    state = joblib.load(path)
    estimator = state["estimator"]
    if state["format"] != FORMAT or not isinstance(estimator, HistGradientBoostingClassifier):
        raise ValueError("unrecognized flow-history HGB model")
    names = checked_features(state["features"])
    if estimator.n_features_in_ != len(names) or not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("HGB feature width or class order does not match its schema")
    return TreeArm(estimator, names, _threads(threads), ThreadpoolController())


def score_arm(model, frame, features):
    if tuple(features) != model.features:
        raise ValueError("features or feature order differ from the fitted arm")
    x = _matrix(frame, model.features)
    with model.pools.limit(limits=model.threads):
        result = np.asarray(model.estimator.decision_function(x), dtype=np.float64)
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError("invalid HGB scores")
    return result


def bench_arm(path, C, E, features, threads, *, reference_c, reference_e, output_dir):
    return benchmark_arm("hgb", path, C, E, features, threads, reference_c=reference_c,
                         reference_e=reference_e, output_dir=output_dir)
