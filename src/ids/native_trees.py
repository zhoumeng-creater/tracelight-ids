"""Native categorical trees as direct baselines and optional soft-target students.

Fitting takes F only. Selection and threshold registration belong to the caller.
Hard and soft students use the same CrossEntropy objective and hard-label weights.
This is an adapter for official CatBoost, not a new distillation algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.utils.class_weight import compute_sample_weight

from .common import read_json, sha256_file, write_json
from .models import _check_frame, _labels, _numeric
from .selection import normalize_feature_schema

CATBOOST_VERSION = "1.2.10"


def _catboost():
    try:
        import catboost
    except ImportError as exc:
        raise ImportError("Native tree baselines require catboost==1.2.10 in the research environment") from exc
    if version("catboost") != CATBOOST_VERSION:
        raise ValueError("Native tree protocol requires catboost==1.2.10")
    return catboost


def native_frame(frame, selected_features, categorical):
    """No fitted transform or label lookup; native quantization/CTRs fit inside F."""
    _check_frame(frame, selected_features, allow_empty=True)
    numeric = [f for f in selected_features if f not in categorical]
    result = pd.DataFrame(index=frame.index)
    if numeric:
        result[numeric] = _numeric(frame, numeric)
    for feature in categorical:
        values = frame[feature].astype("string")
        # Distinguish a missing category from any literal string in the data.
        result[feature] = ("v:" + values).fillna("m:").astype(object)
    return result.loc[:, selected_features]


def mixed_targets(labels, teacher_probabilities=None, kd_weight=0.0):
    """T=1: BCE(y,p)+lambda KL(q||p) has gradient (1+lambda) BCE(target,p).

    We normalize the target objective; its minimizer agrees, its gradient scale
    differs. A tree boosting trajectory need not equal the unnormalized loss.
    """
    y = np.asarray(labels, dtype=np.float64)
    if (y.ndim != 1 or not np.isfinite(y).all() or not np.isin(y, [0., 1.]).all()):
        raise ValueError("binary hard labels required")
    if isinstance(kd_weight, bool) or not np.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("kd_weight must be finite and nonnegative")
    if teacher_probabilities is None:
        if kd_weight != 0:
            raise ValueError("positive kd_weight requires aligned teacher probabilities")
        return y.copy()
    q = np.asarray(teacher_probabilities, dtype=np.float64)
    if q.shape != y.shape or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("teacher probabilities must align with F and lie in [0,1]")
    return (y + kd_weight * q) / (1 + kd_weight)


@dataclass
class NativeTreeBundle:
    estimator: object
    selected_features: list[str]
    categorical_columns: list[str]
    feature_schema: dict
    training_statistics: dict
    num_threads: int

    def predict_raw(self, frame):
        data = native_frame(frame, self.selected_features, self.categorical_columns)
        if data.empty:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self.estimator.predict(data, prediction_type="RawFormulaVal",
                          thread_count=self.num_threads), dtype=np.float64).reshape(-1)

    def predict_scores(self, frame):
        return expit(self.predict_raw(frame))

    def save(self, output_dir):
        root = Path(output_dir)
        model_path, metadata_path = root / "model.cbm", root / "metadata.json"
        if model_path.exists() or metadata_path.exists():
            raise FileExistsError("existing native-tree model is preserved")
        root.mkdir(parents=True, exist_ok=True)
        self.estimator.save_model(str(model_path), format="cbm")
        write_json(metadata_path, {"schema_version": 1, "catboost_version": CATBOOST_VERSION,
            "selected_features": self.selected_features, "categorical_columns": self.categorical_columns,
            "feature_schema": self.feature_schema, "training_statistics": self.training_statistics,
            "num_threads": self.num_threads, "model_sha256": sha256_file(model_path),
            "model_bytes": model_path.stat().st_size,
            "category_encoding": "nonmissing=v:+string; missing=m:",
            "prediction": "native float64 RawFormulaVal; sigmoid scores are not calibrated probabilities"})
        return {"model_bytes": model_path.stat().st_size,
                "deployment_bytes": model_path.stat().st_size + metadata_path.stat().st_size}


def load_native_tree(output_dir):
    root = Path(output_dir)
    metadata = read_json(root / "metadata.json")
    if metadata.get("schema_version") != 1 or metadata.get("catboost_version") != CATBOOST_VERSION:
        raise ValueError("unsupported native-tree artifact")
    if sha256_file(root / "model.cbm") != metadata["model_sha256"]:
        raise ValueError("native-tree model hash mismatch")
    estimator = _catboost().CatBoostClassifier()
    estimator.load_model(str(root / "model.cbm"), format="cbm")
    return NativeTreeBundle(estimator, metadata["selected_features"], metadata["categorical_columns"],
                            metadata["feature_schema"], metadata["training_statistics"], metadata["num_threads"])


def train_native_tree(fit_frame, config, seed, threads, feature_schema,
                      selected_features=None, teacher_probabilities=None, kd_weight=0.0):
    """Fit a fixed-capacity CPU model on F, with no eval_set or hidden early stop.

    Real data are called only from the lab screen; unit tests use synthetic rows.
    teacher_probabilities, if supplied, must already follow the F row identity.
    """
    schema = normalize_feature_schema(feature_schema)
    selected = list(schema["features"] if selected_features is None else selected_features)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(schema["features"]):
        raise ValueError("selected features must be a nonempty subset of the input schema")
    required = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "border_count"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("native tree configuration requires explicit capacity and learning parameters")
    for key in ("iterations", "depth", "border_count"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be positive integer")
    if config["depth"] > 16:
        raise ValueError("native tree depth exceeds CPU support")
    for key in ("learning_rate", "l2_leaf_reg"):
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"invalid {key}")
    if config["learning_rate"] == 0 or type(threads) is not int or threads < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive learning rate/threads and nonnegative seed required")
    y = _labels(fit_frame, require_both=True)
    targets = mixed_targets(y, teacher_probabilities, kd_weight)
    categorical = [f for f in selected if f in schema["categorical"]]
    data = native_frame(fit_frame, selected, categorical)
    weights = compute_sample_weight("balanced", y)
    params = {**config, "loss_function": "CrossEntropy", "task_type": "CPU",
        "grow_policy": "SymmetricTree", "boosting_type": "Plain", "random_seed": seed,
        "thread_count": threads, "allow_writing_files": False, "verbose": False,
        "use_best_model": False, "max_ctr_complexity": 1}
    estimator = _catboost().CatBoostClassifier(**params)
    started = time.perf_counter()
    estimator.fit(data, targets, cat_features=categorical, sample_weight=weights)
    stats = {"fit_seconds": time.perf_counter() - started, "fit_rows": len(y),
        "fit_normal": int((y == 0).sum()), "fit_attack": int((y == 1).sum()),
        "seed": seed, "catboost_version": CATBOOST_VERSION, "parameters": params,
        "resolved_parameters": estimator.get_all_params(),
        "tree_count": int(estimator.tree_count_),
        "leaf_count": int(estimator.get_tree_leaf_counts().sum()),
        "raw_fields": len(selected), "categorical_fields": len(categorical),
        "weight_source": "balanced hard F labels; unchanged by soft targets",
        "target": "hard" if kd_weight == 0 else "(y + lambda * teacher_probability)/(1 + lambda)",
        "kd_weight": float(kd_weight), "target_mean_abs_difference_from_hard": float(np.abs(targets - y).mean()),
        "selection_used": False, "validation_used": False}
    return NativeTreeBundle(estimator, selected, categorical, schema, stats, threads)
