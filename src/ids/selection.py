"""Train-only field selection with auditable group subsampling.

Selection frequencies are construction diagnostics, not an independent estimate
of stability. A caller must rerun the *whole* selector on independent outer
training resamples to compare stability. No calibration or test data are used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from .data import CATEGORICAL, FEATURES

EXCLUDED = frozenset({"id", "sample_id", "feature_hash", "group_hash", "group_id", "label", "attack_cat"})
GROUP_COLUMNS = ("group_hash", "group_id", "feature_hash")
DEFAULT_SELECTION_SEED = 20261101


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def normalize_feature_schema(schema: dict | None = None) -> dict:
    """Return a validated, JSON-safe original-field schema.

    UNSW defaults deliberately preserve its historical MI discrete mask:
    only proto/service/state are discrete. Custom schemas explicitly name all
    discrete fields, including numeric counts/binary fields if appropriate.
    """
    if schema is None or schema == {}:
        schema = {"name": "unsw-nb15", "features": list(FEATURES),
                  "categorical": list(CATEGORICAL), "discrete": list(CATEGORICAL)}
    if not isinstance(schema, dict) or set(schema) != {"name", "features", "categorical", "discrete"}:
        raise ValueError("feature_schema requires name, features, categorical, and discrete")
    if not isinstance(schema["name"], str) or not schema["name"].strip():
        raise ValueError("feature_schema.name must be nonempty")
    for key in ("features", "categorical", "discrete"):
        values = schema[key]
        if (not isinstance(values, (list, tuple)) or any(not isinstance(x, str) or not x for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"feature_schema.{key} must be a list of unique field names")
    names = list(schema["features"])
    if not names or EXCLUDED.intersection(names):
        raise ValueError("Feature schema must exclude identifiers and labels")
    if (not set(schema["categorical"]).issubset(schema["discrete"])
            or not set(schema["discrete"]).issubset(names)):
        raise ValueError("Categorical fields must be discrete and all field types must belong to features")
    return {"name": schema["name"], "features": names,
            "categorical": [x for x in names if x in schema["categorical"]],
            "discrete": [x for x in names if x in schema["discrete"]]}


def _check_frame(frame: pd.DataFrame, schema: dict) -> np.ndarray:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or frame.empty:
        raise ValueError("Selection requires a nonempty DataFrame with unique columns")
    missing = set(schema["features"] + ["label"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required selection columns: {sorted(missing)}")
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Selection labels must be finite binary 0/1")
    if len(np.unique(labels)) != 2:
        raise ValueError("Selection requires both normal and attack labels")
    return labels.astype(np.int64)


def _group_column(frame: pd.DataFrame) -> str:
    for name in GROUP_COLUMNS:
        if name in frame:
            if frame[name].isna().any() or (frame[name].astype(str).str.len() == 0).any():
                raise ValueError(f"Selection group column {name} contains missing identifiers")
            return name
    raise ValueError("Grouped selection requires group_hash, group_id, or feature_hash")


def _frame_binding(frame: pd.DataFrame, schema: dict) -> dict:
    """Bind identity/order and actual field values, without exposing row IDs."""
    identity_columns = [x for x in ("sample_id", "feature_hash", "group_hash", "group_id", "label", "attack_cat") if x in frame]
    if "sample_id" in frame and (frame["sample_id"].isna().any() or not frame["sample_id"].is_unique):
        raise ValueError("sample_id must be nonmissing and unique")
    identity = pd.util.hash_pandas_object(frame[identity_columns], index=False, categorize=False)
    values = pd.util.hash_pandas_object(frame[schema["features"]], index=False, categorize=False)
    return {"n_rows": len(frame), "identity_columns": identity_columns,
            "identity_sha256": hashlib.sha256(identity.to_numpy(dtype="uint64").tobytes()).hexdigest(),
            "feature_values_sha256": hashlib.sha256(values.to_numpy(dtype="uint64").tobytes()).hexdigest(),
            "hash_algorithm": "sha256(pandas.hash_pandas_object(index=False,categorize=False),uint64-bytes)",
            "row_order": "input order", "pandas_version": pd.__version__}


def _settings(settings: dict, seed: int, n_features: int) -> dict:
    if not isinstance(settings, dict):
        raise ValueError("features settings must be a dictionary")
    allowed = {"method", "k", "selection_seed", "n_resamples", "redundancy_threshold", "pool_policy"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown feature selection settings: {sorted(set(settings) - allowed)}")
    method = settings.get("method", "all")
    if method not in {"all", "mutual_info", "stable_mutual_info"}:
        raise ValueError("features.method must be all, mutual_info, or stable_mutual_info")
    pool_policy = settings.get("pool_policy", "complete_groups")
    if pool_policy not in {"complete_groups", "stratified_rows"}:
        raise ValueError("pool_policy must be complete_groups or stratified_rows")
    k = settings.get("k")
    if method == "all":
        if k is not None or settings.get("redundancy_threshold") is not None:
            raise ValueError("features.k and redundancy_threshold must be null for method all")
    elif isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n_features:
        raise ValueError(f"features.k must be an integer between 1 and {n_features}")
    threshold = settings.get("redundancy_threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not np.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("redundancy_threshold must be finite and in (0, 1]")
    default_seed = DEFAULT_SELECTION_SEED if method in {"all", "stable_mutual_info"} or threshold is not None else seed
    selection_seed = settings.get("selection_seed", default_seed)
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int) or not 0 <= selection_seed < 2**32:
        raise ValueError("selection_seed must be an integer in [0, 2**32)")
    repeats = settings.get("n_resamples", 50)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("n_resamples must be an integer >= 2")
    return {"method": method, "k": k, "selection_seed": selection_seed,
            "n_resamples": repeats if method == "stable_mutual_info" else 0,
            "redundancy_threshold": None if threshold is None else float(threshold),
            "pool_policy": pool_policy}


def _mi_scores(frame: pd.DataFrame, labels: np.ndarray, schema: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    names, categorical = schema["features"], schema["categorical"]
    numeric = [x for x in names if x not in categorical]
    columns = {}
    if numeric:
        values = frame[numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError("Numeric selection features contain infinity")
        values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(values)
        columns.update({name: values[:, i] for i, name in enumerate(numeric)})
    if categorical:
        values = OrdinalEncoder().fit_transform(frame[categorical].astype("string").fillna("<MISSING>"))
        columns.update({name: values[:, i] for i, name in enumerate(categorical)})
    matrix = np.column_stack([columns[name] for name in names])
    scores = mutual_info_classif(matrix, labels, discrete_features=[x in schema["discrete"] for x in names],
                                random_state=seed, n_neighbors=min(3, len(frame) - 1))
    return scores, matrix


def _choose(ranking: np.ndarray, matrix: np.ndarray, schema: dict, k: int,
            threshold: float | None) -> tuple[list[int], dict]:
    if threshold is None:
        return ranking[:k].tolist(), {"constant_numeric_fields": [], "rejected_correlated": []}
    names = schema["features"]
    numeric = [i for i, x in enumerate(names) if x not in schema["categorical"]]
    constants = [i for i in numeric if np.unique(matrix[:, i]).size <= 1]
    varying = [i for i in numeric if i not in constants]
    # Ranking then Pearson correlation is Spearman. Categorical ordinal codes
    # never enter this matrix; constant columns are explicitly ineligible.
    corr = pd.DataFrame(matrix[:, varying]).corr(method="spearman").to_numpy()
    corr_index = {index: offset for offset, index in enumerate(varying)}
    chosen, rejected = [], []
    for index in ranking.tolist():
        if index in constants:
            continue
        conflict = next((other for other in chosen if index in corr_index and other in corr_index
                         and abs(corr[corr_index[index], corr_index[other]]) >= threshold), None)
        if conflict is None:
            chosen.append(index)
        else:
            rejected.append({"field": names[index], "retained_field": names[conflict],
                             "absolute_spearman": float(abs(corr[corr_index[index], corr_index[conflict]]))})
        if len(chosen) == k:
            break
    if len(chosen) < k:
        raise ValueError(f"Redundancy constraints cannot satisfy requested k={k}; only {len(chosen)} fields eligible; no backfill")
    return chosen, {"constant_numeric_fields": [names[i] for i in constants], "rejected_correlated": rejected}


def _validate_pool(train: pd.DataFrame, pool: pd.DataFrame, schema: dict,
                   *, require_complete_groups: bool = True) -> None:
    if "sample_id" not in train or "sample_id" not in pool:
        raise ValueError("An external selection pool requires sample_id")
    _frame_binding(pool, schema)
    if not pool["sample_id"].isin(train["sample_id"]).all():
        raise ValueError("Selection pool includes samples outside training")
    matching = train.set_index("sample_id", drop=False).loc[pool["sample_id"]]
    if _frame_binding(matching, schema) != _frame_binding(pool, schema):
        raise ValueError("Selection pool identity or feature values differ from training")
    if require_complete_groups:
        group = _group_column(train)
        selected_groups = set(pool[group].astype(str))
        expected = train.loc[train[group].astype(str).isin(selected_groups), "sample_id"]
        if set(expected) != set(pool["sample_id"]):
            raise ValueError("Selection pool must contain complete training groups")


def select_features(train_df: pd.DataFrame, settings: dict, seed: int, *,
                    feature_schema: dict | None = None,
                    selection_frame: pd.DataFrame | None = None) -> dict:
    """Select raw fields using training data, optionally a verified selection pool.

    Stable selection uniformly draws floor(G/2) groups without replacement each
    round. A one-class subsample is an explicit error, not silently redrawn.
    Final ranks use frequency descending, mean MI rank ascending, schema order.
    The result can be reused by train_model, which verifies the training binding.
    The explicit stratified_rows policy permits partial original-training groups
    in an external pool. Sampling still takes whole groups *within that pool*.
    Without selection_frame, the input itself is the current pool (e.g. an outer
    diagnostic); no assertion about original-training group completeness is made.
    """
    started = time.perf_counter()
    schema = normalize_feature_schema(feature_schema)
    _check_frame(train_df, schema)
    normalized = _settings(settings, seed, len(schema["features"]))
    training_binding = _frame_binding(train_df, schema)
    pool = train_df if selection_frame is None else selection_frame
    if selection_frame is not None:
        _validate_pool(train_df, pool, schema,
                       require_complete_groups=normalized["pool_policy"] == "complete_groups")
    labels = _check_frame(pool, schema)
    names = schema["features"]
    pool_families = sorted(pool["attack_cat"].astype(str).unique().tolist()) if "attack_cat" in pool else None
    metadata = {"schema_version": 1, "method": normalized["method"], "settings": normalized,
                "feature_schema": schema, "training_binding": training_binding,
                "pool_binding": _frame_binding(pool, schema), "uses_split": "train",
                "pool_policy": normalized["pool_policy"], "pool_families": pool_families,
                "pool_scope": "external_selection_frame" if selection_frame is not None else "current_input_frame",
                "original_training_group_completeness": (
                    "not_verified_against_original_training" if selection_frame is None else
                    "required_and_verified" if normalized["pool_policy"] == "complete_groups" else
                    "not_required_partial_training_groups_allowed"),
                "resampling_group_scope": "whole_groups_within_current_selection_pool",
                "stability_scope": "internal construction only; not independent outer stability",
                "requested_k": normalized["k"], "n_resamples": normalized["n_resamples"],
                "resamples": [], "mi_seconds": 0.0, "redundancy_seconds": 0.0}
    if normalized["method"] == "all":
        selected, feature_scores = list(names), {}
        metadata.update({"group_column": None, "n_groups": None, "selection_frequencies": {}, "mean_ranks": {}})
    else:
        k, threshold = normalized["k"], normalized["redundancy_threshold"]
        counts, rank_sum, score_sum = np.zeros(len(names)), np.zeros(len(names)), np.zeros(len(names))
        is_stable = normalized["method"] == "stable_mutual_info"
        group = _group_column(pool) if is_stable else None
        groups = pool[group].astype(str).to_numpy() if group else None
        unique = np.sort(np.unique(groups)) if group else None
        n_rounds = normalized["n_resamples"] if is_stable else 1
        if is_stable and len(unique) < 2:
            raise ValueError("Stable selection requires at least two distinct groups")
        metadata.update({"group_column": group, "n_groups": len(unique) if group else None,
                         "sampling_fraction_of_groups": .5 if is_stable else 1.0,
                         "sampling_policy": "uniform_without_replacement_no_redraw" if is_stable else "full_pool"})
        rng = np.random.default_rng(normalized["selection_seed"])
        last_matrix = None
        for iteration in range(n_rounds):
            begin = time.perf_counter()
            if is_stable:
                sampled_groups = np.sort(rng.choice(unique, size=len(unique) // 2, replace=False))
                positions = np.flatnonzero(np.isin(groups, sampled_groups))
                part = pool.iloc[positions]
                mi_seed = int(rng.integers(0, 2**32))
                part_labels = _check_frame(part, schema)
            else:
                part, part_labels, mi_seed = pool, labels, normalized["selection_seed"]
            mi_start = time.perf_counter()
            scores, matrix = _mi_scores(part, part_labels, schema, mi_seed)
            metadata["mi_seconds"] += time.perf_counter() - mi_start
            ranking = np.lexsort((np.arange(len(names)), -scores))
            rank = np.empty(len(names), dtype=float)
            rank[ranking] = np.arange(1, len(names) + 1)
            red_start = time.perf_counter()
            chosen, diagnostic = _choose(ranking, matrix, schema, k, threshold)
            metadata["redundancy_seconds"] += time.perf_counter() - red_start
            counts[chosen] += 1
            rank_sum += rank
            score_sum += scores
            last_matrix = matrix
            if is_stable:
                family_counts = ({str(key): int(value) for key, value in
                                  part["attack_cat"].astype(str).value_counts().items()}
                                 if pool_families is not None else None)
                metadata["resamples"].append({"iteration": iteration, "mi_seed": mi_seed,
                    "n_groups": len(sampled_groups), "groups_sha256": _hash(sampled_groups.tolist()),
                    "sample_binding": _frame_binding(part, schema), "label_counts": np.bincount(part_labels, minlength=2).tolist(),
                    "family_counts": family_counts,
                    "missing_families": [name for name in pool_families if name not in family_counts]
                    if pool_families is not None else None,
                    "selected_features": [names[i] for i in sorted(chosen)], "actual_k": len(chosen),
                    "redundancy": diagnostic, "seconds": time.perf_counter() - begin})
        frequencies, mean_rank, mean_score = counts / n_rounds, rank_sum / n_rounds, score_sum / n_rounds
        if is_stable:
            ranking = np.lexsort((np.arange(len(names)), mean_rank, -frequencies))
            if threshold is not None:
                # Correlation must be fitted on the full selection pool for the
                # final greedy pass, not on the last half-sample by accident.
                numeric = [x for x in names if x not in schema["categorical"]]
                last_matrix = np.zeros((len(pool), len(names)))
                if numeric:
                    values = pool[numeric].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, na_value=np.nan)
                    if np.isinf(values).any():
                        raise ValueError("Numeric selection features contain infinity")
                    values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(values)
                    for offset, name in enumerate(numeric):
                        last_matrix[:, names.index(name)] = values[:, offset]
        red_start = time.perf_counter()
        chosen, diagnostic = _choose(ranking, last_matrix, schema, k, threshold)
        metadata["redundancy_seconds"] += time.perf_counter() - red_start
        selected = [names[i] for i in sorted(chosen)]
        feature_scores = dict(zip(names, mean_score.tolist()))
        metadata.update({"selection_frequencies": dict(zip(names, frequencies.tolist())),
                         "mean_ranks": dict(zip(names, mean_rank.tolist())), "final_redundancy": diagnostic,
                         "ranking": [names[i] for i in ranking]})
    metadata.update({"actual_k": len(selected), "seconds": time.perf_counter() - started})
    result = {"selected_features": selected, "feature_scores": feature_scores, "metadata": metadata}
    result["result_sha256"] = _hash(result)
    return result


def verify_selection_result(result: dict, train_df: pd.DataFrame, settings: dict, seed: int, *,
                            feature_schema: dict | None = None) -> dict:
    """Validate an in-memory/cached result before reusing its selected fields."""
    schema = normalize_feature_schema(feature_schema)
    _check_frame(train_df, schema)
    normalized = _settings(settings, seed, len(schema["features"]))
    if not isinstance(result, dict) or set(result) != {"selected_features", "feature_scores", "metadata", "result_sha256"}:
        raise ValueError("Invalid selection_result structure")
    payload = {key: value for key, value in result.items() if key != "result_sha256"}
    if _hash(payload) != result["result_sha256"]:
        raise ValueError("selection_result hash mismatch")
    metadata = result["metadata"]
    if (metadata.get("feature_schema") != schema or metadata.get("settings") != normalized
            or metadata.get("training_binding") != _frame_binding(train_df, schema)):
        raise ValueError("selection_result schema/settings/training binding mismatch")
    chosen = result["selected_features"]
    if (not isinstance(chosen, list) or not chosen or len(set(chosen)) != len(chosen)
            or chosen != [x for x in schema["features"] if x in chosen]
            or len(chosen) != (normalized["k"] or len(schema["features"]))):
        raise ValueError("Invalid selected feature budget/order")
    return copy.deepcopy(result)
