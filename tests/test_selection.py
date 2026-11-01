"""Scientific boundaries for reusable, group-resampled field selection."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from ids import selection
from ids.data import CATEGORICAL, FEATURES
from ids.models import load_bundle, save_bundle, train_model
from ids.selection import normalize_feature_schema, select_features, verify_selection_result


@pytest.fixture
def sample():
    rng = np.random.default_rng(90)
    n = 160
    y = np.tile([0, 1], n // 2)
    frame = pd.DataFrame({"signal": y + rng.normal(0, .5, n), "noise": rng.normal(size=n),
                          "count": rng.integers(0, 4, n), "protocol": np.tile(["a", "b", "c", "d"], n // 4),
                          "label": y, "sample_id": [f"sample-{i}" for i in range(n)],
                          "feature_hash": [f"feature-{i}" for i in range(n)],
                          "group_hash": [f"joint-{i // 4:03}" for i in range(n)],
                          "group_id": [f"coarser-{i // 8:03}" for i in range(n)]})
    schema = {"name": "test-flow", "features": ["signal", "noise", "count", "protocol"],
              "categorical": ["protocol"], "discrete": ["count", "protocol"]}
    return frame, schema


def stable_settings(**extra):
    return {"method": "stable_mutual_info", "k": 2, "n_resamples": 4, "selection_seed": 371, **extra}


def without_time(result):
    value = copy.deepcopy(result)
    value.pop("result_sha256")
    for name in ("seconds", "mi_seconds", "redundancy_seconds"):
        value["metadata"].pop(name)
    for sample in value["metadata"]["resamples"]:
        sample.pop("seconds")
    return value


def test_legacy_mi_scores_exactly_match_original_algorithm():
    rng = np.random.default_rng(22)
    labels = np.tile([0, 1], 40)
    frame = pd.DataFrame({name: (np.tile(["a", "b", "c", "a"], 20) if name in CATEGORICAL
                                  else rng.normal(size=80)) for name in FEATURES})
    frame["label"] = labels
    frame.loc[:4, "dur"] = np.nan
    numeric = [x for x in FEATURES if x not in CATEGORICAL]
    columns = {}
    values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(frame[numeric])
    columns.update({name: values[:, i] for i, name in enumerate(numeric)})
    values = OrdinalEncoder().fit_transform(frame[list(CATEGORICAL)].astype("string").fillna("<MISSING>").to_numpy(dtype=object))
    columns.update({name: values[:, i] for i, name in enumerate(CATEGORICAL)})
    expected = mutual_info_classif(np.column_stack([columns[x] for x in FEATURES]), labels,
                                  discrete_features=[x in CATEGORICAL for x in FEATURES], random_state=17,
                                  n_neighbors=3)
    result = select_features(frame, {"method": "mutual_info", "k": 8}, 17)
    np.testing.assert_array_equal(list(result["feature_scores"].values()), expected)
    ranked = set(np.lexsort((np.arange(len(FEATURES)), -expected))[:8])
    assert result["selected_features"] == [x for i, x in enumerate(FEATURES) if i in ranked]
    assert normalize_feature_schema()["discrete"] == list(CATEGORICAL)


def test_group_subsamples_are_complete_and_use_priority_group_hash(sample, monkeypatch):
    frame, schema = sample
    observed = []
    original = selection._mi_scores

    def observe(part, labels, feature_schema, seed):
        observed.append(part.copy())
        for group in part["group_hash"].unique():
            assert set(part.loc[part.group_hash == group, "sample_id"]) == set(frame.loc[frame.group_hash == group, "sample_id"])
        return original(part, labels, feature_schema, seed)

    monkeypatch.setattr(selection, "_mi_scores", observe)
    result = select_features(frame, stable_settings(), 17, feature_schema=schema)
    assert len(observed) == 4
    assert all(len(x) == 80 for x in observed)
    meta = result["metadata"]
    assert meta["group_column"] == "group_hash" and meta["n_groups"] == 40
    assert len({x["groups_sha256"] for x in meta["resamples"]}) > 1
    assert all(x["n_groups"] == 20 and x["actual_k"] == 2 for x in meta["resamples"])
    assert sum(meta["selection_frequencies"].values()) == 2
    assert meta["seconds"] >= meta["mi_seconds"] > 0


@pytest.mark.parametrize("drop,expected", [(["group_hash"], "group_id"), (["group_hash", "group_id"], "feature_hash")])
def test_group_priority_fallback(sample, drop, expected):
    frame, schema = sample
    result = select_features(frame.drop(columns=drop), stable_settings(), 17, feature_schema=schema)
    assert result["metadata"]["group_column"] == expected


def test_stable_selection_fixed_seed_is_independent_of_model_seed(sample):
    frame, schema = sample
    settings = stable_settings()
    del settings["selection_seed"]
    first = select_features(frame, settings, 17, feature_schema=schema)
    second = select_features(frame, settings, 43, feature_schema=schema)
    assert without_time(first) == without_time(second)
    verify_selection_result(first, frame, settings, 999, feature_schema=schema)


def test_all_result_can_be_shared_across_training_seeds(sample):
    frame, schema = sample
    result = select_features(frame, {"method": "all"}, 17, feature_schema=schema)
    verify_selection_result(result, frame, {"method": "all"}, 43, feature_schema=schema)


def test_frequency_mean_rank_then_schema_order_are_explicit_tie_breaks(sample, monkeypatch):
    frame, schema = sample
    scheduled = iter([np.array([4., 3., 2., 1.]), np.array([4., 3., 2., 1.]),
                      np.array([1., 2., 3., 4.]), np.array([1., 2., 3., 4.])])
    monkeypatch.setattr(selection, "mutual_info_classif", lambda *_a, **_k: next(scheduled))
    result = select_features(frame, stable_settings(k=1), 17, feature_schema=schema)
    assert result["metadata"]["selection_frequencies"] == {"signal": .5, "noise": 0, "count": 0, "protocol": .5}
    assert result["metadata"]["mean_ranks"] == dict.fromkeys(schema["features"], 2.5)
    assert result["selected_features"] == ["signal"]


def test_every_half_sample_fits_its_own_preprocessing(sample, monkeypatch):
    frame, schema = sample
    frame.loc[:2, "signal"] = np.nan
    observed = []
    original = selection.SimpleImputer.fit_transform

    def observe(self, values, *args, **kwargs):
        result = original(self, values, *args, **kwargs)
        observed.append((values.copy(), self.statistics_.copy()))
        return result

    monkeypatch.setattr(selection.SimpleImputer, "fit_transform", observe)
    select_features(frame, stable_settings(), 17, feature_schema=schema)
    assert len(observed) == 4
    for values, statistics in observed:
        assert len(values) == len(frame) // 2
        np.testing.assert_allclose(statistics, np.nanmedian(values, axis=0))
    assert not all(np.array_equal(observed[0][1], x[1]) for x in observed[1:])


def test_explicit_schema_discrete_mask_is_forwarded(sample, monkeypatch):
    frame, schema = sample
    seen = []

    def fake(matrix, labels, **kwargs):
        seen.append(kwargs["discrete_features"])
        return np.array([.4, .1, .3, .2])

    monkeypatch.setattr(selection, "mutual_info_classif", fake)
    result = select_features(frame, {"method": "mutual_info", "k": 2}, 1, feature_schema=schema)
    assert seen == [[False, False, True, True]]
    assert result["selected_features"] == ["signal", "count"]


def test_redundancy_uses_numeric_only_constants_and_no_silent_backfill(monkeypatch):
    n = 40
    frame = pd.DataFrame({"first": np.arange(n), "duplicate": -np.arange(n), "constant": 7,
                          "cat_a": np.tile(["x", "y"], n // 2), "cat_b": np.tile(["x", "y"], n // 2),
                          "label": np.tile([0, 1], n // 2)})
    schema = {"name": "redundancy", "features": list(frame.columns[:-1]),
              "categorical": ["cat_a", "cat_b"], "discrete": ["cat_a", "cat_b"]}
    monkeypatch.setattr(selection, "mutual_info_classif", lambda *_a, **_k: np.ones(5))
    setting = {"method": "mutual_info", "k": 3, "redundancy_threshold": .95}
    result = select_features(frame, setting, 1, feature_schema=schema)
    assert result["selected_features"] == ["first", "cat_a", "cat_b"]
    assert result["metadata"]["final_redundancy"]["constant_numeric_fields"] == ["constant"]
    assert result["metadata"]["final_redundancy"]["rejected_correlated"][0]["absolute_spearman"] == 1
    with pytest.raises(ValueError, match="no backfill"):
        select_features(frame, {**setting, "k": 4}, 1, feature_schema=schema)


def test_stable_all_categorical_schema_can_use_redundancy_switch(sample):
    frame, _ = sample
    schema = {"name": "categorical", "features": ["protocol"], "categorical": ["protocol"], "discrete": ["protocol"]}
    result = select_features(frame, stable_settings(k=1, redundancy_threshold=.95), 1, feature_schema=schema)
    assert result["selected_features"] == ["protocol"]


def test_pool_must_be_exact_full_groups_of_train_and_preserves_full_binding(sample):
    frame, schema = sample
    pool = frame.iloc[:80].copy()
    result = select_features(frame, stable_settings(), 1, feature_schema=schema, selection_frame=pool)
    assert result["metadata"]["training_binding"]["n_rows"] == 160
    assert result["metadata"]["pool_binding"]["n_rows"] == 80
    verify_selection_result(result, frame, stable_settings(), 999, feature_schema=schema)
    with pytest.raises(ValueError, match="complete training groups"):
        select_features(frame, stable_settings(), 1, feature_schema=schema, selection_frame=pool.iloc[:-1])
    foreign = pool.copy()
    foreign.loc[0, "sample_id"] = "validation-only"
    with pytest.raises(ValueError, match="outside training"):
        select_features(frame, stable_settings(), 1, feature_schema=schema, selection_frame=foreign)
    altered = pool.copy()
    altered.loc[0, "signal"] += 100
    with pytest.raises(ValueError, match="differ from training"):
        select_features(frame, stable_settings(), 1, feature_schema=schema, selection_frame=altered)


def test_explicit_row_pool_allows_partial_original_groups_but_keeps_pool_groups_whole(sample, monkeypatch):
    frame, schema = sample
    frame["attack_cat"] = np.where(frame.label == 0, "Normal", "Attack")
    frame.loc[1, "attack_cat"] = "Rare"
    # Keep two of the four original rows in each group. Only explicit policy
    # permits this external pool; each inner sample must retain both pool rows.
    pool = frame.loc[frame.index % 4 < 2].copy()
    observed = []
    original = selection._mi_scores

    def inspect(part, *args, **kwargs):
        observed.append(part.copy())
        for group in part.group_hash.unique():
            assert set(part.loc[part.group_hash == group, "sample_id"]) == set(pool.loc[pool.group_hash == group, "sample_id"])
        return original(part, *args, **kwargs)

    monkeypatch.setattr(selection, "_mi_scores", inspect)
    settings = stable_settings(pool_policy="stratified_rows")
    result = select_features(frame, settings, 17, feature_schema=schema, selection_frame=pool)
    meta = result["metadata"]
    assert meta["pool_policy"] == "stratified_rows"
    assert meta["original_training_group_completeness"] == "not_required_partial_training_groups_allowed"
    assert meta["resampling_group_scope"] == "whole_groups_within_current_selection_pool"
    assert len(observed) == settings["n_resamples"]
    for part, record in zip(observed, meta["resamples"]):
        assert record["family_counts"] == part.attack_cat.value_counts().to_dict()
        assert record["missing_families"] == sorted(set(pool.attack_cat) - set(part.attack_cat))
        assert len(part) == 40
    assert any("Rare" in record["missing_families"] for record in meta["resamples"])
    # An outer diagnostic receives its current pool as train_df. It must not
    # claim that these partial groups contain the original training rows.
    outer = select_features(pool, settings, 17, feature_schema=schema)
    assert outer["metadata"]["pool_scope"] == "current_input_frame"
    assert outer["metadata"]["original_training_group_completeness"] == "not_verified_against_original_training"
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_selection_result(result, frame, stable_settings(), 17, feature_schema=schema)


@pytest.mark.parametrize("change", ["outside", "feature", "family", "duplicate"])
def test_partial_row_pool_does_not_relax_identity_or_values(sample, change):
    frame, schema = sample
    frame["attack_cat"] = np.where(frame.label == 0, "Normal", "Attack")
    pool = frame.iloc[:77].copy()
    if change == "outside":
        pool.loc[0, "sample_id"] = "not-in-training"
    elif change == "feature":
        pool.loc[0, "signal"] += 100
    elif change == "family":
        pool.loc[0, "attack_cat"] = "changed-family"
    else:
        pool.loc[0, "sample_id"] = pool.loc[1, "sample_id"]
    with pytest.raises(ValueError, match="outside training|differ from training|unique"):
        select_features(frame, stable_settings(pool_policy="stratified_rows"), 17,
                        feature_schema=schema, selection_frame=pool)


def test_reuse_binds_training_family_identity_for_stratified_pool(sample):
    frame, schema = sample
    frame["attack_cat"] = np.where(frame.label == 0, "Normal", "Attack")
    settings = stable_settings(pool_policy="stratified_rows")
    result = select_features(frame, settings, 17, feature_schema=schema, selection_frame=frame.iloc[:77])
    frame.loc[100, "attack_cat"] = "changed-outside-pool"
    with pytest.raises(ValueError, match="binding mismatch"):
        verify_selection_result(result, frame, settings, 17, feature_schema=schema)


@pytest.mark.parametrize("change", ["label", "feature", "sample_id", "group", "order", "schema", "settings", "hash"])
def test_reuse_rejects_tampering_and_mismatched_binding(sample, change):
    frame, schema = sample
    settings = stable_settings()
    result = select_features(frame, settings, 1, feature_schema=schema)
    frame = frame.copy()
    schema = copy.deepcopy(schema)
    if change == "label":
        frame.loc[0, "label"] = 1 - frame.loc[0, "label"]
    elif change == "feature":
        frame.loc[0, "signal"] += 3
    elif change == "sample_id":
        frame.loc[0, "sample_id"] = "changed"
    elif change == "group":
        frame.loc[0, "group_hash"] = "changed"
    elif change == "order":
        frame = frame.iloc[::-1]
    elif change == "schema":
        schema["discrete"] = ["protocol"]
    elif change == "settings":
        settings["selection_seed"] += 1
    else:
        result["metadata"]["actual_k"] = 99
    with pytest.raises(ValueError, match="mismatch"):
        verify_selection_result(result, frame, settings, 1, feature_schema=schema)


def test_one_class_subsample_fails_explicitly_without_redraw(sample):
    frame, schema = sample
    frame["group_hash"] = frame["label"].astype(str)
    with pytest.raises(ValueError, match="both normal"):
        select_features(frame, stable_settings(), 1, feature_schema=schema)


@pytest.mark.parametrize("bad", ["identifier", "duplicate", "missing_type", "categorical_not_discrete"])
def test_schema_validation(bad, sample):
    _, schema = sample
    schema = copy.deepcopy(schema)
    if bad == "identifier":
        schema["features"].append("group_hash")
    elif bad == "duplicate":
        schema["features"].append("noise")
    elif bad == "missing_type":
        del schema["discrete"]
    else:
        schema["discrete"] = ["count"]
    with pytest.raises(ValueError):
        normalize_feature_schema(schema)


@pytest.mark.parametrize("settings", [stable_settings(k=0), stable_settings(n_resamples=1),
                                     stable_settings(redundancy_threshold=0), stable_settings(selection_seed=True),
                                     stable_settings(typo=3), stable_settings(pool_policy="unknown")])
def test_invalid_settings_fail_before_mi(sample, settings, monkeypatch):
    frame, schema = sample
    monkeypatch.setattr(selection, "_mi_scores", lambda *_: pytest.fail("invalid settings reached MI"))
    with pytest.raises(ValueError):
        select_features(frame, settings, 1, feature_schema=schema)


def test_model_reuses_selection_without_recomputing_and_fits_full_train(sample, monkeypatch, tmp_path):
    frame, schema = sample
    train, val = frame.iloc[:128].copy(), frame.iloc[128:].copy()
    settings = stable_settings()
    result = select_features(train, settings, 1, feature_schema=schema, selection_frame=train.iloc[:64])
    monkeypatch.setattr(selection, "_mi_scores", lambda *_: pytest.fail("reuse recomputed selector"))
    model_config = {"model": "lr", "features": settings}
    first = train_model(train, val, model_config, 17, feature_schema=schema, selection_result=result)
    changed = val.copy()
    changed["signal"] = 999999
    changed["protocol"] = "only-validation"
    changed["label"] = 1 - changed["label"]
    second = train_model(train, changed, model_config, 17, feature_schema=schema, selection_result=result)
    assert first.selected_features == second.selected_features
    np.testing.assert_array_equal(first.estimator.coef_, second.estimator.coef_)
    assert first.preprocessor.scaler.n_samples_seen_ == len(train)
    path = tmp_path / "schema-model.joblib"
    save_bundle(first, path)
    restored = load_bundle(path)
    assert restored.feature_schema == schema
    assert "resamples" not in restored.selection_metadata
    assert len(first.selection_metadata["resamples"]) == 4
    np.testing.assert_array_equal(first.predict_scores(val), restored.predict_scores(val))
    np.testing.assert_array_equal(first.predict_scores(val), first.predict_scores(val.drop(columns=["label"])))
    with pytest.raises(ValueError, match="Missing required"):
        restored.predict_scores(val.drop(columns=["noise"]))
