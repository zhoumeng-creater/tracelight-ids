"""Optional real-package checks; a skipped test is not a TabM validation result."""

import importlib.metadata
import importlib.util

import numpy as np
import pandas as pd
import pytest

from ids import recent_baselines as recent


@pytest.fixture
def tabm_dependencies():
    if any(importlib.util.find_spec(name) is None for name in ("torch", "tabm", "rtdl_num_embeddings")):
        pytest.skip("Optional official TabM packages absent; run in the RECENT_METHODS.md isolated environment")
    return recent.recent_dependency_versions()


@pytest.fixture
def schema_frames():
    schema = {"name": "small-native-fixture", "features": ["value", "constant", "category"],
              "categorical": ["category"], "discrete": ["category"]}
    rng = np.random.default_rng(52)
    frame = pd.DataFrame({"value": rng.normal(size=64), "constant": 5.0,
                          "category": ["train-a", "train-b", None, "train-a"] * 16,
                          "label": np.tile([0, 1], 32),
                          "sample_id": [f"synthetic-{x}" for x in range(64)]})
    frame.loc[0, "value"] = np.nan
    train, val = frame.iloc[:48].copy(), frame.iloc[48:].copy()
    val.loc[val.index[0], "category"] = "only-in-validation"
    return schema, train, val


def settings(**changes):
    return {"model": "tabm", "hyperparameters": {
        "epochs": 3, "patience": 2, "batch_size": 17, "k": 3,
        "n_blocks": 2, "d_block": 8, "dropout": 0.0,
        "embedding_dim": 4, "n_bins": 4, **changes,
    }}


def test_dependency_error_is_actionable_and_does_not_install(monkeypatch):
    original = importlib.metadata.version

    def missing(name):
        if name == "tabm":
            raise importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="explicit optional installation"):
        recent.recent_dependency_versions()


def test_native_preprocessing_train_only_and_unknown_categories(schema_frames):
    schema, train, val = schema_frames
    processor = recent.TabMPreprocessor.fit(train, schema["features"], schema)
    numeric, categorical = processor.transform(val)
    assert numeric.dtype == np.float32 and categorical.dtype == np.int64
    assert categorical[0, 0] == 1  # Unseen category, not a validation-fitted category.
    assert categorical[2, 0] == 0  # Missing category has a distinct reserved index.
    assert "only-in-validation" not in processor.vocabularies["category"]
    assert processor.cat_cardinalities == [4]
    assert processor.imputer.statistics_[0] == pytest.approx(train["value"].median())
    assert processor.scaler.mean_[1] == 5
    assert np.isfinite(numeric).all()
    altered = val.assign(label=1 - val["label"], sample_id="irrelevant-metadata")
    for left, right in zip(processor.transform(altered), (numeric, categorical)):
        np.testing.assert_array_equal(left, right)
    bad = val.copy()
    bad["value"] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        processor.transform(bad)


def test_official_per_head_loss_and_probability_aggregation(tabm_dependencies):
    import torch

    logits = torch.tensor([[[2.0], [-0.1], [-3.0]], [[-1.0], [0.2], [4.0]]], dtype=torch.float64)
    truth = torch.tensor([0., 1.], dtype=torch.float64)
    weight = torch.tensor(1.7, dtype=torch.float64)
    expected = sum(torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, index, 0], truth, pos_weight=weight) for index in range(3)) / 3
    torch.testing.assert_close(recent.tabm_head_loss(logits, truth, weight), expected)
    actual = recent.tabm_probabilities(logits)
    torch.testing.assert_close(actual, logits.sigmoid().mean(dim=1).squeeze(1))
    assert not torch.allclose(actual, logits.mean(dim=1).sigmoid().squeeze(1))


@pytest.mark.parametrize("embedding", ["none", "piecewise_linear"])
def test_real_official_fit_cpu_roundtrip_and_contract(tabm_dependencies, schema_frames, tmp_path, embedding):
    import torch

    schema, train, val = schema_frames
    bundle = recent.train_tabm(train, val, settings(num_embeddings=embedding), 17, feature_schema=schema)
    scores = bundle.predict_scores(val)
    assert scores.shape == (len(val),) and np.isfinite(scores).all()
    assert np.all((scores >= 0) & (scores <= 1))
    assert bundle.training_statistics["dependencies"]["tabm"] == "0.0.3"
    assert bundle.training_statistics["training_objective"] == "mean_independent_head_weighted_bce"
    assert 1 <= bundle.best_epoch <= len(bundle.history) <= 3
    assert all(x.device.type == "cpu" for x in bundle.network_state.values())
    np.testing.assert_array_equal(scores, bundle.predict_scores(val.loc[:, val.columns[::-1]]))
    np.testing.assert_array_equal(scores, bundle.predict_scores(val.assign(label=1 - val["label"])))
    singletons = np.array([bundle.predict_scores(val.iloc[i:i + 1])[0] for i in range(len(val))])
    np.testing.assert_allclose(scores, singletons, atol=1e-6, rtol=1e-5)
    assert bundle.predict_scores(val.iloc[:0]).shape == (0,)
    with pytest.raises(ValueError, match="Missing required"):
        bundle.predict_scores(val.drop(columns=["value"]))
    path = tmp_path / "tabm.joblib"
    recent.save_tabm_bundle(bundle, path)
    restored = recent.load_tabm_bundle(path)
    assert restored.device == "cpu" and restored._cached_network is None
    np.testing.assert_array_equal(scores, restored.predict_scores(val))
    repeated = recent.train_tabm(train, val, settings(num_embeddings=embedding), 17, feature_schema=schema)
    for name, value in bundle.network_state.items():
        torch.testing.assert_close(value, repeated.network_state[name], rtol=0, atol=0)
    if embedding == "piecewise_linear":
        assert bundle.training_statistics["constant_numeric_fields"] == ["constant"]


def test_validation_never_fits_bins_or_vocabulary(tabm_dependencies, schema_frames):
    schema, train, val = schema_frames
    first = recent.train_tabm(train, val, settings(epochs=1), 7, feature_schema=schema)
    changed = val.copy()
    changed["value"] = 10000.0
    changed["category"] = "different-unseen-category"
    second = recent.train_tabm(train, changed, settings(epochs=1), 7, feature_schema=schema)
    assert first.numeric_bins == second.numeric_bins
    assert first.preprocessor.vocabularies == second.preprocessor.vocabularies
    np.testing.assert_array_equal(first.preprocessor.scaler.mean_, second.preprocessor.scaler.mean_)
    np.testing.assert_array_equal(first.preprocessor.imputer.statistics_, second.preprocessor.imputer.statistics_)
    np.testing.assert_array_equal(first.predict_scores(val), second.predict_scores(val))


def test_early_stopping_restores_best_epoch(tabm_dependencies, schema_frames):
    schema, train, val = schema_frames
    stopped = recent.train_tabm(train, val, settings(epochs=6, patience=1, min_delta=100), 5,
                               feature_schema=schema)
    first = recent.train_tabm(train, val, settings(epochs=1), 5, feature_schema=schema)
    assert stopped.best_epoch == 1 and len(stopped.history) == 2
    np.testing.assert_array_equal(stopped.predict_scores(val), first.predict_scores(val))


@pytest.mark.parametrize("changes", [{"k": 0}, {"d_block": True}, {"num_embeddings": "invented"},
                                      {"learning_rate": -1}, {"dropout": 1}, {"unknown": 3}])
def test_invalid_parameters_rejected_before_optional_import(schema_frames, changes):
    schema, train, val = schema_frames
    with pytest.raises(ValueError):
        recent.train_tabm(train, val, settings(**changes), 2, feature_schema=schema)


def test_numeric_only_schema_and_selected_field_contract(tabm_dependencies, schema_frames):
    schema, train, val = schema_frames
    bundle = recent.train_tabm(train, val, settings(epochs=1), 1, feature_schema=schema,
                              selected_features=["value", "constant"])
    assert bundle.preprocessor.cat_cardinalities == []
    assert bundle.predict_scores(val).shape == (len(val),)
    # As with the existing ModelBundle, inference still requires the whole schema.
    with pytest.raises(ValueError, match="Missing required"):
        bundle.predict_scores(val.drop(columns="category"))


def test_categorical_only_schema_is_supported(tabm_dependencies, schema_frames):
    schema, train, val = schema_frames
    bundle = recent.train_tabm(train, val, settings(epochs=1), 3, feature_schema=schema,
                              selected_features=["category"])
    assert bundle.numeric_bins == []
    assert bundle.predict_scores(val).shape == (len(val),)
