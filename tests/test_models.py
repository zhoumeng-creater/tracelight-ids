"""Model tests focus on leakage boundaries, schema, and artifact consistency."""

import numpy as np
import pandas as pd
import pytest

from ids.data import CATEGORICAL, FEATURES
from ids.models import load_bundle, save_bundle, train_model


@pytest.fixture
def frames():
    rng = np.random.default_rng(71)
    labels = np.tile([0, 1], 64)
    values = {}
    for index, name in enumerate(FEATURES):
        if name in CATEGORICAL:
            values[name] = np.array(["category-a", "category-b", "category-c", "category-a"] * 32)
        else:
            values[name] = rng.normal(size=128) + labels * (2.0 if index < 5 else 0.05)
    frame = pd.DataFrame(values)
    frame["label"] = labels
    frame["attack_cat"] = np.where(labels == 1, "Attack", "Normal")
    frame["id"] = np.arange(128)
    frame["sample_id"] = [f"sample-{i}" for i in range(128)]
    frame["feature_hash"] = [f"hash-{i}" for i in range(128)]
    return frame.iloc[:96].copy(), frame.iloc[96:].copy()


def config(name, features=None, **hyperparameters):
    return {"model": name, "features": features or {"method": "all", "k": None}, "hyperparameters": hyperparameters}


@pytest.mark.parametrize("name,parameters", [("lr", {}), ("rf", {"n_estimators": 12}), ("hgb", {"max_iter": 12})])
def test_classical_prediction_and_roundtrip(frames, tmp_path, name, parameters):
    train, val = frames
    bundle = train_model(train, val, config(name, **parameters), seed=12)
    if name == "hgb":
        assert bundle.estimator.early_stopping is False
    scores = bundle.predict_scores(val)
    assert scores.shape == (len(val),)
    assert np.all((0 <= scores) & (scores <= 1))
    assert set(bundle.selected_features).isdisjoint({"id", "sample_id", "feature_hash", "label", "attack_cat"})
    assert bundle.preprocessor.transform(val).dtype == np.float32
    assert bundle.encoded_dim == bundle.preprocessor.transform(val).shape[1]
    path = tmp_path / f"{name}.joblib"
    save_bundle(bundle, path)
    restored = load_bundle(path)
    np.testing.assert_array_equal(scores, restored.predict_scores(val))
    # Predictor access is by field name, not by accidental CSV column order.
    np.testing.assert_array_equal(scores, restored.predict_scores(val.loc[:, val.columns[::-1]]))
    altered = val.copy()
    altered["label"] = 1 - altered["label"]
    altered["attack_cat"] = "changed metadata"
    np.testing.assert_array_equal(scores, restored.predict_scores(altered))


def test_preprocessing_fit_uses_train_only(frames):
    train, val = frames
    first_numeric = next(name for name in FEATURES if name not in CATEGORICAL)
    train.loc[0, first_numeric] = np.nan
    changed_val = val.copy()
    changed_val[first_numeric] = 10_000_000.0
    for name in CATEGORICAL:
        changed_val[name] = "only-in-validation"
    first = train_model(train, val, config("lr"), seed=4)
    second = train_model(train, changed_val, config("lr"), seed=4)
    np.testing.assert_array_equal(first.preprocessor.imputer.statistics_, second.preprocessor.imputer.statistics_)
    np.testing.assert_array_equal(first.preprocessor.scaler.mean_, second.preprocessor.scaler.mean_)
    for categories in second.preprocessor.encoder.categories_:
        assert "only-in-validation" not in categories
    assert np.isfinite(second.predict_scores(changed_val)).all()
    numeric_index = first.preprocessor.numeric_columns.index(first_numeric)
    assert first.preprocessor.imputer.statistics_[numeric_index] == pytest.approx(train[first_numeric].median())


def test_mutual_information_counts_original_fields_and_is_train_only(frames):
    train, val = frames
    settings = config("lr", {"method": "mutual_info", "k": 7})
    original = train_model(train, val, settings, seed=3)
    changed = val.copy()
    changed["label"] = 1 - changed["label"]
    for name in FEATURES:
        changed[name] = "unseen" if name in CATEGORICAL else 1_000_000.0
    other = train_model(train, changed, settings, seed=3)
    assert len(original.selected_features) == 7
    assert original.selected_features == other.selected_features
    assert original.feature_scores == other.feature_scores
    assert len(original.feature_scores) == len(FEATURES)


def test_predict_rejects_bad_schema_and_handles_empty(frames):
    train, val = frames
    bundle = train_model(train, val, config("lr"), seed=1)
    with pytest.raises(ValueError, match="Missing required"):
        bundle.predict_scores(val.drop(columns=[bundle.selected_features[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        bundle.predict_scores(pd.concat([val, val[[FEATURES[0]]]], axis=1))
    numeric = bundle.preprocessor.numeric_columns[0]
    invalid = val.copy()
    invalid[numeric] = np.inf
    with pytest.raises(ValueError, match="infinity"):
        bundle.predict_scores(invalid)
    invalid[numeric] = "invalid"
    with pytest.raises(ValueError, match="non-numeric"):
        bundle.predict_scores(invalid)
    assert bundle.predict_scores(val.iloc[:0]).shape == (0,)


def test_training_rejects_invalid_labels_and_selection(frames):
    train, val = frames
    one_class = train.copy()
    one_class["label"] = 0
    with pytest.raises(ValueError, match="both normal"):
        train_model(one_class, val, config("lr"), seed=0)
    invalid = train.copy()
    invalid.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="binary"):
        train_model(invalid, val, config("lr"), seed=0)
    with pytest.raises(ValueError, match="features.k"):
        train_model(train, val, config("lr", {"method": "mutual_info", "k": 0}), seed=0)


def test_all_missing_training_numeric_keeps_schema(frames):
    train, val = frames
    name = next(name for name in FEATURES if name not in CATEGORICAL)
    train[name] = np.nan
    val[name] = np.nan
    bundle = train_model(train, val, config("lr"), seed=1)
    assert np.isfinite(bundle.predict_scores(val)).all()
    assert name in bundle.preprocessor.numeric_columns


def test_mlp_deterministic_singleton_batch_and_roundtrip(frames, tmp_path):
    pytest.importorskip("torch")
    train, val = frames
    settings = config("mlp", epochs=5, patience=2, batch_size=17, hidden_sizes=[16, 8])
    bundle = train_model(train, val, settings, seed=21, device="cpu")
    scores = bundle.predict_scores(val)
    singletons = np.array([bundle.predict_scores(val.iloc[index:index + 1])[0] for index in range(len(val))])
    np.testing.assert_allclose(scores, singletons, atol=1e-6, rtol=1e-5)
    assert bundle.best_epoch is not None
    assert 1 <= len(bundle.history) <= 5
    assert all(np.isfinite(row["val_loss"]) for row in bundle.history)
    assert all(tensor.device.type == "cpu" for tensor in bundle.network_state.values())
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    restored = load_bundle(path, device="cpu")
    np.testing.assert_array_equal(scores, restored.predict_scores(val))
    repeated = train_model(train, val, settings, seed=21, device="cpu")
    np.testing.assert_array_equal(scores, repeated.predict_scores(val))


def test_mlp_early_stop_restores_best_weights(frames):
    pytest.importorskip("torch")
    train, val = frames
    # An intentionally unreachable improvement stops after one stale epoch.
    settings = config("mlp", epochs=10, patience=1, min_delta=100.0, batch_size=32, hidden_dims=[8])
    stopped = train_model(train, val, settings, seed=5)
    one_epoch = train_model(train, val, config("mlp", epochs=1, batch_size=32, hidden_dims=[8]), seed=5)
    assert len(stopped.history) == 2
    assert stopped.best_epoch == 1
    np.testing.assert_array_equal(stopped.predict_scores(val), one_epoch.predict_scores(val))


def test_mlp_rejects_unsupported_device(frames):
    pytest.importorskip("torch")
    train, val = frames
    with pytest.raises(ValueError, match="Supported MLP devices"):
        train_model(train, val, config("mlp", epochs=1), seed=1, device="meta")
