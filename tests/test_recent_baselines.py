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


def test_piecewise_residual_receives_gradient_from_zero_initialization(tabm_dependencies):
    import torch

    torch.manual_seed(17)
    network = recent.make_tabm_network(2, [], recent._parameters(settings()),
        numeric_bins=[[-1., 0., 1.], [-1., 0., 1.]])
    embedding = network.num_module
    assert torch.count_nonzero(embedding.linear.weight) == 0
    numeric = torch.tensor([[-.7, .3], [.2, -.5], [.8, .9]])
    output = embedding(numeric)
    target = torch.randn_like(output)
    torch.nn.functional.mse_loss(output, target).backward()
    gradient = embedding.linear.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0  # ReLU after the zero projection fails here.


def test_legacy_zero_residual_bundle_keeps_predictions(
        tabm_dependencies, schema_frames, monkeypatch, tmp_path):
    import torch

    schema, train, val = schema_frames
    corrected_factory = recent.make_tabm_network

    def legacy_factory(*args, **kwargs):
        network = corrected_factory(*args, **kwargs)
        network.num_module.activation = torch.nn.ReLU()
        return network

    with monkeypatch.context() as patch:
        patch.setattr(recent, "make_tabm_network", legacy_factory)
        bundle = recent.train_tabm(train, val, settings(epochs=1), 17, feature_schema=schema)
        scores = bundle.predict_scores(val)
    bundle.training_statistics["training_protocol_version"] = 2
    assert torch.count_nonzero(bundle.network_state["num_module.linear.weight"]) == 0
    path = tmp_path / "legacy-tabm.joblib"
    recent.save_tabm_bundle(bundle, path)
    restored = recent.load_tabm_bundle(path)
    assert restored.training_statistics["training_protocol_version"] == 2
    np.testing.assert_array_equal(scores, restored.predict_scores(val))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_ensemble_validation_loss_rewards_complementary_heads(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    # Each class has one very confident correct head and one incorrect head.
    # The deployed mean is more useful than each head's independent loss says.
    probabilities = torch.tensor([[.999, .25], [.001, .75]], dtype=dtype)
    logits = torch.logit(probabilities).unsqueeze(-1)
    truth = torch.tensor([1., 0.], dtype=dtype)
    weight = torch.tensor(2., dtype=dtype)
    expected = -1.5 * torch.log(torch.tensor(.6245, dtype=dtype))
    actual = recent.tabm_ensemble_loss(logits, truth, weight)
    torch.testing.assert_close(actual, expected)
    assert actual < recent.tabm_head_loss(logits, truth, weight)


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_ensemble_validation_loss_saturation_uses_documented_dtype_epsilon(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    logits = torch.tensor([[[-1000.], [-1000.]], [[1000.], [1000.]]], dtype=dtype)
    truth = torch.tensor([1., 0.], dtype=dtype)
    weight = torch.tensor(2., dtype=dtype)
    epsilon = torch.tensor(torch.finfo(dtype).eps, dtype=dtype)
    actual = recent.tabm_ensemble_loss(logits, truth, weight)
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, -1.5 * epsilon.log())


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
    assert bundle.training_statistics["training_protocol_version"] == 3
    assert bundle.training_statistics["early_stopping_metric"] == "val_ensemble_loss"
    assert bundle.training_statistics["validation_objective"] == "weighted_bce_of_mean_sigmoid_probability"
    assert bundle.training_statistics["validation_loss_clamp"] == {
        "policy": "floating_dtype_machine_epsilon", "epsilon": torch.finfo(torch.float32).eps,
        "dtype": "torch.float32"}
    assert all(row["val_loss"] == row["val_ensemble_loss"] for row in bundle.history)
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
    assert restored.training_statistics == bundle.training_statistics
    assert restored.history == bundle.history
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


def test_best_epoch_follows_ensemble_loss_when_head_loss_disagrees(
        tabm_dependencies, schema_frames, monkeypatch, tmp_path):
    import torch

    schema, train, val = schema_frames
    original_forward = recent._forward
    states = []
    # Epoch 1 has lower mean-head loss; epoch 2 has lower deployed-ensemble
    # loss because a confident head compensates for the other head's mistake.
    correct_probabilities = ([.6, .6], [.999, .25], [.55, .55])

    def controlled_validation(network, numeric, categorical, device):
        if network.training:
            return original_forward(network, numeric, categorical, device)
        states.append({name: value.detach().cpu().clone() for name, value in network.state_dict().items()})
        correct = torch.tensor(correct_probabilities[len(states) - 1], dtype=torch.float32,
                               device=device).expand(len(val), -1)
        labels = torch.tensor(val["label"].to_numpy(), device=device)
        probabilities = torch.where(labels[:, None] == 1, correct, 1 - correct)
        return torch.logit(probabilities).unsqueeze(-1)

    with monkeypatch.context() as patch:
        patch.setattr(recent, "_forward", controlled_validation)
        bundle = recent.train_tabm(train, val, settings(epochs=3, patience=1, min_delta=0,
            batch_size=64, k=2, num_embeddings="none"), 17, feature_schema=schema)
    assert len(bundle.history) == 3 and bundle.best_epoch == 2
    assert np.argmin([row["val_head_loss"] for row in bundle.history]) == 0
    assert np.argmin([row["val_ensemble_loss"] for row in bundle.history]) == 1
    for name, value in states[1].items():
        torch.testing.assert_close(bundle.network_state[name], value, rtol=0, atol=0)
    assert bundle.training_statistics["best_validation_loss"] == bundle.history[1]["val_ensemble_loss"]
    assert bundle.training_statistics["positive_weight"] == len(train[train.label == 0]) / len(train[train.label == 1])
    scores = bundle.predict_scores(val)
    path = tmp_path / "ensemble-selected-tabm.joblib"
    recent.save_tabm_bundle(bundle, path)
    restored = recent.load_tabm_bundle(path, device="cpu")
    np.testing.assert_array_equal(scores, restored.predict_scores(val))
    assert restored.best_epoch == 2
    assert restored.history == bundle.history


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
