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


def test_distillation_loss_matches_weighted_hard_unweighted_soft_formula():
    torch = pytest.importorskip("torch")
    from ids.models import _distillation_loss
    logits = torch.tensor([-1.3, .2, 1.7], dtype=torch.float64)
    y = torch.tensor([1., 0., 1.], dtype=torch.float64)
    q = torch.tensor([.15, .6, .8], dtype=torch.float64)
    weight = torch.tensor(3.2, dtype=torch.float64)
    p = torch.sigmoid(logits)
    hard = -(weight * y * torch.log(p) + (1 - y) * torch.log1p(-p)).mean()
    soft = -(q * torch.log(p) + (1 - q) * torch.log1p(-p)).mean()
    assert _distillation_loss(logits, y, q, weight, .4).item() == pytest.approx((.6 * hard + .4 * soft).item())
    assert _distillation_loss(logits, y, None, weight, 0).item() == pytest.approx(hard.item())
    optimum = torch.logit(q).detach().requires_grad_()
    _distillation_loss(optimum, y, q, weight, 1).backward()
    torch.testing.assert_close(optimum.grad, torch.zeros_like(optimum), atol=1e-15, rtol=0)


def test_distillation_alpha_zero_is_exactly_the_hard_training_path(frames):
    torch = pytest.importorskip("torch")
    import copy
    train, val = frames
    hard = config("mlp", epochs=3, patience=2, batch_size=17, hidden_sizes=[8, 4])
    disabled = copy.deepcopy(hard)
    disabled["distillation"] = {"alpha": 0, "temperature": 1, "teacher": {"model": "hgb"}}
    first = train_model(train, val, hard, seed=19)
    second = train_model(train, val, disabled, seed=19)
    for key in first.network_state:
        assert torch.equal(first.network_state[key], second.network_state[key])
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    assert second.training_statistics["training_objective"] == "weighted_hard_bce"
    assert second.training_statistics["distillation"]["enabled"] is False


def test_kd_tensor_dataset_keeps_soft_scores_aligned_and_initialization_paired(frames, monkeypatch):
    torch = pytest.importorskip("torch")
    from ids import models
    train, val = frames
    hard = config("mlp", epochs=2, patience=2, batch_size=17, hidden_sizes=[8, 4])
    import copy
    kd = copy.deepcopy(hard)
    kd["distillation"] = {"alpha": .5, "temperature": 1, "teacher": {"model": "hgb"}}
    preprocessor = models.FeaturePreprocessor.fit(train, list(FEATURES))
    q = 1 / (1 + np.exp(-preprocessor.transform(train)[:, 0]))
    initial_states, batches = [], []
    original_network = models._network
    original_loader = torch.utils.data.DataLoader

    def network(*args):
        result = original_network(*args)
        initial_states.append({k: v.detach().clone() for k, v in result.state_dict().items()})
        return result

    def loader(*args, **kwargs):
        underlying = original_loader(*args, **kwargs)
        observation = []
        batches.append(observation)

        class Recorder:
            def __iter__(self):
                for batch in underlying:
                    observation.append(batch[0].numpy().copy())
                    if len(batch) == 3:
                        torch.testing.assert_close(batch[2], torch.sigmoid(batch[0][:, 0]), atol=1e-7, rtol=1e-6)
                    yield batch

        return Recorder()

    monkeypatch.setattr(models, "_network", network)
    monkeypatch.setattr(torch.utils.data, "DataLoader", loader)
    first = train_model(train, val, hard, seed=13)
    second = train_model(train, val, kd, seed=13, soft_targets=q)
    assert len(initial_states) == 2
    for key in initial_states[0]:
        assert torch.equal(initial_states[0][key], initial_states[1][key])
    assert len(batches[0]) == len(batches[1])
    for left, right in zip(*batches):
        np.testing.assert_array_equal(left, right)
    assert first.training_statistics["parameter_count"] == second.training_statistics["parameter_count"]


def test_kd_early_stopping_uses_hard_validation_loss_and_bundle_stays_student_only(frames, tmp_path):
    torch = pytest.importorskip("torch")
    from ids import models
    train, val = frames
    settings = config("mlp", epochs=3, patience=1, min_delta=100., batch_size=32, hidden_sizes=[8])
    settings["distillation"] = {"alpha": .5, "temperature": 1, "teacher": {"model": "hgb"}}
    bundle = train_model(train, val, settings, seed=5, soft_targets=np.full(len(train), .95))
    assert bundle.best_epoch == 1 and len(bundle.history) == 2
    network = models._network(bundle.encoded_dim, bundle.hidden_dims)
    network.load_state_dict(bundle.network_state)
    logits = network(torch.from_numpy(bundle.preprocessor.transform(val))).squeeze(1)
    truth = torch.from_numpy(val["label"].to_numpy(dtype=np.float32))
    weight = torch.tensor((train["label"] == 0).sum() / (train["label"] == 1).sum(), dtype=torch.float32)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, truth, pos_weight=weight).item()
    assert bundle.training_statistics["best_validation_loss"] == pytest.approx(expected, abs=1e-7)
    assert bundle.training_statistics["parameter_count"] == sum(x.numel() for x in network.parameters())
    assert bundle.training_statistics["epochs_completed"] == 2
    assert not {"teacher", "soft_targets", "oof_scores", "targets"}.intersection(vars(bundle))
    target = tmp_path / "student.joblib"
    save_bundle(bundle, target)
    restored = load_bundle(target)
    np.testing.assert_array_equal(bundle.predict_scores(val), restored.predict_scores(val))
    assert restored.training_statistics == bundle.training_statistics


@pytest.mark.parametrize("case", ["without_config", "classical", "missing", "temperature", "alpha", "shape", "length", "nan", "range", "list"])
def test_invalid_distillation_inputs_are_rejected_before_fitting(frames, monkeypatch, case):
    from ids import models
    train, val = frames
    settings = config("mlp", epochs=1)
    settings["distillation"] = {"alpha": .5, "temperature": 1}
    targets = np.full(len(train), .5)
    if case == "without_config":
        del settings["distillation"]
    elif case == "classical":
        settings["model"] = "hgb"
    elif case == "missing":
        targets = None
    elif case == "temperature":
        settings["distillation"]["temperature"] = 2
    elif case == "alpha":
        settings["distillation"]["alpha"] = float("nan")
    elif case == "shape":
        targets = targets[:, None]
    elif case == "length":
        targets = targets[:-1]
    elif case == "nan":
        targets[0] = float("nan")
    elif case == "range":
        targets[0] = 1.01
    elif case == "list":
        targets = targets.tolist()
    monkeypatch.setattr(models.FeaturePreprocessor, "fit", lambda *_a, **_k: pytest.fail("invalid targets reached fit"))
    with pytest.raises(ValueError, match="distillation|soft_targets"):
        train_model(train, val, settings, seed=1, soft_targets=targets)
