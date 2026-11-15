"""Synthetic checks of weighted learning and the frozen eight-field contract."""
from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.threshold_validation.observations import RAW_FEATURES

torch = pytest.importorskip("torch")
old_mlp = import_module("ids.flow_history.model")
old_hgb = import_module("ids.flow_history.trees")
model = import_module("ids.support_diagnostic.model")


def synthetic(rows=48):
    rng = np.random.default_rng(20261115)
    frame = pd.DataFrame(rng.uniform(0, 100, (rows, 8)), columns=RAW_FEATURES)
    frame["ip_protocol"] = 17
    frame["orig_port"] = rng.integers(1024, 65535, rows)
    frame["resp_port"] = np.where(np.arange(rows) % 2, 443, 53)
    frame["label"] = (np.arange(rows) % 2).astype(int)
    return frame


def test_weighted_bce_ones_matches_loss_and_gradient_and_weights_act():
    logits = torch.tensor([-1.0, 0.2, 2.0, -0.7], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    ordinary = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    ordinary_grad = torch.autograd.grad(ordinary, logits, retain_graph=True)[0]
    actual = model.weighted_bce(logits, labels, torch.ones_like(labels))
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    torch.testing.assert_close(actual, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, ordinary_grad, rtol=0, atol=0)
    weights = torch.tensor([0.5, 1.5, 0.5, 1.5])
    weighted = model.weighted_bce(logits, labels, weights)
    weighted_grad = torch.autograd.grad(weighted, logits)[0]
    torch.testing.assert_close(weighted_grad, ordinary_grad * weights)
    assert not torch.equal(weighted_grad, ordinary_grad)


def test_global_weight_normalization_does_not_use_minibatch_sum():
    labels = np.array([0, 0, 1, 1])
    weights, info = model._weights(np.array([2, 2, 6, 6]), labels)
    np.testing.assert_array_equal(weights, [0.5, 0.5, 1.5, 1.5])
    assert info["normalized_mean"] == 1 and info["input_mean"] == 4
    logits = torch.zeros(2, requires_grad=True)
    targets = torch.zeros(2)
    result = model.weighted_bce(logits, targets, torch.from_numpy(weights[:2]))
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) * 0.5
    torch.testing.assert_close(result, expected.to(result.dtype))
    with pytest.raises(ValueError, match="positive global mean"):
        model._weights(np.zeros(4), labels)


def test_port_mask_changes_only_ports_without_mutation():
    frame = synthetic()
    saved = frame.copy(deep=True)
    actual = model.view_frame(frame, "ports_zero8")
    assert (actual[["orig_port", "resp_port"]].to_numpy() == 0).all()
    other = [name for name in frame.columns if name not in {"orig_port", "resp_port"}]
    pd.testing.assert_frame_equal(actual[other], frame[other])
    pd.testing.assert_frame_equal(frame, saved)
    assert model.view_frame(frame, "raw8") is frame


def test_mlp_p_only_preprocessing_and_legacy_loader_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "EPOCHS", 2)
    p = synthetic()
    future = synthetic(12)
    future.loc[:, "orig_ip_bytes"] = 1e12
    path = model.fit(p, "raw8", tmp_path / "model.pt", group="mlp17",
                     weights=np.ones(len(p)), threads=1, device="cpu")
    fitted = model.load(path, group="mlp17", threads=1, device="cpu")
    transformed = old_mlp._transform(p, RAW_FEATURES)
    np.testing.assert_array_equal(fitted.backend.mean, transformed.mean(axis=0))
    np.testing.assert_allclose(transformed[:, 1], np.log1p(p.orig_port))
    mean_before = fitted.backend.mean.copy()
    scored = model.score(fitted, future, "raw8", "mlp17")
    np.testing.assert_array_equal(fitted.backend.mean, mean_before)
    legacy = old_mlp.load_arm(path, threads=1, device="cpu")
    np.testing.assert_array_equal(scored, old_mlp.score_arm(legacy, future, RAW_FEATURES))
    metadata = read_json(path.with_suffix(".fit.json"))
    assert metadata["base_width"] == 8 and metadata["seed"] == 17
    assert metadata["weighting"]["normalized_mean"] == 1
    assert metadata["preprocessing"]["mean"] == mean_before.tolist()


def test_mlp_all_one_fit_matches_old_training(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "EPOCHS", 2)
    monkeypatch.setattr(old_mlp, "EPOCHS", 2)
    p = synthetic()
    new_path = model.fit(p, "raw8", tmp_path / "weighted.pt", group="mlp29",
                         weights=np.ones(len(p)), threads=1, device="cpu")
    old_path = old_mlp.fit_arm(p, RAW_FEATURES, tmp_path / "old.pt", threads=1,
                               device="cpu", seed=29, base_width=8)
    new_state = torch.load(new_path, weights_only=True)
    old_state = torch.load(old_path, weights_only=True)
    for key in old_state["weights"]:
        # Unreduced BCE then mean has a different FP32 reduction path from
        # fused BCE(mean); compare within the existing arm's score tolerance.
        torch.testing.assert_close(new_state["weights"][key], old_state["weights"][key],
                                   rtol=3e-5, atol=3e-5)
    legacy = model.load(old_path, group="mlp29", threads=1, device="cpu")
    weighted = model.load(new_path, group="mlp29", threads=1, device="cpu")
    np.testing.assert_allclose(model.score(weighted, p, "raw8", "mlp29"),
                               model.score(legacy, p, "raw8", "mlp29"), rtol=3e-5, atol=3e-5)
    assert legacy.view == "raw8"
    with pytest.raises(ValueError, match="view or group"):
        model.score(legacy, p, "ports_zero8", "mlp29")


def test_saved_masked_mlp_ignores_ports_and_refuses_wrong_view(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "EPOCHS", 2)
    p = synthetic()
    path = model.fit(p, "ports_zero8", tmp_path / "masked.pt", group="mlp43",
                     weights=np.where(p.label, 1.5, 0.5), threads=1, device="cpu")
    fitted = model.load(path, group="mlp43", threads=1, device="cpu")
    np.testing.assert_array_equal(fitted.backend.mean[[1, 2]], [0, 0])
    changed = p.assign(orig_port=12345, resp_port=9999)
    np.testing.assert_array_equal(model.score(fitted, p, "ports_zero8", "mlp43"),
                                  model.score(fitted, changed, "ports_zero8", "mlp43"))
    with pytest.raises(ValueError, match="view or group"):
        model.score(fitted, p, "raw8", "mlp43")
    with pytest.raises(ValueError, match="model group"):
        model.load(path, group="mlp17", threads=1, device="cpu")


def test_hgb_uses_weights_and_legacy_fit_and_saved_view(tmp_path, monkeypatch):
    monkeypatch.setitem(old_hgb.PARAMETERS, "max_iter", 3)
    p = synthetic(80)
    weights = np.where(p.label, 1.5, 0.5)
    seen = {}
    original_fit = old_hgb.fit_arm

    def recording_fit(frame, features, path, **kwargs):
        seen["weights"] = kwargs["sample_weight"].copy()
        seen["frame"] = frame.copy()
        return original_fit(frame, features, path, **kwargs)

    monkeypatch.setattr(old_hgb, "fit_arm", recording_fit)
    path = model.fit(p, "ports_zero8", tmp_path / "hgb.joblib", group="hgb",
                     weights=weights, threads=2, device="cpu")
    np.testing.assert_array_equal(seen["weights"], weights)
    assert (seen["frame"][["orig_port", "resp_port"]].to_numpy() == 0).all()
    fitted = model.load(path, group="hgb", threads=2, device="cpu")
    old = old_hgb.load_arm(path, threads=2)
    np.testing.assert_array_equal(model.score(fitted, p, "ports_zero8", "hgb"),
                                  old_hgb.score_arm(old, model.view_frame(p, "ports_zero8"), RAW_FEATURES))
    metadata = read_json(path.with_suffix(".fit.json"))
    assert metadata["device"] == "cpu" and metadata["threads"] == 2
    assert metadata["weighting"]["class_sums"] == {"0": 20.0, "1": 60.0}
