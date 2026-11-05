"""Synthetic-only checks for frozen-model F diagnostics and exact query budgets."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import xlogy

import ids.unsw_diagnostic_f as diagnostic
from ids.models import FeaturePreprocessor, ModelBundle, _network
from ids.risk_research import _state, _state_hash
from ids.unsw_kd import build_targets

SCHEMA = {"name": "synthetic", "features": ["x", "z"], "categorical": [], "discrete": []}


def fixture(n=2051):
    rng = np.random.default_rng(123)
    y = np.asarray([0]*(n//3)+[1]*(n-n//3))
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "label": y, "sample_id": [f"s{i}" for i in range(n)],
                          "feature_hash": [f"g{i:05d}" for i in range(n)],
                          "attack_cat": np.where(y == 0, "Normal", "Synthetic")})
    # One identical-input pair with conflicting hard labels and shared teacher q.
    frame.loc[n-1, ["x", "z", "feature_hash"]] = frame.loc[0, ["x", "z", "feature_hash"]].to_numpy()
    logits = rng.normal(size=n)
    logits[-1] = logits[0]
    prep = FeaturePreprocessor.fit(frame, SCHEMA["features"], SCHEMA)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        net = _network(prep.encoded_dim, (7, 3))
    base = ModelBundle("mlp", prep, prep.selected_features, prep.encoded_dim, [],
                       network_state=_state(net), hidden_dims=(7, 3), feature_schema=SCHEMA)
    bundles = {name: copy.deepcopy(base) for name in diagnostic.MODELS}
    for index, name in enumerate(("H40", "K40"), 1):
        for value in bundles[name].network_state.values():
            value.add_(index*.025)
    _, _, arrays = build_targets(frame, logits)
    return bundles, frame, arrays


def test_three_forwards_four_shared_graph_backwards_and_no_optimizer(tmp_path, monkeypatch):
    bundles, frame, arrays = fixture()
    state_before = {name: _state_hash(b.network_state) for name, b in bundles.items()}
    calls = []
    original = diagnostic._network

    def counting_network(*args, **kwargs):
        net = original(*args, **kwargs)
        net.register_forward_hook(lambda module, inputs, output: calls.append(len(inputs[0])))
        return net

    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError("A diagnostic must never construct an optimizer")

    monkeypatch.setattr(diagnostic, "_network", counting_network)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden_optimizer)
    monkeypatch.setattr(torch.optim, "SGD", forbidden_optimizer)
    initial_threads = torch.get_num_threads()
    initial_rng = torch.random.get_rng_state().clone()
    summary = diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"diagnosis", "cpu")
    n_batches = math.ceil(len(frame)/1024)
    assert sum(calls) == len(frame)*3
    assert len(calls) == 3*n_batches
    assert calls == [1024, 1024, 3]*3
    assert summary["budget"]["backward_calls"] == 4*n_batches
    assert summary["budget"]["optimizer_updates"] == 0
    assert summary["valid"]
    assert all(m["finite"] and m["bounds_valid"] and m["state_checks"]["unchanged"]
               for m in summary["models"].values())
    assert {name: _state_hash(b.network_state) for name, b in bundles.items()} == state_before
    assert torch.get_num_threads() == initial_threads
    torch.testing.assert_close(torch.random.get_rng_state(), initial_rng, rtol=0, atol=0)
    scores = pd.read_parquet(tmp_path/"diagnosis/fit-scores.parquet")
    pd.testing.assert_frame_equal(scores.loc[:, diagnostic.IDENTITY],
                                  frame.loc[:, diagnostic.IDENTITY], check_dtype=True)
    for name in diagnostic.MODELS:
        assert scores[name+"_logit"].dtype == np.float64
        assert np.array_equal(scores[name+"_logit"], scores[name+"_logit"].to_numpy(dtype=np.float32))
    assert len(pd.read_csv(tmp_path/"diagnosis/gradient-batches.csv")) == n_batches
    pattern = pd.read_parquet(tmp_path/"diagnosis/residuals-by-pattern.parquet")
    assert len(pattern) == (len(frame)-1)*3


def test_anchor_gradients_match_independent_whole_f_autograd(tmp_path):
    bundles, frame, arrays = fixture(1033)
    summary = diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"diagnosis", "cpu")
    base = bundles["anchor"]
    net = _network(base.encoded_dim, base.hidden_dims)
    net.load_state_dict(base.network_state)
    x = torch.from_numpy(base.preprocessor.transform(frame))
    z = net(x).squeeze(1).double()
    y = torch.from_numpy(arrays["labels"].astype(np.float64))
    w = torch.from_numpy(arrays["weights"])
    q = torch.from_numpy(arrays["teacher_q"])
    m = torch.from_numpy(arrays["class_mean_q"])
    r = torch.from_numpy(arrays["target_K"])
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    entropy_q = -(torch.special.xlogy(q, q)+torch.special.xlogy(1-q, 1-q))
    entropy_m = -(torch.special.xlogy(m, m)+torch.special.xlogy(1-m, 1-m))
    objectives = {
        "hard_bce": (w*bce(z, y, reduction="none")).sum(),
        "teacher_kl": (w*(bce(z, q, reduction="none")-entropy_q)).sum(),
        "class_mean_kl": (w*(bce(z, m, reduction="none")-entropy_m)).sum(),
        "mixed_bce": (w*bce(z, r, reduction="none")).sum(),
    }
    with np.load(tmp_path/"diagnosis/anchor-gradients.npz", allow_pickle=False) as saved:
        for term, objective in objectives.items():
            grads = torch.autograd.grad(objective, tuple(net.parameters()), retain_graph=True)
            expected = torch.cat([p.reshape(-1) for p in grads]).detach().numpy()
            np.testing.assert_allclose(saved[term], expected, atol=1e-7, rtol=1e-5)
    assert summary["gradient_identity"]["valid"]
    # Whole-F and chunked float32 GEMMs need not produce bit-identical logits.
    assert summary["models"]["anchor"]["losses"]["mixed_bce"] == pytest.approx(
        float(objectives["mixed_bce"].detach()), abs=1e-7)


def test_pattern_bound_matches_direct_group_enumeration_and_residual_totals(tmp_path):
    bundles, frame, arrays = fixture(31)
    summary = diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"diagnosis", "cpu")
    w, r = arrays["weights"], arrays["target_K"]
    expected = 0.
    for pattern in frame.feature_hash.unique():
        selected = frame.feature_hash.to_numpy() == pattern
        pstar = np.sum(w[selected]*r[selected])/np.sum(w[selected])
        expected += float(np.sum(w[selected]*(-xlogy(r[selected], pstar)-xlogy(1-r[selected], 1-pstar))))
    assert summary["pattern_bounds"]["K"]["minimum_bce"] == pytest.approx(expected, abs=1e-14)
    assert summary["pattern_bounds"]["H"]["same_input_conflict"] > 0
    for filename in ("residuals-by-class.csv", "residuals-by-family.csv"):
        table = pd.read_csv(tmp_path/"diagnosis"/filename)
        for name in diagnostic.MODELS:
            sub = table[table.model == name]
            assert sub.rows.sum() == len(frame)
            assert sub.mixed_bce_weighted_sum.sum() == pytest.approx(
                summary["models"][name]["losses"]["mixed_bce"], abs=1e-12)
    quantiles = pd.read_csv(tmp_path/"diagnosis/residuals-by-quantile.csv")
    for _, rows in quantiles.groupby(["model", "driver"]):
        assert rows.rows.sum() == len(frame)
        assert rows.weight_mass.sum() == pytest.approx(1.)


@pytest.mark.parametrize("corruption", ["labels", "weights", "target", "group", "q"])
def test_input_mismatch_rejected_before_forward(tmp_path, monkeypatch, corruption):
    bundles, frame, arrays = fixture(19)
    if corruption == "labels":
        arrays["labels"][0] = 1
    elif corruption == "weights":
        arrays["weights"][0] *= 2
    elif corruption == "target":
        arrays["target_K"][0] += .001
    elif corruption == "group":
        arrays["row_group_index"] = arrays["row_group_index"][::-1]
    else:
        arrays["teacher_q"][0] = np.nan

    def no_network(*args, **kwargs):
        raise AssertionError("Invalid input must be rejected before forward")
    monkeypatch.setattr(diagnostic, "_network", no_network)
    with pytest.raises(ValueError):
        diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"invalid", "cpu")


def test_preprocessing_mismatch_is_rejected(tmp_path):
    bundles, frame, arrays = fixture(19)
    bundles["K40"].preprocessor.scaler.mean_[0] += 1.
    with pytest.raises(ValueError, match="preprocessing"):
        diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"invalid", "cpu")


def test_nonfinite_model_fails_without_mutating_bundle_or_torch_settings(tmp_path):
    bundles, frame, arrays = fixture(19)
    next(iter(bundles["anchor"].network_state.values())).fill_(torch.finfo(torch.float32).max)
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with pytest.raises(FloatingPointError, match="logits"):
        diagnostic.diagnose_f(bundles, frame, arrays, tmp_path/"invalid", "cpu")
    assert torch.get_num_threads() == threads
    assert torch.are_deterministic_algorithms_enabled() == deterministic


def test_zero_norm_cosine_is_undefined():
    assert diagnostic._cosine(np.zeros(3), np.ones(3)) is None
    assert diagnostic._cosine(np.ones(3), np.ones(3)) == pytest.approx(1.)
