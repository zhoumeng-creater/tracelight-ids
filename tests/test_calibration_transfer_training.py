"""Synthetic tiny-network trajectories exercise the common training adapter."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.calibration_transfer import training


class TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, numeric, categorical):
        return self.linear(self.dropout(numeric)).unsqueeze(-1)


def inputs():
    numeric = np.random.default_rng(4).normal(size=(12, 3)).astype(np.float32)
    categorical = np.zeros((12, 1), dtype=np.int64)
    labels = np.array([0] * 8 + [1] * 4, dtype=np.float32)
    episodes = [{"id": "small", "Q": [0, 1, 2, 3], "E": [4, 5, 6, 7]}]
    spec = dict(training.FIXED_TRAINING, optimizer_steps=3, batch_size=4,
                draws_per_class=2, score_batch_size=4, report_every=2)
    return numeric, categorical, labels, episodes, spec


def test_paired_arms_keep_initialization_sampler_and_dropout_streams(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    prototype = TinyHeads()
    numeric, categorical, labels, episodes, spec = inputs()
    results, states = {}, {}
    initial = training.state_digest(prototype.state_dict())
    for arm in training.ARMS:
        model = deepcopy(prototype)
        assert training.state_digest(model.state_dict()) == initial
        out = tmp_path / arm
        out.mkdir()
        results[arm] = training.train_loop(model, numeric, categorical, labels, episodes,
                                          arm=arm, seed=17, training_spec=spec, device="cpu", output_dir=out)
        states[arm] = training.state_digest(model.state_dict())
    for key in ("initial_dropout_rng_sha256", "final_dropout_rng_sha256", "sampling"):
        assert results["A_BCE"][key] == results["B_permuted"][key] == results["C_real"][key]
    assert states["B_permuted"] == states["C_real"]  # Same supplied synthetic episode set.
    assert states["A_BCE"] != states["C_real"]
    assert results["A_BCE"]["cost"]["transfer_discovery_forward_rows"] == 0
    assert results["C_real"]["cost"]["transfer_discovery_forward_rows"] == len(labels) * 3
    assert results["C_real"]["cost"]["bce_forward_rows"] == 4 * 3
    assert results["C_real"]["optimizer_steps"] == 3
    assert [r["step"] for r in results["C_real"]["history"]] == [2, 3]
    assert results["C_real"]["history"][-1]["last_transfer_at_parameter_step"] == 2


def test_class_sampler_cycles_without_replacement_until_exhaustion():
    labels = np.array([0] * 5 + [1] * 3)
    one = training.ClassSampler(labels, seed=29, per_class=2)
    two = training.ClassSampler(labels, seed=29, per_class=2)
    normal, attack = [], []
    for _ in range(4):
        draw = one.next()
        np.testing.assert_array_equal(draw, two.next())
        np.testing.assert_array_equal(labels[draw], [0, 0, 1, 1])
        normal.extend(draw[:2])
        attack.extend(draw[2:])
    assert set(normal[:5]) == set(range(5))
    assert set(attack[:3]) == {5, 6, 7}
    assert one.audit()["classes"]["0"]["completed_cycles"] == 1
    assert one.audit()["classes"]["1"]["completed_cycles"] == 2
    assert one.audit()["draw_sha256"] == two.audit()["draw_sha256"]


def test_production_rejects_shortened_budget_before_model_construction(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A real model must not be constructed")
    monkeypatch.setattr(training, "make_tabm_network", forbidden)
    with pytest.raises(ValueError, match="registration"):
        training.fit([], None, tmp_path / "fit", arm="A_BCE", seed=17,
                     training_spec=dict(training.FIXED_TRAINING, optimizer_steps=3), device="cpu")
    assert not (tmp_path / "fit").exists()


def test_loop_uses_unit_transfer_weight_without_sparse_schedule():
    _, _, _, _, spec = inputs()
    assert training.FIXED_TRAINING["optimizer_steps"] == 15600
    assert training.FIXED_TRAINING["score_batch_size"] == 1024
    assert spec["transfer_weight"] == 1.
    assert training.FIXED_MODEL["n_bins"] == 48
    assert training.FIXED_MODEL["embedding_dim"] == 16


@pytest.mark.parametrize("scoring_fails", [False, True])
def test_predictor_reuses_training_arithmetic_and_restores_callers_flags(monkeypatch, scoring_fails):
    def flags():
        return (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)

    def set_flags(values):
        torch.use_deterministic_algorithms(values[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = values[1:3]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = values[3:5]

    original = flags()
    caller_flags = (False, True, False, True, True)
    numeric = np.ones((2, 3), dtype=np.float32)
    categorical = np.zeros((2, 1), dtype=np.int64)
    rows = [{"synthetic": 1}, {"synthetic": 2}]
    network = object()
    calls = []

    class SyntheticPreprocessor:
        def transform(self, received):
            assert received is rows
            return numeric, categorical

    def synthetic_score(received_network, received_numeric, received_categorical, *, device, batch_size):
        assert received_network is network
        assert received_numeric is numeric and received_categorical is categorical
        assert device == "cpu" and batch_size == 4
        # The scoring entrypoint, rather than only a context helper in isolation,
        # must enforce the same arithmetic as training-time deployed scores.
        assert flags() == (True, False, True, False, False)
        calls.append("score")
        if scoring_fails:
            raise RuntimeError("synthetic scoring failure")
        return SimpleNamespace(scores=torch.tensor([-1., 2.], dtype=torch.float64))

    monkeypatch.setattr(training, "score_full", synthetic_score)
    predictor = training.Predictor(network, SyntheticPreprocessor(), "cpu", 4)
    try:
        set_flags(caller_flags)
        if scoring_fails:
            with pytest.raises(RuntimeError, match="synthetic scoring failure"):
                predictor.score(rows)
        else:
            np.testing.assert_array_equal(predictor.score(rows), [-1., 2.])
        assert calls == ["score"]
        assert flags() == caller_flags
    finally:
        set_flags(original)
