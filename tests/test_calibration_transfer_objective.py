"""Synthetic tensors and tiny networks only: no real model/input evaluation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from ids.route_batch.tabm_training import stable_scores
from scripts.calibration_transfer.objective import backward_transfer, score_full, score_gradient
from scripts.normal_tail_mechanism import transfer_penalty


class TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(.7)

    def forward(self, numeric, categorical):
        return self.linear(self.dropout(numeric)).unsqueeze(-1)


def example():
    torch.manual_seed(17)
    model = TinyHeads()
    x = np.random.default_rng(3).normal(size=(13, 3)).astype(np.float32)
    episodes = [{"id": "first", "Q": [0, 1, 2], "E": [3, 4, 5]},
                {"id": "second", "Q": [0, 1, 2, 3, 4, 5], "E": [6, 7, 8, 9]}]
    return model, x, episodes, [10, 11, 12]


def test_replay_gradient_matches_direct_autograd_and_adds_to_bce():
    model, x, episodes, attacks = example()
    direct = deepcopy(model)
    # Existing BCE gradients must survive the extra backward call.
    for network in (model, direct):
        network.eval()
        torch.nn.functional.binary_cross_entropy_with_logits(
            network(torch.from_numpy(x[:4]), None), torch.ones(4, 3, 1)).backward()
    scored = score_full(model, x, None, device="cpu", batch_size=4)
    replay = backward_transfer(model, x, None, episodes, attacks, device="cpu", batch_size=4,
                               score_pass=scored)
    direct_scores = torch.cat([stable_scores(direct(torch.from_numpy(x[i:i+4]), None))
                               for i in range(0, len(x), 4)])
    objective = transfer_penalty([(direct_scores[e["Q"]], direct_scores[e["E"]])
                                  for e in episodes], direct_scores[attacks])
    objective["penalty"].backward()
    for actual, expected in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-7)
    assert replay["penalty"] == float(objective["penalty"].detach())
    assert replay["cost"]["discovery_forward_rows_this_call"] == 0
    assert 0 < replay["cost"]["replay_backward_rows"] <= len(x)


def test_both_boundaries_and_tie_subgradients_match_frozen_operator():
    gradient, report = score_gradient([0., 2., 10.], [([0], [1])], [2])
    torch.testing.assert_close(gradient, torch.tensor([-1., 1., 0.], dtype=torch.float64))
    assert report["episodes"][0]["normal_fp"] == 1
    assert report["episodes"][0]["normal_budget"] == 0
    # The two equal worst episode terms share max's subgradient; the two equal
    # minimum positives share min's subgradient, exactly as in the original op.
    values = torch.tensor([0., 1., 1., 0., 0.], dtype=torch.float64, requires_grad=True)
    episodes = [([0], [1]), ([0], [2])]
    expected = transfer_penalty([(values[q], values[e]) for q, e in episodes], values[[3, 4]])
    expected["penalty"].backward()
    actual, _ = score_gradient(values, episodes, [3, 4])
    torch.testing.assert_close(actual, values.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual, torch.tensor([0., .5, .5, -.5, -.5], dtype=torch.float64))


def test_score_is_mean_probability_log_odds_not_mean_logit():
    logits = torch.tensor([[[8.], [-1.], [-1.]], [[1000.], [-1000.], [1000.]]])
    actual = stable_scores(logits)
    expected = torch.logit(logits.double().sigmoid().mean(dim=1).squeeze(-1))
    torch.testing.assert_close(actual, expected)
    assert abs(float(actual[0]) - float(logits[0].mean())) > 1.
    assert torch.isfinite(actual).all()


def test_discovery_and_replay_preserve_dropout_rng_and_training_mode():
    model, x, episodes, attacks = example()
    model.train()
    before = torch.get_rng_state().clone()
    report = backward_transfer(model, x, None, episodes, attacks, device="cpu", batch_size=4)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert report["cost"]["discovery_forward_rows_this_call"] == len(x)


def test_updated_parameters_reject_stale_discovery_before_backward():
    model, x, episodes, attacks = example()
    scored = score_full(model, x, None, device="cpu", batch_size=4)
    with torch.no_grad():
        model.linear.bias.add_(.01)
    with pytest.raises(RuntimeError, match="Stale"):
        backward_transfer(model, x, None, episodes, attacks, device="cpu", batch_size=4,
                          score_pass=scored)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_gradient_diagnostics_do_not_replay_or_invent_budget_success():
    # Uniform shrinkage can reduce the gap without reducing strict integer FP.
    _, before = score_gradient([0., 2., 10.], [([0], [1])], [2])
    _, after = score_gradient([0., .4, 2.], [([0], [1])], [2])
    assert after["penalty"] < before["penalty"]
    assert after["episodes"][0]["normal_fp"] == before["episodes"][0]["normal_fp"] == 1
    model, x, episodes, attacks = example()
    report = backward_transfer(model, x, None, episodes, attacks, device="cpu", batch_size=4,
                               backward=False)
    assert report["cost"]["replay_forward_rows"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_invalid_normal_positive_overlap_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        score_gradient([1., 2., 3.], [([0], [1])], [1, 2])
