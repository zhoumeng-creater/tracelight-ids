"""Synthetic scores/networks only; no real F model construction or evaluation."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from ids.route_batch.tabm_training import stable_scores
from scripts.comprehensive_batch.objective import (
    ARMS, backward_objective, objective_terms, score_full, score_gradient,
)


def groups(refs, targets, attacks):
    return {"references": [{"id": f"reference-{i}", "source": str(i), "indices": values}
                            for i, values in enumerate(refs)],
            "targets": [{"id": f"target-{i}", "source": str(i), "indices": values}
                         for i, values in enumerate(targets)], "attacks": attacks}


def test_group_is_weaker_and_does_not_ensure_reference_calibration():
    registered = groups([[0], [1]], [[0], [1]], [2])
    _, plain = score_gradient([0., 2., 10.], registered, "G_group_tail")
    _, reference = score_gradient([0., 2., 10.], registered, "R_reference_envelope")
    assert plain["penalty"] == 0.
    assert reference["penalty"] == 2.
    assert not plain["all_reference_target_budgets_satisfied"]
    # A lower reference boundary alerts on the higher business group's only row.
    pair = next(x for x in reference["reference_target_pairs"]
                if x["reference_id"] == "reference-0" and x["target_id"] == "target-1")
    assert pair["normal_fp"] == 1 and pair["integer_budget"] == 0


def test_group_penalty_never_exceeds_reference_envelope_for_random_scores():
    generator = np.random.default_rng(23)
    registered = groups([list(range(100)), list(range(100, 200))],
                        [list(range(99)), list(range(99, 151)), list(range(151, 200))],
                        list(range(200, 206)))
    for alpha in (0., .01, .2):
        for _ in range(30):
            scores = torch.tensor(generator.normal(size=206), dtype=torch.float64)
            terms = objective_terms(scores, registered, "R_reference_envelope", alpha=alpha)
            assert float(terms["group_penalty"]) <= float(terms["penalty"]) + 1e-14


def test_reference_zero_means_exact_integer_budgets_and_attack_margins():
    registered = groups([list(range(100)), list(range(100, 200))],
                        [list(range(100)), list(range(100, 200))], [200, 201])
    scores = np.zeros(202)
    scores[99] = .5
    scores[199] = .7
    scores[200:] = [1., 2.]
    derivative, report = score_gradient(scores, registered, "R_reference_envelope")
    assert report["penalty"] == 0.
    assert report["group_penalty"] == 0.
    assert report["all_reference_target_budgets_satisfied"]
    assert report["all_reference_attack_margins_satisfied"]
    assert all(x["normal_fp"] == x["integer_budget"] == 1
               for x in report["reference_target_pairs"])
    assert not bool(derivative.ne(0).any())


def test_floor_gap_exposes_tie_requirement_without_relaxing_comparison():
    # Full-source budget 2; disjoint business budgets 0 + 1. Distinct scores at
    # the source rank leave 2 FPs, so the business constraints cannot both pass.
    registered = groups([list(range(200))], [list(range(99)), list(range(99, 200))], [200])
    scores = np.r_[np.arange(200, dtype=float), 300.]
    _, before = score_gradient(scores, registered, "R_reference_envelope")
    assert before["references"][0]["boundary_tie_count"] == 1
    assert not before["all_reference_target_budgets_satisfied"]
    assert before["group_penalty"] == 0. and before["penalty"] == 1.
    # Exactly two tied boundary rows achieve the required pooled slack. Near
    # equality is insufficient; no epsilon tie rule is used.
    scores[198] = np.nextafter(197., np.inf)
    _, near = score_gradient(scores, registered, "R_reference_envelope")
    assert not near["all_reference_target_budgets_satisfied"]
    scores[198] = 197.
    _, after = score_gradient(scores, registered, "R_reference_envelope")
    assert after["references"][0]["boundary_tie_count"] == 2
    assert after["all_reference_target_budgets_satisfied"]
    assert after["penalty"] == 0.


def test_shrinking_gap_does_not_invent_integer_budget_improvement():
    registered = groups([[0]], [[1]], [2])
    _, before = score_gradient([0., 2., 10.], registered, "R_reference_envelope")
    _, after = score_gradient([0., .4, 2.], registered, "R_reference_envelope")
    assert 0. < after["penalty"] < before["penalty"]
    assert before["reference_target_pairs"][0]["normal_fp"] == 1
    assert after["reference_target_pairs"][0]["normal_fp"] == 1


def test_gradients_reach_both_reference_extrema_and_positive_margin():
    registered = groups([[0], [1]], [[0], [1]], [2])
    actual, report = score_gradient([0., 2., 1.], registered, "R_reference_envelope")
    torch.testing.assert_close(actual, torch.tensor([-1., 2., -1.], dtype=torch.float64))
    assert report["normal_penalty"] == report["attack_penalty"] == 2.
    group_gradient, _ = score_gradient([0., 2., 1.], registered, "G_group_tail")
    torch.testing.assert_close(group_gradient, torch.tensor([0., 1., -1.], dtype=torch.float64))


def test_tied_extrema_share_gradient_and_complete_collapse_is_penalized():
    registered = groups([[0], [1]], [[0], [1]], [2, 3])
    actual, report = score_gradient([0., 0., 0., 0.], registered, "R_reference_envelope")
    torch.testing.assert_close(actual, torch.tensor([.5, .5, -.5, -.5], dtype=torch.float64))
    assert report["penalty"] == report["group_penalty"] == 1.
    assert report["all_scores_identical"]
    assert not report["all_reference_attack_margins_satisfied"]


def test_quantile_tie_uses_fixed_canonical_row_not_interpolated_boundary():
    registered = groups([list(range(100))], [list(range(100))], [100])
    scores = np.zeros(101)
    scores[-1] = -2.
    actual, report = score_gradient(scores, registered, "G_group_tail")
    assert report["references"][0]["boundary"] == 0.
    assert report["references"][0]["boundary_tie_count"] == 100
    # The 99th ascending normal is canonical index 98 at alpha 1%.
    assert actual[98] == 1. and actual[100] == -1.
    assert int(actual.ne(0).sum()) == 2


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
    registered = groups([list(range(5)), list(range(5, 10))],
                        [[0, 1], [2, 3, 4], [5, 6], [7, 8, 9]], [10, 11, 12])
    return model, x, registered


@pytest.mark.parametrize("arm", ARMS)
def test_replay_matches_direct_autograd_and_preserves_existing_bce(arm):
    model, x, registered = example()
    direct = deepcopy(model)
    for network in (model, direct):
        network.eval()
        torch.nn.functional.binary_cross_entropy_with_logits(
            network(torch.from_numpy(x[:4]), None), torch.ones(4, 3, 1)).backward()
    scored = score_full(model, x, None, device="cpu", batch_size=4)
    replay = backward_objective(model, x, None, registered, arm=arm,
                                device="cpu", batch_size=4, score_pass=scored)
    scores = torch.cat([stable_scores(direct(torch.from_numpy(x[i:i+4]), None))
                        for i in range(0, len(x), 4)])
    expected = objective_terms(scores, registered, arm)
    expected["penalty"].backward()
    for actual_parameter, expected_parameter in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=2e-6, atol=2e-7)
    assert replay["penalty"] == float(expected["penalty"].detach())
    assert replay["cost"]["discovery_forward_rows_this_call"] == 0
    if arm == "A_BCE":
        assert replay["cost"]["replay_backward_rows"] == 0
    else:
        assert 0 < replay["cost"]["replay_backward_rows"] <= len(x)


def test_replay_preserves_dropout_rng_and_training_mode():
    model, x, registered = example()
    model.train()
    before = torch.get_rng_state().clone()
    report = backward_objective(model, x, None, registered, arm="R_reference_envelope",
                                device="cpu", batch_size=4)
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert report["cost"]["discovery_forward_rows_this_call"] == len(x)


def test_stale_parameter_state_rejected_before_gradient_replay():
    model, x, registered = example()
    scored = score_full(model, x, None, device="cpu", batch_size=4)
    with torch.no_grad():
        model.linear.bias.add_(.01)
    with pytest.raises(RuntimeError, match="Stale"):
        backward_objective(model, x, None, registered, arm="R_reference_envelope",
                           device="cpu", batch_size=4, score_pass=scored)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_diagnostics_only_do_not_replay_or_backpropagate():
    model, x, registered = example()
    report = backward_objective(model, x, None, registered, arm="R_reference_envelope",
                                device="cpu", batch_size=4, backward=False)
    assert report["cost"]["replay_forward_rows"] == 0
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("invalid", ["overlap", "empty", "duplicate", "bad_index", "nonfinite", "bad_arm"])
def test_invalid_memberships_and_scores_are_rejected(invalid):
    registered = groups([[0]], [[1]], [2])
    scores, arm = [0., 2., 10.], "R_reference_envelope"
    if invalid == "overlap":
        registered["targets"][0]["indices"] = [2]
    elif invalid == "empty":
        registered["references"] = []
    elif invalid == "duplicate":
        registered["targets"][0]["indices"] = [1, 1]
    elif invalid == "bad_index":
        registered["attacks"] = [5]
    elif invalid == "nonfinite":
        scores[0] = float("nan")
    else:
        arm = "unknown"
    with pytest.raises(ValueError):
        score_gradient(scores, registered, arm)


def partitioned_gap_case():
    registered = groups([list(range(200)), list(range(200, 400))],
                        [list(range(i, i + 100)) for i in range(0, 400, 100)], [400, 401])
    scores = torch.tensor([0.] * 200 + [-1.] * 200 + [0., .5], dtype=torch.float64)
    scores[98], scores[99] = 2., 4.
    return scores, registered


def test_six_parts_and_bridge_have_exact_additive_values_and_gradients():
    scores, registered = partitioned_gap_case()
    names = dict(zip(("A", "H", "D", "G", "R", "S"), ARMS))
    values, gradients = {}, {}
    for name, arm in names.items():
        gradient, diagnostic = score_gradient(scores, registered, arm)
        gradients[name] = gradient
        values[name] = diagnostic["penalty"]
    assert values == {"A": 0., "H": 1., "D": 3., "G": 3., "R": 4., "S": 6.}
    torch.testing.assert_close(gradients["R"], gradients["D"] + gradients["H"], rtol=0, atol=0)
    torch.testing.assert_close(gradients["S"], gradients["D"] + gradients["G"], rtol=0, atol=0)
    torch.testing.assert_close(gradients["S"] - gradients["R"], gradients["G"] - gradients["H"], rtol=0, atol=0)
    assert values["S"] - values["R"] == values["G"] - values["H"]


def test_ordering_g_le_r_le_s_for_all_registered_weights():
    generator = np.random.default_rng(245)
    scores, registered = partitioned_gap_case()
    for _ in range(30):
        random = torch.tensor(generator.normal(size=len(scores)), dtype=torch.float64)
        terms = objective_terms(random, registered, "S_alignment_group")
        g, r, s = (float(terms[key]) for key in ("group_penalty", "reference_envelope_penalty", "alignment_group_penalty"))
        h, d = float(terms["reference_attack_penalty"]), float(terms["reference_transfer_penalty"])
        assert g <= r + 1e-14 and r <= s + 1e-14
        assert s - r == pytest.approx(g - h, abs=1e-14)
        assert s == pytest.approx(d + g, abs=1e-14)
        for weight in (.25, 1., 4.):
            assert weight * g <= weight * r + 1e-13
            assert weight * r <= weight * s + 1e-13


def test_whole_auxiliary_weight_is_not_a_normal_component_only_weight():
    scores, registered = partitioned_gap_case()
    result = objective_terms(scores, registered, "R_reference_envelope")
    for weight in (.25, 4.):
        actual = weight * result["penalty"]
        normal_only = weight * result["normal_penalty"] + result["attack_penalty"]
        assert float(actual) != float(normal_only)
        assert float(actual) == weight * (float(result["normal_penalty"]) + float(result["attack_penalty"]))


@pytest.mark.parametrize("weight", [.25, 4.])
def test_weighted_s_replay_scales_the_entire_objective_gradient(weight):
    model, x, registered = example()
    direct = deepcopy(model).eval()
    result = backward_objective(model, x, None, registered, arm="S_alignment_group",
                                device="cpu", batch_size=4, weight=weight)
    scores = torch.cat([stable_scores(direct(torch.from_numpy(x[i:i+4]), None))
                        for i in range(0, len(x), 4)])
    (weight * objective_terms(scores, registered, "S_alignment_group")["penalty"]).backward()
    for actual, expected in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-7)
    assert result["applied_weight"] == weight


def test_alignment_only_zero_does_not_certify_attack_detection():
    registered = groups([[0], [1]], [[0], [1]], [2])
    _, d = score_gradient([0., 0., 0.], registered, "D_reference_alignment")
    _, h = score_gradient([0., 0., 0.], registered, "H_reference_attack")
    assert d["penalty"] == 0. and h["penalty"] == 1.
    assert not d["all_reference_attack_margins_satisfied"]
    assert d["all_scores_identical"]
