"""Fabricated scores/logits only; no model, dataset or optimizer fit."""
from dataclasses import replace

import pytest
import torch

from ids.operating_risk_training.constrained_step import projected_dual_update
from ids.operating_risk_training.evaluation import FrozenRiskManifest
from ids.operating_risk_training.operating_point import calibrate
from ids.operating_risk_training.risk_losses import (
    DegenerateNormalScale,
    aggregate_member_scores,
    build_risk_state,
)
from ids.route_batch.tabm_training import stable_scores


def ids(n, prefix):
    return tuple(f"{prefix}:{i}" for i in range(n))


def fixture(q=None, n=None, p=None, groups=None):
    q = torch.linspace(-2, 2, 199, dtype=torch.float64) if q is None else q
    n = torch.tensor([1.0, 1.5, 2.1, 2.8], dtype=torch.float64) if n is None else n
    p = torch.tensor([2.2, 3.0], dtype=torch.float64) if p is None else p
    groups = ("g1",) * len(n) if groups is None else tuple(groups)
    manifest = FrozenRiskManifest(
        reference_ids=ids(len(q), "q"), normal_ids=ids(len(n), "n"),
        normal_group_ids=groups, expected_groups=tuple(dict.fromkeys(groups)),
        protected_positive_ids=ids(len(p), "p"),
    )
    args = dict(reference_scores=q, normal_scores=n, positive_scores=p,
                reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                positive_ids=manifest.protected_positive_ids)
    return calibrate(q, identities=manifest.reference_ids), manifest, args


def state(q=None, n=None, p=None, groups=None):
    point, manifest, args = fixture(q, n, p, groups)
    return build_risk_state(point, manifest, **args)


def test_all_32_members_use_existing_deployment_aggregation_and_receive_gradient():
    logits = torch.linspace(-6, 10, 32, dtype=torch.float32).repeat(3, 1).unsqueeze(-1)
    logits.requires_grad_()
    scores = aggregate_member_scores(logits)
    assert scores.dtype == torch.float64
    assert torch.equal(scores, stable_scores(logits))
    direct = torch.logit(torch.sigmoid(logits.double()).mean(dim=1).squeeze(-1))
    torch.testing.assert_close(scores, direct)
    assert not torch.allclose(scores, logits.double().mean(dim=1).squeeze(-1))
    scores.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad > 0).all()


def test_saturated_logits_are_not_probability_clipped():
    logits = torch.tensor([-1000.0, 1000.0], dtype=torch.float32).reshape(2, 1, 1).repeat(1, 32, 1)
    logits.requires_grad_()
    result = aggregate_member_scores(logits)
    torch.testing.assert_close(result, torch.tensor([-1000.0, 1000.0], dtype=torch.float64))
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.full_like(logits, 1 / 32))


@pytest.mark.parametrize("logits", [torch.zeros(2, 31, 1), torch.zeros(2, 32),
                                   torch.zeros(2, 32, 1, dtype=torch.float64),
                                   torch.zeros(0, 32, 1), torch.full((2, 32, 1), torch.inf)])
def test_aggregation_rejects_another_member_or_precision_contract(logits):
    with pytest.raises(ValueError):
        aggregate_member_scores(logits)


def test_normal_proxy_bounds_true_group_risk_with_full_denominators():
    point, manifest, args = fixture(groups=("g1", "g1", "g1", "g2"))
    result = build_risk_state(point, manifest, **args)
    assert result.group_counts == (3, 1)
    assert result.normal_union_size == 203
    torch.testing.assert_close(result.scale, torch.cat(
        [args["reference_scores"], args["normal_scores"]]
    ).std(correction=0))
    for i, group in enumerate(result.group_ids):
        mask = torch.tensor([g == group for g in manifest.normal_group_ids])
        risk = (args["normal_scores"][mask] > point.threshold).double().mean()
        expected = torch.relu(1 + result.normalized_normal_margin[mask]).mean()
        torch.testing.assert_close(result.normal_group_proxy[i], expected)
        assert result.normal_group_proxy[i] >= risk
    assert result.normal_objective == result.normal_group_proxy.max()


def test_equal_scores_of_distinct_identities_are_not_deduplicated():
    n = torch.tensor([1.0, 1.0, 1.0, 2.8], dtype=torch.float64)
    result = state(n=n, groups=("a", "a", "a", "b"))
    assert result.group_counts == (3, 1)
    assert result.normal_union_size == 203


@pytest.mark.parametrize("role", ["reference", "normal", "positive"])
def test_no_missing_identity_or_minibatch_can_change_scale_or_denominators(role):
    point, manifest, args = fixture()
    args[f"{role}_scores"] = args[f"{role}_scores"][:-1]
    args[f"{role}_ids"] = args[f"{role}_ids"][:-1]
    with pytest.raises(ValueError, match="identities/order"):
        build_risk_state(point, manifest, **args)


def test_duplicate_identity_cannot_be_added_as_a_scale_weight():
    _, manifest, _ = fixture()
    with pytest.raises(ValueError, match="unique"):
        replace(manifest, normal_ids=(manifest.normal_ids[0],) * 4)
    with pytest.raises(ValueError, match="disjoint"):
        replace(manifest, normal_ids=manifest.reference_ids[:4])


def test_wrong_precision_and_omitted_group_are_rejected():
    point, manifest, args = fixture(groups=("a", "a", "b", "b"))
    args["positive_scores"] = args["positive_scores"].float()
    with pytest.raises(ValueError, match="FP64"):
        build_risk_state(point, manifest, **args)
    with pytest.raises(ValueError, match="unknown or empty"):
        replace(manifest, expected_groups=("a",))


def test_positive_affine_invariance_of_proxy_and_continuous_margin():
    point, manifest, args = fixture(groups=("a", "a", "b", "b"))
    before = build_risk_state(point, manifest, **args)
    changed = {k: (4 * v - 11 if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    after = build_risk_state(calibrate(changed["reference_scores"], identities=manifest.reference_ids),
                             manifest, **changed)
    for field in ("normal_group_proxy", "normal_objective", "continuous_positive_margin"):
        torch.testing.assert_close(getattr(before, field), getattr(after, field), rtol=1e-13, atol=1e-13)


def test_affine_invariance_holds_in_the_graph_including_scale_derivative():
    _, manifest, args = fixture()
    factor = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    shift = torch.tensor(-7.0, dtype=torch.float64, requires_grad=True)
    changed = {k: (factor * v + shift if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    result = build_risk_state(calibrate(changed["reference_scores"], identities=manifest.reference_ids),
                              manifest, **changed)
    result.normal_objective.backward()
    assert abs(factor.grad.item()) < 1e-12
    assert abs(shift.grad.item()) < 1e-12


def test_strict_positive_true_residual_and_signed_dual_updates():
    point, _, args = fixture()
    threshold = point.threshold.detach()
    above = torch.nextafter(threshold, torch.tensor(torch.inf, dtype=torch.float64))
    p = torch.stack((threshold - 0.3, threshold, above, threshold + 0.5)).requires_grad_()
    result = state(p=p)
    assert result.strict_positive_threshold == above
    assert (result.positive_true_violation <= 0).tolist() == [False, False, True, True]
    assert result.positive_true_violation[1] == 1
    assert result.positive_true_violation[2] < 0
    assert result.positive_true_violation[3] < 0
    assert result.positive_proxy_violation.requires_grad
    assert not result.positive_true_violation.requires_grad
    updated = projected_dual_update(torch.ones(4, dtype=torch.float64),
                                    result.positive_true_violation, step_size=1.0)
    assert updated[0] > 1
    assert updated[3] < 1
    assert not result.strict_positive_threshold.requires_grad
    zero_updated = projected_dual_update(torch.zeros(4, dtype=torch.float64),
                                         result.positive_true_violation, step_size=0.125)
    assert zero_updated[1] == 0.125
    assert zero_updated[2] == 0
    assert result.positive_proxy_violation[1] == 0


def test_threshold_ties_use_the_reference_graph_even_if_point_was_detached():
    q = torch.cat((torch.zeros(196, dtype=torch.float64), torch.ones(3, dtype=torch.float64)))
    point, manifest, args = fixture(q=q, p=torch.tensor([2.0], dtype=torch.float64))
    args["reference_scores"] = q.clone().requires_grad_()
    result = build_risk_state(point, manifest, **args)
    assert not point.threshold.requires_grad
    threshold_from_proxy = args["positive_scores"][0] + result.positive_proxy_violation[0] * result.scale
    threshold_from_proxy.backward()
    expected = torch.cat((torch.zeros(196, dtype=torch.float64), torch.full((3,), 1 / 3, dtype=torch.float64)))
    torch.testing.assert_close(args["reference_scores"].grad, expected)


def test_tied_worst_group_backward_is_symmetric():
    n = torch.tensor([2.5, 2.5], dtype=torch.float64, requires_grad=True)
    result = state(n=n, groups=("a", "b"))
    result.normal_objective.backward()
    torch.testing.assert_close(n.grad[0], n.grad[1])


@pytest.mark.parametrize("positive", [2.0, 3.0])
def test_constant_normals_have_no_proxy_scale_even_when_true_tp_is_feasible(positive):
    q = torch.full((99,), 2.0, dtype=torch.float64)
    n = torch.full((4,), 2.0, dtype=torch.float64)
    with pytest.raises(DegenerateNormalScale, match="evaluate true risk and TP"):
        state(q=q, n=n, p=torch.tensor([positive], dtype=torch.float64))
    point = calibrate(q, identities=ids(99, "q"))
    assert not point.evaluate(n, identities=ids(4, "n")).alarms.any()
    assert point.evaluate(torch.tensor([positive], dtype=torch.float64),
                          identities=ids(1, "p")).alarms.item() == (positive > 2)


def test_normalization_cannot_silently_remove_a_safe_positive_dual_decrease():
    q = torch.zeros(99, dtype=torch.float64)
    n = torch.tensor([-100.0, 0, 0, 0], dtype=torch.float64)
    smallest = torch.nextafter(torch.tensor(0.0, dtype=torch.float64), torch.tensor(torch.inf))
    p = torch.stack((smallest, torch.tensor(100.0, dtype=torch.float64)))
    with pytest.raises(ValueError, match="lost a positive strict margin"):
        state(q=q, n=n, p=p)


def test_a_single_fn_is_not_cancelled_by_safe_positives_in_the_dual_vector():
    point, _, _ = fixture()
    p = torch.stack((point.threshold, point.threshold + 100))
    result = state(p=p)
    assert result.positive_true_violation.mean() < 0
    updated = projected_dual_update(torch.ones(2, dtype=torch.float64),
                                    result.positive_true_violation, step_size=0.5)
    assert updated[0] == 1.5
    assert updated[1] == 0


def test_exact_zero_threshold_tie_has_unit_dual_signal_without_ulp_division():
    q = torch.zeros(99, dtype=torch.float64)
    n = torch.tensor([-100.0, 0, 0, 0], dtype=torch.float64)
    p = torch.tensor([0.0, 100.0], dtype=torch.float64)
    result = state(q=q, n=n, p=p)
    assert result.positive_true_violation[0] == 1
    assert result.positive_proxy_violation[0] == 0


def test_monotone_warp_can_lower_proxy_without_improving_true_decisions():
    q = torch.linspace(0, 1, 99, dtype=torch.float64)
    n = torch.tensor([0.0, 0.5, 0.9, 1.01, 1.2], dtype=torch.float64)
    p = torch.tensor([2.0], dtype=torch.float64)
    before = state(q=q, n=n, p=p)
    after = state(q=q.log1p(), n=n.log1p(), p=p.log1p())
    assert after.normal_objective < before.normal_objective
    assert (n > q.max()).tolist() == (n.log1p() > q.log1p().max()).tolist()
    assert (before.positive_true_violation <= 0).tolist() == (after.positive_true_violation <= 0).tolist()
    assert before.normal_objective.item() == pytest.approx(0.6725370824239728)
    assert after.normal_objective.item() == pytest.approx(0.6475162409390531)


def test_changing_only_positive_scores_cannot_improve_the_normal_proxy():
    q = torch.linspace(0, 1, 99, dtype=torch.float64)
    n = torch.tensor([0.8, 0.85, 0.9, 1.01], dtype=torch.float64)
    before = state(q=q, n=n, p=torch.tensor([2.0], dtype=torch.float64))
    after = state(q=q, n=n, p=torch.tensor([1.0001], dtype=torch.float64))
    assert torch.equal(before.normal_objective, after.normal_objective)
    assert torch.equal(before.scale, after.scale)
    assert after.continuous_positive_margin < before.continuous_positive_margin


def test_normal_objective_has_no_score_graph_connection_to_positive_scores():
    point, manifest, args = fixture()
    for key in ("reference_scores", "normal_scores", "positive_scores"):
        args[key] = args[key].clone().requires_grad_()
    result = build_risk_state(point, manifest, **args)
    positive_grad = torch.autograd.grad(result.normal_objective, args["positive_scores"], allow_unused=True)[0]
    assert positive_grad is None


def test_complete_score_gradient_is_finite_and_reaches_all_roles():
    point, manifest, args = fixture()
    for key in ("reference_scores", "normal_scores", "positive_scores"):
        args[key] = args[key].clone().requires_grad_()
    result = build_risk_state(point, manifest, **args)
    loss = result.normal_objective + result.positive_proxy_violation.sum()
    loss.backward()
    for key in ("reference_scores", "normal_scores", "positive_scores"):
        assert torch.isfinite(args[key].grad).all()
        assert torch.count_nonzero(args[key].grad) > 0
