"""Fabricated scores only: exact decisions and a declared gradient convention."""
from fractions import Fraction

import numpy as np
import pytest
import torch

from ids.operating_risk_training.operating_point import calibrate, calibration_rank
from ids.route_batch.metrics import decisions as legacy_decisions
from ids.route_batch.metrics import threshold as legacy_threshold


def ids(n, prefix="reference"):
    return tuple(f"{prefix}:{i}" for i in range(n))


def reference(n=199, *, requires_grad=False):
    return torch.arange(n, dtype=torch.float64, requires_grad=requires_grad)


@pytest.mark.parametrize("n", [99, 100, 198, 199, 200, 299, 10000, 2**53 + 7, np.int64(199)])
def test_rank_uses_exact_integer_ceiling(n):
    order = Fraction((int(n) + 1) * 99, 100)
    expected = (order.numerator + order.denominator - 1) // order.denominator
    assert calibration_rank(n) == expected


@pytest.mark.parametrize("n", [0, 1, 98, -1, True, 199.0, "199"])
def test_no_finite_rank_is_not_a_training_reject_all_endpoint(n):
    with pytest.raises(ValueError):
        calibration_rank(n)


@pytest.mark.parametrize("n", [99, 100, 199, 1001])
def test_forward_and_strict_decisions_match_legacy_rule(n):
    values = (reference(n) * 17).remainder(29) / 8
    point = calibrate(values, identities=ids(n))
    old = legacy_threshold(values.numpy())
    assert point.rank == old["rank"]
    assert point.threshold.item() == old["threshold"]
    assert point.n_normal == n
    assert point.threshold_tie_count == int((values == point.threshold).sum())
    below = torch.nextafter(point.threshold, torch.tensor(-torch.inf, dtype=torch.float64))
    above = torch.nextafter(point.threshold, torch.tensor(torch.inf, dtype=torch.float64))
    query = torch.stack([below, point.threshold, above])
    result = point.evaluate(query, identities=ids(3, "query"))
    assert result.alarms.tolist() == [False, False, True]
    assert result.identities == ids(3, "query")
    np.testing.assert_array_equal(result.alarms.numpy(), legacy_decisions(query.numpy(), old))
    assert torch.equal(result.alarms, result.signed_margins > 0)


@pytest.mark.parametrize("values", [
    [0.] * 99,
    torch.zeros(99, dtype=torch.float32),
    torch.zeros(99, dtype=torch.int64),
    torch.zeros((99, 1), dtype=torch.float64),
    torch.empty(0, dtype=torch.float64),
    torch.tensor([float("nan")] * 99, dtype=torch.float64),
    torch.tensor([float("inf")] * 99, dtype=torch.float64),
    torch.tensor([-float("inf")] * 99, dtype=torch.float64),
])
def test_reference_values_must_be_finite_fp64_vectors(values):
    with pytest.raises(ValueError, match="FP64"):
        calibrate(values, identities=ids(99))


@pytest.mark.parametrize("identities", [
    ids(98), ["same"] * 99, ids(98) + ("",), ids(98) + ("  ",),
    ids(98) + (None,), tuple(range(99)), "x" * 99,
])
def test_reference_identity_alignment_and_uniqueness(identities):
    with pytest.raises(ValueError, match="identities"):
        calibrate(reference(99), identities=identities)


def test_query_identity_checks_and_nonempty_finite_values():
    point = calibrate(reference(), identities=ids(199))
    values = torch.tensor([200., 201.], dtype=torch.float64)
    for query_ids in [ids(1, "query"), ["p", "p"], ["p", ""], ["p", None]]:
        with pytest.raises(ValueError, match="identities"):
            point.evaluate(values, identities=query_ids)
    with pytest.raises(ValueError, match="disjoint"):
        point.evaluate(values, identities=["p", "reference:0"])
    for query in [values.float(), values[:, None], values[:0], values * torch.inf]:
        with pytest.raises(ValueError, match="FP64"):
            point.evaluate(query, identities=ids(query.numel(), "query"))


def test_finite_inputs_cannot_silently_overflow_signed_margins():
    point = calibrate(torch.full((99,), -1e308, dtype=torch.float64), identities=ids(99))
    with pytest.raises(ValueError, match="margins"):
        point.evaluate(torch.tensor([1e308], dtype=torch.float64), identities=["positive"])


def test_unique_order_statistic_and_positive_margins_pass_finite_differences():
    q = reference(requires_grad=True)
    positive = torch.tensor([199., 205.], dtype=torch.float64, requires_grad=True)

    def margin(q_values, p_values):
        point = calibrate(q_values, identities=ids(199))
        return point.evaluate(p_values, identities=ids(2, "positive")).signed_margins

    assert torch.autograd.gradcheck(margin, (q, positive), fast_mode=True, eps=1e-6)
    grad_q, grad_p = torch.autograd.grad(margin(q, positive).sum(), (q, positive))
    expected = torch.zeros_like(q)
    expected[197] = -2  # k=198; both positive margins share this current threshold.
    torch.testing.assert_close(grad_q, expected, rtol=0, atol=0)
    torch.testing.assert_close(grad_p, torch.ones_like(positive), rtol=0, atol=0)
    assert (grad_q.sum() + grad_p.sum()).item() == 0  # common-shift derivative


def test_tied_threshold_backward_is_equal_share_and_permutation_equivariant():
    q = torch.cat([torch.zeros(197, dtype=torch.float64), torch.ones(3, dtype=torch.float64)])
    q.requires_grad_()
    point = calibrate(q, identities=ids(200))
    assert point.rank == 199 and point.threshold.item() == 1. and point.threshold_tie_count == 3
    gradient = torch.autograd.grad(point.threshold, q)[0]
    expected = torch.cat([torch.zeros(197, dtype=torch.float64), torch.full((3,), 1/3, dtype=torch.float64)])
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    reversed_q = q.detach().flip(0).requires_grad_()
    reversed_point = calibrate(reversed_q, identities=tuple(reversed(ids(200))))
    reversed_grad = torch.autograd.grad(reversed_point.threshold, reversed_q)[0]
    torch.testing.assert_close(reversed_grad.flip(0), gradient, rtol=0, atol=0)


def test_all_reference_ties_share_gradient_without_jitter_or_lost_tp_semantics():
    q = torch.full((99,), 2., dtype=torch.float64, requires_grad=True)
    positive = torch.tensor([2., 3.], dtype=torch.float64, requires_grad=True)
    point = calibrate(q, identities=ids(99))
    result = point.evaluate(positive, identities=ids(2, "positive"))
    assert result.alarms.tolist() == [False, True]
    grad_q, grad_p = torch.autograd.grad(result.signed_margins.sum(), (q, positive))
    torch.testing.assert_close(grad_q, torch.full_like(q, -2/99), rtol=0, atol=0)
    torch.testing.assert_close(grad_p, torch.ones_like(positive), rtol=0, atol=0)


@pytest.mark.parametrize("transform", [lambda x: x * 7 + 19, lambda x: x**3, torch.exp])
def test_common_strict_monotone_transform_preserves_decisions(transform):
    q = torch.linspace(-3., 3., 199, dtype=torch.float64)
    q[-1] = q[-2]  # exact threshold ties must also preserve strict decisions
    point = calibrate(q, identities=ids(199))
    query = torch.stack([q[0] - 1, point.threshold, point.threshold + .1, q[-1] + 1])
    original = point.evaluate(query, identities=ids(4, "query"))
    changed = calibrate(transform(q), identities=ids(199))
    result = changed.evaluate(transform(query), identities=ids(4, "query"))
    assert changed.threshold.item() == transform(point.threshold).item()
    assert torch.equal(result.alarms, original.alarms)


def test_helper_does_not_consume_rng_or_change_input_values():
    q = reference(requires_grad=True)
    before = q.detach().clone()
    rng = torch.get_rng_state().clone()
    point = calibrate(q, identities=ids(199))
    point.threshold.backward()
    assert torch.equal(q.detach(), before)
    assert torch.equal(torch.get_rng_state(), rng)
