"""Finite-pool event oracles, including exhaustive attainable-threshold checks."""
import itertools
import json

import numpy as np
import pytest
import torch

from ids.risk_distillation import RiskConstraint, build_risk_reference
from ids.threshold_feasibility import solve_threshold_interval


def reference(anchor=(-1., 1.), labels=(0, 1), constraints=None, **kwargs):
    return build_risk_reference(torch.tensor(anchor, dtype=torch.float64), torch.tensor(labels),
        constraints or [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", 0.)],
        sample_ids=[f"c{i}" for i in range(len(anchor))], **kwargs)


def solve(r, c, ref=None, labels=None, alpha=0., **kwargs):
    ref = reference() if ref is None else ref
    return solve_threshold_interval(r, [0] * len(r) if labels is None else labels, c, ref,
        reference_ids=kwargs.pop("reference_ids", [f"r{i}" for i in range(len(r))]),
        constraint_ids=kwargs.pop("constraint_ids", ref.sample_ids), max_fpr=alpha, **kwargs)


def finite_feasible(t, r, c, ref, alpha):
    if np.count_nonzero(r >= t) > int(np.floor(alpha * len(r))):
        return False
    original = ref.anchor_margins.numpy() >= 0
    for spec, member in zip(ref.constraints, ref.members):
        mask = member.numpy()
        count = np.count_nonzero(c[mask] >= t) if spec.kind == "normal_fpr" else np.count_nonzero((original & (c < t))[mask])
        if count > int(np.floor(spec.limit * mask.sum())):
            return False
    return True


def test_strict_normal_and_inclusive_retention_allow_equal_interval_endpoint():
    upper = np.nextafter(np.float64(1.), np.inf)
    result = solve([1.], [0., upper])
    assert result["feasible"] and result["threshold"] == upper
    assert result["lower_bound"] == result["upper_bound"]
    assert all(row["passed"] for row in result["metrics_at_lower"])
    assert result["selection_uses_constraint_labels"] and not result["population_guarantee"]


def test_equal_normal_and_tp_score_produces_empty_interval_without_epsilon():
    result = solve([1.], [0., 1.])
    assert not result["feasible"] and result["threshold"] is None
    assert result["lower_bound"] > result["upper_bound"] == 1.
    assert [row["name"] for row in result["blocking_constraints"]] == ["attack"]
    assert [row["count"] for row in result["metrics_at_lower"] if row["kind"] == "attack_loss"] == [1]


def test_normal_ties_are_rejected_together_and_need_not_fill_budget():
    result = solve([2., 2., 1.], [0., 3.], alpha=.34)
    assert result["threshold"] == np.nextafter(np.float64(2.), np.inf)
    r = result["metrics_at_lower"][0]
    assert r["allowed"] == 1 and r["count"] == 0 and r["passed"]


def test_tp_ties_use_strict_loss_and_inclusive_upper_boundary():
    ref = reference((-1., 1., 1., 1.), (0, 1, 1, 1),
        [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", 1 / 3)])
    result = solve([-1.], [0., 1., 1., 2.], ref)
    assert result["retention_bounds"][0]["allowed"] == 1
    assert result["upper_bound"] == 1.
    c = np.array([0., 1., 1., 2.])
    assert finite_feasible(1., np.array([-1.]), c, ref, 0.)
    assert not finite_feasible(np.nextafter(1., np.inf), np.array([-1.]), c, ref, 0.)


def test_retention_denominator_includes_old_false_negatives():
    ref = reference((-1., 1., -1., -1., -1., -1.), (0, 1, 1, 1, 1, 1),
        [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", .2)])
    result = solve([0.], [-1., -2., -3., -3., -3., -3.], ref)
    assert result["feasible"] and result["upper_bound"] is None
    retention = result["retention_bounds"][0]
    assert retention["support"] == 5 and retention["allowed"] == retention["baseline_tp_count"] == 1
    assert result["metrics_at_lower"][-1]["count"] == 1


def test_zero_anchor_tp_has_no_retention_upper_bound():
    ref = reference((-1., -1.), (0, 1))
    result = solve([10.], [0., -100.], ref)
    assert result["feasible"] and result["upper_bound"] is None
    assert result["retention_bounds"][0]["baseline_tp_count"] == 0
    assert result["metrics_at_lower"][-1]["count"] == 0
    json.dumps(result, allow_nan=False)


def test_normal_stratum_and_pooled_limits_remain_separate():
    specs = [RiskConstraint("normal", "normal_fpr", 1.), RiskConstraint("left", "normal_fpr", 0., group=0),
             RiskConstraint("right", "normal_fpr", 1., group=1), RiskConstraint("attack", "attack_loss", 0.)]
    ref = reference((-1., -1., 1.), (0, 0, 1), specs, normal_strata=torch.tensor([0, 1, 1]))
    result = solve([0.], [3., 4., 5.], ref)
    assert result["threshold"] == np.nextafter(np.float64(3.), np.inf)
    active = [row for row in result["normal_bounds"] if row["active_lower"]]
    assert [(row["population"], row["name"]) for row in active] == [("constraint", "left")]
    assert [row for row in result["normal_bounds"] if row["name"] == "right"][0]["lower_bound"] is None


def test_reference_and_constraint_same_names_are_unambiguous_populations():
    result = solve([3.], [0., 5.])
    assert [(row["population"], row["name"]) for row in result["normal_bounds"]] == [("reference", "normal"), ("constraint", "normal")]
    assert result["normal_bounds"][0]["active_lower"]


def test_reference_attacks_do_not_enter_normal_order_statistic():
    result = solve([-1., 900.], [-2., 0.], labels=[0, 1])
    assert result["threshold"] == np.nextafter(np.float64(-1.), np.inf)
    assert result["normal_bounds"][0]["support"] == 1


def test_float32_inputs_are_cast_before_nextafter_comparison():
    raw = np.float32(-13.815509796142578)
    result = solve(np.array([raw], dtype=np.float32), np.array([-20., 0.], dtype=np.float32))
    assert result["threshold"] > np.float64(raw)
    assert result["metrics_at_lower"][0]["count"] == 0


@pytest.mark.parametrize("transform", [(2., 3.), (.5, -2.)])
@pytest.mark.parametrize("c", [[0., 2.], [2., 0.]])
def test_positive_affine_score_transforms_preserve_feasibility_and_events(transform, c):
    scale, shift = transform
    r = np.array([1., -1.])
    c = np.asarray(c)
    original = solve(r, c)
    mapped = solve(scale * r + shift, scale * c + shift)
    assert mapped["feasible"] == original["feasible"]
    assert [(x["count"], x["passed"]) for x in mapped["metrics_at_lower"]] == [(x["count"], x["passed"]) for x in original["metrics_at_lower"]]
    # nextafter does not commute bit-exactly with arbitrary floating offsets.


def test_exhaustive_small_populations_match_all_attainable_event_thresholds():
    grid = [-1., 0., 1.]
    for limit in (0., .5):
        ref = reference((-1., 1., 1.), (0, 1, 1),
            [RiskConstraint("normal", "normal_fpr", 0.), RiskConstraint("attack", "attack_loss", limit),
             RiskConstraint("family", "attack_loss", 0., group=1)], attack_families=torch.tensor([9, 0, 1]))
        for rr in itertools.product(grid, repeat=2):
            r = np.array(rr)
            for cc in itertools.product(grid, repeat=3):
                c = np.array(cc)
                values = np.unique(np.concatenate((r, c)))
                thresholds = [values[0] - 1., *values, *np.nextafter(values, np.inf)]
                valid = [t for t in thresholds if finite_feasible(t, r, c, ref, .5)]
                result = solve(r, c, ref, alpha=.5)
                assert result["feasible"] == bool(valid), (rr, cc, limit)
                if valid:
                    assert finite_feasible(result["threshold"], r, c, ref, .5)
                    assert result["threshold"] <= min(valid)
                else:
                    assert result["threshold"] is None and result["blocking_constraints"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, -.01, 1., "0.01"])
def test_invalid_reference_fpr_rejected(bad):
    with pytest.raises(ValueError, match="max_fpr"):
        solve([0.], [-1., 1.], alpha=bad)


@pytest.mark.parametrize("r,c,labels", [
    ([], [-1., 1.], []), ([[0.]], [-1., 1.], [0]), ([0.], [-1.], [0]),
    ([np.nan], [-1., 1.], [0]), ([0.], [-1., np.inf], [0]), ([0.], [-1., 1.], [2]),
    ([0.], [-1., 1.], [1]), ([True], [-1., 1.], [0]), (["0"], [-1., 1.], [0]),
])
def test_invalid_vectors_or_missing_normal_support_rejected(r, c, labels):
    with pytest.raises(ValueError):
        solve(r, c, labels=labels)


@pytest.mark.parametrize("rid,cid", [
    (["r0"], ["c1", "c0"]), (["r0", "r0"], ["c0", "c1"]),
    (["c0"], ["c0", "c1"]), ([" "], ["c0", "c1"]),
    (["r0"], ["c0", "c0"]), ([1], ["c0", "c1"]),
])
def test_ids_must_be_aligned_unique_and_disjoint(rid, cid):
    with pytest.raises(ValueError):
        solve([0.], [-1., 1.], reference_ids=rid, constraint_ids=cid)


@pytest.mark.parametrize("mutation", ["mask", "anchor", "labels"])
def test_frozen_reference_tensor_mutation_is_rejected(mutation):
    ref = reference()
    if mutation == "mask":
        ref.members[0][0] = False
    elif mutation == "anchor":
        ref.anchor_margins[0] = -2.
    else:
        ref.labels[0] = 1
    with pytest.raises(ValueError, match="modified"):
        solve([0.], [-1., 1.], ref)


def test_no_finite_threshold_on_extreme_normal_bound_is_explicit():
    with pytest.raises(ValueError, match="representable finite"):
        solve([np.finfo(np.float64).max], [-1., 1.])


def test_solver_detaches_torch_inputs_without_mutation_or_gradients():
    r = torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    c = torch.tensor([-1., 1.], dtype=torch.float32, requires_grad=True)
    rc, cc = r.detach().clone(), c.detach().clone()
    result = solve(r, c)
    assert result["feasible"]
    assert r.grad is None and c.grad is None
    assert torch.equal(r.detach(), rc) and torch.equal(c.detach(), cc)


def test_wrong_reference_type_rejected():
    with pytest.raises(TypeError, match="RiskReference"):
        solve_threshold_interval([0.], [0], [-1., 1.], object(), reference_ids=["r0"], constraint_ids=["c0", "c1"])
