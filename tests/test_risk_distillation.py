"""Synthetic event-count checks for anchored one-sided risk majorizers.

These tests do not train a model or load research data.  Independent Boolean
event counts are the oracle for the bounds, rather than a second tensor loss.
"""

import copy
import itertools
import math

import pytest

torch = pytest.importorskip("torch")

from ids.risk_distillation import (  # noqa: E402 (optional torch)
    ProjectedDual,
    RiskConstraint,
    build_risk_reference,
    constraint_risks,
    risk_lagrangian,
)


def _ids(n):
    return [f"synthetic-{i}" for i in range(n)]


def _constraints():
    return [RiskConstraint("normal", "normal_fpr", .5),
            RiskConstraint("lost", "attack_loss", .25)]


def _reference(anchor, labels, constraints=None, **kwargs):
    anchor = torch.as_tensor(anchor, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.float64)
    ids = _ids(len(anchor))
    ref = build_risk_reference(
        anchor, labels, constraints or _constraints(), sample_ids=ids, **kwargs,
    )
    return ref, ids


def _report(anchor, labels, student, constraints=None, **kwargs):
    ref, ids = _reference(anchor, labels, constraints, **kwargs)
    margins = torch.as_tensor(student, dtype=torch.float64)
    return constraint_risks(margins, ref, sample_ids=ids)


def _actual_events(anchor, labels, student):
    normal = [i for i, y in enumerate(labels) if y == 0]
    attack = [i for i, y in enumerate(labels) if y == 1]
    fpr = sum(student[i] >= 0 for i in normal) / len(normal)
    lost = sum(anchor[i] >= 0 and student[i] < 0 for i in attack) / len(attack)
    return fpr, lost


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
@pytest.mark.parametrize("floor", [.01, .1, 1.])
def test_bounds_cover_independent_event_counts_on_signed_grid(majorizer, floor):
    grid = list(itertools.product([-2., -.01, 0., .01, 2.],
                                  [-3., -.05, -1e-9, 0., 1e-9, .05, 3.]))
    anchor = [a for a, _ in grid] * 2
    student = [m for _, m in grid] * 2
    labels = [0] * len(grid) + [1] * len(grid)
    report = _report(anchor, labels, student, majorizer=majorizer, margin_floor=floor)
    actual = _actual_events(anchor, labels, student)
    torch.testing.assert_close(report.actual_values, torch.tensor(actual, dtype=torch.float64))
    assert bool((report.values + 1e-12 >= report.actual_values).all())
    assert bool((report.values >= 0).all())
    torch.testing.assert_close(report.violations,
                               report.values - torch.tensor([.5, .25], dtype=torch.float64))
    assert tuple(report.names) == ("normal", "lost")


@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
def test_pointwise_bound_including_ties_and_very_small_anchor(label, majorizer):
    kind = "normal_fpr" if label == 0 else "attack_loss"
    constraints = [RiskConstraint("one", kind, .1)]
    for anchor, student in itertools.product([-1., -1e-12, 0., 1e-12, 1.],
                                              [-1., -1e-12, 0., 1e-12, 1.]):
        report = _report([anchor], [label], [student], constraints,
                         majorizer=majorizer, margin_floor=.05)
        event = student >= 0 if label == 0 else anchor >= 0 and student < 0
        assert report.actual_values.item() == float(event)
        assert report.values.item() + 1e-12 >= float(event)


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
def test_weak_anchor_slack_is_kept_and_not_subtracted(majorizer):
    reference, ids = _reference([-.01, .01], [0, 1], margin_floor=.1, majorizer=majorizer)
    at_anchor = constraint_risks(torch.tensor([-.01, .01], dtype=torch.float64),
                                 reference, sample_ids=ids)
    torch.testing.assert_close(at_anchor.actual_values, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(at_anchor.values, torch.tensor([.9, .9], dtype=torch.float64))
    torch.testing.assert_close(at_anchor.baseline_slack, torch.tensor([.9, .9], dtype=torch.float64))
    crossed = constraint_risks(torch.tensor([.001, -.001], dtype=torch.float64),
                               reference, sample_ids=ids)
    torch.testing.assert_close(crossed.values, torch.tensor([1.01, 1.01], dtype=torch.float64))
    torch.testing.assert_close(crossed.actual_values, torch.ones(2, dtype=torch.float64))
    # Subtracting anchor slack would yield .11 for a true event of one.
    assert bool(((crossed.values - crossed.baseline_slack) < crossed.actual_values).all())


def test_fixed_and_adaptive_anchor_slack_equal_but_neither_dominates_away_from_anchor():
    at = [_report([-1., -.01, .01, 1.], [0, 0, 1, 1], [-1., -.01, .01, 1.],
                  margin_floor=.1, majorizer=mode)
          for mode in ["adaptive", "fixed"]]
    torch.testing.assert_close(at[0].baseline_slack, at[1].baseline_slack)
    constraints = [RiskConstraint("normal", "normal_fpr", .5)]
    for candidate, fixed_is_larger in [(-.05, False), (.05, True)]:
        adaptive = _report([-1.], [0], [candidate], constraints,
                           margin_floor=.1, majorizer="adaptive").values.item()
        fixed = _report([-1.], [0], [candidate], constraints,
                        margin_floor=.1, majorizer="fixed").values.item()
        assert (fixed > adaptive) == fixed_is_larger


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
def test_positive_rescaling_of_margins_and_floor_preserves_risks(majorizer):
    anchor = torch.tensor([-.2, -.01, .03, 2., 0.], dtype=torch.float64)
    student = torch.tensor([.02, -.02, -.01, -.2, 0.], dtype=torch.float64)
    labels = [0, 0, 1, 1, 0]
    base = _report(anchor, labels, student, margin_floor=.1, majorizer=majorizer)
    for scale in [.001, 7., 1000.]:
        scaled = _report(anchor * scale, labels, student * scale,
                         margin_floor=.1 * scale, majorizer=majorizer)
        torch.testing.assert_close(scaled.values, base.values)
        torch.testing.assert_close(scaled.baseline_slack, base.baseline_slack)
        torch.testing.assert_close(scaled.actual_values, base.actual_values)


def test_repair_credit_is_a_lower_bound_not_full_repair_event():
    constraints = [RiskConstraint("normal", "normal_fpr", .5)]
    with_credit = _report([1.], [0], [-.01], constraints, margin_floor=.1)
    without = _report([1.], [0], [-.01], constraints,
                      margin_floor=.1, normal_repair_credit=False)
    assert with_credit.actual_values.item() == 0
    assert with_credit.values.item() == pytest.approx(.9)
    assert without.values.item() == pytest.approx(1.)
    repaired = _report([1.], [0], [-1.], constraints, margin_floor=.1)
    assert repaired.values.item() == 0


def test_complete_repairs_do_not_round_normal_bound_below_actual_fpr():
    report = _report([1.] * 10, [0] * 10, [-.1] * 9 + [.1],
                     [RiskConstraint("normal", "normal_fpr", .1)], margin_floor=.1)
    assert report.actual_values.item() == .1
    assert report.values.item() >= report.actual_values.item()
    torch.testing.assert_close(report.values, report.actual_values, rtol=0, atol=0)


def test_new_attack_true_positives_cannot_cancel_lost_baseline_true_positives():
    constraints = [RiskConstraint("lost", "attack_loss", .1)]
    report = _report([1., -1.], [1, 1], [-1., 1.], constraints, margin_floor=.1)
    assert report.actual_values.item() == .5
    assert report.values.item() >= .5
    assert report.violations.item() > 0


def test_family_denominator_is_all_family_attacks_not_only_baseline_positives():
    constraints = [RiskConstraint("family", "attack_loss", .1, group=7)]
    report = _report([1.] + [-1.] * 99, [1] * 100, [-1.] * 100, constraints,
                     attack_families=torch.full((100,), 7, dtype=torch.int64), margin_floor=.1)
    assert report.actual_values.item() == pytest.approx(.01)
    # A 2x upper majorizer on the one lost sample, averaged over all 100.
    assert report.values.item() == pytest.approx(.02)


def test_baseline_missed_family_is_not_falsely_reported_as_new_lost_tp():
    report = _report([-1., -2.], [1, 1], [-1., -2.],
                     [RiskConstraint("lost", "attack_loss", .01)])
    assert report.actual_values.item() == 0
    assert report.values.item() == 0


def test_group_conditional_masks_do_not_mix_normal_strata_and_attack_families():
    constraints = [RiskConstraint("normal7", "normal_fpr", .1, group=7),
                   RiskConstraint("attack7", "attack_loss", .1, group=7)]
    report = _report([-1., -1., 1., 1.], [0, 0, 1, 1], [1., -1., -1., 1.], constraints,
                     normal_strata=torch.tensor([7, 8, -1, -1]),
                     attack_families=torch.tensor([-1, -1, 7, 8]))
    torch.testing.assert_close(report.actual_values, torch.ones(2, dtype=torch.float64))
    assert bool((report.values >= report.actual_values).all())


def test_pooled_normal_rate_is_row_weighted_not_mean_of_group_rates():
    constraints = [RiskConstraint("pooled", "normal_fpr", .5),
                   RiskConstraint("large", "normal_fpr", .5, group=7),
                   RiskConstraint("small", "normal_fpr", .5, group=8)]
    report = _report([-1.] * 1000, [0] * 1000, [1.] * 999 + [-1.], constraints,
                     normal_strata=torch.tensor([7] * 999 + [8]))
    torch.testing.assert_close(report.actual_values,
                               torch.tensor([.999, 1., 0.], dtype=torch.float64))
    assert report.actual_values[0] != report.actual_values[1:].mean()


def test_reference_detaches_and_snapshots_anchor_and_labels():
    anchor = torch.tensor([-1., 1.], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0., 1.], dtype=torch.float64)
    ids = _ids(2)
    reference = build_risk_reference(anchor, labels, _constraints(), sample_ids=ids)
    with torch.no_grad():
        anchor.fill_(99.)
        labels.fill_(1.)
    student = torch.tensor([-.5, .5], dtype=torch.float64, requires_grad=True)
    report = constraint_risks(student, reference, sample_ids=ids)
    torch.testing.assert_close(report.values, torch.tensor([.5, .5], dtype=torch.float64))
    report.values.sum().backward()
    assert anchor.grad is None
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
def test_student_gradients_finite_at_ties_and_tiny_reference_margins(majorizer):
    reference, ids = _reference([-1e-300, 0., 1e-300, -1., 1.], [0, 0, 1, 1, 0],
                                majorizer=majorizer, margin_floor=.05)
    student = torch.tensor([0., -1e-300, 0., -100., 100.], dtype=torch.float64, requires_grad=True)
    report = constraint_risks(student, reference, sample_ids=ids)
    report.values.sum().backward()
    assert bool(torch.isfinite(report.values).all())
    assert bool(torch.isfinite(student.grad).all())


def test_large_majorizer_is_not_silently_clipped_to_probability_one():
    report = _report([-1., 1.], [0, 1], [100., -100.])
    assert bool((report.values > 1).all())
    torch.testing.assert_close(report.actual_values, torch.ones(2, dtype=torch.float64))


def test_lagrangian_keeps_signed_violations_and_detaches_dual_weights():
    reference, ids = _reference([-1., 1.], [0, 1])
    student = torch.tensor([-2., 2.], dtype=torch.float64, requires_grad=True)
    report = constraint_risks(student, reference, sample_ids=ids)
    weights = torch.tensor([2., 4.], dtype=torch.float64, requires_grad=True)
    loss = risk_lagrangian(report, weights)
    assert loss.item() == pytest.approx(-2.)  # 2*(-.5) + 4*(-.25), not positive-only.
    loss.backward()
    assert weights.grad is None
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


@pytest.mark.parametrize("bad", [[1.], [-1., 1.], [math.nan, 1.], [math.inf, 1.]])
def test_invalid_lagrangian_duals_rejected(bad):
    report = _report([-1., 1.], [0, 1], [-1., 1.])
    with pytest.raises((ValueError, TypeError)):
        risk_lagrangian(report, torch.tensor(bad, dtype=torch.float64))


def test_finite_lagrangian_inputs_with_overflowing_product_are_rejected():
    report = _report([-1.], [0], [1.], [RiskConstraint("normal", "normal_fpr", 0.)])
    assert report.violations.item() == 2.
    with pytest.raises(ValueError):
        risk_lagrangian(report, torch.tensor([1e308], dtype=torch.float64))


def test_projected_dual_ascent_descent_clip_and_no_primal_backward():
    reference, ids = _reference([-1., 1.], [0, 1])
    student = torch.tensor([.2, -.2], dtype=torch.float64, requires_grad=True)
    report = constraint_risks(student, reference, sample_ids=ids)
    dual = ProjectedDual(_constraints(), step_size=.1, maximum=.1)
    assert dual.weights.dtype == torch.float64 and dual.weights.device.type == "cpu"
    torch.testing.assert_close(dual.weights, torch.zeros(2, dtype=torch.float64))
    dual.update(report)
    torch.testing.assert_close(dual.weights, torch.tensor([.07, .095], dtype=torch.float64))
    dual.update(report)
    torch.testing.assert_close(dual.weights, torch.full((2,), .1, dtype=torch.float64))
    feasible = constraint_risks(torch.tensor([-1., 1.], dtype=torch.float64), reference, sample_ids=ids)
    dual.update(feasible)
    torch.testing.assert_close(dual.weights, torch.tensor([.05, .075], dtype=torch.float64))
    for _ in range(4):
        dual.update(feasible)
    torch.testing.assert_close(dual.weights, torch.zeros(2, dtype=torch.float64))
    assert student.grad is None and not dual.weights.requires_grad


def test_dual_checkpoint_roundtrip_and_reject_changed_budget_or_order():
    report = _report([-1., 1.], [0, 1], [.2, -.2])
    dual = ProjectedDual(_constraints(), step_size=.2, maximum=3.)
    dual.update(report)
    saved = copy.deepcopy(dual.state_dict())
    resumed = ProjectedDual(_constraints(), step_size=.2, maximum=3.)
    resumed.load_state_dict(saved)
    torch.testing.assert_close(resumed.weights, dual.weights)
    resumed.update(report)
    dual.update(report)
    torch.testing.assert_close(resumed.weights, dual.weights)
    changed = [RiskConstraint("normal", "normal_fpr", .4), _constraints()[1]]
    for constraints in [changed, list(reversed(_constraints()))]:
        with pytest.raises((ValueError, TypeError)):
            ProjectedDual(constraints, step_size=.2, maximum=3.).load_state_dict(saved)


def test_dual_checkpoint_and_updates_reject_different_bound_reference():
    first = _report([-1., 1.], [0, 1], [.2, -.2])
    second = _report([-2., 2.], [0, 1], [.2, -.2])
    original = ProjectedDual(_constraints())
    original.update(first)
    saved = copy.deepcopy(original.state_dict())
    with pytest.raises((ValueError, TypeError)):
        original.update(second)
    bound = ProjectedDual(_constraints())
    bound.update(second)
    with pytest.raises((ValueError, TypeError)):
        bound.load_state_dict(saved)


@pytest.mark.parametrize("field", ["anchor_margins", "labels", "members:0", "members:1"])
@pytest.mark.parametrize("mutation", ["inplace", "data"])
def test_mutated_frozen_reference_rejected_including_data_bypass(field, mutation):
    reference, ids = _reference([-1., 1.], [0, 1])
    tensor = reference.members[int(field[-1])] if field.startswith("members:") else getattr(reference, field)
    target = tensor.data if mutation == "data" else tensor
    with torch.no_grad():
        if field == "anchor_margins":
            target[0] += .125
        elif field == "labels":
            target[0] = 1
        else:
            target[0] = ~target[0]
    # A content digest is required: .data mutations need not increment the
    # original Tensor version counter and must still invalidate the reference.
    with pytest.raises(ValueError, match="signature|frozen|reference"):
        constraint_risks(torch.tensor([-.5, .5], dtype=torch.float64),
                         reference, sample_ids=ids)


def _byte_payload(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()


def test_same_bytes_anchor_and_label_dtype_views_rejected_by_reference_signature():
    for field, dtype in [("anchor_margins", torch.int64), ("labels", torch.float64)]:
        reference, ids = _reference([-1., 1.], [0, 1])
        tensor = getattr(reference, field)
        original = _byte_payload(tensor)
        tensor.data = tensor.data.view(dtype)
        assert tensor.dtype == dtype
        assert _byte_payload(tensor) == original
        with pytest.raises(ValueError, match="frozen risk reference.*signature"):
            constraint_risks(torch.tensor([-.5, .5], dtype=torch.float64),
                             reference, sample_ids=ids)


def test_same_bytes_boolean_members_reinterpreted_as_indices_rejected_by_signature():
    reference, ids = _reference([-1., 1.], [0, 1])
    mask = reference.members[0]
    assert mask.dtype == torch.bool
    original = _byte_payload(mask)
    mask.data = mask.data.view(torch.uint8)
    assert mask.dtype == torch.uint8
    assert _byte_payload(mask) == original
    # Integer indexing has different semantics despite identical bytes/shape.
    with pytest.raises(ValueError, match="frozen risk reference.*signature"):
        constraint_risks(torch.tensor([-.5, .5], dtype=torch.float64),
                         reference, sample_ids=ids)


def test_same_bytes_reshaped_anchor_rejected_by_signature_before_shape_use():
    reference, ids = _reference([-1., 1.], [0, 1])
    anchor = reference.anchor_margins
    original = _byte_payload(anchor)
    anchor.data = anchor.data.reshape(1, 2)
    assert anchor.shape == (1, 2)
    assert _byte_payload(anchor) == original
    # Require the integrity gate's error, not a later broadcasting/masking error.
    with pytest.raises(ValueError, match="frozen risk reference.*signature"):
        constraint_risks(torch.tensor([-.5, .5], dtype=torch.float64),
                         reference, sample_ids=ids)


@pytest.mark.parametrize("signature", [True, 1, b"0" * 64, "", "a" * 63, "a" * 65,
                                        "z" * 64, " " * 64, ["a" * 64]])
def test_dual_constructor_rejects_malformed_reference_signature(signature):
    with pytest.raises((ValueError, TypeError)):
        ProjectedDual(_constraints(), reference_signature=signature)


@pytest.mark.parametrize("previously_bound", [False, True])
@pytest.mark.parametrize("corruption", ["updated_without_signature", "unbound_nonzero_weights"])
def test_inconsistent_dual_checkpoint_rejected_atomically(previously_bound, corruption):
    dual = ProjectedDual(_constraints())
    if previously_bound:
        dual.update(_report([-1., 1.], [0, 1], [.2, -.2]))
    original = copy.deepcopy(dual.state_dict())
    invalid = copy.deepcopy(original)
    invalid["reference_signature"] = None
    if corruption == "updated_without_signature":
        invalid["updates"] = 1
    else:
        invalid["updates"] = 0
        invalid["weights"] = [.1, .1]
    with pytest.raises((ValueError, TypeError)):
        dual.load_state_dict(invalid)
    assert dual.state_dict() == original


@pytest.mark.parametrize("step_size,maximum", [(0., 1.), (-1., 1.), (math.nan, 1.),
                                              (.1, 0.), (.1, -1.), (.1, math.inf)])
def test_invalid_dual_hyperparameters_rejected(step_size, maximum):
    with pytest.raises((ValueError, TypeError)):
        ProjectedDual(_constraints(), step_size=step_size, maximum=maximum)


@pytest.mark.parametrize("partition", ["train", "validation", "calibration", "test", "inner_audit"])
def test_reference_and_risk_call_reject_non_constraint_training_role(partition):
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], partition=partition)
    reference, ids = _reference([-1., 1.], [0, 1])
    with pytest.raises((ValueError, TypeError)):
        constraint_risks(torch.tensor([-1., 1.]), reference, sample_ids=ids, partition=partition)


@pytest.mark.parametrize("ids", [["a", "a"], ["", "b"], [True, "b"], [1, "b"], "ab", ["a"]])
def test_invalid_reference_identity_rejected(ids):
    with pytest.raises((ValueError, TypeError)):
        build_risk_reference(torch.tensor([-1., 1.]), torch.tensor([0., 1.]),
                             _constraints(), sample_ids=ids)


@pytest.mark.parametrize("ids", [["synthetic-1", "synthetic-0"],
                                 ["synthetic-0", "other"], ["synthetic-0"]])
def test_candidate_prediction_identity_must_match_exact_order_and_membership(ids):
    reference, _ = _reference([-1., 1.], [0, 1])
    with pytest.raises((ValueError, TypeError)):
        constraint_risks(torch.tensor([-1., 1.]), reference, sample_ids=ids)


@pytest.mark.parametrize("anchor,labels", [([math.nan, 1.], [0., 1.]),
                                          ([math.inf, 1.], [0., 1.]),
                                          ([-1., 1.], [0., 2.]),
                                          ([-1., 1.], [0., math.nan]),
                                          ([-1., 1.], [0.]), ([], [])])
def test_invalid_reference_values_and_support_rejected(anchor, labels):
    with pytest.raises((ValueError, TypeError)):
        _reference(anchor, labels)


@pytest.mark.parametrize("bad_student", [[math.nan, 0.], [0., math.inf], [0.], [[0., 1.]]])
def test_invalid_student_margins_rejected(bad_student):
    reference, ids = _reference([-1., 1.], [0, 1])
    with pytest.raises((ValueError, TypeError)):
        constraint_risks(torch.tensor(bad_student), reference, sample_ids=ids)


@pytest.mark.parametrize("floor", [0., -.1, math.nan, math.inf, True])
def test_invalid_margin_floor_rejected(floor):
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], margin_floor=floor)


def test_unknown_majorizer_and_duplicate_constraint_name_rejected():
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], majorizer="unknown")
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], [_constraints()[0], _constraints()[0]])


@pytest.mark.parametrize("kind,limit,group", [("unknown", .1, None),
                                            ("normal_fpr", -.01, None),
                                            ("normal_fpr", 1.01, None),
                                            ("normal_fpr", math.nan, None),
                                            ("attack_loss", math.inf, None),
                                            ("normal_fpr", .1, True)])
def test_invalid_constraint_definition_rejected(kind, limit, group):
    with pytest.raises((ValueError, TypeError)):
        constraint = RiskConstraint("bad", kind, limit, group=group)
        _reference([-1., 1.], [0, 1], [constraint])


def test_missing_group_metadata_or_insufficient_group_support_rejected():
    constraint = [RiskConstraint("normal7", "normal_fpr", .1, group=7)]
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], constraint)
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], constraint, normal_strata=torch.tensor([8, -1]))
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], constraint,
                   normal_strata=torch.tensor([7, -1]), min_support=2)
    with pytest.raises((ValueError, TypeError)):
        _reference([-1., 1.], [0, 1], constraint,
                   normal_strata=torch.tensor([7., -1.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable for synthetic tensor check")
def test_cuda_reference_integer_groups_and_student_gradients():
    ids = _ids(4)
    constraints = [RiskConstraint("normal7", "normal_fpr", .1, group=7),
                   RiskConstraint("attack7", "attack_loss", .1, group=7)]
    anchor = torch.tensor([-1., -.1, .1, 1.], device="cuda", requires_grad=True)
    reference = build_risk_reference(
        anchor, torch.tensor([0, 0, 1, 1], device="cuda"), constraints, sample_ids=ids,
        normal_strata=torch.tensor([7, 7, -1, -1], device="cuda"),
        attack_families=torch.tensor([-1, -1, 7, 7], device="cuda"),
    )
    student = torch.tensor([.1, -.1, -.1, .1], device="cuda", requires_grad=True)
    report = constraint_risks(student, reference, sample_ids=ids)
    risk_lagrangian(report, torch.ones(2, dtype=torch.float64)).backward()
    assert anchor.grad is None
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())
    assert bool((report.values >= report.actual_values).all())
