"""Independent scalar equations/gradients for clean-room near-neighbor controls.

Synthetic tensors only: no downloads, fitting, IDS scores or reserved data.
"""
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from ids.risk_baselines import (  # noqa: E402
    adapted_ekd_rank_count, mpt_style_focal_kl, pct_focal_kl,
)
from ids.risk_distillation import (  # noqa: E402
    ProjectedDual, RiskConstraint, build_risk_reference, constraint_risks,
)


def _sigmoid(value):
    return 1 / (1 + math.exp(-value))


def _bernoulli_kl(source, student, temperature):
    q, p = _sigmoid(source / temperature), _sigmoid(student / temperature)
    return q * math.log(q / p) + (1-q) * math.log((1-q) / (1-p))


@pytest.mark.parametrize("temperature", [.5, 1., 4.])
def test_pct_matches_manual_bernoulli_and_symmetric_two_logit_kl(temperature):
    s = torch.tensor([-.8, .3, 1.4, -.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-.2, 1.1, .6, -.4], dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 1])
    result = pct_focal_kl(s, labels, anchor_logits=b, anchor_threshold=.5, temperature=temperature,
                          alpha=1., beta=5., coefficient=.3)
    weights = [6., 6., 1., 1.]
    expected = .3 * temperature**2 * sum(
        w * _bernoulli_kl(float(old), float(new), temperature)
        for w, old, new in zip(weights, b, s.detach())) / 4
    assert result["soft"].item() == pytest.approx(expected, abs=1e-14)
    two_s = torch.stack((-s/2, s/2), dim=1)
    two_b = torch.stack((-b/2, b/2), dim=1)
    row_kl = F.kl_div(F.log_softmax(two_s/temperature, dim=1),
                      F.softmax(two_b/temperature, dim=1), reduction="none").sum(1)
    oracle = .3 * temperature**2 * (torch.tensor(weights) * row_kl).mean()
    torch.testing.assert_close(result["soft"], oracle)
    grad = torch.autograd.grad(result["soft"], s, retain_graph=True)[0]
    analytic = .3 * temperature / 4 * torch.tensor(weights) * (
        torch.sigmoid(s.detach()/temperature) - torch.sigmoid(b/temperature))
    torch.testing.assert_close(grad, analytic)
    torch.testing.assert_close(grad, torch.autograd.grad(oracle, s)[0])
    assert result["stats"]["anchor"]["correct_rows"] == 2


def test_pct_is_row_weighted_not_author_snapshot_cross_broadcast():
    s = torch.tensor([-.7, -.2], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([1., 2.], dtype=torch.float64)
    y = torch.tensor([1, 0])
    actual = pct_focal_kl(s, y, anchor_logits=b)["soft"]
    log_s = F.log_softmax(torch.stack((-s/2, s/2), 1), dim=1)
    prob_b = F.softmax(torch.stack((-b/2, b/2), 1), dim=1)
    rows = F.kl_div(log_s, prob_b, reduction="none").sum(1)
    correct = (rows * torch.tensor([6., 1.])).mean()
    wrong_broadcast = (rows * torch.tensor([[6.], [1.]])).mean()
    torch.testing.assert_close(actual, correct)
    assert not torch.isclose(actual, wrong_broadcast)
    actual_grad = torch.autograd.grad(actual, s, retain_graph=True)[0]
    wrong_grad = torch.autograd.grad(wrong_broadcast, s)[0]
    assert not torch.allclose(actual_grad, wrong_grad)


def test_pct_denominator_is_all_rows_and_does_not_rescale_selected_support():
    s = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    b = torch.ones(4, dtype=torch.float64)
    y = torch.tensor([1, 0, 0, 0])
    result = pct_focal_kl(s, y, anchor_logits=b, alpha=0., beta=1.)
    assert result["soft"].item() == pytest.approx(_bernoulli_kl(1., 0., 1.) / 4)
    result["soft"].backward()
    assert s.grad[1:].tolist() == [0., 0., 0.]


def test_frozen_working_point_changes_only_correctness_weight_not_raw_kl():
    s, b, y = torch.tensor([-.2]), torch.tensor([.2]), torch.tensor([0])
    raw = pct_focal_kl(s, y, anchor_logits=b, alpha=1., beta=0., anchor_threshold=0.)
    shifted = pct_focal_kl(s, y, anchor_logits=b, alpha=1., beta=0., anchor_threshold=100.)
    torch.testing.assert_close(raw["soft"], shifted["soft"], atol=0, rtol=0)
    a = pct_focal_kl(s, y, anchor_logits=b, alpha=0., beta=1., anchor_threshold=0.)
    c = pct_focal_kl(s, y, anchor_logits=b, alpha=0., beta=1., anchor_threshold=.3)
    assert a["soft"].item() == 0 and c["soft"].item() > 0
    threshold = math.nextafter(float(b.item()), math.inf)
    assert pct_focal_kl(s, y, anchor_logits=b, anchor_threshold=threshold)["stats"]["anchor"]["correct_rows"] == 1


def test_optional_zero_threshold_argmax_tie_convention_is_exact():
    s, b, y = torch.tensor([1., -1.]), torch.zeros(2), torch.tensor([0, 1])
    positive = pct_focal_kl(s, y, anchor_logits=b)
    argmax = pct_focal_kl(s, y, anchor_logits=b, correctness_tie_positive=False)
    assert positive["stats"]["anchor"]["correct_rows"] == argmax["stats"]["anchor"]["correct_rows"] == 1
    # Different per-row KL/gradients identify which tied row was protected.
    s = s.double().requires_grad_()
    argmax_loss = pct_focal_kl(s, y, anchor_logits=b, correctness_tie_positive=False)["soft"]
    two_b = torch.stack((-b/2, b/2), 1)
    weights = 1 + 5 * (two_b.argmax(1) == y)
    oracle_rows = F.kl_div(F.log_softmax(torch.stack((-s/2, s/2), 1), dim=1),
                           F.softmax(two_b.double(), dim=1), reduction="none").sum(1)
    torch.testing.assert_close(argmax_loss, (weights * oracle_rows).mean())


def test_pct_detaches_labels_anchor_and_all_scalar_weights():
    s = torch.tensor([-.4, .2], requires_grad=True)
    b = torch.tensor([.6, -.8], requires_grad=True)
    y = torch.tensor([1., 0.], requires_grad=True)
    alpha, beta, coefficient = [torch.tensor(value, requires_grad=True) for value in (1., 5., .3)]
    result = pct_focal_kl(s, y, anchor_logits=b, alpha=alpha, beta=beta, coefficient=coefficient)
    result["soft"].backward()
    assert s.grad is not None and bool(torch.isfinite(s.grad).all())
    assert all(value.grad is None for value in (b, y, alpha, beta, coefficient))


def test_mpt_style_two_sources_have_separate_correctness_masks_and_coefficients():
    s = torch.tensor([-.4, .1, .7], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-.3, .2, .5], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([.6, -.4, -.1], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0., 1., 0.], dtype=torch.float64, requires_grad=True)
    weight_b, weight_t = torch.tensor(.2, requires_grad=True), torch.tensor(.7, requires_grad=True)
    temp = 2.
    result = mpt_style_focal_kl(s, y, anchor_logits=b, teacher_logits=t,
                               anchor_threshold=.1, teacher_threshold=0., temperature=temp,
                               anchor_coefficient=weight_b, teacher_coefficient=weight_t)
    wb, wt = [6, 6, 1], [1, 1, 6]
    expected = temp**2/3 * sum(
        float(weight_b.detach()) * old_w * _bernoulli_kl(float(old.detach()), float(new.detach()), temp)
        + float(weight_t.detach()) * teacher_w * _bernoulli_kl(float(teacher.detach()), float(new.detach()), temp)
        for new, old, teacher, old_w, teacher_w in zip(s, b, t, wb, wt))
    assert result["soft"].item() == pytest.approx(expected, abs=1e-14)
    result["soft"].backward()
    assert bool(torch.isfinite(s.grad).all())
    assert all(value.grad is None for value in (b, t, y, weight_b, weight_t))
    assert result["stats"]["anchor"]["correct_rows"] == 2
    assert result["stats"]["teacher"]["correct_rows"] == 1
    assert not result["stats"]["class_incremental_bias"]


def test_mpt_style_author_no_t2_scaling_is_explicitly_convertible():
    s, b, y = torch.tensor([-.5, .3]), torch.tensor([.4, -.2]), torch.tensor([1, 0])
    temp = 5.
    result = mpt_style_focal_kl(s, y, anchor_logits=b, teacher_logits=b, temperature=temp,
                               anchor_coefficient=1/temp**2, teacher_coefficient=0.)
    expected = sum(6 * _bernoulli_kl(float(old), float(new), temp) for old, new in zip(b, s)) / 2
    assert result["soft"].item() == pytest.approx(expected, abs=1e-14)


def test_identical_logits_have_zero_soft_loss_without_a_hidden_hard_term():
    s = torch.tensor([-.5, .5], requires_grad=True)
    y = torch.tensor([1, 0])
    a = pct_focal_kl(s, y, anchor_logits=s.detach())["soft"]
    b = mpt_style_focal_kl(s, y, anchor_logits=s.detach(), teacher_logits=s.detach())["soft"]
    assert a.item() == b.item() == 0
    (a+b).backward()
    torch.testing.assert_close(s.grad, torch.zeros_like(s))


def _rank(score, thresholds, temperature):
    return sum(_sigmoid((score-threshold)/temperature) for threshold in thresholds)


def test_adapted_ekd_is_rank_count_mse_with_independent_class_denominators_and_gradient():
    s = torch.tensor([-.4, .2, .8, .1], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([-.1, -.2, .4, .5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0., 0., 0., 1.], dtype=torch.float64, requires_grad=True)
    st = torch.tensor([-.3, .3], dtype=torch.float64, requires_grad=True)
    tt = torch.tensor([-.2, .6], dtype=torch.float64, requires_grad=True)
    n, a, c = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (.7, 1.4, .3)]
    tau = .4
    result = adapted_ekd_rank_count(s, y, teacher_logits=t, student_thresholds=st, teacher_thresholds=tt,
                                    temperature=tau, normal_coefficient=n, attack_coefficient=a, coefficient=c)
    diff = [_rank(float(z), st.detach().tolist(), tau)-_rank(float(q), tt.detach().tolist(), tau)
            for z, q in zip(s.detach(), t.detach())]
    expected = .3 * (.7 * sum(d*d for d in diff[:3])/3 + 1.4*diff[3]**2)
    assert result["soft"].item() == pytest.approx(expected, abs=1e-14)
    gradient = []
    for i, z in enumerate(s.detach().tolist()):
        rank_grad = sum((p := _sigmoid((z-threshold)/tau))*(1-p)/tau for threshold in st.detach().tolist())
        gradient.append(.3 * (.7/3 if i < 3 else 1.4) * 2 * diff[i] * rank_grad)
    result["soft"].backward()
    torch.testing.assert_close(s.grad, torch.tensor(gradient, dtype=torch.float64))
    assert all(value.grad is None for value in (t, y, st, tt, n, a, c))
    assert result["stats"]["normal_rows"] == 3 and result["stats"]["attack_rows"] == 1


def test_adapted_ekd_sums_before_mse_and_does_not_normalize_anchor_count():
    s, t, y = torch.tensor([.2, -.3]), torch.tensor([-.2, .1]), torch.tensor([0, 1])
    one = adapted_ekd_rank_count(s, y, teacher_logits=t, student_thresholds=[0.], teacher_thresholds=[0.])
    two = adapted_ekd_rank_count(s, y, teacher_logits=t, student_thresholds=[0., 0.], teacher_thresholds=[0., 0.])
    torch.testing.assert_close(two["soft"], 4*one["soft"])
    assert not two["stats"]["temperature_squared_factor"]


@pytest.mark.parametrize("label", [0, 1])
def test_adapted_ekd_absent_class_is_zero_reported_without_nan(label):
    s = torch.tensor([-.2, .3], requires_grad=True)
    t = torch.tensor([.1, -.2])
    result = adapted_ekd_rank_count(s, torch.full((2,), label), teacher_logits=t,
                                    student_thresholds=[0.], teacher_thresholds=[0.])
    missing = "attack" if label == 0 else "normal"
    assert result["stats"][missing+"_absent"] and result["stats"][missing+"_mse"] == 0
    result["soft"].backward()
    assert bool(torch.isfinite(s.grad).all())


@pytest.mark.parametrize("function", [pct_focal_kl, mpt_style_focal_kl, adapted_ekd_rank_count])
@pytest.mark.parametrize("partition", ["reference", "constraint_train", "validation", "calibration", "test"])
def test_soft_baselines_refuse_other_roles_before_inspecting_inputs(function, partition):
    kwargs = {"anchor_logits": None} if function is pct_focal_kl else (
        {"anchor_logits": None, "teacher_logits": None} if function is mpt_style_focal_kl else
        {"teacher_logits": None, "student_thresholds": None, "teacher_thresholds": None})
    with pytest.raises(ValueError, match="fitting role"):
        function(None, None, partition=partition, **kwargs)


@pytest.mark.parametrize("bad", [torch.ones(2, 1), torch.ones(2, 2), torch.tensor([]),
                                 torch.tensor([0., float("nan")]), torch.tensor([0., float("inf")]),
                                 torch.ones(2, dtype=torch.int64)])
def test_scalar_logit_contract_rejects_shapes_and_nonfinite_values(bad):
    with pytest.raises(ValueError):
        pct_focal_kl(bad, torch.tensor([0, 1]), anchor_logits=torch.ones(2))
    with pytest.raises(ValueError):
        pct_focal_kl(torch.ones(2), torch.tensor([0, 1]), anchor_logits=bad)


@pytest.mark.parametrize("bad_labels", [torch.tensor([True, False]), torch.tensor([0., 2.]),
                                        torch.tensor([0., float("nan")]), torch.zeros(2, 1), torch.zeros(3)])
def test_invalid_labels_are_rejected(bad_labels):
    with pytest.raises(ValueError):
        pct_focal_kl(torch.ones(2), bad_labels, anchor_logits=torch.ones(2))


@pytest.mark.parametrize("key,value", [("temperature", 0.), ("temperature", 1e200), ("temperature", 1e-200),
                                      ("alpha", -1.), ("beta", True), ("coefficient", float("nan")),
                                      ("coefficient", torch.tensor([1.])), ("anchor_threshold", float("inf")),
                                      ("correctness_tie_positive", 1)])
def test_invalid_fixed_parameters_fail_closed(key, value):
    with pytest.raises(ValueError):
        pct_focal_kl(torch.ones(2), torch.tensor([0, 1]), anchor_logits=torch.ones(2), **{key: value})


@pytest.mark.parametrize("student_thresholds,teacher_thresholds", [([], []), ([0., 1.], [0.]),
                                                               ([float("inf")], [0.]),
                                                               (torch.zeros(1, 1), [0.]), ([True], [0.])])
def test_ekd_requires_aligned_finite_fixed_working_point_arrays(student_thresholds, teacher_thresholds):
    with pytest.raises(ValueError):
        adapted_ekd_rank_count(torch.ones(2), torch.tensor([0, 1]), teacher_logits=torch.ones(2),
                               student_thresholds=student_thresholds, teacher_thresholds=teacher_thresholds)


def test_extreme_finite_binary_logits_are_stable_and_teacher_stays_detached():
    s = torch.tensor([-1000., 1000.], requires_grad=True)
    b = torch.tensor([1000., -1000.], requires_grad=True)
    result = pct_focal_kl(s, torch.tensor([1, 0]), anchor_logits=b)
    assert result["soft"].item() == pytest.approx(6000.)
    result["soft"].backward()
    assert bool(torch.isfinite(s.grad).all()) and b.grad is None


def _normal_reference(anchor, *, form="capped", majorizer="adaptive", credit=True):
    ids = [f"synthetic-normal-{i}" for i in range(len(anchor))]
    reference = build_risk_reference(
        anchor, torch.zeros(len(anchor), dtype=torch.int64),
        [RiskConstraint("normal", "normal_fpr", .15)], sample_ids=ids, margin_floor=.1,
        normal_loss_form=form, majorizer=majorizer, normal_repair_credit=credit)
    return reference, ids


def test_normal_hinge_old_false_positive_has_nonzero_gradient_and_true_anchor_slack():
    anchor = torch.tensor([1., -.5, .01, -.01], dtype=torch.float64)
    student = anchor.clone().requires_grad_()
    hinge, ids = _normal_reference(anchor, form="hinge")
    report = constraint_risks(student, hinge, sample_ids=ids)
    # At the frozen anchor: [2,0,1.1,.9], mean=1, hard normal FPR=.5.
    torch.testing.assert_close(report.values, torch.tensor([1.], dtype=torch.float64))
    torch.testing.assert_close(report.baseline_slack, torch.tensor([.5], dtype=torch.float64))
    report.values.sum().backward()
    assert student.grad[0].item() == pytest.approx(.25)
    capped, ids = _normal_reference(anchor)
    old_fp_gradient = torch.autograd.grad(constraint_risks(student, capped, sample_ids=ids).values.sum(), student)[0]
    assert old_fp_gradient[0].item() == 0
    assert report.diagnostics["normal"]["anchor_hinge_upper"] == pytest.approx(1.)
    assert report.diagnostics["normal"]["repair_credit_lower"] == 0


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
def test_normal_hinge_is_pointwise_upper_bound_with_adaptive_anchor_scale(majorizer):
    # Test each individual event, including threshold equality; no aggregate cancellation.
    for old in [-2., -.1, -.01, 0., .01, .1, 2.]:
        reference, ids = _normal_reference(torch.tensor([old], dtype=torch.float64),
                                          form="hinge", majorizer=majorizer)
        for new in [-3., -.1, -.001, 0., .001, .1, 3.]:
            report = constraint_risks(torch.tensor([new], dtype=torch.float64), reference, sample_ids=ids)
            event = float(new >= 0)
            expected = max(0., 1 + new/max(abs(old), .1))
            assert report.actual_values.item() == event
            assert report.values.item() == pytest.approx(expected)
            assert report.values.item() + 1e-12 >= event


def test_normal_loss_form_is_bound_into_signature_and_dual_identity():
    from dataclasses import replace

    anchor = torch.tensor([-1., 1.], dtype=torch.float64)
    capped, ids = _normal_reference(anchor)
    hinge, _ = _normal_reference(anchor, form="hinge")
    assert capped.normal_loss_form == "capped" and hinge.normal_loss_form == "hinge"
    assert capped.signature != hinge.signature
    altered = replace(capped, normal_loss_form="hinge")
    with pytest.raises(ValueError, match="modified"):
        constraint_risks(anchor, altered, sample_ids=ids)
    dual = ProjectedDual(capped.constraints, reference_signature=capped.signature)
    with pytest.raises(ValueError, match="another frozen risk reference"):
        dual.update(constraint_risks(anchor, hinge, sample_ids=ids))


def test_hinge_normal_control_leaves_attack_objective_unchanged_and_ignores_repair_credit():
    anchor = torch.tensor([1., -1., .01, -.01], dtype=torch.float64)
    student = torch.tensor([.5, -.2, -.02, .03], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    constraints = [RiskConstraint("normal", "normal_fpr", .15), RiskConstraint("lost", "attack_loss", .05)]
    ids = [str(i) for i in range(4)]
    reports = []
    for form, credit in [("capped", True), ("hinge", True), ("hinge", False)]:
        ref = build_risk_reference(anchor, labels, constraints, sample_ids=ids, margin_floor=.1,
                                   normal_loss_form=form, normal_repair_credit=credit)
        reports.append(constraint_risks(student, ref, sample_ids=ids))
    torch.testing.assert_close(reports[0].values[1], reports[1].values[1])
    torch.testing.assert_close(reports[0].baseline_slack[1], reports[1].baseline_slack[1])
    torch.testing.assert_close(reports[1].values, reports[2].values)


@pytest.mark.parametrize("form", [None, True, "unknown", ["hinge"]])
def test_unknown_normal_loss_form_rejected(form):
    with pytest.raises(ValueError, match="normal_loss_form"):
        _normal_reference(torch.tensor([-1.]), form=form)
