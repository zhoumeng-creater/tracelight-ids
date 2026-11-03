"""Independent equations and role/teacher-gradient checks for boundary KD."""
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from ids.risk_distillation import (  # noqa: E402
    BoundaryKDConfig, boundary_kd_objective, freeze_margin_threshold, working_margins,
)


def inputs():
    return (torch.tensor([-1.2, -.1, .2, 2.3], dtype=torch.float64, requires_grad=True),
            torch.tensor([0., 1., 0., 1.], dtype=torch.float64))


def test_fixed_hard_term_and_teacher_working_margin_equation():
    z, y = inputs()
    teacher = torch.tensor([-.5, .8, -.1, 1.1], dtype=torch.float64, requires_grad=True)
    settings = BoundaryKDConfig(kd_weight=.4, positive_weight=.3, temperature=2.)
    result = boundary_kd_objective(z, y, settings, student_threshold=.7, teacher_margins=teacher)
    hard = F.binary_cross_entropy_with_logits(z, y, pos_weight=z.new_tensor(.3))
    soft = 4 * F.binary_cross_entropy_with_logits((z - .7) / 2, torch.sigmoid(teacher.detach() / 2))
    torch.testing.assert_close(result["total"], hard + .4 * soft)
    result["total"].backward()
    assert teacher.grad is None
    assert bool(torch.isfinite(z.grad).all())


@pytest.mark.parametrize("temperature", [.5, 1., 2.])
def test_matched_control_is_exact_label_target_with_same_temperature_and_threshold(temperature):
    z, y = inputs()
    config = BoundaryKDConfig(mode="matched_hard", kd_weight=.6, positive_weight=.4, temperature=temperature)
    matched = boundary_kd_objective(z, y, config, student_threshold=.3)
    teacher = (2*y-1)*1000  # finite margins whose sigmoid is exactly 0/1 here
    kd = boundary_kd_objective(z, y, replace(config, mode="kd"), student_threshold=.3,
                               teacher_margins=teacher)
    torch.testing.assert_close(matched["total"], kd["total"], rtol=0, atol=0)
    grad = torch.autograd.grad(matched["total"], z, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.autograd.grad(kd["total"], z)[0], rtol=0, atol=0)


def test_additive_matched_control_has_known_effective_weight_only_at_zero_threshold_and_t_one():
    z, y = inputs()
    beta, positive = .6, .4
    settings = BoundaryKDConfig(mode="matched_hard", kd_weight=beta, positive_weight=positive)
    actual = boundary_kd_objective(z, y, settings, student_threshold=0.)["total"]
    expected = (1+beta)*F.binary_cross_entropy_with_logits(
        z, y, pos_weight=z.new_tensor((positive+beta)/(1+beta)))
    torch.testing.assert_close(actual, expected)
    shifted = boundary_kd_objective(z, y, settings, student_threshold=.7)["total"]
    assert not torch.isclose(shifted, expected)


@pytest.mark.parametrize("mode", ["hard", "kd", "matched_hard"])
def test_disabled_soft_signal_equals_unchanged_hard(mode):
    z, y = inputs()
    result = boundary_kd_objective(z, y, BoundaryKDConfig(mode=mode, kd_weight=0., positive_weight=.3),
                                    student_threshold=2.)
    expected = F.binary_cross_entropy_with_logits(z, y, pos_weight=z.new_tensor(.3))
    torch.testing.assert_close(result["total"], expected, rtol=0, atol=0)
    assert result["soft"].item() == 0


@pytest.mark.parametrize("partition", ["train", "constraint_train", "reference", "validation", "calibration", "test"])
def test_reserved_and_wrong_roles_fail_before_inspecting_values(partition):
    with pytest.raises(ValueError, match="fitting role"):
        boundary_kd_objective(None, None, None, student_threshold=0., partition=partition)


@pytest.mark.parametrize("settings", [dict(mode="gated_kd"), dict(kd_weight=True), dict(kd_weight=-1),
                                       dict(positive_weight=0), dict(temperature=0),
                                       dict(temperature=1e200), dict(temperature=1e-200)])
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        BoundaryKDConfig(**settings)


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), torch.tensor(0.)])
def test_threshold_is_a_frozen_finite_scalar(threshold):
    z, y = inputs()
    with pytest.raises(ValueError):
        boundary_kd_objective(z, y, BoundaryKDConfig(mode="hard"), student_threshold=threshold)


def test_missing_or_nonfinite_teacher_is_rejected():
    z, y = inputs()
    for teacher in (None, torch.zeros(3), torch.tensor([float("nan"), 0., 0., 0.])):
        with pytest.raises(ValueError):
            boundary_kd_objective(z, y, BoundaryKDConfig(), student_threshold=0., teacher_margins=teacher)


def test_reference_scale_is_fixed_and_applied_to_student_margin_only():
    z, y = inputs()
    teacher = torch.tensor([-.4, .2, -.3, .8], dtype=torch.float64)
    result = boundary_kd_objective(z, y, BoundaryKDConfig(), student_threshold=.7,
                                    student_scale=2., teacher_margins=teacher)
    expected = F.binary_cross_entropy_with_logits((z-.7)/2, torch.sigmoid(teacher))
    torch.testing.assert_close(result["soft"], expected)
    for scale in (0, True, -1., float("nan"), 1e-320):
        with pytest.raises(ValueError):
            boundary_kd_objective(z, y, BoundaryKDConfig(), student_threshold=.7,
                                    student_scale=scale, teacher_margins=teacher)


def test_raw_logit_reference_budget_uses_integer_count_and_rejects_ties():
    z = torch.arange(64, dtype=torch.float32)
    result = freeze_margin_threshold(z, torch.zeros_like(z), max_fpr=.15)
    assert result["allowed_false_positives"] == 9
    assert result["false_positives"] == 9
    assert result["achieved_fpr"] <= .15
    # All tied scores are rejected when even one would exceed the budget.
    tied = freeze_margin_threshold(torch.ones(64), torch.zeros(64), max_fpr=.15)
    assert tied["false_positives"] == 0 and tied["threshold"] > 1


def test_nextafter_reference_threshold_is_not_cast_back_to_float32_tie():
    z = torch.ones(3, requires_grad=True)
    result = freeze_margin_threshold(z, torch.zeros(3), max_fpr=0.)
    margins = working_margins(z, threshold=result["threshold"])
    assert margins.dtype == torch.float64 and bool((margins < 0).all())
    assert result["false_positives"] == 0
    margins.sum().backward()
    torch.testing.assert_close(z.grad, torch.ones_like(z))


def test_attack_reference_scores_cannot_influence_normal_threshold():
    y = torch.tensor([0, 0, 0, 1, 1])
    a = freeze_margin_threshold(torch.tensor([-2., 0., 1., 2., 3.]), y, .4)
    b = freeze_margin_threshold(torch.tensor([-2., 0., 1., -500., 1000.]), y, .4)
    assert a == b


@pytest.mark.parametrize("partition", ["fit", "constraint_train", "validation", "calibration", "test"])
def test_reference_threshold_guard_rejects_other_roles_first(partition):
    with pytest.raises(ValueError, match="reference role"):
        freeze_margin_threshold(None, None, partition=partition)


@pytest.mark.parametrize("alpha", [True, -1., 1., float("nan")])
def test_invalid_reference_budget_rejected(alpha):
    with pytest.raises(ValueError):
        freeze_margin_threshold(torch.zeros(3), torch.zeros(3), alpha)


def test_reference_without_normal_or_nonrepresentable_threshold_is_rejected():
    with pytest.raises(ValueError):
        freeze_margin_threshold(torch.zeros(3), torch.ones(3))
    with pytest.raises(ValueError):
        freeze_margin_threshold(torch.tensor([torch.finfo(torch.float64).max]), torch.zeros(1))
