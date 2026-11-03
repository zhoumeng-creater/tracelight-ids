"""Prospective loss equations, gradients and train-batch misuse boundaries."""

from dataclasses import replace
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from ids.research_kd import ResearchKDConfig, research_kd_loss  # noqa: E402 (optional torch)


def _data():
    return (torch.tensor([-2., -.2, .4, 2.], dtype=torch.float64, requires_grad=True),
            torch.tensor([0., 1., 0., 1.], dtype=torch.float64))


def test_hard_equals_existing_weighted_bce_without_hidden_normalization():
    logits, labels = _data()
    result = research_kd_loss(logits, labels, ResearchKDConfig(positive_weight=.4, alpha=.9))
    expected = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(.4))
    torch.testing.assert_close(result.total, expected)
    assert result.statistics["gate_active_count"] == 0
    assert result.statistics["pairwise_pairs"] == 0
    result.total.backward()
    assert bool(torch.isfinite(logits.grad).all())


@pytest.mark.parametrize("temperature", [.5, 1., 2.])
@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_weight_matched_hard_is_exact_q_equals_y_kd_control(temperature, alpha):
    logits, labels = _data()
    config = ResearchKDConfig(mode="matched_hard", alpha=alpha, positive_weight=.4,
                              temperature=temperature)
    matched = research_kd_loss(logits, labels, config)
    kd = research_kd_loss(logits, labels, replace(config, mode="kd"), teacher_scores=labels)
    torch.testing.assert_close(matched.total, kd.total, rtol=0, atol=0)
    grad_matched = torch.autograd.grad(matched.total, logits, retain_graph=True)[0]
    grad_kd = torch.autograd.grad(kd.total, logits)[0]
    torch.testing.assert_close(grad_matched, grad_kd, rtol=0, atol=0)


def test_weight_matched_hard_has_analytic_class_weight_at_temperature_one():
    logits, labels = _data()
    config = ResearchKDConfig(mode="matched_hard", alpha=.3, positive_weight=.4)
    result = research_kd_loss(logits, labels, config)
    effective = (1 - config.alpha) * config.positive_weight + config.alpha
    expected = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(effective))
    torch.testing.assert_close(result.total, expected)
    torch.testing.assert_close(torch.autograd.grad(result.total, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_kd_equation_and_teacher_detachment(alpha):
    logits, labels = _data()
    teacher = torch.tensor([.1, .8, .2, .9], dtype=torch.float64, requires_grad=True)
    config = ResearchKDConfig(mode="kd", alpha=alpha, positive_weight=.4)
    result = research_kd_loss(logits, labels, config, teacher_scores=teacher)
    hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(.4))
    soft = F.binary_cross_entropy_with_logits(logits, teacher.detach())
    torch.testing.assert_close(result.total, (1 - alpha) * hard + alpha * soft)
    result.total.backward()
    assert teacher.grad is None
    assert bool(torch.isfinite(logits.grad).all())


def test_binary_temperature_is_explicit_score_logodds_scaling():
    logits, labels = _data()
    teacher = torch.tensor([0., .2, .8, 1.], dtype=torch.float64)
    config = ResearchKDConfig(mode="kd", alpha=1., temperature=2.)
    result = research_kd_loss(logits, labels, config, teacher_scores=teacher)
    tempered = torch.sigmoid(torch.logit(teacher) / 2)
    expected = 4 * F.binary_cross_entropy_with_logits(logits / 2, tempered)
    assert tempered[0] == 0 and tempered[-1] == 1
    torch.testing.assert_close(result.total, expected)
    result.total.backward()
    assert bool(torch.isfinite(logits.grad).all())


def test_wrong_or_boundary_teachers_restore_original_hard_loss_and_gradients():
    logits, labels = _data()
    # Wrong on each row, including maximally confident errors.
    wrong = 1 - labels
    config = ResearchKDConfig(mode="gated_kd", alpha=1., positive_weight=.4,
                              gate_min_confidence=0.)
    for teacher in (wrong, torch.full_like(labels, .5)):
        result = research_kd_loss(logits, labels, config, teacher_scores=teacher)
        hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(.4))
        torch.testing.assert_close(result.total, hard, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(result.total, logits, retain_graph=True)[0],
                                   torch.autograd.grad(hard, logits, retain_graph=True)[0])
        assert result.statistics["gate_active_count"] == 0
        assert result.statistics["gate_max"] == 0


def test_confidence_gate_uses_raw_score_threshold_and_whole_batch_mean():
    logits = torch.tensor([-.4, .3, .2, 1., -1., .2], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1., 0., 1., 0., 1., 0.], dtype=torch.float64)
    teacher = torch.tensor([.9, .2, .6, .5, .1, .8], dtype=torch.float64, requires_grad=True)
    config = ResearchKDConfig(mode="gated_kd", alpha=.5, positive_weight=.4,
                              gate_min_confidence=.5)
    result = research_kd_loss(logits, labels, config, teacher_scores=teacher)
    gate = torch.tensor([.8, .6, 0., 0., 0., 0.], dtype=torch.float64)
    hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(.4),
                                             reduction="none")
    soft = F.binary_cross_entropy_with_logits(logits, teacher.detach(), reduction="none")
    expected = ((1 - .5 * gate) * hard + .5 * gate * soft).mean()
    torch.testing.assert_close(result.total, expected)
    assert result.statistics["gate_correct_count"] == 3
    assert result.statistics["gate_active_count"] == 2
    assert result.statistics["gate_mean"] == pytest.approx(1.4 / 6)
    assert result.statistics["gate_max"] == pytest.approx(.8)
    result.total.backward()
    assert teacher.grad is None


@pytest.mark.parametrize("gate_policy", ["correct_confidence", "correct_only", "confidence_only"])
@pytest.mark.parametrize("temperature", [.5, 1., 2.])
def test_gated_matched_hard_uses_real_teacher_gate_and_exact_q_equals_y_target(gate_policy, temperature):
    logits, labels = _data()
    teacher = torch.tensor([.1, .9, .8, .6], dtype=torch.float64, requires_grad=True)
    config = ResearchKDConfig(mode="gated_matched_hard", alpha=.4, positive_weight=.3,
                              temperature=temperature, gate_min_confidence=.5, gate_policy=gate_policy)
    result = research_kd_loss(logits, labels, config, teacher_scores=teacher)
    real_kd = research_kd_loss(logits, labels, replace(config, mode="gated_kd"), teacher_scores=teacher)
    assert result.statistics == real_kd.statistics
    confidence = (2 * teacher.detach() - 1).abs()
    correct = torch.tensor([True, True, False, True])
    eligible = confidence >= .5
    if gate_policy != "confidence_only":
        eligible &= correct
    magnitude = torch.ones_like(confidence) if gate_policy == "correct_only" else confidence
    gate = torch.where(eligible, magnitude, torch.zeros_like(confidence))
    hard = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=logits.new_tensor(.3), reduction="none")
    soft = F.binary_cross_entropy_with_logits(logits / temperature, labels, reduction="none") * temperature ** 2
    expected = ((1 - .4 * gate) * hard + .4 * gate * soft).mean()
    torch.testing.assert_close(result.total, expected)
    torch.testing.assert_close(torch.autograd.grad(result.total, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])
    result.total.backward()
    assert teacher.grad is None


def test_gate_policy_ablation_exposes_wrong_teacher_mass_and_confidence_magnitude():
    logits, labels = _data()
    # First two correct and confident, third wrong and confident, last uncertain.
    teacher = torch.tensor([.1, .9, .8, .6], dtype=torch.float64)
    baseline = ResearchKDConfig(mode="gated_kd", gate_min_confidence=.5)
    results = {policy: research_kd_loss(logits, labels, replace(baseline, gate_policy=policy),
                                       teacher_scores=teacher)
               for policy in ("correct_confidence", "correct_only", "confidence_only")}
    assert results["correct_confidence"].statistics["gate_total_mass"] == pytest.approx(1.6)
    assert results["correct_only"].statistics["gate_total_mass"] == pytest.approx(2.)
    assert results["confidence_only"].statistics["gate_total_mass"] == pytest.approx(2.2)
    assert results["correct_confidence"].statistics["gate_wrong_mass"] == 0
    assert results["correct_only"].statistics["gate_wrong_active_count"] == 0
    assert results["confidence_only"].statistics["gate_wrong_mass"] == pytest.approx(.6)
    assert results["confidence_only"].statistics["gate_wrong_active_count"] == 1
    for result in results.values():
        assert result.statistics["gate_total_mass"] == pytest.approx(
            result.statistics["gate_correct_mass"] + result.statistics["gate_wrong_mass"])


@pytest.mark.parametrize("mode", ["gated_kd", "gated_matched_hard"])
def test_all_gated_modes_require_real_teacher_even_with_alpha_zero(mode):
    logits, labels = _data()
    with pytest.raises(ValueError, match="require aligned OOF"):
        research_kd_loss(logits, labels, ResearchKDConfig(mode=mode, alpha=0))


def test_gate_statistics_do_not_imply_non_gated_teachers_are_wrong():
    logits, labels = _data()
    result = research_kd_loss(logits, labels, ResearchKDConfig(mode="kd"), teacher_scores=labels)
    assert all(value == 0 for key, value in result.statistics.items() if key.startswith("gate_"))


def test_pairwise_selects_hardest_train_scores_and_has_expected_gradient_direction():
    logits = torch.tensor([-.5, 3., 1., 2.], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    config = ResearchKDConfig(pairwise_weight=.1, normal_fraction=.5, max_attacks=1,
                              pairwise_margin=.5)
    result = research_kd_loss(logits, labels, config)
    expected = F.softplus(logits.new_tensor(.5 - (1. - 3.)))
    torch.testing.assert_close(result.components["pairwise"], expected)
    grad = torch.autograd.grad(result.components["pairwise"], logits)[0]
    assert grad[0] == 0 and grad[3] == 0
    assert grad[1] > 0 and grad[2] < 0
    assert result.statistics["pairwise_pairs"] == 1
    torch.testing.assert_close(result.total,
                               result.components["base"] + .1 * result.components["pairwise"])


def test_pairwise_group_balancing_assigns_equal_selected_group_mass_on_both_axes():
    logits = torch.tensor([0., 1., 2., -1., -.5, -.2], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    groups = torch.tensor([10, 10, 11, 20, 20, 21])
    config = ResearchKDConfig(pairwise_weight=1, normal_fraction=1)
    result = research_kd_loss(logits, labels, config, group_ids=groups)
    matrix = F.softplus(config.pairwise_margin - (logits[3:, None] - logits[:3][None, :]))
    weights = logits.new_tensor([.25, .25, .5])
    expected = (matrix * weights[:, None] * weights[None, :]).sum()
    torch.testing.assert_close(result.components["pairwise"], expected)
    assert result.statistics["pairwise_pairs"] == 9
    assert result.statistics["pairwise_normal_groups"] == 2
    assert result.statistics["pairwise_attack_groups"] == 2


def test_duplicate_identical_rows_within_group_do_not_change_full_pairwise_group_mean():
    logits = torch.tensor([-.5, 2., -.2, .8], dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    groups = torch.tensor([10, 11, 20, 21])
    config = ResearchKDConfig(pairwise_weight=1, normal_fraction=1)
    reference = research_kd_loss(logits, labels, config, group_ids=groups)
    index = torch.tensor([0, 0, 0, 1, 2, 2, 3])
    duplicate = research_kd_loss(logits[index], labels[index], config, group_ids=groups[index])
    torch.testing.assert_close(reference.components["pairwise"], duplicate.components["pairwise"])


@pytest.mark.parametrize("max_pairs", [1, 2, 7, 12])
def test_pairwise_never_exceeds_declared_cartesian_pair_budget(max_pairs):
    logits = torch.arange(20, dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0] * 10 + [1] * 10)
    config = ResearchKDConfig(pairwise_weight=1, normal_fraction=.5, max_pairs=max_pairs)
    result = research_kd_loss(logits, labels, config)
    stats = result.statistics
    assert 1 <= stats["pairwise_pairs"] <= max_pairs
    assert stats["pairwise_selected_normals"] <= math.ceil(.5 * 10)
    assert stats["pairwise_pairs"] == stats["pairwise_selected_normals"] * stats["pairwise_selected_attacks"]
    result.total.backward()
    assert bool(torch.isfinite(logits.grad).all())


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_batches_have_differentiable_zero_pairwise_loss(label):
    logits = torch.tensor([-.5, .5], requires_grad=True)
    labels = torch.tensor([label, label])
    result = research_kd_loss(logits, labels, ResearchKDConfig(pairwise_weight=1))
    assert result.components["pairwise"] == 0
    result.total.backward()
    assert bool(torch.isfinite(logits.grad).all())
    assert result.statistics["pairwise_pairs"] == 0


@pytest.mark.parametrize("partition", ["validation", "calibration", "test", "", None])
def test_reserved_partitions_rejected_before_accessing_inputs(partition):
    with pytest.raises(ValueError, match="train batches only"):
        research_kd_loss(None, None, ResearchKDConfig(pairwise_weight=1), partition=partition)


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"mode": {}}, {"alpha": -.1}, {"alpha": 1.1}, {"alpha": True},
    {"positive_weight": 0}, {"positive_weight": float("inf")},
    {"temperature": 0}, {"temperature": float("nan")}, {"temperature": 1e200},
    {"temperature": 1e-200},
    {"gate_min_confidence": -1}, {"gate_min_confidence": 1.1},
    {"gate_policy": "unknown"}, {"gate_policy": []},
    {"pairwise_weight": -.1}, {"normal_fraction": 0}, {"normal_fraction": 1.1},
    {"max_normals": 0}, {"max_attacks": True}, {"max_pairs": 1.5},
    {"pairwise_margin": -1}, {"pairwise_temperature": 0},
])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        ResearchKDConfig(**changes)


@pytest.mark.parametrize("kind", ["logit_shape", "empty", "label_shape", "label_value",
                                  "teacher_shape", "teacher_value", "teacher_missing", "group_dtype"])
def test_invalid_batch_contract_rejected(kind):
    logits, labels = _data()
    config = ResearchKDConfig(mode="kd")
    teacher = torch.full_like(logits, .7)
    groups = None
    if kind == "logit_shape":
        logits = logits[:, None]
    elif kind == "empty":
        logits, labels, teacher = logits[:0], labels[:0], teacher[:0]
    elif kind == "label_shape":
        labels = labels[:, None]
    elif kind == "label_value":
        labels[0] = .2
    elif kind == "teacher_shape":
        teacher = teacher[:, None]
    elif kind == "teacher_value":
        teacher[0] = 1.1
    elif kind == "teacher_missing":
        teacher = None
    else:
        groups = torch.zeros_like(labels)
    with pytest.raises(ValueError):
        research_kd_loss(logits, labels, config, teacher_scores=teacher, group_ids=groups)


def test_scalar_component_accounting_and_json_safe_metrics():
    logits, labels = _data()
    config = ResearchKDConfig(mode="gated_kd", positive_weight=.4, pairwise_weight=.1,
                              normal_fraction=1)
    result = research_kd_loss(logits, labels, config, teacher_scores=labels)
    assert all(value.ndim == 0 for value in result.components.values())
    torch.testing.assert_close(result.components["base"],
                               result.components["hard_contribution"] + result.components["soft_contribution"])
    metrics = result.detached_metrics()
    assert all(type(value) in (int, float) for value in metrics.values())
    assert metrics["total"] == pytest.approx(float(result.total))


@pytest.mark.parametrize("changes", [{"positive_weight": 1e-300},
                                      {"mode": "matched_hard", "temperature": 1e-30}])
def test_underflow_cannot_silently_remove_class_weight_or_temperature_scale(changes):
    logits = torch.tensor([-.5, .5], requires_grad=True)
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError, match="remain positive and finite"):
        research_kd_loss(logits, labels, ResearchKDConfig(**changes))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_loss_supports_integer_groups_and_detached_teacher():
    logits = torch.tensor([-.5, .5, .1, -.1], device="cuda", requires_grad=True)
    labels = torch.tensor([0., 1., 0., 1.], device="cuda")
    teacher = torch.tensor([.1, .9, .2, .8], device="cuda", requires_grad=True)
    groups = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int64)
    config = ResearchKDConfig(mode="gated_kd", pairwise_weight=.1, normal_fraction=1)
    result = research_kd_loss(logits, labels, config, teacher_scores=teacher, group_ids=groups)
    result.total.backward()
    assert result.total.device.type == "cuda"
    assert bool(torch.isfinite(logits.grad).all())
    assert teacher.grad is None


def test_mixed_device_teacher_rejected_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional")
    logits, labels = _data()
    with pytest.raises(ValueError, match="same device"):
        research_kd_loss(logits, labels, ResearchKDConfig(mode="kd"),
                         teacher_scores=torch.zeros(4, device="cuda"))
