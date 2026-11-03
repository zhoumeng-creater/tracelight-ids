"""Synthetic equations/gradient controls for ordinary raw binary KL."""
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from ids.upgrade_kd import raw_bernoulli_kl, raw_bernoulli_kl_rows  # noqa: E402


@pytest.mark.parametrize("temperature", [.5, 1., 4.])
def test_rows_match_symmetric_two_logit_kl_and_analytic_gradient(temperature):
    s = torch.tensor([-1.3, .2, 1.8, -.4], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([-.7, 1.3, .2, -2.], dtype=torch.float64, requires_grad=True)
    shift = .3
    rows = raw_bernoulli_kl_rows(s, teacher_logits=t, temperature=temperature, teacher_logit_shift=shift)
    ss, tt = torch.stack((-s/2, s/2), 1), torch.stack((-(t.detach()+shift)/2, (t.detach()+shift)/2), 1)
    oracle = temperature**2 * F.kl_div(F.log_softmax(ss/temperature, 1), F.softmax(tt/temperature, 1),
                                      reduction="none").sum(1)
    torch.testing.assert_close(rows, oracle, atol=1e-14, rtol=1e-12)
    grad = torch.autograd.grad(rows.sum(), (s, t), allow_unused=True)
    expected = temperature * (torch.sigmoid(s.detach()/temperature) - torch.sigmoid((t.detach()+shift)/temperature))
    torch.testing.assert_close(grad[0], expected)
    assert grad[1] is None


def test_manual_bernoulli_values_and_direction_are_teacher_to_student():
    s, t = torch.tensor([-1., .3], dtype=torch.float64), torch.tensor([.7, -2.], dtype=torch.float64)
    rows = raw_bernoulli_kl_rows(s, teacher_logits=t)
    q, p = torch.sigmoid(t), torch.sigmoid(s)
    expected = q * torch.log(q/p) + (1-q) * torch.log((1-q)/(1-p))
    reverse = p * torch.log(p/q) + (1-p) * torch.log((1-p)/(1-q))
    torch.testing.assert_close(rows, expected)
    assert not torch.allclose(rows, reverse)


@pytest.mark.parametrize("balanced", [False, True])
def test_reduction_and_gradient_use_explicit_global_class_weights(balanced):
    s = torch.tensor([-1., -.4, .2, 1.], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([-2., -.1, -1., .3], dtype=torch.float64)
    y = torch.tensor([0, 0, 0, 1])
    rows = raw_bernoulli_kl_rows(s, teacher_logits=t, temperature=2.)
    result = raw_bernoulli_kl(s, y, teacher_logits=t, temperature=2., class_balanced=balanced, coefficient=.3)
    weights = torch.tensor([1/6, 1/6, 1/6, .5] if balanced else [.25]*4, dtype=torch.float64)
    oracle = .3 * (rows * weights).sum()
    torch.testing.assert_close(result["soft"], oracle)
    torch.testing.assert_close(torch.autograd.grad(result["soft"], s, retain_graph=True)[0],
                               torch.autograd.grad(oracle, s)[0])
    stats = result["stats"]
    assert stats["normal_total_weight_before_coefficient"] == (.5 if balanced else .75)
    assert stats["attack_total_weight_before_coefficient"] == (.5 if balanced else .25)
    assert stats["hard_positive_weight_applied"] is False


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 50])
def test_rows_chunking_recovers_full_population_value_and_gradient(balanced, chunk_size):
    s = torch.tensor([-2., -.3, -.4, .2, 1., -.8, .9], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([-1., -2., .4, -.8, .3, -.5, 1.2], dtype=torch.float64)
    y = torch.tensor([0, 0, 0, 0, 1, 0, 1])
    full = raw_bernoulli_kl(s, y, teacher_logits=t, class_balanced=balanced, coefficient=.3)["soft"]
    weights = torch.where(y == 0, .5 / (y == 0).sum(), .5 / (y == 1).sum()) if balanced else torch.ones(7)/7
    weights = weights.to(torch.float64)
    chunks = [(.3 * weights[i:i+chunk_size] * raw_bernoulli_kl_rows(
        s[i:i+chunk_size], teacher_logits=t[i:i+chunk_size])).sum() for i in range(0, 7, chunk_size)]
    assembled = torch.stack(chunks).sum()
    torch.testing.assert_close(assembled, full)
    torch.testing.assert_close(torch.autograd.grad(assembled, s, retain_graph=True)[0],
                               torch.autograd.grad(full, s)[0])


def test_clip_plateau_centering_reverses_normal_gradient_vs_raw_kd():
    # Same very-low normal teacher score: raw KD pushes S DOWN, while alarm
    # centering creates q=.5 and pushes an already-negative margin UP.
    t = torch.tensor([math.log(1e-6/(1-1e-6))], dtype=torch.float64)
    s = torch.tensor([-8.], dtype=torch.float64, requires_grad=True)
    raw = raw_bernoulli_kl_rows(s, teacher_logits=t).sum()
    raw_grad = torch.autograd.grad(raw, s)[0]
    alarm = F.binary_cross_entropy_with_logits(s - (-4.), torch.sigmoid(t-t))
    alarm_grad = torch.autograd.grad(alarm, s)[0]
    assert raw_grad.item() > 0  # descent lowers normal score
    assert alarm_grad.item() < 0  # descent raises normal score


def test_bce_entropy_offset_changes_value_not_student_gradient():
    s = torch.tensor([-8., -.4, .7], dtype=torch.float64, requires_grad=True)
    t = torch.zeros(3, dtype=torch.float64)
    kl = raw_bernoulli_kl_rows(s, teacher_logits=t).mean()
    bce = F.binary_cross_entropy_with_logits(s, torch.full_like(s, .5))
    assert float((bce-kl).detach()) == pytest.approx(math.log(2), abs=1e-14)
    torch.testing.assert_close(torch.autograd.grad(bce, s)[0], torch.autograd.grad(kl, s)[0])


def test_large_saturated_rank_loss_does_not_imply_large_gradient():
    z = torch.tensor([-20.], dtype=torch.float64, requires_grad=True)
    rank = 3 * torch.sigmoid((z-(-4.))/.1)
    loss = .3 * (rank-1.5).square().mean()
    grad = torch.autograd.grad(loss, z)[0]
    assert float(loss.detach()) == pytest.approx(.675)
    assert abs(grad.item()) < 1e-60


def test_teacher_labels_and_scalar_hyperparameters_are_detached():
    s = torch.tensor([-.3, .7], requires_grad=True)
    t = torch.tensor([-.1, 1.4], requires_grad=True)
    y = torch.tensor([0., 1.], requires_grad=True)
    beta, temp, shift = [torch.tensor(v, requires_grad=True) for v in [.3, 2., .1]]
    result = raw_bernoulli_kl(s, y, teacher_logits=t, coefficient=beta, temperature=temp, teacher_logit_shift=shift)
    grads = torch.autograd.grad(result["soft"], (s, t, y, beta, temp, shift), allow_unused=True)
    assert grads[0] is not None and torch.isfinite(grads[0]).all()
    assert all(g is None for g in grads[1:])
    assert result["soft"].dtype == torch.float64


@pytest.mark.parametrize("value", [-1000., 0., 1000.])
def test_equal_raw_logits_have_zero_kl_and_gradient_even_at_extremes(value):
    s = torch.tensor([value], dtype=torch.float64, requires_grad=True)
    rows = raw_bernoulli_kl_rows(s, teacher_logits=s.detach().clone())
    assert rows.item() == 0.
    assert torch.autograd.grad(rows.sum(), s)[0].item() == 0.


def test_opposite_extreme_logits_are_finite_and_zero_coefficient_has_zero_gradient():
    s = torch.tensor([-1000., 1000.], requires_grad=True)
    t = -s.detach()
    rows = raw_bernoulli_kl_rows(s, teacher_logits=t)
    torch.testing.assert_close(rows, torch.tensor([1000., 1000.], dtype=torch.float64))
    result = raw_bernoulli_kl(s, torch.tensor([0, 1]), teacher_logits=t, coefficient=0.)
    assert result["soft"].item() == 0.
    torch.testing.assert_close(torch.autograd.grad(result["soft"], s)[0], torch.zeros_like(s))


def test_balanced_requires_both_classes_but_row_mean_accepts_one():
    s = torch.tensor([-.2, .3])
    y = torch.zeros(2)
    with pytest.raises(ValueError, match="requires both classes"):
        raw_bernoulli_kl(s, y, teacher_logits=s, class_balanced=True)
    result = raw_bernoulli_kl(s, y, teacher_logits=s)
    assert result["stats"]["attack_rows"] == 0
    assert result["stats"]["attack_mean_t2_kl"] is None
    assert result["stats"]["attack_total_weight_before_coefficient"] == 0.


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"temperature": -1}, {"temperature": float("inf")}, {"temperature": 1e-200},
    {"temperature": 1e200}, {"temperature": True}, {"teacher_logit_shift": float("nan")},
    {"teacher_logit_shift": torch.ones(1)}, {"partition": "validation"},
])
def test_rows_reject_invalid_fixed_parameters(kwargs):
    with pytest.raises(ValueError):
        raw_bernoulli_kl_rows(torch.tensor([0.]), teacher_logits=torch.tensor([0.]), **kwargs)


@pytest.mark.parametrize("student,teacher", [
    (torch.tensor([]), torch.tensor([])), (torch.tensor([[0.]]), torch.tensor([[0.]])),
    (torch.tensor([0]), torch.tensor([0.])), (torch.tensor([float("nan")]), torch.tensor([0.])),
    (torch.tensor([0.]), torch.tensor([float("inf")])), (torch.tensor([0.]), torch.tensor([0., 1.])),
])
def test_rows_reject_invalid_vectors(student, teacher):
    with pytest.raises(ValueError):
        raw_bernoulli_kl_rows(student, teacher_logits=teacher)


@pytest.mark.parametrize("labels", [torch.tensor([.5, 1.]), torch.tensor([0]), torch.tensor([False, True]),
                                    torch.tensor([float("nan"), 1.]), torch.tensor([[0, 1]])])
def test_reduction_rejects_invalid_labels(labels):
    with pytest.raises(ValueError, match="labels"):
        raw_bernoulli_kl(torch.tensor([0., 1.]), labels, teacher_logits=torch.tensor([0., 1.]))


@pytest.mark.parametrize("kwargs", [{"coefficient": -1}, {"coefficient": float("inf")},
                                    {"coefficient": True}, {"class_balanced": 1}])
def test_reduction_rejects_invalid_weights(kwargs):
    with pytest.raises(ValueError):
        raw_bernoulli_kl(torch.tensor([0., 1.]), torch.tensor([0, 1]),
                         teacher_logits=torch.tensor([0., 1.]), **kwargs)
