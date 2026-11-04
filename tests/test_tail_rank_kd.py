"""Synthetic falsification checks for the rebuilt transfer object's semantics."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ids.tail_rank_kd import (  # noqa: E402
    binary_channel_rank_rows, sample_same_class_pairs, tail_rank_kd, tail_rank_kd_rows,
)


def test_pair_gradient_moves_both_examples_and_does_not_imitate_class_probability():
    left = torch.tensor([-10., -10., -10.], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([-12., -12., -12.], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([1., 0., .5], dtype=torch.float64, requires_grad=True)
    rows = tail_rank_kd_rows(left, right, target, temperature=2.)
    left_grad, right_grad, target_grad = torch.autograd.grad(rows.sum(), (left, right, target), allow_unused=True)
    expected = (torch.sigmoid(torch.ones(3))-target.detach())/2
    torch.testing.assert_close(left_grad, expected)
    torch.testing.assert_close(right_grad, -expected)
    assert target_grad is None
    assert left_grad[0] < 0 < left_grad[1]  # Order, independent of both negative class scores.


def test_common_shift_and_reversed_endpoints_preserve_pair_loss():
    a, b, target = torch.tensor([-.3, 4.]), torch.tensor([1.3, -.7]), torch.tensor([.5, 1.])
    row = tail_rank_kd_rows(a, b, target)
    torch.testing.assert_close(row, tail_rank_kd_rows(a+100, b+100, target), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(row, tail_rank_kd_rows(b, a, 1-target))


def test_sampler_uses_full_pool_excludes_self_crossclass_and_keeps_global_weights():
    teacher = np.arange(12, dtype=float)
    y = np.array([0]*8+[1]*4)
    pairs = sample_same_class_pairs(teacher, y, reference_normal_scores=np.arange(8.),
                                    max_fpr=.1, tail_multiplier=2, pairs_per_class=73)
    assert len(pairs.left) == 146
    assert (pairs.left != pairs.right).all()
    assert (y[pairs.left] == y[pairs.right]).all()
    assert (teacher[pairs.left] >= pairs.stats["tail_threshold"]).all()
    assert (teacher[pairs.right] < pairs.stats["tail_threshold"]).any()  # Partner uses the full class pool.
    assert pairs.weights[y[pairs.left] == 0].sum() == pytest.approx(.5)
    assert pairs.weights[y[pairs.left] == 1].sum() == pytest.approx(.5)


def test_native_monotone_teacher_transform_preserves_exact_pairs_orders_and_interest():
    teacher = np.array([-200., -150., -100., 0., 3., 4., 10., 12.])
    reference = np.array([-190., -175., -150., -100., -80., -30.])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    kwargs = dict(max_fpr=.1, tail_multiplier=5, pairs_per_class=1000, seed=43)
    a = sample_same_class_pairs(teacher, y, reference_normal_scores=reference, **kwargs)
    def transform(x):
        return np.arctan(x)+9.
    b = sample_same_class_pairs(transform(teacher), y, reference_normal_scores=transform(reference), **kwargs)
    for name in ("left", "right", "targets", "weights"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_uniform_control_samples_low_teacher_rows_even_when_local_interest_cannot():
    teacher = np.arange(20.)
    y = np.array([0]*10+[1]*10)
    kwargs = dict(reference_normal_scores=np.array([15., 17., 19.]), pairs_per_class=1000, seed=17)
    local = sample_same_class_pairs(teacher, y, **kwargs)
    uniform = sample_same_class_pairs(teacher, y, local=False, **kwargs)
    assert local.stats["classes"]["0"]["pairs"] == 0
    assert uniform.stats["classes"]["0"]["pairs"] == 1000
    assert uniform.stats["tail_threshold"] is None
    assert uniform.stats["tail_interest_fraction"] is None


def test_no_active_class_is_zero_not_fabricated_or_renormalized():
    teacher = np.array([-3., -2., 3., 4.])
    pairs = sample_same_class_pairs(teacher, [0, 0, 1, 1], reference_normal_scores=[2., 3., 3.5],
                                    pairs_per_class=5)
    assert pairs.stats["classes"]["0"]["active_rows"] == 0
    assert len(pairs.left) == 5
    assert pairs.weights.sum() == pytest.approx(.5)
    student = torch.tensor(teacher, requires_grad=True)
    loss = tail_rank_kd(student, pairs)
    gradient = torch.autograd.grad(loss, student)[0]
    assert gradient[:2].abs().sum() == 0


def test_no_active_pairs_returns_graph_connected_zero():
    student = torch.tensor([-3., -2., 3., 4.], requires_grad=True)
    pairs = sample_same_class_pairs(student.detach().numpy(), [0, 0, 1, 1],
                                    reference_normal_scores=[10., 11.], pairs_per_class=5)
    loss = tail_rank_kd(student, pairs)
    assert loss.item() == 0
    torch.testing.assert_close(torch.autograd.grad(loss, student)[0], torch.zeros_like(student))


def test_teacher_ties_are_not_artificially_ordered_by_identity():
    teacher = np.array([0., 0., 5., 5.])
    pairs = sample_same_class_pairs(teacher, [0, 0, 1, 1], local=False, pairs_per_class=7)
    np.testing.assert_array_equal(pairs.targets, np.full(14, .5))
    for label in ("0", "1"):
        assert pairs.stats["classes"][label]["target_half"] == 7
    s = torch.zeros(4, requires_grad=True)
    assert tail_rank_kd(s, pairs).item() == pytest.approx(math.log(2))
    torch.testing.assert_close(torch.autograd.grad(tail_rank_kd(s, pairs), s)[0], torch.zeros(4))


def test_within_class_shuffle_preserves_tail_support_but_breaks_instance_knowledge():
    teacher = np.arange(40., dtype=float)
    labels = np.array([0]*20+[1]*20)
    args = dict(reference_normal_scores=np.arange(20.), max_fpr=.1, tail_multiplier=5,
                pairs_per_class=1000, seed=29)
    regular = sample_same_class_pairs(teacher, labels, **args)
    shuffled = sample_same_class_pairs(teacher, labels, shuffle_within_class=True, **args)
    for label in ("0", "1"):
        assert regular.stats["classes"][label]["active_rows"] == shuffled.stats["classes"][label]["active_rows"]
    # With uniform pairs, the independent RNG streams keep identical sampled
    # indices. Only their teacher target relationship is destroyed.
    uniform = sample_same_class_pairs(teacher, labels, local=False, **args)
    shuffled_uniform = sample_same_class_pairs(teacher, labels, local=False, shuffle_within_class=True, **args)
    np.testing.assert_array_equal(uniform.left, shuffled_uniform.left)
    np.testing.assert_array_equal(uniform.right, shuffled_uniform.right)
    assert (uniform.targets != shuffled_uniform.targets).sum() > 500


def test_chunked_pair_reduction_preserves_global_value_and_gradient():
    pairs = sample_same_class_pairs(np.arange(9.), [0]*6+[1]*3, local=False, pairs_per_class=19)
    s = torch.linspace(-2, 3, 9, requires_grad=True)
    full = tail_rank_kd(s, pairs, temperature=2.)
    chunks = []
    for start in range(0, len(pairs.left), 7):
        i, j = pairs.left[start:start+7], pairs.right[start:start+7]
        rows = tail_rank_kd_rows(s[torch.tensor(i.copy())], s[torch.tensor(j.copy())],
                                 torch.tensor(pairs.targets[start:start+7].copy()), temperature=2.)
        chunks.append((rows*torch.tensor(pairs.weights[start:start+7].copy())).sum())
    assembled = torch.stack(chunks).sum()
    torch.testing.assert_close(full, assembled)
    torch.testing.assert_close(torch.autograd.grad(full, s, retain_graph=True)[0],
                               torch.autograd.grad(assembled, s)[0])


def test_binary_rankloss_matches_literal_two_channel_author_expression():
    s = torch.tensor([-2., .7, 0., 1e-7], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([3., .9, -.7, .1], dtype=torch.float64, requires_grad=True)
    channels_s, channels_t = torch.stack((-s/2, s/2), 1), torch.stack((-t/2, t/2), 1)
    channels_s = (channels_s-channels_s.mean(1, keepdim=True))/(channels_s.std(1, keepdim=True)+1e-7)
    channels_t = (channels_t-channels_t.mean(1, keepdim=True))/(channels_t.std(1, keepdim=True)+1e-7)
    ds, dt = channels_s[:, 1]-channels_s[:, 0], channels_t[:, 1]-channels_t[:, 0]
    expected = -(1-2/(1+torch.exp(ds)))*(1-2/(1+torch.exp(dt)))/.0125
    actual = binary_channel_rank_rows(s, teacher_logits=t)
    torch.testing.assert_close(actual, expected)
    assert torch.autograd.grad(actual.sum(), t, allow_unused=True)[0] is None


def test_binary_channel_ranking_degenerates_while_cross_example_order_does_not():
    # Both teachers classify every row as attack but reverse which attack is
    # harder. Binary standardized channel order has essentially no knowledge
    # distinguishing these teachers; same-class cross-example order does.
    s = torch.tensor([2., 3.], dtype=torch.float64)
    first, reverse = torch.tensor([1., 20.], dtype=torch.float64), torch.tensor([20., 1.], dtype=torch.float64)
    a = binary_channel_rank_rows(s, teacher_logits=first)
    b = binary_channel_rank_rows(s, teacher_logits=reverse)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-6)
    pairs_a = sample_same_class_pairs(first.numpy(), [1, 1], local=False, pairs_per_class=10)
    pairs_b = sample_same_class_pairs(reverse.numpy(), [1, 1], local=False, pairs_per_class=10)
    np.testing.assert_array_equal(pairs_a.targets, 1-pairs_b.targets)


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"temperature": float("inf")}])
def test_pair_loss_rejects_invalid_scale(kwargs):
    with pytest.raises(ValueError):
        tail_rank_kd_rows(torch.zeros(1), torch.zeros(1), torch.ones(1), **kwargs)


@pytest.mark.parametrize("kwargs", [{"max_fpr": .3}, {"pairs_per_class": 0}, {"local": 1},
                                    {"seed": -1}, {"tail_multiplier": float("nan")}])
def test_sampler_rejects_invalid_population_parameters(kwargs):
    with pytest.raises(ValueError):
        sample_same_class_pairs([0., 1., 2.], [0, 0, 1], reference_normal_scores=[0., 1.], **kwargs)
