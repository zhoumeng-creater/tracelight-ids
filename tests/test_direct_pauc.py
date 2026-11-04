"""Independent finite-pool CVaR values, subgradients, and global sampling checks."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from ids.direct_pauc import (
    direct_pauc_loss,
    global_negative_tail,
    pairwise_pauc_rows,
    sample_global_tail_pairs,
)


def test_fractional_tail_mass_and_tie_symmetry():
    # beta*N- = 1.5: the top negative gets 1, the two boundary ties get .25 each.
    scores = np.array([7., 5., 5., 2., 0., -1., 100.])
    labels = np.array([0, 0, 0, 0, 0, 0, 1])
    tail = global_negative_tail(scores, labels, beta=.25)
    np.testing.assert_allclose(tail["weights"], [2/3, 1/6, 1/6, 0, 0, 0, 0])
    assert tail["statistics"]["negative_tail_mass"] == 1.5
    assert tail["statistics"]["boundary_tie_rows"] == 2
    assert tail["statistics"]["cutoff"] == 5.


@pytest.mark.parametrize("beta", [.01, .25, .5, 1.])
def test_tied_global_tail_is_invariant_to_row_permutation(beta):
    scores = np.array([4., 4., 3., 3., 3., 0., 5., 1.])
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1])
    perm = np.array([7, 4, 2, 6, 3, 0, 5, 1])
    first = global_negative_tail(scores, labels, beta=beta)["weights"]
    second = global_negative_tail(scores[perm], labels[perm], beta=beta)["weights"]
    np.testing.assert_allclose(first[perm], second)
    assert first[labels == 1].sum() == 0.
    assert first.sum() == pytest.approx(1.)


def test_beta_one_is_full_negative_uniform_distribution():
    tail = global_negative_tail(np.array([4., -4., 1., 8.]), np.array([0, 0, 1, 0]), beta=1)
    np.testing.assert_allclose(tail["weights"], [1/3, 1/3, 0, 1/3])


@pytest.mark.parametrize("beta", [.1, .3, .5, .95, 1.])
def test_exact_loss_equals_independent_minimum_over_cvar_thresholds(beta):
    # For each positive, a CVaR minimum occurs at one of its pair losses.
    scores = torch.tensor([-1., .3, 2., 2., -.2, 1.4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    losses = F.softplus(.4 + scores[y == 0][None, :] - scores[y == 1][:, None])
    independent = []
    for row in losses:
        values = [u + torch.relu(row - u).sum() / (beta * 4) for u in row]
        independent.append(torch.stack(values).min())
    expected = torch.stack(independent).mean()
    result = direct_pauc_loss(scores, y, beta=beta, margin=.4, max_pairs_per_chunk=2)
    torch.testing.assert_close(result["loss"], expected)
    assert result["statistics"]["libauc_reproduction"] is False


def test_exact_loss_and_gradient_match_independent_full_weighted_matrix():
    scores = torch.tensor([3., 1., -2., 5., .2, 2.], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    # beta*N- = 1.5, weights 2/3 top negative and 1/3 runner-up.
    matrix = .8 * F.softplus((.2 + scores[:2][None, :] - scores[3:, None])/.8)
    expected = (matrix * torch.tensor([2/3, 1/3], dtype=torch.float64)).sum() / 3
    reference_grad, = torch.autograd.grad(expected, scores)
    result = direct_pauc_loss(scores, y, beta=.5, margin=.2, temperature=.8, max_pairs_per_chunk=1)
    grad, = torch.autograd.grad(result["loss"], scores)
    torch.testing.assert_close(result["loss"], expected)
    torch.testing.assert_close(grad, reference_grad)
    assert grad[2] == 0
    assert bool((grad[:2] > 0).all()) and bool((grad[3:] < 0).all())


def test_tied_boundary_subgradient_is_symmetric_and_permutation_equivariant():
    scores = torch.tensor([3., 1., 1., 0., .4], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 0, 0, 1])
    first = direct_pauc_loss(scores, y, beta=.375)["loss"]
    grad, = torch.autograd.grad(first, scores)
    assert grad[1] == grad[2] and grad[1] > 0
    permutation = torch.tensor([4, 2, 0, 3, 1])
    reordered = scores.detach()[permutation].requires_grad_()
    second = direct_pauc_loss(reordered, y[permutation], beta=.375)["loss"]
    second_grad, = torch.autograd.grad(second, reordered)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(grad[permutation], second_grad)


def test_global_tail_is_different_from_batch_local_top_fraction():
    # A batch-local rule spends half its negative mass on the easy [1, 0] batch.
    scores, labels = np.array([9., 8., 1., 0., 5.]), np.array([0, 0, 0, 0, 1])
    tail = global_negative_tail(scores, labels, beta=.5)["weights"]
    np.testing.assert_allclose(tail, [.5, .5, 0, 0, 0])
    local = .5 * np.logaddexp(0, 9 - 5) + .5 * np.logaddexp(0, 1 - 5)
    global_loss = direct_pauc_loss(torch.tensor(scores), torch.tensor(labels), beta=.5)["loss"]
    assert float(global_loss) > local + 1


def test_extreme_finite_pair_logits_are_stable_and_keep_float32_gradients():
    pos = torch.tensor([-10000., 10000.], requires_grad=True)
    neg = torch.tensor([10000., -10000.], requires_grad=True)
    rows = pairwise_pauc_rows(pos, neg)
    assert bool(torch.isfinite(rows).all())
    torch.testing.assert_close(rows, torch.tensor([20000., 0.], dtype=torch.float64))
    p_grad, n_grad = torch.autograd.grad(rows.sum(), (pos, neg))
    torch.testing.assert_close(p_grad, torch.tensor([-1., -0.]))
    torch.testing.assert_close(n_grad, torch.tensor([1., 0.]))


def test_chunked_pair_mean_uses_global_denominator_and_matches_gradients():
    pos = torch.tensor([1., -1., .5, 2., -.3], dtype=torch.float64, requires_grad=True)
    neg = torch.tensor([2., .3, 0., -1., 3.], dtype=torch.float64, requires_grad=True)
    whole = pairwise_pauc_rows(pos, neg).mean()
    whole_grad = torch.autograd.grad(whole, (pos, neg))
    chunks = sum(pairwise_pauc_rows(pos[i:i+2], neg[i:i+2]).sum()/len(pos) for i in range(0, len(pos), 2))
    chunk_grad = torch.autograd.grad(chunks, (pos, neg))
    torch.testing.assert_close(whole, chunks)
    for left, right in zip(whole_grad, chunk_grad, strict=True):
        torch.testing.assert_close(left, right)


def test_sampler_rng_state_can_resume_exactly_and_global_rng_is_untouched():
    scores, y = np.array([3., 2., 2., 0., 1., 4.]), np.array([0, 0, 0, 0, 1, 1])
    rng = np.random.default_rng(17)
    _ = sample_global_tail_pairs(scores, y, rng=rng, n_pairs=13, beta=.375)
    state = deepcopy(rng.bit_generator.state)
    global_state = np.random.get_state()
    expected = sample_global_tail_pairs(scores, y, rng=rng, n_pairs=53, beta=.375)
    resumed = np.random.default_rng()
    resumed.bit_generator.state = state
    actual = sample_global_tail_pairs(scores, y, rng=resumed, n_pairs=53, beta=.375)
    for key in ["positive_indices", "negative_indices"]:
        np.testing.assert_array_equal(actual[key], expected[key])
    after = np.random.get_state()
    assert global_state[0] == after[0]
    np.testing.assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]


def test_sampled_gradient_estimates_exact_global_tail_gradient():
    scores = torch.tensor([3., 2., 2., 0., 1., 4.], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    exact = direct_pauc_loss(scores, y, beta=.375)["loss"]
    exact_grad, = torch.autograd.grad(exact, scores)
    pairs = sample_global_tail_pairs(scores.detach().numpy(), y.numpy(), rng=np.random.default_rng(71),
                                     n_pairs=100000, beta=.375)
    sampled = pairwise_pauc_rows(scores[pairs["positive_indices"]], scores[pairs["negative_indices"]]).mean()
    sampled_grad, = torch.autograd.grad(sampled, scores)
    torch.testing.assert_close(sampled, exact, atol=.01, rtol=0)
    torch.testing.assert_close(sampled_grad, exact_grad, atol=.006, rtol=0)
    assert (pairs["negative_indices"] < 3).all()
    assert (pairs["positive_indices"] >= 4).all()


@pytest.mark.parametrize("partition", ["validation", "calibration", "test", "constraint", "reference"])
def test_all_apis_reject_non_fitting_roles(partition):
    s, y = np.array([1., 2.]), np.array([0, 1])
    with pytest.raises(ValueError, match="fitting role"):
        global_negative_tail(s, y, partition=partition)
    with pytest.raises(ValueError, match="fitting role"):
        sample_global_tail_pairs(s, y, rng=np.random.default_rng(1), partition=partition)
    with pytest.raises(ValueError, match="fitting role"):
        direct_pauc_loss(torch.tensor(s), torch.tensor(y), partition=partition)
    with pytest.raises(ValueError, match="fitting role"):
        pairwise_pauc_rows(torch.tensor([1.]), torch.tensor([2.]), partition=partition)


@pytest.mark.parametrize("beta", [0, -.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_beta_rejected(beta):
    with pytest.raises(ValueError, match="beta"):
        global_negative_tail(np.array([1., 2.]), np.array([0, 1]), beta=beta)


@pytest.mark.parametrize("scores,labels", [
    (np.array([1, 2]), np.array([0, 1])),
    (np.array([1., np.nan]), np.array([0, 1])),
    (np.array([[1., 2.]]), np.array([0, 1])),
    (np.array([1., 2.]), np.array([0, 2])),
    (np.array([1., 2.]), np.array([1, 1])),
    (np.array([1., 2.]), np.array([0])),
])
def test_invalid_pool_rejected(scores, labels):
    with pytest.raises(ValueError):
        global_negative_tail(scores, labels)


def test_pair_and_sampler_argument_failures_are_explicit():
    scores, y = np.array([1., 2.]), np.array([0, 1])
    with pytest.raises(ValueError, match="Generator"):
        sample_global_tail_pairs(scores, y, rng=None)
    with pytest.raises(ValueError, match="n_pairs"):
        sample_global_tail_pairs(scores, y, rng=np.random.default_rng(1), n_pairs=0)
    with pytest.raises(ValueError, match="aligned"):
        pairwise_pauc_rows(torch.tensor([1., 2.]), torch.tensor([1.]))
    with pytest.raises(ValueError, match="temperature"):
        pairwise_pauc_rows(torch.tensor([1.]), torch.tensor([1.]), temperature=0)
    with pytest.raises(ValueError, match="margin"):
        pairwise_pauc_rows(torch.tensor([1.]), torch.tensor([1.]), margin=-1)
