"""Only fabricated arrays and freshly initialized tiny official CPU TabM."""
import copy
from itertools import product

import pytest
import torch

from ids.normal_subgroup_training.losses import (
    BatchTargets,
    auxiliary_losses,
    member_bce,
    training_step,
)
from ids.route_batch.tabm_training import stable_scores


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def labels(b=1, dtype=torch.float64):
    return torch.tensor([0, 1, 0, 1], dtype=dtype).repeat_interleave(b)


def logits_from_rows(values):
    return torch.tensor(values, dtype=torch.float64)[:, None, None].expand(-1, 32, 1).clone()


def anchors(b=1, *, normal=None, positive=None, sizes=None, counts=(2, 3), groups=2, scale=1.):
    return BatchTargets(
        normal_targets=[-1.] * (2 * b) if normal is None else normal,
        positive_reference_scores=[0.] * (2 * b) if positive is None else positive,
        normal_group_sizes=[2] * b + [3] * b if sizes is None else sizes,
        profile_normal_counts=counts,
        normal_group_count=groups,
        scale=scale,
        draws_per_group=b,
    )


def test_k32_member_loss_value_gradient_and_no_member_coupling():
    values = torch.linspace(-4., 3., 4 * 32, dtype=torch.float64).reshape(4, 32, 1).requires_grad_()
    loss = member_bce(values, labels(), draws_per_group=1)
    expected = sum(torch.nn.functional.binary_cross_entropy_with_logits(
        values[:, k, 0], labels()) for k in range(32)) / 32
    torch.testing.assert_close(loss, expected)
    grad = torch.autograd.grad(loss, values)[0]
    torch.testing.assert_close(grad[:, :, 0], (values.detach()[:, :, 0].sigmoid() - labels()[:, None]) / 128)
    altered = values.detach().clone()
    altered[0, 31, 0] += 2.
    altered.requires_grad_()
    altered_grad = torch.autograd.grad(member_bce(altered, labels(), draws_per_group=1), altered)[0]
    assert grad[0, 0, 0] == altered_grad[0, 0, 0]


def test_actual_mean_probability_score_and_auxiliary_member_coupling():
    values = torch.linspace(-4., 2.3, 4 * 32, dtype=torch.float64).reshape(4, 32, 1).requires_grad_()
    prob = values.sigmoid().mean(dim=1).squeeze(-1)
    torch.testing.assert_close(stable_scores(values), torch.logit(prob))
    assert not torch.allclose(stable_scores(values), values.mean(dim=1).squeeze(-1))
    target = anchors()
    first = torch.autograd.grad(auxiliary_losses(values, target).normal, values)[0]
    altered = values.detach().clone()
    altered[0, 31, 0] += 2.
    altered.requires_grad_()
    second = torch.autograd.grad(auxiliary_losses(altered, target).normal, altered)[0]
    assert not torch.isclose(first[0, 0, 0], second[0, 0, 0])
    assert auxiliary_losses(values.float(), target).total.dtype == torch.float64


def test_batch64_exact_weights_are_summed_not_divided_again():
    values = torch.zeros((256, 32, 1), dtype=torch.float32, requires_grad=True)
    target = anchors(64, normal=[-1.] * 128, positive=[2.] * 64 + [4.] * 64,
                     sizes=[400] * 64 + [200] * 64, counts=(400, 200))
    result = auxiliary_losses(values, target)
    assert result.normal.item() == pytest.approx(1.)
    assert result.positive.item() == pytest.approx(10.)  # 0.5*(2**2 + 4**2)
    assert result.total.item() == pytest.approx(11.)


def test_enumerated_profile_sampling_equals_group_objective_with_cross_profile_groups():
    # Group 0 spans A and B: global counts 2 and 3, not batch counts.
    a = [(-2., 0), (1., 1)]
    b = [(-1., 0), (2., 1), (3., 1)]
    all_rows = a + b
    target_value, scale = -3., 2.
    expected = sum(
        sum(((value - target_value) / scale) ** 2 for value, group in all_rows if group == g)
        / sum(group == g for _, group in all_rows)
        for g in (0, 1)
    ) / 2
    estimates = []
    for left, right in product(a, b):
        values = logits_from_rows([left[0], 0., right[0], 0.])
        target = anchors(normal=[target_value] * 2, sizes=[(2, 3)[left[1]], (2, 3)[right[1]]],
                         counts=(len(a), len(b)), scale=scale)
        estimates.append(auxiliary_losses(values, target).normal.item())
    assert sum(estimates) / len(estimates) == pytest.approx(expected)


def test_transport_starts_with_nonzero_normal_signal_positive_anchor_zero_gradient():
    values = logits_from_rows([-2., 2., -1., 3.]).requires_grad_()
    original = stable_scores(values).detach()
    target = anchors(normal=original[[0, 2]] - 1, positive=original[[1, 3]])
    result = auxiliary_losses(values, target)
    assert result.normal.item() == pytest.approx(1.)
    assert result.positive.item() == 0.
    positive_grad = torch.autograd.grad(result.positive, values, retain_graph=True)[0]
    assert torch.count_nonzero(positive_grad) == 0
    gradient = torch.autograd.grad(result.normal, values)[0]
    assert (gradient[[0, 2]] > 0).all()  # Gradient descent lowers these normals.
    assert torch.count_nonzero(gradient[[1, 3]]) == 0
    identity = anchors(normal=original[[0, 2]], positive=original[[1, 3]])
    identity_terms = auxiliary_losses(values, identity)
    assert identity_terms.total.item() == 0.
    assert torch.count_nonzero(torch.autograd.grad(identity_terms.total, values)[0]) == 0


def test_two_sided_normal_anchor_and_one_sided_positive_anchor_have_correct_directions():
    values = logits_from_rows([-2., 0., 2., 4.]).requires_grad_()
    target = anchors(normal=[-1., 1.], positive=[1., 3.], scale=2.)
    result = auxiliary_losses(values, target)
    gradient = torch.autograd.grad(result.total, values)[0]
    assert (gradient[0] < 0).all()  # A normal below its target is pulled back up.
    assert (gradient[2] > 0).all()
    assert (gradient[1] < 0).all()  # Only the lowered positive gets protection force.
    assert torch.count_nonzero(gradient[3]) == 0
    wider = anchors(normal=[-1., 1.], positive=[1., 3.], scale=4.)
    torch.testing.assert_close(auxiliary_losses(values, wider).total, result.total / 4)
    assert target.scale == 2.  # Never inferred again from current scores.


def test_auxiliary_gradcheck_k32_and_saturated_scores():
    values = torch.linspace(-.4, .7, 128, dtype=torch.float64).reshape(4, 32, 1).requires_grad_()
    target = anchors(normal=[-2., -2.], positive=[2., 2.], scale=1.7)
    assert torch.autograd.gradcheck(lambda x: auxiliary_losses(x, target).total,
                                   (values,), eps=1e-6, atol=1e-5, rtol=1e-4)
    saturated = logits_from_rows([-1000., 1000., -800., 800.]).requires_grad_()
    extreme_targets = anchors(normal=[-1001., -801.], positive=[1001., 801.])
    result = auxiliary_losses(saturated, extreme_targets)
    assert torch.isfinite(result.total)
    gradient = torch.autograd.grad(result.total, saturated)[0]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) == gradient.numel()


def test_anchor_arrays_are_copied_and_differentiable_or_invalid_anchors_rejected():
    exact_values = [-1.23456789012345, -2.34567890123456]
    torch.testing.assert_close(anchors(normal=exact_values).normal_targets,
                               torch.tensor(exact_values, dtype=torch.float64), rtol=0, atol=0)
    source = torch.tensor([-1., -2.], dtype=torch.float64)
    target = anchors(normal=source)
    source.add_(100)
    torch.testing.assert_close(target.normal_targets, torch.tensor([-1., -2.], dtype=torch.float64))
    with pytest.raises(ValueError, match="must not require gradients"):
        anchors(normal=torch.ones(2, requires_grad=True))
    with pytest.raises(ValueError, match="frozen"):
        anchors(scale=torch.tensor(2., requires_grad=True))
    for invalid in (0., .5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale"):
            anchors(scale=invalid)
    with pytest.raises(ValueError, match="positive integer"):
        anchors(sizes=[1.5, 3])
    with pytest.raises(ValueError, match="aligned finite"):
        anchors(positive=[float("nan"), 1.])
    with pytest.raises(ValueError, match="32"):
        member_bce(torch.zeros(4, 2, 1), labels(), draws_per_group=1)
    with pytest.raises(ValueError, match="A0/A1/B0/B1"):
        member_bce(torch.zeros(4, 32, 1), labels().flip(0), draws_per_group=1)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


class ToyNetwork(torch.nn.Module):
    def __init__(self, *, random_eval=False, raise_eval=False):
        super().__init__()
        self.dropout = torch.nn.Dropout(.3)
        self.linear = torch.nn.Linear(3, 32)
        self.random_eval, self.raise_eval = random_eval, raise_eval
        self.calls = []

    def forward(self, numeric, categorical):
        self.calls.append((self.training, self.dropout.training, torch.is_grad_enabled()))
        if not self.training and self.raise_eval:
            raise RuntimeError("fabricated auxiliary failure")
        extra = torch.rand(()) if not self.training and self.random_eval else 0.
        return (self.linear(self.dropout(numeric)) + extra).unsqueeze(-1)


def make_step_fixture(network):
    numeric = torch.linspace(-1., 1., 8 * 3).reshape(8, 3)
    categorical = None
    network.eval()
    with torch.no_grad():
        base = stable_scores(network(numeric, categorical))
    target = anchors(2, normal=base[[0, 1, 4, 5]] - 100.,
                     positive=base[[2, 3, 6, 7]], sizes=[2, 2, 3, 3])
    return numeric, categorical, labels(2, dtype=torch.float32), target


def test_two_backward_sum_single_clip_single_step_and_mode_rng_restoration(monkeypatch):
    torch.manual_seed(4)
    actual = ToyNetwork()
    numeric, categorical, y, target = make_step_fixture(actual)
    actual.train()
    actual.dropout.eval()  # Preserve an initially mixed set of module modes.
    original_modes = [module.training for module in actual.modules()]
    expected = copy.deepcopy(actual)
    actual.calls.clear()
    expected.calls.clear()
    before_rng = torch.get_rng_state().clone()
    expected.train()
    bce = member_bce(expected(numeric, categorical), y, draws_per_group=2)
    expected.eval()
    auxiliary = auxiliary_losses(expected(numeric, categorical), target)
    (bce + auxiliary.total).backward()
    expected_rng = torch.get_rng_state().clone()
    expected_norm = torch.nn.utils.clip_grad_norm_(expected.parameters(), 1., error_if_nonfinite=True)
    torch.optim.SGD(expected.parameters(), lr=.03).step()
    torch.set_rng_state(before_rng)

    real_clip = torch.nn.utils.clip_grad_norm_
    clip_calls = []

    def tracked_clip(parameters, max_norm, **kwargs):
        clip_calls.append(max_norm)
        return real_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tracked_clip)
    optimizer = CountingSGD(actual.parameters(), lr=.03)
    result = training_step(actual, optimizer, numeric, categorical, y, target)
    assert optimizer.zero_calls == optimizer.step_calls == 1
    assert clip_calls == [1.]
    assert actual.calls == [(True, True, True), (False, False, True)]
    assert [module.training for module in actual.modules()] == original_modes
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not torch.equal(before_rng, expected_rng)  # BCE RNG was not rewound.
    assert result["gradient_norm_before_clip"] == pytest.approx(float(expected_norm))
    assert result["gradient_norm_before_clip"] > 1
    assert result["optimizer_steps"] == 1 and result["auxiliary_rng_unchanged"]
    assert result["total"] == pytest.approx(result["member_bce"] + result["normal"] + result["positive"])
    for a, e in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, e)
        torch.testing.assert_close(a.grad, e.grad)
    norm_after_clip = torch.linalg.vector_norm(torch.cat([p.grad.flatten() for p in actual.parameters()]))
    assert norm_after_clip <= 1.000001


@pytest.mark.parametrize("failure", ["random_eval", "raise_eval"])
def test_auxiliary_failure_restores_mode_and_post_bce_rng_without_update(failure):
    torch.manual_seed(8)
    network = ToyNetwork()
    numeric, categorical, y, target = make_step_fixture(network)
    setattr(network, failure, True)
    network.train()
    before = [p.detach().clone() for p in network.parameters()]
    modes = [m.training for m in network.modules()]
    rng = torch.get_rng_state().clone()
    reference = copy.deepcopy(network)
    reference.train()
    reference(numeric, categorical)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    optimizer = CountingSGD(network.parameters(), lr=.1)
    message = "consumed training RNG" if failure == "random_eval" else "fabricated auxiliary failure"
    with pytest.raises(RuntimeError, match=message):
        training_step(network, optimizer, numeric, categorical, y, target)
    assert optimizer.zero_calls == 1 and optimizer.step_calls == 0
    assert [m.training for m in network.modules()] == modes
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for parameter, original in zip(network.parameters(), before):
        torch.testing.assert_close(parameter, original)


def test_fresh_tiny_official_tabm_k32_ple_cpu_one_update():
    pytest.importorskip("tabm")
    pytest.importorskip("rtdl_num_embeddings")
    from ids.recent_baselines import _parameters, make_tabm_network

    torch.manual_seed(18)
    params = _parameters({"k": 32, "n_blocks": 1, "d_block": 8, "dropout": .1,
                          "embedding_dim": 4, "n_bins": 3})
    network = make_tabm_network(3, [], params,
                               numeric_bins=[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
    numeric, categorical, y, target = make_step_fixture(network)
    before = [p.detach().clone() for p in network.parameters()]
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0002, weight_decay=.0003)
    result = training_step(network, optimizer, numeric, categorical, y, target)
    assert result["normal"] > 0 and result["positive"] == 0
    assert any(not torch.equal(p, original) for p, original in zip(network.parameters(), before))
    assert all(torch.isfinite(p).all() for p in network.parameters())
    assert all(state["step"].item() == 1 for state in optimizer.state.values())
    assert not network.training  # Fixture's original eval mode was restored.
