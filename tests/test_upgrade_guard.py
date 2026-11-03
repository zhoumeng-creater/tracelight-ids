"""Synthetic finite-population acceptance and optimizer transaction checks."""
import copy

import pytest
import torch

from ids.upgrade_guard import guarded_adamw_step


def _network(value=0., *, dtype=torch.float64):
    network = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
    with torch.no_grad():
        network.weight.fill_(value)
    return network


def _report(network, ceiling):
    value = float(network.weight.detach().item())
    return {"feasible": value <= ceiling, "metrics": {"weight": value, "ceiling": ceiling}}


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for one, two in zip(left, right):
            _assert_equal(one, two)
    else:
        assert left == right


def _snapshot(network, optimizer):
    return {"model": copy.deepcopy(network.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()),
            "gradients": [None if p.grad is None else p.grad.clone() for p in network.parameters()],
            "rng": torch.random.get_rng_state().clone(), "modes": [m.training for m in network.modules()]}


def test_full_proposal_matches_unmodified_adamw_bit_for_bit():
    network = _network(.15, dtype=torch.float32)
    expected = copy.deepcopy(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.03, weight_decay=.1)
    ordinary = torch.optim.AdamW(expected.parameters(), lr=.03, weight_decay=.1)
    for step in range(3):
        gradient = torch.full_like(network.weight, -(step + .123))
        network.weight.grad = gradient.clone()
        expected.weight.grad = gradient.clone()
        ordinary.step()
        result = guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 10.))
        assert result["accepted"] and result["fraction"] == 1. and result["backtracks"] == 0
        assert result["evaluations"] == 2
        _assert_equal(network.state_dict(), expected.state_dict())
        _assert_equal(optimizer.state_dict(), ordinary.state_dict())
        assert torch.equal(network.weight.grad, gradient)


def test_backtrack_shortens_real_delta_keeps_full_proposal_moments_and_counter():
    network = _network(0.)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1., weight_decay=0., eps=0.)
    network.weight.grad = torch.full_like(network.weight, -1.)
    result = guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, .3))
    assert result["accepted"] and result["fraction"] == .25 and result["backtracks"] == 2
    assert result["evaluations"] == 4
    assert float(network.weight) == pytest.approx(.25)
    assert result["proposal"]["metrics"]["weight"] == pytest.approx(1.)
    assert result["before"]["metrics"]["weight"] == 0.
    assert result["parameter_delta_l2"] == pytest.approx(.25)
    state = optimizer.state[network.weight]
    assert state["step"].item() == 1
    assert state["exp_avg"].item() == pytest.approx(-.1)
    assert state["exp_avg_sq"].item() == pytest.approx(.001)
    assert "full_proposal" in result["state_semantics"]


def test_boundary_with_no_positive_feasible_step_rejects_exactly():
    network = _network(0.)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1., weight_decay=.1)
    network.weight.grad = -torch.ones_like(network.weight)
    before = _snapshot(network, optimizer)
    result = guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 0.), max_backtracks=3)
    assert not result["accepted"] and result["fraction"] == 0.
    assert result["backtracks"] == 3 and result["evaluations"] == 5
    assert result["after"] == result["before"]
    assert result["parameter_delta_l2"] == 0. and not result["moved"]
    _assert_equal(_snapshot(network, optimizer), before)


def test_rejected_populated_momentum_state_matches_no_proposal_continuation():
    network = _network(-.2)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1, weight_decay=.01)
    network.weight.grad = torch.ones_like(network.weight)
    optimizer.step()
    network.weight.grad = -torch.ones_like(network.weight)
    twin = copy.deepcopy(network)
    twin_optimizer = torch.optim.AdamW(twin.parameters(), lr=.1, weight_decay=.01)
    twin_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    before = _snapshot(network, optimizer)
    initial = float(network.weight)

    def keep_only_original():
        value = float(network.weight.detach())
        return {"feasible": value == initial, "metrics": {"weight": value}}

    result = guarded_adamw_step(network, optimizer, evaluate=keep_only_original, max_backtracks=4)
    assert not result["accepted"]
    _assert_equal(_snapshot(network, optimizer), before)
    # An ordinary next step now takes the same path as if the proposal never happened.
    network.weight.grad = torch.full_like(network.weight, .27)
    twin.weight.grad = torch.full_like(twin.weight, .27)
    optimizer.step()
    twin_optimizer.step()
    _assert_equal(network.state_dict(), twin.state_dict())
    _assert_equal(optimizer.state_dict(), twin_optimizer.state_dict())


def test_callback_rng_and_modes_do_not_escape_accept_or_reject():
    for ceiling in (0., 2.):
        network = _network(0.)
        network.train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=1., weight_decay=0.)
        network.weight.grad = -torch.ones_like(network.weight)
        rng = torch.random.get_rng_state().clone()

        def evaluator():
            network.eval()
            torch.rand(7)
            return _report(network, ceiling)

        result = guarded_adamw_step(network, optimizer, evaluate=evaluator, max_backtracks=2)
        assert result["accepted"] == (ceiling == 2.)
        assert network.training
        assert torch.equal(rng, torch.random.get_rng_state())


def test_infeasible_start_is_rejected_without_any_optimizer_step():
    network = _network(.1)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1.)
    network.weight.grad = -torch.ones_like(network.weight)
    before = _snapshot(network, optimizer)
    with pytest.raises(ValueError, match="feasible starting"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 0.))
    _assert_equal(_snapshot(network, optimizer), before)


@pytest.mark.parametrize("failure_call", [1, 2, 3])
def test_callback_exception_at_each_stage_rolls_back_full_transaction(failure_call):
    network = _network(0.)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1., weight_decay=.1)
    network.weight.grad = -torch.ones_like(network.weight)
    before = _snapshot(network, optimizer)
    calls = []

    def evaluator():
        calls.append(1)
        torch.rand(4)
        network.eval()
        if len(calls) == failure_call:
            raise RuntimeError("deliberate synthetic callback failure")
        return _report(network, .3)

    with pytest.raises(RuntimeError, match="synthetic callback"):
        guarded_adamw_step(network, optimizer, evaluate=evaluator)
    _assert_equal(_snapshot(network, optimizer), before)


@pytest.mark.parametrize("bad", [
    {"feasible": True}, {"feasible": 1, "metrics": {}},
    {"feasible": True, "metrics": {"fpr": float("nan")}},
    {"feasible": True, "metrics": {"fpr": torch.tensor(.1)}},
])
def test_invalid_callback_reports_fail_closed(bad):
    network = _network(0.)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    network.weight.grad = torch.ones_like(network.weight)
    before = _snapshot(network, optimizer)
    with pytest.raises(ValueError, match="evaluate|evaluation"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: bad)
    _assert_equal(_snapshot(network, optimizer), before)


@pytest.mark.parametrize("mutation", ["parameter", "gradient", "buffer", "learning_rate"])
def test_mutating_evaluator_is_detected_and_restored(mutation):
    network = _network(0.)
    network.register_buffer("audit_buffer", torch.tensor([3.]))
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    network.weight.grad = torch.ones_like(network.weight)
    before = _snapshot(network, optimizer)

    def evaluator():
        if mutation == "parameter":
            network.weight.add_(1.)
        elif mutation == "gradient":
            network.weight.grad.add_(1.)
        elif mutation == "buffer":
            network.audit_buffer.add_(1.)
        else:
            optimizer.param_groups[0]["lr"] *= 2
        return {"feasible": True, "metrics": {}}

    with pytest.raises(ValueError, match="must not mutate"):
        guarded_adamw_step(network, optimizer, evaluate=evaluator)
    _assert_equal(_snapshot(network, optimizer), before)


@pytest.mark.parametrize("kwargs", [
    {"max_backtracks": True}, {"max_backtracks": -1}, {"max_backtracks": .5},
    {"shrink": True}, {"shrink": 0.}, {"shrink": 1.}, {"shrink": float("nan")},
])
def test_invalid_backtracking_configuration(kwargs):
    network = _network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    network.weight.grad = torch.ones_like(network.weight)
    with pytest.raises(ValueError):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 1.), **kwargs)


def test_optimizer_cannot_update_an_unguarded_parameter():
    network, other = _network(), _network()
    optimizer = torch.optim.AdamW(other.parameters(), lr=.1)
    other.weight.grad = torch.ones_like(other.weight)
    with pytest.raises(ValueError, match="members"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 1.))


def test_missing_and_nonfinite_gradients_are_rejected_before_mutation():
    network = _network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    with pytest.raises(ValueError, match="accumulated gradients"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 1.))
    network.weight.grad = torch.full_like(network.weight, float("nan"))
    with pytest.raises(ValueError, match="gradients must be finite"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 1.))
    assert not optimizer.state


def test_zero_backtracks_evaluates_only_full_proposal():
    network = _network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1., weight_decay=0.)
    network.weight.grad = -torch.ones_like(network.weight)
    result = guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, .5), max_backtracks=0)
    assert not result["accepted"] and result["evaluations"] == 2 and result["backtracks"] == 0
    assert float(network.weight) == 0.


def test_optimizer_step_exception_restores_partial_parameter_and_state_changes(monkeypatch):
    network = _network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    network.weight.grad = -torch.ones_like(network.weight)
    original_step = optimizer.step
    before = _snapshot(network, optimizer)

    def broken_step():
        original_step()
        torch.rand(4)
        raise RuntimeError("synthetic optimizer failure after update")

    monkeypatch.setattr(optimizer, "step", broken_step)
    with pytest.raises(RuntimeError, match="optimizer failure"):
        guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 1.))
    _assert_equal(_snapshot(network, optimizer), before)


def test_actual_adam_momentum_displacement_is_checked_not_current_gradient_sign():
    network = _network(-.05)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1, weight_decay=0., betas=(.99, .999))
    # Seed a negative first moment. The current positive gradient points inward,
    # but AdamW's actual update still points outwards due to its momentum.
    network.weight.grad = torch.full_like(network.weight, -10.)
    optimizer.step()
    with torch.no_grad():
        network.weight.fill_(-.05)
    network.weight.grad = torch.ones_like(network.weight)
    result = guarded_adamw_step(network, optimizer, evaluate=lambda: _report(network, 0.), max_backtracks=8)
    assert result["proposal"]["metrics"]["weight"] > 0.
    assert result["accepted"] and result["fraction"] < 1.
    assert float(network.weight) <= 0.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable for synthetic RNG/state check")
def test_cuda_transaction_restores_rng_and_optimizer_state_on_reject():
    network = _network().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.1)
    network.weight.grad = -torch.ones_like(network.weight)
    initial = float(network.weight)
    rng = [value.clone() for value in torch.cuda.get_rng_state_all()]
    before = _snapshot(network, optimizer)

    def evaluator():
        torch.rand(3, device="cuda")
        return _report(network, initial)

    result = guarded_adamw_step(network, optimizer, evaluate=evaluator, max_backtracks=2)
    assert not result["accepted"]
    _assert_equal(_snapshot(network, optimizer), before)
    assert all(torch.equal(left, right) for left, right in zip(rng, torch.cuda.get_rng_state_all()))
