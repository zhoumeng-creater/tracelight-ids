"""Artificial scalar/tensor checks; no model, real scores, or dataset is loaded."""
import pytest

torch = pytest.importorskip("torch")

from ids.operating_risk_training.constrained_step import (  # noqa: E402
    full_identity_lagrangian,
    projected_dual_update,
)


def test_primal_dual_have_separate_gradient_paths():
    theta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    multiplier = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    proxy = torch.stack((theta, -theta))
    true = torch.stack((2 * theta, -3 * theta))
    terms = full_identity_lagrangian(
        theta.square(), proxy, true, positive_ids=("p1", "p2"),
        multiplier_ids=("p1", "p2"), multipliers=multiplier,
    )
    assert terms.primal.item() == 2.0
    assert terms.dual.item() == -12.0
    terms.primal.backward()
    assert theta.grad.item() == 3.0
    assert multiplier.grad is None
    terms.dual.backward()
    assert theta.grad.item() == 3.0
    torch.testing.assert_close(multiplier.grad, torch.tensor([4.0, -6.0], dtype=torch.float64))


def test_feasible_signed_residual_decreases_multiplier_and_projects():
    old = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64, requires_grad=True)
    residual = torch.tensor([2.0, -1.0, -9.0], dtype=torch.float64, requires_grad=True)
    result = projected_dual_update(old, residual, step_size=0.25)
    torch.testing.assert_close(result, torch.tensor([1.0, 0.25, 0.0], dtype=torch.float64))
    assert not result.requires_grad
    assert old.tolist() == [0.5, 0.5, 0.5]


def test_zero_initial_multiplier_does_not_protect_first_primal_update():
    theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    terms = full_identity_lagrangian(
        -theta, theta.reshape(1), theta.detach().reshape(1),
        positive_ids=("p1",), multiplier_ids=("p1",),
        multipliers=torch.zeros(1, dtype=torch.float64),
    )
    terms.primal.backward()
    assert theta.grad.item() == -1.0
    assert projected_dual_update(
        torch.zeros(1, dtype=torch.float64), theta.detach().reshape(1), step_size=0.1,
    ).item() == 0.1


@pytest.mark.parametrize("ids", [("p2", "p1"), ("p1",), ("p1", "unknown"), ("p1", "p1")])
def test_missing_reordered_or_changed_positive_identity_is_rejected(ids):
    with pytest.raises(ValueError):
        full_identity_lagrangian(
            torch.tensor(0.0, dtype=torch.float64),
            torch.zeros(len(ids), dtype=torch.float64),
            torch.zeros(len(ids), dtype=torch.float64), positive_ids=ids,
            multiplier_ids=("p1", "p2"), multipliers=torch.zeros(2, dtype=torch.float64),
        )


@pytest.mark.parametrize("step_size", [0, -1, float("inf"), float("nan"), True])
def test_no_implicit_or_invalid_dual_step(step_size):
    with pytest.raises(ValueError):
        projected_dual_update(torch.zeros(1, dtype=torch.float64),
                              torch.ones(1, dtype=torch.float64), step_size=step_size)
