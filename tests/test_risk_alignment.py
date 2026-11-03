"""Research alignment: credit is an anchor-conditioned capped-hinge view.

These checks establish an equivalence/implementation boundary, not an efficacy
result or an additional algorithm to beat in a training table.
"""
import pytest

torch = pytest.importorskip("torch")

from ids.risk_distillation import (  # noqa: E402
    RiskConstraint, build_risk_reference, constraint_risks,
)


@pytest.mark.parametrize("majorizer", ["adaptive", "fixed"])
@pytest.mark.parametrize("floor", [.03, .1, .5])
def test_credit_equals_anchor_conditioned_hinge_and_capped_hinge(majorizer, floor):
    generator = torch.Generator().manual_seed(20261103)
    a = torch.randn(400, generator=generator, dtype=torch.float64)
    m = torch.randn(400, generator=generator, dtype=torch.float64, requires_grad=True)
    y = (torch.arange(400) % 2).to(torch.int64)
    ids = tuple(f"algebra-{i}" for i in range(len(a)))
    constraints = (RiskConstraint("normal", "normal_fpr", .01),
                   RiskConstraint("attack", "attack_loss", .01))
    reference = build_risk_reference(a, y, constraints, sample_ids=ids,
                                    majorizer=majorizer, margin_floor=floor)
    actual = constraint_risks(m, reference, sample_ids=ids)
    b = (a >= 0).to(torch.float64)
    d = a.abs().clamp_min(floor) if majorizer == "adaptive" else torch.full_like(a, floor)
    hinge = (1 + m/d).clamp_min(0)
    # This expression contains no repair variable or FP-credit subtraction.
    capped_hinge = torch.minimum(torch.ones_like(m), (1 + m/floor).clamp_min(0))
    normal = ((1-b)*hinge + b*capped_hinge)[y == 0].mean()
    attack = (b*(1-m/d).clamp_min(0))[y == 1].mean()
    equivalent = torch.stack((normal, attack))
    torch.testing.assert_close(actual.values, equivalent, rtol=1e-14, atol=1e-14)
    # Seeded continuous values avoid knots where autograd subgradient choices
    # may differ between two algebraically equivalent programs.
    assert not bool(((m == 0) | (m == -floor) | (m == -d) | (m == d)).any())
    weights = m.new_tensor([1.3, .7])
    grad_actual = torch.autograd.grad((actual.values*weights).sum(), m, retain_graph=True)[0]
    grad_equivalent = torch.autograd.grad((equivalent*weights).sum(), m)[0]
    torch.testing.assert_close(grad_actual, grad_equivalent, rtol=1e-13, atol=1e-13)


def test_equal_piecewise_values_do_not_force_same_autograd_choice_at_a_knot():
    m = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    credit_view = 1 - (-m/.1).clamp(0, 1)
    capped_hinge = torch.minimum(torch.ones_like(m), torch.relu(1+m/.1))
    assert credit_view.item() == capped_hinge.item() == 1
    left = torch.autograd.grad(credit_view, m, retain_graph=True)[0]
    right = torch.autograd.grad(capped_hinge, m)[0]
    assert left.item() == pytest.approx(10.)
    assert right.item() == pytest.approx(5.)


def test_author_code_shape_counterexample_is_not_samplewise_weighting():
    # Mirrors only the tensor shapes identified in the fixed author blob,
    # without importing/running the external training project.
    weights = torch.tensor([[1.], [2.]])
    row_losses = torch.tensor([3., 7.])
    broadcast = (row_losses*weights).mean()
    intended = (row_losses*weights.squeeze(1)).mean()
    assert broadcast.item() == pytest.approx(7.5)
    assert intended.item() == pytest.approx(8.5)
    assert broadcast != intended


def test_old_normal_false_positive_has_no_credit_gradient_before_crossing_boundary():
    reference = build_risk_reference(torch.tensor([1.], dtype=torch.float64), torch.tensor([0]),
                                    [RiskConstraint("normal", "normal_fpr", .01)],
                                    sample_ids=("old-fp",), margin_floor=.1)
    wrong = torch.tensor([.8], dtype=torch.float64, requires_grad=True)
    report = constraint_risks(wrong, reference, sample_ids=("old-fp",))
    assert report.values.item() == 1
    risk_grad = torch.autograd.grad(report.values.sum(), wrong)[0]
    assert risk_grad.item() == 0
    # The full method DOES include this separate common-hard correction signal.
    hard = torch.nn.functional.binary_cross_entropy_with_logits(wrong, torch.zeros_like(wrong))
    assert torch.autograd.grad(hard, wrong)[0].item() > 0
    repaired = torch.tensor([-.05], dtype=torch.float64, requires_grad=True)
    credit = constraint_risks(repaired, reference, sample_ids=("old-fp",))
    assert credit.values.item() == pytest.approx(.5)
    assert torch.autograd.grad(credit.values.sum(), repaired)[0].item() == pytest.approx(10.)
