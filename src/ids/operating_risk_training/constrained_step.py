"""Full-identity primal/dual algebra, following Cooper's separate graph terms.

This module does not choose an optimizer, step size, risk family or checkpoint.
Signed feasible residuals can decrease multipliers. A zero initial multiplier
provides no first-step protection, and finite updates need not remain feasible.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence
import math

import torch

from .operating_point import _identities, _scores


@dataclass(frozen=True)
class LagrangianTerms:
    primal: torch.Tensor
    dual: torch.Tensor


def full_identity_lagrangian(
    normal_objective: torch.Tensor,
    positive_proxy_violation: torch.Tensor,
    positive_true_violation: torch.Tensor,
    *,
    positive_ids: Sequence[str],
    multiplier_ids: Sequence[str],
    multipliers: torch.Tensor,
) -> LagrangianTerms:
    """Use the whole, identically ordered positive population in both terms.

    The caller binds multiplier_ids once to the complete protected population.
    Passing an observed minibatch as that manifest would change the contract.
    """
    for name, value in (("positive_proxy_violation", positive_proxy_violation),
                        ("positive_true_violation", positive_true_violation),
                        ("multipliers", multipliers)):
        _scores(value, name)
    ids = _identities(positive_ids, positive_proxy_violation.numel())
    expected = _identities(multiplier_ids, multipliers.numel())
    if ids != expected or positive_true_violation.shape != multipliers.shape:
        raise ValueError("Every protected identity must appear in the frozen multiplier order")
    if (not isinstance(normal_objective, torch.Tensor)
            or normal_objective.shape != torch.Size([])
            or normal_objective.dtype != torch.float64
            or not torch.isfinite(normal_objective)):
        raise ValueError("normal_objective must be a finite FP64 scalar")
    if len({normal_objective.device, positive_proxy_violation.device,
            positive_true_violation.device, multipliers.device}) != 1:
        raise ValueError("All primal/dual tensors must share a device")
    if (multipliers < 0).any():
        raise ValueError("Inequality multipliers must be nonnegative")
    primal = normal_objective + torch.sum(multipliers.detach() * positive_proxy_violation)
    dual = torch.sum(multipliers * positive_true_violation.detach())
    if not (torch.isfinite(primal) and torch.isfinite(dual)):
        raise ValueError("Lagrangian terms must remain finite")
    return LagrangianTerms(primal, dual)


def projected_dual_update(
    multipliers: torch.Tensor,
    positive_true_violation: torch.Tensor,
    *,
    step_size: float,
) -> torch.Tensor:
    """One explicit ascent/projection using the SAME pre-update residuals.

    Returns a detached new vector; it does not mutate optimizer or model state.
    step_size has no default because no training schedule is registered yet.
    """
    _scores(multipliers, "multipliers")
    _scores(positive_true_violation, "positive_true_violation")
    if (multipliers.shape != positive_true_violation.shape
            or multipliers.device != positive_true_violation.device
            or (multipliers < 0).any()):
        raise ValueError("Expected aligned nonnegative multipliers and signed residuals")
    if (isinstance(step_size, bool) or not isinstance(step_size, Real)
            or not math.isfinite(float(step_size)) or step_size <= 0):
        raise ValueError("step_size must be an explicit positive finite scalar")
    updated = multipliers.detach() + float(step_size) * positive_true_violation.detach()
    if not torch.isfinite(updated).all():
        raise ValueError("Dual update must remain finite")
    return updated.clamp_min(0)
