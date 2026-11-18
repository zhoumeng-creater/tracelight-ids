"""One differentiable operating point with the existing strict 1% rank rule.

Only score tensors and row identities enter this module. The caller establishes
normal-only membership and data permissions. There is no model, loss or solver.
At threshold ties the backward pass shares the derivative equally among all
tied reference rows. This is a registered generalized-gradient convention, not
a claim that the order statistic is differentiable at ties.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch


def calibration_rank(n_normal: int) -> int:
    """Exact one-based ceil((n+1)*99/100); reject an unavailable finite rank."""
    if isinstance(n_normal, bool) or not isinstance(n_normal, Integral) or n_normal < 1:
        raise ValueError("n_normal must be a positive integer")
    n_normal = int(n_normal)
    rank = ((n_normal + 1) * 99 + 99) // 100
    if rank > n_normal:
        raise ValueError("At least 99 normal reference rows are required for a finite 1% rank")
    return rank


def _identities(values: Sequence[str], size: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("identities must be an aligned sequence of unique nonempty strings")
    result = tuple(values)
    if (len(result) != size or any(not isinstance(v, str) or not v.strip() for v in result)
            or len(set(result)) != size):
        raise ValueError("identities must be an aligned sequence of unique nonempty strings")
    return result


def _scores(values: torch.Tensor, name: str) -> None:
    if (not isinstance(values, torch.Tensor) or values.dtype != torch.float64
            or values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all()):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional FP64 tensor")


class _EqualTieOrderStatistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, rank):
        threshold = torch.kthvalue(values, rank).values
        tied = values == threshold
        weights = tied.to(values.dtype) / tied.sum()
        ctx.save_for_backward(weights)
        return threshold

    @staticmethod
    def backward(ctx, gradient):
        (weights,) = ctx.saved_tensors
        return gradient * weights, None


@dataclass(frozen=True)
class OperatingDecisions:
    """For positive rows, signed_margins > 0 is exactly the TP condition."""

    identities: tuple[str, ...]
    signed_margins: torch.Tensor
    alarms: torch.Tensor


@dataclass(frozen=True)
class OperatingPoint:
    """A current-graph threshold; reuse it across normal and positive queries."""

    threshold: torch.Tensor
    reference_ids: tuple[str, ...]
    rank: int
    threshold_tie_count: int

    @property
    def n_normal(self) -> int:
        return len(self.reference_ids)

    def evaluate(self, scores: torch.Tensor, *, identities: Sequence[str]) -> OperatingDecisions:
        """Evaluate disjoint row identities with the same threshold and its graph.

        Distinct identities may have identical features or scores. This check
        prevents reference-row reuse; it does not establish independence of
        captures, features or distributions.
        """
        _scores(scores, "scores")
        query_ids = _identities(identities, scores.numel())
        if set(query_ids).intersection(self.reference_ids):
            raise ValueError("Reference and evaluated identities must be disjoint")
        if scores.device != self.threshold.device:
            raise ValueError("Scores and reference threshold must be on the same device")
        margins = scores - self.threshold
        if not torch.isfinite(margins).all():
            raise ValueError("Signed margins must remain finite")
        return OperatingDecisions(query_ids, margins, scores > self.threshold)


def calibrate(reference_scores: torch.Tensor, *, identities: Sequence[str]) -> OperatingPoint:
    """Calibrate one current FP64 score graph without detaching the threshold.

    Forward is the exact original order statistic, with no interpolation or
    jitter. For insufficient reference size, training fails instead of using
    the evaluation-only reject-all endpoint of the legacy calibration helper.
    """
    _scores(reference_scores, "reference_scores")
    reference_ids = _identities(identities, reference_scores.numel())
    rank = calibration_rank(reference_scores.numel())
    threshold = _EqualTieOrderStatistic.apply(reference_scores, rank)
    tie_count = int((reference_scores.detach() == threshold.detach()).sum().item())
    return OperatingPoint(threshold, reference_ids, rank, tie_count)
