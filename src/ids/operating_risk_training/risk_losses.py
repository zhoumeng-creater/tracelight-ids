"""Full-population score-space risk proxies; no model or training schedule.

The normal proxy and continuous margins share the current calibration threshold
and a differentiable population scale. Exact decisions remain in evaluation.py:
a lower surrogate alone never certifies a better operating risk.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..route_batch.tabm_training import stable_scores
from .evaluation import FrozenRiskManifest, validate_alignment
from .operating_point import OperatingPoint, _scores, calibrate


class DegenerateNormalScale(ValueError):
    """Normal scores are all equal; use exact evaluation, not a scale floor.

    This can be a zero-normal-risk feasible endpoint if positives exceed the
    shared threshold. It is not evidence that the decision contract is infeasible.
    """


def aggregate_member_scores(logits: torch.Tensor) -> torch.Tensor:
    """Reuse deployed log-odds(mean(sigmoid(head))) with exactly 32 FP32 heads."""
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1:] != (32, 1)
            or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite FP32 logits with shape (rows, 32, 1)")
    result = stable_scores(logits)
    _scores(result, "aggregated_scores")
    return result


@dataclass(frozen=True)
class RiskLossState:
    normal_objective: torch.Tensor
    positive_ids: tuple[str, ...]
    positive_proxy_violation: torch.Tensor
    positive_true_violation: torch.Tensor
    scale: torch.Tensor
    group_ids: tuple[str, ...]
    group_counts: tuple[int, ...]
    normal_group_proxy: torch.Tensor
    normalized_normal_margin: torch.Tensor
    continuous_positive_margin: torch.Tensor
    strict_positive_threshold: torch.Tensor
    normal_union_size: int


def build_risk_state(
    point: OperatingPoint,
    manifest: FrozenRiskManifest,
    *,
    reference_scores: torch.Tensor,
    normal_scores: torch.Tensor,
    positive_scores: torch.Tensor,
    reference_ids: Sequence[str],
    normal_ids: Sequence[str],
    positive_ids: Sequence[str],
) -> RiskLossState:
    """Build a score graph over every frozen reference, normal and positive row.

    The manifest requires pairwise-disjoint unique identities. Reference/normal
    concatenation is their identity union, with each identity weighted once for
    scale. Positive scores never change the normal objective through the scale.
    Every normal group uses its full frozen denominator. No sampled subset,
    duplicate identity, feature-based deduplication or missing group is allowed.
    The supplied point verifies identities/values; the loss rebinds its threshold
    to the supplied reference graph, so an old/detached point cannot detach it.
    """
    validate_alignment(
        point, manifest, reference_scores=reference_scores, normal_scores=normal_scores,
        positive_scores=positive_scores, reference_ids=reference_ids,
        normal_ids=normal_ids, positive_ids=positive_ids,
    )
    # Equal values do not prove graph identity: bind the loss threshold to the
    # supplied reference graph even if an evaluator passed a detached point.
    point = calibrate(reference_scores, identities=manifest.reference_ids)
    scores = torch.cat((reference_scores, normal_scores))
    scale = scores.std(correction=0)
    if not torch.isfinite(scale):
        raise ValueError("The full normal identity union must have a finite score scale")
    if scale <= 0:
        raise DegenerateNormalScale("Constant normal scores: evaluate true risk and TP; do not add a scale floor")
    normalized_normal = (normal_scores - point.threshold) / scale
    continuous_positive = (positive_scores - point.threshold) / scale
    if not (torch.isfinite(normalized_normal).all() and torch.isfinite(continuous_positive).all()):
        raise ValueError("Normalized score margins must remain finite")
    counts = []
    group_losses = []
    for group in manifest.expected_groups:
        mask = torch.tensor(
            [value == group for value in manifest.normal_group_ids],
            dtype=torch.bool, device=normal_scores.device,
        )
        counts.append(int(mask.sum().item()))
        group_losses.append(torch.relu(1 + normalized_normal[mask]).mean())
    normal_group_proxy = torch.stack(group_losses)
    # amax shares its backward contribution uniformly across tied worst groups.
    normal_objective = normal_group_proxy.amax()
    # The exact comparison, not the normalized sign, determines strict TP.
    # A unit indicator jump gives each FN a finite dual signal. The primal
    # proxy differentiates only the continuous margin; this is a declared
    # proxy/true split, not one smooth Lagrangian or per-update protection.
    positive_alarms = positive_scores.detach() > point.threshold.detach()
    positive_proxy = -continuous_positive
    positive_true = (~positive_alarms).to(torch.float64) - continuous_positive.detach()
    if (positive_alarms & (continuous_positive.detach() <= 0)).any():
        raise ValueError("FP64 normalization lost a positive strict margin; no tolerance is applied")
    if not torch.isfinite(positive_true).all():
        raise ValueError("The strict-positive residual must remain finite")
    if not torch.equal(positive_true <= 0, positive_alarms):
        raise ValueError("FP64 normalization lost strict-positive feasibility; no tolerance is applied")
    # Diagnostic only: no nextafter-derived value participates in an update.
    strict_threshold = torch.nextafter(point.threshold.detach(),
                                       torch.full_like(point.threshold, torch.inf))
    if not torch.isfinite(strict_threshold):
        raise ValueError("No finite FP64 score can exceed the current threshold")
    return RiskLossState(
        normal_objective=normal_objective,
        positive_ids=manifest.protected_positive_ids,
        positive_proxy_violation=positive_proxy,
        positive_true_violation=positive_true,
        scale=scale,
        group_ids=manifest.expected_groups,
        group_counts=tuple(counts),
        normal_group_proxy=normal_group_proxy,
        normalized_normal_margin=normalized_normal,
        continuous_positive_margin=continuous_positive,
        strict_positive_threshold=strict_threshold,
        normal_union_size=scores.numel(),
    )
