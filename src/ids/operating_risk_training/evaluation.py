"""True operating-point counts and replayable rows; no model or selection rule."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import torch

from .operating_point import (
    OperatingDecisions,
    OperatingPoint,
    _identities,
    _scores,
    calibration_rank,
)


@dataclass(frozen=True)
class FrozenRiskManifest:
    """One exhaustive, nonoverlapping partition of the declared normal rows.

    This binds row identities and order, not collection independence or the
    representativeness of a proposed risk family. Each positive identity here
    is required to remain detected; there is no average-recall tolerance.
    """

    reference_ids: tuple[str, ...]
    normal_ids: tuple[str, ...]
    normal_group_ids: tuple[str, ...]
    expected_groups: tuple[str, ...]
    protected_positive_ids: tuple[str, ...]

    def __post_init__(self):
        for name in ("reference_ids", "normal_ids", "expected_groups", "protected_positive_ids"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValueError(f"{name} must be a nonempty identity sequence")
            value = tuple(value)
            if not value:
                raise ValueError(f"{name} must be nonempty")
            object.__setattr__(self, name, _identities(value, len(value)))
        calibration_rank(len(self.reference_ids))
        if isinstance(self.normal_group_ids, (str, bytes)):
            raise ValueError("normal_group_ids must align one group per normal identity")
        groups = tuple(self.normal_group_ids)
        if (len(groups) != len(self.normal_ids)
                or any(not isinstance(g, str) or not g.strip() for g in groups)):
            raise ValueError("normal_group_ids must align one group per normal identity")
        if set(groups) != set(self.expected_groups):
            raise ValueError("Normal partition has unknown or empty expected groups")
        object.__setattr__(self, "normal_group_ids", groups)
        q, n, p = map(set, (self.reference_ids, self.normal_ids, self.protected_positive_ids))
        if q & n or q & p or n & p:
            raise ValueError("Reference, normal and protected-positive identities must be disjoint")

    def validate_order(self, *, reference_ids: Sequence[str], normal_ids: Sequence[str],
                       positive_ids: Sequence[str]) -> None:
        """Reject omission, repetition, unknown IDs and silent score reordering."""
        for actual, expected, role in (
            (reference_ids, self.reference_ids, "reference"),
            (normal_ids, self.normal_ids, "normal"),
            (positive_ids, self.protected_positive_ids, "positive"),
        ):
            if isinstance(actual, (str, bytes)) or tuple(actual) != expected:
                raise ValueError(f"{role} score identities/order differ from the frozen manifest")


def tp_protection_feasible(positive_decisions: OperatingDecisions) -> bool:
    """Only all declared positive identities' strict alarms determine this flag."""
    alarms = positive_decisions.alarms
    if (not isinstance(alarms, torch.Tensor) or alarms.dtype != torch.bool or alarms.ndim != 1
            or alarms.numel() == 0 or alarms.numel() != len(positive_decisions.identities)):
        raise ValueError("Nonempty aligned protected-positive decisions are required")
    return bool(alarms.all().item())


def _rows(identities, scores, gaps, alarms, *, groups=None):
    values = scores.detach().cpu().tolist()
    margins = gaps.detach().cpu().tolist()
    decisions = alarms.detach().cpu().tolist()
    rows = []
    for i, identity in enumerate(identities):
        row = {"identity": identity, "score": values[i], "signed_gap": margins[i], "alarm": decisions[i]}
        if groups is not None:
            row["group"] = groups[i]
        rows.append(row)
    return rows


@torch.no_grad()
def validate_alignment(point: OperatingPoint, manifest: FrozenRiskManifest, *,
                       reference_scores: torch.Tensor, normal_scores: torch.Tensor, positive_scores: torch.Tensor,
                       reference_ids: Sequence[str], normal_ids: Sequence[str], positive_ids: Sequence[str]) -> None:
    """Validate complete roles and the current point without changing input graphs.

    Shared by true evaluation and differentiable risk code. This function does
    not construct group risks, row records or any training objective.
    """
    manifest.validate_order(reference_ids=reference_ids, normal_ids=normal_ids, positive_ids=positive_ids)
    for values, identities, role in ((reference_scores, manifest.reference_ids, "reference"),
                                     (normal_scores, manifest.normal_ids, "normal"),
                                     (positive_scores, manifest.protected_positive_ids, "positive")):
        _scores(values, f"{role}_scores")
        if values.numel() != len(identities):
            raise ValueError(f"{role} scores must align with all frozen identities")
    if point.reference_ids != manifest.reference_ids or point.rank != calibration_rank(len(reference_ids)):
        raise ValueError("Operating point differs from the frozen reference manifest")
    if (not isinstance(point.threshold, torch.Tensor) or point.threshold.dtype != torch.float64
            or point.threshold.ndim != 0 or not torch.isfinite(point.threshold)
            or point.threshold.device != reference_scores.device):
        raise ValueError("Operating threshold must be a finite FP64 scalar on the reference device")
    exact = torch.kthvalue(reference_scores, point.rank).values
    ties = int((reference_scores == exact).sum().item())
    if not torch.equal(exact, point.threshold) or ties != point.threshold_tie_count:
        raise ValueError("Operating threshold/ties do not match the supplied reference scores")
    for values in (reference_scores, normal_scores, positive_scores):
        if values.device != point.threshold.device:
            raise ValueError("All score tensors and the operating point must share a device")
        if not torch.isfinite(values - point.threshold).all():
            raise ValueError("All signed gaps must remain finite")


@torch.no_grad()
def evaluate(point: OperatingPoint, manifest: FrozenRiskManifest, *,
             reference_scores: torch.Tensor, normal_scores: torch.Tensor, positive_scores: torch.Tensor,
             reference_ids: Sequence[str], normal_ids: Sequence[str], positive_ids: Sequence[str]) -> dict:
    """Evaluate one shared point and keep every input score for cold replay.

    Fractions remain exact in memory. ``metrics_to_json`` converts them to
    integer numerator/denominator pairs; all score values are Python FP64.
    This reports TP feasibility separately from normal risk and never ranks
    candidates, accepts an experiment, or considers auxiliary loss/H/tie size.
    """
    validate_alignment(point, manifest, reference_scores=reference_scores, normal_scores=normal_scores,
                       positive_scores=positive_scores, reference_ids=reference_ids,
                       normal_ids=normal_ids, positive_ids=positive_ids)
    normal = point.evaluate(normal_scores, identities=normal_ids)
    positive = point.evaluate(positive_scores, identities=positive_ids)
    normal_rows = _rows(manifest.normal_ids, normal_scores, normal.signed_margins, normal.alarms,
                        groups=manifest.normal_group_ids)
    positive_rows = _rows(manifest.protected_positive_ids, positive_scores,
                          positive.signed_margins, positive.alarms)
    reference_gaps = reference_scores - point.threshold
    reference_rows = _rows(manifest.reference_ids, reference_scores, reference_gaps,
                           reference_scores > point.threshold)
    group_metrics = {}
    for group in manifest.expected_groups:
        rows = [row for row in normal_rows if row["group"] == group]
        fp = sum(row["alarm"] for row in rows)
        group_metrics[group] = {"fp_count": fp, "denominator": len(rows), "risk": Fraction(fp, len(rows))}
    passed = [row["identity"] for row in positive_rows if row["alarm"]]
    lost = [row["identity"] for row in positive_rows if not row["alarm"]]
    fp = sum(row["alarm"] for row in normal_rows)
    return {
        "format": "operating-risk-evaluation-v1", "comparison": ">", "score_dtype": "float64",
        "manifest": {name: list(getattr(manifest, name)) for name in (
            "reference_ids", "normal_ids", "normal_group_ids", "expected_groups", "protected_positive_ids")},
        "operating_point": {"threshold": point.threshold.item(), "rank": point.rank,
                            "n_reference": point.n_normal,
                            "reference_threshold_ties": point.threshold_tie_count},
        "normal": {"fp_count": fp, "denominator": len(normal_rows), "risk": Fraction(fp, len(normal_rows)),
                   "groups": group_metrics,
                   "threshold_tie_count": sum(row["signed_gap"] == 0 for row in normal_rows)},
        "protected_positives": {
            "tp_count": len(passed), "fn_count": len(lost), "denominator": len(positive_rows),
            "recall": Fraction(len(passed), len(positive_rows)), "passed_ids": passed, "lost_ids": lost,
            "min_signed_gap": min(row["signed_gap"] for row in positive_rows),
            "threshold_tie_count": sum(row["signed_gap"] == 0 for row in positive_rows),
        },
        "tp_protection_feasible": tp_protection_feasible(positive),
        "rows": {"reference": reference_rows, "normal": normal_rows, "protected_positive": positive_rows},
    }


def metrics_to_json(value):
    """Lossless JSON-ready representation of the exact count ratios."""
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: metrics_to_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [metrics_to_json(item) for item in value]
    return value
