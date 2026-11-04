"""Exact threshold feasibility on finite training/reference score populations.

This is elementary order-statistic event counting, not a new NP classifier or a
population certificate. C participates in choosing the threshold: callers must
declare that training protocol rather than calling the result R-only calibration.
No validation/test rows, fitting, sampling or threshold-gradient approximation.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import torch

from .risk_distillation import RiskReference, constraint_risks


def _vector(values, name, *, length=None):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in "fiu"
            or (length is not None and len(array) != length)):
        raise ValueError(f"{name} must be a nonempty aligned real numeric vector")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _ids(values, length, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must identify every row")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must identify every row") from error
    if (len(result) != length or any(not isinstance(x, str) or not x.strip() for x in result)
            or len(set(result)) != length):
        raise ValueError(f"{name} must contain aligned unique nonempty strings")
    return result


def _normal_bound(scores, name, population, limit):
    support = len(scores)
    allowed = math.floor(limit * support)
    if allowed >= support:
        boundary, lower = None, None
    else:
        # Strict > next-disallowed score because predictions use score >= t.
        boundary = float(np.partition(scores, support - allowed - 1)[support - allowed - 1])
        with np.errstate(over="ignore", invalid="ignore"):
            lower = float(np.nextafter(np.float64(boundary), np.float64(np.inf)))
        if not math.isfinite(lower):
            raise ValueError(f"normal population {population}/{name} has no representable finite threshold")
    return {"name": name, "population": population, "kind": "normal_fpr", "limit": float(limit),
            "support": support, "allowed": allowed, "boundary": boundary, "lower_bound": lower,
            "boundary_rule": "strictly_greater_than_next_disallowed_normal_score"}


def solve_threshold_interval(reference_logits, reference_labels, constraint_logits, reference: RiskReference,
                             *, reference_ids, constraint_ids, max_fpr=.01):
    """Return a JSON-compatible necessary/sufficient finite-pool interval.

    Every prediction is float64(score) >= t. Each normal budget gives a strict
    lower bound, represented by float64 nextafter. Each attack retention budget
    gives an INCLUSIVE upper bound from the original-anchor TP scores. Attack
    denominators include ALL attacks in the frozen mask, not only anchor TPs.

    The reference signature is validated through the existing risk kernel on
    zero margins. That performs O(N) signature checks/CPU copies; this is an
    explicit audit cost, not a free verification. Sample identity correspondence
    and group isolation beyond the checked row-ID boundaries remain the caller's
    responsibility. Torch inputs are detached. Returned metrics are exact finite
    event counts, with no surrogate eligibility or independent-risk claim.
    """
    if not isinstance(reference, RiskReference):
        raise TypeError("reference must be a frozen RiskReference")
    if isinstance(max_fpr, bool) or not isinstance(max_fpr, Real) or not math.isfinite(max_fpr) or not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be a finite number in [0,1)")
    r = _vector(reference_logits, "reference_logits")
    labels = _vector(reference_labels, "reference_labels", length=len(r))
    if not np.isin(labels, [0., 1.]).all():
        raise ValueError("reference_labels must be binary")
    c = _vector(constraint_logits, "constraint_logits", length=len(reference.sample_ids))
    rid = _ids(reference_ids, len(r), "reference_ids")
    cid = _ids(constraint_ids, len(c), "constraint_ids")
    if cid != reference.sample_ids:
        raise ValueError("constraint_ids or their order differ from the frozen reference")
    if set(rid) & set(cid):
        raise ValueError("reference and constraint row IDs overlap")
    # Do not trust frozen=True to make contained tensors immutable.
    constraint_risks(torch.zeros(len(c), dtype=torch.float64), reference, sample_ids=cid)
    if not bool((labels == 0).any()):
        raise ValueError("reference requires normal support")

    normal_bounds = [_normal_bound(r[labels == 0], "normal", "reference", float(max_fpr))]
    retention_bounds = []
    original_tp = reference.anchor_margins.detach().cpu().numpy() >= 0
    masks = [mask.detach().cpu().numpy() for mask in reference.members]
    for spec, mask in zip(reference.constraints, masks):
        support = int(mask.sum())
        if spec.kind == "normal_fpr":
            normal_bounds.append(_normal_bound(c[mask], spec.name, "constraint", spec.limit))
        else:
            allowed = math.floor(spec.limit * support)
            tp_scores = c[mask & original_tp]
            baseline_tp = len(tp_scores)
            upper = (float(np.partition(tp_scores, allowed)[allowed]) if allowed < baseline_tp else None)
            retention_bounds.append({"name": spec.name, "population": "constraint", "kind": "attack_loss",
                "limit": float(spec.limit), "support": support, "allowed": allowed,
                "baseline_tp_count": baseline_tp, "upper_bound": upper,
                "boundary_rule": "at_or_below_first_disallowed_lost_anchor_tp_score"})

    lower = max(row["lower_bound"] for row in normal_bounds if row["lower_bound"] is not None)
    bounded = [row["upper_bound"] for row in retention_bounds if row["upper_bound"] is not None]
    upper = min(bounded) if bounded else None
    feasible = upper is None or lower <= upper
    for row in normal_bounds:
        row["active_lower"] = row["lower_bound"] == lower
    for row in retention_bounds:
        row["active_upper"] = row["upper_bound"] is not None and row["upper_bound"] == upper
    blocking = [{"name": row["name"], "population": "constraint", "kind": "attack_loss",
                 "lower_bound": lower, "upper_bound": row["upper_bound"],
                 "reason": "retention_upper_below_joint_normal_lower"}
                for row in retention_bounds if row["upper_bound"] is not None and row["upper_bound"] < lower]

    normal_r = labels == 0
    count = int((r[normal_r] >= lower).sum())
    metrics = [{"name": "normal", "population": "reference", "kind": "normal_fpr", "support": int(normal_r.sum()),
                "allowed": normal_bounds[0]["allowed"], "count": count,
                "actual": count / int(normal_r.sum()), "passed": count <= normal_bounds[0]["allowed"]}]
    decisions = c >= lower
    for spec, mask in zip(reference.constraints, masks):
        support = int(mask.sum())
        allowed = math.floor(spec.limit * support)
        count = int(decisions[mask].sum()) if spec.kind == "normal_fpr" else int((original_tp & ~decisions)[mask].sum())
        row = {"name": spec.name, "population": "constraint", "kind": spec.kind, "support": support,
               "allowed": allowed, "count": count, "actual": count / support, "passed": count <= allowed}
        if spec.kind == "attack_loss":
            row["baseline_tp_count"] = int(original_tp[mask].sum())
        metrics.append(row)
    if feasible != all(row["passed"] for row in metrics):
        raise ValueError("order-statistic interval and exact event recount disagree")
    return {"feasible": bool(feasible), "threshold": lower if feasible else None,
            "lower_bound": lower, "upper_bound": upper, "normal_bounds": normal_bounds,
            "retention_bounds": retention_bounds, "blocking_constraints": blocking,
            "metrics_at_lower": metrics, "scope": "finite_training_pools_only",
            "selection_uses_constraint_labels": True, "decision_rule": "float64(raw_logit)>=threshold",
            "reference_signature": reference.signature, "n_reference_rows": len(r), "n_constraint_rows": len(c),
            "caller_must_verify_group_isolation": True, "population_guarantee": False}
