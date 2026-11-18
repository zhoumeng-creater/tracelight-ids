"""Fixed F targets in pure NumPy; no network loading or scoring."""
from __future__ import annotations

import numpy as np


def _scores(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array")
    return result


def midranks(values):
    """One-based mean ranks, with exact score equality defining ties."""
    values = _scores(values)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2)[inverse]


def inverse_ecdf(values, u):
    """Empirical inverse CDF, ceil(N*u)-1 with endpoint index clamping."""
    ordered = np.sort(_scores(values))
    u = np.asarray(u, dtype=np.float64)
    if not np.isfinite(u).all() or ((u < 0) | (u > 1)).any():
        raise ValueError("ECDF probabilities must be finite and in [0,1]")
    indices = np.clip(np.ceil(len(ordered) * u).astype(np.int64) - 1, 0, len(ordered) - 1)
    return ordered[indices]


def input_equivalence(numeric, categorical):
    """Classes of actual FP32 numeric plus UNKNOWN/missing network inputs.

    Numeric equality identifies +0 and -0. Do not cast incoming FP64 values to
    FP32 here: callers must pass the already-frozen model transformation.
    """
    numeric, categorical = np.asarray(numeric), np.asarray(categorical)
    if numeric.dtype != np.float32 or numeric.ndim != 2 or numeric.shape[1] != 5:
        raise ValueError("Expected actual FP32 numeric model input with five columns")
    if not np.isfinite(numeric).all():
        raise ValueError("Numeric model inputs must be finite")
    if (categorical.shape != (len(numeric), 6) or not np.issubdtype(categorical.dtype, np.integer)
            or not (categorical[:, 0] == 5).all() or not np.isin(categorical[:, 1:], [0, 1]).all()):
        raise ValueError("Expected integer UNKNOWN=5 plus five binary missing indicators")
    # Every FP32 number is represented exactly in float64; categorical values are 0/1/5.
    _, inverse = np.unique(np.column_stack([numeric, categorical]), axis=0, return_inverse=True)
    return inverse


def build_targets(reference_scores, labels, group_ids, numeric, categorical, *, role="F"):
    """Build both frozen arms, aligned to F including unchanged positive anchors.

    Reference consistency is audited, not silently corrected. In particular old
    GPU caches can differ slightly for identical inputs; those are diagnostic
    geometry only, not formal same-path CPU targets.
    """
    if role != "F":
        raise ValueError("Targets can be constructed only from role F")
    scores = _scores(reference_scores).copy()
    labels, groups = np.asarray(labels), np.asarray(group_ids)
    if (labels.shape != scores.shape or not np.isin(labels, [0, 1]).all()
            or groups.shape != scores.shape):
        raise ValueError("Labels and groups must align with the finite reference scores")
    normal = labels == 0
    if not normal.any() or any(not isinstance(g, str) or not g for g in groups[normal].tolist()):
        raise ValueError("Every normal row requires a nonempty string group ID")
    if any(g != "" for g in groups[~normal].tolist()):
        raise ValueError("Positive rows must have the empty group sentinel")
    equivalence = input_equivalence(numeric, categorical)
    if len(equivalence) != len(scores):
        raise ValueError("Model inputs must align with reference scores")
    n_classes = int(equivalence.max()) + 1
    minimum_label = np.full(n_classes, 2, dtype=np.int8)
    maximum_label = np.full(n_classes, -1, dtype=np.int8)
    np.minimum.at(minimum_label, equivalence, labels)
    np.maximum.at(maximum_label, equivalence, labels)
    if np.any(minimum_label != maximum_label):
        raise ValueError("Opposite-label identical model input collision")
    provisional = scores.copy()
    group_counts = {}
    ordered_normal = np.sort(scores[normal])
    for group in sorted(set(groups[normal].tolist())):
        mask = normal & (groups == group)
        count = int(mask.sum())
        group_counts[group] = count
        _, inverse, ties = np.unique(scores[mask], return_inverse=True, return_counts=True)
        below = np.cumsum(ties) - ties
        # u=(midrank-1/2)/n=(2*below+tie_count)/(2*n). Integer ceiling
        # keeps exact ECDF boundaries from moving up one bin through FP64 rounding.
        numerator = (2 * below + ties)[inverse]
        denominator = 2 * count
        indices = (len(ordered_normal) * numerator + denominator - 1) // denominator - 1
        provisional[mask] = np.minimum(scores[mask], ordered_normal[indices])
    class_minimum = np.full(n_classes, np.inf)
    np.minimum.at(class_minimum, equivalence[normal], provisional[normal])
    transport = scores.copy()
    transport[normal] = class_minimum[equivalence[normal]]
    if np.ptp(transport[normal]) == 0:
        raise ValueError("Normal transport target is constant; frozen design rejects collapse")
    if np.ptp(scores[normal]) == 0:
        raise ValueError("Normal identity target is constant; frozen design rejects collapse")
    reference_min, reference_max = np.full(n_classes, np.inf), np.full(n_classes, -np.inf)
    np.minimum.at(reference_min, equivalence, scores)
    np.maximum.at(reference_max, equivalence, scores)
    provisional_min, provisional_max = np.full(n_classes, np.inf), np.full(n_classes, -np.inf)
    np.minimum.at(provisional_min, equivalence[normal], provisional[normal])
    np.maximum.at(provisional_max, equivalence[normal], provisional[normal])
    normal_classes = np.unique(equivalence[normal])
    discrepancy = reference_max - reference_min
    scale = max(float(inverse_ecdf(scores[normal], 0.75) - inverse_ecdf(scores[normal], 0.25)), 1.0)
    arrays = {"reference_scores": scores, "provisional_target": provisional,
              "transport_target": transport, "identity_target": scores.copy(),
              "equivalence_ids": equivalence}
    for value in arrays.values():
        value.setflags(write=False)
    return {**arrays, "scale": scale, "group_counts": group_counts,
            "equivalence": {"classes": n_classes, "normal_classes": int(len(normal_classes)),
                            "reference_inconsistent_classes": int((discrepancy != 0).sum()),
                            "max_reference_difference": float(discrepancy.max()),
                            "provisional_conflict_normal_classes": int(np.sum(
                                provisional_min[normal_classes] != provisional_max[normal_classes])),
                            "rows_lowered_by_equivalence_min": int(np.sum(transport[normal] < provisional[normal]))}}
