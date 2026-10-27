"""Binary detection metrics and threshold calibration with explicit denominators.

All scores are converted to float64 before comparison. This is important when a
threshold is the next representable float above a tied normal score.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

_Z_95 = 1.959963984540054


def _validated_inputs(y_true: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or values.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional")
    if len(labels) != len(values):
        raise ValueError("y_true and scores must have the same length")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("scores must be finite values in [0, 1]")
    return labels.astype(np.uint8), values


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _wilson(successes: int, denominator: int) -> dict[str, Any]:
    """Two-sided Wilson 95% interval under an IID Bernoulli assumption."""
    low: float | None = None
    high: float | None = None
    if denominator:
        p = successes / denominator
        z2 = _Z_95**2
        divisor = 1 + z2 / denominator
        centre = (p + z2 / (2 * denominator)) / divisor
        half = (
            _Z_95
            * math.sqrt(p * (1 - p) / denominator + z2 / (4 * denominator**2))
            / divisor
        )
        low, high = max(0.0, centre - half), min(1.0, centre + half)
    return {
        "low": low,
        "high": high,
        "confidence": 0.95,
        "method": "wilson",
        "assumption": "IID Bernoulli observations; not a guarantee for correlated flows",
        "successes": int(successes),
        "denominator": int(denominator),
        "small_sample": denominator < 30,
        "few_events": successes < 5 or denominator - successes < 5,
    }


def calibrate_threshold(
    y_true: Any, scores: Any, max_fpr: float = 0.01
) -> dict[str, Any]:
    """Choose a conservative threshold using only the supplied normal scores.

    A record is an attack when ``float64(score) >= threshold``. For n normal
    records, at most floor(max_fpr * n) may be flagged. Ties at the next
    disallowed score are all rejected. A threshold above 1 is valid and rejects
    every possible score. This bounds empirical calibration FPR only.
    """
    labels, values = _validated_inputs(y_true, scores)
    alpha = float(max_fpr)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("max_fpr must be finite and in [0, 1]")
    normal_scores = values[labels == 0]
    n_normal = len(normal_scores)
    if n_normal == 0:
        raise ValueError("threshold calibration requires at least one normal record")
    allowed = math.floor(alpha * n_normal)
    next_disallowed: float | None = None
    if allowed >= n_normal:
        threshold = 0.0
    else:
        next_disallowed = float(np.sort(normal_scores)[::-1][allowed])
        threshold = float(np.nextafter(np.float64(next_disallowed), np.inf))
    false_positives = int(np.count_nonzero(normal_scores >= threshold))
    return {
        "threshold": threshold,
        "max_fpr": alpha,
        "n_normal": int(n_normal),
        "allowed_false_positives": int(allowed),
        "false_positives": false_positives,
        "achieved_fpr": false_positives / n_normal,
        "decision_rule": "float64(score) >= threshold",
        "next_disallowed_normal_score": next_disallowed,
        "rejects_all_scores": threshold > 1,
        "scope": "empirical calibration FPR only; test FPR is not guaranteed",
    }


def evaluate_binary(
    y_true: Any,
    scores: Any,
    threshold: float,
    attack_categories: Any | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed threshold without selecting or modifying it.

    Undefined ratios are JSON null (Python None). Macro F1 uses both fixed
    classes [normal, attack] and is null if either class F1 is undefined.
    Average precision is null without a true attack; ROC AUC requires both
    classes. Per-family recall measures binary detection, not family prediction.
    """
    labels, values = _validated_inputs(y_true, scores)
    if len(labels) == 0:
        raise ValueError("evaluation requires at least one record")
    cutoff = float(threshold)
    if not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("threshold must be finite and non-negative; values above 1 are allowed")
    predicted = values >= cutoff
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    f1_attack = _ratio(2 * tp, 2 * tp + fp + fn)
    f1_normal = _ratio(2 * tn, 2 * tn + fp + fn)
    macro_f1 = (
        (f1_attack + f1_normal) / 2
        if f1_attack is not None and f1_normal is not None
        else None
    )
    categories: list[dict[str, Any]] = []
    if attack_categories is not None:
        raw_categories = np.asarray(attack_categories, dtype=object)
        if raw_categories.ndim != 1 or len(raw_categories) != len(labels):
            raise ValueError("attack_categories must be one-dimensional and match y_true")
        category_labels: list[str | None] = []
        for item in raw_categories:
            missing = item is None or (
                isinstance(item, (float, np.floating)) and math.isnan(float(item))
            )
            category_labels.append(None if missing else str(item))
        # Keep a missing category distinct from a literal string such as 'None'.
        observed = {category_labels[i] for i in np.flatnonzero(positive)}
        for category in sorted(observed, key=lambda x: (x is None, x or "")):
            mask = positive & np.asarray([x == category for x in category_labels])
            count = int(np.count_nonzero(mask))
            detected = int(np.count_nonzero(mask & predicted))
            categories.append(
                {
                    "category": category,
                    "n_attacks": count,
                    "detected": detected,
                    "recall": _ratio(detected, count),
                    "interval": _wilson(detected, count),
                }
            )
    return {
        "n_samples": int(len(labels)),
        "threshold": cutoff,
        "decision_rule": "float64(score) >= threshold",
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "attack_recall": _ratio(tp, tp + fn),
        "normal_fpr": _ratio(fp, fp + tn),
        "precision": _ratio(tp, tp + fp),
        "accuracy": _ratio(tp + tn, len(labels)),
        "f1_attack": f1_attack,
        "f1_normal": f1_normal,
        "macro_f1": macro_f1,
        "average_precision": (
            float(average_precision_score(labels, values)) if tp + fn else None
        ),
        "roc_auc": (
            float(roc_auc_score(labels, values)) if tp + fn and tn + fp else None
        ),
        "intervals": {
            "attack_recall": _wilson(tp, tp + fn),
            "normal_fpr": _wilson(fp, fp + tn),
            "precision": _wilson(tp, tp + fp),
        },
        "attack_categories": categories,
        "attack_category_semantics": "binary attack detection recall within each true attack family; not family classification accuracy",
    }
