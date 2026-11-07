"""Development-only normal-score calibration for binary logit differences.

The 2026 context-cache design uses an exact normal rank and a strict ``>``
decision. It does not use the legacy probability-score/``>=`` evaluation path.
Array inputs are already aligned; this module does not infer row identities,
filter a mixed calibration dataset, or establish dataset independence.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re

import numpy as np


FORMAT = "tracelight-context-threshold-v1"
SCORE_CONTRACT = {
    "score_kind": "attack_minus_normal_logit",
    "normal_label": 0,
    "attack_label": 1,
    "larger_is_attack": True,
    "comparison": "strict_greater_than",
    "comparison_dtype": "float64",
}


def _scores(scores, name="scores") -> np.ndarray:
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric real scores")
    values = values.astype(np.float64, copy=False)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite scores")
    return values


def _labels(labels, length: int) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != length:
        raise ValueError("labels must be one-dimensional and aligned with scores")
    if values.dtype.kind not in "biuf" or not bool(np.isin(values, [0, 1]).all()):
        raise ValueError("labels must contain only normal=0 and attack=1")
    return values.astype(np.uint8, copy=False)


def _alpha(alpha) -> Fraction:
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    try:
        exact = Fraction(str(alpha))
        numeric = float(exact)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError("alpha must satisfy 0 < alpha < 1") from exc
    if not 0 < exact < 1 or not 0 < numeric < 1:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    return exact


def _rank(n: int, alpha: Fraction) -> int:
    target = (n + 1) * (1 - alpha)
    return (target.numerator + target.denominator - 1) // target.denominator


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _binding(model_sha256, preprocessing_sha256, calibration_sha256) -> dict:
    return {name: _sha256(value, name) for name, value in (
        ("model_sha256", model_sha256), ("preprocessing_sha256", preprocessing_sha256),
        ("calibration_sha256", calibration_sha256),
    )}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def register_threshold(
    normal_scores, alpha=.01, *, model_sha256, preprocessing_sha256, calibration_sha256,
) -> dict:
    """Register one model's threshold from its normal-only development scores.

    ``calibration_sha256`` identifies the common normal calibration cohort or
    file, not the candidate-specific scores. Those are bound separately by
    ``normal_scores_sha256``. The caller must verify the input contains only
    normal rows and belongs to the named model/preprocessing/cohort.

    Rank greater than n (including n=0) is encoded as null/+infinity/reject-all,
    never as a finite float sentinel. No final/generalization claim is made.
    """
    values = _scores(normal_scores, "normal_scores")
    exact = _alpha(alpha)
    binding = _binding(model_sha256, preprocessing_sha256, calibration_sha256)
    n = len(values)
    rank = _rank(n, exact)
    reject_all = rank > n
    threshold = None if reject_all else float(np.partition(values, rank - 1)[rank - 1])
    fp = 0 if reject_all else int(np.count_nonzero(values > threshold))
    ties = 0 if reject_all else int(np.count_nonzero(values == threshold))
    return {
        "format": FORMAT,
        "scope": "development_calibration",
        "alpha": float(exact),
        "alpha_exact": str(exact),
        "n_normal": n,
        "rank": rank,
        "threshold": threshold,
        "threshold_kind": "positive_infinity" if reject_all else "finite",
        "reject_all": reject_all,
        "allowed_calibration_fp": max(0, n - rank),
        "calibration_fp": fp,
        "calibration_fpr": _ratio(fp, n),
        "boundary_ties": ties,
        "score_contract": dict(SCORE_CONTRACT),
        "binding": binding,
        "normal_scores_sha256": hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
        "normal_scores_hash_encoding": "ordered_little_endian_float64",
    }


def _registration(registration: dict) -> dict:
    if (not isinstance(registration, dict) or registration.get("format") != FORMAT
            or registration.get("scope") != "development_calibration"
            or registration.get("score_contract") != SCORE_CONTRACT):
        raise ValueError("registration must use the development strict-logit score contract")
    binding = registration.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("registration binding is missing")
    _binding(binding.get("model_sha256"), binding.get("preprocessing_sha256"),
             binding.get("calibration_sha256"))
    exact = _alpha(registration.get("alpha_exact"))
    if registration.get("alpha") != float(exact):
        raise ValueError("registration alpha and alpha_exact disagree")
    n = registration.get("n_normal")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("registration n_normal must be a nonnegative integer")
    rank = _rank(n, exact)
    if registration.get("rank") != rank or registration.get("allowed_calibration_fp") != max(0, n - rank):
        raise ValueError("registration rank or allowed strict calibration FP is inconsistent")
    if rank > n:
        if (registration.get("threshold_kind") != "positive_infinity"
                or registration.get("threshold") is not None or registration.get("reject_all") is not True):
            raise ValueError("insufficient calibration must use the null positive-infinity reject-all endpoint")
    else:
        threshold = registration.get("threshold")
        if (registration.get("threshold_kind") != "finite" or registration.get("reject_all") is not False
                or isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not np.isfinite(threshold)):
            raise ValueError("finite calibration must use a finite numeric threshold and reject_all=false")
    return registration


def apply_threshold(scores, registration) -> np.ndarray:
    """Apply the registered strict rule, returning bools in input row order.

    Bare scores carry no model identity. The consuming runner must match the
    registration's model/preprocessing hashes to the prediction artifacts.
    """
    values = _scores(scores)
    registered = _registration(registration)
    if registered["reject_all"]:
        return np.zeros(len(values), dtype=np.bool_)
    return values > registered["threshold"]


def _counts(labels: np.ndarray, predicted: np.ndarray) -> dict:
    attack = labels == 1
    tp = int(np.count_nonzero(attack & predicted))
    fp = int(np.count_nonzero(~attack & predicted))
    fn = int(np.count_nonzero(attack & ~predicted))
    tn = int(np.count_nonzero(~attack & ~predicted))
    return {"n_rows": len(labels), "n_normal": tn + fp, "n_attack": tp + fn,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "tpr": _ratio(tp, tp + fn), "fpr": _ratio(fp, fp + tn)}


def evaluate(labels, scores, registration, families=None) -> dict:
    """Count fixed-threshold decisions; family names never affect calibration."""
    values = _scores(scores)
    truth = _labels(labels, len(values))
    predicted = apply_threshold(values, registration)
    result = {**_counts(truth, predicted), "binding": dict(registration["binding"]), "families": {}}
    if families is not None:
        groups = np.asarray(families, dtype=object)
        if (groups.ndim != 1 or len(groups) != len(values)
                or not all(isinstance(group, str) and group for group in groups)):
            raise ValueError("families must be aligned nonempty string names")
        for group in sorted(set(groups.tolist())):
            selected = groups == group
            result["families"][group] = _counts(truth[selected], predicted[selected])
    return result


def _row_ids(ids, length: int) -> tuple[list, str]:
    if ids is None:
        return list(range(length)), "position_only"
    supplied = np.asarray(ids, dtype=object)
    if supplied.ndim != 1 or len(supplied) != length:
        raise ValueError("ids must be one-dimensional and aligned with both score arrays")
    result, seen = [], set()
    for value in supplied:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("ids must be unique nonempty strings or integers")
        if isinstance(value, (int, np.integer)):
            value = int(value)
            identity = ("integer", value)
        elif isinstance(value, str) and value:
            identity = ("string", value)
        else:
            raise ValueError("ids must be unique nonempty strings or integers")
        if identity in seen:
            raise ValueError("ids must be unique; duplicated rows cannot be silently paired")
        seen.add(identity)
        result.append(value)
    return result, "caller_aligned_ids"


def compare_decisions(labels, fullscores, fullreg, qcocscores, qcocreg, ids=None, *,
                      require_same_preprocessing=True) -> dict:
    """Compare two separately calibrated models on the same ordered rows.

    IDs bind the common row order; they do not realign independent score files.
    The runner must join/verify prediction IDs before passing these arrays.
    Both registrations must name the same normal cohort and exact alpha.
    Matching preprocessing is required by default. An explicit input-profile
    comparison may relax only that match; both original bindings remain intact.
    """
    left = _scores(fullscores, "fullscores")
    right = _scores(qcocscores, "qcocscores")
    truth = _labels(labels, len(left))
    if len(right) != len(left):
        raise ValueError("fullscores and qcocscores must have the same aligned length")
    fullreg, qcocreg = _registration(fullreg), _registration(qcocreg)
    if _alpha(fullreg["alpha_exact"]) != _alpha(qcocreg["alpha_exact"]):
        raise ValueError("paired comparison requires the same alpha")
    if not isinstance(require_same_preprocessing, bool):
        raise ValueError("require_same_preprocessing must be boolean")
    bindings = ("preprocessing_sha256", "calibration_sha256") if require_same_preprocessing else (
        "calibration_sha256",)
    for name in bindings:
        if fullreg["binding"][name] != qcocreg["binding"][name]:
            raise ValueError(f"paired comparison requires the same {name}")
    row_ids, alignment = _row_ids(ids, len(truth))
    before, after = apply_threshold(left, fullreg), apply_threshold(right, qcocreg)
    attack = truth == 1
    masks = {
        "repaired_fn": attack & ~before & after,
        "new_fn": attack & before & ~after,
        "repaired_fp": ~attack & before & ~after,
        "new_fp": ~attack & ~before & after,
    }
    changes = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
    states = np.where(attack, np.where(before, "unchanged_tp", "unchanged_fn"),
                      np.where(before, "unchanged_fp", "unchanged_tn")).astype(object)
    for name, mask in masks.items():
        states[mask] = name
    pair_counts = {
        kind: {f"{a}{b}": int(np.count_nonzero((truth == label) & (before == a) & (after == b)))
               for a in (0, 1) for b in (0, 1)}
        for kind, label in (("normal", 0), ("attack", 1))
    }
    rows = [{"row_id": row_ids[i], "label": int(truth[i]), "full_score": float(left[i]),
             "qcoc_score": float(right[i]), "full_predicted": bool(before[i]),
             "qcoc_predicted": bool(after[i]), "change": states[i]} for i in range(len(truth))]
    row_bytes = json.dumps(row_ids, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    result = {
        "scope": "development_paired_comparison",
        "alignment": alignment,
        "ordered_row_ids_sha256": hashlib.sha256(row_bytes).hexdigest(),
        "ordered_row_ids_hash_encoding": "utf8_json_array_compact",
        "full": _counts(truth, before), "qcoc": _counts(truth, after),
        "full_binding": dict(fullreg["binding"]), "qcoc_binding": dict(qcocreg["binding"]),
        "changes": changes, "prediction_pairs": pair_counts,
        "net_tp": changes["repaired_fn"] - changes["new_fn"],
        "net_fp": changes["new_fp"] - changes["repaired_fp"],
        "rows": rows,
    }
    if not require_same_preprocessing:
        result["preprocessing_comparison"] = {
            "different_preprocessing_authorized": True,
            "same_preprocessing": fullreg["binding"]["preprocessing_sha256"] == qcocreg["binding"]["preprocessing_sha256"],
        }
    return result
