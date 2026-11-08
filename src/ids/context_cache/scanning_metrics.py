"""Descriptive reports for the frozen Scanning mechanism experiment.

These functions consume aligned, already-scored artifacts. Cohort labels never
select support rows, an input mapping, or a calibration threshold.
"""

from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np
import pandas as pd

from .calibration import _labels, _scores, apply_threshold, evaluate, register_threshold
from .preprocessing import CSVSchema, labels as frame_labels


SCORE_ATOL = 3e-5
SCORE_RTOL = 3e-5
PARTIAL_FPR_MAX = 0.01


def _families(values, size):
    groups = np.asarray(values, dtype=object)
    if (groups.ndim != 1 or len(groups) != size
            or not all(isinstance(v, str) and v.strip() for v in groups)):
        raise ValueError("families must be aligned nonempty strings")
    return groups


def _scanning(groups):
    return np.array([v.strip().casefold() == "scanning" for v in groups], dtype=bool)


def _numeric_rows(frame, features):
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("cohort features must contain finite numeric values")
    return [tuple(row) for row in values]


def _bool_column(values, name):
    parsed = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, str) and value in ("True", "False"):
            parsed.append(value == "True")
        else:
            raise ValueError(f"{name} must contain True/False booleans")
    return np.asarray(parsed, dtype=bool)


def build_cohorts(frames, schema, archived_evaluation_scores):
    """Return reporting masks in the unchanged evaluation row order.

    Archived IDs, labels and families must already be aligned. Exact raw-field
    matching follows the frozen artifact audit's finite float64 equality.
    The dominant pattern is chosen once from old lost rows, with lexicographic
    tie-breaking; absence of old lost rows produces empty target masks.
    """
    if not isinstance(schema, CSVSchema):
        schema = CSVSchema.from_dict(schema)
    if schema.id_column is None or schema.family_column is None:
        raise ValueError("cohort reporting requires registered ID and family columns")
    support, evaluation = frames["support"], frames["evaluation"]
    archived = archived_evaluation_scores
    needed = {"row_id", "label", "legacy_full_predicted", "full_predicted", schema.family_column}
    if not isinstance(archived, pd.DataFrame) or not needed.issubset(archived.columns):
        raise ValueError("archived evaluation scores lack cohort identity/decision columns")
    identities = evaluation[schema.id_column].astype(str).tolist()
    if (len(set(identities)) != len(identities)
            or archived["row_id"].astype(str).tolist() != identities):
        raise ValueError("archived evaluation IDs must match evaluation IDs in order")
    truth = frame_labels(evaluation, schema)
    if not np.array_equal(_labels(pd.to_numeric(archived["label"]).to_numpy(), len(truth)), truth):
        raise ValueError("archived evaluation labels disagree")
    groups = _families(evaluation[schema.family_column].to_numpy(), len(truth))
    if not np.array_equal(archived[schema.family_column].to_numpy(), groups):
        raise ValueError("archived evaluation families disagree")
    scan = _scanning(groups)
    if np.any(scan & (truth != 1)):
        raise ValueError("Scanning family must have attack label 1")
    lost = (scan & _bool_column(archived["legacy_full_predicted"], "legacy_full_predicted")
            & ~_bool_column(archived["full_predicted"], "full_predicted"))
    support_truth = frame_labels(support, schema)
    support_scan = _scanning(_families(support[schema.family_column].to_numpy(), len(support)))
    if np.any(support_scan & (support_truth != 1)):
        raise ValueError("support Scanning family must have attack label 1")
    support_keys = set(_numeric_rows(support.loc[support_scan], schema.features))
    evaluation_keys = _numeric_rows(evaluation, schema.features)
    exact = lost & np.array([key in support_keys for key in evaluation_keys], dtype=bool)
    needed_features = {"L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO"}
    if not needed_features.issubset(schema.features):
        raise ValueError("Scanning cohorts require the four registered identifier fields")
    service = (
        (pd.to_numeric(evaluation["PROTOCOL"]).to_numpy() == 6)
        & (pd.to_numeric(evaluation["L7_PROTO"]).to_numpy() == 91)
        & (pd.to_numeric(evaluation["L4_DST_PORT"]).to_numpy() == 443)
    )
    projection = [f for f in schema.features if f != "L4_SRC_PORT"]
    keys = _numeric_rows(evaluation, projection)
    counts = Counter(key for key, selected in zip(keys, lost) if selected)
    dominant = min(counts, key=lambda key: (-counts[key], key)) if counts else None
    dominant_mask = np.array([key == dominant for key in keys], dtype=bool)
    return {
        "old_54_lost": lost,
        "15_exact_P_positive_matches": exact,
        "TCP6_L791_DST443": service,
        "dominant_47_field_pattern": dominant_mask,
        "dominant_47_scanning": dominant_mask & scan,
        "dominant_47_normal": dominant_mask & (truth == 0),
    }


def empirical_percentiles(scores, normal_scores):
    """Return the fraction of normal calibration scores <= each query score."""
    values, normal = _scores(scores), _scores(normal_scores, "normal_scores")
    if len(normal) == 0:
        raise ValueError("empirical percentiles require nonempty normal calibration scores")
    return np.searchsorted(np.sort(normal), values, side="right") / len(normal)


def scanning_roc(labels, families, scores):
    """Compute grouped-tie ROC AUC and raw partial area for Scanning vs normal.

    Partial area is trapezoidal area on [0, .01], linearly interpolating any
    segment crossing .01. ``partial_auc_divided_by_width`` divides this raw
    area by .01; it is not the chance-corrected sklearn standardized pAUC.
    """
    values = _scores(scores)
    truth = _labels(labels, len(values))
    scan = _scanning(_families(families, len(values)))
    if np.any(scan & (truth != 1)):
        raise ValueError("Scanning family must have attack label 1")
    keep = scan | (truth == 0)
    positive, values = scan[keep], values[keep]
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    result = {
        "n_scanning": n_positive, "n_normal": n_negative,
        "auc": None, "partial_auc_raw": None, "partial_auc_divided_by_width": None,
        "fpr_interval": [0.0, PARTIAL_FPR_MAX],
        "ties": "equal scores enter together", "boundary": "linear interpolation",
        "normalization": "raw area and raw area / 0.01; no chance correction",
    }
    if not n_positive or not n_negative:
        return result
    order = np.argsort(-values, kind="stable")
    ordered_values, positive = values[order], positive[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered_values) != 0), len(values) - 1]
    tpr = np.r_[0.0, np.cumsum(positive)[ends] / n_positive]
    fpr = np.r_[0.0, np.cumsum(~positive)[ends] / n_negative]
    partial = 0.0
    for x0, x1, y0, y1 in zip(fpr[:-1], fpr[1:], tpr[:-1], tpr[1:]):
        if x0 >= PARTIAL_FPR_MAX:
            break
        if x1 <= x0:
            continue
        stop = min(x1, PARTIAL_FPR_MAX)
        y_stop = y0 + (y1 - y0) * (stop - x0) / (x1 - x0)
        partial += (stop - x0) * (y0 + y_stop) / 2
    result.update(auc=float(np.trapz(tpr, fpr)), partial_auc_raw=float(partial),
                  partial_auc_divided_by_width=float(partial / PARTIAL_FPR_MAX))
    return result


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"n": 0, "min": None, "median": None, "max": None, "mean": None}
    return {"n": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "mean": float(values.mean())}


def _paired(truth, before, after):
    attack = truth == 1
    changes = {
        "repaired_fn": int(np.count_nonzero(attack & ~before & after)),
        "new_fn": int(np.count_nonzero(attack & before & ~after)),
        "repaired_fp": int(np.count_nonzero(~attack & before & ~after)),
        "new_fp": int(np.count_nonzero(~attack & ~before & after)),
    }
    return {**changes, "net_tp": changes["repaired_fn"] - changes["new_fn"],
            "net_fp": changes["new_fp"] - changes["repaired_fp"],
            "changed_decisions": int(np.count_nonzero(before != after))}


def _validate_calibration(values, registration):
    # Applying validates the threshold contract without choosing a threshold.
    decisions = apply_threshold(values, registration)
    digest = hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()
    if (registration["n_normal"] != len(values)
            or registration.get("normal_scores_sha256") != digest):
        raise ValueError("calibration scores do not match their registration")
    if registration.get("calibration_fp") != int(decisions.sum()):
        raise ValueError("registered calibration FP disagrees with supplied scores")
    reconstructed = register_threshold(values, registration["alpha"], **registration["binding"])
    if reconstructed != registration:
        raise ValueError("registered threshold differs from its fixed calibration rank")
    return decisions


def analyze_arm(labels, families, scores, calibration_scores, registration,
                baseline_scores, baseline_registration, cohorts, baseline_calibration_scores=None):
    """Summarize one arm against B0; no threshold or intervention is selected."""
    values, before_values = _scores(scores), _scores(baseline_scores, "baseline_scores")
    truth = _labels(labels, len(values))
    groups = _families(families, len(values))
    if len(before_values) != len(values):
        raise ValueError("baseline scores must be aligned")
    normal = _scores(calibration_scores, "calibration_scores")
    cal_decisions = _validate_calibration(normal, registration)
    after = apply_threshold(values, registration)
    before = apply_threshold(before_values, baseline_registration)
    if (registration["alpha_exact"] != baseline_registration["alpha_exact"]
            or registration["binding"]["calibration_sha256"]
            != baseline_registration["binding"]["calibration_sha256"]):
        raise ValueError("paired arms require the same calibration cohort and alpha")
    percentiles = empirical_percentiles(values, normal)
    old_percentiles, old_normal = None, None
    if baseline_calibration_scores is not None:
        old_normal = _scores(baseline_calibration_scores, "baseline_calibration_scores")
        _validate_calibration(old_normal, baseline_registration)
        if len(old_normal) != len(normal):
            raise ValueError("paired calibration rows must have identical counts")
        old_percentiles = empirical_percentiles(before_values, old_normal)
    difference = values - before_values
    beyond = np.abs(difference) > SCORE_ATOL + SCORE_RTOL * np.abs(before_values)
    cohort_reports = {}
    for name, selected in cohorts.items():
        mask = np.asarray(selected)
        if mask.dtype != np.bool_ or mask.shape != truth.shape:
            raise ValueError(f"cohort {name} must be an aligned boolean mask")
        cohort_reports[name] = {
            "metrics": evaluate(truth[mask], values[mask], registration, groups[mask]),
            "paired_changes": _paired(truth[mask], before[mask], after[mask]),
            "score": _summary(values[mask]), "baseline_score": _summary(before_values[mask]),
            "score_delta": _summary(difference[mask]),
            "score_changes_beyond_tolerance": int(np.count_nonzero(beyond & mask)),
            "calibration_percentile": _summary(percentiles[mask]),
            "baseline_calibration_percentile": None if old_percentiles is None else _summary(old_percentiles[mask]),
            "calibration_percentile_delta": None if old_percentiles is None else _summary(
                percentiles[mask] - old_percentiles[mask]),
        }
    threshold, old_threshold = registration["threshold"], baseline_registration["threshold"]
    return {
        "scope": "opened development mechanism report; no selection or independent confirmation",
        "metrics": evaluate(truth, values, registration, groups),
        "scanning_vs_normal_roc": scanning_roc(truth, groups, values),
        "baseline_scanning_vs_normal_roc": scanning_roc(truth, groups, before_values),
        "paired_changes": _paired(truth, before, after),
        "family_changes": {family: _paired(truth[groups == family], before[groups == family],
                                            after[groups == family]) for family in sorted(set(groups))},
        "score_changes_beyond_tolerance": int(beyond.sum()),
        "score_delta": _summary(difference),
        "numerical_tolerance": {"atol": SCORE_ATOL, "rtol": SCORE_RTOL,
                                "reference": "absolute baseline score"},
        "calibration": {
            "rows": len(normal), "fp": int(cal_decisions.sum()),
            "threshold": threshold, "baseline_threshold": old_threshold,
            "threshold_delta": None if threshold is None or old_threshold is None else threshold - old_threshold,
            "score": _summary(normal), "baseline_score": None if old_normal is None else _summary(old_normal),
            "score_delta": None if old_normal is None else _summary(normal - old_normal),
        },
        "percentile_definition": "count(C scores <= E score) / len(C), each arm uses its own C",
        "cohorts": cohort_reports,
    }
