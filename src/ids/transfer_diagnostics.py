"""Training-pool diagnostics for teacher value and an observable-input bottleneck.

No fitting, model/teacher selection, validation lookup or population guarantee.
Thresholds are chosen from calibration normals *before* probe labels are used.
The caller must supply genuine train-only pool provenance and exact encoded-input
equality groups; this array interface cannot establish either fact from scores.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import expit
from scipy.stats import kendalltau


def _vector(values, name, length=None):
    array = np.asarray(values)
    if (array.ndim != 1 or not len(array) or array.dtype.kind not in "fiu"
            or (length is not None and len(array) != length)):
        raise ValueError(f"{name} must be a nonempty aligned numeric vector")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _labels(values, name, length=None):
    result = _vector(values, name, length)
    if not np.isin(result, [0, 1]).all():
        raise ValueError(f"{name} must be binary 0/1")
    return result


def _strings(values, length, name, *, unique=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must identify every row")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must identify every row") from error
    if (len(result) != length or any(not isinstance(x, str) or not x.strip() for x in result)
            or (unique and len(set(result)) != length)):
        raise ValueError(f"{name} must contain aligned {'unique ' if unique else ''}nonempty strings")
    return np.asarray(result)


def _alpha(max_fpr):
    if (isinstance(max_fpr, bool) or not isinstance(max_fpr, Real)
            or not math.isfinite(max_fpr) or not 0 < max_fpr < 1):
        raise ValueError("max_fpr must be a finite number strictly between 0 and 1")
    return float(max_fpr)


def training_threshold(scores, labels, max_fpr=.01):
    """Conservative finite-normal order statistic; score >= threshold alarms.

    Computation stays in float64, including nextafter and the later comparison.
    This controls this calibration pool only, without a distributional claim.
    """
    score = _vector(scores, "scores")
    y = _labels(labels, "labels", len(score))
    alpha = _alpha(max_fpr)
    normal = score[y == 0]
    if not len(normal):
        raise ValueError("calibration requires normal support")
    allowed = math.floor(alpha * len(normal))
    boundary = np.partition(normal, len(normal) - allowed - 1)[len(normal) - allowed - 1]
    with np.errstate(over="ignore", invalid="ignore"):
        threshold = np.nextafter(boundary, np.float64(np.inf))
    if not np.isfinite(threshold):
        raise ValueError("calibration has no representable finite threshold")
    fp = int(np.count_nonzero(normal >= threshold))
    return {"threshold": float(threshold), "n_normal": len(normal), "allowed_false_positives": allowed,
            "false_positives": fp, "achieved_fpr": fp / len(normal), "max_fpr": alpha,
            "decision_rule": "float64(score)>=float64(threshold)",
            "scope": "empirical_training_calibration_only"}


def binary_rank_metrics(labels, scores, max_fpr=.01):
    """ROC AUC and raw / McClish-standardized pAUC, including exact score ties.

    Piecewise-linear ROC interpolation matches sklearn's max_fpr definition.
    Standardization maps a diagonal ROC to .5 and perfect separation to 1.
    It is not an operating-point FPR guarantee or raw partial area.
    """
    score = _vector(scores, "scores")
    y = _labels(labels, "labels", len(score))
    alpha = _alpha(max_fpr)
    n_pos, n_neg = int(y.sum()), int(np.count_nonzero(y == 0))
    if not n_pos or not n_neg:
        return {"roc_auc": None, "partial_auc_raw": None, "partial_auc_standardized": None,
                "max_fpr": alpha, "reason": "requires_both_binary_classes"}
    order = np.argsort(-score, kind="stable")
    sorted_score, positive = score[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(score) - 1]
    tp = np.cumsum(positive)[ends]
    fp = 1 + ends - tp
    fpr, tpr = np.r_[0., fp / n_neg], np.r_[0., tp / n_pos]
    stop = np.searchsorted(fpr, alpha, side="right")
    clipped_fpr = np.r_[fpr[:stop], alpha]
    clipped_tpr = np.r_[tpr[:stop], np.interp(alpha, fpr[stop - 1:stop + 1], tpr[stop - 1:stop + 1])]
    area = float(trapezoid(clipped_tpr, clipped_fpr))
    min_area, max_area = .5 * alpha * alpha, alpha
    return {"roc_auc": float(trapezoid(tpr, fpr)), "partial_auc_raw": area,
            "partial_auc_standardized": .5 * (1 + (area - min_area) / (max_area - min_area)),
            "max_fpr": alpha, "reason": None}


def _operating_metrics(y, scores, threshold, alpha):
    decisions = scores >= threshold
    n_pos, n_neg = int(y.sum()), int(np.count_nonzero(y == 0))
    tp, fp = int(np.count_nonzero(decisions & (y == 1))), int(np.count_nonzero(decisions & (y == 0)))
    return {"n": len(y), "n_attack": n_pos, "n_normal": n_neg,
            "true_positives": tp, "false_positives": fp,
            "tpr": tp / n_pos if n_pos else None, "fpr": fp / n_neg if n_neg else None,
            "threshold": threshold, **binary_rank_metrics(y, scores, alpha)}


def _profile(scores, y):
    result = {}
    for label in (0, 1):
        sample = scores[y == label]
        if not len(sample):
            result[str(label)] = {"n": 0, "unique_scores": 0, "largest_tie_fraction": None,
                                  "quantiles": None, "sigmoid_lower_mass": None, "sigmoid_upper_mass": None}
            continue
        counts = np.unique(sample, return_counts=True)[1]
        probability = expit(sample)
        result[str(label)] = {"n": len(sample), "unique_scores": len(counts),
            "largest_tie_fraction": float(counts.max() / len(sample)),
            "quantiles": np.quantile(sample, [0, .01, .25, .5, .75, .99, 1]).tolist(),
            "sigmoid_lower_mass": float((probability <= 1e-6).mean()),
            "sigmoid_upper_mass": float((probability >= 1 - 1e-6).mean())}
    return result


def _rank_agreement(teacher, student, y, alpha):
    def tau(mask):
        t, s = teacher[mask], student[mask]
        if len(t) < 2 or len(np.unique(t)) < 2 or len(np.unique(s)) < 2:
            return {"n": len(t), "kendall_tau_b": None, "reason": "fewer_than_two_rows_or_constant_score"}
        value = float(kendalltau(t, s).statistic)
        return {"n": len(t), "kendall_tau_b": value, "reason": None}

    result = {"all": tau(np.ones(len(y), dtype=bool)), "normal": tau(y == 0), "attack": tau(y == 1)}
    normal = teacher[y == 0]
    if len(normal):
        k = max(1, math.ceil(alpha * len(normal)))
        boundary = np.partition(normal, len(normal) - k)[len(normal) - k]
        # Include boundary ties rather than arbitrarily choosing normal rows.
        mask = (y == 0) & (teacher >= boundary)
        result["teacher_normal_tail"] = {**tau(mask), "requested_tail_count": k,
                                         "boundary_inclusive_with_ties": float(boundary)}
    else:
        result["teacher_normal_tail"] = {"n": 0, "kendall_tau_b": None, "reason": "no_normal_support"}
    result["interpretation"] = "descriptive_order_agreement_not_teacher_correctness_or_transfer_gain"
    return result


def exact_input_ambiguity(teacher_scores, student_input_groups, teacher_threshold):
    """Empirical oracle floors for identical *encoded* student inputs only.

    For a deterministic group-constant predictor, minimum alarm imitation errors
    are sum_g min(n_g+, n_g-), and minimum mean squared score error is the
    within-group SSE / N. These relax neural capacity, apply only to supplied
    finite rows, and are not a population information-theoretic bound.
    """
    teacher = _vector(teacher_scores, "teacher_scores")
    if (isinstance(teacher_threshold, bool) or not isinstance(teacher_threshold, Real)
            or not math.isfinite(teacher_threshold)):
        raise ValueError("teacher_threshold must be finite")
    if student_input_groups is None:
        return {"status": "not_measured", "reason": "exact_encoded_student_input_groups_not_supplied"}
    groups = _strings(student_input_groups, len(teacher), "student_input_groups")
    _, first, inverse, counts = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = teacher - teacher[first][inverse]
        mean_shift = np.bincount(inverse, weights=shifted / counts[inverse])
        centered = shifted - mean_shift[inverse]
        mse = float(np.mean(centered ** 2))
    if not math.isfinite(mse):
        raise ValueError("teacher score scale overflows the exact-input variance diagnostic")
    n_alarm = np.bincount(inverse, weights=teacher >= teacher_threshold).astype(np.int64)
    errors = int(np.minimum(n_alarm, counts - n_alarm).sum())
    # One O(N log N) lexsort, not a full-row scan per duplicate group.
    order = np.lexsort((teacher, inverse))
    ig, tg = inverse[order], teacher[order]
    starts = np.r_[0, np.flatnonzero((ig[1:] != ig[:-1]) | (tg[1:] != tg[:-1])) + 1, len(teacher)]
    ties = np.diff(starts)
    group_pairs = sum(int(c) * (int(c) - 1) // 2 for c in counts)
    tied_pairs = sum(int(c) * (int(c) - 1) // 2 for c in ties)
    unequal_pairs = group_pairs - tied_pairs
    return {"status": "measured", "n_rows": len(teacher), "n_exact_input_groups": len(counts),
            "n_rows_in_non_singleton_groups": int(counts[counts > 1].sum()),
            "n_groups_with_conflicting_teacher_alarms": int(np.count_nonzero((n_alarm > 0) & (n_alarm < counts))),
            "minimum_teacher_alarm_imitation_errors": errors,
            "minimum_teacher_alarm_imitation_error_fraction": errors / len(teacher),
            "minimum_teacher_score_mse": mse,
            "within_identical_input_teacher_unequal_pairs": unequal_pairs,
            "group_provenance": "caller_declared_exact_encoded_input_equality_not_session_or_split_group",
            "scope": "finite_supplied_rows_relaxes_model_capacity_not_population_bound",
            "absence_of_conflicts_does_not_establish_representability": True}


def _fit_support(counts, family, n):
    if counts is None:
        return None
    if isinstance(counts, Mapping):
        if family is None:
            raise ValueError("family-count mapping requires probe_families")
        values = np.asarray([counts.get(name, 0) for name in family])
    else:
        values = np.asarray(counts)
    if (values.shape != (n,) or values.dtype.kind not in "iu" or np.any(values < 0)):
        raise ValueError("teacher_fit_family_counts must be nonnegative integer counts, per family or per row")
    return values


def transfer_diagnostics(*, calibration_labels, calibration_teacher_scores, calibration_student_scores,
                         probe_labels, probe_teacher_scores, probe_student_scores, calibration_ids, probe_ids,
                         calibration_role="train_calibration", probe_role="train_probe", probe_families=None,
                         student_input_groups=None, teacher_fit_family_counts=None, max_fpr=.01):
    """Describe teacher headroom on a train-only held-out probe (e.g. C).

    Both teachers and students choose their OWN threshold on calibration normals
    (e.g. R). Different probe FPRs remain visible; correction counts are not a
    matched-FPR leaderboard or an achievable learned ensemble. No gate/selection
    rule is hidden here. Family support is diagnostic, not a binary class label.
    Raw scores are interpreted as logits only by the sigmoid saturation profile.
    A family-count map must describe the complete fit population (absent keys
    then mean zero); None denotes unknown support. OOF callers can instead pass
    an aligned per-row count from the specific teacher that produced that row.
    """
    if calibration_role != "train_calibration" or probe_role != "train_probe":
        raise ValueError("only explicit train_calibration and train_probe roles are accepted; no validation/test")
    alpha = _alpha(max_fpr)
    cy = _labels(calibration_labels, "calibration_labels")
    ct = _vector(calibration_teacher_scores, "calibration_teacher_scores", len(cy))
    cs = _vector(calibration_student_scores, "calibration_student_scores", len(cy))
    y = _labels(probe_labels, "probe_labels")
    t = _vector(probe_teacher_scores, "probe_teacher_scores", len(y))
    s = _vector(probe_student_scores, "probe_student_scores", len(y))
    cid = _strings(calibration_ids, len(cy), "calibration_ids", unique=True)
    pid = _strings(probe_ids, len(y), "probe_ids", unique=True)
    if set(cid) & set(pid):
        raise ValueError("calibration and probe row IDs overlap")
    family = _strings(probe_families, len(y), "probe_families") if probe_families is not None else None
    fit_counts = _fit_support(teacher_fit_family_counts, family, len(y))
    tc, sc = training_threshold(ct, cy, alpha), training_threshold(cs, cy, alpha)
    td, sd = t >= tc["threshold"], s >= sc["threshold"]
    correction, harm = (y == 1) & td & ~sd, (y == 1) & ~td & sd
    repair_fp, add_fp = (y == 0) & ~td & sd, (y == 0) & td & ~sd
    attack = y == 1
    support = {"known": fit_counts is not None, "n_attack_rows_with_family_absent_in_teacher_fit":
               int(np.count_nonzero(attack & (fit_counts == 0))) if fit_counts is not None else None,
               "family_is_binary_training_target": False}
    per_family = {}
    if family is not None:
        for name in sorted(set(family[attack])):
            mask = attack & (family == name)
            per_family[name] = {"support": int(mask.sum()), "teacher_tpr": float(td[mask].mean()),
                "student_tpr": float(sd[mask].mean()), "teacher_corrects_student_fn": int(correction[mask].sum()),
                "teacher_loses_student_tp": int(harm[mask].sum()),
                "n_rows_with_family_absent_in_teacher_fit": int((fit_counts[mask] == 0).sum())
                    if fit_counts is not None else None}
    teacher_metrics = _operating_metrics(y, t, tc["threshold"], alpha)
    student_metrics = _operating_metrics(y, s, sc["threshold"], alpha)
    tpauc, spauc = teacher_metrics["partial_auc_standardized"], student_metrics["partial_auc_standardized"]
    return {"scope": "training_probe_diagnostic_not_independent_confirmation", "fits": 0,
        "calibration_role": calibration_role, "probe_role": probe_role,
        "teacher": {"calibration": tc, "probe": teacher_metrics, "score_profile": _profile(t, y)},
        "student": {"calibration": sc, "probe": student_metrics, "score_profile": _profile(s, y)},
        "paired": {"teacher_corrects_student_fn": int(correction.sum()), "teacher_loses_student_tp": int(harm.sum()),
            "teacher_repairs_student_fp": int(repair_fp.sum()), "teacher_adds_student_fp": int(add_fp.sum()),
            "teacher_minus_student_tp": int(correction.sum()) - int(harm.sum()),
            "teacher_minus_student_fp": int(add_fp.sum()) - int(repair_fp.sum()),
            "teacher_minus_student_partial_auc_standardized": tpauc - spauc
                if tpauc is not None and spauc is not None else None,
            "probe_fprs_need_not_be_equal": True},
        "per_attack_family": per_family, "teacher_family_support": support,
        "rank_agreement": _rank_agreement(t, s, y, alpha),
        "exact_input_ambiguity": exact_input_ambiguity(t, student_input_groups, tc["threshold"]),
        "provenance_boundary": "caller_must_establish_real_train_roles_disjoint_groups_and_exact_encoded_inputs"}
