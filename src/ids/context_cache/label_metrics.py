"""Fixed comparisons for binary, semantic-family, and grouped-sham arms."""
from __future__ import annotations

import numpy as np

from .calibration import _labels, _scores, apply_threshold
from .label_inputs import ARMS
from .scanning_metrics import (
    _families, _paired, _scanning, _validate_calibration, analyze_arm, scanning_roc,
)


def score_diagnostics(scores, normal_scores, registration):
    """Within-arm margin and closed calibration rank interval for tied scores.

    A reject-all endpoint gives margin -infinity. Rank bounds are count(C<s)/n
    and count(C<=s)/n, not a randomized rank or a deployable new threshold.
    """
    values, normal = _scores(scores), _scores(normal_scores, "normal_scores")
    if not len(normal):
        raise ValueError("score diagnostics require nonempty normal calibration scores")
    _validate_calibration(normal, registration)
    ordered = np.sort(normal)
    return {
        "score_minus_threshold": (np.full(len(values), -np.inf) if registration["reject_all"]
                                  else values - registration["threshold"]),
        "calibration_percentile_lower": np.searchsorted(ordered, values, side="left") / len(normal),
        "calibration_percentile_upper": np.searchsorted(ordered, values, side="right") / len(normal),
    }


def _candidate_gate(arm_reports, scanning_tp):
    baseline, semantic, sham = (arm_reports[arm] for arm in ARMS)
    metrics = {arm: arm_reports[arm]["metrics"] for arm in ARMS}
    partials = {arm: arm_reports[arm]["scanning_vs_normal_roc"]["partial_auc_divided_by_width"]
                for arm in ARMS}
    available = all(value is not None for value in partials.values())
    checks = {
        "scanning_tp_exceeds_baseline": scanning_tp["semantic_family"] > scanning_tp["baseline"],
        "scanning_tp_exceeds_grouped_sham": scanning_tp["semantic_family"] > scanning_tp["grouped_sham"],
        "overall_tp_not_below_baseline": semantic["metrics"]["tp"] >= baseline["metrics"]["tp"],
        "normal_fp_not_above_baseline": semantic["metrics"]["fp"] <= baseline["metrics"]["fp"],
        "partial_roc_available": available,
        "partial_auc_exceeds_baseline": bool(available and partials["semantic_family"] > partials["baseline"]),
        "partial_auc_exceeds_grouped_sham": bool(available and partials["semantic_family"] > partials["grouped_sham"]),
    }
    semantic_fp_higher = semantic["metrics"]["fp"] > sham["metrics"]["fp"]
    return {
        "passed": bool(all(checks.values())), "checks": checks,
        "scope": "candidate for a separately frozen independent confirmation, not proof or automatic execution",
        "operating_points": {arm: {"scanning_tp": scanning_tp[arm], "overall_tp": metrics[arm]["tp"],
                                    "normal_fp": metrics[arm]["fp"], "normal_fpr": metrics[arm]["fpr"]}
                             for arm in ARMS},
        "normalized_scanning_pauc": partials,
        "partial_roc_definition": {
            "population": "the same evaluation Scanning rows and normal rows in every arm",
            "fpr_interval": [0., .01], "ties": "equal scores enter together",
            "boundary": "linear interpolation", "normalization": "raw area / 0.01; no chance correction",
            "interpretation": "average TPR over the fixed FPR interval, not TPR at the 1% endpoint",
        },
        "semantic_fp_higher_than_sham": semantic_fp_higher,
        "equal_or_lower_fp_cost_than_sham": not semantic_fp_higher,
        "sham_cost_interpretation": (
            "semantic arm has more observed FP than sham: do not claim equal-cost operating-point superiority"
            if semantic_fp_higher else "semantic arm has no more observed FP than sham on this opened evaluation"
        ),
        "selection": "no threshold, label partition, support rows or seed selected from evaluation results",
    }


def analyze_label_results(labels, families, arm_scores, calibration_scores, registrations, cohorts):
    """Recompute all arm reports and the predeclared gate from saved scores."""
    for name, mapping in (("arm_scores", arm_scores), ("calibration_scores", calibration_scores),
                          ("registrations", registrations)):
        if set(mapping) != set(ARMS):
            raise ValueError(f"{name} must contain exactly the three fixed label arms")
    baseline_scores = _scores(arm_scores["baseline"], "baseline_scores")
    truth = _labels(labels, len(baseline_scores))
    groups = _families(families, len(truth))
    scan = _scanning(groups)
    if np.any(scan & (truth != 1)):
        raise ValueError("Scanning family must have attack label 1")
    reports, scanning_tp, decisions = {}, {}, {}
    for arm in ARMS:
        values = _scores(arm_scores[arm])
        if len(values) != len(truth):
            raise ValueError("all arm scores must have the same aligned evaluation rows")
        report = analyze_arm(
            truth, groups, values, calibration_scores[arm], registrations[arm],
            baseline_scores, registrations["baseline"], cohorts, calibration_scores["baseline"],
        )
        # A class-granularity change does not preserve the raw logit scale.
        report["raw_score_comparison_scope"] = (
            "raw score/threshold differences are descriptive only; cross-arm separation uses rank metrics"
        )
        report["cohort_scanning_vs_normal_roc"] = {
            name: scanning_roc(truth[mask], groups[mask], values[mask])
            for name, mask in ((name, np.asarray(selected, dtype=bool)) for name, selected in cohorts.items())
        }
        decisions[arm] = apply_threshold(values, registrations[arm])
        scanning_tp[arm] = int(np.count_nonzero(scan & decisions[arm]))
        report["scanning_tp"] = scanning_tp[arm]
        reports[arm] = report
    return {
        "scope": "opened development label-granularity diagnostic; no independent confirmation",
        "score_semantics": "normal=0; attack is class 1 (binary) or union of classes 1 and 2; temperature=1",
        "arms": reports,
        "semantic_vs_sham_paired_changes": _paired(truth, decisions["grouped_sham"], decisions["semantic_family"]),
        "candidate_gate": _candidate_gate(reports, scanning_tp),
        "diagnostic_columns": {
            "score_minus_threshold": "within-arm margin only; never compare raw margins across arms",
            "calibration_percentile_lower": "count(C < score) / len(C)",
            "calibration_percentile_upper": "count(C <= score) / len(C)",
        },
    }
