"""Fixed operational gates and tie-aware diagnostics on fabricated scores."""
import copy
import json

import numpy as np
import pytest

from ids.context_cache.calibration import register_threshold
from ids.context_cache.label_inputs import ARMS
from ids.context_cache.label_metrics import analyze_label_results, score_diagnostics


def registration(values, arm=0, alpha=.01):
    return register_threshold(values, alpha, model_sha256="abc"[arm] * 64,
                              preprocessing_sha256="d" * 64, calibration_sha256="e" * 64)


def run_report(scores, families=None):
    n = len(scores["baseline"])
    families = ["Scanning", "Scanning", "DDoS"] + ["Benign"] * (n - 3) if families is None else families
    labels = np.array([0 if name == "Benign" else 1 for name in families])
    calibration = {arm: np.arange(99, dtype=np.float64) for arm in ARMS}
    registrations = {arm: registration(calibration[arm], i) for i, arm in enumerate(ARMS)}
    cohorts = {"all": np.ones(n, dtype=bool), "target": np.array(families) != "DDoS"}
    return analyze_label_results(labels, families, scores, calibration, registrations, cohorts)


def test_tie_interval_and_margin_use_own_calibration():
    c = np.array([1., 1., 2., 3.])
    reg = registration(c, alpha=.4)  # rank 3, strict >2
    values = np.array([0., 1., 2., 4.])
    result = score_diagnostics(values, c, reg)
    assert result["score_minus_threshold"].tolist() == [-2., -1., 0., 2.]
    assert result["calibration_percentile_lower"].tolist() == [0., 0., .5, 1.]
    assert result["calibration_percentile_upper"].tolist() == [0., .5, .75, 1.]
    shifted = score_diagnostics(values + 20, c + 20, registration(c + 20, alpha=.4))
    for key in result:
        np.testing.assert_array_equal(result[key], shifted[key])


def test_reject_all_margin_is_negative_infinity():
    c = np.array([0., 1.])
    result = score_diagnostics([5., -1.], c, registration(c))
    assert np.isneginf(result["score_minus_threshold"]).all()
    assert result["calibration_percentile_upper"].tolist() == [1., 0.]


def test_diagnostics_reject_calibration_that_does_not_match_registration():
    c = np.arange(99.)
    with pytest.raises(ValueError, match="registration"):
        score_diagnostics([1., 2.], c + 1, registration(c))


def test_gate_requires_both_semantic_specific_ranking_and_operating_point():
    scores = {
        "baseline": np.array([97., 96., 110., 105., 10., 9.]),
        "semantic_family": np.array([110., 109., 108., 105., 10., 9.]),
        "grouped_sham": np.array([99., 96., 110., 105., 10., 9.]),
    }
    report = run_report(scores)
    gate = report["candidate_gate"]
    assert gate["passed"]
    assert all(gate["checks"].values())
    assert gate["operating_points"]["semantic_family"] == {
        "scanning_tp": 2, "overall_tp": 3, "normal_fp": 1, "normal_fpr": 1 / 3,
    }
    assert report["semantic_vs_sham_paired_changes"]["repaired_fn"] == 1
    assert report["arms"]["semantic_family"]["cohort_scanning_vs_normal_roc"]["target"]["auc"] == 1
    json.dumps(report, allow_nan=False)


def test_more_scanning_with_higher_sham_fp_cost_is_explicit_not_equal_cost_claim():
    scores = {
        "baseline": np.array([97., 96., 110., 105., 10., 9.]),
        "semantic_family": np.array([110., 109., 108., 105., 10., 9.]),
        "grouped_sham": np.array([97., 96., 110., 97.5, 10., 9.]),
    }
    report = run_report(scores)
    gate = report["candidate_gate"]
    assert gate["passed"]
    assert gate["semantic_fp_higher_than_sham"]
    assert not gate["equal_or_lower_fp_cost_than_sham"]
    assert "do not claim equal-cost" in gate["sham_cost_interpretation"]


def test_scanning_count_alone_does_not_pass_if_sham_has_better_tail_ranking():
    scores = {
        "baseline": np.array([97., 96., 110., 105., 10., 9.]),
        "semantic_family": np.array([100., 99., 110., 105., 10., 9.]),
        "grouped_sham": np.array([97., 96., 110., 10., 9., 8.]),
    }
    gate = run_report(scores)["candidate_gate"]
    assert gate["checks"]["scanning_tp_exceeds_grouped_sham"]
    assert not gate["checks"]["partial_auc_exceeds_grouped_sham"]
    assert not gate["passed"]


@pytest.mark.parametrize("change,failed", [("fp", "normal_fp_not_above_baseline"),
                                           ("overall", "overall_tp_not_below_baseline"),
                                           ("tie_sham", "scanning_tp_exceeds_grouped_sham")])
def test_individual_gate_constraints_are_not_traded_off(change, failed):
    scores = {
        "baseline": np.array([99., 97., 110., 105., 10., 9.]),
        "semantic_family": np.array([110., 109., 108., 105., 10., 9.]),
        "grouped_sham": np.array([99., 96., 110., 105., 10., 9.]),
    }
    if change == "fp":
        scores["semantic_family"][-1] = 100
    elif change == "overall":
        # Baseline detects all 3 attacks; semantic loses DDoS and one scanning.
        scores["baseline"][:3] = [99, 99, 110]
        scores["semantic_family"][:3] = [110, 109, 97]
    else:
        scores["grouped_sham"][:2] = [110, 109]
    gate = run_report(scores)["candidate_gate"]
    assert not gate["checks"][failed]
    assert not gate["passed"]


def test_missing_scanning_roc_population_cannot_pass_gate():
    scores = {arm: np.array([110., 105., 90.]) for arm in ARMS}
    gate = run_report(scores, ["DDoS", "DDoS", "Benign"])["candidate_gate"]
    assert not gate["checks"]["partial_roc_available"]
    assert not gate["passed"]


def test_missing_arm_and_bad_cohort_mask_rejected():
    scores = {arm: np.array([110., 105., 90.]) for arm in ARMS}
    missing = copy.deepcopy(scores)
    del missing["grouped_sham"]
    with pytest.raises(ValueError, match="three fixed"):
        run_report(missing)
    normal = {arm: np.arange(99.) for arm in ARMS}
    regs = {arm: registration(normal[arm], i) for i, arm in enumerate(ARMS)}
    with pytest.raises(ValueError, match="boolean mask"):
        analyze_label_results([1, 1, 0], ["Scanning", "DDoS", "Benign"], scores, normal, regs,
                              {"bad": np.array([1, 1, 0])})
