"""Synthetic, independently calculated checks for mechanism reporting."""

import json

import numpy as np
import pandas as pd
import pytest

from ids.context_cache.calibration import register_threshold
from ids.context_cache.preprocessing import CSVSchema
from ids.context_cache.scanning_metrics import (
    analyze_arm, build_cohorts, empirical_percentiles, scanning_roc,
)


def registration(scores, model="a"):
    return register_threshold(scores, .01, model_sha256=model * 64,
                              preprocessing_sha256="b" * 64, calibration_sha256="c" * 64)


@pytest.mark.parametrize("scores,auc,partial", [
    ([4, 3, 2, 1], 1.0, .01),
    ([1, 2, 3, 4], 0.0, 0.0),
    ([1, 1, 1, 1], .5, .00005),
    ([4, 2, 3, 1], .75, .005),
    ([4, 2, 4, 1], .625, .00005),
])
def test_manual_roc_and_boundary_interpolation(scores, auc, partial):
    result = scanning_roc([1, 1, 0, 0], ["Scanning", "Scanning", "Benign", "Benign"], scores)
    assert result["auc"] == pytest.approx(auc)
    assert result["partial_auc_raw"] == pytest.approx(partial)
    assert result["partial_auc_divided_by_width"] == pytest.approx(partial / .01)


def test_roc_groups_ties_and_ignores_other_attack_families():
    result = scanning_roc([1, 0, 1, 1], ["Scanning", "Benign", "DDoS", "DDoS"], [1, 1, 50, -50])
    assert result["n_scanning"] == result["n_normal"] == 1
    assert result["auc"] == .5
    assert result["partial_auc_divided_by_width"] == pytest.approx(.005)


@pytest.mark.parametrize("truth,families", [([1], ["Scanning"]), ([0], ["Benign"])])
def test_single_class_roc_is_undefined(truth, families):
    result = scanning_roc(truth, families, [2])
    assert result["auc"] is None
    assert result["partial_auc_raw"] is None


def test_percentiles_include_ties_and_use_own_calibration():
    assert empirical_percentiles([-1, 1, 2, 4], [1, 1, 3, 4]).tolist() == [0, .5, .5, 1]
    with pytest.raises(ValueError, match="nonempty"):
        empirical_percentiles([0], [])
    with pytest.raises(ValueError, match="finite"):
        empirical_percentiles([np.nan], [0])


@pytest.fixture
def cohort_fixture():
    features = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "signal"]
    schema = CSVSchema.from_dict({"features": features, "family_column": "attack_family"})
    columns = features + ["label", "sample_id", "attack_family"]
    support = pd.DataFrame([
        [100, 443, 6, 91, 1, 1, "p0", "Scanning"],
        [300, 443, 6, 91, 1, 0, "p1", "Benign"],
    ], columns=columns)
    evaluation = pd.DataFrame([
        [100, 443, 6, 91, 1, 1, "e0", "Scanning"],
        [200, 443, 6, 91, 1, 1, "e1", "Scanning"],
        [300, 443, 6, 91, 1, 0, "e2", "Benign"],
        [500, 53, 17, 1, 4, 1, "e3", "Scanning"],
    ], columns=columns)
    archived = pd.DataFrame({"row_id": evaluation.sample_id, "label": evaluation.label,
                             "attack_family": evaluation.attack_family,
                             "legacy_full_predicted": ["True", "True", "False", "True"],
                             "full_predicted": ["False", "False", "False", "True"]})
    return {"support": support, "evaluation": evaluation}, schema, archived


def test_cohorts_preserve_order_and_distinguish_exact_support(cohort_fixture):
    masks = build_cohorts(*cohort_fixture)
    assert masks["old_54_lost"].tolist() == [True, True, False, False]
    assert masks["15_exact_P_positive_matches"].tolist() == [True, False, False, False]
    assert masks["TCP6_L791_DST443"].tolist() == [True, True, True, False]
    assert masks["dominant_47_scanning"].tolist() == [True, True, False, False]
    assert masks["dominant_47_normal"].tolist() == [False, False, True, False]


@pytest.mark.parametrize("mutation", ["order", "label", "family", "boolean"])
def test_cohorts_reject_misaligned_or_invalid_archive(cohort_fixture, mutation):
    frames, schema, archived = cohort_fixture
    if mutation == "order":
        archived = archived.iloc[::-1]
    elif mutation == "label":
        archived.loc[0, "label"] = 0
    elif mutation == "family":
        archived.loc[0, "attack_family"] = "DDoS"
    else:
        archived.loc[0, "legacy_full_predicted"] = "maybe"
    with pytest.raises(ValueError):
        build_cohorts(frames, schema, archived)


def test_empty_lost_cohort_has_empty_dominant_pattern(cohort_fixture):
    frames, schema, archived = cohort_fixture
    archived["full_predicted"] = True
    cohorts = build_cohorts(frames, schema, archived)
    assert not cohorts["old_54_lost"].any()
    assert not cohorts["dominant_47_field_pattern"].any()


def test_report_separates_scores_thresholds_percentiles_and_paired_changes():
    normal = np.arange(99, dtype=float)
    current_c = normal + 10
    before_scores = np.array([99., 97., 99., 97.])
    scores = np.array([107., 109., 107., 109.])
    result = analyze_arm(
        [1, 1, 0, 0], ["Scanning", "Scanning", "Benign", "Benign"], scores,
        current_c, registration(current_c, "d"), before_scores, registration(normal),
        {"scan": np.array([True, True, False, False]), "empty": np.zeros(4, dtype=bool)}, normal,
    )
    assert result["metrics"]["tp"] == result["metrics"]["fp"] == 1
    assert result["paired_changes"] == {"repaired_fn": 1, "new_fn": 1, "repaired_fp": 1,
                                         "new_fp": 1, "net_tp": 0, "net_fp": 0, "changed_decisions": 4}
    assert result["calibration"]["threshold_delta"] == 10
    assert result["cohorts"]["scan"]["score_delta"]["mean"] == 10
    assert result["cohorts"]["scan"]["calibration_percentile_delta"]["mean"] == 0
    assert result["cohorts"]["empty"]["score_delta"]["mean"] is None
    json.dumps(result, allow_nan=False)


def test_tolerance_is_relative_to_baseline_and_strict():
    normal = np.arange(99, dtype=float)
    baseline = np.array([1., 1.])
    values = baseline + np.array([5e-5, 7e-5])
    result = analyze_arm([1, 0], ["Scanning", "Benign"], values, normal, registration(normal),
                         baseline, registration(normal), {"all": np.ones(2, dtype=bool)}, normal)
    assert result["score_changes_beyond_tolerance"] == 1
    assert result["cohorts"]["all"]["score_changes_beyond_tolerance"] == 1
    assert result["paired_changes"]["changed_decisions"] == 0


def test_report_rejects_wrong_calibration_and_misaligned_mask():
    normal = np.arange(99, dtype=float)
    common = ([1, 0], ["Scanning", "Benign"], [2, 0])
    with pytest.raises(ValueError, match="calibration scores"):
        analyze_arm(*common, normal + 1, registration(normal), [2, 0], registration(normal), {})
    with pytest.raises(ValueError, match="boolean mask"):
        analyze_arm(*common, normal, registration(normal), [2, 0], registration(normal), {"bad": [1, 0]})


def test_report_rejects_changed_threshold_even_when_calibration_decisions_match():
    normal = np.arange(99, dtype=float)
    wrong = registration(normal)
    wrong["threshold"] += .25
    wrong["boundary_ties"] = 0
    with pytest.raises(ValueError, match="fixed calibration rank"):
        analyze_arm([1, 0], ["Scanning", "Benign"], [98.1, 0], normal, wrong,
                    [98.1, 0], registration(normal), {})
