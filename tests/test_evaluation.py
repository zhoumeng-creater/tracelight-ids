import json

import numpy as np
import pytest

from ids.evaluation import calibrate_threshold, evaluate_binary


def test_ties_are_conservative_and_attack_scores_do_not_change_calibration():
    result = calibrate_threshold([0, 0, 0, 0, 1], [0.9, 0.8, 0.8, 0.1, 1], 0.5)
    assert result["allowed_false_positives"] == 2
    assert result["threshold"] > 0.8
    assert result["false_positives"] == 1
    assert result["achieved_fpr"] == 0.25
    without_attack = calibrate_threshold([0] * 4, [0.9, 0.8, 0.8, 0.1], 0.5)
    assert result["threshold"] == without_attack["threshold"]


def test_zero_budget_with_score_one_uses_valid_reject_all_threshold():
    calibration = calibrate_threshold([0, 0, 1], [0.2, 1.0, 1.0], 0)
    assert calibration["threshold"] > 1
    assert calibration["rejects_all_scores"] is True
    assert calibration["false_positives"] == 0
    result = evaluate_binary([0, 0, 1], [0.2, 1, 1], calibration["threshold"])
    assert result["confusion"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 0}


def test_float32_ties_remain_rejected_after_nextafter_calibration():
    scores = np.asarray([0.7, 0.7], dtype=np.float32)
    calibration = calibrate_threshold([0, 1], scores, 0)
    result = evaluate_binary([0, 1], scores, calibration["threshold"])
    assert result["confusion"] == {"tn": 1, "fp": 0, "fn": 1, "tp": 0}


def test_unit_budget_accepts_zero_scores_and_fractional_count_is_floored():
    assert calibrate_threshold([0, 0], [0, 1], 1)["threshold"] == 0
    assert calibrate_threshold([0] * 3, [0.1, 0.2, 0.3], 0.25)["allowed_false_positives"] == 0


def test_empirical_calibration_budget_holds_over_ties_and_extremes():
    rng = np.random.default_rng(12)
    for n in [1, 2, 17, 100]:
        scores = rng.integers(0, 11, size=n) / 10
        previous = float("inf")
        for alpha in [0, 0.001, 0.01, 0.1, 0.5, 1]:
            result = calibrate_threshold(np.zeros(n), scores, alpha)
            assert result["false_positives"] <= result["allowed_false_positives"]
            assert result["threshold"] <= previous
            previous = result["threshold"]


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_fpr_budgets_are_rejected(alpha):
    with pytest.raises(ValueError, match="max_fpr"):
        calibrate_threshold([0], [0.1], alpha)


def test_calibration_requires_normals():
    with pytest.raises(ValueError, match="normal"):
        calibrate_threshold([1, 1], [0.2, 0.8])
    with pytest.raises(ValueError, match="normal"):
        calibrate_threshold([], [])


@pytest.mark.parametrize(
    "labels,scores",
    [([0, 2], [0.1, 0.2]), ([0], [float("nan")]), ([0], [float("inf")]),
     ([0], [-0.1]), ([0], [1.01]), ([0, 1], [0.1]), ([[0]], [[0.1]])],
)
def test_invalid_inputs_are_rejected_by_both_entry_points(labels, scores):
    with pytest.raises(ValueError):
        calibrate_threshold(labels, scores)
    with pytest.raises(ValueError):
        evaluate_binary(labels, scores, 0.5)


def test_hand_computed_confusion_and_ranking_metrics():
    result = evaluate_binary([0, 0, 0, 1, 1, 1], [0.1, 0.7, 0.2, 0.8, 0.9, 0.4], 0.5)
    assert result["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert result["attack_recall"] == pytest.approx(2 / 3)
    assert result["normal_fpr"] == pytest.approx(1 / 3)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["average_precision"] == pytest.approx(11 / 12)
    assert result["roc_auc"] == pytest.approx(8 / 9)
    assert result["intervals"]["attack_recall"]["denominator"] == 3
    assert result["intervals"]["attack_recall"]["small_sample"] is True


def test_threshold_equality_is_attack_and_test_fpr_is_not_adjusted():
    calibration = calibrate_threshold([0, 0], [0.1, 0.2], 0)
    result = evaluate_binary([0, 1], [0.9, calibration["threshold"]], calibration["threshold"])
    assert result["normal_fpr"] == 1
    assert result["attack_recall"] == 1
    assert result["threshold"] == calibration["threshold"]


def test_undefined_metrics_are_null_and_wilson_does_not_claim_zero_risk():
    result = evaluate_binary([0] * 10, [0.1] * 10, 0.5)
    for key in ["attack_recall", "precision", "macro_f1", "average_precision", "roc_auc"]:
        assert result[key] is None
    interval = result["intervals"]["normal_fpr"]
    assert interval["low"] == pytest.approx(0)
    assert interval["high"] == pytest.approx(0.2775327998628892)
    assert interval["denominator"] == 10
    assert result["intervals"]["attack_recall"]["low"] is None
    json.dumps(result, allow_nan=False)


def test_only_attacks_have_defined_average_precision_but_not_roc_auc():
    result = evaluate_binary([1, 1], [0.1, 0.9], 0.5)
    assert result["normal_fpr"] is None
    assert result["average_precision"] == 1
    assert result["roc_auc"] is None


def test_per_family_recall_is_binary_detection_and_excludes_normal_categories():
    result = evaluate_binary(
        [0, 1, 1, 1], [0.9, 0.8, 0.1, 0.9], 0.5,
        attack_categories=["normal", "scan", "scan", "dos"],
    )
    by_category = {x["category"]: x for x in result["attack_categories"]}
    assert set(by_category) == {"scan", "dos"}
    assert by_category["scan"]["n_attacks"] == 2
    assert by_category["scan"]["detected"] == 1
    assert by_category["scan"]["recall"] == 0.5


def test_missing_category_and_literal_none_string_are_distinct():
    result = evaluate_binary([1, 1], [0.1, 0.8], 0.5, [None, "None"])
    assert len(result["attack_categories"]) == 2
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("threshold", [-1, float("inf"), float("nan")])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        evaluate_binary([0], [0.1], threshold)


def test_empty_evaluation_and_category_length_mismatch_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_binary([], [], 0.5)
    with pytest.raises(ValueError, match="attack_categories"):
        evaluate_binary([0, 1], [0.1, 0.9], 0.5, ["scan"])
