"""Synthetic tests of train-only teacher value and exact-input limitations."""
import itertools
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ids.transfer_diagnostics import (
    binary_rank_metrics,
    exact_input_ambiguity,
    training_threshold,
    transfer_diagnostics,
)


def _inputs():
    return {"calibration_labels": [0, 0, 0, 0, 1, 1],
        "calibration_teacher_scores": [-3., -2., -1., 0., 2., 3.],
        "calibration_student_scores": [-3., -2., -1., 0., 2., 3.],
        "probe_labels": [1, 1, 1, 1, 0, 0, 0, 0],
        "probe_teacher_scores": [1., -1., -1., 1., -1., 1., 1., -1.],
        "probe_student_scores": [-1., 1., -1., 1., 1., -1., 1., -1.],
        "calibration_ids": [f"R{i}" for i in range(6)], "probe_ids": [f"C{i}" for i in range(8)],
        "probe_families": ["A", "A", "B", "B"] + ["Normal"] * 4, "max_fpr": .25}


def test_corrections_harms_and_different_fprs_remain_visible():
    row = transfer_diagnostics(**_inputs())
    assert row["fits"] == 0
    assert row["paired"]["teacher_corrects_student_fn"] == 1
    assert row["paired"]["teacher_loses_student_tp"] == 1
    assert row["paired"]["teacher_repairs_student_fp"] == 1
    assert row["paired"]["teacher_adds_student_fp"] == 1
    assert row["paired"]["teacher_minus_student_tp"] == row["paired"]["teacher_minus_student_fp"] == 0
    assert row["paired"]["probe_fprs_need_not_be_equal"]
    assert row["per_attack_family"]["A"]["teacher_corrects_student_fn"] == 1
    assert row["per_attack_family"]["A"]["teacher_loses_student_tp"] == 1
    assert row["teacher"]["calibration"]["achieved_fpr"] == .25
    assert row["teacher"]["probe"]["fpr"] == .5
    assert row["teacher_family_support"]["known"] is False
    assert row["exact_input_ambiguity"]["status"] == "not_measured"
    json.dumps(row, allow_nan=False)


def test_each_predictor_calibrates_in_its_own_coordinates_not_on_probe():
    original = transfer_diagnostics(**_inputs())
    data = _inputs()
    for prefix in ("calibration", "probe"):
        data[f"{prefix}_teacher_scores"] = np.asarray(data[f"{prefix}_teacher_scores"]) + 12
        data[f"{prefix}_student_scores"] = np.asarray(data[f"{prefix}_student_scores"]) - 7
    shifted = transfer_diagnostics(**data)
    assert shifted["teacher"]["calibration"]["threshold"] == pytest.approx(11.)
    assert shifted["student"]["calibration"]["threshold"] == pytest.approx(-8.)
    assert shifted["paired"] == original["paired"]
    for model in ("teacher", "student"):
        assert shifted[model]["probe"]["tpr"] == original[model]["probe"]["tpr"]
    data["probe_labels"] = [0, 1] * 4
    relabeled = transfer_diagnostics(**data)
    for model in ("teacher", "student"):
        assert relabeled[model]["calibration"] == shifted[model]["calibration"]


def test_float32_plateau_conservative_threshold_stays_above_plateau():
    score = np.array([-13.81551] * 10, dtype=np.float32)
    frozen = training_threshold(score, np.zeros(10, dtype=int), .1)
    assert frozen["allowed_false_positives"] == 1
    assert frozen["false_positives"] == 0
    assert frozen["threshold"] > float(score[0])
    # Comparing in native f32 would round the threshold back, the original bug.
    assert np.count_nonzero(score.astype(np.float64) >= frozen["threshold"]) == 0


@pytest.mark.parametrize("alpha", [.01, .1, .5, .9])
def test_rank_metrics_match_independent_sklearn_oracle_for_ties(alpha):
    for seed in range(12):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, 200)
        scores = rng.integers(-4, 5, 200).astype(float)
        row = binary_rank_metrics(labels, scores, alpha)
        assert row["roc_auc"] == pytest.approx(roc_auc_score(labels, scores), abs=1e-14)
        assert row["partial_auc_standardized"] == pytest.approx(
            roc_auc_score(labels, scores, max_fpr=alpha), abs=1e-14)


def test_rank_metrics_separate_raw_area_standardized_area_and_missing_support():
    perfect = binary_rank_metrics([0, 0, 1, 1], [0., 0., 1., 1.], .1)
    assert perfect["partial_auc_raw"] == .1
    assert perfect["partial_auc_standardized"] == 1.
    constant = binary_rank_metrics([0, 1], [0., 0.], .1)
    assert constant["partial_auc_raw"] == pytest.approx(.005)
    assert constant["partial_auc_standardized"] == .5
    single_class = binary_rank_metrics([1, 1], [0., 1.])
    assert single_class["roc_auc"] is single_class["partial_auc_standardized"] is None
    json.dumps(single_class, allow_nan=False)


def test_exact_input_alarm_floor_matches_exhaustive_group_constant_decisions():
    scores = np.array([-2., 2., 3., -4., -1., 5., 5.])
    groups = np.array(["a", "a", "a", "b", "b", "c", "c"])
    row = exact_input_ambiguity(scores, groups, 0.)
    y = scores >= 0
    group_names = sorted(set(groups))
    oracle = min(np.count_nonzero(y != np.array([dict(zip(group_names, bits))[g] for g in groups]))
                 for bits in itertools.product([False, True], repeat=3))
    assert row["minimum_teacher_alarm_imitation_errors"] == oracle == 1
    assert row["n_groups_with_conflicting_teacher_alarms"] == 1
    # Group a has 3 unequal pairs; b has 1; c's equal teacher scores add 0.
    assert row["within_identical_input_teacher_unequal_pairs"] == 4
    expected = np.mean(np.concatenate([scores[groups == g] - scores[groups == g].mean()
                                      for g in group_names]) ** 2)
    assert row["minimum_teacher_score_mse"] == pytest.approx(expected)
    assert row["absence_of_conflicts_does_not_establish_representability"]


def test_exact_input_missing_or_singleton_groups_never_claim_general_representability():
    assert exact_input_ambiguity([1., 2.], None, 0.)["status"] == "not_measured"
    row = exact_input_ambiguity([1., 2.], ["one", "two"], 0.)
    assert row["n_rows_in_non_singleton_groups"] == row["minimum_teacher_alarm_imitation_errors"] == 0
    assert row["minimum_teacher_score_mse"] == 0.
    assert row["absence_of_conflicts_does_not_establish_representability"]


def test_many_exact_input_groups_and_teacher_ties_use_same_finite_floor():
    groups = np.repeat(np.arange(2000).astype(str), 4)
    score = np.tile([-2., -2., 2., 2.], 2000)
    row = exact_input_ambiguity(score, groups, 0.)
    assert row["minimum_teacher_alarm_imitation_errors"] == 4000
    assert row["within_identical_input_teacher_unequal_pairs"] == 8000
    assert row["minimum_teacher_score_mse"] == 4.


def test_equal_extreme_scores_have_zero_input_variance_without_sum_overflow():
    row = exact_input_ambiguity([1e308] * 3, ["same"] * 3, 0.)
    assert row["minimum_teacher_score_mse"] == 0.
    with pytest.raises(ValueError, match="overflows"):
        exact_input_ambiguity([-1e308, 1e308], ["same", "same"], 0.)


def test_single_teacher_family_support_is_distinct_from_binary_label_support():
    data = _inputs()
    data["teacher_fit_family_counts"] = {"A": 0, "B": 12, "Normal": 20}
    row = transfer_diagnostics(**data)
    assert row["teacher_family_support"]["n_attack_rows_with_family_absent_in_teacher_fit"] == 2
    assert not row["teacher_family_support"]["family_is_binary_training_target"]
    assert row["per_attack_family"]["A"]["n_rows_with_family_absent_in_teacher_fit"] == 2
    # A complete count map missing B means zero; an unknown map is None.
    data["teacher_fit_family_counts"] = {"A": 2}
    assert transfer_diagnostics(**data)["per_attack_family"]["B"]["n_rows_with_family_absent_in_teacher_fit"] == 2


def test_oof_per_row_family_counts_do_not_hide_fold_specific_absence():
    data = _inputs()
    data["teacher_fit_family_counts"] = [0, 20, 5, 0, 10, 10, 10, 10]
    row = transfer_diagnostics(**data)
    assert row["teacher_family_support"]["n_attack_rows_with_family_absent_in_teacher_fit"] == 2
    assert row["per_attack_family"]["A"]["n_rows_with_family_absent_in_teacher_fit"] == 1
    assert row["per_attack_family"]["B"]["n_rows_with_family_absent_in_teacher_fit"] == 1


def test_saturation_and_rank_constant_state_are_json_safe():
    data = _inputs()
    data["probe_teacher_scores"] = [-1000.] * 4 + [1000.] * 4
    data["probe_student_scores"] = [0.] * 8
    row = transfer_diagnostics(**data)
    assert row["teacher"]["score_profile"]["1"]["sigmoid_lower_mass"] == 1.
    assert row["teacher"]["score_profile"]["0"]["sigmoid_upper_mass"] == 1.
    assert row["teacher"]["score_profile"]["0"]["largest_tie_fraction"] == 1.
    assert row["rank_agreement"]["all"]["kendall_tau_b"] is None
    assert row["rank_agreement"]["teacher_normal_tail"]["n"] == 4
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("field,value,match", [
    ("calibration_role", "validation", "train_calibration"),
    ("probe_role", "test", "no validation/test"),
    ("probe_ids", ["R0"] + [f"C{i}" for i in range(7)], "overlap"),
    ("probe_ids", ["x"] * 8, "unique"),
    ("probe_labels", [2] * 8, "binary"),
    ("probe_teacher_scores", [float("nan")] * 8, "finite"),
    ("probe_student_scores", [0.] * 7, "aligned"),
    ("max_fpr", 0., "strictly"),
    ("max_fpr", True, "strictly"),
    ("teacher_fit_family_counts", [0.] * 8, "integer counts"),
    ("teacher_fit_family_counts", [-1] * 8, "nonnegative"),
])
def test_malformed_roles_and_pool_identity_fail_before_diagnostics(field, value, match):
    data = _inputs()
    data[field] = value
    with pytest.raises(ValueError, match=match):
        transfer_diagnostics(**data)


def test_threshold_requires_normal_support_and_finite_nextafter():
    with pytest.raises(ValueError, match="normal support"):
        training_threshold([0., 1.], [1, 1])
    with pytest.raises(ValueError, match="representable finite"):
        training_threshold([np.finfo(float).max], [0])


def test_probe_can_lack_one_class_without_silently_inventing_auc():
    data = _inputs()
    data["probe_labels"] = [0] * 8
    row = transfer_diagnostics(**data)
    assert row["teacher"]["probe"]["tpr"] is None
    assert row["teacher"]["probe"]["partial_auc_standardized"] is None
    assert row["paired"]["teacher_minus_student_partial_auc_standardized"] is None
    assert row["per_attack_family"] == {}
    json.dumps(row, allow_nan=False)
