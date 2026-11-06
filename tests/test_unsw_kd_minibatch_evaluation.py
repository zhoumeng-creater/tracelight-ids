"""Synthetic H/K endpoint and optimization-interaction acceptance tests."""
import json

import numpy as np
import pandas as pd
import pytest

from ids.unsw_kd_minibatch_evaluation import evaluate_minibatch_endpoints


def fixture(gain=39, n_normal=200):
    frames = {}
    for role in ("reference", "constraint"):
        n_attack = 80
        frames[role] = pd.DataFrame({
            "sample_id": [f"{role}-{i}" for i in range(n_normal + n_attack)],
            "feature_hash": [f"{role}-pattern-{i}" for i in range(n_normal + n_attack)],
            "label": [0] * n_normal + [1] * n_attack,
            "attack_cat": ["Normal"] * n_normal + ["Attack"] * n_attack,
        })
    scores = {arm: {role: np.zeros(len(frame), dtype=np.float64)
                    for role, frame in frames.items()} for arm in ("H", "K")}
    scores["K"]["constraint"][n_normal:n_normal + gain] = 1.
    return frames, scores


@pytest.mark.parametrize("gain,quality,interaction", [
    (33, False, False), (34, True, False), (38, True, False), (39, True, True),
])
def test_distinct_frozen_quality_and_interaction_boundaries(gain, quality, interaction):
    frames, scores = fixture(gain)
    report, tables = evaluate_minibatch_endpoints(frames, scores)
    assert report["metric_acceptance_passed"] is quality
    assert report["interaction_effect_passed"] is interaction
    assert report["interaction_acceptance_passed"] is interaction
    assert report["interaction"]["historical_full_net_TP"] == 5
    assert report["interaction"]["net_TP_interaction"] == gain - 5
    assert report["interaction"]["minimum_mini_net_TP_for_interaction"] == 39
    assert report["parameters"]["minimum_net_tp"] == 34
    assert report["parameters"]["minimum_interaction_tp"] == 34
    assert set(report["endpoints"]) == {"H", "K"}
    assert set(report["comparisons"]) == {"K-H"}
    assert set(tables) == {"K-H_reference", "K-H_constraint", "K-H_attack_families",
                           "K-H_constraint_feature_groups"}
    events = tables["K-H_constraint"]
    assert len(events) == gain and set(events.event) == {"recovered_fn"}
    assert events.candidate_score.dtype == np.float64
    assert tables["K-H_constraint_feature_groups"].net_true_positives.sum() == gain
    assert set(tables["K-H_attack_families"].role) == {"reference", "constraint"}
    assert report["development_only"] and not report["independent_confirmation"]
    assert "new_optimizer_S_M_controls" in report["not_evaluated"]
    assert "formal_CPU_cost" in report["not_evaluated"]
    assert not any("cost" in gate.lower() for gate in report["checks"])
    json.dumps(report, allow_nan=False)


def test_interaction_delta_uses_the_bound_historical_count():
    frames, scores = fixture(39)
    report, _ = evaluate_minibatch_endpoints(frames, scores, historical_full_delta=np.int64(6))
    assert report["metric_acceptance_passed"]
    assert report["interaction"]["net_TP_interaction"] == 33
    assert report["interaction"]["minimum_mini_net_TP_for_interaction"] == 40
    assert not report["interaction_acceptance_passed"]


@pytest.mark.parametrize("historical", [True, 5., np.nan, "5"])
def test_historical_delta_requires_an_integer(historical):
    frames, scores = fixture()
    with pytest.raises(ValueError, match="historical_full_delta"):
        evaluate_minibatch_endpoints(frames, scores, historical_full_delta=historical)


@pytest.mark.parametrize("role", ["reference", "constraint"])
def test_supported_family_regression_blocks_interaction_claim(role):
    frames, scores = fixture(40)
    frames[role].loc[240, "attack_cat"] = "Rare"
    scores["H"][role][240] = 1.
    if role == "reference":
        scores["K"][role][200:202] = 1.
    report, tables = evaluate_minibatch_endpoints(frames, scores)
    pair = report["comparisons"]["K-H"][role]
    assert pair["attack_families"]["Rare"]["support"] == 1
    assert pair["attack_families"]["Rare"]["net_true_positives"] == -1
    assert report["checks"]["C_gain_K_H"]
    assert report["checks"]["R_TP_K_H"]
    assert report["interaction_effect_passed"]
    assert not report["checks"][f"family_TP_K_H_{role}"]
    assert not report["metric_acceptance_passed"]
    assert not report["interaction_acceptance_passed"]
    assert (tables[f"K-H_{role}"].event == "new_fn").sum() == 1


def test_reference_total_regression_blocks_positive_constraint_gain():
    frames, scores = fixture()
    scores["H"]["reference"][200] = 1.
    report, _ = evaluate_minibatch_endpoints(frames, scores)
    assert report["interaction_effect_passed"]
    assert not report["checks"]["R_TP_K_H"]
    assert not report["checks"]["R_pauc_K_H"]
    assert not report["interaction_acceptance_passed"]


def test_pauc_regression_blocks_positive_threshold_count_gain():
    frames, scores = fixture(0)
    for arm in scores:
        scores[arm]["constraint"][:2] = 2.
    scores["H"]["constraint"][200:206] = 3.
    scores["K"]["constraint"][200:245] = 1.
    report, _ = evaluate_minibatch_endpoints(frames, scores)
    assert report["comparisons"]["K-H"]["constraint"]["net_true_positives"] == 39
    assert report["checks"]["C_gain_K_H"]
    assert report["interaction_effect_passed"]
    assert not report["checks"]["C_pauc_K_H"]
    assert not report["interaction_acceptance_passed"]


def test_single_pattern_gain_fails_after_deleting_largest_positive_group():
    frames, scores = fixture()
    frames["constraint"].loc[200:238, "feature_hash"] = "single-pattern"
    report, tables = evaluate_minibatch_endpoints(frames, scores)
    concentration = report["concentration"]
    assert concentration["largest_group_net_true_positives"] == 39
    assert concentration["largest_group_feature_hashes"] == ["single-pattern"]
    assert concentration["remaining_net_true_positives"] == 0
    assert concentration["thresholds_stay_fixed"]
    assert report["interaction_effect_passed"]
    assert not report["checks"]["concentration_K_H_C"]
    assert not report["interaction_acceptance_passed"]
    groups = tables["K-H_constraint_feature_groups"]
    assert groups.loc[groups.feature_hash == "single-pattern", "attack_rows"].item() == 39


def test_normal_error_exchanges_are_saved_even_when_their_net_change_is_zero():
    frames, scores = fixture()
    for role in frames:
        scores["H"][role][:2] = 2.
        scores["K"][role][2:4] = 2.
    report, tables = evaluate_minibatch_endpoints(frames, scores)
    assert report["interaction_acceptance_passed"]
    for role in frames:
        pair = report["comparisons"]["K-H"][role]
        assert pair["repaired_fp"] == 2 and pair["new_fp"] == 2
        assert pair["net_false_positives"] == 0
        assert (tables[f"K-H_{role}"].event == "repaired_fp").sum() == 2
        assert (tables[f"K-H_{role}"].event == "new_fp").sum() == 2


@pytest.mark.parametrize("role", ["reference", "constraint"])
def test_additional_false_positive_fails_even_inside_absolute_cap(role):
    frames, scores = fixture()
    scores["K"][role][0] = 2.
    report, _ = evaluate_minibatch_endpoints(frames, scores)
    assert report["checks"][f"FP_budget_K_{role}"]
    assert report["comparisons"]["K-H"][role]["net_false_positives"] == 1
    assert not report["checks"][f"FP_K_H_{role}"]
    assert not report["interaction_acceptance_passed"]


@pytest.mark.parametrize("role,cap", [("reference", 39), ("constraint", 78)])
def test_absolute_false_positive_caps_are_frozen(role, cap):
    frames, scores = fixture(n_normal=8000)
    for arm in scores:
        scores[arm][role][:cap + 1] = 2.
    report, _ = evaluate_minibatch_endpoints(frames, scores)
    for arm in scores:
        assert report["endpoints"][arm]["roles"][role]["metrics"]["false_positives"] == cap + 1
        assert not report["checks"][f"FP_budget_{arm}_{role}"]
    assert not report["interaction_acceptance_passed"]


def test_max_R_C_normal_threshold_and_float64_boundary_are_preserved():
    frames, scores = fixture(0)
    above_one = np.nextafter(np.float64(1), np.float64(np.inf))
    above_two = np.nextafter(np.float64(2), np.float64(np.inf))
    scores["H"]["constraint"][:200] = 2.
    scores["K"]["reference"][:200] = 1.
    scores["K"]["constraint"][200:202] = [1., above_one]
    report, tables = evaluate_minibatch_endpoints(frames, scores)
    assert report["endpoints"]["H"]["working_point"]["threshold"] == above_two
    assert report["endpoints"]["K"]["working_point"]["threshold"] == above_one
    for arm in scores:
        for role in frames:
            assert report["endpoints"][arm]["roles"][role]["metrics"]["false_positives"] == 0
    events = tables["K-H_constraint"]
    assert events.sample_id.tolist() == ["constraint-201"]
    assert events.candidate_threshold.item() == above_one
    assert events.candidate_score.item() == above_one


@pytest.mark.parametrize("bad", [
    "extra_role", "extra_arm", "missing_arm", "extra_score_role", "unaligned", "nonfinite",
    "matrix", "nonnumeric", "missing_identity", "blank_identity", "duplicate_id",
    "one_class", "cross_role_id", "cross_role_feature",
])
def test_invalid_input_is_a_technical_error_not_a_negative_experiment(bad):
    frames, scores = fixture()
    if bad == "extra_role":
        frames["validation"] = frames["constraint"].copy()
    elif bad == "extra_arm":
        scores["S"] = scores["H"]
    elif bad == "missing_arm":
        scores.pop("K")
    elif bad == "extra_score_role":
        scores["K"]["test"] = scores["K"]["reference"]
    elif bad == "unaligned":
        scores["K"]["constraint"] = scores["K"]["constraint"][:-1]
    elif bad == "nonfinite":
        scores["K"]["reference"][0] = np.inf
    elif bad == "matrix":
        scores["K"]["reference"] = scores["K"]["reference"][:, None]
    elif bad == "nonnumeric":
        scores["K"]["reference"] = scores["K"]["reference"].astype(str)
    elif bad == "missing_identity":
        frames["reference"].loc[0, "feature_hash"] = None
    elif bad == "blank_identity":
        frames["reference"].loc[0, "sample_id"] = " "
    elif bad == "duplicate_id":
        frames["reference"].loc[0, "sample_id"] = frames["reference"].loc[1, "sample_id"]
    elif bad == "one_class":
        frames["reference"]["label"] = 0
    elif bad == "cross_role_id":
        frames["reference"].loc[0, "sample_id"] = frames["constraint"].loc[0, "sample_id"]
    else:
        frames["reference"].loc[0, "feature_hash"] = frames["constraint"].loc[0, "feature_hash"]
    with pytest.raises(ValueError):
        evaluate_minibatch_endpoints(frames, scores)
