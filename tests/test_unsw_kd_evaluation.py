"""Synthetic endpoint checks: effect thresholds, harmed families and concentration."""
import json

import numpy as np
import pandas as pd
import pytest

from ids.unsw_kd_evaluation import evaluate_endpoints


def fixture(gain=34):
    frames = {}
    for role in ("reference", "constraint"):
        n0, n1 = 200, 40
        frames[role] = pd.DataFrame({
            "sample_id": [f"{role}-{i}" for i in range(n0+n1)],
            "feature_hash": [f"{role}-pattern-{i}" for i in range(n0+n1)],
            "label": [0]*n0 + [1]*n1,
            "attack_cat": ["Normal"]*n0 + ["Attack"]*n1,
        })
    scores = {arm: {role: np.zeros(len(frame), dtype=np.float64)
                    for role, frame in frames.items()} for arm in ("H", "K", "S", "M")}
    scores["K"]["constraint"][200:200+gain] = 1.
    return frames, scores


@pytest.mark.parametrize("gain,passed", [(33, False), (34, True)])
def test_default_34_net_tp_boundary_and_three_controls(gain, passed):
    frames, scores = fixture(gain)
    report, events = evaluate_endpoints(frames, scores)
    assert report["metric_acceptance_passed"] is passed
    assert report["parameters"]["minimum_net_tp"] == 34
    assert len(events) == 6
    for control in ("H", "S", "M"):
        assert report["checks"][f"C_gain_K_{control}"] is passed
        table = events[f"K-{control}_constraint"]
        assert len(table) == gain
        assert set(table.event) == {"recovered_fn"}
        assert table.candidate_score.dtype == np.float64
    # No execution, source binding or resource gate is fabricated by this module.
    assert "formal_CPU_cost" in report["not_evaluated"]
    json.dumps(report, allow_nan=False)


def test_win_against_hard_does_not_replace_other_two_comparisons():
    frames, scores = fixture()
    scores["M"]["constraint"][200:233] = 1.
    report, _ = evaluate_endpoints(frames, scores)
    assert report["checks"]["C_gain_K_H"]
    assert not report["checks"]["C_gain_K_M"]
    assert report["comparisons"]["K-M"]["constraint"]["net_true_positives"] == 1
    assert not report["metric_acceptance_passed"]
    assert "quality gate" in report["interpretation"]


def test_harmed_small_family_fails_despite_positive_total_and_pauc():
    frames, scores = fixture(gain=0)
    frames["constraint"].loc[200, "attack_cat"] = "Rare"
    for arm in ("H", "S", "M"):
        scores[arm]["constraint"][200] = 1.
    scores["K"]["constraint"][201:236] = 1.
    report, tables = evaluate_endpoints(frames, scores)
    pair = report["comparisons"]["K-H"]["constraint"]
    assert pair["net_true_positives"] == 34
    assert pair["recovered_fn"] == 35 and pair["new_fn"] == 1
    assert pair["attack_families"]["Rare"]["support"] == 1
    assert pair["attack_families"]["Rare"]["net_true_positives"] == -1
    assert report["checks"]["C_gain_K_H"] and report["checks"]["C_pauc_K_H"]
    assert not report["checks"]["family_TP_K_H_constraint"]
    assert not report["metric_acceptance_passed"]
    assert len(tables["K-H_constraint"]) == 36


def test_one_repeated_pattern_cannot_pass_distributed_gain_gate():
    frames, scores = fixture()
    frames["constraint"].loc[200:233, "feature_hash"] = "single-repeated-pattern"
    report, _ = evaluate_endpoints(frames, scores)
    concentration = report["concentration"]
    assert concentration["net_true_positives"] == 34
    assert concentration["largest_group_net_true_positives"] == 34
    assert concentration["remaining_net_true_positives"] == 0
    assert concentration["thresholds_stay_fixed"] is True
    assert concentration["largest_group_feature_hashes"] == ["single-repeated-pattern"]
    assert report["checks"]["C_gain_K_H"]
    assert not report["checks"]["concentration_K_H_C"]
    assert not report["metric_acceptance_passed"]


def test_normal_error_exchange_remains_visible_even_with_zero_net_change():
    frames, scores = fixture()
    for role in frames:
        for arm in ("H", "S", "M"):
            scores[arm][role][:2] = 2.
        scores["K"][role][2:4] = 2.
    report, tables = evaluate_endpoints(frames, scores)
    assert report["metric_acceptance_passed"]
    for role in frames:
        pair = report["comparisons"]["K-H"][role]
        assert pair["repaired_fp"] == 2 and pair["new_fp"] == 2
        assert pair["net_false_positives"] == 0
        events = tables[f"K-H_{role}"]
        assert (events.event == "repaired_fp").sum() == 2
        assert (events.event == "new_fp").sum() == 2


def test_maximum_of_R_C_and_float64_nextafter_boundary_are_used():
    frames, scores = fixture(gain=0)
    for arm in scores:
        scores[arm]["reference"][:200] = 1.
    above_one = np.nextafter(np.float64(1), np.float64(np.inf))
    scores["K"]["constraint"][200:202] = [1., above_one]
    report, tables = evaluate_endpoints(frames, scores, minimum_net_tp=1)
    for arm in scores:
        endpoint = report["endpoints"][arm]
        assert endpoint["working_point"]["threshold"] == above_one
        assert endpoint["roles"]["reference"]["metrics"]["false_positives"] == 0
    metric = report["endpoints"]["K"]["roles"]["constraint"]["metrics"]
    assert metric["true_positives"] == 1
    table = tables["K-H_constraint"]
    assert table.sample_id.tolist() == ["constraint-201"]
    assert table.candidate_threshold.iloc[0] == above_one
    assert table.candidate_score.iloc[0] == above_one
    assert not report["checks"]["concentration_K_H_C"]


def test_reference_regression_stops_an_otherwise_positive_C_result():
    frames, scores = fixture()
    scores["H"]["reference"][200] = 1.
    report, _ = evaluate_endpoints(frames, scores)
    assert report["checks"]["C_gain_K_H"]
    assert not report["checks"]["R_TP_K_H"]
    assert not report["checks"]["R_pauc_K_H"]
    assert not report["metric_acceptance_passed"]


def test_C_pauc_gate_can_fail_despite_34_more_threshold_detections():
    frames, scores = fixture(gain=0)
    for arm in scores:
        scores[arm]["constraint"][:2] = 2.
    for arm in ("H", "S", "M"):
        scores[arm]["constraint"][200:206] = 3.
    scores["K"]["constraint"][200:] = 1.
    report, _ = evaluate_endpoints(frames, scores)
    assert report["comparisons"]["K-H"]["constraint"]["net_true_positives"] == 34
    assert report["checks"]["C_gain_K_H"]
    assert not report["checks"]["C_pauc_K_H"]
    assert not report["metric_acceptance_passed"]


@pytest.mark.parametrize("bad", ["extra_role", "missing_arm", "unaligned", "nonfinite"])
def test_technical_input_errors_are_not_scientific_negative_results(bad):
    frames, scores = fixture()
    if bad == "extra_role":
        frames["validation"] = frames["constraint"].copy()
    elif bad == "missing_arm":
        scores.pop("S")
    elif bad == "unaligned":
        scores["M"]["constraint"] = scores["M"]["constraint"][:-1]
    else:
        scores["K"]["reference"][0] = np.nan
    with pytest.raises(ValueError):
        evaluate_endpoints(frames, scores)
