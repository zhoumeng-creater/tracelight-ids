"""Synthetic policy tests; no real model or archived score computation."""
import copy

import numpy as np
import pandas as pd
import pytest

from ids.model_training.evaluation import (
    REGISTERED_D_CELLS, calibrate, compare, evaluate, evaluate_decisions, legacy_rule,
)


def frame_for(source="example", normal=100, attack=2, *, port=443):
    size = normal + attack
    return pd.DataFrame({
        "source_id": [source] * size, "row_id": np.arange(size),
        "label": [0] * normal + [1] * attack,
        "family": ["Benign"] * normal + ["ExternalC2"] * attack,
        "flow_end_ms": np.repeat(1700000000000, size),
        "ip_protocol": np.repeat(17, size), "orig_port": np.repeat(12345, size),
        "resp_port": np.repeat(port, size), "duration_seconds": np.ones(size),
    })


@pytest.fixture
def comparison():
    b = [f"TQH-B_i3600_j{j}" for j in (0, 30, 70)]
    d = list(REGISTERED_D_CELLS)
    frame = pd.concat([frame_for(cell) for cell in b+d], ignore_index=True)
    design = {"roles": {"E": {"cells": b, "normal": 300, "attack": 6},
                         "D": {"cells": d, "normal": 1200, "attack": 24}},
              "training": {"neural_seeds": [17, 29, 43]}}
    all_attack = frame.label.eq(1).to_numpy()
    weaker = all_attack & (frame.source_id.isin(b) | frame.row_id.eq(100)).to_numpy()
    decisions = {}
    for model_id, alerts in [("hgb", weaker), ("legacy_hgb_or_rule", weaker)]:
        decisions[model_id] = evaluate_decisions(frame, alerts, model_id=model_id)[2]
    for seed in (17, 29, 43):
        for variant, alerts in (("ple", all_attack), ("no_ple", weaker)):
            model_id = f"tabm_{variant}_s{seed}"
            decisions[model_id] = evaluate_decisions(frame, alerts, model_id=model_id)[2]
    return frame, design, decisions


def test_calibration_exact_existing_rank_and_strict_ties():
    rule = calibrate(np.arange(5511, dtype=float))
    assert rule["rank"] == 5457
    assert rule["threshold"] == 5456
    assert rule["calibration_fp"] == 54
    tied = calibrate(np.zeros(100))
    unit, _, decisions = evaluate(frame_for(normal=100, attack=2), np.zeros(102), tied, model_id="tied")
    assert not decisions.alert.any()
    assert unit[0]["qualified"]


def test_insufficient_calibration_is_unready_not_zero_alert_success():
    rule = calibrate(np.arange(10))
    assert rule["threshold"] is None and not rule["ready"]
    unit, _, _ = evaluate(frame_for(), np.ones(102), rule, model_id="unready")
    assert unit[0]["fp"] == 0 and not unit[0]["qualified"]
    with pytest.raises(ValueError):
        calibrate([])


def test_cell_and_utc_day_integer_budgets_and_absent_families():
    frame = frame_for(normal=199, attack=2)
    frame.loc[:98, "flow_end_ms"] = pd.Timestamp("2026-01-01T23:59:59Z").value // 1_000_000
    frame.loc[99:, "flow_end_ms"] = pd.Timestamp("2026-01-02T00:00:00Z").value // 1_000_000
    scores = np.zeros(len(frame))
    scores[[0, 199]] = 1
    units, days, _ = evaluate(frame, scores, .5, model_id="m")
    assert units[0]["fp_budget"] == 1 and units[0]["qualified"]
    assert [day["utc_date"] for day in days] == ["2026-01-01", "2026-01-02"]
    assert days[0]["normal_n"] == 99 and days[0]["fp_budget"] == 0
    assert not days[0]["qualified"]
    assert units[0]["by_family"]["Scanning"] == {
        "n": 0, "present": False, "tp": 0, "fn": 0, "recall": None}


def test_no_normal_denominator_and_same_service_not_claimed_qualified():
    units, days, _ = evaluate(frame_for(normal=0), [1, 1], .5, model_id="m")
    assert units[0]["fpr"] is None and units[0]["fpr_within_budget"] is None
    assert not units[0]["qualified"] and not days[0]["qualified"]
    frame = frame_for()
    frame.loc[frame.label.eq(1), "resp_port"] = 8443
    units, _, _ = evaluate(frame, frame.label, .5, model_id="m")
    service = units[0]["same_service_normal"][0]
    assert service["resp_port"] == 8443 and service["normal_n"] == 0
    assert service["fpr"] is None and not service["qualified"]


def test_unresolved_truth_is_separate_and_legacy_rule_is_exact():
    frame = frame_for(normal=4, attack=2)
    frame.loc[0, ["label", "family"]] = [-1, "Unresolved"]
    frame["duration_seconds"] = [30, 29.9, 30, 30, 30, 30]
    frame.loc[2, "ip_protocol"] = 6
    frame.loc[3, ["orig_port", "resp_port"]] = [443, 8443]
    frame.loc[4, "resp_port"] = 8443
    assert legacy_rule(frame).tolist() == [True, False, False, True, False, True]
    units, _, _ = evaluate_decisions(frame, legacy_rule(frame), model_id="rule")
    assert units[0]["normal_n"] == 3 and units[0]["unresolved_alerts"] == 1


@pytest.mark.parametrize("fault", ["duplicate", "nan_score", "misaligned", "family", "bool_alert"])
def test_invalid_evaluation_inputs_are_rejected(fault):
    frame = frame_for()
    scores = np.zeros(len(frame))
    if fault == "duplicate":
        frame.loc[1, "row_id"] = 0
    elif fault == "nan_score":
        scores[1] = np.nan
    elif fault == "misaligned":
        scores = scores[:-1]
    elif fault == "family":
        frame.loc[0, "family"] = "ExternalC2"
    with pytest.raises(ValueError):
        if fault == "bool_alert":
            evaluate_decisions(frame, ["False"] * len(frame), model_id="m")
        else:
            evaluate(frame, scores, .5, model_id="m")


def test_all_three_seeds_and_paired_ple_gain(comparison):
    _, design, decisions = comparison
    result = compare(decisions, design)
    assert result["all_seed_development_candidate"] == {"ple": True, "no_ple": False}
    assert result["PLE_detection_benefit"]
    assert result["candidates"]["tabm_ple_s17"]["references"]["hgb"]["added_tp"] == 12
    assert not result["automatic_deployment_adoption"]


def test_shuffled_same_identity_and_historical_frame_only(comparison):
    frame, design, decisions = comparison
    decisions["tabm_ple_s17"] = decisions["tabm_ple_s17"].sample(frac=1, random_state=9)
    decisions["tabm_ple_s17"]["label"] = decisions["tabm_ple_s17"].label.astype("int8")
    decisions["old_rule"] = evaluate_decisions(frame, np.zeros(len(frame), bool), model_id="old_rule")[2]
    result = compare(decisions, design)
    assert result["all_seed_development_candidate"]["ple"]
    assert result["historical_report_only"] == ["old_rule"]


@pytest.mark.parametrize("fault", ["missing_cell", "missing_model", "wrong_identity", "wrong_truth"])
def test_incomplete_or_different_evaluation_never_silently_passes(comparison, fault):
    _, design, decisions = comparison
    key = "tabm_ple_s17"
    if fault == "missing_cell":
        for model_id in decisions:
            decisions[model_id] = decisions[model_id].loc[decisions[model_id].source_id.ne(REGISTERED_D_CELLS[0])]
    elif fault == "missing_model":
        decisions.pop(key)
    elif fault == "wrong_identity":
        decisions[key].loc[0, "row_id"] = 999
    else:
        decisions[key].loc[0, "flow_end_ms"] += 1
    with pytest.raises(ValueError):
        compare(decisions, design)


def test_one_bad_seed_and_one_reference_failure_prevent_adoption(comparison):
    _, design, decisions = comparison
    # Failure against the legacy reference alone must not be hidden by HGB gains.
    cell = REGISTERED_D_CELLS[0]
    reference = decisions["legacy_hgb_or_rule"]
    reference.loc[reference.source_id.eq(cell) & reference.label.eq(1), "alert"] = True
    candidate = decisions["tabm_ple_s29"]
    candidate.loc[candidate.source_id.eq(cell) & candidate.row_id.eq(101), "alert"] = False
    result = compare(decisions, design)
    seed = result["candidates"]["tabm_ple_s29"]
    assert seed["references"]["hgb"]["passed"]
    assert not seed["references"]["legacy_hgb_or_rule"]["passed"]
    assert not result["all_seed_development_candidate"]["ple"]


def test_over_budget_reference_tp_is_reported_not_required_bound(comparison):
    _, design, decisions = comparison
    cell = REGISTERED_D_CELLS[0]
    for model_id in ("hgb", "legacy_hgb_or_rule"):
        reference = decisions[model_id]
        reference.loc[reference.source_id.eq(cell), "alert"] = True
    for seed in (17, 29, 43):
        candidate = decisions[f"tabm_ple_s{seed}"]
        candidate.loc[candidate.source_id.eq(cell), "alert"] = False
    result = compare(decisions, design)
    assert result["all_seed_development_candidate"]["ple"]
    pairs = result["candidates"]["tabm_ple_s17"]["references"]["hgb"]["cells"]
    row = next(row for row in pairs if row["source_id"] == cell)
    assert row["qualification_repair"] and row["reference_over_budget_tp"] == 2
    assert not row["reference_tp_is_required_bound"]
    assert row["lost_tp"] == 2 and len(row["lost_tp_identities"]) == 2


def test_equal_counts_can_lose_old_identity_without_count_regression(comparison):
    _, design, decisions = comparison
    cell = REGISTERED_D_CELLS[0]
    for seed in (17, 29, 43):
        current = decisions[f"tabm_ple_s{seed}"]
        current.loc[current.source_id.eq(cell) & current.row_id.eq(100), "alert"] = False
    result = compare(decisions, design)
    assert result["all_seed_development_candidate"]["ple"]
    pair = result["candidates"]["tabm_ple_s17"]["references"]["hgb"]
    row = next(row for row in pair["cells"] if row["source_id"] == cell)
    assert row["added_tp"] == row["lost_tp"] == 1 and row["net_tp"] == 0
    assert row["lost_tp_identities"][0]["row_id"] == 100
    assert row["by_family"]["ExternalC2"]["lost_tp"] == 1
    assert row["by_family"]["Download"]["present"] is False


def test_b_capability_and_unit_qualification_are_both_required(comparison):
    _, design, decisions = comparison
    good = copy.deepcopy(decisions)
    current = decisions["tabm_ple_s17"]
    current.loc[current.source_id.eq(design["roles"]["E"]["cells"][0]) & current.row_id.eq(101), "alert"] = False
    assert not compare(decisions, design)["all_seed_development_candidate"]["ple"]
    current = good["tabm_ple_s17"]
    current.loc[current.source_id.eq(REGISTERED_D_CELLS[0]) & current.row_id.isin([0, 1]), "alert"] = True
    assert not compare(good, design)["all_seed_development_candidate"]["ple"]


def test_ple_benefit_requires_each_same_seed_not_unpaired_best_seed(comparison):
    _, design, decisions = comparison
    decisions["tabm_no_ple_s43"]["alert"] = decisions["tabm_ple_s43"].alert.copy()
    result = compare(decisions, design)
    assert result["all_seed_development_candidate"]["ple"]
    assert not result["PLE_detection_benefit"]
    assert not result["PLE_same_seed_comparisons"]["43"]["strict_D_improvement"]
