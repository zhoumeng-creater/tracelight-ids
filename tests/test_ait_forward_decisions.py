from copy import deepcopy

import pytest

from scripts.ait_forward.decisions import calibrate_rules, evaluate_candidate, evaluate_comparison


def _calibration():
    small = {f"small-{i}": 0.0 for i in range(98)} | {"small-high": 10.0}
    large = {f"large-{i}": 0.0 for i in range(99)} | {"large-kth": 8.0, "large-high": 9.0}
    return small | large, {"small": list(small), "large": list(large)}


def test_calibration_uses_integer_floors_and_uses_a_shared_strict_threshold():
    scores, groups = _calibration()
    rules = calibrate_rules(scores, groups)
    pooled, business = rules["pooled"], rules["business_max"]
    assert (pooled["threshold"], pooled["C_empirical_FP"], pooled["integer_FP_budget"]) == (8.0, 2, 2)
    assert [row["FP"] for row in pooled["business_groups"]] == [1, 1]
    assert [row["integer_FP_budget"] for row in pooled["business_groups"]] == [0, 1]
    assert business["threshold"] == 10.0
    assert business["C_empirical_FP"] == 0
    assert pooled["comparison"] == business["comparison"] == ">"


def test_calibration_never_splits_a_boundary_tie():
    scores = {f"n{i}": 0.0 for i in range(98)} | {"tie-a": 1.0, "tie-b": 1.0}
    rules = calibrate_rules(scores, {"business": list(scores)})
    for rule in rules.values():
        assert rule["threshold"] == 1.0
        assert rule["integer_FP_budget"] == 1
        assert rule["C_boundary_tie_count"] == 2
        assert rule["C_empirical_FP"] == 0


@pytest.mark.parametrize("groups", [{"first": ["a"]}, {"first": ["a", "b"], "second": ["a"]}])
def test_calibration_requires_a_partition(groups):
    with pytest.raises(ValueError, match="disjoint, exhaustive"):
        calibrate_rules({"a": 1.0, "b": 2.0}, groups)


def test_frozen_rules_report_changed_identities_events_and_ranking_ceiling():
    rules = calibrate_rules(*_calibration())
    scores = {
        "n-high": 10.0, "n-low": 8.0, "n-other": 9.0,
        "c-weak": 9.0, "c-boundary": 10.0, "c-strong": 11.0,
        "download": 10.0, "unknown": 100.0,
    }
    normal_units = {"business-1": ["n-high", "n-low"], "business-2": ["n-other"]}
    attack_ids = {"control": ["c-weak", "c-boundary", "c-strong"], "download": ["download"]}
    events = [
        {"event_id": "ambiguous-control", "kind": "control",
         "candidate_flow_ids": ["c-weak", "c-strong"], "candidate_set_complete": True,
         "association_mode": "one_of"},
        {"event_id": "confirmed-download", "kind": "download",
         "candidate_flow_ids": ["download"], "candidate_set_complete": True,
         "association_mode": "all_confirmed"},
        {"event_id": "incomplete-control", "kind": "control",
         "candidate_flow_ids": ["unknown"], "candidate_set_complete": False,
         "association_mode": "one_of"},
    ]
    frozen_inputs = deepcopy((scores, normal_units, attack_ids, events, rules))
    result = evaluate_comparison(
        scores, normal_units=normal_units, attack_ids=attack_ids, events=events, rules=rules,
        minimum_tp={"control": 2, "download": 1},
    )
    assert (scores, normal_units, attack_ids, events, rules) == frozen_inputs
    pooled, business = result["rules"]["pooled"], result["rules"]["business_max"]
    assert [unit["FP"] for unit in pooled["normal_units"]] == [1, 1]
    assert [unit["FP"] for unit in business["normal_units"]] == [0, 0]
    assert not pooled["all_normal_units_within_budget"]
    assert business["all_normal_units_within_budget"]
    assert pooled["targets"]["control"]["TP"] == 3
    assert business["targets"]["control"]["TP"] == 1
    assert not business["targets"]["control"]["minimum_TP_met"]
    assert business["targets"]["download"]["FN"] == 1
    assert result["paired"]["targets"]["control"]["lost_flow_ids"] == ["c-weak", "c-boundary"]
    assert result["paired"]["targets"]["download"]["lost_flow_ids"] == ["download"]
    assert business["events"][0]["lower_hit"] is False
    assert business["events"][0]["upper_hit"] is True
    assert result["paired"]["events"][0]["lower_hit"]["lost"]
    assert result["paired"]["events"][1]["upper_hit"]["lost"]
    assert business["events"][2]["status"] == "unresolved"
    assert result["paired"]["events"][2]["upper_hit"]["lost"] is None
    assert result["score_ranking_upper_bound"]["control"] == {
        "status": "available", "positive_flows": 3, "minimum_TP": 2,
        "max_TP": 1, "minimum_TP_attainable": False,
    }
    assert result["score_ranking_upper_bound"]["download"]["max_TP"] == 0
    contrast = result["rule_contrast"]
    assert contrast["status"] == "normal_fp_reduction_with_detection_loss"
    assert contrast["normal_budget_repaired_units"] == ["business-1", "business-2"]
    assert contrast["lost_attack_flow_counts"] == {"control": 2, "download": 1}
    assert contrast["lost_positive_event_bound_ids"] == ["ambiguous-control", "confirmed-download"]

    comparison = evaluate_candidate(
        scores, normal_units=normal_units, attack_ids=attack_ids, events=events, rules=rules,
        minimum_tp={"control": 1, "download": 1},
        required_events=[
            {"event_id": "ambiguous-control", "kind": "control"},
            {"event_id": "confirmed-download", "kind": "download"},
        ],
    )
    assert comparison["event_evaluation"]["pooled"]["status"] == "met"
    business_events = comparison["event_evaluation"]["business_max"]
    assert business_events["status"] == "missed"
    assert business_events["required_events"][0]["reason"] == "possible_hit_only"
    assert business_events["required_events"][0]["ordering_diagnosis"] == "rank_can_only_possibly_hit_event"
    assert business_events["required_events"][1]["reason"] == "no_candidate_hit"


def test_unavailable_target_and_normal_unit_do_not_report_success():
    rules = calibrate_rules(*_calibration())
    result = evaluate_comparison(
        {"normal": 1.0, "control": 11.0},
        normal_units={"business": ["normal"], "unavailable-business": []},
        attack_ids={"control": ["control"], "download": None}, events=[],
        rules=rules, minimum_tp={"control": None, "download": 0},
    )
    business = result["rules"]["business_max"]
    assert business["targets"]["download"]["status"] == "unavailable"
    assert business["targets"]["download"]["TP"] is None
    assert business["targets"]["download"]["minimum_TP_met"] is None
    assert business["targets"]["control"]["minimum_TP_met"] is None
    assert business["normal_units"][1]["empirical_budget_met"] is None
    assert not business["all_normal_units_within_budget"]
    assert result["paired"]["targets"]["download"]["lost_flow_ids"] is None
    assert result["score_ranking_upper_bound"] is None


def test_evaluation_rejects_conflicting_labels_and_noninteger_floor():
    rules = calibrate_rules(*_calibration())
    with pytest.raises(ValueError, match="conflicting"):
        evaluate_comparison(
            {"row": 1.0}, normal_units={"business": ["row"]},
            attack_ids={"control": ["row"], "download": None}, events=[],
            rules=rules, minimum_tp={"control": None, "download": None},
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        evaluate_comparison(
            {"normal": 1.0, "control": 11.0}, normal_units={"business": ["normal"]},
            attack_ids={"control": ["control"], "download": None}, events=[],
            rules=rules, minimum_tp={"control": 1.0, "download": None},
        )


@pytest.mark.parametrize(
    ("required_score", "at_observed_feasible_rank", "diagnosis"),
    [
        (9.0, "missed", "rank_cannot_hit_event"),
        (9.5, "met", "frozen_rule_lacks_guaranteed_hit_but_rank_has_room"),
    ],
)
def test_event_ordering_distinguishes_rank_limit_from_frozen_rule(
    required_score, at_observed_feasible_rank, diagnosis,
):
    comparison = evaluate_candidate(
        {"n": 9.0, "c-required": required_score, "c-other": 11.0, "d": 12.0},
        normal_units={"normal": ["n"]},
        attack_ids={"control": ["c-required", "c-other"], "download": ["d"]},
        events=[
            {"event_id": "required-control", "kind": "control", "candidate_flow_ids": ["c-required"],
             "association_mode": "all_confirmed", "candidate_set_complete": True},
            {"event_id": "required-download", "kind": "download", "candidate_flow_ids": ["d"],
             "association_mode": "all_confirmed", "candidate_set_complete": True},
        ],
        rules=calibrate_rules(*_calibration()),
        minimum_tp={"control": 1, "download": 1},
        required_events=[
            {"event_id": "required-control", "kind": "control"},
            {"event_id": "required-download", "kind": "download"},
        ],
    )
    assert comparison["outcomes"]["business_max"] == "criteria_met"
    assert comparison["event_evaluation"]["business_max"]["status"] == "missed"
    assert comparison["confirmation_outcomes"]["business_max"] == "event_criteria_failed"
    best = comparison["result"]["score_ranking_event_feasibility"]
    assert best["status"] == at_observed_feasible_rank
    assert "threshold" not in str(best)
    assert comparison["event_evaluation"]["business_max"]["required_events"][0]["ordering_diagnosis"] == diagnosis
