import numpy as np
import pandas as pd
import pytest

from ids.support_diagnostic.evaluation import evaluate_cell, oracle_cell, paired_cell


def truth(families, source="cell"):
    return pd.DataFrame({"family": families, "row_id": np.arange(len(families)), "source_id": source})


def test_exact_one_percent_budget_and_strict_boundary():
    frame = truth(["Benign"] * 100 + ["ExternalC2"] * 2)
    score = np.array([1.0] + [0.0] * 99 + [1.0, 0.5])
    result = evaluate_cell(frame, score, {"threshold": 0.5})
    assert result["fp_budget"] == result["fp"] == 1
    assert result["qualified"] and result["fpr"] == 0.01
    assert result["by_family"]["ExternalC2"]["tp"] == 1
    assert result["by_family"]["ExternalC2"]["fn"] == 1


def test_one_more_false_positive_disqualifies_cell():
    frame = truth(["Benign"] * 199 + ["ExternalC2"])
    result = evaluate_cell(frame, [1.0, 1.0] + [0.0] * 197 + [1.0], {"threshold": 0.5})
    assert result["fp_budget"] == 1
    assert result["fp"] == 2
    assert not result["qualified"]


@pytest.mark.parametrize("rule", [{"threshold": float("inf"), "ready": True},
                                  {"threshold": None}, {"threshold": 0.5, "ready": False}])
def test_unready_cannot_pass_by_producing_zero_alerts(rule):
    result = evaluate_cell(truth(["Benign"] * 100 + ["ExternalC2"]), np.ones(101), rule)
    assert result["fp"] == result["tp"] == 0
    assert result["fpr_within_budget"]
    assert not result["ready"] and not result["qualified"]


def test_missing_family_is_not_a_perfect_recall_success():
    result = evaluate_cell(truth(["Benign", "ExternalC2"]), [0.0, 1.0], {"threshold": 0.5})
    for family in ("Scanning", "Download"):
        assert result["by_family"][family] == {"n": 0, "present": False, "tp": 0, "fn": 0, "recall": None}


def test_unresolved_is_neither_normal_nor_attack_but_alerts_are_reported():
    result = evaluate_cell(truth(["Benign", "ExternalC2", "Unresolved", "OtherAttack"]),
                           [0.0, 1.0, 1.0, 0.0], {"threshold": 0.5})
    assert result["normal_n"] == 1 and result["attack_n"] == 2
    assert result["unresolved_n"] == result["unresolved_alerts"] == 1
    assert result["tp"] == result["fn"] == 1 and result["alerts"] == 2


def test_paired_changes_preserve_gained_and_lost_even_with_zero_net():
    frame = truth(["Benign"] * 100 + ["ExternalC2"] * 3 + ["Download", "Unresolved"])
    left = np.zeros(105)
    right = np.zeros(105)
    left[[0, 100, 101, 103]] = 1.0
    right[[1, 2, 101, 102, 104]] = 1.0
    result = paired_cell(frame, left, {"threshold": 0.5}, right, {"threshold": 0.5})
    c2 = result["by_family"]["ExternalC2"]
    assert (c2["gained"], c2["lost"], c2["retained"], c2["net"]) == (1, 1, 1, 0)
    assert result["by_family"]["Download"]["lost"] == 1
    assert (result["normal"]["gained"], result["normal"]["lost"]) == (2, 1)
    assert result["unresolved"]["gained"] == 1
    assert result["left_qualified"] and not result["right_qualified"]
    assert not result["both_qualified"]


def test_oracle_does_not_split_normal_ties_to_fill_budget():
    frame = truth(["Benign"] * 100 + ["ExternalC2"] * 3)
    score = [0.9, 0.9] + [0.1] * 98 + [0.95, 0.9, 0.8]
    result = oracle_cell(frame, score)
    assert result["threshold"] == 0.9
    assert result["normal_at_threshold"] == 2
    assert result["fp_budget"] == 1 and result["fp"] == 0
    assert result["tp"] == 1 and result["fn"] == 2
    assert result["diagnostic_only"]
    assert result["threshold_origin"] == "evaluation_normal_labels"


def test_oracle_uses_available_budget_and_counts_attack_families_separately():
    frame = truth(["Benign"] * 100 + ["ExternalC2", "Download", "Scanning"])
    score = list(np.arange(100, dtype=float)) + [99.5, 99.0, 98.0]
    result = oracle_cell(frame, score)
    assert result["threshold"] == 98.0 and result["fp"] == 1
    assert result["by_family"]["ExternalC2"]["tp"] == 1
    assert result["by_family"]["Download"]["tp"] == 1
    assert result["by_family"]["Scanning"]["tp"] == 0


def test_oracle_zero_budget_still_detects_above_all_normals():
    result = oracle_cell(truth(["Benign"] * 99 + ["ExternalC2"]), [1.0] * 99 + [2.0])
    assert result["fp_budget"] == 0 and result["fp"] == 0 and result["tp"] == 1


def test_no_normal_denominator_never_qualifies_or_reports_zero_fpr():
    frame = truth(["ExternalC2"])
    result = evaluate_cell(frame, [1.0], {"threshold": 0.5})
    assert result["fpr"] is None and result["fpr_within_budget"] is None
    assert not result["qualified"]
    oracle = oracle_cell(frame, [1.0])
    assert oracle["status"] == "no_normal_denominator" and not oracle["ready"]


def test_multiple_cells_cannot_hide_failed_cells_in_a_pooled_rate():
    frame = truth(["Benign", "Benign"])
    frame["source_id"] = ["one", "two"]
    with pytest.raises(ValueError, match="one source/cell"):
        evaluate_cell(frame, [0.0, 0.0], {"threshold": 0.5})


@pytest.mark.parametrize("score", [[0.0], [[0.0, 1.0]], [np.nan, 1.0], [0.0, np.inf]])
def test_invalid_scores_fail_instead_of_silently_changing_denominators(score):
    with pytest.raises(ValueError, match="scores must"):
        evaluate_cell(truth(["Benign", "ExternalC2"]), score, {"threshold": 0.5})
