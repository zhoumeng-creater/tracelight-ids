"""Synthetic saved-score checks; no project data or model is loaded."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/calibration_transfer/evaluation.py"
_SPEC = importlib.util.spec_from_file_location("transfer_evaluation_under_test", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def fixture():
    def row(fid, role, label, kind=None):
        return {"flow_id": fid, "stable_flow_id": f"synthetic:{fid}", "dataset_role": role,
                "allocation_group": f"synthetic:{role}", "label": label, "target_kind": kind}
    rows = [row("F-normal", "F_base", 0), row("F-control", "F_base", 1, "control"),
            *[row(f"C{i}", "C", 0) for i in range(3)],
            row("V-normal", "V", 0), row("V-control", "V", 1, "control"),
            row("V-download", "V", 1, "download"), row("V-unknown", "V", None),
            row("X-normal", "external_development", 0),
            row("X-control", "external_development", 1, "control")]
    cohorts = {"F_expanded": ["F-normal", "F-control"], "C": [f"C{i}" for i in range(3)],
               "V": ["V-normal", "V-control", "V-download", "V-unknown"],
               "external_development": ["X-normal", "X-control"]}
    events = [{"event_id": "control", "label": 1, "kind": "control", "role": "V",
               "candidate_flow_ids": ["V-control", "V-unknown"], "association_mode": "one_of",
               "candidate_set_complete": True},
              {"event_id": "download", "label": 1, "kind": "download", "role": "V",
               "candidate_flow_ids": ["V-download"], "association_mode": "all_confirmed",
               "candidate_set_complete": True}]
    units = [{"unit_id": fid, "flow_ids": [fid]} for fid in ("F-normal", "C0", "V-normal", "X-normal")]
    scores = {}
    for run in evaluation.RUNS:
        values = {row["flow_id"]: 2. if row["label"] in (1, None) else 0. for row in rows}
        if run.startswith("A_"):
            values["V-normal"] = values["X-normal"] = 1.
        elif run.startswith("B_"):
            values["X-normal"] = 1.
        scores[run] = values
    thresholds = {run: {"threshold": 0., "comparison": ">", "alpha": .01, "C_count": 3,
                        "integer_FP_budget": 0, "C_empirical_FP": 0} for run in evaluation.RUNS}
    return rows, cohorts, events, units, scores, thresholds


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_nine_saved_score_runs_and_every_pair_are_reported(tmp_path):
    values = fixture()
    result = evaluation.evaluate_rows(*values, tmp_path)
    assert result["model_calls"] == result["threshold_calculations"] == 0
    assert result["paired_comparisons"] == 9
    assert len(list(tmp_path.glob("*.scores.jsonl.gz"))) == 9
    assert len(list(tmp_path.glob("*.paired.jsonl.gz"))) == 9
    decisions = read(tmp_path / "decision_summary.json")
    for name in evaluation.PAIRS:
        assert decisions["paired_consistency"][name]["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]
    assert decisions["whole_project_adoption_decided"] is False
    assert decisions["automatic_model_or_seed_selection"] is False
    assert decisions["ordinary_BCE_all_three_seeds_normal_units_within_budget"] is False
    targets = read(tmp_path / "control_download_metrics.json")["C_real_s17"]
    assert {row["target_kind"] for row in targets} == {"control", "download"}
    assert next(row for row in targets if row["target_kind"] == "download")["TP"] == 1
    metrics = read(tmp_path / "metrics.json")["C_real_s17"]
    v = next(row for row in metrics if row["role"] == "V")
    assert v["unknown_label_rows"] == 1
    assert v["FP"] == 0


def test_strict_threshold_tie_and_raw_score_changes(tmp_path):
    values = fixture()
    evaluation.evaluate_rows(*values, tmp_path)
    rows = evaluation.report.read_rows(tmp_path / "C_real_s17.scores.jsonl.gz")
    normal = next(row for row in rows if row["flow_id"] == "V-normal")
    assert normal["score"] == normal["threshold"] == 0.
    assert normal["prediction"] == 0
    pairs = evaluation.report.read_rows(tmp_path / "C_vs_A_s17.paired.jsonl.gz")
    normal_pair = next(row for row in pairs if row["flow_id"] == "V-normal")
    assert normal_pair["score_delta"] == -1.
    assert normal_pair["baseline_margin"] == 1.
    assert normal_pair["context_margin"] == 0.
    assert normal_pair["alert_removed"] is True


def test_positive_loss_one_seed_prevents_consistent_claim(tmp_path):
    values = fixture()
    values[4]["C_real_s29"]["V-download"] = 0.
    evaluation.evaluate_rows(*values, tmp_path)
    decisions = read(tmp_path / "decision_summary.json")
    for name in ("C_vs_A", "C_vs_B"):
        item = decisions["paired_consistency"][name]
        assert item["any_seed_positive_flow_or_event_bound_loss"] is True
        assert item["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"] is False
    pairs = read(tmp_path / "paired_summary.json")["C_vs_B_s29"]
    assert pairs["development"]["lost_positive_flow_ids"] == ["V-download"]
    assert pairs["development"]["lost_positive_event_bounds"] == ["download"]
    assert next(row for row in pairs["target_changes"] if row["target_kind"] == "download")["TP_lost"] == ["V-download"]


def test_one_of_event_lower_bound_loss_retained_without_labeled_flow_loss(tmp_path):
    values = fixture()
    values[4]["C_real_s17"]["V-unknown"] = 0.
    evaluation.evaluate_rows(*values, tmp_path)
    pair = read(tmp_path / "paired_summary.json")["C_vs_B_s17"]
    assert pair["development"]["positive_flow_loss"] == 0
    assert pair["development"]["lost_positive_event_bounds"] == ["control"]
    assert pair["development"]["unit_excess_Pareto_with_no_positive_loss"] is False
    event = next(row for row in pair["events"] if row["event_id"] == "control")
    assert event["lower_lost"] is True
    assert event["upper_lost"] is False


def test_F_and_C_do_not_decide_development_benefit(tmp_path):
    values = fixture()
    for run in evaluation.RUNS:
        if run.startswith("C_"):
            values[4][run]["F-normal"] = values[4][run]["C0"] = 100.
            values[4][run]["F-control"] = -100.
    evaluation.evaluate_rows(*values, tmp_path)
    decision = read(tmp_path / "decision_summary.json")["paired_consistency"]["C_vs_B"]
    assert decision["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"] is True
    units = evaluation.report.read_rows(tmp_path / "C_real_s17.normal_units.jsonl.gz")
    assert next(row for row in units if row["unit_id"] == "F-normal")["in_sample_for_this_run"] is True
    assert next(row for row in units if row["unit_id"] == "C0")["development_comparison"] is False


def test_unchanged_decisions_cannot_be_claimed_as_unit_benefit(tmp_path):
    values = fixture()
    for seed in evaluation.SEEDS:
        values[4][f"C_real_s{seed}"] = {fid: 2 * score for fid, score in values[4][f"B_permuted_s{seed}"].items()}
    evaluation.evaluate_rows(*values, tmp_path)
    result = read(tmp_path / "decision_summary.json")["paired_consistency"]["C_vs_B"]
    assert result["all_three_seeds_exact_development_decisions_unchanged"] is True
    assert result["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"] is False


@pytest.mark.parametrize("change", ["missing_run", "missing_flow", "nonfinite", "wrong_comparator", "wrong_C_count", "reserved_role", "overlap"])
def test_incomplete_or_changed_contract_rejected_before_outputs(tmp_path, change):
    values = fixture()
    if change == "missing_run":
        del values[4]["C_real_s43"]
    elif change == "missing_flow":
        del values[4]["C_real_s43"]["V-normal"]
    elif change == "nonfinite":
        values[4]["C_real_s43"]["V-normal"] = float("nan")
    elif change == "wrong_comparator":
        values[5]["C_real_s43"]["comparison"] = ">="
    elif change == "wrong_C_count":
        values[5]["C_real_s43"]["C_count"] = 4
    elif change == "reserved_role":
        values[0][0]["dataset_role"] = "reserved_confirmation"
    else:
        values[1]["C"].append("F-normal")
    with pytest.raises(ValueError):
        evaluation.evaluate_rows(*values, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_reports_are_not_overwritten(tmp_path):
    values = fixture()
    evaluation.evaluate_rows(*values, tmp_path)
    previous = (tmp_path / "paired_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.evaluate_rows(*copy.deepcopy(values), tmp_path)
    assert (tmp_path / "paired_summary.json").read_bytes() == previous
