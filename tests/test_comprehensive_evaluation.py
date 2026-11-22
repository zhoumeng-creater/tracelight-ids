"""Synthetic frozen-score evaluation; no project rows or model is opened."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from scripts.comprehensive_batch import evaluation, matrix


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def fixture(tmp_path):
    def row(fid, role, label, kind=None):
        return {"flow_id": fid, "stable_flow_id": f"synthetic:{fid}", "dataset_role": role,
                "allocation_group": f"synthetic:{role}", "label": label, "target_kind": kind}
    rows = [row("F-normal", "F_base", 0), row("F-control", "F_base", 1, "control"),
            row("F-extra", "F_expansion", 0), *[row(f"C{i}", "C", 0) for i in range(3)],
            row("V-normal", "V", 0), row("V-control", "V", 1, "control"),
            row("V-download", "V", 1, "download"), row("V-unknown", "V", None),
            row("X-normal", "external_development", 0), row("X-control", "external_development", 1, "control")]
    cohorts = {"F_base": ["F-normal", "F-control"], "F_expanded": ["F-normal", "F-control", "F-extra"],
               "C": [f"C{i}" for i in range(3)], "V": ["V-normal", "V-control", "V-download", "V-unknown"],
               "external_development": ["X-normal", "X-control"]}
    events = [{"event_id": "control", "label": 1, "kind": "control", "role": "V",
               "candidate_flow_ids": ["V-control", "V-unknown"], "association_mode": "one_of", "candidate_set_complete": True},
              {"event_id": "download", "label": 1, "kind": "download", "role": "V",
               "candidate_flow_ids": ["V-download"], "association_mode": "all_confirmed", "candidate_set_complete": True},
              {"event_id": "uncertain", "label": None, "kind": "unknown", "role": "V",
               "candidate_flow_ids": ["V-unknown"], "association_mode": "one_of", "candidate_set_complete": False}]
    units = [{"unit_id": fid, "flow_ids": [fid]} for fid in ("F-normal", "F-extra", "C0", "V-normal", "X-normal")]
    plan = matrix.build_plan()
    plan["tabm_jobs"] = [job for job in plan["tabm_jobs"] if job["primary_configuration"] and job["arm"] in {"A_BCE", "G_group_tail", "R_reference_envelope"}]
    for job in plan["tabm_jobs"]:
        job["checkpoints"] = [15600, 31200]
        job["snapshot_ids"] = [f'{job["job_id"]}_step{step}' for step in job["checkpoints"]]
    plan["comparisons"] = [pair for pair in plan["comparisons"] if pair["kind"] in {"hgb_factorial", "architecture_descriptive"}
                           or (pair["pair_name"] == "R_vs_G" and pair["primary_configuration"])]
    for pair in plan["comparisons"]:
        if pair["kind"] != "hgb_factorial":
            pair["checkpoint_steps"] = [15600, 31200]
    chosen = {job["job_id"] for job in plan["tabm_jobs"] if job["arm"] == "R_reference_envelope"}
    plan["budget_comparisons"] = [pair for pair in plan["budget_comparisons"] if pair["job_id"] in chosen]
    _, registered = evaluation._registered(plan)
    models = {identity: {key: value for key, value in spec.items() if key in
                        {"job_id", "family", "arm", "seed", "learning_rate", "auxiliary_weight", "step"}}
              for identity, spec in registered.items()}
    thresholds = {identity: {"threshold": 0., "comparison": ">", "alpha": .01, "C_count": 3,
                             "integer_FP_budget": 0, "C_empirical_FP": 0} for identity in models}
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    for identity, model in models.items():
        values = {r["flow_id"]: 2. if r["label"] in (1, None) else 0. for r in rows}
        if model["arm"] == "A_BCE":
            values["V-normal"] = values["X-normal"] = 1.
        elif model["arm"] == "G_group_tail":
            values["X-normal"] = 1.
        np.savez_compressed(score_dir / f"{identity}.npz", scores=np.array([values[r["flow_id"]] for r in rows], dtype=np.float64))
    return {"rows": rows, "cohorts": cohorts, "events": events, "units": units,
            "models": models, "thresholds": thresholds, "score_dir": score_dir, "output": tmp_path / "report", "plan": plan}


def modify_scores(values, snapshot, **changes):
    path = values["score_dir"] / f"{snapshot}.npz"
    with np.load(path, allow_pickle=False) as source:
        scores = source["scores"]
    positions = {row["flow_id"]: i for i, row in enumerate(values["rows"])}
    for fid, score in changes.items():
        scores[positions[fid]] = score
    np.savez_compressed(path, scores=scores)


def target(seed=17, step=15600, arm="R_reference_envelope"):
    return matrix.snapshot_id(matrix.job_id(arm, seed), step)


def primary_group(values):
    result = read(values["output"] / "decision_summary.json")["paired_consistency"]
    return next(value for key, value in result.items() if key.startswith("matched_auxiliary") and key.endswith("step15600"))


def test_full_plan_registers_292_models_and_all_784_comparison_instances():
    plan = matrix.build_plan()
    jobs, snapshots = evaluation._registered(plan)
    pairs = evaluation._comparisons(plan, jobs)
    assert len(snapshots) == 292
    assert len(pairs) == 784
    assert sum(pair["kind"] == "budget" for pair in pairs) == 72
    assert sum(pair["kind"] == "hgb_factorial" for pair in pairs) == 4


def test_saved_scores_complete_with_compact_reconstructible_outputs(tmp_path):
    values = fixture(tmp_path)
    result = evaluation.evaluate_saved(**values)
    assert result["matrix_complete"]
    assert result["completed_snapshots"] == 22
    assert result["completed_comparisons"] == 19
    assert result["model_calls"] == result["threshold_calculations"] == 0
    assert primary_group(values)["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]
    summary = read(values["output"] / "model_summaries.json")[target()]
    assert next(x for x in summary["by_role"] if x["role"] == "V")["unknown_label_rows"] == 1
    assert not list(values["output"].rglob("*.scores.jsonl.gz"))
    pair = next(item for item in read(values["output"] / "paired_summary.json").values()
                if item["after_snapshot"] == target() and item["kind"] == "matched_auxiliary")
    location = values["output"] / pair["summary_path"]
    changes = evaluation.report.read_rows(location.parent / "changed_flows.jsonl.gz")
    assert [row["flow_id"] for row in changes] == ["X-normal"]
    assert changes[0] == {"flow_id": "X-normal", "target_kind": None, "change": "alert_removed"}
    # Raw values and margins remain reconstructible without repeating them in
    # every pair's changed-flow records.
    position = next(i for i, row in enumerate(values["rows"]) if row["flow_id"] == "X-normal")
    with np.load(values["score_dir"] / f'{pair["after_snapshot"]}.npz') as right:
        after = right["scores"][position]
    with np.load(values["score_dir"] / f'{pair["before_snapshot"]}.npz') as left:
        before = left["scores"][position]
    assert after - before == -1.
    assert int(after > values["thresholds"][pair["after_snapshot"]]["threshold"]) == 0
    assert read(location)["unchanged_flow_count"] == len(values["rows"]) - 1
    assert (values["output"] / summary["scores_relative_to_report"]).resolve() == (values["score_dir"] / summary["score_file"]).resolve()
    assert len(evaluation.report.read_rows(values["output"] / "common_identities.jsonl.gz")) == len(values["rows"])


@pytest.mark.parametrize("explicit_failure", [False, True])
def test_missing_or_failed_seed_cannot_count_as_complete(tmp_path, explicit_failure):
    values = fixture(tmp_path)
    missing = target(43)
    if explicit_failure:
        values["models"][missing]["status"] = "failed"
    else:
        del values["models"][missing]
    del values["thresholds"][missing]
    result = evaluation.evaluate_saved(**values)
    group = primary_group(values)
    assert result["status"] == "partial"
    assert not group["complete"]
    assert len(group["completed_comparisons"]) == 2
    assert not group["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]
    assert missing in result["decision_summary"]["coverage"]["missing_snapshots"]


def test_positive_download_loss_and_ambiguous_event_lower_loss_are_retained(tmp_path):
    values = fixture(tmp_path)
    modify_scores(values, target(29), **{"V-download": 0., "V-unknown": 0.})
    evaluation.evaluate_saved(**values)
    group = primary_group(values)
    assert group["any_completed_seed_positive_flow_or_event_bound_loss"]
    assert not group["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]
    pair = next(item for item in read(values["output"] / "paired_summary.json").values()
                if item["after_snapshot"] == target(29) and item["kind"] == "matched_auxiliary")
    detail = read(values["output"] / pair["summary_path"])
    assert detail["development"]["lost_positive_flow_ids"] == ["V-download"]
    assert set(detail["development"]["lost_positive_event_bounds"]) == {"control", "download"}
    download = next(item for item in detail["target_changes"] if item["target_kind"] == "download")
    assert download["TP_lost"] == ["V-download"]
    assert download["TP_added"] == []
    events = read(values["output"] / "models" / target(29) / "events.json")
    uncertain = next(item for item in events if item["event_id"] == "uncertain")
    assert uncertain["label"] is None
    assert uncertain["lower_hit"] is None and uncertain["upper_hit"] is None


def test_new_attack_detection_is_saved_separately_from_losses(tmp_path):
    values = fixture(tmp_path)
    modify_scores(values, target(17, arm="G_group_tail"), **{"V-download": 0.})
    modify_scores(values, target(17), **{"V-control": 0.})
    evaluation.evaluate_saved(**values)
    pair = next(item for item in read(values["output"] / "paired_summary.json").values()
                if item["after_snapshot"] == target() and item["kind"] == "matched_auxiliary")
    detail = read(values["output"] / pair["summary_path"])
    assert next(x for x in detail["target_changes"] if x["target_kind"] == "download")["TP_added"] == ["V-download"]
    assert next(x for x in detail["target_changes"] if x["role"] == "V" and x["target_kind"] == "control")["TP_lost"] == ["V-control"]
    assert not primary_group(values)["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]


def test_budget_comparison_uses_both_explicit_checkpoints(tmp_path):
    values = fixture(tmp_path)
    modify_scores(values, target(17, 31200), **{"V-download": 0.})
    evaluation.evaluate_saved(**values)
    pair = next(item for item in read(values["output"] / "paired_summary.json").values()
                if item["kind"] == "budget" and item["seed"] == 17)
    assert pair["before_snapshot"] == target(17, 15600)
    assert pair["after_snapshot"] == target(17, 31200)
    assert not pair["is_primary"]
    assert pair["development"]["positive_flow_loss"] == 1
    assert primary_group(values)["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]


def test_HGB_fit_membership_and_descriptive_comparisons_are_not_three_seed_ablation(tmp_path):
    values = fixture(tmp_path)
    evaluation.evaluate_saved(**values)
    for arm, expected in (("narrow_raw8", False), ("mixed_raw8", True)):
        units = read(values["output"] / "models" / f"HGB_{arm}" / "normal_units.json")
        assert next(x for x in units if x["unit_id"] == "F-extra")["in_sample_for_this_run"] is expected
    groups = read(values["output"] / "decision_summary.json")["paired_consistency"]
    hgb = [value for value in groups.values() if value["kind"] in {"hgb_factorial", "architecture_descriptive"}]
    assert len(hgb) == 6
    assert all(value["complete"] and not value["three_seed_comparison"] for value in hgb)
    assert all(not value["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"] for value in hgb)


def test_score_rescaling_without_decision_changes_does_not_claim_improvement(tmp_path):
    values = fixture(tmp_path)
    for seed in (17, 29, 43):
        modify_scores(values, target(seed), **{"X-normal": 2., "V-control": 4., "V-download": 4.})
    evaluation.evaluate_saved(**values)
    group = primary_group(values)
    assert group["all_three_seeds_exact_development_decisions_unchanged"]
    assert not group["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]


def test_training_and_calibration_changes_do_not_decide_development_benefit(tmp_path):
    values = fixture(tmp_path)
    for seed in (17, 29, 43):
        modify_scores(values, target(seed), **{"F-normal": 100., "F-control": -100., "C0": 100.})
    evaluation.evaluate_saved(**values)
    assert primary_group(values)["all_three_seeds_unit_excess_Pareto_with_no_positive_loss"]


@pytest.mark.parametrize("change", ["wrong_dtype", "short_vector", "nan", "threshold", "metadata", "reserved_role", "overlap"])
def test_changed_contract_rejected_before_report_creation(tmp_path, change):
    values = fixture(tmp_path)
    if change in {"wrong_dtype", "short_vector", "nan"}:
        score = np.zeros(len(values["rows"]), dtype=np.float64)
        if change == "wrong_dtype":
            score = score.astype(np.float32)
        elif change == "short_vector":
            score = score[:-1]
        else:
            score[0] = np.nan
        np.savez_compressed(values["score_dir"] / f"{target()}.npz", scores=score)
    elif change == "threshold":
        values["thresholds"][target()]["comparison"] = ">="
    elif change == "metadata":
        values["models"][target()]["seed"] = 42
    elif change == "reserved_role":
        values["rows"][0]["dataset_role"] = "reserved_confirmation"
    else:
        values["cohorts"]["V"].append("F-normal")
    with pytest.raises(ValueError):
        evaluation.evaluate_saved(**values)
    assert not values["output"].exists()


def test_completed_earlier_snapshot_from_failed_job_remains_usable(tmp_path):
    values = fixture(tmp_path)
    values["models"][target()]["job_status"] = "failed_job"
    del values["models"][target(17, 31200)]
    del values["thresholds"][target(17, 31200)]
    result = evaluation.evaluate_saved(**values)
    assert result["status"] == "partial"
    assert target() in result["decision_summary"]["coverage"]["completed_snapshots_from_failed_jobs"]
    assert primary_group(values)["complete"]


def test_reports_can_be_rebuilt_after_interruption_without_touching_scores(tmp_path):
    values = fixture(tmp_path)
    original = {path.name: path.read_bytes() for path in values["score_dir"].glob("*.npz")}
    first = evaluation.evaluate_saved(**values)
    before = (values["output"] / "paired_summary.json").read_bytes()
    (values["output"] / "comparison.md").write_text("interrupted", encoding="utf8")
    second = evaluation.evaluate_saved(**copy.deepcopy(values))
    assert first == second
    assert (values["output"] / "paired_summary.json").read_bytes() == before
    assert "interrupted" not in (values["output"] / "comparison.md").read_text(encoding="utf8")
    assert {path.name: path.read_bytes() for path in values["score_dir"].glob("*.npz")} == original


def test_run_collect_receipts_pass_real_evaluator_including_HGB_step_200(tmp_path):
    from scripts.comprehensive_batch import run

    values = fixture(tmp_path)
    batch_root = tmp_path / "synthetic_batch"
    collected = {}
    original_receipts = {}
    for job in [*values["plan"]["tabm_jobs"], *values["plan"]["hgb_jobs"]]:
        location = batch_root / "fits" / job["job_id"]
        location.mkdir(parents=True)
        receipt_steps = [200] if job["family"] == "hgb" else job["checkpoints"]
        receipts = []
        for step in receipt_steps:
            # Only receipt binding is under test. Neither _collect nor the real
            # evaluator may load these deliberately non-model bytes.
            artifact = location / f"synthetic_step{step}.bin"
            artifact.write_bytes(f'{job["job_id"]}:{step}:not-a-model'.encode())
            receipts.append({"step": step, "path": artifact.name,
                             "sha256": run.data.file_digest(artifact), "bytes": artifact.stat().st_size})
        original_receipts[job["job_id"]] = copy.deepcopy(receipts)
        collected.update(run._collect(location, job, job["family"], receipts, "complete", batch_root))
        assert receipts == original_receipts[job["job_id"]]
    assert set(collected) == set(values["models"])
    values["models"] = collected
    for job in values["plan"]["hgb_jobs"]:
        assert original_receipts[job["job_id"]][0]["step"] == 200
        assert collected[job["snapshot_id"]]["step"] is None
    result = evaluation.evaluate_saved(**values)
    assert result["status"] == "complete"
    assert result["completed_snapshots"] == len(collected)
    assert result["completed_comparisons"] == 19
    _, registered = evaluation._registered(values["plan"])
    summaries = read(values["output"] / "model_summaries.json")
    for identity, metadata in collected.items():
        saved = summaries[identity]["model"]
        assert saved["status"] == saved["job_status"] == "complete"
        assert saved["path"] == metadata["path"]
        assert saved["sha256"] == metadata["sha256"]
        assert run.data.file_digest(batch_root / saved["path"]) == saved["sha256"]
        for field in ("job_id", "family", "arm", "seed", "learning_rate", "auxiliary_weight", "step"):
            if field in registered[identity]:
                assert saved[field] == registered[identity][field]
