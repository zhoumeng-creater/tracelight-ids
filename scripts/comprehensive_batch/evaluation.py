"""Stream the preregistered comparison from frozen scalar score files only.

One NPZ vector corresponds to the provided row order. Shared identities and
unit/event definitions are written once. At most two models' score dictionaries
are resident for a comparison; no model is constructed or threshold selected.
Missing registered snapshots remain missing evidence, never successful runs.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from scripts.calibration_repair.evaluation import (
    ALLOWED_ROLES, DEVELOPMENT_ROLES, _is_development, _pair, _targets, report,
)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf8", newline="\n")


def _safe(identity):
    if not isinstance(identity, str) or not identity or Path(identity).name != identity or any(
            x in identity for x in ("/", "\\", ":")) or identity in {".", ".."}:
        raise ValueError("Expected a plain registered snapshot or comparison identity")
    return identity


def _registered(plan):
    jobs, snapshots = {}, {}
    for family, entries in (("tabm", plan["tabm_jobs"]), ("hgb", plan.get("hgb_jobs", []))):
        for job in entries:
            jid = _safe(job["job_id"])
            if jid in jobs or job["family"] != family:
                raise ValueError("Duplicate or inconsistent registered job")
            jobs[jid] = job
            if family == "tabm":
                steps = job["checkpoints"]
                identities = job["snapshot_ids"]
                if len(steps) != len(identities) or len(set(steps)) != len(steps):
                    raise ValueError("Registered checkpoint identities differ")
                items = zip(identities, steps)
            else:
                items = [(job["snapshot_id"], None)]
            for identity, step in items:
                _safe(identity)
                if identity in snapshots:
                    raise ValueError("Duplicate registered snapshot")
                snapshots[identity] = {**job, "step": step, "snapshot_id": identity}
    return jobs, snapshots


def _snapshot(job, step):
    if job["family"] == "hgb":
        return job["snapshot_id"]
    try:
        position = job["checkpoints"].index(step)
    except ValueError as error:
        raise ValueError("Comparison names an unregistered checkpoint") from error
    return job["snapshot_ids"][position]


def _comparisons(plan, jobs):
    """Expand matched, HGB and within-trajectory budget comparisons explicitly."""
    result = []
    for item in [*plan["comparisons"], *plan.get("hgb_comparisons", [])]:
        before, after = jobs[item["before_job_id"]], jobs[item["after_job_id"]]
        for step in item["checkpoint_steps"]:
            suffix = "terminal" if step is None else f"step{step}"
            result.append({**item, "comparison_instance_id": f'{item["comparison_id"]}_{suffix}',
                           "consistency_group": f'{item["group_id"]}_{suffix}',
                           "before_snapshot": _snapshot(before, step),
                           "after_snapshot": _snapshot(after, step), "checkpoint_step": step,
                           "is_primary": bool(item.get("primary_configuration"))
                           and step == item.get("primary_step", 15600)})
    for item in plan.get("budget_comparisons", []):
        job = jobs[item["job_id"]]
        before, after = item["before_step"], item["after_step"]
        suffix = f"step{before}_to{after}"
        result.append({**item, "kind": "budget", "comparison_instance_id": f'{item["comparison_id"]}_{suffix}',
                       "consistency_group": f'{item["group_id"]}_{suffix}',
                       "before_snapshot": _snapshot(job, before), "after_snapshot": _snapshot(job, after),
                       "checkpoint_step": None, "is_primary": False})
    names = [_safe(item["comparison_instance_id"]) for item in result]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate registered comparison")
    return result


def _validate_rows(rows, cohorts, events, units):
    index = {row["flow_id"]: row for row in rows}
    if len(index) != len(rows) or not rows:
        raise ValueError("Need unique nonempty score identities")
    if any(row["dataset_role"] not in ALLOWED_ROLES or row.get("label") not in {None, 0, 1}
           for row in rows):
        raise ValueError("Reserved/unregistered role or unsupported label")
    sets = []
    for name in ("F_expanded", "C", "V", "external_development"):
        ids = cohorts[name]
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= index.keys():
            raise ValueError(f"Missing or duplicate {name} identities")
        sets.append(set(ids))
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("Training/calibration/development overlap")
    if any(index[fid]["label"] != 0 or index[fid]["dataset_role"] != "C" for fid in cohorts["C"]):
        raise ValueError("C must contain registered normal calibration rows")
    if not report.validate_metadata(rows, events, units)["all_event_references_resolve"]:
        raise ValueError("Event references do not resolve")
    return index


def _load_scores(score_dir, snapshot, rows, threshold, fit_ids):
    path = Path(score_dir) / f"{snapshot}.npz"
    with np.load(path, allow_pickle=False) as source:
        values = source["scores"]
    if values.dtype != np.float64 or values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite FP64 score per common identity: {snapshot}")
    return {row["flow_id"]: {
        "run": snapshot, "flow_id": row["flow_id"], "stable_flow_id": row.get("stable_flow_id"),
        "dataset_role": row["dataset_role"], "allocation_group": row.get("allocation_group"),
        "label": row.get("label"), "target_kind": row.get("target_kind"),
        "score": float(score), "prediction": int(score > threshold),
        "threshold": threshold, "comparison": ">", "used_for_fitting": row["flow_id"] in fit_ids,
    } for row, score in zip(rows, values)}


def _unit_events(scores, units, events, index, fit_ids):
    unit_rows = []
    for value in report.normal_unit_results(units, scores):
        ids = value["flow_ids"]
        unit_rows.append({**value, "FP_excess": max(0, value["FP"] - value["integer_FP_budget"]),
                          "development_comparison": _is_development(ids, index),
                          "fit_member_count": len(set(ids) & fit_ids),
                          "in_sample_for_this_run": bool(set(ids) & fit_ids)})
    event_rows = []
    for value in report.event_results(events, scores):
        ids = value["candidate_flow_ids"]
        event_rows.append({**value, "development_comparison": _is_development(ids, index),
                           "fit_candidate_count": len(set(ids) & fit_ids),
                           "source_used_for_fitting_this_run": bool(set(ids) & fit_ids)})
    return unit_rows, event_rows


def _consistency(registered, completed):
    groups = defaultdict(list)
    for item in registered:
        groups[item["consistency_group"]].append(item)
    output = {}
    for key, members in groups.items():
        expected = [x["comparison_instance_id"] for x in members]
        available = [name for name in expected if name in completed]
        seeds = [x.get("seed") for x in members]
        complete = bool(expected) and len(available) == len(expected)
        paired_mechanism = members[0].get("kind") not in {"hgb_factorial", "architecture_descriptive"}
        three = paired_mechanism and len(seeds) == 3 and set(seeds) == {17, 29, 43}
        decisions = [completed[name]["development"] for name in available]
        output[key] = {
            "kind": members[0].get("kind"), "claim_scope": members[0]["claim_scope"],
            "is_primary": any(x["is_primary"] for x in members),
            "expected_comparisons": expected, "completed_comparisons": available,
            "missing_comparisons": [name for name in expected if name not in completed],
            "expected_seeds": seeds, "complete": complete, "three_seed_comparison": three,
            "all_registered_comparisons_unit_excess_Pareto_with_no_positive_loss": bool(
                complete and all(x["unit_excess_Pareto_with_no_positive_loss"] for x in decisions)),
            "all_three_seeds_unit_excess_Pareto_with_no_positive_loss": bool(
                complete and three and all(x["unit_excess_Pareto_with_no_positive_loss"] for x in decisions)),
            "any_completed_seed_positive_flow_or_event_bound_loss": any(
                x["positive_flow_loss"] or x["positive_event_bound_loss"] for x in decisions),
            "all_three_seeds_exact_development_decisions_unchanged": bool(
                complete and three and all(x["exact_decisions_unchanged"] for x in decisions)),
        }
    return output


def evaluate_saved(rows, cohorts, events, units, models, thresholds, score_dir, output, plan):
    """Evaluate only completed, preregistered snapshots and frozen thresholds.

    ``models`` is a snapshot-id mapping. Failed or absent snapshots are recorded
    as missing coverage; completed snapshots require complete scores and a saved
    C threshold. The runner owns score-row alignment and all phase barriers.
    """
    index = _validate_rows(rows, cohorts, events, units)
    jobs, registered = _registered(plan)
    pairs = _comparisons(plan, jobs)
    if not set(models) <= registered.keys():
        raise ValueError("Unregistered model snapshot")
    complete = {key: value for key, value in models.items()
                if value.get("status", "complete") in {"complete", "completed"}}
    if set(thresholds) != set(complete):
        raise ValueError("Completed snapshots require exactly their frozen thresholds")
    resolved, fits, fit_sets = {}, {}, {}
    for snapshot, metadata in complete.items():
        expected = registered[snapshot]
        for name in ("job_id", "family", "arm", "seed", "learning_rate", "auxiliary_weight", "step"):
            if name in metadata and name in expected and metadata[name] != expected[name]:
                raise ValueError(f"Model metadata differs from registration: {snapshot} {name}")
        resolved[snapshot] = {**expected, **metadata}
        fit_cohort = expected.get("fit_cohort", "F_expanded")
        if fit_cohort not in cohorts or not set(cohorts[fit_cohort]) <= index.keys():
            raise ValueError("Missing registered fitting cohort")
        if fit_cohort not in fit_sets:
            fit_sets[fit_cohort] = set(cohorts[fit_cohort])
        fits[snapshot] = fit_sets[fit_cohort]
        rule = thresholds[snapshot]
        if (not math.isfinite(float(rule["threshold"])) or rule.get("comparison") != ">"
                or rule.get("alpha") != .01 or rule.get("C_count") != len(cohorts["C"])
                or rule.get("integer_FP_budget") != len(cohorts["C"]) // 100):
            raise ValueError("Frozen empirical C threshold contract differs")
        # Validate all vector files before creating an evaluation directory.
        with np.load(Path(score_dir) / f"{snapshot}.npz", allow_pickle=False) as source:
            value = source["scores"]
            if value.dtype != np.float64 or value.shape != (len(rows),) or not np.isfinite(value).all():
                raise ValueError(f"Expected finite FP64 scores: {snapshot}")
    output = Path(output)
    # Derived reports may be rebuilt after interruption. Coverage and index JSON
    # identify the current results; raw predictions and thresholds are read-only.
    output.mkdir(parents=True, exist_ok=True)
    identity_keys = ("flow_id", "stable_flow_id", "dataset_role", "allocation_group", "source_group",
                     "label", "target_kind", "label_evidence", "clock_basis", "replay_export_time")
    report.save_rows(output / "common_identities.jsonl.gz",
                     ({key: row[key] for key in identity_keys if key in row} for row in rows))
    report.save_rows(output / "event_definitions.jsonl.gz", events)
    report.save_rows(output / "normal_unit_definitions.jsonl.gz", units)
    _save(output / "cohorts.json", cohorts)
    order_hash = hashlib.sha256(json.dumps([r["flow_id"] for r in rows], ensure_ascii=True,
                                          separators=(",", ":")).encode()).hexdigest()
    model_summaries = {}
    for snapshot in sorted(complete):
        scores = _load_scores(score_dir, snapshot, rows, float(thresholds[snapshot]["threshold"]), fits[snapshot])
        unit_values, event_values = _unit_events(scores, units, events, index, fits[snapshot])
        location = output / "models" / snapshot
        location.mkdir(parents=True, exist_ok=True)
        # Membership is shared in normal_unit_definitions; no repeated 60k-ID lists.
        compact_units = [{k: v for k, v in item.items() if k != "flow_ids"} for item in unit_values]
        _save(location / "normal_units.json", compact_units)
        _save(location / "events.json", event_values)
        model_summaries[snapshot] = {
            "model": resolved[snapshot], "threshold": thresholds[snapshot],
            "score_file": f"{snapshot}.npz",
            "scores_relative_to_report": os.path.relpath(Path(score_dir) / f"{snapshot}.npz", output),
            "common_row_order_sha256": order_hash, "rows": len(rows),
            "by_role": report.role_results(scores), "control_download": _targets(scores),
            "development_normal_units": len([x for x in unit_values if x["development_comparison"]]),
            "development_normal_units_within_budget": sum(
                x["empirical_budget_met"] is True for x in unit_values if x["development_comparison"]),
        }
        _save(location / "summary.json", model_summaries[snapshot])
        del scores, unit_values, event_values, compact_units
    comparisons, unavailable = {}, []
    for item in pairs:
        name, before, after = (item[k] for k in ("comparison_instance_id", "before_snapshot", "after_snapshot"))
        if before not in complete or after not in complete:
            unavailable.append({**item, "missing_snapshots": [x for x in (before, after) if x not in complete]})
            continue
        left = _load_scores(score_dir, before, rows, float(thresholds[before]["threshold"]), fits[before])
        right = _load_scores(score_dir, after, rows, float(thresholds[after]["threshold"]), fits[after])
        lu, le = _unit_events(left, units, events, index, fits[before])
        ru, re = _unit_events(right, units, events, index, fits[after])
        changed, summary = _pair(left, right, {x["unit_id"]: x for x in lu},
                                {x["unit_id"]: x for x in ru}, {x["event_id"]: x for x in le},
                                {x["event_id"]: x for x in re}, index)
        destination = output / "pairs" / name
        destination.mkdir(parents=True, exist_ok=True)
        changed = [x for x in changed if x["alert_added"] or x["alert_removed"]]
        report.save_rows(destination / "changed_flows.jsonl.gz", (
            {"flow_id": row["flow_id"], "target_kind": row["target_kind"],
             "change": "alert_added" if row["alert_added"] else "alert_removed"}
            for row in changed))
        summary = {**item, **summary, "changed_flow_count": len(changed),
                   "unchanged_flow_count": len(rows) - len(changed),
                   "full_pairs_reconstructible_from": "common_identities + both NPZ score vectors + thresholds"}
        _save(destination / "summary.json", summary)
        comparisons[name] = {**item, "development": summary["development"],
                             "changed_flow_count": len(changed), "summary_path": str(destination.relative_to(output) / "summary.json")}
        del left, right, lu, ru, le, re, changed, summary
    consistency = _consistency(pairs, comparisons)
    coverage = {"expected_snapshots": len(registered), "completed_snapshots": len(complete),
                "missing_snapshots": sorted(set(registered) - complete.keys()),
                "failed_model_records": {k: v for k, v in models.items() if k not in complete},
                "completed_snapshots_from_failed_jobs": sorted(
                    k for k, v in complete.items() if v.get("job_status") not in {None, "complete", "completed"}),
                "matrix_complete": bool(registered) and set(complete) == set(registered),
                "expected_comparisons": len(pairs), "completed_comparisons": len(comparisons),
                "unavailable_comparisons": unavailable}
    decisions = {"coverage": coverage, "paired_consistency": consistency,
                 "whole_project_adoption_decided": False, "automatic_model_or_seed_selection": False,
                 "primary_rule": "Default lr=.002, auxiliary weight=1, step=15600 is primary; other fixed cells are sensitivity/budget evidence, not post-hoc winners.",
                 "interpretation_rules": [
                     "Report actual business-unit integer budgets and control/download flow/event changes at each fixed checkpoint.",
                     "R-vs-G compares full recipes; S-vs-G and R-vs-H isolate the registered alignment increment with matched separation terms.",
                     "Missing runs prevent complete three-seed claims; failure is not replaced by a different seed or configuration.",
                     "A lower training surrogate with unchanged development decisions provides no demonstrated detection benefit.",
                     "Improvement only after a longer budget or at another fixed learning rate/weight is reported as training sensitivity, not hidden model selection.",
                     "Record every added/lost control/download flow and positive event bound; no pooled TP offset for download losses.",
                     "Observed development improvements require measured cost and later independent forward confirmation.",
                     "HGB-vs-TabM is an architecture comparison with different training algorithms, not a paired-initialization mechanism ablation."],
                 "normal_unit_overlap_policy": "Separate registered denominators; never pool overlapping units as independent samples.",
                 "evidence_scope": "Opened historical development only; F is training, C is calibration, reserved confirmation is excluded."}
    _save(output / "model_summaries.json", model_summaries)
    _save(output / "paired_summary.json", comparisons)
    _save(output / "decision_summary.json", decisions)
    _save(output / "coverage.json", coverage)
    lines = ["# 一次批次完整比较", "", f"完成 {len(complete)}/{len(registered)} 份预登记模型快照，完成 {len(comparisons)}/{len(pairs)} 项预登记配对。缺失运行保留为空缺证据。", "",
             "默认学习率 0.002、辅助权重 1、15,600 步是主比较；其余固定配置与时长完整展示，不自动挑最好模型或种子。", "",
             "| 模型快照 | 正常业务预算通过数 | 控制 TP/总数 | 下载 TP/总数 |", "|---|---:|---:|---:|"]
    for snapshot in sorted(registered):
        if snapshot not in model_summaries:
            lines.append(f"| {snapshot} | 缺失 | 缺失 | 缺失 |")
            continue
        value = model_summaries[snapshot]
        counts = {}
        for kind in ("control", "download"):
            selected = [x for x in value["control_download"] if x["role"] in DEVELOPMENT_ROLES and x["target_kind"] == kind]
            counts[kind] = f"{sum(x['TP'] for x in selected)}/{sum(x['positive_flows'] for x in selected)}"
        lines.append(f"| {snapshot} | {value['development_normal_units_within_budget']}/{value['development_normal_units']} | {counts['control']} | {counts['download']} |")
    lines += ["", "| 配对组 | 预定配对齐全 | 三种子均有业务超额改善且无攻击／事件边界丢失 |", "|---|---|---|"]
    for name, value in consistency.items():
        passed = value['all_three_seeds_unit_excess_Pareto_with_no_positive_loss'] if value['three_seed_comparison'] else "不适用：单次 HGB 或架构描述比较"
        lines.append(f"| {name} | {value['complete']} | {passed} |")
    lines += ["", "控制／下载按来源、逐连接和事件上下界分别保存。正常预算通过不自动证明攻击能力充分；不把重叠业务单元当作独立样本。", "",
              "R 对 G 是整套目标比较；S 对 G、R 对 H 用于检查对应分离项下的对齐增量。权重、学习率和训练时长的敏感性须连同反例展示。", "",
              "变化连接保存在每项配对的 changed_flows.jsonl.gz；未变化连接可由共享身份、两份原始分数及各自冻结阈值复算。", "",
              "缺失运行不能形成三种子一致结论。训练约束未达到、达到后未迁移和攻击取舍分别解释；本报告不自动选择模型或宣布独立确认完成。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf8", newline="\n")
    return {"status": "complete" if coverage["matrix_complete"] else "partial",
            "model_calls": 0, "threshold_calculations": 0, "rows_per_snapshot": len(rows),
            "completed_snapshots": len(complete), "expected_snapshots": len(registered),
            "completed_comparisons": len(comparisons), "expected_comparisons": len(pairs),
            "matrix_complete": coverage["matrix_complete"], "decision_summary": decisions}
