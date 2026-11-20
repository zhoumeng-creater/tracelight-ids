"""Evaluate frozen scalar scores; never construct a model or choose a threshold.

The nine fits and nine C thresholds must already be persisted by the runner.
This module retains every row and every registered normal unit/event. Development
comparisons exclude both F and C, and cannot certify whole-project adoption.
"""
from __future__ import annotations

from collections import defaultdict
import importlib.util
import math
from pathlib import Path


_HELPER = Path(__file__).resolve().parents[1] / "fit_contract/run_minimal_comparison.py"
_SPEC = importlib.util.spec_from_file_location("transfer_frozen_report_helpers", _HELPER)
report = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(report)

ARMS = ("A_BCE", "B_permuted", "C_real")
SEEDS = (17, 29, 43)
RUNS = tuple(f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS)
PAIRS = {"B_vs_A": ("A_BCE", "B_permuted"),
         "C_vs_A": ("A_BCE", "C_real"),
         "C_vs_B": ("B_permuted", "C_real")}
DEVELOPMENT_ROLES = {"V", "external_development"}
ALLOWED_ROLES = {"F_base", "F_expansion", "C", "C_background", *DEVELOPMENT_ROLES}


def _is_development(ids, index):
    return bool(ids) and all(index[fid]["dataset_role"] in DEVELOPMENT_ROLES for fid in ids)


def _targets(scores):
    grouped = defaultdict(list)
    for row in scores.values():
        if row["label"] == 1:
            grouped[row["dataset_role"], row.get("target_kind")].append(row)
    return [{"role": role, "target_kind": kind, "positive_flows": len(group),
             "TP": sum(row["prediction"] for row in group),
             "FN": sum(1 - row["prediction"] for row in group)}
            for (role, kind), group in sorted(grouped.items(), key=lambda item: str(item[0]))]


def _pair(before, after, before_units, after_units, before_events, after_events, index):
    rows, role_summary = report.paired_results(before, after)
    for row in rows:
        row["target_kind"] = index[row["flow_id"]].get("target_kind")
        row["score_delta"] = row["context_score"] - row["baseline_score"]
        row["threshold_delta"] = row["context_threshold"] - row["baseline_threshold"]
        row["baseline_margin"] = row["baseline_score"] - row["baseline_threshold"]
        row["context_margin"] = row["context_score"] - row["context_threshold"]

    unit_pairs = []
    for uid, left in before_units.items():
        right = after_units[uid]
        if left["flow_ids"] != right["flow_ids"]:
            raise ValueError("Paired normal denominators differ")
        unit_pairs.append({"unit_id": uid, "development_comparison": left["development_comparison"],
                           "normal_count": left["normal_count"],
                           "integer_FP_budget": left["integer_FP_budget"],
                           "before_FP": left["FP"], "after_FP": right["FP"],
                           "before_excess": left["FP_excess"], "after_excess": right["FP_excess"],
                           "FP_delta": right["FP"] - left["FP"],
                           "excess_delta": right["FP_excess"] - left["FP_excess"]})

    event_pairs = []
    for eid, left in before_events.items():
        right = after_events[eid]
        row = {"event_id": eid, "label": left.get("label"), "kind": left.get("kind"),
               "development_comparison": left["development_comparison"],
               "before_lower_hit": left["lower_hit"], "after_lower_hit": right["lower_hit"],
               "before_upper_hit": left["upper_hit"], "after_upper_hit": right["upper_hit"]}
        for bound in ("lower", "upper"):
            a, b = left[f"{bound}_hit"], right[f"{bound}_hit"]
            row[f"{bound}_lost"] = (a is True and b is False) if a is not None and b is not None else None
            row[f"{bound}_gained"] = (a is False and b is True) if a is not None and b is not None else None
        event_pairs.append(row)

    dev_units = [row for row in unit_pairs if row["development_comparison"]]
    dev_rows = [row for row in rows if row["dataset_role"] in DEVELOPMENT_ROLES]
    positives = [row for row in dev_rows if row["label"] == 1]
    dev_events = [row for row in event_pairs if row["development_comparison"] and row["label"] == 1]
    lost = [row["flow_id"] for row in positives if row["alert_removed"]]
    event_lost = [row["event_id"] for row in dev_events if row["lower_lost"] or row["upper_lost"]]
    no_worse = bool(dev_units) and all(row["excess_delta"] <= 0 for row in dev_units)
    strict = any(row["excess_delta"] < 0 for row in dev_units)
    exact_normal_change = any(row["FP_delta"] != 0 for row in dev_units)
    grouped = defaultdict(lambda: {"TP_added": [], "TP_lost": []})
    for row in positives:
        key = row["dataset_role"], row.get("target_kind")
        if row["alert_added"]:
            grouped[key]["TP_added"].append(row["flow_id"])
        if row["alert_removed"]:
            grouped[key]["TP_lost"].append(row["flow_id"])
    return rows, {"by_role": role_summary, "normal_units": unit_pairs,
                  "events": event_pairs,
                  "target_changes": [{"role": role, "target_kind": kind, **values}
                                     for (role, kind), values in sorted(grouped.items(), key=lambda item: str(item[0]))],
                  "development": {
                      "normal_units": len(dev_units), "positive_flows": len(positives),
                      "unit_excess_no_worse": no_worse, "unit_excess_strict_improvement": strict,
                      "unit_FP_no_worse": bool(dev_units) and all(row["FP_delta"] <= 0 for row in dev_units),
                      "unit_FP_strict_improvement": any(row["FP_delta"] < 0 for row in dev_units),
                      "normal_decisions_changed": exact_normal_change,
                      "lost_positive_flow_ids": lost, "lost_positive_event_bounds": event_lost,
                      "positive_flow_loss": len(lost), "positive_event_bound_loss": len(event_lost),
                      "unit_excess_Pareto_with_no_positive_loss": bool(positives) and no_worse and strict and not lost and not event_lost,
                      "exact_decisions_unchanged": not any(row["alert_added"] or row["alert_removed"] for row in dev_rows),
                      "decision_only_comparison": True}}


def _decision_summary(unit_metrics, comparisons):
    consistency = {}
    for name in PAIRS:
        per_seed = {str(seed): comparisons[f"{name}_s{seed}"]["development"] for seed in SEEDS}
        consistency[name] = {
            "all_three_seeds_unit_excess_Pareto_with_no_positive_loss": all(
                value["unit_excess_Pareto_with_no_positive_loss"] for value in per_seed.values()),
            "any_seed_positive_flow_or_event_bound_loss": any(
                value["positive_flow_loss"] or value["positive_event_bound_loss"] for value in per_seed.values()),
            "all_three_seeds_exact_development_decisions_unchanged": all(
                value["exact_decisions_unchanged"] for value in per_seed.values()),
            "per_seed": per_seed}
    a_budget = {}
    for seed in SEEDS:
        units = [row for row in unit_metrics[f"A_BCE_s{seed}"] if row["development_comparison"]]
        a_budget[str(seed)] = bool(units) and all(row["empirical_budget_met"] for row in units)
    return {
        "ordinary_BCE_development_normal_units_all_within_budget_by_seed": a_budget,
        "ordinary_BCE_all_three_seeds_normal_units_within_budget": all(a_budget.values()),
        "paired_consistency": consistency,
        "whole_project_adoption_decided": False,
        "whole_project_adoption_reason": "No numeric control/download acceptance requirement has been supplied; report each class and event without inferring acceptance from normal budgets.",
        "automatic_model_or_seed_selection": False,
        "interpretation_rules": [
            "If ordinary BCE already meets the application's attack requirements and all required normal units, use the mature baseline; normal budgets alone do not establish this.",
            "If B/C improve A but C has no consistent benefit over B, retain only the calibration-aware adaptation claim and end the temporal/business correspondence claim for this version.",
            "Consistent C-vs-B and C-vs-A exact unit improvements without attack loss support a fixed candidate for independent forward confirmation, subject to measured cost; they are not independent confirmation.",
            "Lower F surrogate gaps with unchanged development decisions are no demonstrated detection benefit; do not rescue by scaling scores or moving thresholds.",
            "Lower false positives with lost control/download detections are a recorded tradeoff, not a joint solution. List every gained and lost flow/event bound.",
            "F feasibility with failed cross-run development, or failure of all fixed arms, ends this version without additional sources, windows, seeds, thresholds, or model searches."],
        "normal_unit_overlap_policy": "Units remain separate. Their excess counts are not pooled or treated as independent observations.",
        "temporal_claim": "C-vs-B tests the fixed observed temporal/business correspondence, not an isolated causal time effect.",
        "evidence_scope": "Opened historical development; C is calibration; F is training; no reserved or forward confirmation is scored here."}


def evaluate_rows(rows, cohorts, events, units, scores_by_run, thresholds, output):
    """Save the full fixed comparison from already-computed scores and thresholds.

    ``scores_by_run[run][flow_id]`` is a finite float. ``thresholds[run]`` must
    contain threshold, comparison='>', alpha=.01 and the fixed C count/budget.
    No calibration order statistic is recomputed here.
    """
    if set(scores_by_run) != set(RUNS) or set(thresholds) != set(RUNS):
        raise ValueError("Need all nine registered runs and already-frozen thresholds")
    index = {row["flow_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate scored identity")
    if any(row["dataset_role"] not in ALLOWED_ROLES or row.get("label") not in {None, 0, 1} for row in rows):
        raise ValueError("Reserved/unregistered role or unsupported label")
    for role in ("F_expanded", "C", "V", "external_development"):
        ids = cohorts[role]
        if not ids or len(ids) != len(set(ids)) or any(fid not in index for fid in ids):
            raise ValueError(f"Missing or duplicate registered {role} identities")
    cohort_sets = [set(cohorts[role]) for role in ("F_expanded", "C", "V", "external_development")]
    if sum(map(len, cohort_sets)) != len(set.union(*cohort_sets)):
        raise ValueError("Training/calibration/development overlap")
    if any(index[fid]["label"] != 0 for fid in cohorts["C"]):
        raise ValueError("C contains a non-normal row")
    metadata = report.validate_metadata(rows, events, units)
    if not metadata["all_event_references_resolve"]:
        raise ValueError("Registered event references do not all resolve")
    for run in RUNS:
        if set(scores_by_run[run]) != set(index) or not all(math.isfinite(float(value)) for value in scores_by_run[run].values()):
            raise ValueError(f"Incomplete or nonfinite score vector: {run}")
        threshold = thresholds[run]
        if (not math.isfinite(float(threshold["threshold"])) or threshold.get("comparison") != ">"
                or threshold.get("alpha") != .01 or threshold.get("C_count") != len(cohorts["C"])
                or threshold.get("integer_FP_budget") != len(cohorts["C"]) // 100):
            raise ValueError("Threshold metadata differs from the frozen empirical C rule")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [f"{run}.{suffix}" for run in RUNS for suffix in ("scores.jsonl.gz", "events.jsonl.gz", "normal_units.jsonl.gz")]
    names += [f"{pair}_s{seed}.paired.jsonl.gz" for pair in PAIRS for seed in SEEDS]
    names += ["metrics.json", "control_download_metrics.json", "paired_summary.json", "decision_summary.json", "comparison.md"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Preserve existing evaluation outputs; no automatic overwrite")
    score_rows, metrics, targets, unit_metrics, event_metrics = {}, {}, {}, {}, {}
    fit_ids = set(cohorts["F_expanded"])
    for run in RUNS:
        threshold = float(thresholds[run]["threshold"])
        scores = {fid: {"run": run, "flow_id": fid, "stable_flow_id": row.get("stable_flow_id"),
                        "dataset_role": row["dataset_role"], "allocation_group": row.get("allocation_group"),
                        "label": row.get("label"), "target_kind": row.get("target_kind"),
                        "score": float(scores_by_run[run][fid]), "threshold": threshold, "comparison": ">",
                        "prediction": int(float(scores_by_run[run][fid]) > threshold),
                        "used_for_fitting": fid in fit_ids}
                  for fid, row in index.items()}
        score_rows[run] = scores
        report.save_rows(output / f"{run}.scores.jsonl.gz", scores.values())
        unit_results = []
        for meta, result in zip(units, report.normal_unit_results(units, scores)):
            unit_results.append({**meta, **result, "FP_excess": max(0, result["FP"] - result["integer_FP_budget"]),
                                 "development_comparison": _is_development(result["flow_ids"], index),
                                 "fit_member_count": len(set(result["flow_ids"]) & fit_ids),
                                 "in_sample_for_this_run": bool(set(result["flow_ids"]) & fit_ids)})
        event_results = []
        for meta, result in zip(events, report.event_results(events, scores)):
            ids = result["candidate_flow_ids"]
            event_results.append({**meta, **result, "development_comparison": _is_development(ids, index),
                                  "fit_candidate_count": len(set(ids) & fit_ids),
                                  "source_used_for_fitting_this_run": any(index[fid]["dataset_role"] in {"F_base", "F_expansion"} for fid in ids)})
        unit_metrics[run], event_metrics[run] = unit_results, event_results
        report.save_rows(output / f"{run}.normal_units.jsonl.gz", unit_results)
        report.save_rows(output / f"{run}.events.jsonl.gz", event_results)
        metrics[run], targets[run] = report.role_results(scores), _targets(scores)
    comparisons = {}
    for name, (before, after) in PAIRS.items():
        for seed in SEEDS:
            a, b = f"{before}_s{seed}", f"{after}_s{seed}"
            paired, summary = _pair(score_rows[a], score_rows[b],
                                    {row["unit_id"]: row for row in unit_metrics[a]},
                                    {row["unit_id"]: row for row in unit_metrics[b]},
                                    {row["event_id"]: row for row in event_metrics[a]},
                                    {row["event_id"]: row for row in event_metrics[b]}, index)
            key = f"{name}_s{seed}"
            comparisons[key] = {"before": a, "after": b, **summary}
            report.save_rows(output / f"{key}.paired.jsonl.gz", paired)
    decisions = _decision_summary(unit_metrics, comparisons)
    for filename, value in (("metrics.json", metrics), ("control_download_metrics.json", targets),
                            ("paired_summary.json", comparisons), ("decision_summary.json", decisions)):
        report.save_json(output / filename, value)
    lines = ["# 固定三臂开发比较", "", "九份模型与 C 阈值冻结后生成本表。每个种子单独列出；不按结果选择模型或种子。", "",
             "| 运行 | 开发正常单元预算通过数 | 控制 TP/总数 | 下载 TP/总数 |", "|---|---:|---:|---:|"]
    for run in RUNS:
        selected = [row for row in unit_metrics[run] if row["development_comparison"]]
        counts = {}
        for kind in ("control", "download"):
            selected_targets = [row for row in targets[run] if row["role"] in DEVELOPMENT_ROLES and row["target_kind"] == kind]
            counts[kind] = f"{sum(row['TP'] for row in selected_targets)}/{sum(row['positive_flows'] for row in selected_targets)}"
        lines.append(f"| {run} | {sum(row['empirical_budget_met'] is True for row in selected)}/{len(selected)} | {counts['control']} | {counts['download']} |")
    lines += ["", "表中控制与下载按连接统计；各来源、角色和事件候选上下界见 JSON/JSONL。正常单元逐项保留，重叠单元不累加为独立样本。", "",
              "| 配对 | 三种子均降低至少一个单元超额、其余不增且无正例/事件边界丢失 | 任一种子有正例或事件边界丢失 |", "|---|---|---|"]
    for name, item in decisions["paired_consistency"].items():
        lines.append(f"| {name} | {item['all_three_seeds_unit_excess_Pareto_with_no_positive_loss']} | {item['any_seed_positive_flow_or_event_bound_loss']} |")
    lines += ["", "正常预算合格不能代替控制／下载能力要求。项目尚未给定两类攻击的采用门槛，本报告不据此自动宣布整体成功。", "",
              "若 C 相对 B 的实际判决无一致增益，则结束本版本的先后对应创新主张；若只有 F 代理下降而开发判决未改善，则没有检测收益依据。存在攻击丢失时按逐流和事件上下界展示取舍。", "",
              "满足固定比较的候选还需结合实测成本判断，再做独立时间前向确认。固定三臂失败后不自动增加数据、阈值、种子或窗口搜索。", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return {"runs": list(RUNS), "rows_per_run": len(rows), "normal_units_per_run": len(units),
            "events_per_run": len(events), "paired_comparisons": len(comparisons),
            "model_calls": 0, "threshold_calculations": 0, "decision_summary": decisions}
