"""Summarize saved four-arm coverage results without loading or calling models.

Usage: python scripts/summarize_coverage_outcomes.py --results RETURN --output NEW_DIR
Only Python's standard library is used. Predictions and event bounds are read
from the return files; no score, threshold, model or experiment is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path


ARMS = ("narrow_raw8", "mixed_raw8", "narrow_ports_zero", "mixed_ports_zero")
PAIRS = {
    "coverage_raw8": ("narrow_raw8", "mixed_raw8"),
    "coverage_ports_zero": ("narrow_ports_zero", "mixed_ports_zero"),
    "ports_narrow": ("narrow_raw8", "narrow_ports_zero"),
    "ports_mixed": ("mixed_raw8", "mixed_ports_zero"),
}
DEV = {"V", "external_development"}
ROLES = DEV | {"F_base", "F_expansion", "C", "C_background"}
PAIR_FIELDS = ("rows", "TP_added", "TP_lost", "FP_added", "FP_removed",
               "unknown_added_alerts", "unknown_removed_alerts")
COUNT_FIELDS = ("TP", "FN", "FP", "TN", "unknown_label_rows")
SECTIONS = ("development", "calibration", "fit_or_expansion_sources")


def require(row, keys, where):
    missing = [key for key in keys if key not in row]
    if missing:
        raise ValueError(f"{where}: 缺少字段 {', '.join(missing)}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def keyed(rows, field, where):
    result = {}
    for number, row in enumerate(rows, 1):
        require(row, [field], f"{where}:{number}")
        if row[field] in result:
            raise ValueError(f"{where}: 重复 {field}={row[field]}")
        result[row[field]] = row
    return result


def section(role):
    if role in DEV:
        return "development"
    if role in {"C", "C_background"}:
        return "calibration"
    return "fit_or_expansion_sources"


def counts(rows):
    c = Counter((r["label"], r["prediction"]) for r in rows)
    return dict(TP=c[1, 1], FN=c[1, 0], FP=c[0, 1], TN=c[0, 0],
                unknown_label_rows=c[None, 0] + c[None, 1])


def flow_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset_role"], row["allocation_group"]].append(row)
    output = []
    for (role, source), group in sorted(groups.items()):
        targets = {}
        for kind in ("control", "download"):
            selected = [r for r in group if r["label"] == 1 and r["target_kind"] == kind]
            tp = sum(r["prediction"] for r in selected)
            targets[kind] = {"positive_flows": len(selected), "TP": tp,
                             "FN": len(selected) - tp}
        output.append({"role": role, "source": source, "rows": len(group),
                       **counts(group), "targets": targets,
                       "unknown_alerts": sum(r["prediction"] for r in group if r["label"] is None)})
    return output


def pair_counts(before, after):
    c = Counter({key: 0 for key in PAIR_FIELDS})
    for a in before:
        b = after[a["flow_id"]]
        c["rows"] += 1
        added = a["prediction"] == 0 and b["prediction"] == 1
        lost = a["prediction"] == 1 and b["prediction"] == 0
        if a["label"] == 1:
            c["TP_added"] += added
            c["TP_lost"] += lost
        elif a["label"] == 0:
            c["FP_added"] += added
            c["FP_removed"] += lost
        else:
            c["unknown_added_alerts"] += added
            c["unknown_removed_alerts"] += lost
    return dict(c)


def validate_scores(arm, rows):
    index = keyed(rows, "flow_id", f"{arm}.scores.jsonl.gz")
    if not index:
        raise ValueError(f"{arm}.scores.jsonl.gz: 没有判决记录")
    for row in rows:
        where = f"{arm}.scores:{row['flow_id']}"
        require(row, ["arm", "dataset_role", "allocation_group", "label", "target_kind", "prediction"], where)
        if row["arm"] != arm or row["dataset_role"] not in ROLES:
            raise ValueError(f"{where}: arm 或 dataset_role 不属于固定四臂")
        if row["label"] not in (0, 1, None) or row["prediction"] not in (0, 1):
            raise ValueError(f"{where}: label/prediction 必须是已登记的二值或未知标签")
        if row["label"] == 1 and row["target_kind"] not in {"control", "download"}:
            raise ValueError(f"{where}: 正例缺少 control/download 目标类别")
        if not row["allocation_group"]:
            raise ValueError(f"{where}: 缺少来源 allocation_group")
    missing_roles = {"F_base", "F_expansion", "C", "V", "external_development"} - {
        row["dataset_role"] for row in rows}
    if missing_roles:
        raise ValueError(f"{arm}.scores: 缺少固定实验角色 {sorted(missing_roles)}")
    return index


def validate_metrics(arm, index, metrics, targets):
    actual = defaultdict(list)
    for row in index.values():
        actual[row["dataset_role"]].append(row)
    saved = keyed(metrics, "role", f"metrics.json/{arm}")
    if saved.keys() != actual.keys():
        raise ValueError(f"metrics.json/{arm}: 角色集合与逐流判决不同")
    for role, rows in actual.items():
        require(saved[role], COUNT_FIELDS, f"metrics.json/{arm}/{role}")
        if any(saved[role][k] != v for k, v in counts(rows).items()):
            raise ValueError(f"metrics.json/{arm}/{role}: 汇总与逐流判决不一致")
    expected = Counter()
    expected_tp = Counter()
    for row in index.values():
        if row["label"] == 1:
            key = row["dataset_role"], row["target_kind"]
            expected[key] += 1
            expected_tp[key] += row["prediction"]
    found = set()
    for row in targets:
        require(row, ["role", "target_kind", "positive_flows", "TP", "FN"], f"control_download_metrics.json/{arm}")
        key = row["role"], row["target_kind"]
        if key in found or key not in expected:
            raise ValueError(f"control_download_metrics.json/{arm}: 重复或多余类别 {key}")
        found.add(key)
        if (row["positive_flows"], row["TP"], row["FN"]) != (expected[key], expected_tp[key], expected[key] - expected_tp[key]):
            raise ValueError(f"control_download_metrics.json/{arm}/{key}: 与逐流判决不一致")
    if found != expected.keys():
        raise ValueError(f"control_download_metrics.json/{arm}: 缺少目标类别 {set(expected) - found}")


def summarize_events(arm, rows, index):
    keyed(rows, "event_id", f"{arm}.events.jsonl.gz")
    grouped, positives = defaultdict(Counter), []
    for event in rows:
        where = f"{arm}.events:{event['event_id']}"
        require(event, ["role", "kind", "label", "candidate_flow_ids", "association_mode",
                        "candidate_set_complete", "status", "lower_hit", "upper_hit",
                        "source_used_for_fitting_this_arm"], where)
        ids = event["candidate_flow_ids"]
        if len(ids) != len(set(ids)) or any(fid not in index for fid in ids):
            raise ValueError(f"{where}: 重复或缺少候选流引用")
        if event["role"] not in ROLES or event["label"] not in (0, 1, None):
            raise ValueError(f"{where}: 无效角色或标签")
        if any(index[fid]["dataset_role"] != event["role"] for fid in ids):
            # Normal calibration events may also contain unknown C_background candidates.
            roles = {index[fid]["dataset_role"] for fid in ids}
            if event["role"] != "C" or not roles <= {"C", "C_background"}:
                raise ValueError(f"{where}: 候选流跨实验角色")
        fitted = event["role"] == "F_base" or (event["role"] == "F_expansion" and arm.startswith("mixed_"))
        if event["source_used_for_fitting_this_arm"] is not fitted:
            raise ValueError(f"{where}: 训练来源标志与角色不一致")
        resolved = bool(ids) and event["candidate_set_complete"] is True and event["association_mode"] in {"one_of", "all_confirmed"}
        lower = upper = None
        if resolved:
            predictions = [bool(index[fid]["prediction"]) for fid in ids]
            lower = all(predictions) if event["association_mode"] == "one_of" else any(predictions)
            upper = any(predictions)
        if (event["lower_hit"], event["upper_hit"], event["status"]) != (
                lower, upper, "resolved_bounds" if resolved else "unresolved"):
            raise ValueError(f"{where}: 既有事件界限与候选判决或完整性不一致")
        sources = sorted({index[fid]["allocation_group"] for fid in ids})
        key = event["role"], tuple(sources), event["kind"], str(event["label"])
        group = grouped[key]
        group["registered_records"] += 1
        group["multiple_candidates"] += len(ids) > 1
        if not resolved:
            group["unresolved"] += 1
        elif lower:
            group["certain_alert"] += 1
        elif upper:
            group["possible_only_alert"] += 1
        else:
            group["no_candidate_alert"] += 1
        if event["label"] == 1:
            positives.append({**event, "sources": sources,
                              "unit_note": "按原 kind 保留；不当作独立链数，不与流 TP 相加"})
    result = []
    for (role, sources, kind, label), values in sorted(grouped.items()):
        result.append({"role": role, "sources": list(sources), "kind": kind,
                       "label": None if label == "None" else int(label),
                       **{key: values[key] for key in ("registered_records", "multiple_candidates",
                           "certain_alert", "possible_only_alert", "no_candidate_alert", "unresolved")}})
    return result, positives


def normal_unit_role(unit, index):
    """Legacy runner units need not contain role metadata; use all members."""
    if unit.get("role") is not None:
        return unit["role"], "reported"
    roles = {index[fid]["dataset_role"] for fid in unit["flow_ids"]}
    if len(roles) != 1:
        raise ValueError(f"normal_units:{unit['unit_id']}: 缺少 role，且不能从全部成员确定唯一角色")
    return roles.pop(), "derived_from_all_member_score_roles"


def summarize_units(arm, rows, index):
    keyed(rows, "unit_id", f"{arm}.normal_units.jsonl.gz")
    output = []
    for unit in rows:
        where = f"{arm}.normal_units:{unit['unit_id']}"
        require(unit, ["flow_ids", "normal_count", "FP", "integer_FP_budget",
                       "empirical_budget_met", "in_sample_for_this_arm"], where)
        ids = unit["flow_ids"]
        if len(ids) != len(set(ids)) or any(fid not in index for fid in ids):
            raise ValueError(f"{where}: 重复或缺少正常成员引用")
        role, role_basis = normal_unit_role(unit, index)
        if role not in ROLES:
            raise ValueError(f"{where}: 未登记的角色 {role}")
        if any(index[fid]["label"] != 0 or index[fid]["dataset_role"] != role for fid in ids):
            raise ValueError(f"{where}: 正常分母含未知、攻击或不同角色")
        fp = sum(index[fid]["prediction"] for fid in ids)
        expected = len(ids), fp, len(ids) // 100, fp <= len(ids) // 100 if ids else None
        if tuple(unit[k] for k in ("normal_count", "FP", "integer_FP_budget", "empirical_budget_met")) != expected:
            raise ValueError(f"{where}: 正常单位的既有预算结果与判决不一致")
        fitted = role == "F_base" or (role == "F_expansion" and arm.startswith("mixed_"))
        # The runner's in_sample flag depends on optional original unit metadata.
        # Preserve it; source use can be established separately from member roles.
        output.append({k: v for k, v in unit.items() if k != "flow_ids"} | {
            "role": role, "role_basis": role_basis,
            "source_used_for_fitting_this_arm": fitted,
            "sources": sorted({index[fid]["allocation_group"] for fid in ids}),
            "members_verified_known_normal": True})
    return output


def summarize_pair(name, before, after, saved_rows, saved_summary):
    pairs = keyed(saved_rows, "flow_id", f"{name}.paired.jsonl.gz")
    if pairs.keys() != before.keys():
        raise ValueError(f"{name}.paired.jsonl.gz: 缺少或多余流身份")
    grouped = defaultdict(list)
    for fid, a in before.items():
        b, pair = after[fid], pairs[fid]
        require(pair, ["dataset_role", "label", "baseline_prediction", "context_prediction",
                       "alert_added", "alert_removed"], f"{name}.paired:{fid}")
        expected = (a["dataset_role"], a["label"], a["prediction"], b["prediction"],
                    a["prediction"] == 0 and b["prediction"] == 1,
                    a["prediction"] == 1 and b["prediction"] == 0)
        if tuple(pair[k] for k in ("dataset_role", "label", "baseline_prediction", "context_prediction", "alert_added", "alert_removed")) != expected:
            raise ValueError(f"{name}.paired:{fid}: 配对与两臂既有判决不一致")
        grouped[a["dataset_role"], a["allocation_group"], a["target_kind"], str(a["label"])].append(a)
    require(saved_summary, ["before", "after", "by_role"], f"paired_summary.json/{name}")
    if (saved_summary["before"], saved_summary["after"]) != PAIRS[name]:
        raise ValueError(f"paired_summary.json/{name}: 比较方向不一致")
    saved_roles = keyed(saved_summary["by_role"], "role", f"paired_summary.json/{name}")
    actual_roles = {r["dataset_role"] for r in before.values()}
    if saved_roles.keys() != actual_roles:
        raise ValueError(f"paired_summary.json/{name}: 角色集合不完整")
    for role in actual_roles:
        require(saved_roles[role], PAIR_FIELDS, f"paired_summary.json/{name}/{role}")
        c = pair_counts([r for r in before.values() if r["dataset_role"] == role], after)
        if any(saved_roles[role][key] != value for key, value in c.items()):
            raise ValueError(f"paired_summary.json/{name}/{role}: 配对计数不一致")
    by_source_kind = [{"role": role, "source": source, "target_kind": kind,
                       "label": None if label == "None" else int(label),
                       **pair_counts(group, after)}
                      for (role, source, kind, label), group in sorted(grouped.items(), key=lambda item: str(item[0]))]
    dev = pair_counts([r for r in before.values() if r["dataset_role"] in DEV], after)
    favorable = dev["TP_added"] > 0 or dev["FP_removed"] > 0
    adverse = dev["TP_lost"] > 0 or dev["FP_added"] > 0
    return {"before": PAIRS[name][0], "after": PAIRS[name][1],
            "by_role": saved_summary["by_role"], "by_source_kind": by_source_kind,
            "development_changes": dev,
            "development_flags": {
                "coverage_gains_observed": name.startswith("coverage_") and favorable,
                "coverage_losses_observed": name.startswith("coverage_") and adverse,
                "port_dependence_observed": name.startswith("ports_") and (favorable or adverse),
                "tradeoff_observed": favorable and adverse,
                "no_known_flow_decision_change": not (favorable or adverse)}}


def summarize(results):
    results = Path(results)
    names = ["summary.json", "metrics.json", "control_download_metrics.json", "paired_summary.json"]
    names += [f"{arm}.{kind}.jsonl.gz" for arm in ARMS for kind in ("scores", "events", "normal_units")]
    names += [f"{name}.paired.jsonl.gz" for name in PAIRS]
    missing = [name for name in names if not (results / name).is_file()]
    if missing:
        raise ValueError("四臂回传不完整，缺少文件：" + ", ".join(missing))
    run = read_json(results / "summary.json")
    require(run, ["arms", "fit_calls", "score_rows_per_arm"], "summary.json")
    if len(run["arms"]) != 4 or set(run["arms"]) != set(ARMS) or run["fit_calls"] != 4:
        raise ValueError("summary.json: 不是四次固定拟合的完整回传")
    metrics, targets, pairs = [read_json(results / name) for name in names[1:4]]
    for label, document, expected in (("metrics.json", metrics, ARMS),
                                      ("control_download_metrics.json", targets, ARMS),
                                      ("paired_summary.json", pairs, PAIRS)):
        require(document, expected, label)
    indices, arms, identity_sets = {}, {}, {}
    for arm in ARMS:
        index = validate_scores(arm, read_rows(results / f"{arm}.scores.jsonl.gz"))
        if len(index) != run["score_rows_per_arm"]:
            raise ValueError(f"{arm}.scores: 行数与 summary.json 不一致")
        if indices:
            first = indices[ARMS[0]]
            if index.keys() != first.keys():
                raise ValueError(f"{arm}.scores: 四臂流身份集合不同")
            for fid, row in index.items():
                keys = ("dataset_role", "allocation_group", "label", "target_kind")
                if any(row[key] != first[fid][key] for key in keys):
                    raise ValueError(f"{arm}.scores:{fid}: 四臂角色、来源或真值不同")
        indices[arm] = index
        validate_metrics(arm, index, metrics[arm], targets[arm])
        event_rows = read_rows(results / f"{arm}.events.jsonl.gz")
        unit_rows = read_rows(results / f"{arm}.normal_units.jsonl.gz")
        if not event_rows or not unit_rows:
            missing_kind = "events" if not event_rows else "normal_units"
            raise ValueError(f"{arm}.{missing_kind}.jsonl.gz: 空文件不能代表完整回传")
        event_summary, positive_events = summarize_events(arm, event_rows, index)
        units = summarize_units(arm, unit_rows, index)
        signatures = {
            "events": {r["event_id"]: (r["role"], r["label"], r["kind"], tuple(r["candidate_flow_ids"]),
                        r["association_mode"], r["candidate_set_complete"]) for r in event_rows},
            "units": {r["unit_id"]: (normal_unit_role(r, index)[0], tuple(r["flow_ids"])) for r in unit_rows}}
        if identity_sets and signatures != identity_sets:
            raise ValueError(f"{arm}: 四臂事件候选集或正常单位身份不一致")
        identity_sets = signatures
        flows = flow_summary(index.values())
        arms[arm] = {scope: {"flow_metrics": [r for r in flows if section(r["role"]) == scope],
                             "normal_units": [r for r in units if section(r["role"]) == scope],
                             "event_groups": [r for r in event_summary if section(r["role"]) == scope],
                             "positive_event_records": [r for r in positive_events if section(r["role"]) == scope]}
                     for scope in SECTIONS}
    comparisons = {name: summarize_pair(name, indices[a], indices[b],
                        read_rows(results / f"{name}.paired.jsonl.gz"), pairs[name])
                   for name, (a, b) in PAIRS.items()}
    cost_path = results / "cost.json"
    reported_cost = {"status": "not_provided", "data": None}
    if cost_path.is_file():
        reported_cost = {"status": "provided", "data": read_json(cost_path)}
        names.append("cost.json")
    return {
        "schema": "coverage-outcome-description-v2", "status": "complete_saved_results_summarized",
        "primary_arm": "mixed_ports_zero", "primary_conclusion_roles": sorted(DEV),
        "new_algorithm_inference": False, "overall_pass_defined": False, "winner_selected": False,
        "model_loads": 0, "model_calls": 0, "fit_calls": 0, "threshold_calculations": 0,
        "interpretation": {
            "sources": "source 使用原 allocation_group；保留完整来源／链身份。",
            "excluded_from_primary": "C/C_background 另列。F_base 与 F_expansion 均不支持主结论；后者即使在 narrow 未拟合，也已用于 mixed。",
            "events": "按原 kind 分开；lower_hit 才是确定命中，upper-only 为可能命中。未闭合候选的界限保持 null。事件／对象不与流 TP 相加，不当作独立攻击链。",
            "normal_units": "逐单位保留既有整数预算判断；缺 role 时只从全部成员的唯一角色推导并标明依据。原 in_sample 元数据保留，来源是否用于拟合另据角色记录。重叠单位不可相加成独立总体。C 经验结果不等于未来保证。",
            "cost": "cost.json 原样传递；仅原记录的拟合、批评分和工件字节，不是逐条告警端到端延迟、采集／提取成本或运行内存。缺成本记为未提供，不影响已有判决解释。",
            "flags": "仅 V/external_development 的已知流标签。扩充后出现有利变化，指存在新增 TP 或移除 FP；扩充后出现身份回退，指存在丢失 TP 或新增 FP。两者可以同时成立，不等于整体优劣，也不是纯来源多样性的因果效应。端口两流程判决变化包含重新拟合及各自校准，不是冻结函数单字段因果检验。有利与不利变化并存时标记取舍。未知流标签翻转不计这些标记；事件上下界另列，不能以流级无变化代替事件判断。",
            "research": "描述性基线结果，不产生创新成立、整体通过、赢家选择、自动蒸馏或自动转向历史／图的结论。"},
        "arms": arms, "comparisons": comparisons, "reported_cost": reported_cost,
        "inputs": {name: {"sha256": hashlib.sha256((results / name).read_bytes()).hexdigest(),
                          "bytes": (results / name).stat().st_size} for name in names},
    }


def markdown(data):
    lines = ["# 四臂既有结果汇总", "", "主参照：`mixed_ports_zero`。主结论只使用 V 和 external_development；这些是已开开发评价，不是新盲测。",
             "没有整体通过判定或赢家选择。本报告不支持新算法、自动蒸馏或自动转向历史／图的结论。", ""]
    titles = {"development": "未拟合的开发评价", "calibration": "校准及未定校准背景",
              "fit_or_expansion_sources": "拟合／扩充来源诊断（不进入主结论）"}
    for scope in SECTIONS:
        lines += [f"## {titles[scope]}", "", "流统计与事件统计分别报告；未知标签不计 TP、FN 或正常分母。", "",
                  "| 臂 | 角色／来源 | 控制 TP/FN | 下载 TP/FN | 正常 FP/分母 | 未知行／未知告警 |",
                  "|---|---|---:|---:|---:|---:|"]
        for arm in ARMS:
            for row in data["arms"][arm][scope]["flow_metrics"]:
                c, d = row["targets"]["control"], row["targets"]["download"]
                lines.append(f"| {arm} | {row['role']} / {row['source']} | {c['TP']}/{c['FN']} | {d['TP']}/{d['FN']} | {row['FP']}/{row['FP'] + row['TN']} | {row['unknown_label_rows']}/{row['unknown_alerts']} |")
        lines += ["", "正常单位单独判断，单位可能重叠，以下行数不构成独立样本数。", "",
                  "| 臂 | 单位 | FP/正常数 | 整数预算 | 预算内 |", "|---|---|---:|---:|---|"]
        for arm in ARMS:
            for unit in data["arms"][arm][scope]["normal_units"]:
                met = {True: "是", False: "否", None: "无分母"}[unit["empirical_budget_met"]]
                lines.append(f"| {arm} | {unit['unit_id']} | {unit['FP']}/{unit['normal_count']} | {unit['integer_FP_budget']} | {met} |")
        lines += ["", "攻击事件／对象按原类别分开；确定、仅可能、未命中和未决不能混算。", "",
                  "| 臂 | 角色／来源／原 kind | 原登记记录 | 确定命中 | 仅可能命中 | 无候选告警 | 未决 | 多候选 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            for event in data["arms"][arm][scope]["event_groups"]:
                if event["label"] == 1:
                    ident = f"{event['role']} / {','.join(event['sources'])} / {event['kind']}"
                    lines.append(f"| {arm} | {ident} | {event['registered_records']} | {event['certain_alert']} | {event['possible_only_alert']} | {event['no_candidate_alert']} | {event['unresolved']} | {event['multiple_candidates']} |")
        lines.append("")
    lines += ["## 四项固定配对", "", "下表只汇总 V/external_development 的身份变化；各角色、来源和控制／下载分项保存在 JSON 的 by_source_kind。", "",
              "| 比较（前→后） | 新增 TP | 丢失 TP | 新增 FP | 移除 FP | 未知新增／移除告警 | 描述性标记 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    flag_labels = {
        "coverage_gains_observed": "扩充后有利变化",
        "coverage_losses_observed": "扩充后身份回退",
        "port_dependence_observed": "端口两流程判决变化",
        "tradeoff_observed": "有利与不利变化并存",
        "no_known_flow_decision_change": "已知流判决无变化",
    }
    for name, pair in data["comparisons"].items():
        c = pair["development_changes"]
        flags = "；".join(flag_labels[k] for k, v in pair["development_flags"].items() if v)
        lines.append(f"| {name}: {pair['before']} → {pair['after']} | {c['TP_added']} | {c['TP_lost']} | {c['FP_added']} | {c['FP_removed']} | {c['unknown_added_alerts']}/{c['unknown_removed_alerts']} | {flags} |")
    lines += ["", data["interpretation"]["flags"], "", "逐个攻击事件的完整候选、来源和上下界保存在 outcome_summary.json 的 positive_event_records；未决事件保留 null。", "",
              "## 原回传成本", "", data["interpretation"]["cost"], ""]
    cost = data["reported_cost"]
    if cost["status"] == "not_provided":
        lines += ["原回传未提供 cost.json；不补估成本。", ""]
    else:
        reported = cost["data"]
        lines += [f"原 scope：{reported.get('scope', '未提供')}。",
                  f"原设备／线程：{reported.get('device', '未提供')}／{reported.get('threads', '未提供')}。", "",
                  "| 臂 | 拟合秒 | C批评分秒 | 其他批评分秒 | 工件字节 | 拟合行 | 总评分行 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            values = reported.get("arms", {}).get(arm, {})
            fields = ("fit_seconds", "calibration_score_seconds", "remaining_score_seconds",
                      "model_bytes", "fit_rows", "scored_rows")
            lines.append("| " + arm + " | " + " | ".join(str(values.get(key, "未提供")) for key in fields) + " |")
        lines.append("")
    lines += [
              "输入文件 SHA-256 与大小写入 JSON。此汇总读取现成判决，模型加载、模型调用、拟合、阈值计算均为 0。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results, output = args.results.resolve(), args.output.resolve()
    if output == results or results in output.parents:
        parser.error("--output 必须在原回传目录之外；原目录保持只读")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output 必须是新目录或空目录，不覆盖已有结果")
    try:
        data = summarize(results)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    (output / "outcome_summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "outcome_summary.md").write_text(markdown(data), encoding="utf-8")
    print(json.dumps({"status": data["status"], "output": str(output), "model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
