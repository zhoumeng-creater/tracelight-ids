"""Static F/V role and normal-group audit: no models, scores, C, or E are opened.

python data_contract_review.py --thesis-root <thesis root>
Only frozen F/V rows, preprocessing metadata and the existing F group tree are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


COUNTS = ["orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
VISIBLE = ["duration_seconds", *COUNTS]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def keys(frame, columns):
    return [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]


def overlap(left, right):
    ls, rs = set(left), set(right)
    return {
        "shared_distinct_keys": len(ls & rs),
        "left_rows_with_shared_key": sum(k in rs for k in left),
        "right_rows_with_shared_key": sum(k in ls for k in right),
        "left_rows": len(left), "right_rows": len(right),
    }


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def durations(frame):
    a = frame.duration_seconds.to_numpy(float)
    valid = np.isfinite(a) & (a >= 0)
    return a[valid], int((~valid).sum())


def ks_distance(a, b):
    if not len(a) or not len(b):
        return None
    a, b = np.sort(a), np.sort(b)
    grid = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def duration_compare(f, v):
    a, fm = durations(f)
    b, vm = durations(v)
    outside = int(np.count_nonzero((b < a.min()) | (b > a.max()))) if len(a) else len(b)
    return {
        "F_n": len(f), "V_n": len(v), "F_duration_missing": fm, "V_duration_missing": vm,
        "F_duration": stats(a), "V_duration": stats(b),
        "V_valid_duration_outside_F_observed_range": outside,
        "V_valid_duration_denominator": len(b),
        "duration_empirical_KS": ks_distance(a, b),
    }


def build_tree(frame, depth=0, key="P"):
    # A static coverage comparator, not a registered training group contract.
    node = {"key": key, "n": len(frame), "depth": depth}
    if depth >= 5 or len(frame) < 512:
        return node
    for offset in range(4):
        feature = COUNTS[(depth + offset) % 4]
        vals, counts = np.unique(frame[feature], return_counts=True)
        cumulative = np.cumsum(counts)
        candidates = np.flatnonzero((cumulative[:-1] >= 256) & (len(frame) - cumulative[:-1] >= 256))
        if not len(candidates):
            continue
        # np.argmin chooses the first/smaller cut for equal imbalance.
        chosen = candidates[np.argmin(np.abs(2 * cumulative[candidates] - len(frame)))]
        cut = int(vals[chosen])
        selected = frame[feature] <= cut
        node.update({"feature": feature, "cut_le": cut,
                     "left": build_tree(frame[selected], depth + 1, key + "0"),
                     "right": build_tree(frame[~selected], depth + 1, key + "1")})
        return node
    return node


def route(row, node):
    while "feature" in node:
        node = node["left"] if row[node["feature"]] <= node["cut_le"] else node["right"]
    return node["key"]


def network_keys(frame, preprocessing):
    base = preprocessing["base"]
    a = frame[VISIBLE].to_numpy(float)
    missing = np.isnan(a) | (a == -1)
    transformed = ((np.log1p(np.where(missing, np.asarray(base["median"]), a))
                    - np.asarray(base["mean"])) / np.asarray(base["scale"])).astype(np.float32)
    # The protocol-blind channel is constant; the five missing masks remain visible.
    assert preprocessing["protocol_policy"] == "constant_unknown"
    assert np.isfinite(transformed).all()
    return [x.tobytes() + m.tobytes() for x, m in zip(transformed, missing)]


def group_summary(f, v, f_groups, v_groups):
    result = []
    for group in sorted(set(f_groups)):
        a, b = f.loc[f_groups == group], v.loc[v_groups == group]
        fc = Counter(keys(a, COUNTS))
        result.append({
            "group": group, **duration_compare(a, b),
            "F_exact_patterns": len(fc), "F_singleton_patterns": sum(n == 1 for n in fc.values()),
            "F_profile_counts": a.profile.value_counts().to_dict(),
            "V_profile_counts": b.profile.value_counts().to_dict(),
            "one_singleton_pattern_all_alerts_leaf_risk": 1 / len(a) if any(n == 1 for n in fc.values()) else None,
            "F_1percent_integer_budget_if_misused_as_hard_test": len(a) // 100,
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.thesis_root.resolve()
    output = args.output_dir or root / "工作/操作风险修正_20261118"
    output.mkdir(parents=True, exist_ok=True)
    archive = root / "实验归档/20261118_正常子群训练结果/回传"
    paths = {
        "F": archive / "inputs/F.parquet", "V": archive / "inputs/V.parquet",
        "old_groups": archive / "preflight/groups.json",
        "preprocessing": archive / "models/transport_target_s17/trajectory/preprocessing.json",
        "preprocessing_s29": archive / "models/transport_target_s29/trajectory/preprocessing.json",
        "preprocessing_s43": archive / "models/transport_target_s43/trajectory/preprocessing.json",
    }
    f, v = pd.read_parquet(paths["F"]), pd.read_parquet(paths["V"])
    old = json.loads(paths["old_groups"].read_text(encoding="utf8"))
    preprocessing = json.loads(paths["preprocessing"].read_text(encoding="utf8"))
    assert all(json.loads(paths[name].read_text(encoding="utf8")) == preprocessing
               for name in ["preprocessing_s29", "preprocessing_s43"])
    f0, v0 = f[f.label == 0].copy(), v[v.label == 0].copy()
    assert len(f0) == 14691 and len(v0) == 18645
    assert not f[COUNTS].isna().any().any() and not v[COUNTS].isna().any().any()
    assert (f[COUNTS] >= 0).all().all() and (v[COUNTS] >= 0).all().all()
    fk, vk = keys(f0, COUNTS), keys(v0, COUNTS)
    fc, vc = Counter(fk), Counter(vk)
    f_pattern_groups = f0.groupby(COUNTS, sort=True)
    v_pattern_groups = v0.groupby(COUNTS, sort=True)
    empty = v0.iloc[:0]
    pattern_rows = []
    for pattern, a in f_pattern_groups:
        b = v_pattern_groups.get_group(pattern) if pattern in v_pattern_groups.groups else empty
        pattern_rows.append({"pattern": pattern, **duration_compare(a, b),
                             "F_profile_counts": a.profile.value_counts().to_dict(),
                             "V_profile_counts": b.profile.value_counts().to_dict()})
    known = np.array([key in fc for key in vk])
    range_lookup = {}
    for pattern, a in f_pattern_groups:
        valid, missing = durations(a)
        range_lookup[pattern] = (valid.min(), valid.max()) if len(valid) else None
    outside = []
    known_missing = []
    no_valid_f_support = []
    for key, duration in zip(vk, v0.duration_seconds):
        bounds = range_lookup.get(key)
        known_missing.append(key in fc and (pd.isna(duration) or duration < 0))
        no_valid_f_support.append(key in fc and duration >= 0 and bounds is None)
        outside.append(key in fc and duration >= 0 and bounds is not None and (duration < bounds[0] or duration > bounds[1]))
    v0["F_pattern_known"] = known
    v0["duration_outside_F_pattern_observed_range"] = outside
    coverage = v0.groupby("source_id").agg(
        n=("label", "size"), F_pattern_known=("F_pattern_known", "sum"),
        duration_outside=("duration_outside_F_pattern_observed_range", "sum")).reset_index().to_dict("records")
    support = []
    for n in [1, 2, 5, 10, 20, 50, 100, 256]:
        selected = {key for key, count in fc.items() if count >= n}
        support.append({"display_cut_only_not_a_training_filter": n, "F_groups": len(selected),
                        "F_rows": sum(fc[key] for key in selected), "V_rows": sum(vc[key] for key in selected)})
    eligible_descriptive = [x for x in pattern_rows if x["F_n"] >= 20 and x["V_n"] >= 20]
    pooled_tree = build_tree(f0)
    f_old_groups = np.array([route(row, old["trees"][row["profile"]]) for row in f0.to_dict("records")])
    v_old_groups = np.array([route(row, old["trees"][row["profile"]]) for row in v0.to_dict("records")])
    f_pooled_groups = np.array([route(row, pooled_tree) for row in f0.to_dict("records")])
    v_pooled_groups = np.array([route(row, pooled_tree) for row in v0.to_dict("records")])
    old_leaves = group_summary(f0, v0, f_old_groups, v_old_groups)
    pooled_leaves = group_summary(f0, v0, f_pooled_groups, v_pooled_groups)
    f_min, f_max = f0[COUNTS].min(), f0[COUNTS].max()
    count_outside = ((v0[COUNTS] < f_min) | (v0[COUNTS] > f_max)).any(axis=1)

    # A source-level candidate. It is not the only legal role allocation.
    roles = {
        "Q_F": f[(f.source_id == "TQH-B_i30_j0") & (f.label == 0)],
        "N_F_risk": f[(f.source_id != "TQH-B_i30_j0") & (f.label == 0)],
        "P_F_protection": f[f.label == 1],
        "Q_V": v[(v.source_id == "TQH-B_i300_j0") & (v.label == 0)],
        "N_V_risk": v[(v.source_id != "TQH-B_i300_j0") & (v.label == 0)],
        "P_V_protection": v[(v.source_id != "TQH-B_i300_j0") & (v.label == 1)],
        "V_excluded_Q_cell_positive": v[(v.source_id == "TQH-B_i300_j0") & (v.label == 1)],
        "F_all": f, "V_all": v,
    }
    for original, components in [(f, ["Q_F", "N_F_risk", "P_F_protection"]),
                                 (v, ["Q_V", "N_V_risk", "P_V_protection", "V_excluded_Q_cell_positive"])]:
        pieces = [key for name in components for key in keys(roles[name], ["source_id", "row_id"])]
        assert len(pieces) == len(set(pieces)) == len(original)
        assert set(pieces) == set(keys(original, ["source_id", "row_id"]))
    role_rows = {}
    for name, frame in roles.items():
        role_rows[name] = {
            "n": len(frame), "by_source": frame.groupby(["source_id", "label"]).size().rename("n").reset_index().to_dict("records"),
            "identity_sha256": hashlib.sha256(json.dumps(sorted(keys(frame, ["source_id", "row_id"])), separators=(",", ":")).encode()).hexdigest(),
            "four_count_patterns": len(set(keys(frame, COUNTS))),
            "raw_model_inputs": len(set(keys(frame, VISIBLE))),
            "FP32_model_inputs": len(set(network_keys(frame, preprocessing))),
        }
    pairs = [("Q_F", "N_F_risk"), ("Q_F", "P_F_protection"), ("Q_V", "N_V_risk"),
             ("Q_V", "P_V_protection"), ("F_all", "Q_V"), ("F_all", "N_V_risk"),
             ("F_all", "P_V_protection"), ("F_all", "V_all")]
    overlaps = {}
    for a, b in pairs:
        overlaps[f"{a}__{b}"] = {
            "identities": overlap(keys(roles[a], ["source_id", "row_id"]), keys(roles[b], ["source_id", "row_id"])),
            "four_counts": overlap(keys(roles[a], COUNTS), keys(roles[b], COUNTS)),
            "five_visible_raw_fields": overlap(keys(roles[a], VISIBLE), keys(roles[b], VISIBLE)),
            "actual_frozen_FP32_inputs": overlap(network_keys(roles[a], preprocessing), network_keys(roles[b], preprocessing)),
        }
    assert all(x["identities"]["shared_distinct_keys"] == 0 for x in overlaps.values())
    combined = pd.concat([f, v], ignore_index=True)
    cross_label = {
        "four_counts": overlap(keys(combined[combined.label == 0], COUNTS), keys(combined[combined.label == 1], COUNTS)),
        "FP32_model_inputs": overlap(network_keys(combined[combined.label == 0], preprocessing), network_keys(combined[combined.label == 1], preprocessing)),
    }
    timeline = []
    for role, frame in [("F", f), ("V", v)]:
        for source, group in frame.groupby("source_id"):
            valid = group.duration_seconds >= 0
            timeline.append({"role": role, "source_id": source, "profile": group.profile.iloc[0],
                             "rows": len(group), "normal": int((group.label == 0).sum()), "positive": int((group.label == 1).sum()),
                             "flow_end_min_ms": int(group.flow_end_ms.min()), "flow_end_max_ms": int(group.flow_end_ms.max()),
                             "known_flow_start_min_ms": float((group.loc[valid, "flow_end_ms"] - 1000 * group.loc[valid, "duration_seconds"]).min()),
                             "unknown_start_duration_rows": int((~valid).sum())})
    gap = {}
    for profile in ["A", "B"]:
        f_end = max(x["flow_end_max_ms"] for x in timeline if x["profile"] == profile and x["role"] == "F")
        v_start = min(x["known_flow_start_min_ms"] for x in timeline if x["profile"] == profile and x["role"] == "V")
        gap[profile] = {"F_last_end_ms": f_end, "V_first_known_start_ms": v_start, "known_interval_gap_ms": v_start - f_end}
    q_ranks = {name: {"n_normal": len(roles[name]), "registered_formula_rank_only_not_a_threshold": math.ceil((len(roles[name]) + 1) * 0.99),
                      "valid_rank": math.ceil((len(roles[name]) + 1) * 0.99) <= len(roles[name])}
               for name in ["Q_F", "Q_V"]}
    result = {
        "scope": "Only frozen F/V inputs and metadata; no score vector, C, D, E, model, calibration, or training used",
        "new_models": 0, "new_forwards": 0, "new_training": 0, "new_thresholds": 0,
        "input_files": {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for name, path in paths.items()},
        "denominators": {"F": len(f), "F_normal": len(f0), "F_positive": int((f.label == 1).sum()), "V": len(v), "V_normal": len(v0), "V_positive": int((v.label == 1).sum())},
        "four_count_exact_patterns": {
            "definition": COUNTS, "profile_and_duration_not_in_definition": True,
            "F_patterns": len(fc), "F_singletons": sum(n == 1 for n in fc.values()),
            "V_patterns": len(vc), "V_known_patterns": len(set(fc) & set(vc)), "V_known_rows": int(known.sum()),
            "V_unknown_rows": int((~known).sum()), "V_known_fraction": float(known.mean()),
            "F_patterns_absent_V": len(set(fc) - set(vc)), "support_table": support,
            "coverage_by_V_source": coverage,
            "V_known_pattern_missing_duration_rows": sum(known_missing),
            "V_known_pattern_valid_duration_without_any_valid_F_duration": sum(no_valid_f_support),
            "V_known_pattern_valid_duration_outside_F_valid_range": sum(outside),
            "F_missing_duration_rows": int((f0.duration_seconds < 0).sum()), "V_missing_duration_rows": int((v0.duration_seconds < 0).sum()),
            "descriptive_both_n_at_least_20": {
                "not_a_selection_or_training_filter": True, "groups": len(eligible_descriptive),
                "F_rows": sum(x["F_n"] for x in eligible_descriptive), "V_rows": sum(x["V_n"] for x in eligible_descriptive),
                "duration_empirical_KS": stats([x["duration_empirical_KS"] for x in eligible_descriptive if x["duration_empirical_KS"] is not None]),
            },
            "all_F_pattern_details": pattern_rows,
        },
        "original_32_leaves": {"requires_profile": True, "leaves": old_leaves},
        "profile_free_coarse_static_comparator": {
            "not_a_registered_training_risk_family": True,
            "method": "Same deterministic count-balanced split as old tree; pooled F normal; min child 256; maximum depth 5; at most 32 leaves",
            "tree": pooled_tree, "leaves": pooled_leaves, "V_routed_rows": len(v0),
            "V_count_rows_outside_global_F_normal_coordinate_range": int(count_outside.sum()),
            "leaves_containing_at_least_one_F_singleton_pattern": sum(x["F_singleton_patterns"] > 0 for x in pooled_leaves),
        },
        "collection_cell_risk_family": {
            "F_normal_cells": f0.groupby("source_id").size().to_dict(), "V_normal_cells": v0.groupby("source_id").size().to_dict(),
            "new_capture_source_has_no_natural_existing_cell_assignment": True,
            "meaning": "Empirical per-capture stress test; does not imply component-wise normal risk control",
        },
        "candidate_roles": {
            "status": "source-role candidate, not unique or final frozen registration", "roles": role_rows,
            "threshold_rank_formula_only": q_ranks, "overlap_audit": overlaps,
            "all_F_positive_retained": True,
            "Q_F_is_training_reference_not_independent_calibration": True,
            "Q_V_not_in_training_but_used_for_selection_not_independent_final_confirmation": True,
            "V_B_j0_30_positive_rows_excluded_from_selection": True,
            "normal_risk_training_group_counts_must_be_refit_on_N_F_after_Q_F_allocation": True,
            "F_and_V_roles_each_partition_original_identities_exactly": True,
            "three_seed_preprocessing_metadata_identical": True,
        },
        "all_FV_normal_positive_input_collisions": cross_label,
        "timeline": timeline, "F_before_V_known_interval_gaps": gap,
        "verdict": {
            "exact_patterns_as_exhaustive_primary_risk_family": "reject: sparse plus 45.67 percent unseen V rows; four counts do not fix duration or missingness",
            "old_profile_trees_as_deployable_condition_family": "reject: source route required and heterogeneous within leaves",
            "coarse_profile_free_tree": "routes every row but does not establish component risk invariance; rare patterns are absorbed",
            "collection_cells": "retain as fixed empirical capture-level diagnostics; not a substitute for component-risk coverage",
            "ready_for_data_contract_and_operating_point_implementation": True,
            "ready_for_final_risk_family_or_real_training_registration": False,
            "open_item": "A fixed visible risk family with support and a justified relation to targeted composition changes; plus independent normal evidence for unobserved captures if a deployment risk claim is required",
        },
    }
    result = plain(result)
    (output / "data_contract_review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    pattern = result["four_count_exact_patterns"]
    lines = ["# 操作风险修正：F/V 数据角色与正常风险族静态审查", "",
             "**数据角色可以实施；正常风险族尚不能冻结。** 精确四计数模式过稀疏，粗分组虽然能给所有行分配编号，却不能据此声称覆盖了条件风险变化。现有数据没有同输入异标签冲突，不能把问题归结为有限 F/V 本身不可分。", "",
             "本审查仅打开已有 F/V 输入、旧分组元数据和预处理元数据。模型、缓存分数、C、D、E 均未读取；模型加载、前向、训练、新校准与阈值搜索均为 0。", "",
             "## 1. 精确四计数模式不适合作为完整主风险族", "",
             f"F 的 14691 条正常产生 {len(fc)} 个精确四计数模式，其中 {pattern['F_singletons']} 个只有一条。V 的 18645 条正常中，仅 {int(known.sum())} 条（{known.mean():.2%}）来自 F 出现过的模式，另 {int((~known).sum())} 条没有精确模式支持。分组不含 profile、duration、协议、分数或事后错误筛选。", "",
             "| F 模式最小样本数（仅描述） | 模式数 | F 行 | 覆盖 V 行 |", "|---:|---:|---:|---:|"]
    for row in support:
        lines.append(f"| {row['display_cut_only_not_a_training_filter']} | {row['F_groups']} | {row['F_rows']} | {row['V_rows']} |")
    lines += ["", "提高支持门槛会删除稀有模式，也会进一步降低覆盖；把未见模式塞进 OTHER 只解决编号问题，不会建立它们与已见模式相同的风险。", "",
              f"即使四计数相同，duration 仍会变化：在已知模式的 V 行中，{sum(outside)} 条有效时长超出对应 F 模式的有效时长观测范围，{sum(known_missing)} 条为缺失时长。全 F/V 正常分别有 {int((f0.duration_seconds < 0).sum())}/{int((v0.duration_seconds < 0).sum())} 条时长为 -1；已按缺失处理，没有把 -1 当实际时长。观测范围越界是经验条件变化，不能直接解释为总体分布检验。", "",
              "| V 来源 | 正常分母 | F 精确模式已见 | 已见模式有效时长越界 |", "|---|---:|---:|---:|"]
    for row in coverage:
        lines.append(f"| {row['source_id']} | {row['n']} | {row['F_pattern_known']} | {row['duration_outside']} |")
    lines += ["", "## 2. 粗分组的覆盖与少数成分保护是两回事", "",
              "旧 32 叶需先按 A/B profile 路由；它能作为已有采集数据的诊断分层，但不能成为新来源可自然套用的机制条件。采集 cell 同样只能描述整次采集的平均经验风险；cell 内一个低占比模式全部误报，平均值仍可能很低。", "",
              f"另作一个静态覆盖对照：沿用旧确定性四计数平衡规则，取消 A/B 路由，在 pooled F 正常上最多建 32 叶、最小子叶 256 行。得到 {len(pooled_leaves)} 叶，F 样本数范围 {min(x['F_n'] for x in pooled_leaves)}–{max(x['F_n'] for x in pooled_leaves)}，V 范围 {min(x['V_n'] for x in pooled_leaves)}–{max(x['V_n'] for x in pooled_leaves)}。全部 18645 条 V 能路由，但仍有 {int(count_outside.sum())} 行四计数至少一维超出 F 正常全局观测范围。该树仅用于说明覆盖问题，不是登记的新候选。", "",
              f"{sum(x['F_singleton_patterns'] > 0 for x in pooled_leaves)}/{len(pooled_leaves)} 叶含 F 单例模式。每叶至少 256 行，因此一个单例模式 100% 误报只贡献至多 1/256=0.390625% 的叶风险，仍可低于 1%。这是粗分组掩盖稀有成分的明确反例；编号全覆盖不能保证成分风险全覆盖。", "",
              f"条件时长也没有因粗分组固定：旧32叶有 {sum(x['V_valid_duration_outside_F_observed_range'] for x in old_leaves)} 条 V 有效时长超出各自 F 叶范围，最大经验 KS 距离为 {max(x['duration_empirical_KS'] for x in old_leaves):.3f}；无 profile 粗树分别为 {sum(x['V_valid_duration_outside_F_observed_range'] for x in pooled_leaves)} 条和 {max(x['duration_empirical_KS'] for x in pooled_leaves):.3f}。这些是经验分布差异，不是独立样本假设下的显著性检验。", "",
              "推荐保留来源 cell 和可见计数分层作固定诊断，否决把精确模式、旧32叶或新粗树直接宣布为已经闭合的主风险族。下一步需要固定的是机制风险族及其支持解释，不是从这些表中挑一个样本数门槛后立即开训。", "",
              "## 3. 一个按既有来源角色划分的完整候选", "",
              "下表是一种可执行的来源级分割，不是唯一合法方案。不随机拆同行、不改原 F/V 归属：", "",
              "| 角色 | 固定身份规则 | 行数 |", "|---|---|---:|",
              "| Q_F | F 中 B_i30_j0 的正常 | 1935 |",
              "| 训练正常风险 N_F | F 其余五个来源的正常 | 12756 |",
              "| 训练攻击保护 P_F | F 全部攻击，含 Q_F 来源中的30条攻击 | 5213 |",
              "| Q_V | V 中 B_i300_j0 的正常 | 1569 |",
              "| 验证正常风险 N_V | V 其余五个来源的正常 | 17076 |",
              "| 验证攻击保护 P_V | V 其余五个来源的攻击 | 2775 |",
              "| 不参与选择的 V 攻击 | B_i300_j0 的30条攻击 | 30 |", "",
              "F 三个角色和 V 四个角色分别完整、互斥地划分原身份。选择 B 的最早采集作 Q，是为了复用原 B-only 正常参照的来源角色并保持采集完整；这项选择不宣称该 cell 代表所有未来 B 或 D 正常。", "",
              f"仅核对序位公式有效性：Q_F 的 99% 序位为 {q_ranks['Q_F']['registered_formula_rank_only_not_a_threshold']}，Q_V 为 {q_ranks['Q_V']['registered_formula_rank_only_not_a_threshold']}；未计算任何分数阈值。Q_F 本来就是训练材料，warm start 也已经见过，不能称独立校准。Q_V 不进入训练，但参与选轮，因此不能称最终独立确认。", "",
              "若采用此分割，训练正常风险组的支持与统计必须在 N_F 重新确定，不能把被划入 Q_F 的行继续计入风险组分母。现有 F-only 分组统计只是覆盖审查。完整 F 正例仍可保护；Q_F 来源的30条正例是训练约束，不是独立验证。", "",
              "## 4. 身份不重叠，不等于输入和采集过程独立", "",
              "全部上述跨角色身份交集为 0。按原冻结预处理实际 FP32 五维数值加缺失掩码检查：", "",
              "| 角色对 | 重复输入数 | 左侧重复行 | 右侧重复行 |", "|---|---:|---:|---:|"]
    for name, report in overlaps.items():
        x = report["actual_frozen_FP32_inputs"]
        lines.append(f"| {name} | {x['shared_distinct_keys']} | {x['left_rows_with_shared_key']} | {x['right_rows_with_shared_key']} |")
    lines += ["", "这些是相同可见流型的重复，不是 source_id/row_id 身份泄漏。输入相同不能证明同一物理流被复制，也不能证明两次采集独立；现有表没有足以鉴别重放关系的原始事件身份。不为制造独立性而删除常见流型，那会改变 Q 的正常组成。", "",
              f"全 F/V 正常与攻击的四计数冲突为 {cross_label['four_counts']['shared_distinct_keys']}，实际 FP32 输入异标签冲突也为 {cross_label['FP32_model_inputs']['shared_distinct_keys']}。这只排除了已观测有限数据的同输入异标签障碍，不证明当前模型优化能达到所有风险约束。", "",
              f"按可恢复的流起止区间，A 的最后 F 与最早 V 间隔 {gap['A']['known_interval_gap_ms']/1000:.3f} 秒，B 间隔 {gap['B']['known_interval_gap_ms']/1000:.3f} 秒。缺失 duration 的行无法恢复流起点，不能用该计算证明所有物理流区间独立。", "",
              "## 5. 不能把 minimax 目标误写成小组验收", "",
              "风险族用于优化经验风险或其明确代理；它不是给每个小组强加原15单元的1%通过线。若 n<100，按 floor(0.01n) 自动设硬预算会变成0FP；对单例组尤其会把 max 经验风险退化成任何一条正常告警都失败。这样的统计与优化对象必须先说明，不能靠分组规模隐式改变任务。", "",
              "最终15单元的原分母与预算保留，由冻结工件按原 C 程序评价；开发 minimax 风险降低不自动等于15单元通过。零误报也不是1%总体风险证书：在理想独立 Bernoulli 条件下，零误报的单组单侧95%上界仍为 1−0.05^(1/n)，n=100 时约2.95%，n=256 时约1.16%；当前重复流型与采集相关性还不能直接套用该独立假设。", "",
              "## 6. 当前可以实施到哪一步", "",
              "可以实现并核验：上述候选身份分配、分母与重叠检查、严格序位判决、完整检查点分数记录。不能据此冻结主正常风险族或登记真实训练。具体未闭合项是：一个仅由可见信息定义、有足够支持、又不把稀有正常成分平均掉的风险族，以及它与要抵抗的采集组成变化之间的依据。若要作部署风险保证，还需另立新采集的正常证据合同；F/V 身份划开并不能补出这份证据。", "",
              "全部来源时间、分组支持、时长比较、候选身份摘要和 SHA-256 见同名 JSON；完整7134个 F 模式亦保存在其中。复现：", "", "```powershell",
              f'python data_contract_review.py --thesis-root "{root}"', "```"]
    (output / "data_contract_review.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(json.dumps({"F_patterns": len(fc), "V_known": int(known.sum()), "duration_outside": int(sum(outside)), "coarse_leaves": len(pooled_leaves), "role_overlap_zero": True, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
