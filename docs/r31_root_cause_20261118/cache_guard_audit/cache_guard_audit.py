"""R31 guard audit from frozen caches only; no model import or scoring.

Run: python cache_guard_audit.py --thesis-root <root> [--output-dir <dir>]
The original C threshold is read once as a descriptive reference, never fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return str(value[0]), int(value[1])


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0}
    return {
        "n": int(values.size), "min": float(values.min()),
        "median": float(np.median(values)), "max": float(values.max()),
        "sum": float(values.sum()),
    }


def count_rows(frame, columns):
    return [dict(zip(columns, key if isinstance(key, tuple) else (key,)), n=int(n))
            for key, n in frame.groupby(columns, dropna=False).size().items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.thesis_root.resolve()
    output = args.output_dir or root / "工作/正常子群根因研究_20261118"
    output.mkdir(parents=True, exist_ok=True)
    r31 = root / "实验归档/20261118_正常子群训练结果/回传"
    r30 = root / "实验归档/20261117_第三十轮模型训练重设计/回传"
    audit_path = root / "工作/正常子群结果分析_20261118/training_audit.json"
    inputs = {}

    def record(path):
        inputs[str(path.relative_to(root))] = {
            "sha256": sha256(path), "bytes": path.stat().st_size,
        }
        return path

    def read_json(path):
        return json.loads(record(path).read_text(encoding="utf-8-sig"))

    audit = read_json(audit_path)
    v = pd.read_parquet(record(r31 / "inputs/V.parquet"))
    old_v = pd.read_parquet(record(r30 / "inputs/V.parquet"))
    assert v.equals(old_v)
    v = v.sort_values(["source_id", "row_id"], kind="stable").reset_index(drop=True)
    ids = [identity(row) for row in v[["source_id", "row_id"]].itertuples(index=False, name=None)]
    assert len(ids) == len(set(ids)) == 21450
    declared_ids = read_json(r31 / "preflight/identities.json")
    assert ids == list(map(identity, declared_ids["V"]))
    index = {key: i for i, key in enumerate(ids)}
    positive = v.label.to_numpy() == 1
    normal = ~positive
    positive_indices = np.flatnonzero(positive)
    normal_indices = np.flatnonzero(normal)
    pids = [ids[i] for i in positive_indices]
    assert int(positive.sum()) == 2805 and int(normal.sum()) == 18645

    all_rejected = set()
    by_trajectory = {}
    event_rows = []
    panel_failure_events = []
    rejected_events_with_panels_strict = 0
    # Rebuild every worsening identity directly from archived event metrics;
    # compare with the separately generated training_audit, rather than trust it.
    for name, analysis in audit["trajectories"].items():
        history_path = r31 / "models" / name / "trajectory/history.json"
        baseline_path = r31 / "models" / name / "trajectory/baseline_metrics.json"
        history, baseline = read_json(history_path), read_json(baseline_path)
        assert len(history) == len(analysis["events"]) == 16
        assert list(map(identity, baseline["V_positive_ids"])) == pids
        base_inv = np.asarray(baseline["V_positive_inversions"], dtype=np.int64)
        seen = set()
        failed_events = 0
        event_summary = []
        for original, reviewed in zip(history, analysis["events"]):
            assert original["step"] == reviewed["step"]
            metrics = original["metrics"]
            assert list(map(identity, metrics["V_positive_ids"])) == pids
            current = np.asarray(metrics["V_positive_inversions"], dtype=np.int64)
            worse = np.flatnonzero(current > base_inv)
            extracted = {
                identity(row["identity"]): (int(row["baseline"]), int(row["current"]), int(row["delta"]))
                for row in reviewed["positive_inversion_change"]["V"]["worse_rows"]
            }
            rebuilt = {pids[i]: (int(base_inv[i]), int(current[i]), int(current[i] - base_inv[i]))
                       for i in worse}
            assert extracted == rebuilt
            assert bool(len(worse)) == (not reviewed["guard_checks"]["V_each_positive_inversions_not_increased"])
            failed_events += bool(len(worse))
            panels_strict = all(panel["strictly_separated"] for panel in metrics["panels"].values())
            assert metrics["panels"] == reviewed["panels"]
            rejected_events_with_panels_strict += bool(len(worse)) and panels_strict
            if not panels_strict:
                panel_failure_events.append({"trajectory": name, "step": int(original["step"]), "panels": metrics["panels"]})
            seen.update(rebuilt)
            for key, (before, after, delta) in rebuilt.items():
                row = v.iloc[index[key]]
                event_rows.append({
                    "trajectory": name, "seed": int(analysis["execution"]["seed"]),
                    "arm": analysis["execution"]["arm"], "step": int(original["step"]),
                    "source_id": key[0], "row_id": key[1],
                    "profile": row.profile, "ip_protocol": int(row.ip_protocol),
                    "baseline_pooled_normal_ge_count": before,
                    "current_pooled_normal_ge_count": after, "delta": delta,
                    "normal_denominator": int(normal.sum()),
                    "current_rank_fraction": after / int(normal.sum()),
                })
            event_summary.append({
                "step": int(original["step"]), "worse_positive_count": int(len(worse)),
                "current_max_inversions": int(current.max()),
                "current_max_rank_fraction": float(current.max() / normal.sum()),
                "worsening_delta_sum": int((current[worse] - base_inv[worse]).sum()),
                "V_panel_strict_separation": reviewed["guard_checks"]["V_panel_strict_separation"],
                "V_pooled_pauc_not_lower": reviewed["guard_checks"]["V_pooled_pauc_not_lower"],
                "only_failed_guard": reviewed["failed_guards"] == ["V_each_positive_inversions_not_increased"],
            })
        assert seen == {identity(row["identity"]) for row in analysis["positive_changes"]["V"]["rows"]}
        all_rejected.update(seen)
        by_trajectory[name] = {
            "seed": int(analysis["execution"]["seed"]), "rejected_events": int(failed_events),
            "distinct_worsened_positive_rows": len(seen),
            "identities": [list(x) for x in sorted(seen)], "events": event_summary,
        }

    rejected_indices = np.array([index[key] for key in sorted(all_rejected)], dtype=int)
    baseline_rows, normal_rows, pair_rows = [], [], []
    seed_results = {}
    for seed in [17, 29, 43]:
        cache_path = r31 / f"preflight/protocol_blind_s{seed}/reference_scores.npz"
        with np.load(record(cache_path), allow_pickle=False) as cache:
            scores = cache["V"].copy()
        old_path = r30 / f"models/protocol_blind_s{seed}/trajectory/selected_scores.npz"
        with np.load(record(old_path), allow_pickle=False) as cache:
            old_scores = cache["validation_scores"].copy()
        old_ids = read_json(r30 / f"models/protocol_blind_s{seed}/trajectory/score_identities.json")
        assert list(map(identity, old_ids["V"])) == ids
        assert scores.shape == old_scores.shape == (len(v),)
        assert scores.dtype == old_scores.dtype == np.dtype("float64")
        assert np.isfinite(scores).all() and np.isfinite(old_scores).all()
        calibration = read_json(r30 / f"evidence/protocol_blind_s{seed}/calibration.json")
        assert calibration["n_normal"] == 5511 and calibration["rank"] == 5457
        assert calibration["comparison"] == ">"
        threshold = float(calibration["threshold"])
        ns, ps = scores[normal], scores[positive]
        inverse = np.array([np.count_nonzero(ns >= score) for score in ps], dtype=np.int64)
        old_inverse = np.array([np.count_nonzero(old_scores[normal] >= score) for score in old_scores[positive]], dtype=np.int64)
        assert {pids[i] for i in np.flatnonzero(inverse)} == all_rejected
        for arm in ["identity_target", "transport_target"]:
            base = read_json(r31 / f"models/{arm}_s{seed}/trajectory/baseline_metrics.json")
            assert np.array_equal(inverse, base["V_positive_inversions"])
        pairs = []
        offending_normal = Counter()
        for pi, positive_index in enumerate(positive_indices):
            row = v.iloc[positive_index]
            score = float(scores[positive_index])
            mask = ns >= score
            relevant = normal_indices[mask]
            by_source = Counter()
            by_profile_protocol = Counter()
            same_profile_count = 0
            panel_count = 0
            for ni in relevant:
                nrow = v.iloc[ni]
                offending_normal[int(ni)] += 1
                by_source[str(nrow.source_id)] += 1
                by_profile_protocol[f"{nrow.profile}/{int(nrow.ip_protocol)}"] += 1
                same_profile_count += nrow.profile == row.profile
                panel_count += nrow.profile == row.profile and nrow.ip_protocol == row.ip_protocol
                pair = {
                    "seed": seed, "positive_source_id": str(row.source_id),
                    "positive_row_id": int(row.row_id), "positive_score": score,
                    "normal_source_id": str(nrow.source_id), "normal_row_id": int(nrow.row_id),
                    "normal_profile": str(nrow.profile), "normal_ip_protocol": int(nrow.ip_protocol),
                    "normal_score": float(scores[ni]),
                }
                pairs.append(pair)
            baseline_rows.append({
                "seed": seed, "source_id": str(row.source_id), "row_id": int(row.row_id),
                "profile": str(row.profile), "ip_protocol": int(row.ip_protocol),
                "score": score, "R30_cached_score": float(old_scores[positive_index]),
                "old_C_threshold": threshold, "score_minus_old_C_threshold": score - threshold,
                "above_old_C_threshold": bool(score > threshold),
                "rejected_in_any_R31_event": ids[positive_index] in all_rejected,
                "pooled_normal_ge_count": int(inverse[pi]),
                "same_profile_normal_ge_count": int(same_profile_count),
                "same_profile_protocol_normal_ge_count": int(panel_count),
                "normal_ge_by_source": dict(sorted(by_source.items())),
                "normal_ge_by_profile_protocol": dict(sorted(by_profile_protocol.items())),
            })
        pair_rows.extend(pairs)
        for ni, count in sorted(offending_normal.items()):
            row = v.iloc[ni]
            normal_rows.append({
                "seed": seed, "source_id": str(row.source_id), "row_id": int(row.row_id),
                "profile": str(row.profile), "ip_protocol": int(row.ip_protocol),
                "family": str(row.family), "score": float(scores[ni]),
                "score_minus_old_C_threshold": float(scores[ni] - threshold),
                "positive_identities_outranked": int(count),
            })
        current_rows = [x for x in baseline_rows if x["seed"] == seed]
        rejected_rows = [x for x in current_rows if x["rejected_in_any_R31_event"]]
        current_normals = pd.DataFrame([x for x in normal_rows if x["seed"] == seed])
        pair_frame = pd.DataFrame(pairs)
        source_normal_counts = []
        for profile, frame in v[normal].groupby("profile"):
            indices = frame.index.to_numpy()
            source_normal_counts.append({
                "profile": profile, "normal_denominator": len(frame),
                "normal_above_old_C_threshold": int(np.count_nonzero(scores[indices] > threshold)),
            })
        seed_results[str(seed)] = {
            "calibration_reference_only": calibration,
            "R30_cache_vs_R31_CPU_anchor": {
                "same_identities_and_order": True,
                "bitwise_equal_rows": int(np.count_nonzero(scores.view(np.uint64) == old_scores.view(np.uint64))),
                "rows": len(v), "max_abs_score_difference": float(np.max(np.abs(scores - old_scores))),
                "positive_inversion_counts_identical": bool(np.array_equal(inverse, old_inverse)),
                "old_R30_all_positive_above_old_C_threshold": bool(np.all(old_scores[positive] > threshold)),
                "old_R30_min_positive_margin": float(np.min(old_scores[positive] - threshold)),
            },
            "all_positive": {
                "n": int(positive.sum()), "above_old_C_threshold": int(np.count_nonzero(ps > threshold)),
                "score": stats(ps), "margin_to_old_C": stats(ps - threshold),
                "positive_with_pooled_inverse": int(np.count_nonzero(inverse)),
                "pooled_inverse_pairs": int(inverse.sum()),
                "max_inverse_count": int(inverse.max()),
                "max_inverse_fraction": float(inverse.max() / normal.sum()),
            },
            "ever_rejected_positive_union": {
                "n": len(rejected_rows),
                "score": stats([x["score"] for x in rejected_rows]),
                "margin_to_old_C": stats([x["score_minus_old_C_threshold"] for x in rejected_rows]),
                "same_profile_protocol_inverse_pairs": sum(x["same_profile_protocol_normal_ge_count"] for x in rejected_rows),
            },
            "normal_reference_threshold_region": {
                "normal_denominator": int(normal.sum()),
                "normal_above_old_C_threshold": int(np.count_nonzero(ns > threshold)),
                "by_profile": source_normal_counts,
                "distinct_normal_outranking_any_positive": len(offending_normal),
                "baseline_inverse_normal_by_profile_protocol": count_rows(current_normals, ["profile", "ip_protocol"]),
                "baseline_inverse_pairs_by_profile_protocol": count_rows(pair_frame, ["normal_profile", "normal_ip_protocol"]),
                "baseline_inverse_pairs_by_source": count_rows(pair_frame, ["normal_source_id"]),
            },
            "panels": base["panels"], "pooled": base["pooled"],
            "rejected_identity_baseline_details": rejected_rows,
        }

    events_frame = pd.DataFrame(event_rows)
    output_data = {
        "scope": "Frozen cached-score/metric diagnosis only; no selection change or rejected-checkpoint scoring",
        "new_model_loads": 0, "new_forwards": 0, "new_training": 0,
        "new_calibrations": 0, "threshold_searches": 0,
        "identity_columns": ["source_id", "row_id"],
        "inverse_definition": "count(V normal score >= each V positive score); ties count against positive",
        "V_rows": len(v), "normal_denominator": int(normal.sum()), "positive_denominator": int(positive.sum()),
        "denominators_by_profile_protocol_label": count_rows(v, ["profile", "ip_protocol", "label"]),
        "denominators_by_source_label": count_rows(v, ["source_id", "label"]),
        "all_rejected_identity_union": [list(x) for x in sorted(all_rejected)],
        "all_rejected_identity_union_count": len(all_rejected),
        "rejected_identity_metadata": v.iloc[rejected_indices][["source_id", "row_id", "profile", "ip_protocol", "family"]].to_dict("records"),
        "rejected_events": sum(x["rejected_events"] for x in by_trajectory.values()),
        "rejected_events_with_both_panels_strictly_separated": int(rejected_events_with_panels_strict),
        "panel_failure_events": panel_failure_events,
        "worsened_positive_event_rows": len(event_rows),
        "worsened_event_rows_by_profile_protocol": count_rows(events_frame, ["profile", "ip_protocol"]),
        "max_rejected_checkpoint_inverse_count": int(events_frame.current_pooled_normal_ge_count.max()),
        "max_rejected_checkpoint_inverse_fraction": float(events_frame.current_rank_fraction.max()),
        "trajectories": by_trajectory, "seeds": seed_results,
        "baseline_outranking_normal_identities": normal_rows,
        "verified_R31_and_R30_V_frames_equal": True,
        "verified_training_audit_rows_against_all_96_original_histories": True,
        "inputs": dict(sorted(inputs.items())),
        "interpretation_limits": [
            "Baseline outranking-normal identities are not the identities responsible for new inversions at rejected checkpoints.",
            "Rejected checkpoints contain per-positive inversion counts but no complete V score vector; their scores, old-threshold margins, and crossing-normal identities cannot be reconstructed.",
            "The old C threshold is read as a descriptive baseline reference; no new C calibration or candidate decision is produced.",
            "All baseline V attacks above old C threshold does not establish candidate C/E recall or justify post-hoc guard removal.",
        ],
    }
    (output / "cache_guard_audit.json").write_text(json.dumps(output_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Dict columns in CSV are rendered as JSON to preserve all source partitions.
    for row in baseline_rows:
        for field in ["normal_ge_by_source", "normal_ge_by_profile_protocol"]:
            row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True)
    for name, rows in [
        ("cache_guard_positive_baseline.csv", baseline_rows),
        ("cache_guard_worsened_events.csv", event_rows),
        ("cache_guard_baseline_normals.csv", normal_rows),
        ("cache_guard_baseline_pairs.csv", pair_rows),
    ]:
        pd.DataFrame(rows).to_csv(output / name, index=False, encoding="utf-8-sig")
    lines = [
        "# R31 逐正例逆序保护：冻结缓存复核", "",
        f"逐事件回查原始 history 后，确认 96 个验证事件中 {output_data['rejected_events']} 个触发 V 逐正例逆序保护，涉及 {len(all_rejected)} 个不同攻击身份，全部来自 B/UDP。完整 V 为 21450 行：18645 正常、2805 攻击。", "",
        "原模型的 2805 条 V 攻击在三个 seed 都高于各自原登记 C 阈值。被保护的逆序集中在高于旧工作阈值的极端正常尾部与 B/UDP 攻击之间；它约束完整跨来源排序，不能等同于工作点漏检保护。", "",
        "V 的来源分母为 A：13991 正常、2715 TCP 攻击；B：4654 正常、90 UDP 攻击。A_TCP 面板含 13315 正常和 2715 攻击，B_UDP 面板含 477 正常和 90 攻击。", "",
        "| seed | 原 C 阈值 | V 攻击最小旧阈值间隔 | 原有逆序攻击 | 正常－攻击逆序对 | 参与的正常身份 | 原有最大逆序数 / 18645 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, result in seed_results.items():
        a, n = result["all_positive"], result["normal_reference_threshold_region"]
        lines.append(f"| {seed} | {result['calibration_reference_only']['threshold']:.9f} | {a['margin_to_old_C']['min']:.6f} | {a['positive_with_pooled_inverse']} | {a['pooled_inverse_pairs']} | {n['distinct_normal_outranking_any_positive']} | {a['max_inverse_count']} / 18645 ({a['max_inverse_fraction']:.6%}) |")
    lines += ["", "旧 C 只读取原登记的 5511 正常、rank 5457、严格 `>` 阈值；这里的间隔属于 log-odds 数值描述，没有重新校准。", "", "## 正常身份与来源", ""]
    for seed, result in seed_results.items():
        n = result["normal_reference_threshold_region"]
        unique = "; ".join(f"{x['profile']}/协议{x['ip_protocol']}：{x['n']} 条" for x in n["baseline_inverse_normal_by_profile_protocol"])
        pairs = "; ".join(f"{x['normal_profile']}/协议{x['normal_ip_protocol']}：{x['n']} 对" for x in n["baseline_inverse_pairs_by_profile_protocol"])
        lines.append(f"- seed {seed}：正常身份 {unique}；逆序对 {pairs}。")
    lines += ["", "协议 1 为 ICMP，6 为 TCP，17 为 UDP。原有逆序中没有 B/UDP 正常；B/UDP 面板严格分离。逆序来自 A 正常或 B 的其他协议，pooled 保护将这些跨域排序一起锁定。每一对身份、分数和来源见 `cache_guard_baseline_pairs.csv`；每个攻击完整来源分解见 `cache_guard_positive_baseline.csv`。", "", "## 被拒事件实际知道什么", "", "| 轨迹 | 拒绝事件 / 16 | 不同恶化攻击身份 |", "|---|---:|---:|"]
    for name, trajectory in by_trajectory.items():
        lines.append(f"| {name} | {trajectory['rejected_events']} | {trajectory['distinct_worsened_positive_rows']} |")
    lines += ["", f"89 次逆序拒绝中有 {rejected_events_with_panels_strict} 次 A_TCP 与 B_UDP 面板仍严格分离。由于恶化攻击全部属于 B/UDP，这 {rejected_events_with_panels_strict} 次新增逆序关系必在 B/UDP 面板之外；只能定位到这个集合，不能恢复具体正常身份。另 {len(panel_failure_events)} 次 B_UDP 面板也失去严格分离：", ""]
    for event in panel_failure_events:
        lines.append(f"- {event['trajectory']}，step {event['step']}：B_UDP pAUC={event['panels']['B_UDP']['pauc']:.12f}，A_TCP 仍严格分离。")
    lines += ["", f"被拒事件记录的最大逐正例逆序数为 {output_data['max_rejected_checkpoint_inverse_count']} / 18645（{output_data['max_rejected_checkpoint_inverse_fraction']:.6%}）。该比例是相对完整 V 正常分母的排序位置，不是 B-only C 的校准结果或 E 误报率。它说明硬门槛会拒绝很小的极端尾部排序增量，不能据此重选 checkpoint。", "", "原始 history 保存了每条攻击的逆序数，没有保存被拒 checkpoint 的完整 V 分数。可确认哪些攻击逆序增加以及增加多少，不能确认新越过它们的正常身份、攻击分数降了多少、候选旧阈值检出或重新校准后的 C/E 表现。本文列出的正常身份均属于原模型，不能升级为被拒 checkpoint 的因果解释。", "", "## 机制判断与下一版设计含义", "", "已证实的错位是：训练中 F 正例分数软锚，并没有直接控制 V 全来源逐身份逆序；选轮却以每个身份一条逆序也不得增加为硬条件。该条件额外限制了固定工作点 TP 保留未必需要限制的排序变化，也不能单独保证 C/E 检出。原有冲突主要位于 B/UDP 攻击与 A/ICMP 等正常的跨域极端尾部。正常子群目标改善不能自动满足逐身份保护；4 次 B_UDP 面板退化也证明不能把全部拒绝解释为无关的跨域排序。", "", "下一版首先要明确保护的业务端点，再为训练保护目标、验证保护和最终 C/E 判决建立对应关系。不能在看完 R31 后删掉失败条件重选，也不能把这里的旧阈值诊断当作新选轮规则。若保留逐身份排序保护，就需说明如何让训练作用于这些排序关系，并记录完整 F/V 检查点分数以复原越序来源；若改为工作点保护，必须在新注册中冻结开发校准角色与指标，而不是沿用已查看的 C/E 调规则。", "", "## 缓存一致性与复现", "", "R30 与 R31 的 V 数据帧逐列相同、身份顺序相同。R30 原训练缓存与 R31 CPU 参照有微小数值差异，不能写成逐位一致；三 seed 的逐正例逆序计数完全一致。原 R30 缓存也同样满足全部 2805 攻击高于旧 C 阈值。", ""]
    for seed, result in seed_results.items():
        comp = result["R30_cache_vs_R31_CPU_anchor"]
        lines.append(f"- seed {seed}：21450 行中 {comp['bitwise_equal_rows']} 行逐位相同，最大绝对差 {comp['max_abs_score_difference']:.12g}；原缓存最小旧阈值间隔 {comp['old_R30_min_positive_margin']:.6f}。")
    lines += ["", "使用 Python、NumPy、Pandas：", "", "```powershell", f'python cache_guard_audit.py --thesis-root "{root}"', "```", "", "输入文件 SHA-256、完整分母与全部事件核对结果见 `cache_guard_audit.json`。本脚本只读已有分数和元数据，模型加载、前向、训练、校准、阈值搜索均为 0。"]
    (output / "cache_guard_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rejected_events": output_data["rejected_events"], "distinct_rejected": len(all_rejected), "max_rejected_inverse": output_data["max_rejected_checkpoint_inverse_count"], "inputs_hashed": len(inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
