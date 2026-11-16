"""Localize exact archived-input label conflicts; never load or execute a model.

Run: python conflict_localization.py
The six-field view removes only orig_port and resp_port. All other values use
exact numeric equality; no rounding, nearest-neighbor search, or clustering.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TRAINING = HERE / "recovered-training.parquet"
REFERENCE = HERE / "input-audit/evidence/input_conflicts.json"
R26_RUN = HERE.parents[1] / "实验归档/20261115_第二十六轮历史监督/回传"
RAW = ["ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts",
       "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
SIX = [name for name in RAW if name not in ("orig_port", "resp_port")]


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def jsonable(value):
    return json.loads(json.dumps(value, default=native, ensure_ascii=False))


def weighted_floor(negative, positive):
    total = negative + positive
    return (-negative * math.log(negative / total) - positive * math.log(positive / total)
            if negative > 0 and positive > 0 else 0.0)


def provenance(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def weights_for(frame, support):
    if support == "source":
        return np.ones(len(frame))
    keys = ["domain", "label"] if support == "mixed" else ["label"]
    weights = 1.0 / frame.groupby(keys)["label"].transform("size").to_numpy()
    return weights / weights.mean()


def audit_view(frame, columns, support, view):
    data = frame.copy()
    data["audit_weight"] = weights_for(data, support)
    total_weight = float(data.audit_weight.sum())
    by_input = data.groupby(columns, sort=True, dropna=False)
    records = []
    for key, rows in by_input:
        if rows.label.nunique() < 2:
            continue
        negative = float(rows.loc[rows.label.eq(0), "audit_weight"].sum())
        positive = float(rows.loc[rows.label.eq(1), "audit_weight"].sum())
        domains = sorted(rows.domain.unique().tolist())
        source_rows, target_rows = rows.loc[rows.domain.eq("source")], rows.loc[rows.domain.eq("target")]
        source_internal = source_rows.label.nunique() > 1
        target_internal = target_rows.label.nunique() > 1
        cross_domain = (any(a != b for a in source_rows.label.unique() for b in target_rows.label.unique()))
        families = rows.groupby(["domain", "family", "label"]).size()
        captures = rows.groupby(["source_id", "family", "label"]).size()
        capture_internal = any(group.label.nunique() > 1 for _, group in rows.groupby("source_id"))
        labels_by_capture = {capture: set(group.label) for capture, group in rows.groupby("source_id")}
        captures_cross_label = any(a != b for ca, ya in labels_by_capture.items()
                                  for cb, yb in labels_by_capture.items() if ca < cb for a in ya for b in yb)
        records.append({
            "support": support, "view": view, "input": dict(zip(columns, key)),
            "rows": len(rows), "negative_rows": int(rows.label.eq(0).sum()),
            "positive_rows": int(rows.label.eq(1).sum()), "domains": domains,
            "source_internal_conflict": source_internal, "target_internal_conflict": target_internal,
            "cross_source_target_conflict": cross_domain,
            "within_at_least_one_capture_conflict": capture_internal,
            "between_source_captures_opposite_labels": captures_cross_label,
            "negative_weight": negative, "positive_weight": positive,
            "unrestricted_bce_optimal_positive_probability": positive / (negative + positive),
            "bce_floor_contribution": weighted_floor(negative, positive) / total_weight,
            "weighted_error_floor_contribution": min(negative, positive) / total_weight,
            "raw8_unique_inputs_inside_group": len(rows[RAW].drop_duplicates()),
            "distinct_port_pairs": len(rows[["orig_port", "resp_port"]].drop_duplicates()),
            "family_counts": [{"domain": d, "family": fam, "label": int(label), "rows": int(n)}
                              for (d, fam, label), n in families.items()],
            "capture_family_counts": [{"source_id": cap, "family": fam, "label": int(label), "rows": int(n)}
                                      for (cap, fam, label), n in captures.items()],
        })
    conflict_mask = by_input.label.transform("nunique").gt(1)
    affected = data.loc[conflict_mask]
    family_counts = affected.groupby(["domain", "family", "label"]).size()
    control = data.loc[data.domain.eq("source") & data.family.eq("C&C")]
    affected_control = affected.loc[affected.domain.eq("source") & affected.family.eq("C&C")]
    summary = {
        "support": support, "view": view, "rows": len(data), "unique_inputs": by_input.ngroups,
        "conflicting_inputs": len(records), "rows_in_conflicting_inputs": len(affected),
        "weighted_bce_infimum": sum(row["bce_floor_contribution"] for row in records),
        "weighted_error_infimum": sum(row["weighted_error_floor_contribution"] for row in records),
        "conflict_groups_by_origin": {
            "source_internal": sum(row["source_internal_conflict"] for row in records),
            "target_internal": sum(row["target_internal_conflict"] for row in records),
            "cross_source_target": sum(row["cross_source_target_conflict"] for row in records),
            "within_at_least_one_capture": sum(row["within_at_least_one_capture_conflict"] for row in records),
            "between_source_captures_opposite_labels": sum(row["between_source_captures_opposite_labels"] for row in records),
        },
        "family_counts_in_conflicts": [{"domain": d, "family": fam, "label": int(label), "rows": int(n)}
                                      for (d, fam, label), n in family_counts.items()],
        "source_control_rows": len(control), "source_control_rows_in_conflicts": len(affected_control),
        "source_control_identities_in_conflicts": affected_control[["source_id", "row_id"]].to_dict("records"),
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r26-run", type=Path, default=R26_RUN,
                        help="R26 run directory containing evidence/replay/ (default: local archived run)")
    parser.add_argument("--training", type=Path, default=TRAINING,
                        help="Recovered training parquet (default: recovered-training.parquet beside script)")
    parser.add_argument("--input-conflicts", type=Path, default=REFERENCE,
                        help="Existing input-conflict audit JSON used for cross-checking")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Directory for localization JSON and CSV (default: script directory)")
    args = parser.parse_args()
    training_path, reference_path = args.training.resolve(), args.input_conflicts.resolve()
    replay_dir, output_dir = args.r26_run.resolve() / "evidence/replay", args.output_dir.resolve()
    frame = pd.read_parquet(training_path)
    assert len(frame) == 18874
    assert not frame[["source_id", "row_id"]].duplicated().any()
    assert np.isfinite(frame[RAW].to_numpy(dtype=np.float64)).all()
    assert set(frame.domain) == {"source", "target"}
    np.testing.assert_allclose(frame.weight.to_numpy(), weights_for(frame, "mixed"), rtol=1e-14, atol=1e-14)
    summaries, records_by_view = [], {}
    for support in ("source", "target", "mixed"):
        selected = frame if support == "mixed" else frame.loc[frame.domain.eq(support)]
        for view, columns in (("raw8", RAW), ("ports_zero8", SIX)):
            summary, records = audit_view(selected, columns, support, view)
            summaries.append(summary)
            records_by_view[(support, view)] = records
    mixed_summary = next(row for row in summaries if (row["support"], row["view"]) == ("mixed", "ports_zero8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected = next(row for row in reference["results"] if row["arm"] == "mixed-ports_zero8-hgb"
                    and row["representation"] == "common_raw_ports_zero8")
    for key in ("rows", "unique_inputs", "conflicting_inputs", "rows_in_conflicting_inputs"):
        assert mixed_summary[key] == expected[key], (key, mixed_summary[key], expected[key])
    assert math.isclose(mixed_summary["weighted_bce_infimum"], expected["weighted_bce_infimum"], rel_tol=1e-12)
    assert math.isclose(mixed_summary["weighted_error_infimum"], expected["weighted_error_infimum"], rel_tol=1e-12)
    groups = sorted(records_by_view[("mixed", "ports_zero8")], key=lambda row: (-row["bce_floor_contribution"], tuple(row["input"].values())))
    key_to_group = {}
    for i, group in enumerate(groups, 1):
        group["group_id"] = f"C{i:03d}"
        group["fraction_of_exact_conflict_floor"] = group["bce_floor_contribution"] / mixed_summary["weighted_bce_infimum"]
        key_to_group[tuple(group["input"][name] for name in SIX)] = group["group_id"]
    evaluation = []
    replay_paths = sorted(replay_dir.glob("TQH-*.parquet"))
    assert len(replay_paths) == 15
    matching_identities = []
    for path in replay_paths:
        data = pd.read_parquet(path, columns=["source_id", "row_id", "family", *RAW])
        matches = [key_to_group.get(key) for key in data[SIX].itertuples(index=False, name=None)]
        data["conflict_group"] = matches
        matched = data.loc[data.conflict_group.notna()]
        evaluation.append({
            "source_id": path.stem, "rows": len(data),
            "normal_rows": int(data.family.eq("Benign").sum()),
            "attack_rows": int(data.family.eq("ExternalC2").sum()),
            "exact_match_rows": len(matched),
            "exact_match_normal_rows": int(matched.family.eq("Benign").sum()),
            "exact_match_attack_rows": int(matched.family.eq("ExternalC2").sum()),
            "matched_groups": sorted(matched.conflict_group.unique().tolist()),
        })
        matching_identities.extend(matched.to_dict("records"))
    profile_matches = []
    for profile in ("B", "D"):
        selected = [row for row in evaluation if row["source_id"].startswith("TQH-" + profile + "_")]
        profile_matches.append({"profile": profile, "cells": len(selected), **{
            key: sum(row[key] for row in selected) for key in ("rows", "normal_rows", "attack_rows", "exact_match_rows",
                                                            "exact_match_normal_rows", "exact_match_attack_rows")}})
    output = jsonable({
        "audit_date": "2026-11-16", "operation": "exact_input_group_counts_only",
        "model_loaded": False, "model_forward_executed": False, "training_executed": False,
        "matching_columns": SIX,
        "definitions": {
            "equality": "原始数值完全相同；只移除两端口，不做容差匹配、近邻或聚类。",
            "infimum": "当前有限训练样本和登记权重下，无约束确定性评分函数的经验BCE下界；不是网络最优值或泛化下界。",
            "origin_counts": "同组可以同时有采集内冲突与跨采集异标签；来源计数项不保证互斥。",
            "floor_fraction": "某精确输入组占全部精确冲突BCE下界的份额；不表示训练失败或外域漏检的因果解释比例。",
            "raw8": "完整保留两端口的八字段对照；原始输入无精确冲突不保证模型分箱或泛化后仍可分。",
            "evaluation_match": "只检查既有B/D回放输入是否精确落入训练冲突组；无命中不排除参数共享带来的间接影响。",
        },
        "inputs": [provenance(path) for path in [training_path, reference_path, *replay_paths]],
        "checks": {"reproduced_existing_mixed_exact_conflict_counts_and_floor": True,
                   "unique_training_identities": True, "finite_raw_inputs": True,
                   "reconstructed_mixed_weights_match": True},
        "view_and_support_summaries": summaries,
        "mixed_ports_zero8_groups_sorted_by_floor_contribution": groups,
        "exact_B_D_matches_by_profile": profile_matches,
        "exact_B_D_matches_by_cell": evaluation,
        "exact_B_D_matching_row_identities_and_inputs": matching_identities,
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "conflict_localization.json", output_dir / "conflict_localization.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows = []
    for group in groups:
        row = {"group_id": group["group_id"], **group["input"]}
        row.update({key: group[key] for key in ("rows", "negative_rows", "positive_rows", "negative_weight", "positive_weight",
                   "unrestricted_bce_optimal_positive_probability", "bce_floor_contribution", "fraction_of_exact_conflict_floor",
                   "weighted_error_floor_contribution", "raw8_unique_inputs_inside_group", "distinct_port_pairs",
                   "source_internal_conflict", "target_internal_conflict", "cross_source_target_conflict",
                   "within_at_least_one_capture_conflict", "between_source_captures_opposite_labels")})
        row["family_counts_json"] = json.dumps(group["family_counts"], ensure_ascii=False, default=native)
        row["capture_family_counts_json"] = json.dumps(group["capture_family_counts"], ensure_ascii=False, default=native)
        csv_rows.append(row)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    concise = [{key: row[key] for key in ("support", "view", "conflicting_inputs", "rows_in_conflicting_inputs",
                                         "weighted_bce_infimum", "source_control_rows_in_conflicts")} for row in summaries]
    print(json.dumps({"summaries": concise, "top_group": groups[0], "evaluation": profile_matches},
                     ensure_ascii=False, default=native, indent=2))


if __name__ == "__main__":
    main()
