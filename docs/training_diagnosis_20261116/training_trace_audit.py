"""Audit archived R26 traces and metrics without loading or executing any model.

Run: python training_trace_audit.py
Inputs are existing JSON and compressed CSV evidence. Outputs are placed next
to this script. Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
R26_RUN = HERE.parents[1] / "实验归档/20261115_第二十六轮历史监督/回传"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compressed_rows(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def slope(values):
    center = (len(values) - 1) / 2
    return sum((i - center) * value for i, value in enumerate(values)) / sum(
        (i - center) ** 2 for i in range(len(values))
    )


def summaries(rows, arm):
    result = {}
    for profile, cells in (("B", 3), ("D", 12)):
        selected = [row for row in rows if row["arm"] == arm and row["scope"] == "cell"
                    and row["source_id"].startswith("TQH-" + profile + "_")]
        assert len(selected) == cells, (arm, profile, len(selected))
        assert len({row["source_id"] for row in selected}) == cells
        sums = {key: sum(int(row[key]) for row in selected)
                for key in ("normal_n", "fp", "attack_n", "tp")}
        assert (sums["normal_n"], sums["attack_n"]) == ({"B": (5459, 72), "D": (9564, 1964)}[profile])
        assert all(int(row["Scanning_n"]) == int(row["Download_n"]) == 0 for row in selected)
        result[profile] = {
            "cells": cells,
            "qualified_cells": sum(row["qualified"].lower() == "true" for row in selected),
            **sums,
            "pooled_fpr_descriptive_only": sums["fp"] / sums["normal_n"],
            "pooled_recall_descriptive_only": sums["tp"] / sums["attack_n"],
        }
    return result


def input_record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r26-run", type=Path, default=R26_RUN,
                        help="R26 run directory containing evidence/ (default: local archived run)")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Directory for audit JSON and CSV (default: script directory)")
    args = parser.parse_args()
    evidence = args.r26_run.resolve() / "evidence"
    output_dir = args.output_dir.resolve()
    metrics_path, oracle_path = evidence / "metrics.csv.gz", evidence / "oracles.csv.gz"
    support_path = evidence / "training-support.json"
    metrics, oracles = compressed_rows(metrics_path), compressed_rows(oracle_path)
    supports = read_json(support_path)["supports"]
    inputs = [metrics_path, oracle_path, support_path]
    arms, csv_rows, fit_records = [], [], {}
    for path in sorted((evidence / "fits").glob("*.json")):
        fit = read_json(path)
        if fit.get("new_fits") != 1 or not fit.get("group", "").startswith("mlp"):
            continue
        inputs.append(path)
        arm = path.stem
        fit_records[arm] = fit
        losses = fit["training_loss_per_epoch"]
        assert len(losses) == fit["epochs"] == 80
        assert fit["training_batch_size"] == 512
        assert all(math.isfinite(value) and value >= 0 for value in losses)
        assert fit["rows"] == supports[fit["support"]]["rows"]
        means, scales = fit["preprocessing"]["mean"], fit["preprocessing"]["scale"]
        assert len(means) == len(scales) == len(fit["features"]) == 8
        assert all(math.isfinite(value) for value in means)
        assert all(math.isfinite(value) and value > 0 for value in scales)
        tail = losses[-20:]
        last20_slope = slope(tail)
        trace = {
            "epoch1": losses[0], "epoch40": losses[39], "epoch60": losses[59],
            "epoch61": losses[60], "epoch80": losses[79],
            "minimum": min(losses), "minimum_epoch": losses.index(min(losses)) + 1,
            "last20_ols_slope_per_epoch": last20_slope,
            "last20_ols_slope_divided_by_mean": last20_slope / statistics.mean(tail),
            "last20_relative_reduction_epoch61_to80": (tail[0] - tail[-1]) / tail[0],
            "last20_mean": statistics.mean(tail), "last20_population_sd": statistics.pstdev(tail),
            "last20_increase_steps": sum(b > a for a, b in zip(tail, tail[1:])),
            "epoch_loss_series": losses,
        }
        batches = math.ceil(fit["rows"] / fit["training_batch_size"])
        metrics_summary, oracle_summary = summaries(metrics, arm), summaries(oracles, arm)
        row = {
            "arm": arm, "support": fit["support"], "view": fit["view"], "seed": fit["seed"],
            "training_rows": fit["rows"], "positive_rows": fit["positive_rows"],
            "epochs": fit["epochs"], "batch_size": fit["training_batch_size"],
            "batches_per_epoch": batches, "optimizer_steps_total": batches * fit["epochs"],
            **{key: value for key, value in trace.items() if key != "epoch_loss_series"},
        }
        for profile in ("B", "D"):
            row.update({profile + "_" + key: value for key, value in metrics_summary[profile].items()})
            row[profile + "_oracle_tp_diagnostic_only"] = oracle_summary[profile]["tp"]
        csv_rows.append(row)
        probe_values = {"ip_protocol": {"tcp_6": 6, "udp_17": 17},
                        "resp_port": {"port_443": 443, "port_8443": 8443}}
        probes = {}
        for feature, values in probe_values.items():
            index = fit["features"].index(feature)
            probes[feature] = {name: ((math.log1p(raw) if not (
                fit["view"] == "ports_zero8" and feature.endswith("port")) else 0.0)
                - means[index]) / scales[index] for name, raw in values.items()}
        arms.append({**row, "trace": trace, "preprocessing": fit["preprocessing"],
                     "weighting": fit["weighting"], "fixed_value_standardized_probes": probes})

    assert len(arms) == 15
    assert {(row["support"], row["view"]) for row in arms} == {
        ("source", "ports_zero8"), ("target", "raw8"), ("target", "ports_zero8"),
        ("mixed", "raw8"), ("mixed", "ports_zero8")}
    by_support_view = []
    for support, view in sorted({(row["support"], row["view"]) for row in arms}):
        selected = [row for row in arms if (row["support"], row["view"]) == (support, view)]
        assert {row["seed"] for row in selected} == {17, 29, 43}
        by_support_view.append({
            "support": support, "view": view,
            "epoch80_mean_across_seeds": statistics.mean(row["epoch80"] for row in selected),
            "epoch80_min_across_seeds": min(row["epoch80"] for row in selected),
            "epoch80_max_across_seeds": max(row["epoch80"] for row in selected),
            "all_last20_ols_slopes_negative": all(row["last20_ols_slope_per_epoch"] < 0 for row in selected),
            "last20_relative_reductions": {str(row["seed"]): row["last20_relative_reduction_epoch61_to80"]
                                           for row in selected},
            "D_tp_by_seed": {str(row["seed"]): row["D_tp"] for row in selected},
            "D_qualified_cells_by_seed": {str(row["seed"]): row["D_qualified_cells"] for row in selected},
        })

    support_changes = {}
    for name, support in supports.items():
        n = support["rows"]
        cells = [{**cell, "mass_fraction": cell["mass"] / n, "per_row_weight": cell["mass"] / cell["rows"]}
                 for cell in support["domain_class_mass"]]
        target_positives = sum(cell["rows"] for cell in cells if cell["domain"] == "target" and cell["label"] == 1)
        support_changes[name] = {
            **support, "domain_class_mass": cells,
            "batches_per_epoch": math.ceil(n / 512), "steps_80_epochs": math.ceil(n / 512) * 80,
            "target_control_row_visits_80_epochs": target_positives * 80,
            "expected_target_control_rows_per_full_uniform_batch": 512 * target_positives / n,
        }

    preprocessing_changes = []
    for view in ("raw8", "ports_zero8"):
        target = fit_records["target-" + view + "-mlp29"]["preprocessing"]
        mixed = fit_records["mixed-" + view + "-mlp29"]["preprocessing"]
        features = fit_records["target-" + view + "-mlp29"]["features"]
        # Training identities and deterministic transforms make these parameters seed-independent.
        for support in ("target", "mixed"):
            assert all(fit_records[f"{support}-{view}-mlp{seed}"]["preprocessing"]
                       == fit_records[f"{support}-{view}-mlp29"]["preprocessing"] for seed in (17, 43))
        for i, feature in enumerate(features):
            preprocessing_changes.append({
                "view": view, "feature": feature, "target_mean": target["mean"][i],
                "mixed_mean": mixed["mean"][i], "target_scale": target["scale"][i],
                "mixed_scale": mixed["scale"][i],
                "mixed_to_target_scale_ratio": mixed["scale"][i] / target["scale"][i],
                "mean_shift_in_target_sd": (mixed["mean"][i] - target["mean"][i]) / target["scale"][i],
                "masked_slot": view == "ports_zero8" and feature in ("orig_port", "resp_port"),
            })

    output = {
        "audit_date": "2026-11-16", "operation": "archived_json_and_csv_analysis_only",
        "model_loaded": False, "model_forward_executed": False, "training_executed": False,
        "definitions": {
            "epoch_loss": "更新过程中各批次BCE按批次行数加权的平均；不是用轮末权重重新计算的全训练损失。",
            "last20": "第61至80轮；OLS趋势只描述已有轨迹，不证明继续训练能改善开发或外域效果。",
            "loss_comparison": "source、target、mixed使用不同数据和权重；损失绝对值不构成模型容量或迁移优劣的单独比较。",
            "evaluation": "仅聚合scope=cell，避免与UTC日记录重复计数；B/D都是已开开发材料。",
            "oracle": "使用评价期正常标签的事后排序诊断，不是可部署阈值或盲测成绩。",
            "weight_mass": "权重总量反映声明的训练目标份额；不是实测梯度大小。",
            "row_visits": "同一行多轮访问次数；不增加独立事件或训练样本数。",
        },
        "inputs": [input_record(path) for path in inputs],
        "checks": {"new_mlp_arms": len(arms), "epochs_each": 80, "seeds_each_support_view": [17, 29, 43],
                   "evaluation_cells_each": {"B": 3, "D": 12}, "trace_and_preprocessing_values_finite": True,
                   "preprocessing_seed_independent_within_support_view": True},
        "group_summaries": by_support_view, "support_and_step_changes": support_changes,
        "mixed_vs_target_preprocessing": preprocessing_changes, "arms": arms,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output_dir / "training_trace_audit.json", output_dir / "training_trace_audit.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"outputs": [str(json_path), str(csv_path)], "groups": by_support_view}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
