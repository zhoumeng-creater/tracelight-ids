"""Two fixed calibration rules on existing R30 caches; no model calls or searches.

The rank calculation is the no-preserve case of the existing
ids.risk_transfer.audit.order_rule from risk-audit-evidence-v1. The old
branch is not imported: this report must work from a clean checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
IDENTITY = ["source_id", "row_id"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fixed_rank(n: int, alpha: float, delta: float | None) -> tuple[int, float | None]:
    if n < 1 or not 0 < alpha < 1 or (delta is not None and not 0 < delta < 1):
        raise ValueError("Invalid calibration size or fixed risk level")
    if delta is None:
        # Original alpha is exactly 1/100; use integer arithmetic for its rank.
        if alpha != 0.01:
            raise ValueError("This comparison freezes original alpha at 0.01")
        return (99 * (n + 1) + 99) // 100, None
    ranks = np.arange(1, n + 1)
    admissible = ranks[binom.sf(ranks - 1, n, 1 - alpha) <= delta]
    if not len(admissible):
        raise ValueError("No finite NP threshold at the registered sample size")
    rank = int(admissible[0])
    tail = float(binom.sf(rank - 1, n, 1 - alpha))
    assert rank == 1 or binom.sf(rank - 2, n, 1 - alpha) > delta
    return rank, tail


def calibrate(scores, method):
    scores = np.asarray(scores, dtype=np.float64)
    assert scores.ndim == 1 and np.isfinite(scores).all()
    assert len(scores) == method["expected_n"]
    rank, tail = fixed_rank(len(scores), method["alpha"], method.get("delta"))
    assert rank == method["expected_rank"] and rank <= len(scores)
    threshold = float(np.partition(scores, rank - 1)[rank - 1])
    return {"rank": rank, "n": len(scores), "threshold": threshold,
            "binomial_tail": tail, "calibration_fp": int((scores > threshold).sum()),
            "alpha": method["alpha"], "delta": method.get("delta"), "comparison": ">"}


def metrics(frame, alerts):
    y = frame.label.to_numpy(dtype=int)
    a = np.asarray(alerts, dtype=bool)
    normal_n = int((y == 0).sum())
    attack_n = int((y == 1).sum())
    fp = int(((y == 0) & a).sum())
    tp = int(((y == 1) & a).sum())
    return {"normal_n": normal_n, "attack_n": attack_n, "fp": fp, "tp": tp,
            "fn": attack_n - tp, "budget": normal_n // 100,
            "fpr": fp / normal_n if normal_n else None,
            "recall": tp / attack_n if attack_n else None,
            "within_budget": bool(normal_n and fp <= normal_n // 100),
            "fp_excess": max(0, fp - normal_n // 100)}


def run(return_dir: Path, output_dir: Path, spec_path: Path):
    spec = read_json(spec_path)
    assert spec["models"] == [f"protocol_blind_s{x}" for x in (17, 29, 43)]
    assert [m["id"] for m in spec["methods"]] == ["original_order_statistic", "fixed_np_reference"]
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = {"spec_sha256": sha(spec_path), "phase": "C_only_before_ED_read", "models": {}}

    # All three calibrations finish and are written before any evaluation scores
    # or labels are opened. No feasible-interval report is read by this program.
    for model_id in spec["models"]:
        c_path = return_dir / "evidence" / model_id / "C-scores.parquet"
        c = pd.read_parquet(c_path).sort_values(IDENTITY).reset_index(drop=True)
        assert not c.duplicated(IDENTITY).any()
        old_meta_path = c_path.with_name("calibration.json")
        old_meta = read_json(old_meta_path)
        rules = {m["id"]: calibrate(c.score, m) for m in spec["methods"]}
        assert rules["original_order_statistic"]["threshold"] == old_meta["threshold"]
        assert rules["original_order_statistic"]["calibration_fp"] == old_meta["calibration_fp"]
        assert rules["fixed_np_reference"]["threshold"] >= rules["original_order_statistic"]["threshold"]
        thresholds["models"][model_id] = {
            "C_sha256": sha(c_path), "old_calibration_sha256": sha(old_meta_path), "rules": rules}
    thresholds_path = output_dir / "calibration_only.json"
    write_json(thresholds_path, thresholds)

    summaries, cells, changes, families, cost_context = [], [], [], [], []
    input_files = {}
    common_identities = None
    for model_id in spec["models"]:
        p = return_dir / "reports" / f"{model_id}-decisions.parquet"
        ed_path = return_dir / "evidence" / model_id / "ED-scores.parquet"
        d = pd.read_parquet(p).sort_values(IDENTITY).reset_index(drop=True)
        ed = pd.read_parquet(ed_path).sort_values(IDENTITY).reset_index(drop=True)
        assert len(d) == 17059 and d.source_id.nunique() == 15
        assert not d.duplicated(IDENTITY).any() and np.isfinite(d.score).all()
        assert set(d.label.unique()) == {0, 1} and int(d.label.sum()) == 2036
        assert d[IDENTITY].equals(ed[IDENTITY]) and np.array_equal(d.score, ed.score)
        assert d.family.eq("Benign").equals(d.label.eq(0))
        assert (d.alert == (d.score > d.threshold)).all()
        ids = d[IDENTITY + ["label"]]
        if common_identities is None:
            common_identities = ids.copy()
        else:
            assert common_identities.equals(ids)
        original = d.alert.to_numpy(dtype=bool)
        assert int((original & d.label.eq(1).to_numpy()).sum()) == 2036
        input_files[model_id] = {"decisions_sha256": sha(p), "ED_scores_sha256": sha(ed_path)}
        cost_path = return_dir / "cost" / f"{model_id}.json"
        cost = read_json(cost_path)
        cost_context.append({"model_id": model_id, "original_cost_sha256": sha(cost_path),
                             "scope": cost["timing"]["scope"],
                             "model_artifact_bytes": cost["model_artifact_bytes"],
                             "single_cpu_p50_ms": cost["timing"]["single"]["p50_ms"],
                             "single_cpu_p95_ms": cost["timing"]["single"]["p95_ms"],
                             "status": "original_R30_cost_reused_no_new_measurement"})
        rules = thresholds["models"][model_id]["rules"]
        for method_id, rule in rules.items():
            alerts = d.score.to_numpy() > rule["threshold"]
            if method_id == "original_order_statistic":
                assert np.array_equal(alerts, original)
            # The fixed NP threshold is nested: no new positive decisions.
            assert not (alerts & ~original).any()
            per_cell = []
            for source, g in d.groupby("source_id", sort=True):
                row = {"model_id": model_id, "method": method_id, "source_id": source,
                       **metrics(g, alerts[g.index])}
                per_cell.append(row)
                cells.append(row)
            base = metrics(d, alerts)
            # Only per-unit compliance is an acceptance endpoint. Pooled fields
            # remain descriptive and are named explicitly to avoid confusion.
            for key in ("budget", "within_budget", "fp_excess"):
                base["pooled_" + key] = base.pop(key)
            lost_tp = (~alerts & original & d.label.eq(1).to_numpy())
            removed_fp = (~alerts & original & d.label.eq(0).to_numpy())
            summaries.append({"model_id": model_id, "method": method_id, **base,
                              "threshold": rule["threshold"], "calibration_fp": rule["calibration_fp"],
                              "units_within_budget": sum(x["within_budget"] for x in per_cell),
                              "all_15_units_within_budget": all(x["within_budget"] for x in per_cell),
                              "sum_unit_fp_excess": sum(x["fp_excess"] for x in per_cell),
                              "max_unit_fpr": max(x["fpr"] for x in per_cell),
                              "old_tp_lost": int(lost_tp.sum()), "tp_added": 0,
                              "fp_removed": int(removed_fp.sum()), "fp_added": 0})
            for family, g in d[d.label.eq(1)].groupby("family", sort=True):
                families.append({"model_id": model_id, "method": method_id, "family": family,
                                 "attack_n": len(g), "tp": int(alerts[g.index].sum()),
                                 "old_tp_lost": int(lost_tp[g.index].sum())})
            for idx in np.flatnonzero(lost_tp | removed_fp):
                changes.append({"model_id": model_id, "method": method_id,
                                "source_id": d.at[idx, "source_id"], "row_id": int(d.at[idx, "row_id"]),
                                "family": d.at[idx, "family"], "score": float(d.at[idx, "score"]),
                                "change": "old_tp_lost" if lost_tp[idx] else "old_fp_removed"})

    result = {"status": "development_comparison_complete", "spec_sha256": sha(spec_path),
              "calibration_only_sha256": sha(thresholds_path), "input_files": input_files,
              "model_loads": 0, "forward_passes": 0, "optimizer_updates": 0,
              "new_calibration_rules": 1, "parameter_grid_searches": 0,
              "independent_confirmation": False, "deployment_adoption": False,
              "summaries": summaries, "families": families, "original_cost_context": cost_context}
    write_json(output_dir / "comparison.json", result)
    pd.DataFrame(cells).to_csv(output_dir / "comparison_cells.csv", index=False, encoding="utf-8")
    pd.DataFrame(changes, columns=["model_id", "method", "source_id", "row_id", "family", "score", "change"]).to_csv(
        output_dir / "comparison_changes.csv", index=False, encoding="utf-8")
    lines = ["# R30 固定操作点开发比较", "", "原模型与C/ED身份保持不变，只新增一个固定NP参照。"
             "模型加载、前向、拟合均为0；原R30/R31验收不改。", "",
             "| seed | 规则 | C误报 | ED检出/2036 | ED误报 | 合规单元/15 | 丢失旧TP | 移除旧FP |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        label = "原顺位" if s["method"] == "original_order_statistic" else "固定NP参照"
        lines.append(f"| {s['model_id'].split('_s')[-1]} | {label} | {s['calibration_fp']} | "
                     f"{s['tp']} | {s['fp']} | {s['units_within_budget']} | {s['old_tp_lost']} | {s['fp_removed']} |")
    lines += ["", "逐单元计数见 `comparison_cells.csv`，全部新增／丢失身份见 `comparison_changes.csv`。"
              "NP阈值提高，故没有新增TP或FP；不能将移除FP与丢失TP合并成一个净收益。", "",
              "| seed | 原工件字节数 | 原CPU单条p50/ms | 原CPU单条p95/ms |",
              "|---|---:|---:|---:|"]
    for c in cost_context:
        lines.append(f"| {c['model_id'].split('_s')[-1]} | {c['model_artifact_bytes']} | "
                     f"{c['single_cpu_p50_ms']:.6f} | {c['single_cpu_p95_ms']:.6f} |")
    lines += ["",
              "本批是已开数据上的事后开发比较。参数固定于新参照计算前，但此前已观察原ED及可行区间。"
              "C被多轮研究利用且存在来源变化，二项尾概率不构成本项目的部署风险保证。", "",
              "模型、输入和前向路径未改，原模型成本仅可复用为上下文；本次缓存分析耗时不是模型推理成本。"
              "没有新轻量化收益，也没有原创算法或独立确认结果。", "",
              "本次比较在此结束：不扫alpha/delta，不挑种子，不自动追加训练或另一条校准规则。"]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_dir), "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs/operating_point_reset_20261118")
    parser.add_argument("--spec", type=Path, default=REPO / "configs/operating-point-reset-20261118.json")
    args = parser.parse_args()
    run(args.return_dir, args.output_dir, args.spec)
