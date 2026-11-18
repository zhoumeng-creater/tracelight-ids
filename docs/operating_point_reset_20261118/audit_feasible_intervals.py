"""Audit R30 cached-score feasibility; never select a deployment threshold.

Dependencies: numpy, pandas, and a parquet engine (pyarrow).
Reads archived decisions only. No fitting, model loading, forward pass, score
generation, or candidate decision file is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (17, 29, 43)
ID = ["source_id", "row_id"]
EXPECTED_CELLS = {
    "TQH-B_i3600_j0": (1805, 24),
    "TQH-B_i3600_j30": (1839, 24),
    "TQH-B_i3600_j70": (1815, 24),
    "TQH-D_i1800_j0": (811, 36),
    "TQH-D_i1800_j30": (800, 35),
    "TQH-D_i1800_j70": (799, 27),
    "TQH-D_i300_j0": (550, 134),
    "TQH-D_i300_j30": (547, 115),
    "TQH-D_i300_j70": (564, 99),
    "TQH-D_i30_j0": (676, 554),
    "TQH-D_i30_j30": (687, 482),
    "TQH-D_i30_j70": (656, 411),
    "TQH-D_i3600_j0": (1151, 24),
    "TQH-D_i3600_j30": (1152, 24),
    "TQH-D_i3600_j70": (1171, 23),
}
EXPECTED_ORIGINAL = {
    17: {"fp": 154, "qualified_cells": 8},
    29: {"fp": 175, "qualified_cells": 5},
    43: {"fp": 202, "qualified_cells": 4},
}
DEFAULT_RETURN = Path(
    r"E:\OneDrive - Coventry University\中传\毕业论文\实验归档"
    r"\20261117_第三十轮模型训练重设计\回传"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(frame: pd.DataFrame) -> list[dict]:
    return [
        {"source_id": str(r.source_id), "row_id": int(r.row_id), "score": float(r.score)}
        for r in frame[ID + ["score"]].itertuples(index=False)
    ]


def audit(path: Path, seed: int) -> tuple[dict, pd.DataFrame]:
    d = pd.read_parquet(path).sort_values(ID).reset_index(drop=True)
    assert set(ID + ["label", "score", "threshold", "alert"]).issubset(d.columns)
    assert not d[ID].isna().any().any() and not d.duplicated(ID).any()
    assert set(d.source_id) == set(EXPECTED_CELLS)
    assert len(d) == 17059 and set(d.label.unique()) == {0, 1}
    assert d.score.dtype == np.dtype("float64")
    assert d.threshold.dtype == np.dtype("float64")
    assert pd.api.types.is_bool_dtype(d.alert)
    assert np.isfinite(d.score).all() and np.isfinite(d.threshold).all()
    assert d.threshold.nunique() == 1
    original_t = float(d.threshold.iloc[0])
    assert np.array_equal(d.alert.to_numpy(), d.score.to_numpy() > original_t)
    if "ready" in d:
        assert d.ready.all()

    pos = d.loc[d.label.eq(1)]
    assert len(pos) == 2036 and pos.alert.all()
    assert int(d.label.eq(0).sum()) == 15023
    for prefix, normal_n, attack_n in (("TQH-B_", 5459, 72), ("TQH-D_", 9564, 1964)):
        role = d.loc[d.source_id.str.startswith(prefix)]
        assert int(role.label.eq(0).sum()) == normal_n
        assert int(role.label.eq(1).sum()) == attack_n

    upper = float(pos.score.min())
    below_upper = float(np.nextafter(upper, -np.inf))
    cells = []
    for sid, group in d.groupby("source_id", sort=True):
        normal = np.sort(group.loc[group.label.eq(0), "score"].to_numpy())
        positives = group.loc[group.label.eq(1)]
        normal_n, attack_n = len(normal), len(positives)
        assert (normal_n, attack_n) == EXPECTED_CELLS[sid]
        budget = normal_n // 100
        lower_index = normal_n - budget - 1
        lower = float(normal[lower_index])
        lower_prev = float(np.nextafter(lower, -np.inf))
        fp_at_lower = int((normal > lower).sum())
        fp_below_lower = int((normal > lower_prev).sum())
        # Strict > makes L inclusive even when the lower-bound score is tied.
        assert fp_at_lower <= budget < fp_below_lower
        fp_at_upper = int((normal > upper).sum())
        fp_below_upper = int((normal > below_upper).sum())
        assert fp_below_upper == int((normal >= upper).sum())
        assert (lower < upper) == (fp_below_upper <= budget)
        original_fp = int((group.label.eq(0) & group.alert).sum())
        cells.append({
            "source_id": sid,
            "normal_n": normal_n,
            "attack_n": attack_n,
            "fp_budget": budget,
            "original_fp": original_fp,
            "original_tp": int((group.label.eq(1) & group.alert).sum()),
            "original_qualified": original_fp <= budget,
            "L_j": lower,
            "L_j_order_index_zero_based": lower_index,
            "normal_ties_at_L_j": int((normal == lower).sum()),
            "fp_at_L_j": fp_at_lower,
            "fp_at_nextafter_L_j_negative_infinity": fp_below_lower,
            "fp_at_U": fp_at_upper,
            "fp_at_nextafter_U_negative_infinity": fp_below_upper,
            "normal_ties_at_U": int((normal == upper).sum()),
            "can_meet_budget_and_preserve_all_global_TP": lower < upper,
        })

    lower = max(c["L_j"] for c in cells)
    exists = lower < upper
    assert exists == all(c["can_meet_budget_and_preserve_all_global_TP"] for c in cells)
    minimum_losses = pos.loc[pos.score.le(lower)]
    # Any valid common threshold is >= L; exactly these positives are then lost
    # at the smallest valid threshold. This is an oracle bound, not a selection.
    assert exists == (len(minimum_losses) == 0)
    original_fp = sum(c["original_fp"] for c in cells)
    original_qualified = sum(c["original_qualified"] for c in cells)
    assert original_fp == EXPECTED_ORIGINAL[seed]["fp"]
    assert original_qualified == EXPECTED_ORIGINAL[seed]["qualified_cells"]
    assert original_qualified < 15
    result = {
        "model_id": f"protocol_blind_s{seed}",
        "seed": seed,
        "diagnostic_only": True,
        "input_path": str(path.resolve()),
        "input_sha256": sha256(path),
        "rows": len(d),
        "normal_n": 15023,
        "protected_original_tp_n": 2036,
        "original_C_threshold": original_t,
        "original_fp": original_fp,
        "original_qualified_cells": original_qualified,
        "original_acceptance_passed": False,
        "common_lower_L": lower,
        "all_original_TP_upper_U_exclusive": upper,
        "U_minus_L": upper - lower,
        "common_full_TP_interval_exists": exists,
        "interval_notation": "[L, U)" if exists else "empty",
        "binding_lower_sources": [c["source_id"] for c in cells if c["L_j"] == lower],
        "conflicting_sources": [c["source_id"] for c in cells if not c["can_meet_budget_and_preserve_all_global_TP"]],
        "upper_attaining_positive_identities": identities(pos.loc[pos.score.eq(upper)]),
        "minimum_TP_losses_for_all_15_budgets": len(minimum_losses),
        "unavoidable_loss_identities": identities(minimum_losses),
        "oracle_minimum_FP_preserving_all_original_TP": sum(c["fp_at_nextafter_U_negative_infinity"] for c in cells),
        "normal_ties_at_U": sum(c["normal_ties_at_U"] for c in cells),
        "cells": cells,
    }
    return result, d[ID + ["label"]]


def render(result: dict, script: Path) -> str:
    lines = [
        "# R30 固定评分的共同阈值可行性诊断",
        "",
        "日期：2026-11-18。`diagnostic_only=true`。读取三个 R30 blind 的原判决缓存，核对完整身份、标签、分数与原判决后，只计算固定排序的可行性界限。没有加载模型、前向、训练、新评分或输出候选判决。",
        "",
        "seed17、seed43 的固定评分存在同时保住全部 2036 个原 TP、满足全部 15 单元误报预算的共同阈值；seed29 不存在，至少必须损失 1 个原 TP。三份模型按原 C 阈值的验收仍然失败，本诊断不改判旧结果。",
        "",
        "这些界限使用了已开的 B/D 标签，不能用于选择部署阈值或种子。用户已同意新方案按给定 FPR 上限下的召回与成本评价，不默认把保住 R30 全部 TP 作为新的采用硬门。本报告保留旧全 TP 条件，是为了区分操作点限制与固定排序冲突。",
        "",
        "判决固定为 `score > t`。每单元正常数为 n、预算 b=floor(n/100)，升序正常分数的零基第 n-b-1 项是下界 L_j。共同下界 L=max(L_j)；全部原 TP 的最低分是严格上界 U。共同可行区间为 [L,U)，当且仅当 L<U 时非空。正常在 L_j 处并列也允许 t=L_j；正例在 U 处并列均要求 t<U。",
        "",
        "| seed | 原 C 阈值 | 共同下界 L | 全 TP 上界 U（不含） | U−L | 区间非空 | 满足 15 单元预算的最少 TP 损失 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for m in result["models"]:
        lines.append(f"| {m['seed']} | {m['original_C_threshold']:.17g} | {m['common_lower_L']:.17g} | {m['all_original_TP_upper_U_exclusive']:.17g} | {m['U_minus_L']:.17g} | {'是' if m['common_full_TP_interval_exists'] else '否'} | {m['minimum_TP_losses_for_all_15_budgets']} |")
    lines += [
        "",
        "seed29 的约束冲突来自 D_i300_j0 和 D_i3600_j0：当阈值从下方逼近 U、仍保留全部 TP 时，分别有 6/5、13/11 条误报/预算。若阈值满足全部 15 单元预算，至少损失 TQH-D_i30_j30 的 row1147。三个种子的 U 都由该正例取得。",
        "",
        "各模型的正常数为 15023、攻击数为 2036；其中 B 为 5459 正常和 72 攻击，D 为 9564 正常和 1964 攻击。原 C 阈值下三个模型分别有 154、175、202 FP，只有 8/15、5/15、4/15 单元合规。",
    ]
    for m in result["models"]:
        lines += [
            "",
            f"**seed{m['seed']}：逐单元下界**",
            "",
            f"共同下界由 {', '.join(m['binding_lower_sources'])} 取得。",
            "",
            "| 单元 | 正常/攻击 | FP 预算 | 原 FP | L_j | FP(L_j) | FP(nextafter(L_j,−∞)) | 保全全部 TP 时最低 FP |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in m["cells"]:
            lines.append(f"| {c['source_id']} | {c['normal_n']}/{c['attack_n']} | {c['fp_budget']} | {c['original_fp']} | {c['L_j']:.17g} | {c['fp_at_L_j']} | {c['fp_at_nextafter_L_j_negative_infinity']} | {c['fp_at_nextafter_U_negative_infinity']} |")
        lines += [
            "",
            f"输入：`{m['input_path']}`",
            "",
            f"SHA-256：`{m['input_sha256']}`",
        ]
    lines += [
        "",
        "所有原判决逐行满足严格 `>`；三份身份及标签完全相同，无重复或缺失。45 个单元均核对了原正常/攻击分母、L_j 的包含边界和其相邻浮点值的排除边界；另在 U 与 nextafter(U,−∞) 核对正常并列。此次三个 U 上均无正常并列。",
        "",
        "复算使用 Python、NumPy、Pandas 与 PyArrow：",
        "",
        "```powershell",
        f'python "{script.resolve()}" --return-dir "{result["return_dir"]}" --output-dir "{result["output_dir"]}"',
        "```",
        "",
        "完整精度、每单元边界、原 TP 冲突身份、输入哈希和核验项见同目录 feasible_intervals.json。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-dir", type=Path, default=DEFAULT_RETURN)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    models, baseline_ids = [], None
    for seed in SEEDS:
        path = args.return_dir / "reports" / f"protocol_blind_s{seed}-decisions.parquet"
        model, ids = audit(path, seed)
        if baseline_ids is None:
            baseline_ids = ids
        else:
            assert baseline_ids.equals(ids)
        models.append(model)
    result = {
        "schema_version": 1,
        "diagnostic_only": True,
        "permitted_use": "Opened-development fixed-score feasibility diagnosis; not deployment threshold or seed selection.",
        "data_use": "Existing R30 B/D scores and labels; results do not restore blind-test status or change original acceptance.",
        "new_evaluation_policy": "Report recall at the stated FPR constraint and cost; all-R30-TP preservation is not an automatic new adoption gate.",
        "return_dir": str(args.return_dir.resolve()),
        "output_dir": str(output),
        "script_sha256": sha256(Path(__file__)),
        "comparison": ">",
        "score_dtype": "float64",
        "expected_cells": {k: {"normal_n": n, "attack_n": p} for k, (n, p) in EXPECTED_CELLS.items()},
        "checks": {
            "unique_complete_identities": True,
            "identical_identity_labels_across_three_seeds": True,
            "original_decisions_equal_score_strictly_greater_than_threshold": True,
            "exact_15_cell_denominators": True,
            "all_2036_positives_originally_detected": True,
            "finite_float64_scores": True,
            "strict_boundary_and_nextafter_checks": True,
            "old_results_remain_failed": True,
        },
        "models": models,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "feasible_intervals.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "feasible_intervals.md").write_text(render(result, Path(__file__)), encoding="utf-8")
    print(json.dumps({"diagnostic_only": True, "models": [
        {"seed": m["seed"], "L": m["common_lower_L"], "U": m["all_original_TP_upper_U_exclusive"], "exists": m["common_full_TP_interval_exists"], "minimum_TP_losses": m["minimum_TP_losses_for_all_15_budgets"]}
        for m in models
    ]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
