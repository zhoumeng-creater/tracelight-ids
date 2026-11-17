"""Pure probability design arithmetic; never reads scores or model artifacts.

Reads only normal denominators from the already published units.json.
Uses deterministic binomial-tail inversion and order-index calculations.
No calibration scores, thresholds, fits, forwards or simulations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(r"E:\OneDrive - Coventry University\中传\毕业论文")
UNITS = ROOT / "实验归档/20261117_第二十九轮模型训练/回传/reports/units.json"
OUT = Path(__file__).with_suffix(".json")
DELTA = 0.05
DELTA_CAL = 0.025
DELTA_EVAL = 0.025


def binomial_cdf(n: int, p: float, k: int) -> float:
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    term = math.exp(n * math.log1p(-p))
    total = term
    for j in range(1, k + 1):
        term *= (n - j + 1) / j * p / (1.0 - p)
        total += term
    return min(1.0, max(0.0, total))


def invert_cell_tail(n: int, budget: int, allowance: float) -> float:
    low, high = 0.0, 0.01
    for _ in range(90):
        mid = (low + high) / 2.0
        if 1.0 - binomial_cdf(n, mid, budget) <= allowance:
            low = mid
        else:
            high = mid
    return low


def tolerance_index(n: int, risk_cap: float, beta: float) -> dict:
    # If even the sample maximum cannot certify risk_cap, no finite
    # observed order statistic can provide this confidence.
    maximum_failure_bound = math.exp(n * math.log1p(-risk_cap))
    result = {
        "calibration_normal_n": n,
        "risk_cap": risk_cap,
        "failure_allowance": beta,
        "minimum_n_for_any_finite_order_statistic": math.ceil(
            math.log(beta) / math.log1p(-risk_cap)
        ),
        "failure_bound_at_sample_maximum": maximum_failure_bound,
        "finite_order_statistic_feasible": maximum_failure_bound <= beta,
    }
    if not result["finite_order_statistic_feasible"]:
        return result
    r = 0
    while binomial_cdf(n, risk_cap, r + 1) <= beta:
        r += 1
    result.update(
        calibration_above_order_index_max=r,
        ascending_one_based_index=n-r,
        achieved_failure_bound=binomial_cdf(n, risk_cap, r),
        next_less_conservative_failure_bound=binomial_cdf(n, risk_cap, r+1),
    )
    return result


def main() -> None:
    units = json.loads(UNITS.read_text(encoding="utf-8"))
    cells = sorted(
        [{"cell": x["source_id"], "normal_n": x["normal_n"],
          "budget": x["normal_n"] // 100}
         for x in units if x["model_id"] == "hgb"],
        key=lambda x: x["cell"],
    )
    assert len(cells) == 15
    assert sum(x["normal_n"] for x in cells) == 15023
    results = {}
    for model_count in (1, 3, 7):
        # Main proposed family is 3 predeclared primary seeds.
        # 1 and 7 are scope arithmetic, not additional experiments.
        beta = DELTA_CAL / model_count
        eta = DELTA_EVAL / (model_count * len(cells))
        maxima = [dict(x, max_true_risk_for_cell=invert_cell_tail(
            x["normal_n"], x["budget"], eta)) for x in cells]
        equal_cell_cap = min(x["max_true_risk_for_cell"] for x in maxima)
        # Main allocation uses the entire finite-cell sampling budget:
        # M * sum_c BinomialTail(n_c, p*, b_c) <= delta_eval.
        # This is still a union bound, not a product/independence claim.
        low, high = 0.0, 0.01
        for _ in range(90):
            mid = (low + high) / 2.0
            total = model_count * sum(
                1.0-binomial_cdf(x["normal_n"], mid, x["budget"])
                for x in cells
            )
            if total <= DELTA_EVAL:
                low = mid
            else:
                high = mid
        # Round down, so displayed/registered cap does not exceed the
        # inverted sufficient bound due to decimal rounding.
        cap = math.floor(low*1e12)/1e12
        tails = [1.0-binomial_cdf(x["normal_n"], cap, x["budget"])
                 for x in cells]
        results[str(model_count)] = {
            "guarantee_scope_models": model_count,
            "cells_per_model": len(cells),
            "calibration_failure_per_model": beta,
            "risk_allocation": "delta_cal/M per score; delta_eval divided across model-cell events by the fixed binomial tails at common p*",
            "equal_cell_allocation_reference": {
                "evaluation_failure_per_model_cell": eta,
                "cellwise_risk_caps": maxima,
                "uniform_risk_cap": equal_cell_cap,
            },
            "risk_cap_before_downward_decimal_rounding": low,
            "uniform_true_risk_cap": cap,
            "cell_allocations": [dict(x, failure_allowance_per_model=tail)
                                 for x, tail in zip(cells, tails)],
            "evaluation_union_bound_at_cap": model_count*sum(tails),
            "allocated_overall_bound": DELTA_CAL+DELTA_EVAL,
            "calibration_sample_feasibility": {
                str(n): tolerance_index(n, cap, beta) for n in (1569, 5511)
            },
            "same_Q_used_to_certify_up_to_200_checkpoints": {
                str(n): tolerance_index(n, cap, beta/200) for n in (1569, 5511)
            },
            "separate_calibration_for_all_15_normal_populations": {
                str(n): tolerance_index(n, cap, beta/15) for n in (1569, 5511)
            },
        }
    report = {
        "scope": "analytic probability and sample-size/index design only",
        "not_executed": ["model load", "training", "forward", "scoring",
                         "threshold computation", "threshold search", "simulation"],
        "joint_failure_budget_is_proposed_design_choice_not_user_requirement": DELTA,
        "delta_calibration": DELTA_CAL,
        "delta_finite_evaluation_sampling": DELTA_EVAL,
        "main_scope": "three predeclared primary seeds and all fifteen fixed-size cells",
        "independence_not_needed": ["between cells", "between seeds", "between calibration failure events"],
        "necessary_unverified_conditions": [
            "each score fixed independently of calibration values",
            "calibration normal rows iid from a stated reference population",
            "every target-cell normal tail risk bounded by that reference risk, or separately calibrated",
            "conditional on frozen model and threshold, normal indicators within each evaluated cell independent with risk at most cap",
            "future evaluated data not reused to adapt model or risk protocol",
        ],
        "old_R29_thresholds_and_acceptance_unchanged": True,
        "denominator_source_sha256": hashlib.sha256(UNITS.read_bytes()).hexdigest(),
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                   encoding="utf-8", newline="\n")
    for m, result in results.items():
        print("M", m, "p*", result["uniform_true_risk_cap"],
              "evaluation_union_bound", result["evaluation_union_bound_at_cap"])
        for n, x in result["calibration_sample_feasibility"].items():
            print("n", n, json.dumps(x, ensure_ascii=False))
        print("200-checkpoint nmin", result["same_Q_used_to_certify_up_to_200_checkpoints"]["1569"]["minimum_n_for_any_finite_order_statistic"])


if __name__ == "__main__":
    main()
