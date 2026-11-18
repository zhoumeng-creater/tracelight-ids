"""Exact arithmetic for a design counterexample; no data or model execution."""
from fractions import Fraction as F
import json
from pathlib import Path


def solve():
    alpha = F(1, 100)
    # C is 1/2 U(0,1) + 1/2 U(1,2), hence U(0,2).
    old_cutoff = 2 * (1 - alpha)
    old_g2_risk = 2 - old_cutoff
    # The single-sided target yields U(0,1), U(0,2).
    # For t in (1,2), its new C CDF is 1/2 + t/4.
    new_cutoff = 4 * ((1 - alpha) - F(1, 2))
    new_g2_risk = (2 - new_cutoff) / 2
    d_g2_weight = F(3, 4)
    old_d_risk = d_g2_weight * old_g2_risk
    new_d_risk = d_g2_weight * new_g2_risk
    assert old_cutoff == F(99, 50) and new_cutoff == F(49, 25)
    assert old_g2_risk == new_g2_risk == F(1, 50)
    assert old_d_risk == new_d_risk == F(3, 200) > alpha
    assert 3 > old_cutoff and 3 > new_cutoff
    values = dict(alpha=alpha, old_cutoff=old_cutoff, new_cutoff=new_cutoff,
                  old_g2_risk=old_g2_risk, new_g2_risk=new_g2_risk,
                  old_d_risk=old_d_risk, new_d_risk=new_d_risk)
    return {
        "kind": "analytic_counterexample_not_IDS_experiment",
        "claim_refuted": "Exact fitting of the single-sided pooled-quantile target suffices to repair arbitrary normal-mixture change",
        "fractions": {k: str(v) for k, v in values.items()},
        "decimals": {k: float(v) for k, v in values.items()},
        "positive_score": 3, "positive_recall_before_and_after": 1,
        "normal_target_realization_error": 0,
        "normal_scores_never_increase": True,
        "actual_IDS_model_loads": 0, "actual_IDS_forwards": 0,
        "actual_IDS_fits": 0, "actual_IDS_threshold_searches": 0,
    }


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(solve(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Exact counterexample verified: 1.5% risk before and after perfect target fit.")
