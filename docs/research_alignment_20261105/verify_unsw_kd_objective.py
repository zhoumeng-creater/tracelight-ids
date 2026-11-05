"""CPU synthetic algebra check for the preregistered four-arm objective. No fits."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from ids.upgrade_kd import raw_bernoulli_kl_rows


def main():
    y = torch.tensor([0, 1, 1, 0, 1, 0, 1], dtype=torch.float64)
    t = torch.tensor([-20, 20, -3, 2, .2, -.3, 8], dtype=torch.float64)
    q = torch.sigmoid(t)
    z = torch.tensor([-4, 2, .1, -1, 5, .7, -2], dtype=torch.float64,
                     requires_grad=True)
    w = torch.where(y == 0,
                    torch.full_like(y, .5 / int((y == 0).sum())),
                    torch.full_like(y, .5 / int((y == 1).sum())))
    lam = .3
    r = (y + lam * q) / (1 + lam)
    fused_rows = F.binary_cross_entropy_with_logits(z, r, reduction="none")
    hard_rows = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    kl_rows = raw_bernoulli_kl_rows(z, teacher_logits=t, temperature=1.)
    explicit = (w * (hard_rows + lam * kl_rows) / (1 + lam)).sum()
    fused = (w * fused_rows).sum()
    entropy = -(q * F.logsigmoid(t) + (1-q) * F.logsigmoid(-t))
    expected_constant = (w * lam / (1 + lam) * entropy).sum()
    g_explicit = torch.autograd.grad(explicit, z, retain_graph=True)[0]
    g_fused = torch.autograd.grad(fused, z, retain_graph=True)[0]
    g_analytic = w * (torch.sigmoid(z.detach()) - r)
    chunked = sum((w[i:i+3] * fused_rows[i:i+3]).sum()
                  for i in range(0, len(y), 3))
    g_chunked = torch.autograd.grad(chunked, z)[0]
    checks = {
        "normalized_KL_vs_fused_gradient_max_abs": float((g_explicit-g_fused).abs().max()),
        "analytic_gradient_max_abs": float((g_analytic-g_fused).abs().max()),
        "chunked_global_denominator_gradient_max_abs": float((g_chunked-g_fused).abs().max()),
        "teacher_entropy_constant_error": float((fused-explicit-expected_constant).abs()),
        "class0_weight_sum": float(w[y == 0].sum()),
        "class1_weight_sum": float(w[y == 1].sum()),
        "hard_equals_fused_with_q_equal_y": bool(torch.equal(y, (y+lam*y)/(1+lam))),
    }
    for key, value in checks.items():
        if key.endswith("max_abs") or key.endswith("error"):
            assert value < 1e-12, (key, value)
    assert checks["hard_equals_fused_with_q_equal_y"]
    assert abs(checks["class0_weight_sum"]-.5) < 1e-7
    assert abs(checks["class1_weight_sum"]-.5) < 1e-7
    result = {
        "scope": "CPU synthetic algebra only; no dataset, teacher inference, or training",
        "torch": torch.__version__, "dtype": "float64", "passed": True,
        "checks": checks,
        "budget": {"arms": 4, "steps_per_arm": 40, "F_rows": 85916,
                   "optimizer_updates": 160, "training_row_forwards": 4*40*85916,
                   "chunks_per_step": 42, "backward_calls": 4*40*42,
                   "R_C_forward_rows_step0_shared_plus_four_endpoints": 5*(12274+24547)},
    }
    target = Path(__file__).with_name("unsw_kd_objective_verification.json")
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
