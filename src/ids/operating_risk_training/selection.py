"""Compare complete true evaluations; no surrogate or threshold search enters."""
from __future__ import annotations

from fractions import Fraction


def compare_checkpoint(candidate_f: dict, candidate_v: dict, *,
                       reference_f: dict, reference_v: dict) -> dict:
    """Require every F/V protected TP and strict worst-group V risk improvement.

    References are the currently selected complete evaluations. On equal risk,
    retain the earlier checkpoint. A caller must supply candidates in increasing
    training-step order; no training schedule or risk family is chosen here.
    """
    for candidate, reference in ((candidate_f, reference_f), (candidate_v, reference_v)):
        if (candidate.get("format") != "operating-risk-evaluation-v1"
                or reference.get("format") != "operating-risk-evaluation-v1"
                or candidate["manifest"] != reference["manifest"]):
            raise ValueError("Checkpoint comparison requires the same complete frozen role manifest")
        if not reference["tp_protection_feasible"]:
            raise ValueError("A selected reference must preserve all declared positive identities")
    role_ids = []
    for metrics in (candidate_f, candidate_v):
        manifest = metrics["manifest"]
        role_ids.append(set(manifest["reference_ids"] + manifest["normal_ids"]
                            + manifest["protected_positive_ids"]))
    if role_ids[0] & role_ids[1]:
        raise ValueError("Training and validation identities must be disjoint")
    old = max(group["risk"] for group in reference_v["normal"]["groups"].values())
    new = max(group["risk"] for group in candidate_v["normal"]["groups"].values())
    if not isinstance(old, Fraction) or not isinstance(new, Fraction):
        raise ValueError("Checkpoint comparison requires exact in-memory count ratios")
    lost_f = list(candidate_f["protected_positives"]["lost_ids"])
    lost_v = list(candidate_v["protected_positives"]["lost_ids"])
    feasible = (candidate_f["tp_protection_feasible"] and candidate_v["tp_protection_feasible"]
                and not lost_f and not lost_v)
    reasons = []
    if not feasible:
        reasons.append("protected_positive_identity_lost")
    if new >= old:
        reasons.append("no_strict_true_worst_group_risk_improvement")
    return {"replace_reference": bool(feasible and new < old),
            "tp_protection_feasible": bool(feasible), "lost_f_ids": lost_f, "lost_v_ids": lost_v,
            "reference_worst_group_risk": old, "candidate_worst_group_risk": new,
            "reasons": reasons}
