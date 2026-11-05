"""Fixed-endpoint R/C evaluation for the four-arm UNSW ordinary KD pilot.

Array/DataFrame interface only: no training, model selection, or file loading.
The caller separately verifies source bindings, training execution and costs.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .risk_research import _fixed_metrics
from .transfer_diagnostics import binary_rank_metrics, training_threshold

ARMS = ("H", "K", "S", "M")
ROLES = ("reference", "constraint")
IDENTITY = ("sample_id", "feature_hash", "label", "attack_cat")
EVENTS = ("recovered_fn", "new_fn", "repaired_fp", "new_fp")


def _validate(frames, scores):
    if set(frames) != set(ROLES) or set(scores) != set(ARMS):
        raise ValueError("Evaluation requires exactly R/C and H/K/S/M")
    checked = {}
    for role in ROLES:
        frame = frames[role]
        if not isinstance(frame, pd.DataFrame) or not len(frame):
            raise ValueError(f"{role}: nonempty DataFrame required")
        if not set(IDENTITY).issubset(frame.columns):
            raise ValueError(f"{role}: missing identity columns")
        if frame.loc[:, IDENTITY].isna().any().any():
            raise ValueError(f"{role}: identity contains missing values")
        for key in ("sample_id", "feature_hash", "attack_cat"):
            if not frame[key].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{role}: {key} must contain nonempty strings")
        if frame.sample_id.duplicated().any():
            raise ValueError(f"{role}: duplicate sample_id")
        labels = frame.label.to_numpy()
        if labels.dtype.kind not in "fiu" or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{role}: both binary classes required")
    for arm in ARMS:
        if set(scores[arm]) != set(ROLES):
            raise ValueError(f"{arm}: exactly R/C scores required")
        checked[arm] = {}
        for role in ROLES:
            array = np.asarray(scores[arm][role])
            if (array.ndim != 1 or len(array) != len(frames[role])
                    or array.dtype.kind not in "fiu"):
                raise ValueError(f"{arm}/{role}: aligned numeric score vector required")
            array = array.astype(np.float64, copy=False)
            if not np.isfinite(array).all():
                raise ValueError(f"{arm}/{role}: scores must be finite")
            checked[arm][role] = array
    return checked


def _event_masks(y, baseline, candidate):
    return {
        "recovered_fn": (y == 1) & ~baseline & candidate,
        "new_fn": (y == 1) & baseline & ~candidate,
        "repaired_fp": (y == 0) & baseline & ~candidate,
        "new_fp": (y == 0) & ~baseline & candidate,
    }


def _counts(events, mask):
    result = {name: int(np.count_nonzero(event & mask)) for name, event in events.items()}
    result["net_true_positives"] = result["recovered_fn"] - result["new_fn"]
    result["net_false_positives"] = result["new_fp"] - result["repaired_fp"]
    return result


def _paired(frame, baseline_scores, candidate_scores, baseline_tau, candidate_tau):
    y = frame.label.to_numpy()
    baseline = baseline_scores >= np.float64(baseline_tau)
    candidate = candidate_scores >= np.float64(candidate_tau)
    events = _event_masks(y, baseline, candidate)
    result = _counts(events, np.ones(len(frame), dtype=bool))
    families = {}
    for family in sorted(frame.loc[y == 1, "attack_cat"].unique()):
        mask = (y == 1) & (frame.attack_cat.to_numpy() == family)
        families[family] = {
            "support": int(mask.sum()),
            "baseline_true_positives": int(np.count_nonzero(mask & baseline)),
            "candidate_true_positives": int(np.count_nonzero(mask & candidate)),
            **_counts(events, mask),
        }
    result["attack_families"] = families
    result["normal_support"] = int(np.count_nonzero(y == 0))
    result["attack_support"] = int(np.count_nonzero(y == 1))
    table = frame.loc[:, IDENTITY].copy().reset_index(drop=True)
    table["baseline_score"] = baseline_scores
    table["candidate_score"] = candidate_scores
    table["baseline_threshold"] = np.float64(baseline_tau)
    table["candidate_threshold"] = np.float64(candidate_tau)
    table["baseline_alarm"] = baseline
    table["candidate_alarm"] = candidate
    table["event"] = ""
    for name, mask in events.items():
        table.loc[mask, "event"] = name
    return result, table.loc[table.event != ""].reset_index(drop=True), events


def _concentration(frame, events):
    # Fixed original decisions: removing a group never re-registers a threshold.
    table = frame.loc[:, ["feature_hash", "label"]].copy()
    for name in EVENTS:
        table[name] = events[name].astype(np.int64)
    table["attack_rows"] = (table.label == 1).astype(np.int64)
    table["normal_rows"] = (table.label == 0).astype(np.int64)
    groups = table.groupby("feature_hash", sort=True)[
        [*EVENTS, "attack_rows", "normal_rows"]
    ].sum()
    groups["net_true_positives"] = groups.recovered_fn - groups.new_fn
    groups["net_false_positives"] = groups.new_fp - groups.repaired_fp
    total = int(groups.net_true_positives.sum())
    largest = int(groups.net_true_positives.max())
    largest_hashes = groups.index[groups.net_true_positives == largest].tolist()
    remaining = total - largest
    return {
        "comparison": "K-H", "role": "constraint", "group_key": "feature_hash",
        "semantics": "exact input duplicate pattern; not an independent collection event",
        "thresholds_stay_fixed": True,
        "feature_patterns": len(groups),
        "net_true_positives": total,
        "largest_group_net_true_positives": largest,
        "largest_group_feature_hashes": largest_hashes,
        "remaining_net_true_positives": remaining,
        "passed": bool(largest > 0 and remaining > 0),
        "groups": groups.reset_index().to_dict("records"),
    }


def evaluate_endpoints(frames, scores, *, minimum_net_tp=34, max_fpr=.01,
                       pauc_tolerance=1e-12):
    """Return a JSON-compatible report and six changed-event DataFrames.

    scores[arm][role] must follow the supplied identity row order.
    H/K/S/M each register a single max(R-normal, C-normal) float64 threshold.
    Minimum effect is fixed at 34 for the formal pilot; the argument permits
    smaller synthetic fixtures. This routine reports negative findings normally.
    It does not assert budget, source integrity, deployment size, or CPU cost.
    """
    if (isinstance(minimum_net_tp, bool) or not isinstance(minimum_net_tp, Integral)
            or minimum_net_tp < 1):
        raise ValueError("minimum_net_tp must be a positive integer")
    if (isinstance(max_fpr, bool) or not isinstance(max_fpr, Real)
            or not math.isfinite(max_fpr) or not 0 < max_fpr < 1):
        raise ValueError("max_fpr must be between zero and one")
    if (isinstance(pauc_tolerance, bool) or not isinstance(pauc_tolerance, Real)
            or not math.isfinite(pauc_tolerance) or pauc_tolerance < 0):
        raise ValueError("pauc_tolerance must be finite and nonnegative")
    scores = _validate(frames, scores)
    endpoints, tables, comparisons = {}, {}, {}
    for arm in ARMS:
        registered = {
            role: training_threshold(scores[arm][role], frames[role].label.to_numpy(), max_fpr)
            for role in ROLES
        }
        tau = max(item["threshold"] for item in registered.values())
        roles = {}
        for role in ROLES:
            metric = _fixed_metrics(frames[role], scores[arm][role], tau)
            metric["decision_rule"] = "float64(score)>=registered_max_R_C_normal_threshold"
            roles[role] = {
                "metrics": metric,
                "rank": binary_rank_metrics(frames[role].label.to_numpy(), scores[arm][role], max_fpr),
            }
        endpoints[arm] = {
            "working_point": {"threshold": tau, "normal_registration": registered,
                              "scope": "empirical R/C development pools", "population_guarantee": False},
            "roles": roles,
        }
    concentration = None
    for baseline in ("H", "S", "M"):
        key = f"K-{baseline}"
        comparisons[key] = {}
        for role in ROLES:
            pair, events, masks = _paired(
                frames[role], scores[baseline][role], scores["K"][role],
                endpoints[baseline]["working_point"]["threshold"],
                endpoints["K"]["working_point"]["threshold"],
            )
            base = endpoints[baseline]["roles"][role]
            candidate = endpoints["K"]["roles"][role]
            pair.update({
                "delta_tpr": candidate["metrics"]["tpr"] - base["metrics"]["tpr"],
                "delta_fpr": candidate["metrics"]["fpr"] - base["metrics"]["fpr"],
                "delta_pauc_standardized": (
                    candidate["rank"]["partial_auc_standardized"]
                    - base["rank"]["partial_auc_standardized"]),
            })
            comparisons[key][role] = pair
            tables[f"{key}_{role}"] = events
            if baseline == "H" and role == "constraint":
                concentration = _concentration(frames[role], masks)
    checks = {}
    for baseline in ("H", "S", "M"):
        pair = comparisons[f"K-{baseline}"]["constraint"]
        checks[f"C_gain_K_{baseline}"] = pair["net_true_positives"] >= minimum_net_tp
        checks[f"C_pauc_K_{baseline}"] = pair["delta_pauc_standardized"] >= -pauc_tolerance
    reference = comparisons["K-H"]["reference"]
    checks["R_TP_K_H"] = reference["net_true_positives"] >= 0
    checks["R_pauc_K_H"] = reference["delta_pauc_standardized"] >= -pauc_tolerance
    for role in ROLES:
        for arm in ARMS:
            metric = endpoints[arm]["roles"][role]["metrics"]
            checks[f"FP_budget_{arm}_{role}"] = (
                metric["false_positives"] <= math.floor(max_fpr * metric["n_normal"]))
        pair = comparisons["K-H"][role]
        checks[f"FP_K_H_{role}"] = pair["net_false_positives"] <= 0
        checks[f"family_TP_K_H_{role}"] = all(
            family["net_true_positives"] >= 0 for family in pair["attack_families"].values())
    checks["concentration_K_H_C"] = concentration["passed"]
    passed = bool(all(checks.values()))
    if not (checks["C_gain_K_H"] and checks["C_pauc_K_H"]):
        interpretation = "KD gain over hard did not meet the frozen quality gate"
    elif not (checks["C_gain_K_M"] and checks["C_pauc_K_M"]):
        interpretation = "increment over mean smoothing did not meet the frozen quality gate"
    elif not (checks["C_gain_K_S"] and checks["C_pauc_K_S"]):
        interpretation = "increment over the fixed permutation did not meet the frozen quality gate"
    elif not passed:
        interpretation = "C gains passed but another metric or concentration gate failed"
    else:
        interpretation = "metric gates passed; execution, deployment and cost gates still required"
    report = {
        "schema_version": 1, "scope": "fixed endpoint R/C development evaluation",
        "parameters": {"minimum_net_tp": int(minimum_net_tp), "max_fpr": float(max_fpr),
                       "pauc_tolerance": float(pauc_tolerance)},
        "endpoints": endpoints, "comparisons": comparisons,
        "concentration": concentration, "checks": checks,
        "metric_acceptance_passed": passed, "interpretation": interpretation,
        "not_evaluated": ["source_bindings", "target_controls", "training_budget_and_endpoint",
                          "deployment_structure_and_bundle_size", "formal_CPU_cost"],
    }
    return report, tables
