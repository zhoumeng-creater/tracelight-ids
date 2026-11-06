"""Frozen H/K mini-batch endpoint gates; no model calls or threshold search."""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np
import pandas as pd

from .risk_research import _fixed_metrics
from .transfer_diagnostics import binary_rank_metrics, training_threshold
from .unsw_kd_evaluation import IDENTITY, ROLES, _concentration, _paired

ARMS = ("H", "K")
MAX_FPR = .01
MINIMUM_NET_TP = 34
MINIMUM_INTERACTION_TP = 34
PAUC_TOLERANCE = 1e-12
FP_CAPS = {"reference": 39, "constraint": 78}


def _validate(frames, scores):
    if set(frames) != set(ROLES) or set(scores) != set(ARMS):
        raise ValueError("Mini-batch evaluation requires exactly R/C and H/K")
    for role in ROLES:
        frame = frames[role]
        if not isinstance(frame, pd.DataFrame) or not len(frame):
            raise ValueError(f"{role}: nonempty DataFrame required")
        if not set(IDENTITY).issubset(frame.columns) or frame.loc[:, IDENTITY].isna().any().any():
            raise ValueError(f"{role}: complete identity columns required")
        for column in ("sample_id", "feature_hash", "attack_cat"):
            if not frame[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"{role}: {column} must contain nonempty strings")
        if frame.sample_id.duplicated().any():
            raise ValueError(f"{role}: duplicate sample_id")
        labels = frame.label.to_numpy()
        if labels.dtype.kind not in "fiu" or set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{role}: both binary classes required")
    for column in ("sample_id", "feature_hash"):
        if set(frames["reference"][column]) & set(frames["constraint"][column]):
            raise ValueError(f"R/C must not share {column}")
    checked = {}
    for arm in ARMS:
        if set(scores[arm]) != set(ROLES):
            raise ValueError(f"{arm}: exactly R/C scores required")
        checked[arm] = {}
        for role in ROLES:
            values = np.asarray(scores[arm][role])
            if (values.ndim != 1 or len(values) != len(frames[role])
                    or values.dtype.kind not in "fiu"):
                raise ValueError(f"{arm}/{role}: aligned numeric score vector required")
            values = values.astype(np.float64, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(f"{arm}/{role}: scores must be finite")
            checked[arm][role] = values
    return checked


def evaluate_minibatch_endpoints(frames, scores, *, historical_full_delta=5):
    """Return quality/interaction gates and changed-event/family/group tables.

    Inputs are final H/K endpoint scores only, aligned with the frozen R/C
    identities. Each endpoint gets exactly one max(R-normal, C-normal)
    threshold using the existing conservative float64 order statistic.
    The caller binds the historical full-batch delta and verifies execution.
    No S/M result or resource gate is imported from the previous experiment.
    """
    if isinstance(historical_full_delta, bool) or not isinstance(historical_full_delta, Integral):
        raise ValueError("historical_full_delta must be an integer bound to the old experiment")
    historical_full_delta = int(historical_full_delta)
    scores = _validate(frames, scores)
    endpoints = {}
    for arm in ARMS:
        registered = {
            role: training_threshold(scores[arm][role], frames[role].label.to_numpy(), MAX_FPR)
            for role in ROLES
        }
        threshold = max(record["threshold"] for record in registered.values())
        if not math.isfinite(threshold):
            raise ValueError(f"{arm}: registered threshold is not finite")
        roles = {}
        for role in ROLES:
            metric = _fixed_metrics(frames[role], scores[arm][role], threshold)
            metric["decision_rule"] = "float64(score)>=registered_max_R_C_normal_threshold"
            roles[role] = {
                "metrics": metric,
                "rank": binary_rank_metrics(frames[role].label.to_numpy(), scores[arm][role], MAX_FPR),
            }
        endpoints[arm] = {
            "working_point": {"threshold": threshold, "normal_registration": registered,
                              "scope": "empirical R/C development pools", "population_guarantee": False},
            "roles": roles,
        }
    comparisons, tables, families = {}, {}, []
    for role in ROLES:
        pair, events, masks = _paired(
            frames[role], scores["H"][role], scores["K"][role],
            endpoints["H"]["working_point"]["threshold"], endpoints["K"]["working_point"]["threshold"],
        )
        baseline, candidate = (endpoints[arm]["roles"][role] for arm in ARMS)
        pair.update({
            "delta_tpr": candidate["metrics"]["tpr"] - baseline["metrics"]["tpr"],
            "delta_fpr": candidate["metrics"]["fpr"] - baseline["metrics"]["fpr"],
            "delta_pauc_standardized": (candidate["rank"]["partial_auc_standardized"]
                                        - baseline["rank"]["partial_auc_standardized"]),
        })
        comparisons[role] = pair
        tables[f"K-H_{role}"] = events
        families.extend({"role": role, "attack_cat": name, **value}
                        for name, value in pair["attack_families"].items())
        if role == "constraint":
            concentration = _concentration(frames[role], masks)
    tables["K-H_attack_families"] = pd.DataFrame(families)
    tables["K-H_constraint_feature_groups"] = pd.DataFrame(concentration["groups"])
    checks = {
        "C_gain_K_H": comparisons["constraint"]["net_true_positives"] >= MINIMUM_NET_TP,
        "C_pauc_K_H": comparisons["constraint"]["delta_pauc_standardized"] >= -PAUC_TOLERANCE,
        "R_TP_K_H": comparisons["reference"]["net_true_positives"] >= 0,
        "R_pauc_K_H": comparisons["reference"]["delta_pauc_standardized"] >= -PAUC_TOLERANCE,
    }
    for role in ROLES:
        for arm in ARMS:
            metric = endpoints[arm]["roles"][role]["metrics"]
            checks[f"FP_budget_{arm}_{role}"] = metric["false_positives"] <= min(
                FP_CAPS[role], math.floor(MAX_FPR * metric["n_normal"]))
        pair = comparisons[role]
        checks[f"FP_K_H_{role}"] = pair["net_false_positives"] <= 0
        checks[f"family_TP_K_H_{role}"] = all(
            value["net_true_positives"] >= 0 for value in pair["attack_families"].values())
    checks["concentration_K_H_C"] = concentration["passed"]
    metric_passed = bool(all(checks.values()))
    mini_delta = comparisons["constraint"]["net_true_positives"]
    interaction = mini_delta - historical_full_delta
    effect_passed = interaction >= MINIMUM_INTERACTION_TP
    interaction_passed = bool(metric_passed and effect_passed)
    if not metric_passed:
        interpretation = "mini-batch KD failed at least one frozen quality gate; stop this branch"
    elif not effect_passed:
        interpretation = "quality gates passed; the preregistered optimization interaction did not"
    else:
        interpretation = "quality and interaction gates passed; source and execution gates remain separate"
    report = {
        "schema_version": 1, "scope": "fixed H/K mini-batch endpoint R/C development evaluation",
        "parameters": {"minimum_net_tp": MINIMUM_NET_TP, "max_fpr": MAX_FPR,
                       "pauc_tolerance": PAUC_TOLERANCE, "max_FP": dict(FP_CAPS),
                       "minimum_interaction_tp": MINIMUM_INTERACTION_TP},
        "endpoints": endpoints, "comparisons": {"K-H": comparisons},
        "concentration": concentration, "checks": checks,
        "metric_acceptance_passed": metric_passed,
        "interaction": {"formula": "(Kmini-Hmini) C net TP minus (Kfull-Hfull) C net TP",
                        "mini_net_TP": mini_delta, "historical_full_net_TP": historical_full_delta,
                        "net_TP_interaction": interaction, "minimum_net_TP_interaction": MINIMUM_INTERACTION_TP,
                        "minimum_mini_net_TP_for_interaction": historical_full_delta + MINIMUM_INTERACTION_TP,
                        "effect_size_gate_passed": bool(effect_passed),
                        "interpretation": "whole optimization scheme interaction; not a significance test"},
        "interaction_effect_passed": bool(effect_passed),
        "interaction_acceptance_passed": interaction_passed,
        "interpretation": interpretation, "development_only": True, "independent_confirmation": False,
        "not_evaluated": ["source_bindings", "training_budget_and_fixed_endpoint", "deployment_structure",
                          "new_optimizer_S_M_controls", "formal_CPU_cost", "independent_generalization"],
    }
    return report, tables
