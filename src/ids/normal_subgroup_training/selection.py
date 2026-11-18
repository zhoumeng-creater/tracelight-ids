"""Frozen-contract selection; no calibration, gradients, or performance early stop."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from sklearn.metrics import roc_auc_score


QUANTILES = (("0.90", Fraction(9, 10)), ("0.95", Fraction(19, 20)),
             ("0.99", Fraction(99, 100)))
VALIDATION_STEPS = tuple(139 * event for event in range(1, 17))


def _scores(values, length):
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError("Expected finite one-dimensional row-aligned scores")
    return result


def normal_tail_metrics(scores, group_ids, expected_groups):
    """Exact group tail mass; fractional boundary ties never split equal scores."""
    groups = np.asarray(group_ids, dtype=object)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("Normal group IDs must be a nonempty vector")
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("Normal group IDs must be nonempty strings")
    expected = tuple(expected_groups)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("Expected group manifest must be nonempty and unique")
    if set(groups) != set(expected):
        raise ValueError("All frozen groups require V support; unknown groups are forbidden")
    values = _scores(scores, len(groups))
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    # Above each distinct score: all tied rows receive the same boundary weight.
    above = len(values) - np.cumsum(counts)
    result = {"normal": len(values), "unique_scores": len(unique),
              "max_tie_block": int(counts.max()),
              "tied_pairs": sum(int(n) * (int(n) - 1) // 2 for n in counts),
              "groups": {}, "H_q": {}, "H": Fraction(0)}
    group_counts = {g: np.bincount(inverse[groups == g], minlength=len(unique)) for g in expected}
    for g in expected:
        result["groups"][g] = {"normal": int(np.sum(groups == g)), "T": {}, "h": {}}
    for name, q in QUANTILES:
        mass = len(values) * (1 - q)
        take = -(-mass.numerator // mass.denominator)
        boundary = int(np.searchsorted(np.cumsum(counts), len(values) - take + 1, side="left"))
        boundary_weight = (mass - int(above[boundary])) / int(counts[boundary])
        if int(above[boundary]) + int(counts[boundary]) * boundary_weight != mass:
            raise AssertionError("Fractional tail mass must equal its exact registered value")
        excesses = []
        for g in expected:
            group = result["groups"][g]
            tail = (int(group_counts[g][boundary + 1:].sum())
                    + int(group_counts[g][boundary]) * boundary_weight)
            proportion = tail / group["normal"]
            excess = max(Fraction(0), proportion - (1 - q))
            group["T"][name], group["h"][name] = proportion, excess
            excesses.append(excess)
        result["H_q"][name] = max(excesses)
    result["H"] = max(result["H_q"].values())
    return result


def _frame_contract(frame):
    required = {"source_id", "row_id", "label", "profile", "ip_protocol"}
    if not required.issubset(frame.columns) or not len(frame):
        raise ValueError("F/V frames require nonempty physical identity and label metadata")
    if frame[list(required)].isna().any().any():
        raise ValueError("F/V identity and label metadata cannot be missing")
    if not frame.label.isin([0, 1]).all() or not frame.profile.isin(["A", "B"]).all():
        raise ValueError("F/V require binary labels and known A/B profiles")
    row_ids = frame.row_id.to_numpy(dtype=np.float64)
    if not np.isfinite(row_ids).all() or np.any(row_ids != np.floor(row_ids)):
        raise ValueError("Physical row identities must be integers")
    identities = tuple((str(source), int(row)) for source, row in zip(frame.source_id, row_ids))
    if len(set(identities)) != len(identities):
        raise ValueError("Physical row identities must be unique")
    return {"identities": identities, "labels": tuple(map(int, frame.label)),
            "profiles": tuple(frame.profile), "protocols": tuple(map(int, frame.ip_protocol))}


def _ranking(scores, labels):
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Each validation panel requires both labels")
    return {"pauc": float(roc_auc_score(labels, scores, max_fpr=0.01)),
            "strictly_separated": bool(scores[labels == 1].min() > scores[labels == 0].max())}


def selection_metrics(f_scores, f_frame, v_scores, v_frame, v_normal_group_ids, expected_groups):
    """Pure metrics over already scored F/V rows, with complete identity binding."""
    v_normal_group_ids = tuple(v_normal_group_ids)
    expected_groups = tuple(expected_groups)
    f_contract, v_contract = _frame_contract(f_frame), _frame_contract(v_frame)
    if set(f_contract["identities"]) & set(v_contract["identities"]):
        raise ValueError("F and V physical identities overlap")
    fs, vs = _scores(f_scores, len(f_frame)), _scores(v_scores, len(v_frame))
    fy, vy = f_frame.label.to_numpy(), v_frame.label.to_numpy()
    if set(np.unique(fy)) != {0, 1}:
        raise ValueError("F requires both labels")
    f_normals = np.sort(fs[fy == 0])
    f_inversions = len(f_normals) - np.searchsorted(f_normals, fs[fy == 1], side="left")
    normals = np.sort(vs[vy == 0])
    # side='left' counts equal normal scores against the positive.
    inversions = len(normals) - np.searchsorted(normals, vs[vy == 1], side="left")
    panels = {}
    for name, profile, protocol in (("A_TCP", "A", 6), ("B_UDP", "B", 17)):
        mask = (v_frame.profile.to_numpy() == profile) & (v_frame.ip_protocol.to_numpy() == protocol)
        panels[name] = _ranking(vs[mask], vy[mask])
    tail = normal_tail_metrics(vs[vy == 0], v_normal_group_ids, expected_groups)
    normal_profiles = v_frame.profile.to_numpy()[vy == 0]
    normal_groups = np.asarray(v_normal_group_ids, dtype=object)
    if any(not group.startswith(profile + "_") for group, profile in zip(normal_groups, normal_profiles)):
        raise ValueError("Frozen normal group roots must match their A/B source profile")
    source_tails = {}
    for profile in ("A", "B"):
        mask = normal_profiles == profile
        source_groups = tuple(g for g in expected_groups if g.startswith(profile + "_"))
        source_tails[profile] = normal_tail_metrics(vs[vy == 0][mask], normal_groups[mask], source_groups)
    return {"F_contract": f_contract, "V_contract": v_contract,
            "F_positive_ids": tuple(i for i, y in zip(f_contract["identities"], fy) if y == 1),
            "F_positive_scores": tuple(map(float, fs[fy == 1])),
            "F_positive_inversions": tuple(map(int, f_inversions)),
            "V_positive_ids": tuple(i for i, y in zip(v_contract["identities"], vy) if y == 1),
            "V_positive_inversions": tuple(map(int, inversions)),
            "V_normal_group_ids": tuple(v_normal_group_ids),
            "panels": panels, "pooled": _ranking(vs, vy),
            "tail": tail, "source_tails": source_tails}


def checkpoint_guards(metrics, baseline):
    """V2 protects F ordering; raw F score decreases remain visible diagnostics."""
    for field_name in ("F_contract", "V_contract", "F_positive_ids", "V_positive_ids",
                       "V_normal_group_ids"):
        if metrics[field_name] != baseline[field_name]:
            raise ValueError(f"Frozen selection identity/metadata changed: {field_name}")
    f_lost = [identity for identity, score, old in zip(
        metrics["F_positive_ids"], metrics["F_positive_scores"], baseline["F_positive_scores"])
        if score < old]
    f_worse = [identity for identity, count, old in zip(
        metrics["F_positive_ids"], metrics["F_positive_inversions"], baseline["F_positive_inversions"])
        if count > old]
    v_worse = [identity for identity, count, old in zip(
        metrics["V_positive_ids"], metrics["V_positive_inversions"], baseline["V_positive_inversions"])
        if count > old]
    tail, old_tail = metrics["tail"], baseline["tail"]
    source_tail_checks = {
        profile: {q: metrics["source_tails"][profile]["H_q"][q]
                  <= baseline["source_tails"][profile]["H_q"][q] for q, _ in QUANTILES}
        for profile in ("A", "B")}
    checks = {"F_each_positive_inversions_not_increased": not f_worse,
              "V_each_positive_inversions_not_increased": not v_worse,
              "V_panel_strict_separation": all(p["strictly_separated"] for p in metrics["panels"].values()),
              "V_pooled_pauc_not_lower": metrics["pooled"]["pauc"] >= baseline["pooled"]["pauc"],
              "V_panel_pauc_not_lower": all(metrics["panels"][p]["pauc"] >= baseline["panels"][p]["pauc"]
                                            for p in ("A_TCP", "B_UDP")),
              "normal_max_tie_block_not_increased": tail["max_tie_block"] <= old_tail["max_tie_block"],
              "normal_tied_pairs_not_increased": tail["tied_pairs"] <= old_tail["tied_pairs"],
              "each_H_q_not_increased": all(tail["H_q"][q] <= old_tail["H_q"][q] for q, _ in QUANTILES),
              "each_source_H_q_not_increased": all(all(checks.values())
                                                   for checks in source_tail_checks.values())}
    return {"passed": all(checks.values()), "checks": checks,
            "F_lower_positive_ids": f_lost, "F_lower_positive_count": len(f_lost),
            "F_worse_positive_ids": f_worse, "V_worse_positive_ids": v_worse,
            "source_tail_checks": source_tail_checks}


@dataclass
class Selector:
    """Exactly sixteen validation events; step0 remains a non-success fallback."""

    baseline: dict
    best_step: int = field(default=0, init=False)
    events: int = field(default=0, init=False)
    best_H: Fraction = field(init=False)

    def __post_init__(self):
        self.baseline = deepcopy(self.baseline)
        self.best_H = self.baseline["tail"]["H"]
        if any(self.baseline["F_positive_inversions"]):
            raise ValueError("R30 step0 requires complete F strict separation")
        if not checkpoint_guards(self.baseline, self.baseline)["passed"]:
            raise ValueError("R30 step0 must pass its own registered protection contract")

    def update(self, metrics, step):
        if (self.events == len(VALIDATION_STEPS) or isinstance(step, bool)
                or step != VALIDATION_STEPS[self.events]):
            raise ValueError("Selection requires the next fixed step in 139..2224, exactly once")
        guards = checkpoint_guards(metrics, self.baseline)
        h = metrics["tail"]["H"]
        eligible = guards["passed"] and h < self.baseline["tail"]["H"]
        improved = eligible and h < self.best_H
        # Events arrive in increasing order, so an equal H retains the earlier step.
        if improved:
            self.best_H, self.best_step = h, int(step)
        self.events += 1
        complete = self.events == len(VALIDATION_STEPS)
        return {"improved": improved, "eligible": eligible, "guards": guards,
                "selected_step": self.best_step, "selected_H": self.best_H,
                "candidate_improved": self.best_step != 0, "validation_events": self.events,
                "stop": complete, "stop_reason": "fixed_budget_complete" if complete else None,
                "performance_early_stop": False}

    def finish(self):
        if self.events != len(VALIDATION_STEPS):
            raise ValueError("Cannot finalize selection before all sixteen validation events")
        return {"selected_step": self.best_step, "selected_H": self.best_H,
                "candidate_improved": self.best_step != 0, "fallback": self.best_step == 0,
                "validation_events": self.events, "optimizer_steps": VALIDATION_STEPS[-1]}


def metrics_to_json(value):
    """Preserve exact fractions in receipts; float is only a display convenience."""
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}
    if isinstance(value, dict):
        return {key: metrics_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metrics_to_json(item) for item in value]
    return value
