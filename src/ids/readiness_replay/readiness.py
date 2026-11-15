"""Repair R25 readiness from frozen observations and scores, without truth or inference."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pandas as pd

from ids.behavior_transfer import policy
from ids.behavior_transfer.decisions import COMPONENTS, COUNT_COLUMNS, HOUR_MS

ALPHA = Fraction(1, 100)
SCORE_COLUMNS = ("score_raw", "score_contact", "score_quality")
META = ("row_id", "flow_end_ms", "phase")
FIXED_POLICIES = (
    "raw_old", "contact_old", "quality_old", "add_quality_old",
    "ports_old", "targets_old", "counts_old", "quality_counts_old", "no_reply_scan_proxy",
)
LOCAL_POLICIES = tuple(name + "_local" for name in COMPONENTS)
UNION_POLICIES = ("frozen_plus_ports_local", "frozen_plus_counts_local")
POLICIES = (*FIXED_POLICIES, *LOCAL_POLICIES, *UNION_POLICIES)


def _minimum_rows(components):
    inverse = 1 / (ALPHA / len(components))
    return (inverse.numerator + inverse.denominator - 1) // inverse.denominator - 1


MINIMUM_ROWS_BY_POLICY = {name: _minimum_rows(parts) for name, parts in COMPONENTS.items()}
MIN_REFERENCE_ROWS = max(MINIMUM_ROWS_BY_POLICY.values())


def _integer_column(frame, name):
    values = frame[name].to_numpy()
    if not np.issubdtype(values.dtype, np.integer) or (
        len(values) and (values.min() < 0 or values.max() > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must be nonnegative integer values representable as int64")
    return values.astype(np.int64, copy=False)


def _validate(features, scores, old_decisions, old_registration):
    n = len(features)
    expected = np.arange(1, n + 1, dtype=np.int64)
    if not n:
        raise ValueError("Cannot replay an empty source")
    for frame, columns in (
        (features, ("row_id", "flow_end_ms", *COUNT_COLUMNS)),
        (scores, ("row_id", *SCORE_COLUMNS)),
        (old_decisions, (*META, *POLICIES)),
    ):
        if not frame.columns.is_unique or not set(columns).issubset(frame.columns):
            raise ValueError("Missing or duplicated replay input columns")
        if not np.array_equal(_integer_column(frame, "row_id"), expected):
            raise ValueError("Every input must have the same row_id order 1..n")
    if set(old_decisions.columns) != set((*META, *POLICIES)):
        raise ValueError("Expected exactly the 18 R25 policies and three identity/phase columns")
    if any(old_decisions[name].dtype != bool for name in POLICIES):
        raise ValueError("Sealed R25 policies must be Boolean")
    clock = _integer_column(features, "flow_end_ms")
    if (clock[1:] < clock[:-1]).any():
        raise ValueError("Replay requires a nondecreasing integer-millisecond clock")
    for frame in (old_decisions, scores):
        if "flow_end_ms" in frame and not np.array_equal(_integer_column(frame, "flow_end_ms"), clock):
            raise ValueError("Feature, score and decision clocks differ")
    for name in SCORE_COLUMNS:
        policy.scores(scores[name])
    for name in COUNT_COLUMNS:
        policy.scores(features[name])
    hour = int(clock[0]) + HOUR_MS
    if hour > np.iinfo(np.int64).max:
        raise ValueError("First-hour cutoff is outside int64 milliseconds")
    if not np.array_equal(old_decisions.phase.to_numpy(), np.where(clock < hour, "startup", "after")):
        raise ValueError("Old phase does not match the R25 first-hour cutoff")
    if (old_registration["rows"] != n or old_registration["prefix_start_ms"] != int(clock[0])
            or old_registration["cutoff_ms"] != hour):
        raise ValueError("Old registration source/time identity differs")
    if old_registration["components"] != {k: list(v) for k, v in COMPONENTS.items()}:
        raise ValueError("The registered component allocation differs from R25")
    if set(old_registration["registration"]) != set(COMPONENTS):
        raise ValueError("Expected all seven registered component policies")
    for name, parts in COMPONENTS.items():
        rules = old_registration["registration"][name]["old"]
        if set(rules) != set(parts):
            raise ValueError("Fallback components differ from the registered policy")
        for rule in rules.values():
            value = rule["threshold"]
            if (rule["comparison"] != ">" or rule["comparison_dtype"] != "float64"
                    or rule["alpha"] != float(ALPHA / len(parts))
                    or (value is not None and not np.isfinite(value))):
                raise ValueError("Fallback threshold or equal component allocation changed")
    return clock, hour


def _cutoff(clock, hour):
    """Return the original hour or the first later group with sufficient strict predecessors."""
    hour_rows = int(np.searchsorted(clock, hour, side="left"))
    if clock[-1] < hour:
        return None, "hour_not_elapsed", hour_rows, 0
    if hour_rows >= MIN_REFERENCE_ROWS:
        return hour, "original_hour_ready", hour_rows, 0
    position = max(hour_rows, MIN_REFERENCE_ROWS)
    blocked_group_rows = 0
    if position < len(clock):
        candidate = clock[position]
        first = int(np.searchsorted(clock, candidate, side="left"))
        if first < MIN_REFERENCE_ROWS:
            last = int(np.searchsorted(clock, candidate, side="right"))
            blocked_group_rows = last - first
            position = last
        if position < len(clock):
            return int(clock[position]), "waited_for_samples", hour_rows, blocked_group_rows
    return None, "insufficient_strictly_prior_rows", hour_rows, blocked_group_rows


def repair_decisions(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    old_decisions: pd.DataFrame,
    old_registration: dict,
) -> tuple[pd.DataFrame, dict]:
    """Return repaired 18-policy decisions and a truth-free readiness registration.

    All seven local policies share one strictly preceding prefix. No model is loaded,
    no labels are inspected, and none of the four input objects is modified.
    """
    clock, hour = _validate(features, scores, old_decisions, old_registration)
    cutoff, reason, hour_rows, blocked_group_rows = _cutoff(clock, hour)
    ready = cutoff is not None
    prefix_rows = int(np.searchsorted(clock, cutoff, side="left")) if ready else 0
    before = clock < cutoff if ready else np.ones(len(clock), dtype=bool)
    blind = features.loc[:, list(COUNT_COLUMNS)].reset_index(drop=True).copy()
    for name in ("score_raw", "score_quality"):
        blind[name] = scores[name].to_numpy(dtype=np.float64)
    out = old_decisions.reset_index(drop=True).copy(deep=True)
    out["phase"] = np.where(before, "startup", "after")
    registrations = {}
    for name, parts in COMPONENTS.items():
        original = old_registration["registration"][name]
        old_rules = deepcopy(original["old"])
        fallback = policy.apply(blind, old_rules)
        local_rules = policy.register(blind.iloc[:prefix_rows], parts, ALPHA) if ready else deepcopy(old_rules)
        if ready:
            for rule in local_rules.values():
                rule["scope"] = "local_unlabeled_alert_rate"
                rule["n_reference_rows"] = rule["n_normal"]
                if rule["threshold"] is None:
                    raise ValueError("Ready prefix unexpectedly produced an insufficient-sample threshold")
        out[name + "_local"] = np.where(before, fallback, policy.apply(blind, local_rules))
        registrations[name] = {
            "old": old_rules,
            "local": {
                "rules": local_rules,
                "status": "local_unlabeled_prefix" if ready else "old_calibration_fallback",
                "local_rows": prefix_rows,
                "cutoff_ms": cutoff,
                "threshold_status": {
                    field: "finite" if rule["threshold"] is not None else "insufficient_reference_rows"
                    for field, rule in local_rules.items()
                },
            },
            "original_local_status": original["local"]["status"],
            "original_local_insufficient_components": [
                field for field, rule in original["local"]["rules"].items() if rule["threshold"] is None
            ],
        }
    out["frozen_plus_ports_local"] = out.add_quality_old | out.ports_local
    out["frozen_plus_counts_local"] = out.add_quality_old | out.counts_local
    first_after = prefix_rows if ready else None
    switch_clock = int(clock[first_after]) if ready else None
    prefix_summary = {}
    if ready:
        for field in blind:
            values = blind[field].to_numpy()[:prefix_rows]
            prefix_summary[field] = {
                "minimum": float(values.min()), "maximum": float(values.max()),
                "all_prefix_equal": bool(values.min() == values.max()),
            }
    report = {
        "format": "tracelight-readiness-replay-v1",
        "status": "ready" if ready else "not_ready", "ready": ready, "reason": reason,
        "rows": len(clock), "prefix_start_ms": int(clock[0]), "old_hour_cutoff_ms": hour,
        "cutoff_ms": cutoff, "first_after_clock_ms": switch_clock,
        "minimum_reference_rows": MIN_REFERENCE_ROWS,
        "minimum_reference_rows_by_policy": MINIMUM_ROWS_BY_POLICY.copy(),
        "maximum_components": max(map(len, COMPONENTS.values())), "alpha": float(ALPHA),
        "prefix_rows": prefix_rows, "first_hour_rows": hour_rows,
        "startup_rows": int(before.sum()), "after_rows": int((~before).sum()),
        "startup_before_first_hour_rows": hour_rows,
        "waiting_for_readiness_rows": int(before.sum()) - hour_rows,
        "hour_elapsed": bool(clock[-1] >= hour),
        "rows_strictly_before_last_clock": int(np.searchsorted(clock, clock[-1], side="left")),
        "blocked_candidate_clock_group_rows": blocked_group_rows,
        "first_after_clock_group_rows": int(np.searchsorted(clock, switch_clock, side="right")
                                              - np.searchsorted(clock, switch_clock, side="left")) if ready else 0,
        "prefix_score_summary": prefix_summary, "registration": registrations,
        "components": {k: list(v) for k, v in COMPONENTS.items()},
        "changed_alerts_vs_r25": {name: int((out[name].to_numpy() != old_decisions[name].to_numpy()).sum())
                                  for name in POLICIES},
        "scope": "Readiness repair on already opened observations; finite ranks are not a future false-positive guarantee.",
        "reference": "All observations strictly before the common cutoff; no truth, event labels or future scores used.",
        "not_ready_behavior": "Keep the registered older-C component rules for the entire block; all rows remain startup.",
        "ties": "Whole switch-clock group excluded. Finite all-equal prefix scores are valid ready thresholds.",
    }
    return out, report
