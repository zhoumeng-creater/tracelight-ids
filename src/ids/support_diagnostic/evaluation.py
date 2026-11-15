"""Per-cell low-FPR evaluation; evaluation labels never choose deployed rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


FOCUS_FAMILIES = ("Scanning", "ExternalC2", "Download")
BENIGN = "Benign"
UNRESOLVED = "Unresolved"


def _prepare(truth_frame: pd.DataFrame, score) -> tuple[np.ndarray, np.ndarray, str | None]:
    required = {"family", "row_id", "source_id"}
    missing = required.difference(truth_frame.columns)
    if missing:
        raise ValueError(f"truth is missing columns: {sorted(missing)}")
    if truth_frame[list(required)].isna().any().any():
        raise ValueError("truth identity and family columns must not contain nulls")
    sources = truth_frame["source_id"].unique()
    if len(sources) > 1:
        raise ValueError("evaluate one source/cell at a time; pooled FPR can hide failed cells")
    if truth_frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique within a source/cell")
    scores = np.asarray(score, dtype=np.float64)
    if scores.ndim != 1 or len(scores) != len(truth_frame):
        raise ValueError("scores must be a one-dimensional array aligned with truth rows")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    families = truth_frame["family"].astype(str).to_numpy()
    if np.any(families == ""):
        raise ValueError("family must not be empty")
    return families, scores, str(sources[0]) if len(sources) else None


def _decisions(scores: np.ndarray, rule: Mapping[str, Any]) -> tuple[np.ndarray, float | None, bool]:
    raw_threshold = rule.get("threshold")
    threshold = float(raw_threshold) if raw_threshold is not None else float("inf")
    finite = bool(np.isfinite(threshold))
    ready = bool(rule.get("ready", finite)) and finite
    alerts = scores > threshold if ready else np.zeros(len(scores), dtype=bool)
    return alerts, threshold if finite else None, ready


def _family_names(families: np.ndarray) -> list[str]:
    observed = set(families.tolist()).difference({BENIGN, UNRESOLVED})
    return sorted(observed.union(FOCUS_FAMILIES))


def _detection(mask: np.ndarray, alerts: np.ndarray) -> dict[str, Any]:
    n = int(np.count_nonzero(mask))
    tp = int(np.count_nonzero(mask & alerts))
    return {"n": n, "present": n > 0, "tp": tp, "fn": n - tp, "recall": tp / n if n else None}


def _summary(families, scores, source_id, rule) -> tuple[dict[str, Any], np.ndarray]:
    alerts, threshold, ready = _decisions(scores, rule)
    normal = families == BENIGN
    unresolved = families == UNRESOLVED
    attack = ~(normal | unresolved)
    normal_n = int(np.count_nonzero(normal))
    fp = int(np.count_nonzero(normal & alerts))
    # Fixed alpha=1/100, using integer arithmetic at the exact budget boundary.
    fp_budget = normal_n // 100
    fpr_within_budget = fp <= fp_budget if normal_n else None
    total = _detection(attack, alerts)
    result = {
        "source_id": source_id,
        "rows": len(families),
        "threshold": threshold,
        "ready": ready,
        "normal_n": normal_n,
        "fp": fp,
        "fp_budget": fp_budget,
        "fpr": fp / normal_n if normal_n else None,
        "fpr_within_budget": fpr_within_budget,
        "qualified": bool(ready and normal_n > 0 and fpr_within_budget),
        "attack_n": total["n"],
        "tp": total["tp"],
        "fn": total["fn"],
        "recall": total["recall"],
        "unresolved_n": int(np.count_nonzero(unresolved)),
        "unresolved_alerts": int(np.count_nonzero(unresolved & alerts)),
        "alerts": int(np.count_nonzero(alerts)),
        "by_family": {family: _detection(families == family, alerts) for family in _family_names(families)},
    }
    return result, alerts


def evaluate_cell(truth_frame: pd.DataFrame, score, rule: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a previously fixed rule in one cell, with strict score > threshold.

    ``rule`` contains ``threshold`` and optional ``ready``. A non-finite threshold
    is never ready. An unready rule cannot qualify through zero alerts. Missing
    attack families retain n=0 and recall=None rather than being counted as a pass.
    """
    families, scores, source_id = _prepare(truth_frame, score)
    return _summary(families, scores, source_id, rule)[0]


def _changes(mask: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    n = int(np.count_nonzero(mask))
    gained = int(np.count_nonzero(mask & ~left & right))
    lost = int(np.count_nonzero(mask & left & ~right))
    return {
        "n": n,
        "present": n > 0,
        "left_alerts": int(np.count_nonzero(mask & left)),
        "right_alerts": int(np.count_nonzero(mask & right)),
        "gained": gained,
        "lost": lost,
        "retained": int(np.count_nonzero(mask & left & right)),
        "neither": int(np.count_nonzero(mask & ~left & ~right)),
        "net": gained - lost,
    }


def paired_cell(
    truth_frame: pd.DataFrame,
    left_score,
    left_rule: Mapping[str, Any],
    right_score,
    right_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Return right-minus-left paired changes, retaining both arms' qualification."""
    families, left_scores, source_id = _prepare(truth_frame, left_score)
    _, right_scores, _ = _prepare(truth_frame, right_score)
    left_summary, left = _summary(families, left_scores, source_id, left_rule)
    right_summary, right = _summary(families, right_scores, source_id, right_rule)
    normal = families == BENIGN
    unresolved = families == UNRESOLVED
    return {
        "source_id": source_id,
        "direction": "right_minus_left",
        "left": left_summary,
        "right": right_summary,
        "left_qualified": left_summary["qualified"],
        "right_qualified": right_summary["qualified"],
        "both_qualified": left_summary["qualified"] and right_summary["qualified"],
        "all_rows": _changes(np.ones(len(families), dtype=bool), left, right),
        "normal": _changes(normal, left, right),
        "attacks": _changes(~(normal | unresolved), left, right),
        "unresolved": _changes(unresolved, left, right),
        "by_family": {family: _changes(families == family, left, right) for family in _family_names(families)},
    }


def oracle_cell(truth_frame: pd.DataFrame, score) -> dict[str, Any]:
    """Diagnostic-only maximal strict upper-threshold detection under a 1% FP budget.

    The (budget+1)-th largest normal score is the lowest admissible boundary.
    Equal-score groups are never split to fill the FP budget. This function reads
    evaluation labels and must not supply an operational threshold.
    """
    families, scores, source_id = _prepare(truth_frame, score)
    normal_scores = scores[families == BENIGN]
    n = len(normal_scores)
    if n:
        budget = n // 100
        ascending_index = n - 1 - budget
        threshold = float(np.partition(normal_scores, ascending_index)[ascending_index])
        rule = {"threshold": threshold, "ready": True}
    else:
        rule = {"threshold": None, "ready": False}
    summary, _ = _summary(families, scores, source_id, rule)
    summary.update(
        diagnostic_only=True,
        threshold_origin="evaluation_normal_labels",
        comparison="strict_greater",
        normal_at_threshold=int(np.count_nonzero(normal_scores == rule["threshold"])) if n else 0,
        status="ok" if n else "no_normal_denominator",
    )
    return summary
