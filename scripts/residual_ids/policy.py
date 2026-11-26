"""Frozen operating points and paired evaluation for a residual IDS branch."""
from __future__ import annotations

from scripts.ait_forward.decisions import (
    TARGETS,
    _changes,
    _event_changes,
    _groups,
    _required_event_report,
    _score_index,
    _target_results,
    calibrate_rules,
    validate_required_event_kinds,
)
from scripts.coverage_batch.run import report


def freeze_residual(incumbent_scores, branch_scores, groups, incumbent_threshold):
    incumbent = _score_index(incumbent_scores)
    groups = _groups(groups, incumbent, allow_empty=False)
    if (
        set().union(*map(set, groups.values())) != incumbent.keys()
        or sum(map(len, groups.values())) != len(incumbent)
    ):
        raise ValueError("C units must partition the entire normal calibration cohort")
    negatives = {fid for fid, value in incumbent.items() if value <= incumbent_threshold}
    branch = _score_index(branch_scores) if branch_scores else {}
    if branch.keys() != negatives:
        raise ValueError("Branch C scores must cover exactly the incumbent nonalerts")

    units = []
    for name, ids in groups.items():
        remaining_scores = sorted((branch[fid] for fid in ids if fid in negatives), reverse=True)
        prior_fp = len(ids) - len(remaining_scores)
        budget = len(ids) // 100
        remaining = budget - prior_fp
        threshold = remaining_scores[remaining] if remaining >= 0 else None
        units.append({
            "unit_id": name, "normal_count": len(ids), "FP_budget": budget,
            "incumbent_FP": prior_fp, "remaining_FP_budget": remaining, "own_threshold": threshold,
        })
    enabled = all(unit["remaining_FP_budget"] >= 0 for unit in units)
    threshold = max(unit["own_threshold"] for unit in units) if enabled else None
    for unit in units:
        extra = sum(
            fid in branch and branch[fid] > threshold for fid in groups[unit["unit_id"]]
        ) if enabled else 0
        unit["rescue_FP"] = extra
        unit["total_FP"] = unit["incumbent_FP"] + extra
    return {
        "enabled": enabled, "threshold": threshold, "comparison": ">",
        "incumbent_threshold": incumbent_threshold, "alpha": 0.01,
        "disabled_reason": None if enabled else "incumbent_already_exceeds_C_business_budget",
        "zero_remaining_FP_units": [u["unit_id"] for u in units if u["remaining_FP_budget"] == 0],
        "business_units": units, "population_guarantee": False,
    }


def predict_residual(incumbent_alerts, branch_scores, rule):
    if not rule["enabled"]:
        return dict(incumbent_alerts)
    if branch_scores.keys() != {fid for fid, alert in incumbent_alerts.items() if not alert}:
        raise ValueError("Branch scores are misaligned with incumbent nonalerts")
    return {
        fid: alert or branch_scores[fid] > rule["threshold"]
        for fid, alert in incumbent_alerts.items()
    }


def evaluate_alerts(alerts, *, normal_units, attack_ids, events, required_events=()):
    units = _groups(normal_units, alerts, allow_empty=False)
    attacks = {kind: attack_ids[kind] for kind in TARGETS}
    floors = {kind: 1 if attacks[kind] else None for kind in TARGETS}
    targets = _target_results({kind: attacks[kind] or () for kind in TARGETS}, alerts, floors)
    normal_ids = {fid for ids in units.values() for fid in ids}
    positives = {fid for ids in attacks.values() if ids for fid in ids}
    score_rows = {
        fid: {"label": 0 if fid in normal_ids else 1 if fid in positives else None,
              "prediction": prediction}
        for fid, prediction in alerts.items()
    }
    normal = report.normal_unit_results(
        [{"unit_id": name, "flow_ids": ids} for name, ids in units.items()], score_rows,
    )
    event_bounds = report.event_results(events, score_rows)
    if required_events:
        validate_required_event_kinds(events, required_events)
    return {
        "normal_units": normal,
        "all_normal_units_within_budget": all(x["empirical_budget_met"] is True for x in normal),
        "targets": targets, "events": event_bounds,
        "required_events": _required_event_report(event_bounds, targets, required_events)
        if required_events else None,
        "unknown_alerts": sum(alerts[fid] for fid in alerts.keys() - normal_ids - positives),
        "unknown_rows": len(alerts.keys() - normal_ids - positives),
    }


def paired_alerts(before, after, *, normal_units, attack_ids, before_report, after_report):
    if before.keys() != after.keys():
        raise ValueError("Paired predictions must share precisely the same identities")
    event_changes = _event_changes(before_report["events"], after_report["events"])
    for item in event_changes:
        for bound in ("lower_hit", "upper_hit"):
            item[bound]["before"] = item[bound].pop("pooled")
            item[bound]["after"] = item[bound].pop("business_max")
    return {
        "normal_units": [
            {"unit_id": name, **_changes(ids, before, after)} for name, ids in normal_units.items()
        ],
        "targets": {kind: _changes(attack_ids[kind] or (), before, after) for kind in TARGETS},
        "events": event_changes,
    }


def definite_event_counts(events):
    return {
        kind: sum(event["lower_hit"] is True for event in events if event["kind"] == kind)
        if any(event["kind"] == kind and event["lower_hit"] is not None for event in events) else None
        for kind in TARGETS
    }


def rank_diagnostic(incumbent_scores, branch_scores, units, attack_ids, threshold, incumbent_alerts, events=()):
    """Use opened normal labels for a feasibility bound, without exporting its cutoff."""
    normal_ids = {fid for ids in units.values() for fid in ids}
    feasible = freeze_residual(
        {fid: incumbent_scores[fid] for fid in normal_ids},
        {fid: score for fid, score in branch_scores.items() if fid in normal_ids},
        units, threshold,
    )
    if not feasible["enabled"]:
        return {"status": "incumbent_over_budget", "max_TP": None}
    upper = predict_residual(incumbent_alerts, branch_scores, feasible)
    return _rank_result(upper, attack_ids, events)


def _rank_result(alerts, attack_ids, events):
    event_bounds = report.event_results(events, {fid: {"prediction": alert} for fid, alert in alerts.items()})
    return {
        "status": "observed_rank_ceiling",
        "max_TP": {
            kind: sum(alerts[fid] for fid in attack_ids[kind]) if attack_ids[kind] else None
            for kind in TARGETS
        },
        "max_definite_events": definite_event_counts(event_bounds),
    }


def standalone_rank_diagnostic(scores, units, attack_ids, events=()):
    normal = {fid: scores[fid] for members in units.values() for fid in members}
    feasible_threshold = calibrate_rules(normal, units)["business_max"]["threshold"]
    return _rank_result({fid: value > feasible_threshold for fid, value in scores.items()}, attack_ids, events)
