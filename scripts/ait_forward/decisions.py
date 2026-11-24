"""Apply two frozen C rules to scores from the same fitted detector.

Scores, cohort membership, and event associations are provided by the caller.
These functions never fit, score, load a model, or select a deployment rule.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real

from scripts.coverage_batch.run import calibration_threshold, report


TARGETS = ("control", "download")
RULES = ("pooled", "business_max")


def _finite_score(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Scores and thresholds must be finite real scalars")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("Scores and thresholds must be finite real scalars") from error
    if not isfinite(result):
        raise ValueError("Scores and thresholds must be finite real scalars")
    return result


def _score_index(scores):
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("Provide a nonempty identity-to-score mapping")
    if any(not isinstance(fid, str) or not fid.strip() for fid in scores):
        raise ValueError("Score identities must be nonempty strings")
    return {fid: _finite_score(value) for fid, value in scores.items()}


def _groups(groups, scores, *, allow_empty):
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("Provide named groups")
    result = {}
    for name, ids in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Group names must be nonempty strings")
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
            raise ValueError(f"Expected an identity sequence for {name}")
        members = tuple(ids)
        if any(not isinstance(fid, str) or not fid.strip() for fid in members):
            raise ValueError(f"Invalid identity in {name}")
        if len(set(members)) != len(members):
            raise ValueError(f"Duplicate identity in {name}")
        if not members and not allow_empty:
            raise ValueError(f"Empty group {name}")
        if set(members) - scores.keys():
            raise ValueError(f"Missing score for a member of {name}")
        result[name] = members
    return result


def calibrate_rules(calibration_scores, business_groups):
    """Freeze the pooled rule and the maximum of the C business thresholds."""
    scores = _score_index(calibration_scores)
    groups = _groups(business_groups, scores, allow_empty=False)
    if set().union(*map(set, groups.values())) != scores.keys() or (
        sum(map(len, groups.values())) != len(scores)
    ):
        raise ValueError("C business groups must be a disjoint, exhaustive partition")
    pooled = calibration_threshold(list(scores.values()))
    working_points = {
        name: calibration_threshold([scores[fid] for fid in ids]) for name, ids in groups.items()
    }
    thresholds = {
        "pooled": pooled["threshold"],
        "business_max": max(point["threshold"] for point in working_points.values()),
    }
    return {
        rule: {
            "threshold": threshold,
            "comparison": ">",
            "alpha": 0.01,
            "C_count": len(scores),
            "integer_FP_budget": pooled["integer_FP_budget"],
            "C_empirical_FP": sum(score > threshold for score in scores.values()),
            "C_boundary_tie_count": sum(score == threshold for score in scores.values()),
            "population_guarantee": False,
            "business_groups": [
                {
                    "unit_id": name,
                    "normal_count": len(ids),
                    "integer_FP_budget": working_points[name]["integer_FP_budget"],
                    "own_threshold": working_points[name]["threshold"],
                    "FP": sum(scores[fid] > threshold for fid in ids),
                }
                for name, ids in groups.items()
            ],
        }
        for rule, threshold in thresholds.items()
    }


def _validate_floors(minimum_tp):
    if not isinstance(minimum_tp, Mapping) or set(minimum_tp) != set(TARGETS):
        raise ValueError("Provide absolute minimum TP counts for control and download")
    result = {}
    for kind, value in minimum_tp.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0):
            raise ValueError(f"{kind} minimum TP must be a nonnegative integer or None")
        result[kind] = None if value is None else int(value)
    return result


def _target_results(attack_ids, predictions, floors):
    results = {}
    for kind, ids in attack_ids.items():
        detected = [fid for fid in ids if predictions[fid]] if ids else None
        tp = len(detected) if detected is not None else None
        floor = floors[kind]
        results[kind] = {
            "status": "available" if ids else "unavailable",
            "positive_flows": len(ids) if ids else None,
            "TP": tp,
            "FN": len(ids) - tp if tp is not None else None,
            "minimum_TP": floor,
            "minimum_TP_met": tp >= floor if tp is not None and floor is not None else None,
            "detected_flow_ids": detected,
        }
    return results


def _changes(ids, before, after):
    if not ids:
        return {"gained_flow_ids": None, "lost_flow_ids": None, "alert_count_delta": None}
    gains = [fid for fid in ids if not before[fid] and after[fid]]
    losses = [fid for fid in ids if before[fid] and not after[fid]]
    return {
        "gained_flow_ids": gains,
        "lost_flow_ids": losses,
        "alert_count_delta": len(gains) - len(losses),
    }


def _event_changes(before, after):
    output = []
    for left, right in zip(before, after, strict=True):
        row = {"event_id": left["event_id"], "kind": left["kind"]}
        for bound in ("lower_hit", "upper_hit"):
            a, b = left[bound], right[bound]
            row[bound] = {
                "pooled": a,
                "business_max": b,
                "gained": (not a and b) if a is not None and b is not None else None,
                "lost": (a and not b) if a is not None and b is not None else None,
            }
        output.append(row)
    return output


def validate_required_event_kinds(events, required_events):
    observed = {event["event_id"]: event["kind"] for event in events}
    for row in required_events or ():
        identity = row["event_id"]
        if identity in observed and observed[identity] != row["kind"]:
            raise ValueError(f"Required event has a different qualified target: {identity}")


def _required_event_report(observed_events, targets, required_events):
    if required_events is None:
        return {"status": "not_registered", "required_events": []}
    observed = {event["event_id"]: event for event in observed_events}
    events = []
    for requirement in required_events:
        kind = requirement["kind"]
        result_row = observed.get(requirement["event_id"])
        lower = result_row["lower_hit"] if result_row else None
        upper = result_row["upper_hit"] if result_row else None
        if targets[kind]["status"] == "unavailable":
            status, reason = "indeterminate", "target_unavailable"
        elif result_row is None:
            status, reason = "indeterminate", "required_event_missing"
        elif lower is True:
            status, reason = "met", "guaranteed_hit"
        elif upper is False:
            status, reason = "missed", "no_candidate_hit"
        else:
            status = "indeterminate"
            reason = "possible_hit_only" if upper is True else "unresolved_candidates"
        events.append({**requirement, "lower_hit": lower, "upper_hit": upper, "status": status, "reason": reason})
    states = {event["status"] for event in events}
    status = "missed" if "missed" in states else "met" if states == {"met"} else "indeterminate"
    return {"status": status, "required_events": events}


def evaluate_event_protocol(result, required_events):
    """Apply the preregistered must-hit requirement to each required event."""
    requirements = None if required_events is None else tuple(required_events)
    return {
        name: _required_event_report(rule["events"], rule["targets"], requirements)
        for name, rule in result["rules"].items()
    }


def _event_ordering_diagnosis(current, best):
    if best["status"] == "missed":
        return "rank_cannot_hit_event"
    if best["reason"] == "possible_hit_only":
        return "rank_can_only_possibly_hit_event"
    if best["status"] == "indeterminate":
        return "event_evidence_indeterminate"
    if current["status"] != "met":
        return "frozen_rule_lacks_guaranteed_hit_but_rank_has_room"
    return "frozen_rule_guaranteed_hit_and_rank_has_room"


def _rule_contrast(results, paired, predictions):
    reductions = [unit["unit_id"] for unit in paired["normal_units"] if (unit["alert_count_delta"] or 0) < 0]
    repaired = [
        before["unit_id"]
        for before, after in zip(results["pooled"]["normal_units"], results["business_max"]["normal_units"],
                                 strict=True)
        if before["empirical_budget_met"] is False and after["empirical_budget_met"] is True
    ]
    lost_attacks = {
        kind: None if rows["lost_flow_ids"] is None else len(rows["lost_flow_ids"])
        for kind, rows in paired["targets"].items()
    }
    lost_event_bounds = [
        event["event_id"] for event in paired["events"]
        if event["lower_hit"]["lost"] is True or event["upper_hit"]["lost"] is True
    ]
    if predictions["pooled"] == predictions["business_max"]:
        status = "identical_alerts"
    elif not reductions:
        status = "no_measured_normal_fp_reduction"
    elif any(lost_attacks.values()) or lost_event_bounds:
        status = "normal_fp_reduction_with_detection_loss"
    else:
        status = "normal_fp_reduction_without_observed_detection_loss"
    return {"status": status, "normal_fp_reduction_units": reductions, "normal_budget_repaired_units": repaired,
            "lost_attack_flow_counts": lost_attacks, "lost_positive_event_bound_ids": lost_event_bounds}


def evaluate_comparison(evaluation_scores, *, normal_units, attack_ids, events, rules, minimum_tp,
                        required_events=None):
    """Apply frozen thresholds once and report paired and ranking-limit diagnostics.

    The caller supplies qualified evaluation identities. An unknown flow may be
    an event candidate but cannot enter a normal denominator or attack count.
    """
    scores = _score_index(evaluation_scores)
    units = _groups(normal_units, scores, allow_empty=True)
    if not isinstance(attack_ids, Mapping) or set(attack_ids) != set(TARGETS):
        raise ValueError("Provide attack identities for control and download")
    attacks = _groups(
        {kind: () if ids is None else ids for kind, ids in attack_ids.items()},
        scores,
        allow_empty=True,
    )
    normal_ids = set().union(*map(set, units.values()))
    positive_ids = set(attacks["control"]) | set(attacks["download"])
    if normal_ids & positive_ids or set(attacks["control"]) & set(attacks["download"]):
        raise ValueError("An evaluation identity has conflicting target or normal membership")
    floors = _validate_floors(minimum_tp)
    if set(rules) != set(RULES) or any(rules[name]["comparison"] != ">" for name in RULES):
        raise ValueError("Provide the two frozen strict-greater-than rules")
    thresholds = {name: _finite_score(rules[name]["threshold"]) for name in RULES}
    if thresholds["pooled"] > thresholds["business_max"]:
        raise ValueError("The C business rule cannot be more permissive than the C pooled rule")
    events = tuple(events)
    if any(event.get("kind") not in TARGETS or event.get("label", 1) != 1 for event in events):
        raise ValueError("Provide registered positive control/download events")
    positive_events = [{**event, "label": 1} for event in events]
    if required_events is not None:
        required_events = tuple(required_events)
        validate_required_event_kinds(positive_events, required_events)
    unit_rows = [{"unit_id": name, "flow_ids": list(ids)} for name, ids in units.items()]
    predictions, results = {}, {}
    for rule, threshold in thresholds.items():
        prediction = {fid: score > threshold for fid, score in scores.items()}
        predictions[rule] = prediction
        score_rows = {
            fid: {"label": 0 if fid in normal_ids else 1 if fid in positive_ids else None,
                  "prediction": alert}
            for fid, alert in prediction.items()
        }
        by_unit = report.normal_unit_results(unit_rows, score_rows)
        event_bounds = report.event_results(positive_events, score_rows)
        targets = _target_results(attacks, prediction, floors)
        results[rule] = {
            "threshold": threshold,
            "normal_units": by_unit,
            "num_normal_units_within_budget": sum(row["empirical_budget_met"] is True for row in by_unit),
            "all_normal_units_within_budget": all(row["empirical_budget_met"] is True for row in by_unit),
            "targets": targets,
            "events": event_bounds,
        }

    before = predictions["pooled"]
    after = predictions["business_max"]
    ceilings = event_ceiling = None
    if all(units.values()):
        # The maximal unit boundary is the lowest possible threshold satisfying
        # every observed evaluation unit. Report only attainable counts.
        feasible = max(calibration_threshold([scores[fid] for fid in ids])["threshold"] for ids in units.values())
        upper = _target_results(attacks, {fid: scores[fid] > feasible for fid in positive_ids}, floors)
        ceilings = {
            kind: {key: row[key] for key in ("status", "positive_flows", "minimum_TP")}
            | {"max_TP": row["TP"], "minimum_TP_attainable": row["minimum_TP_met"]}
            for kind, row in upper.items()
        }
        if required_events is not None:
            candidate_predictions = {
                fid: {"prediction": scores[fid] > feasible}
                for event in positive_events for fid in event["candidate_flow_ids"] if fid in scores
            }
            best = report.event_results(positive_events, candidate_predictions)
            event_ceiling = _required_event_report(best, upper, required_events)
    paired = {
        "normal_units": [{"unit_id": name, **_changes(ids, before, after)} for name, ids in units.items()],
        "targets": {kind: _changes(ids, before, after) for kind, ids in attacks.items()},
        "events": _event_changes(results["pooled"]["events"], results["business_max"]["events"]),
    }
    return {
        "rules": results,
        "paired": paired,
        "score_ranking_upper_bound": ceilings,
        "score_ranking_event_feasibility": event_ceiling,
        "rule_contrast": _rule_contrast(results, paired, predictions),
    }


def _flow_outcomes(result, attack_ids):
    """Report the evaluated scope; diagnostics never substitute an oracle cutoff."""
    outcome = {}
    for name, rule in result["rules"].items():
        targets = rule["targets"]
        floors_met = all(targets[kind]["minimum_TP_met"] for kind in TARGETS if attack_ids[kind] is not None)
        ranking = result["score_ranking_upper_bound"]
        if rule["all_normal_units_within_budget"] and floors_met:
            status = "criteria_met" if all(attack_ids.values()) else "partial_scope_criteria_met"
        elif ranking and all(
            ranking[kind]["minimum_TP_attainable"] for kind in TARGETS if attack_ids[kind] is not None
        ):
            status = "fixed_rule_failed_score_ordering_has_room"
        else:
            status = "fixed_score_ordering_insufficient"
        outcome[name] = status
    return outcome


def _confirmation_outcomes(flow_outcomes, event_evaluation):
    outcomes = {}
    for name, flow_status in flow_outcomes.items():
        event_status = event_evaluation[name]["status"]
        if flow_status not in {"criteria_met", "partial_scope_criteria_met"}:
            outcomes[name] = "flow_criteria_failed"
        elif event_status in {"missed", "indeterminate"}:
            outcomes[name] = "event_criteria_failed" if event_status == "missed" else "event_criteria_indeterminate"
        else:
            scope = "partial_scope_" if flow_status == "partial_scope_criteria_met" else ""
            criteria = "event_and_flow_criteria_met" if event_status == "met" else "flow_only_criteria_met"
            outcomes[name] = scope + criteria
    return outcomes


def evaluate_candidate(evaluation_scores, *, normal_units, attack_ids, events, rules, minimum_tp,
                       required_events=None):
    """Evaluate one fixed candidate with its own precomputed C-only rules."""
    events = tuple(events)
    if required_events is not None:
        required_events = tuple(required_events)
    result = evaluate_comparison(
        evaluation_scores, normal_units=normal_units, attack_ids=attack_ids, events=events,
        rules=rules, minimum_tp=minimum_tp, required_events=required_events,
    )
    flow_outcomes = _flow_outcomes(result, attack_ids)
    event_evaluation = evaluate_event_protocol(result, required_events)
    best = result["score_ranking_event_feasibility"]
    if best is not None:
        for rule in event_evaluation.values():
            for current, feasible in zip(rule["required_events"], best["required_events"], strict=True):
                current["ordering_diagnosis"] = _event_ordering_diagnosis(current, feasible)
    return {
        "outcomes": flow_outcomes,
        "result": result,
        "event_evaluation": event_evaluation,
        "confirmation_outcomes": _confirmation_outcomes(flow_outcomes, event_evaluation),
    }
