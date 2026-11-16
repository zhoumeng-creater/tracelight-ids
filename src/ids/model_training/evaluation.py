"""Fixed-C evaluation and registered, identity-paired development decisions.

Evaluation labels are used only for reporting, never to change a threshold.
Qualification is per complete cell; UTC-day and same-service results remain
visible diagnostics rather than additional, retrospectively chosen criteria.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ids.route_batch.metrics import threshold as register_threshold


IDENTITY = ["source_id", "row_id"]
TRUTH_COLUMNS = [*IDENTITY, "label", "family", "flow_end_ms", "ip_protocol", "orig_port", "resp_port"]
FOCUS_FAMILIES = ("ExternalC2", "Scanning", "Download")
REGISTERED_D_CELLS = tuple(
    f"TQH-D_i{interval}_j{jitter}"
    for interval in (30, 300, 1800, 3600) for jitter in (0, 30, 70)
)


def calibrate(scores) -> dict:
    """Register the existing ceil((n+1)*.99) normal-only order statistic.

    Caller supplies the fixed C normal identities only. A small calibration
    sample whose rank exceeds n has no finite threshold and is not ready.
    """
    result = register_threshold(scores, alpha=0.01)
    result["ready"] = result["threshold"] is not None
    result["shared_across_cells"] = True
    return result


def _frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TRUTH_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation frame missing columns: {sorted(missing)}")
    if frame.empty or frame[TRUTH_COLUMNS].isna().any().any():
        raise ValueError("Evaluation requires nonempty rows and nonnull identity/truth fields")
    if frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate (source_id, row_id) evaluation identity")
    if not frame.source_id.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("source_id must contain nonempty strings")
    if not frame.family.map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("family must contain nonempty strings")
    if not frame.label.isin([-1, 0, 1]).all():
        raise ValueError("label must be -1 (unresolved), 0 (normal), or 1 (attack)")
    expected = np.where(frame.family.eq("Benign"), 0, np.where(frame.family.eq("Unresolved"), -1, 1))
    if not np.array_equal(frame.label.to_numpy(), expected):
        raise ValueError("label and family disagree")
    for name in ("flow_end_ms", "ip_protocol", "orig_port", "resp_port"):
        values = frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"{name} must contain finite integers")
    result = frame.copy().reset_index(drop=True)
    result["utc_date"] = pd.to_datetime(result.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return result


def _boolean(values, length, name):
    result = np.asarray(values)
    if result.shape != (length,) or result.dtype.kind != "b":
        raise ValueError(f"{name} must be an aligned boolean array")
    return result


def _counts(frame: pd.DataFrame) -> dict:
    normal = frame.label.eq(0).to_numpy()
    attack = frame.label.eq(1).to_numpy()
    unknown = frame.label.eq(-1).to_numpy()
    alert = frame.alert.to_numpy(dtype=bool)
    normal_n, attack_n = int(normal.sum()), int(attack.sum())
    fp, tp = int((normal & alert).sum()), int((attack & alert).sum())
    ready = bool(frame.ready.all())
    within = fp <= normal_n // 100 if normal_n else None
    families = sorted(set(FOCUS_FAMILIES) | set(frame.loc[attack, "family"]))
    by_family = {}
    for family in families:
        selected = frame.family.eq(family).to_numpy()
        n, found = int(selected.sum()), int((selected & alert).sum())
        by_family[family] = {"n": n, "present": bool(n), "tp": found, "fn": n-found,
                             "recall": found/n if n else None}
    return {
        "rows": len(frame), "normal_n": normal_n, "attack_n": attack_n,
        "fp": fp, "fp_budget": normal_n // 100, "fpr": fp/normal_n if normal_n else None,
        "fpr_within_budget": within, "ready": ready,
        "qualified": bool(ready and normal_n and within),
        "tp": tp, "fn": attack_n-tp, "recall": tp/attack_n if attack_n else None,
        "unresolved_n": int(unknown.sum()), "unresolved_alerts": int((unknown & alert).sum()),
        "alerts": int(alert.sum()), "by_family": by_family,
    }


def _same_service(frame: pd.DataFrame) -> list[dict]:
    """Normal denominators for each observed attack (protocol, responder-port).

    Service definitions come from observed attack metadata solely for descriptive
    reporting. No service-specific threshold is fitted or deployed.
    """
    services = frame.loc[frame.label.eq(1), ["ip_protocol", "resp_port"]].drop_duplicates()
    rows = []
    for protocol, port in services.sort_values(["ip_protocol", "resp_port"]).itertuples(index=False, name=None):
        selected = frame.ip_protocol.eq(protocol) & frame.resp_port.eq(port)
        rows.append({"ip_protocol": int(protocol), "resp_port": int(port),
                     **_counts(frame.loc[selected]), "descriptive_only": True})
    return rows


def _reports(decision_frame: pd.DataFrame, model_id: str, threshold):
    units, days = [], []
    for source, group in decision_frame.groupby("source_id", sort=True):
        units.append({"model_id": model_id, "source_id": source, "threshold": threshold,
                      **_counts(group), "same_service_normal": _same_service(group)})
        for day, daily in group.groupby("utc_date", sort=True):
            days.append({"model_id": model_id, "source_id": source, "utc_date": day,
                         "threshold": threshold, **_counts(daily),
                         "same_service_normal": _same_service(daily)})
    return units, days, decision_frame


def evaluate_decisions(frame, alerts, *, model_id, threshold=None):
    """Report an already fixed Boolean policy, including the historical OR rule."""
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model_id must be a nonempty string")
    prepared = _frame(frame)
    prepared["alert"] = _boolean(alerts, len(prepared), "alerts")
    prepared["ready"] = True
    prepared["model_id"] = model_id
    prepared["threshold"] = threshold
    return _reports(prepared, model_id, threshold)


def evaluate(frame, scores, threshold, *, model_id):
    """Evaluate all supplied cells and UTC days using one fixed model threshold."""
    prepared = _frame(frame)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(prepared),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite and aligned with evaluation rows")
    registration = threshold if isinstance(threshold, Mapping) else {"threshold": threshold}
    boundary = registration["threshold"]
    if boundary is not None and not np.isfinite(float(boundary)):
        raise ValueError("threshold must be finite or None for an unready registration")
    boundary = None if boundary is None else float(boundary)
    ready = boundary is not None and bool(registration.get("ready", True))
    alerts = values > float(boundary) if ready else np.zeros(len(values), dtype=bool)
    _, _, decisions = evaluate_decisions(prepared, alerts, model_id=model_id, threshold=boundary)
    decisions["score"] = values
    decisions["ready"] = ready
    return _reports(decisions, model_id, boundary)


def legacy_rule(frame: pd.DataFrame) -> np.ndarray:
    """Frozen UDP/either-port-443 and duration >=30 seconds rule."""
    _frame(frame)
    if "duration_seconds" not in frame or not np.isfinite(frame.duration_seconds.to_numpy(dtype=float)).all():
        raise ValueError("Legacy rule requires finite duration_seconds")
    return (frame.ip_protocol.eq(17) & (frame.orig_port.eq(443) | frame.resp_port.eq(443))
            & frame.duration_seconds.ge(30)).to_numpy(dtype=bool)


def _identities(frame, mask):
    return frame.loc[mask, [*IDENTITY, "family"]].to_dict(orient="records")


def _paired(candidate, reference, d_cells):
    rows = []
    for source, current in candidate.groupby("source_id", sort=True):
        previous = reference.loc[current.index]
        new, old = _counts(current), _counts(previous)
        attack = current.label.eq(1)
        added = attack & current.alert & ~previous.alert
        lost = attack & ~current.alert & previous.alert
        old_over_budget = old["normal_n"] > 0 and not old["fpr_within_budget"]
        repair = bool(new["qualified"] and old_over_budget)
        tp_gain = bool(new["qualified"] and old["qualified"] and new["tp"] > old["tp"])
        no_drop = not old["qualified"] or new["tp"] >= old["tp"]
        family_changes = {}
        for family in new["by_family"]:
            selected = current.family.eq(family)
            gained, dropped = int((selected & added).sum()), int((selected & lost).sum())
            family_changes[family] = {"n": int(selected.sum()), "present": bool(selected.any()),
                                      "added_tp": gained, "lost_tp": dropped, "net_tp": gained-dropped}
        rows.append({
            "source_id": source, "candidate": new, "reference": old,
            "reference_over_budget": bool(old_over_budget),
            "reference_over_budget_tp": old["tp"] if old_over_budget else None,
            "reference_tp_is_required_bound": old["qualified"],
            "no_tp_drop_in_reference_qualified_cell": bool(no_drop),
            "qualification_repair": repair, "qualified_tp_gain": tp_gain,
            "strict_D_improvement": bool(source in d_cells and (repair or tp_gain)),
            "added_tp": int(added.sum()), "lost_tp": int(lost.sum()),
            "net_tp": int(added.sum()-lost.sum()),
            "added_tp_identities": _identities(current, added),
            "lost_tp_identities": _identities(current, lost),
            "by_family": family_changes,
        })
    all_qualified = all(row["candidate"]["qualified"] for row in rows)
    no_drop = all(row["no_tp_drop_in_reference_qualified_cell"] for row in rows)
    strict = any(row["strict_D_improvement"] for row in rows)
    return {"cells": rows, "all_candidate_cells_qualified": all_qualified,
            "no_tp_drop_in_reference_qualified_cells": no_drop, "strict_D_improvement": strict,
            "passed": bool(all_qualified and no_drop and strict),
            "added_tp": sum(row["added_tp"] for row in rows),
            "lost_tp": sum(row["lost_tp"] for row in rows)}


def _registered_cells(design):
    roles = design["roles"]
    b_cells = roles["E"]["cells"]
    d_cells = roles["D"]["cells"]
    # The reviewed design uses this exact description; the executable manifest
    # may replace it with the explicit twelve registered source identities.
    if d_cells == "all 12 registered TQH-D cells":
        d_cells = list(REGISTERED_D_CELLS)
    for name, cells, expected in (("E", b_cells, 3), ("D", d_cells, 12)):
        if not isinstance(cells, list) or len(cells) != expected or len(set(cells)) != expected:
            raise ValueError(f"{name} must explicitly register {expected} distinct cells")
    if set(b_cells) & set(d_cells):
        raise ValueError("E and D cells must be disjoint")
    return set(b_cells), set(d_cells)


def compare(decision_frames: dict[str, pd.DataFrame], design: dict, *,
            hgb_id="hgb", legacy_union_id="legacy_hgb_or_rule", ple_ids=None, no_ple_ids=None) -> dict:
    """Apply the registered three-seed, two-reference and paired-PLE rules.

    Every supplied historical frame is identity/truth checked but only the named
    two references and six TabM candidates participate in adoption decisions.
    """
    seeds = design["training"]["neural_seeds"]
    if list(seeds) != [17, 29, 43]:
        raise ValueError("The fixed comparison requires seeds 17, 29, 43")
    ple_ids = ple_ids or {seed: f"tabm_ple_s{seed}" for seed in seeds}
    no_ple_ids = no_ple_ids or {seed: f"tabm_no_ple_s{seed}" for seed in seeds}
    if set(ple_ids) != set(seeds) or set(no_ple_ids) != set(seeds):
        raise ValueError("Exactly one full and ablated model per registered seed is required")
    required = [hgb_id, legacy_union_id, *ple_ids.values(), *no_ple_ids.values()]
    if len(set(required)) != 8:
        raise ValueError("References and all seed candidates must have distinct model IDs")
    missing = set(required).difference(decision_frames)
    if missing:
        raise ValueError(f"Missing registered model decisions: {sorted(missing)}")
    b_cells, d_cells = _registered_cells(design)
    expected_cells = b_cells | d_cells
    aligned = {}
    anchor = None
    for model_id, frame in decision_frames.items():
        checked = _frame(frame)
        if set(checked.source_id) != expected_cells:
            raise ValueError(f"{model_id}: evaluation cells differ from registered E/D cells")
        if "model_id" not in checked or not checked.model_id.eq(model_id).all():
            raise ValueError(f"{model_id}: decision model_id differs from its mapping key")
        for name in ("alert", "ready"):
            if name not in checked:
                raise ValueError(f"{model_id}: missing {name} decisions")
            _boolean(checked[name], len(checked), name)
        if (checked.alert & ~checked.ready).any():
            raise ValueError("An unready policy cannot emit alerts")
        checked = checked.set_index(IDENTITY, drop=False)
        if anchor is None:
            anchor = checked
        elif len(checked) != len(anchor) or not checked.index.isin(anchor.index).all():
            raise ValueError(f"{model_id}: evaluation identities differ")
        checked = checked.reindex(anchor.index)
        if not np.array_equal(checked[TRUTH_COLUMNS].to_numpy(), anchor[TRUTH_COLUMNS].to_numpy()):
            raise ValueError(f"{model_id}: aligned truth or observation metadata differ")
        aligned[model_id] = checked.reset_index(drop=True)
    for role, cells in (("E", b_cells), ("D", d_cells)):
        selected = aligned[hgb_id].source_id.isin(cells)
        for key, label in (("normal", 0), ("attack", 1)):
            expected = design["roles"][role].get(key)
            if expected is not None and int((selected & aligned[hgb_id].label.eq(label)).sum()) != expected:
                raise ValueError(f"{role}: {key} count differs from registered role")
    expected_b_tp = int(design["roles"]["E"]["attack"])
    candidate_results, ple_results = {}, {}
    for seed in seeds:
        for variant, ids in (("ple", ple_ids), ("no_ple", no_ple_ids)):
            model_id = ids[seed]
            current = aligned[model_id]
            b_tp = int((current.source_id.isin(b_cells) & current.label.eq(1) & current.alert).sum())
            references = {reference_id: _paired(current, aligned[reference_id], d_cells)
                          for reference_id in (hgb_id, legacy_union_id)}
            candidate_results[model_id] = {
                "seed": seed, "variant": variant, "B_tp": b_tp, "required_B_tp": expected_b_tp,
                "B_capability_preserved": b_tp == expected_b_tp, "references": references,
                "development_candidate": bool(b_tp == expected_b_tp
                                              and all(pair["passed"] for pair in references.values())),
                "lost_tp_to_references": {key: value["lost_tp"] for key, value in references.items()},
            }
        ple_results[str(seed)] = {"full_model_id": ple_ids[seed], "ablation_model_id": no_ple_ids[seed],
                                  **_paired(aligned[ple_ids[seed]], aligned[no_ple_ids[seed]], d_cells)}
    variant_qualified = {
        name: all(candidate_results[ids[seed]]["development_candidate"] for seed in seeds)
        for name, ids in (("ple", ple_ids), ("no_ple", no_ple_ids))
    }
    return {
        "status": "already_opened_development_comparison",
        "primary_reference": hgb_id, "legacy_capability_reference": legacy_union_id,
        "expected_cells": {"E": sorted(b_cells), "D": sorted(d_cells)},
        "candidates": candidate_results, "all_seed_development_candidate": variant_qualified,
        "PLE_same_seed_comparisons": ple_results,
        "PLE_detection_benefit": all(pair["passed"] for pair in ple_results.values()),
        "historical_report_only": sorted(set(decision_frames).difference(required)),
        "HGB_all_cells_qualified": all(_counts(g)["qualified"]
                                       for _, g in aligned[hgb_id].groupby("source_id")),
        "automatic_deployment_adoption": False,
        "identity_loss_interpretation": "Per-cell TP counts do not guarantee retention of each old TP; identities are reported",
        "failure_action": "retain tradeoffs or negative result; no threshold search or parameter expansion",
    }
