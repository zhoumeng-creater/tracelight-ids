"""Fixed-C, identity-paired evaluation of the registered protocol-input design.

This module consumes scores and already frozen historical decisions. It never
fits a model, searches thresholds, combines candidate alerts or selects seeds.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ids.model_training.evaluation import (
    IDENTITY, TRUTH_COLUMNS, _boolean, _counts, _frame, _paired,
    calibrate as _calibrate, evaluate, evaluate_decisions, legacy_rule,
)


SEEDS = (17, 29, 43)
HISTORICAL_IDS = ("r26_target_mlp17", "r26_target_mlp29", "r26_target_mlp43")


def calibrate(scores, *, expected_normal_rows=5511):
    """Apply the unchanged normal-only order statistic to the fixed C pool."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_normal_rows,) or not np.isfinite(values).all():
        raise ValueError("Calibration scores must match the registered finite C normal pool")
    result = _calibrate(values)
    if expected_normal_rows == 5511 and result["rank"] != 5457:
        raise ValueError("The unchanged C calibration rank must be 5457")
    return result


def _roles(design):
    result = {}
    for role, count in (("E_B", 3), ("E_D", 12)):
        cells = design["roles"][role]["cells"]
        if not isinstance(cells, list) or len(cells) != count:
            raise ValueError(f"{role} must register {count} complete cells")
        ids = [cell["source_id"] for cell in cells]
        if len(set(ids)) != count:
            raise ValueError(f"{role} contains duplicate cells")
        result[role] = {cell["source_id"]: cell for cell in cells}
    if set(result["E_B"]) & set(result["E_D"]):
        raise ValueError("E_B and E_D must be disjoint")
    return result


def _finite_scores(frame, model_id):
    if "score" not in frame:
        raise ValueError(f"{model_id}: frozen candidate scores are required")
    values = frame.score.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{model_id}: scores must be finite")
    return values


def _check_candidate_decisions(frame, model_id):
    values = _finite_scores(frame, model_id)
    if "threshold" not in frame or frame.ready.nunique() != 1:
        raise ValueError(f"{model_id}: one shared calibration readiness is required")
    boundaries = frame.threshold.unique()
    if len(boundaries) != 1:
        raise ValueError(f"{model_id}: one shared calibration threshold is required")
    boundary = boundaries[0]
    if bool(frame.ready.iloc[0]):
        if pd.isna(boundary) or not np.isfinite(float(boundary)):
            raise ValueError(f"{model_id}: ready calibration requires a finite threshold")
        expected = values > float(boundary)
    else:
        if not pd.isna(boundary):
            raise ValueError(f"{model_id}: unready calibration must have no threshold")
        expected = np.zeros(len(frame), dtype=bool)
    if not np.array_equal(expected, frame.alert.to_numpy(dtype=bool)):
        raise ValueError(f"{model_id}: decisions disagree with the strict shared threshold")


def _align(decision_frames, required, candidate_ids, roles, design):
    if not isinstance(decision_frames, Mapping):
        raise ValueError("decision_frames must map model IDs to full decision frames")
    missing = set(required) - set(decision_frames)
    if missing:
        raise ValueError(f"Missing registered model decisions: {sorted(missing)}")
    cells = set(roles["E_B"]) | set(roles["E_D"])
    aligned, anchor = {}, None
    for model_id, frame in decision_frames.items():
        checked = _frame(frame)
        if set(checked.source_id) != cells:
            raise ValueError(f"{model_id}: evaluation cells differ from the registration")
        if "model_id" not in checked or not checked.model_id.eq(model_id).all():
            raise ValueError(f"{model_id}: decision model_id differs from its mapping key")
        for column in ("ready", "alert"):
            if column not in checked:
                raise ValueError(f"{model_id}: missing {column}")
            _boolean(checked[column], len(checked), column)
        if (checked.alert & ~checked.ready).any():
            raise ValueError(f"{model_id}: an unready policy cannot emit alerts")
        if "duration_seconds" not in checked or not np.isfinite(
                checked.duration_seconds.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{model_id}: finite duration metadata is required")
        checked = checked.set_index(IDENTITY, drop=False)
        if anchor is None:
            anchor = checked.sort_index()
        elif len(checked) != len(anchor) or not checked.index.isin(anchor.index).all():
            raise ValueError(f"{model_id}: physical row identities differ")
        checked = checked.reindex(anchor.index)
        truth = [*TRUTH_COLUMNS, "duration_seconds"]
        if not np.array_equal(checked[truth].to_numpy(), anchor[truth].to_numpy()):
            raise ValueError(f"{model_id}: aligned truth or observation metadata differ")
        aligned[model_id] = checked.reset_index(drop=True)
        if model_id in candidate_ids:
            _check_candidate_decisions(aligned[model_id], model_id)
    anchor = next(iter(aligned.values()))
    if not anchor.label.isin([0, 1]).all():
        raise ValueError("The registered E roles contain only resolved binary labels")
    for role, role_cells in roles.items():
        selected = anchor.source_id.isin(role_cells)
        spec = design["roles"][role]
        for field, label in (("normal", 0), ("attack", 1)):
            if int((selected & anchor.label.eq(label)).sum()) != int(spec[field]):
                raise ValueError(f"{role}: {field} count differs from registration")
        if len(anchor.loc[selected]) != int(spec["rows"]):
            raise ValueError(f"{role}: row count differs from registration")
        for source, cell in role_cells.items():
            for field, label in (("normal", 0), ("attack", 1)):
                actual = int((anchor.source_id.eq(source) & anchor.label.eq(label)).sum())
                if actual != int(cell[field]):
                    raise ValueError(f"{source}: {field} denominator differs from registration")
    return aligned


def _rank_metrics(frame):
    scores = _finite_scores(frame, "ranking panel")
    labels = frame.label.to_numpy(dtype=np.int64)
    normal, positive = int((labels == 0).sum()), int((labels == 1).sum())
    return {
        "normal_n": normal, "attack_n": positive,
        "pauc_0_01": float(roc_auc_score(labels, scores, max_fpr=0.01)) if normal and positive else None,
        "tp": int((frame.label.eq(1) & frame.alert).sum()),
        "fp": int((frame.label.eq(0) & frame.alert).sum()),
        "strict_positive_negative_separation": bool(
            normal and positive and scores[labels == 1].min() > scores[labels == 0].max()),
    }


def ranking_report(frame, d_cells):
    """Report pooled D and TCP-only D without changing any decision."""
    d = frame.loc[frame.source_id.isin(d_cells)]
    return {"D_pooled": _rank_metrics(d), "D_TCP": _rank_metrics(d.loc[d.ip_protocol.eq(6)])}


def _pair(candidate, reference, d_cells):
    # Reuse the frozen identity accounting, but not its old adoption predicate:
    # a repaired FP failure alone is not the new strict D-TP gain condition.
    previous = _paired(candidate, reference, d_cells)
    result = {key: value for key, value in previous.items()
              if key not in {"passed", "strict_D_improvement"}}
    for row in result["cells"]:
        row.pop("strict_D_improvement")
        row["D_cell_strict_tp_gain"] = bool(
            row["source_id"] in d_cells and row["candidate"]["tp"] > row["reference"]["tp"])
    result["strict_D_cell_tp_gain"] = any(row["D_cell_strict_tp_gain"] for row in result["cells"])
    result["D_net_tp"] = sum(row["net_tp"] for row in result["cells"] if row["source_id"] in d_cells)
    return result


def _identity_rows(frame, mask):
    return frame.loc[mask, [*IDENTITY, "family", "ip_protocol", "duration_seconds"]].to_dict("records")


def _historical_loss_diagnostic(candidate, r29, reference, d_cells):
    """Trace the complete old TP-loss set, without a post-hoc duration cutoff."""
    reference_tp = reference.source_id.isin(d_cells) & reference.label.eq(1) & reference.alert
    old_lost = reference_tp & ~r29.alert
    old_retained = reference_tp & r29.alert
    recovered, still_lost = old_lost & candidate.alert, old_lost & ~candidate.alert
    newly_lost = old_retained & ~candidate.alert
    durations = reference.loc[old_lost, "duration_seconds"].to_numpy(dtype=np.float64)
    return {
        "definition": "D positive identities detected by the named historical reference but missed by paired R29 PLE",
        "duration_cutoff_applied": False,
        "old_lost_n": int(old_lost.sum()),
        "old_lost_protocol_counts": {str(int(k)): int(v) for k, v in
                                     reference.loc[old_lost, "ip_protocol"].value_counts().sort_index().items()},
        "old_lost_duration_seconds": ({"min": float(durations.min()), "median": float(np.median(durations)),
                                        "max": float(durations.max())} if len(durations) else None),
        "old_lost_identities": _identity_rows(reference, old_lost),
        "recovered_n": int(recovered.sum()), "recovered_identities": _identity_rows(reference, recovered),
        "still_lost_n": int(still_lost.sum()), "still_lost_identities": _identity_rows(reference, still_lost),
        "newly_lost_from_R29_retained_n": int(newly_lost.sum()),
        "newly_lost_from_R29_retained_identities": _identity_rows(reference, newly_lost),
    }


def compare(decision_frames, design, *, implementation_contract_pass,
            main_ids=None, control_ids=None, legacy_union_id="legacy_hgb_or_rule",
            historical_ids=HISTORICAL_IDS, r29_ple_ids=None):
    """Apply all three paired seeds and the explicit implementation/meaning/effect AND.

    New candidates must contain the frozen score, shared threshold and its exact
    strict-greater decision. Historical frames are reused as supplied; their
    calibration is not recalculated. Only the registered main arm can be adopted.
    """
    if type(implementation_contract_pass) is not bool:
        raise ValueError("implementation_contract_pass must be an explicit boolean")
    if tuple(design["training"]["seeds"]) != SEEDS:
        raise ValueError("Exactly the registered seeds 17, 29, 43 are required")
    if tuple(historical_ids) != HISTORICAL_IDS:
        raise ValueError("All three registered R26 capability references are required")
    main_ids = main_ids or {seed: f"protocol_blind_s{seed}" for seed in SEEDS}
    control_ids = control_ids or {seed: f"protocol_aware_s{seed}" for seed in SEEDS}
    r29_ple_ids = r29_ple_ids or {seed: f"r29_tabm_ple_s{seed}" for seed in SEEDS}
    if any(set(ids) != set(SEEDS) for ids in (main_ids, control_ids, r29_ple_ids)):
        raise ValueError("Every arm and R29 diagnostic must supply exactly three registered seeds")
    references = (legacy_union_id, *historical_ids)
    required = [*references, *main_ids.values(), *control_ids.values(), *r29_ple_ids.values()]
    if len(set(required)) != len(required):
        raise ValueError("Candidate, control and historical IDs must be distinct")
    roles = _roles(design)
    b_cells, d_cells = set(roles["E_B"]), set(roles["E_D"])
    aligned = _align(decision_frames, required, set(main_ids.values()) | set(control_ids.values()), roles, design)
    seeds, model_metrics = {}, {}
    for model_id in [*main_ids.values(), *control_ids.values()]:
        current = aligned[model_id]
        model_metrics[model_id] = {
            "cells": [{"source_id": source, **_counts(group)} for source, group in
                      current.groupby("source_id", sort=True)],
            **ranking_report(current, d_cells),
        }
    for seed in SEEDS:
        main_id, control_id = main_ids[seed], control_ids[seed]
        main = aligned[main_id]
        a, b = model_metrics[main_id], model_metrics[control_id]
        mechanism = bool(a["D_pooled"]["pauc_0_01"] > b["D_pooled"]["pauc_0_01"]
                         and a["D_pooled"]["tp"] >= b["D_pooled"]["tp"])
        comparisons = {name: _pair(main, aligned[name], d_cells) for name in (*references, control_id)}
        checks = {
            "all_15_cells_qualified": all(cell["qualified"] for cell in a["cells"]),
            "B_each_cell_24_TP_total_72": bool(
                sum(cell["tp"] for cell in a["cells"] if cell["source_id"] in b_cells) == 72
                and all(cell["tp"] == 24 for cell in a["cells"] if cell["source_id"] in b_cells)),
            "no_TP_drop_in_qualified_historical_cells": all(
                comparisons[name]["no_tp_drop_in_reference_qualified_cells"] for name in references),
            "strict_D_cell_TP_gain_over_legacy_union": comparisons[legacy_union_id]["strict_D_cell_tp_gain"],
            "no_TP_drop_in_qualified_control_cells": comparisons[control_id][
                "no_tp_drop_in_reference_qualified_cells"],
            "strict_D_total_TP_gain_over_control": a["D_pooled"]["tp"] > b["D_pooled"]["tp"],
        }
        seeds[str(seed)] = {
            "main_model_id": main_id, "control_model_id": control_id,
            "mechanism_pass": mechanism,
            "mechanism_differences": {
                "D_pooled_pauc_delta": a["D_pooled"]["pauc_0_01"] - b["D_pooled"]["pauc_0_01"],
                "D_TP_delta": a["D_pooled"]["tp"] - b["D_pooled"]["tp"],
            },
            "effect_checks": checks, "effect_pass": all(checks.values()),
            "comparisons": comparisons,
            "historical_R29_loss_diagnostics": {
                name: _historical_loss_diagnostic(main, aligned[r29_ple_ids[seed]], aligned[name], d_cells)
                for name in references},
        }
    mechanism_pass = all(item["mechanism_pass"] for item in seeds.values())
    effect_pass = all(item["effect_pass"] for item in seeds.values())
    return {
        "status": "already_opened_development_comparison",
        "implementation_contract_pass": implementation_contract_pass,
        "all_seed_mechanism_pass": mechanism_pass, "all_seed_effect_pass": effect_pass,
        "adoption_pass": bool(implementation_contract_pass and mechanism_pass and effect_pass),
        "adoption_formula": "implementation_contract_pass AND all_seed_mechanism_pass AND all_seed_effect_pass",
        "automatic_deployment_adoption": False,
        "expected_cells": {role: sorted(cells) for role, cells in roles.items()},
        "models": model_metrics, "seeds": seeds,
        "historical_report_only": sorted(set(decision_frames) - set(required)),
        "identity_loss_interpretation": "Nondecreasing TP counts do not imply retention of every old TP identity",
        "mechanism_interpretation": "D pooled ranking improves with nonregressed detection; TCP-only ranking is separate",
        "failure_action": "retain negative result; no threshold/loss/weight/capacity/seed changes or promotion of control",
    }


__all__ = ["calibrate", "compare", "evaluate", "evaluate_decisions", "legacy_rule", "ranking_report"]
