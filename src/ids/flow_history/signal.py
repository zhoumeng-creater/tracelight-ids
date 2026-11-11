"""Label-stratified summaries of the four registered measurements; no fitted score."""

import numpy as np
import pandas as pd

from .evaluation import syn_cohort
from .features import HISTORY_COLUMNS


def _distribution(values):
    if not len(values):
        return {"n": 0, "n_zero": 0, "n_unique": 0, "min_q25_q50_q75_max": None}
    return {
        "n": len(values),
        "n_zero": int(np.count_nonzero(values == 0)),
        "n_unique": int(np.unique(values).size),
        "min_q25_q50_q75_max": np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
    }


def _ordered_pairs(scans, normals):
    normal = np.sort(normals)
    below = np.searchsorted(normal, scans, side="left")
    at_or_below = np.searchsorted(normal, scans, side="right")
    pairs = len(scans) * len(normal)
    wins = int(below.sum())
    tied = int((at_or_below - below).sum())
    return {
        "pairs": pairs,
        "scan_above": wins,
        "tied": tied,
        "scan_below": pairs - wins - tied,
        "auc_higher_means_scanning": (wins + 0.5 * tied) / pairs if pairs else None,
    }


def _mixed_groups(selected, columns):
    grouped = selected.groupby(columns, sort=False, dropna=False, observed=True)["label"].agg(["sum", "size"])
    scans = grouped["sum"].astype(np.int64)
    normals = grouped["size"] - scans
    mixed = (scans > 0) & (normals > 0)
    counts = {
        "distinct_inputs": len(grouped),
        "mixed_inputs": int(mixed.sum()),
        "scans_in_mixed_inputs": int(scans[mixed].sum()),
        "normals_in_mixed_inputs": int(normals[mixed].sum()),
        "cross_label_pairs_with_identical_input": int((scans[mixed] * normals[mixed]).sum()),
        "minimum_errors_for_deterministic_per_row_decision": int(np.minimum(scans, normals).sum()),
    }
    return counts, grouped.index[mixed]


def diagnose(frame: pd.DataFrame) -> dict:
    """Audit one registered role independently, retaining the predeclared measurement directions."""
    if not frame.label.isin([0, 1]).all() or (frame.attack_family.eq("Scanning") & frame.label.ne(1)).any():
        raise ValueError("invalid binary/Scanning label in history signal diagnostic")
    if not np.isfinite(frame[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("nonfinite history signal")
    scan = frame.attack_family.eq("Scanning")
    normal = frame.label.eq(0)
    pattern = syn_cohort(frame)
    selected = frame.loc[pattern & (normal | scan)]
    pos = selected.label.eq(1)
    empty = frame[HISTORY_COLUMNS[0]].eq(0)
    variables = {}
    for name in HISTORY_COLUMNS:
        scans = selected.loc[pos, name].to_numpy(dtype=np.float64)
        normals = selected.loc[~pos, name].to_numpy(dtype=np.float64)
        variables[name] = {
            **_ordered_pairs(scans, normals),
            "scan_distribution": _distribution(scans),
            "normal_distribution": _distribution(normals),
        }
    # syn_cohort fixes all 47 other raw columns exactly; source port is the
    # remaining coordinate. Equality uses the saved numeric values, no tolerance.
    raw, mixed_ports = _mixed_groups(selected, ["L4_SRC_PORT"])
    augmented, mixed_children = _mixed_groups(selected, ["L4_SRC_PORT", *HISTORY_COLUMNS])
    unresolved = mixed_children.get_level_values("L4_SRC_PORT").nunique(dropna=False)
    if (
        unresolved > len(mixed_ports)
        or augmented["minimum_errors_for_deterministic_per_row_decision"]
        > raw["minimum_errors_for_deterministic_per_row_decision"]
    ):
        raise RuntimeError("adding coordinates cannot merge distinct raw inputs")
    return {
        "scope": "one role, fixed round-16 47-field signature, Scanning and normal only; "
        "positive direction for all four variables, ties count one-half; no fitted weights or threshold",
        "total_role_rows": len(frame),
        "role_empty_history_rows": int(empty.sum()),
        "syn47_rows": int(pattern.sum()),
        "syn47_scans": int(pos.sum()),
        "syn47_normals": int((~pos).sum()),
        "syn47_other_attack_rows_excluded_from_pairing": int(pattern.sum()) - len(selected),
        "syn47_scans_with_nonempty_history": int((pattern & scan & ~empty).sum()),
        "syn47_normals_with_nonempty_history": int((pattern & normal & ~empty).sum()),
        "fixed_direction_variables": variables,
        "identical_input_scan_normal_collisions": {
            "48_fields": raw,
            "52_fields": augmented,
            "formerly_mixed_48_patterns_with_any_mixed_52_child": int(unresolved),
            "formerly_mixed_48_patterns_with_no_mixed_52_child": len(mixed_ports) - int(unresolved),
            "note": "An unambiguous observed input (especially a singleton) does not establish a predictive rule. "
            "The finite-sample error bound uses min(scan,normal) per identical input within this slice.",
        },
    }
