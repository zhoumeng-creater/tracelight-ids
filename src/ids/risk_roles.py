"""Deterministic whole-group F/R/C allocation for development-only research.

This is protocol plumbing, not a new learning algorithm. See
docs/ALGORITHM_V3_DESIGN_20261103.md section 3 and RESEARCH_GROUNDING.md.
No model scores, validation labels or attack-family objectives choose the split.
All rows are assigned once; fractions are soft targets and no seed is retried.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real

from .risk_data import role_identity, validate_risk_roles


_ROLES = ("fit", "reference", "constraint")
_REQUIRED_LABELS = ((0, 1), (0,), (0, 1))
_ALGORITHM = "size_desc_seeded_sha256_greedy_binary_support_v1"


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric fraction")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _support_available(counts, vector, candidate, remaining):
    # Only three roles/two labels: at most 2**3 ways to fill their outstanding
    # requirements using normal-only, attack-only, or mixed future groups.
    # Counting normal and attack groups separately would double-book a mixed
    # group when different roles need its two labels.
    options = []
    for index, required in enumerate(_REQUIRED_LABELS):
        missing = tuple(label for label in required if counts[index][label]
                        + (vector[label] if index == candidate else 0) == 0)
        if not missing:
            options.append(((0, 0, 0),))
        elif len(missing) == 2:
            options.append(((1, 1, 0), (0, 0, 1)))
        elif missing == (0,):
            options.append(((1, 0, 0), (0, 0, 1)))
        else:
            options.append(((0, 1, 0), (0, 0, 1)))

    def complete(index, available):
        if index == len(options):
            return True
        for usage in options[index]:
            if all(need <= have for need, have in zip(usage, available)):
                if complete(index + 1, tuple(have - need for need, have in zip(usage, available))):
                    return True
        return False

    return complete(0, remaining)


def make_risk_roles(train, validation, *, seed, reference_fraction=.1, constraint_fraction=.2):
    """Return ({fit, reference, constraint}, JSON-safe allocation manifest).

    Inputs are already verified prepared frames (or a declared whole-group
    pilot cohort). Group priority is group_id, group_hash, then feature_hash;
    risk_data rejects inconsistent feature/session-to-union relations.

    Groups are processed largest-first; equal sizes are ordered by SHA256 of
    [algorithm, seed, group], then group identity. Each step chooses the role
    minimizing the increase in squared row-count AND binary-label-count error,
    divided by max(target_count, 1). Fixed role order breaks equal costs. A
    constant-size support guard checks whether remaining normal-only,
    attack-only and mixed groups can fill the outstanding requirements:
    normal in all roles, attack in fit/constraint. No repair,
    random resampling or alternative seed search follows a failure.

    This fixed heuristic does not optimize a global row/family balance. The
    guard preserves binary support, not family coverage or exact fractions.
    Every original row is retained in one role;
    returned frames preserve original row order and reset only their index.
    Constraint is training development, not an independent risk certificate.
    No files are read/written and no models are trained here.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    ref_fraction = _fraction(reference_fraction, "reference_fraction")
    con_fraction = _fraction(constraint_fraction, "constraint_fraction")
    if ref_fraction + con_fraction >= 1:
        raise ValueError("reference_fraction + constraint_fraction must be less than one")
    fractions = (1 - ref_fraction - con_fraction, ref_fraction, con_fraction)
    identity = role_identity(train)
    group_column = next(key for key in ("group_id", "group_hash", "feature_hash") if key in identity)
    # Validate the supplied development identity before allocating anything.
    # Its labels are deliberately absent from the allocation objective below.
    role_identity(validation, group_column=group_column)
    table = (identity.groupby([group_column, "label"], sort=True).size().unstack(fill_value=0)
             .reindex(columns=[0, 1], fill_value=0))
    groups = []
    for group, values in table.iterrows():
        vector = (int(values[0]), int(values[1]))
        groups.append((group, sum(vector), vector, _hash([_ALGORITHM, seed, group])))
    groups.sort(key=lambda item: (-item[1], item[3], item[0]))
    label_groups = [sum(vector[label] > 0 for _, _, vector, _ in groups) for label in (0, 1)]
    required_groups = [sum(label in required for required in _REQUIRED_LABELS) for label in (0, 1)]
    if len(groups) < 3 or any(label_groups[label] < required_groups[label] for label in (0, 1)):
        raise ValueError("Whole-group role support is impossible: require >=3 normal-bearing groups and "
                         f">=2 attack-bearing groups; found groups={len(groups)}, normal={label_groups[0]}, "
                         f"attack={label_groups[1]}, largest_group_rows={groups[0][1]}; no rows split/removed")
    total_labels = [int((identity.label == label).sum()) for label in (0, 1)]
    target_rows = [fraction * len(identity) for fraction in fractions]
    target_labels = [[fraction * total for total in total_labels] for fraction in fractions]
    actual_rows = [0, 0, 0]
    actual_labels = [[0, 0] for _ in _ROLES]
    remaining = [sum(vector[0] > 0 and vector[1] == 0 for _, _, vector, _ in groups),
                 sum(vector[1] > 0 and vector[0] == 0 for _, _, vector, _ in groups),
                 sum(vector[0] > 0 and vector[1] > 0 for _, _, vector, _ in groups)]
    assignment = {}
    order_digest = hashlib.sha256()
    for group, size, vector, order_key in groups:
        order_digest.update(json.dumps([group, size, list(vector), order_key],
                                       separators=(",", ":")).encode("utf-8") + b"\n")
        category = 2 if all(vector) else (0 if vector[0] else 1)
        remaining[category] -= 1
        candidates = []
        for index in range(len(_ROLES)):
            if not _support_available(actual_labels, vector, index, remaining):
                continue
            cost = (2 * (actual_rows[index] - target_rows[index]) * size + size * size) / max(
                target_rows[index], 1.)
            cost += sum((2 * (actual_labels[index][label] - target_labels[index][label]) * vector[label]
                         + vector[label] * vector[label]) / max(target_labels[index][label], 1.)
                        for label in (0, 1))
            if not math.isfinite(cost):
                raise ValueError("Role allocation objective is non-finite")
            candidates.append((cost, index))
        if not candidates:
            raise ValueError("Fixed whole-group allocation cannot retain required binary support; "
                             f"seed={seed}, group={group}, size={size}; no retry, repair or row removal")
        _, chosen = min(candidates)
        assignment[group] = _ROLES[chosen]
        actual_rows[chosen] += size
        for label in (0, 1):
            actual_labels[chosen][label] += vector[label]
    for index, required in enumerate(_REQUIRED_LABELS):
        if any(actual_labels[index][label] == 0 for label in required):
            raise ValueError(f"Fixed allocation left {_ROLES[index]} without required labels; no retry or repair")
    labels = identity[group_column].map(assignment)
    roles = {}
    for name in _ROLES:
        frame = train.loc[labels.eq(name).to_numpy()].reset_index(drop=True).copy(deep=True)
        frame.attrs = copy.deepcopy(train.attrs)
        roles[name] = frame
    manifest = validate_risk_roles(train, *(roles[name] for name in _ROLES), validation,
                                   group_column=group_column)
    manifest["allocation"] = {
        "algorithm": _ALGORITHM, "seed": seed,
        "group_order": "descending row count, seeded SHA256, lexical group identity",
        "role_tie_order": list(_ROLES),
        "objective": "sum_roles ((rows-target_rows)^2/max(target_rows,1) + "
                     "sum_binary_labels (count-target_count)^2/max(target_count,1))",
        "support_guard": "exact binary-support completion using remaining normal-only/attack-only/mixed "
                         "groups; at most eight role-completion combinations",
        "required_labels": {name: list(required) for name, required in zip(_ROLES, _REQUIRED_LABELS)},
        "requested_fractions": dict(zip(_ROLES, fractions)),
        "target_rows": dict(zip(_ROLES, target_rows)),
        "actual_rows": dict(zip(_ROLES, actual_rows)),
        "actual_fractions": {name: count / len(identity) for name, count in zip(_ROLES, actual_rows)},
        "row_delta_from_target": {name: count - target for name, count, target
                                  in zip(_ROLES, actual_rows, target_rows)},
        "fraction_targets_are_soft": True, "binary_label_stratification": True,
        "family_stratification": False, "family_coverage_guaranteed": False,
        "binary_support_feasibility_preserved": True, "global_row_balance_optimality_guaranteed": False,
        "n_groups": len(groups), "rows_removed": 0, "groups_split": 0, "retries": 0,
        "group_order_sha256": order_digest.hexdigest(),
        "group_assignment_sha256": _hash(sorted(assignment.items())),
        "largest_groups_summary_limit": 10,
        "largest_groups": [{"group": group, "n_rows": size,
                            "labels": {"0": vector[0], "1": vector[1]}, "role": assignment[group]}
                           for group, size, vector, _ in groups[:10]],
        "n_groups_larger_than_target_rows": {name: sum(size > target for _, size, _, _ in groups)
                                           for name, target in zip(_ROLES, target_rows)},
    }
    return roles, manifest
