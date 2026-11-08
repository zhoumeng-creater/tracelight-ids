"""Support-only label views for the fixed Scanning granularity diagnostic.

Inputs are the already-frozen FP32 matrix and support metadata. No query
features, labels, scores or alternate seeds enter these transformations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json

import numpy as np

ARMS = ("baseline", "semantic_family", "grouped_sham")
CLASS_NAMES = {
    "baseline": ("Normal", "Attack"),
    "semantic_family": ("Normal", "Scanning", "OtherAttack"),
    "grouped_sham": ("Normal", "AttackA", "AttackB"),
}
SHAM_VERSION = "scanning-label-group-sham-v1"
SHAM_SEED = 20261108


def _binary(labels):
    values = np.asarray(labels)
    if (values.ndim != 1 or values.dtype.kind not in "biuf"
            or not np.isin(values, [0, 1]).all() or set(values.tolist()) != {0, 1}):
        raise ValueError("support labels must be aligned binary values containing both classes")
    return values.astype(np.int64, copy=True)


def _matrix(values, rows=None):
    x = np.asarray(values)
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 48
            or (rows is not None and x.shape[0] != rows) or not np.isfinite(x).all()):
        raise ValueError("supply the aligned finite frozen FP32 support matrix with 48 columns")
    return x


def _row_bytes(values):
    canonical = np.array(values, dtype="<f4", copy=True)
    canonical[canonical == 0] = 0
    return canonical.tobytes()


def canonical_support_group_ids(frozen_support_x):
    """Canonical input SHA per support row; signed zero does not split groups."""
    x = _matrix(frozen_support_x)
    return np.asarray([hashlib.sha256(_row_bytes(row)).hexdigest() for row in x], dtype=object)


def semantic_support_labels(binary_labels, support_families):
    """Normal=0, Scanning=1, OtherAttack=2; preserve the original row order."""
    binary = _binary(binary_labels)
    families = np.asarray(support_families, dtype=object)
    if (families.shape != binary.shape
            or not all(isinstance(value, str) and value.strip() for value in families)):
        raise ValueError("support families must be aligned nonempty strings")
    if not np.array_equal(families == "Benign", binary == 0):
        raise ValueError("normalized Benign family and binary normal must agree")
    result = np.where(binary == 0, 0, np.where(families == "Scanning", 1, 2)).astype(np.int64)
    if set(result.tolist()) != {0, 1, 2}:
        raise ValueError("all three semantic support classes must be present")
    return result


def label_sha256(labels):
    """Ordered little-endian int64 labels, without any row reordering."""
    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError("label digest requires one-dimensional nonnegative integer labels")
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def grouped_sham_support_labels(frozen_support_x, binary_labels, small_class_rows):
    """One fixed group-hash pass, exact quota or failure; never split a group."""
    binary = _binary(binary_labels)
    x = _matrix(frozen_support_x, len(binary))
    if type(small_class_rows) is not int or not 0 < small_class_rows < int(binary.sum()):
        raise ValueError("small class quota must be an integer between zero and all attacks")
    groups = defaultdict(list)
    for index in np.flatnonzero(binary):
        groups[_row_bytes(x[index])].append(int(index))
    prefix = json.dumps([SHAM_VERSION, SHAM_SEED], separators=(",", ":")).encode("ascii") + b"\0"
    priorities = {key: hashlib.sha256(prefix + key).digest() for key in groups}
    ordered = sorted(groups, key=lambda key: (priorities[key], key))
    result = np.where(binary == 0, 0, 2).astype(np.int64)
    remaining = small_class_rows
    for key in ordered:
        indices = groups[key]
        if len(indices) <= remaining:
            result[indices] = 1
            remaining -= len(indices)
        if remaining == 0:
            break
    if remaining:
        raise ValueError("fixed grouped sham did not fill quota; alternate seeds or row splits are forbidden")
    records = [
        {"input_sha256": hashlib.sha256(key).hexdigest(), "rows": len(groups[key]),
         "priority_sha256": priorities[key].hex(), "sham_class": int(result[groups[key][0]])}
        for key in ordered
    ]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    audit = {
        "version": SHAM_VERSION, "seed": SHAM_SEED,
        "small_class_rows": small_class_rows,
        "class_counts": np.bincount(result, minlength=3).tolist(),
        "attack_input_groups": len(groups),
        "attack_singleton_groups": sum(len(indices) == 1 for indices in groups.values()),
        "new_attack_subclass_conflicts": 0,
        "selection": "fixed hash order; accept a whole group only if its size fits the remaining quota",
        "grouping": "complete finite little-endian FP32 attack input bytes; signed zeros canonicalized",
        "groups": records, "groups_sha256": digest,
    }
    return result, audit


def build_support_label_views(frozen_support_x, binary_labels, support_families, *, small_class_rows=None):
    """Build all three P label vectors and their reproducible input/label audit."""
    binary = _binary(binary_labels)
    x = _matrix(frozen_support_x, len(binary))
    semantic = semantic_support_labels(binary, support_families)
    semantic_count = int(np.count_nonzero(semantic == 1))
    if small_class_rows is None:
        small_class_rows = semantic_count
    if type(small_class_rows) is not int or small_class_rows != semantic_count:
        raise ValueError("sham quota must equal the number of semantic Scanning support rows")
    sham, sham_audit = grouped_sham_support_labels(x, binary, small_class_rows)
    views = {"baseline": binary, "semantic_family": semantic, "grouped_sham": sham}
    row_groups = canonical_support_group_ids(x)
    groups = defaultdict(list)
    for index, key in enumerate(row_groups):
        groups[key].append(index)
    conflicts = {}
    for arm, values in views.items():
        conflicts[arm] = {
            "all_label_conflict_groups": sum(np.unique(values[indices]).size > 1 for indices in groups.values()),
            "attack_subclass_conflict_groups": sum(
                {1, 2}.issubset(set(values[indices].tolist())) for indices in groups.values()
            ) if arm != "baseline" else 0,
        }
    if conflicts["grouped_sham"]["attack_subclass_conflict_groups"]:
        raise AssertionError("grouped sham split an identical attack input")
    cross = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cross, (semantic, sham), 1)
    families = np.asarray(support_families, dtype=object)
    audit = {
        "format": "tracelight-support-label-views-v1",
        "scope": "support only; no query metadata, labels or scores used",
        "rows": len(binary), "features": 48,
        "input_sha256": hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest(),
        "input_hash_encoding": "unchanged ordered little-endian float32 matrix",
        "row_input_group_sha256": row_groups.tolist(),
        "unique_input_groups": len(groups),
        "input_group_size_histogram": {str(k): v for k, v in sorted(Counter(map(len, groups.values())).items())},
        "class_names": {arm: list(CLASS_NAMES[arm]) for arm in ARMS},
        "class_counts": {arm: np.bincount(values, minlength=len(CLASS_NAMES[arm])).tolist()
                         for arm, values in views.items()},
        "label_sha256": {arm: label_sha256(values) for arm, values in views.items()},
        "label_hash_encoding": "ordered_little_endian_int64",
        "semantic_by_sham_class_counts": cross.tolist(),
        "family_by_sham_class_counts": {
            family: np.bincount(sham[families == family], minlength=3).tolist()
            for family in sorted(set(families.tolist()))
        },
        "input_label_conflicts": conflicts,
        "sham": sham_audit,
    }
    return views, audit


def attack_union_log_odds(logits):
    """Unchanged binary subtraction or stable sum-of-both-attacks log odds."""
    z = np.asarray(logits)
    if (z.ndim < 1 or z.shape[-1] not in (2, 3) or z.dtype.kind != "f"
            or not np.isfinite(z).all()):
        raise ValueError("expected finite floating logits with normal first and one/two attack columns")
    if z.shape[-1] == 2:
        return z[..., 1] - z[..., 0]
    return np.logaddexp(z[..., 1], z[..., 2]) - z[..., 0]
