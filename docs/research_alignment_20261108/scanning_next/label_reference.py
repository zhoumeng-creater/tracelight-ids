"""Pure reference for a support-label diagnostic; no model or data-file I/O.

The laboratory supplies frozen FP32 support inputs, binary labels, and support
families. Query families are never accepted. This is not a production adapter.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

SHAM_VERSION = "scanning-label-group-sham-v1"
SHAM_SEED = 20261108


def _binary(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("support labels must be a one-dimensional binary array")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("support must contain normal and attack rows")
    return labels.astype(np.int64, copy=True)


def semantic_support_labels(binary_labels, support_families):
    """Normal=0, Scanning=1, OtherAttack=2, preserving support row order."""
    binary = _binary(binary_labels)
    families = np.asarray(support_families)
    if families.shape != binary.shape:
        raise ValueError("support family and binary labels must align")
    if not np.array_equal(families == "Benign", binary == 0):
        raise ValueError("Benign and binary normal must agree")
    result = np.where(binary == 0, 0, np.where(families == "Scanning", 1, 2))
    if set(result.tolist()) != {0, 1, 2}:
        raise ValueError("all three semantic support classes must be present")
    return result.astype(np.int64)


def grouped_sham_support_labels(frozen_support_x, binary_labels, small_class_rows):
    """Count-matched sham split without new identical-input attack conflicts.

    Group attack rows by their complete little-endian FP32 input bytes. Sort
    groups by a single versioned SHA-256 key. Select a whole group iff it fits
    the remaining quota; fail if the exact quota is impossible for this pass.
    No retries, new seeds, C/E input, or family identities are accepted.
    """
    binary = _binary(binary_labels)
    x = np.asarray(frozen_support_x)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape != (len(binary), 48):
        raise ValueError("supply the frozen FP32 support matrix in original row order")
    if not np.isfinite(x).all():
        raise ValueError("support matrix must be finite")
    attack_count = int(binary.sum())
    if type(small_class_rows) is not int or not 0 < small_class_rows < attack_count:
        raise ValueError("small class quota must be between zero and all attacks")
    groups = {}
    for index in np.flatnonzero(binary):
        canonical = np.array(x[index], dtype="<f4", copy=True)
        canonical[canonical == 0] = 0  # Do not split arithmetically equal signed zeros.
        key = canonical.tobytes()
        groups.setdefault(key, []).append(int(index))
    prefix = json.dumps([SHAM_VERSION, SHAM_SEED], separators=(",", ":")).encode("ascii") + b"\0"
    ordered = sorted(groups, key=lambda key: (hashlib.sha256(prefix + key).digest(), key))
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
        raise ValueError("fixed grouped sham did not fill quota; no alternative seed or row split permitted")
    metadata = {
        "version": SHAM_VERSION,
        "seed": SHAM_SEED,
        "class_counts": np.bincount(result, minlength=3).tolist(),
        "attack_input_groups": len(groups),
        "attack_singleton_groups": sum(len(indices) == 1 for indices in groups.values()),
        "new_attack_subclass_conflicts": 0,
        "selection": "fixed hash order; accept whole group only when its size fits remaining quota",
        "grouping": "complete finite little-endian FP32 attack input bytes; signed zeros canonicalized",
    }
    return result, metadata


def attack_union_log_odds(logits):
    """Binary alarm score: log(sum attack exp(logit)) minus normal logit.

    No temperature, class reweighting, argmax, or scanning-only score. The binary
    branch is the exact prior subtraction operation, including its FP32 dtype.
    """
    z = np.asarray(logits)
    if z.ndim < 1 or z.shape[-1] not in (2, 3) or z.dtype.kind != "f":
        raise ValueError("expected floating logits with normal first and one/two attack columns")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite")
    if z.shape[-1] == 2:
        return z[..., 1] - z[..., 0]
    return np.logaddexp(z[..., 1], z[..., 2]) - z[..., 0]
