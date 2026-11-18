"""F-only deterministic count bins, used during training and validation only."""
from __future__ import annotations

import numpy as np


FORMAT = "normal-subgroup-count-bins-v1"
FIELDS = ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
ROOTS = ("A", "B")
MIN_CHILD_ROWS = 256
MAX_DEPTH = 4


def _columns(frame):
    labels = np.asarray(frame["label"])
    profiles = np.asarray(frame["profile"])
    if labels.ndim != 1 or profiles.shape != labels.shape:
        raise ValueError("Labels and profiles must be aligned one-dimensional columns")
    if not np.isin(labels, [0, 1]).all() or not np.isin(profiles, ROOTS).all():
        raise ValueError("F/V require binary labels and known A/B profiles")
    values = np.column_stack([np.asarray(frame[c], dtype=np.float64) for c in FIELDS])
    if values.shape != (len(labels), 4):
        raise ValueError("Count columns must align with labels")
    if np.isinf(values).any() or ((values < 0) & (values != -1)).any():
        raise ValueError("Counts require nonnegative values, NaN, or -1")
    values[np.isnan(values)] = -1
    return values, labels, profiles


def _tree(values, indices, depth, key):
    node = {"key": key, "depth": depth, "n": int(len(indices))}
    if depth == MAX_DEPTH or len(indices) < 2 * MIN_CHILD_ROWS:
        return node
    for offset in range(4):
        feature = (depth + offset) % 4
        distinct, counts = np.unique(values[indices, feature], return_counts=True)
        left_n = np.cumsum(counts)[:-1]
        legal = np.flatnonzero((left_n >= MIN_CHILD_ROWS)
                               & (len(indices) - left_n >= MIN_CHILD_ROWS))
        if not len(legal):
            continue
        # np.unique sorts ascending, so first argmin implements the lower-cut tie rule.
        cut_index = legal[np.argmin(np.abs(2 * left_n[legal] - len(indices)))]
        cut = float(distinct[cut_index])
        left = values[indices, feature] <= cut
        node.update(feature=FIELDS[feature], cut_le=cut,
                    left=_tree(values, indices[left], depth + 1, key + "0"),
                    right=_tree(values, indices[~left], depth + 1, key + "1"))
        return node
    return node


def leaf_nodes(spec):
    """Return leaves in fixed A/B, left/right order."""
    if spec.get("format") != FORMAT:
        raise ValueError("Unsupported group specification")
    result = []

    def visit(node):
        if "left" in node:
            visit(node["left"])
            visit(node["right"])
        else:
            result.append(node)

    for profile in ROOTS:
        visit(spec["trees"][profile])
    return result


def fit_groups(frame, *, role="F"):
    """Fit the frozen 256-row, depth-4 tree from F normal rows only.

    frame may be a dataframe or a mapping of aligned arrays. Role and file/identity
    binding are separate contracts: the caller must bind the actual F artifact.
    """
    if role != "F":
        raise ValueError("Groups can be constructed only from role F")
    values, labels, profiles = _columns(frame)
    trees, patterns = {}, {}
    for profile in ROOTS:
        indices = np.flatnonzero((labels == 0) & (profiles == profile))
        if len(indices) < MIN_CHILD_ROWS:
            raise ValueError(f"F normal root {profile} has fewer than 256 rows")
        trees[profile] = _tree(values, indices, 0, profile + "_")
        patterns[profile] = np.unique((values[indices] == -1).astype(np.uint8), axis=0).tolist()
    result = {"format": FORMAT, "fit_role": "F", "fields": list(FIELDS),
              "min_child_rows": MIN_CHILD_ROWS, "max_depth": MAX_DEPTH,
              "trees": trees, "normal_missing_patterns": patterns}
    result["group_keys"] = [node["key"] for node in leaf_nodes(result)]
    return result


def map_groups(frame, spec, *, role="V", require_all=True):
    """Return aligned group strings; positives have the empty-string sentinel.

    Unknown profiles or missing patterns and absent frozen groups fail the F/V
    contract. Numeric values outside the original range follow the same cuts.
    """
    if role not in ("F", "V"):
        raise ValueError("Groups are only mapped for roles F and V, never deployment")
    leaves = leaf_nodes(spec)
    if (spec.get("fields") != list(FIELDS) or spec.get("min_child_rows") != MIN_CHILD_ROWS
            or spec.get("max_depth") != MAX_DEPTH):
        raise ValueError("Group specification differs from the frozen design")
    values, labels, profiles = _columns(frame)
    result = np.full(len(labels), "", dtype="<U6")
    for profile in ROOTS:
        indices = np.flatnonzero((labels == 0) & (profiles == profile))
        allowed = {tuple(x) for x in spec["normal_missing_patterns"][profile]}
        for i in indices:
            if tuple((values[i] == -1).astype(int)) not in allowed:
                raise ValueError(f"UNSUPPORTED_MISSING in {role} profile {profile}")
            node = spec["trees"][profile]
            while "left" in node:
                side = "left" if values[i, FIELDS.index(node["feature"])] <= node["cut_le"] else "right"
                node = node[side]
            result[i] = node["key"]
    expected = {node["key"] for node in leaves}
    present = set(result[labels == 0].tolist())
    if require_all and present != expected:
        raise ValueError(f"Frozen groups without {role} normal support: {sorted(expected - present)}")
    return result
