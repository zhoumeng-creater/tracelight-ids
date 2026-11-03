"""Validate explicit training roles in memory; never split, load or save data.

``constraint`` participates in training development, not certification. Hashes
bind the supplied prepared identities; this module does not recompute feature
hashes from network features or establish statistical session independence.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import pandas as pd


_TRAINING_ROLES = ("fit", "reference", "constraint")
_BASE_COLUMNS = ("sample_id", "feature_hash", "label", "attack_cat")
_GROUP_COLUMNS = ("group_id", "group_hash", "feature_hash")
_OPTIONAL_IDENTITIES = ("group_id", "group_hash", "session_hash")


def _group_column(frame, requested=None):
    if requested is not None and requested not in _GROUP_COLUMNS:
        raise ValueError("group_column must be group_id, group_hash or feature_hash")
    preferred = next((name for name in _GROUP_COLUMNS if name in frame), None)
    if preferred is None:
        raise ValueError("A feature/group identity column is required")
    if requested is not None and requested != preferred:
        raise ValueError(f"group_column cannot weaken the preferred {preferred} boundary")
    schema = frame.attrs.get("feature_schema", {})
    schema_name = schema.get("name", "") if isinstance(schema, dict) else ""
    cic = ("session_hash" in frame or frame.attrs.get("group_column") in {"group_id", "group_hash"}
           or str(frame.attrs.get("schema_version", "")).startswith("cicids2017-")
           or str(schema_name).startswith("cicids2017-"))
    if cic and preferred == "feature_hash":
        raise ValueError("CIC/session data requires its prepared joint group_id or group_hash")
    return preferred


def _identity(frame, name, group_column, columns=None):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{name}: expected a DataFrame with unique column names")
    if frame.empty:
        raise ValueError(f"{name}: an explicit role cannot be empty")
    optional = [key for key in _OPTIONAL_IDENTITIES if key in frame]
    expected = list(_BASE_COLUMNS) + optional
    if columns is not None and expected != columns:
        raise ValueError(f"{name}: identity columns differ from original_train")
    if not set(expected).issubset(frame) or group_column not in frame:
        raise ValueError(f"{name}: missing required identity columns")
    identity = frame.loc[:, expected].reset_index(drop=True).copy()
    for key in expected:
        if key == "label":
            continue
        if not identity[key].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name}: {key} must contain nonempty string identities")
    labels = identity["label"]
    if (pd.api.types.is_bool_dtype(labels.dtype)
            or not labels.isin([0, 1]).all()
            or (labels.dtype == object and labels.map(lambda value: isinstance(value, bool)).any())):
        raise ValueError(f"{name}: label must contain binary numeric values, not booleans")
    identity["label"] = labels.astype("int64")
    if identity["sample_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate sample_id")
    if {"group_id", "group_hash"}.issubset(identity) and not identity.group_id.equals(identity.group_hash):
        raise ValueError(f"{name}: group_id and group_hash disagree")
    for relation in ("feature_hash", "session_hash"):
        if relation in identity and relation != group_column:
            if identity.groupby(relation, sort=False)[group_column].nunique().gt(1).any():
                raise ValueError(f"{name}: {relation} maps to multiple joint groups")
    return identity


def role_identity(frame, *, group_column=None):
    """Return a validated copy of identity/label/family columns, in row order.

    Prefer group_id, then group_hash; an explicit override cannot discard an
    available stronger boundary. Marked CIC/session data cannot use only hashes
    of exact feature rows. Feature values themselves are outside this contract.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("role: expected a DataFrame")
    group = _group_column(frame, group_column)
    return _identity(frame, "role", group)


def _digest(identity, *, ordered):
    # Stream records instead of materializing a million-row list of dictionaries.
    values = identity if ordered else identity.sort_values("sample_id", kind="stable")
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema_version": 1, "columns": list(values.columns)},
                             sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    for row in values.itertuples(index=False, name=None):
        digest.update(json.dumps(row, ensure_ascii=True, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _support(identity, group_column, original_families, original_attack_families):
    families = identity.attack_cat.value_counts().to_dict()
    group_sizes = identity.groupby(group_column, sort=False).size()
    present_attacks = set(identity.loc[identity.label == 1, "attack_cat"])
    return {
        "n_rows": len(identity), "n_groups": len(group_sizes),
        "n_feature_groups": int(identity.feature_hash.nunique()),
        "largest_group_rows": int(group_sizes.max()),
        "labels": {str(label): int((identity.label == label).sum()) for label in (0, 1)},
        "families": {family: int(families.get(family, 0))
                     for family in sorted(original_families | set(families))},
        "missing_families": sorted(original_families - set(families)),
        "missing_attack_families": sorted(original_attack_families - present_attacks),
        "families_not_in_original_train": sorted(set(families) - original_families),
        "identity_sha256": _digest(identity, ordered=False),
        "identity_order_sha256": _digest(identity, ordered=True),
    }


def validate_risk_roles(original_train, fit, reference, constraint, validation, *, group_column=None):
    """Verify an explicit, exhaustive original_train -> three-role partition.

    All five frames must be nonempty and have the same identity columns. No
    class/family quota is imposed: a normal-only reference is legal and missing
    families are reported. Roles may reorder rows, but may not alter their
    original identities, binary labels or attack categories. Validation must be
    disjoint from original_train by every available relation, not just row IDs.
    This function neither chooses a split nor reads/writes any file.
    """
    if not isinstance(original_train, pd.DataFrame):
        raise ValueError("original_train: expected a DataFrame")
    group = _group_column(original_train, group_column)
    original = _identity(original_train, "original_train", group)
    columns = list(original.columns)
    supplied = {"fit": fit, "reference": reference, "constraint": constraint, "validation": validation}
    identities = {name: _identity(frame, name, group, columns) for name, frame in supplied.items()}
    # Check all retained aliases/session relations too, even when group_id wins.
    relations = list(dict.fromkeys(["sample_id", "feature_hash", group,
                                   *[key for key in _OPTIONAL_IDENTITIES if key in original]]))
    training = {name: identities[name] for name in _TRAINING_ROLES}
    for (left_name, left), (right_name, right) in itertools.combinations(training.items(), 2):
        for key in relations:
            if set(left[key]) & set(right[key]):
                raise ValueError(f"{left_name}/{right_name}: overlapping {key}")
    for key in relations:
        if set(original[key]) & set(identities["validation"][key]):
            raise ValueError(f"original_train/validation: overlapping {key}")
    original_ids = set(original.sample_id)
    assigned_ids = set().union(*(set(identity.sample_id) for identity in training.values()))
    if assigned_ids != original_ids:
        raise ValueError("Training roles must exhaust original_train without missing or foreign sample_id")
    indexed = original.set_index("sample_id")
    for name, identity in training.items():
        supplied_indexed = identity.set_index("sample_id")
        if not supplied_indexed.equals(indexed.loc[supplied_indexed.index]):
            raise ValueError(f"{name}: original labels, family or identity were changed")
    families = set(original.attack_cat)
    attack_families = set(original.loc[original.label == 1, "attack_cat"])
    support = {name: _support(identity, group, families, attack_families)
               for name, identity in {"original_train": original, **identities}.items()}
    binding = {name: support[name]["identity_sha256"] for name in _TRAINING_ROLES}
    assignment_hash = hashlib.sha256(json.dumps(binding, sort_keys=True,
                                              separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1, "purpose": "risk_v3_explicit_role_validation",
        "scope": "development_only_not_independent_confirmation",
        "training_roles": list(_TRAINING_ROLES),
        "constraint_scope": "training_development_not_certificate",
        "reference_scope": "training_reference_not_certificate",
        "group_column": group, "identity_columns": columns,
        "group_semantics": "prepared_joint_groups" if group != "feature_hash" else "exact_feature_groups_only",
        "statistical_independence_verified": False,
        "checks": {"exhaustive": True, "mutually_disjoint": True,
                   "original_identity_preserved": True, "validation_isolated": True},
        "roles": support, "role_assignment_sha256": assignment_hash,
    }
