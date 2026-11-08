"""Read registered P/C/E inputs and split P without fitting a preprocessing model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from ids.common import canonical_hash, sha256_file
from ids.context_cache.csv_workflow import ROLES, _feature_signatures, audit_roles
from ids.context_cache.preprocessing import CSVPreprocessor, CSVSchema, frame_features, labels, read_csv


def _registered_path(base: Path, item: dict, name: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
        raise ValueError(f"registered {name} requires a nonempty path")
    path = Path(item["path"])
    return (path if path.is_absolute() else base / path).resolve()


def _registered_hash(item: dict, name: str) -> str:
    value = item.get("sha256")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"registered {name} requires a SHA-256 digest")
    return value.lower()


def load_inputs(config_path, *, load_preprocessor=False) -> dict:
    """Validate registered CSVs, without loading model/checkpoint artifacts.

    This accepts the existing label configuration or a batch configuration
    containing the same ``schema`` and ``files`` fields. It does not invoke an
    old experiment's code freeze or require its arm names. Other registered
    artifact paths are resolved for the caller but are not loaded or checked.
    An explicitly requested frozen preprocessor is hash-checked on load and
    belongs only to the final full-P comparison, never the internal T/V split.
    """
    if not isinstance(load_preprocessor, bool):
        raise ValueError("load_preprocessor must be boolean")
    config_path = Path(config_path).resolve()
    spec = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(spec, dict):
        raise ValueError("batch configuration must be a JSON object")
    schema = CSVSchema.from_dict(spec.get("schema"))
    files = spec.get("files")
    if not isinstance(files, dict) or set(files) != set(ROLES):
        raise ValueError("registered files must contain support, calibration and evaluation")
    paths, frames = {}, {}
    for role in ROLES:
        item = files[role]
        path = _registered_path(config_path.parent, item, role)
        expected = _registered_hash(item, role)
        if type(item.get("rows")) is not int or item["rows"] < 1:
            raise ValueError(f"registered {role} requires a positive row count")
        if sha256_file(path) != expected:
            raise ValueError(f"registered SHA-256 mismatch for {role}")
        frame = read_csv(path, schema)
        if len(frame) != item["rows"]:
            raise ValueError(f"registered row count differs for {role}")
        paths[role], frames[role] = path, frame
    audit = audit_roles(frames, schema, exact_feature_policy="report")
    artifacts = spec.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")
    for name, item in artifacts.items():
        if name in paths or name == "checkpoint":
            raise ValueError(f"artifact name conflicts with a reserved role: {name}")
        paths[name] = _registered_path(config_path.parent, item, name)
    if "checkpoint" in spec:
        paths["checkpoint"] = _registered_path(config_path.parent, spec["checkpoint"], "checkpoint")
    preprocessor = None
    if load_preprocessor:
        if "preprocessor" not in artifacts:
            raise ValueError("no frozen preprocessor is registered")
        preprocessor = CSVPreprocessor.load(
            paths["preprocessor"], expected_sha256=_registered_hash(artifacts["preprocessor"], "preprocessor"))
        if preprocessor.schema.to_dict() != schema.to_dict():
            raise ValueError("frozen preprocessor schema differs from registered CSV schema")
    return {"spec": spec, "schema": schema, "frames": frames, "paths": paths,
            "audit": audit, "preprocessor": preprocessor}


def split_support(frame, schema, seed, val_fraction=.25, *, min_class_rows=2):
    """Assign connected P components once by hash, returning positional indices.

    Communication groups and exact full-input duplicates both join rows. The
    union is transitive. Sorted sample IDs identify each component independently
    of frame order; labels and family names never affect assignment. No seed
    search or balancing follows assignment. Insufficient class support fails.
    """
    if not isinstance(schema, CSVSchema):
        schema = CSVSchema.from_dict(schema)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    if (isinstance(val_fraction, (bool, np.bool_)) or not isinstance(val_fraction, (int, float))
            or not np.isfinite(val_fraction) or not 0 < val_fraction < 1):
        raise ValueError("val_fraction must be strictly between zero and one")
    if type(min_class_rows) is not int or min_class_rows < 1:
        raise ValueError("min_class_rows must be a positive integer")
    if schema.id_column is None or schema.group_column is None:
        raise ValueError("support split requires explicit sample IDs and communication groups")
    features = frame_features(frame, schema)
    for name in (schema.id_column, schema.group_column):
        if name not in frame or frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"support split requires nonempty metadata: {name}")
    ids = frame[schema.id_column].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("support split requires unique sample IDs")
    groups = frame[schema.group_column].astype(str).tolist()
    signatures = _feature_signatures(features, schema)
    parent = list(range(len(frame)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for keys in (groups, signatures):
        first = {}
        for index, key in enumerate(keys):
            if key in first:
                union(first[key], index)
            else:
                first[key] = index
    components = {}
    for index in range(len(frame)):
        components.setdefault(find(index), []).append(index)
    validation = np.zeros(len(frame), dtype=bool)
    component_assignments = []
    cutoff = int(float(val_fraction) * 2**256)
    for indices in components.values():
        members = sorted(ids[index] for index in indices)
        component_id = canonical_hash(members)
        digest = hashlib.sha256(f"route-batch-support-v1:{seed}:{component_id}".encode("ascii")).hexdigest()
        is_validation = int(digest, 16) < cutoff
        validation[indices] = is_validation
        component_assignments.append({"component_sha256": component_id,
                                      "rows": len(indices),
                                      "role": "validation" if is_validation else "train"})
    train_indices, validation_indices = np.flatnonzero(~validation), np.flatnonzero(validation)
    truth = labels(frame, schema)
    counts = {}
    for role, indices in (("train", train_indices), ("validation", validation_indices)):
        counts[role] = {"rows": len(indices), "normal": int((truth[indices] == 0).sum()),
                        "attack": int((truth[indices] == 1).sum())}
        if min(counts[role]["normal"], counts[role]["attack"]) < min_class_rows:
            raise ValueError(f"fixed support split has insufficient class support in {role}: {counts[role]}")
    train_ids = [ids[index] for index in train_indices]
    validation_ids = [ids[index] for index in validation_indices]
    audit = {
        "format": "tracelight-route-support-split-v1", "seed": seed,
        "target_validation_fraction": float(val_fraction),
        "actual_validation_fraction": len(validation_indices) / len(frame),
        "min_class_rows": min_class_rows, "counts": counts,
        "component_count": len(components),
        "largest_component_rows": max(map(len, components.values())),
        "components": sorted(component_assignments, key=lambda item: item["component_sha256"]),
        "train_ids": train_ids, "validation_ids": validation_ids,
        "ordered_support_ids_sha256": canonical_hash(ids),
        "split_ids_sha256": canonical_hash({"train": sorted(train_ids), "validation": sorted(validation_ids)}),
        "group_overlap": len({groups[i] for i in train_indices} & {groups[i] for i in validation_indices}),
        "exact_feature_overlap": len({signatures[i] for i in train_indices}
                                     & {signatures[i] for i in validation_indices}),
        "assignment_uses_labels": False, "assignment_uses_families": False,
        "preprocessor_fits": 0,
    }
    return train_indices, validation_indices, audit
