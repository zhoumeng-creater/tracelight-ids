"""Offline, hash-bound A/B supervision and unchanged historical evaluation rows.

This module never imports a model or downloads data. Native UID labels are
joined by the existing TQH parser; row_id retains the original physical order.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from ids.response_evidence import tqh
from ids.threshold_validation.observations import RAW_FEATURES

REPO = Path(__file__).resolve().parents[3]
IDENTITY = ["source_id", "row_id"]
ROLES = ("F", "V", "C", "E_B", "E_D")
BASE_COLUMNS = [*IDENTITY, "flow_end_ms", "family", "label", *RAW_FEATURES]
PREDICTORS = ["duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "ip_protocol"]
REFERENCE_COLUMNS = [*BASE_COLUMNS, "alert", "ready", "model_id", "threshold", "score"]


def _contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its registered root: {relative}")
    return path


def verify_files(root, required):
    if not required:
        raise ValueError("Historical inputs must have registered SHA-256 bindings")
    for relative, digest in required.items():
        path = _contained(root, relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen input missing or changed: {relative}")


def _profile(source):
    if not isinstance(source, str) or not source.startswith("TQH-"):
        raise ValueError("Expected a registered TQH source identity")
    return source.split("-", 1)[1].split("_", 1)[0]


def _frame(frame):
    missing = set(BASE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Input frame lacks {sorted(missing)}")
    result = frame.copy().reset_index(drop=True)
    if result.empty or result[IDENTITY].isna().any().any() or result.duplicated(IDENTITY).any():
        raise ValueError("Physical source/row identities must be nonempty and unique")
    row_id = result.row_id.to_numpy(dtype=float)
    if not np.isfinite(row_id).all() or np.any(row_id != np.floor(row_id)) or np.any(row_id < 1):
        raise ValueError("row_id must retain positive native physical row numbers")
    if not result.family.isin(["Benign", "ExternalC2"]).all():
        raise ValueError("Unresolved or non-C2 labels must not enter these roles")
    if not np.array_equal(result.label.to_numpy(), result.family.ne("Benign").astype(int).to_numpy()):
        raise ValueError("Family and binary label disagree")
    if not np.isfinite(result[["flow_end_ms", *RAW_FEATURES]].to_numpy(dtype=float)).all():
        raise ValueError("Frozen replay numeric inputs must be finite; missing values use -1")
    profiles = result.source_id.map(_profile)
    if "profile" in result and not result.profile.equals(profiles):
        raise ValueError("Profile metadata disagrees with source identity")
    result["profile"] = profiles
    return result


def align_frozen(current, frozen, *, name):
    """Reject changed identities, values or truth; row ordering may differ."""
    current, frozen = _frame(current), _frame(frozen)
    left = current[BASE_COLUMNS].sort_values(IDENTITY).reset_index(drop=True)
    right = frozen[BASE_COLUMNS].sort_values(IDENTITY).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        raise ValueError(f"Native inputs differ from frozen R29 {name}") from error


def _identity_hash(frame):
    ordered = frame.sort_values(IDENTITY)
    value = "".join(f"{source}\t{int(row)}\n" for source, row in ordered[IDENTITY].itertuples(index=False, name=None))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def protocol_label_counts(frame):
    return [{"ip_protocol": int(protocol), "label": int(label), "rows": int(len(group))}
            for (protocol, label), group in frame.groupby(["ip_protocol", "label"], sort=True)]


def validate_roles(roles, design):
    """Check complete cells, labels, identities and the actual chronological claim."""
    normalized, seen, bounds = {}, set(), {}
    for name in ROLES:
        frame = _frame(roles[name])
        expected = design["roles"][name]
        cells = {cell["source_id"]: cell for cell in expected["cells"]}
        if set(frame.source_id) != set(cells):
            raise ValueError(f"Role cells differ: {name}")
        if (len(frame), int(frame.label.eq(0).sum()), int(frame.label.eq(1).sum())) != (
                expected["rows"], expected["normal"], expected["attack"]):
            raise ValueError(f"Role counts differ: {name}")
        identities = set(map(tuple, frame[IDENTITY].to_numpy()))
        if identities & seen:
            raise ValueError(f"Role identities overlap: {name}")
        seen |= identities
        for source, group in frame.groupby("source_id", sort=True):
            spec = cells[source]
            if (int(group.label.eq(0).sum()), int(group.label.eq(1).sum())) != (spec["normal"], spec["attack"]):
                raise ValueError(f"Complete cell counts differ: {source}")
            if not group.profile.eq(spec["profile"]).all():
                raise ValueError(f"Profile changed: {source}")
            if not group.flow_end_ms.between(spec["first_flow_end_ms"], spec["last_flow_end_ms"]).all():
                raise ValueError(f"Flow clocks lie outside registered cell: {source}")
        bounds[name] = {"first": int(frame.flow_end_ms.min()), "last": int(frame.flow_end_ms.max())}
        normalized[name] = frame
    for profile in ("A", "B"):
        f = normalized["F"].loc[normalized["F"].profile.eq(profile)]
        v = normalized["V"].loc[normalized["V"].profile.eq(profile)]
        if f.empty or v.empty or f.flow_end_ms.max() >= v.flow_end_ms.min():
            raise ValueError(f"Within-profile F must precede V: {profile}")
    if max(bounds["F"]["last"], bounds["V"]["last"]) >= bounds["C"]["first"]:
        raise ValueError("All F/V must precede C")
    if bounds["C"]["last"] >= min(bounds["E_B"]["first"], bounds["E_D"]["first"]):
        raise ValueError("C must precede all E cells")
    # Deliberately do not require global F<V: the A and B captures have different dates.
    return normalized, bounds


def _read_raw_cell(archive, spec, expected):
    prefix = spec["profile"] + "/" + spec["unit"]
    observations, labels, receipt = tqh._cell(archive, prefix)
    for key, value in receipt.items():
        if key in spec and value != spec[key]:
            raise ValueError(f"TQH source binding changed: {spec['name']} / {key}")
    for key, value in expected["source_binding"].items():
        if key not in spec or spec[key] != value:
            raise ValueError(f"Design and TQH source binding disagree: {spec['name']} / {key}")
    frame = observations.select(["row_id", "flow_end_ms", *RAW_FEATURES]).to_pandas()
    truth = labels.select(["row_id", "family", "native_label"]).to_pandas()
    if not np.array_equal(frame.row_id.to_numpy(), truth.row_id.to_numpy()):
        raise ValueError("Native observation/label physical order differs")
    counts = dict(Counter(truth.native_label))
    if counts != expected["author_label_counts"]:
        raise ValueError(f"Author label counts differ: {spec['name']}")
    if not truth.native_label.isin(["benign", "benign_external", "malicious_c2"]).all():
        raise ValueError("Undecided/recon labels cannot be silently mapped to normal")
    frame["family"] = truth.family.to_numpy()
    frame["label"] = truth.native_label.eq("malicious_c2").astype("int64").to_numpy()
    frame["source_id"] = spec["name"]
    frame["profile"] = spec["profile"]
    # Keep the original UID-to-row_id mapping, not just a label count.
    native_rows = [json.loads(line) for line in archive.read(prefix + "/conn.log").splitlines()]
    uid = pd.DataFrame({"source_id": spec["name"], "row_id": frame.row_id, "uid": [row["uid"] for row in native_rows]})
    manifest = json.loads(archive.read(prefix + "/manifest.json"))
    details = {"source_id": spec["name"], **receipt, "author_label_counts": counts,
               "framework": manifest.get("framework"), "encryption_profile": manifest.get("encryption_profile"),
               "tooling": manifest.get("tooling", {}), "archive": spec["archive"]}
    return _frame(frame), uid, details


def load_reference_decisions(r29_root, files, evaluation):
    """Read already-made decisions, never recalculate a historical policy."""
    frames = []
    for name, relative in files.items():
        saved = pd.read_parquet(_contained(r29_root, relative))
        align_frozen(saved, evaluation, name=relative)
        if saved.alert.dtype.kind != "b" or saved.ready.dtype.kind != "b" or not saved.ready.all():
            raise ValueError(f"Historical alert/ready values differ: {name}")
        selected = [column for column in REFERENCE_COLUMNS if column in saved]
        saved = saved[selected].copy()
        saved["model_id"] = name
        for column in ("threshold", "score"):
            saved[column] = pd.to_numeric(saved[column], errors="raise").astype("float64") if column in saved else np.nan
        frames.append(saved)
    if not frames:
        raise ValueError("Strong historical decisions must be bound")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["model_id", *IDENTITY]).any():
        raise ValueError("Historical decisions duplicate model/physical identity")
    return result


def load_inputs(r29_root, r26_root, raw_dir, output_dir, design, registration):
    """Return roles, binding, calibration references without running a model.

    roles also contains ED, uid_mapping and reference_decisions as audit/evaluation
    artifacts. r26_root is optional: all required original scores and thresholds
    are already copied and hash-bound in R29. output_dir is retained for runner
    API symmetry; persist_inputs is the only writing operation.
    """
    del output_dir
    r29_root, raw_dir = Path(r29_root), Path(raw_dir)
    verify_files(r29_root, registration["required_r29_artifacts"])
    if r26_root is not None:
        verify_files(Path(r26_root), registration["required_r26_artifacts"])
    source_path = REPO / design["data_binding"]["source_metadata"]["path"]
    actual = hashlib.sha256(source_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if actual != design["data_binding"]["source_metadata"]["sha256"]:
        raise ValueError("TQH metadata differs from frozen design")
    specs = {source["name"]: source for source in read_json(source_path)["sources"]}
    raw_cells, uid_frames, sources, archives = {}, [], [], {}
    expected_cells = {cell["source_id"]: cell for role in design["roles"].values() for cell in role["cells"]}
    for source, expected in expected_cells.items():
        spec = specs[source]
        archive_path = _contained(raw_dir, spec["archive"])
        if spec["archive"] not in archives:
            if (not archive_path.is_file() or archive_path.stat().st_size != spec["archive_bytes"]
                    or sha256_file(archive_path) != spec["archive_sha256"]):
                raise ValueError(f"Offline source archive missing or changed: {spec['archive']}")
            archives[spec["archive"]] = {"bytes": archive_path.stat().st_size,
                                        "sha256": spec["archive_sha256"]}
        with ZipFile(archive_path) as archive:
            frame, uid, receipt = _read_raw_cell(archive, spec, expected)
        excluded = int(expected["excluded_attack"])
        if excluded:
            if frame.label.eq(1).sum() != excluded or expected["attack"] != 0:
                raise ValueError("Only registered C controls may be excluded")
            frame = frame.loc[frame.label.eq(0)].copy()
        raw_cells[source] = frame
        uid_frames.append(uid)
        sources.append(receipt)
    built = {name: pd.concat([raw_cells[cell["source_id"]] for cell in design["roles"][name]["cells"]],
                            ignore_index=True) for name in ROLES}
    frozen = {name: pd.read_parquet(r29_root / f"inputs/{name}.parquet") for name in ("F", "V", "P", "C", "E", "D", "ED")}
    for name in ("F", "V"):
        align_frozen(built[name].loc[built[name].profile.eq("B")], frozen[name], name=name)
    align_frozen(pd.concat([frozen["F"], frozen["V"]], ignore_index=True), frozen["P"], name="P=F+V")
    for name, old_name in (("C", "C"), ("E_B", "E"), ("E_D", "D")):
        align_frozen(built[name], frozen[old_name], name=old_name)
        # Preserve the original complete cached score columns and row order.
        built[name] = _frame(frozen[old_name])
    align_frozen(pd.concat([built["E_B"], built["E_D"]], ignore_index=True), frozen["ED"], name="ED")
    roles, bounds = validate_roles(built, design)
    roles["ED"] = _frame(frozen["ED"])
    roles["uid_mapping"] = pd.concat(uid_frames, ignore_index=True)
    if roles["uid_mapping"].duplicated(IDENTITY).any():
        raise ValueError("UID mapping duplicates a physical identity")
    roles["reference_decisions"] = load_reference_decisions(
        r29_root, registration["reference_decision_files"], roles["ED"])
    references = read_json(r29_root / "inputs/references.json")
    if r26_root is not None:
        for name in ("hgb", "mlp17", "mlp29", "mlp43"):
            if references[name] != read_json(Path(r26_root) / f"evidence/calibration/target-ports_zero8-{name}.json"):
                raise ValueError(f"R26 and R29 calibration references disagree: {name}")
    binding = {"r29_path": str(r29_root.resolve()), "raw_path": str(raw_dir.resolve()),
               "r29_artifacts": registration["required_r29_artifacts"], "raw_archives": archives,
               "sources": sources, "chronology": bounds, "global_F_before_V_required": False,
               "identity_sha256": {name: _identity_hash(roles[name]) for name in (*ROLES, "ED")},
               "protocol_label_counts": {name: protocol_label_counts(roles[name]) for name in ROLES},
               "F_group_counts": {f"{p}{int(y)}": int(len(g)) for (p, y), g in roles["F"].groupby(["profile", "label"])},
               "new_fits": 0, "model_forwards": 0, "downloads": 0}
    for name, expected in registration.get("protocol_label_counts", {}).items():
        if binding["protocol_label_counts"][name] != expected:
            raise ValueError(f"Registered protocol/label counts changed: {name}")
    return roles, binding, references


def persist_inputs(root, roles, binding, references):
    folder = Path(root) / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in (*ROLES, "ED", "uid_mapping", "reference_decisions"):
        path = folder / f"{name}.parquet"
        roles[name].to_parquet(path, index=False, compression="zstd")
        paths.append(path)
    write_json(folder / "references.json", references)
    write_json(folder / "input_audit.json", {"binding": binding,
        "roles": {name: {"rows": len(roles[name]), "normal": int(roles[name].label.eq(0).sum()),
                         "attack": int(roles[name].label.eq(1).sum()), "cells": sorted(roles[name].source_id.unique())}
                  for name in (*ROLES, "ED")}, "new_fits": 0, "model_forwards": 0,
        "new_blind_test": False, "calibration_controls_excluded": 72,
        "weight_contract": "No weight columns in F/V; four-group sampling only",
        "flow_clock": "flow end; native nondecreasing export order, not measured alert availability"})
    paths.extend([folder / "references.json", folder / "input_audit.json"])
    write_json(folder / "manifest.json", {path.name: sha256_file(path) for path in paths})
    return [*paths, folder / "manifest.json"]


def load_persisted_inputs(root):
    folder = Path(root) / "inputs"
    verify_files(folder, read_json(folder / "manifest.json"))
    roles = {name: pd.read_parquet(folder / f"{name}.parquet")
             for name in (*ROLES, "ED", "uid_mapping", "reference_decisions")}
    return roles, read_json(folder / "input_audit.json")["binding"], read_json(folder / "references.json")


def blind(frame):
    """Prediction gets the five numeric fields and protocol, never truth/identity."""
    return frame[PREDICTORS].copy()
