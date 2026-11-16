"""Hash-bound R26 inputs, natural row weights and whole-cell temporal roles."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from ids.support_diagnostic.training_diagnosis import load_frame
from ids.threshold_validation.observations import RAW_FEATURES
from .protocol import REPO

IDENTITY = ["source_id", "row_id"]


def verify_r26(root, registration):
    for rel, digest in registration["required_r26_artifacts"].items():
        path = root / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Frozen R26 input missing or changed: {path}")


def with_truth(frame):
    frame = frame.copy().reset_index(drop=True)
    if not frame.family.isin(["Benign", "ExternalC2"]).all():
        raise ValueError("Unexpected target family; do not reinterpret unresolved labels")
    truth = frame.family.ne("Benign").astype("int64")
    if "label" in frame and not np.array_equal(frame.label, truth):
        raise ValueError("Family/label disagreement")
    frame["label"] = truth
    if frame.duplicated(IDENTITY).any() or frame[IDENTITY].isna().any().any():
        raise ValueError("Invalid physical row identity")
    if not np.isfinite(frame[list(RAW_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite replay input")
    return frame


def split_roles(training, calibration, evaluation, design, registration, source_specs):
    target = with_truth(training.loc[training.domain.eq("target")])
    # The frozen R26 mixed frame contains source/class balancing weights.
    # Preserve them as audit metadata; R29 deliberately fits natural row mass.
    target["historical_weight"] = target["weight"]
    target["weight"] = 1.0
    roles = {name: target.loc[target.source_id.isin(design["roles"][name]["cells"])].copy()
             for name in ("F", "V")}
    roles["P"] = pd.concat([roles["F"], roles["V"]], ignore_index=True)
    if set(map(tuple, roles["P"][IDENTITY].to_numpy())) != set(map(tuple, target[IDENTITY].to_numpy())):
        raise ValueError("Target training rows are not exactly F union V")
    if not target.weight.eq(1).all():
        raise ValueError("Target training rows must retain natural unit weights")
    roles["C"] = with_truth(calibration)
    if roles["C"].label.any():
        raise ValueError("C must contain only the frozen normal rows")
    combined = with_truth(evaluation)
    if set(combined.source_id) != set(registration["evaluation_cells"]):
        raise ValueError("Evaluation must contain exactly the registered E and D cells")
    roles["E"] = combined.loc[combined.source_id.isin(design["roles"]["E"]["cells"])].copy()
    roles["D"] = combined.loc[~combined.source_id.isin(design["roles"]["E"]["cells"])].copy()
    for name, frame in roles.items():
        expected = design["roles"][name]
        if (int(frame.label.eq(0).sum()), int(frame.label.eq(1).sum())) != (
                expected["normal"], expected.get("attack", 0)):
            raise ValueError(f"Registered role counts changed: {name}")
        if name in ("F", "V", "C", "E") and set(frame.source_id) != set(expected["cells"]):
            raise ValueError(f"Registered role cells changed: {name}")
    seen = set()
    for name in ("F", "V", "C", "E", "D"):
        keys = set(map(tuple, roles[name][IDENTITY].to_numpy()))
        if seen & keys:
            raise ValueError(f"Physical identities overlap across roles: {name}")
        seen |= keys
    bounds = {}
    for name in ("F", "V", "C", "E"):
        specs = [source_specs[cell] for cell in design["roles"][name]["cells"]]
        bounds[name] = {"first": min(s["first_flow_end_ms"] for s in specs),
                        "last": max(s["last_flow_end_ms"] for s in specs)}
    if any(bounds[a]["last"] >= bounds[b]["first"] for a, b in zip(("F", "V", "C"), ("V", "C", "E"))):
        raise ValueError("Whole-cell F/V/C/E chronology is not strictly ordered")
    for name in ("C", "E", "D"):
        for cell, frame in roles[name].groupby("source_id"):
            spec = source_specs[cell]
            if not frame.flow_end_ms.between(spec["first_flow_end_ms"], spec["last_flow_end_ms"]).all():
                raise ValueError(f"Replay clocks outside registered source bounds: {cell}")
    roles["ED"] = combined
    return {name: frame.reset_index(drop=True) for name, frame in roles.items()}, bounds


def load_inputs(r26, recovered, raw_dir, root, registration, design):
    verify_r26(r26, registration)
    if recovered is None and not (r26 / "local/training/mixed.parquet").is_file() and raw_dir:
        recovered = root / "recovered/recovered-training.parquet"
        if not recovered.exists():
            recovered.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(REPO / "scripts/recover_r26_training_rows.py"),
                            "--r26-run", str(r26), "--raw-dir", str(raw_dir), "--output", str(recovered)], check=True)
    training, path = load_frame(r26, recovered)
    calibration = pd.read_parquet(r26 / "evidence/replay/calibration.parquet")
    evaluation = pd.concat([pd.read_parquet(r26 / f"evidence/replay/{cell}.parquet")
                            for cell in registration["evaluation_cells"]], ignore_index=True)
    roles, bounds = split_roles(training, calibration, evaluation, design, registration,
                               read_json(r26 / "protocol.json")["source_specs"])
    binding = {"training_numeric_sha256": sha256_file(path),
               "r26_artifacts": registration["required_r26_artifacts"], "chronology": bounds,
               "training_path": str(path), "r26_path": str(r26)}
    references = {name: read_json(r26 / f"evidence/calibration/target-ports_zero8-{name}.json")
                  for name in ("hgb", "mlp17", "mlp29", "mlp43")}
    references["legacy_hgb"] = read_json(REPO / "configs/support-diagnostic-20261115/legacy-hgb-calibration.json")
    return roles, binding, references


def persist_inputs(root, roles, binding, references):
    folder = root / "inputs"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in roles.items():
        path = folder / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        paths.append(path)
    write_json(folder / "references.json", references)
    write_json(folder / "input_audit.json", {"binding": binding,
        "roles": {name: {"rows": len(frame), "normal": int(frame.label.eq(0).sum()),
                         "attack": int(frame.label.eq(1).sum()), "cells": sorted(frame.source_id.unique())}
                  for name, frame in roles.items()},
        "new_model_fits": 0, "new_model_forwards": 0,
        "new_blind_test": False, "source_training_rows_used": 0,
        "calibration_attack_rows_excluded": 72, "flow_clock": "flow end, not independently measured log availability"})
    paths += [folder / "references.json", folder / "input_audit.json"]
    return paths


def blind(frame):
    """The prediction API receives only the eight raw features (ports are audit-only)."""
    return frame[list(RAW_FEATURES)].copy()
