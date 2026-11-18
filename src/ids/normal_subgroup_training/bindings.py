"""Bind a zero-update preflight to the three R30 models and F/V only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ids.model_training_redesign.sampling import canonical_frame, group_indices


REPO = Path(__file__).resolve().parents[3]
DEFAULT_BINDINGS = REPO / "configs/normal-subgroup-training-20261118/input-bindings.json"
SEEDS = (17, 29, 43)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound_inputs(archive, *, bindings_path=DEFAULT_BINDINGS):
    """No C/E data, cached scores, or model deserialization is needed here."""
    archive = Path(archive).resolve()
    bindings = json.loads(Path(bindings_path).read_text(encoding="utf-8"))
    expected_ids = {f"protocol_blind_s{s}" for s in SEEDS}
    if set(bindings["models"]) != expected_ids:
        raise ValueError("Expected exactly the three registered R30 blind models")
    required = ["freeze.json", "inputs/F.parquet", "inputs/V.parquet"]
    required.extend(bindings["models"][key]["path"] for key in sorted(expected_ids))
    checked = {}
    for relative in required:
        path = (archive / relative).resolve()
        if not path.is_relative_to(archive):
            raise ValueError("Bound file must stay inside archive")
        item = bindings["files"][relative]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Bound artifact hash mismatch: {relative}")
        checked[relative] = item["sha256"]
    freeze = json.loads((archive / "freeze.json").read_text(encoding="utf-8"))
    for key in expected_ids:
        item, frozen = bindings["models"][key], freeze["artifacts"][key]
        if item["path"] != frozen["path"] or item["sha256"] != frozen["sha256"]:
            raise ValueError(f"Original freeze differs: {key}")
    frames = {role: canonical_frame(pd.read_parquet(archive / f"inputs/{role}.parquet"))
              for role in ("F", "V")}
    expected = {"F": (19904, 14691, 5213), "V": (21450, 18645, 2805)}
    for role, frame in frames.items():
        counts = (len(frame), int(frame.label.eq(0).sum()), int(frame.label.eq(1).sum()))
        if counts != expected[role] or not frame.profile.isin(["A", "B"]).all():
            raise ValueError(f"Unexpected {role} identity/label contract")
    groups = group_indices(frames["F"])
    if {key: len(value) for key, value in groups.items()} != dict(A0=8843, A1=5123, B0=5848, B1=90):
        raise ValueError("F source/class counts differ")
    f_ids = set(map(tuple, frames["F"][["source_id", "row_id"]].to_numpy()))
    if f_ids.intersection(map(tuple, frames["V"][["source_id", "row_id"]].to_numpy())):
        raise ValueError("F/V identities overlap")
    return frames, bindings, checked
