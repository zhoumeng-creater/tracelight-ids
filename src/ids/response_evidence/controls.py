"""Reuse published R22 control weights when the original lab checkout is present."""

import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from ids.route_batch.metrics import decisions, threshold
from ids.route_batch.protocol import REPO

FOLDER = REPO / "configs/response-evidence-20261112"


def previous_output(reg, runtime):
    ref = read_json(FOLDER / "r22-model-reference.json")
    inputs = {s["name"]: s["sha256"] for s in reg["sources"] if s["role"] in {"P", "C"}}
    if inputs != ref["inputs"] or runtime != ref["runtime"]:
        return None
    configured = os.environ.get("TRACELIGHT_R22_OUTPUT")
    base = Path(configured) if configured else REPO.parent / "tracelight-contact-r22/local/contact-r22"
    for entries in ref["artifacts"].values():
        for name, digest in entries.items():
            file = base / name
            if not file.is_file() or sha256_file(file) != digest:
                return None
    return str(base.resolve())


def reuse(root, base, group, arm):
    entries = read_json(FOLDER / "r22-model-reference.json")["artifacts"][group + "-" + arm]
    paths = []
    for name, digest in entries.items():
        source, target = Path(base) / name, root / name
        if sha256_file(source) != digest:
            raise ValueError("Published R22 control changed: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".copying")
        shutil.copyfile(source, tmp)
        tmp.replace(target)
        paths.append(target)
    return paths


def replay_calibration(root, reg):
    ref = read_json(FOLDER / "r22-model-reference.json")
    inputs = {s["name"]: s["sha256"] for s in reg["sources"] if s["role"] in {"P", "C"}}
    result = root / "evidence/r22-control-replay.json"
    if inputs != ref["inputs"]:
        write_json(result, {"status": "different_source_identities"})
        return [result]
    reference = FOLDER / "r22-normal-reference.parquet"
    if sha256_file(reference) != ref["calibration_reference_sha256"]:
        raise ValueError("Damaged R22 calibration reference")
    old = pd.read_parquet(reference)
    report = {}
    for group, rows in old.groupby("learner", sort=False):
        actual = pd.read_parquet(root / "evidence/calibration" / (group + ".parquet"))
        if not np.array_equal(rows[["capture", "row_id"]], actual[["capture", "row_id"]]):
            raise ValueError("Changed R22 normal calibration identities")
        for arm in ("raw", "contact"):
            a, b = rows[arm].to_numpy(), actual[arm].to_numpy()
            delta = np.abs(a - b)
            registered = threshold(a)
            report[group + "-" + arm] = {
                "rows": len(rows),
                "max_abs_score_delta": float(delta.max()),
                "old_threshold": registered["threshold"],
                "new_threshold": threshold(b)["threshold"],
                "changed_alerts_own_cuts": int(
                    np.sum(decisions(a, registered) != decisions(b, threshold(b)))
                ),
            }
    write_json(result, {"status": "measured", "arms": report})
    return [result]
