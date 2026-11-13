"""Pinned Round 23 detectors, with a real early-normal replay before the holdout is scored."""

from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from ids.contact_evidence.experiment import TREE_GROUPS
from ids.route_batch.protocol import REPO

from .scoring import POLICIES, VIEWS, make_decisions, register, score_group

CONFIG = REPO / "configs/raw-workpoint-20261113"


def unpack(root):
    ref = read_json(CONFIG / "controls.json")
    source = CONFIG / "frozen-controls.zip"
    if sha256_file(source) != ref["bundle_sha256"]:
        raise ValueError("Frozen controls bundle has changed")
    folder = root / "local/controls"
    outputs = []
    with ZipFile(source) as archive:
        if set(archive.namelist()) != set(ref["members"]):
            raise ValueError("Frozen controls member list has changed")
        for name, digest in ref["members"].items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(archive.read(name))
            if sha256_file(tmp) != digest:
                raise ValueError(f"Damaged frozen control: {name}")
            tmp.replace(target)
            outputs.append(target)
    return outputs


def _paired_policies(frozen, actual):
    result = {}
    n = len(actual)
    for name in POLICIES:
        old = np.asarray(frozen[name], dtype=bool)
        new = actual[name].to_numpy(dtype=bool)
        added = int((~old & new).sum())
        lost = int((old & ~new).sum())
        fp = int(new.sum())
        result[name] = {
            "registered_fp": int(old.sum()),
            "actual_fp": fp,
            "added": added,
            "lost": lost,
            "changed_alerts": added + lost,
            "fp_cap": n // 100,
            "within_budget": fp <= n // 100,
        }
    return result


def probe(root, group, threads, batch_rows=65_536):
    """Replay the production scorer on each earlier capture, keeping original strict thresholds."""
    base = root / "local/controls"
    path = CONFIG / "early-normal-features.parquet"
    if sha256_file(path) != read_json(CONFIG / "controls.json")["calibration_features_sha256"]:
        raise ValueError("Early-normal features changed")
    features = pd.read_parquet(path).set_index(["capture", "row_id"])
    directory = base / "evidence/calibration"
    reference = pd.read_parquet(directory / (group + ".parquet")).set_index(["capture", "row_id"])
    if len(reference) != 1826 or not reference.index.is_unique or not features.index.equals(reference.index):
        raise ValueError("Early-normal calibration identities changed")
    rules, supplements = register(directory, group)
    frozen_bits = pd.DataFrame(make_decisions(reference, rules, supplements, features), index=reference.index)
    cpu = group in TREE_GROUPS
    device = "cpu" if cpu else "cuda:0"
    actual = []
    by_capture = {}
    seconds = dict.fromkeys(VIEWS, 0.0)
    with TemporaryDirectory(dir=root / "local") as tmp:
        work = Path(tmp)
        for capture, frame in features.groupby(level="capture", sort=False):
            # Use physical positions within this replay file after checking the original identities above.
            observed = frame.reset_index(drop=True).assign(row_id=np.arange(1, len(frame) + 1))
            observed.to_parquet(work / "observed.parquet", index=False)
            observed[["row_id"]].to_parquet(work / "physical.parquet", index=False)
            result = score_group(
                work / "observed.parquet",
                work / "physical.parquet",
                base / "models/frozen",
                directory,
                group,
                work / "sealed.parquet",
                threads=threads,
                device=device,
                batch_rows=batch_rows,
            )
            scored = pd.read_parquet(work / "sealed.parquet")
            if not np.array_equal(scored.row_id, observed.row_id):
                raise ValueError("Calibration replay changed physical row order")
            scored.index = frame.index
            actual.append(scored)
            by_capture[capture] = {
                "rows": len(frame),
                "policies": _paired_policies(frozen_bits.loc[frame.index], scored),
            }
            for view in VIEWS:
                seconds[view] += result["scoring_seconds"][view]
    actual = pd.concat(actual).loc[reference.index]
    paired = _paired_policies(frozen_bits, actual)
    measured = {}
    for view in VIEWS:
        observed = actual["score_" + view].to_numpy()
        archived = reference[view].to_numpy()
        # HGB serialization should be bit-identical. GPU batch shape and hardware can change FP32 logits.
        close = (
            np.array_equal(observed, archived)
            if cpu
            else np.allclose(observed, archived, rtol=1e-5, atol=1e-5)
        )
        measured[view] = {
            "max_abs_score_delta": float(np.abs(observed - archived).max()),
            "within_score_tolerance": bool(close),
            "changed_alerts": paired[view]["changed_alerts"],
            "rows_per_second": len(reference) / seconds[view],
            "score_batch": result["model_score_batch"][view],
            "device": device,
            "threads": threads,
        }
    out = root / "evidence/probes" / (group + ".json")
    write_json(
        out,
        {
            "calibration_rows": len(reference),
            "device": device,
            "batch_rows": batch_rows,
            "comparison": "strict greater than original or residual C threshold",
            "continuous_score_tolerance": {"rtol": 0.0 if cpu else 1e-5, "atol": 0.0 if cpu else 1e-5},
            "measured": measured,
            "policies": paired,
            "by_capture": by_capture,
        },
    )
    if not all(v["within_score_tolerance"] for v in measured.values()):
        raise ValueError(f"Frozen {group} scores exceed replay tolerance; see {out}")
    if cpu and any(v["changed_alerts"] for v in paired.values()):
        raise ValueError(f"Frozen {group} CPU decision differs from the registered reference; see {out}")
    return [out]
