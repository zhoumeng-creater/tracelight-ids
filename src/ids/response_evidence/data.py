"""Strictly row-ordered blind observations and physically separate author truth."""

from collections import Counter
from contextlib import ExitStack
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.contact_evidence.data import BLIND, TRUTH, _audit, _fields, prepared, truth
from ids.threshold_validation.observations import CHUNK_ROWS, REQUIRED, _prepare_chunk
from .observations import QUALITY_INPUTS, QUALITY_COLUMNS, GAP_COLUMNS, classify_response, counter_gap_frame
from .sources import resolve

REGISTRY = Path(__file__).resolve().parents[3] / "configs/response-evidence-20261112/registration.json"


def read_registration():
    from . import tqh

    reg = read_json(REGISTRY)
    reg["sources"].extend(tqh.read_specs())
    return reg


def _sfc_inputs(raw):
    raw = raw.rename(
        columns={
            "orig_h": "id.orig_h",
            "resp_h": "id.resp_h",
            "orig_p": "id.orig_p",
            "resp_p": "id.resp_p",
        }
    )
    source = raw.label_source
    unknown = source.isin(["unmatched_v7", "scan_heuristic"]) | source.str.startswith("ambiguous")
    normal = raw.attack.eq("0") & raw.type_of_attack.eq("normal") & ~unknown
    attack = raw.attack.eq("1") & ~raw.type_of_attack.isin(["normal", "-", ""]) & ~unknown
    if not (normal | attack | unknown).all():
        raise ValueError("Inconsistent IoT-SFC binary/type label pair")
    raw["label"] = np.select((normal, attack), ("Benign", "Malicious"), default="Unresolved")
    raw["detailed-label"] = raw.type_of_attack.mask(normal, "-").replace(
        {"scanning": "PartOfAHorizontalPortScan"}
    )
    return raw


def prepare_log(path, features, labels, *, sfc=False, chunk_rows=CHUNK_ROWS):
    rows = inversions = ties = max_lag_ms = 0
    first, watermark, previous_end = None, -1, -1
    counts, native, missing = Counter(), Counter(), Counter()
    features.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=features.parent, prefix="parse-") as temp:
        tmp = Path(temp)
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream, ExitStack() as stack:
            kwargs = {}
            if not sfc:
                fields = _fields(stream)
                if not set(QUALITY_INPUTS).issubset(fields):
                    raise ValueError("Missing Zeek response evidence fields")
                kwargs.update(
                    sep=r"\s+",
                    comment="#",
                    names=fields,
                    usecols=list(REQUIRED | set(QUALITY_INPUTS)),
                    quoting=3,
                )
            reader = stack.enter_context(
                pd.read_csv(stream, dtype=str, na_filter=False, chunksize=chunk_rows, **kwargs)
            )
            for raw in reader:
                if raw.empty:
                    continue
                if sfc:
                    raw = _sfc_inputs(raw)
                records, time, end, lag, decreases = _prepare_chunk(
                    raw, rows, watermark, previous_end, counts, native, missing
                )
                obs = records.select(BLIND)
                target = records.select(TRUTH)
                if sfc:
                    # Preserve the publisher's original values beside our mapped evaluation family.
                    for name, original in (("native_label", "attack"), ("native_detail", "type_of_attack")):
                        target = target.set_column(
                            target.schema.get_field_index(name), name, pa.array(raw[original])
                        )
                    for name, original in (
                        ("author_window", "window_id"),
                        ("label_source", "label_source"),
                        ("attacker_role", "attacker_role"),
                    ):
                        target = target.append_column(name, pa.array(raw[original]))
                # Encode once per date; window is never a predictor or a threshold input.
                days, indices = np.unique(time // 86400000, return_inverse=True)
                windows = pa.array(days.astype("datetime64[D]").astype(str)).take(pa.array(indices))
                target = target.append_column("window", windows)
                codes = classify_response(raw)
                obs = obs.append_column("response_evidence_code", pa.array(codes))
                for code, name in zip((-1, 0, 1), QUALITY_COLUMNS):
                    obs = obs.append_column(name, pa.array((codes == code).astype(np.int8)))
                gaps = counter_gap_frame(raw)
                for name in GAP_COLUMNS:
                    obs = obs.append_column(name, pa.array(gaps[name]))
                if not rows:
                    first = int(time[0])
                    bw = stack.enter_context(pq.ParquetWriter(tmp / "blind", obs.schema, compression="zstd"))
                    tw = stack.enter_context(
                        pq.ParquetWriter(tmp / "truth", target.schema, compression="zstd")
                    )
                bw.write_table(obs)
                tw.write_table(target)
                rows += len(obs)
                inversions += decreases
                ties += int((time[1:] == time[:-1]).sum()) + int(time[0] == watermark)
                max_lag_ms = max(max_lag_ms, int(lag.max()))
                watermark, previous_end = int(time[-1]), int(end[-1])
        if not rows:
            raise ValueError("No observed connection records")
        (tmp / "blind").replace(features)
        (tmp / "truth").replace(labels)
    return dict(
        rows=rows,
        first_flow_end_ms=first,
        last_flow_end_ms=watermark,
        adjacent_end_time_decreases=inversions,
        logical_time_ties=ties,
        max_clock_lag_ms=max_lag_ms,
    )


def prepare_one(root, source_dir, spec):
    if spec.get("format") == "tqh":
        from . import tqh

        return tqh.prepare_one(root, source_dir, spec)
    path = resolve(source_dir, spec, read_registration())
    name = spec["name"]
    features, labels = prepared(root, name), truth(root, name)
    info = prepare_log(path, features, labels, sfc=spec.get("format") == "iot-sfc")
    for key in ("rows", "first_flow_end_ms", "last_flow_end_ms"):
        if key in spec and info[key] != spec[key]:
            raise ValueError(f"Registered {key} differs for {name}")
    evidence = _audit(root, name)
    write_json(evidence, {"source_sha256": spec["sha256"], "bytes": spec["bytes"], **info})
    return [features, labels, evidence]


def check_chronology(root, reg):
    roles = {r: {} for r in ("P", "C", "R", "H", "E", "X")}
    names, hashes = set(), set()
    cutoff = reg["latest_opened_flow_end_ms"]
    previous_open = set(reg["previously_scored"])
    for spec in reg["sources"]:
        name, digest, role = spec["name"], spec["sha256"], spec["role"]
        if name in names or digest in hashes:
            raise ValueError("Capture name or bytes reused across registered roles")
        if role in ("H", "E", "X") and name in previous_open or role == "R" and name not in previous_open:
            raise ValueError(f"Inspection status disagrees with registry for {name}")
        names.add(name)
        hashes.add(digest)
        audit = read_json(_audit(root, name))
        if audit["source_sha256"] != digest or audit["bytes"] != spec["bytes"]:
            raise ValueError(f"Receipt mismatch: {name}")
        start, end = audit["first_flow_end_ms"], audit["last_flow_end_ms"]
        if end < start or audit["rows"] <= 0:
            raise ValueError(f"Invalid period: {name}")
        if (role in ("E", "X") and start <= cutoff) or (role in ("H", "R") and end > cutoff):
            raise ValueError(f"Incorrectly classified evaluation period: {name}")
        roles[role][name] = (start, end)
    if not all(roles.values()):
        raise ValueError("Registration lacks a train/calibration/replay/historical/forward cohort")
    if max(e for _, e in roles["P"].values()) >= min(s for s, _ in roles["C"].values()):
        raise ValueError("Training overlaps calibration")
    last_c = max(e for _, e in roles["C"].values())
    if any(s <= last_c for r in ("E", "R", "H", "X") for s, _ in roles[r].values()):
        raise ValueError("Evaluation period precedes end of calibration")
    # Our forward snapshots come from one sensor and must represent disjoint periods.
    forward = sorted(roles["E"].values())
    if any(a[1] >= b[0] for a, b in zip(forward, forward[1:])):
        raise ValueError("Forward sensor snapshots overlap")
    out = Path(root) / "evidence/chronology.json"
    write_json(
        out,
        {
            "captured_periods_ms": roles,
            "latest_opened_flow_end_ms": cutoff,
            "window": "IoT-23 and IoT-SFC: UTC day of cumulative flow-end clock. TQH: original test cell.",
            "clock": "Strict earlier logical times only. Cold start for every capture / single-sensor backup.",
        },
    )
    return [out]
