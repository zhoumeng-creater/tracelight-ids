"""Independent TQH-C2 cells; preserve native Zeek physical order and join author labels by UID."""

from collections import Counter
from decimal import Decimal
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, sha256_file, write_json
from ids.contact_evidence.data import BLIND, _audit, prepared, truth
from ids.contact_evidence.download import ssl_context
from ids.threshold_validation.observations import REQUIRED, _prepare_chunk
from .observations import GAP_COLUMNS, QUALITY_COLUMNS, QUALITY_INPUTS, classify_response, counter_gap_frame


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/response-evidence-20261112/tqh.json"
CACHE = ROOT / "data/raw/response-evidence-20261112"
BASE = "https://zenodo.org/records/23905375/files/"
FAMILIES = {
    "benign": "Benign",
    "benign_external": "Benign",
    "malicious_c2": "ExternalC2",
    "malicious_recon": "ExternalRecon",
    "unknown": "Unresolved",
    "malicious_lateral": "OtherAttack",
}


def read_specs():
    return read_json(REGISTRY)["sources"]


@lru_cache(maxsize=32)
def _digest(path, size, mtime):
    return sha256_file(path)


def _valid(file, spec):
    if not file.is_file():
        return False
    st = file.stat()
    return (
        st.st_size == spec["archive_bytes"]
        and _digest(file, st.st_size, st.st_mtime_ns) == spec["archive_sha256"]
    )


def _archive(spec, source_dir):
    name = spec["archive"]
    path = (Path(source_dir) if source_dir is not None else CACHE) / name
    if _valid(path, spec):
        return path
    if source_dir is not None:
        raise ValueError(f"Offline TQH archive missing or invalid: {path}")
    CACHE.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        with NamedTemporaryFile(dir=CACHE, prefix="tqh-", suffix=".part", delete=False) as target:
            temp = Path(target.name)
        try:
            req = Request(BASE + name + "?download=1", headers={"Accept-Encoding": "identity"})
            with urlopen(req, timeout=90, context=ssl_context()) as resp, temp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            if not _valid(temp, spec):
                raise ValueError("TQH author archive digest differs")
            temp.replace(path)
            return path
        except (OSError, EOFError, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(f"Failed to fetch pinned TQH archive {name}") from exc
            time.sleep(2**attempt)
        finally:
            temp.unlink(missing_ok=True)


def copy_offline(spec, output):
    dest = Path(output) / spec["archive"]
    if _valid(dest, spec):
        return dest
    source = _archive(spec, None)
    if source.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=dest.parent, prefix="copy-tqh-", delete=False) as temp:
            pending = Path(temp.name)
        try:
            shutil.copyfile(source, pending)
            if not _valid(pending, spec):
                raise ValueError("Copied TQH archive digest differs")
            pending.replace(dest)
        finally:
            pending.unlink(missing_ok=True)
    return dest


def _cell(archive, prefix):
    native = archive.read(prefix + "/conn.log")
    labels = archive.read(prefix + "/labeled.jsonl")
    manifest = json.loads(archive.read(prefix + "/manifest.json"))
    # Zeek JSON can carry sub-microsecond durations. Keep the exact published
    # decimal through ceiling, even when TS is a large epoch timestamp.
    rows = [json.loads(line, parse_float=Decimal) for line in native.splitlines()]
    end = np.fromiter(
        (math.ceil((row["ts"] + (row.get("duration") or 0)) * 1000) for row in rows), dtype=np.int64
    )
    raw = pd.DataFrame.from_records(rows)
    author = pd.DataFrame.from_records(
        [json.loads(line) for line in labels.splitlines()], columns=["uid", "label"]
    )
    if (
        raw.empty
        or raw.uid.isna().any()
        or not raw.uid.is_unique
        or author.uid.isna().any()
        or not author.uid.is_unique
    ):
        raise ValueError("Missing or duplicated TQH flow UID")
    if not author.uid.isin(raw.uid).all() or not raw.uid.isin(author.uid).all():
        raise ValueError("TQH conn.log and author labels have different UID sets")
    ordered = author.set_index("uid").loc[raw.uid, "label"].reset_index(drop=True)
    if not ordered.isin(FAMILIES).all():
        raise ValueError("Unknown native TQH label")
    # JSON Zeek logs omit optional keys for unset fields (e.g. history for ICMP).
    for name in REQUIRED | set(QUALITY_INPUTS):
        if name not in raw:
            if name in {"history", "duration"}:
                raw[name] = "-"
            elif name not in {"label", "detailed-label"}:
                raise ValueError(f"Missing TQH capture field {name}")
    # Blind preparation receives no author outcome. _prepare_chunk builds the frozen clock/inputs.
    raw = raw.fillna("-")
    raw["label"], raw["detailed-label"] = "Benign", "-"
    records = _prepare_chunk(raw, 0, -1, -1, Counter(), Counter(), Counter())[0]
    clock = np.maximum.accumulate(end)
    obs = records.select(BLIND)
    obs = obs.set_column(obs.schema.get_field_index("flow_end_ms"), "flow_end_ms", pa.array(clock))
    codes = classify_response(raw)
    obs = obs.append_column("response_evidence_code", pa.array(codes))
    for code, name in zip((-1, 0, 1), QUALITY_COLUMNS, strict=True):
        obs = obs.append_column(name, pa.array((codes == code).astype(np.int8)))
    gaps = counter_gap_frame(raw)
    for name in GAP_COLUMNS:
        obs = obs.append_column(name, pa.array(gaps[name]))
    unit = prefix.split("/")[-1]
    target = pa.table(
        {
            "row_id": records["row_id"],
            "family": ordered.map(FAMILIES),
            "native_label": ordered,
            "native_detail": ordered,
            "window": pa.array([unit] * len(raw)),
        }
    )
    receipt = {
        "sha256": hashlib.sha256(native).hexdigest(),
        "bytes": len(native),
        "labels_sha256": hashlib.sha256(labels).hexdigest(),
        "uid_order_sha256": hashlib.sha256(("\n".join(raw.uid) + "\n").encode()).hexdigest(),
        "rows": len(raw),
        "first_flow_end_ms": int(clock[0]),
        "last_flow_end_ms": int(clock[-1]),
        "manifest_start_utc": manifest["start_utc"],
        "manifest_end_utc": manifest["end_utc"],
        "adjacent_end_time_decreases": int(np.sum(np.diff(end) < 0)),
        "logical_time_ties": int(np.sum(np.diff(clock) == 0)),
        "max_clock_lag_ms": int((clock - end).max()),
    }
    return obs, target, receipt


def prepare_one(root, source_dir, spec):
    source = _archive(spec, source_dir)
    with ZipFile(source) as archive:
        obs, labels, audit = _cell(archive, spec["profile"] + "/" + spec["unit"])
    for key in (
        "sha256",
        "bytes",
        "labels_sha256",
        "uid_order_sha256",
        "rows",
        "first_flow_end_ms",
        "last_flow_end_ms",
        "manifest_start_utc",
        "manifest_end_utc",
    ):
        if audit[key] != spec[key]:
            raise ValueError(f"TQH cell {spec['name']} violates registered {key}")
    name = spec["name"]
    features, targets = prepared(root, name), truth(root, name)
    for path, table in ((features, obs), (targets, labels)):
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(".tmp")
        pq.write_table(table, pending, compression="zstd")
        pending.replace(path)
    report = _audit(root, name)
    write_json(report, {**audit, "source_sha256": audit["sha256"], "archive_sha256": spec["archive_sha256"]})
    return [features, targets, report]
