"""Prepare physically separate blind observations and author truth for the registered captures."""

from collections import Counter
from contextlib import ExitStack
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.threshold_validation.observations import (
    CHUNK_ROWS,
    HISTORY_FEATURES,
    RAW_FEATURES,
    REQUIRED,
    _prepare_chunk,
)
from .download import resolve_source
from .source_history import SourceHistory


REGISTRY = Path(__file__).resolve().parents[3] / "configs/contact-evidence-20261112/registration.json"
BLIND = ("row_id", "flow_end_ms", "source", "destination", "native_protocol", *RAW_FEATURES)
TRUTH = ("row_id", "family", "native_label", "native_detail")


def read_registration():
    return read_json(REGISTRY)


def prepared(root, name):
    return Path(root) / "local/prepared" / (name + ".parquet")


def truth(root, name):
    return Path(root) / "local/truth" / (name + ".parquet")


def _audit(root, name):
    return Path(root) / "evidence/sources" / (name + ".json")


def _fields(stream):
    for line in stream:
        if line.startswith("#fields"):
            fields = ["detailed-label" if c == "det_label" else c for c in line.split()[1:]]
            if len(set(fields)) != len(fields) or not REQUIRED.issubset(fields):
                raise ValueError("Missing or ambiguous IoT-23 connection columns")
            return fields
        if not line.startswith("#"):
            break
    raise ValueError("Missing Zeek #fields header")


def prepare_log(path, features, labels, *, chunk_rows=CHUNK_ROWS):
    """Preserve physical record identity. Expose no held-out class counts to the experiment."""
    rows = reversed_rows = ties = max_lag_ms = 0
    first, watermark, previous_end = None, -1, -1
    histogram, native, missing = Counter(), Counter(), Counter()
    state = SourceHistory()
    features.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=features.parent, prefix="zeek-") as temp:
        tmp = Path(temp)
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream, ExitStack() as writers:
            fields = _fields(stream)
            with pd.read_csv(
                stream,
                sep=r"\s+",
                comment="#",
                names=fields,
                usecols=list(REQUIRED),
                dtype=str,
                na_filter=False,
                chunksize=chunk_rows,
                quoting=3,
            ) as reader:
                for raw in reader:
                    if raw.empty:
                        continue
                    records, time, end, lag, inversions = _prepare_chunk(
                        raw, rows, watermark, previous_end, histogram, native, missing
                    )
                    obs = records.select(BLIND)
                    target = records.select(TRUTH)
                    for column, values in zip(HISTORY_FEATURES, state.transform(obs)):
                        obs = obs.append_column(column, pa.array(values))
                    if not rows:
                        first = int(time[0])
                        blind_writer = writers.enter_context(
                            pq.ParquetWriter(tmp / "blind", obs.schema, compression="zstd")
                        )
                        truth_writer = writers.enter_context(
                            pq.ParquetWriter(tmp / "truth", target.schema, compression="zstd")
                        )
                    blind_writer.write_table(obs)
                    truth_writer.write_table(target)
                    rows += len(obs)
                    reversed_rows += inversions
                    ties += int((time[1:] == time[:-1]).sum()) + int(time[0] == watermark)
                    max_lag_ms = max(max_lag_ms, int(lag.max()))
                    watermark, previous_end = int(time[-1]), int(end[-1])
        if not rows:
            raise ValueError("No observed records")
        (tmp / "blind").replace(features)
        (tmp / "truth").replace(labels)
    return {
        "rows": rows,
        "first_flow_end_ms": first,
        "last_flow_end_ms": watermark,
        "adjacent_end_time_decreases": reversed_rows,
        "logical_time_ties": ties,
        "max_clock_lag_ms": max_lag_ms,
    }


def prepare_one(root, source_dir, spec):
    path = resolve_source(source_dir, spec)
    name = spec["name"]
    features, labels = prepared(root, name), truth(root, name)
    audit = prepare_log(path, features, labels)
    for field in ("rows", "first_flow_end_ms", "last_flow_end_ms"):
        if field in spec and audit[field] != spec[field]:
            raise ValueError(f"Registered {field} does not match for {name}")
    evidence = _audit(root, name)
    write_json(evidence, {"source_sha256": spec["sha256"], "bytes": spec["bytes"], **audit})
    return [features, labels, evidence]


def check_chronology(root, reg):
    sources = reg["sources"]
    if len({s["name"] for s in sources}) != len(sources) or len({s["sha256"] for s in sources}) != len(
        sources
    ):
        raise ValueError("Capture names and uncompressed hashes must be unique across roles")
    roles = {role: {} for role in ("P", "C", "D", "E")}
    for spec in sources:
        name, role = spec["name"], spec["role"]
        audit = read_json(_audit(root, name))
        if audit["source_sha256"] != spec["sha256"] or audit["bytes"] != spec["bytes"]:
            raise ValueError(f"Source receipt does not match {name}")
        start, end = audit["first_flow_end_ms"], audit["last_flow_end_ms"]
        if end < start or audit["rows"] < 1:
            raise ValueError(f"Empty or negative span for {name}")
        if role == "E" and start <= reg["latest_opened_flow_end_ms"]:
            raise ValueError(f"Holdout must follow all previously inspected outcome periods: {name}")
        roles[role][name] = (start, end)
    if not all(roles[r] for r in ("P", "C", "E")):
        raise ValueError("Missing train, calibration or evaluation capture")
    previous = None
    for role in ("P", "C", "D", "E"):
        spans = list(roles[role].values())
        if not spans:
            continue
        if previous is not None and min(t[0] for t in spans) <= previous:
            raise ValueError("Train, control, development and evaluation periods overlap")
        if role == "E":
            ordered = sorted(spans)
            if any(a[1] >= b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError("Held-out capture periods overlap")
        previous = max(t[1] for t in spans)
    output = Path(root) / "evidence/chronology.json"
    write_json(
        output,
        {
            "captured_periods_ms": roles,
            "latest_opened_flow_end_ms": reg["latest_opened_flow_end_ms"],
            "identity": "Whole captures kept intact; row key is capture plus physical data-line number. Hosts can recur in different captures.",
            "clock": "Physical Zeek record order and cumulative maximum of ceil((start + duration) in ms); missing duration uses start. Equal logical times never observe each other.",
        },
    )
    return [output]
