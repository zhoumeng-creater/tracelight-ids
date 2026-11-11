"""Causal IoT-23 Zeek-record observations in the publisher's physical log order."""

from collections import Counter
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ids.flow_history.features import HISTORY_COLUMNS, WINDOW_MS, _distinct_in_window


RAW_FEATURES = (
    "ip_protocol",
    "orig_port",
    "resp_port",
    "duration_seconds",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
)
HISTORY_FEATURES = tuple(HISTORY_COLUMNS)
PORTS = {"id.orig_p": "orig_port", "id.resp_p": "resp_port"}
COUNTS = ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
REQUIRED = {
    "ts",
    "id.orig_h",
    "id.resp_h",
    "proto",
    "duration",
    "label",
    "detailed-label",
    *PORTS,
    *COUNTS,
}
PROTOCOL = {"icmp": 1, "tcp": 6, "udp": 17, "icmp6": 58, "icmpv6": 58, "sctp": 132}
MISSING = {"", "-", "(empty)"}
CHUNK_ROWS = 131072


def _family(label, detail):
    label, detail = label.casefold(), detail.casefold()
    if label == "benign" and detail in MISSING | {"benign"}:
        return "Benign"
    if label != "malicious" or detail in MISSING | {"unknown", "unresolved", "benign"}:
        return "Unresolved"
    if detail == "partofahorizontalportscan":
        return "Scanning"
    if detail == "c&c":
        return "C&C"
    if {"c&c", "partofahorizontalportscan"}.intersection(detail.split("-")):
        return "Mixed"
    return "OtherAttack"


def _numbers(series, name, *, integer=False, required=False):
    cleaned = series.mask(series.isin(MISSING))
    nums = pd.to_numeric(cleaned, errors="raise").to_numpy(dtype=np.float64)
    valid = ~np.isnan(nums)
    if not np.isfinite(nums[valid]).all() or (nums[valid] < 0).any() or (required and not valid.all()):
        raise ValueError(f"Invalid Zeek {name}")
    if integer and (nums[valid] != np.floor(nums[valid])).any():
        raise ValueError(f"Fractional count in {name}")
    nums[~valid] = -1
    return nums.astype(np.int64) if integer else nums


def _sum_window(indicator, low, high):
    cumulative = np.empty(len(indicator) + 1, dtype=np.int64)
    cumulative[0] = 0
    np.cumsum(indicator, dtype=np.int64, out=cumulative[1:])
    return cumulative[high] - cumulative[low]


def _history_input(path):
    columns = ["flow_end_ms", "source", "native_protocol", "destination", "resp_port", "resp_pkts"]
    table = pq.read_table(path, columns=columns)
    codes = {}
    for col in ["source", "native_protocol", "destination"]:
        encoded = pc.dictionary_encode(table[col].combine_chunks())
        codes[col] = encoded.indices.to_numpy()
    keys = (codes["source"].astype(np.int64) << 32) | codes["native_protocol"].astype(np.int64)
    return (
        table["flow_end_ms"].to_numpy(),
        keys,
        codes["destination"],
        table["resp_port"].to_numpy(),
        table["resp_pkts"].to_numpy(),
    )


def _history(clock, keys, destination, port, received):
    # Within one source/protocol the input clock is nondecreasing. Separate keys on a
    # synthetic time axis by >60s so one vectorized window query covers all keys.
    order = keys.argsort(kind="stable")
    key = keys[order]
    ordered_clock = clock[order]
    delta = np.diff(ordered_clock)
    delta[key[1:] != key[:-1]] = WINDOW_MS + 1
    axis = np.empty(len(clock), dtype=np.int64)
    axis[0] = 0
    np.cumsum(delta, out=axis[1:])
    low = axis.searchsorted(axis - WINDOW_MS, side="left")
    high = axis.searchsorted(axis, side="left")
    spans = high - low
    recv = received[order]
    known = _sum_window(recv >= 0, low, high)
    unanswered = _sum_window(recv == 0, low, high)
    fractions = np.divide(unanswered, known, out=np.full(len(key), -1.0, dtype=float), where=known > 0)
    fractions[spans == 0] = 0.0
    # Unknown destination ports have sentinel -1 in the raw part; never count the sentinel as a port.
    p = port[order]
    valid = p >= 0
    if valid.any():
        port_counts = _distinct_in_window(axis[valid], p[valid], axis)
    else:
        port_counts = np.zeros(len(axis), dtype=np.int32)
    dst_counts = _distinct_in_window(axis, destination[order], axis)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return (
        spans[inverse].astype(np.int32),
        dst_counts[inverse].astype(np.int32),
        port_counts[inverse].astype(np.int32),
        fractions[inverse],
    )


def _prepare_chunk(raw, offset, last_clock, prior_end, family_counts, native_counts, missing):
    raw = raw.fillna("-")
    ts = _numbers(raw.ts, "ts", required=True)
    duration = _numbers(raw.duration, "duration")
    # IoT-23 timestamps and durations use Zeek's fixed microsecond input precision.
    start_us = np.rint(ts * 1_000_000).astype(np.int64)
    elapsed_us = np.rint(np.maximum(duration, 0.0) * 1_000_000).astype(np.int64)
    end = (start_us + elapsed_us + 999) // 1000
    clock = np.maximum.accumulate(np.maximum(end, last_clock))
    adjacent_decreases = int((np.diff(end, prepend=prior_end) < 0).sum())
    lag = clock - end

    label = raw.label.astype(str)
    detail = raw["detailed-label"].astype(str)
    pairs = pd.MultiIndex.from_arrays([label, detail])
    codes, unique = pd.factorize(pairs, sort=False)
    family_values = []
    for (a, b), count in zip(unique, np.bincount(codes)):
        family = _family(a, b)
        family_values.append(family)
        native_counts[(a, b, family)] += int(count)
        family_counts[family] += int(count)
    proto = raw.proto.str.lower()
    if (
        raw["id.orig_h"].isin(MISSING).any()
        or raw["id.resp_h"].isin(MISSING).any()
        or proto.isin(MISSING).any()
    ):
        raise ValueError("Zeek connection lacks a source, destination or protocol")
    output = {
        "row_id": np.arange(offset + 1, offset + 1 + len(raw), dtype=np.int64),
        "flow_end_ms": clock,
        "family": np.array(family_values, dtype=object)[codes],
        "native_label": label,
        "native_detail": detail,
        "source": raw["id.orig_h"],
        "native_protocol": proto,
        "destination": raw["id.resp_h"],
        "ip_protocol": proto.map(PROTOCOL).fillna(-1).to_numpy(dtype=np.int16),
        "duration_seconds": duration,
    }
    for col, target in list(PORTS.items()) + [(c, c) for c in COUNTS]:
        values = _numbers(raw[col], col, integer=True)
        if col in PORTS and (values > 65535).any():
            raise ValueError(f"Out-of-range port in {col}")
        output[target] = values
        missing[target] += int((values < 0).sum())
    missing["duration_seconds"] += int((duration < 0).sum())
    missing["unmapped_protocol"] += int((output["ip_protocol"] < 0).sum())
    return pa.Table.from_pydict(output), clock, end, lag, adjacent_decreases


def prepare(path, dest):
    """Write row-ordered compact Parquet; return coverage and surrogate-clock diagnostics."""
    path, dest = Path(path), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    family_counts, native_counts, missing = Counter(), Counter(), Counter()
    n = inversions = clamped = largest_lag = ties = 0
    first_ms, last_ms, previous_end = None, -1, -1
    with TemporaryDirectory(dir=dest.parent, prefix="zeek-observations-") as tmp:
        stage = Path(tmp) / "raw.parquet"
        opener = gzip.open if path.suffix.casefold() == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.startswith("#fields"):
                    fields = line.split()[1:]
                    break
                if not line.startswith("#"):
                    raise ValueError("Zeek data appeared before #fields")
            else:
                raise ValueError("No Zeek #fields directive")
            if not REQUIRED.issubset(fields) or len(set(fields)) != len(fields):
                raise ValueError("Missing or ambiguous IoT-23 columns")
            reader = pd.read_csv(
                stream,
                sep=r"\s+",
                comment="#",
                names=fields,
                usecols=list(REQUIRED),
                dtype=str,
                na_filter=False,
                chunksize=CHUNK_ROWS,
                quoting=3,
            )
            writer = None
            try:
                for raw in reader:
                    if raw.empty:
                        continue
                    table, clock, end, lag, reversed_rows = _prepare_chunk(
                        raw, n, last_ms, previous_end, family_counts, native_counts, missing
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(stage, table.schema, compression="zstd")
                        first_ms = int(clock[0])
                    writer.write_table(table)
                    n += len(clock)
                    inversions += reversed_rows
                    clamped += int((lag != 0).sum())
                    ties += int((np.diff(clock, prepend=last_ms) == 0).sum())
                    largest_lag = max(largest_lag, int(lag.max()))
                    last_ms, previous_end = int(clock[-1]), int(end[-1])
            finally:
                reader.close()
                if writer is not None:
                    writer.close()
        if not n:
            raise ValueError("No observations in IoT-23 capture")
        histories = _history(*_history_input(stage))
        compact = Path(tmp) / "observations.parquet"
        keep = ["row_id", "flow_end_ms", "family", "native_label", "native_detail", *RAW_FEATURES]
        start = 0
        with pq.ParquetFile(stage) as pf:
            writer = None
            try:
                for batch in pf.iter_batches(batch_size=CHUNK_ROWS, columns=keep):
                    table = pa.Table.from_batches([batch])
                    end = start + len(table)
                    for name, values in zip(HISTORY_FEATURES, histories):
                        table = table.append_column(name, pa.array(values[start:end]))
                    if writer is None:
                        writer = pq.ParquetWriter(compact, table.schema, compression="zstd")
                    writer.write_table(table)
                    start = end
            finally:
                if writer is not None:
                    writer.close()
        if start != n:
            raise ValueError("Lost physical log rows while writing histories")
        compact.replace(dest)
    return {
        "rows": n,
        "family_counts": dict(sorted(family_counts.items())),
        "native_label_mapping": [
            {"native_label": a, "native_detail": b, "family": f, "rows": count}
            for (a, b, f), count in sorted(native_counts.items())
        ],
        "first_flow_end_ms": first_ms,
        "last_flow_end_ms": last_ms,
        "physical_end_adjacent_decreases": inversions,
        "physical_end_clamped_to_previous_watermark": clamped,
        "logical_time_tied_with_previous": ties,
        "max_clock_lag_ms": largest_lag,
        "clock": "Physical Zeek log order. Cumulative maximum of ceil((ts + known duration)*1000); missing duration uses ts. Actual exporter arrival delay unavailable.",
        "history_window": "[logical time - 60000ms, logical time), per source and protocol, all incoming records; no-reply fraction uses only known responder packet counts. Empty window 0; nonempty window with no valid responder counts -1.",
        "missing_or_unknown": dict(sorted(missing.items())),
    }
