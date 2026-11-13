"""Evaluate preregistered alert streams against raw HGB on the same physical flows."""

from collections import defaultdict
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import write_json
from ids.response_evidence.evaluation import endpoint


ARMS = ("raw", "contact", "quality", "add_contact", "add_quality", "add_quic30", "add_web64k")
CELLS = np.arange(1 << len(ARMS))
RAW = (CELLS & 1) != 0


def _aligned(parquets, columns, batch_rows):
    """Split on the earliest physical batch boundary across the three inputs."""
    readers = [
        f.iter_batches(batch_size=batch_rows, columns=cols, use_threads=False)
        for f, cols in zip(parquets, columns, strict=True)
    ]
    pending = [next(reader, None) for reader in readers]
    while all(batch is not None for batch in pending):
        n = min(batch.num_rows for batch in pending)
        if n:
            yield [batch.slice(0, n) for batch in pending]
        pending = [
            batch.slice(n) if batch.num_rows > n else next(reader, None)
            for batch, reader in zip(pending, readers, strict=True)
        ]
    if any(batch is not None and batch.num_rows for batch in pending):
        raise ValueError("Source streams end at different physical rows")


def _compare(hist, bit):
    candidate = (CELLS & (1 << bit)) != 0
    before, after = int(hist[RAW].sum()), int(hist[candidate].sum())
    return dict(
        rows=int(hist.sum()),
        before=before,
        after=after,
        gained=int(hist[candidate & ~RAW].sum()),
        lost=int(hist[~candidate & RAW].sum()),
        net=after - before,
    )


def _report(groups):
    endpoints = defaultdict(lambda: np.zeros(len(CELLS), dtype=np.int64))
    native = []
    for (family, label, detail), hist in sorted(groups.items()):
        group = endpoint(family, detail)
        endpoints[group] += hist
        native.append(
            dict(
                family=family,
                native_label=label,
                native_detail=detail,
                endpoint=group,
                paired={arm: _compare(hist, bit) for bit, arm in enumerate(ARMS)},
            )
        )
    normal = endpoints.get("Benign", np.zeros(len(CELLS), dtype=np.int64))
    n = int(normal.sum())
    budget = n // 100 if n else None
    predictions = {}
    for bit, arm in enumerate(ARMS):
        benign = _compare(normal, bit)
        predictions[arm] = dict(
            normal_rows=n,
            allowed_fp=budget,
            raw_fp=benign["before"],
            candidate_fp=benign["after"],
            raw_within_budget=benign["before"] <= budget if n else None,
            candidate_within_budget=benign["after"] <= budget if n else None,
            both_within_budget=max(benign["before"], benign["after"]) <= budget if n else None,
            endpoints={group: _compare(hist, bit) for group, hist in sorted(endpoints.items())},
        )
    return dict(rows=sum(int(h.sum()) for h in groups.values()), comparisons=predictions, native=native)


def evaluate(pred: Path, truth: Path, blind: Path, destination: Path, *, batch_rows: int = 131072):
    """Report one whole-capture and separate UTC-day results, including over-budget days."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    columns = (
        ["row_id", *ARMS],
        ["row_id", "family", "native_label", "native_detail"],
        ["row_id", "flow_end_ms"],
    )
    day_groups = defaultdict(lambda: defaultdict(lambda: np.zeros(len(CELLS), dtype=np.int64)))
    seen, previous_clock = 0, None
    with ExitStack() as stack:
        files = [stack.enter_context(pq.ParquetFile(path)) for path in (pred, truth, blind)]
        sizes = [f.metadata.num_rows for f in files]
        if not sizes[0] or len(set(sizes)) != 1:
            raise ValueError("Require three nonempty aligned capture sizes")
        for arm in ARMS:
            if not pa.types.is_boolean(files[0].schema_arrow.field(arm).type):
                raise ValueError(f"Expected sealed boolean alert column: {arm}")
        for f in files:
            if not pa.types.is_integer(f.schema_arrow.field("row_id").type):
                raise ValueError("Expected integer physical row IDs")
        if not pa.types.is_integer(files[2].schema_arrow.field("flow_end_ms").type):
            raise ValueError("Expected integer UTC millisecond clock")
        for p, t, b in _aligned(files, columns, batch_rows):
            ids = np.arange(seen + 1, seen + 1 + len(p), dtype=np.int64)
            for stream in (p, t, b):
                if any(col.null_count for col in stream.columns):
                    raise ValueError("Missing decision, truth, row ID or observation clock")
                if not np.array_equal(stream["row_id"].to_numpy(), ids):
                    raise ValueError("Three streams must follow identical consecutive physical row IDs")
            clock = b["flow_end_ms"].to_numpy()
            if np.any(clock[1:] < clock[:-1]) or (previous_clock is not None and clock[0] < previous_clock):
                raise ValueError("Observation clock regresses")
            previous_clock = clock[-1]
            dates = clock // 86400000
            codes, keys = pd.factorize(
                pd.MultiIndex.from_frame(t.select(columns[1][1:]).to_pandas()), sort=False
            )
            flags = np.zeros(len(ids), dtype=np.int64)
            raw = p["raw"].to_numpy(zero_copy_only=False)
            for bit, arm in enumerate(ARMS):
                decision = p[arm].to_numpy(zero_copy_only=False)
                if arm.startswith("add_") and np.any(raw & ~decision):
                    raise ValueError(f"Additive policy removed a raw alert: {arm}")
                flags |= decision.astype(np.int64) << bit
            for day in np.unique(dates):
                mask = dates == day
                counts = np.bincount(codes[mask] * len(CELLS) + flags[mask], minlength=len(keys) * len(CELLS))
                lookup = day_groups[int(day)]
                for key, val in zip(keys, counts.reshape(-1, len(CELLS)), strict=True):
                    if val.any():
                        lookup[key] += val
            seen += len(ids)
    if seen != sizes[0]:
        raise ValueError("Incomplete physical capture")
    total = defaultdict(lambda: np.zeros(len(CELLS), dtype=np.int64))
    days = {}
    for day, groups in sorted(day_groups.items()):
        days[(date(1970, 1, 1) + timedelta(days=day)).isoformat()] = _report(groups)
        for key, hist in groups.items():
            total[key] += hist
    result = dict(baseline="raw", arms=list(ARMS), rows=seen, capture=_report(total), utc_days=days)
    write_json(destination, result)
    return result
