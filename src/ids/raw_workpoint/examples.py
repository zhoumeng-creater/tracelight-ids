"""Small replayable samples of misses, recoveries and normal controls."""

from contextlib import ExitStack

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import write_json
from ids.response_evidence.experiment import FULL
from .evaluate import ARMS, _aligned

# Take at most 32 per (day, native label, raw decision, candidate decision pattern).
# The smallest 64-bit hashes give a repeatable sample across the entire day.
LIMIT = 32


def _hash(rows):
    x = rows.astype(np.uint64)
    x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def collect(pred, truth, features, destination, *, batch_rows):
    groups = {}
    totals = {}
    selected = ["row_id", *ARMS, "score_raw", "score_contact", "score_quality"]
    meta = ["row_id", "family", "native_detail"]
    cols = ["row_id", "flow_end_ms", *FULL]
    with ExitStack() as stack:
        files = [stack.enter_context(pq.ParquetFile(p)) for p in (pred, truth, features)]
        for p, t, f in _aligned(files, (selected, meta, cols), batch_rows):
            ids = p["row_id"].to_numpy()
            if any(not np.array_equal(ids, b["row_id"].to_numpy()) for b in (t, f)):
                raise ValueError("Misaligned example records")
            dates = f["flow_end_ms"].to_numpy() // 86400000
            details = t["native_detail"].dictionary_encode()
            families = t["family"].dictionary_encode()
            # Build keys in integer arrays, independent of dictionary ordering in other batches.
            bits = np.zeros(len(ids), dtype=np.int64)
            for bit, arm in enumerate(ARMS):
                bits |= p[arm].to_numpy(zero_copy_only=False).astype(np.int64) << bit
            stride = 1 << len(ARMS)
            first = int(dates[0])
            codes = (
                ((dates - first) * len(families.dictionary) + families.indices.to_numpy())
                * len(details.dictionary)
                + details.indices.to_numpy()
            ) * stride + bits
            order = np.argsort(codes, kind="stable")
            values = codes[order]
            splits = np.flatnonzero(np.diff(values)) + 1
            for indexes in np.split(order, splits):
                i = indexes[0]
                key = (int(dates[i]), families[i].as_py(), details[i].as_py(), int(bits[i]))
                totals[key] = totals.get(key, 0) + len(indexes)
                hashes = _hash(ids[indexes])
                if len(hashes) > LIMIT:
                    keep = np.argpartition(hashes, LIMIT - 1)[:LIMIT]
                    indexes, hashes = indexes[keep], hashes[keep]
                prior = groups.get(key)
                if prior and len(prior) == LIMIT:
                    below = hashes < prior[-1][0]
                    indexes, hashes = indexes[below], hashes[below]
                if not len(indexes):
                    continue
                ix = pa.array(indexes)
                sample = pa.Table.from_arrays(
                    [c.take(ix) for c in p.columns] + [f[n].take(ix) for n in cols[1:]],
                    names=selected + cols[1:],
                )
                pairs = list(zip(hashes.tolist(), sample.to_pylist(), strict=True))
                groups[key] = sorted([*(prior or []), *pairs], key=lambda pair: pair[0])[:LIMIT]
    rows = []
    for (day, family, detail, bits), records in sorted(groups.items()):
        for _, record in records:
            rows.append(
                {
                    "utc_day": str(np.datetime64(day, "D")),
                    "family": family,
                    "native_detail": detail,
                    "stratum_rows": totals[day, family, detail, bits],
                    **record,
                }
            )
    if not rows:
        raise ValueError("Empty example stream")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows), pending, compression="zstd")
    pending.replace(destination)
    spec = destination.with_suffix(".json")
    write_json(
        spec,
        {
            "rows": len(rows),
            "strata": len(totals),
            "per_stratum_max": LIMIT,
            "total_scored": sum(totals.values()),
            "sampling": "lowest splitmix finalizer(row_id) per day/family/detail/alerts",
        },
    )
    return [destination, spec]
