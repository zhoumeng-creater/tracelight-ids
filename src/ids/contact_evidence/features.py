"""Causal service/contact histories. External sorting keeps full addresses off the Python heap."""

from collections import deque
from contextlib import ExitStack, closing
from itertools import groupby
from operator import itemgetter
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ids.flow_history.features import WINDOW_MS
from ids.threshold_validation.observations import HISTORY_FEATURES, RAW_FEATURES

SERVICE_FEATURES = ("svc_prior_flows_60s", "svc_prior_reply_fraction_60s")
CONTACT_FEATURES = ("contact_prior_flows_60s", "contact_prior_reply_fraction_60s")
NEW_FEATURES = SERVICE_FEATURES + CONTACT_FEATURES
ARMS = {
    "raw": tuple(RAW_FEATURES),
    "history": RAW_FEATURES + HISTORY_FEATURES,
    "service": RAW_FEATURES + HISTORY_FEATURES + SERVICE_FEATURES,
    "contact": RAW_FEATURES + HISTORY_FEATURES + NEW_FEATURES,
}
META = ("source", "native_protocol", "destination", "resp_port", "resp_pkts", "flow_end_ms", "row_id")
CHUNK = 32_768


def _stage(source, db, chunk):
    db.execute("CREATE TABLE events(src TEXT, proto TEXT, dst TEXT, p INTEGER, r INTEGER, t INTEGER)")
    n = 0
    previous = -1
    for batch in source.iter_batches(columns=list(META), batch_size=chunk):
        if any(col.null_count for col in batch.columns):
            raise ValueError("Contact metadata contains nulls")
        times = batch.column(5).to_numpy()
        row_ids = batch.column(6).to_numpy()
        if (
            times[0] < previous
            or (np.diff(times) < 0).any()
            or not np.array_equal(row_ids, np.arange(n + 1, n + len(batch) + 1))
        ):
            raise ValueError("Contact features require uninterrupted row IDs and monotone observation time")
        previous = times[-1]
        n += len(batch)
        # Bound the Python strings and tuples to one batch. rowid auto-increments in input order.
        db.executemany(
            "INSERT INTO events VALUES(?,?,?,?,?,?)", zip(*(col.to_pylist() for col in batch.columns[:6]))
        )
    db.commit()
    return n


def _calculate(db, grouping, counts, ratios):
    """Scan one service/contact and clock at a time. Equal times never update each other's evidence."""
    keys = ", ".join(grouping)
    # Insert straight into a disk B-tree. No large ORDER BY scratch files on the system drive.
    types = ", ".join(f"{k} {'INTEGER' if k == 'p' else 'TEXT'}" for k in grouping)
    db.execute(
        f"CREATE TABLE ordered({types}, t INTEGER, r INTEGER, id INTEGER, "
        f"PRIMARY KEY ({keys}, t, r, id)) WITHOUT ROWID"
    )
    db.execute(f"INSERT INTO ordered SELECT {keys}, t, r, rowid FROM events WHERE p >= 0")
    db.commit()
    query = f"SELECT {keys}, t, r, id FROM ordered ORDER BY {keys}, t, r, id"
    # A window holds at most 60,001 distinct millisecond bins, even for a massive clock tie.
    with closing(db.execute(query)) as cursor:
        for _, rows in groupby(cursor, key=itemgetter(*range(len(grouping)))):
            prior = deque()
            count = answered = known = 0
            for instant, same_time in groupby(rows, key=itemgetter(-3)):
                while prior and prior[0][0] < instant - WINDOW_MS:
                    _, n, a, k = prior.popleft()
                    count -= n
                    answered -= a
                    known -= k
                fraction = answered / known if known else -1.0
                n = a = k = 0
                for row in same_time:
                    response, pos = row[-2:]
                    counts[pos - 1] = count
                    ratios[pos - 1] = fraction
                    n += 1
                    a += response > 0
                    k += response >= 0
                count += n
                answered += a
                known += k
                prior.append((instant, n, a, k))
    db.execute("DROP TABLE ordered")
    db.commit()


def enrich_prepared(source_path, dest_path, *, chunk=CHUNK):
    """Extend a label-free Zeek parquet capture; keep physical row order and replace output atomically."""
    source_path, dest_path = Path(source_path), Path(dest_path)
    if source_path.resolve() == dest_path.resolve() or chunk < 1:
        raise ValueError("Expected distinct input/output paths and a positive chunk size")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with TemporaryDirectory(prefix="contact-", dir=dest_path.parent) as name, ExitStack() as stack:
        scratch = Path(name)
        source = stack.enter_context(pq.ParquetFile(source_path))
        names = source.schema_arrow.names
        if (
            not set(META).issubset(names)
            or set(NEW_FEATURES).intersection(names)
            or {"family", "label", "native_label", "native_detail"}.intersection(names)
        ):
            raise ValueError("Expected blind endpoint-preserving input with no precomputed contact features")
        count = source.metadata.num_rows
        if not count:
            raise ValueError("Empty observation stream")
        db = stack.enter_context(closing(sqlite3.connect(scratch / "contacts.sqlite3")))
        db.execute("PRAGMA journal_mode=OFF")  # Scratch DB is discarded if the operation is interrupted.
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA temp_store=FILE")
        if _stage(source, db, chunk) != count:
            raise ValueError("Parquet metadata and physical row count differ")
        arrays = {}
        for field in NEW_FEATURES:
            is_ratio = "fraction" in field
            arr = np.memmap(
                scratch / (field + ".bin"),
                mode="w+",
                dtype="float64" if is_ratio else "int32",
                shape=(count,),
            )
            stack.callback(arr._mmap.close)
            arr[:] = -1 if is_ratio else 0  # Unknown destination ports have no matching service.
            arrays[field] = arr
        _calculate(db, ("src", "proto", "p"), *(arrays[k] for k in SERVICE_FEATURES))
        _calculate(db, ("src", "proto", "p", "dst"), *(arrays[k] for k in CONTACT_FEATURES))
        pending = scratch / "enriched.parquet"
        schema = source.schema_arrow
        for field, arr in arrays.items():
            schema = schema.append(pa.field(field, pa.from_numpy_dtype(arr.dtype)))
        pos = 0
        with pq.ParquetWriter(pending, schema, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=chunk):
                end = pos + len(batch)
                table = pa.Table.from_batches([batch])
                for field, arr in arrays.items():
                    table = table.append_column(field, pa.array(arr[pos:end]))
                writer.write_table(table)
                pos = end
        sqlite_bytes = (scratch / "contacts.sqlite3").stat().st_size
        mapped_bytes = sum(arr.nbytes for arr in arrays.values())
        output_bytes = pending.stat().st_size
        pending.replace(dest_path)
    return {
        "rows": count,
        "seconds": perf_counter() - started,
        "added": list(NEW_FEATURES),
        "window_ms": WINDOW_MS,
        "working_batch_rows": chunk,
        "allocated_sqlite_bytes_before_cleanup": sqlite_bytes,
        "allocated_feature_map_bytes": mapped_bytes,
        "enriched_parquet_bytes": output_bytes,
    }
