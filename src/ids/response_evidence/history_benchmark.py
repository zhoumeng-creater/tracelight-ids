"""Measure three histories on one physical, label-blind capture prefix with identical inputs."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import sha256_file
from ids.contact_evidence.features import enrich_prepared as sqlite_history
from ids.contact_evidence.source_history import SourceHistory
from ids.threshold_validation.observations import HISTORY_FEATURES
from .history import ADDED, LEGACY_COLUMNS, _resources, enrich_prepared


def _legacy(prefix, dest):
    temp = dest.with_name("old-source.parquet")
    state = SourceHistory()
    with pq.ParquetFile(prefix) as stream:
        fields = [
            pa.field(name, pa.float64() if "fraction" in name else pa.int32()) for name in HISTORY_FEATURES
        ]
        schema = pa.schema([*stream.schema_arrow, *fields], metadata=stream.schema_arrow.metadata)
        with pq.ParquetWriter(temp, schema, compression="zstd") as writer:
            for batch in stream.iter_batches(batch_size=32768):
                table = pa.Table.from_batches([batch])
                cols = state.transform(table)
                writer.write_table(pa.Table.from_arrays([*table.columns, *cols], schema=schema))
    return sqlite_history(temp, dest)


def _compare(left, right, names):
    checks = {}
    with pq.ParquetFile(left) as a, pq.ParquetFile(right) as b:
        if a.metadata.num_rows != b.metadata.num_rows:
            raise ValueError("Mismatched history row counts")

        def stream(file):
            return file.iter_batches(columns=list(names), batch_size=32768)

        digest = {name: hashlib.sha256() for name in names}
        for lhs, rhs in zip(stream(a), stream(b), strict=True):
            for name in names:
                x, y = lhs.column(name), rhs.column(name)
                if not x.equals(y):
                    raise ValueError(f"History parity failed for {name}")
                digest[name].update(np.ascontiguousarray(x.to_numpy()).tobytes())
        for name in names:
            checks[name] = digest[name].hexdigest()
    return checks


def benchmark_history(source_path, *, rows=131072, workers=4, block_rows=32768):
    """Report exact legacy and serial parity and historical-build-only timing on the same prefix."""
    path = Path(source_path)
    if rows < 1:
        raise ValueError("Expected a positive benchmark size")
    with TemporaryDirectory(prefix="benchmark-history-", dir=path.parent) as name:
        scratch = Path(name)
        prefix = scratch / "prefix.parquet"
        with pq.ParquetFile(path) as stream:
            n = min(rows, stream.metadata.num_rows)
            if n == 0:
                raise ValueError("Empty benchmark input")
            with pq.ParquetWriter(prefix, stream.schema_arrow, compression="zstd") as writer:
                seen = 0
                for batch in stream.iter_batches(batch_size=min(32768, n)):
                    part = batch.slice(0, min(len(batch), n - seen))
                    writer.write_table(pa.Table.from_batches([part]))
                    seen += len(part)
                    if seen == n:
                        break
        old, one, multi = (scratch / (v + ".parquet") for v in ("legacy", "serial", "parallel"))
        with _resources() as legacy:
            old_stats = _legacy(prefix, old)
        single = enrich_prepared(prefix, one, workers=1, block_rows=block_rows)
        parallel = enrich_prepared(prefix, multi, workers=workers, block_rows=block_rows)
        old_hashes = _compare(old, multi, ("row_id", *LEGACY_COLUMNS))
        all_hashes = _compare(one, multi, ("row_id", *ADDED))
        return {
            "source": path.name,
            "blind_prefix_parquet_sha256": sha256_file(prefix),
            "physical_first_row": 1,
            "physical_last_row": n,
            "scope": "Identical blind Parquet prefix including I/O; no labels, fits or GPU work. RSS sampled in shared process.",
            "legacy": {**legacy, "sqlite_stage_seconds": old_stats["seconds"]},
            "one_worker": single,
            "parallel": parallel,
            "speedup_vs_legacy": legacy["seconds"] / parallel["seconds"],
            "speedup_vs_one_worker": single["seconds"] / parallel["seconds"],
            "all_rows_legacy_column_sha256": old_hashes,
            "all_rows_serial_column_sha256": all_hashes,
        }
