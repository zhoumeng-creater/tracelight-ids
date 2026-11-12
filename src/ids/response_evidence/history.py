"""Parallel, label-free 60-second histories in Zeek's physical observation order."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter, process_time, thread_time

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from ids.contact_evidence.features import CONTACT_FEATURES, SERVICE_FEATURES
from ids.flow_history.features import WINDOW_MS, _distinct_in_window
from ids.threshold_validation.observations import HISTORY_FEATURES, _sum_window


CODE = "response_evidence_code"
SCOPES = ("src", "svc", "contact")
QUALITY_HISTORY_COLUMNS = tuple(
    f"{scope}_prior_{stat}_fraction_60s" for scope in SCOPES for stat in ("observed_no_reply", "unknown")
)
LEGACY_COLUMNS = HISTORY_FEATURES + SERVICE_FEATURES + CONTACT_FEATURES
ADDED = LEGACY_COLUMNS + QUALITY_HISTORY_COLUMNS
META = ("flow_end_ms", "source", "native_protocol", "destination", "resp_port", "resp_pkts", CODE)
LABELS = {"family", "label", "native_label", "native_detail", "detailed-label", "det_label"}


@contextmanager
def _resources():
    result = {}
    proc = psutil.Process()
    rss = [proc.memory_info().rss]
    stop = Event()

    def sample():
        while not stop.wait(0.05):
            rss[0] = max(rss[0], proc.memory_info().rss)

    thread = Thread(target=sample, daemon=True)
    thread.start()
    wall, cpu = perf_counter(), process_time()
    try:
        yield result
    finally:
        result["seconds"] = perf_counter() - wall
        result["process_cpu_seconds"] = process_time() - cpu
        result["mean_cpu_cores"] = result["process_cpu_seconds"] / max(result["seconds"], 1e-9)
        stop.set()
        thread.join()
        result["sampled_peak_rss_bytes"] = max(rss[0], proc.memory_info().rss)


def _stage(source, scratch, n, block_rows):
    # A memory-mapped Arrow file permits concurrent random slices, including a 60s warm-up,
    # without a capture-wide Python address dictionary or repeated Parquet decompression.
    clock = np.empty(n, dtype=np.int64)
    offsets = [0]
    schema = pa.schema([source.schema_arrow.field(col) for col in META])
    with pa.OSFile(str(scratch / "events.arrow"), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
        for batch in source.iter_batches(columns=["row_id", *META], batch_size=max(4096, block_rows)):
            pos, end = offsets[-1], offsets[-1] + len(batch)
            if any(col.null_count for col in batch.columns):
                raise ValueError("History metadata contains nulls")
            if not np.array_equal(batch.column("row_id").to_numpy(), np.arange(pos + 1, end + 1)):
                raise ValueError("Expected uninterrupted physical row IDs")
            times = batch.column("flow_end_ms").to_numpy()
            if (np.diff(times) < 0).any() or times[0] < (clock[pos - 1] if pos else 0):
                raise ValueError("Expected monotone nonnegative observation time")
            if not np.isin(batch.column(CODE).to_numpy(), [-1, 0, 1]).all():
                raise ValueError("Invalid response evidence code")
            clock[pos:end] = times
            writer.write_batch(batch.select(list(META)))
            offsets.append(end)
    if offsets[-1] != n:
        raise ValueError("Input physical row count does not match metadata")
    return clock, np.asarray(offsets)


def _partitions(clock, block_rows):
    # Partition time even if every flow has the same source. Charge a very dense minute to
    # one block so subsequent jobs do not reread its entire prefix repeatedly.
    first = 0
    while first < len(clock):
        end = min(first + block_rows, len(clock))
        minute_end = int(clock.searchsorted(int(clock[first]) + WINDOW_MS, side="right"))
        if minute_end - first > 3 * block_rows:
            end = minute_end
        # Never split a clock tie. A multi-million-row tie has a constant empty within-tie history.
        if end < len(clock) and clock[end] == clock[end - 1]:
            end = int(clock.searchsorted(clock[end], side="right"))
        warm = int(clock.searchsorted(int(clock[first]) - WINDOW_MS, side="left"))
        yield warm, first, end
        first = end


def _slice(reader, offsets, lo, end):
    i = int(offsets.searchsorted(lo, side="right") - 1)
    j = int(offsets.searchsorted(end, side="left"))
    batches = []
    for group in range(i, j):
        left, right = max(lo, offsets[group]), min(end, offsets[group + 1])
        batches.append(reader.get_batch(group).slice(int(left - offsets[group]), int(right - left)))
    return pa.Table.from_batches(batches)


def _dictionary(table, name):
    return table[name].combine_chunks().dictionary_encode().indices.to_numpy()


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(denominator), -1.0), where=denominator > 0)


def _scope(scope, data, first):
    times, source, proto, dst, port, replies, quality = data
    key_cols = [source, proto]
    if scope != "src":
        key_cols.append(port)
    if scope == "contact":
        key_cols.append(dst)
    # NumPy's stable lexsort keeps physical time nondecreasing inside each key.
    order = np.lexsort(tuple(reversed(key_cols)))
    change = np.zeros(len(order) - 1, dtype=bool)
    for col in key_cols:
        k = col[order]
        change |= k[1:] != k[:-1]
    delta = np.diff(times[order])
    delta[change] = WINDOW_MS + 1
    axis = np.empty(len(order), dtype=np.int64)
    axis[0] = 0
    np.cumsum(delta, out=axis[1:])
    targets = np.flatnonzero(order >= first)
    # Only query output rows. Warm-up observations never receive output.
    pos = order[targets] - first
    query = axis[targets]
    lo, hi = axis.searchsorted(query - WINDOW_MS), axis.searchsorted(query)
    count = hi - lo
    recv, code = replies[order], quality[order]
    known = _sum_window(recv >= 0, lo, hi)
    legacy = _ratio(_sum_window(recv == 0 if scope == "src" else recv > 0, lo, hi), known)
    q_known = _sum_window(code >= 0, lo, hi)
    q_zero = _ratio(_sum_window(code == 0, lo, hi), q_known)
    unknown = _ratio(count - q_known, count)
    if scope == "src":
        legacy[count == 0] = 0
        p = port[order]
        good = p >= 0
        ports = _distinct_in_window(axis[good], p[good], query) if good.any() else np.zeros_like(count)
        dests = _distinct_in_window(axis, dst[order], query)
        names = HISTORY_FEATURES
        values = [count.astype("int32"), dests.astype("int32"), ports.astype("int32"), legacy]
    else:
        missing = port[order[targets]] < 0
        count[missing] = 0
        legacy[missing] = q_zero[missing] = unknown[missing] = -1.0
        names = SERVICE_FEATURES if scope == "svc" else CONTACT_FEATURES
        values = [count.astype("int32"), legacy]
    names += tuple(c for c in QUALITY_HISTORY_COLUMNS if c.startswith(scope + "_"))
    output = {}
    for name, col in zip(names, [*values, q_zero, unknown], strict=True):
        physical = np.empty_like(col)
        physical[pos] = col
        output[name] = physical
    return output


def _empty_window(rows):
    counts = np.zeros(rows, dtype=np.int32)
    missing = np.full(rows, -1.0)
    zero = np.zeros(rows, dtype=np.float64)
    return pa.table(
        {
            name: zero if name == HISTORY_FEATURES[-1] else (missing if "fraction" in name else counts)
            for name in ADDED
        }
    )


def _part_index(path, n, offsets, window, scope_workers):
    warm, first, end = window
    with pa.memory_map(str(path / "events.arrow"), "r") as mmap:
        table = _slice(pa.ipc.open_file(mmap), offsets, warm, end)
        data = tuple(
            _dictionary(table, col)
            if col in ("source", "native_protocol", "destination")
            else table[col].to_numpy()
            for col in META
        )

        def calculate(scope):
            started = thread_time()
            values = _scope(scope, data, first - warm)
            return values, thread_time() - started

        if scope_workers > 1:
            with ThreadPoolExecutor(max_workers=scope_workers) as executor:
                groups = list(executor.map(calculate, SCOPES))
        else:
            groups = [calculate(scope) for scope in SCOPES]
        output = {k: v for values, _ in groups for k, v in values.items()}
        result = pa.table({k: output[k] for k in ADDED})
        with (
            pa.OSFile(str(path / f"{n}.arrow"), "wb") as sink,
            pa.ipc.new_file(sink, result.schema) as writer,
        ):
            writer.write_table(result)
    return sum(cpu for _, cpu in groups)


def _join(source, pending, blocks, block_rows, empty_blocks):
    # Arrow history blocks can span arbitrary Parquet row groups. Decode at most one
    # original output batch, and close the writer even if a subsequent block is unreadable.
    schema = pa.schema(
        [*source.schema_arrow, *_empty_window(0).schema], metadata=source.schema_arrow.metadata
    )
    raw = iter(source.iter_batches(batch_size=min(block_rows, 262144)))
    batch = next(raw)
    with pq.ParquetWriter(pending, schema, compression="zstd") as writer:
        for i, (_, first, end) in enumerate(blocks):
            with ExitStack() as stack:
                if i not in empty_blocks:
                    mmap = stack.enter_context(pa.memory_map(str(pending.parent / f"{i}.arrow"), "r"))
                    features = pa.ipc.open_file(mmap).read_all()
                pos = 0
                while pos < end - first:
                    size = min(len(batch), end - first - pos)
                    part = _empty_window(size) if i in empty_blocks else features.slice(pos, size)
                    table = pa.Table.from_arrays(
                        [*batch.slice(0, size).columns, *part.columns], schema=schema
                    )
                    writer.write_table(table)
                    pos += size
                    batch = batch.slice(size) if size < len(batch) else next(raw, None)


def enrich_prepared(source_path, dest_path, *, workers=4, block_rows=262144):
    """Append old and quality histories. Publish complete output only after row-order-preserving join."""
    source_path, dest_path = Path(source_path), Path(dest_path)
    if workers < 1 or block_rows < 1 or source_path.resolve() == dest_path.resolve():
        raise ValueError("Expected positive workers/block_rows and distinct paths")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with _resources() as stats, TemporaryDirectory(prefix="history-", dir=dest_path.parent) as temp:
        path = Path(temp)
        with pq.ParquetFile(source_path) as source:
            names = set(source.schema_arrow.names)
            if not {"row_id", *META} <= names or names.intersection(set(ADDED) | LABELS):
                raise ValueError("Expected blind observations without precomputed histories")
            rows = source.metadata.num_rows
            if not rows:
                raise ValueError("Empty observation stream")
            start = perf_counter()
            clock, offsets = _stage(source, path, rows, block_rows)
            stage_seconds = perf_counter() - start
            blocks = list(_partitions(clock, block_rows))
            largest = max(end - warm for warm, _, end in blocks)
            # Reserve half of currently available RAM for other pipeline jobs, Arrow and file cache.
            # 320 bytes per row and concurrent scope accommodates sorting/prefix/dictionary temporaries.
            available_bytes = psutil.virtual_memory().available
            slots = max(1, min(workers, (available_bytes // 2) // (320 * largest)))
            outer = min(len(blocks), slots)
            # A dense capture can have one 700k-row block and a small tail. Give free
            # workers to the heaviest remaining block instead of dividing them equally.
            inner = [1] * len(blocks)
            extra = min(slots - outer, len(blocks) * (len(SCOPES) - 1))
            weights = [end - warm for warm, _, end in blocks]
            for _ in range(extra):
                options = (i for i in range(len(blocks)) if inner[i] < len(SCOPES))
                i = max(options, key=lambda idx: weights[idx] / inner[idx])
                inner[i] += 1

            # This covers an arbitrarily large isolated burst at one timestamp without
            # allocating or sorting one entry per tied endpoint.
            empty_blocks = {i for i, (warm, _, end) in enumerate(blocks) if clock[warm] == clock[end - 1]}

            def work(index):
                if index in empty_blocks:
                    return 0.0
                return _part_index(path, index, offsets, blocks[index], inner[index])

            start = perf_counter()
            with ThreadPoolExecutor(max_workers=outer) as executor:
                worker_cpu = sum(executor.map(work, range(len(blocks))))
            kernel_seconds = perf_counter() - start
            pending = path / "result.parquet"
            join_start = perf_counter()
            _join(source, pending, blocks, block_rows, empty_blocks)
            join_seconds = perf_counter() - join_start
            scratch_bytes = sum(file.stat().st_size for file in path.iterdir())
            output_bytes = pending.stat().st_size
        pending.replace(dest_path)
    return {
        **stats,
        "rows": rows,
        "added": list(ADDED),
        "window_ms": WINDOW_MS,
        "requested_workers": workers,
        "time_block_workers": outer,
        "scope_workers_per_block": max(inner),
        "scope_workers_per_block_min": min(inner),
        "blocks": len(blocks),
        "longest_block_with_warmup_rows": largest,
        "all_tied_empty_blocks": len(empty_blocks),
        "block_rows": block_rows,
        "kernel_seconds": kernel_seconds,
        "stage_seconds": stage_seconds,
        "join_seconds": join_seconds,
        "available_memory_at_dispatch_bytes": available_bytes,
        "scratch_bytes_including_output": scratch_bytes,
        "enriched_parquet_bytes": output_bytes,
        "kernel_thread_cpu_seconds": worker_cpu,
        "kernel_mean_active_scope_cores": worker_cpu / kernel_seconds,
        "rows_per_second": rows / stats["seconds"],
    }
