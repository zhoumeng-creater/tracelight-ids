"""Frozen IoT-23 43-1 source and parallel preparation in physical Zeek log order."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import gzip
from io import BytesIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.contact_evidence.data import BLIND, TRUTH, _audit, prepared, truth
from ids.contact_evidence.download import resolve_source
from ids.response_evidence.observations import (
    GAP_COLUMNS,
    QUALITY_COLUMNS,
    QUALITY_INPUTS,
    classify_response,
    counter_gap_frame,
)
from ids.threshold_validation.observations import REQUIRED, _prepare_chunk

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "configs/raw-workpoint-20261113/source.json"
CACHE = REPO / "data/raw/raw-workpoint-20261113"
BLOCK_ROWS = 131072
# Input is read sequentially only for decompression/framing. Actual parsing, dtype
# conversion, label mapping, response classification and IPC writes run in children.
BYTES_PER_BLOCK = 16 << 20


def read_spec():
    return read_json(REGISTRY)


def resolve(source_dir=None, *, workers=8):
    return resolve_source(source_dir, read_spec(), cache=CACHE, workers=workers)


def _header(stream):
    for line in stream:
        if line.startswith(b"#fields"):
            names = ["detailed-label" if x == "det_label" else x for x in line.decode("utf8").split()[1:]]
            if len(names) != len(set(names)) or not REQUIRED.union(QUALITY_INPUTS).issubset(names):
                raise ValueError("Incomplete or repeated Zeek fields")
            return names
        if not line.startswith(b"#"):
            break
    raise ValueError("Missing Zeek header")


def _blocks(stream, target_bytes):
    # One trailing readline ensures every chunk ends at a physical line boundary.
    while body := stream.read(target_bytes):
        if not body.endswith(b"\n"):
            body += stream.readline()
        yield body


def _store(table, path):
    with pa.OSFile(str(path), "wb") as file, pa.ipc.new_file(file, table.schema) as writer:
        writer.write_table(table)


def _parse(body, names, prefix):
    start = process_time()
    proc = psutil.Process()
    raw = pd.read_csv(
        BytesIO(body),
        sep=r"\s+",
        comment="#",
        names=names,
        usecols=sorted(REQUIRED.union(QUALITY_INPUTS)),
        dtype=str,
        na_filter=False,
        quoting=3,
    )
    if raw.empty:
        return None
    records, _, end, _, drops = _prepare_chunk(raw, 0, -1, -1, Counter(), Counter(), Counter())
    # Write raw ends to scratch. The parent assigns the cross-block prefix maximum.
    blind = records.select(BLIND)
    blind = blind.set_column(blind.schema.get_field_index("flow_end_ms"), "flow_end_ms", pa.array(end))
    code = classify_response(raw)
    blind = blind.append_column("response_evidence_code", pa.array(code))
    for name, value in zip(QUALITY_COLUMNS, (-1, 0, 1), strict=True):
        blind = blind.append_column(name, pa.array((code == value).astype(np.int8)))
    gaps = counter_gap_frame(raw)
    for name in GAP_COLUMNS:
        blind = blind.append_column(name, pa.array(gaps[name]))
    _store(blind, prefix.with_suffix(".blind"))
    _store(records.select(TRUTH), prefix.with_suffix(".truth"))
    return {
        "rows": len(raw),
        "pid": os.getpid(),
        "cpu": process_time() - start,
        "rss": proc.memory_info().rss,
        "within_block_clock_decreases": int(drops),
    }


def _finish(prefix, info, state, stack, writers, pending):
    with ExitStack() as reads:
        tables = []
        for part in ("blind", "truth"):
            f = reads.enter_context(pa.memory_map(str(prefix.with_suffix("." + part)), "r"))
            tables.append(pa.ipc.open_file(f).read_all())
        blind, target = tables
        n, watermark, previous = state["rows"], state["last_flow_end_ms"], state["prev_end"]
        raw_end = blind["flow_end_ms"].to_numpy()
        clock = np.maximum.accumulate(np.maximum(watermark, raw_end))
        ids = pa.array(np.arange(n + 1, n + 1 + len(clock), dtype=np.int64))
        blind = blind.set_column(blind.schema.get_field_index("row_id"), "row_id", ids)
        blind = blind.set_column(blind.schema.get_field_index("flow_end_ms"), "flow_end_ms", pa.array(clock))
        target = target.set_column(target.schema.get_field_index("row_id"), "row_id", ids)
        days, ix = np.unique(clock // 86400000, return_inverse=True)
        target = target.append_column(
            "window", pa.array(days.astype("datetime64[D]").astype(str)).take(pa.array(ix))
        )
        if len(target) != len(blind) or len(target) != info["rows"]:
            raise ValueError("Different blind/truth chunk sizes")
        if not writers:
            state["first_flow_end_ms"] = int(clock[0])
            for dest, table in zip(pending, (blind, target), strict=True):
                writers.append(stack.enter_context(pq.ParquetWriter(dest, table.schema, compression="zstd")))
        for writer, table in zip(writers, (blind, target), strict=True):
            writer.write_table(table, row_group_size=131072)
        state["rows"] += len(raw_end)
        state["last_flow_end_ms"] = int(clock[-1])
        state["prev_end"] = int(raw_end[-1])
        state["adjacent_end_time_decreases"] += (
            int(raw_end[0] < previous) + info["within_block_clock_decreases"]
        )
        state["logical_time_ties"] += int(clock[0] == watermark) + int(np.sum(clock[1:] == clock[:-1]))
        state["max_clock_lag_ms"] = max(state["max_clock_lag_ms"], int((clock - raw_end).max()))
        # Arrow arrays can keep mapped pages alive after file close on Windows.
        del blind, target, tables, table, raw_end
    for suffix in (".blind", ".truth"):
        prefix.with_suffix(suffix).unlink()


def prepare_log(path, features, labels, *, workers=8, block_rows=BLOCK_ROWS, expected=None):
    """Create R23-compatible blind/truth parquet. No model, score or test-label feedback."""
    path, features, labels = Path(path), Path(features), Path(labels)
    if len({p.resolve() for p in (path, features, labels)}) != 3 or workers < 1 or block_rows < 1:
        raise ValueError("Distinct files and positive worker/block sizes required")
    features.parent.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    began, parent_cpu = perf_counter(), process_time()
    requested_workers = workers
    workers = min(workers, os.cpu_count() or 1)
    # Cap live payloads on Windows; the largest allocation is the parsed string frame per worker.
    memory_per_worker = 384 << 20
    max_workers = max(1, min(workers, psutil.virtual_memory().available // (2 * memory_per_worker)))
    target_bytes = max(1, min(BYTES_PER_BLOCK, block_rows * 128))
    state = {
        "rows": 0,
        "first_flow_end_ms": None,
        "last_flow_end_ms": -1,
        "prev_end": -1,
        "adjacent_end_time_decreases": 0,
        "logical_time_ties": 0,
        "max_clock_lag_ms": 0,
    }
    used_pids = set()
    worker_cpu = 0.0
    max_child_rss = 0
    nonempty = 0
    with TemporaryDirectory(dir=features.parent, prefix="parse43-") as temp:
        scratch = Path(temp)
        pending = [scratch / "blind.parquet", scratch / "truth.parquet"]
        with gzip.open(path, "rb") as stream, ExitStack() as stack:
            names = _header(stream)

            def consume(item):
                nonlocal worker_cpu, max_child_rss, nonempty
                prefix, result = item
                info = result.result() if pool else result
                if info is None:
                    return
                _finish(prefix, info, state, stack, writers, pending)
                used_pids.add(info["pid"])
                worker_cpu += info["cpu"]
                max_child_rss = max(max_child_rss, info["rss"])
                nonempty += 1

            pool = (
                stack.enter_context(ProcessPoolExecutor(max_workers=max_workers)) if max_workers > 1 else None
            )
            queue = deque()
            writers = []
            for i, body in enumerate(_blocks(stream, target_bytes)):
                prefix = scratch / str(i)
                queue.append(
                    (
                        prefix,
                        pool.submit(_parse, body, names, prefix) if pool else _parse(body, names, prefix),
                    )
                )
                if len(queue) >= max_workers + 1:
                    consume(queue.popleft())
            while queue:
                consume(queue.popleft())
        if not state["rows"]:
            raise ValueError("Empty capture")
        if expected:
            for field in ("rows", "first_flow_end_ms", "last_flow_end_ms"):
                if field in expected and state[field] != expected[field]:
                    raise ValueError(f"Source violates registered {field}")
        pending[0].replace(features)
        pending[1].replace(labels)
    state.pop("prev_end")
    elapsed = perf_counter() - began
    cpu = process_time() - parent_cpu
    return {
        **state,
        "seconds": elapsed,
        "requested_workers": requested_workers,
        "allocated_worker_slots": max_workers,
        "actual_parser_processes": len(used_pids),
        "blocks": nonempty,
        "raw_target_bytes_per_block": target_bytes,
        "parse_cpu_seconds": worker_cpu,
        "parent_process_cpu_seconds": cpu,
        "mean_observed_cpu_cores": (cpu + (worker_cpu if max_workers > 1 else 0)) / elapsed,
        "largest_sampled_worker_rss_bytes": max_child_rss,
    }


def prepare(root, *, source_dir=None, workers=8, block_rows=BLOCK_ROWS):
    spec = read_spec()
    source = resolve(source_dir, workers=workers)
    name = spec["name"]
    blind, labels = prepared(root, name), truth(root, name)
    info = prepare_log(source, blind, labels, workers=workers, block_rows=block_rows, expected=spec)
    receipt = _audit(root, name)
    write_json(receipt, {"source_sha256": spec["sha256"], "bytes": spec["bytes"], **info})
    return [blind, labels, receipt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the checksum-verified offline 43-1 source")
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    print(resolve(args.source_dir, workers=args.workers), flush=True)
