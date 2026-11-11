"""History at flow completion for the fixed NF-ToN-IoT-v3 input.

Disk staging preserves CSV record identities. Replay shards never share a
(source IPv4, IP protocol) state. Their inputs contain no labels, absolute
host identities as features, or scores. The public entry accepts CSV or ZIP.
"""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from ipaddress import IPv4Address
from itertools import groupby
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
import hashlib
import json
import time
import zipfile

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.csv as pacsv

from ids.common import canonical_hash, sha256_file, write_json
from ids.context_cache import lab_data as lab
from ids.route_batch.confirmation import _HashingReader


SOURCE_SHA256 = "53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78"
SOURCE_RECORDS = 27_520_260
HISTORY_COLUMNS = ("hist_flows_60s", "hist_dst_ips_60s", "hist_dst_ports_60s", "hist_no_reply_fraction_60s")
WINDOW_MS = 60_000
EVENT = np.dtype(
    [
        ("ms", "<i8"),
        ("key", "<u8"),
        ("dst", "<u4"),
        ("port", "<u2"),
        ("no_reply", "u1"),
        ("target", "<i4"),
        ("record", "<u4"),
    ]
)


@contextmanager
def _source_stream(source):
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            members = [
                entry
                for entry in archive.infolist()
                if not entry.is_dir() and PurePosixPath(entry.filename).name == "NF-ToN-IoT-v3.csv"
            ]
            if len(members) != 1:
                raise ValueError("expected one NF-ToN-IoT-v3.csv ZIP member")
            with archive.open(members[0]) as stream:
                yield stream
    else:
        with source.open("rb") as stream:
            yield stream


@lru_cache(maxsize=65536)
def _ip_code(address):
    return int(IPv4Address(address.strip()))


def _addresses(batch, column):
    codes = batch[column].dictionary_encode()
    if codes.null_count:
        raise ValueError(f"missing {column}")
    values = np.array([_ip_code(addr) for addr in codes.dictionary.to_pylist()], dtype=np.uint32)
    return values[codes.indices.to_numpy()]


def _integers(batch, column, maximum):
    values = batch[column].to_numpy(zero_copy_only=False)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > maximum).any()
        or not np.array_equal(values, np.floor(values))
    ):
        raise ValueError(f"invalid integer in {column}")
    return values.astype(np.int64)


def _target_numbers(targets, digest):
    if not len(targets) or len(targets) >= np.iinfo(np.int32).max or not targets["row_id"].is_unique:
        raise ValueError("expected nonempty targets with unique row_id")
    parts = targets["row_id"].astype(str).str.rpartition(":")
    if not parts[0].eq(digest).all() or not parts[2].str.fullmatch("[1-9][0-9]*").all():
        raise ValueError("row_id must contain the source SHA-256 and a one-based CSV record number")
    return parts[2].to_numpy(dtype=np.int64)


def _verify_target_batch(batch, offsets, subset):
    raw = batch.take(pa.array(offsets)).to_pandas()
    left = raw[list(lab.FEATURES)].to_numpy(dtype=np.float64)
    right = subset[list(lab.FEATURES)].to_numpy(dtype=np.float64)
    equal = (left == right) | (np.isnan(left) & np.isnan(right))
    if not equal.all():
        row, col = np.argwhere(~equal)[0]
        raise ValueError(f"target feature mismatch at {subset.row_id.iloc[row]}, {lab.FEATURES[col]}")
    if not np.array_equal(raw.Label, subset.label):
        raise ValueError("target label mismatch with source CSV")
    if not np.array_equal(
        raw.Attack.str.strip().str.casefold(), subset.attack_family.str.strip().str.casefold()
    ):
        raise ValueError("target attack_family mismatch with source CSV")
    if "communication_group" in subset:
        groups = [
            lab._communication_group_from_ids(src, dst, int(sp), int(dp), int(proto))
            for src, dst, sp, dp, proto in zip(
                raw.IPV4_SRC_ADDR, raw.IPV4_DST_ADDR, raw.L4_SRC_PORT, raw.L4_DST_PORT, raw.PROTOCOL
            )
        ]
        if not np.array_equal(groups, subset.communication_group):
            raise ValueError("target communication_group mismatch with source CSV")
    if "flow_end_ms" in subset:
        present = subset.flow_end_ms.notna().to_numpy()
        if not np.array_equal(
            subset.flow_end_ms.to_numpy()[present], raw.FLOW_END_MILLISECONDS.to_numpy()[present]
        ):
            raise ValueError("target flow_end_ms mismatch with source CSV")
    return raw[["IPV4_SRC_ADDR", "PROTOCOL", "FLOW_END_MILLISECONDS"]].rename(
        columns={"IPV4_SRC_ADDR": "source_host", "PROTOCOL": "protocol", "FLOW_END_MILLISECONDS": "end_ms"}
    )


def _stage(source, targets, row_numbers, paths, expected):
    order = np.argsort(row_numbers)
    sorted_numbers = row_numbers[order]
    metadata = []
    checked = scanned = backwards = 0
    minimum = maximum = previous = None
    src, dst = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
    start, end = "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
    types = {name: pa.float64() for name in lab.FEATURES}
    types.update(
        {
            src: pa.string(),
            dst: pa.string(),
            "Label": pa.float64(),
            "Attack": pa.string(),
            start: pa.int64(),
            end: pa.int64(),
        }
    )
    with ExitStack() as stack:
        sinks = [stack.enter_context(p.open("wb")) for p in paths]
        binary = stack.enter_context(_source_stream(source))
        digest = _HashingReader(binary)
        reader = stack.enter_context(
            pacsv.open_csv(
                digest,
                read_options=pacsv.ReadOptions(block_size=8 * 1024**2),
                convert_options=pacsv.ConvertOptions(column_types=types, include_columns=list(types)),
            )
        )
        for batch in reader:
            timestamps = batch[end].to_numpy(zero_copy_only=False)
            starts = batch[start].to_numpy(zero_copy_only=False)
            if (
                batch[end].null_count
                or batch[start].null_count
                or (starts < 0).any()
                or (timestamps < starts).any()
            ):
                raise ValueError("invalid flow start/end timestamp; history extraction stopped")
            if not batch.num_rows:
                continue
            backwards += int(np.count_nonzero(timestamps[1:] < timestamps[:-1]))
            backwards += int(previous is not None and timestamps[0] < previous)
            previous = timestamps[-1]
            lo, hi = int(timestamps.min()), int(timestamps.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
            keys = (_addresses(batch, src).astype(np.uint64) << 8) | _integers(batch, "PROTOCOL", 255).astype(
                np.uint64
            )
            events = np.empty(batch.num_rows, dtype=EVENT)
            events["ms"], events["key"], events["dst"] = timestamps, keys, _addresses(batch, dst)
            events["port"] = _integers(batch, "L4_DST_PORT", 65535)
            events["no_reply"] = _integers(batch, "OUT_PKTS", 2**53 - 1) == 0
            events["target"] = -1
            events["record"] = np.arange(scanned + 1, scanned + batch.num_rows + 1, dtype=np.uint32)
            until = sorted_numbers.searchsorted(scanned + batch.num_rows, side="right")
            if until > checked:
                target = order[checked:until]
                offsets = sorted_numbers[checked:until] - scanned - 1
                events["target"][offsets] = target
                selected = targets.iloc[target]
                meta = _verify_target_batch(batch, offsets, selected)
                meta.insert(0, "row_id", selected.row_id.to_numpy())
                metadata.append(meta)
            scanned += batch.num_rows
            checked = until
            # Mix all address octets and the protocol (the low byte). A sheer
            # modulo of key would assign almost every TCP event to one shard.
            mixed = keys ^ (keys >> 23)
            mixed = (mixed * np.uint64(0x9E3779B97F4A7C15)) >> 32
            shards = mixed % len(paths)
            for slot, sink in enumerate(sinks):
                events[shards == slot].tofile(sink)
        actual = {"sha256": digest.sha256.hexdigest(), "bytes": digest.byte_count, "records": scanned}
    if actual != expected:
        raise ValueError(f"source identity mismatch: {actual}")
    if checked != len(targets):
        raise ValueError("target row number beyond end of CSV")
    info = {
        "source": actual,
        "checked_target_rows": checked,
        "checked_input_fields_per_row": len(lab.FEATURES),
        "earliest_completion_ms": minimum,
        "latest_completion_ms": maximum,
        "source_end_time_inversions": backwards,
    }
    return pd.concat(metadata, ignore_index=True), info


class _RollingState:
    """Time-window state for one independent stream of ordered completion batches."""

    def __init__(self):
        self.events = deque()
        self.groups = {}
        self.distinct_entries = 0

    def expire(self, instant):
        while self.events and self.events[0][0] < instant - WINDOW_MS:
            _, key, dst, port, no_reply = self.events.popleft()
            state = self.groups[key]
            state[0] -= 1
            state[1] -= no_reply
            for counter, value in ((state[2], dst), (state[3], port)):
                counter[value] -= 1
                if not counter[value]:
                    del counter[value]
                    self.distinct_entries -= 1
            if not state[0]:
                del self.groups[key]

    def features(self, key):
        state = self.groups.get(key)
        if state is None:
            return 0, 0, 0, 0.0
        return state[0], len(state[2]), len(state[3]), state[1] / state[0]

    def update(self, instant, key, dst, port, no_reply):
        state = self.groups.get(key)
        if state is None:
            state = self.groups[key] = [0, 0, Counter(), Counter()]
        self.events.append((instant, key, dst, port, no_reply))
        state[0] += 1
        state[1] += no_reply
        for counter, value in ((state[2], dst), (state[3], port)):
            self.distinct_entries += int(value not in counter)
            counter[value] += 1


def _sorted_records(events):
    for offset in range(0, len(events), 32768):
        yield from events[offset : offset + 32768].tolist()


def _replay(path):
    began = time.perf_counter()
    cpu_began = time.process_time()
    events = np.fromfile(path, dtype=EVENT)
    events = events[np.argsort(events["ms"], kind="stable")]
    started = time.perf_counter()
    state = _RollingState()
    results = []
    peak_records = peak_keys = peak_entries = largest_tie = 0
    batches = seen = 0
    process = psutil.Process()
    sorted_rss = sampled_rss = process.memory_info().rss
    next_sample = 65536
    for instant, simultaneous in groupby(_sorted_records(events), key=lambda row: row[0]):
        batch = list(simultaneous)
        state.expire(instant)
        if state.events and not instant - WINDOW_MS <= state.events[0][0] <= state.events[-1][0] < instant:
            raise RuntimeError("history contains out-of-window or simultaneous completions")
        batches += 1
        for _, key, _, _, _, target, _ in batch:
            values = state.features(key)
            if target >= 0:
                results.append((target, *values))
        for _, key, dst, port, no_reply, _, _ in batch:
            state.update(instant, key, dst, port, no_reply)
        peak_records = max(peak_records, len(state.events))
        peak_keys = max(peak_keys, len(state.groups))
        peak_entries = max(peak_entries, state.distinct_entries)
        largest_tie = max(largest_tie, len(batch))
        seen += len(batch)
        if seen >= next_sample:
            sampled_rss = max(sampled_rss, process.memory_info().rss)
            next_sample = seen + 65536
    elapsed = time.perf_counter() - started
    mem = process.memory_info()
    keys, counts = np.unique(events["key"], return_counts=True)
    top = int(np.argmax(counts)) if len(counts) else None
    audit = {
        "records": len(events),
        "targets": len(results),
        "sort_read_seconds": started - began,
        "replay_seconds": elapsed,
        "cpu_seconds_including_sort": time.process_time() - cpu_began,
        "replay_rows_per_second": len(events) / elapsed if elapsed else None,
        "strict_window_batches_checked": batches,
        "rss_after_offline_sort_bytes": sorted_rss,
        "sampled_replay_peak_process_rss_bytes": max(sampled_rss, mem.rss),
        "peak_window_records": peak_records,
        "peak_active_source_protocol_keys": peak_keys,
        "peak_distinct_address_plus_port_counter_entries": peak_entries,
        "largest_same_ms_batch": largest_tie,
        "most_frequent_key": int(keys[top]) if top is not None else None,
        "most_frequent_key_rows": int(counts[top]) if top is not None else 0,
        "worker_rss_at_finish_bytes": mem.rss,
        "worker_process_lifetime_peak_wset_bytes": getattr(mem, "peak_wset", None),
    }
    return results, audit


def _checkpointed_replay(job):
    path, receipt_path = job
    if receipt_path.exists():
        try:
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload = saved["payload"]
            if saved["sha256"] == canonical_hash(payload):
                return payload["rows"], {**payload["stats"], "reused_shard": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass  # A torn or corrupt shard is rebuilt from the verified input.
    rows, stats = _replay(path)
    payload = {"stats": stats, "rows": rows}
    write_json(receipt_path, {"payload": payload, "sha256": canonical_hash(payload)})
    return rows, {**stats, "reused_shard": False}


def _distinct_in_window(times, values, query):
    """Count active value lifetimes (t, t+W] at each query, independent of replay state."""
    order = np.lexsort((times, values))
    t, val = times[order], values[order]
    breaks = (val[1:] != val[:-1]) | (np.diff(t) > WINDOW_MS)
    starts = np.sort(t[np.r_[True, breaks]])
    ends = np.sort(t[np.r_[breaks, True]] + WINDOW_MS)
    return starts.searchsorted(query, side="left") - ends.searchsorted(query, side="left")


def _benchmark_rows(source, records, expected_digest):
    """Read real base features for the benchmark's timed records; they are not de-normalized approximations."""
    columns = [*lab.FEATURES, "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_END_MILLISECONDS"]
    types = {k: pa.float64() for k in lab.FEATURES}
    types.update(
        {"IPV4_SRC_ADDR": pa.string(), "IPV4_DST_ADDR": pa.string(), "FLOW_END_MILLISECONDS": pa.int64()}
    )
    selected = []
    ordered = np.sort(records)
    checked = scanned = 0
    with _source_stream(source) as stream:
        digest = _HashingReader(stream)
        with pacsv.open_csv(
            digest,
            read_options=pacsv.ReadOptions(block_size=8 * 1024**2),
            convert_options=pacsv.ConvertOptions(column_types=types, include_columns=columns),
        ) as reader:
            for batch in reader:
                until = ordered.searchsorted(scanned + batch.num_rows, side="right")
                if until > checked:
                    ids = ordered[checked:until]
                    rows = batch.take(pa.array(ids.astype(np.int64) - scanned - 1)).to_pandas()
                    rows.insert(0, "source_record", ids)
                    selected.append(rows)
                checked = until
                scanned += batch.num_rows
        if checked != len(records) or digest.sha256.hexdigest() != expected_digest:
            raise ValueError("benchmark raw input differs from source staging")
    return pd.concat(selected, ignore_index=True).set_index("source_record").loc[records].reset_index()


def _prepare_benchmark(source, work, shards, audits, source_sha):
    # Preselected independent of labels and predictions: busiest (source, protocol),
    # then numeric key for ties. Every arrival in the chosen timed interval is scored.
    _, key, slot = min(
        (-a["most_frequent_key_rows"], a["most_frequent_key"], i)
        for i, a in enumerate(audits)
        if a["records"]
    )
    mapped = np.memmap(shards[slot], mode="r", dtype=EVENT)
    seq = mapped[mapped["key"] == key]
    del mapped
    seq = seq[np.argsort(seq["ms"], kind="stable")]
    t = seq["ms"]
    target_size = 512
    has_usable_past = (np.diff(t) > 0) & (np.diff(t) <= WINDOW_MS)
    eligible = np.flatnonzero(np.r_[False, has_usable_past] & (t >= t[0] + WINDOW_MS))
    recipe = {
        "selection": "highest complete-source record count per (source IPv4, protocol), lowest key on ties; "
        "first completion with 60s of source coverage and nonempty past, 512 consecutive arrivals plus all end-time ties",
        "source_sha256": source_sha,
        "chosen_key_records": len(seq),
        "scope": "single busiest source/protocol; "
        "time-ordered state and raw model inputs, offline end-time watermark assumption",
    }
    if not len(eligible) or len(t) - eligible[0] < target_size:
        return {**recipe, "status": "unavailable", "reason": "insufficient warmed contiguous flow stream"}
    begin = int(eligible[0])
    end = int(t.searchsorted(t[begin + target_size - 1], side="right"))
    first = int(t.searchsorted(t[begin] - WINDOW_MS, side="left"))
    segment = seq[first:end]
    warm = begin - first
    meta = pd.DataFrame(
        {
            "source_record": segment["record"],
            "src_key": segment["key"],
            "dst_key": segment["dst"],
            "history_dst_port": segment["port"],
            "history_no_reply": segment["no_reply"],
            "flow_end_ms": segment["ms"],
        }
    )
    warmup = meta.iloc[:warm].copy()
    measured = meta.iloc[warm:].reset_index(drop=True)
    raw = _benchmark_rows(source, measured.source_record.to_numpy(), source_sha)
    keys = [(_ip_code(src) << 8) | int(proto) for src, proto in zip(raw.IPV4_SRC_ADDR, raw.PROTOCOL)]
    if (
        not np.array_equal(measured.src_key, keys)
        or not np.array_equal(measured.dst_key, [_ip_code(dst) for dst in raw.IPV4_DST_ADDR])
        or not np.array_equal(measured.history_dst_port, raw.L4_DST_PORT)
        or not np.array_equal(measured.history_no_reply, raw.OUT_PKTS == 0)
        or not np.array_equal(measured.flow_end_ms, raw.FLOW_END_MILLISECONDS)
    ):
        raise ValueError("benchmark source identity or time mismatch")
    t = segment["ms"]
    query = t[warm:]
    lo, hi = t.searchsorted(query - WINDOW_MS), t.searchsorted(query)
    counts = hi - lo
    no_reply = np.r_[0, np.cumsum(segment["no_reply"], dtype=np.int64)]
    measured[HISTORY_COLUMNS[0]] = counts
    measured[HISTORY_COLUMNS[1]] = _distinct_in_window(t, segment["dst"], query)
    measured[HISTORY_COLUMNS[2]] = _distinct_in_window(t, segment["port"], query)
    measured[HISTORY_COLUMNS[3]] = np.divide(
        no_reply[hi] - no_reply[lo], counts, out=np.zeros(len(counts), dtype=float), where=counts > 0
    )
    measured = pd.concat([measured, raw[list(lab.FEATURES)]], axis=1)
    for fname, frame in [("benchmark_warmup.parquet", warmup), ("benchmark_stream.parquet", measured)]:
        target = work / fname
        temporary = target.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
    return {
        **recipe,
        "status": "ready",
        "warmup_rows": len(warmup),
        "timed_rows": len(measured),
        "timed_start_ms": int(query[0]),
        "timed_end_ms": int(query[-1]),
        "history_lower_inclusive_ms": int(query[0]) - WINDOW_MS,
        "benchmark_warmup_sha256": sha256_file(work / "benchmark_warmup.parquet"),
        "benchmark_stream_sha256": sha256_file(work / "benchmark_stream.parquet"),
        "reference": "time bounds + merged active intervals per destination IP/port; independent of deque replay",
    }


def _extract(source, targets, work, threads, expected):
    source, work = Path(source), Path(work)
    if type(threads) is not int or threads < 1:
        raise ValueError("positive integer history worker count required")
    missing = {"row_id", "label", "attack_family", *lab.FEATURES} - set(targets)
    if missing:
        raise ValueError(f"missing target alignment columns: {sorted(missing)}")
    targets = targets.reset_index(drop=True)
    numbers = _target_numbers(targets, expected["sha256"])
    work.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    output = np.empty((len(targets), len(HISTORY_COLUMNS)), dtype=np.float64)
    filled = np.zeros(len(targets), dtype=bool)
    audits = []
    # A shard stores only the selected rows and scalar timing evidence. Recover
    # this work across attempts; every invocation revalidates source and targets.
    identity = canonical_hash(
        [
            expected,
            hashlib.sha256(numbers.astype("<i8").tobytes()).hexdigest(),
            sha256_file(Path(__file__)),
            threads,
        ]
    )
    progress = work / ".history-progress" / identity
    progress.mkdir(parents=True, exist_ok=True)
    receipts = [progress / f"shard-{i}.json" for i in range(threads)]
    with TemporaryDirectory(prefix="history-", dir=work) as temporary:
        shards = [Path(temporary) / f"source-{i}.bin" for i in range(threads)]
        meta, audit = _stage(source, targets, numbers, shards, expected)
        stage_finished = time.perf_counter()
        with ExitStack() as stack:
            jobs = list(zip(shards, receipts))
            if threads > 1:
                pool = stack.enter_context(ProcessPoolExecutor(max_workers=threads))
                completed = pool.map(_checkpointed_replay, jobs)
            else:
                completed = map(_checkpointed_replay, jobs)
            for values, receipt in completed:
                for slot, *features in values:
                    if filled[slot]:
                        raise ValueError("target emitted twice")
                    output[slot] = features
                    filled[slot] = True
                audits.append(receipt)
        replay_finished = time.perf_counter()
        benchmark = _prepare_benchmark(source, work, shards, audits, expected["sha256"])
        write_json(work / "benchmark_segment.json", benchmark)
        benchmark_finished = time.perf_counter()
    if not filled.all() or sum(a["records"] for a in audits) != expected["records"]:
        raise ValueError("history replay incomplete")
    result = pd.DataFrame(output, columns=HISTORY_COLUMNS)
    result.insert(0, "row_id", targets.row_id.to_numpy())
    meta = meta.set_index("row_id").loc[result.row_id].reset_index()
    meta_path = work / "target_metadata.parquet"
    pending = meta_path.with_suffix(".parquet.tmp")
    meta.to_parquet(pending, index=False)
    pending.replace(meta_path)
    parallel_seconds = replay_finished - stage_finished
    computed = [item for item in audits if not item["reused_shard"]]
    audit.update(
        {
            "features": list(HISTORY_COLUMNS),
            "group_key": ["IPV4_SRC_ADDR", "PROTOCOL"],
            "unit": "completed flow, assuming available at recorded end time",
            "window": "[end_ms - 60000, end_ms); ties queried before any updates",
            "empty_history_target_rows": int(np.count_nonzero(output[:, 0] == 0)),
            "benchmark_segment_status": benchmark["status"],
            "benchmark_segment_preparation_seconds": benchmark_finished - replay_finished,
            "labels_used_in_history": False,
            "shards": [{k: v for k, v in a.items() if k != "most_frequent_key"} for a in audits],
            "workers_requested": threads,
            "workers_with_records": sum(a["records"] > 0 for a in audits),
            "shards_reused_after_revalidating_source": len(audits) - len(computed),
            "strict_window_batches_checked": sum(a["strict_window_batches_checked"] for a in audits),
            "csv_parse_and_staging_seconds": stage_finished - began,
            "parallel_replay_wall_seconds_including_sort_and_processes": parallel_seconds,
            "computed_replay_rows_per_wall_second": sum(a["records"] for a in computed) / parallel_seconds,
            "computed_worker_cpu_seconds_per_wall_second": sum(
                a["cpu_seconds_including_sort"] for a in computed
            )
            / parallel_seconds,
            "end_to_end_seconds_before_receipt": time.perf_counter() - began,
            "staged_bytes": expected["records"] * EVENT.itemsize,
            "per_shard_peak_scope": "exact cardinalities for the 60-second state; process RSS also includes offline sort buffers",
            "feature_cost_scope": "each completed flow queried and updated; offline CSV staging, sort and worker startup separately reported",
            "model_inputs_exclude": ["raw IPs", "absolute time", "labels", "row_id"],
        }
    )
    write_json(work / "history_audit.json", audit)
    return result


def extract_history(source: Path, targets: pd.DataFrame, work: Path, threads: int) -> pd.DataFrame:
    """Verify the official source and all target inputs, then emit the four history columns."""
    return _extract(
        source,
        targets,
        work,
        threads,
        {"sha256": SOURCE_SHA256, "bytes": lab.SOURCE_BYTES, "records": SOURCE_RECORDS},
    )
