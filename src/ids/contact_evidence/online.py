"""Independent online reference and deployed row-cost probe for the registered Zeek observations."""

from collections import Counter, defaultdict, deque
from itertools import islice
from time import perf_counter_ns

import numpy as np
import psutil
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ids.flow_history.features import WINDOW_MS
from .features import ARMS, CHUNK, RAW_FEATURES


def _triple():
    return [0, 0, 0]  # flows, known responder counts, positive responder counts


def _adjust(table, key, amounts):
    values = table[key]
    for i, amount in enumerate(amounts):
        values[i] += amount
    if not values[0]:
        del table[key]


def _reply(table, key):
    values = table.get(key)
    if values is None:
        return 0, -1.0
    count, known, answered = values
    return count, answered / known if known else -1.0


class OnlineHistory:
    """Retain the current millisecond separately; count equal endpoint tuples once per millisecond."""

    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.arm = arm
        self.wants_service = arm in {"service", "contact"}
        self.wants_contact = arm == "contact"
        self.clock = None
        self.bins = deque()
        self.retained_time_keys = 0
        self.pending = defaultdict(_triple)
        self.base = defaultdict(_triple)
        self.peers = defaultdict(Counter)
        self.ports = defaultdict(Counter)
        self.services = defaultdict(_triple)
        self.contacts = defaultdict(_triple)

    def _apply(self, bucket, direction):
        for (src, proto, dst, port), stats in bucket.items():
            key = (src, proto)
            amounts = stats if direction == 1 else [-v for v in stats]
            _adjust(self.base, key, amounts)
            peer_counts = self.peers[key]
            peer_counts[dst] += amounts[0]
            if not peer_counts[dst]:
                del peer_counts[dst]
            if not peer_counts:
                del self.peers[key]
            if port < 0:
                continue
            port_counts = self.ports[key]
            port_counts[port] += amounts[0]
            if not port_counts[port]:
                del port_counts[port]
            if not port_counts:
                del self.ports[key]
            if self.wants_service:
                _adjust(self.services, (src, proto, port), amounts)
            if self.wants_contact:
                _adjust(self.contacts, (src, proto, port, dst), amounts)

    def _advance(self, instant):
        if self.clock == instant:
            return
        if self.clock is not None and instant < self.clock:
            raise ValueError("Observation time moved backwards")
        lower = instant - WINDOW_MS
        while self.bins and self.bins[0][0] < lower:
            _, expired = self.bins.popleft()
            self._apply(expired, -1)
            self.retained_time_keys -= len(expired)
        if self.clock is not None and self.clock >= lower:
            self._apply(self.pending, 1)
            self.bins.append((self.clock, self.pending))
            self.retained_time_keys += len(self.pending)
        self.pending = defaultdict(_triple)
        self.clock = instant

    def _put(self, src, proto, dst, port, received):
        group = self.pending[(src, proto, dst, port)]
        group[0] += 1
        group[1] += received >= 0
        group[2] += received > 0

    def warm(self, instant, src, proto, dst, port, received):
        if self.arm != "raw":
            self._advance(instant)
            self._put(src, proto, dst, port, received)

    def observe(self, instant, src, proto, dst, port, received):
        if self.arm == "raw":
            return ()
        self._advance(instant)
        key = (src, proto)
        count, known, answered = self.base.get(key, (0, 0, 0))
        no_reply = (known - answered) / known if known else (-1.0 if count else 0.0)
        values = (count, len(self.peers.get(key, ())), len(self.ports.get(key, ())), no_reply)
        if self.wants_service:
            values += _reply(self.services, (src, proto, port))
        if self.wants_contact:
            values += _reply(self.contacts, (src, proto, port, dst))
        self._put(src, proto, dst, port, received)
        return values


def _range(table, columns, predicate):
    scanner = table.scanner(
        columns=list(columns),
        filter=predicate,
        batch_size=CHUNK,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
    )
    for batch in scanner.to_batches():
        yield from zip(*(col.to_pylist() for col in batch.columns))


def benchmark_online(path, score_one, arm, *, rows=96, threshold=None):
    """Three contiguous sample blocks. score_one takes one 1-D feature vector and returns a scalar."""
    if arm not in ARMS or rows < 1:
        raise ValueError("Unknown arm or invalid sample size")
    n = pq.read_metadata(path).num_rows
    if not n:
        raise ValueError("Empty cost capture")
    rows = min(n, rows)
    # Split the timeline into up to three disjoint regions and measure a middle block in each.
    blocks = min(3, rows)
    sample_sizes = np.diff(np.linspace(0, rows, blocks + 1, dtype=int))
    regions = np.linspace(0, n, blocks + 1, dtype=int)
    source = ds.dataset(path, format="parquet")
    meta_names = ["flow_end_ms", "source", "native_protocol", "destination", "resp_port", "resp_pkts"]
    lengths = []
    largest_temporal_state = 0
    alerts = []
    traces = []
    checked_scores = 0
    worst_error = 0.0
    process = psutil.Process()
    rss_before = process.memory_info().rss
    rss_observed = [rss_before]
    largest_state = {"source_protocols": 0, "service_keys": 0, "contact_keys": 0, "pending_contacts": 0}
    # Scan columns individually once per block; time feature construction, model and any threshold check.
    # Feature-row decoding, file access and arriving at the logical clock are outside the reported latency.
    columns = list(dict.fromkeys(["row_id", *meta_names, *ARMS[arm]]))
    pos = {name: i for i, name in enumerate(columns)}
    row_id = ds.field("row_id")
    clock = ds.field("flow_end_ms")
    for size, start, end in zip(sample_sizes, regions[:-1], regions[1:]):
        first = int(start + (end - start - size) // 2 + 1)
        selected = list(
            islice(_range(source, columns, (row_id >= first) & (row_id < first + int(size))), int(size) + 1)
        )
        if [row[pos["row_id"]] for row in selected] != list(range(first, first + int(size))):
            raise ValueError("Cost sample identity changed")
        begin_time = selected[0][pos["flow_end_ms"]]
        engine = OnlineHistory(arm)
        warm_rows = 0
        if arm != "raw":
            warm = (clock >= begin_time - WINDOW_MS) & (clock <= begin_time) & (row_id < first)
            for event in _range(source, meta_names, warm):
                engine.warm(*event)
                warm_rows += 1
        rss_observed.append(process.memory_info().rss)
        # Warm model code paths and capture the independent batch-feature predictions outside timing.
        stored = [np.array([row[pos[field]] for field in ARMS[arm]], dtype=np.float64) for row in selected]
        reference = [float(score_one(vector)) for vector in stored]
        for row, expected, ref in zip(selected, stored, reference):
            metadata = tuple(row[pos[field]] for field in meta_names)
            raw = tuple(row[pos[field]] for field in RAW_FEATURES)
            then = perf_counter_ns()
            extras = engine.observe(*metadata)
            vector = np.asarray(raw + extras, dtype=np.float64)
            score = float(score_one(vector))
            alert = score > threshold if threshold is not None else None
            lengths.append(perf_counter_ns() - then)
            if not np.array_equal(vector, expected):
                raise ValueError(f"Online history mismatch for {arm}, row {row[pos['row_id']]}")
            if not np.isfinite(score) or not np.isclose(score, ref, rtol=1e-6, atol=1e-6):
                raise ValueError("Online and batch-feature model scores differ")
            if threshold is not None and alert != (ref > threshold):
                raise ValueError("Online and batch-feature alerts differ")
            worst_error = max(worst_error, abs(score - ref))
            largest_temporal_state = max(
                largest_temporal_state, engine.retained_time_keys + len(engine.pending)
            )
            if threshold is not None:
                alerts.append(bool(alert))
            checked_scores += 1
            for key, mapping in (
                ("source_protocols", engine.base),
                ("service_keys", engine.services),
                ("contact_keys", engine.contacts),
                ("pending_contacts", engine.pending),
            ):
                largest_state[key] = max(largest_state[key], len(mapping))
        rss_observed.append(process.memory_info().rss)
        traces.append({"first_row_id": first, "rows": len(selected), "warm_rows": warm_rows})
    times = np.asarray(lengths, dtype=np.int64)
    return {
        "arm": arm,
        "rows": checked_scores,
        "p50_ms": float(np.quantile(times, 0.50) / 1e6),
        "p95_ms": float(np.quantile(times, 0.95) / 1e6),
        "latencies_ns": lengths,
        "score_max_abs_error": worst_error,
        "checked_alerts": threshold is not None,
        **({"alerts": alerts} if threshold is not None else {}),
        "blocks": traces,
        "largest_sampled_state": {**largest_state, "keys_at_distinct_times": largest_temporal_state},
        "rss_before_bytes": rss_before,
        "max_rss_at_block_boundaries_bytes": max(rss_observed),
        "scope": "One decoded completed record through historical state, feature assembly, scorer and optional threshold",
    }
