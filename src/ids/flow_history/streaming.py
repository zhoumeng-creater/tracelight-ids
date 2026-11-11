"""CPU flow-completion service cost on a fixed, independently replayable live-state segment."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from threadpoolctl import threadpool_info, threadpool_limits

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from .features import HISTORY_COLUMNS, WINDOW_MS, _RollingState


META = ["source_record", "src_key", "dst_key", "history_dst_port", "history_no_reply", "flow_end_ms"]


class PendingHistory:
    """Make completed records visible only upon the first strictly later arrival."""

    def __init__(self):
        self.window = _RollingState()
        self.pending = []
        self.instant = None

    def flush(self):
        for key, dst, port, no_reply in self.pending:
            self.window.update(self.instant, key, dst, port, no_reply)
        self.pending.clear()

    def advance(self, instant):
        if self.instant != instant:
            if self.instant is not None and self.instant > instant:
                raise ValueError("completion times are out of order")
            self.flush()
            self.window.expire(instant)
            self.instant = instant


def _warm(path, first_ms, expected_key):
    engine = PendingHistory()
    count = 0
    for batch in pq.ParquetFile(path).iter_batches(columns=META, batch_size=8192):
        for _, key, dst, port, no_reply, instant in batch.to_pandas().itertuples(index=False, name=None):
            if key != expected_key or not first_ms - WINDOW_MS <= instant < first_ms:
                raise ValueError("warm input violates segment key or time bounds")
            engine.advance(instant)
            engine.pending.append((key, dst, port, no_reply))
        count += batch.num_rows
    # Keep the last warm millisecond pending, as it would be across a live request boundary.
    return engine, count


def _stream_worker(request):
    req = read_json(request)
    out = Path(request).parent
    path, segment_path = Path(req["model"]), Path(req["segment"])
    if path.suffix.lower() == ".pt":
        from .model import load_arm, score_arm
        kind, kwargs = "mlp", {"device": "cpu", "threads": req["threads"]}
    elif path.suffix.lower() == ".joblib":
        from .trees import load_arm, score_arm
        kind, kwargs = "hgb", {"threads": req["threads"]}
    else:
        raise ValueError("expected a registered MLP .pt or HGB .joblib model")
    parent = segment_path.parent
    provenance = read_json(parent / "benchmark_segment.json")
    if provenance["status"] != "ready":
        write_json(out/"stream-benchmark.json", {"status": "unavailable", "kind": kind, "segment": provenance})
        return
    warm_file = parent / "benchmark_warmup.parquet"
    warm_hash, segment_hash = sha256_file(warm_file), sha256_file(segment_path)
    if (warm_hash != provenance["benchmark_warmup_sha256"] or segment_hash != provenance["benchmark_stream_sha256"]):
        raise ValueError("stream fixture differs from the extractor's registered files")
    segment = pd.read_parquet(segment_path)
    if not len(segment) or len(segment.src_key.unique()) != 1:
        raise ValueError("benchmark requires a nonempty single-key trace")
    times = segment.flow_end_ms.to_numpy(dtype=np.int64)
    records = segment.source_record.to_numpy(dtype=np.int64)
    if not segment.source_record.is_unique or not np.array_equal(np.lexsort((records, times)), np.arange(len(segment))):
        raise ValueError("trace must preserve completion order and entire millisecond groups")
    names = tuple(req["features"])
    historical = names == (*FEATURES, *HISTORY_COLUMNS)
    if not historical and names != FEATURES:
        raise ValueError("stream benchmark accepts raw or true-history model")
    expected = segment[list(HISTORY_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    raw = segment[list(FEATURES)].to_numpy(dtype=np.float64, copy=True)
    source_ips = segment.src_key.to_numpy(dtype=np.uint64) >> 8
    dest_ips = segment.dst_key.to_numpy(dtype=np.uint32)
    proto_i, port_i, out_packets_i = [FEATURES.index(col) for col in ("PROTOCOL", "L4_DST_PORT", "OUT_PKTS")]
    if (not np.array_equal((source_ips << 8) | raw[:, proto_i].astype(np.uint64), segment.src_key)
            or not np.array_equal(raw[:, port_i], segment.history_dst_port)
            or not np.array_equal(raw[:, out_packets_i] == 0, segment.history_no_reply)):
        raise ValueError("numeric flow fields differ from staged streaming metadata")
    process = psutil.Process()
    with threadpool_limits(limits=req["threads"]):
        model = load_arm(path, **kwargs)
        # Independent expected features from the extractor also bind the segment's batch scores.
        reference = score_arm(model, segment, names)
        registration = req["registration"]
        threshold_value = float(registration["threshold"])
        warm_rows = pq.ParquetFile(warm_file).metadata.num_rows
        if historical:
            engine, count = _warm(warm_file, int(times[0]), int(segment.src_key.iloc[0]))
            assert count == warm_rows
        # Model warm-up uses bounded precomputed vectors, and cannot change live history.
        for i in range(8):
            score_arm(model, segment.iloc[[i % len(segment)]], names)
        observed = np.empty_like(expected) if historical else None
        scores = np.empty(len(segment), dtype=float)
        alerts = np.empty(len(segment), dtype=bool)
        elapsed_ns = np.empty(len(segment), dtype=np.int64)
        rss_start = process.memory_info().rss
        samples = [rss_start]
        stop = threading.Event()

        def sample():
            while not stop.wait(.01):
                samples.append(process.memory_info().rss)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        started = time.perf_counter_ns()
        try:
            for i, values in enumerate(raw):
                then = time.perf_counter_ns()
                if historical:
                    # Input is a numeric NetFlow record with binary-form IPv4 addresses.
                    key = (int(source_ips[i]) << 8) | int(values[proto_i])
                    engine.advance(int(times[i]))
                    current = engine.window.features(key)
                    row = np.concatenate((values, current))
                    engine.pending.append((key, int(dest_ips[i]), int(values[port_i]), int(values[out_packets_i] == 0)))
                else:
                    row = values
                frame = pd.DataFrame(row.reshape(1, -1), columns=names)
                score = score_arm(model, frame, names)[0]
                alert = score > threshold_value
                elapsed_ns[i] = time.perf_counter_ns() - then
                scores[i], alerts[i] = score, alert
                if historical:
                    observed[i] = current
            if historical:
                engine.flush()
            wall_s = (time.perf_counter_ns() - started)/1e9
        finally:
            stop.set()
            sampler.join()
        samples.append(process.memory_info().rss)
        delta = np.abs(reference - scores)
        history_ok = bool(np.allclose(observed, expected, rtol=0, atol=1e-14)) if historical else None
        change = alerts != (reference > threshold_value)
        comparable = bool((delta <= 3e-5 + 3e-5 * np.abs(reference)).all() and not change.any())
        # Only nonidentifying evidence leaves the lab: per-event position, latency and predictions.
        evidence = pd.DataFrame({"source_record": records, "relative_ms": times-times[0],
                                 "service_ms": elapsed_ns/1e6, "score": scores, "reference_score": reference,
                                 "alert": alerts, "decision_changed": change})
        if historical:
            for j, col in enumerate(HISTORY_COLUMNS):
                evidence["observed_" + col] = observed[:, j]
                evidence["expected_" + col] = expected[:, j]
        evidence.to_csv(out / "stream-rows.csv.gz", index=False)
        write_json(out/"stream-threshold.json", registration)
        write_json(out/"stream-benchmark.json", {
            "status": "complete" if comparable and history_ok is not False else "replay_mismatch",
            "kind": kind, "device": "cpu", "threads": req["threads"], "native_threadpools": threadpool_info(),
            "torch_imported_in_worker": "torch" in sys.modules,
            "features": list(names), "history_enabled": historical, "history_matches_offline": history_ok,
            "max_abs_feature_difference": float(np.max(np.abs(observed - expected))) if historical else None,
            "scores_within_tolerance_and_alerts_unchanged": comparable,
            "max_abs_score_difference": float(delta.max()), "decision_changes": int(change.sum()),
            "warm_prior_key_rows": warm_rows, "measured_rows": len(segment),
            "measured_completion_ms_groups": len(np.unique(times)),
            "max_records_sharing_completion_ms": int(segment.flow_end_ms.value_counts().max()),
            "service_p50_ms": float(np.percentile(elapsed_ns, 50)/1e6),
            "service_p95_ms": float(np.percentile(elapsed_ns, 95)/1e6),
            "service_max_ms": float(elapsed_ns.max()/1e6), "segment_wall_s": wall_s,
            "throughput_rows_per_second": len(segment)/wall_s,
            "hot_worker_rss_before": rss_start, "hot_worker_sampled_peak_rss": max(samples),
            "rss_samples": len(samples), "rss_sampling_interval_ms": 10,
            "model_artifact_bytes": path.stat().st_size,
            "model_and_threshold_bytes": path.stat().st_size + (out/"stream-threshold.json").stat().st_size,
            "model_sha256": sha256_file(path), "segment_sha256": segment_hash,
            "warmup_sha256": warm_hash, "segment_receipt_sha256": sha256_file(parent/"benchmark_segment.json"),
            "input_scope": "ordered exported numeric flow + binary IPv4 addresses, one busy source/protocol key",
            "timing_scope": "synchronous one-row CPU prediction through strict alert; history arm includes "
                            "key formation, expiry, previous-tie insertion, query and current-tie buffering; "
                            "throughput includes final flush; no file I/O, upstream sort, exporter or network",
            "rss_scope": "whole warmed process including runtime, trace fixture, sampler and live state; "
                         "not per-host memory or whole-network capacity",
            "pid": os.getpid(),
        })


def bench_stream_arm(path, segment_path, features, threads, *, threshold_value, output_dir):
    """Launch in a fresh CPU process; compare raw/history on the identical recorded segment."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    reg = threshold_value if isinstance(threshold_value, dict) else {"threshold": threshold_value, "comparison": ">"}
    if reg["comparison"] != ">" or not np.isfinite(float(reg["threshold"])):
        raise ValueError("CPU registration requires a finite threshold and strict > comparison")
    request = out / "stream-request.json"
    write_json(request, {"model": str(Path(path).resolve()), "segment": str(Path(segment_path).resolve()),
                         "features": list(features), "threads": threads, "registration": reg})
    env = dict(os.environ)
    env.update({key: str(threads) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH"))))
    subprocess.run([sys.executable, "-m", "ids.flow_history.streaming", str(request)], env=env, check=True)
    return read_json(out/"stream-benchmark.json")


if __name__ == "__main__":
    _stream_worker(sys.argv[1])
