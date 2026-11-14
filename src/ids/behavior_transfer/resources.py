"""Bounded CPU/GPU resource measurements using frozen or already prepared observations."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, perf_counter_ns

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

from ids.common import read_json, sha256_file, write_json
from ids.contact_evidence.experiment import NEURAL_GROUPS
from ids.contact_evidence.online import OnlineHistory
from ids.flow_history import model
from ids.raw_workpoint import controls
from ids.raw_workpoint.scoring import VIEWS, make_decisions, register
from ids.response_evidence.experiment import ARMS
from ids.response_evidence.history import enrich_prepared
from ids.threshold_validation.observations import HISTORY_FEATURES
from .policy import register as register_counts

GPU_ROWS = 65_536
GPU_BATCHES = (4096, 16384, 65536)
REPEATS = 3
HISTORY_ROWS = 50_000


def _online_counts(frame, expected):
    """Measure the existing online state and the two registered count decisions."""
    reference = controls.CONFIG / "early-normal-features.parquet"
    if sha256_file(reference) != read_json(controls.CONFIG / "controls.json")["calibration_features_sha256"]:
        raise ValueError("Online count calibration changed")
    names = ("hist_dst_ports_60s", "hist_dst_ips_60s")
    rules = register_counts(pd.read_parquet(reference, columns=list(names)), names)
    cuts = {name: rules[name]["threshold"] for name in names}
    columns = ("flow_end_ms", "source", "native_protocol", "destination", "resp_port", "resp_pkts")
    events = list(zip(*(frame[name].to_pylist() for name in columns), strict=True))
    vectors = np.column_stack([expected[name].to_numpy() for name in HISTORY_FEATURES])
    # Warm code paths on an independent state, then time the full cold-start prefix.
    warm = OnlineHistory("history")
    for event in events[:64]:
        warm.observe(*event)
    del warm
    engine = OnlineHistory("history")
    process = psutil.Process()
    before = process.memory_info().rss
    peak = before
    latencies = []
    flags = {"ports": 0, "targets": 0, "union": 0}
    state = dict(source_protocol_keys=0, keys_at_distinct_times=0, pending_endpoint_keys=0)
    with threadpool_limits(limits=1):
        for i, event in enumerate(events):
            start = perf_counter_ns()
            vector = engine.observe(*event)
            ports = cuts[names[0]] is not None and vector[2] > cuts[names[0]]
            targets = cuts[names[1]] is not None and vector[1] > cuts[names[1]]
            combined = ports or targets
            latencies.append(perf_counter_ns() - start)
            if not np.array_equal(vector, vectors[i]):
                raise ValueError(f"Online count state differs from batch history at prefix row {i + 1}")
            flags["ports"] += int(ports)
            flags["targets"] += int(targets)
            flags["union"] += int(combined)
            state["source_protocol_keys"] = max(state["source_protocol_keys"], len(engine.base))
            state["keys_at_distinct_times"] = max(
                state["keys_at_distinct_times"], engine.retained_time_keys + len(engine.pending)
            )
            state["pending_endpoint_keys"] = max(state["pending_endpoint_keys"], len(engine.pending))
            if i % 1024 == 0:
                peak = max(peak, process.memory_info().rss)
    peak = max(peak, process.memory_info().rss)
    elapsed = np.asarray(latencies, dtype=np.int64) / 1e6
    return {
        "rows": len(events),
        "arm": "OnlineHistory(history)",
        "native_threads": 1,
        "p50_ms": float(np.median(elapsed)),
        "p95_ms": float(np.quantile(elapsed, 0.95)),
        "max_ms": float(elapsed.max()),
        "total_timed_ms": float(elapsed.sum()),
        "exact_history_columns": list(HISTORY_FEATURES),
        "exact_rows": len(events),
        "rules": rules,
        "alert_counts": flags,
        "maximum_retained_state": state,
        "rss_before_bytes": before,
        "sampled_peak_process_rss_bytes": peak,
        "rss_sample_spacing_rows": 1024,
        "scope": "One decoded record through existing 60-second OnlineHistory state, both count strict thresholds, and OR. No parsing, model inference or response-quality histories.",
        "reference": "Equal 0.5% allocations on older normal C; observed row alerts counted, no new labels.",
        "memory_scope": "Process RSS includes previously allocated input and vectorized history tables; sampled every 1024 rows and after the loop.",
    }


def _gpu_reference(root, group):
    probe = root / "evidence/probes" / (group + ".json")
    prior = read_json(probe)
    if not all(v["within_score_tolerance"] for v in prior["measured"].values()):
        raise ValueError("The earlier normal replay must pass before throughput measurement")
    reference = controls.CONFIG / "early-normal-features.parquet"
    if sha256_file(reference) != read_json(controls.CONFIG / "controls.json")["calibration_features_sha256"]:
        raise ValueError("Early-normal throughput input changed")
    calibration_dir = root / "local/controls/evidence/calibration"
    c = pd.read_parquet(calibration_dir / (group + ".parquet"))
    frame = pd.read_parquet(reference)
    if not frame[["capture", "row_id"]].equals(c[["capture", "row_id"]]):
        raise ValueError("Throughput observations do not match earlier calibration identities")
    rules, supplements = register(calibration_dir, group)
    index = np.arange(GPU_ROWS) % len(frame)
    return (
        frame.iloc[index].reset_index(drop=True),
        {v: c[v].to_numpy()[index] for v in VIEWS},
        rules,
        supplements,
        sha256_file(probe),
        len(frame),
    )


def gpu_batch_probe(root, group, threads):
    """Select only a numerically equivalent batch size on repeated early normal records."""
    root = Path(root)
    if group not in NEURAL_GROUPS or threads < 1 or not torch.cuda.is_available():
        raise ValueError("GPU batch probe requires a registered MLP, positive CPU threads and CUDA")
    frame, archived, rules, extra, prior_digest, unique_rows = _gpu_reference(root, group)
    models = {
        v: model.load_arm(
            root / "local/controls/models/frozen" / (group + "-" + v + ".pt"),
            threads=threads,
            device="cuda:0",
        )
        for v in VIEWS
    }
    device = next(iter(models.values())).device
    results, predictions = {}, {}
    for batch in GPU_BATCHES:
        for m in models.values():
            m.score_batch = batch
        times = {v: [] for v in VIEWS}
        values = {}
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(2):
            for view in VIEWS:
                model.score_arm(models[view], frame, ARMS[view])
        for repeat in range(REPEATS):
            shift = repeat % len(VIEWS)
            for view in VIEWS[shift:] + VIEWS[:shift]:
                torch.cuda.synchronize(device)
                started = perf_counter()
                values[view] = model.score_arm(models[view], frame, ARMS[view])
                torch.cuda.synchronize(device)
                times[view].append(perf_counter() - started)
        reference = values if batch == GPU_BATCHES[0] else predictions[GPU_BATCHES[0]]
        ref_flags = make_decisions(reference, rules, extra, frame)
        flags = make_decisions(values, rules, extra, frame)
        maximum_error = {v: float(np.abs(values[v] - reference[v]).max()) for v in VIEWS}
        close = all(np.allclose(values[v], reference[v], rtol=1e-5, atol=1e-5) for v in VIEWS)
        changes = {k: int(np.count_nonzero(flags[k] != ref_flags[k])) for k in flags}
        combined = sum(float(np.median(times[v])) for v in VIEWS)
        results[str(batch)] = {
            "scoring_seconds_by_view": times,
            "sum_of_view_median_seconds": combined,
            "three_view_rows_per_second": len(frame) / combined,
            "max_abs_score_delta_vs_4096": maximum_error,
            "changed_decisions_vs_4096": changes,
            "equivalent": bool(close and not any(changes.values())),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        predictions[batch] = values
    selected = min(
        (b for b in GPU_BATCHES if results[str(b)]["equivalent"]),
        key=lambda b: results[str(b)]["sum_of_view_median_seconds"],
    )
    old = make_decisions(archived, rules, extra, frame)
    chosen = make_decisions(predictions[selected], rules, extra, frame)
    path = root / "evidence/gpu-batch" / (group + ".json")
    write_json(
        path,
        {
            "group": group,
            "selected": selected,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "threads": threads,
            "rows": len(frame),
            "unique_reference_rows": unique_rows,
            "warmup_repetitions": 2,
            "timed_repetitions": REPEATS,
            "reference_probe_sha256": prior_digest,
            "candidates": results,
            "changed_decisions_vs_repeated_archived_C": {
                k: int(np.count_nonzero(chosen[k] != old[k])) for k in old
            },
            "scope": "Three frozen views; repeated earlier C observations; feature conversion, GPU transfer, forward pass and CPU score return. Each view is measured separately.",
            "selection": "Lowest sum of per-view median times among batches matching the 4096-score tolerance and all seven prior decision policies.",
            "numerical_tolerance": {"atol": 1e-5, "rtol": 1e-5},
        },
    )
    return [path]


def history_benchmark(source_file, output_folder, threads):
    """Compare serial and parallel builds of the same first 50,000 blind rows."""
    source_file, output_folder = Path(source_file), Path(output_folder)
    if threads < 1:
        raise ValueError("History threads must be positive")
    output_folder.mkdir(parents=True, exist_ok=True)
    with pq.ParquetFile(source_file) as reader:
        total = reader.metadata.num_rows
        if not total:
            raise ValueError("Empty history benchmark input")
        batch = next(reader.iter_batches(batch_size=HISTORY_ROWS))
        frame = pa.Table.from_batches([batch])
    with TemporaryDirectory(dir=output_folder, prefix="history-probe-") as td:
        scratch = Path(td)
        source, serial, parallel = [scratch / (v + ".parquet") for v in ("input", "serial", "parallel")]
        pq.write_table(frame, source, compression="zstd")
        input_digest = sha256_file(source)
        a = enrich_prepared(source, serial, workers=1)
        b = enrich_prepared(source, parallel, workers=threads)
        left, right = pq.read_table(serial), pq.read_table(parallel)
        if not left.equals(right, check_metadata=False):
            raise ValueError("Serial and parallel history differ")
        columns = left.column_names
        online = _online_counts(frame, left)
    path = output_folder / "history-benchmark.json"
    write_json(
        path,
        {
            "source": source_file.name,
            "source_rows": total,
            "prefix_rows": len(frame),
            "sample_parquet_sha256": input_digest,
            "selection": "First min(50000, source rows), physical row order",
            "serial": a,
            "parallel": b,
            "all_columns_identical": columns,
            "counts_online": online,
            "wall_speedup": a["seconds"] / b["seconds"],
            "scope": "Materialized blind log -> full 60-second history build, including scratch I/O and output serialization; labels unavailable.",
            "timing_order": "Single serial pass followed by single parallel pass; filesystem caches can affect elapsed time.",
        },
    )
    return [path]
