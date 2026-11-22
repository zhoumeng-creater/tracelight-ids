"""Isolated CPU inference measurements on fixed F rows, never training or selection."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "src")]

import numpy as np
import psutil
from scripts.calibration_repair import data
from scripts.comprehensive_batch.run import read, write, predictor_for


def measure(predictor, rows, *, clock=time.perf_counter):
    if len(rows) < 1024:
        raise ValueError("Registered CPU measurements use 1024 fixed training rows")
    result = {}
    for size, repetitions, warmups in ((1, 256, 16), (256, 10, 3), (1024, 5, 2)):
        for index in range(warmups):
            block = rows[(index * size) % (len(rows) - size + 1):][:size]
            predictor.score(block)
        timings = []
        for repeat in range(3):
            for index in range(repetitions):
                start = (index * size + repeat) % (len(rows) - size + 1)
                before = clock()
                values = np.asarray(predictor.score(rows[start:start + size]))
                elapsed = clock() - before
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError("Invalid inference measurement output")
                timings.append(elapsed)
        milliseconds = np.asarray(timings) * 1000
        result[str(size)] = {"batch_size": size, "rounds": 3, "calls": len(timings),
                             "warmup_calls": warmups, "p50_ms": float(np.median(milliseconds)),
                             "p95_ms": float(np.quantile(milliseconds, .95)),
                             "rows_per_second": len(timings) * size / sum(timings),
                             "seconds_per_call": timings}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    output = args.output.resolve()
    if not (output / "evaluation_complete.json").exists():
        raise RuntimeError("Measurements follow the completed fixed evaluation")
    meta = read(output / "models_frozen.json")["models"][args.snapshot]
    model_path = output / meta["path"]
    if data.file_digest(model_path) != meta["sha256"]:
        raise ValueError("Model modified before cost measurement")
    rows = read(output / "benchmark_rows.json")
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    predictor = predictor_for(model_path, meta, device="cpu")
    loaded = process.memory_info().rss
    timings = measure(predictor, rows)
    memory = process.memory_info()
    write(output / "benchmarks" / f"{args.snapshot}.json", {
        "snapshot_id": args.snapshot, "model_sha256": meta["sha256"],
        "model_bytes": model_path.stat().st_size, "device": "cpu", "threads": 4,
        "fresh_process": True, "baseline_rss_bytes": baseline, "loaded_rss_bytes": loaded,
        "final_rss_bytes": memory.rss, "peak_working_set_bytes": getattr(memory, "peak_wset", None),
        "batch_measurements": timings,
        "row_identity_sha256": data.json_digest([row["flow_id"] for row in rows]),
        "scope": "Feature dictionaries to fixed deployed scalar scores, including preprocessing; excludes packet capture, feature extraction and network alert delivery. F-only fixed sample, not an IID traffic workload guarantee."})


if __name__ == "__main__":
    main()
