"""Small synthetic aggregation benchmark; no model, training, or source data access."""

import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import pandas as pd
import psutil

from ids.readiness_replay.evaluation import evaluate_frames, prepare_context

POLICIES = (
    "raw_old",
    "contact_old",
    "quality_old",
    "add_quality_old",
    "ports_local",
    "ports_old",
    "targets_local",
    "targets_old",
    "counts_local",
    "counts_old",
    "raw_local",
    "quality_local",
    "quality_counts_local",
    "quality_counts_old",
    "raw_quality_counts_local",
    "frozen_plus_ports_local",
    "frozen_plus_counts_local",
    "no_reply_scan_proxy",
)


def benchmark(size, scan_heavy):
    rng = np.random.default_rng(1425)
    ids = np.arange(1, size + 1, dtype=np.int64)
    clock = 1736121600000 + np.arange(size) // 3 * 100
    probabilities = (0.25, 0.60, 0.10, 0.05) if scan_heavy else (0.94, 0.02, 0.03, 0.01)
    families = rng.choice(["Benign", "Scanning", "OtherAttack", "Unresolved"], size=size, p=probabilities)
    families[:45] = "ExternalC2"
    families[45:61] = "Download"
    features = pd.DataFrame(
        dict(
            row_id=ids,
            flow_end_ms=clock,
            ip_protocol=6,
            resp_port=rng.choice([22, 53, 80, 443, 445, 8080], size),
            orig_port=40000 + ids % 20000,
            duration_seconds=rng.random(size),
            hist_dst_ports_60s=ids % 32,
            hist_dst_ips_60s=ids % 12,
        )
    )
    truth = pd.DataFrame(
        dict(
            row_id=ids,
            uid=["S" + str(i) for i in ids],
            window="2025-01-06",
            family=families,
            native_detail=families,
            event_id=np.where(
                families == "Scanning", "scan1", np.where(families == "OtherAttack", "other", "")
            ),
            evidence_level="synthetic",
            exposure="synthetic",
        )
    )
    scores = pd.DataFrame(
        dict(
            row_id=ids,
            score_raw=rng.random(size),
            score_contact=rng.random(size),
            score_quality=rng.random(size),
        )
    )
    flags = rng.random((size, len(POLICIES))) < 0.02
    old = pd.DataFrame(dict(row_id=ids, flow_end_ms=clock, phase=np.where(ids <= 130, "startup", "after")))
    old = pd.concat([old, pd.DataFrame(flags, columns=POLICIES)], axis=1)
    new = old.copy()
    new["phase"] = np.where(ids <= 399, "startup", "after")
    for name in POLICIES:
        if name.endswith("_local"):
            new.loc[130:398, name] = ~new.loc[130:398, name]
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.025):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start, cpu = time.perf_counter(), time.process_time()
    context = prepare_context(features, truth)
    prepared = dict(wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
    records = []
    root = Path(__file__).resolve().parents[2] / "artifacts/readiness-replay-synthetic"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix="evaluation-") as temporary:
        for group in ("synthetic1", "synthetic2"):
            start, cpu = time.perf_counter(), time.process_time()
            paths = evaluate_frames(
                features,
                scores,
                truth,
                old,
                new,
                source=dict(name="synthetic", role="synthetic"),
                group=group,
                output=Path(temporary) / group,
                context=context,
            )
            records.append(
                dict(
                    group=group,
                    wall_seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu,
                    output_bytes=sum(p.stat().st_size for p in paths),
                )
            )
    stop.set()
    thread.join()
    return dict(
        rows=size,
        case="scan_heavy" if scan_heavy else "normal_heavy",
        policies=len(POLICIES),
        context=prepared,
        groups=records,
        peak_process_rss_bytes=peak[0],
        rare_rows=61,
        aggregation_groups={k: len(v.labels) for k, v in context.axes.items()},
    )


if __name__ == "__main__":
    report = dict(
        scope="Synthetic sealed-flag aggregation and file writing only; no models, fit, real source reads or GPU.",
        cases=[benchmark(100000, False), benchmark(500000, True)],
    )
    dest = Path(__file__).with_name("synthetic_evaluation_benchmark.json")
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
