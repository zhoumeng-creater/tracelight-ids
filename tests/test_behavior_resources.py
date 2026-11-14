"""Synthetic measurements reject numerically different GPU batches and mismatched histories."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ids.behavior_transfer import resources
from ids.common import read_json
from ids.response_evidence.observations import GAP_COLUMNS, QUALITY_COLUMNS


def test_gpu_selects_fastest_equivalent_candidate_with_archived_reference(monkeypatch, tmp_path):
    n = 300
    f = pd.DataFrame(
        {
            "ip_protocol": 17,
            "orig_port": 42,
            "resp_port": 443,
            "duration_seconds": 0.0,
            "resp_ip_bytes": 0.0,
            "x": np.linspace(-2, 2, n),
        }
    )
    original = {v: f.x.to_numpy().copy() for v in resources.VIEWS}
    rules = {v: {"threshold": 0.5} for v in resources.VIEWS}
    extras = {v: {"threshold": 1.0} for v in ("contact", "quality")}
    monkeypatch.setattr(
        resources, "_gpu_reference", lambda root, group: (f, original, rules, extras, "hash", n)
    )
    monkeypatch.setattr(
        resources.model, "load_arm", lambda *a, **kw: SimpleNamespace(device="cuda:0", score_batch=0)
    )
    tick = {"seconds": 0.0, "batch": 4096}

    def score(arm, frame, features):
        tick["batch"] = arm.score_batch
        return frame.x.to_numpy() + (1.0 if arm.score_batch == 65536 else 0.0)

    monkeypatch.setattr(resources.model, "score_arm", score)

    def clock():
        tick["seconds"] += 1 / tick["batch"]
        return tick["seconds"]

    monkeypatch.setattr(resources, "perf_counter", clock)
    for method in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(resources.torch.cuda, method, lambda *a: None)
    monkeypatch.setattr(resources.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resources.torch.cuda, "get_device_name", lambda *a: "synthetic GPU")
    monkeypatch.setattr(resources.torch.cuda, "max_memory_allocated", lambda *a: 17)
    monkeypatch.setattr(resources.torch.cuda, "max_memory_reserved", lambda *a: 18)
    report = read_json(resources.gpu_batch_probe(tmp_path, "mlp29", 1)[0])
    assert report["selected"] == 16384
    assert not report["candidates"]["65536"]["equivalent"]
    assert report["candidates"]["65536"]["changed_decisions_vs_4096"]["raw"] > 0
    assert not any(report["changed_decisions_vs_repeated_archived_C"].values())
    assert report["gpu"] == "synthetic GPU"


def test_history_benchmark_real_serial_parallel_kernel_on_synthetic_records(tmp_path):
    n = 60
    f = pd.DataFrame(
        {
            "row_id": np.arange(1, n + 1),
            "flow_end_ms": np.arange(n) * 20000,
            "source": np.resize(["s1", "s2"], n),
            "destination": np.resize(["d1", "d2", "d3"], n),
            "native_protocol": "tcp",
            "resp_port": np.arange(n) % 4,
            "resp_pkts": np.arange(n) % 3,
            "response_evidence_code": np.resize([-1, 0, 1], n),
        }
    )
    for k in (*GAP_COLUMNS, *QUALITY_COLUMNS):
        f[k] = 0
    source = tmp_path / "input.parquet"
    f.to_parquet(source, index=False)
    report = read_json(resources.history_benchmark(source, tmp_path / "evidence", 4)[0])
    assert report["prefix_rows"] == report["source_rows"] == n
    assert report["serial"]["requested_workers"] == 1
    assert report["parallel"]["requested_workers"] == 4
    assert "hist_dst_ports_60s" in report["all_columns_identical"]
    assert report["wall_speedup"] > 0
    online = report["counts_online"]
    assert online["exact_rows"] == n
    assert online["native_threads"] == 1
    assert 0 < online["p50_ms"] <= online["p95_ms"]
    assert online["maximum_retained_state"]["keys_at_distinct_times"] > 0
    assert online["maximum_retained_state"]["source_protocol_keys"] > 0
