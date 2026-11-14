"""Exercise the actual R25 orchestration, resume, decisions, labels and return ZIP with synthetic inputs."""

from concurrent.futures import Future
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer import runner, resources
from ids.common import read_json, write_json
from ids.raw_workpoint.scoring import VIEWS
from ids.response_evidence.observations import GAP_COLUMNS, QUALITY_COLUMNS
from ids.threshold_validation.observations import RAW_FEATURES


class ImmediatePool:
    """Deterministic synthetic scheduling; production process entry functions remain unchanged."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, callback, *args):
        future = Future()
        try:
            future.set_result(callback(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def test_full_runner_failed_day_resumes_completed_sources_and_packs(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    cache = tmp_path / "cache"
    calls = []
    pending = {"bad": True}
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr(runner, "ResourceLog", lambda root: DummyMonitor())
    monkeypatch.setattr(runner, "environment_snapshot", lambda: {"packages": {"synthetic": "only"}})
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"source_sha256": "fixed-synthetic-test"})
    monkeypatch.setattr(runner, "own_windows_workers", lambda: None)

    def marker(base, name):
        file = base / ("evidence/" + name + ".json")
        write_json(file, {"synthetic": True})
        return [file]

    monkeypatch.setattr(runner, "checks", lambda root: marker(root, "checks-fixture"))
    monkeypatch.setattr(runner.controls, "unpack", lambda root: marker(root, "unpack-fixture"))
    monkeypatch.setattr(runner.controls, "probe", lambda root, g, *a, **kw: marker(root, "probes/" + g))

    def gpu(root, group, threads):
        file = root / "evidence/gpu-batch" / (group + ".json")
        write_json(file, {"selected": 4096, "synthetic": True})
        return [file]

    monkeypatch.setattr(resources, "gpu_batch_probe", gpu)
    monkeypatch.setattr(
        resources, "history_benchmark", lambda src, out, threads: marker(out, "synthetic-history-benchmark")
    )

    cal = pd.DataFrame(
        {
            "hist_dst_ports_60s": np.zeros(600),
            "hist_dst_ips_60s": np.ones(600),
            **{"score_" + v: np.linspace(-2, 2, 600) for v in VIEWS},
        }
    )
    monkeypatch.setattr(runner, "calibration_frame", lambda *args: cal)

    def prepare(cache, folder, workers):
        specs = []
        for i in range(3):
            name = "IoT-SFC-synthetic-" + str(i)
            if i == 1 and pending["bad"]:
                specs.append(
                    {"name": name, "role": "synthetic", "status": "failed", "error": "injected missing day"}
                )
                continue
            folder.mkdir(parents=True, exist_ok=True)
            n = 700
            f = pd.DataFrame({c: np.zeros(n) for c in RAW_FEATURES})
            f["row_id"] = np.arange(1, n + 1)
            f["flow_end_ms"] = np.arange(n) * 10000 + i * 86400000
            f["source"] = "s"
            f["destination"] = "d"
            f["native_protocol"] = "tcp"
            f["ip_protocol"] = 6
            f["resp_port"] = np.r_[np.ones(360) * 80, np.arange(340) % 8 + 100]
            f["response_evidence_code"] = 0
            for c in (*GAP_COLUMNS, *QUALITY_COLUMNS):
                f[c] = 0
            labels = pd.DataFrame(
                {
                    "row_id": f.row_id,
                    "window": f"2025-01-0{i + 1}",
                    "family": ["Benign"] * 400 + ["Scanning"] * 299 + ["Unresolved"],
                    "native_detail": ["normal"] * 400 + ["scan"] * 299 + ["related"],
                    "event_id": [""] * 400 + ["one-event"] * 299 + [""],
                    "uid": [name + str(j) for j in range(n)],
                }
            )
            a, b, c = [folder / (name + s) for s in (".parquet", "-truth.parquet", "-audit.json")]
            f.to_parquet(a, index=False)
            labels.to_parquet(b, index=False)
            write_json(c, {"fixture": name})
            specs.append(
                {
                    "name": name,
                    "role": "synthetic",
                    "status": "complete",
                    "features": a,
                    "truth": b,
                    "audit": c,
                }
            )
        return specs

    monkeypatch.setattr(runner.sfc, "prepare_sources", prepare)

    def score(src, raw, frozen, cdir, group, out, **kwargs):
        frame = pd.read_parquet(src)
        n = len(frame)
        value = np.where(frame.hist_dst_ports_60s > 5, 3.0, -3.0)
        bits = {v: value > 2 for v in VIEWS}
        bits.update(add_quality=bits["raw"])
        result = pd.DataFrame({"row_id": frame.row_id, **bits, **{"score_" + v: value for v in VIEWS}})
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False)
        calls.append((src.name, group))
        return {"synthetic": True, "rows": n}

    monkeypatch.setattr(runner, "score_group", score)
    monkeypatch.setattr(
        "sys.argv",
        [
            "behavior-transfer",
            "--output",
            str(root),
            "--cache",
            str(cache),
            "--providers",
            "sfc",
            "--threads",
            "4",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert read_json(root / "outcome.json")["evaluated_group_sources"] == 10
    assert len(calls) == 10
    path = next(root.glob("behavior-incomplete-*.zip"))
    with ZipFile(path) as z:
        assert "archive_manifest.json" in z.namelist()
        assert not any(k.startswith("local/") for k in z.namelist())
    pending["bad"] = False
    runner.main()
    assert read_json(root / "outcome.json")["evaluated_group_sources"] == 15
    assert read_json(root / "last_invocation.json")["status"] == "complete"
    assert len(calls) == 15
    one = pd.read_parquet(root / "local/decisions/IoT-SFC-synthetic-0/hgb.parquet")
    assert len(one.columns) - 3 == 18
    assert set(one.phase) == {"startup", "after"}
    events = pd.read_csv(root / "evidence/evaluation/IoT-SFC-synthetic-0/hgb/events.csv.gz")
    assert events.event_id.nunique() == 1
    assert (events.before + events.gained - events.lost == events.after).all()


class DummyMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
