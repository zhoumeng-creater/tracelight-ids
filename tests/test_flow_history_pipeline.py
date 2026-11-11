"""Synthetic integration of stage ordering, deployment timing, receipts and lab delivery."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import psutil
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from ids.flow_history import __main__ as cli, execution, pipeline as pipe, trees
from ids.flow_history.evaluation import measure, paired_comparison
from ids.route_batch import protocol


def role_frame(rng, role, n):
    frame = pd.DataFrame(rng.uniform(0, 1, (n, len(FEATURES))), columns=FEATURES)
    frame["label"] = 0 if role.startswith("C") else np.arange(n) % 2
    frame["IN_BYTES"] += 3 * frame.label
    frame["attack_family"] = np.where(frame.label, "Scanning", "Benign")
    frame["row_id"] = [f"{role}-{i}" for i in range(n)]
    frame["communication_group"] = frame.row_id
    return frame


def _assert_pack(path):
    assert read_json(Path(str(path) + ".sha256.json"))["sha256"] == sha256_file(path)
    with ZipFile(path) as contents:
        manifest = json.loads(contents.read("archive_manifest.json"))
        assert set(manifest) == set(contents.namelist()) - {"archive_manifest.json"}
        for name, entry in manifest.items():
            data = contents.read(name)
            assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert not any(s.startswith(("source/", "history-work/")) for s in manifest)
        return manifest


def test_registered_pipeline_fit_score_isolated_bench_and_resume(tmp_path, monkeypatch):
    """Real HGB fit -> two independent CPU workers -> packaging; only source and history are simulated."""
    from ids.common import environment_snapshot

    root, old = tmp_path / "lab19", tmp_path / "old"
    root.mkdir()
    old.mkdir()
    rng = np.random.default_rng(48119)
    roles = {role: role_frame(rng, role, count) for role, count in
             zip(pipe.ROLES, [160, 125, 540, 0, 0, 0])}
    # The anchor loader checks the known group independently of its CSV hashes.
    e = roles["E"]
    syn = e.index[e.label.eq(1)][:217].union(e.index[e.label.eq(0)][:33])
    fixed = {key: 0 for key in FEATURES if key != "L4_SRC_PORT"}
    fixed.update(L4_DST_PORT=443, PROTOCOL=6, L7_PROTO=91, IN_BYTES=44, IN_PKTS=1, TCP_FLAGS=2,
                 FLOW_DURATION_MILLISECONDS=1, CLIENT_TCP_FLAGS=2, MAX_IP_PKT_LEN=44,
                 SHORTEST_FLOW_PKT=44, LONGEST_FLOW_PKT=44, SRC_TO_DST_AVG_THROUGHPUT=352000,
                 NUM_PKTS_UP_TO_128_BYTES=1, TCP_WIN_MAX_IN=1024)
    for key, value in fixed.items():
        e.loc[syn, key] = value
    forward = {"binary_evaluable": False, "scanning_evaluable": False}

    def cohorts(original, source, work, threads):
        assert original == old and threads == 2
        write_json(work / "cohort_manifest.json", {"forward": forward})
        return roles

    def history(source, frame, work, threads):
        work.mkdir()
        meta = frame[["row_id"]].assign(source_host="192.0.2.1", end_ms=1556043001000)
        meta.to_parquet(work / "target_metadata.parquet", index=False)
        write_json(work / "history_audit.json", {"rows": len(frame)})
        write_json(work / "benchmark_segment.json", {"status": "unavailable", "reason": "fixture"})
        return frame[["row_id"]].assign(**{name: np.arange(len(frame)) % 7 for name in pipe.HISTORY_COLUMNS})

    monkeypatch.setattr(pipe, "resolve_source", lambda *args, **kwargs: tmp_path / "synthetic.csv")
    monkeypatch.setattr(pipe, "prepare_cohorts", cohorts)
    monkeypatch.setattr(pipe, "extract_history", history)
    def preflight(path):
        marker = path / "evidence/preflight/test-fixture.json"
        write_json(marker, {"synthetic_test_body": True})
        return [marker]
    monkeypatch.setattr(pipe, "synthetic_checks", preflight)
    monkeypatch.setattr(pipe, "ARMS", {"hgb52_history": "hgb"})
    monkeypatch.setattr(trees, "PARAMETERS", {**trees.PARAMETERS, "max_iter": 4})
    monkeypatch.setattr(protocol, "source_snapshot", lambda: {"source_sha256": "synthetic-build-A"})
    monkeypatch.setattr("ids.common.environment_snapshot", lambda: environment_snapshot(include_torch=False))
    hashes = {}
    for name in pipe.REFERENCE_HASHES:
        paths = []
        folder = old / "tasks" / name / "attempt-001"
        folder.mkdir(parents=True)
        for role, key in [("calibration", "C"), ("evaluation", "E")]:
            frame = roles[key]
            path = folder / (role + ".csv")
            frame[["row_id", "label", "attack_family"]].assign(score=frame.IN_BYTES).to_csv(path, index=False)
            paths.append(sha256_file(path))
        hashes[name] = tuple(paths)
    monkeypatch.setattr(pipe, "REFERENCE_HASHES", hashes)

    # Run real lane logic in this interpreter so the synthetic parameters are visible.
    # Record the lifetime boundary; real cost measurement still spawns fresh Python workers.
    lane_pool = []

    class Pool(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            assert max_workers == 2
            super().__init__(max_workers)

        def __enter__(self):
            lane_pool.append(True)
            return super().__enter__()

        def __exit__(self, *args):
            value = super().__exit__(*args)
            lane_pool.pop()
            return value

    monkeypatch.setattr(pipe, "ProcessPoolExecutor", Pool)
    regular = pipe.benchmark
    bench_calls = []

    def observed_bench(*args):
        assert lane_pool == []  # All parallel training jobs exited before benchmarking starts.
        bench_calls.append(args[-1])
        return regular(*args)

    monkeypatch.setattr(pipe, "benchmark", observed_bench)
    activity = SimpleNamespace(activity="")
    pipe.run(root, old, None, 2, activity, device="cpu")
    report = read_json(root / "summary.json")
    assert report["status"] == "development_complete_forward_not_evaluable" and bench_calls == [1, 2]
    assert report["forward_cohort"] == forward and not any(k.startswith("forward/") for k in report["models"])
    result = report["models"]["development/hgb52_history"]
    assert result["scan_n"] == 270 and result["tp"] + result["fp"] <= len(e)
    assert set(result["cpu_cost"]) == {"1", "2"}
    assert all(cost["equivalent"] and cost["single_p95_ms"] > 0 for cost in result["cpu_cost"].values())
    assert "development/reference_full_to_hgb52_history" in report["comparisons"]
    controlled, (zero,) = pipe.inputs(root / "inputs", "mlp52_zero", ("E",))
    assert len(controlled) == 52 and not zero[list(pipe.HISTORY_COLUMNS)].to_numpy().any()
    prediction = pd.read_parquet(root / "evidence/development/hgb52_history/evaluation.parquet")
    assert np.isfinite(prediction.score).all() and prediction.row_id.tolist() == roles["E"].row_id.tolist()
    saved = _assert_pack(execution.pack(root))
    assert "summary.json" in saved and "evidence/development/hgb52_history/strata.json" in saved
    assert "models/development/hgb52_history.joblib" in saved
    prior_receipts = {f: sha256_file(f) for f in (root / "receipts").glob("*.json")}
    pipe.run(root, old, None, 2, activity, device="cpu")
    assert bench_calls == [1, 2] and all(sha256_file(f) == digest for f, digest in prior_receipts.items())
    monkeypatch.setattr(protocol, "source_snapshot", lambda: {"source_sha256": "synthetic-build-B"})
    with pytest.raises(ValueError, match="registered"):
        pipe.run(root, old, None, 2, activity, device="cpu")


def test_partial_stage_retry_preserves_completed_evidence_and_rejects_mutation(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    file = root / "calibration.json"
    calls = []

    def run():
        calls.append(True)
        write_json(file, {"attempt": len(calls)})
        if len(calls) == 1:
            raise RuntimeError("interrupted")
        return [file]

    with pytest.raises(RuntimeError, match="interrupted"):
        execution.stage(root, "score", "bound-A", run)
    assert not (root / "receipts/score.json").exists()
    execution.stage(root, "score", "bound-A", run)
    execution.stage(root, "score", "bound-A", run)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="identity"):
        execution.stage(root, "score", "bound-B", run)
    file.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        execution.stage(root, "score", "bound-A", run)


def test_frozen_point_and_missing_scans():
    rng = np.random.default_rng(1)
    c, e = role_frame(rng, "C", 99), role_frame(rng, "E", 200)
    e["attack_family"] = np.where(e.label, "DDoS", "Benign")
    cal = np.arange(99, dtype=float)
    scores = np.full(len(e), 98.0)  # Strict > excludes ties at registered rank 99.
    result, row = measure(c, e, cal, scores)
    assert result["threshold"]["rank"] == 99 and not row.alert.any()
    assert result["scan_n"] == 0 and result["syn_scan_normal_auc"] is None
    assert not paired_comparison(row, row, result, result)["passes_detection_comparison"]
    with pytest.raises(ValueError, match="finite"):
        measure(c.iloc[:-1], e, cal[:-1], scores)
    with pytest.raises(ValueError, match="normal-only"):
        measure(c.assign(label=1), e, cal, scores)
    with pytest.raises(ValueError, match="finite"):
        measure(c, e, cal, scores + np.nan)
    e["attack_family"] = np.where(e.label, "Scanning", "Benign")
    base, old_rows = measure(c, e, cal, scores)
    new_score = np.where(e.label, 99.0, 0.0)
    new, new_rows = measure(c, e, cal, new_score)
    assert paired_comparison(old_rows, new_rows, base, new)["passes_detection_comparison"]
    new_score[:5] = 99
    higher_fp, rows = measure(c, e, cal, new_score)
    assert not paired_comparison(old_rows, rows, base, higher_fp)["passes_detection_comparison"]


def test_forward_scan_not_claimed_with_no_training_scans_and_failed_stage_excluded(tmp_path):
    root = tmp_path / "no-training-scans"
    write_json(root / "evidence/inputs/cohort_manifest.json", {"forward": {
        "binary_evaluable": True, "scanning_evaluable": False}})
    rng = np.random.default_rng(97)
    cal, ev = role_frame(rng, "C", 199), role_frame(rng, "E", 200)
    cpu = np.zeros(len(cal))
    for arm, scores in [("hgb48", np.zeros(len(ev))), ("hgb52_history", ev.label.to_numpy(dtype=float))]:
        execution.stage(root, f"forward-{arm}-score", "test", lambda: pipe.save_scores(
            root, arm, "forward", cal, ev, cpu, scores, timing={"synthetic": True}))
    pipe.summary(root, ["forward"], {})
    summary = read_json(root / "summary.json")
    comparison = summary["comparisons"]["forward/hgb48_to_hgb52_history"]
    assert comparison["delta_scan_tp"] == 100 and comparison["passes_detection_comparison"] is None
    pipe.summary(root, ["forward"], {"forward-hgb52_history": "score integrity check failed on resume"})
    summary = read_json(root / "summary.json")
    assert summary["status"] == "partial" and "forward/hgb52_history" not in summary["models"]
    assert "forward/hgb48_to_hgb52_history" not in summary["comparisons"]


def test_locked_and_failed_cli_still_returns_an_integrity_checked_package(tmp_path, monkeypatch):
    root, original = tmp_path / "lab", tmp_path / "old"
    original.mkdir()
    monkeypatch.setattr(sys, "argv", ["flow_history", "--original-run", str(original), "--output", str(root)])
    monkeypatch.setattr(cli, "own_windows_workers", lambda: None)
    with execution.run_lock(root):
        with pytest.raises(RuntimeError, match="lock"):
            cli.main()
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic interrupted run")
    monkeypatch.setattr(pipe, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        cli.main()
    packed = _assert_pack(next(root.glob("flow-history-results-*.zip")))
    assert "last_invocation.json" in packed and read_json(root / "last_invocation.json")["status"] == "failed"


@pytest.mark.parametrize("shell", [s for s in ("powershell", "pwsh") if shutil.which(s)] or [None])
def test_checkonly_powershell_defaults_and_spaces(tmp_path, shell):
    if not shell:
        pytest.skip("PowerShell unavailable")
    checkout = tmp_path / "synthetic lab checkout"
    scripts, tests = checkout / "scripts", checkout / "tests"
    scripts.mkdir(parents=True)
    tests.mkdir()
    wrapper = scripts / "RUN_FLOW_HISTORY.ps1"
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / wrapper.name, wrapper)
    (checkout / "pytest.ini").write_text("[pytest]\n")
    (tests / "test_flow_history_marker.py").write_text("def test_flow_history_marker():\n    assert True\n")
    # A non-history module must never be collected by the task-specific preflight.
    (tests / "test_other_task.py").write_text("raise RuntimeError('unrelated optional lab dependency')\n")
    run = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                          str(wrapper), "-Python", sys.executable, "-CheckOnly"],
                         cwd=checkout, capture_output=True, encoding="utf-8", errors="replace", timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed" in run.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object cleanup applies to the laboratory OS")
def test_windows_coordinator_exit_reaps_synthetic_worker():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    code = """
from ids.flow_history.execution import own_windows_workers
import os, subprocess, sys
own_windows_workers()
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
os._exit(0)
"""
    completed = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                               text=True, check=True, timeout=15)
    pid = int(completed.stdout.strip())
    timer = time.monotonic() + 3
    while psutil.pid_exists(pid) and time.monotonic() < timer:
        time.sleep(.02)
    assert not psutil.pid_exists(pid)
