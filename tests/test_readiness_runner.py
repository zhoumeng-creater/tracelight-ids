"""Exercise real spawned cached-score replay, old-metric parity, resume and packing."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from ids.common import read_json, sha256_file, write_json
from ids.flow_history.execution import pack
from ids.readiness_replay import inputs, runner
from readiness_fixture import make_previous


class QuietMonitor:
    activity = "test"

    def __init__(self, root):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def marker(root):
    path = root / "evidence/synthetic-check-marker.json"
    write_json(path, {"test_prevents_recursive_pytest": True})
    return [path]


def test_spawned_replay_uses_cache_matches_original_and_resume_does_not_create_pool(tmp_path, monkeypatch):
    previous, registration = make_previous(tmp_path / "previous")
    config = tmp_path / "registration.json"
    write_json(config, registration)
    monkeypatch.setattr(inputs, "CONFIG", config)
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "checks", marker)
    monkeypatch.setattr(runner, "ResourceLog", QuietMonitor)
    monkeypatch.setattr(runner, "environment_snapshot", lambda **kw: {"python": "synthetic", "packages": {}})
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"source_sha256": "synthetic-fixed-code"})
    old = {str(p.relative_to(previous)): sha256_file(p) for p in previous.rglob("*") if p.is_file()}
    output = tmp_path / "repaired"
    assert runner.run(previous, output, threads=4, workers=2, memory_gib=1.25)
    outcome = read_json(output / "outcome.json")
    assert outcome["completed_group_sources"] == 10
    assert outcome["fits"] == outcome["model_forward_calls"] == 0
    report = read_json(output / "evidence/readiness/synthetic-dedale/mlp29.json")
    assert report["first_hour_rows"] == 2 and report["prefix_rows"] == 399
    assert report["ready"]
    performance = [read_json(p) for p in (output / "evidence/performance").glob("*/*.json")]
    assert len(performance) == 10
    assert all(p["original_budget_parity"]["status"] == "exact" for p in performance)
    assert all(p["fits"] == p["model_forward_calls"] == 0 for p in performance)
    assert len({p["pid"] for p in performance}) == 2
    assert old == {str(p.relative_to(previous)): sha256_file(p) for p in previous.rglob("*") if p.is_file()}
    receipts = {p.name: sha256_file(p) for p in (output / "receipts").glob("replay-*.json")}

    def forbidden_pool(**kwargs):
        pytest.fail("Completed replay must not create worker processes")
    monkeypatch.setattr(runner, "ProcessPoolExecutor", forbidden_pool)
    assert runner.run(previous, output, threads=4, workers=2, memory_gib=1.25)
    assert read_json(output / "evidence/dispatch.json")["pending_group_sources"] == 0
    assert receipts == {p.name: sha256_file(p) for p in (output / "receipts").glob("replay-*.json")}
    archive = pack(output, prefix="synthetic-replay")
    with ZipFile(archive) as z:
        assert "archive_manifest.json" in z.namelist()
        assert not any(name.startswith("local/") for name in z.namelist())
        assert any(name.endswith("ranking_targets.csv.gz") for name in z.namelist())


def test_memory_concurrency_is_bounded_and_changed_output_receipt_is_rejected(tmp_path, monkeypatch):
    class Memory:
        available = 8 * 1024**3
    monkeypatch.setattr(runner.psutil, "virtual_memory", lambda: Memory())
    specs = [{"estimated_worker_bytes": 2 * 1024**3}] * 20
    plan = runner.profile(specs, threads=24, requested_workers=8, memory_gib=5)
    assert plan["workers"] == 2 and plan["threads_per_worker"] == 12
    with pytest.raises(ValueError, match="largest source"):
        runner.profile(specs, threads=24, requested_workers=8, memory_gib=1)
    root = tmp_path / "output"
    artifact = root / "evidence/file.json"
    write_json(artifact, {"value": 1})
    write_json(root / "receipts/sample.json", {"binding": "fixed", "outputs": {"evidence/file.json": sha256_file(artifact)}})
    assert runner.completed(root, "sample", "fixed")
    write_json(artifact, {"value": 2})
    with pytest.raises(ValueError, match="output changed"):
        runner.completed(root, "sample", "fixed")


def test_missing_source_cache_packs_partial_then_restores_under_same_binding(tmp_path, monkeypatch):
    previous, registration = make_previous(tmp_path / "previous")
    config = tmp_path / "registration.json"
    write_json(config, registration)
    monkeypatch.setattr(inputs, "CONFIG", config)
    monkeypatch.setattr(runner, "CONFIG", config)
    monkeypatch.setattr(runner, "checks", marker)
    monkeypatch.setattr(runner, "ResourceLog", QuietMonitor)
    monkeypatch.setattr(runner, "own_windows_workers", lambda: None)
    monkeypatch.setattr(runner, "environment_snapshot", lambda **kw: {"python": "synthetic", "packages": {}})
    monkeypatch.setattr(runner, "source_snapshot", lambda: {"source_sha256": "synthetic-fixed-code"})
    missing = previous / "local/scores/synthetic-sfc/mlp29.parquet"
    original_bytes = missing.read_bytes()
    missing.unlink()  # Only this generated pytest fixture is removed and then restored byte-for-byte.
    output = tmp_path / "repaired"
    monkeypatch.setattr("sys.argv", [
        "readiness-replay", "--previous", str(previous), "--output", str(output),
        "--threads", "4", "--workers", "2", "--memory-gib", "1.25",
    ])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    partial = read_json(output / "outcome.json")
    assert partial["status"] == "incomplete" and partial["completed_group_sources"] == 5
    assert {entry["source"] for entry in partial["completed"]} == {"synthetic-dedale"}
    preflight = read_json(output / "evidence/input_preflight.json")
    assert len(preflight["failures"]) == 1
    assert preflight["failures"][0]["source"] == "synthetic-sfc"
    assert preflight["failures"][0]["stage"] == "cached_inputs"
    assert read_json(output / "last_invocation.json")["status"] == "incomplete"
    first_protocol_hash = sha256_file(output / "protocol.json")
    completed_receipts = {
        path.name: (sha256_file(path), path.stat().st_mtime_ns)
        for path in (output / "receipts").glob("replay-*.json")
    }
    assert len(completed_receipts) == 5
    with ZipFile(next(output.glob("readiness-replay-incomplete-*.zip"))) as archive:
        assert json.loads(archive.read("outcome.json"))["completed_group_sources"] == 5
        assert "archive_manifest.json" in archive.namelist()
        assert not any(name.startswith("local/") for name in archive.namelist())

    missing.write_bytes(original_bytes)
    runner.main()
    assert sha256_file(output / "protocol.json") == first_protocol_hash
    complete = read_json(output / "outcome.json")
    assert complete["status"] == "complete" and complete["completed_group_sources"] == 10
    assert not complete["failures"]
    dispatch = read_json(output / "evidence/dispatch.json")
    assert dispatch["pending_sources"] == 1 and dispatch["pending_group_sources"] == 5
    assert dispatch["reused_group_sources"] == 5
    assert read_json(output / "last_invocation.json")["status"] == "complete"
    for name, evidence in completed_receipts.items():
        path = output / "receipts" / name
        assert (sha256_file(path), path.stat().st_mtime_ns) == evidence
    with ZipFile(next(output.glob("readiness-replay-results-*.zip"))) as archive:
        result = json.loads(archive.read("outcome.json"))
        assert result["status"] == "complete" and result["completed_group_sources"] == 10
        assert result["fits"] == result["model_forward_calls"] == 0


def test_previous_output_tree_must_be_separate():
    with pytest.raises(ValueError, match="separate"):
        inputs.distinct_roots(Path("previous"), Path("previous/repair"))
    with pytest.raises(ValueError, match="separate"):
        inputs.distinct_roots(Path("previous"), Path("."))
