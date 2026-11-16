"""Synthetic orchestration tests; the backend never constructs a real model."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.model_training import runner


def frame(role, marker):
    result = pd.DataFrame({name: [float(marker), float(marker + 0.25)]
                           for name in runner.data.RAW_FEATURES})
    result["ip_protocol"] = 6
    result["source_id"] = role
    result["row_id"] = [0, 1]
    result["label"] = [0, 0] if role == "C" else [0, 1]
    result["family"] = np.where(result.label, "ExternalC2", "Benign")
    return result


class FakeBackend:
    def __init__(self, root, events):
        self.root = root
        self.events = events
        self.fit_calls = 0
        self.loads = 0
        self.fail_next = False

    def _fit(self, output, phase, selected, **kwargs):
        output = Path(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty unfinished fit cannot be retrained")
        output.mkdir(parents=True, exist_ok=True)
        self.fit_calls += 1
        self.events.append((phase, output.parent.name))
        write_json(output / "attempt.json", {"arm": kwargs["arm"], "phase": phase})
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("synthetic interrupted fit")
        artifact = output / "model.fake"
        artifact.write_text(output.parent.name, encoding="utf-8")
        result = {"status": "complete", "selected_stage": selected, "artifact_path": str(artifact)}
        write_json(output / "summary.json", result)
        return result

    def fit_trajectory(self, f, v, output, **kwargs):
        assert set(f.source_id) == {"F"} and set(v.source_id) == {"V"}
        return self._fit(output, "trajectory", 2, **kwargs)

    def fit_final(self, p, output, *, stage, **kwargs):
        assert set(p.source_id) == {"F", "V"} and stage == 2
        return self._fit(output, "final", stage, **kwargs)

    def load_predictor(self, path, *, device, threads):
        assert self.fit_calls == 14
        assert device == "cpu" and threads == 1
        frozen = read_json(self.root / "freeze.json")
        assert len(frozen["artifacts"]) == 7
        assert all((self.root / f"receipts/{name}-final.json").is_file()
                   for name in frozen["artifacts"])
        self.loads += 1
        model_id = Path(path).read_text(encoding="utf-8")

        def score(raw):
            assert list(raw.columns) == list(runner.data.RAW_FEATURES)
            assert "label" not in raw and "source_id" not in raw
            role = "C" if raw.duration_seconds.iloc[0] == 3 else "ED"
            self.events.append(("score", model_id, role))
            return raw.duration_seconds.to_numpy(dtype=np.float64)

        return SimpleNamespace(score=score)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    design = {"training": {"hgb": {"random_state": 20261111},
                           "neural_seeds": [17, 29, 43], "max_fits": 14}}
    protocol = {"version": runner.VERSION, "tasks": runner.tasks(design), "design": design,
                "threads": 4, "device": "cpu", "code": {}, "registration": {}}
    write_json(root / "protocol.json", protocol)
    f, v, c, ed = frame("F", 1), frame("V", 2), frame("C", 3), frame("ED", 4)
    roles = {"F": f, "V": v, "P": pd.concat([f, v], ignore_index=True), "C": c, "ED": ed}
    key = canonical_hash(protocol)
    runner.checked_stage(root, "inputs", key,
                         lambda: runner.data.persist_inputs(root, roles, {}, {}))

    def snapshot():
        path = root / "source_snapshot/fake.py"
        path.parent.mkdir()
        path.write_text("# synthetic source snapshot\n", encoding="utf-8")
        return [path]

    runner.checked_stage(root, "sources", key, snapshot)
    events = []
    backend = FakeBackend(root, events)
    monkeypatch.setattr(runner, "verify_live_code", lambda protocol: None)
    monkeypatch.setattr(runner, "verify_runtime", lambda registration, device: {"synthetic": True})
    monkeypatch.setattr(runner, "initialize", lambda *args: protocol)
    monkeypatch.setattr(runner, "own_windows_workers", lambda: None)
    monkeypatch.setattr(runner, "ResourceLog", lambda root: nullcontext(SimpleNamespace(activity="")))

    def launch(root, mode, task_id):
        if mode == "worker":
            runner.train_task(root, task_id, backend=backend)
        else:
            assert mode == "cost-worker"
            assert len([event for event in events if event[0] == "score"]) == 14
            write_json(root / f"cost/{task_id}.json", {"synthetic": True})
            events.append(("cost", task_id))

    original_score = runner.score_task
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "score_task",
                        lambda root, protocol, task: original_score(root, protocol, task, backend=backend))

    def report(root, protocol):
        assert len([event for event in events if event[0] == "score"]) == 14
        path = root / "reports/comparison.json"
        write_json(path, {"synthetic": True})
        events.append(("report",))
        return [path]

    monkeypatch.setattr(runner, "report", report)
    original_read = pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if Path(path).name in {"C.parquet", "ED.parquet"}:
            assert backend.fit_calls == 14, "C/ED must not enter fitting or pre-freeze scoring"
            assert (root / "receipts/freeze.json").is_file()
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    return SimpleNamespace(root=root, protocol=protocol, backend=backend, events=events,
                           r26=tmp_path / "frozen-r26")


def complete(fixture):
    return runner.run(fixture.root, r26=fixture.r26, threads=4, device="cpu")


def test_seven_trajectories_refits_freeze_then_cpu_scoring_and_resume(run_fixture):
    f = run_fixture
    result = complete(f)
    assert result["status"] == "complete"
    assert f.backend.fit_calls == 14 and f.backend.loads == 7
    assert sum(event[0] == "trajectory" for event in f.events) == 7
    assert sum(event[0] == "final" for event in f.events) == 7
    first_score = next(i for i, event in enumerate(f.events) if event[0] == "score")
    assert all(event[0] in {"trajectory", "final"} for event in f.events[:first_score])
    assert first_score == 14
    for task in f.protocol["tasks"]:
        scores = [event[2] for event in f.events if event[:2] == ("score", task["id"])]
        assert scores == ["C", "ED"]
        assert read_json(f.root / f"evidence/{task['id']}/calibration.json")["n_normal"] == 2
    assert read_json(f.root / "freeze.json")["actual_fit_calls"] == 14
    saved_events = list(f.events)
    for task in f.protocol["tasks"]:
        runner.train_task(f.root, task["id"], backend=f.backend)
    second = complete(f)
    assert second["status"] == "complete" and f.backend.fit_calls == 14
    assert f.events == saved_events


def test_unfinished_nonempty_fit_is_not_retried(run_fixture):
    f = run_fixture
    f.backend.fail_next = True
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.train_task(f.root, "hgb", backend=f.backend)
    assert f.backend.fit_calls == 1
    assert not (f.root / "receipts/hgb-trajectory.json").exists()
    assert (f.root / "models/hgb/trajectory/attempt.json").is_file()
    with pytest.raises(ValueError, match="Nonempty"):
        runner.train_task(f.root, "hgb", backend=f.backend)
    assert f.backend.fit_calls == 1


def test_incomplete_global_freeze_prevents_predictor_loading(run_fixture):
    f = run_fixture
    runner.train_task(f.root, "hgb", backend=f.backend)
    with pytest.raises(FileNotFoundError):
        runner.freeze(f.root, f.protocol)
    with pytest.raises(FileNotFoundError):
        runner.score_task(f.root, f.protocol, f.protocol["tasks"][0])
    assert f.backend.loads == 0


def test_freeze_artifact_set_must_include_all_seven_models(run_fixture):
    f = run_fixture
    for task in f.protocol["tasks"]:
        runner.train_task(f.root, task["id"], backend=f.backend)
    paths = runner.freeze(f.root, f.protocol)
    frozen = read_json(paths[0])
    frozen["artifacts"].pop("hgb")
    write_json(paths[0], frozen)
    runner.checked_stage(f.root, "freeze", canonical_hash(f.protocol), lambda: paths)
    with pytest.raises(ValueError, match="All seven"):
        runner.score_task(f.root, f.protocol, f.protocol["tasks"][0])
    assert f.backend.loads == 0


def test_completed_stage_receipt_and_manifest_detect_artifact_changes(run_fixture):
    f = run_fixture
    complete(f)
    artifact = f.root / "models/hgb/final/model.fake"
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.train_task(f.root, "hgb", backend=f.backend)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(f.root)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.pack(f.root)
    assert f.backend.fit_calls == 14


def test_receipt_tampering_is_detected_even_if_manifest_is_rebuilt(run_fixture):
    f = run_fixture
    complete(f)
    receipt = f.root / "receipts/hgb-final.json"
    state = read_json(receipt)
    state["outputs"]["models/hgb/final/model.fake"] = "0" * 64
    write_json(receipt, state)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(f.root)
    runner.manifest(f.root)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify(f.root)
    assert f.backend.fit_calls == 14


def test_changed_protocol_binding_cannot_reuse_completed_models(run_fixture):
    f = run_fixture
    runner.train_task(f.root, "hgb", backend=f.backend)
    protocol = read_json(f.root / "protocol.json")
    protocol["tasks"][0]["seed"] += 1
    write_json(f.root / "protocol.json", protocol)
    with pytest.raises(ValueError, match="binding changed"):
        runner.train_task(f.root, "hgb", backend=f.backend)
    assert f.backend.fit_calls == 2


def test_offline_verify_pack_need_no_runtime_or_backend(run_fixture, monkeypatch):
    f = run_fixture
    complete(f)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline archive verification must not call a live runtime or model")

    monkeypatch.setattr(runner, "verify_live_code", forbidden)
    monkeypatch.setattr(runner, "verify_runtime", forbidden)
    monkeypatch.setattr(f.backend, "load_predictor", forbidden)
    monkeypatch.setattr(f.backend, "fit_trajectory", forbidden)
    monkeypatch.setattr(f.backend, "fit_final", forbidden)
    report = runner.verify(f.root)
    assert report["status"] == "complete" and report["model_forwards"] == 0
    packed = runner.pack(f.root)
    archive = Path(packed["archive"])
    assert sha256_file(archive) == packed["sha256"]
    assert archive.with_suffix(".zip.sha256").read_text(encoding="utf-8").split()[0] == packed["sha256"]
    expected = set(read_json(f.root / "archive_manifest.json")["files"])
    with ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == expected | {"archive_manifest.json"}
        assert zipped.testzip() is None
    assert f.backend.fit_calls == 14


def test_failed_evidence_archive_keeps_broken_receipt_and_original_error(run_fixture, monkeypatch):
    f = run_fixture
    runner.train_task(f.root, "hgb", backend=f.backend)
    receipt = f.root / "receipts/hgb-final.json"
    original_receipt = receipt.read_bytes()
    artifact = f.root / "models/hgb/final/model.fake"
    artifact.write_text("damaged output retained as failure evidence", encoding="utf-8")
    original_error = RuntimeError("original training-stage integrity failure")
    write_json(f.root / "status.json", {"status": "failed", "error": str(original_error)})

    def forbidden(*args, **kwargs):
        raise AssertionError("Failure evidence cannot require successful experiment verification or model calls")

    monkeypatch.setattr(runner, "verify", forbidden)
    monkeypatch.setattr(f.backend, "load_predictor", forbidden)
    result = runner.archive_failure(f.root, original_error)
    assert result["status"] == "failed_evidence_only"
    assert result["completed_experiment_verified"] is False
    assert result["original_error_type"] == "RuntimeError"
    assert result["original_error"] == str(original_error)
    assert result["model_forwards"] == 0
    assert [item["receipt"] for item in result["broken_receipts"]] == ["receipts/hgb-final.json"]
    assert "models/hgb/final/model.fake" in result["broken_receipts"][0]["error"]
    assert receipt.read_bytes() == original_receipt
    assert read_json(f.root / "status.json")["status"] == "failed"
    assert read_json(f.root / "failure_integrity.json")["broken_receipts"] == result["broken_receipts"]
    with ZipFile(result["archive"]) as archive:
        assert archive.read("receipts/hgb-final.json") == original_receipt
        assert archive.read("models/hgb/final/model.fake") == artifact.read_bytes()
        assert "failure_integrity.json" in archive.namelist()
        assert archive.testzip() is None
    assert f.backend.fit_calls == 2
