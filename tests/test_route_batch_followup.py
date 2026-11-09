"""Synthetic artifacts and mock workers for the three-task follow-up contract."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.route_batch import followup
from ids.route_batch.protocol import FORMAT, default_settings, make_tasks


@pytest.fixture
def fake_followup(tmp_path, monkeypatch):
    original, output = tmp_path / "original-run", tmp_path / "followup-run"
    original.mkdir()
    features = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "bytes", "duration"]
    schema = {"features": features, "numeric": features[4:], "categorical": features[:4],
              "label_column": "label", "id_column": "sample_id", "group_column": "communication_group",
              "family_column": "attack_family"}
    files = {}
    for offset, (role, n) in enumerate((("support", 16), ("calibration", 100), ("evaluation", 20))):
        label = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        frame = pd.DataFrame({"L4_SRC_PORT": np.arange(n)+100+offset*1000, "L4_DST_PORT": 443,
            "PROTOCOL": 6, "L7_PROTO": 91, "bytes": np.arange(n)+offset*1000, "duration": 1.,
            "label": label, "attack_family": np.where(label, "Scanning", "Benign"),
            "sample_id": [f"{role}-{i}" for i in range(n)],
            "communication_group": [f"{role}-group-{i}" for i in range(n)]})
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "sha256": sha256_file(path), "rows": n}
    settings = default_settings()
    settings.update(device="cpu", reference=[])
    spec = {"protocol": FORMAT, "data_kind": "synthetic", "schema": schema, "files": files,
            "artifacts": {}, "settings": settings, "tasks": make_tasks(settings), "wall_clock_limit": None}
    original_files = {"src/ids/original.py": "synthetic-original-source"}
    original_source = {"files": original_files, "source_sha256": canonical_hash(original_files)}
    write_json(original / "protocol.json", spec)
    write_json(original / "source_snapshot.json", original_source)
    write_json(original / "preflight.json", {"protocol_sha256": canonical_hash(spec), "source": original_source})
    write_json(original / "summary.json", {"sentinel": "original summary must not change"})
    for task in spec["tasks"]:
        if task["kind"] == "tabm":
            path = original / "tasks" / task["id"] / "attempt-001" / "refit" / "model.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(("synthetic model sentinel "+task["id"]).encode())
    write_json(original / "confirmation" / "freeze.json", {"sentinel": "opened original confirmation"})
    (original / "confirmation" / "evaluation.csv").write_bytes(b"must never be read as evaluation inputs")
    before = followup._inventory(original)
    python = tmp_path / "pfn-python.exe"
    python.write_bytes(b"synthetic interpreter marker; never executed")
    checkpoint = tmp_path / followup.TABPFN_CHECKPOINT
    checkpoint.write_bytes(b"synthetic checkpoint; never deserialized")
    monkeypatch.setattr(followup, "TABPFN_CHECKPOINT_SHA256", sha256_file(checkpoint))
    preparation = tmp_path / "prepare-report.json"
    write_json(preparation, {"status": "access_required", "failed_stage": "checkpoint_download", "error": "synthetic access gate"})
    source = {"files": {"src/ids/followup.py": "fixed-synthetic-source"}, "source_sha256": "fixed-followup-source"}
    monkeypatch.setattr(followup, "source_snapshot", lambda: source.copy())
    monkeypatch.setattr(followup, "environment_snapshot", lambda: {"synthetic": True})
    counters = {"dispatch": [], "replay": [], "pfn": 0, "cost": [], "exit_nonzero": set(), "interrupt": set()}
    from ids.route_batch import replay

    def mock_replay(original_run, task, destination):
        assert Path(original_run) == original
        assert task in ("tabicl_official_1", "tabicl_official_8")
        counters["replay"].append(task)
        cpu = destination / "cpu"
        write_json(cpu / "deployment.json", {"synthetic": True})
        (cpu / "evaluation.csv").write_text("row_id,label,score\nsynthetic,0,0\n", encoding="utf-8")
        return {"paths": {"original": {"status": "complete"}, "cpu": {"status": "complete"}},
                "cross_build_path": {"max_absolute_score_difference": .25}}

    def mock_cost(config, directory):
        assert Path(config) == original / "protocol.json"
        counters["cost"].append(str(directory))
        return {"status": "complete", "measurements": {"synthetic": True}}

    real_tabpfn_task = followup._tabpfn_task

    def mock_pfn(binding, destination):
        assert binding["original_run"] == str(original)
        counters["pfn"] += 1
        (destination / "model.joblib").write_bytes(b"synthetic recipe")
        return {"status": "complete", "confirmation_evaluated": False}

    def mock_subprocess(command, **kwargs):
        assert command[1:4] == ["-m", "ids.route_batch", "_followup-worker"]
        task = command[command.index("--task")+1]
        destination = Path(command[command.index("--output")+1])
        counters["dispatch"].append((task, command[0]))
        if task in counters["interrupt"]:
            destination.mkdir(parents=True)
            (destination / "interrupted.txt").write_text("preserve attempt", encoding="utf-8")
            raise OSError("synthetic worker startup failure")
        followup.worker(output, task, destination)
        return SimpleNamespace(returncode=7 if task in counters["exit_nonzero"] else 0)

    monkeypatch.setattr(replay, "run_replay_diagnostic", mock_replay)
    monkeypatch.setattr(followup, "_cpu_cost", mock_cost)
    monkeypatch.setattr(followup, "_tabpfn_task", mock_pfn)
    monkeypatch.setattr(followup.subprocess, "run", mock_subprocess)
    return SimpleNamespace(original=original, output=output, python=python, checkpoint=checkpoint,
        preparation=preparation, counters=counters, source=source, before=before, spec=spec,
        kwargs={"tabpfn_python": python, "tabpfn_checkpoint": checkpoint, "preparation_report": preparation},
        real_tabpfn_task=real_tabpfn_task)


def start(f, **kwargs):
    return followup.run(f.original, f.output, **f.kwargs, **kwargs)


def test_preflight_does_not_create_followup_or_modify_original(fake_followup):
    f = fake_followup
    report = followup.preflight(f.original, f.output, **f.kwargs)
    assert report["tasks"] == list(followup.TASKS)
    assert report["tabpfn"]["status"] == "unavailable"
    assert report["real_model_loads"] == 0
    assert not f.output.exists()
    assert followup._inventory(f.original) == f.before


def test_missing_pfn_keeps_both_diagnostics_and_resume_runs_only_pfn(fake_followup):
    f = fake_followup
    first = start(f)
    assert [x[0] for x in f.counters["dispatch"]] == list(followup.TASKS[:2])
    assert f.counters["replay"] == list(followup.TASKS[:2])
    assert not first["all_tasks_complete"] and first["tabm_training_calls"] == 0
    assert first["confirmation_evaluated"] is False
    assert first["tasks"][-1]["status"] == "unavailable"
    assert followup._inventory(f.original) == f.before
    snapshots = {task: sha256_file(f.output / "tasks" / task / "attempt-001" / "result.json") for task in followup.TASKS[:2]}
    write_json(f.preparation, {"status": "ready", "failed_stage": None, "error": None})
    second = start(f, resume=True)
    assert second["all_tasks_complete"]
    assert [x[0] for x in f.counters["dispatch"]] == list(followup.TASKS)
    assert f.counters["dispatch"][-1][1] == str(f.python)
    assert f.counters["pfn"] == 1
    for task, digest in snapshots.items():
        assert sha256_file(f.output / "tasks" / task / "attempt-001" / "result.json") == digest
        assert len(list((f.output / "tasks" / task).glob("attempt-*"))) == 2  # directory + log
    assert (f.output / "tasks/tabpfn25_official_8/attempt-001/result.json").is_file()
    assert (f.output / "tasks/tabpfn25_official_8/attempt-002/result.json").is_file()
    assert len(list((f.output / "preparation").glob("attempt-*.json"))) == 2
    assert followup._inventory(f.original) == f.before
    start(f, resume=True)
    assert len(f.counters["dispatch"]) == 3


@pytest.mark.parametrize("kind", ["missing", "malformed", "non_object"])
def test_unreadable_preparation_never_blocks_tabicl(fake_followup, kind):
    f = fake_followup
    if kind == "missing":
        f.preparation.unlink()
    else:
        f.preparation.write_text("{" if kind == "malformed" else "[]", encoding="utf-8")
    result = start(f)
    assert len(f.counters["replay"]) == 2
    assert not result["all_tasks_complete"]
    record = read_json(f.output / "tasks/tabpfn25_official_8/attempt-001/result.json")
    assert record["stage"] == "preparation_report"
    assert "missing" in record["reason"] if kind == "missing" else "cannot read" in record["reason"]


@pytest.mark.parametrize("what", ["source", "original_file", "original_protocol", "original_snapshot", "pfn_path"])
def test_resume_rejects_changed_binding_or_original_identity(fake_followup, what):
    f = fake_followup
    start(f)
    count = len(f.counters["dispatch"])
    if what == "source":
        f.source["source_sha256"] = "changed-followup-source"
    elif what == "original_file":
        (f.original / "confirmation/evaluation.csv").write_bytes(b"changed opened confirmation sentinel")
    elif what == "original_protocol":
        spec = read_json(f.original / "protocol.json")
        spec["scope"] = "changed"
        write_json(f.original / "protocol.json", spec)
    elif what == "original_snapshot":
        write_json(f.original / "source_snapshot.json", {"files": {}, "source_sha256": canonical_hash({})})
    else:
        f.kwargs["tabpfn_python"] = f.python.with_name("different-python.exe")
    with pytest.raises(ValueError):
        start(f, resume=True)
    assert len(f.counters["dispatch"]) == count


@pytest.mark.parametrize("location", ["same", "inside", "parent", "python_inside", "checkpoint_inside"])
def test_output_and_preparation_paths_cannot_modify_original(fake_followup, location):
    f = fake_followup
    output = {"same": f.original, "inside": f.original / "new", "parent": f.original.parent}.get(location, f.output)
    kwargs = f.kwargs.copy()
    if location == "python_inside":
        kwargs["tabpfn_python"] = f.original / "venv/python.exe"
    elif location == "checkpoint_inside":
        kwargs["tabpfn_checkpoint"] = f.original / "weights/model.ckpt"
    with pytest.raises(ValueError):
        followup.preflight(f.original, output, **kwargs)
    assert followup._inventory(f.original) == f.before
    assert not f.counters["dispatch"]


def test_pack_partial_includes_preparation_and_excludes_model_bytes(fake_followup, tmp_path):
    f = fake_followup
    start(f)
    evidence = f.output / "preparation-evidence" / "attempt-001"
    evidence.mkdir(parents=True)
    (evidence / "setup.log").write_text("synthetic preparation evidence", encoding="utf-8")
    for suffix in (".pt", ".pth", ".ckpt", ".joblib", ".safetensors"):
        (f.output / ("do-not-pack"+suffix)).write_bytes(b"synthetic model sentinel")
    target = tmp_path / "partial.zip"
    receipt = followup.pack(f.output, target)
    assert receipt["models_included"] is False and receipt["tabm_rerun"] is False
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        assert "preparation-evidence/attempt-001/setup.log" in names
        assert {"binding.json", "source_snapshot.json", "original_files.json", "summary.json"}.issubset(names)
        assert not any(name.startswith("do-not-pack") for name in names)
        manifest = json.loads(archive.read("archive_manifest.json"))["files"]
        assert set(manifest) == set(names)-{"archive_manifest.json"}
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == item["sha256"] for name, item in manifest.items())
        assert json.loads(archive.read("summary.json"))["all_tasks_complete"] is False
    assert followup._inventory(f.original) == f.before


@pytest.mark.parametrize("mode", ["original_root", "original_target", "inside_followup"])
def test_pack_rejects_paths_that_could_write_original(fake_followup, tmp_path, mode):
    f = fake_followup
    start(f)
    root = f.original if mode == "original_root" else f.output
    target = f.original / "bad.zip" if mode == "original_target" else f.output / "bad.zip" if mode == "inside_followup" else tmp_path / "bad.zip"
    with pytest.raises(ValueError):
        followup.pack(root, target)
    assert not target.exists()
    assert followup._inventory(f.original) == f.before


def test_nonzero_worker_exit_cannot_be_hidden_by_complete_result(fake_followup):
    f = fake_followup
    f.counters["exit_nonzero"].add("tabicl_official_1")
    result = start(f)
    assert result["tasks"][0]["status"] == "failed"
    assert result["tasks"][1]["status"] == "complete"
    original_worker = f.output / "tasks/tabicl_official_1/attempt-001/worker-result-before-exit-check.json"
    assert read_json(original_worker)["status"] == "complete"
    f.counters["exit_nonzero"].clear()
    start(f, resume=True)
    assert [x[0] for x in f.counters["dispatch"]] == ["tabicl_official_1", "tabicl_official_8", "tabicl_official_1"]


def test_interrupted_attempt_is_preserved_and_only_unfinished_task_restarts(fake_followup):
    f = fake_followup
    f.counters["interrupt"].add("tabicl_official_1")
    first = start(f)
    assert first["tasks"][0]["status"] == "unavailable"
    f.counters["interrupt"].clear()
    start(f, resume=True)
    assert (f.output / "tasks/tabicl_official_1/attempt-001/interrupted.txt").read_text() == "preserve attempt"
    assert (f.output / "tasks/tabicl_official_1/attempt-002/result.json").is_file()
    assert f.counters["replay"] == ["tabicl_official_8", "tabicl_official_1"]


def test_worker_rejects_tabm_and_confirmation_tasks_before_writing(fake_followup):
    f = fake_followup
    start(f)
    for task in ("tabm_log1p_standard_w128_lr0.002_s17", "confirm", "reference_full"):
        destination = f.output / "tasks" / task / "attempt-001"
        with pytest.raises(ValueError, match="only the two TabICL"):
            followup.worker(f.output, task, destination)
        assert not destination.exists()


def test_pfn_task_scores_only_old_calibration_and_development_E(fake_followup, monkeypatch, tmp_path):
    f = fake_followup
    fitted, predicted = [], []

    class MockFoundation:
        metadata = {"score": "synthetic score", "kind": "tabpfn25"}

        def save(self, path):
            path.write_bytes(b"synthetic rebuild recipe")

        def predict_outputs(self, x):
            predicted.append(len(x))
            scores = np.linspace(-1., 1., len(x))
            p = 1. / (1.+np.exp(-scores))
            return {"scores": scores, "probabilities": np.column_stack([1.-p, p])}

    def fit(kind, x, y, **kwargs):
        fitted.append((kind, len(x), len(y), kwargs))
        return MockFoundation()

    monkeypatch.setattr("ids.route_batch.foundation.fit_foundation", fit)
    target = tmp_path / "pfn-task"
    target.mkdir()
    result = f.real_tabpfn_task({"original_run":str(f.original), "tabpfn_checkpoint":str(f.checkpoint)}, target)
    assert fitted[0][:3] == ("tabpfn25", 16, 16)
    assert fitted[0][3]["device"] == "cpu" and fitted[0][3]["n_estimators"] == 8
    assert predicted == [100, 20]
    assert result["confirmation_evaluated"] is False
    assert pd.read_csv(target / "evaluation.csv")["row_id"].tolist() == [f"evaluation-{i}" for i in range(20)]
    assert followup._inventory(f.original) == f.before
