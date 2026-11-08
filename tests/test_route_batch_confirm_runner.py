"""Mock-model confirmation lifecycle; all data are small synthetic fixtures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.route_batch import confirm_runner as confirmation


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    schema = {"features": ["x"], "numeric": ["x"], "categorical": [], "label_column": "label",
              "id_column": "sample_id", "group_column": "communication_group", "family_column": "attack_family"}
    def csv_file(path, role, n, new=False):
        labels = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        x = np.linspace(0., 1., n) if role == "calibration" else np.where(labels, 2., .2)
        frame = pd.DataFrame({"x": x, "label": labels, "attack_family": np.where(labels, "Scanning", "Benign"),
                              "sample_id": [f"{'new' if new else 'old'}-{role}-{i}" for i in range(n)],
                              "communication_group": [f"{'new' if new else 'old'}-{role}-group-{i}" for i in range(n)]})
        frame.to_csv(path, index=False)
        return {"path": str(path), "rows": n, "sha256": sha256_file(path)}
    files = {role: csv_file(tmp_path / f"old-{role}.csv", role, n)
             for role, n in (("support", 20), ("calibration", 100), ("evaluation", 20))}
    tasks = [{"id": "reference_full", "kind": "reference", "name": "full"},
             {"id": "tabm_seed17", "kind": "tabm", "seed": 17},
             {"id": "tabm_seed29", "kind": "tabm", "seed": 29},
             {"id": "tabm_cost_failed", "kind": "tabm", "seed": 43},
             {"id": "optional_pfn", "kind": "foundation", "model_kind": "tabpfn25"},
             {"id": "training_failed", "kind": "tabm", "seed": 99}]
    spec = {"schema": schema, "files": files, "artifacts": {}, "round14_root": str(tmp_path / "round14"),
            "settings": {"alpha": .01, "sensitivity_alphas": [.005, .02], "query_batch_size": 128,
                         "reference": ["full"]}, "tasks": tasks, "tabpfn_python": "pfn-python"}
    reference = tmp_path / "round14" / "comparison" / "full"
    reference.mkdir(parents=True)
    reference_files = {}
    for name in ("model.pt", "preprocessor.joblib", "threshold.json"):
        (reference / name).write_text("original-reference", encoding="utf-8")
        reference_files[name] = sha256_file(reference / name)
    write_json(reference / "deployment.json", {"candidate": "full", "files": reference_files,
        "calibration_sha256": files["calibration"]["sha256"]})
    spec["baseline_model_sha256"] = reference_files["model.pt"]
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_text("synthetic-checkpoint", encoding="utf-8")
    for task in tasks:
        attempt = root / "tasks" / task["id"] / "attempt-001"
        attempt.mkdir(parents=True)
        if task["id"] == "training_failed":
            record = {"status": "failed", "error": "synthetic training failed"}
        elif task["kind"] == "reference":
            record = {"status": "scores_only", "model": {"kind": "reference", "directory": str(reference),
                      "score_contract": "synthetic score"}}
        elif task["kind"] == "foundation":
            (attempt / "model.joblib").write_text("synthetic recipe", encoding="utf-8")
            record = {"status": "complete", "model": {"kind": "foundation", "model_file": "model.joblib",
                      "model_sha256": sha256_file(attempt / "model.joblib"), "score_contract": "synthetic score",
                      "metadata": {"kind": "tabpfn25", "checkpoint_path": str(checkpoint),
                                   "checkpoint_sha256": sha256_file(checkpoint)}}}
        else:
            (attempt / "refit").mkdir()
            (attempt / "refit" / "model.pt").write_text(task["id"], encoding="utf-8")
            (attempt / "preprocessing.joblib").write_text("synthetic representation", encoding="utf-8")
            write_json(attempt / "refit" / "summary.json", {"status": "complete"})
            record = {"status": "complete", "model": {"kind": "tabm", "model_file": "refit/model.pt",
                      "preprocessing_file": "preprocessing.joblib", "score_contract": "synthetic score",
                      "model_sha256": sha256_file(attempt / "refit" / "model.pt"),
                      "preprocessing_sha256": sha256_file(attempt / "preprocessing.joblib")}}
            if task["id"] == "tabm_cost_failed":
                record = {"status": "failed", "error": "synthetic cost measurement failed"}
        write_json(attempt / "result.json", record)
    write_json(root / "protocol.json", spec)
    write_json(root / "source_snapshot.json", {"source_sha256": "fixed-synthetic-code"})
    previous = tmp_path / "previous"
    previous.mkdir()
    for name in ("preparation.json", "support.csv", "normal_calibration.csv", "development_evaluation.csv"):
        (previous / name).write_text("synthetic previous evidence", encoding="utf-8")
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"synthetic source, never opened as a real archive")
    counters = {"prepare": 0, "workers": [], "loads": []}
    monkeypatch.setattr(confirmation, "source_snapshot", lambda: {"source_sha256": "fixed-synthetic-code"})
    monkeypatch.setattr(confirmation, "CALIBRATION_ROWS", 100)
    monkeypatch.setattr(confirmation, "EVALUATION_ROWS", 20)
    def prepare(source_zip, inputs, output, **recipe):
        assert (root / "confirmation" / "freeze.json").is_file()
        counters["prepare"] += 1
        output.mkdir()
        fresh_files = {role: csv_file(output / f"{role}.csv", role, n, new=True)
                       for role, n in (("calibration", 100), ("evaluation", 20))}
        report = {"status": "prepared", "seed": recipe["seed"], "schema": inputs["schema"].to_dict(),
                  "source": {"archive_sha256": sha256_file(source_zip)}, "files": fresh_files,
                  "scope": "synthetic same-source unseen-group confirmation"}
        previous_names = {"support": "support.csv", "calibration": "normal_calibration.csv",
                          "evaluation": "development_evaluation.csv"}
        report["seen_evidence"] = [
            {"round": "round13", "role": role, "sha256": sha256_file(previous / filename)}
            for role, filename in previous_names.items()]
        report["seen_evidence"] += [{"round": "round14", "role": role, "sha256": item["sha256"]}
                                    for role, item in files.items()]
        write_json(output / "preparation.json", report)
        return report
    monkeypatch.setattr(confirmation, "prepare_confirmation", prepare)
    class Predictor:
        def __init__(self, sign=1):
            self.sign = sign
        def predict_scores(self, x, batch_size=128):
            return self.sign * x[:, 0].astype(np.float64)
    def load_tabm(path, device="cpu"):
        counters["loads"].append(str(path))
        return Predictor(-1 if "seed29" in str(path) else 1)
    monkeypatch.setattr("ids.route_batch.tabm_training.load_predictor", load_tabm)
    monkeypatch.setattr("ids.route_batch.preprocessing.Representation.load", lambda path:
                        SimpleNamespace(transform=lambda frame: frame[["x"]].to_numpy(dtype=np.float32)))
    monkeypatch.setattr("ids.route_batch.foundation.load_foundation", lambda *args, **kwargs: Predictor())
    monkeypatch.setattr("ids.context_cache.csv_workflow.CSVDetector.load", lambda path:
                        SimpleNamespace(predict_frame=lambda frame: pd.DataFrame({"score": frame["x"]})))
    monkeypatch.setattr(confirmation.runner, "_archived_scores", lambda *args: pytest.fail("Old E cannot be paired with new E"))
    def subprocess_run(command, **kwargs):
        model_id = command[command.index("--model-id") + 1]
        counters["workers"].append((model_id, command[0]))
        counters.setdefault("retry_flags", []).append("--retry-failed" in command)
        confirmation.confirm_worker(root, model_id, retry_failed="--retry-failed" in command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(confirmation.subprocess, "run", subprocess_run)
    return SimpleNamespace(root=root, archive=archive, previous=previous, counters=counters, spec=spec)


def test_freezes_all_seeds_and_scores_only_new_data(fake_run):
    f = fake_run
    report = confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert report["models_scored"] == 5 and report["planned_models"] == 6
    assert report["status"] == "complete_with_unavailable_models"
    assert f.counters["prepare"] == 1 and len(f.counters["workers"]) == 5
    assert ("optional_pfn", "pfn-python") in f.counters["workers"]
    freeze = read_json(f.root / "confirmation" / "freeze.json")
    entries = {entry["id"]: entry for entry in freeze["models"]}
    assert entries["tabm_cost_failed"]["eligible"]
    assert entries["tabm_cost_failed"]["original_status"] == "failed"
    assert set(entries["reference_full"]["files"]) == {
        "model.pt", "preprocessor.joblib", "threshold.json", "deployment.json"}
    assert report["paired_vs_new_full"]["tabm_seed29"]["new_fn"] == 10
    metrics = read_json(f.root / "confirmation" / "models" / "reference_full" / "attempt-001" / "metrics.json")
    assert metrics["threshold"]["n_normal"] == 100
    assert "unseen communication groups" in metrics["threshold"]["scope"]
    assert "paired_vs_full" not in metrics


def test_resume_does_not_prepare_again_or_add_later_models(fake_run):
    f = fake_run
    confirmation.run_confirmation(f.root, f.archive, f.previous)
    freeze_path = f.root / "confirmation" / "freeze.json"
    original = freeze_path.read_bytes()
    later = f.root / "tasks" / "training_failed" / "attempt-002"
    later.mkdir()
    write_json(later / "result.json", {"status": "complete", "model": {"kind": "tabm"}})
    report = confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert freeze_path.read_bytes() == original
    assert report["models_scored"] == 5
    assert f.counters["prepare"] == 1 and len(f.counters["workers"]) == 5


def test_prepared_data_tamper_cannot_trigger_a_new_sample(fake_run):
    f = fake_run
    confirmation.run_confirmation(f.root, f.archive, f.previous)
    path = f.root / "confirmation" / "prepared" / "evaluation.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed or missing"):
        confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert f.counters["prepare"] == 1


def test_frozen_model_tamper_is_rejected_before_reuse(fake_run):
    f = fake_run
    confirmation.run_confirmation(f.root, f.archive, f.previous)
    model = f.root / "tasks" / "tabm_seed17" / "attempt-001" / "refit" / "model.pt"
    model.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or missing"):
        confirmation.confirm_worker(f.root, "tabm_seed17")


def test_one_confirmation_failure_does_not_suppress_other_models(fake_run, monkeypatch):
    f = fake_run
    original = confirmation._predictor
    def fail_one(entry, schema, batch_size):
        if entry["id"] == "tabm_seed17":
            raise RuntimeError("synthetic predictor failure")
        return original(entry, schema, batch_size)
    monkeypatch.setattr(confirmation, "_predictor", fail_one)
    report = confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert report["status"] == "partial" and report["models_scored"] == 4
    assert len(f.counters["workers"]) == 5
    failed = next(row for row in report["results"] if row["id"] == "tabm_seed17")
    assert failed["status"] == "failed"


def test_orphan_receipt_must_match_the_same_six_seen_csvs(fake_run):
    f = fake_run
    frozen, inputs = confirmation._create_freeze(f.root, f.archive, f.previous, None)
    prepared = f.root / "confirmation" / "prepared"
    report = confirmation.prepare_confirmation(f.archive, inputs, prepared,
        previous_prepared_dir=f.previous, **frozen["data_recipe"])
    report["seen_evidence"][0]["sha256"] = "0" * 64
    write_json(prepared / "preparation.json", report)
    with pytest.raises(ValueError, match="six seen CSVs"):
        confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert f.counters["prepare"] == 1
    assert not f.counters["workers"]


def test_initial_freeze_rejects_code_changed_since_original_run(fake_run, monkeypatch):
    f = fake_run
    monkeypatch.setattr(confirmation, "source_snapshot", lambda: {"source_sha256": "changed-code"})
    with pytest.raises(ValueError, match="original run source snapshot"):
        confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert not (f.root / "confirmation" / "freeze.json").exists()
    assert f.counters["prepare"] == 0


def test_foundation_measurement_failure_preserves_eligible_model(fake_run):
    f = fake_run
    path = f.root / "tasks" / "optional_pfn" / "attempt-001" / "result.json"
    record = read_json(path)
    record.update(status="measurement_failed", cost={"status": "failed", "error": "synthetic cost failure"})
    write_json(path, record)
    report = confirmation.run_confirmation(f.root, f.archive, f.previous)
    row = next(item for item in report["results"] if item["id"] == "optional_pfn")
    assert row["original_task_status"] == "measurement_failed"
    assert row["freeze_status"] == "ready" and row["status"] == "complete"


@pytest.mark.parametrize("failure_status", ["failed", "unavailable"])
def test_explicit_retry_only_retries_original_eligible_failures(fake_run, monkeypatch, failure_status):
    from ids.route_batch.foundation import FoundationUnavailable

    f = fake_run
    original = confirmation._predictor
    def fail_one(entry, schema, batch_size):
        if entry["id"] == "tabm_seed17":
            if failure_status == "unavailable":
                raise FoundationUnavailable("synthetic", "temporarily unavailable")
            raise RuntimeError("synthetic failure")
        return original(entry, schema, batch_size)
    monkeypatch.setattr(confirmation, "_predictor", fail_one)
    first = confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert first["models_scored"] == 4
    model_root = f.root / "confirmation" / "models" / "tabm_seed17"
    assert read_json(model_root / "attempt-001" / "result.json")["status"] == failure_status
    monkeypatch.setattr(confirmation, "_predictor", original)
    default_resume = confirmation.run_confirmation(f.root, f.archive, f.previous)
    assert default_resume["models_scored"] == 4 and len(f.counters["workers"]) == 5
    later = f.root / "tasks" / "training_failed" / "attempt-002"
    later.mkdir()
    write_json(later / "result.json", {"status": "complete", "model": {"kind": "tabm"}})
    retried = confirmation.run_confirmation(f.root, f.archive, f.previous, retry_failed=True)
    assert retried["models_scored"] == 5 and f.counters["prepare"] == 1
    assert len(f.counters["workers"]) == 6
    assert f.counters["workers"][-1][0] == "tabm_seed17" and f.counters["retry_flags"][-1]
    assert read_json(model_root / "attempt-001" / "result.json")["status"] == failure_status
    assert read_json(model_root / "attempt-002" / "result.json")["status"] == "complete"
    successful = f.root / "confirmation" / "models" / "tabm_seed29"
    assert len(list(successful.glob("attempt-*"))) == 1
    with pytest.raises(ValueError, match="unavailable at confirmation freeze"):
        confirmation.confirm_worker(f.root, "training_failed", retry_failed=True)
