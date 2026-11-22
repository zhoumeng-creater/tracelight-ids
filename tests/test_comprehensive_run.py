"""Synthetic task queues and phase barriers; never construct real-data models."""
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts.comprehensive_batch import run


def plan():
    return {"hgb_jobs": [{"job_id": "HGB_mixed_ports_zero"}],
            "tabm_jobs": [{"job_id": arm, "arm": arm, "seed": 17, "learning_rate": .002,
                           "auxiliary_weight": 1., "checkpoints": [1, 2], "optimizer_steps": 2}
                          for arm in ("A_BCE", "R_reference_envelope")]}


class Batch:
    def __init__(self, root, failure=None):
        self.root, self.failure, self.calls = root, failure, []

    def fit(self, rows, processor, directory, *, job, **kwargs):
        self.calls.append(job["job_id"])
        directory.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for step in (1, 2):
            if step == 2 and job["arm"].startswith("R_") and self.failure:
                run.write(directory / "snapshots.json", snapshots)
                raise self.failure("synthetic interruption")
            path = directory / f"step{step}.pt"
            path.write_text(f"synthetic {job['job_id']} {step}", encoding="utf8")
            snapshots.append({"step": step, "path": path.name, "sha256": run.data.file_digest(path)})
        return {"snapshots": snapshots, "initial_state_sha256": "paired-init",
                "sampler_draw_sha256": "paired-sampler"}

    def hgb(self, rows, directory, *, job, reuse=None):
        self.calls.append(job["job_id"])
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.joblib"
        path.write_text("synthetic hgb", encoding="utf8")
        return {"snapshots": [{"step": 200, "path": path.name, "sha256": run.data.file_digest(path)}]}

    def train(self, **kwargs):
        return run.train_phase(self.root, [], None, plan(), fit_fn=self.fit, hgb_fn=self.hgb, **kwargs)


def test_numerical_failure_retains_snapshots_and_other_jobs_without_refitting(tmp_path):
    batch = Batch(tmp_path, FloatingPointError)
    models = batch.train()
    assert len(models) == 4
    frozen = run.read(tmp_path / "models_frozen.json")
    assert frozen["all_registered_jobs_terminal"]
    assert frozen["jobs"]["R_reference_envelope"]["status"] == "failed_numerical"
    assert models["R_reference_envelope_step1"]["job_status"] == "failed_numerical"
    attempts = run.read(tmp_path / "job_invocations/R_reference_envelope.json")
    assert len(attempts) == 1 and attempts[0]["status"] == "failed_numerical"
    assert attempts[0]["wall_seconds"] >= 0
    batch.calls.clear()
    assert batch.train(resume=True) == models
    assert not batch.calls
    assert run.read(tmp_path / "job_invocations/R_reference_envelope.json") == attempts


def test_resource_interruption_does_not_open_calibration_barrier(tmp_path):
    batch = Batch(tmp_path, OSError)
    with pytest.raises(OSError):
        batch.train()
    assert not (tmp_path / "models_frozen.json").exists()
    assert run.read(tmp_path / "job_statuses/R_reference_envelope.json")["status"] == "interrupted_resource"
    assert run.read(tmp_path / "job_invocations/R_reference_envelope.json")[0]["status"] == "interrupted_resource"


@pytest.mark.parametrize("message", ["CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered", "CUBLAS_STATUS_ALLOC_FAILED", "CUDA out of memory"])
def test_device_fault_interrupts_batch_instead_of_becoming_algorithm_failure(tmp_path, message):
    batch = Batch(tmp_path, lambda ignored: RuntimeError(message))
    with pytest.raises(RuntimeError, match=message):
        batch.train()
    assert not (tmp_path / "models_frozen.json").exists()
    assert run.read(tmp_path / "job_statuses/R_reference_envelope.json")["status"] == "interrupted_resource"


def test_non_device_runtime_error_does_not_hide_as_resource_interruption():
    assert not run._resource_error(RuntimeError("invalid registered tensor shape"))


def test_complete_snapshots_all_calibrate_before_any_development_and_cache_resume(tmp_path):
    batch = Batch(tmp_path)
    models = batch.train()
    calls = []
    rows = [{"flow_id": "c", "dataset_role": "C", "label": 0},
            {"flow_id": "v", "dataset_role": "V", "label": 1}]

    def calibration(repo):
        assert (tmp_path / "models_frozen.json").exists()
        calls.append("C")
        return rows[:1]

    def development(repo):
        assert set(run.read(tmp_path / "thresholds.json")) == set(models)
        assert (tmp_path / "thresholds_frozen.json").exists()
        calls.append("V")
        return rows, {"C": ["c"]}, [], []

    def loader(path, meta):
        class Predictor:
            def score(self, received):
                calls.append("score_" + received[0]["dataset_role"])
                if received[0]["dataset_role"] != "C":
                    assert (tmp_path / "thresholds_frozen.json").exists()
                return np.asarray([float(row["label"]) for row in received])
        return Predictor()

    def writer(received_rows, cohorts, events, units, metadata, thresholds, score_dir, out, received_plan):
        assert received_rows == rows
        assert set(metadata) == set(thresholds) == set(models)
        for identity in models:
            with np.load(score_dir / f"{identity}.npz") as f:
                np.testing.assert_array_equal(f["scores"], [0., 1.])
        return {"synthetic": True}

    kwargs = dict(calibration_loader=calibration, evaluation_loader=development, loader=loader, writer=writer)
    run.evaluate_phase(tmp_path, plan(), models, **kwargs)
    assert calls == ["C"] + ["score_C"] * len(models) + ["V"] + ["score_V"] * len(models)
    calls.clear()
    run.evaluate_phase(tmp_path, plan(), models, **kwargs)
    assert calls == ["C", "V"]  # Completed raw scores are reused; no model scoring repeats.
    with pytest.raises(RuntimeError, match="training can never resume"):
        batch.train(resume=True)


def test_modified_artifact_prevents_scoring_on_resume(tmp_path):
    batch = Batch(tmp_path)
    models = batch.train()
    (tmp_path / next(iter(models.values()))["path"]).write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="binding differs"):
        batch.train(resume=True)


def test_package_contains_all_inference_snapshots_and_excludes_lab_recovery(tmp_path):
    output = tmp_path / "this_batch"
    output.mkdir()
    for name, value in (("model.pt", "model"), ("recovery.pt", "optimizer state"),
                        ("failure.json", '{"failed":true}')):
        (output / name).write_text(value, encoding="utf8")
    archive = run.package(output)
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"this_batch/model.pt", "this_batch/failure.json", "this_batch/output_receipt.json"}
    assert (output / "recovery.pt").is_file()
    assert run.read(Path(str(archive) + ".sha256.json"))["sha256"] == run.data.file_digest(archive)
