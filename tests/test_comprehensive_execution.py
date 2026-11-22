"""Generated job artifacts exercise scheduling, stopping and explicit adoption."""
from io import BytesIO
import copy
from pathlib import Path

import pytest

from scripts.comprehensive_batch import run, execution


def tiny_plan():
    return {"hgb_jobs": [{"job_id": "HGB_generated", "family": "hgb"}],
            "tabm_jobs": [{"job_id": name, "family": "tabm", "seed": 17,
                           "checkpoints": [2], "arm": name} for name in ("A", "G", "R")]}


class Launcher:
    def __init__(self, failure=None):
        self.processes, self.started = [], []
        self.peak_gpu = self.peak_cpu = 0
        self.failure = failure

    def __call__(self, output, job, **kwargs):
        owner = self
        class Process:
            def __init__(self):
                self.finished = False
                self.calls = 0
                self.family = job["family"]
            def poll(self):
                if not self.finished:
                    self.calls += 1
                    if self.calls < 3:
                        return None
                    directory = output / "fits" / job["job_id"]
                    directory.mkdir(parents=True, exist_ok=True)
                    artifact = directory / "model.pt"
                    artifact.write_bytes(b"generated artifact")
                    status = "interrupted_resource" if owner.failure == job["job_id"] else "complete"
                    run.write(output / "job_statuses" / f"{job['job_id']}.json", {
                        "status": status, "job": job, "initial_state_sha256": "same-generated-init",
                        "sampler_draw_sha256": "same-generated-draws",
                        "snapshots": [{"step": 200 if self.family == "hgb" else 2,
                                       "path": artifact.name, "sha256": run.data.file_digest(artifact)}]})
                    self.finished = True
                return 1 if owner.failure == job["job_id"] else 0
            def terminate(self):
                self.finished = True
            def wait(self, timeout=None):
                self.finished = True
                return 0
            def kill(self):
                self.finished = True
        process = Process()
        self.processes.append(process)
        self.started.append(job["job_id"])
        self.peak_gpu = max(self.peak_gpu, sum(not p.finished and p.family == "tabm" for p in self.processes))
        self.peak_cpu = max(self.peak_cpu, sum(not p.finished and p.family == "hgb" for p in self.processes))
        return {"process": process, "stream": BytesIO(), "job": job, "log": "generated.log"}


def test_two_queues_overlap_freeze_at_end_and_resume_skips_every_completed_job(tmp_path):
    launcher = Launcher()
    models = execution.train_parallel(tmp_path, tiny_plan(), {"gpu_workers": 2, "cpu_workers": 1},
                                      launcher=launcher, poll_interval=0)
    assert len(models) == 4
    assert launcher.started[:3] == ["A", "G", "HGB_generated"]
    assert launcher.peak_gpu == 2 and launcher.peak_cpu == 1
    assert run.read(tmp_path / "models_frozen.json")["all_registered_jobs_terminal"]
    resumed = Launcher()
    assert execution.train_parallel(tmp_path, tiny_plan(), {"gpu_workers": 2, "cpu_workers": 1},
                                    launcher=resumed, poll_interval=0, resume=True) == models
    assert resumed.started == []


def test_resource_failure_stops_queues_and_never_freezes_or_opens_calibration(tmp_path):
    launcher = Launcher(failure="A")
    with pytest.raises(RuntimeError, match="Worker interrupted"):
        execution.train_parallel(tmp_path, tiny_plan(), {"gpu_workers": 2, "cpu_workers": 1},
                                 launcher=launcher, poll_interval=0)
    assert (tmp_path / "execution_stop.request").exists()
    assert "R" not in launcher.started
    assert all(process.finished for process in launcher.processes)
    assert not (tmp_path / "models_frozen.json").exists()
    assert not (tmp_path / "calibration_started.json").exists()


def test_parallel_training_cannot_resume_after_calibration(tmp_path):
    run.write(tmp_path / "calibration_started.json", {})
    with pytest.raises(RuntimeError, match="no more training"):
        execution.train_parallel(tmp_path, tiny_plan(), {"gpu_workers": 2, "cpu_workers": 1})


def test_output_lock_excludes_another_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "output_processes", lambda output: [])
    with execution.exclusive_output(tmp_path):
        with pytest.raises(OSError):
            with execution.exclusive_output(tmp_path):
                pytest.fail("two coordinators acquired the same output")
    with execution.exclusive_output(tmp_path):
        pass


def test_existing_output_worker_prevents_adoption_or_new_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "output_processes", lambda output: [{"pid": 123}])
    with pytest.raises(RuntimeError, match="active workers"):
        with execution.exclusive_output(tmp_path):
            pytest.fail("live worker ignored")


def legacy_fixture(tmp_path):
    repo, output = tmp_path / "repo", tmp_path / "result"
    output.mkdir()
    baseline_path = repo / run.DIRECTORY / "resource_v1_registration.json"
    baseline = {"release_tag": "comprehensive-calibration-v1", "generated": True}
    run.write(baseline_path, baseline)
    plan = {"hgb_jobs": [], "tabm_jobs": []}
    old = {"registration_sha256_lf": run.text_digest(baseline_path), "registration": baseline,
           "plan_sha256": run.data.json_digest(plan), "coverage_results": None}
    run.write(output / "source_receipt.json", old)
    new = {**old, "registration_sha256_lf": "new-generated-binding",
           "registration": {"release_tag": "comprehensive-calibration-v1.1",
                            "legacy_registration_sha256_lf": run.text_digest(baseline_path)}}
    return repo, output, plan, old, new


def test_explicit_adoption_preserves_old_receipt_and_plan_without_refitting(tmp_path, monkeypatch):
    repo, output, plan, old, new = legacy_fixture(tmp_path)
    monkeypatch.setattr(execution, "output_processes", lambda output: [])
    original = (output / "source_receipt.json").read_bytes()
    execution.adopt_v1(output, new, plan, repo=repo)
    assert (output / "source_receipt_v1.json").read_bytes() == original
    assert run.read(output / "source_receipt.json") == new
    assert not run.read(output / "execution_adoption.json")["scientific_job_definitions_changed"]
    assert not (output / "fits").exists()


def test_unregistered_old_output_cannot_be_adopted(tmp_path, monkeypatch):
    repo, output, plan, old, new = legacy_fixture(tmp_path)
    monkeypatch.setattr(execution, "output_processes", lambda output: [])
    old["registration"]["generated"] = False
    run.write(output / "source_receipt.json", old)
    with pytest.raises(ValueError, match="unchanged"):
        execution.adopt_v1(output, new, plan, repo=repo)
    assert not (output / "source_receipt_v1.json").exists()
    assert run.read(output / "source_receipt.json") == old
