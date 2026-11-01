"""Stage-3 integration: matched training, OOF provenance, resume and packaging."""
import copy
import importlib.util
from pathlib import Path
import shutil
import zipfile

import pytest
import yaml

from ids import benchmark_workflow, distillation, runner
from ids.common import code_fingerprint, read_json, sha256_file, write_json
from ids.data import make_synthetic_dataset, prepare_dataset, load_partition

REPO = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(f"stage3_test_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage3_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("stage3-template")
    raw = make_synthetic_dataset(root / "raw", n_train=600, n_test=120)
    data = root / "prepared"
    prepare_dataset(*raw, data, data_kind="synthetic", seed=17)
    suite = yaml.safe_load((REPO / "configs" / "distillation-smoke.yaml").read_text(encoding="utf-8"))
    suite["threads"] = 1
    fingerprint = code_fingerprint()
    loaded = []

    def development_load(path, split):
        assert split in {"train", "validation"}, "Stage 3 opened a reserved partition"
        loaded.append(split)
        return load_partition(path, split)

    with pytest.MonkeyPatch.context() as patch:
        for module in (runner, distillation):
            patch.setattr(module, "code_fingerprint", lambda: fingerprint)
            patch.setattr(module, "load_partition", development_load)
        runs = runner.run_suite(data, suite, root / "runs", device="cpu")
    return {"root": root, "data": data, "suite": suite, "runs": runs,
            "fingerprint": fingerprint, "loaded": loaded}


@pytest.fixture
def stage3(stage3_template, tmp_path, monkeypatch):
    root = tmp_path / "stage3"
    shutil.copytree(stage3_template["root"], root)
    for module in (runner, distillation):
        monkeypatch.setattr(module, "code_fingerprint", lambda: stage3_template["fingerprint"])
    return {"root": root, "data": root / "prepared", "suite": copy.deepcopy(stage3_template["suite"]),
            "run_root": root / "runs" / stage3_template["suite"]["suite_name"]}


def test_matched_student_training_records_verified_targets_but_deploys_only_student(stage3_template):
    from ids.models import load_bundle
    from ids.study import build_study
    root = stage3_template["root"] / "runs" / stage3_template["suite"]["suite_name"]
    hard, kd = root / "02-mlp-s17", root / "03-mlp-s17"
    assert len(stage3_template["runs"]) == 4
    assert set(stage3_template["loaded"]) == {"train", "validation"}
    hard_stats = read_json(hard / "training_statistics.json")
    kd_stats = read_json(kd / "training_statistics.json")
    assert hard_stats["parameter_count"] == kd_stats["parameter_count"]
    assert hard_stats["training_objective"] == "weighted_hard_bce"
    assert kd_stats["training_objective"] == "weighted_hard_bce_plus_unweighted_soft_bce"
    provenance = read_json(kd / "distillation.json")
    teacher = root / "teachers" / "03-oof-s17"
    assert provenance["targets_sha256"] == sha256_file(teacher / "targets.parquet")
    assert provenance["teacher_manifest_sha256"] == sha256_file(teacher / "manifest.json")
    assert provenance["training_split"] == "train"
    assert provenance["oof_manifest"]["folds"] == 3
    bundle = load_bundle(kd / "model.joblib", device="cpu")
    assert not any(key in bundle.__dict__ for key in ("soft_targets", "teacher", "teacher_models"))
    assert len(bundle.selected_features) == 42
    study = build_study(root)
    assert study["n_runs"] == 4
    assert len(study["groups"]) == 4


def test_suite_resume_does_not_retrain_completed_teacher_or_students(stage3, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Completed stage-3 resume must not fit again")
    monkeypatch.setattr(runner, "train_model", forbidden)
    monkeypatch.setattr(distillation, "train_model", forbidden)
    runs = runner.run_suite(stage3["data"], stage3["suite"], stage3["root"] / "runs", resume=True)
    assert len(runs) == 4


def test_resume_rejects_tampered_teacher_targets(stage3):
    target = stage3["run_root"] / "teachers" / "03-oof-s17" / "targets.parquet"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.run_suite(stage3["data"], stage3["suite"], stage3["root"] / "runs", resume=True)


def test_completed_student_cannot_silently_regenerate_its_teacher(stage3, monkeypatch):
    teacher = stage3["run_root"] / "teachers" / "03-oof-s17"
    (teacher / "completion.json").unlink()
    def forbidden(*_args, **_kwargs):
        pytest.fail("A completed student's teacher must never be regenerated")
    monkeypatch.setattr(distillation, "build_oof_targets", forbidden)
    with pytest.raises(ValueError, match="original complete OOF"):
        runner.run_suite(stage3["data"], stage3["suite"], stage3["root"] / "runs", resume=True)


def test_completion_fast_path_still_checks_development_input_bytes(stage3):
    launcher = script("run_stage3")
    (stage3["data"] / "train.parquet").write_bytes(b"damaged-input")
    from types import SimpleNamespace
    args = SimpleNamespace(data_dir=stage3["data"], work_dir=stage3["root"] / "completed", device="cpu", resume=True)
    with pytest.raises(ValueError, match="Partition checksum mismatch: train"):
        launcher.prepare_plan(args, stage3["suite"])


def test_zero_alpha_never_generates_a_teacher(stage3, monkeypatch):
    suite = copy.deepcopy(stage3["suite"])
    suite["suite_name"] = "alpha-zero-integration"
    suite["models"] = [suite["models"][-1]]
    suite["models"][0]["distillation"]["alpha"] = 0

    def forbidden(*_args, **_kwargs):
        pytest.fail("alpha=0 must not build teacher targets")
    monkeypatch.setattr(distillation, "build_oof_targets", forbidden)
    run = runner.run_suite(stage3["data"], suite, stage3["root"] / "runs")[0]
    assert not (run / "distillation.json").exists()
    assert read_json(run / "training_statistics.json")["training_objective"] == "weighted_hard_bce"


def test_launcher_dry_run_has_three_sequential_rounds_and_no_final_evaluation(stage3, monkeypatch, capsys):
    launcher = script("run_stage3")
    def forbidden(*_args, **_kwargs):
        pytest.fail("dry-run must not launch processes")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    output = stage3["root"] / "new work with spaces"
    assert launcher.main(["--data-dir", str(stage3["data"]), "--work-dir", str(output), "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert text.count("'benchmark-suite'") == 3
    assert "'20261101'" in text and "'20261102'" in text and "'20261103'" in text
    assert "'evaluate'" not in text and "'freeze'" not in text
    assert not output.exists()


def test_launcher_rejects_unmatched_student_training_budgets():
    launcher = script("run_stage3")
    suite = yaml.safe_load((REPO / "configs" / "distillation.yaml").read_text(encoding="utf-8"))
    launcher.validate_design(suite)
    suite["models"][-1]["hyperparameters"] = {**suite["models"][-1]["hyperparameters"], "epochs": 30}
    with pytest.raises(ValueError, match="identical architecture"):
        launcher.validate_design(suite)


def test_benchmark_round_is_bound_to_resume_plan(stage3, monkeypatch):
    calls = []
    fingerprint = code_fingerprint()
    monkeypatch.setattr(benchmark_workflow, "code_fingerprint", lambda: fingerprint)
    monkeypatch.setattr(benchmark_workflow, "environment_snapshot", lambda: {"cpu_physical": 1})
    def fast(run, data, output, threads, measurement):
        calls.append(run.name)
        write_json(output, {**benchmark_workflow.run_identity(run, data), **measurement,
                            "threads": threads, "benchmark_code_sha256": fingerprint["source_sha256"]})
    monkeypatch.setattr(benchmark_workflow, "benchmark_run", fast)
    output = stage3["root"] / "round-1"
    benchmark_workflow.benchmark_suite(stage3["run_root"], stage3["data"], output,
                                       measurement_round=1, planned_rounds=3)
    assert len(calls) == 4
    with pytest.raises(ValueError, match="Cannot resume"):
        benchmark_workflow.benchmark_suite(stage3["run_root"], stage3["data"], output,
                                           resume=True, measurement_round=2, planned_rounds=3)


def test_package_validates_completion_bytes_and_contains_inventory(tmp_path):
    packager = script("package_run")
    root = tmp_path / "complete"
    root.mkdir()
    (root / "evidence.json").write_text('{"synthetic": true}\n', encoding="utf-8")
    suite = {"models": [{"model": "mlp"}], "seeds": [17], "benchmark_rounds": 1}
    write_json(root / "STAGE3_PLAN.json", {"identity": {"suite": suite}, "expected_runs": 1})
    write_json(root / "reports" / "distillation-study.json", {"n_runs": 1,
        "groups": [{"seeds": [17], "cpu_benchmarks": [{}]}]})
    (root / "reports" / "distillation-study.html").write_text("synthetic test report", encoding="utf-8")
    (root / "reports" / "validation-records.html").write_text("synthetic test records", encoding="utf-8")
    write_json(root / "STAGE3_COMPLETE.json", {"phase": "validation_development", "n_runs": 1,
        "benchmark_rounds": 1,
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}})
    archive = packager.package_run(root, tmp_path / "verified.zip")
    with zipfile.ZipFile(archive) as zipped:
        assert "complete/MANIFEST.sha256" in zipped.namelist()
        assert sha256_file(root / "evidence.json") in zipped.read("complete/MANIFEST.sha256").decode()
    assert archive.with_suffix(".zip.sha256").read_text().strip() == sha256_file(archive)
    extra = root / "unsealed-model.joblib"
    extra.write_bytes(b"unsealed")
    with pytest.raises(ValueError, match="Unsealed additional"):
        packager.package_run(root, tmp_path / "unsealed.zip")
    extra.unlink()
    (root / "evidence.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        packager.package_run(root, tmp_path / "must-not-exist.zip")
    assert not (tmp_path / "must-not-exist.zip").exists()
