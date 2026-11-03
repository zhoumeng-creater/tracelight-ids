"""V4 lab orchestration uses mock GPU/manifests/runner, never real-data fits."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import yaml

from ids.cicids import CIC_CATEGORICAL, CIC_DISCRETE, CIC_FEATURES, CIC_SCHEMA_VERSION
from ids.data import FEATURES, SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upgrade_lab_script", REPO / "scripts/run_upgrade_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def hardware(name="NVIDIA GeForce RTX 4090 D"):
    return {"recorded_at": "fake-clock", "cpu_logical": 16, "packages": {"torch": "2.6.0+cu126"},
            "torch": {"cuda_available": True, "cuda_runtime": "12.6",
                      "gpus": [{"index": 0, "name": name, "total_memory_bytes": 24 * 1024 ** 3}]}}


def config(profile):
    return {"schema_version": 1, "purpose": "upgrade_v4_development_only",
            "seeds": [17] if profile == "pilot" else [17, 29, 43], "threads": 2, "max_fpr": .01,
            "primary_working_point": "recalibrated_reference", "conditions": sorted(lab.CONDITIONS),
            "optimization": {"macro_steps": 30 if profile == "pilot" else 60}}


def manifest(name):
    if name == "unsw":
        value = {"schema_version": SCHEMA_VERSION, "features": list(FEATURES), "data_kind": "mirror"}
    else:
        value = {"schema_version": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
                 "data_kind": "official_repaired", "feature_schema": {
                     "name": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
                     "categorical": list(CIC_CATEGORICAL), "discrete": list(CIC_DISCRETE)}}
    value["partitions"] = {split: {"file": split + ".parquet", "n_rows": 100, "sha256": "b" * 64}
                           for split in ("train", "validation", "calibration", "test")}
    return value


def save_manifest(path, value):
    put_json(path / "manifest.json", value)
    (path / "manifest.sha256").write_text(lab.sha256_file(path / "manifest.json") + "\n", encoding="ascii")


def old_v3(path, data):
    plan = {"config": {"purpose": "risk_v3_development_only", "cohort_policy": "complete_development",
                       "seeds": [17, 29, 43], "max_train_rows": None, "max_validation_rows": None},
            "data": data, "source": {"commit": "f" * 40, "source_sha256": "d" * 64},
            "role_manifest_hash": "c" * 64, "reserved_partitions_read": []}
    plan["plan_id"] = lab.canonical_hash(plan)
    put_json(path / "plan.json", plan)
    # These nonexistent teacher and frame payloads are intentional: metadata-only
    # dry-run must not try loading models, parquet or the full V3 verifier.
    put_json(path / lab.V3_COMPLETE, {"status": "development_complete", "reserved_partitions_read": [],
             "plan_id": plan["plan_id"], "files": {"plan.json": lab.sha256_file(path / "plan.json"),
             "teachers/s17/reference.joblib": "e" * 64}})


def seal(path, cfg, v3, data, *, skipped=0):
    path.mkdir(parents=True)
    put_json(path / "plan.json", {"config": cfg, "v3_dir": str(Path(v3).resolve()),
                                  "data_dir": str(Path(data).resolve())})
    (path / "result.txt").write_text("mock diagnostics; no model fitted", encoding="utf-8")
    planned = len(cfg["seeds"]) * len(cfg["conditions"])
    put_json(path / lab.COMPLETE, {"status": "development_complete", "reserved_partitions_read": [],
             "n_students_planned": planned, "n_students": planned - skipped,
             "n_skipped_no_feasible_start": skipped, "n_teacher_fits": 0, "n_anchor_fits": 0,
             "files": {name: lab.sha256_file(path / name) for name in ("plan.json", "result.txt")}})


def fake_verify(path):
    path = Path(path)
    complete = lab.read_json(path / lab.COMPLETE)
    for name, digest in complete["files"].items():
        if lab.sha256_file(path / name) != digest:
            raise ValueError("mock completed artifact changed")
    return complete


@pytest.fixture
def ready(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    for profile in ("pilot", "full"):
        (repo / "configs" / f"upgrade-v4-{profile}.yaml").write_text(yaml.safe_dump(config(profile)), encoding="utf-8")
    monkeypatch.setattr(lab, "REPO", repo)
    monkeypatch.setattr(lab, "validate_upgrade_config", lambda cfg: None)
    monkeypatch.setattr(lab, "check_environment", hardware)
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: None)
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "a" * 64, "files": {}})
    monkeypatch.setattr(lab, "verify_upgrade_research", fake_verify)
    inputs, calls = {}, []
    for name in ("unsw", "cicids"):
        data, source = tmp_path / ("data-" + name), tmp_path / ("v3-" + name)
        value = manifest(name)
        save_manifest(data, value)
        old_v3(source, value)
        inputs[name] = {"data_dir": data, "v3_dir": source}

    def runner(v3, data, cfg, output, **kwargs):
        calls.append((Path(v3), Path(data), copy.deepcopy(cfg), Path(output), kwargs))
        if kwargs["resume"]:
            if lab.read_json(Path(output) / "plan.json") != {
                    "config": cfg, "v3_dir": str(Path(v3).resolve()), "data_dir": str(Path(data).resolve())}:
                raise ValueError("mock runner plan identity mismatch")
        else:
            seal(Path(output), cfg, v3, data)
        return fake_verify(output)

    monkeypatch.setattr(lab, "run_upgrade_research", runner)
    return inputs, calls


def arguments(inputs, *names):
    return {f"{name}_{key}": value for name in names for key, value in inputs[name].items()}


@pytest.mark.parametrize("kwargs", [{}, {"unsw_v3_dir": "v3"}, {"cicids_data_dir": "data"},
                                    {"unsw_v3_dir": "v3", "cicids_data_dir": "data"}])
def test_environment_pairs_required_before_any_probe(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("probe after invalid paths"))
    with pytest.raises(ValueError, match="environment|Each"):
        lab.run_lab(work_dir=tmp_path / "work", **kwargs)


def test_wrong_gpu_rejected_before_source_or_prepared_access(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "check_environment", lambda: hardware("RTX4050 Laptop GPU"))
    monkeypatch.setattr(lab, "_dataset_record", lambda *a: pytest.fail("metadata read on wrong GPU"))
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="CUDA0"):
        lab.run_lab(work_dir=work, dry_run=True, **arguments(inputs, "unsw"))
    assert not calls and not work.exists()


def test_pilot_both_environments_metadata_only_no_payloads(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, dry_run=True, **arguments(inputs, "unsw", "cicids"))
    assert result["status"] == "upgrade_v4_lab_plan_only" and calls == []
    assert result["expected_all_environments"]["students_planned"] == 16
    assert result["expected_all_environments"]["teacher_fits"] == 0
    assert not (work / "upgrade-v4-unsw").exists() and not (work / "packages").exists()
    saved = lab.read_json(work / lab.PREFLIGHT)
    assert saved["binding"]["interpreter"]["executable"] == str(Path(sys.executable).absolute())
    assert saved["binding"]["inputs"]["cicids"]["v3"]["completion_sha256"] == lab.sha256_file(inputs["cicids"]["v3_dir"] / lab.V3_COMPLETE)
    assert saved["reserved_partitions_read"] == []


@pytest.mark.parametrize("profile,n", [("pilot", 8), ("full", 24)])
def test_serial_complete_package_and_original_receipt(ready, tmp_path, profile, n):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, profile=profile, **arguments(inputs, "unsw", "cicids"))
    assert result["status"] == "upgrade_v4_lab_development_complete"
    assert [call[0] for call in calls] == [inputs["unsw"]["v3_dir"], inputs["cicids"]["v3_dir"]]
    assert result["expected_per_environment"]["students_planned"] == n
    for call in calls:
        assert call[-1] == {"device": "cuda", "threads": 2, "resume": False}
    for name in ("UNSW", "CICIDS"):
        path = work / "packages" / f"UPGRADE_V4_{name}.zip"
        assert path.with_name(path.name + ".sha256").read_text().strip() == lab.sha256_file(path)
        with zipfile.ZipFile(path) as archive:
            assert lab.PACKAGE_MANIFEST in archive.namelist()
            assert not any("teachers/" in member or "reference.joblib" in member for member in archive.namelist())
            assert "upgrade-v4-" + name.lower() + "/plan.json" in archive.namelist()
    receipt = work / "packages" / lab.PREFLIGHT
    assert receipt.read_bytes() == (work / lab.PREFLIGHT).read_bytes()
    assert receipt.with_name(receipt.name + ".sha256").read_text().strip() == lab.sha256_file(receipt)


def test_no_feasible_start_is_valid_diagnostic_completion(ready, tmp_path, monkeypatch):
    inputs, calls = ready

    def skipped(v3, data, cfg, output, **kwargs):
        seal(Path(output), cfg, v3, data, skipped=8)

    monkeypatch.setattr(lab, "run_upgrade_research", skipped)
    result = lab.run_lab(work_dir=tmp_path / "work", **arguments(inputs, "unsw"))
    assert result["status"] == "upgrade_v4_lab_development_complete"
    assert not calls


def test_completed_resume_calls_runner_to_revalidate_its_plan(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    args = arguments(inputs, "unsw")
    lab.run_lab(work_dir=work, **args)
    original = (work / lab.PREFLIGHT).read_bytes()
    result = lab.run_lab(work_dir=work, resume=True, **args)
    assert calls[-1][-1]["resume"] is True
    assert result["environments"]["unsw"]["package"]["status"] == "verified_existing"
    assert (work / lab.PREFLIGHT).read_bytes() == original


def test_dry_run_same_directory_can_become_run_with_explicit_resume(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    args = arguments(inputs, "cicids")
    lab.run_lab(work_dir=work, dry_run=True, **args)
    lab.run_lab(work_dir=work, resume=True, **args)
    assert len(calls) == 1 and calls[0][-1]["resume"] is False


def test_partial_output_is_preserved_and_never_dispatched(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    args = arguments(inputs, "unsw")
    lab.run_lab(work_dir=work, dry_run=True, **args)
    partial = work / "upgrade-v4-unsw"
    partial.mkdir()
    (partial / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        lab.run_lab(work_dir=work, resume=True, **args)
    assert calls == [] and (partial / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("field", ["source", "threads", "v3", "interpreter"])
def test_resume_binding_changes_rejected(ready, tmp_path, monkeypatch, field):
    inputs, calls = ready
    work = tmp_path / "work"
    args = arguments(inputs, "unsw")
    lab.run_lab(work_dir=work, dry_run=True, **args)
    if field == "source":
        monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "b" * 64, "files": {}})
    elif field == "threads":
        args["threads"] = 3
    elif field == "v3":
        path = inputs["unsw"]["v3_dir"] / lab.V3_COMPLETE
        value = lab.read_json(path)
        value["created_at"] = "different"
        put_json(path, value)
    else:
        monkeypatch.setattr(sys, "executable", str(tmp_path / "different-python.exe"))
    with pytest.raises(ValueError, match="binding changed"):
        lab.run_lab(work_dir=work, resume=True, **args)
    assert calls == []


def test_prepared_identity_differs_from_source_v3_rejected(ready, tmp_path):
    inputs, calls = ready
    path = inputs["unsw"]["data_dir"]
    value = manifest("unsw")
    value["partitions"]["train"]["n_rows"] += 1
    save_manifest(path, value)
    with pytest.raises(ValueError, match="original V3 data identity"):
        lab.run_lab(work_dir=tmp_path / "work", dry_run=True, **arguments(inputs, "unsw"))
    assert calls == []


@pytest.mark.parametrize("change", ["plan_checksum", "plan_id", "reserved", "bounded", "missing_seed"])
def test_source_v3_metadata_defects_rejected_before_training(ready, tmp_path, change):
    inputs, calls = ready
    root = inputs["unsw"]["v3_dir"]
    plan = lab.read_json(root / "plan.json")
    done = lab.read_json(root / lab.V3_COMPLETE)
    if change == "reserved":
        done["reserved_partitions_read"] = ["test"]
    else:
        if change == "plan_id":
            plan["plan_id"] = "0" * 64
        elif change == "bounded":
            plan["config"]["cohort_policy"] = "bounded_pilot"
        elif change == "missing_seed":
            plan["config"]["seeds"] = [29, 43]
        else:
            plan["extra"] = 1
        if change not in {"plan_checksum", "plan_id"}:
            plan["plan_id"] = lab.canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})
        put_json(root / "plan.json", plan)
        if change != "plan_checksum":
            done["plan_id"] = plan["plan_id"]
            done["files"]["plan.json"] = lab.sha256_file(root / "plan.json")
    put_json(root / lab.V3_COMPLETE, done)
    with pytest.raises(ValueError):
        lab.run_lab(work_dir=tmp_path / "work", **arguments(inputs, "unsw"))
    assert calls == []


@pytest.mark.parametrize("which", ["data_dir", "v3_dir"])
def test_work_cannot_nest_in_immutable_inputs(ready, which):
    inputs, calls = ready
    work = inputs["unsw"][which] / "new-work"
    with pytest.raises(ValueError, match="non-nested"):
        lab.run_lab(work_dir=work, **arguments(inputs, "unsw"))
    assert calls == [] and not work.exists()


@pytest.mark.parametrize("which", ["source", "v3"])
def test_input_change_during_runner_blocks_package(ready, tmp_path, monkeypatch, which):
    inputs, calls = ready
    original = lab.run_upgrade_research

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if which == "source":
            monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "f" * 64, "files": {}})
        else:
            path = inputs["unsw"]["v3_dir"] / lab.V3_COMPLETE
            value = lab.read_json(path)
            value["changed"] = True
            put_json(path, value)
        return result

    monkeypatch.setattr(lab, "run_upgrade_research", changed)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="changed during"):
        lab.run_lab(work_dir=work, **arguments(inputs, "unsw"))
    assert len(calls) == 1 and not (work / "packages").exists()


def test_corrupt_existing_zip_rejected_on_resume(ready, tmp_path):
    inputs, _ = ready
    work = tmp_path / "work"
    args = arguments(inputs, "unsw")
    lab.run_lab(work_dir=work, **args)
    package = work / "packages" / "UPGRADE_V4_UNSW.zip"
    with package.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        lab.run_lab(work_dir=work, resume=True, **args)


@pytest.mark.parametrize("change", [{"seeds": [29]}, {"conditions": ["hard"]},
                                    {"purpose": "final_test"}, {"max_train_rows": 100}])
def test_profile_rejects_ambiguous_matrix_or_caps(ready, change):
    value = config("pilot")
    value.update(change)
    lab._profile_path("pilot").write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError):
        lab.load_profile("pilot")


def test_help_available_from_other_directory_without_trainer(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_upgrade_lab.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0 and "--unsw-v3-dir" in result.stdout and "--cicids-v3-dir" in result.stdout
