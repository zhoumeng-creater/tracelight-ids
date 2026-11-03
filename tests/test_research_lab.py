"""Lab-location gates and verified packaging without real-data training."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("research_lab_script", REPO / "scripts/run_research_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def hardware(name="NVIDIA GeForce RTX 4090 D", *, available=True, index=0):
    return {"recorded_at": "2026-11-03T12:00:00+08:00", "cpu_logical": 16,
            "ram_total_bytes": 64 * 1024 ** 3, "packages": {"torch": "2.6.0+cu126"},
            "torch": {"cuda_available": available, "cuda_runtime": "12.6",
                      "gpus": [{"index": index, "name": name, "total_memory_bytes": 24 * 1024 ** 3}]}}


def manifest(name):
    if name == "unsw":
        return {"schema_version": lab.SCHEMA_VERSION, "features": list(lab.FEATURES), "data_kind": "mirror",
                "partitions": {split: {"n_rows": 10} for split in ("train", "validation")}}
    schema = {"name": lab.CIC_SCHEMA_VERSION, "features": list(lab.CIC_FEATURES),
              "categorical": list(lab.CIC_CATEGORICAL),
              "discrete": [x for x in lab.CIC_FEATURES if x in lab.CIC_DISCRETE]}
    return {"schema_version": lab.CIC_SCHEMA_VERSION, "features": list(lab.CIC_FEATURES),
            "feature_schema": schema, "data_kind": "official_repaired",
            "partitions": {split: {"n_rows": 20} for split in ("train", "validation")}}


def seal(root):
    root.mkdir(parents=True)
    (root / "result.txt").write_text("fake engineering result, not trained", encoding="utf8")
    lab.write_json(root / "RESEARCH_COMPLETE.json", {"status": "development_complete",
                   "files": {"result.txt": lab.sha256_file(root / "result.txt")}})


def fake_verify(root):
    root = Path(root)
    result = lab.read_json(root / "RESEARCH_COMPLETE.json")
    assert result["status"] == "development_complete"
    for name, digest in result["files"].items():
        if lab.sha256_file(root / name) != digest:
            raise ValueError("fixture completed research artifact changed")
    return result


@pytest.fixture
def ready(tmp_path, monkeypatch):
    inputs = {}
    for name in ("unsw", "cicids"):
        path = tmp_path / ("data-" + name)
        path.mkdir()
        lab.write_json(path / "manifest.json", manifest(name))
        inputs[name] = path
    monkeypatch.setattr(lab, "environment_snapshot", hardware)
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "a" * 64, "files": {}})
    # These tests verify the wrapper, not the already-tested partition hash
    # reader. Metadata are fake; no real or reserved data files exist here.
    monkeypatch.setattr(lab, "verify_development_inputs", lambda p: lab.read_json(Path(p) / "manifest.json"))
    monkeypatch.setattr(lab, "verify_research", fake_verify)
    calls = []

    def runner(data, config, output, **kwargs):
        calls.append((Path(data).name, copy.deepcopy(config), Path(output), kwargs))
        if kwargs["dry_run"]:
            return {"status": "plan_only", "reserved_partitions_read": []}
        if Path(output).exists():
            assert kwargs["resume"]
        else:
            seal(Path(output))
        return fake_verify(output)

    monkeypatch.setattr(lab, "run_research", runner)
    return inputs, calls


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX4090D", "NVIDIA GeForce RTX 4090 D", "RTX-4090_D"])
def test_gpu_name_normalization_accepts_real_default_4090d(name):
    assert lab.require_lab_gpu(hardware(name))["index"] == 0


@pytest.mark.parametrize("snapshot", [hardware("NVIDIA GeForce RTX 4050 Laptop GPU"),
                                      hardware("NVIDIA GeForce RTX 4090"),
                                      hardware(available=False), hardware(index=1), {}])
def test_4050_cpu_or_wrong_default_gpu_are_rejected(snapshot):
    with pytest.raises(ValueError):
        lab.require_lab_gpu(snapshot)


def test_any_matching_secondary_gpu_cannot_bypass_default_gpu_gate():
    snapshot = hardware("NVIDIA GeForce RTX 4050 Laptop GPU")
    snapshot["torch"]["gpus"].append({"index": 1, "name": "NVIDIA GeForce RTX 4090 D",
                                       "total_memory_bytes": 24 * 1024 ** 3})
    with pytest.raises(ValueError, match="Default CUDA GPU"):
        lab.require_lab_gpu(snapshot)


def test_full_profile_has_65_students_20_teacher_fits_and_no_row_caps():
    config = lab.load_profile("full")
    assert len(config["models"]) * len(config["seeds"]) == 65
    assert (config["folds"] + 1) * len(config["seeds"]) == 20
    assert config["max_train_rows"] is None and config["max_validation_rows"] is None


def test_4050_rejection_precedes_data_access_and_writes(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "environment_snapshot", lambda: hardware("NVIDIA RTX 4050 Laptop GPU"))
    monkeypatch.setattr(lab, "verify_development_inputs", lambda *_: pytest.fail("data read before GPU gate"))
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="Default CUDA GPU"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"])
    assert not work.exists() and not calls


def test_dry_run_saves_complete_hardware_plan_without_training(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, profile="full", unsw_data_dir=inputs["unsw"], dry_run=True)
    assert result["status"] == "lab_plan_only"
    assert result["expected_per_environment"] == {"students": 65, "teacher_fits": 20, "total_fits": 85}
    assert calls[0][3] == {"device": "cuda", "resume": False, "dry_run": True}
    saved = lab.read_json(work / "LAB_PREFLIGHT.json")
    assert saved["environment"]["ram_total_bytes"] == 64 * 1024 ** 3
    assert saved["reserved_partitions_read"] == []
    assert not (work / "research-unsw").exists()


def test_two_full_environments_are_serially_run_and_safely_packaged(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, profile="full", unsw_data_dir=inputs["unsw"],
                         cicids_data_dir=inputs["cicids"])
    assert [call[0] for call in calls] == ["data-unsw", "data-cicids"]
    assert all(call[3]["device"] == "cuda" for call in calls)
    assert result["status"] == "lab_development_complete"
    receipt = work / "packages/LAB_PREFLIGHT.json"
    receipt_bytes = receipt.read_bytes()
    assert receipt_bytes == (work / "LAB_PREFLIGHT.json").read_bytes()
    assert receipt.with_name(receipt.name + ".sha256").read_text().strip() == lab.sha256_file(receipt)
    for name in inputs:
        package = Path(result["environments"][name]["package"]["zip"])
        assert package.with_name(package.name + ".sha256").read_text().strip() == lab.sha256_file(package)
        with zipfile.ZipFile(package) as archive:
            assert "PACKAGE_MANIFEST.sha256" in archive.namelist()
            assert f"research-{name}/RESEARCH_COMPLETE.json" in archive.namelist()
    resumed = lab.run_lab(work_dir=work, profile="full", unsw_data_dir=inputs["unsw"],
                          cicids_data_dir=inputs["cicids"], resume=True)
    assert all(item["package"]["status"] == "verified_existing" for item in resumed["environments"].values())
    assert receipt.read_bytes() == receipt_bytes
    assert resumed["preflight_receipt"] == result["preflight_receipt"]


def test_preflight_receipt_resume_refuses_tampering_and_preserves_saved_timestamp(ready, tmp_path):
    plan = tmp_path / "LAB_PREFLIGHT.json"
    lab.write_json(plan, {"created_at": "saved_original_time", "binding": {"hardware": "fake_4090D"}})
    packages = tmp_path / "packages"
    first = lab._copy_saved_preflight(plan, packages, resume=False)
    assert (packages / "LAB_PREFLIGHT.json").read_bytes() == plan.read_bytes()
    assert lab._copy_saved_preflight(plan, packages, resume=True) == first
    target = packages / "LAB_PREFLIGHT.json"
    target.write_text("tampered", encoding="utf8")
    with pytest.raises(ValueError, match="does not match"):
        lab._copy_saved_preflight(plan, packages, resume=True)
    assert target.read_text() == "tampered"


def test_cicids_pilot_missing_inputs_and_wrong_schema_fail_without_runner(ready, tmp_path):
    inputs, calls = ready
    with pytest.raises(ValueError, match="at least one"):
        lab.run_lab(work_dir=tmp_path / "none")
    with pytest.raises(ValueError, match="CICIDS pilot is forbidden"):
        lab.run_lab(work_dir=tmp_path / "pilot", cicids_data_dir=inputs["cicids"])
    bad = manifest("unsw")
    bad["features"] = bad["features"][:-1]
    lab.write_json(inputs["unsw"] / "manifest.json", bad)
    with pytest.raises(ValueError, match="exact feature schema"):
        lab.run_lab(work_dir=tmp_path / "bad", unsw_data_dir=inputs["unsw"])
    assert not calls


def test_synthetic_manifest_is_rejected_in_real_lab_wrapper(ready, tmp_path):
    inputs, calls = ready
    value = manifest("unsw")
    value["data_kind"] = "synthetic"
    lab.write_json(inputs["unsw"] / "manifest.json", value)
    with pytest.raises(ValueError, match="synthetic is forbidden"):
        lab.run_lab(work_dir=tmp_path / "work", unsw_data_dir=inputs["unsw"])
    assert not calls


def test_unknown_work_or_changed_source_cannot_be_overwritten(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    work = tmp_path / "unknown"
    work.mkdir()
    (work / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="must be empty"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"])
    assert (work / "keep.txt").read_text() == "keep"
    work = tmp_path / "planned"
    lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "b" * 64})
    with pytest.raises(ValueError, match="source, configuration, hardware, or data changed"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], resume=True)


def test_package_refuses_unknown_partial_or_tampered_archives(ready, tmp_path):
    source = tmp_path / "research-unsw"
    seal(source)
    target = tmp_path / "result.zip"
    target.write_bytes(b"unknown package")
    with pytest.raises(FileExistsError):
        lab.package_completed_environment(source, target, resume=True)
    assert target.read_bytes() == b"unknown package"
    target = tmp_path / "new.zip"
    lab.package_completed_environment(source, target)
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sidecar"):
        lab.package_completed_environment(source, target, resume=True)


def test_package_rejects_internal_destination_and_linked_members(ready, tmp_path):
    source = tmp_path / "research-unsw"
    seal(source)
    with pytest.raises(ValueError, match="outside"):
        lab.package_completed_environment(source, source / "bad.zip")
    outside = tmp_path / "outside.txt"
    outside.write_text("not a package member")
    link = source / "linked.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic-link creation is unavailable in this test environment")
    with pytest.raises(ValueError, match="links"):
        lab.package_completed_environment(source, tmp_path / "linked.zip")
    assert outside.read_text() == "not a package member"


def test_zip_readback_rejects_traversal_even_with_matching_hash(tmp_path):
    target = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        lab._verify_zip(target, {"../escape": hashlib.sha256(b"bad").hexdigest()})


def test_direct_script_help_works_outside_repo_without_pythonpath(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_research_lab.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--unsw-data-dir" in result.stdout and "--cicids-data-dir" in result.stdout
