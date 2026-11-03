"""V3 lab orchestration tests use fake manifests/GPU/runner, never real training."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("risk_lab_script", REPO / "scripts/run_risk_lab.py")
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf8")


def hardware(name="NVIDIA GeForce RTX 4090 D", *, available=True, index=0):
    return {"recorded_at": "mock-clock", "cpu_logical": 16, "ram_total_bytes": 64 * 1024 ** 3,
            "packages": {"torch": "2.6.0+cu126"},
            "torch": {"cuda_available": available, "cuda_runtime": "12.6",
                      "gpus": [{"index": index, "name": name, "total_memory_bytes": 24 * 1024 ** 3}]}}


def config(profile):
    full = profile == "full"
    return {"schema_version": 1, "purpose": "risk_v3_development_only",
            "cohort_policy": "complete_development" if full else "bounded_pilot",
            "seeds": [17, 29, 43] if full else [17], "folds": 3, "threads": 2,
            "max_train_rows": None if full else 12000, "max_validation_rows": None if full else 6000,
            "conditions": ["hard", "boundary_kd", "risk_hard", "risk_matched", "risk_kd",
                           "normal_hinge_kd", "pct", "dual_source_fd", "ekd"]}


def manifest(name):
    if name == "unsw":
        data = {"schema_version": lab.SCHEMA_VERSION, "features": list(lab.FEATURES), "data_kind": "mirror"}
    else:
        data = {"schema_version": lab.CIC_SCHEMA_VERSION, "features": list(lab.CIC_FEATURES),
                "data_kind": "official_repaired",
                "feature_schema": {"name": lab.CIC_SCHEMA_VERSION, "features": list(lab.CIC_FEATURES),
                    "categorical": list(lab.CIC_CATEGORICAL), "discrete": list(lab.CIC_DISCRETE)}}
    data["partitions"] = {split: {"file": split + ".parquet", "n_rows": 100,
                                   "sha256": "b" * 64} for split in ("train", "validation")}
    # These intentionally missing reserved payloads must never be opened by
    # preflight. A manifest can describe them without granting data access.
    data["partitions"].update({split: {"file": split + ".parquet", "n_rows": 20,
                                       "sha256": "c" * 64} for split in ("calibration", "test")})
    return data


def save_manifest(path, value):
    put_json(path / "manifest.json", value)
    (path / "manifest.sha256").write_text(lab.sha256_file(path / "manifest.json") + "\n", encoding="ascii")


def seal(path, cfg, data_dir):
    path.mkdir(parents=True)
    put_json(path / "plan.json", {"config": cfg, "data_dir": str(Path(data_dir).resolve())})
    (path / "result.txt").write_text("mock engineering result; no model fitted", encoding="utf8")
    put_json(path / lab.COMPLETE, {"status": "development_complete", "reserved_partitions_read": [],
             "n_students": len(cfg["seeds"]) * len(cfg["conditions"]),
             "n_teacher_fits": len(cfg["seeds"]) * (cfg["folds"] + 1),
             "files": {name: lab.sha256_file(path / name) for name in ("result.txt", "plan.json")}})


def fake_verify(path):
    path = Path(path)
    done = lab.read_json(path / lab.COMPLETE)
    for name, digest in done["files"].items():
        if lab.sha256_file(path / name) != digest:
            raise ValueError("mock completed artifact changed")
    return done


@pytest.fixture
def ready(tmp_path, monkeypatch):
    repo = tmp_path / "fake-repo"
    (repo / "configs").mkdir(parents=True)
    for profile, filename in (("pilot", "risk-v3-pilot.yaml"), ("full", "risk-v3.yaml")):
        (repo / "configs" / filename).write_text(yaml.safe_dump(config(profile)), encoding="utf8")
    monkeypatch.setattr(lab, "REPO", repo)
    monkeypatch.setattr(lab, "validate_risk_config", lambda cfg: None)
    monkeypatch.setattr(lab, "check_environment", hardware)
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: None)
    monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "a" * 64, "files": {}})
    monkeypatch.setattr(lab, "verify_risk_research", fake_verify)
    inputs, calls = {}, []
    for name in ("unsw", "cicids"):
        path = tmp_path / ("data-" + name)
        save_manifest(path, manifest(name))
        inputs[name] = path

    def runner(data, cfg, output, **kwargs):
        calls.append((Path(data), copy.deepcopy(cfg), Path(output), kwargs))
        if kwargs["resume"]:
            plan = lab.read_json(Path(output) / "plan.json")
            if plan != {"config": cfg, "data_dir": str(Path(data).resolve())}:
                raise ValueError("mock runner plan identity mismatch")
        else:
            seal(Path(output), cfg, data)
        return fake_verify(output)

    monkeypatch.setattr(lab, "run_risk_research", runner)
    return inputs, calls


@pytest.mark.parametrize("name", ["RTX4090D", "NVIDIA GeForce RTX 4090 D", "RTX-4090_D"])
def test_accepts_only_normalized_default_4090d(name):
    assert lab.require_lab_gpu(hardware(name))["index"] == 0


@pytest.mark.parametrize("snapshot", [hardware("RTX 4050 Laptop GPU"), hardware("RTX 4090"),
                                      hardware(available=False), hardware(index=1), {}])
def test_rejects_cpu_4050_non_d_or_secondary_only(snapshot):
    with pytest.raises(ValueError):
        lab.require_lab_gpu(snapshot)


def test_matching_secondary_gpu_cannot_bypass_cuda_zero():
    snapshot = hardware("RTX4050")
    snapshot["torch"]["gpus"].append(hardware()["torch"]["gpus"][0] | {"index": 1})
    with pytest.raises(ValueError, match="CUDA0"):
        lab.require_lab_gpu(snapshot)


@pytest.mark.parametrize("available,name", [(False, "RTX4090D"), (True, "RTX4050 Laptop GPU")])
def test_actual_torch_cuda_is_rechecked_before_activation(monkeypatch, available, name):
    cuda = SimpleNamespace(is_available=lambda: available, get_device_name=lambda index: name,
                           set_device=lambda index: pytest.fail("wrong CUDA activated"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    with pytest.raises(ValueError):
        lab._activate_cuda_zero()


def test_actual_same_process_device_is_explicitly_pinned_to_zero(monkeypatch):
    selected = []
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: "RTX4090D",
                           set_device=selected.append, current_device=lambda: selected[-1])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    lab._activate_cuda_zero()
    assert selected == [0]


def test_dependency_mismatch_fails_without_environment_probe_or_installer(ready, monkeypatch):
    (lab.REPO / "pyproject.toml").write_text('[project]\ndependencies = ["numpy==1.26.4"]\n', encoding="utf8")
    monkeypatch.setattr(lab.importlib.metadata, "version", lambda name: "0.0.0")
    # Recover the real function, which the fixture replaces for fake GPU tests.
    original = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(original)
    monkeypatch.setattr(original, "REPO", lab.REPO)
    monkeypatch.setattr(original, "environment_snapshot", lambda: pytest.fail("probe after bad dependency"))
    with pytest.raises(ValueError, match="dependency mismatch"):
        original.check_environment()


def test_rejection_precedes_data_access_and_output(ready, tmp_path, monkeypatch):
    inputs, calls = ready
    monkeypatch.setattr(lab, "check_environment", lambda: hardware("RTX4050 Laptop GPU"))
    monkeypatch.setattr(lab, "_dataset_record", lambda *_: pytest.fail("data accessed before GPU rejection"))
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: pytest.fail("wrong GPU activated"))
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="CUDA0"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    assert not calls and not work.exists()


def test_dry_run_manifest_only_no_payloads_roles_or_runner(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    result = lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    assert result["status"] == "risk_v3_lab_plan_only" and not calls
    assert result["expected_per_environment"] == {"students": 9, "teacher_fits": 4,
                                                 "anchor_fits": 1, "total_fits": 14}
    saved = lab.read_json(work / lab.PREFLIGHT)
    assert saved["environment"]["ram_total_bytes"] == 64 * 1024 ** 3
    assert saved["binding"]["interpreter"]["executable"] == str(Path(sys.executable).absolute())
    assert saved["binding"]["device"] == "cuda" and saved["binding"]["cuda_index"] == 0
    assert saved["binding"]["config_file_sha256"] == lab.sha256_file(lab._profile_path("pilot"))
    assert saved["binding"]["datasets"]["unsw"]["data_kind"] == "mirror"
    assert saved["reserved_partitions_read"] == []
    assert list(work.iterdir()) == [work / lab.PREFLIGHT]


def test_dry_run_to_real_plan_resume_retains_original_bytes(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    saved = (work / lab.PREFLIGHT).read_bytes()
    result = lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], resume=True)
    assert len(calls) == 1 and calls[0][3] == {"device": "cuda", "threads": 2, "resume": False}
    assert result["status"] == "risk_v3_lab_development_complete"
    assert (work / "packages" / lab.PREFLIGHT).read_bytes() == saved


def test_full_serial_environments_counts_zip_and_complete_only_resume(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    args = {"work_dir": work, "profile": "full", "unsw_data_dir": inputs["unsw"],
            "cicids_data_dir": inputs["cicids"]}
    first = lab.run_lab(**args)
    assert first["expected_per_environment"] == {"students": 27, "teacher_fits": 12,
                                                 "anchor_fits": 3, "total_fits": 42}
    assert [call[0].name for call in calls] == ["data-unsw", "data-cicids"]
    assert all(call[3] == {"device": "cuda", "threads": 2, "resume": False} for call in calls)
    saved = (work / lab.PREFLIGHT).read_bytes()
    for name, record in first["environments"].items():
        package = Path(record["package"]["zip"])
        assert package.with_name(package.name + ".sha256").read_text().strip() == lab.sha256_file(package)
        with zipfile.ZipFile(package) as archive:
            assert lab.PACKAGE_MANIFEST in archive.namelist()
            assert f"risk-v3-{name}/{lab.COMPLETE}" in archive.namelist()
    second = lab.run_lab(**args, resume=True)
    assert len(calls) == 4 and all(call[3]["resume"] for call in calls[2:])
    assert all(item["package"]["status"] == "verified_existing" for item in second["environments"].values())
    assert (work / "packages" / lab.PREFLIGHT).read_bytes() == saved


@pytest.mark.parametrize("change", ["source", "manifest", "config", "interpreter", "gpu"])
def test_resume_rejects_changed_identity(ready, tmp_path, monkeypatch, change):
    inputs, calls = ready
    work = tmp_path / "work"
    lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    if change == "source":
        monkeypatch.setattr(lab, "code_fingerprint", lambda: {"source_sha256": "d" * 64})
    elif change == "manifest":
        value = manifest("unsw")
        value["partitions"]["train"]["sha256"] = "d" * 64
        save_manifest(inputs["unsw"], value)
    elif change == "config":
        value = config("pilot") | {"threads": 3}
        lab._profile_path("pilot").write_text(yaml.safe_dump(value), encoding="utf8")
    elif change == "interpreter":
        monkeypatch.setattr(lab.sys, "prefix", "different-environment")
    else:
        snapshot = hardware()
        snapshot["torch"]["gpus"][0]["total_memory_bytes"] -= 1
        monkeypatch.setattr(lab, "check_environment", lambda: snapshot)
    with pytest.raises(ValueError, match="binding changed"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], resume=True)
    assert not calls


def test_incomplete_output_preserved_and_not_passed_to_runner(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    output = work / "risk-v3-unsw"
    output.mkdir()
    sentinel = output / "partial.txt"
    sentinel.write_text("keep interrupted state", encoding="utf8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], resume=True)
    assert sentinel.read_text() == "keep interrupted state" and not calls


def test_completed_resume_runner_rechecks_plan_identity(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    # Correct fitting counts but a different frozen plan must not be reused.
    seal(work / "risk-v3-unsw", config("pilot") | {"threads": 99}, inputs["unsw"])
    with pytest.raises(ValueError, match="plan identity"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], resume=True)
    assert len(calls) == 1 and calls[0][3]["resume"]
    assert not (work / "packages").exists()


@pytest.mark.parametrize("field,value", [("n_students", 8), ("n_teacher_fits", 3),
                                       ("reserved_partitions_read", ["calibration"]),
                                       ("status", "started"), ("files", {})])
def test_bad_completion_metadata_cannot_be_accepted_or_packaged(ready, tmp_path, monkeypatch, field, value):
    inputs, _ = ready
    runner = lab.run_risk_research

    def bad_runner(data, cfg, output, **kwargs):
        runner(data, cfg, output, **kwargs)
        complete = lab.read_json(Path(output) / lab.COMPLETE)
        complete[field] = value
        put_json(Path(output) / lab.COMPLETE, complete)

    monkeypatch.setattr(lab, "run_risk_research", bad_runner)
    with pytest.raises(ValueError):
        lab.run_lab(work_dir=tmp_path / "work", unsw_data_dir=inputs["unsw"])
    assert not (tmp_path / "work" / "packages").exists()


@pytest.mark.parametrize("bad", ["synthetic", "schema", "hash", "partition"])
def test_bad_dataset_manifest_refused_without_runner(ready, tmp_path, bad):
    inputs, calls = ready
    value = manifest("unsw")
    if bad == "synthetic":
        value["data_kind"] = "synthetic"
    elif bad == "schema":
        value["features"] = value["features"][:-1]
    elif bad == "partition":
        value["partitions"]["train"]["file"] = "test.parquet"
    save_manifest(inputs["unsw"], value)
    if bad == "hash":
        (inputs["unsw"] / "manifest.sha256").write_text("a" * 64, encoding="ascii")
    with pytest.raises(ValueError):
        lab.run_lab(work_dir=tmp_path / "work", unsw_data_dir=inputs["unsw"], dry_run=True)
    assert not calls and not (tmp_path / "work").exists()


def test_profile_missing_input_pilot_cic_and_full_caps_rejected(ready, tmp_path):
    inputs, calls = ready
    with pytest.raises(ValueError, match="at least one"):
        lab.run_lab(work_dir=tmp_path / "none")
    with pytest.raises(ValueError, match="UNSW only"):
        lab.run_lab(work_dir=tmp_path / "cic-pilot", cicids_data_dir=inputs["cicids"])
    full = config("full") | {"max_train_rows": 100}
    lab._profile_path("full").write_text(yaml.safe_dump(full), encoding="utf8")
    with pytest.raises(ValueError, match="every development"):
        lab.load_profile("full")
    assert not calls


def test_unknown_nonempty_work_and_nested_data_paths_preserved(ready, tmp_path):
    inputs, calls = ready
    work = tmp_path / "work"
    work.mkdir()
    (work / "unknown.txt").write_text("keep", encoding="utf8")
    with pytest.raises(FileExistsError, match="empty"):
        lab.run_lab(work_dir=work, unsw_data_dir=inputs["unsw"], dry_run=True)
    with pytest.raises(ValueError, match="non-nested"):
        lab.run_lab(work_dir=inputs["unsw"] / "work", unsw_data_dir=inputs["unsw"], dry_run=True)
    assert (work / "unknown.txt").read_text() == "keep" and not calls


def test_receipt_exact_saved_bytes_and_tamper_rejected(tmp_path):
    source = tmp_path / lab.PREFLIGHT
    put_json(source, {"created_at": "original saved time"})
    packages = tmp_path / "packages"
    result = lab._copy_saved_preflight(source, packages, resume=False)
    target = packages / lab.PREFLIGHT
    assert target.read_bytes() == source.read_bytes()
    assert result == lab._copy_saved_preflight(source, packages, resume=True)
    target.write_text("modified", encoding="utf8")
    with pytest.raises(ValueError, match="original bytes"):
        lab._copy_saved_preflight(source, packages, resume=True)
    assert target.read_text() == "modified"


def test_package_refuses_partial_unknown_and_inside_environment(ready, tmp_path):
    inputs, _ = ready
    output = tmp_path / "sealed"
    seal(output, config("pilot"), inputs["unsw"])
    with pytest.raises(ValueError, match="outside"):
        lab.package_completed_environment(output, output / "bad.zip")
    package = tmp_path / "existing.zip"
    package.write_bytes(b"unknown")
    with pytest.raises(FileExistsError, match="preserved"):
        lab.package_completed_environment(output, package, resume=True)
    assert package.read_bytes() == b"unknown"


def test_package_rejects_symbolic_link_without_reading_external_member(ready, tmp_path):
    inputs, _ = ready
    output = tmp_path / "sealed"
    seal(output, config("pilot"), inputs["unsw"])
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf8")
    try:
        (output / "linked.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not allow symbolic links in this test account")
    with pytest.raises(ValueError, match="links"):
        lab.package_completed_environment(output, tmp_path / "bad-link.zip")
    assert outside.read_text() == "outside" and not (tmp_path / "bad-link.zip").exists()


def test_zip_member_hash_verification_catches_payload_tampering(tmp_path):
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("safe/result.txt", b"changed")
    expected = {"safe/result.txt": hashlib.sha256(b"original").hexdigest()}
    with pytest.raises(ValueError, match="payload hash"):
        lab._verify_zip(package, expected)


def test_zip_path_traversal_rejected_even_when_payload_hash_matches(tmp_path):
    package = tmp_path / "escape.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../escape", b"data")
    with pytest.raises(ValueError, match="unsafe"):
        lab._verify_zip(package, {"../escape": hashlib.sha256(b"data").hexdigest()})


def test_package_tamper_and_completion_reserved_access_are_rejected(ready, tmp_path):
    inputs, _ = ready
    output = tmp_path / "sealed"
    seal(output, config("pilot"), inputs["unsw"])
    result = lab.package_completed_environment(output, tmp_path / "result.zip")
    package = Path(result["zip"])
    package.write_bytes(package.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="sidecar"):
        lab.package_completed_environment(output, package, resume=True)
    complete = lab.read_json(output / lab.COMPLETE)
    complete["reserved_partitions_read"] = ["test"]
    put_json(output / lab.COMPLETE, complete)
    with pytest.raises(ValueError, match="reserved"):
        lab._verify_complete(output)


def test_direct_help_outside_repository_uses_self_contained_path(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_risk_lab.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout and "--profile" in result.stdout
    assert "--device" not in result.stdout and "--allow-cpu" not in result.stdout
