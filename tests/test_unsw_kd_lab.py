from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unsw_kd_lab_test_module", REPO / "scripts/run_unsw_kd_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def ready(tmp_path, monkeypatch):
    root = tmp_path / "lab"
    screen = root / "experiments/screen/budget-transfer-unsw"
    data = root / "data/prepared-v1"
    marker = screen / lab.SCREEN_COMPLETE
    put(marker, {"status": "screen_complete"})
    manifest = {"schema_version": "unsw-nb15-split-v1", "dataset_id": "frozen-unsw",
                "partitions": {"train": {"file": "train.parquet"},
                               "test": {"file": "MUST_NOT_READ.parquet"}}}
    put(screen / "plan.json", {"binding": {"data": manifest}})
    put(data / "manifest.json", manifest)
    protocol = tmp_path / "protocol.json"
    put(protocol, {"data": {"screen_complete_sha256": lab.sha256_file(marker)}})
    monkeypatch.setattr(lab, "PROTOCOL", protocol)
    source = {"dirty": False, "commit": "a" * 40, "source_sha256": "b" * 64, "files": {}}
    monkeypatch.setattr(lab, "code_fingerprint", lambda: source)
    monkeypatch.setattr(lab, "check_environment", lambda: {"test_hardware": True})
    monkeypatch.setattr(lab, "require_lab_gpu", lambda env: {"name": "RTX4090D"})
    calls = []
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: calls.append("activate_cuda"))
    def preflight(*args, **kwargs):
        calls.append(("preflight", args, kwargs))
        return {"status": "preflight_ok", "student_fits": 0}
    def run(screen_arg, data_arg, output, **kwargs):
        calls.append(("run", screen_arg, data_arg, kwargs))
        output.mkdir()
        put(output / "UNSW_KD_COMPLETE.json", {"student_fits": 4, "acceptance": False})
        return {"status": "unsw_kd_complete", "student_fits": 4, "acceptance": False}
    monkeypatch.setattr(lab, "preflight_unsw_kd", preflight)
    monkeypatch.setattr(lab, "run_unsw_kd", run)
    return root, screen, data, calls


def assert_package(record):
    target = Path(record["zip"])
    assert target.with_name(target.name + ".sha256").read_text().strip() == lab.sha256_file(target)
    with zipfile.ZipFile(target) as archive:
        manifest = archive.read(lab.PACKAGE_MANIFEST).decode()
        expected = dict((name, digest) for digest, name in
                        (row.split("  ", 1) for row in manifest.splitlines()))
        assert set(archive.namelist()) == set(expected) | {lab.PACKAGE_MANIFEST}
        for name, digest in expected.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        assert not any("/packages/" in n for n in archive.namelist())
        return set(archive.namelist())


def test_discovery_uses_exact_screen_and_manifest_only(ready):
    root, screen, data, _ = ready
    other = root / "experiments/wrong-screen"
    put(other / lab.SCREEN_COMPLETE, {"status": "different"})
    put(root / "data/cicids/manifest.json", {"dataset_id": "cicids"})
    assert lab.resolve_inputs(root) == (screen.resolve(), data.resolve())


def test_ambiguous_discovery_requires_explicit_paths(ready):
    root, screen, data, _ = ready
    duplicate = root / "experiments/duplicate"
    duplicate.mkdir()
    (duplicate / lab.SCREEN_COMPLETE).write_bytes((screen / lab.SCREEN_COMPLETE).read_bytes())
    with pytest.raises(ValueError, match="exactly one.*screen"):
        lab.resolve_inputs(root)
    assert lab.resolve_inputs(root, screen, data) == (screen.resolve(), data.resolve())
    put(root / "data/copied/manifest.json", lab.read_json(data / "manifest.json"))
    with pytest.raises(ValueError, match="exactly one.*prepared"):
        lab.resolve_inputs(root, screen)


def test_wrong_explicit_screen_is_not_accepted(ready):
    root, screen, data, _ = ready
    put(screen / lab.SCREEN_COMPLETE, {"modified": True})
    with pytest.raises(ValueError, match="completion SHA"):
        lab.resolve_inputs(root, screen, data)


def test_preflight_only_never_dispatches_training_or_activates_cuda(ready, tmp_path, monkeypatch):
    root, screen, data, calls = ready
    monkeypatch.setattr(lab, "run_unsw_kd", lambda *a, **k: pytest.fail("preflight trained"))
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "check", preflight_only=True)
    assert result["status"] == "preflight_complete"
    assert len(calls) == 1 and calls[0][0] == "preflight"
    assert not (tmp_path / "check/experiment").exists()
    assert_package(result["package"])


def test_four_arm_negative_result_is_completed_and_packaged(ready, tmp_path):
    root, screen, data, calls = ready
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "formal")
    assert result["status"] == "experiment_complete"
    assert result["result"]["acceptance"] is False
    assert calls[0] == "activate_cuda" and calls[1][0] == "run"
    assert calls[1][-1]["device"] == "cuda" and calls[1][-1]["measure_cpu"] is True
    names = assert_package(result["package"])
    assert any("UNSW_KD_COMPLETE.json" in n for n in names)
    before = {str(p): p.read_bytes() for p in (tmp_path / "formal").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="no overwrite"):
        lab.run_lab(lab_root=root, work_dir=tmp_path / "formal")
    assert {str(p): p.read_bytes() for p in (tmp_path / "formal").rglob("*") if p.is_file()} == before


def test_failure_keeps_and_packages_partial_model(ready, tmp_path, monkeypatch):
    root, screen, data, calls = ready
    def fail(s, d, output, **kwargs):
        output.mkdir()
        (output / "partial.joblib").write_bytes(b"retain partial")
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(lab, "run_unsw_kd", fail)
    work = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated failure"):
        lab.run_lab(lab_root=root, work_dir=work)
    assert (work / "experiment/partial.joblib").read_bytes() == b"retain partial"
    assert (work / "LAB_FAILURE.json").is_file()
    zips = list((work / "packages").glob("*.zip"))
    assert len(zips) == 1
    names = assert_package({"zip": str(zips[0])})
    assert any("partial.joblib" in n for n in names)


@pytest.mark.parametrize("failure", ["dirty", "gpu", "incomplete", "changed_source"])
def test_invalid_execution_retains_failure_record(ready, tmp_path, monkeypatch, failure):
    root, screen, data, calls = ready
    if failure == "dirty":
        monkeypatch.setattr(lab, "code_fingerprint", lambda: {"dirty": True, "commit": "a"*40})
    elif failure == "gpu":
        def bad_gpu(_):
            raise ValueError("wrong GPU")
        monkeypatch.setattr(lab, "require_lab_gpu", bad_gpu)
    elif failure == "incomplete":
        monkeypatch.setattr(lab, "run_unsw_kd", lambda *a, **k: {"status": "partial", "student_fits": 3})
    else:
        original = lab.run_unsw_kd
        def changed(*a, **k):
            value = original(*a, **k)
            monkeypatch.setattr(lab, "code_fingerprint", lambda: {"dirty": True})
            return value
        monkeypatch.setattr(lab, "run_unsw_kd", changed)
    work = tmp_path / failure
    with pytest.raises(ValueError):
        lab.run_lab(lab_root=root, work_dir=work)
    assert (work / "LAB_FAILURE.json").is_file()
    assert len(list((work / "packages").glob("*.zip"))) == 1
    if failure in {"dirty", "gpu"}:
        assert not calls


def test_work_must_not_overlap_frozen_inputs(ready):
    root, screen, data, calls = ready
    with pytest.raises(ValueError, match="non-nested"):
        lab.run_lab(lab_root=root, work_dir=screen / "new-run")
    assert not (screen / "new-run").exists() and calls == []


def test_package_only_preserves_previous_package_without_running(ready, tmp_path, monkeypatch):
    root, screen, data, calls = ready
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "check", preflight_only=True)
    first_zip = Path(result["package"]["zip"])
    first_bytes = first_zip.read_bytes()
    monkeypatch.setattr(lab, "run_unsw_kd", lambda *a, **k: pytest.fail("package trained"))
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("package probed hardware"))
    second = lab.package_work(result["work_dir"])
    assert Path(second["zip"]) != first_zip and first_zip.read_bytes() == first_bytes
    assert_package(second)


def test_optional_frozen_targets_passed_to_core(ready, tmp_path):
    root, screen, data, calls = ready
    path = tmp_path / "frozen.npz"
    path.write_bytes(b"core validates this")
    lab.run_lab(lab_root=root, work_dir=tmp_path / "check", preflight_only=True, frozen_targets=path)
    assert calls[0][2]["frozen_targets_path"] == path.resolve()


def test_help_available_outside_repo_without_trainer(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_unsw_kd_lab.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0
    assert "--package-only" in result.stdout and "--preflight-only" in result.stdout
    assert "--profile" not in result.stdout and "--resume" not in result.stdout
