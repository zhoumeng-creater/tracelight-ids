from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unsw_kd_minibatch_lab_test_module", REPO / "scripts/run_unsw_kd_minibatch_lab.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def frozen_result(status, **extra):
    preflight = status == "minibatch_preflight_complete"
    return {"status": status, "student_fits": 0 if preflight else 2, "teacher_fits": 0,
            "optimizer_updates": 0 if preflight else 6720, **extra}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    root = tmp_path / "lab"
    screen = root / "experiments/screen/budget-transfer-unsw"
    round10 = root / "experiments/round10/experiment"
    data = root / "data/prepared-v1"
    put(screen / lab.SCREEN_COMPLETE, {"status": "screen_complete"})
    put(round10 / lab.ROUND10_COMPLETE, {"status": "unsw_kd_complete"})
    data.mkdir(parents=True)
    (data / "train.parquet").write_bytes(b"synthetic original train payload")
    manifest = {"schema_version": "unsw-nb15-split-v1", "dataset_id": "frozen-unsw",
                "partitions": {"train": {"file": "train.parquet", "sha256": lab.sha256_file(data / "train.parquet")},
                               "test": {"file": "MUST_NOT_READ.parquet", "sha256": "b" * 64}}}
    put(data / "manifest.json", manifest)
    binding = tmp_path / "binding.json"
    put(binding, {"files": [
        {"root_alias": "screen", "path": lab.SCREEN_COMPLETE, "sha256": lab.sha256_file(screen / lab.SCREEN_COMPLETE)},
        {"root_alias": "round10", "path": lab.ROUND10_COMPLETE, "sha256": lab.sha256_file(round10 / lab.ROUND10_COMPLETE)},
    ], "prepared_contract": {"data_manifest": manifest}})
    monkeypatch.setattr(lab, "BINDING", binding)
    protocol = tmp_path / "protocol.json"
    put(protocol, {"test_frozen_protocol": True})
    monkeypatch.setattr(lab, "PROTOCOL", protocol)
    source = {"dirty": False, "commit": "a" * 40, "source_sha256": "b" * 64, "files": {}}
    monkeypatch.setattr(lab, "code_fingerprint", lambda: source)
    monkeypatch.setattr(lab, "check_environment", lambda: {"test_hardware": True})
    monkeypatch.setattr(lab, "check_preflight_environment", lambda: {"metadata_only": True})
    monkeypatch.setattr(lab, "require_lab_gpu", lambda env: {"name": "RTX4090D"})
    calls = []
    monkeypatch.setattr(lab, "_activate_cuda_zero", lambda: calls.append("activate_cuda"))

    def preflight(*args, **kwargs):
        calls.append(("preflight", args, kwargs))
        return frozen_result("minibatch_preflight_complete")

    def run(screen_arg, round10_arg, data_arg, output, **kwargs):
        calls.append(("run", screen_arg, round10_arg, data_arg, kwargs))
        output.mkdir()
        result = frozen_result("unsw_kd_minibatch_complete", acceptance_passed=False)
        put(output / "UNSW_KD_MINIBATCH_COMPLETE.json", result)
        return result

    monkeypatch.setattr(lab, "preflight_minibatch", preflight)
    monkeypatch.setattr(lab, "run_minibatch", run)
    return root, screen, round10, data, calls


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
        assert not any("/packages/" in name for name in archive.namelist())
        return set(archive.namelist())


def test_discovery_binds_both_rounds_and_original_train_only(ready, monkeypatch):
    root, screen, round10, data, _ = ready
    put(root / "experiments/wrong-screen" / lab.SCREEN_COMPLETE, {"wrong": True})
    put(root / "experiments/wrong-round" / lab.ROUND10_COMPLETE, {"wrong": True})
    put(root / "data/cicids/manifest.json", {"dataset_id": "cicids"})
    opened = []
    original = lab.sha256_file

    def track(path):
        opened.append(Path(path))
        assert "MUST_NOT_READ" not in str(path)
        return original(path)

    monkeypatch.setitem(lab._verify_prepared.__globals__, "sha256_file", track)
    assert lab.resolve_inputs(root) == (screen.resolve(), round10.resolve(), data.resolve())
    assert data / "train.parquet" in opened


@pytest.mark.parametrize("kind", ["screen", "round10", "prepared"])
def test_multiple_exact_copies_require_explicit_paths(ready, kind):
    root, screen, round10, data, _ = ready
    source = {"screen": screen, "round10": round10, "prepared": data}[kind]
    destination = root / ("data" if kind == "prepared" else "experiments") / "duplicate"
    shutil.copytree(source, destination)
    with pytest.raises(ValueError, match="exactly one.*" + kind):
        lab.resolve_inputs(root)
    assert lab.resolve_inputs(root, screen, round10, data) == (screen.resolve(), round10.resolve(), data.resolve())


@pytest.mark.parametrize("which", ["screen", "round10", "manifest", "train"])
def test_explicit_inputs_cannot_bypass_binding(ready, which):
    root, screen, round10, data, calls = ready
    if which == "screen":
        put(screen / lab.SCREEN_COMPLETE, {"modified": True})
    elif which == "round10":
        put(round10 / lab.ROUND10_COMPLETE, {"modified": True})
    elif which == "manifest":
        put(data / "manifest.json", {"modified": True})
    else:
        (data / "train.parquet").write_bytes(b"different physical parquet")
    with pytest.raises(ValueError, match="frozen binding"):
        lab.resolve_inputs(root, screen, round10, data)
    assert calls == []


def test_preflight_never_dispatches_run_or_cuda_environment(ready, tmp_path, monkeypatch):
    root, screen, round10, data, calls = ready
    monkeypatch.setattr(lab, "run_minibatch", lambda *a, **k: pytest.fail("preflight dispatched run"))
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("preflight initialized CUDA probe"))
    monkeypatch.setattr(lab, "require_lab_gpu", lambda *a: pytest.fail("preflight checked CUDA"))
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "check", preflight_only=True)
    assert result["status"] == "minibatch_preflight_complete"
    assert len(calls) == 1 and calls[0][0] == "preflight"
    assert calls[0][1] == (screen.resolve(), round10.resolve(), data.resolve())
    assert calls[0][2] == {}
    assert not (tmp_path / "check/experiment").exists()
    environment = lab.read_json(tmp_path / "check/LAB_ENVIRONMENT.json")
    assert environment["cuda_checked"] is False and environment["selected_gpu"] is None
    request = lab.read_json(tmp_path / "check/LAB_REQUEST.json")
    for key in ("planned_student_fits", "planned_teacher_fits", "planned_optimizer_updates",
                "planned_teacher_queries", "planned_endpoint_rc_rows", "planned_cpu_cost_measurements"):
        assert request[key] == 0
    assert_package(result["package"])


@pytest.mark.parametrize("key", ["student_fits", "teacher_fits", "optimizer_updates"])
def test_preflight_rejects_nonzero_or_boolean_execution_counters(ready, tmp_path, monkeypatch, key):
    root, _, _, _, calls = ready
    result = frozen_result("minibatch_preflight_complete")
    result[key] = True if key == "teacher_fits" else 1
    monkeypatch.setattr(lab, "preflight_minibatch", lambda *a: result)
    work = tmp_path / "failed-check"
    with pytest.raises(ValueError, match=key + "=0"):
        lab.run_lab(lab_root=root, work_dir=work, preflight_only=True)
    assert calls == [] and not (work / "experiment").exists()
    assert (work / "LAB_FAILURE.json").is_file()
    assert len(list((work / "packages").glob("*.zip"))) == 1


def test_invalid_scientific_result_is_complete_and_packaged_without_retry(ready, tmp_path):
    root, screen, round10, data, calls = ready
    work = tmp_path / "formal"
    result = lab.run_lab(lab_root=root, work_dir=work)
    assert result["status"] == "unsw_kd_minibatch_complete"
    assert result["result"]["acceptance_passed"] is False
    assert len(calls) == 2 and calls[0] == "activate_cuda" and calls[1][0] == "run"
    assert calls[1][1:4] == (screen.resolve(), round10.resolve(), data.resolve())
    assert calls[1][-1] == {"device": "cuda"}
    request = lab.read_json(work / "LAB_REQUEST.json")
    assert request["entrypoint"] == "unsw_kd_minibatch"
    assert request["protocol_sha256"] == lab.sha256_file(lab.PROTOCOL)
    assert request["planned_student_fits"] == 2
    assert request["planned_optimizer_updates"] == 6720
    assert request["planned_endpoint_rc_rows"] == 73642
    assert request["planned_teacher_fits"] == request["planned_teacher_queries"] == 0
    assert request["planned_cpu_cost_measurements"] == 0
    assert request["automatic_expansion"] is request["automatic_rerun"] is False
    assert any("UNSW_KD_MINIBATCH_COMPLETE.json" in name for name in assert_package(result["package"]))
    before = {str(p): p.read_bytes() for p in work.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError, match="no overwrite"):
        lab.run_lab(lab_root=root, work_dir=work)
    assert {str(p): p.read_bytes() for p in work.rglob("*") if p.is_file()} == before


def test_python_entry_fixes_cublas_before_cuda_probe(ready, tmp_path, monkeypatch):
    root, _, _, _, _ = ready
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    def check():
        assert lab.os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        return {"test_hardware": True}

    monkeypatch.setattr(lab, "check_environment", check)
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "fixed-cublas")
    assert result["status"] == "unsw_kd_minibatch_complete"


def test_failure_keeps_and_packages_partial_diagnostics(ready, tmp_path, monkeypatch):
    root, screen, round10, data, calls = ready

    def fail(s, r, d, output, **kwargs):
        output.mkdir()
        (output / "partial.jsonl").write_bytes(b'{"retained":true}\n')
        raise RuntimeError("simulated mini-batch failure")

    monkeypatch.setattr(lab, "run_minibatch", fail)
    work = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated mini-batch failure"):
        lab.run_lab(lab_root=root, work_dir=work)
    assert (work / "experiment/partial.jsonl").read_bytes() == b'{"retained":true}\n'
    assert (work / "LAB_FAILURE.json").is_file()
    zips = list((work / "packages").glob("*.zip"))
    assert len(zips) == 1
    assert any("partial.jsonl" in name for name in assert_package({"zip": str(zips[0])}))


@pytest.mark.parametrize("failure", ["dirty", "gpu", "incomplete", "student_fits", "teacher_fits", "optimizer_updates", "missing_counter", "changed_source"])
def test_invalid_execution_retains_failure_and_does_not_retry(ready, tmp_path, monkeypatch, failure):
    root, screen, round10, data, calls = ready
    if failure == "dirty":
        monkeypatch.setattr(lab, "code_fingerprint", lambda: {"dirty": True, "commit": "a" * 40})
    elif failure == "gpu":
        def bad_gpu(_):
            raise ValueError("wrong GPU")
        monkeypatch.setattr(lab, "require_lab_gpu", bad_gpu)
    elif failure == "changed_source":
        original = lab.run_minibatch

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            monkeypatch.setattr(lab, "code_fingerprint", lambda: {"dirty": True})
            return result

        monkeypatch.setattr(lab, "run_minibatch", changed)
    else:
        result = frozen_result("partial" if failure == "incomplete" else "unsw_kd_minibatch_complete")
        if failure == "missing_counter":
            del result["optimizer_updates"]
        elif failure != "incomplete":
            result[failure] = 1
        monkeypatch.setattr(lab, "run_minibatch", lambda *a, **k: result)
    work = tmp_path / failure
    with pytest.raises(ValueError):
        lab.run_lab(lab_root=root, work_dir=work)
    assert (work / "LAB_FAILURE.json").is_file()
    assert len(list((work / "packages").glob("*.zip"))) == 1
    if failure in {"dirty", "gpu"}:
        assert not calls


@pytest.mark.parametrize("which", ["screen", "round10", "data"])
def test_work_cannot_be_nested_in_frozen_inputs(ready, which):
    root, screen, round10, data, calls = ready
    work = {"screen": screen, "round10": round10, "data": data}[which] / "new-run"
    with pytest.raises(ValueError, match="non-nested"):
        lab.run_lab(lab_root=root, work_dir=work)
    assert not work.exists() and calls == []


def test_package_only_cli_does_not_start_experiment_or_probe_hardware(ready, tmp_path, monkeypatch):
    root, screen, round10, data, calls = ready
    result = lab.run_lab(lab_root=root, work_dir=tmp_path / "check", preflight_only=True)
    first_zip = Path(result["package"]["zip"])
    first_bytes = first_zip.read_bytes()
    monkeypatch.setattr(lab, "run_lab", lambda **k: pytest.fail("package dispatched experiment"))
    monkeypatch.setattr(lab, "check_environment", lambda: pytest.fail("package probed hardware"))
    assert lab.main(["--package-only", result["work_dir"]]) == 0
    assert first_zip.read_bytes() == first_bytes
    zips = list(first_zip.parent.glob("*.zip"))
    assert len(zips) == 2
    for path in zips:
        assert_package({"zip": str(path)})
    with pytest.raises(SystemExit):
        lab.main(["--package-only", result["work_dir"], "--preflight-only"])


def test_package_only_rejects_other_experiment_directory(tmp_path):
    put(tmp_path / "LAB_REQUEST.json", {"planned_student_continuations": 4})
    with pytest.raises(ValueError, match="UNSW H/K mini-batch"):
        lab.package_work(tmp_path)


def test_help_available_outside_repo_and_has_no_training_controls(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "scripts/run_unsw_kd_minibatch_lab.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0
    assert "--round10-dir" in result.stdout and "--package-only" in result.stdout
    for option in ("--profile", "--resume", "--seed", "--epochs", "--train", "--binding", "--device", "--round11-dir"):
        assert option not in result.stdout
