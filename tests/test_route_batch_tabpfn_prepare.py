"""Small mocked downloads and preparation lifecycle; no real weights or installs."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from ids.route_batch import tabpfn_prepare as prepare


class Response(io.BytesIO):
    def __init__(self, data, *, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


@pytest.fixture
def public_files(tmp_path, monkeypatch):
    checkpoint, license_bytes = b"small synthetic checkpoint, never deserialized", b"synthetic license"
    monkeypatch.setattr(prepare, "CHECKPOINT_BYTES", len(checkpoint))
    monkeypatch.setattr(prepare, "CHECKPOINT_SHA256", hashlib.sha256(checkpoint).hexdigest())
    monkeypatch.setattr(prepare, "LICENSE_BYTES", len(license_bytes))
    monkeypatch.setattr(prepare, "LICENSE_SHA256", hashlib.sha256(license_bytes).hexdigest())
    metadata = {"id": prepare.HF_REPO, "sha": prepare.HF_REVISION, "private": False, "gated": False,
                "siblings": [{"rfilename": prepare.CHECKPOINT_NAME, "size": len(checkpoint),
                              "lfs": {"sha256": hashlib.sha256(checkpoint).hexdigest()}}]}
    calls = []
    def request(url, headers=None):
        calls.append((url, headers))
        if "/api/models/" in url:
            return Response(json.dumps(metadata).encode())
        data = license_bytes if url == prepare.LICENSE_URL else checkpoint
        if headers and "Range" in headers:
            offset = int(headers["Range"].split("=")[1].split("-")[0])
            return Response(data[offset:], status=206, headers={"Content-Range": f"bytes {offset}-{len(data)-1}/{len(data)}"})
        return Response(data)
    monkeypatch.setattr(prepare, "_request", request)
    return SimpleNamespace(checkpoint=checkpoint, license=license_bytes, calls=calls, metadata=metadata,
                           path=tmp_path / "weights" / prepare.CHECKPOINT_NAME, reports=tmp_path / "reports")


def test_public_fixed_revision_download_and_local_reuse(public_files, monkeypatch):
    f = public_files
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["status"] == "ready" and result["download"] == "downloaded"
    assert f.path.read_bytes() == f.checkpoint
    assert f.path.with_name(f.path.name + ".LICENSE").read_bytes() == f.license
    assert result["account_license_acceptance_claimed"] is False
    assert all(prepare.HF_REVISION in url for url, _ in f.calls)
    monkeypatch.setattr(prepare, "_request", lambda *a, **k: pytest.fail("valid local weight must not contact network"))
    assert prepare.prepare_checkpoint(f.path, f.reports)["download"] == "reused"


def test_gated_repository_stops_without_download_or_simulated_acceptance(public_files):
    f = public_files
    f.metadata["gated"] = "manual"
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["status"] == "access_required" and result["failed_stage"] == "checkpoint_access"
    assert len(f.calls) == 1 and not f.path.exists()


def test_http_403_is_reported_as_access_required(monkeypatch):
    def blocked(*args, **kwargs):
        raise urllib.error.HTTPError("https://huggingface.co/blocked", 403, "forbidden", {}, None)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", blocked)
    with pytest.raises(prepare.PrepareError) as error:
        prepare._request("https://huggingface.co/blocked")
    assert error.value.status == "access_required"


def test_transient_http_errors_retry_but_report_failure_as_status_string(public_files, monkeypatch):
    f = public_files
    # HTTPError itself exposes numeric .status; this must not become report status.
    def unavailable(*args, **kwargs):
        raise urllib.error.HTTPError(prepare.CHECKPOINT_URL, 502, "gateway unavailable", {}, None)
    monkeypatch.setattr(prepare, "_request", unavailable)
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["status"] == "failed" and result["error_type"] == "HTTPError"


def test_request_retries_transient_gateway_failure(monkeypatch):
    calls = []
    def open_url(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise urllib.error.HTTPError(prepare.CHECKPOINT_URL, 502, "gateway unavailable", {}, None)
        return Response(b"ok")
    monkeypatch.setattr(prepare.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(prepare.time, "sleep", lambda seconds: None)
    with prepare._request(prepare.CHECKPOINT_URL) as response:
        assert response.read() == b"ok"
    assert len(calls) == 2


def test_partial_download_resumes_at_registered_offset(public_files):
    f = public_files
    f.path.parent.mkdir()
    f.path.with_name(f.path.name + ".download").write_bytes(f.checkpoint[:7])
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["status"] == "ready" and f.path.read_bytes() == f.checkpoint
    assert (prepare.CHECKPOINT_URL, {"Range": "bytes=7-"}) in f.calls


def test_existing_wrong_checkpoint_is_preserved_without_network(public_files):
    f = public_files
    f.path.parent.mkdir()
    f.path.write_bytes(b"user file")
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["status"] == "failed" and result["failed_stage"] == "checkpoint_verify"
    assert f.path.read_bytes() == b"user file" and not f.calls


def test_wrong_remote_digest_stops_before_weight_download(public_files):
    f = public_files
    f.metadata["siblings"][0]["lfs"]["sha256"] = "0" * 64
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["failed_stage"] == "checkpoint_metadata" and len(f.calls) == 1


def test_wrong_download_bytes_never_promote_final_file(public_files, monkeypatch):
    f = public_files
    request = prepare._request
    monkeypatch.setattr(prepare, "_request", lambda url, **kwargs:
                        Response(b"x" * len(f.checkpoint)) if url == prepare.CHECKPOINT_URL else request(url, **kwargs))
    result = prepare.prepare_checkpoint(f.path, f.reports)
    assert result["failed_stage"] == "checkpoint_verify" and not f.path.exists()
    assert f.path.with_name(f.path.name + ".download").exists()


@pytest.fixture
def fake_lab(tmp_path, monkeypatch, public_files):
    core = tmp_path / "core" / "Scripts" / "python.exe"
    core.parent.mkdir(parents=True)
    core.write_bytes(b"existing core, never mutated")
    target = tmp_path / "optional" / "Scripts" / "python.exe"
    installed = {"pip": "24.0"}
    calls = []
    pins = {**prepare._pins(prepare.REPO / "requirements-route-batch-tabpfn.txt"),
            "torch": prepare.TORCH_VERSION, "pip": prepare.PIP_VERSION}
    state = {"fail_stage": None}
    def inspect(python, attempt, env):
        assert Path(python).is_file()
        return {"version": [3, 12], "bits": 64, "prefix": str(Path(python).parent.parent),
                "base_prefix": str(tmp_path / "base-python"), "packages": installed.copy()}
    def run(command, stage, attempt, env, **kwargs):
        calls.append((list(map(str, command)), stage))
        log = attempt / f"mock-{len(calls)}-{stage}.log"
        log.write_text("synthetic subprocess evidence", encoding="utf-8")
        if state["fail_stage"] == stage:
            raise prepare.PrepareError(stage, "synthetic interruption")
        if stage == "venv_create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"mock new interpreter")
            (target.parent.parent / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
        elif stage == "pip_bootstrap":
            installed["pip"] = prepare.PIP_VERSION
        elif stage == "torch_install":
            installed["torch"] = prepare.TORCH_VERSION
        elif stage == "dependency_install":
            installed.update(pins)
        elif stage == "synthetic_tests":
            Path(command[command.index("--junitxml") + 1]).write_text('<testsuites tests="1" failures="0"/>')
        return log
    monkeypatch.setattr(prepare, "_inspect_python", inspect)
    monkeypatch.setattr(prepare, "_run", run)
    monkeypatch.setattr(prepare, "_probe", lambda *a: {"synthetic_cuda_probe": "passed", "real_model_loads": 0})
    def perform():
        return prepare.prepare_tabpfn(core_python=core, tabpfn_python=target,
                                      checkpoint_path=public_files.path, report_dir=tmp_path / "prepare-report")
    return SimpleNamespace(core=core, python=target, perform=perform, installed=installed,
                           calls=calls, state=state, pins=pins, files=public_files)


def test_new_environment_ready_binds_dependencies_checkpoint_freeze_and_junit(fake_lab):
    f = fake_lab
    result = f.perform()
    assert result["status"] == "ready" and result["checkpoint"]["sha256"] == prepare.CHECKPOINT_SHA256
    assert Path(result["junit"]).is_file() and Path(result["pip_freeze"]).is_file()
    assert result["packages"]["tabpfn"] == "9.2.0" and result["packages"]["torch"] == "2.6.0+cu126"
    assert f.core.read_bytes() == b"existing core, never mutated"
    assert [stage for command, stage in f.calls if command[0] == str(f.core)] == ["venv_create"]
    assert all(command[0] == str(f.python) for command, stage in f.calls if stage != "venv_create")
    assert not result["environment_existed_before"]
    previous = len(f.calls)
    assert f.perform()["status"] == "ready"
    assert all(stage not in {"venv_create", "pip_bootstrap", "torch_install", "dependency_install", "project_install"}
               for _, stage in f.calls[previous:])


def test_owned_interrupted_dependency_install_resumes_without_replacing_core(fake_lab):
    f = fake_lab
    f.state["fail_stage"] = "dependency_install"
    assert f.perform()["failed_stage"] == "dependency_install"
    f.state["fail_stage"] = None
    assert f.perform()["status"] == "ready"
    assert sum(stage == "venv_create" for _, stage in f.calls) == 1
    assert sum(stage == "torch_install" for _, stage in f.calls) == 1


def test_interrupted_bootstrap_can_finish_its_owned_partial_venv(fake_lab):
    f = fake_lab
    f.state["fail_stage"] = "pip_bootstrap"
    assert f.perform()["failed_stage"] == "pip_bootstrap"
    f.state["fail_stage"] = None
    assert f.perform()["status"] == "ready"
    assert sum(stage == "venv_create" for _, stage in f.calls) == 2
    assert f.core.read_bytes() == b"existing core, never mutated"


def test_existing_incompatible_environment_is_not_upgraded(fake_lab):
    f = fake_lab
    f.python.parent.mkdir(parents=True)
    f.python.write_bytes(b"old optional")
    (f.python.parent.parent / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    f.installed.update(f.pins, torch="2.7.0")
    result = f.perform()
    assert result["failed_stage"] == "dependency_versions" and not f.calls
    assert f.python.read_bytes() == b"old optional"


def test_cuda_failure_cannot_report_ready_or_download_weights(fake_lab, monkeypatch):
    f = fake_lab
    def no_cuda(*args):
        raise prepare.PrepareError("cuda_import_probe", "CUDA unavailable")
    monkeypatch.setattr(prepare, "_probe", no_cuda)
    result = f.perform()
    assert result["failed_stage"] == "cuda_import_probe"
    assert not f.files.path.exists() and not f.files.calls


def test_pins_keep_fixed_official_and_project_runtime():
    pins = prepare._pins(prepare.REPO / "requirements-route-batch-tabpfn.txt")
    assert pins["tabpfn"] == "9.2.0"
    assert pins["scikit-learn"] == "1.5.1" and pins["numpy"] == "1.26.4"
    assert pins["huggingface-hub"] == "0.28.1" and pins["pydantic"] == "2.11.7"
