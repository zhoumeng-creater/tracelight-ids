"""Prepare the optional pinned TabPFN environment and public official checkpoint.

No estimator is constructed and no pretrained weights are deserialized here.
The core interpreter only creates an isolated venv. Existing unowned or
incompatible environments are never upgraded. Owned interrupted installs resume.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[3]
HF_REPO = "Prior-Labs/tabpfn_2_5"
HF_REVISION = "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b"
CHECKPOINT_NAME = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
CHECKPOINT_BYTES = 42_935_499
CHECKPOINT_SHA256 = "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"
LICENSE_BYTES = 16_905
LICENSE_SHA256 = "ac1d23429431a622b91c73db741868e69a0140d401ebb20b9e869a835e036e28"
TORCH_VERSION = "2.6.0+cu126"
PIP_VERSION = "25.1.1"
FORMAT = "tracelight-tabpfn-preparation-v1"
MARKER = ".tracelight-tabpfn-preparation.json"
LICENSE_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/LICENSE"
CHECKPOINT_URL = f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{CHECKPOINT_NAME}"


class PrepareError(RuntimeError):
    def __init__(self, stage, message, *, status="failed"):
        self.stage, self.status = stage, status
        super().__init__(message)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path):
    path = Path(path).resolve()
    if path.name != CHECKPOINT_NAME:
        raise PrepareError("checkpoint_verify", f"Retain the version-bearing filename: {CHECKPOINT_NAME}")
    if not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES or _sha256(path) != CHECKPOINT_SHA256:
        raise PrepareError("checkpoint_verify", f"Checkpoint missing or size/SHA-256 differs: {path}")
    return {"path": str(path), "bytes": CHECKPOINT_BYTES, "sha256": CHECKPOINT_SHA256}


def _request(url, *, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "TraceLight-TabPFN-Preparation/1", **(headers or {})})
    for attempt in range(3):
        try:
            # System CA verification remains enabled. No token, browser or account mutation.
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PrepareError("checkpoint_access", "Official Hub requires authenticated access; complete the official access flow.",
                                   status="access_required") from exc
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
        time.sleep(attempt + 1)
    raise AssertionError("download request loop exhausted")


def _download_file(url, target, expected_size, expected_sha):
    """Resume our partial file; never replace an existing final file."""
    target = Path(target)
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size or _sha256(target) != expected_sha:
            raise PrepareError("checkpoint_verify", f"Existing file differs; preserved without overwrite: {target}")
        return "reused"
    partial = target.with_name(target.name + ".download")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        raise PrepareError("checkpoint_verify", f"Partial download exceeds expected length: {partial}")
    if offset < expected_size:
        with _request(url, headers={"Range": f"bytes={offset}-"} if offset else {}) as response:
            status = response.status
            if status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_size}"):
                    raise PrepareError("checkpoint_download", "Official download returned an inconsistent Content-Range")
            elif status == 200:
                offset = 0  # Server may ignore Range; restart only our partial file.
            else:
                raise PrepareError("checkpoint_download", f"Unexpected download status: {status}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("ab" if offset else "wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    if stream.tell() > expected_size:
                        raise PrepareError("checkpoint_verify", "Downloaded file exceeds registered byte count")
    if partial.stat().st_size != expected_size or _sha256(partial) != expected_sha:
        raise PrepareError("checkpoint_verify", f"Downloaded size/SHA-256 differs; partial retained: {partial}")
    # Windows rename refuses a destination created concurrently, preserving it.
    partial.rename(target)
    return "downloaded"


def prepare_checkpoint(checkpoint_path, report_dir):
    """Retrieve a public official file at one revision; never load the checkpoint.

    The caller has authorized downloading the model under its published terms.
    This does not assert or simulate Prior Labs account license acceptance.
    An actual access gate stops preparation; there is no alternate-source fallback.
    """
    target, directory = Path(checkpoint_path).resolve(), Path(report_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stage": "checkpoint_verify", "repo_id": HF_REPO,
              "revision": HF_REVISION, "checkpoint_url": CHECKPOINT_URL, "license_url": LICENSE_URL,
              "license_sha256": LICENSE_SHA256, "account_license_acceptance_claimed": False,
              "real_model_loads": 0}
    try:
        if target.name != CHECKPOINT_NAME:
            raise PrepareError("checkpoint_verify", f"Expected filename: {CHECKPOINT_NAME}")
        if target.exists():
            report.update(checkpoint=verify_checkpoint(target), download="reused", status="ready")
        else:
            report["stage"] = "checkpoint_access"
            url = f"https://huggingface.co/api/models/{HF_REPO}/revision/{HF_REVISION}?blobs=true"
            with _request(url) as response:
                metadata = json.load(response)
            report["hub_access"] = {key: metadata.get(key) for key in ("id", "sha", "private", "gated")}
            if metadata.get("private") is not False or metadata.get("gated") is not False:
                raise PrepareError("checkpoint_access", "Official repository now requires access approval; use its official access flow.",
                                   status="access_required")
            if metadata.get("id") != HF_REPO or metadata.get("sha") != HF_REVISION:
                raise PrepareError("checkpoint_metadata", "Official repository identity/revision differs")
            entry = next((item for item in metadata.get("siblings", []) if item.get("rfilename") == CHECKPOINT_NAME), {})
            if entry.get("size") != CHECKPOINT_BYTES or entry.get("lfs", {}).get("sha256") != CHECKPOINT_SHA256:
                raise PrepareError("checkpoint_metadata", "Official checkpoint size/SHA-256 differs from registration")
            report["stage"] = "license_download"
            license_path = target.with_name(target.name + ".LICENSE")
            _download_file(LICENSE_URL, license_path, LICENSE_BYTES, LICENSE_SHA256)
            report["license_file"] = {"path": str(license_path), "bytes": LICENSE_BYTES, "sha256": LICENSE_SHA256}
            report["stage"] = "checkpoint_download"
            report["download"] = _download_file(CHECKPOINT_URL, target, CHECKPOINT_BYTES, CHECKPOINT_SHA256)
            report.update(checkpoint=verify_checkpoint(target), status="ready")
        report["stage"] = "complete"
    except Exception as exc:
        report.update(status=exc.status if isinstance(exc, PrepareError) else "failed", failed_stage=getattr(exc, "stage", report["stage"]),
                      error=str(exc), error_type=type(exc).__name__)
    _write(directory / "checkpoint-report.json", report)
    return report


def _pins(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            result.update(_pins(Path(path).parent / line[3:].strip()))
        else:
            name, expected = line.split("==")
            result[re.sub(r"[-_.]+", "-", name).lower()] = expected
    return result


def _environment(repo):
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(repo / "src"), "PYTHONNOUSERSITE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TABPFN_DISABLE_TELEMETRY": "1", "TABPFN_NO_BROWSER": "1", "DO_NOT_TRACK": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    return env


def _run(command, stage, attempt, env, *, capture_json=False):
    number = len(list(attempt.glob("command-*.log"))) + 1
    log = attempt / f"command-{number:02d}-{stage}.log"
    if capture_json:
        completed = subprocess.run([str(x) for x in command], env=env, capture_output=True, text=True, encoding="utf-8")
        log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    else:
        print(f"TabPFN prepare: {stage}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run([str(x) for x in command], env=env, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise PrepareError(stage, f"Command failed ({completed.returncode}); see {log}")
    return json.loads(completed.stdout) if capture_json else log


def _inspect_python(python, attempt, env):
    code = ("import json,sys,struct,importlib.metadata as m; "
            "print(json.dumps({'version':list(sys.version_info[:2]),'bits':struct.calcsize('P')*8,"
            "'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable,"
            "'packages':{d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}}))")
    return _run([python, "-X", "utf8", "-c", code], "interpreter", attempt, env, capture_json=True)


def _probe(python, attempt, env):
    code = """
import json, sys, importlib.metadata as metadata
import torch, tabpfn, skrub, lightgbm, pydantic_settings, huggingface_hub, ids
from tabpfn.constants import ModelVersion
assert torch.__version__ == '2.6.0+cu126', torch.__version__
assert torch.version.cuda == '12.6', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA device unavailable'
assert metadata.version('tabpfn') == '9.2.0'
assert ModelVersion.V2_5.value == 'v2.5'
assert callable(tabpfn.TabPFNClassifier.create_default_for_version)
x = torch.arange(16, dtype=torch.float32, device='cuda:0').reshape(4,4)
torch.testing.assert_close(x @ torch.eye(4, device='cuda:0'), x)
torch.cuda.synchronize()
print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
'cuda_name':torch.cuda.get_device_name(0),'tabpfn':metadata.version('tabpfn'),
'model_version':ModelVersion.V2_5.value,'ids_path':str(ids.__file__),
'real_model_loads':0,'real_model_predictions':0,'synthetic_cuda_probe':'passed'}))
"""
    report = _run([python, "-X", "utf8", "-c", code], "cuda_import_probe", attempt, env, capture_json=True)
    _write(attempt / "environment.json", report)
    return report


def prepare_tabpfn(*, core_python, tabpfn_python, checkpoint_path, report_dir):
    """Prepare everything required by the optional worker; return a saved report."""
    core, python = Path(core_python).resolve(), Path(tabpfn_python).resolve()
    report_root, checkpoint = Path(report_dir).resolve(), Path(checkpoint_path).resolve()
    venv = python.parent.parent
    request = {"core_python": str(core), "tabpfn_python": str(python), "checkpoint_path": str(checkpoint),
               "repo": str(REPO.resolve())}
    report = {"format": FORMAT, "status": "running", "request": request, "stage": "paths",
              "real_model_loads": 0, "real_model_predictions": 0, "core_environment_modified": False}
    try:
        if python.name.lower() != "python.exe" or python.parent.name.lower() != "scripts":
            raise PrepareError("paths", "TabPFNPython must be a Windows venv Scripts/python.exe path")
        core_venv = core.parent.parent
        if (core == python or venv.is_relative_to(core_venv) or core_venv.is_relative_to(venv)
                or report_root.is_relative_to(core_venv) or report_root.is_relative_to(venv)
                or checkpoint.is_relative_to(core_venv)):
            raise PrepareError("paths", "Keep the optional venv, report and checkpoint outside the existing core environment")
        report_root.mkdir(parents=True, exist_ok=True)
        binding = report_root / "preparation-spec.json"
        if binding.exists() and _read(binding) != request:
            raise PrepareError("paths", "Report directory belongs to a different preparation request")
        _write(binding, request)
        attempt = report_root / f"attempt-{len(list(report_root.glob('attempt-*')))+1:03d}"
        attempt.mkdir()
        report["attempt_directory"] = str(attempt)
        report["started_utc"] = datetime.now(timezone.utc).isoformat()
        env = _environment(REPO)
        report["stage"] = "core_interpreter"
        core_info = _inspect_python(core, attempt, env)
        if core_info["version"] != [3, 12] or core_info["bits"] != 64:
            raise PrepareError("core_interpreter", "Use the existing 64-bit Python 3.12 core interpreter")
        requirements = REPO / "requirements-route-batch-tabpfn.txt"
        pins = {**_pins(requirements), "torch": TORCH_VERSION, "pip": PIP_VERSION}
        identity = {"format": FORMAT, "python": str(python), "repo": str(REPO.resolve()), "pins": pins}
        marker = venv / MARKER
        existed_before = venv.exists()
        owned = marker.is_file()
        report["stage"] = "environment_ownership"
        if owned:
            state = _read(marker)
            if state.get("identity") != identity:
                raise PrepareError("environment_ownership", "Existing managed environment has a different preparation specification")
        elif venv.exists():
            state = {"phase": "unowned_verify_only"}
        else:
            venv.mkdir(parents=True)
            state, owned = {"identity": identity, "phase": "bootstrap"}, True
            _write(marker, state)
        if owned and state["phase"] == "bootstrap":
            # An interrupted venv creation can leave python.exe but no complete
            # configuration/pip. Recreate only a directory owned by this request.
            report["stage"] = "venv_create"
            _run([core, "-m", "venv", venv], "venv_create", attempt, env)
        elif not python.is_file():
            raise PrepareError("environment_ownership", "Existing environment cannot be started; preserved without modification")
        report["stage"] = "dependency_versions"
        info = _inspect_python(python, attempt, env)
        configuration = venv / "pyvenv.cfg"
        if (info["version"] != [3, 12] or info["bits"] != 64 or Path(info["prefix"]).resolve() != venv
                or info["prefix"] == info["base_prefix"] or not configuration.is_file()
                or "include-system-site-packages = false" not in configuration.read_text().lower()):
            raise PrepareError("dependency_versions", "Optional interpreter is not the requested isolated 64-bit Python 3.12 venv")
        installed = info["packages"]
        conflicts = {name: installed[name] for name, expected in pins.items()
                     if name in installed and installed[name] != expected
                     and not (name == "pip" and owned and state["phase"] == "bootstrap")}
        missing = [name for name in pins if name not in installed]
        if conflicts or (missing and not owned):
            raise PrepareError("dependency_versions", f"Preserving incompatible/unowned environment; conflicts={conflicts}, missing={missing}")
        if owned and state["phase"] != "complete":
            if state["phase"] == "bootstrap":
                report["stage"] = "pip_bootstrap"
                _run([python, "-m", "pip", "install", f"pip=={PIP_VERSION}"], "pip_bootstrap", attempt, env)
                state["phase"] = "dependencies"
                _write(marker, state)
            if "torch" not in installed:
                report["stage"] = "torch_install"
                _run([python, "-m", "pip", "install", "--no-deps", f"torch=={TORCH_VERSION}",
                      "--index-url", "https://download.pytorch.org/whl/cu126", "--report", attempt / "torch-install.json"],
                     "torch_install", attempt, env)
            report["stage"] = "dependency_install"
            _run([python, "-m", "pip", "install", "--only-binary=:all:", "-r", requirements,
                  "--report", attempt / "dependencies-install.json"], "dependency_install", attempt, env)
            report["stage"] = "project_install"
            _run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable", REPO,
                  "--report", attempt / "project-install.json"], "project_install", attempt, env)
        report["stage"] = "dependency_verify"
        installed = _inspect_python(python, attempt, env)["packages"]
        mismatch = {name: installed.get(name) for name, expected in pins.items() if installed.get(name) != expected}
        if mismatch:
            raise PrepareError("dependency_verify", f"Pinned dependency versions differ: {mismatch}")
        report["packages"] = {name: installed[name] for name in sorted(pins)}
        _run([python, "-m", "pip", "check"], "pip_check", attempt, env)
        freeze_log = _run([python, "-m", "pip", "freeze", "--all"], "pip_freeze", attempt, env)
        report["pip_freeze"] = str(freeze_log)
        report["stage"] = "cuda_import_probe"
        report["environment"] = _probe(python, attempt, env)
        report["stage"] = "synthetic_tests"
        junit = attempt / "synthetic-tests.xml"
        test_files = [REPO / "tests/test_route_batch_foundation.py", *sorted((REPO / "tests").glob("test_route_batch_tabpfn_prepare*.py"))]
        test_env = {**env, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        _run([python, "-m", "pytest", "-q", "--junitxml", junit, *test_files], "synthetic_tests", attempt, test_env)
        if not junit.is_file():
            raise PrepareError("synthetic_tests", "Pytest returned without its JUnit evidence")
        report["junit"] = str(junit)
        if owned:
            state["phase"] = "complete"
            _write(marker, state)
        report["stage"] = "checkpoint_prepare"
        checkpoint_report = prepare_checkpoint(checkpoint, attempt)
        report["checkpoint_preparation"] = checkpoint_report
        if checkpoint_report["status"] != "ready":
            raise PrepareError(checkpoint_report["failed_stage"], checkpoint_report["error"], status=checkpoint_report["status"])
        report["checkpoint"] = verify_checkpoint(checkpoint)
        report.update(status="ready", stage="complete", environment_existed_before=existed_before)
    except Exception as exc:
        report.update(status=exc.status if isinstance(exc, PrepareError) else "failed", failed_stage=getattr(exc, "stage", report["stage"]),
                      error=str(exc), error_type=type(exc).__name__)
    # Do not write an invalid requested report path inside the protected core env.
    if report.get("failed_stage") != "paths":
        _write(report_root / "prepare-report.json", report)
        if "attempt_directory" in report:
            _write(Path(report["attempt_directory"]) / "prepare-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-python")
    parser.add_argument("--tabpfn-python")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        result = prepare_checkpoint(args.checkpoint, args.output_directory)
    else:
        if not args.core_python or not args.tabpfn_python:
            parser.error("--core-python and --tabpfn-python are required without --download-only")
        result = prepare_tabpfn(core_python=args.core_python, tabpfn_python=args.tabpfn_python,
                               checkpoint_path=args.checkpoint, report_dir=args.output_directory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    sys.exit(main())
