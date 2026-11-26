"""Validate and run one copy of the offline Windows residual IDS experiment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from tempfile import mkdtemp
from zipfile import ZIP_DEFLATED, ZipFile

from scripts.residual_lab.common import (
    BUNDLE_MANIFEST, BUNDLE_SCHEMA, HERE, WHEEL_LOCK, checked_member, fingerprint, read_json,
    verify_file, wheel_entries, write_json,
)

ROOT = HERE.parents[1]
RUNTIME_PROBE = """
import importlib.metadata as metadata
import json
import platform
import sys
import sysconfig

import joblib
import numpy
import pandas
import psutil
import rtdl_num_embeddings
import scipy
import sklearn
import tabm
import threadpoolctl
import torch

torch.ones((1, 8), dtype=torch.float32).sum().item()
print(json.dumps({
    'python_version': platform.python_version(),
    'platform': sysconfig.get_platform(),
    'torch_cuda': torch.version.cuda,
    'distributions': {name: metadata.version(name) for name in sys.argv[1:]},
}))
"""


def verify_bundle(root):
    print("1/3 Verifying source, experiment inputs and offline dependency files.", flush=True)
    manifest = read_json(root / BUNDLE_MANIFEST)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("Unknown laboratory bundle schema.")
    for name, expected in manifest["files"].items():
        verify_file(checked_member(root, name), expected)
    return manifest


@contextmanager
def install_lock(path):
    import msvcrt

    with path.open("a+b") as lock:
        if not lock.tell():
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Another launcher is preparing this runtime. Wait for its window to finish.") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def runtime_report(python, expected, env):
    if not python.is_file():
        return None
    process = subprocess.run(
        [str(python), "-s", "-c", RUNTIME_PROBE, *expected],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, encoding="utf-8", errors="replace",
    )
    if process.returncode != 0:
        return None
    report = json.loads(process.stdout)
    if report["distributions"] != expected or report["torch_cuda"] is not None:
        return None
    return report


def setup_runtime(root, env):
    print("2/3 Checking the Windows CPU runtime.", flush=True)
    entries = wheel_entries()
    expected = {entry["package"]: entry["version"] for entry in entries}
    signature = fingerprint(WHEEL_LOCK)["sha256"][:12]
    cache = root / ".lab-runtime"
    cache.mkdir(exist_ok=True)
    patch = ".".join(map(str, sys.version_info[:3]))
    runtime = cache / f"py{patch}-{signature}"
    python = runtime / "Scripts" / "python.exe"
    with install_lock(cache / "install.lock"):
        report = runtime_report(python, expected, env)
        if report is not None:
            print(f"Using offline CPU environment with Python {report['python_version']}.", flush=True)
            return python, report
        print("Creating the package-local environment and installing pinned wheels offline.", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(runtime)], check=True, env=env)
        requirements = cache / f"requirements-{signature}.txt"
        requirements.write_text(
            "\n".join(f"{e['package']}=={e['version']} --hash=sha256:{e['sha256']}" for e in entries) + "\n",
            encoding="ascii",
        )
        subprocess.run([
            str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "install", "--no-index",
            "--find-links", str(root / "wheelhouse"), "--require-hashes", "--only-binary=:all:",
            "--no-cache-dir", "--force-reinstall", "-r", str(requirements),
        ], check=True, env=env)
        subprocess.run(
            [str(python), "-m", "pip", "--isolated", "--disable-pip-version-check", "check"], check=True, env=env,
        )
        report = runtime_report(python, expected, env)
        if report is None:
            raise RuntimeError("Installed wheels failed runtime checks. Check the Python and CPU runtime on this PC.")
    return python, report


def experiment_command(python, root):
    return [
        str(python), "-s", "-u", "-m", "scripts.residual_ids.run", "--archive",
        str(root / "data/frozen-models.zip"), "--bundle", str(root / "data"), "--device", "cpu",
    ]


def build_return(run, status, returns):
    returns.mkdir(exist_ok=True)
    suffix = "" if status == "complete" else "-failed"
    archive = returns / f"residual-rescue-{run.name}{suffix}.zip"
    temporary = archive.with_suffix(".zip.part")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
            for file in sorted(run.rglob("*")):
                if file.is_file():
                    bundle.write(file, file.relative_to(run).as_posix())
        with ZipFile(temporary) as bundle:
            if bad := bundle.testzip():
                raise ValueError(f"Could not verify the result ZIP member: {bad}")
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    check = archive.with_suffix(".zip.sha256.json")
    write_json(check, {"filename": archive.name, "status": status, **fingerprint(archive)})
    print(f"Result ZIP: {archive}", flush=True)
    print(f"Result checksum: {check}", flush=True)


def verify_result(result):
    receipt_path = result / "output_receipt.json"
    artifacts = read_json(receipt_path)
    observed = {
        path.relative_to(result).as_posix()
        for path in result.rglob("*") if path.is_file() and path != receipt_path
    }
    if set(artifacts) != observed or not {"registration.json", "summary.json"} <= observed:
        raise ValueError("The output receipt does not cover the expected run artifacts.")
    for name, expected in artifacts.items():
        verify_file(checked_member(result, name), expected)


def run_once(root, command, manifest, report, env):
    print("3/3 Running the registered residual IDS experiment on CPU.", flush=True)
    logs = root / "runs"
    logs.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    run = Path(mkdtemp(prefix=started.strftime("%Y%m%dT%H%M%SZ-"), dir=logs))
    result = run / "result"
    returncode = 1
    status = "failed"
    try:
        with (run / "console.log").open("w", encoding="utf-8", newline="\n") as log:
            with subprocess.Popen(
                [*command, "--output", str(result)], cwd=root, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            ) as process:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                returncode = process.wait()
            if returncode == 0:
                try:
                    verify_result(result)
                except (OSError, ValueError) as error:
                    returncode = 1
                    message = f"Result verification failed: {error}"
                    log.write(message + "\n")
                    print(message, flush=True)
                else:
                    status = "complete"
    finally:
        write_json(run / "lab-run.json", {
            "status": status, "exit_code": returncode, "device": "cpu", "runtime": report,
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_manifest_sha256": fingerprint(root / BUNDLE_MANIFEST)["sha256"],
            "source_head": manifest["source_head"], "source_tree_dirty": manifest["source_tree_dirty"],
        })
        build_return(run, status, root / "returns")
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--setup-only", action="store_true", help="verify and install, then stop")
    modes.add_argument("--check-only", action="store_true", help="verify, install and check inputs without fitting")
    options = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sysconfig.get_platform() != "win-amd64":
        parser.error("Windows x64 CPython 3.12 is required.")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PIP_CONFIG_FILE=os.devnull)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSAFEPATH", None)
    try:
        manifest = verify_bundle(ROOT)
        python, report = setup_runtime(ROOT, env)
        if options.setup_only:
            print(json.dumps(report, indent=2))
            return 0
        env["PYTHONPATH"] = str(ROOT / "src")
        command = experiment_command(python, ROOT)
        if options.check_only:
            print("3/3 Checking frozen inputs without fitting or scoring.", flush=True)
            return subprocess.call([*command, "--check-only"], cwd=ROOT, env=env)
        return run_once(ROOT, command, manifest, report, env)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"Launch failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
