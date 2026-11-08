"""Exercise the real wrapper in Windows PowerShell and pwsh with a Python stub."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/RUN_ROUTE_BATCH.ps1"
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.fixture(params=SHELLS or [None], ids=lambda path: Path(path).stem if path else "unavailable")
def shell(request):
    if request.param is None:
        pytest.skip("PowerShell is required for the Windows wrapper checks")
    return request.param


def exercise(shell, tmp_path, *, resume=False, failure="", mismatch=False, relative=False, confirmation="",
             missing_confirmation_file=""):
    output = tmp_path / "output with spaces"
    source = tmp_path / "round14 with spaces"
    checkpoint = tmp_path / "weights with spaces" / "model.ckpt"
    pfn_checkpoint = tmp_path / "weights with spaces" / "pfn.ckpt"
    pfn_python = tmp_path / "pfn environment" / "python.exe"
    log = tmp_path / "calls.jsonl"
    if confirmation == "both":
        (tmp_path / "raw source.zip").write_bytes(b"synthetic-source")
        prepared = tmp_path / "round13 prepared"
        prepared.mkdir()
        for name in ("support.csv", "normal_calibration.csv", "development_evaluation.csv", "preparation.json"):
            (prepared / name).write_bytes(b"synthetic-prepared")
        if missing_confirmation_file:
            missing = (tmp_path / "raw source.zip" if missing_confirmation_file == "source_zip"
                       else prepared / missing_confirmation_file)
            missing.unlink()
    if resume:
        (output / "run").mkdir(parents=True)
        (output / "config.json").write_text(json.dumps({
            "round14_root": str(source), "checkpoint": {"path": str(checkpoint)},
            "tabpfn_checkpoint": {"path": str(pfn_checkpoint)}, "tabpfn_python": str(pfn_python),
        }), encoding="utf-8")
    harness = tmp_path / "exercise.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tokens = $null
$errors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:ROUTE_WRAPPER, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw ($errors | Out-String) }
function Stub-Python {
    $arguments = @($args)
    ConvertTo-Json -Compress -InputObject $arguments | Add-Content -LiteralPath $env:ROUTE_CALLS -Encoding UTF8
    if ($arguments[2] -eq $env:ROUTE_FAIL) { $global:LASTEXITCODE = 7; return }
    if ($arguments[2] -eq 'run') {
        $outputIndex = [Array]::IndexOf($arguments, '--output')
        [IO.Directory]::CreateDirectory($arguments[$outputIndex + 1]) | Out-Null
    }
    $global:LASTEXITCODE = 0
}
$params = @{Python='Stub-Python'; OutputRoot=$env:ROUTE_OUTPUT;
            Round14Root=$env:ROUTE_SOURCE; Checkpoint=$env:ROUTE_CHECKPOINT;
            TabPFNCheckpoint=$env:ROUTE_PFN_CHECKPOINT; TabPFNPython=$env:ROUTE_PFN_PYTHON}
if ($env:ROUTE_RESUME -eq '1') { $params.Resume=$true; $params.RetryUnavailable=$true }
if ($env:ROUTE_CONFIRM -ne '') { $params.SourceZip=$env:ROUTE_SOURCE_ZIP }
if ($env:ROUTE_CONFIRM -eq 'both') { $params.Round13Prepared=$env:ROUTE_ROUND13 }
& $env:ROUTE_WRAPPER @params
''', encoding="utf-8")
    env = dict(os.environ, ROUTE_WRAPPER=str(WRAPPER), ROUTE_CALLS=str(log),
               ROUTE_OUTPUT=str(output), ROUTE_SOURCE=str(source),
               ROUTE_CHECKPOINT=str(checkpoint if not mismatch else checkpoint.with_name("other.ckpt")),
               ROUTE_PFN_CHECKPOINT=str(pfn_checkpoint), ROUTE_PFN_PYTHON=str(pfn_python),
               ROUTE_RESUME="1" if resume else "0", ROUTE_FAIL=failure,
               ROUTE_CONFIRM=confirmation, ROUTE_SOURCE_ZIP=str(tmp_path / "raw source.zip"),
               ROUTE_ROUND13=str(tmp_path / "round13 prepared"))
    if relative:
        for key in ("ROUTE_OUTPUT", "ROUTE_SOURCE", "ROUTE_CHECKPOINT", "ROUTE_PFN_CHECKPOINT", "ROUTE_PFN_PYTHON"):
            env[key] = os.path.relpath(env[key], tmp_path)
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(harness)], cwd=tmp_path if relative else ROOT, env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []
    return result, calls, output, checkpoint, pfn_checkpoint, pfn_python


def test_wrapper_parses_and_forwards_complete_batch_with_space_paths(shell, tmp_path):
    result, calls, output, checkpoint, pfn_checkpoint, pfn_python = exercise(shell, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[2] for call in calls] == ["register", "preflight", "run", "pack"]
    register = calls[0]
    assert register[register.index("--checkpoint") + 1] == str(checkpoint)
    assert register[register.index("--tabpfn-checkpoint") + 1] == str(pfn_checkpoint)
    assert register[register.index("--tabpfn-python") + 1] == str(pfn_python)
    assert register[register.index("--output") + 1] == str(output / "config.json")
    assert calls[2][-2:] == ["--output", str(output / "run")]
    assert calls[3][calls[3].index("--run-dir") + 1] == str(output / "run")
    assert Path(calls[3][-1]).name.startswith("route-results-")
    assert Path(calls[3][-1]).suffix == ".zip"
    assert "--include-models" not in calls[3]


def test_wrapper_resume_reuses_config_and_forwards_retry(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, resume=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[2] for call in calls] == ["preflight", "run", "pack"]
    assert calls[1][-2:] == ["--resume", "--retry-unavailable"]
    assert (output / "config.json").is_file()


def test_relative_paths_are_resolved_before_changing_to_repo(shell, tmp_path):
    result, calls, output, checkpoint, pfn_checkpoint, _ = exercise(shell, tmp_path, relative=True)
    assert result.returncode == 0, result.stdout + result.stderr
    register = calls[0]
    assert register[register.index("--checkpoint") + 1] == str(checkpoint)
    assert register[register.index("--tabpfn-checkpoint") + 1] == str(pfn_checkpoint)
    assert register[register.index("--output") + 1] == str(output / "config.json")


def test_confirmation_runs_between_batch_and_analysis_pack(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, confirmation="both")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [call[2] for call in calls] == ["register", "preflight", "run", "confirm", "pack"]
    assert calls[3] == ["-m", "ids.route_batch", "confirm", "--run-dir", str(output / "run"),
                        "--source-zip", str(tmp_path / "raw source.zip"),
                        "--round13-prepared", str(tmp_path / "round13 prepared")]


def test_confirmation_requires_both_paths_before_any_external_call(shell, tmp_path):
    result, calls, *_ = exercise(shell, tmp_path, confirmation="one")
    assert result.returncode != 0
    assert calls == []
    assert "must be supplied together" in result.stdout + result.stderr


def test_confirmation_failure_still_packs_results_and_returns_original_error(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, confirmation="both", failure="confirm")
    assert result.returncode != 0
    assert [call[2] for call in calls] == ["register", "preflight", "run", "confirm", "pack"]
    errors = list((output / "run").glob("confirmation-wrapper-*-error.txt"))
    assert len(errors) == 1
    assert "exit code 7" in errors[0].read_text(encoding="utf-8")
    assert "exit code 7" in result.stdout + result.stderr


def test_confirmation_retry_forwards_original_freeze_retry_flag(shell, tmp_path):
    result, calls, *_ = exercise(shell, tmp_path, resume=True, confirmation="both")
    assert result.returncode == 0, result.stdout + result.stderr
    confirmation = next(call for call in calls if call[2] == "confirm")
    assert confirmation[-1] == "--retry-failed"
    assert "Task-level resume" in result.stdout
    assert "optimizer state is not resumed" in result.stdout


@pytest.mark.parametrize("missing", ["source_zip", "support.csv", "normal_calibration.csv",
                                     "development_evaluation.csv", "preparation.json"])
def test_missing_confirmation_inputs_stop_before_training(shell, tmp_path, missing):
    result, calls, *_ = exercise(shell, tmp_path, confirmation="both", missing_confirmation_file=missing)
    assert result.returncode != 0
    assert calls == []
    assert "existing file" in result.stdout + result.stderr or "missing required file" in result.stdout + result.stderr


@pytest.mark.parametrize("failure,expected", [("register", ["register"]),
                                              ("preflight", ["register", "preflight"]),
                                              ("run", ["register", "preflight", "run"]),
                                              ("pack", ["register", "preflight", "run", "pack"])])
def test_wrapper_stops_after_each_external_failure(shell, tmp_path, failure, expected):
    result, calls, *_ = exercise(shell, tmp_path, failure=failure)
    assert result.returncode != 0
    assert [call[2] for call in calls] == expected
    assert "exit code 7" in result.stdout + result.stderr


def test_existing_config_rejects_silently_changed_checkpoint(shell, tmp_path):
    result, calls, output, checkpoint, *_ = exercise(shell, tmp_path, resume=True, mismatch=True)
    assert result.returncode != 0
    assert calls == []
    assert json.loads((output / "config.json").read_text(encoding="utf-8"))["checkpoint"]["path"] == str(checkpoint)
