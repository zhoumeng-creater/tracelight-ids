"""Synthetic shell checks for the dedicated follow-up; no models or installers run."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "RUN_ROUTE_FOLLOWUP.ps1"
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.fixture(params=SHELLS or [None], ids=lambda p: Path(p).stem if p else "unavailable")
def shell(request):
    if request.param is None:
        pytest.skip("PowerShell is required for wrapper checks")
    return request.param


def exercise(shell, tmp_path, *, preparation="ready", failure="", complete=True, resume=False,
             occupied=False, overlap=False, missing_original=False, relative=False, checkpoint_default=False,
             protected_target=""):
    checkout = tmp_path / "new checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    wrapper = scripts / WRAPPER.name
    shutil.copyfile(WRAPPER, wrapper)
    original = tmp_path / "original checkout" / "local" / "old batch" / "run"
    original.mkdir(parents=True)
    checkpoint = tmp_path / "weights with spaces" / "tabpfn.ckpt"
    pfn_python = tmp_path / "optional venv" / "Scripts" / "python.exe"
    if protected_target == "checkpoint":
        checkpoint = original / "weights" / "tabpfn.ckpt"
    elif protected_target == "python":
        pfn_python = original / "venv" / "Scripts" / "python.exe"
    for name in ("source_snapshot.json", "summary.json"):
        (original / name).write_text("{}", encoding="utf-8")
    (original / "protocol.json").write_text(json.dumps({"tabpfn_checkpoint": {"path": str(checkpoint)}}), encoding="utf-8")
    if missing_original:
        (original / "source_snapshot.json").unlink()
    originals = {p.name: p.read_bytes() for p in original.iterdir()}
    output = original if overlap else tmp_path / "followup output"
    if resume or occupied:
        (output / "run").mkdir(parents=True)
    log, after = tmp_path / "calls.jsonl", tmp_path / "environment-after.json"
    (scripts / "PREPARE_ROUTE_TABPFN.ps1").write_text(r'''
param([string]$CorePython, [string]$TabPFNPython, [string]$Checkpoint, [string]$OutputDirectory)
$ErrorActionPreference='Stop'
@{stage='prepare'; args=@{CorePython=$CorePython; TabPFNPython=$TabPFNPython; Checkpoint=$Checkpoint; OutputDirectory=$OutputDirectory}} |
    ConvertTo-Json -Compress | Add-Content -LiteralPath $env:FOLLOWUP_CALLS -Encoding UTF8
[IO.Directory]::CreateDirectory((Join-Path $OutputDirectory 'attempt-001')) | Out-Null
[IO.File]::WriteAllText((Join-Path $OutputDirectory 'attempt-001\setup.log'), 'synthetic preparation log')
$status = 'ready'
if ($env:FOLLOWUP_PREPARE -in @('fail', 'not_ready')) { $status='access_required' }
if ($env:FOLLOWUP_PREPARE -ne 'missing_report') {
    @{status=$status; failed_stage='checkpoint_access'; error='synthetic status'} | ConvertTo-Json |
        Set-Content -LiteralPath (Join-Path $OutputDirectory 'prepare-report.json') -Encoding UTF8
}
if ($env:FOLLOWUP_PREPARE -in @('fail', 'ready_nonzero')) { exit 7 }
exit 0
''', encoding="utf-8")
    harness = tmp_path / "harness.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$tokens=$null; $parseErrors=$null
$null=[System.Management.Automation.Language.Parser]::ParseFile($env:FOLLOWUP_WRAPPER,[ref]$tokens,[ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
function Stub-Python {
    $arguments=@($args)
    $stage = if ($arguments[1] -eq 'pytest') { 'core-synthetic-tests' } else { $arguments[2] }
    @{stage=$stage; args=$arguments} | ConvertTo-Json -Compress |
        Add-Content -LiteralPath $env:FOLLOWUP_CALLS -Encoding UTF8
    if ($stage -eq 'core-synthetic-tests') {
        if ($env:FOLLOWUP_FAIL -ne 'missing_junit') {
            $index=[Array]::IndexOf($arguments,'--junitxml')
            [IO.File]::WriteAllText($arguments[$index+1], '<testsuites><testsuite tests="1" failures="0" errors="0" /></testsuites>')
        }
        Write-Output 'synthetic core check output'
    }
    if ($arguments[2] -eq 'followup-preflight' -and $env:FOLLOWUP_FAIL -ne 'followup-preflight') {
        $index=[Array]::IndexOf($arguments,'--report')
        [IO.File]::WriteAllText($arguments[$index+1], '{}')
    }
    if ($arguments[2] -eq 'followup') {
        $index=[Array]::IndexOf($arguments,'--output')
        $runPath=$arguments[$index+1]
        [IO.Directory]::CreateDirectory($runPath) | Out-Null
        @{all_tasks_complete=($env:FOLLOWUP_COMPLETE -eq '1')} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $runPath 'summary.json') -Encoding UTF8
    }
    if ($arguments[2] -eq 'followup-pack' -and $env:FOLLOWUP_FAIL -ne 'followup-pack') {
        $index=[Array]::IndexOf($arguments,'--output')
        [IO.File]::WriteAllText($arguments[$index+1], 'synthetic archive marker')
    }
    $global:LASTEXITCODE=0
    if ($stage -eq $env:FOLLOWUP_FAIL) { $global:LASTEXITCODE=9 }
}
$params=@{Python='Stub-Python'; OriginalRun=$env:FOLLOWUP_ORIGINAL; OutputRoot=$env:FOLLOWUP_OUTPUT; TabPFNPython=$env:FOLLOWUP_PYTHON}
if ($env:FOLLOWUP_DEFAULT_CHECKPOINT -ne '1') { $params.TabPFNCheckpoint=$env:FOLLOWUP_CHECKPOINT }
if ($env:FOLLOWUP_RESUME -eq '1') { $params.Resume=$true }
$env:PYTHONPATH='original-python-path'
$env:OMP_NUM_THREADS='7'
try { & $env:FOLLOWUP_WRAPPER @params }
finally {
    @{pythonpath=$env:PYTHONPATH; omp=$env:OMP_NUM_THREADS; cwd=(Get-Location).Path} |
        ConvertTo-Json | Set-Content -LiteralPath $env:FOLLOWUP_AFTER -Encoding UTF8
}
''', encoding="utf-8")
    env = dict(os.environ, FOLLOWUP_WRAPPER=str(wrapper), FOLLOWUP_CALLS=str(log), FOLLOWUP_AFTER=str(after),
               FOLLOWUP_ORIGINAL=str(original), FOLLOWUP_OUTPUT=str(output), FOLLOWUP_PYTHON=str(pfn_python),
               FOLLOWUP_CHECKPOINT=str(checkpoint), FOLLOWUP_PREPARE=preparation, FOLLOWUP_FAIL=failure,
               FOLLOWUP_COMPLETE="1" if complete else "0", FOLLOWUP_RESUME="1" if resume else "0",
               FOLLOWUP_DEFAULT_CHECKPOINT="1" if checkpoint_default else "0")
    if relative:
        for name in ("FOLLOWUP_ORIGINAL", "FOLLOWUP_OUTPUT", "FOLLOWUP_PYTHON", "FOLLOWUP_CHECKPOINT"):
            env[name] = os.path.relpath(env[name], tmp_path)
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=40)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []
    assert {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()} == originals
    restored = json.loads(after.read_text(encoding="utf-8-sig"))
    assert restored == {"pythonpath": "original-python-path", "omp": "7", "cwd": str(tmp_path)}
    return result, calls, output, original, checkpoint, pfn_python


def stages(calls):
    return [item["stage"] for item in calls]


def report(output):
    files = list(output.glob("followup-wrapper-*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_complete_followup_forwards_only_dedicated_commands_and_preserves_original(shell, tmp_path):
    result, calls, output, original, checkpoint, pfn_python = exercise(shell, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert stages(calls) == ["prepare", "followup-preflight", "core-synthetic-tests", "followup", "followup-pack"]
    assert calls[0]["args"] == {"CorePython": "Stub-Python", "TabPFNPython": str(pfn_python),
                                "Checkpoint": str(checkpoint), "OutputDirectory": str(output / "preparation")}
    arguments = calls[3]["args"]
    for name, value in (("--original-run", original), ("--output", output / "run"),
                        ("--preparation-report", output / "preparation" / "followup-preparation-report.json")):
        assert arguments[arguments.index(name)+1] == str(value)
    assert report(output)["all_tasks_complete"] is True
    assert report(output)["preparation"] == "ready"
    assert report(output)["core_synthetic_tests"] == "passed"
    assert report(output)["pack"] == "complete"
    assert (output / "run/preparation-evidence/attempt-001/setup.log").is_file()
    assert list((output / "run/preparation-evidence").glob("core-synthetic-tests-*.xml"))
    assert list((output / "run/preparation-evidence").glob("core-synthetic-tests-*.log"))
    test_args = calls[2]["args"]
    assert test_args[:4] == ["-m", "pytest", "-q", "--junitxml"]
    assert {Path(name).name for name in test_args[5:]} == {
        "test_route_batch_foundation.py", "test_route_batch_replay.py", "test_route_batch_deployment_replay.py",
        "test_route_batch_cost_worker.py", "test_route_batch_followup.py", "test_route_batch_followup_cli.py"}
    assert list(output.glob("route-followup-results-*.zip"))


@pytest.mark.parametrize("preparation", ["fail", "not_ready", "missing_report", "ready_nonzero"])
def test_preparation_failure_keeps_diagnostics_and_pack_but_returns_nonzero(shell, tmp_path, preparation):
    result, calls, output, *_ = exercise(shell, tmp_path, preparation=preparation)
    assert result.returncode != 0
    assert stages(calls) == ["prepare", "followup-preflight", "core-synthetic-tests", "followup", "followup-pack"]
    assert report(output)["preparation"] == "failed"
    assert report(output)["pack"] == "complete"
    assert report(output)["errors"]
    assert (output / "run/preparation-evidence/attempt-001/setup.log").is_file()
    effective = json.loads((output / "preparation/followup-preparation-report.json").read_text(encoding="utf-8"))
    assert effective["status"] != "ready"
    assert effective["wrapper_status"] == "failed"
    if preparation == "ready_nonzero":
        raw = json.loads((output / "preparation/prepare-report.json").read_text(encoding="utf-8-sig"))
        assert raw["status"] == "ready"
        assert effective["failed_stage"] == "preparation_wrapper"


def test_queue_return_zero_with_incomplete_tasks_is_not_success(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, complete=False)
    assert result.returncode != 0
    assert stages(calls)[-1] == "followup-pack"
    assert report(output)["all_tasks_complete"] is False
    assert "unfinished" in result.stdout + result.stderr


def test_followup_command_failure_still_packs_existing_results(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, failure="followup")
    assert result.returncode != 0
    assert stages(calls)[-1] == "followup-pack"
    assert report(output)["followup"] == "command_failed"
    assert "exit code 9" in result.stdout + result.stderr


def test_preflight_identity_failure_stops_before_followup(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, failure="followup-preflight")
    assert result.returncode != 0
    assert stages(calls) == ["prepare", "followup-preflight"]
    assert report(output)["preflight"] == "failed"
    assert not (output / "run").exists()


def test_pack_failure_is_reported_and_nonzero(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, failure="followup-pack")
    assert result.returncode != 0
    assert stages(calls)[-1] == "followup-pack"
    assert report(output)["pack"] == "failed"


def test_resume_is_forwarded_only_to_new_followup(shell, tmp_path):
    result, calls, output, *_ = exercise(shell, tmp_path, resume=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[3]["args"][-1] == "--resume"
    assert "--resume" not in calls[1]["args"]
    assert report(output)["resume"] is True
    assert "original batch is read-only" in result.stdout


@pytest.mark.parametrize("mode", ["occupied", "overlap", "missing_original"])
def test_invalid_paths_stop_before_preparation(shell, tmp_path, mode):
    result, calls, *_ = exercise(shell, tmp_path, **{mode: True})
    assert result.returncode != 0
    assert calls == []


def test_relative_paths_resolve_before_switching_checkout(shell, tmp_path):
    result, calls, output, original, checkpoint, pfn_python = exercise(shell, tmp_path, relative=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0]["args"]["TabPFNPython"] == str(pfn_python)
    assert calls[0]["args"]["Checkpoint"] == str(checkpoint)
    args = calls[3]["args"]
    assert args[args.index("--original-run")+1] == str(original)
    assert args[args.index("--output")+1] == str(output / "run")


def test_checkpoint_default_comes_from_original_protocol(shell, tmp_path):
    result, calls, _, _, checkpoint, _ = exercise(shell, tmp_path, checkpoint_default=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0]["args"]["Checkpoint"] == str(checkpoint)


@pytest.mark.parametrize("protected_target", ["python", "checkpoint"])
def test_preparation_targets_cannot_modify_original_run(shell, tmp_path, protected_target):
    result, calls, *_ = exercise(shell, tmp_path, protected_target=protected_target)
    assert result.returncode != 0
    assert calls == []
    assert "outside OriginalRun" in result.stdout + result.stderr


@pytest.mark.parametrize("failure", ["core-synthetic-tests", "missing_junit"])
def test_core_synthetic_failure_blocks_real_followup(shell, tmp_path, failure):
    result, calls, output, *_ = exercise(shell, tmp_path, failure=failure)
    assert result.returncode != 0
    assert stages(calls) == ["prepare", "followup-preflight", "core-synthetic-tests"]
    assert report(output)["core_synthetic_tests"] == "failed"
    assert not (output / "run").exists()
    assert list(output.glob("core-synthetic-tests-*.log"))
