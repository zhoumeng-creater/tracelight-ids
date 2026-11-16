"""Run the actual PowerShell entry against synthetic files and a Python stub."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]
PROTOCOL = b'{"synthetic": "r26"}\n'


@pytest.fixture(params=SHELLS or [None], ids=lambda path: Path(path).stem if path else "unavailable")
def shell(request):
    if request.param is None:
        pytest.skip("PowerShell is unavailable")
    return request.param


@pytest.fixture
def layout(tmp_path):
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/RUN_MODEL_TRAINING.ps1", repo / "scripts")
    for folder, value in (
        ("model-training-20261116", {
            "version": "model-training-r29-v1",
            "design_file": "configs/model-design-20261116/design.json",
            "r26_protocol_sha256": hashlib.sha256(PROTOCOL).hexdigest(),
        }),
        ("model-design-20261116", {
            "roles": {"F": "fit", "V": "selection"},
            "training": {"development_trajectories": 7, "final_refits": 7, "max_fits": 14},
            "calibration": {"partition": "C normal only"},
        }),
    ):
        directory = repo / "configs" / folder
        directory.mkdir(parents=True)
        filename = "registration.json" if folder.startswith("model-training") else "design.json"
        (directory / filename).write_text(json.dumps(value), encoding="utf-8")
    (repo / "tests").mkdir()
    for filename in ("test_model_training.py", "test_model_training_powershell.py",
                     "test_recover_r26_training_rows.py", "test_unrelated.py"):
        (repo / "tests" / filename).touch()
    source = tmp_path / "r26 return with spaces"
    source.mkdir()
    (source / "protocol.json").write_bytes(PROTOCOL)
    recovered = tmp_path / "input with spaces" / "recovered-training.parquet"
    recovered.parent.mkdir()
    recovered.write_bytes(b"synthetic numeric frame")
    return {"repo": repo, "source": source, "recovered": recovered,
            "output": tmp_path / "output with spaces", "tmp": tmp_path}


def exercise(shell, layout, *, modes=(), explicit_python=True, no_inputs=False,
             auto_source=False, original=False, raw=False, fail=False, packaged=False, threads=3):
    repo, source = layout["repo"], layout["source"]
    if original:
        numeric = source / "local/training/mixed.parquet"
        numeric.parent.mkdir(parents=True)
        numeric.write_bytes(b"retained numeric frame")
    if packaged:
        numeric = repo / "model-training-inputs/recovered-training.parquet"
        numeric.parent.mkdir(parents=True)
        shutil.copy2(layout["recovered"], numeric)
    if raw:
        (layout["tmp"] / "raw archives").mkdir()
    log = layout["tmp"] / "calls.jsonl"
    harness = layout["tmp"] / "exercise.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:MT_WRAPPER, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
function Stub-Python {
    [ordered]@{args=@($args); pythonpath=$env:PYTHONPATH} |
        ConvertTo-Json -Depth 10 -Compress | Add-Content -LiteralPath $env:MT_CALLS -Encoding UTF8
    $global:LASTEXITCODE = [int]$env:MT_EXIT
}
$params = @{OutputDir=$env:MT_OUTPUT}
if ($env:MT_EXPLICIT_PYTHON -eq '1') { $params.Python = 'Stub-Python' }
if ($env:MT_NO_INPUTS -ne '1') {
    if ($env:MT_AUTO_SOURCE -ne '1') { $params.R26Run = $env:MT_SOURCE }
    if ($env:MT_RAW -eq '1') { $params.RawDir = $env:MT_RAW_DIR }
    elseif ($env:MT_PACKAGED -ne '1') { $params.RecoveredTraining = $env:MT_RECOVERED }
}
$params.Threads = [int]$env:MT_THREADS
$params.Device = 'cuda:0'
foreach ($mode in ($env:MT_MODES -split ',')) {
    if ($mode) { $params[$mode] = $true }
}
$env:PYTHONPATH = 'preserved-original-path'
try { & $env:MT_WRAPPER @params }
finally {
    if ($env:PYTHONPATH -ne 'preserved-original-path') { throw 'PYTHONPATH was not restored' }
}
''', encoding="utf-8")
    env = dict(os.environ, MT_WRAPPER=str(repo / "scripts/RUN_MODEL_TRAINING.ps1"),
               MT_CALLS=str(log), MT_OUTPUT=str(layout["output"]), MT_SOURCE=str(source),
               MT_RECOVERED=str(layout["recovered"]), MT_EXPLICIT_PYTHON=str(int(explicit_python)),
               MT_NO_INPUTS=str(int(no_inputs)), MT_AUTO_SOURCE=str(int(auto_source)),
               MT_RAW=str(int(raw)), MT_RAW_DIR=str(layout["tmp"] / "raw archives"),
               MT_PACKAGED=str(int(packaged)), MT_MODES=",".join(modes), MT_THREADS=str(threads),
               MT_EXIT="7" if fail else "0")
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(harness)], cwd=layout["tmp"], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []
    return result, calls


def test_plan_requires_neither_python_nor_r26(shell, layout):
    shutil.rmtree(layout["source"])
    result, calls = exercise(shell, layout, modes=("PlanOnly",), explicit_python=False, no_inputs=True)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["version"] == "model-training-r29-v1"
    assert plan["training"]["max_fits"] == 14
    assert calls == []
    assert not layout["output"].exists()


def test_full_call_preserves_paths_and_sets_pythonpath(shell, layout):
    result, calls = exercise(shell, layout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[:2] == ["-m", "ids.model_training"]
    for flag, value in (("--output-dir", layout["output"]), ("--r26-run", layout["source"]),
                        ("--recovered-training", layout["recovered"]), ("--threads", 3), ("--device", "cuda:0")):
        assert args[args.index(flag) + 1] == str(value)
    assert calls[0]["pythonpath"] == str(layout["repo"] / "src")
    assert not any(arg.endswith("-only") for arg in args)


@pytest.mark.parametrize("mode,flag", [("InputsOnly", "--inputs-only"),
                                      ("VerifyOnly", "--verify-only"), ("PackOnly", "--pack-only")])
def test_modes_do_not_accidentally_enter_training(shell, layout, mode, flag):
    no_inputs = mode != "InputsOnly"
    if no_inputs:
        shutil.rmtree(layout["source"])
    result, calls = exercise(shell, layout, modes=(mode,), no_inputs=no_inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    assert calls[0]["args"][-1] == flag
    assert ("--r26-run" in calls[0]["args"]) is not no_inputs


def test_check_only_has_explicit_scoped_tests_without_inputs(shell, layout):
    result, calls = exercise(shell, layout, modes=("CheckOnly",), no_inputs=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[:3] == ["-m", "pytest", "-q"]
    assert {Path(path).name for path in args[3:]} == {
        "test_model_training.py", "test_model_training_powershell.py", "test_recover_r26_training_rows.py"}


def test_check_only_requires_explicit_python(shell, layout):
    result, calls = exercise(shell, layout, modes=("CheckOnly",), explicit_python=False, no_inputs=True)
    assert result.returncode != 0
    assert "requires an explicit -Python" in result.stdout + result.stderr
    assert calls == []


def test_multiple_modes_rejected_before_python(shell, layout):
    result, calls = exercise(shell, layout, modes=("InputsOnly", "PackOnly"))
    assert result.returncode != 0
    assert "Choose only one" in result.stdout + result.stderr
    assert calls == []


@pytest.mark.parametrize("threads", [1, 65])
def test_invalid_thread_budget_rejected_before_python(shell, layout, threads):
    result, calls = exercise(shell, layout, threads=threads)
    assert result.returncode != 0
    assert "Threads" in result.stdout + result.stderr
    assert calls == []


def test_retained_original_takes_priority_over_recovered(shell, layout):
    result, calls = exercise(shell, layout, original=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--recovered-training" not in calls[0]["args"]


def test_missing_frame_uses_existing_raw_without_download(shell, layout):
    result, calls = exercise(shell, layout, raw=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args = calls[0]["args"]
    assert args[args.index("--raw-dir") + 1] == str(layout["tmp"] / "raw archives")
    assert "--recovered-training" not in args


def test_packaged_frame_discovery_is_narrow(shell, layout):
    result, calls = exercise(shell, layout, packaged=True)
    assert result.returncode == 0, result.stdout + result.stderr
    args = calls[0]["args"]
    assert args[args.index("--recovered-training") + 1] == str(
        layout["repo"] / "model-training-inputs/recovered-training.parquet")


@pytest.mark.parametrize("count", [1, 2])
def test_adjacent_r26_search_requires_exactly_one_hash_match(shell, layout, count):
    for name in ("tracelight-support-diagnostic", "task-reset-20261115")[:count]:
        adjacent = layout["repo"].parent / name / "local/support-diagnostic-r26"
        adjacent.mkdir(parents=True)
        (adjacent / "protocol.json").write_bytes(PROTOCOL)
    result, calls = exercise(shell, layout, auto_source=True)
    if count == 1:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(calls) == 1
    else:
        assert result.returncode != 0
        assert "found 2 matching" in result.stdout + result.stderr
        assert calls == []


def test_wrong_protocol_fails_before_python(shell, layout):
    (layout["source"] / "protocol.json").write_bytes(b"wrong run")
    result, calls = exercise(shell, layout)
    assert result.returncode != 0
    assert "protocol hash differs" in result.stdout + result.stderr
    assert calls == []


def test_python_failure_does_not_retry_or_start_packing(shell, layout):
    result, calls = exercise(shell, layout, fail=True)
    assert result.returncode != 0
    assert len(calls) == 1
    assert "exited with code 7" in result.stdout + result.stderr
    assert "incomplete fit" in result.stdout + result.stderr
