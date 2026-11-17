"""Exercise the real wrapper with synthetic directories and a PowerShell Python stub.

No real Python child, GPU, training command, historical input or model is used.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.fixture(params=SHELLS or [None], ids=lambda path: Path(path).stem if path else "unavailable")
def shell(request):
    if request.param is None:
        pytest.skip("PowerShell is unavailable")
    return request.param


@pytest.fixture
def layout(tmp_path):
    repo = tmp_path / "repo with spaces"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/RUN_MODEL_TRAINING_REDESIGN.ps1", repo / "scripts")
    config = repo / "configs/model-training-redesign-20261117"
    config.mkdir(parents=True)
    (config / "registration.json").write_text(json.dumps({
        "version": "model-training-redesign-r30-v1",
        "design_file": "configs/model-training-redesign-20261117/design.json",
    }), encoding="utf-8")
    (config / "design.json").write_text(json.dumps({
        "arms": [{"id": "protocol_blind"}, {"id": "protocol_aware"}],
        "training": {"seeds": [17, 29, 43]},
        "selection": {"nll_affects_selection": False, "patience_validation_events": 16},
    }), encoding="utf-8")
    (repo / "tests").mkdir()
    for name in ("test_model_training_redesign_data.py", "test_model_training_redesign_training.py",
                 "test_model_training_redesign_powershell.py", "test_model_training.py", "test_unrelated.py"):
        (repo / "tests" / name).touch()
    r29, raw, r26 = (tmp_path / name for name in ("r29 return with spaces", "raw ZIP directory", "optional r26"))
    for folder in (r29, raw, r26):
        folder.mkdir()
    (r29 / "protocol.json").write_text('{"synthetic":true}', encoding="utf-8")
    return {"repo": repo, "r29": r29, "raw": raw, "r26": r26,
            "output": tmp_path / "output with spaces", "tmp": tmp_path}


def exercise(shell, layout, *, modes=(), python="stub", omit=(), r26=False,
             fail_call=0, threads=4, prior_pythonpath="original-python-path"):
    log, harness = layout["tmp"] / "calls.jsonl", layout["tmp"] / "exercise.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:TRD_WRAPPER, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$global:RedesignStubCalls = 0
function Stub-Python {
    $global:RedesignStubCalls += 1
    [ordered]@{args=@($args); pythonpath=$env:PYTHONPATH} |
        ConvertTo-Json -Depth 10 -Compress | Add-Content -LiteralPath $env:TRD_CALLS -Encoding UTF8
    $global:LASTEXITCODE = 0
    if ($global:RedesignStubCalls -eq [int]$env:TRD_FAIL_CALL) { $global:LASTEXITCODE = 7 }
}
$params = @{OutputDir=$env:TRD_OUTPUT; Threads=[int]$env:TRD_THREADS; Device='cuda:0'}
if ($env:TRD_PYTHON -eq 'stub') { $params.Python = 'Stub-Python' }
elseif ($env:TRD_PYTHON -eq 'missing') { $params.Python = 'missing-redesign-python-93845.exe' }
$omit = @($env:TRD_OMIT -split ',')
if ('R29Run' -notin $omit) { $params.R29Run = $env:TRD_R29 }
if ('RawDir' -notin $omit) { $params.RawDir = $env:TRD_RAW }
if ($env:TRD_R26_ENABLED -eq '1') { $params.R26Run = $env:TRD_R26 }
foreach ($mode in ($env:TRD_MODES -split ',')) {
    if ($mode) { $params[$mode] = $true }
}
if ($env:TRD_PRIOR_PRESENT -eq '1') { $env:PYTHONPATH = $env:TRD_PRIOR_VALUE }
else { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }
$expectedPythonPath = $env:PYTHONPATH
try { & $env:TRD_WRAPPER @params }
finally {
    if ($env:PYTHONPATH -ne $expectedPythonPath) { throw 'PYTHONPATH was not restored' }
}
''', encoding="utf-8")
    env = dict(os.environ, TRD_WRAPPER=str(layout["repo"] / "scripts/RUN_MODEL_TRAINING_REDESIGN.ps1"),
               TRD_CALLS=str(log), TRD_OUTPUT=str(layout["output"]), TRD_R29=str(layout["r29"]),
               TRD_RAW=str(layout["raw"]), TRD_R26=str(layout["r26"]), TRD_R26_ENABLED=str(int(r26)),
               TRD_PYTHON=python, TRD_OMIT=",".join(omit), TRD_MODES=",".join(modes),
               TRD_FAIL_CALL=str(fail_call), TRD_THREADS=str(threads),
               TRD_PRIOR_PRESENT=str(int(prior_pythonpath is not None)), TRD_PRIOR_VALUE=prior_pythonpath or "")
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(harness)], cwd=layout["tmp"], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []
    return result, calls


def test_plan_only_requires_no_python_input_or_gpu(shell, layout):
    shutil.rmtree(layout["r29"])
    shutil.rmtree(layout["raw"])
    result, calls = exercise(shell, layout, modes=("PlanOnly",), python="missing", omit=("R29Run", "RawDir"))
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["max_fits"] == 6 and plan["final_refits"] == 0
    assert plan["model_execution"] is False
    assert plan["seeds"] == [17, 29, 43]
    assert plan["selection"]["nll_affects_selection"] is False
    assert calls == [] and not layout["output"].exists()


@pytest.mark.parametrize("modes", [("PlanOnly", "CheckOnly"), ("InputsOnly", "PackOnly")])
def test_only_modes_are_mutually_exclusive_before_python(shell, layout, modes):
    result, calls = exercise(shell, layout, modes=modes)
    assert result.returncode != 0 and "Choose exactly one" in result.stdout + result.stderr
    assert calls == []


def test_check_only_requires_explicit_python(shell, layout):
    result, calls = exercise(shell, layout, modes=("CheckOnly",), python="default", omit=("R29Run", "RawDir"))
    assert result.returncode != 0 and "requires an explicit -Python" in result.stdout + result.stderr
    assert calls == []


def test_check_only_checks_pinned_packages_then_only_scoped_tests(shell, layout):
    result, calls = exercise(shell, layout, modes=("CheckOnly",), omit=("R29Run", "RawDir"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 2 and calls[0]["args"][0] == "-c"
    assert "0.0.3" in calls[0]["args"][1] and "0.0.12" in calls[0]["args"][1]
    args = calls[1]["args"]
    assert args[:3] == ["-m", "pytest", "-q"]
    assert {Path(path).name for path in args[3:]} == {
        "test_model_training_redesign_data.py", "test_model_training_redesign_training.py",
        "test_model_training_redesign_powershell.py"}
    assert not layout["output"].exists()


def test_missing_author_dependencies_stop_before_pytest(shell, layout):
    result, calls = exercise(shell, layout, modes=("CheckOnly",), omit=("R29Run", "RawDir"), fail_call=1)
    assert result.returncode != 0 and "dependencies are missing" in result.stdout + result.stderr
    assert len(calls) == 1 and calls[0]["args"][0] == "-c"


@pytest.mark.parametrize("mode,flag", [("InputsOnly", "--inputs-only"), ("VerifyOnly", "--verify-only"),
                                      ("PackOnly", "--pack-only")])
def test_nontraining_modes_forward_exactly_one_flag(shell, layout, mode, flag):
    no_input = mode != "InputsOnly"
    result, calls = exercise(shell, layout, modes=(mode,), omit=("R29Run", "RawDir") if no_input else ())
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[:2] == ["-m", "ids.model_training_redesign"] and args[-1] == flag
    assert ("--r29-run" in args) is not no_input
    assert ("--raw-dir" in args) is not no_input


@pytest.mark.parametrize("prior", [None, "original-python-path"])
def test_run_arguments_reach_only_stub_with_space_paths_and_restore_environment(shell, layout, prior):
    # This mode is sent to Stub-Python, never to a Python executable.
    result, calls = exercise(shell, layout, r26=True, threads=1, prior_pythonpath=prior)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(calls) == 1
    args = calls[0]["args"]
    for flag, value in (("--output-dir", layout["output"]), ("--r29-run", layout["r29"]),
                        ("--raw-dir", layout["raw"]), ("--r26-run", layout["r26"]),
                        ("--threads", 1), ("--device", "cuda:0")):
        assert args[args.index(flag) + 1] == str(value)
    assert not any(arg.endswith("-only") for arg in args)
    expected_path = str(layout["repo"] / "src") + (os.pathsep + prior if prior else "")
    assert calls[0]["pythonpath"] == expected_path
    assert not layout["output"].exists()


@pytest.mark.parametrize("omitted", ["R29Run", "RawDir"])
def test_missing_required_input_argument_fails_before_python(shell, layout, omitted):
    result, calls = exercise(shell, layout, modes=("InputsOnly",), omit=(omitted,))
    assert result.returncode != 0 and "Pass -R29Run and -RawDir" in result.stdout + result.stderr
    assert calls == []


@pytest.mark.parametrize("folder", ["r29", "raw"])
def test_missing_input_directory_fails_before_python(shell, layout, folder):
    shutil.rmtree(layout[folder])
    result, calls = exercise(shell, layout, modes=("InputsOnly",))
    assert result.returncode != 0 and "Input directory missing" in result.stdout + result.stderr
    assert calls == []


def test_missing_r29_protocol_is_not_accepted_as_a_run(shell, layout):
    (layout["r29"] / "protocol.json").unlink()
    result, calls = exercise(shell, layout, modes=("InputsOnly",))
    assert result.returncode != 0 and "directory containing R29 protocol.json" in result.stdout + result.stderr
    assert calls == []


def test_failure_restores_environment_without_retry(shell, layout):
    result, calls = exercise(shell, layout, modes=("InputsOnly",), fail_call=1)
    assert result.returncode != 0 and "exited with code 7" in result.stdout + result.stderr
    assert "PYTHONPATH was not restored" not in result.stdout + result.stderr
    assert len(calls) == 1 and calls[0]["args"][-1] == "--inputs-only"


@pytest.mark.parametrize("threads", [0, 65])
def test_invalid_thread_budget_rejected_before_python(shell, layout, threads):
    result, calls = exercise(shell, layout, modes=("InputsOnly",), threads=threads)
    assert result.returncode != 0 and "Threads" in result.stdout + result.stderr
    assert calls == []
