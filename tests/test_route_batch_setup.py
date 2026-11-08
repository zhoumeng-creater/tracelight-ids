"""Exercise PowerShell argument forwarding and existing-environment refusal."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup_route_batch_lab.ps1"
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


def invoke(shell, script, args=(), *, env=None):
    return subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), *args],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )


@pytest.mark.parametrize("shell", SHELLS or [None], ids=lambda p: Path(p).stem if p else "unavailable")
def test_setup_parser_and_actual_helper_keep_paths_and_pip_arguments(shell, tmp_path):
    if shell is None:
        pytest.skip("PowerShell unavailable")
    harness = tmp_path / "exercise setup.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:ROUTE_SETUP_SCRIPT, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw ($parseErrors | Out-String) }
$helper = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Invoke-RouteCommand'
}, $true)
if ($null -eq $helper) { throw 'Actual command helper missing' }
Invoke-Expression $helper.Extent.Text
function Capture-Arguments {
    $global:LASTEXITCODE = 0
    ConvertTo-Json -Compress -InputObject @($args)
}
$tabpfnPython = 'Capture-Arguments'
$repoRoot = $env:ROUTE_SETUP_PROJECT
$tabpfnReports = $env:ROUTE_SETUP_REPORTS
$commands = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.CommandAst] -and
    $node.GetCommandName() -eq 'Invoke-RouteCommand' -and
    $node.Extent.Text.Contains("'--editable'")
}, $true))
if ($commands.Count -ne 1) { throw 'Expected one actual project install invocation' }
Invoke-Expression $commands[0].Extent.Text
''', encoding="utf-8")
    project = str(tmp_path / "project with spaces")
    reports = str(tmp_path / "reports with spaces")
    result = invoke(shell, harness, env=dict(os.environ, ROUTE_SETUP_SCRIPT=str(SCRIPT),
                                           ROUTE_SETUP_PROJECT=project, ROUTE_SETUP_REPORTS=reports))
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in reversed(result.stdout.splitlines()) if line.startswith("["))
    assert json.loads(line) == [
        "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable", project,
        "--report", str(Path(reports) / "project-install-report.json"),
    ]


@pytest.mark.parametrize("shell", SHELLS or [None], ids=lambda p: Path(p).stem if p else "unavailable")
def test_existing_core_is_rejected_before_python_or_writes(shell, tmp_path):
    if shell is None:
        pytest.skip("PowerShell unavailable")
    core = tmp_path / "existing core"
    core.mkdir()
    marker = core / "user-file.txt"
    marker.write_text("preserve", encoding="utf-8")
    report = tmp_path / "report"
    result = invoke(shell, SCRIPT, ["-CoreVenvDir", str(core), "-Python", "must-not-run",
                                    "-CheckOutput", str(report)])
    assert result.returncode != 0
    assert "CoreVenvDir must be new" in result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert not report.exists()


@pytest.mark.parametrize("shell", SHELLS or [None], ids=lambda p: Path(p).stem if p else "unavailable")
def test_tabpfn_only_keeps_core_and_refuses_existing_optional_env(shell, tmp_path):
    if shell is None:
        pytest.skip("PowerShell unavailable")
    core = tmp_path / "core"
    (core / "Scripts").mkdir(parents=True)
    interpreter = core / "Scripts" / "python.exe"
    interpreter.write_bytes(b"placeholder; never execute")
    optional = core.parent / "tracelight-route-tabpfn"
    optional.mkdir()
    result = invoke(shell, SCRIPT, ["-CoreVenvDir", str(core), "-TabPFNOnly", "-Python", "must-not-run"])
    assert result.returncode != 0
    assert "TabPFNVenvDir must be new" in result.stdout + result.stderr
    assert interpreter.read_bytes() == b"placeholder; never execute"
