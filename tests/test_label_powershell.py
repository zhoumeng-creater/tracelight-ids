"""Execute the wrapper's actual helper and pip call without installing anything."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.mark.parametrize("shell", SHELLS or [None], ids=lambda p: Path(p).stem if p else "unavailable")
@pytest.mark.parametrize("spaced_path", [False, True], ids=["script-dot", "space-path"])
def test_real_powershell_helper_forwards_pip_editable_arguments(shell, spaced_path, tmp_path):
    if shell is None:
        pytest.skip("PowerShell is required to exercise the Windows laboratory wrapper")
    harness = tmp_path / "exercise-wrapper.ps1"
    harness.write_text(r'''
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    $env:LABEL_WRAPPER_PATH, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw 'Wrapper parse failed' }
$helper = $ast.Find({ param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Invoke-LabelPython'
}, $true)
if ($null -eq $helper) { throw 'Actual helper missing' }
Invoke-Expression $helper.Extent.Text

# This stub receives exactly the argv that would be passed to the Python executable.
# The advanced helper still binds the real command first, so -e reproduces the bug.
function Capture-Arguments {
    $global:LASTEXITCODE = 0
    ConvertTo-Json -Compress -InputObject @($args)
}
$Python = 'Capture-Arguments'
$calls = @($ast.FindAll({ param($node)
    $node -is [System.Management.Automation.Language.CommandAst] -and
    $node.GetCommandName() -eq 'Invoke-LabelPython' -and
    $node.CommandElements.Count -gt 2 -and $node.CommandElements[2].Extent.Text -eq 'pip'
}, $true))
if ($calls.Count -ne 1) { throw 'Expected the actual single pip invocation' }
$call = $calls[0]
$command = $call.Extent.Text
if ($env:LABEL_USE_SPACED_PATH -eq '1') {
    $last = $call.CommandElements[$call.CommandElements.Count - 1]
    $offset = $last.Extent.StartOffset - $call.Extent.StartOffset
    $quoted = "'" + $env:LABEL_PROJECT_PATH.Replace("'", "''") + "'"
    $command = $command.Substring(0, $offset) + $quoted
}
Invoke-Expression $command
''', encoding="utf-8")
    project = str(tmp_path / "project with spaces")
    env = dict(os.environ, LABEL_WRAPPER_PATH=str(ROOT / "scripts/RUN_SCANNING_LABEL.ps1"),
               LABEL_USE_SPACED_PATH="1" if spaced_path else "0", LABEL_PROJECT_PATH=project)
    process = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                              "-File", str(harness)], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout.strip()) == [
        "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable",
        project if spaced_path else ".",
    ]
