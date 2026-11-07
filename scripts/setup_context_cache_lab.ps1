[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$VenvDir,
    [string]$Python = 'python',
    [string]$CheckOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvPath = [IO.Path]::GetFullPath($VenvDir)
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$tabiclPath = Join-Path $repoRoot 'local\vendor\tabicl'
$tabiclCommit = 'd83091b845a8c5f7c087bd54964f755a9e560f45'
if (-not $CheckOutput) {
    $runName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $CheckOutput = Join-Path $repoRoot "local\context-cache-setup\$runName"
}
$checkPath = [IO.Path]::GetFullPath($CheckOutput)
if (Test-Path -LiteralPath $checkPath) {
    throw 'CheckOutput must be a new directory; installation reports are not overwritten.'
}

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string]$Executable, [string[]]$ArgumentList)
    Write-Host "Running: $Executable $($ArgumentList -join ' ')"
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

Invoke-Checked -Executable $Python -ArgumentList @(
    '-c', 'import sys, struct; assert sys.version_info[:2] == (3, 12), ''Use Python 3.12''; assert struct.calcsize(''P'') == 8, ''Use 64-bit Python'''
)
if (Test-Path -LiteralPath $venvPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $venvPath 'pyvenv.cfg')) -or -not (Test-Path -LiteralPath $venvPython)) {
        throw 'VenvDir exists but is not a Windows virtual environment. Choose a dedicated directory.'
    }
} else {
    Invoke-Checked -Executable $Python -ArgumentList @('-m', 'venv', $venvPath)
}
Invoke-Checked -Executable $venvPython -ArgumentList @(
    '-c', 'import sys, pathlib; assert sys.version_info[:2] == (3, 12); assert sys.prefix != sys.base_prefix; cfg = (pathlib.Path(sys.prefix) / ''pyvenv.cfg'').read_text(); assert ''include-system-site-packages = false'' in cfg.lower(), ''Use an isolated venv'''
)

# Reuse an intact fixed checkout; never reset an existing modified checkout.
if (-not (Test-Path -LiteralPath $tabiclPath)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $tabiclPath) | Out-Null
    Invoke-Checked -Executable 'git' -ArgumentList @(
        'clone', '--filter=blob:none', '--no-checkout', 'https://github.com/soda-inria/tabicl.git', $tabiclPath
    )
    Invoke-Checked -Executable 'git' -ArgumentList @('-C', $tabiclPath, 'checkout', '--detach', $tabiclCommit)
}
$actualCommit = (Invoke-Checked -Executable 'git' -ArgumentList @('-C', $tabiclPath, 'rev-parse', 'HEAD') | Out-String).Trim()
if ($actualCommit -ne $tabiclCommit) {
    throw "TabICL HEAD must be $tabiclCommit; found $actualCommit. Use the pinned clean checkout."
}
$sourceChanges = (Invoke-Checked -Executable 'git' -ArgumentList @('-C', $tabiclPath, 'status', '--porcelain', '--untracked-files=no') | Out-String).Trim()
if ($sourceChanges) {
    throw 'The existing TabICL checkout has tracked modifications. Preserve it and provide the pinned clean checkout.'
}
New-Item -ItemType Directory -Path $checkPath | Out-Null

$savedEnvironment = @{}
foreach ($name in @('PYTHONPATH', 'PYTHONNOUSERSITE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'CUBLAS_WORKSPACE_CONFIG')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location -LiteralPath $repoRoot
try {
    # Test the installed packages without inherited source-path/user-site overrides.
    $env:PYTHONPATH = $null
    $env:PYTHONNOUSERSITE = '1'
    Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'install', 'pip==25.1.1')
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'pip', 'install', '--only-binary=:all:', '-r', (Join-Path $repoRoot 'requirements-context-cache-lab.txt'),
        '--report', (Join-Path $checkPath 'pip-install-report.json')
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'pip', 'install', '--no-deps', 'torch==2.6.0+cu126', '--index-url', 'https://download.pytorch.org/whl/cu126'
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', $tabiclPath
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', $repoRoot
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'check')
    Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'freeze', '--all') |
        Set-Content -LiteralPath (Join-Path $checkPath 'pip-freeze.txt') -Encoding UTF8

    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $probe = @'
import importlib.metadata as metadata
import json
import pathlib
import sys
import torch
import tabicl
from ids.context_cache.preprocessing import require_pinned_preprocessing
assert torch.__version__ == '2.6.0+cu126', torch.__version__
assert torch.version.cuda == '12.6', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA unavailable: verify NVIDIA driver and GPU visibility'
assert torch.cuda.device_count() > 0
device = torch.device('cuda:0')
x = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
torch.testing.assert_close(x @ torch.eye(4, device=device), x)
torch.cuda.synchronize(device)
_, source = require_pinned_preprocessing()
report = {
    'stage': 'environment_and_synthetic_cuda_probe',
    'python': sys.version,
    'python_executable': sys.executable,
    'torch': str(torch.__version__),
    'cuda_runtime': torch.version.cuda,
    'device': str(device),
    'cuda_matmul_check': 'passed',
    'devices': [
        {'index': i, 'name': torch.cuda.get_device_name(i),
         'capability': list(torch.cuda.get_device_capability(i)),
         'total_memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ],
    'tabicl_package_version': metadata.version('tabicl'),
    'tabicl_import_path': tabicl.__file__,
    'upstream_source': source,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
'@
    Invoke-Checked -Executable $venvPython -ArgumentList @('-c', $probe, (Join-Path $checkPath 'environment.json'))
    $testFiles = @(
        'tests/test_weighted_context_cache.py',
        'tests/test_tabicl_cache_adapter.py',
        'tests/test_context_compiler.py',
        'tests/test_context_cache_cli.py',
        'tests/test_context_csv_preprocessing.py',
        'tests/test_context_calibration.py',
        'tests/test_context_csv_workflow.py',
        'tests/test_context_lab_data.py'
    )
    Invoke-Checked -Executable $venvPython -ArgumentList (
        @('-m', 'pytest', '-q', '--junitxml', (Join-Path $checkPath 'synthetic-tests.xml')) + $testFiles
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'ids.context_cache', 'synthetic-check', '--output', (Join-Path $checkPath 'native-synthetic')
    )
    Invoke-Checked -Executable $venvPython -ArgumentList @(
        '-m', 'ids.context_cache', 'synthetic-csv-check', '--output', (Join-Path $checkPath 'csv-synthetic')
    )
    Write-Output "Environment ready: $venvPython"
    Write-Output "Installation and synthetic reports: $checkPath"
} finally {
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
