[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CoreVenvDir,
    [string]$Python = 'python',
    [switch]$WithTabPFN,
    [string]$TabPFNVenvDir,
    [switch]$TabPFNOnly,
    [string]$CheckOutput
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$corePath = [IO.Path]::GetFullPath($CoreVenvDir)
$corePython = Join-Path $corePath 'Scripts\python.exe'
$installTabPFN = $WithTabPFN -or $TabPFNOnly -or [bool]$TabPFNVenvDir
if (-not $TabPFNVenvDir) {
    $TabPFNVenvDir = Join-Path (Split-Path -Parent $corePath) 'tracelight-route-tabpfn'
}
$tabpfnPath = [IO.Path]::GetFullPath($TabPFNVenvDir)
$tabpfnPython = Join-Path $tabpfnPath 'Scripts\python.exe'
if ($corePath.TrimEnd('\') -ieq $tabpfnPath.TrimEnd('\')) {
    throw 'Core and TabPFN environments must be different directories.'
}
if ($TabPFNOnly) {
    if (-not (Test-Path -LiteralPath $corePython)) {
        throw 'TabPFNOnly requires the existing route-batch core interpreter; it will not modify it.'
    }
} elseif (Test-Path -LiteralPath $corePath) {
    throw 'CoreVenvDir must be new. Existing environments are not modified.'
}
if ($installTabPFN -and (Test-Path -LiteralPath $tabpfnPath)) {
    throw 'TabPFNVenvDir must be new. Existing environments are not modified.'
}
if (-not $CheckOutput) {
    $runName = (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $CheckOutput = Join-Path $repoRoot "local\route-batch-setup\$runName"
}
$checkPath = [IO.Path]::GetFullPath($CheckOutput)
if (Test-Path -LiteralPath $checkPath) {
    throw 'CheckOutput must be a new directory; installation reports are not overwritten.'
}
foreach ($environmentPath in @($corePath, $tabpfnPath)) {
    if ($checkPath.TrimEnd('\') -ieq $environmentPath.TrimEnd('\') -or
        $checkPath.StartsWith($environmentPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Keep CheckOutput outside the environment directories.'
    }
}

function Invoke-RouteCommand {
    param([Parameter(Mandatory = $true)][string]$Executable, [string[]]$ArgumentList)
    Write-Host "Running: $Executable $($ArgumentList -join ' ')"
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

function Save-RouteEnvironment {
    param([string]$Interpreter, [string]$ReportDirectory)
    Invoke-RouteCommand -Executable $Interpreter -ArgumentList @('-m', 'pip', 'check') |
        Set-Content -LiteralPath (Join-Path $ReportDirectory 'pip-check.txt') -Encoding UTF8
    Invoke-RouteCommand -Executable $Interpreter -ArgumentList @('-m', 'pip', 'freeze', '--all') |
        Set-Content -LiteralPath (Join-Path $ReportDirectory 'pip-freeze.txt') -Encoding UTF8
}

Invoke-RouteCommand -Executable $Python -ArgumentList @(
    '-c', 'import sys, struct; assert sys.version_info[:2] == (3, 12), ''Use Python 3.12''; assert struct.calcsize(''P'') == 8, ''Use 64-bit Python'''
)
New-Item -ItemType Directory -Path $checkPath | Out-Null
$savedEnvironment = @{}
$offlineVariables = @('PYTHONPATH', 'PYTHONNOUSERSITE', 'HF_HUB_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY',
    'TRANSFORMERS_OFFLINE', 'TABPFN_NO_BROWSER', 'TABPFN_DISABLE_TELEMETRY', 'DO_NOT_TRACK',
    'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
foreach ($name in $offlineVariables) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$summary = [ordered]@{
    status = 'running'; core_python = $corePython; core_status = 'not_run'
    tabpfn_python = $tabpfnPython; tabpfn_status = 'not_requested'
    real_model_loads = 0; pretrained_weight_downloads = 0
}
Push-Location -LiteralPath $repoRoot
try {
    $env:PYTHONPATH = $null
    $env:PYTHONNOUSERSITE = '1'
    $env:HF_HUB_OFFLINE = '1'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    $env:TABPFN_NO_BROWSER = '1'
    $env:TABPFN_DISABLE_TELEMETRY = '1'
    $env:DO_NOT_TRACK = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'

    if (-not $TabPFNOnly) {
        $coreReports = Join-Path $checkPath 'core'
        New-Item -ItemType Directory -Path $coreReports | Out-Null
        # The existing installer owns pinned TabICL, CUDA and its synthetic checks.
        # It receives a new venv, never an older laboratory environment.
        & (Join-Path $PSScriptRoot 'setup_context_cache_lab.ps1') -VenvDir $corePath -Python $Python `
            -CheckOutput (Join-Path $coreReports 'context-cache')
        Invoke-RouteCommand -Executable $corePython -ArgumentList @(
            '-m', 'pip', 'install', '--dry-run', '--only-binary=:all:',
            '-r', (Join-Path $repoRoot 'requirements-route-batch.txt'),
            '--report', (Join-Path $coreReports 'dependency-dry-run.json')
        )
        Invoke-RouteCommand -Executable $corePython -ArgumentList @(
            '-m', 'pip', 'install', '--no-deps', '--only-binary=:all:',
            'tabm==0.0.3', 'rtdl_num_embeddings==0.0.12',
            '--report', (Join-Path $coreReports 'addons-install-report.json')
        )
        Save-RouteEnvironment -Interpreter $corePython -ReportDirectory $coreReports
        Invoke-RouteCommand -Executable $corePython -ArgumentList @(
            '-c', 'import importlib.metadata as m; import tabm, rtdl_num_embeddings; assert m.version(''tabm'') == ''0.0.3''; assert m.version(''rtdl_num_embeddings'') == ''0.0.12''; print(''TabM and numerical embeddings imports passed; no pretrained model loaded'')'
        )
        $routeTests = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'tests') -Filter 'test_route_batch*.py' |
            Sort-Object Name | ForEach-Object { $_.FullName })
        if ($routeTests.Count -eq 0) { throw 'Route-batch tests are missing from the checkout.' }
        Invoke-RouteCommand -Executable $corePython -ArgumentList (
            @('-m', 'pytest', '-q', '--junitxml', (Join-Path $coreReports 'route-synthetic-tests.xml')) + $routeTests
        )
        $summary.core_status = 'ready'
    } else {
        $summary.core_status = 'existing_unmodified'
    }

    if ($installTabPFN) {
        $summary.tabpfn_status = 'installing'
        $tabpfnReports = Join-Path $checkPath 'tabpfn'
        New-Item -ItemType Directory -Path $tabpfnReports | Out-Null
        Invoke-RouteCommand -Executable $Python -ArgumentList @('-m', 'venv', $tabpfnPath)
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @('-m', 'pip', 'install', 'pip==25.1.1')
        # Install the registered CUDA wheel first so the resolver cannot choose a different Torch.
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-m', 'pip', 'install', '--no-deps', 'torch==2.6.0+cu126',
            '--index-url', 'https://download.pytorch.org/whl/cu126',
            '--report', (Join-Path $tabpfnReports 'torch-install-report.json')
        )
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-m', 'pip', 'install', '--dry-run', '--only-binary=:all:',
            '-r', (Join-Path $repoRoot 'requirements-route-batch-tabpfn.txt'),
            '--report', (Join-Path $tabpfnReports 'dependency-dry-run.json')
        )
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-m', 'pip', 'install', '--only-binary=:all:',
            '-r', (Join-Path $repoRoot 'requirements-route-batch-tabpfn.txt'),
            '--report', (Join-Path $tabpfnReports 'pip-install-report.json')
        )
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--editable', $repoRoot,
            '--report', (Join-Path $tabpfnReports 'project-install-report.json')
        )
        Save-RouteEnvironment -Interpreter $tabpfnPython -ReportDirectory $tabpfnReports
        $probe = @'
import importlib.metadata as metadata
import json
import pathlib
import sys
import torch
from ids.route_batch.foundation import _tabpfn_class, TABPFN_CHECKPOINT, TABPFN_CHECKPOINT_SHA256
classifier, model_version, _, _ = _tabpfn_class()
assert str(model_version.value) == 'v2.5'
assert torch.__version__ == '2.6.0+cu126', torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable in the optional TabPFN environment'
report = {
    'stage': 'optional_tabpfn_dependency_import_only', 'python_executable': sys.executable,
    'python': sys.version, 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
    'packages': {name: metadata.version(name) for name in ('tabpfn', 'skrub', 'scikit-learn', 'numpy',
        'pydantic', 'pydantic-settings', 'safetensors', 'lightgbm')},
    'model_version': model_version.value, 'checkpoint_filename': TABPFN_CHECKPOINT,
    'checkpoint_sha256': TABPFN_CHECKPOINT_SHA256,
    'pretrained_weight_downloads': 0, 'real_model_loads': 0,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, indent=2))
'@
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-c', $probe, (Join-Path $tabpfnReports 'environment.json')
        )
        Invoke-RouteCommand -Executable $tabpfnPython -ArgumentList @(
            '-m', 'pytest', '-q', 'tests/test_route_batch_foundation.py',
            '--junitxml', (Join-Path $tabpfnReports 'foundation-synthetic-tests.xml')
        )
        $summary.tabpfn_status = 'dependencies_ready_checkpoint_still_required'
    }
    $summary.status = 'ready'
    Write-Output "Core interpreter: $corePython"
    Write-Output "Optional TabPFN interpreter: $tabpfnPython ($($summary.tabpfn_status))"
    Write-Output "Setup reports: $checkPath"
    Write-Output 'This setup does not download weights or accept model license terms.'
} catch {
    $summary.status = 'failed'
    $summary.error = $_.Exception.Message
    throw
} finally {
    $summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $checkPath 'setup-summary.json') -Encoding UTF8
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
