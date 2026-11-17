[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$R29Run,
    [string]$R26Run,
    [string]$RawDir,
    [Alias('OutputRoot')][string]$OutputDir,
    [ValidateRange(1,64)][int]$Threads = 4,
    [string]$Device = 'cuda:0',
    [switch]$PlanOnly,
    [switch]$CheckOnly,
    [switch]$InputsOnly,
    [switch]$VerifyOnly,
    [switch]$PackOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$modes = [int]$PlanOnly.IsPresent + [int]$CheckOnly.IsPresent + [int]$InputsOnly.IsPresent +
    [int]$VerifyOnly.IsPresent + [int]$PackOnly.IsPresent
if ($modes -gt 1) { throw 'Choose exactly one Only mode, or omit them for the registered run.' }
$registration = Get-Content -LiteralPath (Join-Path $repo 'configs/model-training-redesign-20261117/registration.json') -Raw | ConvertFrom-Json
if ($PlanOnly) {
    $design = Get-Content -LiteralPath (Join-Path $repo $registration.design_file) -Raw | ConvertFrom-Json
    [ordered]@{
        version = $registration.version
        max_fits = 6
        final_refits = 0
        arms = $design.arms
        seeds = $design.training.seeds
        selection = $design.selection
        execution = 'F/V only; selected checkpoint is final; freeze all six before C/E; no download'
        model_execution = $false
    } | ConvertTo-Json -Depth 20
    return
}
if ($CheckOnly -and -not $PSBoundParameters.ContainsKey('Python')) {
    throw '-CheckOnly requires an explicit -Python path.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -and
    -not (Get-Command -Name $Python -CommandType Application,Function -ErrorAction SilentlyContinue)) {
    throw "Python was not found: $Python. Use the existing compatible environment; no packages are installed by this entry."
}
if (-not $OutputDir) { $OutputDir = Join-Path $repo 'local/model-training-redesign-r30' }
$needsInputs = -not ($CheckOnly -or $VerifyOnly -or $PackOnly)
if ($needsInputs) {
    if (-not $R29Run -or -not $RawDir) { throw 'Pass -R29Run and -RawDir with the retained R29 return and TQH ZIP directory.' }
    foreach ($inputFolder in @($R29Run, $RawDir)) {
        if (-not (Test-Path -LiteralPath $inputFolder -PathType Container)) { throw "Input directory missing: $inputFolder" }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $R29Run 'protocol.json') -PathType Leaf)) {
        throw '-R29Run must be the directory containing R29 protocol.json, inputs and reports.'
    }
    if ($R26Run -and -not (Test-Path -LiteralPath $R26Run -PathType Container)) { throw "Optional R26 input missing: $R26Run" }
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
    if ($CheckOnly) {
        & $Python -c "import importlib.metadata as m; import torch, tabm, rtdl_num_embeddings; assert m.version('tabm') == '0.0.3'; assert m.version('rtdl_num_embeddings') == '0.0.12'; print('Pinned author packages available; synthetic CPU checks only')"
        if ($LASTEXITCODE -ne 0) { throw 'Pinned author dependencies are missing; checks did not run.' }
        $checks = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_model_training_redesign*.py' -File |
            Sort-Object Name | ForEach-Object FullName)
        $arguments = @('-m', 'pytest', '-q') + $checks
    } else {
        $arguments = @('-m', 'ids.model_training_redesign', '--output-dir', $OutputDir,
            '--threads', "$Threads", '--device', $Device)
        if ($needsInputs) {
            $arguments += @('--r29-run', $R29Run, '--raw-dir', $RawDir)
            if ($R26Run) { $arguments += @('--r26-run', $R26Run) }
        }
        if ($InputsOnly) { $arguments += '--inputs-only' }
        if ($VerifyOnly) { $arguments += '--verify-only' }
        if ($PackOnly) { $arguments += '--pack-only' }
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Redesign entry exited with code $LASTEXITCODE. Preserve the output and failure evidence; incomplete fits are not automatically retried."
    }
} finally { $env:PYTHONPATH = $previousPythonPath }
