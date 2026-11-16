[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$R26Run,
    [Alias('OutputRoot')][string]$OutputDir,
    [string]$RecoveredTraining,
    [string]$RawDir,
    [ValidateRange(2,64)][int]$Threads = 4,
    [string]$Device = 'cuda:0',
    [switch]$CheckOnly,
    [switch]$PlanOnly,
    [switch]$InputsOnly,
    [switch]$VerifyOnly,
    [switch]$PackOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$modeCount = [int]$CheckOnly.IsPresent + [int]$PlanOnly.IsPresent + [int]$InputsOnly.IsPresent +
    [int]$VerifyOnly.IsPresent + [int]$PackOnly.IsPresent
if ($modeCount -gt 1) {
    throw 'Choose only one of -CheckOnly, -PlanOnly, -InputsOnly, -VerifyOnly, or -PackOnly.'
}
$registrationPath = Join-Path $repo 'configs\model-training-20261116\registration.json'
$registration = Get-Content -LiteralPath $registrationPath -Raw | ConvertFrom-Json
if ($PlanOnly) {
    $design = Get-Content -LiteralPath (Join-Path $repo $registration.design_file) -Raw | ConvertFrom-Json
    [ordered]@{
        version = $registration.version
        registration = $registration
        roles = $design.roles
        training = $design.training
        calibration = $design.calibration
        execution = '7 development trajectories + 7 final refits; freeze all final models before C/E/D; no download'
    } | ConvertTo-Json -Depth 40
    return
}
if ($CheckOnly -and -not $PSBoundParameters.ContainsKey('Python')) {
    throw '-CheckOnly requires an explicit -Python path to the environment used for checking.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -and
    -not (Get-Command -Name $Python -CommandType Application,Function -ErrorAction SilentlyContinue)) {
    throw "Pass -Python with the existing compatible Python path: $Python. This wrapper does not install or upgrade packages."
}
if (-not $OutputDir) { $OutputDir = Join-Path $repo 'local\model-training-r29' }
function Get-ModelTrainingSha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $sha256 = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $sha256.Dispose(); $stream.Dispose() }
}
$needsInputs = -not ($CheckOnly -or $VerifyOnly -or $PackOnly)
if ($needsInputs -and -not $R26Run) {
    $parent = Split-Path -Parent $repo
    $candidates = @(
        (Join-Path $parent 'tracelight-support-diagnostic\local\support-diagnostic-r26'),
        (Join-Path $parent 'task-reset-20261115\local\support-diagnostic-r26'),
        (Join-Path $parent 'task-reset\local\support-diagnostic-r26'),
        (Join-Path $repo '.worktrees\task-reset-20261115\local\support-diagnostic-r26')
    ) | Select-Object -Unique
    $verified = @()
    foreach ($candidate in $candidates) {
        $protocolPath = Join-Path $candidate 'protocol.json'
        if ((Test-Path -LiteralPath $protocolPath -PathType Leaf) -and
            (Get-ModelTrainingSha256 $protocolPath) -eq $registration.r26_protocol_sha256) {
            $verified += (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $verified = @($verified | Select-Object -Unique)
    if ($verified.Count -ne 1) {
        throw "Pass -R26Run pointing to the retained R26 run or extracted R26 return. The adjacent-path search found $($verified.Count) matching runs; no first-match choice or recursive search was made."
    }
    $R26Run = $verified[0]
    Write-Host "Using registered R26 run: $R26Run"
}
if ($needsInputs) {
    $protocolPath = Join-Path $R26Run 'protocol.json'
    if (-not (Test-Path -LiteralPath $protocolPath -PathType Leaf)) {
        throw "R26 protocol.json is missing: $R26Run. Pass -R26Run with the retained run or extracted R26 return."
    }
    if ((Get-ModelTrainingSha256 $protocolPath) -ne $registration.r26_protocol_sha256) {
        throw "R26 protocol hash differs from this registration: $protocolPath"
    }
    $originalTraining = Join-Path $R26Run 'local\training\mixed.parquet'
    if (Test-Path -LiteralPath $originalTraining -PathType Leaf) {
        $RecoveredTraining = $null
    } elseif (-not $RecoveredTraining) {
        $recoveryCandidates = @(
            (Join-Path $repo 'model-training-inputs\recovered-training.parquet'),
            (Join-Path (Split-Path -Parent $repo) 'input\recovered-training.parquet')
        ) | Select-Object -Unique
        $available = @($recoveryCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
        if ($available.Count -gt 1) {
            throw 'Multiple packaged training frames exist. Pass -RecoveredTraining explicitly.'
        }
        if ($available.Count -eq 1) { $RecoveredTraining = $available[0] }
    }
    if ($RecoveredTraining -and -not (Test-Path -LiteralPath $RecoveredTraining -PathType Leaf)) {
        throw "Recovered training frame is missing: $RecoveredTraining. Keep its matching .json receipt beside it."
    }
    if ($RawDir -and -not (Test-Path -LiteralPath $RawDir -PathType Container)) {
        throw "Retained raw-archive directory is missing: $RawDir. No download is performed."
    }
    if (-not (Test-Path -LiteralPath $originalTraining -PathType Leaf) -and -not $RecoveredTraining -and -not $RawDir) {
        throw 'Training numeric rows are missing. Pass -RecoveredTraining with the packaged parquet and adjacent .json receipt, or -RawDir with retained archives for scripts/recover_r26_training_rows.py. Do not rerun R26 training.'
    }
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $tests = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_model_training*.py' -File |
            Sort-Object Name | ForEach-Object FullName)
        $tests += (Join-Path $repo 'tests\test_recover_r26_training_rows.py')
        $arguments = @('-m', 'pytest', '-q') + $tests
    } else {
        $arguments = @('-m', 'ids.model_training', '--output-dir', $OutputDir, '--threads', "$Threads", '--device', $Device)
        if ($needsInputs) {
            $arguments += @('--r26-run', $R26Run)
            if ($RecoveredTraining) { $arguments += @('--recovered-training', $RecoveredTraining) }
            if ($RawDir) { $arguments += @('--raw-dir', $RawDir) }
        }
        if ($InputsOnly) { $arguments += '--inputs-only' }
        if ($VerifyOnly) { $arguments += '--verify-only' }
        if ($PackOnly) { $arguments += '--pack-only' }
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Model training entry exited with code $LASTEXITCODE. Keep the output directory and failure receipt. Completed fits are reused; an incomplete fit must not be restarted automatically."
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
