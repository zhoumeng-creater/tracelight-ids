[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$R26Run,
    [string]$OutputRoot,
    [string]$RecoveredTraining,
    [ValidateRange(1,64)][int]$Threads = 4,
    [switch]$CheckOnly,
    [switch]$PlanOnly,
    [switch]$InputsOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (([int]$CheckOnly.IsPresent + [int]$PlanOnly.IsPresent + [int]$InputsOnly.IsPresent) -gt 1) {
    throw 'Choose only one of -CheckOnly, -PlanOnly, or -InputsOnly.'
}
$registrationPath = Join-Path $repo 'configs\training-diagnosis-20261116\registration.json'
$registration = Get-Content -LiteralPath $registrationPath -Raw | ConvertFrom-Json
if ($PlanOnly) {
    $registration | ConvertTo-Json -Depth 32
    return
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Pass -Python with the existing compatible Python path: $Python"
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\training-diagnosis-r28' }
if (-not $CheckOnly -and -not $R26Run) {
    $parent = Split-Path -Parent $repo
    $candidates = @(
        (Join-Path $parent 'tracelight-support-diagnostic\local\support-diagnostic-r26'),
        (Join-Path $parent 'task-reset-20261115\local\support-diagnostic-r26'),
        (Join-Path $parent 'task-reset\local\support-diagnostic-r26'),
        (Join-Path $repo '.worktrees\task-reset-20261115\local\support-diagnostic-r26')
    ) | Select-Object -Unique
    $expectedProtocol = $registration.source_artifacts.PSObject.Properties['protocol.json'].Value
    $verified = @()
    foreach ($candidate in $candidates) {
        $protocolPath = Join-Path $candidate 'protocol.json'
        if ((Test-Path -LiteralPath $protocolPath -PathType Leaf) -and
            (Get-FileHash -LiteralPath $protocolPath -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expectedProtocol) {
            $verified += (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $verified = @($verified | Select-Object -Unique)
    if ($verified.Count -ne 1) {
        throw "Pass -R26Run pointing to the retained R26 run. The small adjacent-path search found $($verified.Count) matching runs; it did not choose a first match or scan other directories."
    }
    $R26Run = $verified[0]
    Write-Host "Using registered R26 run: $R26Run"
}
if (-not $CheckOnly) {
    if (-not (Test-Path -LiteralPath (Join-Path $R26Run 'protocol.json') -PathType Leaf)) {
        throw "R26 protocol.json is missing: $R26Run. Pass the retained run or extracted R26 return directory."
    }
    $trainingPath = if ($RecoveredTraining) { $RecoveredTraining } else { Join-Path $R26Run 'local\training\mixed.parquet' }
    if (-not (Test-Path -LiteralPath $trainingPath -PathType Leaf)) {
        throw "Training numeric rows are missing: $trainingPath. Use the retained R26 local/training/mixed.parquet, or run scripts/recover_r26_training_rows.py on retained raw archives and pass -RecoveredTraining. Do not rerun R26 training."
    }
}
$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $tests = @(
            (Join-Path $repo 'tests\test_training_diagnosis.py'),
            (Join-Path $repo 'tests\test_recover_r26_training_rows.py')
        )
        $tests += @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_support_diagnostic_*.py' -File |
            Sort-Object Name | ForEach-Object FullName)
        $arguments = @('-m', 'pytest', '-q') + $tests
    } else {
        $arguments = @('-m', 'ids.support_diagnostic.training_diagnosis', '--r26-run', $R26Run,
            '--output', $OutputRoot, '--threads', "$Threads")
        if ($RecoveredTraining) { $arguments += @('--recovered-training', $RecoveredTraining) }
        if ($InputsOnly) { $arguments += '--inputs-only' }
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training diagnosis exited with code $LASTEXITCODE. Keep the failure report and completed scoring receipts; fix the reported input or environment issue before resuming."
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
