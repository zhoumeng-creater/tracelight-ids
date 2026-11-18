[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$R30Archive,
    [string]$OutputDir,
    [ValidateRange(1,1)][int]$Threads = 1,
    [string]$Device = 'cuda:0',
    [switch]$Run,
    [switch]$PrepareOnly,
    [switch]$CheckOnly,
    [switch]$PlanOnly,
    [switch]$VerifyOnly,
    [switch]$PackOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$modes = [int]$Run.IsPresent + [int]$PrepareOnly.IsPresent + [int]$CheckOnly.IsPresent +
    [int]$PlanOnly.IsPresent + [int]$VerifyOnly.IsPresent + [int]$PackOnly.IsPresent
if ($modes -ne 1) { throw 'Choose one explicit mode: Run, PrepareOnly, CheckOnly, PlanOnly, VerifyOnly or PackOnly.' }
if ($PlanOnly) {
    Get-Content -LiteralPath (Join-Path $repo 'configs/normal-subgroup-lab-20261118/registration.json') -Raw
    return
}
if ($CheckOnly -and -not $PSBoundParameters.ContainsKey('Python')) { throw '-CheckOnly requires an explicit -Python.' }
if (-not $CheckOnly -and -not $OutputDir) { throw 'Pass an explicit -OutputDir inside the project experiment directory.' }
if (($Run -or $PrepareOnly) -and -not $R30Archive) { throw '-R30Archive must name the retained R30 return root.' }
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $repo
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
    if ($CheckOnly) {
        $checks = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_normal_subgroup_*.py' -File |
            Sort-Object Name | ForEach-Object FullName)
        & $Python -m pytest -q @checks
    } else {
        $arguments = @('-m', 'ids.normal_subgroup_training', '--output', $OutputDir,
            '--threads', "$Threads", '--device', $Device)
        if ($Run) { $arguments += '--run' }
        if ($PrepareOnly) { $arguments += '--prepare-only' }
        if ($VerifyOnly) { $arguments += '--verify-only' }
        if ($PackOnly) { $arguments += '--pack-only' }
        if ($Run -or $PrepareOnly) { $arguments += @('--r30-archive', $R30Archive) }
        & $Python @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Normal subgroup entry failed ($LASTEXITCODE). Preserve this output; do not restart an incomplete fit in another directory." }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
