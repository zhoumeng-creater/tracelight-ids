[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$SourceDir,
    [string]$OutputRoot,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateRange(1,6)][int]$PrepareWorkers = 2,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\forward-threshold-r21' }
$old = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $tests = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_threshold_*.py' -File | ForEach-Object FullName)
        & $Python -m pytest -q @tests
    } else {
        $arguments = @('-m','ids.threshold_validation','--output',$OutputRoot,'--prepare-workers',"$PrepareWorkers")
        if ($SourceDir) { $arguments += @('--source-dir',$SourceDir) }
        if ($Threads) { $arguments += @('--threads',"$Threads") }
        & $Python @arguments
    }
    if ($LASTEXITCODE) { throw "Forward validation exited with code $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $old }
