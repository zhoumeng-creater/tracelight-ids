[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$PreviousRoot = 'D:\zhoumeng\Graduate\tracelight-behavior-transfer\local\behavior-transfer-r25',
    [string]$OutputRoot,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateRange(0,8)][int]$Workers = 0,
    [ValidateRange(0,1024)][double]$MemoryGiB = 0,
    [switch]$CheckOnly,
    [switch]$PlanOnly,
    [switch]$PreflightOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing route-core interpreter. Pass -Python with its full path: $Python"
}
if (([int]$CheckOnly.IsPresent + [int]$PlanOnly.IsPresent + [int]$PreflightOnly.IsPresent) -gt 1) {
    throw 'Choose only one of CheckOnly, PlanOnly and PreflightOnly.'
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\readiness-replay-r25' }
$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    $arguments = @('-m','ids.readiness_replay','--output',$OutputRoot)
    if ($Threads) { $arguments += @('--threads',"$Threads") }
    if ($Workers) { $arguments += @('--workers',"$Workers") }
    if ($MemoryGiB) { $arguments += @('--memory-gib',$MemoryGiB.ToString([Globalization.CultureInfo]::InvariantCulture)) }
    if ($PlanOnly) { $arguments += '--plan' }
    elseif ($CheckOnly) { $arguments += '--check' }
    else {
        $arguments += @('--previous',$PreviousRoot)
        if ($PreflightOnly) { $arguments += '--preflight-only' }
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Readiness replay exited with code $LASTEXITCODE. Inspect the incomplete package or preflight report." }
} finally { $env:PYTHONPATH = $previousPath }
