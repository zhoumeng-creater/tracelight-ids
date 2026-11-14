[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$CacheDir,
    [string]$OutputRoot,
    [ValidateRange(0,64)][int]$Threads = 0,
    [switch]$CheckOnly,
    [switch]$PlanOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Missing route-core interpreter. Pass -Python with its full path: $Python"
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\behavior-transfer-r25' }
$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $checks = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -File -Filter 'test_behavior_*.py' | Sort-Object Name | ForEach-Object FullName)
        & $Python -m pytest -q -p no:cacheprovider @checks
    } else {
        $arguments = @('-m','ids.behavior_transfer','--output',$OutputRoot)
        if ($CacheDir) { $arguments += @('--cache',$CacheDir) }
        if ($Threads) { $arguments += @('--threads',"$Threads") }
        if ($PlanOnly) { $arguments += '--plan' }
        & $Python @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Round 25 exited with code $LASTEXITCODE. See the output directory for an incomplete package." }
} finally { $env:PYTHONPATH = $previousPath }
