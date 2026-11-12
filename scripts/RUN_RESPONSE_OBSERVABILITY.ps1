[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$SourceDir,
    [string]$OutputRoot,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateRange(1,4)][int]$PrepareWorkers = 2,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing route-core interpreter: $Python. Pass -Python with its full path." }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\response-r23' }
$oldPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $checks = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -File -Filter 'test_response_*.py' | Sort-Object Name | ForEach-Object FullName)
        & $Python -m pytest -q -p no:cacheprovider @checks
    } else {
        $arguments = @('-m','ids.response_evidence','--output',$OutputRoot,'--prepare-workers',"$PrepareWorkers")
        if ($SourceDir) { $arguments += @('--source-dir',$SourceDir) }
        if ($Threads) { $arguments += @('--threads',"$Threads") }
        & $Python @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Round 23 exited with code $LASTEXITCODE." }
} finally { $env:PYTHONPATH = $oldPath }
