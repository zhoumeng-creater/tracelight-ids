[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$OriginalRun = 'D:\zhoumeng\Graduate\tracelight-route-batch\local\route-batch-20261108\run',
    [string]$OutputRoot,
    [string]$Source,
    [ValidateRange(0, 64)][int]$Threads = 0,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repo 'local\flow-history-r19'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing route-core Python: $Python" }
$oldPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $testFiles = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_flow_history*.py' -File |
            Sort-Object Name | ForEach-Object { $_.FullName })
        & $Python -m pytest -q @testFiles
        if ($LASTEXITCODE -ne 0) { throw "Synthetic flow-history checks failed (exit $LASTEXITCODE)" }
        return
    }
    $argsList = @('-m','ids.flow_history','--original-run',$OriginalRun,'--output',$OutputRoot)
    if ($Threads -gt 0) { $argsList += @('--threads',"$Threads") }
    if ($Source) { $argsList += @('--source',$Source) }
    & $Python @argsList
    if ($LASTEXITCODE -ne 0) { throw "Flow-history batch failed (exit $LASTEXITCODE); inspect the returned evidence." }
}
finally { $env:PYTHONPATH = $oldPath }
