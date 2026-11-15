[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$ReferenceRun,
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$OutputRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\r26-risk-audit' }
$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    & $Python -m ids.risk_transfer.audit --reference-run $ReferenceRun --output $OutputRoot
    if ($LASTEXITCODE -ne 0) { throw "R26 risk audit failed with exit code $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $previousPath }
