[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$Output = '',
    [switch]$Resume,
    [switch]$InputsOnly,
    [string]$CoverageResults = ''
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Output) {
    $suffix = if ($InputsOnly) { 'inputs' } else { 'results' }
    $Output = Join-Path $repo "results/comprehensive-20261122-$suffix"
}
$runnerArgs = @('-X', 'utf8', '-u', (Join-Path $PSScriptRoot 'comprehensive_batch/run.py'), '--output', [IO.Path]::GetFullPath($Output))
if ($Resume) { $runnerArgs += '--resume' }
if ($InputsOnly) { $runnerArgs += '--inputs-only' }
if ($CoverageResults) { $runnerArgs += @('--coverage-results', [IO.Path]::GetFullPath($CoverageResults)) }
& $Python @runnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Batch interrupted. Preserve this output. Resolve the reported interruption and use the same command with -Resume; do not create a fresh training batch."
}
if (-not $InputsOnly) {
    Write-Output "Return only $Output.zip and $Output.zip.sha256.json. Keep recovery states at the laboratory. Do not rebuild TraceLight-ALL."
}
