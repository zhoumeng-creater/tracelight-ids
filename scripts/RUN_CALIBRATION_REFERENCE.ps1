[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [Alias('OutputRoot')][string]$Output = '',
    [switch]$InputsOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf) -and
    -not (Get-Command -Name $Python -CommandType Application -ErrorAction SilentlyContinue)) {
    throw "Python was not found: $Python. Pass -Python with the existing project interpreter."
}
if (-not $Output) {
    $suffix = if ($InputsOnly) { 'inputs' } else { 'nine-fits' }
    $Output = Join-Path $repo "results/calibration-reference-20261122-$suffix"
}
$Output = [IO.Path]::GetFullPath($Output)
if (-not $InputsOnly -and (Test-Path -LiteralPath "$Output.zip")) {
    throw "Preserve the existing result archive: $Output.zip"
}
$runnerArgs = @('-u', (Join-Path $PSScriptRoot 'calibration_repair/run.py'), '--output', $Output)
if ($InputsOnly) { $runnerArgs += '--inputs-only' }
& $Python @runnerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Batch stopped with exit code $LASTEXITCODE. Preserve the output and failure.json; no automatic refit."
}
if (-not $InputsOnly) {
    Compress-Archive -LiteralPath $Output -DestinationPath "$Output.zip"
    Write-Output "Nine fixed fits and evaluation completed. Return the complete archive: $Output.zip"
}
