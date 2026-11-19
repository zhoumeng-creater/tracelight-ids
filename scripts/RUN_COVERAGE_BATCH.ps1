param(
    [switch]$InputsOnly,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$registration = Join-Path $repoRoot "docs/coverage_basis_20261119/batch/registration.json"
if (-not (Test-Path -LiteralPath $registration)) {
    throw "The complete source-coverage batch has not been materialized in this checkout."
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $suffix = if ($InputsOnly) { "inputs" } else { "four-comparisons" }
    $Output = Join-Path $repoRoot "results/coverage-basis-20261119-$suffix"
}
$runnerArgs = @(
    (Join-Path $PSScriptRoot "coverage_batch/run.py"),
    "--registration", $registration,
    "--output", $Output
)
if ($InputsOnly) { $runnerArgs += "--inputs-only" }
& python @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "Source-coverage batch stopped with exit code $LASTEXITCODE" }
if (-not $InputsOnly) {
    $archivePath = "$Output.zip"
    if (Test-Path -LiteralPath $archivePath) { throw "Results completed; preserve the existing archive: $archivePath" }
    Compress-Archive -LiteralPath $Output -DestinationPath $archivePath
    Write-Output "All four comparisons completed. Return the full archive: $archivePath"
}
