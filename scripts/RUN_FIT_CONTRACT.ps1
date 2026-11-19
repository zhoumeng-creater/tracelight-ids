param(
    [switch]$InputsOnly,
    [int]$Threads = 4,
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$deliveryRoot = Join-Path $repoRoot "docs/fit_contract_20261119"
$dataRoot = Join-Path $deliveryRoot "dataset"
$registrationPath = Join-Path $deliveryRoot "fit_contract/registration.json"
$registration = Get-Content -LiteralPath $registrationPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $registration.data_materialized) { throw "The registered dataset has not been materialized." }
foreach ($item in $registration.runner_inputs) {
    $itemPath = Join-Path $dataRoot $item.name
    $actualHash = (Get-FileHash -LiteralPath $itemPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $item.sha256) { throw "Registered input changed: $($item.name)" }
}
if ([string]::IsNullOrWhiteSpace($Output)) {
    $suffix = if ($InputsOnly) { "inputs" } else { "comparison" }
    $Output = Join-Path $repoRoot "results/fit-contract-20261119-$suffix"
}
$runnerArgs = @(
    (Join-Path $PSScriptRoot "fit_contract/run_minimal_comparison.py"),
    "--model-rows", (Join-Path $dataRoot "model_rows.jsonl.gz"),
    "--inference-rows", (Join-Path $dataRoot "inference_rows.jsonl.gz"),
    "--events", (Join-Path $dataRoot "event_metadata.jsonl.gz"),
    "--normal-units", (Join-Path $dataRoot "normal_units.jsonl.gz"),
    "--output", $Output
)
if ($InputsOnly) { $runnerArgs += "--inputs-only" }
else { $runnerArgs += @("--threads", $Threads) }
& python @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "Fixed comparison command failed with exit code $LASTEXITCODE" }
