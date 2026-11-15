[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$ReferenceRun,
    [string]$RawDir,
    [string]$OutputRoot,
    [ValidateRange(1,8)][int]$CpuWorkers = 2,
    [ValidateRange(1,2)][int]$GpuWorkers = 2,
    [ValidateRange(0,64)][int]$Threads = 0,
    [switch]$CheckOnly,
    [switch]$PlanOnly,
    [switch]$PreflightOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Pass -Python with the existing route-core Python path: $Python" }
if (([int]$CheckOnly.IsPresent + [int]$PlanOnly.IsPresent + [int]$PreflightOnly.IsPresent) -gt 1) { throw 'Choose only one check mode.' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\support-diagnostic-r26' }
$searchRoot = Split-Path -Parent $repo
if (-not $CheckOnly -and -not $PlanOnly -and -not $ReferenceRun) {
    $registry = Get-Content -LiteralPath (Join-Path $repo 'configs\response-evidence-20261112\r22-model-reference.json') -Raw | ConvertFrom-Json
    $matches = @(Get-ChildItem -LiteralPath $searchRoot -Filter 'hgb-raw.joblib' -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.Directory.Name -eq 'frozen' -and $_.Directory.Parent.Name -eq 'models' } | Sort-Object FullName)
    foreach ($candidate in $matches) {
        $candidateRoot = $candidate.Directory.Parent.Parent.FullName
        $valid = $true
        foreach ($group in @('hgb','mlp17','mlp29','mlp43')) {
            $extension = if ($group -eq 'hgb') { '.joblib' } else { '.pt' }
            $relative = "models/frozen/$group-raw$extension"
            $path = Join-Path $candidateRoot $relative
            $expected = $registry.artifacts.PSObject.Properties["$group-raw"].Value.PSObject.Properties[$relative].Value
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { $valid = $false; break }
            if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { $valid = $false; break }
        }
        if ($valid) { $ReferenceRun = $candidateRoot; break }
    }
    if (-not $ReferenceRun) { throw 'Pinned old raw models were not found beside this checkout. Pass -ReferenceRun pointing to the retained R22/R23 run that contains models/frozen. Do not rerun the old models.' }
    Write-Host "Reusing verified raw models: $ReferenceRun"
}
if (-not $CheckOnly -and -not $PlanOnly -and -not $RawDir) {
    $archives = @(Get-ChildItem -LiteralPath $searchRoot -Filter 'TQH-C2_labels_v110.zip' -File -Recurse -ErrorAction SilentlyContinue | Sort-Object FullName)
    $required = @('CTU-IoT-Malware-Capture-1-1.conn.log.labeled.gz','CTU-IoT-Malware-Capture-8-1.conn.log.labeled.gz','TQH-C2_labels_v110.zip','TQH-C2_labels_D.zip')
    foreach ($archive in $archives) {
        $candidateCache = $archive.Directory.FullName
        $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $candidateCache $_) -PathType Leaf) })
        if ($missing.Count -eq 0) { $RawDir = $candidateCache; break }
    }
    if ($RawDir) { Write-Host "Checking retained raw cache before use: $RawDir" }
}
$previousPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    $arguments = @('-m','ids.support_diagnostic','--output',$OutputRoot,'--cpu-workers',"$CpuWorkers",'--gpu-workers',"$GpuWorkers")
    if ($Threads) { $arguments += @('--threads',"$Threads") }
    if ($CheckOnly) { $arguments += '--check' }
    elseif ($PlanOnly) { $arguments += '--plan' }
    else {
        $arguments += @('--reference-run',$ReferenceRun)
        if ($RawDir) { $arguments += @('--raw-dir',$RawDir) }
        if ($PreflightOnly) { $arguments += '--preflight-only' }
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Support diagnostic exited with code $LASTEXITCODE. Keep the partial return package and resume the same command after fixing the reported cause." }
} finally { $env:PYTHONPATH = $previousPath }
