param(
    [string]$LabRoot = 'D:\zhoumeng\Graduate',
    [string]$ScreenDir = '',
    [string]$UnswData = '',
    [string]$WorkDir = '',
    [string]$FrozenTargets = '',
    [switch]$PreflightOnly,
    [string]$PackageOnly = ''
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $LabRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "Existing lab Python is missing: $py"
}
$env:PYTHONPATH = Join-Path $repo 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$labArgs = @((Join-Path $PSScriptRoot 'run_unsw_kd_lab.py'), '--lab-root', $LabRoot)
if ($PackageOnly) {
    if ($ScreenDir -or $UnswData -or $WorkDir -or $FrozenTargets -or $PreflightOnly) {
        throw 'PackageOnly cannot be combined with run options.'
    }
    $labArgs += @('--package-only', $PackageOnly)
} else {
    $actualHead = & git -C $repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Invalid research checkout.' }
    $dirty = & git -C $repo status --porcelain
    if ($LASTEXITCODE -ne 0 -or $dirty) {
        throw 'Use the clean reviewed checkout. Current work is preserved.'
    }
    if ($ScreenDir) { $labArgs += @('--screen-dir', $ScreenDir) }
    if ($UnswData) { $labArgs += @('--data-dir', $UnswData) }
    if ($WorkDir) { $labArgs += @('--work-dir', $WorkDir) }
    if ($FrozenTargets) { $labArgs += @('--frozen-targets', $FrozenTargets) }
    if ($PreflightOnly) { $labArgs += '--preflight-only' }
    Write-Host "Code commit: $actualHead"
    Write-Host 'UNSW only; H/K/S/M; 4 continuations x 40 full-F updates; no new teacher/warmup.'
}
& $py @labArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Stopped. Preserve the work directory and return the failure ZIP/SHA if printed. Do not restart training automatically.'
}
