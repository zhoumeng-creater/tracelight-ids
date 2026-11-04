param(
    [string]$LabRoot = 'D:\zhoumeng\Graduate',
    [string]$V3Root = '',
    [string]$UnswData = '',
    [string]$CicidsData = '',
    [ValidateSet('screen', 'pilot', 'full')][string]$Profile = 'screen',
    [switch]$DryRun,
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $LabRoot 'venv\Scripts\python.exe'
if (-not $V3Root) { $V3Root = Join-Path $LabRoot 'experiments\risk-v3-full-20261103' }
if (-not $UnswData) { $UnswData = Join-Path $LabRoot 'data\prepared-v1' }
if (-not $CicidsData) { $CicidsData = Join-Path $LabRoot 'experiments\stage4-stable-selection\data\cicids-prepared' }
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw "Existing lab Python is missing: $py" }
$actualHead = & git -C $repo rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Invalid research checkout.' }
$dirty = & git -C $repo status --porcelain
if ($LASTEXITCODE -ne 0 -or $dirty) {
    throw 'Use the clean reviewed checkout containing this script; current work is preserved.'
}
Set-Location -LiteralPath $repo
$env:PYTHONPATH = Join-Path $repo 'src'
$kind = if ($DryRun) { 'check-' } else { '' }
$work = Join-Path $LabRoot ('experiments\budget-transfer-' + $Profile + '-' + $kind + $actualHead.Substring(0, 8))
$labArgs = @(
    'scripts/run_budget_transfer_lab.py',
    '--unsw-v3-dir', (Join-Path $V3Root 'risk-v3-unsw'),
    '--unsw-data-dir', $UnswData,
    '--cicids-v3-dir', (Join-Path $V3Root 'risk-v3-cicids'),
    '--cicids-data-dir', $CicidsData,
    '--work-dir', $work, '--profile', $Profile
)
if ($DryRun) { $labArgs += '--dry-run' }
if ($Resume) { $labArgs += '--resume' }
Write-Host "Pinned commit: $actualHead"
Write-Host "Profile: $Profile. Screen stops at F/R/C; no automatic KD matrix or V evaluation."
if ($Profile -ne 'screen') { Write-Host 'Historical rank-transfer matrix: retained for explicit comparison, no longer the default recommendation.' }
Write-Host "Output: $work"
& $py @labArgs
if ($LASTEXITCODE -ne 0) { throw 'Study stopped; keep the work directory and the terminal error. Do not overwrite.' }
if ($DryRun) {
    Write-Host 'Metadata preflight passed; no models fitted. Run again without -DryRun to use a separate formal work directory.'
} else {
    Write-Host ('Return ALL SIX files under: ' + (Join-Path $work 'packages'))
}
