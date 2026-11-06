param(
    [string]$LabRoot = 'D:\zhoumeng\Graduate',
    [string]$ScreenDir = '',
    [string]$Round10Dir = '',
    [string]$UnswData = '',
    [string]$WorkDir = '',
    [switch]$PreflightOnly,
    [string]$PackageOnly = ''
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path $LabRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "Existing lab Python not found: $py"
}
$env:PYTHONPATH = Join-Path $repo 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$labArgs = @((Join-Path $PSScriptRoot 'run_unsw_kd_minibatch_lab.py'), '--lab-root', $LabRoot)

if ($PackageOnly) {
    if ($ScreenDir -or $Round10Dir -or $UnswData -or $WorkDir -or $PreflightOnly) {
        throw '-PackageOnly cannot be combined with run options.'
    }
    $labArgs += @('--package-only', $PackageOnly)
} else {
    $actualHead = & git -C $repo rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the checkout commit.' }
    $dirty = & git -C $repo status --porcelain
    if ($LASTEXITCODE -ne 0 -or $dirty) {
        throw 'Use a clean committed checkout. Existing work is preserved.'
    }
    if ($ScreenDir) { $labArgs += @('--screen-dir', $ScreenDir) }
    if ($Round10Dir) { $labArgs += @('--round10-dir', $Round10Dir) }
    if ($UnswData) { $labArgs += @('--data-dir', $UnswData) }
    if ($WorkDir) { $labArgs += @('--work-dir', $WorkDir) }
    if ($PreflightOnly) {
        $labArgs += '--preflight-only'
        Write-Host "Frozen H/K mini-batch preflight; commit=$actualHead; fits/updates=0"
    } else {
        Write-Host "Frozen H/K mini-batch; commit=$actualHead; student fits=2; teacher fits/queries=0; updates=6720"
    }
}

& $py @labArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Stopped. Preserve the printed work directory and failure package. No resume or automatic restart.'
}
