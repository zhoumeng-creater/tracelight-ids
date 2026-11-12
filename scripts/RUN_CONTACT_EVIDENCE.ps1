[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$SourceDir,
    [string]$OutputRoot,
    [ValidateRange(0,64)][int]$Threads = 0,
    [ValidateRange(1,4)][int]$PrepareWorkers = 2,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Cannot find the laboratory route-core Python: $Python. Pass -Python with its actual path." }
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\contact-r22' }
$oldPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $files = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -File -Filter 'test_contact_*.py' | Sort-Object Name | ForEach-Object FullName)
        & $Python -m pytest -q @files
    } else {
        $arguments = @('-m','ids.contact_evidence','--output',$OutputRoot,'--prepare-workers',"$PrepareWorkers")
        if ($SourceDir) { $arguments += @('--source-dir',$SourceDir) }
        if ($Threads) { $arguments += @('--threads',"$Threads") }
        & $Python @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Contact study exited with code $LASTEXITCODE." }
} finally { $env:PYTHONPATH = $oldPath }
