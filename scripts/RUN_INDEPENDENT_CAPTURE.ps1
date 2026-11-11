[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$OutputRoot,
    [string]$Source,
    [ValidateRange(0,64)][int]$Threads = 0,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\independent-capture-r20' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing route-core Python: $Python" }
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($CheckOnly) {
        $files = @(Get-ChildItem -LiteralPath (Join-Path $repo 'tests') -Filter 'test_capture_validation*.py' -File |
            Sort-Object Name | ForEach-Object FullName)
        & $Python -m pytest -q @files
    } else {
        $arguments = @('-m','ids.capture_validation','--output',$OutputRoot)
        if ($Source) { $arguments += @('--source',$Source) }
        if ($Threads -gt 0) { $arguments += @('--threads',"$Threads") }
        & $Python @arguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Independent capture exited with code $LASTEXITCODE" }
} finally { $env:PYTHONPATH = $oldPythonPath }
