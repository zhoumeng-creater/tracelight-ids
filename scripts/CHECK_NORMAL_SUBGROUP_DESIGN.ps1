[CmdletBinding(DefaultParameterSetName='Check')]
param(
    [Parameter(Mandatory=$true)] [string]$Python,
    [Parameter(Mandatory=$true, ParameterSetName='Check')] [switch]$CheckOnly,
    [Parameter(Mandatory=$true, ParameterSetName='Replay')] [switch]$ZeroUpdate,
    [Parameter(Mandatory=$true, ParameterSetName='Replay')] [string]$R30Archive,
    [Parameter(Mandatory=$true, ParameterSetName='Replay')] [string]$OutputDir
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $repo
try {
    $env:PYTHONPATH = Join-Path $repo 'src'
    if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
    if ($CheckOnly) {
        & $Python -m pytest tests/test_normal_subgroup_targets.py tests/test_normal_subgroup_losses.py tests/test_normal_subgroup_selection.py tests/test_normal_subgroup_preflight.py -q
    } else {
        & $Python -m ids.normal_subgroup_training --preflight --r30-archive $R30Archive --output $OutputDir
    }
    if ($LASTEXITCODE -ne 0) { throw "Normal subgroup check failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
