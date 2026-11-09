[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CorePython,
    [Parameter(Mandatory = $true)][string]$TabPFNPython,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$savedPythonPath = [Environment]::GetEnvironmentVariable('PYTHONPATH', 'Process')
$savedNoUserSite = [Environment]::GetEnvironmentVariable('PYTHONNOUSERSITE', 'Process')
try {
    $env:PYTHONPATH = Join-Path $repoRoot 'src'
    $env:PYTHONNOUSERSITE = '1'
    $arguments = @('-X', 'utf8', '-m', 'ids.route_batch.tabpfn_prepare',
        '--core-python', $CorePython, '--tabpfn-python', $TabPFNPython,
        '--checkpoint', $Checkpoint, '--output-directory', $OutputDirectory)
    & $CorePython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "TabPFN preparation did not reach ready; inspect $OutputDirectory\prepare-report.json"
    }
    $report = Get-Content -LiteralPath (Join-Path $OutputDirectory 'prepare-report.json') -Raw | ConvertFrom-Json
    if ($report.status -ne 'ready') { throw 'Preparation returned without a ready report.' }
    Write-Output "TabPFN ready: $TabPFNPython"
} finally {
    [Environment]::SetEnvironmentVariable('PYTHONPATH', $savedPythonPath, 'Process')
    [Environment]::SetEnvironmentVariable('PYTHONNOUSERSITE', $savedNoUserSite, 'Process')
}
