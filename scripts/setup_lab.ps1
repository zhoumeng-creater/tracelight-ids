[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$VenvDir,
    [string]$Python = 'python',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvPath = [IO.Path]::GetFullPath($VenvDir)
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$ArgumentList)
    & $Executable @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable"
    }
}

Invoke-Checked -Executable $Python -ArgumentList @('-c', 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the laboratory recipe"')
if (Test-Path -LiteralPath $venvPath) {
    if (-not (Test-Path -LiteralPath (Join-Path $venvPath 'pyvenv.cfg')) -or -not (Test-Path -LiteralPath $venvPython)) {
        throw 'VenvDir exists but is not a Windows virtual environment. Choose a new directory.'
    }
} else {
    Invoke-Checked -Executable $Python -ArgumentList @('-m', 'venv', $venvPath)
}
Invoke-Checked -Executable $venvPython -ArgumentList @('-c', 'import sys; assert sys.version_info[:2] == (3, 12), "Existing virtual environment is not Python 3.12"')
Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'install', '-r', (Join-Path $repoRoot 'requirements-lock.txt'))
Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'install', '-e', $repoRoot, '--no-deps')
$torchIndex = if ($Device -eq 'cuda') { 'https://download.pytorch.org/whl/cu126' } else { 'https://download.pytorch.org/whl/cpu' }
$torchVersion = if ($Device -eq 'cuda') { 'torch==2.6.0+cu126' } else { 'torch==2.6.0+cpu' }
Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'install', $torchVersion, '--index-url', $torchIndex)
Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pip', 'check')
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$doctorArguments = @('-m', 'ids', 'doctor')
if ($Device -eq 'cuda') { $doctorArguments += '--require-cuda' }
Invoke-Checked -Executable $venvPython -ArgumentList $doctorArguments
if (-not $SkipTests) {
    Invoke-Checked -Executable $venvPython -ArgumentList @('-m', 'pytest', (Join-Path $repoRoot 'tests'), '-q')
}
Write-Output "Environment ready: $venvPython"
Write-Output 'No dataset was downloaded and no research experiment was started.'
