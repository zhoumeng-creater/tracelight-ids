param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Round14Root,
    [string]$Checkpoint,
    [string]$TabPFNCheckpoint,
    [string]$TabPFNPython,
    [string]$SourceZip,
    [string]$Round13Prepared,
    [switch]$Resume,
    [switch]$RetryUnavailable
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$outputPath = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python = [IO.Path]::GetFullPath($Python) }
if (-not [string]::IsNullOrWhiteSpace($Round14Root)) { $Round14Root = [IO.Path]::GetFullPath($Round14Root) }
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) { $Checkpoint = [IO.Path]::GetFullPath($Checkpoint) }
if (-not [string]::IsNullOrWhiteSpace($TabPFNCheckpoint)) { $TabPFNCheckpoint = [IO.Path]::GetFullPath($TabPFNCheckpoint) }
if (-not [string]::IsNullOrWhiteSpace($TabPFNPython)) { $TabPFNPython = [IO.Path]::GetFullPath($TabPFNPython) }
if (-not [string]::IsNullOrWhiteSpace($SourceZip)) { $SourceZip = [IO.Path]::GetFullPath($SourceZip) }
if (-not [string]::IsNullOrWhiteSpace($Round13Prepared)) { $Round13Prepared = [IO.Path]::GetFullPath($Round13Prepared) }
$configPath = Join-Path $outputPath 'config.json'
$runPath = Join-Path $outputPath 'run'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$archivePath = Join-Path $outputPath ('route-results-' + $stamp + '.zip')

function Invoke-RoutePython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code ${exitCode}: $($Arguments -join ' ')"
    }
}

function Assert-SameRegisteredPath {
    param([string]$Requested, [string]$Registered, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($Requested)) { return }
    if ([string]::IsNullOrWhiteSpace($Registered) -or
        [IO.Path]::GetFullPath($Requested) -ne [IO.Path]::GetFullPath($Registered)) {
        throw "$Name differs from the existing config.json. The registered configuration was not overwritten."
    }
}

if ($RetryUnavailable -and -not $Resume) {
    throw 'RetryUnavailable requires Resume.'
}
if ([string]::IsNullOrWhiteSpace($SourceZip) -ne [string]::IsNullOrWhiteSpace($Round13Prepared)) {
    throw 'SourceZip and Round13Prepared must be supplied together.'
}
if (-not [string]::IsNullOrWhiteSpace($SourceZip)) {
    if (-not (Test-Path -LiteralPath $SourceZip -PathType Leaf)) {
        throw 'SourceZip must be an existing file before the batch starts.'
    }
    if (-not (Test-Path -LiteralPath $Round13Prepared -PathType Container)) {
        throw 'Round13Prepared must be an existing prepared-data directory before the batch starts.'
    }
    foreach ($requiredName in @('support.csv', 'normal_calibration.csv', 'development_evaluation.csv', 'preparation.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Round13Prepared $requiredName) -PathType Leaf)) {
            throw "Round13Prepared is missing required file: $requiredName"
        }
    }
}
if ((Test-Path -LiteralPath $runPath) -and -not $Resume) {
    throw 'The run directory already exists. Use Resume for the unchanged registered batch.'
}
if (-not (Test-Path -LiteralPath $configPath)) {
    if ([string]::IsNullOrWhiteSpace($Round14Root) -or [string]::IsNullOrWhiteSpace($Checkpoint)) {
        throw 'Round14Root and Checkpoint are required for the first registration.'
    }
} else {
    $registered = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Assert-SameRegisteredPath -Requested $Round14Root -Registered $registered.round14_root -Name 'Round14Root'
    Assert-SameRegisteredPath -Requested $Checkpoint -Registered $registered.checkpoint.path -Name 'Checkpoint'
    Assert-SameRegisteredPath -Requested $TabPFNCheckpoint -Registered $registered.tabpfn_checkpoint.path -Name 'TabPFNCheckpoint'
    Assert-SameRegisteredPath -Requested $TabPFNPython -Registered $registered.tabpfn_python -Name 'TabPFNPython'
}

[IO.Directory]::CreateDirectory($outputPath) | Out-Null
$oldPythonPath = $env:PYTHONPATH
$oldEncoding = $env:PYTHONIOENCODING
$oldOmp = $env:OMP_NUM_THREADS
$oldMkl = $env:MKL_NUM_THREADS
$oldOpenBlas = $env:OPENBLAS_NUM_THREADS
$oldCublas = $env:CUBLAS_WORKSPACE_CONFIG
$sourcePaths = @((Join-Path $repoRoot 'src'), (Join-Path $repoRoot 'local\vendor\tabicl\src'))
if (-not [string]::IsNullOrWhiteSpace($oldPythonPath)) { $sourcePaths += $oldPythonPath }
$env:PYTHONPATH = $sourcePaths -join [IO.Path]::PathSeparator
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'

Push-Location -LiteralPath $repoRoot
try {
    if (-not (Test-Path -LiteralPath $configPath)) {
        $registrationArguments = @('-m', 'ids.route_batch', 'register', '--round14-root',
                                  $Round14Root, '--checkpoint', $Checkpoint, '--output', $configPath)
        if (-not [string]::IsNullOrWhiteSpace($TabPFNCheckpoint)) {
            $registrationArguments += @('--tabpfn-checkpoint', [IO.Path]::GetFullPath($TabPFNCheckpoint))
        }
        if (-not [string]::IsNullOrWhiteSpace($TabPFNPython)) {
            $registrationArguments += @('--tabpfn-python', [IO.Path]::GetFullPath($TabPFNPython))
        }
        Invoke-RoutePython -Arguments $registrationArguments
    }
    Invoke-RoutePython -Arguments @('-m', 'ids.route_batch', 'preflight', '--config', $configPath,
                                   '--output', (Join-Path $outputPath ('preflight-' + $stamp + '.json')))
    $runArguments = @('-m', 'ids.route_batch', 'run', '--config', $configPath, '--output', $runPath)
    if ($Resume) {
        Write-Output 'Task-level resume reuses completed tasks. An interrupted training trajectory preserves its previous attempt and restarts from the same seed; optimizer state is not resumed.'
        $runArguments += '--resume'
    }
    if ($RetryUnavailable) { $runArguments += '--retry-unavailable' }
    Invoke-RoutePython -Arguments $runArguments
    $confirmationError = $null
    if (-not [string]::IsNullOrWhiteSpace($SourceZip)) {
        $confirmationArguments = @('-m', 'ids.route_batch', 'confirm', '--run-dir', $runPath,
                                   '--source-zip', $SourceZip, '--round13-prepared', $Round13Prepared)
        if ($RetryUnavailable) {
            # Confirmation retries use the original model freeze, without adding newly available models.
            $confirmationArguments += '--retry-failed'
        }
        try {
            Invoke-RoutePython -Arguments $confirmationArguments
        } catch {
            $confirmationError = $_
            $errorPath = Join-Path $runPath ('confirmation-wrapper-' + $stamp + '-error.txt')
            [IO.File]::WriteAllText($errorPath, ($confirmationError | Out-String), (New-Object System.Text.UTF8Encoding($false)))
            Write-Warning 'Confirmation failed. Existing development results and the error will still be packed.'
        }
    }
    Invoke-RoutePython -Arguments @('-m', 'ids.route_batch', 'pack', '--run-dir', $runPath,
                                   '--output', $archivePath)
    Write-Output "Analysis archive: $archivePath"
    Write-Output "Full model artifacts remain in: $runPath"
    if ($null -ne $confirmationError) { throw $confirmationError }
} finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldEncoding
    $env:OMP_NUM_THREADS = $oldOmp
    $env:MKL_NUM_THREADS = $oldMkl
    $env:OPENBLAS_NUM_THREADS = $oldOpenBlas
    $env:CUBLAS_WORKSPACE_CONFIG = $oldCublas
}
