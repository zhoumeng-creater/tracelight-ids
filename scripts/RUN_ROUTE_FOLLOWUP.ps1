[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$OriginalRun = 'D:\zhoumeng\Graduate\tracelight-route-batch\local\route-batch-20261108\run',
    [string]$OutputRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) 'local\route-followup-20261109'),
    [string]$TabPFNPython = 'D:\zhoumeng\Graduate\venvs\tracelight-route-tabpfn-followup\Scripts\python.exe',
    [string]$TabPFNCheckpoint,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$originalPath = [IO.Path]::GetFullPath($OriginalRun)
$outputPath = [IO.Path]::GetFullPath($OutputRoot)
$runPath = Join-Path $outputPath 'run'
$preparationPath = Join-Path $outputPath 'preparation'
$preparationReport = Join-Path $preparationPath 'prepare-report.json'
$effectivePreparationReport = Join-Path $preparationPath 'followup-preparation-report.json'
$prepareScript = Join-Path $PSScriptRoot 'PREPARE_ROUTE_TABPFN.ps1'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$preflightReport = Join-Path $outputPath ('preflight-' + $stamp + '.json')
$coreTestReport = Join-Path $outputPath ('core-synthetic-tests-' + $stamp + '.xml')
$coreTestLog = Join-Path $outputPath ('core-synthetic-tests-' + $stamp + '.log')
$wrapperName = 'followup-wrapper-' + $stamp + '.json'
$archivePath = Join-Path $outputPath ('route-followup-results-' + $stamp + '.zip')

function Test-SameOrChildPath {
    param([string]$Candidate, [string]$Parent)
    $candidateName = $Candidate.TrimEnd('\', '/')
    $parentName = $Parent.TrimEnd('\', '/')
    return $candidateName.Equals($parentName, [StringComparison]::OrdinalIgnoreCase) -or
        $candidateName.StartsWith($parentName + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}

if ((Test-SameOrChildPath -Candidate $outputPath -Parent $originalPath) -or
    (Test-SameOrChildPath -Candidate $originalPath -Parent $outputPath)) {
    throw 'OutputRoot and OriginalRun must be separate directories; neither may contain the other.'
}
foreach ($name in @('protocol.json', 'source_snapshot.json', 'summary.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $originalPath $name) -PathType Leaf)) {
        throw "OriginalRun is missing required original-batch evidence: $name"
    }
}
if ((Test-Path -LiteralPath $runPath) -and -not $Resume) {
    throw 'The follow-up run already exists. Use Resume to reuse completed follow-up tasks.'
}
if (-not (Test-Path -LiteralPath $prepareScript -PathType Leaf)) {
    throw 'PREPARE_ROUTE_TABPFN.ps1 is missing from this checkout.'
}
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python = [IO.Path]::GetFullPath($Python) }
$TabPFNPython = [IO.Path]::GetFullPath($TabPFNPython)
if ([string]::IsNullOrWhiteSpace($TabPFNCheckpoint)) {
    $originalProtocol = Get-Content -LiteralPath (Join-Path $originalPath 'protocol.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $TabPFNCheckpoint = $originalProtocol.tabpfn_checkpoint.path
    if ([string]::IsNullOrWhiteSpace($TabPFNCheckpoint)) {
        throw 'Specify TabPFNCheckpoint; the original protocol has no registered target path.'
    }
}
$TabPFNCheckpoint = [IO.Path]::GetFullPath($TabPFNCheckpoint)
foreach ($target in @($TabPFNPython, $TabPFNCheckpoint)) {
    if (Test-SameOrChildPath -Candidate $target -Parent $originalPath) {
        throw 'TabPFNPython and TabPFNCheckpoint must be outside OriginalRun before preparation starts.'
    }
}
[IO.Directory]::CreateDirectory($outputPath) | Out-Null
$errorsSeen = New-Object 'System.Collections.Generic.List[string]'
$report = [ordered]@{
    original_run = $originalPath; followup_run = $runPath
    scope = 'two TabICL replay diagnostics and one TabPFN development-E task; no TabM or confirmation'
    resume = [bool]$Resume; preparation = 'not_started'; preflight = 'not_started'; core_synthetic_tests = 'not_started'
    followup = 'not_started'; all_tasks_complete = $false; pack = 'not_started'; archive = $null; errors = @()
}

function Save-WrapperReport {
    $report.errors = @($errorsSeen.ToArray())
    $json = ($report | ConvertTo-Json -Depth 8) + [Environment]::NewLine
    [IO.File]::WriteAllText((Join-Path $outputPath $wrapperName), $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Invoke-FollowupPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code ${exitCode}: $($Arguments -join ' ')"
    }
}

$savedEnvironment = @{}
foreach ($name in @('PYTHONPATH', 'PYTHONIOENCODING', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$sourcePaths = @((Join-Path $repoRoot 'src'), (Join-Path $repoRoot 'local\vendor\tabicl\src'))
if (-not [string]::IsNullOrWhiteSpace($savedEnvironment.PYTHONPATH)) { $sourcePaths += $savedEnvironment.PYTHONPATH }
$env:PYTHONPATH = $sourcePaths -join [IO.Path]::PathSeparator
$env:PYTHONIOENCODING = 'utf-8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Push-Location -LiteralPath $repoRoot
try {
    $preparationError = $null
    try {
        $shellExecutable = (Get-Process -Id $PID).Path
        & $shellExecutable -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $prepareScript `
            -CorePython $Python -TabPFNPython $TabPFNPython -Checkpoint $TabPFNCheckpoint -OutputDirectory $preparationPath
        $prepareExitCode = $LASTEXITCODE
        if ($prepareExitCode -ne 0) { throw "TabPFN preparation exited with code $prepareExitCode." }
        if (-not (Test-Path -LiteralPath $preparationReport -PathType Leaf)) {
            throw 'TabPFN preparation did not produce prepare-report.json.'
        }
        $prepared = Get-Content -LiteralPath $preparationReport -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($prepared.status -ne 'ready') { throw "TabPFN preparation status is $($prepared.status), not ready." }
        $report.preparation = 'ready'
    } catch {
        $report.preparation = 'failed'
        $preparationError = $_.Exception.Message
        $errorsSeen.Add($_.Exception.Message)
        Write-Warning 'TabPFN preparation failed. The two independent TabICL diagnostics will still be attempted.'
    }
    # Bind this invocation's readiness, never a stale ready report left by an earlier attempt.
    $effectivePreparation = [ordered]@{}
    if (Test-Path -LiteralPath $preparationReport -PathType Leaf) {
        try {
            $rawPreparation = Get-Content -LiteralPath $preparationReport -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($property in $rawPreparation.PSObject.Properties) {
                $effectivePreparation[$property.Name] = $property.Value
            }
        } catch {
            $effectivePreparation['original_report_read_error'] = $_.Exception.Message
        }
    }
    $effectivePreparation['wrapper_status'] = $report.preparation
    $effectivePreparation['wrapper_error'] = $preparationError
    $effectivePreparation['original_report'] = $preparationReport
    if ($report.preparation -eq 'ready') {
        $effectivePreparation['status'] = 'ready'
    } elseif (-not $effectivePreparation.Contains('status') -or $effectivePreparation['status'] -eq 'ready') {
        $effectivePreparation['status'] = 'failed'
        $effectivePreparation['failed_stage'] = 'preparation_wrapper'
        $effectivePreparation['error'] = $preparationError
    }
    [IO.Directory]::CreateDirectory($preparationPath) | Out-Null
    [IO.File]::WriteAllText($effectivePreparationReport, (($effectivePreparation | ConvertTo-Json -Depth 30) + [Environment]::NewLine),
                          (New-Object System.Text.UTF8Encoding($false)))
    Save-WrapperReport
    $commonArguments = @('--original-run', $originalPath, '--output', $runPath,
                         '--tabpfn-python', $TabPFNPython, '--tabpfn-checkpoint', $TabPFNCheckpoint,
                         '--preparation-report', $effectivePreparationReport)
    try {
        Invoke-FollowupPython -Arguments (@('-m', 'ids.route_batch', 'followup-preflight') + $commonArguments + @('--report', $preflightReport))
        $report.preflight = 'passed'
    } catch {
        $report.preflight = 'failed'
        $errorsSeen.Add($_.Exception.Message)
        Save-WrapperReport
        throw
    }
    $coreTests = @(
        'test_route_batch_foundation.py',
        'test_route_batch_replay.py',
        'test_route_batch_deployment_replay.py',
        'test_route_batch_cost_worker.py',
        'test_route_batch_followup.py',
        'test_route_batch_followup_cli.py'
    ) | ForEach-Object { Join-Path (Join-Path $repoRoot 'tests') $_ }
    Write-Output "Checking the core environment with synthetic data and random tiny models. Log: $coreTestLog"
    try {
        Invoke-FollowupPython -Arguments (@('-m', 'pytest', '-q', '--junitxml', $coreTestReport) + $coreTests) *> $coreTestLog
        if (-not (Test-Path -LiteralPath $coreTestReport -PathType Leaf)) {
            throw 'Core synthetic checks returned without a JUnit report.'
        }
        $report.core_synthetic_tests = 'passed'
    } catch {
        $report.core_synthetic_tests = 'failed'
        $errorsSeen.Add($_.Exception.Message)
        Save-WrapperReport
        throw
    }
    try {
        $runArguments = @('-m', 'ids.route_batch', 'followup') + $commonArguments
        if ($Resume) {
            $runArguments += '--resume'
            Write-Output 'Resume applies only to this follow-up: completed tasks are reused; unfinished tasks are retried. The original batch is read-only.'
        }
        Invoke-FollowupPython -Arguments $runArguments
        $report.followup = 'queue_finished'
    } catch {
        $report.followup = 'command_failed'
        $errorsSeen.Add($_.Exception.Message)
    }
    $summaryPath = Join-Path $runPath 'summary.json'
    if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
        try {
            $summary = Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $report.all_tasks_complete = ($summary.all_tasks_complete -eq $true)
            if (-not $report.all_tasks_complete) { $errorsSeen.Add('The follow-up summary contains unfinished, failed or unavailable tasks.') }
        } catch {
            $errorsSeen.Add('Cannot read follow-up completion state: ' + $_.Exception.Message)
        }
    } else {
        $errorsSeen.Add('The follow-up did not produce summary.json.')
    }
    $report.pack = 'requested'
    $report.archive = $archivePath
    Save-WrapperReport
    if (-not (Test-Path -LiteralPath $runPath -PathType Container)) {
        throw ('No follow-up run directory was created. ' + ($errorsSeen -join ' | '))
    }
    $evidencePath = Join-Path $runPath 'preparation-evidence'
    [IO.Directory]::CreateDirectory($evidencePath) | Out-Null
    if (Test-Path -LiteralPath $preparationPath -PathType Container) {
        foreach ($item in (Get-ChildItem -LiteralPath $preparationPath -Force)) {
            Copy-Item -LiteralPath $item.FullName -Destination $evidencePath -Recurse -Force
        }
    }
    foreach ($pattern in @('preflight-*.json', 'core-synthetic-tests-*', 'followup-wrapper-*.json')) {
        foreach ($item in (Get-ChildItem -LiteralPath $outputPath -Filter $pattern -File)) {
            Copy-Item -LiteralPath $item.FullName -Destination $evidencePath -Force
        }
    }
    try {
        Invoke-FollowupPython -Arguments @('-m', 'ids.route_batch', 'followup-pack', '--run-dir', $runPath, '--output', $archivePath)
        $report.pack = 'complete'
        $report.archive = $archivePath
        Save-WrapperReport
    } catch {
        $report.pack = 'failed'
        $errorsSeen.Add($_.Exception.Message)
        Save-WrapperReport
        throw
    }
    Write-Output "Follow-up archive: $archivePath"
    if ($errorsSeen.Count -gt 0) { throw ($errorsSeen -join ' | ') }
} finally {
    Pop-Location
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
