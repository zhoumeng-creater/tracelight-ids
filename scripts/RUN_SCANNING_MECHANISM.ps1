[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateSet('Preflight', 'Run', 'Verify', 'Pack')][string]$Stage,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Round14Root,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$RunRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $repoRoot
if (-not $RunRoot) { $RunRoot = Join-Path $repoRoot 'local\context-scanning-v1' }
$runPath = [IO.Path]::GetFullPath($RunRoot)
$protocolPath = Join-Path $runPath 'protocol.json'
$comparisonPath = Join-Path $runPath 'comparison'
$utf8 = New-Object System.Text.UTF8Encoding($false)
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

function Invoke-ScanPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $($Arguments -join ' ')" }
}

# Fail before scoring if the checkout is not the released code.
$expectedCommit = & git rev-parse 'scanning-mechanism-lab-v1.1^{commit}'
if ($LASTEXITCODE -ne 0) { throw 'Fetch the frozen scanning-mechanism-lab-v1.1 tag first.' }
$actualCommit = & git rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $expectedCommit) { throw 'Checkout scanning-mechanism-lab-v1.1 first.' }

if ($Stage -eq 'Preflight') {
    if (Test-Path -LiteralPath $runPath) { throw 'RunRoot already exists; preserve it and diagnose before another run.' }
    New-Item -ItemType Directory -Path $runPath | Out-Null
} elseif (-not (Test-Path -LiteralPath $runPath -PathType Container)) {
    throw 'Run Preflight first.'
}

# Do not put an open transcript in a directory while it is being hashed and packed.
$transcribing = $false
try {
    if ($Stage -ne 'Pack') {
        $logPath = Join-Path $runPath ($Stage.ToLowerInvariant() + '.log')
        if (Test-Path -LiteralPath $logPath) { throw 'Stage log exists; do not overwrite prior evidence.' }
        Start-Transcript -LiteralPath $logPath | Out-Null
        $transcribing = $true
    }
    switch ($Stage) {
        'Preflight' {
            Invoke-ScanPython -m pip install --no-deps --no-build-isolation --editable .
            $tests = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'tests') -File | Where-Object {
                $_.Name -like 'test_context_*.py' -or $_.Name -like 'test_scanning_*.py' -or
                $_.Name -in @('test_weighted_context_cache.py', 'test_tabicl_cache_adapter.py')
            } | Sort-Object Name | ForEach-Object { $_.FullName })
            $junitPath = Join-Path $runPath 'synthetic-junit.xml'
            $syntheticTemp = Join-Path $repoRoot 'local\synthetic\scanning-v1'
            [IO.Directory]::CreateDirectory((Split-Path -Parent $syntheticTemp)) | Out-Null
            $pytestArguments = @('-m', 'pytest', '-q') + $tests + @('--basetemp', $syntheticTemp, '--junitxml', $junitPath)
            Invoke-ScanPython @pytestArguments
            [xml]$junit = Get-Content -LiteralPath $junitPath -Raw
            $suites = @($junit.testsuites.testsuite)
            foreach ($suite in $suites) {
                if ([int]$suite.failures -ne 0 -or [int]$suite.errors -ne 0 -or [int]$suite.skipped -ne 0) {
                    throw 'Synthetic suite must have zero failures, errors and skips.'
                }
            }
            Invoke-ScanPython -m ids.context_cache scanning-register --round14-root $Round14Root --checkpoint $Checkpoint --output $protocolPath
            Invoke-ScanPython -m ids.context_cache scanning-preflight --config $protocolPath --output (Join-Path $runPath 'preflight.json')
        }
        'Run' {
            if (-not (Test-Path -LiteralPath (Join-Path $runPath 'preflight.json'))) { throw 'Preflight receipt missing.' }
            Invoke-ScanPython -m ids.context_cache lab-scanning-compare --config $protocolPath --output $comparisonPath
        }
        'Verify' {
            Invoke-ScanPython -m ids.context_cache scanning-verify --results $comparisonPath --output (Join-Path $runPath 'verification.json')
        }
        'Pack' {
            $archivePath = $runPath + '-results.zip'
            Invoke-ScanPython -m ids.context_cache scanning-pack --run-root $runPath --output $archivePath
            $digest = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
            [IO.File]::WriteAllText($archivePath + '.sha256', $digest + '  ' + [IO.Path]::GetFileName($archivePath) + "`n", $utf8)
            Write-Host "Return: $archivePath"
            Write-Host "SHA-256: $digest"
        }
    }
} catch {
    $failurePath = Join-Path $runPath ($Stage.ToLowerInvariant() + '-failure.txt')
    if (-not (Test-Path -LiteralPath $failurePath)) {
        [IO.File]::WriteAllText($failurePath, ($_ | Out-String), $utf8)
    }
    throw
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
}
