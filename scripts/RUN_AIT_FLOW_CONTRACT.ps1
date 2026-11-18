[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$Pcap,
    [string]$OutputRoot,
    [ValidateSet('auto','wsl','native')][string]$Runtime = 'auto',
    [string]$Distro = 'Ubuntu',
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\ait-flow-contract' }
if ($CheckOnly) {
    & $Python -X utf8 -m pytest -q -p no:cacheprovider (Join-Path $repo 'tests\test_ait_zeek_runtime.py') (Join-Path $repo 'tests\test_ait_packet_diagnostics.py') (Join-Path $repo 'tests\test_ait_contract_analysis.py') (Join-Path $repo 'tests\test_ait_contract_runner.py') (Join-Path $repo 'tests\test_ait_compare_exports.py')
    if ($LASTEXITCODE -ne 0) { throw "AIT contract checks failed: $LASTEXITCODE" }
    return
}
$arguments = @((Join-Path $PSScriptRoot 'ait_contract\run.py'), '--output', $OutputRoot, '--runtime', $Runtime, '--distro', $Distro)
if ($Pcap) { $arguments += @('--pcap', $Pcap) }
& $Python -X utf8 @arguments
if ($LASTEXITCODE -ne 0) { throw 'AIT data-contract batch incomplete. Return the generated ZIP; retained successful stages are reusable.' }
