[CmdletBinding()]
param(
    [string]$Python = 'D:\zhoumeng\Graduate\venvs\tracelight-route-core\Scripts\python.exe',
    [string]$Pcap,
    [string]$OutputRoot,
    [switch]$CheckOnly
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Pcap) {
    $Pcap = Join-Path $repo 'data\raw\ait-lds\source-audit-20261115\gather\intranet_server\logs\suricata\log.pcap.1642684613'
}
if (-not $OutputRoot) { $OutputRoot = Join-Path $repo 'local\ait-packet-evidence' }
$evidence = Join-Path $repo 'docs\research_alignment_20261115\risk_transfer\source\ait-small-evidence.json'
if ($CheckOnly) {
    & $Python -m pytest -q -p no:cacheprovider (Join-Path $repo 'tests\test_ait_packet_evidence.py')
    if ($LASTEXITCODE -ne 0) { throw "AIT packet evidence checks failed: $LASTEXITCODE" }
    return
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$OutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
& $Python (Join-Path $PSScriptRoot 'prepare_ait_packet_evidence.py') --pcap $Pcap --output $OutputRoot --evidence-json $evidence
$runExit = $LASTEXITCODE
$payloads = @(Get-ChildItem -LiteralPath $OutputRoot -File | Where-Object { $_.Extension -in @('.json','.csv','.log') })
if ($payloads.Count -gt 0) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
    $package = Join-Path (Split-Path -Parent $OutputRoot) "ait-packet-evidence-results-$stamp.zip"
    Compress-Archive -LiteralPath $payloads.FullName -DestinationPath $package -CompressionLevel Optimal
    (Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant() | Set-Content -LiteralPath "$package.sha256" -Encoding ascii
    Write-Host "Return package (no PCAP or download blocks): $package"
}
if ($runExit -ne 0) { throw "Source evidence preparation stopped with code $runExit. Keep the verified cached blocks; repeat the same command to resume." }
