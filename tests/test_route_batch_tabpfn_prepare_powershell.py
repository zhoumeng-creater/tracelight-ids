"""Run the actual wrapper with an argument-recording fake interpreter."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELLS = [path for name in ("powershell", "pwsh") if (path := shutil.which(name))]


@pytest.mark.parametrize("shell", SHELLS or [None], ids=lambda path: Path(path).stem if path else "unavailable")
def test_wrapper_forwards_paths_and_restores_environment(shell, tmp_path):
    if shell is None:
        pytest.skip("PowerShell unavailable")
    report_dir = tmp_path / "reports with spaces"
    report_dir.mkdir()
    (report_dir / "prepare-report.json").write_text('{"status":"ready"}', encoding="utf-8")
    capture = tmp_path / "captured.json"
    harness = tmp_path / "wrapper harness.ps1"
    harness.write_text(r'''
$ErrorActionPreference = 'Stop'
function Capture-Python {
    ConvertTo-Json -Compress -InputObject @($args) | Set-Content -LiteralPath $env:CAPTURE -Encoding UTF8
    $global:LASTEXITCODE = 0
}
$env:PYTHONPATH = 'original source path'
$env:PYTHONNOUSERSITE = 'original site value'
& $env:PREPARE_SCRIPT -CorePython Capture-Python -TabPFNPython $env:TARGET_PYTHON -Checkpoint $env:TARGET_WEIGHT -OutputDirectory $env:REPORT_DIR
if ($env:PYTHONPATH -ne 'original source path' -or $env:PYTHONNOUSERSITE -ne 'original site value') {
    throw 'Wrapper did not restore inherited environment'
}
''', encoding="utf-8")
    target = str(tmp_path / "optional env" / "Scripts" / "python.exe")
    weight = str(tmp_path / "weights with spaces" / "tabpfn-v2.5-classifier-v2.5_default.ckpt")
    env = dict(os.environ, CAPTURE=str(capture), PREPARE_SCRIPT=str(ROOT / "scripts/PREPARE_ROUTE_TABPFN.ps1"),
               TARGET_PYTHON=target, TARGET_WEIGHT=weight, REPORT_DIR=str(report_dir))
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(harness)],
                            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(capture.read_text(encoding="utf-8-sig")) == [
        "-X", "utf8", "-m", "ids.route_batch.tabpfn_prepare", "--core-python", "Capture-Python",
        "--tabpfn-python", target, "--checkpoint", weight, "--output-directory", str(report_dir)]
