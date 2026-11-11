import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("shell", [name for name in ("powershell", "pwsh") if shutil.which(name)])
@pytest.mark.parametrize("check_only", [True, False])
def test_lab_wrapper_uses_current_checkout_and_scoped_tests(tmp_path, shell, check_only):
    stub = tmp_path / "capture.ps1"
    stub.write_text("ConvertTo-Json -InputObject @($args) | Set-Content -Encoding UTF8 -LiteralPath $env:IDS_STUB_OUT\nexit 0\n",
                    encoding="ascii")
    dest = tmp_path / "args.json"
    args = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO / "scripts/RUN_FLOW_HISTORY.ps1"),
            "-Python", str(stub)]
    if check_only:
        args.append("-CheckOnly")
    subprocess.run(args, check=True, cwd=tmp_path, env={**os.environ, "IDS_STUB_OUT": str(dest)})
    argv = json.loads(dest.read_text(encoding="utf-8-sig"))
    if check_only:
        assert argv[:3] == ["-m", "pytest", "-q"]
        assert all(Path(path).is_file() and Path(path).name.startswith("test_flow_history") for path in argv[3:])
        assert len(argv) > 3
    else:
        assert argv[:2] == ["-m", "ids.flow_history"]
        assert Path(argv[argv.index("--output") + 1]) == REPO / "local/flow-history-r19"


@pytest.mark.skipif(os.name != "nt", reason="Windows laboratory process ownership")
def test_interrupted_coordinator_releases_its_worker(tmp_path):
    source = ("from ids.flow_history.execution import own_windows_workers\n"
              "import subprocess, sys, time\n"
              "own_windows_workers()\n"
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              "print(p.pid, flush=True)\n"
              "time.sleep(60)\n")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen([sys.executable, "-c", source], cwd=tmp_path, stdout=subprocess.PIPE, text=True, env=env)
    child = None
    try:
        child = psutil.Process(int(proc.stdout.readline()))
        proc.kill()
        proc.wait(timeout=10)
        for _ in range(100):
            if not child.is_running():
                break
            time.sleep(.05)
        assert not child.is_running()
    finally:
        if proc.poll() is None:
            proc.kill()
        if child and child.is_running():
            child.kill()
