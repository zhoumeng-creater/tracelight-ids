"""Run a development suite on a prepared dataset; never open final test data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


def build_commands(args, repo: Path, suite: dict, stamp: str) -> list[list[str]]:
    """Construct argv lists, so spaces in paths never become shell syntax."""
    prefix = [sys.executable, "-m", "ids"]
    doctor = prefix + ["doctor", "--output", str(args.work_dir / "environment" / f"doctor-{stamp}.json")]
    if args.device == "cuda":
        doctor.append("--require-cuda")
    suite_command = prefix + ["suite", "--data-dir", str(args.data_dir), "--config", str(args.config),
                              "--output-root", str(args.work_dir / "runs"), "--device", args.device]
    if args.resume:
        suite_command.append("--resume")
    report = prefix + ["report", "--run-root", str(args.work_dir / "runs" / suite["suite_name"]),
                       "--output", str(args.work_dir / "reports" / f"{suite['suite_name']}-{stamp}.html")]
    return [doctor, suite_command, report]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Existing output of ids prepare")
    parser.add_argument("--work-dir", type=Path, required=True, help="Persistent local disk directory for runs/reports")
    parser.add_argument("--config", type=Path, required=True, help="Development suite YAML")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed runs with identical inputs/code")
    parser.add_argument("--dry-run", action="store_true", help="Print exact argv lists without creating outputs or launching jobs")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    args.data_dir = args.data_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if not (args.data_dir / "manifest.json").is_file():
        parser.error("--data-dir must contain the manifest.json created by ids prepare")
    if not args.config.is_file():
        parser.error("--config is not a file")
    try:
        suite = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        parser.error(f"Cannot read suite: {exc}")
    if not isinstance(suite, dict) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", str(suite.get("suite_name", ""))):
        parser.error("suite_name must be a short ASCII filename")
    threads = suite.get("threads", 1)
    if type(threads) is not int or threads < 1:
        parser.error("Suite threads must be a positive integer")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commands = build_commands(args, repo, suite, stamp)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["PYTHONHASHSEED"] = "0"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONUNBUFFERED"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = str(threads)
    if not args.dry_run:
        (args.work_dir / "environment").mkdir(parents=True, exist_ok=True)
        (args.work_dir / "reports").mkdir(parents=True, exist_ok=True)
    for command in commands:
        print("argv:", repr(command), flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(command, cwd=repo, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Stopped: {command[3]} exited with code {exc.returncode}. No later phase was started.", file=sys.stderr)
            return exc.returncode or 1
    if not args.dry_run:
        print("Development complete. Calibration and test were not requested by this launcher.", flush=True)
        print("Validation report:", commands[-1][-1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
