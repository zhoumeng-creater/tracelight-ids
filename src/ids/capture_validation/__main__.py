"""Run the registered external comparison on the laboratory's existing route-core environment."""

import argparse
from functools import partial
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd
import psutil

from ids.common import canonical_hash, environment_snapshot, read_json, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.flow_history.features import HISTORY_COLUMNS, _extract
from ids.route_batch.protocol import source_snapshot
from . import source

REPO = Path(__file__).resolve().parents[3]
ROLES = ("P_time", "C_time", "E_time")


def synthetic_preflight(root):
    files = sorted((REPO / "tests").glob("test_capture_validation*.py"))
    if not files:
        raise RuntimeError("Capture replication tests missing from checkout")
    out = root / "evidence" / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    result = out / "synthetic-junit.xml"
    log = out / "synthetic.log"
    with log.open("w", encoding="utf-8") as f:
        status = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", f"--junitxml={result}", *map(str, files)],
            cwd=REPO,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if status.returncode:
        raise RuntimeError(f"Lab synthetic tests failed; see {log}")
    return [log, result]


def prepare(root, official, index, packet_index, threads):
    from .cohort import prepare_targets

    data, manifest = prepare_targets(official, index, packet_rows=packet_index, threads=threads)
    out = root / "evidence/inputs"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "coverage.json", manifest)
    data.to_parquet(root / "targets.parquet", index=False)
    manifest_rows = out / "registered_rows.parquet"
    data[
        [
            "row_id",
            "role",
            "source_label",
            "source_family",
            "truth_source",
            "label",
            "attack_family",
            "communication_group",
            "flow_end_ms",
            "host_key",
            "end_day_utc",
        ]
    ].to_parquet(manifest_rows, index=False)
    return [out / "coverage.json", manifest_rows, root / "targets.parquet"]


def history(root, official, threads, expected):
    data = pd.read_parquet(root / "targets.parquet")
    # The shared verifier checks the publisher label and family against the raw source.
    original = data.assign(label=data.source_label, attack_family=data.source_family)
    work = root / "history-work"
    obs = _extract(official, original, work, threads, expected)
    if not np.array_equal(obs.row_id, data.row_id):
        raise ValueError("Registered history target rows changed")
    out = root / "evidence/inputs"
    dest = root / "inputs"
    dest.mkdir(exist_ok=True)
    artifacts = []
    for name in ("history_audit.json", "benchmark_segment.json"):
        artifacts.append(Path(shutil.copyfile(work / name, out / name)))
    if read_json(work / "benchmark_segment.json")["status"] == "ready":
        artifacts.extend(
            work / name
            for name in ("benchmark_stream.parquet", "benchmark_warmup.parquet", "benchmark_segment.json")
        )
    for col in HISTORY_COLUMNS:
        data[col] = obs[col].to_numpy()
    compact = out / "registered_history.parquet"
    data[["row_id", "role", *HISTORY_COLUMNS]].to_parquet(compact, index=False)
    for role in ROLES:
        path = dest / (role + ".parquet")
        data.loc[data.role.eq(role)].reset_index(drop=True).to_parquet(path, index=False)
        artifacts.append(path)
    return [*artifacts, compact]


def execute(root, supplied, threads):
    from .experiment import run_pair

    own_windows_workers()
    index, packet_index, registration = source.label_index()
    env = environment_snapshot()
    spec = {
        "study": "external-cse-hgb-20261111-v1",
        "registration": registration,
        "source_sha256": source_snapshot()["source_sha256"],
        "cpu_threads": threads,
        "runtime": {k: env.get(k) for k in ("python", "packages", "torch")},
    }
    binding = canonical_hash(spec)
    spec_path = root / "protocol.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("Run directory already has another registered source, runtime or CPU budget")
    write_json(spec_path, spec)
    write_json(root / "environment.json", env)
    with ResourceLog(root) as monitor:
        monitor.activity = "capture_synthetic"
        stage(root, "synthetic", binding, partial(synthetic_preflight, root))
        monitor.activity = "verify_uq_source"
        official = source.resolve(root / "source", supplied)
        monitor.activity = "verify_targets_and_labels"
        stage(root, "cohort", binding, partial(prepare, root, official, index, packet_index, threads))
        audit = read_json(root / "evidence/inputs/coverage.json")
        if audit.get("role_identity_sha256") != registration["role_identity_sha256"]:
            raise ValueError("Cohort rows or adjudicated labels differ from the registered audit")
        if not audit["eligible_for_training"]:
            raise ValueError("Registered external cohort is not evaluable; see evidence/inputs/coverage.json")
        monitor.activity = "external_history_replay"
        stage(root, "history", binding, partial(history, root, official, threads, source.CSV))
        return run_pair(root, binding, threads, monitor)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source", type=Path, help="Existing official UQ ZIP or plain CSV")
    ap.add_argument("--threads", type=int, default=min(24, psutil.cpu_count(logical=False) or 2))
    args = ap.parse_args()
    root = args.output.resolve()
    supplied = args.source.resolve() if args.source else None
    if args.threads < 2 or args.threads > 64:
        ap.error("The paired CPU run requires 2–64 threads")
    if root.exists() and not (root / "run.lock").is_file() and any(root.iterdir()):
        ap.error("Use a new directory or the original capture-validation output")
    with run_lock(root):
        try:
            result = execute(root, supplied, args.threads)
        except Exception:
            write_json(
                root / "last_invocation.json",
                {"status": "failed", "at": utc_now(), "error": traceback.format_exc()},
            )
            pack(root, prefix="independent-capture-results")
            raise
        write_json(root / "last_invocation.json", {"status": result["status"], "at": utc_now()})
        pack(root, prefix="independent-capture-results")


if __name__ == "__main__":
    main()
