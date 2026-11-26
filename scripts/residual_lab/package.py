"""Fetch the Windows wheel cache and build the self-contained residual IDS bundle."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from scripts.residual_lab.common import (
    BUNDLE_MANIFEST, BUNDLE_SCHEMA, HERE, checked_member, fingerprint, read_json, verify_file, wheel_entries,
    write_json,
)

ROOT = HERE.parents[1]
SOURCE = """
LICENSE
docs/RESIDUAL_RESCUE_DATA_NOTICES.md
docs/RESIDUAL_RESCUE_LAB_20261126.md
docs/coverage_basis_20261119/batch/cohorts.json
scripts/ait_forward/baseline.py
scripts/ait_forward/comparators.py
scripts/ait_forward/decisions.py
scripts/calibration_repair/data.py
scripts/calibration_repair/evaluation.py
scripts/calibration_repair/objective.py
scripts/calibration_repair/run.py
scripts/calibration_repair/training.py
scripts/calibration_transfer/data.py
scripts/calibration_transfer/objective.py
scripts/comprehensive_batch/benchmark.py
scripts/comprehensive_batch/evaluation.py
scripts/comprehensive_batch/execution.py
scripts/comprehensive_batch/matrix.py
scripts/comprehensive_batch/objective.py
scripts/comprehensive_batch/resource_profile.py
scripts/comprehensive_batch/resources.py
scripts/comprehensive_batch/run.py
scripts/comprehensive_batch/training.py
scripts/coverage_batch/materialize.py
scripts/coverage_batch/run.py
scripts/fit_contract/run_minimal_comparison.py
scripts/normal_tail_mechanism.py
scripts/residual_lab/bootstrap.py
scripts/residual_lab/common.py
scripts/residual_lab/wheels-win-cp312.tsv
src/ids/__init__.py
src/ids/cicids.py
src/ids/common.py
src/ids/context_cache/__init__.py
src/ids/context_cache/attention.py
src/ids/context_cache/cache.py
src/ids/context_cache/calibration.py
src/ids/context_cache/compiler.py
src/ids/context_cache/csv_workflow.py
src/ids/context_cache/lab_data.py
src/ids/context_cache/preprocessing.py
src/ids/context_cache/representation_audit.py
src/ids/context_cache/tabicl_adapter.py
src/ids/context_cache/upstream.py
src/ids/data.py
src/ids/flow_history/__init__.py
src/ids/flow_history/arm_cost.py
src/ids/flow_history/model.py
src/ids/flow_history/trees.py
src/ids/model_training/__init__.py
src/ids/model_training/preprocessing.py
src/ids/models.py
src/ids/recent_baselines.py
src/ids/route_batch/__init__.py
src/ids/route_batch/benchmark.py
src/ids/route_batch/metrics.py
src/ids/route_batch/tabm_training.py
src/ids/selection.py
""".split()
PAYLOAD = {
    "fit_rows.jsonl.gz", "calibration_rows.jsonl.gz", "evaluation_rows.jsonl.gz",
    "memberships.json", "data_audit.json", "frozen-models.zip",
}


def fetch_wheel(cache, entry):
    destination = checked_member(cache, entry["filename"])
    if destination.exists():
        verify_file(destination, entry)
        return destination.name
    with NamedTemporaryFile(prefix=destination.stem, suffix=".part", dir=cache, delete=False) as out:
        temporary = Path(out.name)
        try:
            with urlopen(entry["url"], timeout=60) as response:
                shutil.copyfileobj(response, out, length=1024 * 1024)
        except Exception:
            out.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        verify_file(temporary, entry)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.name


def fetch_wheels(cache):
    cache.mkdir(parents=True, exist_ok=True)
    entries = wheel_entries()
    with ThreadPoolExecutor(max_workers=6) as pool:
        for filename in pool.map(lambda entry: fetch_wheel(cache, entry), entries):
            print(f"Verified {filename}", flush=True)


def payload_files(payload):
    manifest_path = payload / "manifest.json"
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema") != "tracelight-residual-rescue-input-v1"
        or manifest.get("scope") != "opened_development_only"
        or manifest.get("lineage", {}).get("confirmation_used") is not False
    ):
        raise ValueError("Input must declare the frozen opened-development experiment.")
    if set(manifest["artifacts"]) != PAYLOAD:
        raise ValueError("Unexpected payload files. Check the input bundle manifest.")
    result = {"data/manifest.json": manifest_path}
    for name, expected in manifest["artifacts"].items():
        path = checked_member(payload, name)
        verify_file(path, expected)
        result[f"data/{name}"] = path
    return result


def build(payload, cache, output, *, allow_dirty=False):
    receipt = output.with_suffix(output.suffix + ".sha256.json")
    if output.exists() or receipt.exists():
        raise FileExistsError(f"Use a new destination: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Package output must have the .zip extension.")
    source_files = {name: ROOT / name for name in SOURCE}
    for path in (ROOT / "scripts/residual_ids").glob("*.py"):
        if path.name != "build_inputs.py":
            source_files[path.relative_to(ROOT).as_posix()] = path
    if "scripts/residual_ids/run.py" not in source_files:
        raise FileNotFoundError("The residual experiment runner is missing.")
    source_files["RUN_RESIDUAL_LAB.cmd"] = ROOT / "scripts/RUN_RESIDUAL_LAB.cmd"
    source_files["README.md"] = HERE / "README.md"
    source_paths = sorted({path.relative_to(ROOT).as_posix() for path in source_files.values()})
    source_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *source_paths],
        cwd=ROOT, text=True,
    ).strip()
    if source_status and not allow_dirty:
        raise ValueError("Packaged source has uncommitted changes. Commit it before building the release.")
    files = {**payload_files(payload), **source_files}
    for entry in wheel_entries():
        wheel = checked_member(cache, entry["filename"])
        verify_file(wheel, entry)
        files[f"wheelhouse/{wheel.name}"] = wheel
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_tree_dirty": bool(source_status),
        "files": {name: fingerprint(path) for name, path in sorted(files.items())},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=output.stem, suffix=".part", dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=4) as bundle:
            bundle.writestr(BUNDLE_MANIFEST, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            for name, path in sorted(files.items()):
                codec = ZIP_STORED if path.suffix in {".whl", ".gz", ".zip"} else ZIP_DEFLATED
                bundle.write(path, name, compress_type=codec)
        with ZipFile(temporary) as bundle:
            if bad := bundle.testzip():
                raise ValueError(f"ZIP member verification failed: {bad}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(receipt, {"filename": output.name, **fingerprint(output)})
    print(f"Bundle: {output}")
    print(f"Checksum: {receipt}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch-wheels")
    download.add_argument("--wheelhouse", type=Path, required=True)
    pack = commands.add_parser("build")
    pack.add_argument("--payload", type=Path, required=True)
    pack.add_argument("--wheelhouse", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--allow-dirty", action="store_true", help="build an engineering test bundle from local edits")
    options = parser.parse_args()
    if options.command == "fetch-wheels":
        fetch_wheels(options.wheelhouse)
    else:
        build(options.payload, options.wheelhouse, options.output, allow_dirty=options.allow_dirty)


if __name__ == "__main__":
    main()
