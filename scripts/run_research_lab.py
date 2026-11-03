"""Run development-only research on the verified laboratory CUDA GPU.

No download/preparation/final-test entry is called. Pilot is UNSW-only because
the CICIDS row cap would remove its large linked Portscan training group.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ids.cicids import CIC_CATEGORICAL, CIC_DISCRETE, CIC_FEATURES, CIC_SCHEMA_VERSION  # noqa: E402
from ids.common import code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json  # noqa: E402
from ids.data import FEATURES, SCHEMA_VERSION  # noqa: E402
from ids.research import run_research, validate_config, verify_research  # noqa: E402
from ids.runner import verify_development_inputs  # noqa: E402
from ids.selection import normalize_feature_schema  # noqa: E402


def normalize_gpu_name(value):
    if not isinstance(value, str):
        raise ValueError("GPU names must be nonempty strings")
    result = re.sub(r"[^a-z0-9]", "", value.casefold())
    if not result:
        raise ValueError("GPU names must contain letters or digits")
    return result


def require_lab_gpu(environment, required_name="RTX4090D"):
    """Require usable torch CUDA and a matching default GPU, without fallback."""
    required = normalize_gpu_name(required_name)
    torch_info = environment.get("torch", {})
    if torch_info.get("cuda_available") is not True or not torch_info.get("cuda_runtime"):
        raise ValueError("Laboratory research requires torch CUDA; CPU fallback is forbidden")
    gpu = next((g for g in torch_info.get("gpus", []) if g.get("index") == 0), None)
    if gpu is None or not normalize_gpu_name(gpu.get("name", "")).endswith(required):
        actual = None if gpu is None else gpu.get("name")
        raise ValueError(f"Default CUDA GPU must match {required_name!r}; found {actual!r}")
    if type(gpu.get("total_memory_bytes")) is not int or gpu["total_memory_bytes"] <= 0:
        raise ValueError("CUDA GPU memory information is missing or invalid")
    return copy.deepcopy(gpu)


def load_profile(profile):
    if profile not in {"pilot", "full"}:
        raise ValueError("profile must be pilot or full")
    name = "research-v2-pilot.yaml" if profile == "pilot" else "research-v2.yaml"
    config = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
    validate_config(config)
    if profile == "full" and (config["cohort_policy"] != "complete_development" or any(
            config[key] is not None for key in ("max_train_rows", "max_validation_rows"))):
        raise ValueError("The full laboratory profile must retain every development row/group")
    return config


def _assert_no_links(path):
    """Reject symlinks/junctions in the path, allowing OneDrive cloud tags."""
    path = Path(path).absolute()
    for item in [*reversed(path.parents), path]:
        if item.is_symlink():
            raise ValueError(f"Symbolic links are not accepted: {item}")
        try:
            metadata = item.lstat()
        except FileNotFoundError:
            continue
        if getattr(metadata, "st_reparse_tag", 0) in {0xA0000003, 0xA000000C}:
            raise ValueError(f"Junctions/symbolic links are not accepted: {item}")
    return path.resolve()


def _inventory(root):
    """Enumerate only regular files inside root; never follow a linked folder."""
    root = _assert_no_links(root)
    result = {}
    pending = [root]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                path = _assert_no_links(Path(entry.path))
                if not path.is_relative_to(root):
                    raise ValueError("Package member escapes the completed research directory")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError(f"Only regular package files are allowed: {path}")
                else:
                    name = path.relative_to(root).as_posix()
                    if "\\" in name or any(ord(c) < 32 for c in name):
                        raise ValueError("Package filenames must have unambiguous manifest paths")
                    result[name] = sha256_file(path)
    return dict(sorted(result.items()))


def _verify_zip(path, expected):
    """Read every member and compare payload hashes, not just the ZIP CRC."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("ZIP inventory is incomplete, duplicate, or unexpected")
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            mode = member.external_attr >> 16
            if (member.is_dir() or member.filename.startswith("/") or ".." in parts
                    or "\\" in member.filename or stat.S_ISLNK(mode)):
                raise ValueError("ZIP contains an unsafe member")
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[member.filename]:
                raise ValueError(f"ZIP member hash mismatch: {member.filename}")


def package_completed_environment(output, zip_path, *, resume=False):
    """Package a verified completed environment without modifying its files."""
    root, target = _assert_no_links(output), _assert_no_links(zip_path)
    if target.is_relative_to(root):
        raise ValueError("The result ZIP must be outside the sealed environment")
    verify_research(root)
    inventory = _inventory(root)
    prefix = root.name
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix) or prefix in {".", ".."}:
        raise ValueError("The archive root name must be a safe path component")
    payload = {prefix + "/" + name: digest for name, digest in inventory.items()}
    manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(payload.items())).encode("utf-8")
    expected = {**payload, "PACKAGE_MANIFEST.sha256": hashlib.sha256(manifest).hexdigest()}
    sidecar = target.with_name(target.name + ".sha256")
    _assert_no_links(sidecar)
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial package is preserved; refusing to overwrite it")
        if sidecar.read_text(encoding="ascii").strip() != sha256_file(target):
            raise ValueError("Existing package SHA-256 sidecar does not match")
        _verify_zip(target, expected)
        return {"zip": str(target), "sha256": sha256_file(target), "status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement even if another process
    # races this check. Failed attempts are preserved, not silently overwritten.
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in inventory:
            archive.write(root / name, prefix + "/" + name)
        archive.writestr("PACKAGE_MANIFEST.sha256", manifest)
    _verify_zip(target, expected)
    verify_research(root)
    if _inventory(root) != inventory:
        raise ValueError("Completed environment changed during packaging; package is not accepted")
    digest = sha256_file(target)
    with sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "\n")
    return {"zip": str(target), "sha256": digest, "status": "verified_new"}


def _dataset_record(name, path):
    path = _assert_no_links(path)
    manifest = verify_development_inputs(path)
    schema = normalize_feature_schema(manifest.get("feature_schema"))
    if name == "unsw":
        expected = normalize_feature_schema()
        valid = (manifest.get("data_kind") in {"official", "mirror"}
                 and manifest.get("schema_version") == SCHEMA_VERSION
                 and manifest.get("features") == list(FEATURES))
    else:
        expected = normalize_feature_schema({"name": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
            "categorical": list(CIC_CATEGORICAL), "discrete": [x for x in CIC_FEATURES if x in CIC_DISCRETE]})
        valid = (manifest.get("data_kind") == "official_repaired"
                 and manifest.get("schema_version") == CIC_SCHEMA_VERSION
                 and manifest.get("features") == list(CIC_FEATURES))
    if not valid or schema != expected:
        raise ValueError(f"{name}: expected its pinned real-data kind and exact feature schema; synthetic is forbidden")
    return {"data_dir": str(path), "manifest_sha256": sha256_file(path / "manifest.json"),
            "data_kind": manifest["data_kind"], "schema": schema,
            "development_rows": {split: manifest["partitions"][split]["n_rows"]
                                 for split in ("train", "validation")}}


def _copy_saved_preflight(plan_path, packages_dir, *, resume):
    """Return the exact saved receipt alongside ZIPs; never regenerate dates."""
    source = _assert_no_links(plan_path)
    target = _assert_no_links(Path(packages_dir) / "LAB_PREFLIGHT.json")
    sidecar = target.with_name(target.name + ".sha256")
    _assert_no_links(sidecar)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial preflight receipt is preserved; refusing to overwrite")
        if target.read_bytes() != payload or sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError("Saved package preflight receipt does not match the original preflight")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
        with sidecar.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
    return {"file": str(target), "sha256": digest}


def run_lab(*, work_dir, unsw_data_dir=None, cicids_data_dir=None, profile="pilot", dry_run=False,
            resume=False, require_gpu_name="RTX4090D"):
    """Preflight both inputs, run environments serially, then verify their ZIPs."""
    if unsw_data_dir is None and cicids_data_dir is None:
        raise ValueError("Provide at least one prepared data directory")
    if profile == "pilot" and cicids_data_dir is not None:
        raise ValueError("CICIDS pilot is forbidden: row caps would drop whole attack groups; use full")
    config = load_profile(profile)
    validate_config(config)
    if profile == "full" and (config["cohort_policy"] != "complete_development" or any(
            config[key] is not None for key in ("max_train_rows", "max_validation_rows"))):
        raise ValueError("The full profile cannot cap rows or split the development cohort")
    environment = environment_snapshot()
    gpu = require_lab_gpu(environment, require_gpu_name)
    datasets = {name: _dataset_record(name, path) for name, path in
                (("unsw", unsw_data_dir), ("cicids", cicids_data_dir)) if path is not None}
    work = _assert_no_links(work_dir)
    for record in datasets.values():
        data_path = Path(record["data_dir"])
        if work.is_relative_to(data_path) or data_path.is_relative_to(work):
            raise ValueError("Work and prepared-data directories must be separate, non-nested paths")
    stable_environment = copy.deepcopy(environment)
    stable_environment.pop("recorded_at", None)
    source = code_fingerprint()
    binding = {"profile": profile, "config": config, "datasets": datasets, "device": "cuda",
               "required_gpu_name": normalize_gpu_name(require_gpu_name),
               "source_sha256": source["source_sha256"], "environment": stable_environment}
    n_students = len(config["models"]) * len(config["seeds"])
    n_teacher_fits = (config["folds"] + 1) * len(config["seeds"])
    preflight = {"schema_version": 1, "created_at": utc_now(), "binding": binding,
                 "environment": environment, "selected_default_gpu": gpu, "source": source,
                 "expected_per_environment": {"students": n_students, "teacher_fits": n_teacher_fits,
                                              "total_fits": n_students + n_teacher_fits},
                 "expected_all_environments": {"students": n_students * len(datasets),
                    "teacher_fits": n_teacher_fits * len(datasets)},
                 "execution": "environments run serially; HGB teachers use CPU, students use CUDA",
                 "scope": "development train/validation only; never prepare/download/calibration/test",
                 "reserved_partitions_read": []}
    plan_path = work / "LAB_PREFLIGHT.json"
    if plan_path.exists():
        if not resume:
            raise FileExistsError("Laboratory work already has a preflight; use --resume for identical inputs")
        if read_json(plan_path).get("binding") != binding:
            raise ValueError("Laboratory source, configuration, hardware, or data changed")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("New laboratory work directory must be empty")
        work.mkdir(parents=True, exist_ok=True)
        write_json(plan_path, preflight)
    results = {}
    for name, record in datasets.items():
        output = work / ("research-" + name)
        _assert_no_links(output)
        result = run_research(record["data_dir"], config, output, device="cuda", resume=resume, dry_run=dry_run)
        if dry_run:
            results[name] = result
        else:
            verify_research(output)
            results[name] = {"research_status": result["status"], "package": package_completed_environment(
                output, work / "packages" / ("research-" + name + ".zip"), resume=resume)}
    receipt = None if dry_run else _copy_saved_preflight(plan_path, work / "packages", resume=resume)
    return {"status": "lab_plan_only" if dry_run else "lab_development_complete", "profile": profile,
            "work_dir": str(work), "expected_per_environment": preflight["expected_per_environment"],
            "environments": results, "preflight_receipt": receipt, "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--unsw-data-dir", type=Path)
    parser.add_argument("--cicids-data-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--require-gpu-name", default="RTX4090D")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_lab(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"Laboratory preflight/run rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
