"""Preflight and run development-only V3 research on laboratory CUDA0 RTX4090D.

Dry-run reads the source/configuration and prepared manifest metadata only. It
never prepares/downloads data, reads reserved partitions, or calls the trainer.
Resume reuses verified COMPLETE results; interrupted training is not resumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tomllib
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ids.cicids import CIC_CATEGORICAL, CIC_DISCRETE, CIC_FEATURES, CIC_SCHEMA_VERSION  # noqa: E402
from ids.common import code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now  # noqa: E402
from ids.data import FEATURES, SCHEMA_VERSION  # noqa: E402
from ids.selection import normalize_feature_schema  # noqa: E402


PREFLIGHT = "RISK_V3_LAB_PREFLIGHT.json"
COMPLETE = "RISK_RESEARCH_COMPLETE.json"
PACKAGE_MANIFEST = "RISK_V3_PACKAGE_MANIFEST.sha256"
REQUIRED_GPU = "RTX4090D"


# Lazy imports keep --help available without importing a training module. These
# wrappers are also the explicit seams used by synthetic/mocked wrapper tests.
def validate_risk_config(config):
    from ids.risk_research import validate_risk_config as validate
    return validate(config)


def run_risk_research(data_dir, config, output, **kwargs):
    from ids.risk_research import run_risk_research as run
    return run(data_dir, config, output, **kwargs)


def verify_risk_research(output):
    from ids.risk_research import verify_risk_research as verify
    return verify(output)


def _assert_no_links(path):
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


def normalize_gpu_name(value):
    if not isinstance(value, str):
        raise ValueError("GPU name must be a string")
    result = re.sub(r"[^a-z0-9]", "", value.casefold())
    if not result:
        raise ValueError("GPU name is empty")
    return result


def require_lab_gpu(environment):
    info = environment.get("torch", {})
    if info.get("cuda_available") is not True or not info.get("cuda_runtime"):
        raise ValueError("V3 laboratory research requires torch CUDA; CPU fallback is forbidden")
    gpu = next((g for g in info.get("gpus", []) if g.get("index") == 0), None)
    if gpu is None or not normalize_gpu_name(gpu.get("name", "")).endswith(normalize_gpu_name(REQUIRED_GPU)):
        actual = None if gpu is None else gpu.get("name")
        raise ValueError(f"Default CUDA0 must be RTX4090D; found {actual!r}")
    if type(gpu.get("total_memory_bytes")) is not int or gpu["total_memory_bytes"] <= 0:
        raise ValueError("CUDA0 memory information is invalid")
    return copy.deepcopy(gpu)


def check_environment():
    requirements = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    mismatches = []
    for requirement in requirements + ["torch==2.6.0"]:
        name, expected = requirement.split("==")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual.split("+")[0] != expected:
            mismatches.append(f"{name}: need {expected}, found {actual}")
    if mismatches:
        raise ValueError("Locked project dependency mismatch: " + "; ".join(mismatches))
    return environment_snapshot()


def _activate_cuda_zero():
    """Pin the same process that calls the runner; never dispatch another Python."""
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA became unavailable; CPU fallback is forbidden")
    name = torch.cuda.get_device_name(0)
    if not normalize_gpu_name(name).endswith(normalize_gpu_name(REQUIRED_GPU)):
        raise ValueError(f"Actual CUDA0 must be RTX4090D; found {name!r}")
    torch.cuda.set_device(0)
    if torch.cuda.current_device() != 0:
        raise ValueError("Could not select CUDA0")


def _profile_path(profile):
    if profile not in {"pilot", "full"}:
        raise ValueError("profile must be pilot or full")
    return _assert_no_links(REPO / "configs" / ("risk-v3-pilot.yaml" if profile == "pilot" else "risk-v3.yaml"))


def load_profile(profile):
    config = yaml.safe_load(_profile_path(profile).read_text(encoding="utf-8"))
    validate_risk_config(config)
    if config.get("purpose") != "risk_v3_development_only":
        raise ValueError("The laboratory config must be V3 development-only")
    if profile == "pilot":
        if (config.get("cohort_policy") != "bounded_pilot" or config.get("seeds") != [17]
                or config.get("max_train_rows") != 12000 or config.get("max_validation_rows") != 6000):
            raise ValueError("V3 pilot must retain the pinned UNSW seed17 and 12000/6000 limits")
    elif (config.get("cohort_policy") != "complete_development" or len(config.get("seeds", [])) < 3
          or any(config.get(key) is not None for key in ("max_train_rows", "max_validation_rows"))):
        raise ValueError("V3 full requires at least three seeds and every development row/group")
    return config


def _dataset_record(name, directory):
    """Read metadata only; the V3 runner verifies train/validation file payloads."""
    root = _assert_no_links(directory)
    manifest_path = _assert_no_links(root / "manifest.json")
    checksum = _assert_no_links(root / "manifest.sha256")
    digest = sha256_file(manifest_path)
    if checksum.read_text(encoding="ascii").strip() != digest:
        raise ValueError("Prepared manifest checksum mismatch")
    manifest = read_json(manifest_path)
    if name == "unsw":
        expected = normalize_feature_schema()
        valid = (manifest.get("data_kind") in {"official", "mirror"}
                 and manifest.get("schema_version") == SCHEMA_VERSION
                 and manifest.get("features") == list(FEATURES))
    elif name == "cicids":
        expected = normalize_feature_schema({"name": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
            "categorical": list(CIC_CATEGORICAL), "discrete": [f for f in CIC_FEATURES if f in CIC_DISCRETE]})
        valid = (manifest.get("data_kind") == "official_repaired"
                 and manifest.get("schema_version") == CIC_SCHEMA_VERSION
                 and manifest.get("features") == list(CIC_FEATURES))
    else:
        raise ValueError("Unknown laboratory environment")
    if not valid or normalize_feature_schema(manifest.get("feature_schema")) != expected:
        raise ValueError(f"{name}: pinned real-data kind and exact schema required; synthetic is forbidden")
    development = {}
    for split in ("train", "validation"):
        record = manifest.get("partitions", {}).get(split, {})
        if (record.get("file") != split + ".parquet" or type(record.get("n_rows")) is not int
                or record["n_rows"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", ""))):
            raise ValueError(f"{name}: invalid declared development partition {split}")
        _assert_no_links(root / record["file"])
        development[split] = copy.deepcopy(record)
    return {"data_dir": str(root), "manifest_sha256": digest, "data_kind": manifest["data_kind"],
            "schema": expected, "development": development,
            "preflight_verification": "manifest_only; runner verifies development payloads before fitting"}


def _inventory(root):
    root = _assert_no_links(root)
    files, pending = {}, [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = _assert_no_links(entry.path)
                if not path.is_relative_to(root):
                    raise ValueError("Package member escapes completed environment")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError("Package accepts regular files only")
                else:
                    name = path.relative_to(root).as_posix()
                    if "\\" in name or any(ord(c) < 32 for c in name):
                        raise ValueError("Ambiguous package filename")
                    files[name] = sha256_file(path)
    return dict(sorted(files.items()))


def _verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or set(names) != set(expected):
            raise ValueError("ZIP inventory is incomplete, duplicated, or unexpected")
        for member in archive.infolist():
            if (member.is_dir() or member.filename.startswith("/") or ".." in PurePosixPath(member.filename).parts
                    or "\\" in member.filename or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("ZIP contains an unsafe member")
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected[member.filename]:
                raise ValueError("ZIP payload hash mismatch: " + member.filename)


def _verify_complete(output):
    output = _assert_no_links(output)
    if not _assert_no_links(output / COMPLETE).is_file():
        raise FileExistsError("Incomplete V3 output is preserved; epoch resume is not supported")
    result = verify_risk_research(output)
    if (result.get("status") != "development_complete" or result.get("reserved_partitions_read") != []
            or not isinstance(result.get("files"), dict) or not result["files"]):
        raise ValueError("V3 completion metadata is invalid or reserved partitions were accessed")
    return result


def package_completed_environment(output, zip_path, *, resume=False):
    root, target = _assert_no_links(output), _assert_no_links(zip_path)
    if target.is_relative_to(root):
        raise ValueError("V3 ZIP must be outside the sealed environment")
    _verify_complete(root)
    inventory = _inventory(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", root.name) or root.name in {".", ".."}:
        raise ValueError("Archive root name must be a safe component")
    payload = {root.name + "/" + name: digest for name, digest in inventory.items()}
    content = "".join(f"{digest}  {name}\n" for name, digest in sorted(payload.items())).encode("utf-8")
    expected = {**payload, PACKAGE_MANIFEST: hashlib.sha256(content).hexdigest()}
    sidecar = _assert_no_links(target.with_name(target.name + ".sha256"))
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial V3 package is preserved; refusing overwrite")
        if sidecar.read_text(encoding="ascii").strip() != sha256_file(target):
            raise ValueError("Existing V3 package sidecar mismatch")
        _verify_zip(target, expected)
        return {"zip": str(target), "sha256": sha256_file(target), "status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in inventory:
            archive.write(root / name, root.name + "/" + name)
        archive.writestr(PACKAGE_MANIFEST, content)
    _verify_zip(target, expected)
    _verify_complete(root)
    if _inventory(root) != inventory:
        raise ValueError("V3 environment changed while packaging")
    digest = sha256_file(target)
    with sidecar.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "\n")
    return {"zip": str(target), "sha256": digest, "status": "verified_new"}


def _copy_saved_preflight(source, packages, *, resume):
    source = _assert_no_links(source)
    target = _assert_no_links(Path(packages) / PREFLIGHT)
    sidecar = _assert_no_links(target.with_name(target.name + ".sha256"))
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial V3 preflight receipt is preserved")
        if target.read_bytes() != content or sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError("V3 preflight receipt differs from saved original bytes")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        with sidecar.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
    return {"file": str(target), "sha256": digest}


def run_lab(*, work_dir, unsw_data_dir=None, cicids_data_dir=None, profile="pilot",
            dry_run=False, resume=False, threads=None):
    if unsw_data_dir is None and cicids_data_dir is None:
        raise ValueError("Provide at least one prepared environment")
    if profile == "pilot" and cicids_data_dir is not None:
        raise ValueError("V3 pilot accepts UNSW only; CICIDS requires full without row caps")
    config = load_profile(profile)
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        config = {**config, "threads": threads}
        validate_risk_config(config)
    environment = check_environment()
    gpu = require_lab_gpu(environment)
    _activate_cuda_zero()
    datasets = {name: _dataset_record(name, path) for name, path in
                (("unsw", unsw_data_dir), ("cicids", cicids_data_dir)) if path is not None}
    work = _assert_no_links(work_dir)
    for record in datasets.values():
        data = Path(record["data_dir"])
        if work.is_relative_to(data) or data.is_relative_to(work):
            raise ValueError("Work and prepared-data directories must be separate non-nested paths")
    stable_environment = copy.deepcopy(environment)
    stable_environment.pop("recorded_at", None)
    interpreter = {"executable": str(Path(sys.executable).absolute()),
                   "real_executable": str(Path(sys.executable).resolve()),
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix}
    source = code_fingerprint()
    expected = {"students": len(config["conditions"]) * len(config["seeds"]),
                "teacher_fits": (config["folds"] + 1) * len(config["seeds"]),
                "anchor_fits": len(config["seeds"])}
    expected["total_fits"] = sum(expected.values())
    binding = {"profile": profile, "config": config, "config_path": str(_profile_path(profile)),
               "config_file_sha256": sha256_file(_profile_path(profile)), "datasets": datasets,
               "source_sha256": source["source_sha256"], "environment": stable_environment,
               "interpreter": interpreter, "device": "cuda", "cuda_index": 0, "required_gpu": REQUIRED_GPU}
    preflight = {"schema_version": 1, "purpose": "risk_v3_lab_preflight", "created_at": utc_now(),
                 "binding": binding, "environment": environment, "selected_gpu": gpu, "source": source,
                 "expected_per_environment": expected,
                 "expected_all_environments": {key: value * len(datasets) for key, value in expected.items()},
                 "scope": "development only; no prepare/download/calibration/test; serial environments",
                 "dry_run_scope": "manifest metadata and plan only; no training or role generation",
                 "resume_scope": "verified complete outputs only; no interrupted-epoch training resume",
                 "reserved_partitions_read": []}
    plan_path = _assert_no_links(work / PREFLIGHT)
    if plan_path.exists():
        if not resume:
            raise FileExistsError("V3 work has a preflight; use --resume only with identical inputs")
        if read_json(plan_path).get("binding") != binding:
            raise ValueError("V3 source/configuration/interpreter/hardware/data binding changed")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("New V3 work directory must be empty")
        work.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(preflight, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    results = {}
    for name, record in datasets.items():
        output = _assert_no_links(work / ("risk-v3-" + name))
        if dry_run:
            results[name] = {"status": "plan_only", "output": str(output),
                             "data_kind": record["data_kind"], "reserved_partitions_read": []}
            continue
        if output.exists():
            if not resume:
                raise FileExistsError("Existing V3 output is preserved; refusing overwrite")
            _verify_complete(output)
            # The runner recomputes its full plan binding (including roles and
            # development payloads) before accepting a completed-only resume.
            run_risk_research(record["data_dir"], copy.deepcopy(config), output,
                              device="cuda", threads=config["threads"], resume=True)
            complete = _verify_complete(output)
        else:
            run_risk_research(record["data_dir"], copy.deepcopy(config), output,
                              device="cuda", threads=config["threads"], resume=False)
            complete = _verify_complete(output)
        if (complete.get("n_students") != expected["students"]
                or complete.get("n_teacher_fits") != expected["teacher_fits"]):
            raise ValueError("V3 completion fitting counts differ from the frozen plan")
        if code_fingerprint()["source_sha256"] != source["source_sha256"]:
            raise ValueError("V3 source changed during execution; preserve outputs without accepting package")
        results[name] = {"status": "verified_development_complete", "package": package_completed_environment(
            output, work / "packages" / ("RISK_V3_" + name.upper() + ".zip"), resume=resume)}
    receipt = None if dry_run else _copy_saved_preflight(plan_path, work / "packages", resume=resume)
    return {"status": "risk_v3_lab_plan_only" if dry_run else "risk_v3_lab_development_complete",
            "profile": profile, "work_dir": str(work), "expected_per_environment": expected,
            "environments": results, "preflight_receipt": receipt, "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--unsw-data-dir", type=Path)
    parser.add_argument("--cicids-data-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_lab(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"V3 laboratory preflight/run rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
