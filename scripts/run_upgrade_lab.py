"""Run the V4 mechanism study using immutable V3 artifacts on lab RTX4090D.

Pilot and full both retain the complete saved V3 cohort. No teacher/anchor is
refitted, no prepared data is rebuilt, and calibration/test are never opened.
Dry-run inspects JSON metadata only. Resume accepts sealed complete outputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ids.common import canonical_hash, code_fingerprint, read_json, sha256_file, utc_now  # noqa: E402
from run_risk_lab import (  # noqa: E402
    REQUIRED_GPU, _activate_cuda_zero, _assert_no_links, _dataset_record,
    _inventory, _verify_zip, check_environment, require_lab_gpu,
)

PREFLIGHT = "UPGRADE_V4_LAB_PREFLIGHT.json"
COMPLETE = "UPGRADE_COMPLETE.json"
V3_COMPLETE = "RISK_RESEARCH_COMPLETE.json"
PACKAGE_MANIFEST = "UPGRADE_V4_PACKAGE_MANIFEST.sha256"
CONDITIONS = {"hard", "legacy_boundary_kd", "legacy_ekd", "raw_kd", "balanced_raw_kd",
              "guarded_hard", "guarded_raw_kd", "guarded_balanced_raw_kd"}


def validate_upgrade_config(config):
    from ids.upgrade_research import validate_upgrade_config as validate
    return validate(config)


def run_upgrade_research(v3_dir, data_dir, config, output, **kwargs):
    from ids.upgrade_research import run_upgrade_research as run
    return run(v3_dir, data_dir, config, output, **kwargs)


def verify_upgrade_research(output):
    from ids.upgrade_research import verify_upgrade_research as verify
    return verify(output)


def _profile_path(profile):
    if profile not in {"pilot", "full"}:
        raise ValueError("profile must be pilot or full")
    return _assert_no_links(REPO / "configs" / f"upgrade-v4-{profile}.yaml")


def load_profile(profile):
    config = yaml.safe_load(_profile_path(profile).read_text(encoding="utf-8"))
    validate_upgrade_config(config)
    seeds = [17] if profile == "pilot" else [17, 29, 43]
    if (config.get("purpose") != "upgrade_v4_development_only" or config.get("seeds") != seeds
            or set(config.get("conditions", [])) != CONDITIONS
            or len(config.get("conditions", [])) != len(CONDITIONS)):
        raise ValueError("V4 laboratory profile requires the pinned seeds and eight controls")
    if any(config.get(key) is not None for key in ("max_train_rows", "max_validation_rows")):
        raise ValueError("V4 reuses the complete saved V3 cohort; row caps are forbidden")
    return config


def _v3_record(name, directory, data_record, config):
    """Inspect only old completion/plan metadata; trainer verifies all payloads."""
    root = _assert_no_links(directory)
    completion_path = _assert_no_links(root / V3_COMPLETE)
    complete = read_json(completion_path)
    files = complete.get("files", {})
    if (complete.get("status") != "development_complete" or complete.get("reserved_partitions_read") != []
            or not isinstance(files, dict) or not files):
        raise ValueError(f"{name}: source V3 must be completed development-only research")
    plan_path = _assert_no_links(root / "plan.json")
    if files.get("plan.json") != sha256_file(plan_path):
        raise ValueError(f"{name}: source V3 plan checksum mismatch")
    plan = read_json(plan_path)
    if (plan.get("reserved_partitions_read") != [] or complete.get("plan_id") != plan.get("plan_id")
            or plan.get("plan_id") != canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})):
        raise ValueError(f"{name}: source V3 plan identity or reserved-partition scope is invalid")
    old_config = plan.get("config", {})
    if (old_config.get("purpose") != "risk_v3_development_only"
            or old_config.get("cohort_policy") != "complete_development"
            or any(old_config.get(k) is not None for k in ("max_train_rows", "max_validation_rows"))
            or not set(config["seeds"]).issubset(old_config.get("seeds", []))):
        raise ValueError(f"{name}: complete-cohort V3 source with every selected seed required")
    manifest = read_json(Path(data_record["data_dir"]) / "manifest.json")
    if plan.get("data") != manifest:
        raise ValueError(f"{name}: prepared manifest differs from original V3 data identity")
    return {"v3_dir": str(root), "completion_sha256": sha256_file(completion_path),
            "plan_sha256": files["plan.json"], "plan_id": plan.get("plan_id"),
            "original_source": copy.deepcopy(plan.get("source", {})),
            "roles_hash": plan.get("role_manifest_hash"),
            "verification": "completion/plan metadata only; runner verifies full source artifacts before fitting"}


def _verify_complete(output):
    root = _assert_no_links(output)
    if not _assert_no_links(root / COMPLETE).is_file():
        raise FileExistsError("Incomplete V4 output is preserved; interrupted-epoch resume is unsupported")
    done = verify_upgrade_research(root)
    if (done.get("status") != "development_complete" or done.get("reserved_partitions_read") != []
            or not isinstance(done.get("files"), dict) or not done["files"]):
        raise ValueError("V4 completion is invalid or reserved partitions were accessed")
    return done


def package_completed_environment(output, zip_path, *, resume=False):
    """Package only this new environment, never recursively follow its V3 path."""
    root, target = _assert_no_links(output), _assert_no_links(zip_path)
    if target.is_relative_to(root):
        raise ValueError("V4 ZIP must be outside the sealed environment")
    _verify_complete(root)
    inventory = _inventory(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", root.name) or root.name in {".", ".."}:
        raise ValueError("Archive root must be a safe component")
    payload = {root.name + "/" + name: digest for name, digest in inventory.items()}
    content = "".join(f"{digest}  {name}\n" for name, digest in sorted(payload.items())).encode("utf-8")
    expected = {**payload, PACKAGE_MANIFEST: hashlib.sha256(content).hexdigest()}
    sidecar = _assert_no_links(target.with_name(target.name + ".sha256"))
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial V4 package is preserved; refusing overwrite")
        if sidecar.read_text(encoding="ascii").strip() != sha256_file(target):
            raise ValueError("Existing V4 package sidecar mismatch")
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
        raise ValueError("V4 environment changed during packaging")
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
            raise FileExistsError("Existing/partial V4 preflight receipt is preserved")
        if target.read_bytes() != content or sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError("V4 preflight receipt differs from saved original bytes")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        with sidecar.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
    return {"file": str(target), "sha256": digest}


def run_lab(*, work_dir, unsw_v3_dir=None, cicids_v3_dir=None, unsw_data_dir=None,
            cicids_data_dir=None, profile="pilot", dry_run=False, resume=False, threads=None):
    pairs = {name: (v3, data) for name, v3, data in
             (("unsw", unsw_v3_dir, unsw_data_dir), ("cicids", cicids_v3_dir, cicids_data_dir))}
    if not any(v3 is not None or data is not None for v3, data in pairs.values()):
        raise ValueError("Provide at least one environment with both V3 and prepared-data paths")
    if any((v3 is None) != (data is None) for v3, data in pairs.values()):
        raise ValueError("Each environment requires both its source V3 directory and prepared data")
    config = load_profile(profile)
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        config = {**config, "threads": threads}
        validate_upgrade_config(config)
    environment = check_environment()
    gpu = require_lab_gpu(environment)
    _activate_cuda_zero()
    inputs = {}
    for name, (v3, data) in pairs.items():
        if v3 is not None:
            dataset = _dataset_record(name, data)
            inputs[name] = {"data": dataset, "v3": _v3_record(name, v3, dataset, config)}
    work = _assert_no_links(work_dir)
    for record in inputs.values():
        for value in (record["data"]["data_dir"], record["v3"]["v3_dir"]):
            old = Path(value)
            if work.is_relative_to(old) or old.is_relative_to(work):
                raise ValueError("New work, prepared data and old V3 must be separate non-nested paths")
    stable_environment = copy.deepcopy(environment)
    stable_environment.pop("recorded_at", None)
    interpreter = {"executable": str(Path(sys.executable).absolute()), "real_executable": str(Path(sys.executable).resolve()),
                   "prefix": sys.prefix, "base_prefix": sys.base_prefix}
    source = code_fingerprint()
    expected = {"students_planned": len(config["conditions"]) * len(config["seeds"]),
                "teacher_fits": 0, "anchor_fits": 0, "source_seeds_reused": len(config["seeds"])}
    binding = {"profile": profile, "config": config, "config_path": str(_profile_path(profile)),
               "config_file_sha256": sha256_file(_profile_path(profile)), "inputs": inputs,
               "source_sha256": source["source_sha256"], "environment": stable_environment,
               "interpreter": interpreter, "device": "cuda", "cuda_index": 0, "required_gpu": REQUIRED_GPU}
    preflight = {"schema_version": 1, "purpose": "upgrade_v4_lab_preflight", "created_at": utc_now(),
                 "binding": binding, "environment": environment, "selected_gpu": gpu, "source": source,
                 "expected_per_environment": expected,
                 "expected_all_environments": {key: value * len(inputs) for key, value in expected.items()},
                 "scope": "development only; complete V3 cohorts; no prepare/download/calibration/test; serial environments",
                 "dry_run_scope": "source/config/environment, prepared manifest and V3 completion/plan metadata only",
                 "resume_scope": "verified complete outputs only; no interrupted-epoch resume",
                 "reserved_partitions_read": []}
    plan_path = _assert_no_links(work / PREFLIGHT)
    if plan_path.exists():
        if not resume:
            raise FileExistsError("V4 work has a preflight; use identical-input --resume only")
        if read_json(plan_path).get("binding") != binding:
            raise ValueError("V4 source/configuration/interpreter/hardware/data/V3 binding changed")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("New V4 work directory must be empty")
        work.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(preflight, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    results = {}
    for name, record in inputs.items():
        output = _assert_no_links(work / ("upgrade-v4-" + name))
        if dry_run:
            results[name] = {"status": "plan_only", "output": str(output), "reserved_partitions_read": []}
            continue
        if output.exists():
            if not resume:
                raise FileExistsError("Existing V4 output is preserved; refusing overwrite")
            _verify_complete(output)
        run_upgrade_research(record["v3"]["v3_dir"], record["data"]["data_dir"], copy.deepcopy(config), output,
                             device="cuda", threads=config["threads"], resume=output.exists())
        complete = _verify_complete(output)
        counts = [complete.get(key) for key in ("n_students_planned", "n_students", "n_skipped_no_feasible_start",
                                               "n_teacher_fits", "n_anchor_fits")]
        if (any(type(n) is not int or n < 0 for n in counts)
                or counts[0] != expected["students_planned"] or counts[1] + counts[2] != counts[0]
                or counts[3:] != [0, 0]):
            raise ValueError("V4 completion contradicts the planned reused-source study")
        if code_fingerprint()["source_sha256"] != source["source_sha256"]:
            raise ValueError("V4 source changed during execution; preserve outputs without accepting package")
        if sha256_file(Path(record["v3"]["v3_dir"]) / V3_COMPLETE) != record["v3"]["completion_sha256"]:
            raise ValueError("Original V3 completion changed during the V4 run")
        results[name] = {"status": "verified_development_complete", "package": package_completed_environment(
            output, work / "packages" / ("UPGRADE_V4_" + name.upper() + ".zip"), resume=resume)}
    receipt = None if dry_run else _copy_saved_preflight(plan_path, work / "packages", resume=resume)
    return {"status": "upgrade_v4_lab_plan_only" if dry_run else "upgrade_v4_lab_development_complete",
            "profile": profile, "work_dir": str(work), "expected_per_environment": expected,
            "expected_all_environments": {key: value * len(inputs) for key, value in expected.items()},
            "environments": results, "preflight_receipt": receipt, "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    for name in ("unsw", "cicids"):
        parser.add_argument(f"--{name}-v3-dir", type=Path)
        parser.add_argument(f"--{name}-data-dir", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_lab(**vars(args))
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(2, f"V4 laboratory preflight/run rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
