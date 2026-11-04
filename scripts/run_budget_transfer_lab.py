"""Run the staged budget/teacher/low-FPR development study on lab RTX4090D.

Default screen stops on F/R/C before KD or V. Legacy pilot/full are explicit
historical comparisons, not the default recommendation. Dry-run reads JSON/source metadata only; fits
and complete-only resume require an unchanged clean checkout and bound inputs.
No prepared-data rebuild, download, historical calibration or test lookup.
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

from ids.common import code_fingerprint, read_json, sha256_file, utc_now  # noqa: E402
from run_risk_lab import (  # noqa: E402
    REQUIRED_GPU, _activate_cuda_zero, _assert_no_links, _dataset_record,
    _inventory, _verify_zip, check_environment, require_lab_gpu,
)
from run_upgrade_lab import _v3_record  # noqa: E402

PREFLIGHT = "BUDGET_TRANSFER_LAB_PREFLIGHT.json"
COMPLETE = "BUDGET_TRANSFER_COMPLETE.json"
SCREEN_COMPLETE = "TRANSFER_SCREEN_COMPLETE.json"
V3_COMPLETE = "RISK_RESEARCH_COMPLETE.json"
PACKAGE_MANIFEST = "BUDGET_TRANSFER_PACKAGE_MANIFEST.sha256"


def validate_config(config):
    from ids.budget_transfer import validate_config as validate
    return validate(config)


def run_transfer_research(v3_dir, data_dir, config, output, **kwargs):
    from ids.budget_transfer import run_transfer_research as run
    return run(v3_dir, data_dir, config, output, **kwargs)


def verify_transfer(output):
    from ids.budget_transfer import verify_transfer as verify
    return verify(output)


def run_transfer_screen(v3_dir, data_dir, config, output, **kwargs):
    from ids.transfer_screen import run_transfer_screen as run
    return run(v3_dir, data_dir, config, output, **kwargs)


def verify_transfer_screen(output):
    from ids.transfer_screen import verify_transfer_screen as verify
    return verify(output)


def require_screen_dependencies():
    from ids.native_trees import _catboost
    _catboost()


def _profile_path(profile):
    if profile not in {"screen", "pilot", "full"}:
        raise ValueError("profile must be screen, pilot or full")
    return _assert_no_links(REPO / "configs" / f"budget-transfer-{profile}.yaml")


def load_profile(profile):
    configs = {}
    for name in ("screen", "pilot", "full"):
        cfg = yaml.safe_load(_profile_path(name).read_text(encoding="utf-8"))
        validate_config(cfg)
        if (cfg["seeds"] != ([17, 29, 43] if name == "full" else [17])
                or cfg["warmup_epochs"] != 40 or cfg["macro_steps"] != 40):
            raise ValueError("Laboratory profiles require pinned seeds and the same 40-epoch/40-step budgets")
        configs[name] = cfg
    if ({k: v for k, v in configs["pilot"].items() if k != "seeds"}
            != {k: v for k, v in configs["full"].items() if k != "seeds"}):
        raise ValueError("Pilot/full may differ only in seeds, not cohorts or training budgets")
    _profile_path(profile)  # Reject an unknown requested profile, too.
    return configs[profile]


def _source_identity(source):
    if (source.get("dirty") is not False
            or not isinstance(source.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", source["commit"])):
        raise ValueError("Laboratory study requires a clean committed Git checkout; preserve current work")
    return {"commit": source["commit"], "source_sha256": source["source_sha256"]}


def _verify_complete(output, *, profile="pilot"):
    root = _assert_no_links(output)
    marker = SCREEN_COMPLETE if profile == "screen" else COMPLETE
    if not _assert_no_links(root / marker).is_file():
        raise FileExistsError("Incomplete budget-transfer output is preserved; interrupted-fit resume is unsupported")
    done = verify_transfer_screen(root) if profile == "screen" else verify_transfer(root)
    expected_status = "screen_complete" if profile == "screen" else "development_complete"
    if (done.get("schema_version") != 1 or done.get("status") != expected_status
            or done.get("reserved_partitions_read") != []
            or not isinstance(done.get("files"), dict) or not done["files"]):
        raise ValueError("Budget-transfer completion/schema/reserved-partition scope is invalid")
    return done


def _verify_counts(output, done, config, *, profile="pilot"):
    """Every registered student runs; only the exact full-input alias is skipped."""
    from ids.budget_transfer import METHODS
    seeds = config["seeds"]
    n = len(seeds)
    teacher, warmup, students = (done.get(k) for k in ("n_teacher_fits", "n_warmup_fits", "n_student_fits"))
    if (any(type(x) is not int or x < 0 for x in (teacher, warmup, students))
            or teacher != 2*n or warmup != 4*n or students != (4 if profile == "screen" else 17)*n):
        raise ValueError("Budget-transfer completion contradicts the planned fit counts")
    summary = read_json(Path(output) / "summary.json")
    if (summary.get("teacher_fits") != teacher or summary.get("warmup_fits") != warmup
            or summary.get("student_fits") != students or len(summary.get("runs", [])) != students
            or summary.get("independent_confirmation") is not False):
        raise ValueError("Budget-transfer summary/counts or development scope mismatch")
    observed = [(r.get("seed"), r.get("budget"), r.get("method")) for r in summary["runs"]]
    skipped = [(r.get("seed"), r.get("budget"), r.get("method")) for r in summary.get("skipped", [])]
    if profile == "screen":
        if done.get("n_native_tree_fits") != 2*n or summary.get("native_tree_fits") != 2*n:
            raise ValueError("Screen native-tree fit count changed")
        expected = {(seed, budget, method) for seed in seeds for budget in ("all", "reduced")
                    for method in ("hard", "direct_pauc")}
        if (len(set(observed)) != len(observed) or set(observed) != expected or skipped
                or done.get("validation_loaded") is not False or summary.get("validation_loaded") is not False
                or summary.get("automatic_expansion") is not False):
            raise ValueError("Screen condition inventory or F/R/C-only scope changed")
        return
    expected = {(seed, budget, method) for seed in seeds for budget in ("all", "reduced") for method in METHODS}
    if len(set(observed + skipped)) != len(observed + skipped) or set(observed + skipped) != expected:
        raise ValueError("Budget-transfer condition inventory is missing, duplicated or unexpected")
    aliases = {(seed, "all", "direct_rank_matched") for seed in seeds}
    if set(skipped) != aliases or set(observed) != expected - aliases:
        raise ValueError("Budget-transfer full condition matrix or exact-alias-only policy changed")


def package_completed_environment(output, zip_path, *, resume=False, profile="pilot"):
    root, target = _assert_no_links(output), _assert_no_links(zip_path)
    if target.is_relative_to(root):
        raise ValueError("Budget-transfer ZIP must be outside the sealed environment")
    _verify_complete(root, profile=profile)
    inventory = _inventory(root)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", root.name) or root.name in {".", ".."}:
        raise ValueError("Archive root must be a safe component")
    payload = {root.name + "/" + name: digest for name, digest in inventory.items()}
    content = "".join(f"{digest}  {name}\n" for name, digest in sorted(payload.items())).encode("utf-8")
    expected = {**payload, PACKAGE_MANIFEST: hashlib.sha256(content).hexdigest()}
    sidecar = _assert_no_links(target.with_name(target.name + ".sha256"))
    if target.exists() or sidecar.exists():
        if not resume or not target.is_file() or not sidecar.is_file():
            raise FileExistsError("Existing/partial budget-transfer package is preserved; refusing overwrite")
        if sidecar.read_text(encoding="ascii").strip() != sha256_file(target):
            raise ValueError("Existing budget-transfer package sidecar mismatch")
        _verify_zip(target, expected)
        return {"zip": str(target), "sha256": sha256_file(target), "status": "verified_existing"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in inventory:
            archive.write(root / name, root.name + "/" + name)
        archive.writestr(PACKAGE_MANIFEST, content)
    _verify_zip(target, expected)
    _verify_complete(root, profile=profile)
    if _inventory(root) != inventory:
        raise ValueError("Budget-transfer environment changed during packaging")
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
            raise FileExistsError("Existing/partial budget-transfer preflight receipt is preserved")
        if target.read_bytes() != content or sidecar.read_text(encoding="ascii").strip() != digest:
            raise ValueError("Preflight receipt differs from saved original bytes")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        with sidecar.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(digest + "\n")
    return {"file": str(target), "sha256": digest}


def run_lab(*, work_dir, unsw_v3_dir=None, cicids_v3_dir=None, unsw_data_dir=None,
            cicids_data_dir=None, profile="screen", dry_run=False, resume=False, threads=None):
    pairs = {name: (v3, data) for name, v3, data in
             (("unsw", unsw_v3_dir, unsw_data_dir), ("cicids", cicids_v3_dir, cicids_data_dir))}
    if not any(v3 is not None or data is not None for v3, data in pairs.values()):
        raise ValueError("Provide at least one environment with both V3 and prepared-data paths")
    if any((v3 is None) != (data is None) for v3, data in pairs.values()):
        raise ValueError("Each environment requires both its source V3 and prepared data")
    config = load_profile(profile)
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        config = {**config, "threads": threads}
        validate_config(config)
    source = code_fingerprint()
    source_id = _source_identity(source)
    environment = check_environment()
    if profile == "screen":
        require_screen_dependencies()
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
    n = len(config["seeds"])
    expected = {"teacher_fits": 2*n, "warmup_fits": 4*n,
                "student_fits": (4 if profile == "screen" else 17)*n,
                "total_fits": (12 if profile == "screen" else 23)*n}
    if profile == "screen":
        expected["native_tree_fits"] = 2*n
    binding = {"profile": profile, "config": config, "config_path": str(_profile_path(profile)),
               "config_file_sha256": sha256_file(_profile_path(profile)), "inputs": inputs,
               "source": source_id, "environment": stable_environment, "interpreter": interpreter,
               "execution_mode": "dry_run" if dry_run else "lab_run",
               "device": "cuda", "cuda_index": 0, "required_gpu": REQUIRED_GPU}
    preflight = {"schema_version": 1, "purpose": "budget_transfer_lab_preflight", "created_at": utc_now(),
                 "binding": binding, "environment": environment, "selected_gpu": gpu, "source": source,
                 "expected_per_environment": expected,
                 "expected_all_environments": {key: value * len(inputs) for key, value in expected.items()},
                 "scope": ("F/R/C screen only; no V loading, KD or automatic expansion" if profile == "screen"
                           else "historical comparison; complete V3 cohorts; V after frozen selection; serial environments"),
                 "dry_run_scope": "source/config/environment, prepared manifest and V3 completion/plan JSON metadata only",
                 "resume_scope": "verified complete outputs only; no interrupted-fit resume",
                 "pilot_scope": "legacy matrix, explicit opt-in; one seed with full 40/40 budget",
                 "screen_scope": "one seed; two HGB teachers, four warmups, four hard/direct continuations, two native CatBoost trees; F/R/C only",
                 "reserved_partitions_read": []}
    plan_path = _assert_no_links(work / PREFLIGHT)
    if plan_path.exists():
        if not resume:
            raise FileExistsError("Budget-transfer work has a preflight; use identical-input --resume only")
        if read_json(plan_path).get("binding") != binding:
            raise ValueError("Budget-transfer source/configuration/interpreter/hardware/data/V3/mode binding changed")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError("New budget-transfer work directory must be empty")
        work.mkdir(parents=True, exist_ok=True)
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(preflight, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    results = {}
    for name, record in inputs.items():
        output = _assert_no_links(work / ("budget-transfer-" + name))
        if dry_run:
            results[name] = {"status": "plan_only", "output": str(output), "reserved_partitions_read": []}
            continue
        if output.exists():
            if not resume:
                raise FileExistsError("Existing budget-transfer output is preserved; refusing overwrite")
            _verify_complete(output, profile=profile)
        run = run_transfer_screen if profile == "screen" else run_transfer_research
        run(record["v3"]["v3_dir"], record["data"]["data_dir"], copy.deepcopy(config), output,
            device="cuda", resume=output.exists())
        done = _verify_complete(output, profile=profile)
        _verify_counts(output, done, config, profile=profile)
        if _source_identity(code_fingerprint()) != source_id:
            raise ValueError("Budget-transfer source changed during execution; preserve outputs without accepting package")
        if sha256_file(Path(record["v3"]["v3_dir"]) / V3_COMPLETE) != record["v3"]["completion_sha256"]:
            raise ValueError("Original V3 completion changed during the budget-transfer run")
        if _dataset_record(name, record["data"]["data_dir"]) != record["data"]:
            raise ValueError("Prepared manifest changed during the budget-transfer run")
        results[name] = {"status": "verified_screen_complete" if profile == "screen" else "verified_development_complete",
            "package": package_completed_environment(output, work / "packages" / ("BUDGET_TRANSFER_" + name.upper() + ".zip"),
                resume=resume, profile=profile)}
    receipt = None if dry_run else _copy_saved_preflight(plan_path, work / "packages", resume=resume)
    return {"status": "budget_transfer_lab_plan_only" if dry_run else
            ("budget_transfer_lab_screen_complete" if profile == "screen" else "budget_transfer_lab_development_complete"),
            "profile": profile, "work_dir": str(work), "expected_per_environment": expected,
            "expected_all_environments": {key: value * len(inputs) for key, value in expected.items()},
            "environments": results, "preflight_receipt": receipt, "reserved_partitions_read": []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("screen", "pilot", "full"), default="screen")
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
    except (ValueError, FileExistsError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Budget-transfer laboratory preflight/run rejected: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
