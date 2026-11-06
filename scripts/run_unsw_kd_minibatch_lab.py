"""Preflight, run once, or package the frozen UNSW H/K mini-batch validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback
import uuid
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ids.common import code_fingerprint, read_json, sha256_file, utc_now, write_json  # noqa: E402
from run_risk_lab import (  # noqa: E402
    _activate_cuda_zero, _assert_no_links, _verify_zip, check_environment, require_lab_gpu,
)
from run_unsw_diagnostic_lab import (  # noqa: E402
    BINDING, ROUND10_COMPLETE, SCREEN_COMPLETE, _bound_marker, _payload_inventory,
    _resolve_artifact, _unique, _verify_prepared, check_preflight_environment,
)

DEFAULT_LAB_ROOT = Path(r"D:\zhoumeng\Graduate")
PROTOCOL = REPO / "configs/unsw-kd-minibatch-minimal-20261106.json"
ENTRYPOINT = "unsw_kd_minibatch"
PACKAGE_MANIFEST = "UNSW_KD_MINIBATCH_PACKAGE_MANIFEST.sha256"


def preflight_minibatch(screen_dir, round10_dir, data_dir):
    from ids.unsw_kd_minibatch_experiment import preflight_minibatch as run
    return run(screen_dir, round10_dir, data_dir)


def run_minibatch(screen_dir, round10_dir, data_dir, output, *, device="cuda"):
    from ids.unsw_kd_minibatch_experiment import run_minibatch as run
    return run(screen_dir, round10_dir, data_dir, output, device=device)


def resolve_inputs(lab_root, screen_dir=None, round10_dir=None, data_dir=None):
    """Reuse sealed discovery; only the original train file is physically hashed."""
    lab = _assert_no_links(lab_root)
    binding = read_json(BINDING)
    screen = _resolve_artifact(lab, screen_dir, SCREEN_COMPLETE,
                               _bound_marker(binding, "screen", SCREEN_COMPLETE), "frozen UNSW screen directory")
    round10 = _resolve_artifact(lab, round10_dir, ROUND10_COMPLETE,
                                _bound_marker(binding, "round10", ROUND10_COMPLETE), "frozen round10 directory")
    expected = binding["prepared_contract"]["data_manifest"]
    if data_dir is None:
        candidates = []
        for base in (lab / "data", lab / "experiments"):
            if not base.is_dir():
                continue
            for manifest in base.rglob("manifest.json"):
                manifest = _assert_no_links(manifest)
                if not manifest.is_file():
                    continue
                try:
                    matched = read_json(manifest) == expected
                except json.JSONDecodeError:
                    continue
                if matched:
                    candidates.append(manifest.parent)
        data_dir = _unique(candidates, "matching UNSW prepared directory")
    return screen, round10, _verify_prepared(data_dir, expected)


def _source():
    result = code_fingerprint()
    if (result.get("dirty") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", result.get("commit") or "")):
        raise ValueError("Use a clean committed checkout; existing work is preserved")
    return result


def package_work(work_dir):
    """Package complete or partial work without invoking either experimental arm."""
    root = _assert_no_links(work_dir)
    if not root.is_dir() or not (root / "LAB_REQUEST.json").is_file():
        raise ValueError("Package-only requires a work directory created by this lab wrapper")
    request = read_json(_assert_no_links(root / "LAB_REQUEST.json"))
    if request.get("entrypoint") != ENTRYPOINT:
        raise ValueError("Package-only requires an UNSW H/K mini-batch work directory")
    inventory = _payload_inventory(root)
    packages = _assert_no_links(root / "packages")
    packages.mkdir(exist_ok=True)
    token = utc_now().replace("-", "").replace(":", "").replace("+", "").replace(".", "") + "-" + uuid.uuid4().hex[:8]
    target = packages / ("UNSW_KD_MINIBATCH_" + token + ".zip")
    names = {root.name + "/" + name: digest for name, digest in inventory.items()}
    listing = "".join(f"{digest}  {name}\n" for name, digest in sorted(names.items())).encode("utf-8")
    expected = {**names, PACKAGE_MANIFEST: hashlib.sha256(listing).hexdigest()}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in inventory:
            archive.write(root / name, root.name + "/" + name)
        archive.writestr(PACKAGE_MANIFEST, listing)
    _verify_zip(target, expected)
    if _payload_inventory(root) != inventory:
        raise ValueError("Work changed while packaging; preserve the ZIP and original work")
    digest = sha256_file(target)
    with target.with_name(target.name + ".sha256").open("x", encoding="ascii", newline="\n") as stream:
        stream.write(digest + "\n")
    return {"zip": str(target), "sha256": digest, "files": len(inventory)}


def _verify_result(result, *, preflight_only):
    status = "minibatch_preflight_complete" if preflight_only else "unsw_kd_minibatch_complete"
    if result.get("status") != status:
        raise ValueError("Mini-batch validation did not return its complete status")
    counts = {"student_fits": 0 if preflight_only else 2, "teacher_fits": 0,
              "optimizer_updates": 0 if preflight_only else 6720}
    for key, expected in counts.items():
        if type(result.get(key)) is not int or result[key] != expected:
            raise ValueError(f"Frozen mini-batch validation requires {key}={expected}")


def run_lab(*, lab_root=DEFAULT_LAB_ROOT, screen_dir=None, round10_dir=None,
            data_dir=None, work_dir=None, preflight_only=False):
    lab = _assert_no_links(lab_root)
    if work_dir is None:
        token = utc_now().replace("-", "").replace(":", "").replace("+", "").replace(".", "")
        work_dir = lab / "experiments" / ("unsw-kd-minibatch-" + ("check-" if preflight_only else "")
                                         + token + "-" + uuid.uuid4().hex[:8])
    work = _assert_no_links(work_dir)
    if work.exists():
        raise FileExistsError("Existing work is preserved; no overwrite or automatic rerun")
    screen, round10, data = resolve_inputs(lab, screen_dir, round10_dir, data_dir)
    if any(work.is_relative_to(p) or p.is_relative_to(work) for p in (screen, round10, data, REPO)):
        raise ValueError("Work and immutable inputs/source must be non-nested")
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / "LAB_REQUEST.json", {
        "entrypoint": ENTRYPOINT, "created_at": utc_now(),
        "screen_dir": str(screen), "round10_dir": str(round10), "data_dir": str(data),
        "work_dir": str(work), "preflight_only": bool(preflight_only),
        "binding": str(BINDING), "binding_sha256": sha256_file(BINDING),
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "planned_student_fits": 0 if preflight_only else 2, "planned_teacher_fits": 0,
        "planned_optimizer_updates": 0 if preflight_only else 6720,
        "planned_teacher_queries": 0, "planned_endpoint_rc_rows": 0 if preflight_only else 73642,
        "planned_cpu_cost_measurements": 0, "reserved_partitions_read": [],
        "automatic_expansion": False, "automatic_rerun": False,
    })
    print(f"Work directory: {work}", flush=True)
    try:
        source = _source()
        if preflight_only:
            environment = check_preflight_environment()
            gpu = None
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            environment = check_environment()
            gpu = require_lab_gpu(environment)
        write_json(work / "LAB_ENVIRONMENT.json", {
            "source": source, "environment": environment, "selected_gpu": gpu,
            "cuda_checked": not preflight_only,
            "python_executable": sys.executable, "python_prefix": sys.prefix,
        })
        if preflight_only:
            result = preflight_minibatch(screen, round10, data)
        else:
            _activate_cuda_zero()
            result = run_minibatch(screen, round10, data, work / "experiment", device="cuda")
        _verify_result(result, preflight_only=preflight_only)
        status = result["status"]
        if code_fingerprint() != source:
            raise ValueError("Source changed during the run; retain all outputs")
        write_json(work / "LAB_RESULT.json", {"status": status, "result": result,
                                              "completed_at": utc_now()})
    except BaseException as exc:
        write_json(work / "LAB_FAILURE.json", {
            "status": "execution_failed", "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "recorded_at": utc_now(),
        })
        try:
            package = package_work(work)
            print("Failure package: " + json.dumps(package, ensure_ascii=False), flush=True)
        except Exception as package_error:
            print(f"Packaging failed; preserve {work}: {package_error}", file=sys.stderr, flush=True)
        raise
    package = package_work(work)
    print("Return ZIP and SHA256: " + json.dumps(package, ensure_ascii=False), flush=True)
    return {"status": status, "work_dir": str(work), "result": result, "package": package}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, default=DEFAULT_LAB_ROOT)
    parser.add_argument("--screen-dir", type=Path, help="exact frozen budget-transfer-unsw directory")
    parser.add_argument("--round10-dir", type=Path, help="exact frozen four-arm experiment directory")
    parser.add_argument("--data-dir", type=Path, help="original bound UNSW prepared directory")
    parser.add_argument("--work-dir", type=Path, help="new non-existing work directory")
    parser.add_argument("--preflight-only", action="store_true", help="verify inputs only; no CUDA, inference, or training")
    parser.add_argument("--package-only", type=Path, metavar="WORK_DIR",
                        help="package existing complete/partial mini-batch work; no experiment or GPU check")
    args = parser.parse_args(argv)
    if args.package_only is not None:
        if any(x is not None for x in (args.screen_dir, args.round10_dir, args.data_dir,
                                      args.work_dir)) or args.preflight_only:
            parser.error("--package-only cannot be combined with run options")
        print(json.dumps(package_work(args.package_only), ensure_ascii=False, indent=2))
        return 0
    run_lab(lab_root=args.lab_root, screen_dir=args.screen_dir, round10_dir=args.round10_dir,
            data_dir=args.data_dir, work_dir=args.work_dir, preflight_only=args.preflight_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
