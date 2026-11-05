"""Run/package the frozen UNSW H/K/S/M lab pilot; never expand the matrix."""
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
    _activate_cuda_zero, _assert_no_links, _inventory, _verify_zip,
    check_environment, require_lab_gpu,
)

DEFAULT_LAB_ROOT = Path(r"D:\zhoumeng\Graduate")
PROTOCOL = REPO / "configs/unsw-small-kd-minimal-20261105.json"
SCREEN_COMPLETE = "TRANSFER_SCREEN_COMPLETE.json"
PACKAGE_MANIFEST = "UNSW_KD_PACKAGE_MANIFEST.sha256"


def preflight_unsw_kd(screen_dir, data_dir, **kwargs):
    from ids.unsw_kd_experiment import preflight_unsw_kd as run
    return run(screen_dir, data_dir, **kwargs)


def run_unsw_kd(screen_dir, data_dir, output, **kwargs):
    from ids.unsw_kd_experiment import run_unsw_kd as run
    return run(screen_dir, data_dir, output, **kwargs)


def _unique(paths, kind):
    paths = sorted(set(paths))
    if len(paths) != 1:
        found = "\n".join(str(p) for p in paths) or "(none)"
        raise ValueError(f"Need exactly one {kind}; specify its path explicitly. Found:\n{found}")
    return paths[0]


def resolve_inputs(lab_root, screen_dir=None, data_dir=None):
    """Discover only metadata bound to the frozen UNSW screen, never split rows."""
    lab = _assert_no_links(lab_root)
    cfg = read_json(PROTOCOL)
    if screen_dir is None:
        search = lab / "experiments"
        candidates = []
        if search.is_dir():
            for marker in search.rglob(SCREEN_COMPLETE):
                marker = _assert_no_links(marker)
                if marker.is_file() and sha256_file(marker) == cfg["data"]["screen_complete_sha256"]:
                    candidates.append(marker.parent)
        screen = _unique(candidates, "frozen UNSW screen directory")
    else:
        screen = _assert_no_links(screen_dir)
    marker = _assert_no_links(screen / SCREEN_COMPLETE)
    if not marker.is_file() or sha256_file(marker) != cfg["data"]["screen_complete_sha256"]:
        raise ValueError("Screen completion SHA differs from the frozen UNSW protocol")
    if data_dir is not None:
        data = _assert_no_links(data_dir)
    else:
        expected = read_json(_assert_no_links(screen / "plan.json"))["binding"]["data"]
        candidates = []
        for base in (lab / "data", lab / "experiments"):
            if not base.is_dir():
                continue
            for manifest in base.rglob("manifest.json"):
                manifest = _assert_no_links(manifest)
                if manifest.is_file() and read_json(manifest) == expected:
                    candidates.append(manifest.parent)
        data = _unique(candidates, "matching UNSW prepared directory")
    if not data.is_dir():
        raise FileNotFoundError(f"Prepared directory does not exist: {data}")
    return screen, data


def _source():
    result = code_fingerprint()
    if (result.get("dirty") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", result.get("commit") or "")):
        raise ValueError("Use a clean committed checkout; existing work is preserved")
    return result


def _payload_inventory(root):
    """Exclude only this wrapper's package destination; keep partial run artifacts."""
    result = {}
    for child in sorted(root.iterdir()):
        child = _assert_no_links(child)
        if child.name == "packages":
            continue
        if child.is_file():
            result[child.name] = sha256_file(child)
        elif child.is_dir():
            result.update({child.name + "/" + k: v for k, v in _inventory(child).items()})
        else:
            raise ValueError(f"Cannot package non-regular entry: {child}")
    return dict(sorted(result.items()))


def package_work(work_dir):
    """Package successful, rejected, preflight, or interrupted work without fitting."""
    root = _assert_no_links(work_dir)
    if not root.is_dir() or not (root / "LAB_REQUEST.json").is_file():
        raise ValueError("Package-only requires a work directory created by this lab wrapper")
    inventory = _payload_inventory(root)
    if not inventory:
        raise ValueError("No run artifacts to package")
    packages = _assert_no_links(root / "packages")
    packages.mkdir(exist_ok=True)
    token = utc_now().replace("-", "").replace(":", "").replace("+", "").replace(".", "") + "-" + uuid.uuid4().hex[:8]
    target = packages / ("UNSW_KD_" + token + ".zip")
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


def run_lab(*, lab_root=DEFAULT_LAB_ROOT, screen_dir=None, data_dir=None,
            work_dir=None, preflight_only=False, frozen_targets=None):
    lab = _assert_no_links(lab_root)
    if work_dir is None:
        token = utc_now().replace("-", "").replace(":", "").replace("+", "").replace(".", "")
        work_dir = lab / "experiments" / ("unsw-kd-" + ("check-" if preflight_only else "")
                                         + token + "-" + uuid.uuid4().hex[:8])
    work = _assert_no_links(work_dir)
    if work.exists():
        raise FileExistsError("Existing work is preserved; no overwrite or automatic refit")
    # Resolve input locations before creating a work directory nested in old inputs.
    screen, data = resolve_inputs(lab, screen_dir, data_dir)
    if any(work.is_relative_to(p) or p.is_relative_to(work) for p in (screen, data, REPO)):
        raise ValueError("Work and immutable inputs/source must be non-nested")
    work.mkdir(parents=True, exist_ok=False)
    write_json(work / "LAB_REQUEST.json", {
        "created_at": utc_now(), "screen_dir": str(screen), "data_dir": str(data),
        "work_dir": str(work), "preflight_only": bool(preflight_only),
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "frozen_targets": None if frozen_targets is None else str(frozen_targets),
        "new_teacher_fits": 0, "new_warmup_fits": 0,
        "planned_student_continuations": 0 if preflight_only else 4,
        "reserved_partitions_read": [], "automatic_expansion": False,
    })
    print(f"Work directory: {work}", flush=True)
    try:
        source = _source()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        environment = check_environment()
        gpu = require_lab_gpu(environment)
        write_json(work / "LAB_ENVIRONMENT.json", {
            "source": source, "environment": environment, "selected_gpu": gpu,
            "python_executable": sys.executable, "python_prefix": sys.prefix,
        })
        options = {"protocol_path": PROTOCOL}
        if frozen_targets is not None:
            options["frozen_targets_path"] = _assert_no_links(frozen_targets)
        if preflight_only:
            result = preflight_unsw_kd(screen, data, **options)
            status = "preflight_complete"
        else:
            _activate_cuda_zero()
            result = run_unsw_kd(screen, data, work / "experiment", device="cuda",
                                 measure_cpu=True, **options)
            if result.get("status") != "unsw_kd_complete" or result.get("student_fits") != 4:
                raise ValueError("Runner did not complete the frozen four student continuations")
            status = "experiment_complete"
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
    parser.add_argument("--screen-dir", type=Path, help="explicit frozen budget-transfer-unsw directory")
    parser.add_argument("--data-dir", type=Path, help="explicit existing UNSW prepared directory")
    parser.add_argument("--work-dir", type=Path, help="new non-existing work directory")
    parser.add_argument("--preflight-only", action="store_true", help="check bound inputs, no training or timing")
    parser.add_argument("--frozen-targets", type=Path,
                        help="optional sealed target NPZ; hash must match the protocol")
    parser.add_argument("--package-only", type=Path, metavar="WORK_DIR",
                        help="package existing successful/partial work, no trainer or CUDA check")
    args = parser.parse_args(argv)
    if args.package_only is not None:
        if any(x is not None for x in (args.screen_dir, args.data_dir, args.work_dir,
                                      args.frozen_targets)) or args.preflight_only:
            parser.error("--package-only cannot be combined with run options")
        print(json.dumps(package_work(args.package_only), ensure_ascii=False, indent=2))
        return 0
    run_lab(lab_root=args.lab_root, screen_dir=args.screen_dir, data_dir=args.data_dir,
            work_dir=args.work_dir, preflight_only=args.preflight_only,
            frozen_targets=args.frozen_targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
