"""Package a completed stage-3/4 experiment with verified SHA-256 inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_run(work_dir: Path, output: Path):
    work_dir, output = work_dir.resolve(), output.resolve()
    if not work_dir.is_dir() or output.is_relative_to(work_dir):
        raise ValueError("Choose a work directory and a ZIP output outside it")
    if output.exists():
        raise FileExistsError("Output already exists; no archive will be overwritten")
    if (work_dir / "STAGE4_COMPLETE.json").is_file():
        return package_stage4(work_dir, output)
    completion = json.loads((work_dir / "STAGE3_COMPLETE.json").read_text(encoding="utf-8"))
    required = {"STAGE3_PLAN.json", "reports/distillation-study.json", "reports/distillation-study.html",
                "reports/validation-records.html"}
    if completion.get("phase") != "validation_development" or not required.issubset(completion.get("files", {})):
        raise ValueError("Requires a completed development workflow")
    for name, expected in completion["files"].items():
        path = work_dir / name
        if not path.resolve().is_relative_to(work_dir) or digest(path) != expected:
            raise ValueError(f"Completed artifact changed: {name}")
    plan = json.loads((work_dir / "STAGE3_PLAN.json").read_text(encoding="utf-8"))
    study = json.loads((work_dir / "reports" / "distillation-study.json").read_text(encoding="utf-8"))
    suite = plan["identity"]["suite"]
    expected = len(suite["models"]) * len(suite["seeds"])
    if (completion.get("n_runs") != expected or study.get("n_runs") != expected
            or plan.get("expected_runs") != expected or expected < 1
            or completion.get("benchmark_rounds") != suite["benchmark_rounds"]
            or len(study.get("groups", [])) != len(suite["models"])):
        raise ValueError("Completion, plan and study coverage disagree")
    for group in study["groups"]:
        if (group.get("seeds") != sorted(suite["seeds"])
                or len(group.get("cpu_benchmarks", [])) != suite["benchmark_rounds"]):
            raise ValueError("Study lacks complete seed/benchmark coverage")
        if suite["benchmark_rounds"] > 1 and not any(
                b.get("status") == "complete" and b.get("seeds") == sorted(suite["seeds"])
                for b in group.get("repeated_benchmarks", [])):
            raise ValueError("Study lacks complete repeated CPU measurements")
    files = sorted(p for p in work_dir.rglob("*") if p.is_file())
    inventory = {}
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(work_dir):
            raise ValueError("Do not package linked files outside the work directory")
        name = path.relative_to(work_dir).as_posix()
        if name == "MANIFEST.sha256":
            raise ValueError("Work directory must not contain a conflicting MANIFEST.sha256")
        if name not in completion["files"] and name != "STAGE3_COMPLETE.json" and path.suffix != ".log":
            raise ValueError(f"Unsealed additional file; keep extra notes outside work directory: {name}")
        inventory[name] = digest(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            name = path.relative_to(work_dir).as_posix()
            archive.write(path, f"{work_dir.name}/{name}")
        manifest = "".join(f"{value}  {name}\n" for name, value in inventory.items())
        archive.writestr(f"{work_dir.name}/MANIFEST.sha256", manifest.encode("utf-8"))
    with zipfile.ZipFile(output) as archive:
        for name, expected in inventory.items():
            h = hashlib.sha256()
            with archive.open(f"{work_dir.name}/{name}") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("File changed during packaging; archive is not verified")
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "\n", encoding="ascii")
    print(f"Verified ZIP: {output}\nSHA-256: {checksum}\nFiles: {len(inventory)} plus MANIFEST.sha256")
    return output


def package_stage4(work_dir, output):
    # The standalone entry must work even without an editable package install.
    import sys
    import uuid
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ids.stage4 import verify_completion
    completion = verify_completion(work_dir)
    inventory = {}
    files = sorted(p for p in work_dir.rglob("*") if p.is_file())
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(work_dir):
            raise ValueError("Linked files are not allowed in the experiment package")
        name = path.relative_to(work_dir).as_posix()
        if name not in completion["files"] and name != "STAGE4_COMPLETE.json" and path.suffix != ".log":
            raise ValueError(f"Unsealed additional file: {name}")
        inventory[name] = digest(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary archive leaves any hard-interrupted attempt untouched
    # and permits --resume without a manual cleanup step.
    temporary = output.with_name(f"{output.name}.{uuid.uuid4().hex}.partial")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                archive.write(path, f"{work_dir.name}/{path.relative_to(work_dir).as_posix()}")
            archive.writestr(f"{work_dir.name}/MANIFEST.sha256",
                             "".join(f"{value}  {name}\n" for name, value in inventory.items()).encode("utf-8"))
        with zipfile.ZipFile(temporary) as archive:
            for name, expected_hash in inventory.items():
                h = hashlib.sha256()
                with archive.open(f"{work_dir.name}/{name}") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        h.update(block)
                if h.hexdigest() != expected_hash:
                    raise ValueError("Stage-4 file changed during packaging")
        if output.exists():
            raise FileExistsError("Output ZIP appeared during packaging")
        temporary.rename(output)
    except BaseException:
        # Only the exact uncommitted temporary file created by this call is removed.
        temporary.unlink(missing_ok=True)
        raise
    checksum = digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(checksum + "\n", encoding="ascii")
    print(f"Verified ZIP: {output}\nSHA-256: {checksum}\nFiles: {len(inventory)} plus MANIFEST.sha256")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        package_run(args.work_dir, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
