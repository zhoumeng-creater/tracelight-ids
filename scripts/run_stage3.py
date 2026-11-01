"""Record the stage-3 protocol, run matched models, then benchmark all three rounds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ids.common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json  # noqa: E402
from ids.runner import validate_suite, verify_run, verify_run_teacher, verify_development_inputs  # noqa: E402


def validate_design(suite):
    validate_suite(suite)
    expected = {"hgb_reference", "mlp_reference", "student_hard", "student_kd"}
    models = suite["models"]
    if len(models) != 4 or {m.get("variant") for m in models} != expected:
        raise ValueError("Stage 3 requires exactly the four declared reference/student variants")
    by_variant = {m["variant"]: m for m in models}
    if any(m.get("features", {}).get("method") != "all" for m in models):
        raise ValueError("Stage 3 retains all 42 raw fields")
    teacher, large, hard, kd = (by_variant[k] for k in ("hgb_reference", "mlp_reference", "student_hard", "student_kd"))
    if teacher["model"] != "hgb" or any(m["model"] != "mlp" for m in (large, hard, kd)):
        raise ValueError("Incorrect stage-3 model families")
    if hard["hyperparameters"] != kd["hyperparameters"]:
        raise ValueError("Hard and KD students must have identical architecture and training parameters")
    if any("distillation" in m for m in (teacher, large, hard)) or "distillation" not in kd:
        raise ValueError("Only student_kd may enable distillation")
    large_hp = {k: v for k, v in large["hyperparameters"].items() if k != "hidden_sizes"}
    small_hp = {k: v for k, v in hard["hyperparameters"].items() if k != "hidden_sizes"}
    if large_hp != small_hp:
        raise ValueError("Large and small MLP training budgets must match")
    distillation = kd["distillation"]
    teacher_config = {k: v for k, v in teacher.items() if k != "variant"}
    if distillation["teacher"] != teacher_config:
        raise ValueError("OOF and reference teacher training rules must match")
    if distillation["temperature"] != 1 or not 0 < distillation["alpha"] <= 1:
        raise ValueError("Stage 3 requires active distillation with temperature 1")


def build_commands(args, suite):
    prefix = [sys.executable, "-m", "ids"]
    root = args.work_dir / "runs" / suite["suite_name"]
    doctor = prefix + ["doctor", "--output", str(args.work_dir / "environment" / "doctor.json")]
    if args.device == "cuda":
        doctor.append("--require-cuda")
    train = prefix + ["suite", "--data-dir", str(args.data_dir), "--config", str(args.config),
                      "--output-root", str(args.work_dir / "runs"), "--device", args.device]
    if args.resume:
        train.append("--resume")
    commands = [doctor, train]
    for measurement_round in range(1, suite["benchmark_rounds"] + 1):
        command = prefix + ["benchmark-suite", "--data-dir", str(args.data_dir),
                            "--run-root", str(root), "--output-dir",
                            str(args.work_dir / "benchmarks" / f"round-{measurement_round}"),
                            "--threads", "1", "--order-seed", str(20261100 + measurement_round),
                            "--measurement-round", str(measurement_round),
                            "--planned-rounds", str(suite["benchmark_rounds"])]
        if args.resume:
            command.append("--resume")
        commands.append(command)
    commands.extend([
        prefix + ["report", "--run-root", str(root), "--output", str(args.work_dir / "reports" / "validation-records.html")],
        prefix + ["study", "--run-root", str(root), "--benchmark-dir", str(args.work_dir / "benchmarks"),
                  "--output", str(args.work_dir / "reports" / "distillation-study.html")],
    ])
    return commands


def prepare_plan(args, suite):
    verify_development_inputs(args.data_dir)
    source = code_fingerprint()
    environment = environment_snapshot()
    environment.pop("recorded_at", None)
    identity = {"suite": suite, "dataset_manifest_sha256": sha256_file(args.data_dir / "manifest.json"),
                "source_sha256": source["source_sha256"], "requested_device": args.device,
                "environment_sha256": canonical_hash(environment)}
    path = args.work_dir / "STAGE3_PLAN.json"
    if path.exists():
        if not args.resume:
            raise FileExistsError("Work directory already has a plan; use --resume for identical inputs")
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError("Stage-3 source, data, configuration, device or environment changed")
        for name, digest in saved["snapshot_files"].items():
            if not (args.work_dir / name).resolve().is_relative_to(args.work_dir) or sha256_file(args.work_dir / name) != digest:
                raise ValueError(f"Protocol snapshot changed: {name}")
        if (args.work_dir / "STAGE3_COMPLETE.json").exists():
            for name, digest in read_json(args.work_dir / "STAGE3_COMPLETE.json")["files"].items():
                if not (args.work_dir / name).resolve().is_relative_to(args.work_dir) or sha256_file(args.work_dir / name) != digest:
                    raise ValueError(f"Completed artifact changed: {name}")
            print("Stage 3 already complete; verified saved outputs. No commands relaunched.")
            return False
        return True
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError("New stage-3 work directory must be empty")
    snapshots = {}
    for name, digest in source["files"].items():
        target = args.work_dir / "protocol" / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
        if sha256_file(target) != digest:
            raise ValueError("Source changed while snapshotting; restart in a new directory")
        snapshots[target.relative_to(args.work_dir).as_posix()] = digest
    for name in ("STAGE3_PROTOCOL.md", "STAGE3_LAB.md", "DATA.md", "PROTOCOL.md"):
        target = args.work_dir / "protocol" / name
        shutil.copyfile(REPO / "docs" / name, target)
        snapshots[target.relative_to(args.work_dir).as_posix()] = sha256_file(target)
    write_json(path, {"schema_version": 1, "created_at": utc_now(), "identity": identity,
                      "source": source, "environment": environment, "snapshot_files": snapshots,
                      "expected_runs": len(suite["models"]) * len(suite["seeds"]),
                      "benchmark_rounds": suite["benchmark_rounds"],
                      "scope": "development train/validation; calibration and final test evaluation not requested"})
    return True


def finish(args, suite):
    root = args.work_dir / "runs" / suite["suite_name"]
    for i, model in enumerate(suite["models"]):
        for seed in suite["seeds"]:
            run = root / f"{i:02d}-{model['model']}-s{seed}"
            verify_run(run)
            if model.get("distillation", {}).get("alpha", 0) > 0:
                verify_run_teacher(run, root / "teachers" / f"{i:02d}-oof-s{seed}", args.data_dir)
    study = read_json(args.work_dir / "reports" / "distillation-study.json")
    expected = len(suite["models"]) * len(suite["seeds"])
    if (study["n_runs"] != expected or len(study["groups"]) != len(suite["models"])
            or any(g["seeds"] != sorted(suite["seeds"]) for g in study["groups"])):
        raise ValueError("Incomplete study or inconsistent environments/configurations across seeds")
    for group in study["groups"]:
        rounds = group["cpu_benchmarks"]
        if len(rounds) != suite["benchmark_rounds"] or any(
                b["seeds"] != sorted(suite["seeds"]) for b in rounds):
            raise ValueError("Incomplete CPU measurement rounds")
        actual = {b["spec"].get("measurement_round", 1) for b in rounds}
        if actual != set(range(1, suite["benchmark_rounds"] + 1)):
            raise ValueError("CPU measurement round identities differ from plan")
        if suite["benchmark_rounds"] > 1:
            repeated = group.get("repeated_benchmarks", [])
            if (len(repeated) != 1 or repeated[0]["status"] != "complete"
                    or repeated[0]["seeds"] != sorted(suite["seeds"])):
                raise ValueError("CPU rounds must share identical environment/settings and cover every seed")
    files = {p.relative_to(args.work_dir).as_posix(): sha256_file(p)
             for p in sorted(args.work_dir.rglob("*"))
             if p.is_file() and p.name != "STAGE3_COMPLETE.json" and p.suffix != ".log"}
    write_json(args.work_dir / "STAGE3_COMPLETE.json", {
        "schema_version": 1, "finished_at": utc_now(), "n_runs": expected,
        "benchmark_rounds": suite["benchmark_rounds"], "files": files,
        "phase": "validation_development", "final_evaluation_performed": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "distillation.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark-rounds", type=int, help="Must match the selected configuration")
    args = parser.parse_args(argv)
    args.data_dir, args.work_dir, args.config = (p.expanduser().resolve() for p in (args.data_dir, args.work_dir, args.config))
    try:
        suite = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        validate_design(suite)
        if type(suite.get("benchmark_rounds")) is not int or suite["benchmark_rounds"] < 1:
            raise ValueError("Configuration needs positive benchmark_rounds")
        if args.benchmark_rounds is not None and args.benchmark_rounds != suite["benchmark_rounds"]:
            raise ValueError("Do not change the planned number of benchmark rounds via command line")
        if not (args.data_dir / "manifest.json").is_file():
            raise ValueError("Prepared manifest.json not found; reuse the existing prepared dataset")
        data_kind = read_json(args.data_dir / "manifest.json")["data_kind"]
        if suite["suite_name"] == "distillation-smoke" and data_kind != "synthetic":
            raise ValueError("Smoke configuration is for synthetic data only")
        commands = build_commands(args, suite)
        if args.dry_run:
            for command in commands:
                print("argv:", repr(command))
            return 0
        if not prepare_plan(args, suite):
            return 0
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(REPO / "src"), PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
                           PYTHONUTF8="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = str(suite["threads"])
        for command in commands:
            print("argv:", repr(command), flush=True)
            subprocess.run(command, cwd=REPO, env=environment, check=True)
        finish(args, suite)
        print("Stage 3 development complete: training, CPU rounds and reports verified. No final evaluation requested.")
        print("Study:", args.work_dir / "reports" / "distillation-study.html")
        return 0
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as exc:
        print(f"Stopped: {exc}. Later phases were not started.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
