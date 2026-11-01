"""One entry: preflight, fixed data preparation, two studies, verified ZIP."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import tomllib
import zipfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def check_environment(device):
    """Check only project requirements, without upgrading an existing lab venv."""
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
    from ids.common import environment_snapshot
    value = environment_snapshot()
    if device == "cuda" and not value.get("torch", {}).get("cuda_available"):
        raise ValueError("CUDA requested but unavailable; no silent CPU fallback")
    return value


def preflight(work, device, config, resume):
    from ids.common import canonical_hash, code_fingerprint, read_json, sha256_file, write_json
    from ids.data import load_partition, make_synthetic_dataset, prepare_dataset
    from ids.models import train_model
    environment = check_environment(device)
    environment.pop("recorded_at", None)
    identity = {"source_sha256": code_fingerprint()["source_sha256"],
                "environment_sha256": canonical_hash(environment), "device": device}
    target = work / "preflight.json"
    if target.exists():
        checksum = target.with_suffix(".sha256")
        saved = read_json(target)
        if (not resume or not checksum.is_file() or checksum.read_text().strip() != sha256_file(target)
                or saved["identity"] != identity):
            raise ValueError("Preflight identity changed; preserve the previous work directory")
        return saved
    raw = work / "data" / "preflight-raw"
    prepared = work / "data" / "preflight-prepared"
    if not (prepared / "manifest.json").exists():
        if not (raw / "synthetic_source.json").exists():
            make_synthetic_dataset(raw)
        prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv",
                        prepared, seed=17, data_kind="synthetic")
    train, val = load_partition(prepared, "train"), load_partition(prepared, "validation")
    model = {"model": "mlp", "features": {"method": "stable_mutual_info", "k": 4,
             "selection_seed": 20261101, "n_resamples": 3},
             "hyperparameters": {"hidden_sizes": [8, 4], "epochs": 2, "patience": 2,
                                  "batch_size": 128, "num_threads": config["threads"]}}
    bundle = train_model(train, val, model, device=device, seed=17)
    scores = bundle.predict_scores(val)
    import numpy as np
    if len(scores) != len(val) or not np.isfinite(scores).all():
        raise ValueError("Synthetic device preflight prediction failed")
    result = {"identity": identity, "environment": environment, "status": "complete",
              "synthetic_device_check": {"device": device, "n_rows": len(val), "finite_scores": True},
              "scope": "engineering check only; not research evidence"}
    write_json(target, result)
    target.with_suffix(".sha256").write_text(sha256_file(target) + "\n", encoding="ascii")
    return result


def prepare_inputs(args, config):
    from ids.cicids import make_synthetic_cicids_dataset, prepare_cicids_dataset, load_cic_prepared_verified
    from ids.common import read_json, sha256_file
    from ids.data import make_synthetic_dataset, prepare_dataset
    from ids.runner import verify_development_inputs
    base = args.work_dir / "data"
    if args.smoke:
        unsw = base / "unsw-prepared"
        raw = base / "unsw-raw"
        if not (unsw / "manifest.json").exists():
            if not (raw / "synthetic_source.json").exists():
                make_synthetic_dataset(raw)
            prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv",
                            unsw, seed=17, data_kind="synthetic")
    else:
        if args.unsw_data_dir is None:
            raise ValueError("Provide --unsw-data-dir pointing to the previous real prepared dataset")
        unsw = args.unsw_data_dir.resolve()
    manifest = verify_development_inputs(unsw)
    if not args.smoke and sha256_file(unsw / "manifest.json") != config["dataset"]["expected_unsw_manifest_sha256"]:
        raise ValueError("UNSW prepared identity differs from the accepted Stage-2/3 archive; use the original prepared directory")
    if (manifest["data_kind"] == "synthetic") != args.smoke:
        raise ValueError("UNSW data kind does not match smoke/research profile")
    cic = base / "cicids-prepared"
    if not (cic / "manifest.json").exists():
        raw = args.cic_raw_dir or base / "cicids-raw"
        if args.smoke:
            if not (raw / "synthetic_source.json").exists():
                make_synthetic_cicids_dataset(raw)
        elif args.cic_raw_dir is None:
            from fetch_cicids import fetch_cicids
            fetch_cicids(raw)
        prepare_cicids_dataset(raw, cic, seed=config["dataset"]["split_seed"],
                               data_kind="synthetic" if args.smoke else "official_repaired",
                               session_gap_seconds=config["dataset"]["session_gap_seconds"], reuse=args.resume)
    load_cic_prepared_verified(cic, splits=("train", "validation"))
    if (read_json(cic / "manifest.json")["data_kind"] == "synthetic") != args.smoke:
        raise ValueError("CIC data kind does not match smoke/research profile")
    return {"unsw": unsw, "cicids": cic}


def package_completed(work, experiment, resume):
    from ids.common import sha256_file
    from package_run import package_run
    output = work / "TraceLight-Stage4.zip"
    if output.exists():
        checksum = output.with_suffix(".zip.sha256")
        if not resume or not checksum.is_file() or checksum.read_text().strip() != sha256_file(output):
            raise ValueError("Existing ZIP is not verified; preserve it and inspect the error")
        with zipfile.ZipFile(output) as archive:
            if archive.read(f"{experiment.name}/STAGE4_COMPLETE.json") != (experiment / "STAGE4_COMPLETE.json").read_bytes():
                raise ValueError("Existing ZIP belongs to a different completed experiment")
        print("Verified existing ZIP:", output, flush=True)
    else:
        package_run(experiment, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsw-data-dir", type=Path, help="Existing real UNSW prepared directory")
    parser.add_argument("--work-dir", type=Path, required=True, help="New stage-4 container; data and output are kept separate")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", action="store_true", help="Verify and continue the exact same frozen workflow")
    parser.add_argument("--smoke", action="store_true", help="Synthetic engineering validation only; never research results")
    parser.add_argument("--cic-raw-dir", type=Path, help="Optional already downloaded, verified CNS2022 source directory")
    args = parser.parse_args(argv)
    args.work_dir = args.work_dir.expanduser().resolve()
    config_path = REPO / "configs" / ("stage4-smoke.yaml" if args.smoke else "stage4.yaml")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = str(config["threads"])
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        from ids.stage4 import run_stage4, validate_config
        validate_config(config)
        if args.work_dir.exists() and any(args.work_dir.iterdir()) and not args.resume:
            raise FileExistsError("Work directory already exists; use --resume for unchanged inputs")
        print("Stage 4: checking locked environment and a short device fixture", flush=True)
        checked = preflight(args.work_dir, args.device, config, args.resume)
        print("Stage 4: verifying UNSW and preparing the independent CIC dataset", flush=True)
        data_dirs = prepare_inputs(args, config)
        experiment = run_stage4(data_dirs, args.work_dir / "experiment", config,
                                device=args.device, resume=args.resume, preflight=checked)
        output = package_completed(args.work_dir, experiment, args.resume)
        print("Stage 4 COMPLETE. Research data kind:", config["profile"], flush=True)
        print("Report:", experiment / "reports" / "index.html", flush=True)
        print("Return ZIP:", output, flush=True)
        print("Return checksum:", output.with_suffix(".zip.sha256"), flush=True)
        print("Keep the work directory and existing UNSW prepared files for later frozen evaluation.", flush=True)
        return 0
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError, zipfile.BadZipFile) as exc:
        print(f"Stopped: {exc}. Completed evidence was not overwritten.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    if os.environ.get("PYTHONHASHSEED") != "0":
        import subprocess
        environment = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
        raise SystemExit(subprocess.call([sys.executable, "-u", *sys.argv], env=environment))
    raise SystemExit(main())
