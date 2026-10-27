"""Command-line research workflows; no command trains on the official test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .common import environment_snapshot, read_json, write_json


def build_parser():
    parser = argparse.ArgumentParser(prog="tracelight", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor", help="Record software, CPU, RAM and GPU; no account or environment secrets")
    p.add_argument("--output", type=Path)
    p.add_argument("--require-cuda", action="store_true")
    p = sub.add_parser("synthetic", help="Generate clearly labelled engineering fixture, never research evidence")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-train", type=int, default=600)
    p.add_argument("--n-test", type=int, default=120)
    p = sub.add_parser("prepare", help="Audit CSVs and create immutable train/validation/calibration/test files")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-kind", choices=["official", "mirror", "synthetic"], default="official")
    p.add_argument("--seed", type=int, default=20261027)
    p = sub.add_parser("suite", help="Train development models; reads only train and validation")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("freeze", help="Freeze ONE selected model before opening calibration/test; no training")
    p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("evaluate", help="Calibrate a frozen model, then evaluate sealed test once")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p = sub.add_parser("benchmark", help="CPU single-row latency and separate batch throughput on validation")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=1)
    p = sub.add_parser("report", help="Build offline inspection report from real validation run artifacts")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("predict", help="Predict strict-schema CSV; labels optional; trusted project model only")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = environment_snapshot()
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if args.require_cuda and not result.get("torch", {}).get("cuda_available"):
                raise ValueError("CUDA is required but unavailable; inspect driver and PyTorch wheel")
        elif args.command == "synthetic":
            from .data import make_synthetic_dataset
            files = make_synthetic_dataset(args.output, args.n_train, args.n_test)
            print("Synthetic engineering fixture only:", *files, sep="\n")
        elif args.command == "prepare":
            from .data import prepare_dataset
            result = prepare_dataset(args.train_csv, args.test_csv, args.output,
                                     seed=args.seed, data_kind=args.data_kind)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "suite":
            from .runner import run_suite
            config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
            run_suite(args.data_dir, config, args.output_root, args.device, args.resume)
        elif args.command == "freeze":
            from .runner import freeze_run
            print(json.dumps(freeze_run(args.run_dir), indent=2))
        elif args.command == "evaluate":
            from .runner import evaluate_frozen
            print(json.dumps(evaluate_frozen(args.run_dir, args.data_dir), indent=2))
        elif args.command == "benchmark":
            from .benchmark import benchmark_predict
            from .data import load_partition
            from .models import load_bundle
            from .runner import verify_run
            from .common import sha256_file
            meta = verify_run(args.run_dir)
            if meta["dataset_manifest_sha256"] != sha256_file(args.data_dir / "manifest.json"):
                raise ValueError("Dataset differs from training")
            if args.output.exists():
                raise FileExistsError("Benchmark output exists")
            bundle = load_bundle(args.run_dir / "model.joblib", device="cpu")
            bundle.num_threads = args.threads
            if bundle.model_name == "rf":
                bundle.estimator.n_jobs = args.threads
            if bundle.model_name == "mlp":
                import torch
                torch.set_num_threads(args.threads)
            result = benchmark_predict(bundle.predict_scores, load_partition(args.data_dir, "validation"),
                                       threads=args.threads)
            result.update({"run_id": args.run_dir.name, "environment": environment_snapshot(),
                           "model_file_bytes": (args.run_dir / "model.joblib").stat().st_size})
            write_json(args.output, result)
        elif args.command == "report":
            from .reporting import build_report
            build_report(args.run_root, args.output)
            print("Validation report:", args.output)
        elif args.command == "predict":
            import numpy as np
            import pandas as pd
            from .data import load_unsw_csv
            from .models import load_bundle
            from .runner import verify_run, verify_evaluation
            verify_run(args.run_dir)
            if args.output.exists():
                raise FileExistsError("Prediction output exists")
            threshold_file = args.run_dir / "evaluation" / "threshold.json"
            stage = "calibration"
            if threshold_file.exists():
                verify_evaluation(args.run_dir)
            else:
                threshold_file = args.run_dir / "validation_threshold.json"
                stage = "validation_development_only"
            threshold = read_json(threshold_file)["threshold"]
            frame = load_unsw_csv(args.input, require_labels=False)
            bundle = load_bundle(args.run_dir / "model.joblib", device="cpu")
            from threadpoolctl import threadpool_limits
            if bundle.model_name == "mlp":
                import torch
                torch.set_num_threads(bundle.num_threads)
            with threadpool_limits(limits=bundle.num_threads):
                scores = bundle.predict_scores(frame)
            out = pd.DataFrame({"sample_id": frame["sample_id"], "score": scores,
                                "predicted_label": (np.asarray(scores, dtype=np.float64) >= threshold).astype(int),
                                "threshold": threshold, "threshold_source": stage,
                                "run_id": args.run_dir.name})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(args.output, index=False)
            print(f"Predicted {len(out)} rows. Scores are not calibrated probabilities. No accuracy inferred.")
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
