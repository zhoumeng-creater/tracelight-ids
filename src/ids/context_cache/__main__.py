"""Full/QCOC cache compilation and raw-CSV lab comparison; no weight downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


INPUT_FORMAT = "tracelight-tabicl-native-input-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, content: dict):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _new_output_dir(path: Path):
    # Research outputs are not overwritten by a repeated command.
    path.mkdir(parents=True, exist_ok=False)


def _read_input(path: Path, role: str):
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("format") != INPUT_FORMAT or data.get("role") != role:
        raise ValueError(f"Expected {INPUT_FORMAT} with role={role}")
    if data.get("data_kind") not in {"synthetic", "development"}:
        raise ValueError("This entry point accepts synthetic/development inputs only")
    x = data.get("X")
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or min(x.shape) < 1
            or x.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(x).all())):
        raise ValueError("X must be finite float32/float64 [batch, rows, features]")
    features = data.get("feature_names")
    if (not isinstance(features, list) or len(features) != x.shape[-1]
            or not all(isinstance(f, str) and f for f in features) or len(set(features)) != len(features)):
        raise ValueError("feature_names must uniquely name each ordered input feature")
    preprocessing = data.get("preprocessing_id")
    if not isinstance(preprocessing, str) or not preprocessing:
        raise ValueError("preprocessing_id must identify the externally frozen support-fitted transform")
    if data.get("label_mapping") != {"normal": 0, "attack": 1}:
        raise ValueError("Native labels must use normal=0, attack=1 without class shuffles")
    if role == "support":
        y = data.get("y")
        if (not isinstance(y, torch.Tensor) or y.shape != x.shape[:2]
                or not bool(((y == 0) | (y == 1)).all())):
            raise ValueError("support y must match [batch,rows] and contain only 0/1")
    else:
        if "y" in data:
            raise ValueError("Query input contains no labels; keep evaluation labels in a separate file")
    contract = {"feature_names": features, "preprocessing_id": preprocessing,
                "label_mapping": data["label_mapping"]}
    return data, contract


def compile_command(args) -> dict:
    """Lab entry: read one prepared support tensor, build and export both caches."""
    from .tabicl_adapter import FullCacheAdapter

    data, contract = _read_input(args.support, "support")
    if not 1 <= args.slots <= data["X"].shape[1]:
        raise ValueError("slots must be between 1 and the number of support rows")
    _new_output_dir(args.output)
    adapter = FullCacheAdapter.from_checkpoint(
        args.checkpoint, device=args.device, expected_sha256=args.checkpoint_sha256)
    adapter.provenance.update({"input_contract": contract, "data_kind": data["data_kind"],
                               "support_file_sha256": _sha256(args.support)})
    adapter.build(data["X"].to(args.device), data["y"].to(args.device))
    full = adapter.as_predictor()
    compact, compilation = adapter.compile(slots=args.slots, query_chunk_size=args.query_chunk_size)
    full_path, compact_path = args.output / "full.pt", args.output / "compact.pt"
    full.save(full_path)
    compact.save(compact_path)
    report = {"stage": "cache_compilation_only", "data_kind": data["data_kind"],
              "input_contract": contract, "full": full.size_report(), "compact": compact.size_report(),
              "compilation": compilation, "artifacts": {
                  "full": {"file": full_path.name, "sha256": _sha256(full_path),
                           "file_bytes": full_path.stat().st_size},
                  "compact": {"file": compact_path.name, "sha256": _sha256(compact_path),
                              "file_bytes": compact_path.stat().st_size}},
              "scope": "Native tensors only; no risk calibration, final test, IDS comparison or resource benchmark"}
    _write_json(args.output / "compile_report.json", report)
    return {"output": str(args.output.resolve()), "stage": report["stage"],
            "source_count": data["X"].shape[1], "slots": args.slots}


def predict_command(args) -> dict:
    import torch
    from .tabicl_adapter import CompactPredictor

    if args.output.exists():
        raise FileExistsError(args.output)
    data, contract = _read_input(args.queries, "query")
    predictor = CompactPredictor.load(args.bundle, device=args.device)
    if predictor.provenance.get("input_contract") != contract:
        raise ValueError("Query input contract differs from the compiled support contract")
    logits = predictor.predict_logits(data["X"].to(args.device)).detach().cpu()
    scores = logits[..., 1] - logits[..., 0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "tracelight-tabicl-native-predictions-v1", "data_kind": data["data_kind"],
                "logits": logits, "scores": scores, "score_kind": "attack_minus_normal_logit",
                "bundle_sha256": _sha256(args.bundle), "query_sha256": _sha256(args.queries),
                "input_contract": contract}, args.output)
    return {"output": str(args.output.resolve()), "rows_per_batch": logits.shape[1],
            "stage": "uncalibrated_native_prediction"}


def synthetic_command(args) -> dict:
    """Exercise the actual official architecture with small random weights."""
    import torch
    from .tabicl_adapter import FullCacheAdapter
    from .upstream import require_pinned_tabicl

    source = require_pinned_tabicl()
    from tabicl._model.tabicl import TabICL

    _new_output_dir(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(20261106)
    config = {"max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
              "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
              "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.0,
              "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
              "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise",
              "icl_ssmax": "qassmax-mlp-elementwise"}
    model = TabICL(**config).eval()
    # Default query modulation is exactly zero. Make the nonlinear path
    # observable so a mistaken order of QASSMax and sqrt(d) cannot pass.
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "query_mlp"):
                torch.nn.init.normal_(module.query_mlp[-1].weight, std=0.2)
                torch.nn.init.normal_(module.query_mlp[-1].bias, std=0.1)
    support = torch.randn(1, 12, 3)
    labels = torch.tensor([[0, 1] * 6])
    queries = torch.randn(1, 7, 3)
    contract = {"feature_names": ["synthetic_a", "synthetic_b", "synthetic_c"],
                "preprocessing_id": "synthetic_identity_no_fit",
                "label_mapping": {"normal": 0, "attack": 1}}
    adapter = FullCacheAdapter(model, model_config=config, provenance={
        "data_kind": "synthetic", "weights": "random_tiny_nonzero", "input_contract": contract})
    adapter.build(support, labels)
    official = adapter.predict_logits(queries)
    full = adapter.as_predictor()
    full_logits = full.predict_logits(queries)
    torch.testing.assert_close(full_logits, official, atol=2e-6, rtol=2e-5)
    identity, _ = adapter.compile(slots=12, query_chunk_size=5)
    torch.testing.assert_close(identity.predict_logits(queries), official, atol=2e-6, rtol=2e-5)
    compact, compilation = adapter.compile(slots=3, query_chunk_size=5)
    compact_logits = compact.predict_logits(queries)
    pieces = torch.cat([compact.predict_logits(queries[:, :2]), compact.predict_logits(queries[:, 2:])], dim=1)
    torch.testing.assert_close(pieces, compact_logits, atol=2e-6, rtol=2e-5)
    full.save(args.output / "full.pt")
    compact_path = args.output / "compact.pt"
    compact.save(compact_path)
    query_path = args.output / "queries.pt"
    torch.save({"format": INPUT_FORMAT, "role": "query", "data_kind": "synthetic", "X": queries,
                **contract}, query_path)

    # A separate interpreter is given only the compiled artifact and queries.
    # No raw support tensor, source checkpoint, full cache or fitting Q is read.
    cold_dir = args.output / "cold_process"
    cold_dir.mkdir()
    cold_path = cold_dir / "predictions.pt"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
    command = [sys.executable, "-m", "ids.context_cache", "predict", "--bundle", str(compact_path.resolve()),
               "--queries", str(query_path.resolve()), "--output", str(cold_path.resolve()), "--device", "cpu"]
    child = subprocess.run(command, cwd=cold_dir, env=env, capture_output=True, text=True, timeout=60)
    if child.returncode:
        raise RuntimeError(f"Cold-process prediction failed: {child.stderr[-3000:]}")
    cold = torch.load(cold_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(cold["logits"], compact_logits.cpu(), atol=2e-6, rtol=2e-5)
    full_size, compact_size = full.size_report(), compact.size_report()
    report = {"stage": "synthetic_engineering_check", "data_kind": "synthetic", "source": source,
              "weights": "random_tiny_nonzero_no_checkpoint", "support_rows": 12, "query_rows": 7,
              "slots": 3, "torch_version": str(torch.__version__), "device": "cpu",
              "checks": {"official_full_vs_adapter": True, "identity_M_equals_N": True,
                         "query_batch_partition_invariance": True, "cold_process_round_trip": True},
              "full_max_abs_logit_error": float((full_logits - official).abs().max()),
              "cold_max_abs_logit_error": float((cold["logits"] - compact_logits.cpu()).abs().max()),
              "full": full_size, "compact": compact_size, "compilation": compilation,
              "artifact_file_bytes": {"full": (args.output / "full.pt").stat().st_size,
                                      "compact": compact_path.stat().st_size},
              "scope": "Synthetic interface verification only; not IDS quality, real-model equivalence or measured speedup"}
    _write_json(args.output / "synthetic_report.json", report)
    return {"output": str(args.output.resolve()), "stage": report["stage"], "checks": report["checks"]}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabicl-source", type=Path, help="Optional pinned source checkout; no installation")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synthetic-check", help="CPU-only random tiny official model; no checkpoint/data download")
    synth.add_argument("--output", type=Path, required=True)
    compile_parser = sub.add_parser("compile", help="Lab: build full and QCOC caches from one prepared support tensor")
    compile_parser.add_argument("--checkpoint", type=Path, required=True)
    compile_parser.add_argument("--checkpoint-sha256")
    compile_parser.add_argument("--support", type=Path, required=True)
    compile_parser.add_argument("--slots", type=int, required=True)
    compile_parser.add_argument("--query-chunk-size", type=int, default=256)
    compile_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    compile_parser.add_argument("--output", type=Path, required=True)
    predict = sub.add_parser("predict", help="Native logits/scores only; no labels, calibration or final-test metrics")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--queries", type=Path, required=True)
    predict.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    predict.add_argument("--output", type=Path, required=True)
    csv_synth = sub.add_parser("synthetic-csv-check", help="Random tiny model: raw CSV, separate thresholds and cold CSV prediction")
    csv_synth.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("lab-prepare", help="Lab: frozen NF-ToN-IoT-v3 support/calibration/evaluation CSVs")
    prepare.add_argument("--raw-csv", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare_v2 = sub.add_parser("lab-prepare-v2", help="Lab: group-disjoint flow sampling without sequential feature exclusion")
    prepare_v2.add_argument("--raw-csv", type=Path, required=True)
    prepare_v2.add_argument("--output", type=Path, required=True)
    register = sub.add_parser("lab-register", help="Register explicit CSV files and verified checkpoint; no model loading")
    register.add_argument("--template", type=Path, required=True)
    register.add_argument("--support-csv", type=Path, required=True)
    register.add_argument("--calibration-csv", type=Path, required=True)
    register.add_argument("--evaluation-csv", type=Path, required=True)
    register.add_argument("--checkpoint", type=Path, required=True)
    register.add_argument("--checkpoint-sha256", required=True)
    register.add_argument("--output", type=Path, required=True)
    preflight = sub.add_parser("lab-preflight", help="Validate pinned files, CSV roles and overlaps without loading model weights")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path)
    comparison = sub.add_parser("lab-compare", help="Lab: exactly full KV and one QCOC budget from three development CSVs")
    comparison.add_argument("--config", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    input_comparison = sub.add_parser("lab-input-compare", help="Lab: legacy full, protected-ID full, and protected-ID QCOC on the same v2 data")
    input_comparison.add_argument("--config", type=Path, required=True)
    input_comparison.add_argument("--output", type=Path, required=True)
    raw_predict = sub.add_parser("predict-csv", help="Load model + fitted preprocessing + registered threshold for raw CSV")
    raw_predict.add_argument("--deployment", type=Path, required=True)
    raw_predict.add_argument("--input", type=Path, required=True)
    raw_predict.add_argument("--output", type=Path, required=True)
    scan_register = sub.add_parser("scanning-register", help="Bind the frozen round14 inputs and references; no fitting")
    scan_register.add_argument("--round14-root", type=Path, required=True)
    scan_register.add_argument("--checkpoint", type=Path, required=True)
    scan_register.add_argument("--output", type=Path, required=True)
    for command, help_text in (
        ("scanning-preflight", "Lab: verify frozen identities, runtime and CUDA without loading weights"),
        ("lab-scanning-compare", "Lab: B0 replay then source-port bijection and masking; full KV only"),
    ):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    scan_predict = sub.add_parser("scanning-predict", help="Cold arm-specific full-KV prediction from raw features")
    scan_predict.add_argument("--deployment", type=Path, required=True)
    scan_predict.add_argument("--input", type=Path, required=True)
    scan_predict.add_argument("--output", type=Path, required=True)
    scan_verify = sub.add_parser("scanning-verify", help="Recompute saved results without loading a model")
    scan_verify.add_argument("--results", type=Path, required=True)
    scan_verify.add_argument("--output", type=Path, required=True)
    scan_pack = sub.add_parser("scanning-pack", help="Archive this run only, with member SHA verification")
    scan_pack.add_argument("--run-root", type=Path, required=True)
    scan_pack.add_argument("--output", type=Path, required=True)
    label_register = sub.add_parser("label-register", help="Bind the frozen round14 inputs and references; no fitting")
    label_register.add_argument("--round14-root", type=Path, required=True)
    label_register.add_argument("--checkpoint", type=Path, required=True)
    label_register.add_argument("--output", type=Path, required=True)
    for command, help_text in (
        ("label-preflight", "Lab: verify frozen identities, runtime and CUDA without loading weights"),
        ("lab-label-compare", "Lab: binary replay, semantic family labels and grouped sham; full KV only"),
    ):
        item = sub.add_parser(command, help=help_text)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
    label_predict = sub.add_parser("label-predict", help="Cold arm-specific full-KV prediction from raw features")
    label_predict.add_argument("--deployment", type=Path, required=True)
    label_predict.add_argument("--input", type=Path, required=True)
    label_predict.add_argument("--output", type=Path, required=True)
    label_verify = sub.add_parser("label-verify", help="Recompute saved results without loading a model")
    label_verify.add_argument("--results", type=Path, required=True)
    label_verify.add_argument("--output", type=Path, required=True)
    label_pack = sub.add_parser("label-pack", help="Archive this run only, with member SHA verification")
    label_pack.add_argument("--run-root", type=Path, required=True)
    label_pack.add_argument("--output", type=Path, required=True)
    return parser


def scanning_command(args):
    from .scanning_protocol import (
        load_scanning_protocol, pack_scanning_results, preflight_scanning, register_scanning_protocol,
    )

    if args.command == "scanning-register":
        return register_scanning_protocol(round14_root=args.round14_root, checkpoint=args.checkpoint,
                                          output=args.output)
    if args.command == "scanning-pack":
        return pack_scanning_results(args.run_root, args.output)
    if args.command == "scanning-predict":
        from .scanning_workflow import ScanningDetector
        return ScanningDetector.load(args.deployment).predict_csv(args.input, args.output)
    if args.command == "scanning-verify":
        from .scanning_workflow import verify_scanning_result
        if args.output.exists():
            raise FileExistsError(args.output)
        result = verify_scanning_result(args.results)
    else:
        spec, _, _ = load_scanning_protocol(args.config)
        if spec["data_kind"] != "development":
            raise ValueError("laboratory scanning CLI requires frozen development data; synthetic fixtures use Python API")
        if args.command == "lab-scanning-compare":
            from .scanning_workflow import run_scanning
            report = run_scanning(args.config, args.output)
            return {"status": "complete", "output": str(args.output.resolve()), "workload": report["workload"]}
        if args.output.exists():
            raise FileExistsError(args.output)
        _, _, _, _, result = preflight_scanning(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    return result


def label_command(args):
    from .label_protocol import (
        load_label_protocol, pack_label_results, preflight_label, register_label_protocol,
    )

    if args.command == "label-register":
        return register_label_protocol(round14_root=args.round14_root, checkpoint=args.checkpoint,
                                          output=args.output)
    if args.command == "label-pack":
        return pack_label_results(args.run_root, args.output)
    if args.command == "label-predict":
        from .label_workflow import LabelDetector
        return LabelDetector.load(args.deployment).predict_csv(args.input, args.output)
    if args.command == "label-verify":
        from .label_workflow import verify_label_result
        if args.output.exists():
            raise FileExistsError(args.output)
        result = verify_label_result(args.results)
    else:
        spec, _, _ = load_label_protocol(args.config)
        if spec["data_kind"] != "development":
            raise ValueError("laboratory label CLI requires frozen development data; synthetic fixtures use Python API")
        if args.command == "lab-label-compare":
            from .label_workflow import run_label
            report = run_label(args.config, args.output)
            return {"status": "complete", "output": str(args.output.resolve()), "workload": report["workload"]}
        if args.output.exists():
            raise FileExistsError(args.output)
        _, _, _, _, result = preflight_label(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    return result


def csv_command(args):
    from .csv_workflow import CSVDetector, preflight_protocol, run_comparison

    if args.command in {"lab-prepare", "lab-prepare-v2"}:
        from .lab_data import prepare_lab_data
        if args.command == "lab-prepare-v2":
            return prepare_lab_data(args.raw_csv, args.output, sampling_profile="group_disjoint_flow_v2")
        return prepare_lab_data(args.raw_csv, args.output)
    if args.command == "lab-register":
        from .csv_workflow import register_protocol
        return register_protocol(args.template, support_csv=args.support_csv,
                                 calibration_csv=args.calibration_csv, evaluation_csv=args.evaluation_csv,
                                 checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
                                 output=args.output)
    if args.command == "predict-csv":
        return CSVDetector.load(args.deployment).predict_csv(args.input, args.output)
    if args.command == "lab-preflight":
        spec, _, _, _, audit = preflight_protocol(args.config)
        result = {"stage": "csv_preflight_without_model_loading", "data_kind": spec["data_kind"],
                  "slots": spec["slots"], "audit": audit}
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.output, result)
        return result
    if args.command == "synthetic-csv-check":
        from .csv_synthetic import synthetic_csv_check
        return synthetic_csv_check(args.output)
    if args.command == "lab-input-compare":
        from .csv_workflow import run_input_comparison
        report = run_input_comparison(args.config, args.output)
        return {"output": str(args.output.resolve()), "scope": report["scope"],
                "candidate_count": report["candidate_count"],
                "evaluation": report["evaluation"],
                "paired_changes": {name: value["changes"] for name, value in report["paired"].items()}}
    report = run_comparison(args.config, args.output)
    return {"output": str(args.output.resolve()), "scope": report["scope"],
            "candidate_count": report["candidate_count"], "nontrivial_thresholds": report["nontrivial_thresholds"],
            "evaluation": report["evaluation"], "paired_changes": report["paired"]["changes"]}


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.tabicl_source:
        source = (args.tabicl_source / "src").resolve()
        if not (source / "tabicl" / "_model" / "tabicl.py").is_file():
            raise ValueError("--tabicl-source must point to the official source repository")
        sys.path.insert(0, str(source))
    actions = {"synthetic-check": synthetic_command, "compile": compile_command, "predict": predict_command}
    if "scanning" in args.command:
        actions[args.command] = scanning_command
    if "label" in args.command:
        actions[args.command] = label_command
    result = actions.get(args.command, csv_command)(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
