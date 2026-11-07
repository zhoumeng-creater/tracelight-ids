"""Frozen two-candidate development comparison from raw CSV files.

Only support rows fit the official 2026 preprocessing and full context. Each
candidate registers its own threshold on a separate normal-only development
file. The third file is development evaluation, never an opened final test.
"""

from __future__ import annotations

from itertools import combinations
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, write_json
from . import TABICL_COMMIT
from .calibration import apply_threshold, compare_decisions, evaluate, register_threshold
from .preprocessing import CSVPreprocessor, CSVSchema, labels, read_csv


PROTOCOL = "tracelight-context-csv-comparison-v1"
DEPLOYMENT = "tracelight-context-csv-detector-v1"
ROLES = ("support", "calibration", "evaluation")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _hash(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value.lower()


def _resolve(base, name):
    if not isinstance(name, str) or not name:
        raise ValueError("manifest file paths must be nonempty strings")
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_protocol(path):
    """Validate the fixed single-budget manifest before loading any model."""
    path = Path(path).resolve()
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    return validate_protocol(spec, path.parent)


def validate_protocol(spec, base):
    """Validate an in-memory registration using its declared path base."""
    base = Path(base).resolve()
    expected = {"protocol", "data_kind", "schema", "files", "checkpoint", "slots",
                "alpha", "build_device", "query_batch_size", "query_chunk_size", "threads", "seed"}
    if not isinstance(spec, dict) or set(spec) != expected or spec.get("protocol") != PROTOCOL:
        raise ValueError(f"manifest must contain exactly the fields of {PROTOCOL}")
    if spec["data_kind"] not in {"synthetic", "development"}:
        raise ValueError("comparison accepts synthetic/development data only, not final test")
    if spec["build_device"] not in {"cpu", "cuda"}:
        raise ValueError("build_device must be cpu or cuda")
    for name in ("slots", "query_batch_size", "query_chunk_size", "threads"):
        _positive(spec[name], name)
    if isinstance(spec["seed"], bool) or not isinstance(spec["seed"], int) or not 0 <= spec["seed"] < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    # Let the exact-rank implementation validate alpha even before CSV/model IO.
    register_threshold([], spec["alpha"], model_sha256="0" * 64,
                       preprocessing_sha256="0" * 64, calibration_sha256="0" * 64)
    schema = CSVSchema.from_dict(spec["schema"])
    if not isinstance(spec["files"], dict) or set(spec["files"]) != set(ROLES):
        raise ValueError("manifest must name support, calibration and evaluation CSVs")
    paths = {}
    for role in ROLES:
        item = spec["files"][role]
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "rows"}:
            raise ValueError(f"{role} requires path, sha256 and rows")
        _positive(item["rows"], f"{role}.rows")
        item["sha256"] = _hash(item["sha256"], f"{role}.sha256")
        paths[role] = _resolve(base, item["path"])
    if len(set(paths.values())) != 3:
        raise ValueError("each data role requires a separate CSV")
    checkpoint = spec["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"path", "sha256"}:
        raise ValueError("checkpoint requires local path and sha256")
    checkpoint["sha256"] = _hash(checkpoint["sha256"], "checkpoint.sha256")
    paths["checkpoint"] = _resolve(base, checkpoint["path"])
    if spec["slots"] > spec["files"]["support"]["rows"]:
        raise ValueError("slots exceed the registered support row count")
    n = spec["files"]["calibration"]["rows"]
    position = (n + 1) * (1 - Fraction(str(spec["alpha"])))
    rank = (position.numerator + position.denominator - 1) // position.denominator
    if spec["data_kind"] == "development" and rank > n:
        raise ValueError("normal calibration rows are insufficient for a finite threshold at the registered alpha")
    return spec, schema, paths


def register_protocol(template_path, *, support_csv, calibration_csv, evaluation_csv,
                      checkpoint, checkpoint_sha256, output):
    """Register explicit lab files; inspect CSVs, but never load model weights.

    This fills hashes and row counts without choosing a split, sampling rows,
    inferring features, or changing the single budget in the supplied template.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = json.loads(Path(template_path).read_text(encoding="utf-8-sig"))
    schema = CSVSchema.from_dict(spec["schema"])
    checkpoint = Path(checkpoint).resolve()
    expected = _hash(checkpoint_sha256, "checkpoint_sha256")
    if sha256_file(checkpoint) != expected:
        raise ValueError("checkpoint SHA-256 differs from the independently registered source")
    supplied = dict(zip(ROLES, (support_csv, calibration_csv, evaluation_csv)))
    frames, files = {}, {}
    for role, value in supplied.items():
        path = Path(value).resolve()
        frames[role] = read_csv(path, schema)
        files[role] = {"path": str(path), "sha256": sha256_file(path), "rows": len(frames[role])}
    spec["files"] = files
    spec["checkpoint"] = {"path": str(checkpoint), "sha256": expected}
    spec, _, _ = validate_protocol(spec, output.parent)
    audit = audit_roles(frames, schema)
    write_json(output, spec)
    return {"stage": "registered_explicit_csv_files_without_model_loading",
            "output": str(output), "protocol_sha256": sha256_file(output),
            "data_kind": spec["data_kind"], "slots": spec["slots"], "audit": audit}


def _row_ids(frame, schema, role):
    if schema.id_column is not None and schema.id_column in frame:
        return frame[schema.id_column].astype(str).tolist()
    return [f"{role}:{index}" for index in range(len(frame))]


def _feature_signatures(frame, schema):
    """Canonical full input signatures exclude all labels/IDs/family metadata."""
    numeric = set(schema.numeric)
    result = []
    for row in frame.loc[:, list(schema.features)].itertuples(index=False, name=None):
        values = []
        for name, value in zip(schema.features, row):
            if pd.isna(value):
                values.append(None)
            elif name in numeric:
                number = float(value)
                values.append((0.0 if number == 0 else number).hex())
            else:
                values.append(str(value))
        result.append(hashlib.sha256(json.dumps(values, separators=(",", ":"),
                                               ensure_ascii=False).encode()).hexdigest())
    return result


def audit_roles(frames, schema):
    """Reject cross-role exact-feature, explicit-ID and declared-group overlap."""
    if set(frames) != set(ROLES):
        raise ValueError("all three development roles are required")
    signatures, ids, groups, counts = {}, {}, {}, {}
    for role, frame in frames.items():
        truth = labels(frame, schema)
        if role == "calibration" and not bool((truth == 0).all()):
            raise ValueError("calibration CSV must contain normal rows only; no silent filtering")
        if role in {"support", "evaluation"} and set(truth.tolist()) != {0, 1}:
            raise ValueError(f"{role} CSV must contain both normal and attack rows")
        signature_list = _feature_signatures(frame, schema)
        signatures[role] = set(signature_list)
        ids[role] = set(_row_ids(frame, schema, role))
        groups[role] = set(frame[schema.group_column].astype(str)) if schema.group_column else set()
        counts[role] = {"rows": len(frame), "normal": int((truth == 0).sum()),
                        "attack": int((truth == 1).sum()),
                        "within_role_exact_feature_duplicates": len(frame) - len(signatures[role]),
                        "ordered_row_ids_sha256": canonical_hash(_row_ids(frame, schema, role))}
    pairs = {}
    for left, right in combinations(ROLES, 2):
        common = {"exact_feature_overlap": len(signatures[left] & signatures[right]),
                  "explicit_id_overlap": len(ids[left] & ids[right]) if schema.id_column else 0,
                  "declared_group_overlap": len(groups[left] & groups[right]) if schema.group_column else 0}
        if any(common.values()):
            raise ValueError(f"cross-role overlap {left}/{right}: {common}")
        pairs[f"{left}/{right}"] = common
    return {"roles": counts, "cross_role_pairs": pairs,
            "explicit_ids": "checked" if schema.id_column else "absent_generated_role_row_keys",
            "session_groups": "checked_declared_group" if schema.group_column else "not_checked_no_group_column",
            "duplicate_policy": "within-role rows retained; cross-role exact inputs rejected",
            "scope": "declared identities and exact inputs only; this audit does not establish population independence"}


def _code_snapshot():
    folder = Path(__file__).resolve().parent
    files = {p.name: sha256_file(p) for p in sorted(folder.glob("*.py"))}
    return {"source_files_sha256": files, "combined_sha256": canonical_hash(files)}


def preflight_protocol(config_path):
    """Read explicitly listed CSVs and hashes, but never load a checkpoint."""
    spec, schema, paths = load_protocol(config_path)
    for role in (*ROLES, "checkpoint"):
        expected = spec["checkpoint"]["sha256"] if role == "checkpoint" else spec["files"][role]["sha256"]
        if sha256_file(paths[role]) != expected.lower():
            raise ValueError(f"registered SHA-256 mismatch for {role}")
    frames = {role: read_csv(paths[role], schema) for role in ROLES}
    for role, frame in frames.items():
        if len(frame) != spec["files"][role]["rows"]:
            raise ValueError(f"registered row count differs for {role}")
    audit = audit_roles(frames, schema)
    return spec, schema, paths, frames, audit


def _score_frame(predictor, preprocessor, frame, batch_size):
    import torch

    scores = []
    start = time.perf_counter()
    with torch.no_grad():
        for begin in range(0, len(frame), batch_size):
            transformed = preprocessor.transform(frame.iloc[begin:begin + batch_size])
            logits = predictor.predict_logits(torch.from_numpy(transformed).unsqueeze(0))
            # Subtract in the model's FP32 arithmetic, then widen for comparison.
            scores.append((logits[0, :, 1] - logits[0, :, 0]).cpu().numpy().astype(np.float64))
    return np.concatenate(scores), time.perf_counter() - start


class CSVDetector:
    """Cold raw-CSV detector: frozen preprocessing + native model + threshold."""

    def __init__(self, predictor, preprocessor, registration, manifest):
        self.predictor, self.preprocessor = predictor, preprocessor
        self.registration, self.manifest = registration, manifest

    @classmethod
    def load(cls, directory):
        import torch
        from .tabicl_adapter import CompactPredictor

        directory = Path(directory)
        manifest = read_json(directory / "deployment.json")
        if manifest.get("format") != DEPLOYMENT or manifest.get("prediction_device") != "cpu":
            raise ValueError("unsupported CSV deployment format/profile")
        torch.set_num_threads(_positive(manifest.get("threads"), "deployment threads"))
        _positive(manifest.get("query_batch_size"), "deployment query_batch_size")
        expected_files = {"model.pt", "preprocessor.joblib", "threshold.json"}
        if set(manifest.get("files", {})) != expected_files:
            raise ValueError("deployment must contain model, preprocessing and threshold registrations")
        for name, expected in manifest["files"].items():
            if sha256_file(directory / name) != _hash(expected, name):
                raise ValueError(f"deployment hash mismatch for {name}")
        prep_hash = manifest["files"]["preprocessor.joblib"]
        preprocessor = CSVPreprocessor.load(directory / "preprocessor.joblib", expected_sha256=prep_hash)
        registration = read_json(directory / "threshold.json")
        if registration.get("binding") != {
            "model_sha256": manifest["files"]["model.pt"],
            "preprocessing_sha256": prep_hash,
            "calibration_sha256": manifest["calibration_sha256"],
        }:
            raise ValueError("threshold binding differs from the deployment artifacts")
        apply_threshold([], registration)  # Validate the exact decision contract, including +infinity.
        predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
        contract = predictor.provenance.get("input_contract", {})
        if (contract.get("preprocessing_id") != prep_hash
                or contract.get("feature_names") != preprocessor.report()["output_features"]
                or contract.get("label_mapping") != {"normal": 0, "attack": 1}
                or predictor.train_shape[-1] != preprocessor.n_features_out_):
            raise ValueError("preprocessing and model feature/label contracts differ")
        return cls(predictor, preprocessor, registration, manifest)

    def predict_frame(self, frame):
        scores, _ = _score_frame(self.predictor, self.preprocessor, frame, self.manifest["query_batch_size"])
        predicted = apply_threshold(scores, self.registration)
        return pd.DataFrame({"row_id": _row_ids(frame, self.preprocessor.schema, "query"),
                             "score": scores, "predicted_attack": predicted.astype(np.uint8)})

    def predict_csv(self, input_path, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(output_path)
        frame = read_csv(input_path, self.preprocessor.schema, require_label=False)
        result = self.predict_frame(frame)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        return {"output": str(output_path.resolve()), "rows": len(result),
                "alerts": int(result["predicted_attack"].sum()),
                "threshold_scope": self.registration["scope"]}


def run_comparison(config_path, output_dir):
    """Execute exactly full KV and one QCOC budget; no model/threshold search."""
    import torch
    from .tabicl_adapter import FullCacheAdapter

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    spec, schema, paths, frames, audit = preflight_protocol(config_path)
    output_dir.mkdir(parents=True)
    torch.set_num_threads(spec["threads"])
    torch.manual_seed(spec["seed"])
    snapshot = _code_snapshot()
    write_json(output_dir / "protocol.json", spec)
    write_json(output_dir / "data_audit.json", audit)
    write_json(output_dir / "source_snapshot.json", snapshot)
    write_json(output_dir / "environment.json", environment_snapshot())
    timing = {}
    start = time.perf_counter()
    preprocessor = CSVPreprocessor(schema).fit(frames["support"])
    prep_path = preprocessor.save(output_dir / "preprocessor.joblib")
    prep_hash = sha256_file(prep_path)
    matrix = preprocessor.transform(frames["support"])
    timing["support_preprocessing_seconds"] = time.perf_counter() - start
    contract = {"preprocessing_id": prep_hash, "feature_names": preprocessor.report()["output_features"],
                "label_mapping": {"normal": 0, "attack": 1}}
    adapter = FullCacheAdapter.from_checkpoint(
        paths["checkpoint"], device=spec["build_device"], expected_sha256=spec["checkpoint"]["sha256"])
    adapter.provenance.update({"input_contract": contract, "data_kind": spec["data_kind"],
                               "support_csv_sha256": spec["files"]["support"]["sha256"],
                               "protocol_sha256": canonical_hash(spec), "adapter_source": snapshot})
    if spec["build_device"] == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    adapter.build(torch.from_numpy(matrix).unsqueeze(0),
                  torch.from_numpy(labels(frames["support"], schema)).unsqueeze(0))
    if spec["build_device"] == "cuda":
        torch.cuda.synchronize()
    timing["full_context_build_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    compact, compilation = adapter.compile(slots=spec["slots"], query_chunk_size=spec["query_chunk_size"])
    if spec["build_device"] == "cuda":
        torch.cuda.synchronize()
    timing["qcoc_compile_seconds"] = time.perf_counter() - start
    candidates = {"full": adapter.as_predictor(), "qcoc": compact}
    model_hashes, tensor_sizes = {}, {}
    for name, predictor in candidates.items():
        directory = output_dir / name
        directory.mkdir()
        predictor.save(directory / "model.pt")
        shutil.copyfile(prep_path, directory / "preprocessor.joblib")
        model_hashes[name] = sha256_file(directory / "model.pt")
        tensor_sizes[name] = predictor.size_report()
    # Scoring loads each saved candidate independently on the same CPU profile.
    # This avoids live full/compact backbone aliases or build-device effects.
    del candidates, compact, adapter, matrix, predictor
    registrations, evaluation_scores, results = {}, {}, {}
    calibration_ids = _row_ids(frames["calibration"], schema, "calibration")
    families = None
    if schema.family_column:
        families = frames["evaluation"][schema.family_column].fillna("<missing>").astype(str).tolist()
    from .tabicl_adapter import CompactPredictor
    for name in ("full", "qcoc"):
        predictor = CompactPredictor.load(output_dir / name / "model.pt", device="cpu")
        normal_scores, timing[f"{name}_calibration_csv_score_seconds"] = _score_frame(
            predictor, preprocessor, frames["calibration"], spec["query_batch_size"])
        registration = register_threshold(
            normal_scores, spec["alpha"], model_sha256=model_hashes[name], preprocessing_sha256=prep_hash,
            calibration_sha256=spec["files"]["calibration"]["sha256"])
        registrations[name] = registration
        directory = output_dir / name
        threshold_path = directory / "threshold.json"
        write_json(threshold_path, registration)
        pd.DataFrame({"row_id": calibration_ids, "label": 0, "score": normal_scores,
                      "predicted_attack": apply_threshold(normal_scores, registration).astype(np.uint8)}).to_csv(
                          output_dir / f"{name}_calibration.csv", index=False)
        deployment = {"format": DEPLOYMENT, "candidate": name, "data_kind": spec["data_kind"],
                      "prediction_device": "cpu", "dtype": "float32", "threads": spec["threads"],
                      "query_batch_size": spec["query_batch_size"],
                      "calibration_sha256": spec["files"]["calibration"]["sha256"],
                      "upstream_commit": TABICL_COMMIT,
                      "files": {"model.pt": model_hashes[name], "preprocessor.joblib": prep_hash,
                                "threshold.json": sha256_file(threshold_path)}}
        write_json(directory / "deployment.json", deployment)
        scores, timing[f"{name}_evaluation_csv_score_seconds"] = _score_frame(
            predictor, preprocessor, frames["evaluation"], spec["query_batch_size"])
        evaluation_scores[name] = scores
        results[name] = evaluate(labels(frames["evaluation"], schema), scores, registration, families)
        del predictor
    paired = compare_decisions(
        labels(frames["evaluation"], schema), evaluation_scores["full"], registrations["full"],
        evaluation_scores["qcoc"], registrations["qcoc"],
        ids=_row_ids(frames["evaluation"], schema, "evaluation"))
    rows = pd.DataFrame(paired.pop("rows"))
    if families is not None:
        rows["attack_family"] = families
    rows.to_csv(output_dir / "paired_evaluation.csv", index=False)
    packaged = {name: {"files": {p.name: p.stat().st_size for p in (output_dir / name).iterdir() if p.is_file()},
                       "total_file_bytes": sum(p.stat().st_size for p in (output_dir / name).iterdir() if p.is_file())}
               for name in ("full", "qcoc")}
    report = {"protocol": PROTOCOL, "scope": "development_comparison_not_final_confirmation",
              "data_kind": spec["data_kind"], "candidate_count": 2, "slots": spec["slots"],
              "support_rows": len(frames["support"]), "alpha": registrations["full"]["alpha"],
              "single_view": True,
              "nontrivial_thresholds": {name: not value["reject_all"] for name, value in registrations.items()},
              "prediction_profile": {"device": "cpu", "dtype": "float32",
                  "threads": spec["threads"], "query_batch_size": spec["query_batch_size"]},
              "preprocessing": preprocessor.report(), "registrations": registrations, "evaluation": results,
              "paired": paired, "data_audit": audit, "compilation": compilation,
              "native_tensor_sizes": tensor_sizes, "complete_csv_deployment_files": packaged,
              "timing": {"kind": "single_run_wall_clock_not_latency_or_RSS_benchmark", **timing},
              "source_snapshot": snapshot,
              "input_hashes": {role: spec["files"][role]["sha256"] for role in ROLES},
              "checkpoint_sha256": spec["checkpoint"]["sha256"]}
    write_json(output_dir / "comparison.json", report)
    return report
