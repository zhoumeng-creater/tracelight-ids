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
PROTOCOL_V2 = "tracelight-context-input-comparison-v2"
DEPLOYMENT = "tracelight-context-csv-detector-v1"
ROLES = ("support", "calibration", "evaluation")
INPUT_PROFILES = {"legacy": "official_single_view_v1", "new": "nf_v3_protected_ids_v2"}
V2_POLICIES = {"sampling_profile": "group_disjoint_flow_v2", "risk_unit": "flow",
               "exact_feature_policy": "report"}


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
    if not isinstance(spec, dict) or spec.get("protocol") not in {PROTOCOL, PROTOCOL_V2}:
        raise ValueError("manifest must declare a supported comparison protocol")
    is_v2 = spec["protocol"] == PROTOCOL_V2
    if is_v2:
        expected |= {*V2_POLICIES, "preparation"}
    if set(spec) != expected:
        raise ValueError(f"manifest must contain exactly the fields of {spec['protocol']}")
    if is_v2 and any(spec[key] != value for key, value in V2_POLICIES.items()):
        raise ValueError("v2 requires group_disjoint_flow_v2 sampling, flow risk and report exact-feature policy")
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
    if is_v2 and (schema.id_column is None or schema.group_column is None):
        raise ValueError("v2 requires explicit sample IDs and communication groups")
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
    if is_v2:
        preparation = spec["preparation"]
        if not isinstance(preparation, dict) or set(preparation) != {"path", "sha256"}:
            raise ValueError("v2 preparation requires path and sha256")
        preparation["sha256"] = _hash(preparation["sha256"], "preparation.sha256")
        paths["preparation"] = _resolve(base, preparation["path"])
        if spec["data_kind"] == "development":
            from .lab_data import SCHEMA, SEED

            frozen = {"slots": 512, "seed": SEED, "alpha": .01, "build_device": "cuda",
                      "threads": 1, "query_batch_size": 128, "query_chunk_size": 256}
            if (schema.to_dict() != CSVSchema.from_dict(SCHEMA).to_dict()
                    or any(spec[key] != value for key, value in frozen.items())
                    or {role: spec["files"][role]["rows"] for role in ROLES}
                    != {"support": 8192, "calibration": 5000, "evaluation": 20000}):
                raise ValueError("v2 development requires the frozen NF-v3 schema, seed, P8192/C5000/E20000/M512/alpha.01 and CUDA/1-thread/batch128/chunk256 profile")
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
    if spec.get("protocol") == PROTOCOL_V2:
        parents = {Path(item["path"]).parent for item in files.values()}
        if len(parents) != 1:
            raise ValueError("v2 CSVs must share one prepared directory and preparation.json")
        preparation = next(iter(parents)) / "preparation.json"
        spec["preparation"] = {"path": str(preparation), "sha256": sha256_file(preparation)}
    spec, _, paths = validate_protocol(spec, output.parent)
    audit = audit_roles(frames, schema, exact_feature_policy=spec.get("exact_feature_policy", "reject"))
    if spec["protocol"] == PROTOCOL_V2:
        audit["preparation"] = _audit_preparation(spec, schema, paths)
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


def audit_roles(frames, schema, *, exact_feature_policy="reject"):
    """Always reject ID/group overlap; explicitly reject or report equal inputs."""
    if exact_feature_policy not in {"reject", "report"}:
        raise ValueError("exact_feature_policy must be reject or report")
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
        if (common["explicit_id_overlap"] or common["declared_group_overlap"]
                or (exact_feature_policy == "reject" and common["exact_feature_overlap"])):
            raise ValueError(f"cross-role overlap {left}/{right}: {common}")
        pairs[f"{left}/{right}"] = common
    return {"roles": counts, "cross_role_pairs": pairs,
            "explicit_ids": "checked" if schema.id_column else "absent_generated_role_row_keys",
            "session_groups": "checked_declared_group" if schema.group_column else "not_checked_no_group_column",
            "duplicate_policy": ("within-role rows retained; cross-role exact inputs rejected"
                                 if exact_feature_policy == "reject" else
                                 "flow rows retained; cross-role exact inputs reported; ID/group overlaps rejected"),
            "scope": "declared identities and exact inputs only; this audit does not establish population independence"}


def _code_snapshot():
    folder = Path(__file__).resolve().parent
    files = {p.name: sha256_file(p) for p in sorted(folder.glob("*.py"))}
    return {"source_files_sha256": files, "combined_sha256": canonical_hash(files)}


def _audit_preparation(spec, schema, paths):
    """Bind v2 sampling evidence to exactly the registered source and three files."""
    from .lab_data import DATASET, IDENTIFIER_DOMAINS, SOURCE_BYTES, SOURCE_SHA1

    path = paths["preparation"]
    actual = sha256_file(path)
    if actual != spec["preparation"]["sha256"]:
        raise ValueError("registered SHA-256 mismatch for preparation")
    receipt = read_json(path)
    if (receipt.get("format") != "tracelight-nf-ton-v3-preparation-v2"
            or any(receipt.get(key) != value for key, value in V2_POLICIES.items())
            or receipt.get("data_kind") != spec["data_kind"] or receipt.get("seed") != spec["seed"]
            or CSVSchema.from_dict(receipt.get("schema", {})).to_dict() != schema.to_dict()):
        raise ValueError("v2 preparation receipt format, policies, data kind, seed or schema differ")
    if any(paths[role].parent != path.parent for role in ROLES):
        raise ValueError("v2 CSVs and bound preparation.json must share one prepared directory")
    receipt_files = receipt.get("files", {})
    if set(receipt_files) != set(ROLES):
        raise ValueError("v2 preparation receipt must bind all three CSVs")
    for role in ROLES:
        item = receipt_files[role]
        if (not isinstance(item, dict) or item.get("sha256", "").lower() != spec["files"][role]["sha256"]
                or item.get("rows") != spec["files"][role]["rows"]
                or _resolve(path.parent, item.get("path")) != paths[role]):
            raise ValueError(f"v2 preparation CSV binding differs for {role}")
    budget = receipt.get("budgets", {})
    if (budget.get("support_per_class", 0) * 2 != spec["files"]["support"]["rows"]
            or budget.get("calibration_rows") != spec["files"]["calibration"]["rows"]
            or budget.get("evaluation_rows") != spec["files"]["evaluation"]["rows"]
            or budget.get("alpha") != spec["alpha"]):
        raise ValueError("v2 preparation budgets differ from the registered comparison")
    source = receipt.get("source", {})
    if spec["data_kind"] == "development":
        if (source.get("dataset") != DATASET or source.get("sha1") != SOURCE_SHA1
                or source.get("bytes") != SOURCE_BYTES or source.get("official_release_verified") is not True
                or budget.get("slots") != spec["slots"]):
            raise ValueError("v2 development preparation must verify the frozen official NF-v3 source and slots")
    elif source.get("official_release_verified") is not False:
        raise ValueError("synthetic preparation must not claim an official release")
    scans = receipt.get("scans", {})
    if set(scans) != set(ROLES):
        raise ValueError("v2 preparation requires three full-source identifier validation scans")
    domains, source_rows = dict(IDENTIFIER_DOMAINS), set()
    for role in ROLES:
        scan = scans[role]
        count = _positive(scan.get("source_rows_scanned"), "preparation.source_rows_scanned")
        validation = scan.get("identifier_validation", {})
        fields = validation.get("fields", {})
        if validation.get("rows_validated") != count or set(fields) != set(domains):
            raise ValueError(f"v2 preparation identifier scan coverage differs for {role}")
        for name, upper in domains.items():
            field = fields[name]
            low, high = field.get("minimum"), field.get("maximum")
            if (field.get("accepted_minimum") != 0 or field.get("accepted_maximum") != upper
                    or type(low) is not int or type(high) is not int or not 0 <= low <= high <= upper):
                raise ValueError(f"v2 preparation identifier domain evidence differs for {role}/{name}")
        source_rows.add(count)
    if len(source_rows) != 1:
        raise ValueError("v2 preparation scans must cover the same entire source")
    return {"path": str(path), "sha256": actual, "format": receipt["format"],
            **V2_POLICIES, "source": source, "budgets": budget,
            "identifier_validation": {"all_three_full_source_scans_verified": True,
                                      "source_rows": next(iter(source_rows)), "accepted_maxima": domains}}


def _representation_audits(frames, schema):
    from .representation_audit import audit_final_inputs

    return {name: audit_final_inputs(CSVPreprocessor(schema, profile=profile).fit(frames["support"]),
                                    frames, schema) for name, profile in INPUT_PROFILES.items()}


def preflight_protocol(config_path, *, audit_representations=True):
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
    audit = audit_roles(frames, schema, exact_feature_policy=spec.get("exact_feature_policy", "reject"))
    if spec["protocol"] == PROTOCOL_V2:
        audit["preparation"] = _audit_preparation(spec, schema, paths)
        if audit_representations:
            audit["representations"] = _representation_audits(frames, schema)
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


def _save_candidates(candidates, prep_path, output_dir):
    hashes, sizes = {}, {}
    for name, predictor in candidates.items():
        directory = output_dir / name
        directory.mkdir()
        predictor.save(directory / "model.pt")
        shutil.copyfile(prep_path, directory / "preprocessor.joblib")
        hashes[name] = sha256_file(directory / "model.pt")
        sizes[name] = predictor.size_report()
    return hashes, sizes


def _score_saved_candidates(names, spec, schema, frames, output_dir, model_hashes, timing):
    """Score each arm from its own saved model and preprocessing state."""
    from .tabicl_adapter import CompactPredictor

    registrations, evaluation_scores, results = {}, {}, {}
    calibration_ids = _row_ids(frames["calibration"], schema, "calibration")
    families = (frames["evaluation"][schema.family_column].fillna("<missing>").astype(str).tolist()
                if schema.family_column else None)
    for name in names:
        directory = output_dir / name
        prep_path = directory / "preprocessor.joblib"
        prep_hash = sha256_file(prep_path)
        preprocessor = CSVPreprocessor.load(prep_path, expected_sha256=prep_hash)
        predictor = CompactPredictor.load(directory / "model.pt", device="cpu")
        normal_scores, timing[f"{name}_calibration_csv_score_seconds"] = _score_frame(
            predictor, preprocessor, frames["calibration"], spec["query_batch_size"])
        registration = register_threshold(
            normal_scores, spec["alpha"], model_sha256=model_hashes[name], preprocessing_sha256=prep_hash,
            calibration_sha256=spec["files"]["calibration"]["sha256"])
        registrations[name] = registration
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
        del predictor, preprocessor
    return registrations, evaluation_scores, results, families


def _write_pair(left, right, *, schema, frame, scores, registrations, families, output, rename=False):
    paired = compare_decisions(
        labels(frame, schema), scores[left], registrations[left], scores[right], registrations[right],
        ids=_row_ids(frame, schema, "evaluation"),
        require_same_preprocessing=not (rename and (left, right) == ("legacy_full", "full")))
    rows = pd.DataFrame(paired.pop("rows"))
    if rename:
        rows.rename(columns={f"full_{suffix}": f"{left}_{suffix}" for suffix in ("score", "predicted")}
                    | {f"qcoc_{suffix}": f"{right}_{suffix}" for suffix in ("score", "predicted")}, inplace=True)
        paired = {"reference_name": left, "candidate_name": right,
                  "reference": paired.pop("full"), "candidate": paired.pop("qcoc"),
                  "reference_binding": paired.pop("full_binding"),
                  "candidate_binding": paired.pop("qcoc_binding"), **paired}
    if families is not None:
        rows["attack_family"] = families
    rows.to_csv(output, index=False)
    return paired


def _run_comparison(config_path, output_dir, *, input_comparison):
    import torch
    from .tabicl_adapter import FullCacheAdapter

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    spec, schema, paths, frames, audit = preflight_protocol(config_path, audit_representations=False)
    expected_protocol = PROTOCOL_V2 if input_comparison else PROTOCOL
    if spec["protocol"] != expected_protocol:
        raise ValueError(f"this runner requires {expected_protocol}")
    output_dir.mkdir(parents=True)
    torch.set_num_threads(spec["threads"])
    torch.manual_seed(spec["seed"])
    snapshot = _code_snapshot()
    write_json(output_dir / "protocol.json", spec)
    write_json(output_dir / "source_snapshot.json", snapshot)
    write_json(output_dir / "environment.json", environment_snapshot())
    timing, preprocessing_reports, model_hashes, tensor_sizes = {}, {}, {}, {}
    groups = (("legacy", INPUT_PROFILES["legacy"], "legacy_full", False),
              ("new", INPUT_PROFILES["new"], "full", True)) if input_comparison else (
                  ("new", None, "full", True),)
    if input_comparison:
        from .representation_audit import audit_final_inputs

        audit["representations"] = {}
    compilation = None
    for group, profile, full_name, compile_compact in groups:
        start = time.perf_counter()
        preprocessor = (CSVPreprocessor(schema) if profile is None else
                        CSVPreprocessor(schema, profile=profile)).fit(frames["support"])
        prep_path = output_dir / (f"{group}_preprocessor.joblib" if input_comparison else "preprocessor.joblib")
        preprocessor.save(prep_path)
        prep_hash = sha256_file(prep_path)
        matrix = preprocessor.transform(frames["support"])
        prefix = f"{group}_" if input_comparison else ""
        timing[f"{prefix}support_preprocessing_seconds"] = time.perf_counter() - start
        preprocessing_reports[group] = preprocessor.report()
        if input_comparison:
            representation = audit_final_inputs(preprocessor, frames, schema)
            audit["representations"][group] = representation
            write_json(output_dir / f"representation_audit_{group}.json", representation)
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
        timing[f"{full_name}_context_build_seconds"] = time.perf_counter() - start
        candidates = {full_name: adapter.as_predictor()}
        if compile_compact:
            start = time.perf_counter()
            compact, compilation = adapter.compile(slots=spec["slots"], query_chunk_size=spec["query_chunk_size"])
            if spec["build_device"] == "cuda":
                torch.cuda.synchronize()
            timing["qcoc_compile_seconds"] = time.perf_counter() - start
            candidates["qcoc"] = compact
            del compact
        hashes, sizes = _save_candidates(candidates, prep_path, output_dir)
        model_hashes.update(hashes)
        tensor_sizes.update(sizes)
        # Release build-device objects before any arm is reloaded for CPU scoring.
        del candidates, adapter, matrix, preprocessor
    write_json(output_dir / "data_audit.json", audit)
    names = ("legacy_full", "full", "qcoc") if input_comparison else ("full", "qcoc")
    registrations, evaluation_scores, results, families = _score_saved_candidates(
        names, spec, schema, frames, output_dir, model_hashes, timing)
    if input_comparison:
        paired = {}
        for left, right in (("legacy_full", "full"), ("full", "qcoc")):
            paired[f"{left}_vs_{right}"] = _write_pair(
                left, right, schema=schema, frame=frames["evaluation"], scores=evaluation_scores,
                registrations=registrations, families=families,
                output=output_dir / f"paired_{left}_{right}.csv", rename=True)
        rows = pd.DataFrame({"row_id": _row_ids(frames["evaluation"], schema, "evaluation"),
                             "label": labels(frames["evaluation"], schema)})
        for name in names:
            rows[f"{name}_score"] = evaluation_scores[name]
            rows[f"{name}_predicted"] = apply_threshold(evaluation_scores[name], registrations[name])
        if families is not None:
            rows["attack_family"] = families
        rows.to_csv(output_dir / "evaluation_scores.csv", index=False)
    else:
        paired = _write_pair("full", "qcoc", schema=schema, frame=frames["evaluation"], scores=evaluation_scores,
                             registrations=registrations, families=families,
                             output=output_dir / "paired_evaluation.csv")
    packaged = {name: {"files": {p.name: p.stat().st_size for p in (output_dir / name).iterdir() if p.is_file()},
                       "total_file_bytes": sum(p.stat().st_size for p in (output_dir / name).iterdir() if p.is_file())}
                for name in names}
    report = {"protocol": spec["protocol"], "scope": "development_comparison_not_final_confirmation",
              "data_kind": spec["data_kind"], "candidate_count": len(names), "slots": spec["slots"],
              "support_rows": len(frames["support"]), "alpha": registrations["full"]["alpha"],
              "single_view": True,
              "nontrivial_thresholds": {name: not value["reject_all"] for name, value in registrations.items()},
              "prediction_profile": {"device": "cpu", "dtype": "float32",
                                     "threads": spec["threads"], "query_batch_size": spec["query_batch_size"]},
              "preprocessing": preprocessing_reports if input_comparison else preprocessing_reports["new"],
              "registrations": registrations, "evaluation": results, "paired": paired, "data_audit": audit,
              "compilation": compilation, "native_tensor_sizes": tensor_sizes,
              "complete_csv_deployment_files": packaged,
              "timing": {"kind": "single_run_wall_clock_not_latency_or_RSS_benchmark", **timing},
              "source_snapshot": snapshot,
              "input_hashes": {role: spec["files"][role]["sha256"] for role in ROLES},
              "checkpoint_sha256": spec["checkpoint"]["sha256"]}
    if input_comparison:
        report.update({**V2_POLICIES, "preparation": dict(spec["preparation"]),
                       "arm_profiles": {"legacy_full": INPUT_PROFILES["legacy"],
                                        "full": INPUT_PROFILES["new"], "qcoc": INPUT_PROFILES["new"]},
                       "selection": "three fixed arms; no evaluation-based model, budget or threshold selection"})
    write_json(output_dir / "comparison.json", report)
    return report


def run_comparison(config_path, output_dir):
    """Execute the unchanged v1 full KV / QCOC comparison contract."""
    return _run_comparison(config_path, output_dir, input_comparison=False)


def run_input_comparison(config_path, output_dir):
    """Execute exactly legacy full, protected-input full, and protected-input QCOC."""
    return _run_comparison(config_path, output_dir, input_comparison=True)
