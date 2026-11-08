"""Frozen support-label experiment registration, read-only preflight and packaging."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, write_json
from .calibration import apply_threshold, register_threshold
from .csv_workflow import ROLES, V2_POLICIES, audit_roles
from .lab_data import SCHEMA
from .preprocessing import CSVSchema, labels, read_csv, require_pinned_preprocessing


PROTOCOL = "tracelight-scanning-label-v1"
ARMS = ["baseline", "semantic_family", "grouped_sham"]
FILE_IDENTITIES = {
    "support": ("prepared/support.csv", 8192, "ee6c1470f90caa3fcf5d47e884db71b82a6a5ab574ad8ebfe61cc5ba120a8fbf"),
    "calibration": ("prepared/normal_calibration.csv", 5000, "fe0d56b6d87411f6b24af534a5543ea0774fdd1812fdf224572b5f6c750e24b2"),
    "evaluation": ("prepared/development_evaluation.csv", 20000, "20883897adfa29498765174666a88dc2cc206a2f30f659476b07e416ee81f408"),
}
ARTIFACT_IDENTITIES = {
    "preparation": ("prepared/preparation.json", "f6de37c295fb82a59d73e17b0f17184636f32b2e033d9847be245f1fa35c12e7"),
    "preprocessor": ("comparison/full/preprocessor.joblib", "7a3189d5c3d6393b16600e9c84e4aea4ab4977d24f660ed4a8a996b5212161fb"),
    "baseline_calibration": ("comparison/full_calibration.csv", "0c2093830579ccfa7d1b2f213810c63bb0fd6275afb9dc937d58ad01a6305333"),
    "baseline_evaluation": ("comparison/evaluation_scores.csv", "64ad5b5b0dc9133ba78a909741157253e63290cc16c2abb6f403c79f1d855a26"),
    "baseline_threshold": ("comparison/full/threshold.json", "198e312e6f928d16f01b8d1a882c69ece91d45eb883f4c0b431ec4497626aad6"),
}
CHECKPOINT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
BASELINE_MODEL_SHA256 = "61ffaf8a225026cfbe67f7e79dc3a9e7e53b9fed6297dbe26c88d7cee5432179"
FIXED = {"seed": 20261107, "sham_seed": 20261108, "threads": 1,
         "sham_class_rows": 512, "query_batch_size": 128, "build_device": "cuda", "alpha": .01,
         "center": 35989.91845703125, "scale": 21268.044830665465,
         "width": 48, "score_atol": 3e-5, "score_rtol": 3e-5}


def verify_label_code_freeze(root=None):
    """Verify the release's LF-normalized source inventory, without loading models."""
    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[3]
    manifest_path = root / "configs/scanning-label-code-freeze-v1.json"
    manifest = read_json(manifest_path)
    files = manifest.get("files")
    if (manifest.get("format") != "tracelight-scanning-label-code-freeze-v1"
            or not isinstance(files, dict) or not files):
        raise ValueError("invalid Scanning-label code freeze manifest")
    required = {"src/ids/common.py", "configs/context-scanning-label-v1.json", "scripts/RUN_SCANNING_LABEL.ps1",
                "pyproject.toml", "requirements-lock.txt"}
    required.update(p.relative_to(root).as_posix() for p in (root / "src/ids/context_cache").glob("*.py"))
    if not required <= set(files):
        raise ValueError(f"code freeze omits required files: {sorted(required - set(files))}")
    observed = {}
    for name, expected in files.items():
        _sha(expected)
        path = (root / name).resolve()
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or not path.is_relative_to(root) or path == manifest_path):
            raise ValueError("code freeze contains an external or self-referential path")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"frozen source SHA mismatch: {name}")
        observed[name] = actual
    return {"manifest_sha256": sha256_file(manifest_path), "verified_files": len(observed),
            "lf_normalized_sha256": observed, "combined_sha256": canonical_hash(observed)}


def _verify_runtime(environment):
    frozen = {"torch": "2.6.0+cu126", "numpy": "1.26.4", "pandas": "2.2.2",
              "scikit-learn": "1.5.1", "joblib": "1.4.2"}
    if (not environment.get("python", "").startswith("3.12.")
            or any(environment.get("packages", {}).get(k) != v for k, v in frozen.items())
            or environment.get("torch", {}).get("cuda_runtime") != "12.6"
            or not environment.get("torch", {}).get("cuda_available")):
        raise RuntimeError("development requires frozen Python3.12/Torch2.6.0+cu126/CUDA12.6 and preprocessing packages")


def frozen_template():
    """Return the only development experiment; registration changes paths only."""
    return {"protocol": PROTOCOL, "data_kind": "development", "schema": copy.deepcopy(SCHEMA),
            "files": {k: {"path": f"ROUND14/{p}", "rows": n, "sha256": h}
                      for k, (p, n, h) in FILE_IDENTITIES.items()},
            "artifacts": {k: {"path": f"ROUND14/{p}", "sha256": h}
                          for k, (p, h) in ARTIFACT_IDENTITIES.items()},
            "checkpoint": {"path": "CHECKPOINT/tabicl-classifier-v2-20260212.ckpt",
                           "sha256": CHECKPOINT_SHA256},
            "baseline_model_sha256": BASELINE_MODEL_SHA256, "arms": ARMS.copy(), **FIXED}


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA-256")
    return value


def _resolve(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be nonempty")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_label_protocol(config):
    config = Path(config).resolve()
    spec = json.loads(config.read_text(encoding="utf-8-sig"))
    expected = frozen_template()
    if set(spec) != set(expected) or spec.get("protocol") != PROTOCOL:
        raise ValueError("unsupported Scanning-label protocol fields")
    if spec["data_kind"] not in {"development", "synthetic"}:
        raise ValueError("Scanning-label accepts development or explicitly synthetic data")
    schema = CSVSchema.from_dict(spec["schema"])
    if (len(schema.features) != 48 or schema.features[0] != "L4_SRC_PORT"
            or not all((schema.id_column, schema.group_column, schema.family_column))):
        raise ValueError("Scanning-label requires 48 columns, source port first, ID/group/family metadata")
    if spec["arms"] != ARMS:
        raise ValueError("Scanning-label requires exactly the three frozen arms in order")
    if set(spec["files"]) != set(ROLES) or set(spec["artifacts"]) != set(ARTIFACT_IDENTITIES):
        raise ValueError("Scanning-label requires all frozen data and artifact roles")
    for key in ("threads", "query_batch_size", "width", "sham_class_rows"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"invalid {key}")
    for key in ("seed", "sham_seed"):
        if type(spec[key]) is not int or not 0 <= spec[key] < 2**32:
            raise ValueError(f"invalid {key}")
    if (spec["width"] != 48 or spec["sham_seed"] != 20261108 or spec["alpha"] != .01
            or spec["score_atol"] != 3e-5 or spec["score_rtol"] != 3e-5
            or spec["build_device"] not in {"cpu", "cuda"}
            or not np.isfinite([spec["center"], spec["scale"]]).all() or spec["scale"] <= 0):
        raise ValueError("invalid fixed intervention, tolerance, calibration or scaling contract")
    _sha(spec["baseline_model_sha256"])
    paths = {}
    all_items = {**spec["files"], **spec["artifacts"], "checkpoint": spec["checkpoint"]}
    for role, item in all_items.items():
        if set(item) != ({"path", "sha256", "rows"} if role in ROLES else {"path", "sha256"}):
            raise ValueError(f"invalid artifact fields for {role}")
        _sha(item["sha256"])
        paths[role] = _resolve(config.parent, item["path"])
        if role in ROLES and (type(item["rows"]) is not int or item["rows"] < 1):
            raise ValueError("rows must be positive integers")
    if len(set(paths.values())) != len(paths):
        raise ValueError("data and artifact roles must use distinct files")
    if spec["data_kind"] == "development":
        if (schema.to_dict() != CSVSchema.from_dict(SCHEMA).to_dict()
                or any(spec[k] != v for k, v in FIXED.items())
                or spec["baseline_model_sha256"] != BASELINE_MODEL_SHA256):
            raise ValueError("development Scanning-label parameters differ from frozen protocol")
        for role, (_, rows, digest) in FILE_IDENTITIES.items():
            if spec["files"][role]["rows"] != rows or spec["files"][role]["sha256"] != digest:
                raise ValueError(f"development identity differs for {role}")
        for role, (_, digest) in ARTIFACT_IDENTITIES.items():
            if spec["artifacts"][role]["sha256"] != digest:
                raise ValueError(f"development identity differs for {role}")
        if spec["checkpoint"]["sha256"] != CHECKPOINT_SHA256:
            raise ValueError("development checkpoint differs")
    elif spec["build_device"] != "cpu":
        raise ValueError("synthetic checks use CPU")
    return spec, schema, paths


def register_label_protocol(*, round14_root, checkpoint, output):
    output, root = Path(output).resolve(), Path(round14_root).resolve()
    if output.exists():
        raise FileExistsError(output)
    spec = frozen_template()
    for role, (relative, _, _) in FILE_IDENTITIES.items():
        spec["files"][role]["path"] = str(root / relative)
    for role, (relative, _) in ARTIFACT_IDENTITIES.items():
        spec["artifacts"][role]["path"] = str(root / relative)
    spec["checkpoint"]["path"] = str(Path(checkpoint).resolve())
    for item in [*spec["files"].values(), *spec["artifacts"].values(), spec["checkpoint"]]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen artifact SHA mismatch: {item['path']}")
    write_json(output, spec)
    return {"stage": "registered_frozen_label_paths", "output": str(output),
            "protocol_sha256": sha256_file(output), "data_kind": "development"}


def _preparation_audit(spec, schema, paths):
    receipt = read_json(paths["preparation"])
    if (receipt.get("format") != "tracelight-nf-ton-v3-preparation-v2"
            or receipt.get("data_kind") != spec["data_kind"]
            or receipt.get("seed") != spec["seed"]
            or CSVSchema.from_dict(receipt.get("schema", {})).to_dict() != schema.to_dict()
            or any(receipt.get(k) != v for k, v in V2_POLICIES.items())):
        raise ValueError("preparation contract differs")
    for role in ROLES:
        item = receipt.get("files", {}).get(role, {})
        if any(item.get(k) != spec["files"][role][k] for k in ("sha256", "rows")):
            raise ValueError(f"preparation binding differs for {role}")
    if spec["data_kind"] == "development" and receipt.get("source", {}).get("official_release_verified") is not True:
        raise ValueError("official source receipt missing")
    return {"sha256": spec["artifacts"]["preparation"]["sha256"],
            "csv_hashes_and_rows_match": True, "paths": "original paths retained as provenance; relocation allowed"}


def _predictions(values):
    strings = pd.Series(values).astype(str).str.lower()
    if not strings.isin(["true", "false", "0", "1"]).all():
        raise ValueError("invalid archived predicted labels")
    return strings.isin(["true", "1"]).to_numpy()


def preflight_label(config):
    """Read hashes, CSVs and JSON only; no pickle/checkpoint loading or fitting."""
    spec, schema, paths = load_label_protocol(config)
    code_freeze = verify_label_code_freeze() if spec["data_kind"] == "development" else None
    items = {**spec["files"], **spec["artifacts"], "checkpoint": spec["checkpoint"]}
    for role, item in items.items():
        if sha256_file(paths[role]) != item["sha256"]:
            raise ValueError(f"registered SHA-256 mismatch for {role}")
    frames = {role: read_csv(paths[role], schema) for role in ROLES}
    for role in ROLES:
        if len(frames[role]) != spec["files"][role]["rows"]:
            raise ValueError(f"row count differs for {role}")
    audit = audit_roles(frames, schema, exact_feature_policy="report")
    support_families = frames["support"][schema.family_column].astype(str)
    support_binary = labels(frames["support"], schema)
    if not np.array_equal(support_families.eq("Benign").to_numpy(), support_binary == 0):
        raise ValueError("support family and binary labels disagree")
    scan_rows = int(support_families.eq("Scanning").sum())
    if scan_rows != spec["sham_class_rows"] or not 0 < scan_rows < int(support_binary.sum()):
        raise ValueError("support Scanning count must match frozen sham quota and leave other attacks")
    if spec["data_kind"] == "development" and int((support_binary == 0).sum()) != 4096:
        raise ValueError("development requires the frozen 4096 normal support rows")
    audit["support_family_counts"] = support_families.value_counts().sort_index().to_dict()
    audit["three_class_counts"] = [int((support_binary == 0).sum()), scan_rows,
                                    int(support_binary.sum()) - scan_rows]
    audit["preparation"] = _preparation_audit(spec, schema, paths)
    registration = read_json(paths["baseline_threshold"])
    baseline = {}
    for role, name, score, prediction in (
        ("calibration", "baseline_calibration", "score", "predicted_attack"),
        ("evaluation", "baseline_evaluation", "full_score", "full_predicted"),
    ):
        saved = pd.read_csv(paths[name], dtype={"row_id": str}, float_precision="round_trip")
        if (len(saved) != len(frames[role]) or saved["row_id"].astype(str).tolist()
                != frames[role][schema.id_column].astype(str).tolist()
                or not np.array_equal(saved["label"].to_numpy(), labels(frames[role], schema))):
            raise ValueError(f"archived baseline row order or labels differ for {role}")
        values = saved[score].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite baseline scores for {role}")
        if not np.array_equal(_predictions(saved[prediction]), apply_threshold(values, registration)):
            raise ValueError(f"archived baseline decisions differ for {role}")
        if role == "evaluation":
            if saved["attack_family"].astype(str).tolist() != frames[role][schema.family_column].astype(str).tolist():
                raise ValueError("archived baseline attack families differ")
            _predictions(saved["legacy_full_predicted"])
            if not np.isfinite(saved["legacy_full_score"].to_numpy(dtype=float)).all():
                raise ValueError("nonfinite legacy scores")
        baseline[role] = values
    rebuilt = register_threshold(baseline["calibration"], spec["alpha"],
                                 model_sha256=spec["baseline_model_sha256"],
                                 preprocessing_sha256=spec["artifacts"]["preprocessor"]["sha256"],
                                 calibration_sha256=spec["files"]["calibration"]["sha256"])
    if registration != rebuilt:
        raise ValueError("archived baseline threshold rank, scores hash or bindings differ")
    _, upstream = require_pinned_preprocessing()
    environment = environment_snapshot()
    if spec["data_kind"] == "development":
        _verify_runtime(environment)
    if spec["build_device"] == "cuda" and not environment.get("torch", {}).get("cuda_available"):
        raise RuntimeError("frozen development build requires CUDA")
    audit.update({"baseline_row_order_labels_predictions_threshold_verified": True,
                  "upstream": upstream, "environment": environment,
                  "code_freeze": code_freeze,
                  "model_loads": 0, "preprocessor_loads": 0, "preprocessor_fits": 0,
                  "support_label_assignments": 0})
    return spec, schema, paths, frames, audit


def pack_label_results(run_root, output_zip):
    """Archive one run, including failures; verify every stored file after writing."""
    root, target = Path(run_root).resolve(), Path(output_zip).resolve()
    if not root.is_dir() or target.exists() or target.is_relative_to(root):
        raise ValueError("result root must exist; new ZIP must be outside the result root")
    status_path = root / "comparison" / "state.json"
    status = read_json(status_path) if status_path.is_file() else {"status": "partial"}
    if status.get("status") == "complete":
        from .label_workflow import verify_label_result
        verify_label_result(root / "comparison")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p != root / "archive_manifest.json")
    for path in files:
        if (path.is_symlink() or not path.resolve().is_relative_to(root)
                or path.suffix.lower() in {".ckpt", ".zip"}
                or path.name.lower() == "nf-ton-iot-v3.csv"
                or "raw" in path.relative_to(root).parts):
            raise ValueError(f"raw/checkpoint/nested archive or external file in result root: {path}")
    manifest = {"format": "tracelight-scanning-label-result-archive-v1", "run_status": status,
                "files": {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in files}}
    manifest_path = root / "archive_manifest.json"
    write_json(manifest_path, manifest)
    files.append(manifest_path)
    expected_hashes = {f"{root.name}/{name}": item["sha256"] for name, item in manifest["files"].items()}
    expected_hashes[f"{root.name}/archive_manifest.json"] = sha256_file(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, arcname=f"{root.name}/{path.relative_to(root).as_posix()}")
    with zipfile.ZipFile(target) as archive:
        expected_names = {f"{root.name}/{p.relative_to(root).as_posix()}" for p in files}
        if set(archive.namelist()) != expected_names:
            raise ValueError("archive file set differs")
        for path in files:
            name = f"{root.name}/{path.relative_to(root).as_posix()}"
            h = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected_hashes[name]:
                raise ValueError(f"archive readback hash differs: {name}")
    return {"archive": str(target), "sha256": sha256_file(target),
            "bytes": target.stat().st_size, "files": len(files), "run_status": status,
            "readback_verified": True}
