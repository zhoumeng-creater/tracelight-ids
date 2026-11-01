"""Group-disjoint teacher targets generated exclusively inside training data.

OOF teacher scores are class-balanced attack scores, not calibrated posterior
probabilities. Each teacher fits its own preprocessing and HGB on other folds;
no validation, calibration or test partition is loaded. Verification hashes
teacher model bytes without deserializing them.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import load_partition
from .models import save_bundle, train_model

_IDENTITY_COLUMNS = ["sample_id", "feature_hash", "label"]
_BASE_FILES = {"config.json", "dataset.json", "code.json", "environment.json", "targets.parquet", "folds.json"}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(frame):
    if not set(_IDENTITY_COLUMNS).issubset(frame.columns):
        raise ValueError("OOF identity requires sample_id, feature_hash and label")
    identity = frame.loc[:, _IDENTITY_COLUMNS].reset_index(drop=True).copy()
    for key in ("sample_id", "feature_hash"):
        if identity[key].isna().any() or not identity[key].map(lambda v: isinstance(v, str) and bool(v)).all():
            raise ValueError(f"OOF {key} must contain nonempty strings")
    if identity["sample_id"].duplicated().any():
        raise ValueError("OOF sample IDs must be unique")
    if not identity["label"].isin([0, 1]).all():
        raise ValueError("OOF labels must be binary")
    identity["label"] = identity["label"].astype(int)
    return identity


def _identity_hash(frame):
    return canonical_hash(_identity(frame).to_dict(orient="records"))


def _teacher_settings(teacher_config, threads):
    if not isinstance(teacher_config, dict) or teacher_config.get("model") != "hgb":
        raise ValueError("OOF teacher must be an HGB configuration")
    if "distillation" in teacher_config:
        raise ValueError("OOF teacher cannot itself use distillation")
    effective = copy.deepcopy(teacher_config)
    features = effective.get("features") or {"method": "all"}
    if features.get("method", "all") != "all" or features.get("k") is not None:
        raise ValueError("OOF teacher must use all raw fields")
    effective["features"] = features
    hp = effective.setdefault("hyperparameters", {})
    if hp.get("early_stopping", False) is not False:
        raise ValueError("OOF HGB teacher requires early_stopping=False")
    if "num_threads" in hp and hp["num_threads"] != threads:
        raise ValueError("Teacher num_threads must match OOF threads")
    hp["early_stopping"] = False
    hp["num_threads"] = threads
    return effective


def _fold_assignment(identity, folds, fold_seed):
    _integer(folds, "folds", 2)
    _integer(fold_seed, "fold_seed")
    labels = identity["label"].to_numpy()
    groups = identity["feature_hash"].to_numpy()
    if set(labels) != {0, 1} or len(set(groups)) < folds:
        raise ValueError("OOF training requires both labels and enough feature groups")
    for label in (0, 1):
        if len(set(groups[labels == label])) < folds:
            raise ValueError("OOF training needs at least folds feature groups containing each class")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    assignment = np.full(len(identity), -1, dtype=np.int64)
    partitions = []
    for index, (train_index, heldout_index) in enumerate(splitter.split(np.zeros(len(identity)), labels, groups)):
        if set(labels[train_index]) != {0, 1} or not len(heldout_index):
            raise ValueError("OOF fold has insufficient training class support or no heldout rows")
        if set(groups[train_index]) & set(groups[heldout_index]):
            raise ValueError("OOF feature groups crossed a train/heldout boundary")
        if np.any(assignment[heldout_index] != -1):
            raise ValueError("OOF heldout folds overlap")
        assignment[heldout_index] = index
        partitions.append((train_index, heldout_index))
    if np.any(assignment < 0):
        raise ValueError("OOF folds do not cover all training rows")
    return assignment, partitions


def _subset_identity(frame, prefix):
    identity = _identity(frame)
    return {f"n_{prefix}_rows": len(identity),
            f"n_{prefix}_groups": identity["feature_hash"].nunique(),
            f"{prefix}_sample_ids_sha256": canonical_hash(sorted(identity["sample_id"].tolist())),
            f"{prefix}_feature_groups_sha256": canonical_hash(sorted(set(identity["feature_hash"]))),
            f"{prefix}_identity_sha256": _identity_hash(identity)}


def build_oof_targets(data_dir, teacher_config, output_dir, seed, folds=3, fold_seed=20261101, threads=4) -> Path:
    """Create or verify a completed OOF artifact; never overwrite partial work.

    A repeated call on a complete matching artifact performs verification only.
    An incomplete directory must be preserved/restarted by the orchestration
    layer, rather than silently reused. All fold models are analysis artifacts;
    they are not included in a student's deployment bundle.
    """
    _integer(seed, "seed")
    _integer(threads, "threads", 1)
    _integer(folds, "folds", 2)
    _integer(fold_seed, "fold_seed")
    effective = _teacher_settings(teacher_config, threads)
    data_dir, root = Path(data_dir), Path(output_dir)
    code = code_fingerprint()
    expected = {"teacher_config": teacher_config, "seed": seed, "folds": folds,
                "fold_seed": fold_seed, "threads": threads, "source_sha256": code["source_sha256"],
                "dataset_manifest_sha256": sha256_file(data_dir / "manifest.json")}
    if root.exists():
        if not (root / "completion.json").is_file():
            raise FileExistsError("Incomplete OOF output exists; preserve it and restart in a new directory")
        verify_oof_targets(root, data_dir, expected)
        return root
    begin = time.perf_counter()
    train = load_partition(data_dir, "train")
    identity = _identity(train)
    assignment, partitions = _fold_assignment(identity, folds, fold_seed)
    dataset = read_json(data_dir / "manifest.json")
    config = {"teacher_config": copy.deepcopy(teacher_config), "effective_teacher_config": effective,
              "seed": seed, "folds": folds, "fold_seed": fold_seed, "threads": threads}
    root.mkdir(parents=True)
    write_json(root / "config.json", config)
    write_json(root / "dataset.json", dataset)
    write_json(root / "code.json", code)
    write_json(root / "environment.json", environment_snapshot(include_torch=False))
    scores = np.full(len(train), np.nan, dtype=np.float64)
    fold_records = []
    for index, (train_index, heldout_index) in enumerate(partitions):
        fit_frame, heldout = train.iloc[train_index], train.iloc[heldout_index]
        fit_begin = time.perf_counter()
        with threadpool_limits(limits=threads):
            # HGB has no validation selection. Passing only its fitting rows
            # for diagnostic metrics keeps even heldout labels out of fitting.
            bundle = train_model(fit_frame, fit_frame, effective, seed=seed, device="cpu")
        fit_seconds = time.perf_counter() - fit_begin
        predict_begin = time.perf_counter()
        with threadpool_limits(limits=threads):
            fold_scores = np.asarray(bundle.predict_scores(heldout), dtype=np.float64)
        predict_seconds = time.perf_counter() - predict_begin
        if fold_scores.shape != (len(heldout),) or not np.isfinite(fold_scores).all() or np.any(
                (fold_scores < 0) | (fold_scores > 1)):
            raise ValueError("OOF teacher produced invalid heldout scores")
        scores[heldout_index] = fold_scores
        model_file = f"fold-{index:02d}.joblib"
        serialize_begin = time.perf_counter()
        save_bundle(bundle, root / model_file)
        serialization_seconds = time.perf_counter() - serialize_begin
        fold_records.append({"fold": index, "teacher_seed": seed,
                             **_subset_identity(fit_frame, "train"), **_subset_identity(heldout, "holdout"),
                             "model_file": model_file, "model_sha256": sha256_file(root / model_file),
                             "selected_features": bundle.selected_features, "encoded_dim": bundle.encoded_dim,
                             "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                             "serialization_seconds": serialization_seconds,
                             "teacher_diagnostics_split": "fold_training_rows_only_no_validation_selection"})
    targets = identity.assign(fold=assignment, teacher_score=scores)
    temporary = root / "targets.parquet.tmp"
    targets.to_parquet(temporary, index=False)
    temporary.replace(root / "targets.parquet")
    write_json(root / "folds.json", {"splitter": "StratifiedGroupKFold", "shuffle": True,
                                     "fold_seed": fold_seed, "folds": fold_records})
    if code_fingerprint()["source_sha256"] != code["source_sha256"]:
        raise ValueError("Source changed during OOF generation; incomplete artifact is not reusable")
    manifest = {"schema_version": 1, "kind": "group_oof_teacher_targets", **expected,
                "train_partition_sha256": dataset["partitions"]["train"]["sha256"],
                "train_identity_sha256": _identity_hash(identity),
                "train_identity_order": "original prepared train row order; canonical list of sample_id,feature_hash,label records",
                "n_training_rows": len(train), "config_sha256": canonical_hash(config),
                "score_semantics": "class-balanced teacher attack scores; not calibrated probabilities",
                "splitter": "StratifiedGroupKFold", "group_column": "feature_hash",
                "generation_seconds": time.perf_counter() - begin,
                "fold_fit_seconds_total": sum(f["fit_seconds"] for f in fold_records),
                "heldout_predict_seconds_total": sum(f["predict_seconds"] for f in fold_records),
                "model_serialization_seconds_total": sum(f["serialization_seconds"] for f in fold_records),
                "timing_scope": "fit includes train-only preprocessing and fitting-row diagnostics; generation includes preparation and artifact writes",
                "files": {p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}}
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(sha256_file(root / "manifest.json") + "\n", encoding="ascii")
    # Completion is written last via atomic replace. Its absence seals a failed
    # attempt as incomplete; it never enables partial-target reuse.
    write_json(root / "completion.json", {"schema_version": 1, "status": "complete",
                                          "manifest_sha256": sha256_file(root / "manifest.json"),
                                          "completed_at": utc_now()})
    verify_oof_targets(root, data_dir, expected)
    return root


def verify_oof_targets(path, data_dir=None, expected=None) -> dict:
    """Verify completion, bytes, fold isolation and identity without loading models.

    When data_dir is provided, load only its train partition and require exact
    original row identity. expected is a subset of manifest binding fields;
    callers can require the current teacher/source/seed settings for resume.
    """
    root = Path(path)
    if not (root / "completion.json").is_file():
        raise ValueError("OOF artifact is incomplete: completion.json is missing")
    completion = read_json(root / "completion.json")
    manifest_hash = sha256_file(root / "manifest.json")
    if completion.get("status") != "complete" or completion.get("manifest_sha256") != manifest_hash or (
            (root / "manifest.sha256").read_text().strip() != manifest_hash):
        raise ValueError("OOF completion or manifest checksum mismatch")
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "group_oof_teacher_targets":
        raise ValueError("Unsupported OOF artifact schema")
    folds = _integer(manifest["folds"], "folds", 2)
    allowed = _BASE_FILES | {f"fold-{index:02d}.joblib" for index in range(folds)}
    if set(manifest["files"]) != allowed:
        raise ValueError("OOF artifact file inventory is invalid")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not (root / name).is_file() or sha256_file(root / name) != digest:
            raise ValueError(f"OOF artifact changed: {name}")
    if set(p.name for p in root.iterdir()) != allowed | {"manifest.json", "manifest.sha256", "completion.json"}:
        raise ValueError("OOF artifact contains unexpected files or subdirectories")
    if expected and any(key not in manifest or manifest[key] != value for key, value in expected.items()):
        raise ValueError("OOF artifact does not match expected configuration or identity")
    config, dataset, code = (read_json(root / name) for name in ("config.json", "dataset.json", "code.json"))
    if canonical_hash(config) != manifest["config_sha256"]:
        raise ValueError("OOF config identity mismatch")
    for key in ("teacher_config", "seed", "folds", "fold_seed", "threads"):
        if config[key] != manifest[key]:
            raise ValueError(f"OOF configuration binding mismatch: {key}")
    _integer(config["seed"], "seed")
    _integer(config["threads"], "threads", 1)
    if config["effective_teacher_config"] != _teacher_settings(config["teacher_config"], config["threads"]):
        raise ValueError("OOF effective teacher configuration mismatch")
    if sha256_file(root / "dataset.json") != manifest["dataset_manifest_sha256"] or (
            dataset["partitions"]["train"]["sha256"] != manifest["train_partition_sha256"]):
        raise ValueError("OOF dataset/train partition binding mismatch")
    if canonical_hash(code["files"]) != code["source_sha256"] or code["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("OOF source fingerprint mismatch")
    targets = pd.read_parquet(root / "targets.parquet")
    if set(targets.columns) != set(_IDENTITY_COLUMNS) | {"fold", "teacher_score"}:
        raise ValueError("OOF target columns are invalid")
    identity = _identity(targets)
    if len(identity) != manifest["n_training_rows"] or len(identity) != dataset["partitions"]["train"]["n_rows"] or (
            _identity_hash(identity) != manifest["train_identity_sha256"]):
        raise ValueError("OOF training identity or row coverage mismatch")
    scores = targets["teacher_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("OOF teacher scores must be finite and in [0,1]")
    assignment, partitions = _fold_assignment(identity, folds, config["fold_seed"])
    if not np.array_equal(targets["fold"].to_numpy(), assignment):
        raise ValueError("OOF fold assignments differ from the declared group split")
    fold_info = read_json(root / "folds.json")
    if fold_info.get("splitter") != "StratifiedGroupKFold" or fold_info.get("shuffle") is not True or (
            fold_info.get("fold_seed") != config["fold_seed"] or len(fold_info["folds"]) != folds):
        raise ValueError("OOF fold metadata is inconsistent")
    for index, (fit, heldout) in enumerate(partitions):
        item = fold_info["folds"][index]
        expected_fold = {"fold": index, "teacher_seed": config["seed"],
                         **_subset_identity(identity.iloc[fit], "train"),
                         **_subset_identity(identity.iloc[heldout], "holdout"),
                         "model_file": f"fold-{index:02d}.joblib"}
        if any(item.get(key) != value for key, value in expected_fold.items()):
            raise ValueError("OOF fold train/holdout identity mismatch")
        if item["model_sha256"] != manifest["files"][item["model_file"]] or (
                len(item["selected_features"]) != len(dataset["features"])
                or set(item["selected_features"]) != set(dataset["features"])
                or type(item["encoded_dim"]) is not int or item["encoded_dim"] < len(dataset["features"])):
            raise ValueError("OOF fold model binding or teacher feature set mismatch")
        for key in ("fit_seconds", "predict_seconds", "serialization_seconds"):
            if not math.isfinite(item[key]) or item[key] < 0:
                raise ValueError("OOF timing must be finite and nonnegative")
    for total, per_fold in (("fold_fit_seconds_total", "fit_seconds"),
                            ("heldout_predict_seconds_total", "predict_seconds"),
                            ("model_serialization_seconds_total", "serialization_seconds")):
        if not math.isclose(manifest[total], sum(x[per_fold] for x in fold_info["folds"]), rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("OOF total timing differs from fold records")
    if not math.isfinite(manifest["generation_seconds"]) or manifest["generation_seconds"] < 0:
        raise ValueError("OOF generation timing is invalid")
    measured_total = sum(manifest[key] for key in ("fold_fit_seconds_total", "heldout_predict_seconds_total",
                                                  "model_serialization_seconds_total"))
    if manifest["generation_seconds"] < measured_total:
        raise ValueError("OOF generation duration is shorter than measured fold operations")
    if data_dir is not None:
        data_dir = Path(data_dir)
        if sha256_file(data_dir / "manifest.json") != manifest["dataset_manifest_sha256"]:
            raise ValueError("OOF targets were generated for a different dataset")
        train = load_partition(data_dir, "train")
        if not identity.equals(_identity(train)):
            raise ValueError("OOF target identity differs from the prepared training partition")
    return manifest


def load_oof_scores(path, train_df, data_dir=None, expected=None) -> np.ndarray:
    """Return verified scores in train_df order, rejecting extra/missing/moved IDs."""
    verify_oof_targets(path, data_dir, expected)
    supplied = _identity(train_df)
    targets = pd.read_parquet(Path(path) / "targets.parquet").set_index("sample_id")
    ordered = supplied.set_index("sample_id")
    if len(ordered) != len(targets) or set(ordered.index) != set(targets.index):
        raise ValueError("OOF targets do not cover exactly the supplied training sample IDs")
    targets = targets.loc[ordered.index]
    if not np.array_equal(targets["feature_hash"], ordered["feature_hash"]) or not np.array_equal(
            targets["label"], ordered["label"]):
        raise ValueError("OOF supplied training labels or feature groups are misaligned")
    return targets["teacher_score"].to_numpy(dtype=np.float64, copy=True)
