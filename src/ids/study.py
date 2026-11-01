"""Verified, validation-only study summaries; no model deserialization.

Seed standard deviations describe training variability on one fixed partition,
not independent test samples or confidence intervals. OR alerts below are an
exploratory complementarity diagnostic, not a calibrated ensemble.
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import canonical_hash, read_json, sha256_file
from .evaluation import calibrate_threshold, evaluate_binary
from .runner import verify_run

_REQUIRED = {
    "config.json", "dataset.json", "code.json", "environment.json", "model.joblib",
    "selected_features.json", "validation_predictions.parquet",
    "validation_metrics.json", "validation_threshold.json",
}
_ALLOWED = _REQUIRED | {"training_history.json", "examples.json", "distillation.json", "training_statistics.json", "selection.json"}
_METRICS = (
    "attack_recall", "normal_fpr", "precision", "accuracy", "macro_f1",
    "average_precision", "roc_auc", "training_seconds", "model_file_bytes", "encoded_dim",
    "parameter_count", "trainable_parameter_count", "actual_epochs", "best_epoch",
    "teacher_generation_seconds", "teacher_fold_fit_seconds_total",
)
_VARIANT_LABELS = {"hgb_reference": "全字段 HGB", "mlp_reference": "大 MLP",
                   "student_hard": "小 MLP（硬标签）", "student_kd": "蒸馏小 MLP"}
_ENVIRONMENT_KEYS = (
    "python", "os", "os_release", "machine", "processor", "cpu_physical", "cpu_logical",
    "ram_total_bytes", "packages", "torch", "nvidia_smi",
)


def _aggregate(values):
    """Keep seed-ordered nulls; use only defined observations for summaries."""
    values = [None if x is None else float(x) for x in values]
    observed = [x for x in values if x is not None]
    if not all(math.isfinite(x) for x in observed):
        raise ValueError("Study metrics must be finite or null")
    return {"mean": float(np.mean(observed)) if observed else None,
            "std": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
            "min": min(observed) if observed else None,
            "max": max(observed) if observed else None,
            "n": len(observed), "values": values}


def _environment(value):
    # Do not propagate arbitrary paths, account names or additional local metadata.
    return {key: value[key] for key in _ENVIRONMENT_KEYS if key in value}


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _same_number(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return math.isfinite(float(actual)) and math.isclose(
        float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)


def _validate_training_statistics(run_path, meta, config, code, encoded):
    """Check manifest-bound training metadata, without loading model weights."""
    if "training_statistics.json" not in meta["files"]:
        if config["model_config"].get("variant") in _VARIANT_LABELS:
            raise ValueError("Named study variant requires training_statistics.json")
        return None
    value = read_json(run_path / "training_statistics.json")
    expected = {"schema_version": 1, "run_id": meta["run_id"],
                "model_family": config["model_config"]["model"],
                "config_sha256": meta["config_sha256"], "source_sha256": code["source_sha256"],
                "dataset_manifest_sha256": meta["dataset_manifest_sha256"]}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Training statistics identity mismatch")
    if config["model_config"]["model"] != "mlp":
        if any(value.get(k) is not None for k in ("parameter_count", "trainable_parameter_count")):
            raise ValueError("Tree parameter counts must remain undefined")
        return value
    hyper = config["model_config"].get("hyperparameters", {})
    hidden = hyper.get("hidden_sizes", hyper.get("hidden_dims", [64, 32]))
    if not isinstance(hidden, (list, tuple)) or not hidden or any(type(x) is not int or x < 1 for x in hidden):
        raise ValueError("Invalid MLP hidden dimensions")
    widths = [encoded, *hidden, 1]
    expected_count = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
    if any(type(value.get(k)) is not int or value[k] != expected_count
           for k in ("parameter_count", "trainable_parameter_count")):
        raise ValueError("MLP parameter count differs from configured architecture")
    epochs, best = value.get("epochs_completed"), value.get("best_epoch")
    if (type(epochs) is not int or type(best) is not int
            or not 1 <= best <= epochs <= hyper.get("epochs", 60)
            or value.get("actual_epochs") != epochs):
        raise ValueError("MLP actual/best epoch metadata is inconsistent")
    if "training_history.json" not in meta["files"]:
        raise ValueError("MLP training statistics require a bound training history")
    history = read_json(run_path / "training_history.json")
    if (len(history) != epochs or [x.get("epoch") for x in history] != list(range(1, epochs + 1))
            or not _same_number(value.get("best_validation_loss"), history[best - 1].get("val_loss"))):
        raise ValueError("Training statistics disagree with saved history")
    positive_weight = value.get("positive_weight")
    if isinstance(positive_weight, bool) or not isinstance(positive_weight, (float, int)) or not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("Training positive weight must be positive and finite")
    return value


def _validate_distillation(run_path, meta, config, code, dataset, statistics):
    """Validate embedded OOF provenance; never reopen teacher/cache/test files."""
    model = config["model_config"]
    setting = model.get("distillation")
    variant = model.get("variant")
    if variant in _VARIANT_LABELS:
        expected_model = "hgb" if variant == "hgb_reference" else "mlp"
        if model["model"] != expected_model or model.get("features", {}).get("method", "all") != "all":
            raise ValueError("Named distillation-study variants require the correct model and all raw fields")
        if (variant == "student_kd") != bool(setting):
            raise ValueError("Distillation setting does not match named variant")
    if not setting:
        if "distillation.json" in meta["files"]:
            raise ValueError("Non-distilled run must not carry a distillation artifact")
        if statistics and model["model"] == "mlp" and statistics.get("training_objective") != "weighted_hard_bce":
            raise ValueError("Hard-label run has an inconsistent training objective")
        return None
    if not isinstance(setting, dict) or model["model"] != "mlp" or "distillation.json" not in meta["files"]:
        raise ValueError("Distilled MLP requires a bound distillation.json")
    value = read_json(run_path / "distillation.json")
    alpha, temperature = setting.get("alpha"), setting.get("temperature")
    if (isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not 0 < alpha <= 1
            or temperature != 1):
        raise ValueError("Unsupported distillation alpha or temperature")
    teacher = setting.get("teacher")
    if not isinstance(teacher, dict) or teacher.get("model") != "hgb" or teacher.get("features", {}).get("method", "all") != "all":
        raise ValueError("Distillation requires an all-field HGB teacher")
    expected = {"schema_version": 1, "method": "group_oof_hgb", "run_id": meta["run_id"],
                "config_sha256": meta["config_sha256"], "source_sha256": code["source_sha256"],
                "dataset_manifest_sha256": meta["dataset_manifest_sha256"], "training_split": "train",
                "alpha": alpha, "temperature": temperature, "teacher_config": teacher,
                "folds": setting.get("folds"), "fold_seed": setting.get("fold_seed")}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Distillation artifact identity mismatch")
    manifest = value.get("oof_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Distillation requires embedded OOF provenance")
    train = dataset.get("partitions", {}).get("train", {})
    expected_oof = {"schema_version": 1, "kind": "group_oof_teacher_targets", "teacher_config": teacher,
                    "seed": config["seed"], "folds": value["folds"], "fold_seed": value["fold_seed"],
                    "threads": config["threads"], "source_sha256": code["source_sha256"],
                    "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
                    "train_partition_sha256": train.get("sha256"), "n_training_rows": train.get("n_rows"),
                    "splitter": "StratifiedGroupKFold", "group_column": "feature_hash"}
    if any(manifest.get(k) != v or v is None for k, v in expected_oof.items()):
        raise ValueError("OOF provenance differs from training data, source, seed or fold settings")
    folds = manifest["folds"]
    if type(folds) is not int or folds < 2 or type(manifest["fold_seed"]) is not int:
        raise ValueError("Invalid OOF fold metadata")
    files = manifest.get("files", {})
    allowed = {"config.json", "dataset.json", "code.json", "environment.json", "targets.parquet", "folds.json"}
    allowed.update(f"fold-{i:02d}.joblib" for i in range(folds))
    if not isinstance(files, dict) or set(files) != allowed:
        raise ValueError("Invalid OOF artifact whitelist")
    hashes = [*files.values(), manifest.get("train_identity_sha256"), manifest.get("train_partition_sha256"),
              manifest.get("config_sha256"), value.get("targets_sha256"), value.get("teacher_manifest_sha256"),
              value.get("teacher_completion_sha256")]
    if any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{64}", x) for x in hashes):
        raise ValueError("Invalid OOF provenance hash")
    if value["targets_sha256"] != files["targets.parquet"]:
        raise ValueError("OOF target hash differs from embedded manifest")
    effective = {**teacher, "features": teacher.get("features", {"method": "all"}),
                 "hyperparameters": {**teacher.get("hyperparameters", {}),
                                      "early_stopping": False, "num_threads": config["threads"]}}
    expected_config = {"teacher_config": teacher, "effective_teacher_config": effective,
                       "seed": config["seed"], "folds": folds, "fold_seed": manifest["fold_seed"],
                       "threads": config["threads"]}
    if manifest["config_sha256"] != canonical_hash(expected_config):
        raise ValueError("OOF teacher configuration hash mismatch")
    if not manifest.get("train_identity_order") or not manifest.get("score_semantics"):
        raise ValueError("OOF sample identity or score semantics are missing")
    distill_stats = (statistics or {}).get("distillation", {})
    if (not statistics or statistics.get("training_objective") != "weighted_hard_bce_plus_unweighted_soft_bce"
            or distill_stats.get("enabled") is not True or distill_stats.get("alpha") != alpha
            or distill_stats.get("temperature") != temperature or distill_stats.get("teacher_model") != "hgb"):
        raise ValueError("Distillation objective metadata is inconsistent")
    return {**value, "study_verification_scope": "embedded provenance and training bindings; external OOF rows and fold models are not reopened by study"}


def _validate_predictions(path, meta, dataset, operating):
    frame = pd.read_parquet(path)
    required = {"sample_id", "score", "threshold", "predicted_label", "true_label",
                "attack_category", "split", "run_id", "data_kind"}
    if not required.issubset(frame.columns):
        raise ValueError("Validation predictions lack required identity columns")
    if frame.empty or frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("Validation sample IDs must be nonempty and unique")
    if not frame["sample_id"].map(lambda x: isinstance(x, str) and bool(x)).all():
        raise ValueError("Validation sample IDs must be nonempty strings")
    for column, expected in (("split", "validation"), ("run_id", meta["run_id"]),
                             ("data_kind", meta["data_kind"])):
        if not frame[column].eq(expected).all():
            raise ValueError(f"Validation identity mismatch: {column}")
    if len(frame) != dataset["partitions"]["validation"]["n_rows"]:
        raise ValueError("Validation row count differs from dataset manifest")
    if not frame["threshold"].eq(operating["threshold"]).all():
        raise ValueError("Prediction thresholds differ from saved validation threshold")
    # Sorting once lets all later comparisons use positionally aligned NumPy masks.
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    metrics = evaluate_binary(frame["true_label"], frame["score"], operating["threshold"],
                              frame["attack_category"])
    predicted = frame["score"].to_numpy(dtype=np.float64) >= operating["threshold"]
    if not frame["predicted_label"].isin([0, 1]).all() or not np.array_equal(
            frame["predicted_label"].to_numpy(), predicted.astype(int)):
        raise ValueError("Saved decisions differ from float64 score >= threshold")
    checked = calibrate_threshold(frame["true_label"], frame["score"], operating["max_fpr"])
    for key in ("threshold", "n_normal", "false_positives", "allowed_false_positives"):
        if operating.get(key) != checked[key]:
            raise ValueError(f"Saved validation operating point is inconsistent: {key}")
    if not _same_number(operating.get("achieved_fpr"), checked["achieved_fpr"]):
        raise ValueError("Saved validation operating FPR is inconsistent")
    # Missing family values are allowed, but kept distinct from the literal 'None'.
    categories = frame["attack_category"].map(lambda x: None if pd.isna(x) else str(x))
    identity = pd.DataFrame({"sample_id": frame["sample_id"],
                             "true_label": frame["true_label"].astype(int),
                             "attack_category": categories})
    return identity, predicted, metrics


def _read_run(root, run_path):
    preliminary = read_json(run_path / "run.json")
    artifacts = set(preliminary.get("files", {}))
    # verify_run hashes all manifest entries. Reject unexpected names *before*
    # that call so an altered manifest cannot make this analysis read test files.
    if not _REQUIRED.issubset(artifacts) or not artifacts.issubset(_ALLOWED):
        raise ValueError("Run must contain only recognized development artifacts")
    meta = verify_run(run_path)
    if meta.get("phase") != "development" or meta["run_id"] != run_path.name:
        raise ValueError("Study requires development runs with consistent run IDs")
    config = read_json(run_path / "config.json")
    dataset = read_json(run_path / "dataset.json")
    code = read_json(run_path / "code.json")
    saved = read_json(run_path / "validation_metrics.json")
    operating = read_json(run_path / "validation_threshold.json")
    selected = read_json(run_path / "selected_features.json")
    environment = _environment(read_json(run_path / "environment.json"))
    if canonical_hash(config) != meta["config_sha256"]:
        raise ValueError("Run configuration identity is inconsistent")
    if sha256_file(run_path / "dataset.json") != meta["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest identity is inconsistent")
    if canonical_hash(code["files"]) != code["source_sha256"]:
        raise ValueError("Training source fingerprint is inconsistent")
    if dataset["data_kind"] != meta["data_kind"]:
        raise ValueError("Dataset kind identity is inconsistent")
    if operating.get("source_split") != "validation" or (
            saved.get("split") != "validation" or saved.get("phase") != "development"):
        raise ValueError("Study uses validation-only development operating points")
    if operating["max_fpr"] != config["max_fpr"] or not 0 <= config["max_fpr"] < 1:
        raise ValueError("Validation FPR budget differs from configuration")
    if type(config["seed"]) is not int:
        raise ValueError("Run seed must be an integer")
    model_config = config["model_config"]
    if meta["model_name"] != model_config["model"]:
        raise ValueError("Run model identity is inconsistent")
    if (not isinstance(selected, list) or not selected
            or any(not isinstance(x, str) for x in selected)
            or len(selected) != len(set(selected))
            or not set(selected).issubset(dataset["features"])):
        raise ValueError("Selected raw fields must be distinct dataset features")
    feature_config = model_config.get("features", {"method": "all"})
    method = feature_config.get("method", "all")
    if method == "all":
        if set(selected) != set(dataset["features"]):
            raise ValueError("All-field run did not select every raw feature")
    elif feature_config.get("k") != len(selected):
        raise ValueError("Selected raw-field count differs from configured budget")
    identity, predicted, metrics = _validate_predictions(
        run_path / "validation_predictions.parquet", meta, dataset, operating)
    for key in ("n_samples", "threshold", "confusion"):
        if saved.get(key) != metrics[key]:
            raise ValueError(f"Saved validation metrics disagree with predictions: {key}")
    for key in _METRICS[:7]:
        if not _same_number(saved.get(key), metrics[key]):
            raise ValueError(f"Saved validation metrics disagree with predictions: {key}")
    expected_categories = [{k: row[k] for k in ("category", "n_attacks", "detected")}
                           for row in metrics["attack_categories"]]
    saved_categories = [{k: row[k] for k in ("category", "n_attacks", "detected")}
                        for row in saved.get("attack_categories", [])]
    if expected_categories != saved_categories:
        raise ValueError("Saved attack-category counts disagree with predictions")
    if saved.get("n_raw_features") != len(selected):
        raise ValueError("Saved raw-field count differs from selected features")
    encoded = saved["encoded_dim"]
    training_seconds = saved["training_seconds"]
    if (type(encoded) is not int or encoded < len(selected)
            or not math.isfinite(training_seconds) or training_seconds < 0):
        raise ValueError("Invalid encoded dimension or training duration")
    metrics.update(training_seconds=training_seconds, encoded_dim=encoded,
                   model_file_bytes=(run_path / "model.joblib").stat().st_size)
    statistics = _validate_training_statistics(run_path, meta, config, code, encoded)
    distillation = _validate_distillation(run_path, meta, config, code, dataset, statistics)
    metrics.update({key: statistics.get(key) if statistics else None for key in
                    ("parameter_count", "trainable_parameter_count", "actual_epochs", "best_epoch")})
    oof = distillation["oof_manifest"] if distillation else {}
    for metric, source_key in (("teacher_generation_seconds", "generation_seconds"),
                               ("teacher_fold_fit_seconds_total", "fold_fit_seconds_total")):
        value = oof.get(source_key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid teacher generation timing")
        metrics[metric] = value
    spec = {"model": model_config["model"], "feature_method": method,
            "raw_feature_budget": len(selected), "feature_config": feature_config,
            "hyperparameters": model_config.get("hyperparameters", {}),
            "model_config": model_config, "max_fpr": config["max_fpr"],
            "requested_device": config["requested_device"], "threads": config["threads"],
            "source_sha256": code["source_sha256"],
            "training_environment_sha256": canonical_hash(environment)}
    if "variant" in model_config:
        spec["variant"] = model_config["variant"]
    run = {"run_key": run_path.relative_to(root).as_posix() or ".", "run_id": meta["run_id"],
           "seed": config["seed"], "run_manifest_sha256": sha256_file(run_path / "run.json"),
           "model_sha256": meta["files"]["model.joblib"],
           "source_sha256": code["source_sha256"], "selected_features": selected,
           "metrics": metrics, "environment": environment,
           "threshold": operating["threshold"], "max_fpr": config["max_fpr"]}
    if statistics is not None:
        run["training_statistics"] = statistics
    if distillation is not None:
        run["distillation"] = distillation
    return spec, run, dataset, meta["dataset_manifest_sha256"], identity, predicted


def _feature_stability(runs):
    selections = [{"seed": r["seed"], "features": r["selected_features"]} for r in runs]
    pairs = []
    for left, right in itertools.combinations(runs, 2):
        a, b = set(left["selected_features"]), set(right["selected_features"])
        pairs.append({"left_seed": left["seed"], "right_seed": right["seed"],
                      "jaccard": len(a & b) / len(a | b)})
    frequencies = {name: sum(name in r["selected_features"] for r in runs)
                   for name in sorted(set().union(*(set(r["selected_features"]) for r in runs)))}
    return {"selections": selections, "pairwise_jaccard": pairs,
            "mean_pairwise_jaccard": float(np.mean([p["jaccard"] for p in pairs])) if pairs else None,
            "selection_counts": frequencies,
            "scope": "raw field selection stability across training seeds, not encoded columns"}


def _comparisons(groups, decisions, identity):
    common_seeds = sorted(set.intersection(*(set(g["seeds"]) for g in groups)))
    group_ids = [g["id"] for g in groups]
    labels = identity["true_label"].to_numpy() == 1
    n_attacks, n_normal = int(labels.sum()), int((~labels).sum())
    categories = identity["attack_category"].to_numpy()
    families = sorted(set(categories[labels]), key=lambda x: (x is None, x or ""))
    per_seed = []
    for seed in common_seeds:
        matrix = np.stack([decisions[(group_id, seed)] for group_id in group_ids])
        counts = matrix.sum(axis=0)
        union = counts > 0
        unique = [{"group_id": gid, "count": int(np.count_nonzero(labels & row & (counts == 1)))}
                  for gid, row in zip(group_ids, matrix)]
        pairwise = []
        for i, j in itertools.combinations(range(len(groups)), 2):
            left, right = matrix[i], matrix[j]
            either = left | right
            pairwise.append({
                "left": group_ids[i], "right": group_ids[j],
                "left_only_detected": int(np.count_nonzero(labels & left & ~right)),
                "right_only_detected": int(np.count_nonzero(labels & right & ~left)),
                "both_detected": int(np.count_nonzero(labels & left & right)),
                "both_missed": int(np.count_nonzero(labels & ~either)),
                "or_attack_recall": _ratio(int(np.count_nonzero(labels & either)), n_attacks),
                "or_normal_fpr": _ratio(int(np.count_nonzero(~labels & either)), n_normal),
                "or_false_positives": int(np.count_nonzero(~labels & either)),
                "decision_agreement": float(np.mean(left == right)),
            })
        missed = int(np.count_nonzero(labels & ~union))
        family_missed = []
        for family in families:
            mask = labels & np.array([x == family for x in categories])
            count = int(np.count_nonzero(mask & ~union))
            support = int(mask.sum())
            family_missed.append({"category": family, "support": support,
                                  "count": count, "rate": _ratio(count, support)})
        per_seed.append({
            "seed": seed, "n_attacks": n_attacks, "n_normal": n_normal,
            "all_missed": {"count": missed, "rate": _ratio(missed, n_attacks)},
            "or_alerts": {"attack_recall": _ratio(int(np.count_nonzero(labels & union)), n_attacks),
                          "normal_fpr": _ratio(int(np.count_nonzero(~labels & union)), n_normal),
                          "false_positives": int(np.count_nonzero(~labels & union))},
            "unique_detections": unique, "pairwise": pairwise,
            "per_attack_common_missed": family_missed,
        })
    return {"common_seeds": common_seeds, "candidate_group_ids": group_ids, "per_seed": per_seed,
            "scope": "same validation sample IDs and labels; intersection of seeds across all groups",
            "unique_detection_definition": "true attacks detected by this candidate and by no other candidate",
            "or_alerts_notice": "Exploratory OR alerts reuse individual validation thresholds; no FPR-budget guarantee and no final-test claim"}


def _attach_benchmarks(groups, benchmark_dir, dataset_hash):
    if benchmark_dir is None:
        return
    root = Path(benchmark_dir)
    if not root.is_dir():
        raise ValueError("benchmark_dir must be a directory")
    run_index = {r["run_manifest_sha256"]: (g, r) for g in groups for r in g["runs"]}
    collected = {}
    for path in sorted(root.rglob("*.json")):
        value = read_json(path)
        if not isinstance(value, dict) or not {"single", "batch"}.intersection(value):
            continue
        binding = ("run_manifest_sha256", "dataset_manifest_sha256", "model_sha256", "source_sha256")
        if any(key not in value for key in binding) or value.get("split") != "validation":
            raise ValueError("CPU benchmark lacks validation/run/dataset/model/source binding")
        if value["run_manifest_sha256"] not in run_index:
            raise ValueError("CPU benchmark is bound to an unknown run")
        group, run = run_index[value["run_manifest_sha256"]]
        expected = {"run_id": run["run_id"], "dataset_manifest_sha256": dataset_hash,
                    "model_sha256": run["model_sha256"], "source_sha256": run["source_sha256"]}
        if any(value.get(key) != item for key, item in expected.items()):
            raise ValueError("CPU benchmark identity mismatch")
        if value.get("device") != "cpu" or value.get("scope") != "preprocessing_plus_predict_fn":
            raise ValueError("Study supports CPU preprocessing-plus-prediction benchmarks only")
        if value.get("model_file_bytes") != run["metrics"]["model_file_bytes"]:
            raise ValueError("CPU benchmark model size differs from verified model")
        single, batch = value["single"], value["batch"]
        environment = _environment(value.get("environment", {}))
        if not environment or not value.get("benchmark_code_sha256"):
            raise ValueError("CPU benchmark lacks environment or measurement source identity")
        if type(value["threads"]) is not int or value["threads"] < 1 or single["batch_size"] != 1:
            raise ValueError("Invalid CPU benchmark thread count or single-row batch size")
        measurement_round = value.get("measurement_round", 1)
        planned_rounds = value.get("planned_rounds", 1)
        if (type(measurement_round) is not int or type(planned_rounds) is not int
                or not 1 <= measurement_round <= planned_rounds):
            raise ValueError("Invalid CPU measurement round or planned rounds")
        spec = {"scope": value["scope"], "threads": value["threads"],
                "batch_size": batch["batch_size"], "environment_sha256": canonical_hash(environment),
                "benchmark_code_sha256": value["benchmark_code_sha256"],
                "warmup": value.get("warmup"), "repeats": value.get("repeats"),
                "single_rows": single["n_unique_rows"], "single_observations": single["n_observations"],
                "single_row_indices": single["row_indices"],
                "requested_batch_size": batch["requested_batch_size"],
                "batch_rows": batch["n_rows"], "batch_observations": batch["n_observations"],
                "measurement_round": measurement_round, "planned_rounds": planned_rounds}
        metrics = {"p50_ms": single["p50_ms"], "p95_ms": single["p95_ms"],
                   "throughput_rows_per_second": batch["throughput_rows_per_second"]}
        if any(not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in metrics.values()):
            raise ValueError("CPU benchmark metrics must be positive and finite")
        single_times = np.asarray(single["latencies_ms"], dtype=float)
        batch_times = np.asarray(batch["latencies_ms"], dtype=float)
        for timings, record in ((single_times, single), (batch_times, batch)):
            if (timings.ndim != 1 or len(timings) != record["n_observations"] or not len(timings)
                    or not np.all(np.isfinite(timings)) or np.any(timings <= 0)):
                raise ValueError("CPU benchmark raw timing observations are inconsistent")
        if (not _same_number(metrics["p50_ms"], np.percentile(single_times, 50))
                or not _same_number(metrics["p95_ms"], np.percentile(single_times, 95))
                or not _same_number(metrics["throughput_rows_per_second"],
                                    batch["n_rows"] * len(batch_times) / (batch_times.sum() / 1000))):
            raise ValueError("CPU benchmark summaries disagree with raw timing observations")
        if (len(single_times) != single["n_unique_rows"] * value["repeats"]
                or len(single["row_indices"]) != single["n_unique_rows"]
                or len(set(single["row_indices"])) != single["n_unique_rows"]
                or batch["n_rows"] != batch["batch_size"]
                or len(batch_times) != value["repeats"]
                or batch["total_rows_processed"] != batch["n_rows"] * value["repeats"]):
            raise ValueError("CPU benchmark measurement counts are inconsistent")
        bucket = collected.setdefault((group["id"], canonical_hash(spec)),
                                      {"spec": spec, "environment": environment, "runs": []})
        if run["seed"] in [x["seed"] for x in bucket["runs"]]:
            raise ValueError("Duplicate CPU benchmark for the same run, round and measurement setting")
        bucket["runs"].append({"seed": run["seed"], "run_key": run["run_key"],
                               "artifact": path.relative_to(root).as_posix(),
                               "artifact_sha256": sha256_file(path), "metrics": metrics})
    for (group_id, _), bucket in sorted(collected.items()):
        bucket["runs"].sort(key=lambda r: r["seed"])
        bucket["seeds"] = [r["seed"] for r in bucket["runs"]]
        bucket["metrics"] = {key: _aggregate([r["metrics"][key] for r in bucket["runs"]])
                             for key in ("p50_ms", "p95_ms", "throughput_rows_per_second")}
        bucket["statistical_scope"] = "summary of per-run measured quantiles/throughputs; not a pooled latency quantile"
        next(g for g in groups if g["id"] == group_id)["cpu_benchmarks"].append(bucket)
    for group in groups:
        group["repeated_benchmarks"] = _repeat_benchmarks(group)


def _repeat_benchmarks(group):
    """Summarize rounds within each seed before aggregating across seeds."""
    buckets = {}
    for bucket in group["cpu_benchmarks"]:
        if bucket["spec"]["planned_rounds"] == 1:
            continue
        spec = {k: v for k, v in bucket["spec"].items() if k != "measurement_round"}
        item = buckets.setdefault(canonical_hash(spec), {"spec": spec, "environment": bucket["environment"],
                                                        "records": {}})
        for run in bucket["runs"]:
            item["records"].setdefault(run["seed"], []).append({"round": bucket["spec"]["measurement_round"],
                                                              "artifact": run["artifact"],
                                                              "metrics": run["metrics"]})
    result = []
    keys = ("p50_ms", "p95_ms", "throughput_rows_per_second")
    for item in buckets.values():
        planned = item["spec"]["planned_rounds"]
        per_seed = []
        for seed, rows in sorted(item.pop("records").items()):
            rows.sort(key=lambda r: r["round"])
            rounds = [r["round"] for r in rows]
            missing = sorted(set(range(1, planned + 1)) - set(rounds))
            per_seed.append({"seed": seed, "rounds": rounds, "missing_rounds": missing,
                             "status": "complete" if not missing else "incomplete_rounds",
                             "metrics": {key: {"median": float(np.median([r["metrics"][key] for r in rows])),
                                               **_aggregate([r["metrics"][key] for r in rows])} for key in keys},
                             "measurements": rows})
        complete = [r for r in per_seed if r["status"] == "complete"]
        complete_seeds = [r["seed"] for r in complete]
        item.update(per_seed=per_seed, seeds=complete_seeds, observed_seeds=[r["seed"] for r in per_seed],
                    missing_seeds=sorted(set(group["seeds"]) - set(complete_seeds)),
                    status="complete" if set(complete_seeds) == set(group["seeds"]) else "incomplete_rounds",
                    metrics={key: _aggregate([r["metrics"][key]["median"] for r in complete]) for key in keys},
                    statistical_scope="per-seed median across complete planned rounds, then mean/range across training seeds; not a pooled latency quantile or confidence interval")
        result.append(item)
    return result


def _feature_budget_comparisons(groups):
    """Pair reduced-field runs with an all-field run under identical fit settings."""
    def fit_key(group):
        spec = {k: v for k, v in group["spec"].items()
                if k not in {"feature_method", "raw_feature_budget", "feature_config", "model_config"}}
        spec["model_config"] = {k: v for k, v in group["spec"]["model_config"].items() if k != "features"}
        return canonical_hash(spec)

    def fuzzers(run):
        return next((c["recall"] for c in run["metrics"]["attack_categories"]
                     if c["category"] == "Fuzzers"), None)

    def delta(left, right):
        return (left - right) * 100 if left is not None and right is not None else None

    baselines = {}
    for group in groups:
        if group["spec"]["feature_method"] == "all":
            key = fit_key(group)
            if key in baselines:
                raise ValueError("Ambiguous all-field baseline under identical training settings")
            baselines[key] = group
    output = []
    for candidate in groups:
        if candidate["spec"]["feature_method"] == "all":
            continue
        baseline = baselines.get(fit_key(candidate))
        common = sorted(set(candidate["seeds"]) & set(baseline["seeds"])) if baseline else []
        per_seed = []
        if baseline:
            left = {r["seed"]: r for r in candidate["runs"]}
            right = {r["seed"]: r for r in baseline["runs"]}
            for seed in common:
                a, b = left[seed], right[seed]
                per_seed.append({"seed": seed,
                                 "attack_recall_delta_pp": delta(a["metrics"]["attack_recall"], b["metrics"]["attack_recall"]),
                                 "fuzzers_recall_delta_pp": delta(fuzzers(a), fuzzers(b)),
                                 "model_size_ratio": _ratio(a["metrics"]["model_file_bytes"], b["metrics"]["model_file_bytes"]),
                                 "raw_feature_ratio": len(a["selected_features"]) / len(b["selected_features"]),
                                 "encoded_dim_ratio": a["metrics"]["encoded_dim"] / b["metrics"]["encoded_dim"]})
        output.append({"candidate_group_id": candidate["id"],
                       "baseline_group_id": baseline["id"] if baseline else None,
                       "model": candidate["spec"]["model"],
                       "raw_feature_budget": candidate["spec"]["raw_feature_budget"],
                       "status": "paired" if common else "no_common_seed" if baseline else "no_matching_all_field_baseline",
                       "common_seeds": common, "n_pairs": len(common), "per_seed": per_seed,
                       "metrics": {key: _aggregate([row[key] for row in per_seed]) for key in (
                           "attack_recall_delta_pp", "fuzzers_recall_delta_pp", "model_size_ratio",
                           "raw_feature_ratio", "encoded_dim_ratio")},
                       "scope": "candidate minus all-field baseline in percentage points; ratios are candidate / baseline; identical fit settings, paired training seeds, fixed validation partition"})
    return output


def _distillation_match_key(group, kind):
    spec = {k: v for k, v in group["spec"].items()
            if k not in {"variant", "model_config", "hyperparameters"}}
    model = {k: v for k, v in group["spec"]["model_config"].items() if k not in {"variant", "distillation"}}
    hyper = dict(model.get("hyperparameters", {}))
    if kind == "capacity_effect":
        hyper.pop("hidden_sizes", None)
        hyper.pop("hidden_dims", None)
    model["hyperparameters"] = hyper
    spec["model_config"] = model
    return canonical_hash(spec)


def _paired_cpu(candidate, control, common):
    """Keep every measurement setting/round, then pair complete round medians."""
    output = []
    for mode, key in (("single_round", "cpu_benchmarks"), ("within_seed_round_median", "repeated_benchmarks")):
        controls = {canonical_hash(b["spec"]): b for b in control.get(key, [])}
        for bucket in candidate.get(key, []):
            other = controls.get(canonical_hash(bucket["spec"]))
            if not other:
                continue
            if mode == "single_round":
                left = {r["seed"]: r["metrics"] for r in bucket["runs"]}
                right = {r["seed"]: r["metrics"] for r in other["runs"]}
            else:
                left = {r["seed"]: {k: v["median"] for k, v in r["metrics"].items()}
                        for r in bucket["per_seed"] if r["status"] == "complete"}
                right = {r["seed"]: {k: v["median"] for k, v in r["metrics"].items()}
                         for r in other["per_seed"] if r["status"] == "complete"}
            seeds = sorted(set(left) & set(right) & set(common))
            rows = [{"seed": seed, "p50_ratio": left[seed]["p50_ms"] / right[seed]["p50_ms"],
                     "p95_ratio": left[seed]["p95_ms"] / right[seed]["p95_ms"],
                     "throughput_ratio": left[seed]["throughput_rows_per_second"] / right[seed]["throughput_rows_per_second"]}
                    for seed in seeds]
            output.append({"mode": mode, "spec": bucket["spec"], "common_seeds": seeds,
                           "missing_seeds": sorted(set(common) - set(seeds)), "n_pairs": len(seeds),
                           "status": "paired" if seeds else "no_common_measured_seed", "per_seed": rows,
                           "metrics": {k: _aggregate([r[k] for r in rows])
                                       for k in ("p50_ratio", "p95_ratio", "throughput_ratio")},
                           "scope": "candidate / control; latency below 1 and throughput above 1 are faster; identical measurement settings"})
    return output


def _distillation_comparisons(groups, decisions, identity):
    """Separate KD-vs-hard treatment from hard-small-vs-large capacity changes."""
    output = []
    labels = identity["true_label"].to_numpy() == 1
    categories = identity["attack_category"].to_numpy()
    families = sorted(set(categories[labels]), key=lambda x: (x is None, x or ""))
    definitions = (("distillation_effect", "student_kd", "student_hard"),
                   ("capacity_effect", "student_hard", "mlp_reference"))
    for kind, candidate_variant, control_variant in definitions:
        candidates = [g for g in groups if g["spec"].get("variant") == candidate_variant]
        controls = [g for g in groups if g["spec"].get("variant") == control_variant]
        for candidate in candidates:
            matched = [g for g in controls if _distillation_match_key(g, kind) ==
                       _distillation_match_key(candidate, kind)]
            if len(matched) > 1:
                raise ValueError("Ambiguous matched distillation control")
            control = matched[0] if matched else None
            common = sorted(set(candidate["seeds"]) & set(control["seeds"])) if control else []
            rows = []
            for seed in common:
                a = next(r for r in candidate["runs"] if r["seed"] == seed)
                b = next(r for r in control["runs"] if r["seed"] == seed)
                left, right = decisions[(candidate["id"], seed)], decisions[(control["id"], seed)]
                recovered, lost = labels & left & ~right, labels & ~left & right
                per_attack = []
                for family in families:
                    mask = labels & np.array([x == family for x in categories])
                    gain, loss = int(np.count_nonzero(mask & recovered)), int(np.count_nonzero(mask & lost))
                    support = int(mask.sum())
                    per_attack.append({"category": family, "support": support,
                                       "recovered": gain, "newly_missed": loss, "net_detected_gain": gain - loss,
                                       "recall_delta_pp": 100 * (gain - loss) / support,
                                       "recovered_sample_ids": identity.loc[mask & recovered, "sample_id"].head(10).tolist(),
                                       "newly_missed_sample_ids": identity.loc[mask & lost, "sample_id"].head(10).tolist()})
                def delta(key, scale=1):
                    x, y = a["metrics"].get(key), b["metrics"].get(key)
                    return (x - y) * scale if x is not None and y is not None else None

                parameters = a["metrics"].get("parameter_count"), b["metrics"].get("parameter_count")
                rows.append({"seed": seed,
                             "attack_recall_delta_pp": delta("attack_recall", 100),
                             "average_precision_delta": delta("average_precision"),
                             "normal_fpr_delta_pp": delta("normal_fpr", 100),
                             "recovered_attacks": int(recovered.sum()), "newly_missed_attacks": int(lost.sum()),
                             "net_detected_gain": int(recovered.sum() - lost.sum()),
                             "new_false_positives": int(np.count_nonzero(~labels & left & ~right)),
                             "resolved_false_positives": int(np.count_nonzero(~labels & ~left & right)),
                             "model_size_ratio": a["metrics"]["model_file_bytes"] / b["metrics"]["model_file_bytes"],
                             "parameter_ratio": _ratio(*parameters) if None not in parameters else None,
                             "per_attack": per_attack})
            keys = ("attack_recall_delta_pp", "average_precision_delta", "normal_fpr_delta_pp",
                    "recovered_attacks", "newly_missed_attacks", "net_detected_gain", "new_false_positives",
                    "resolved_false_positives", "model_size_ratio", "parameter_ratio")
            family_summary = []
            for index, family in enumerate(families):
                family_summary.append({"category": family, "support": int(np.count_nonzero(
                    labels & np.array([x == family for x in categories]))),
                    **{key: _aggregate([row["per_attack"][index][key] for row in rows]) for key in
                       ("recovered", "newly_missed", "net_detected_gain", "recall_delta_pp")}})
            output.append({"kind": kind, "candidate_variant": candidate_variant, "control_variant": control_variant,
                           "candidate_group_id": candidate["id"], "control_group_id": control["id"] if control else None,
                           "status": "paired" if common else "no_common_seed" if control else "no_matching_control",
                           "common_seeds": common, "n_pairs": len(common), "expected_seeds": [17, 29, 43],
                           "missing_expected_seeds": sorted({17, 29, 43} - set(common)),
                           "complete_expected_seeds": set(common) == {17, 29, 43},
                           "per_seed": rows, "metrics": {key: _aggregate([r[key] for r in rows]) for key in keys},
                           "per_attack": family_summary,
                           "cpu_comparisons": _paired_cpu(candidate, control, common) if control else [],
                           "scope": "candidate minus matched control on the same validation identities and seed; KD-vs-hard isolates training objective, capacity comparison changes hidden widths; HGB is reference only",
                           "example_scope": "first up to 10 stable sample IDs per family and outcome per seed; counts use every validation record"})
    return output


def build_study(run_root, benchmark_dir=None) -> dict:
    """Build a JSON-serializable study from verified development artifacts only.

    All runs must refer to exactly the same dataset manifest and validation
    identities. Different hyperparameters, source versions, FPR budgets or
    training environments form separate groups. A repeated seed in a group is
    rejected rather than counted as an independent replicate. Failed-attempt
    archives are excluded explicitly; model bytes are hashed, never loaded.
    """
    root = Path(run_root)
    if not root.is_dir():
        raise ValueError("run_root must be a directory")
    paths = [p.parent for p in sorted(root.rglob("run.json"))
             if "failed_attempts" not in p.relative_to(root).parts]
    if not paths:
        raise ValueError("No completed development run manifests found")
    grouped, decisions = {}, {}
    dataset_hash = dataset = identity = None
    warnings = [
        "Development validation only: thresholds and model choices use this partition; results are not final-test evidence.",
        "Seed standard deviations quantify training variation on a fixed partition, not independent datasets or confidence intervals.",
        "Exploratory OR alerts are not calibrated to the individual FPR budget.",
        "Raw-field count, encoded dimension and serialized model bytes do not measure feature-extraction cost.",
    ]
    for path in paths:
        spec, run, run_dataset, manifest_hash, run_identity, predicted = _read_run(root, path)
        if dataset_hash is None:
            dataset_hash, dataset, identity = manifest_hash, run_dataset, run_identity
        elif dataset_hash != manifest_hash:
            raise ValueError("Study cannot mix different dataset manifests")
        elif not identity.equals(run_identity):
            raise ValueError("Validation sample IDs, labels or attack categories differ across runs")
        group_id = "group-" + canonical_hash(spec)[:16]
        if group_id in grouped and grouped[group_id]["spec"] != spec:
            raise ValueError("Group identity hash collision")
        group = grouped.setdefault(group_id, {"id": group_id, "spec": spec, "runs": []})
        if run["seed"] in [r["seed"] for r in group["runs"]]:
            raise ValueError("Duplicate seed within the same study group")
        group["runs"].append(run)
        decisions[(group_id, run["seed"])] = predicted
    groups = sorted(grouped.values(), key=lambda g: (g["spec"]["model"],
                    g["spec"]["raw_feature_budget"], g["id"]))
    for group in groups:
        runs = sorted(group["runs"], key=lambda r: r["seed"])
        group.update(runs=runs, seeds=[r["seed"] for r in runs], n_runs=len(runs), cpu_benchmarks=[], repeated_benchmarks=[])
        spec = group["spec"]
        group["label"] = (_VARIANT_LABELS.get(spec.get("variant")) or
                          f"{spec['model'].upper()} / {spec['feature_method']} {spec['raw_feature_budget']}") + f" / {group['id'][-6:]}"
        group["metrics"] = {key: _aggregate([r["metrics"][key] for r in runs]) for key in _METRICS}
        group["features"] = _feature_stability(runs)
        group["per_attack"] = []
        for index, category in enumerate(runs[0]["metrics"]["attack_categories"]):
            rows = [r["metrics"]["attack_categories"][index] for r in runs]
            group["per_attack"].append({"category": category["category"], "support": category["n_attacks"],
                                        "recall": _aggregate([x["recall"] for x in rows]),
                                        "detected": _aggregate([x["detected"] for x in rows])})
        group["attack_category_semantics"] = "binary detection within each true attack family, not family classification"
    comparison = _comparisons(groups, decisions, identity)
    if len(groups) < 2:
        warnings.append("Only one candidate group: complementarity comparisons require additional candidates.")
    if not comparison["common_seeds"]:
        warnings.append("No seed is shared across all candidate groups; no complementarity statistics were computed.")
    if len({tuple(g["seeds"]) for g in groups}) > 1:
        warnings.append("Groups have different seed sets: group summaries use their own seeds; complementarity uses only the common intersection.")
    if len({g["spec"]["max_fpr"] for g in groups}) > 1:
        warnings.append("Candidates have different individual FPR budgets; comparisons are not a controlled fixed-budget experiment.")
    if dataset["data_kind"] not in {"official", "official_repaired"}:
        warnings.append("Dataset is not declared official; retain its provenance limits in every research claim.")
    if dataset["data_kind"] == "official_repaired":
        warnings.append("Author-repaired dataset with a pinned observed archive hash; group-safe same-capture evaluation, not time/host transfer.")
    _attach_benchmarks(groups, benchmark_dir, dataset_hash)
    distillation_comparisons = _distillation_comparisons(groups, decisions, identity)
    for paired_comparison in distillation_comparisons:
        if not paired_comparison["complete_expected_seeds"]:
            warnings.append(f"Incomplete matched three-seed comparison: {paired_comparison['kind']} / {paired_comparison['candidate_group_id']}")
    if not any(g["cpu_benchmarks"] for g in groups):
        warnings.append("No identity-bound CPU benchmarks are included; no latency or throughput conclusion is available.")
    return {"schema_version": 1, "phase": "validation_development",
            "dataset_manifest_sha256": dataset_hash, "data_kind": dataset["data_kind"],
            "dataset": {key: dataset[key] for key in ("dataset_id", "split_method", "split_seed",
                                                       "provenance_status") if key in dataset},
            "n_validation_samples": len(identity), "n_runs": sum(g["n_runs"] for g in groups),
            "validation_identity_sha256": canonical_hash(identity.to_dict(orient="records")),
            "identity_scope": "cross-run validation predictions and declared manifest row count; original dataset files are not opened",
            "groups": groups, "comparisons": comparison,
            "feature_budget_comparisons": _feature_budget_comparisons(groups),
            "distillation_comparisons": distillation_comparisons, "warnings": warnings}
