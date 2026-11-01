"""Verified, validation-only study summaries; no model deserialization.

Seed standard deviations describe training variability on one fixed partition,
not independent test samples or confidence intervals. OR alerts below are an
exploratory complementarity diagnostic, not a calibrated ensemble.
"""
from __future__ import annotations

import itertools
import math
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
_ALLOWED = _REQUIRED | {"training_history.json", "examples.json"}
_METRICS = (
    "attack_recall", "normal_fpr", "precision", "accuracy", "macro_f1",
    "average_precision", "roc_auc", "training_seconds", "model_file_bytes", "encoded_dim",
)
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
    spec = {"model": model_config["model"], "feature_method": method,
            "raw_feature_budget": len(selected), "feature_config": feature_config,
            "hyperparameters": model_config.get("hyperparameters", {}),
            "model_config": model_config, "max_fpr": config["max_fpr"],
            "requested_device": config["requested_device"], "threads": config["threads"],
            "source_sha256": code["source_sha256"],
            "training_environment_sha256": canonical_hash(environment)}
    run = {"run_key": run_path.relative_to(root).as_posix() or ".", "run_id": meta["run_id"],
           "seed": config["seed"], "run_manifest_sha256": sha256_file(run_path / "run.json"),
           "model_sha256": meta["files"]["model.joblib"],
           "source_sha256": code["source_sha256"], "selected_features": selected,
           "metrics": metrics, "environment": environment,
           "threshold": operating["threshold"], "max_fpr": config["max_fpr"]}
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
        spec = {"scope": value["scope"], "threads": value["threads"],
                "batch_size": batch["batch_size"], "environment_sha256": canonical_hash(environment),
                "benchmark_code_sha256": value["benchmark_code_sha256"],
                "warmup": value.get("warmup"), "repeats": value.get("repeats"),
                "single_rows": single["n_unique_rows"], "single_observations": single["n_observations"],
                "single_row_indices": single["row_indices"],
                "requested_batch_size": batch["requested_batch_size"],
                "batch_rows": batch["n_rows"], "batch_observations": batch["n_observations"]}
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
            raise ValueError("Duplicate CPU benchmark for the same run and measurement setting")
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
        group.update(runs=runs, seeds=[r["seed"] for r in runs], n_runs=len(runs), cpu_benchmarks=[])
        spec = group["spec"]
        group["label"] = f"{spec['model'].upper()} / {spec['feature_method']} {spec['raw_feature_budget']} / {group['id'][-6:]}"
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
    if dataset["data_kind"] != "official":
        warnings.append("Dataset is not declared official; retain its provenance limits in every research claim.")
    _attach_benchmarks(groups, benchmark_dir, dataset_hash)
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
            "feature_budget_comparisons": _feature_budget_comparisons(groups), "warnings": warnings}
