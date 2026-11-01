"""Small hand-built, checksummed artifacts exercise study arithmetic and gates.

The fake model is deliberately not a pickle. No training, model loading, raw
dataset access or calibration/test access is needed for research aggregation.
"""
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids import study
from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.evaluation import calibrate_threshold, evaluate_binary


def _seal(run):
    config, dataset = read_json(run / "config.json"), read_json(run / "dataset.json")
    meta = {"schema_version": 1, "run_id": run.name, "status": "complete", "phase": "development",
            "config_sha256": canonical_hash(config), "dataset_manifest_sha256": sha256_file(run / "dataset.json"),
            "data_kind": dataset["data_kind"], "model_name": config["model_config"]["model"],
            "files": {p.name: sha256_file(p) for p in run.iterdir()
                      if p.is_file() and p.name != "run.json"}}
    write_json(run / "run.json", meta)


def _run(root, name="a-s17", *, seed=17, model="rf", attacks=(0, 1, 2), normal=(6,),
         selected=None, hyperparameters=None, reverse=False, labels=None, categories=None,
         sample_ids=None, float32=False, dataset_id="fixture", environment=None):
    path = root / name
    path.mkdir(parents=True)
    selected = list(selected) if selected is not None else ["f0", "f1", "f2", "f3"]
    config = {"model_config": {"model": model,
                              "features": {"method": "all"} if len(selected) == 4 else {
                                  "method": "mutual_info", "k": len(selected)},
                              "hyperparameters": hyperparameters or {"n_estimators": 100}},
              "seed": seed, "requested_device": "cpu", "max_fpr": .01, "threads": 1}
    dataset = {"dataset_id": dataset_id, "data_kind": "synthetic",
               "features": ["f0", "f1", "f2", "f3"],
               "partitions": {"validation": {"n_rows": 106, "file": "validation.parquet"}},
               "split_method": "fixture_only", "split_seed": 1}
    source_files = {"src/ids/example.py": "a" * 64}
    scores = np.full(106, .5 if float32 else .1, dtype=np.float32 if float32 else np.float64)
    scores[list(attacks) + list(normal)] = .9
    labels = np.array([1] * 6 + [0] * 100) if labels is None else np.array(labels)
    categories = (["Fuzzers"] * 3 + ["Exploits"] * 3 + ["Normal"] * 100
                  if categories is None else categories)
    operating = calibrate_threshold(labels, scores, .01)
    operating.update(source_split="validation", usage="development_only_not_final_threshold")
    metrics = evaluate_binary(labels, scores, operating["threshold"], categories)
    metrics.update(split="validation", phase="development", data_kind="synthetic", training_seconds=2.0,
                   n_raw_features=len(selected), encoded_dim=len(selected))
    frame = pd.DataFrame({"sample_id": sample_ids or [f"row-{i:03d}" for i in range(106)],
                          "score": scores, "true_label": labels, "attack_category": categories,
                          "threshold": operating["threshold"], "predicted_label": (
                              scores.astype(np.float64) >= operating["threshold"]).astype(int),
                          "split": "validation", "run_id": name, "data_kind": "synthetic"})
    if reverse:
        frame = frame.iloc[::-1]
    frame.to_parquet(path / "validation_predictions.parquet", index=False)
    (path / "model.joblib").write_bytes(b"deliberately-not-a-pickle")
    for filename, value in {"config.json": config, "dataset.json": dataset,
                            "code.json": {"files": source_files, "source_sha256": canonical_hash(source_files)},
                            "environment.json": environment or {"cpu_physical": 8, "recorded_at": name},
                            "selected_features.json": selected, "validation_threshold.json": operating,
                            "validation_metrics.json": metrics}.items():
        write_json(path / filename, value)
    _seal(path)
    return path


def _benchmark(root, run, *, filename="bench.json", threads=1, processor="fixture-cpu"):
    meta = read_json(run / "run.json")
    value = {"run_id": run.name, "run_manifest_sha256": sha256_file(run / "run.json"),
             "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
             "model_sha256": meta["files"]["model.joblib"],
             "source_sha256": read_json(run / "code.json")["source_sha256"],
             "benchmark_code_sha256": "b" * 64, "split": "validation", "device": "cpu",
             "scope": "preprocessing_plus_predict_fn", "threads": threads, "warmup": 5, "repeats": 2,
             "model_file_bytes": (run / "model.joblib").stat().st_size,
             "environment": {"processor": processor, "recorded_at": filename},
             "single": {"batch_size": 1, "n_unique_rows": 2, "n_observations": 4,
                        "row_indices": [0, 53], "p50_ms": 2.5, "p95_ms": 3.85,
                        "latencies_ms": [1., 2., 3., 4.]},
             "batch": {"batch_size": 2, "requested_batch_size": 2, "n_rows": 2,
                       "n_observations": 2, "total_rows_processed": 4,
                       "latencies_ms": [2., 2.], "throughput_rows_per_second": 1000.}}
    target = root / filename
    write_json(target, value)
    return target


def test_global_unique_detections_pairwise_union_and_blindspots(tmp_path):
    _run(tmp_path, "a", model="rf", attacks=(0, 1, 2), normal=(6,))
    _run(tmp_path, "b", model="hgb", attacks=(1, 2, 3), normal=(7,), reverse=True)
    _run(tmp_path, "c", model="mlp", attacks=(2, 4), normal=(6,))
    result = study.build_study(tmp_path)
    names = {g["spec"]["model"]: g["id"] for g in result["groups"]}
    row = result["comparisons"]["per_seed"][0]
    assert row["all_missed"] == {"count": 1, "rate": 1 / 6}
    assert row["or_alerts"] == {"attack_recall": 5 / 6, "normal_fpr": .02, "false_positives": 2}
    # Historical subtraction formula would report 2, 2, 1. The true global
    # unique counts are 1, 1, 1 because attack 1 is shared by only two models.
    assert {x["group_id"]: x["count"] for x in row["unique_detections"]} == dict.fromkeys(names.values(), 1)
    pair = next(x for x in row["pairwise"] if {x["left"], x["right"]} == {names["rf"], names["hgb"]})
    assert pair["left_only_detected"] == pair["right_only_detected"] == 1
    assert pair["both_detected"] == 2 and pair["both_missed"] == 2
    assert pair["or_normal_fpr"] == .02 and pair["or_attack_recall"] == 4 / 6
    assert row["per_attack_common_missed"] == [
        {"category": "Exploits", "support": 3, "count": 1, "rate": 1 / 3},
        {"category": "Fuzzers", "support": 3, "count": 0, "rate": 0.0}]
    assert "no FPR-budget guarantee" in result["comparisons"]["or_alerts_notice"]
    json.dumps(result, allow_nan=False)


def test_seed_statistics_feature_stability_and_null_single_seed_std(tmp_path):
    _run(tmp_path, "rf17", selected=["f0", "f1"], attacks=(0, 1), seed=17)
    _run(tmp_path, "rf29", selected=["f1", "f2"], attacks=(0, 1, 2, 3), seed=29)
    _run(tmp_path, "hgb17", model="hgb", seed=17)
    result = study.build_study(tmp_path)
    group = next(g for g in result["groups"] if g["spec"]["model"] == "rf")
    assert group["seeds"] == [17, 29]
    assert group["metrics"]["attack_recall"]["mean"] == .5
    assert group["metrics"]["attack_recall"]["std"] == pytest.approx(np.std([1 / 3, 2 / 3], ddof=1))
    assert group["features"]["mean_pairwise_jaccard"] == 1 / 3
    assert group["features"]["selection_counts"] == {"f0": 1, "f1": 2, "f2": 1}
    assert group["per_attack"][1]["category"] == "Fuzzers"
    assert group["per_attack"][1]["support"] == 3
    assert group["per_attack"][1]["recall"]["values"] == [2 / 3, 1.0]
    single = next(g for g in result["groups"] if g["spec"]["model"] == "hgb")
    assert single["metrics"]["attack_recall"]["std"] is None
    assert single["features"]["mean_pairwise_jaccard"] is None
    assert result["comparisons"]["common_seeds"] == [17]


def test_training_hyperparameters_and_raw_budgets_are_distinct_groups(tmp_path):
    _run(tmp_path, "rf100", hyperparameters={"n_estimators": 100})
    _run(tmp_path, "rf200", hyperparameters={"n_estimators": 200})
    _run(tmp_path, "rf100-mi2", selected=["f0", "f1"], hyperparameters={"n_estimators": 100})
    result = study.build_study(tmp_path)
    assert len(result["groups"]) == 3
    assert sorted(g["spec"]["raw_feature_budget"] for g in result["groups"]) == [2, 4, 4]


def test_duplicate_seed_is_not_an_independent_replicate(tmp_path):
    _run(tmp_path, "first")
    _run(tmp_path, "second")
    with pytest.raises(ValueError, match="Duplicate seed"):
        study.build_study(tmp_path)


@pytest.mark.parametrize("mismatch", ["ids", "labels", "category", "manifest"])
def test_inconsistent_sample_or_dataset_identity_is_rejected(tmp_path, mismatch):
    _run(tmp_path, "a")
    args = {"model": "hgb"}
    if mismatch == "ids":
        args["sample_ids"] = [f"changed-{i}" for i in range(106)]
    elif mismatch == "labels":
        args["labels"] = [1] * 5 + [0] * 101
    elif mismatch == "category":
        args["categories"] = ["Other"] * 6 + ["Normal"] * 100
    else:
        args["dataset_id"] = "different"
    _run(tmp_path, "b", **args)
    with pytest.raises(ValueError, match="different dataset|sample IDs, labels or attack categories"):
        study.build_study(tmp_path)


def test_float64_threshold_and_saved_decision_integrity(tmp_path):
    run = _run(tmp_path, float32=True)
    result = study.build_study(tmp_path)
    assert result["groups"][0]["metrics"]["normal_fpr"]["mean"] == .01
    pred = pd.read_parquet(run / "validation_predictions.parquet")
    assert pred["score"].dtype == np.float32
    assert pred["threshold"].iloc[0] == np.nextafter(.5, np.inf)
    pred.loc[7, "predicted_label"] = 1
    pred.to_parquet(run / "validation_predictions.parquet", index=False)
    _seal(run)
    with pytest.raises(ValueError, match="float64"):
        study.build_study(tmp_path)


def test_validation_only_no_pickle_and_every_run_verified(tmp_path, monkeypatch):
    run = _run(tmp_path)
    (run / "evaluation").mkdir()
    (run / "evaluation" / "test_predictions.parquet").write_bytes(b"reserved")
    (tmp_path / "calibration.parquet").write_bytes(b"reserved")
    calls = []
    original = study.verify_run

    def verify(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(study, "verify_run", verify)
    original_read = pd.read_parquet

    def read(path, *args, **kwargs):
        assert Path(path).name == "validation_predictions.parquet"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", read)
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *_a, **_kw: pytest.fail("study deserialized a model"))
    result = study.build_study(tmp_path)
    assert calls == [run]
    assert result["phase"] == "validation_development"


def test_tampered_artifact_fails_checksum(tmp_path):
    run = _run(tmp_path)
    (run / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        study.build_study(tmp_path)


def test_unlisted_artifact_and_reserved_manifest_entry_are_rejected(tmp_path, monkeypatch):
    run = _run(tmp_path)
    meta = read_json(run / "run.json")
    meta["files"]["test.parquet"] = "0" * 64
    write_json(run / "run.json", meta)
    monkeypatch.setattr(study, "verify_run", lambda *_a: pytest.fail("hashed reserved artifact"))
    with pytest.raises(ValueError, match="recognized development"):
        study.build_study(tmp_path)


def test_resealed_wrong_summary_is_detected_from_predictions(tmp_path):
    run = _run(tmp_path)
    metrics = read_json(run / "validation_metrics.json")
    metrics["attack_recall"] = .999
    write_json(run / "validation_metrics.json", metrics)
    _seal(run)
    with pytest.raises(ValueError, match="metrics disagree"):
        study.build_study(tmp_path)


def test_no_common_seed_does_not_invent_paired_results(tmp_path):
    _run(tmp_path, "a", seed=17)
    _run(tmp_path, "b", seed=29, model="hgb")
    result = study.build_study(tmp_path)
    assert result["comparisons"]["common_seeds"] == []
    assert result["comparisons"]["per_seed"] == []
    assert any("No seed is shared" in x for x in result["warnings"])


def test_failed_attempt_archive_is_explicitly_excluded(tmp_path):
    run = _run(tmp_path, "a")
    shutil.copytree(run, tmp_path / "failed_attempts" / "a")
    assert study.build_study(tmp_path)["n_runs"] == 1


def test_cpu_benchmarks_preserve_settings_and_ignore_plan(tmp_path):
    root, bench = tmp_path / "runs", tmp_path / "bench"
    a, b = _run(root, "a", seed=17), _run(root, "b", seed=29)
    _benchmark(bench, a, filename="a.json")
    _benchmark(bench, b, filename="b.json")
    _benchmark(bench, a, filename="a-two-threads.json", threads=2)
    _benchmark(bench, a, filename="a-other-cpu.json", processor="other-cpu")
    write_json(bench / "plan.json", {"kind": "benchmark_suite_plan"})
    result = study.build_study(root, bench)
    buckets = result["groups"][0]["cpu_benchmarks"]
    assert len(buckets) == 3
    aggregate = next(b for b in buckets if b["seeds"] == [17, 29])
    assert aggregate["metrics"]["p95_ms"]["mean"] == 3.85
    assert aggregate["metrics"]["p95_ms"]["std"] == 0
    assert aggregate["metrics"]["throughput_rows_per_second"]["mean"] == 1000
    assert all("environment_sha256" in b["spec"] for b in buckets)


@pytest.mark.parametrize("mutation", ["unbound", "dataset", "model", "source", "split", "run_id", "unknown"])
def test_cpu_benchmark_binding_is_mandatory(tmp_path, mutation):
    run = _run(tmp_path / "runs")
    path = _benchmark(tmp_path / "bench", run)
    value = read_json(path)
    if mutation == "unbound":
        del value["run_manifest_sha256"]
    else:
        key = {"dataset": "dataset_manifest_sha256", "model": "model_sha256",
               "source": "source_sha256", "split": "split", "run_id": "run_id",
               "unknown": "run_manifest_sha256"}[mutation]
        value[key] = "wrong"
    write_json(path, value)
    with pytest.raises(ValueError, match="binding|identity mismatch|unknown run"):
        study.build_study(tmp_path / "runs", tmp_path / "bench")


def test_duplicate_benchmark_cannot_inflate_seed_sample_size(tmp_path):
    run = _run(tmp_path / "runs")
    _benchmark(tmp_path / "bench", run, filename="a.json")
    _benchmark(tmp_path / "bench", run, filename="b.json")
    with pytest.raises(ValueError, match="Duplicate CPU benchmark"):
        study.build_study(tmp_path / "runs", tmp_path / "bench")


def test_undefined_metrics_stay_null_and_empty_root_is_rejected(tmp_path):
    assert study._aggregate([None]) == {"mean": None, "std": None, "min": None, "max": None,
                                        "n": 0, "values": [None]}
    with pytest.raises(ValueError, match="No completed"):
        study.build_study(tmp_path)


def test_different_training_environment_is_not_one_seed_distribution(tmp_path):
    env = {"cpu_physical": 8, "processor": "first", "recorded_at": "ignored"}
    _run(tmp_path, "a", environment=env)
    other = copy.deepcopy(env)
    other["processor"] = "second"
    _run(tmp_path, "b", environment=other)
    assert len(study.build_study(tmp_path)["groups"]) == 2


def test_feature_budget_deltas_pair_seeds_and_require_identical_fit_settings(tmp_path):
    _run(tmp_path, "all17", seed=17, attacks=(0, 1, 2, 3))
    _run(tmp_path, "all29", seed=29, attacks=(0, 1, 2, 3, 4))
    _run(tmp_path, "mi17", seed=17, attacks=(0, 1), selected=["f0", "f1"])
    _run(tmp_path, "mi29", seed=29, attacks=(0, 1, 2), selected=["f1", "f2"])
    _run(tmp_path, "unmatched-budget", seed=17, selected=["f0"], hyperparameters={"n_estimators": 200})
    output = study.build_study(tmp_path)["feature_budget_comparisons"]
    paired = next(x for x in output if x["raw_feature_budget"] == 2)
    assert paired["status"] == "paired" and paired["common_seeds"] == [17, 29]
    assert paired["metrics"]["attack_recall_delta_pp"]["mean"] == pytest.approx(-100 / 3)
    assert paired["metrics"]["fuzzers_recall_delta_pp"]["values"] == pytest.approx([-100 / 3, 0])
    assert paired["metrics"]["model_size_ratio"]["mean"] == 1
    assert paired["metrics"]["raw_feature_ratio"]["mean"] == .5
    unpaired = next(x for x in output if x["raw_feature_budget"] == 1)
    assert unpaired["baseline_group_id"] is None
    assert unpaired["status"] == "no_matching_all_field_baseline"
    assert unpaired["metrics"]["attack_recall_delta_pp"]["n"] == 0


def test_benchmark_summary_is_recomputed_from_raw_observations(tmp_path):
    run = _run(tmp_path / "runs")
    path = _benchmark(tmp_path / "bench", run)
    value = read_json(path)
    value["single"]["p95_ms"] = 0.01
    write_json(path, value)
    with pytest.raises(ValueError, match="raw timing"):
        study.build_study(tmp_path / "runs", tmp_path / "bench")


def _variant_run(root, variant, seed=17, attacks=(0, 1, 2), normal=(6,), learning_rate=.001):
    model = "hgb" if variant == "hgb_reference" else "mlp"
    hidden = [64, 32] if variant == "mlp_reference" else [32, 16]
    hyper = {"max_iter": 200} if model == "hgb" else {
        "hidden_sizes": hidden, "epochs": 120, "patience": 12, "learning_rate": learning_rate,
        "weight_decay": .0001, "batch_size": 1024}
    run = _run(root, f"{variant}-{seed}", model=model, seed=seed, attacks=attacks,
               normal=normal, hyperparameters=hyper)
    config, dataset, code = (read_json(run / name) for name in ("config.json", "dataset.json", "code.json"))
    config["model_config"]["variant"] = variant
    dataset["partitions"]["train"] = {"file": "train.parquet", "sha256": "d" * 64, "n_rows": 12}
    teacher = {"model": "hgb", "features": {"method": "all"}, "hyperparameters": {"max_iter": 200}}
    if variant == "student_kd":
        config["model_config"]["distillation"] = {"alpha": .5, "temperature": 1, "teacher": teacher,
                                                    "folds": 3, "fold_seed": 20261101}
    write_json(run / "config.json", config)
    write_json(run / "dataset.json", dataset)
    stats = {"schema_version": 1, "run_id": run.name, "model_family": model,
             "config_sha256": canonical_hash(config), "source_sha256": code["source_sha256"],
             "dataset_manifest_sha256": sha256_file(run / "dataset.json"),
             "parameter_count": None, "trainable_parameter_count": None, "best_epoch": None,
             "epochs_completed": None, "actual_epochs": None, "best_validation_loss": None}
    if model == "mlp":
        widths = [4, *hidden, 1]
        count = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
        stats.update(parameter_count=count, trainable_parameter_count=count, best_epoch=1,
                     epochs_completed=2, actual_epochs=2, best_validation_loss=.7, positive_weight=2.,
                     training_objective="weighted_hard_bce", distillation={"enabled": False})
        write_json(run / "training_history.json", [{"epoch": 1, "val_loss": .7}, {"epoch": 2, "val_loss": .8}])
    if variant == "student_kd":
        stats.update(training_objective="weighted_hard_bce_plus_unweighted_soft_bce",
                     distillation={"enabled": True, "alpha": .5, "temperature": 1, "teacher_model": "hgb"})
        effective = {**teacher, "hyperparameters": {**teacher["hyperparameters"], "early_stopping": False, "num_threads": 1}}
        oof_config = {"teacher_config": teacher, "effective_teacher_config": effective,
                      "seed": seed, "folds": 3, "fold_seed": 20261101, "threads": 1}
        names = ["config.json", "dataset.json", "code.json", "environment.json", "targets.parquet", "folds.json",
                 "fold-00.joblib", "fold-01.joblib", "fold-02.joblib"]
        manifest = {"schema_version": 1, "kind": "group_oof_teacher_targets", "teacher_config": teacher,
                    "seed": seed, "folds": 3, "fold_seed": 20261101, "threads": 1,
                    "source_sha256": code["source_sha256"], "dataset_manifest_sha256": stats["dataset_manifest_sha256"],
                    "train_partition_sha256": "d" * 64, "n_training_rows": 12,
                    "train_identity_sha256": "e" * 64, "train_identity_order": "original prepared train order",
                    "config_sha256": canonical_hash(oof_config), "score_semantics": "fixture uncalibrated teacher scores",
                    "splitter": "StratifiedGroupKFold", "group_column": "feature_hash",
                    "files": dict.fromkeys(names, "f" * 64)}
        write_json(run / "distillation.json", {
            "schema_version": 1, "method": "group_oof_hgb", "run_id": run.name,
            "config_sha256": stats["config_sha256"], "source_sha256": stats["source_sha256"],
            "dataset_manifest_sha256": stats["dataset_manifest_sha256"], "training_split": "train",
            "alpha": .5, "temperature": 1, "teacher_config": teacher, "folds": 3, "fold_seed": 20261101,
            "targets_sha256": "f" * 64, "teacher_manifest_sha256": "a" * 64,
            "teacher_completion_sha256": "b" * 64, "oof_manifest": manifest})
    write_json(run / "training_statistics.json", stats)
    _seal(run)
    return run


def test_distillation_and_capacity_have_distinct_matched_controls_and_family_evidence(tmp_path):
    for seed in [17, 29, 43]:
        _variant_run(tmp_path, "hgb_reference", seed, attacks=(0, 1, 2, 3, 4, 5))
        _variant_run(tmp_path, "mlp_reference", seed, attacks=(0, 1, 3, 4))
        _variant_run(tmp_path, "student_hard", seed, attacks=(0, 3))
        _variant_run(tmp_path, "student_kd", seed, attacks=(1, 3, 4), normal=(7,))
    result = study.build_study(tmp_path)
    assert result["n_runs"] == 12 and len(result["groups"]) == 4
    assert len(result["comparisons"]["per_seed"][0]["pairwise"]) == 6
    assert "or_alerts" in result["comparisons"]["per_seed"][0]
    assert {g["label"].split(" / ")[0] for g in result["groups"]} == {
        "全字段 HGB", "大 MLP", "小 MLP（硬标签）", "蒸馏小 MLP"}
    comparisons = {x["kind"]: x for x in result["distillation_comparisons"]}
    assert set(comparisons) == {"distillation_effect", "capacity_effect"}
    kd = comparisons["distillation_effect"]
    assert kd["common_seeds"] == [17, 29, 43] and kd["complete_expected_seeds"]
    assert kd["control_variant"] == "student_hard"
    assert kd["metrics"]["recovered_attacks"]["values"] == [2, 2, 2]
    assert kd["metrics"]["newly_missed_attacks"]["values"] == [1, 1, 1]
    assert kd["metrics"]["attack_recall_delta_pp"]["mean"] == pytest.approx(100 / 6)
    assert kd["metrics"]["parameter_ratio"]["mean"] == 1
    assert kd["metrics"]["normal_fpr_delta_pp"]["mean"] == 0
    assert kd["metrics"]["new_false_positives"]["mean"] == 1
    assert kd["metrics"]["resolved_false_positives"]["mean"] == 1
    families = {x["category"]: x for x in kd["per_seed"][0]["per_attack"]}
    assert families["Fuzzers"]["recovered_sample_ids"] == ["row-001"]
    assert families["Fuzzers"]["newly_missed_sample_ids"] == ["row-000"]
    assert families["Fuzzers"]["net_detected_gain"] == 0
    assert families["Exploits"]["net_detected_gain"] == 1
    assert comparisons["capacity_effect"]["metrics"]["parameter_ratio"]["mean"] < 1
    assert not kd["cpu_comparisons"]
    json.dumps(result, allow_nan=False)


def test_distillation_incomplete_seed_and_mismatched_optimizer_are_not_completed_controls(tmp_path):
    _variant_run(tmp_path, "student_hard", seed=17)
    _variant_run(tmp_path, "student_kd", seed=17)
    value = study.build_study(tmp_path)["distillation_comparisons"][0]
    assert not value["complete_expected_seeds"] and value["missing_expected_seeds"] == [29, 43]
    other = tmp_path / "mismatched-study"
    _variant_run(other, "student_hard", seed=17)
    _variant_run(other, "student_kd", seed=17, learning_rate=.002)
    value = next(x for x in study.build_study(other)["distillation_comparisons"] if x["kind"] == "distillation_effect")
    assert value["status"] == "no_matching_control"
    assert value["metrics"]["attack_recall_delta_pp"]["n"] == 0


@pytest.mark.parametrize("mutation", ["parameter_count", "actual_epochs", "best_validation_loss", "source_sha256"])
def test_resealed_training_metadata_mismatch_is_rejected(tmp_path, mutation):
    run = _variant_run(tmp_path, "student_hard")
    value = read_json(run / "training_statistics.json")
    value[mutation] = "wrong" if mutation == "source_sha256" else 999
    write_json(run / "training_statistics.json", value)
    _seal(run)
    with pytest.raises(ValueError, match="statistics|parameter count|epoch|history"):
        study.build_study(tmp_path)


@pytest.mark.parametrize("mutation", ["training_split", "targets_sha256", "oof_seed", "train_sha", "source", "config", "group_column", "files"])
def test_resealed_distillation_provenance_mismatch_is_rejected(tmp_path, mutation):
    run = _variant_run(tmp_path, "student_kd")
    value = read_json(run / "distillation.json")
    if mutation in {"training_split", "targets_sha256"}:
        value[mutation] = "test" if mutation == "training_split" else "0" * 64
    elif mutation == "files":
        value["oof_manifest"]["files"]["test.parquet"] = "0" * 64
    else:
        key = {"oof_seed": "seed", "train_sha": "train_partition_sha256", "source": "source_sha256",
               "config": "config_sha256", "group_column": "group_column"}[mutation]
        value["oof_manifest"][key] = 999 if mutation == "oof_seed" else "wrong"
    write_json(run / "distillation.json", value)
    _seal(run)
    with pytest.raises(ValueError, match="Distillation|OOF"):
        study.build_study(tmp_path)


def _round_benchmark(root, run, round_number, planned=3):
    path = _benchmark(root / f"round-{round_number}", run, filename=f"{run.name}.json")
    value = read_json(path)
    value.update(measurement_round=round_number, planned_rounds=planned)
    # Scale actual raw observations as well as summaries, preserving validation.
    for mode in ("single", "batch"):
        value[mode]["latencies_ms"] = [x * round_number for x in value[mode]["latencies_ms"]]
    value["single"]["p50_ms"] *= round_number
    value["single"]["p95_ms"] *= round_number
    value["batch"]["throughput_rows_per_second"] /= round_number
    write_json(path, value)
    return path


def test_round_medians_are_within_seed_and_missing_rounds_remain_incomplete(tmp_path):
    root, bench = tmp_path / "runs", tmp_path / "bench"
    a = _variant_run(root, "student_hard", seed=17)
    b = _variant_run(root, "student_hard", seed=29)
    for round_number in [1, 2, 3]:
        _round_benchmark(bench, a, round_number)
    _round_benchmark(bench, b, 1)
    result = study.build_study(root, bench)
    group = result["groups"][0]
    assert group["n_runs"] == 2 and len(group["cpu_benchmarks"]) == 3
    repeated = group["repeated_benchmarks"][0]
    assert repeated["status"] == "incomplete_rounds" and repeated["seeds"] == [17]
    assert repeated["missing_seeds"] == [29]
    assert repeated["metrics"]["p95_ms"]["mean"] == pytest.approx(7.7)
    assert repeated["metrics"]["p95_ms"]["n"] == 1
    assert repeated["per_seed"][1]["missing_rounds"] == [2, 3]
    assert repeated["per_seed"][0]["metrics"]["throughput_rows_per_second"]["median"] == 500


def test_cpu_ratios_pair_only_same_round_and_complete_round_medians(tmp_path):
    root, bench = tmp_path / "runs", tmp_path / "bench"
    a, b = _variant_run(root, "student_hard"), _variant_run(root, "student_kd")
    for run in (a, b):
        for round_number in [1, 2, 3]:
            _round_benchmark(bench, run, round_number)
    value = study.build_study(root, bench)["distillation_comparisons"][0]
    assert len(value["cpu_comparisons"]) == 4
    paired = next(x for x in value["cpu_comparisons"] if x["mode"] == "within_seed_round_median")
    assert paired["metrics"]["p95_ratio"]["values"] == [1.]
    assert paired["common_seeds"] == [17]
    duplicate = _round_benchmark(bench, a, 1)
    shutil.copyfile(duplicate, duplicate.parent / "duplicate.json")
    with pytest.raises(ValueError, match="Duplicate CPU benchmark"):
        study.build_study(root, bench)
