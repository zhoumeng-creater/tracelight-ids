"""Integration checks for research gates, split isolation and artifact integrity.

One real LR fit on a marked synthetic fixture is reused through isolated copies.
Hardware snapshots and source fingerprints are controlled test inputs; these
tests neither collect workstation details nor modify project source files.
"""

import copy
import shutil

import numpy as np
import pandas as pd
import pytest

from ids import runner
from ids.common import read_json, sha256_file, write_json
from ids.data import load_partition, make_synthetic_dataset, prepare_dataset
from ids.evaluation import calibrate_threshold, evaluate_binary
from ids.models import load_bundle


def _fingerprint(digest="a" * 64):
    return {"commit": None, "dirty": None, "source_sha256": digest, "files": {}}


@pytest.fixture(autouse=True)
def controlled_runtime(monkeypatch):
    monkeypatch.setattr(runner, "code_fingerprint", _fingerprint)
    monkeypatch.setattr(runner, "environment_snapshot", lambda: {"purpose": "integration_test_only"})


@pytest.fixture(scope="module")
def templates(tmp_path_factory):
    root = tmp_path_factory.mktemp("runner-template")
    sources = make_synthetic_dataset(root / "raw", n_train=600, n_test=120, seed=42)
    prepared = root / "prepared"
    prepare_dataset(*sources, prepared, seed=17, data_kind="synthetic")
    suite = {
        "suite_name": "integrity-fixture",
        "models": [{"model": "lr", "features": {"method": "all", "k": None},
                    "hyperparameters": {"max_iter": 100, "num_threads": 1}}],
        "seeds": [17],
        "max_fpr": 0.0,
        "threads": 1,
        "purpose": "integration_test_only",
    }
    # Module fixture precedes function-scoped autouse fixtures.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "code_fingerprint", _fingerprint)
        patch.setattr(runner, "environment_snapshot", lambda: {"purpose": "integration_test_only"})
        runs = runner.run_suite(prepared, suite, root / "runs")
    return {"data": prepared, "suite_dir": root / "runs" / suite["suite_name"],
            "suite": suite, "run_name": runs[0].name}


@pytest.fixture
def workspace(templates, tmp_path):
    data = tmp_path / "prepared"
    output_root = tmp_path / "runs"
    suite_dir = output_root / templates["suite"]["suite_name"]
    shutil.copytree(templates["data"], data)
    shutil.copytree(templates["suite_dir"], suite_dir)
    return {"data": data, "output_root": output_root, "suite_dir": suite_dir,
            "suite": copy.deepcopy(templates["suite"]),
            "run": suite_dir / templates["run_name"]}


def _append_bytes(path):
    with path.open("ab") as stream:
        stream.write(b"\nmodified-by-integrity-test\n")


def test_development_loads_only_training_and_validation(templates, tmp_path, monkeypatch):
    loaded = []

    def development_load(data_dir, split):
        assert split in {"train", "validation"}, "development opened a reserved partition"
        loaded.append(split)
        return load_partition(data_dir, split)

    monkeypatch.setattr(runner, "load_partition", development_load)
    run = runner.run_development(
        templates["data"], copy.deepcopy(templates["suite"]["models"][0]),
        tmp_path / "development", seed=17, max_fpr=0, threads=1,
    )
    assert loaded == ["train", "validation"]
    assert not (run / "evaluation").exists()
    assert not (run / "evaluation_plan.json").exists()
    metrics = read_json(run / "validation_metrics.json")
    threshold = read_json(run / "validation_threshold.json")
    assert metrics["split"] == "validation"
    assert metrics["phase"] == "development"
    assert metrics["data_kind"] == "synthetic"
    assert threshold["source_split"] == "validation"
    assert threshold["usage"] == "development_only_not_final_threshold"
    runner.verify_run(run)


def test_unfrozen_evaluation_fails_before_reading_reserved_data(workspace, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("evaluation without a freeze must not read any partition")

    monkeypatch.setattr(runner, "load_partition", forbidden)
    with pytest.raises(ValueError, match="freeze"):
        runner.evaluate_frozen(workspace["run"], workspace["data"])
    assert not (workspace["run"] / "evaluation").exists()


def test_freeze_requires_a_complete_run_and_is_not_overwritten(workspace):
    path = workspace["run"] / "run.json"
    original = read_json(path)
    write_json(path, {**original, "status": "running"})
    with pytest.raises(ValueError, match="not complete"):
        runner.freeze_run(workspace["run"])
    assert not (workspace["run"] / "evaluation_plan.json").exists()
    write_json(path, original)
    plan = runner.freeze_run(workspace["run"])
    assert plan["model_sha256"] == sha256_file(workspace["run"] / "model.joblib")
    assert plan["dataset_manifest_sha256"] == sha256_file(workspace["data"] / "manifest.json")
    frozen_bytes = (workspace["run"] / "evaluation_plan.json").read_bytes()
    with pytest.raises(FileExistsError, match="frozen"):
        runner.freeze_run(workspace["run"])
    assert (workspace["run"] / "evaluation_plan.json").read_bytes() == frozen_bytes


def test_completed_evaluation_uses_calibration_before_test_and_preserves_provenance(workspace, monkeypatch):
    run, data = workspace["run"], workspace["data"]
    calibration = load_partition(data, "calibration")
    test = load_partition(data, "test")
    assert set(calibration["sample_id"]).isdisjoint(test["sample_id"])
    bundle = load_bundle(run / "model.joblib", device="cpu")
    expected_scores = bundle.predict_scores(calibration)
    expected_operating = calibrate_threshold(calibration["label"], expected_scores, max_fpr=0)
    loaded, calibrations = [], []

    def recorded_load(data_dir, split):
        loaded.append(split)
        if split == "test":
            threshold = read_json(run / "evaluation" / "threshold.json")
            assert threshold["source_split"] == "calibration"
            assert threshold["threshold"] == expected_operating["threshold"]
        return load_partition(data_dir, split)

    def recorded_calibration(labels, scores, max_fpr=0.01):
        np.testing.assert_array_equal(np.asarray(labels), calibration["label"].to_numpy())
        np.testing.assert_array_equal(scores, expected_scores)
        calibrations.append(len(labels))
        return calibrate_threshold(labels, scores, max_fpr)

    monkeypatch.setattr(runner, "load_partition", recorded_load)
    monkeypatch.setattr(runner, "calibrate_threshold", recorded_calibration)
    runner.freeze_run(run)
    result = runner.evaluate_frozen(run, data)
    assert loaded == ["calibration", "test"]
    assert calibrations == [len(calibration)]
    assert result["phase"] == "frozen_evaluation"
    assert result["split"] == "test"
    assert result["threshold_source"] == "calibration"
    assert result["data_kind"] == "synthetic"
    assert result["n_samples"] == 120
    predictions = pd.read_parquet(run / "evaluation" / "test_predictions.parquet")
    assert predictions["sample_id"].tolist() == test["sample_id"].tolist()
    assert set(predictions["split"]) == {"test"}
    assert predictions["threshold"].nunique() == 1
    measured = evaluate_binary(test["label"], predictions["score"], expected_operating["threshold"])
    assert result["confusion"] == measured["confusion"]
    np.testing.assert_array_equal(
        predictions["predicted_label"],
        (predictions["score"].to_numpy(dtype=np.float64) >= expected_operating["threshold"]).astype(int),
    )
    completion = runner.verify_evaluation(run)
    assert completion["plan_sha256"] == sha256_file(run / "evaluation_plan.json")


def test_saved_validation_decisions_match_float64_threshold_and_metrics(workspace):
    run = workspace["run"]
    validation = load_partition(workspace["data"], "validation")
    bundle = load_bundle(run / "model.joblib", device="cpu")
    scores = bundle.predict_scores(validation)
    operating = read_json(run / "validation_threshold.json")
    predictions = pd.read_parquet(run / "validation_predictions.parquet")
    expected = scores.astype(np.float64) >= operating["threshold"]
    np.testing.assert_array_equal(predictions["predicted_label"], expected.astype(int))
    # At zero budget, the normal record defining the nextafter cutoff is rejected.
    normals = validation["label"].to_numpy() == 0
    assert np.any(scores[normals].astype(np.float64) == operating["next_disallowed_normal_score"])
    assert not expected[normals].any()
    assert not predictions.loc[normals, "predicted_label"].any()
    metrics = read_json(run / "validation_metrics.json")
    assert metrics["confusion"] == evaluate_binary(validation["label"], scores, operating["threshold"])["confusion"]


def test_float32_boundary_event_export_agrees_with_evaluator():
    scores = np.array([0.7, 0.7, 0.9], dtype=np.float32)
    frame = pd.DataFrame({"sample_id": ["n", "a1", "a2"], "label": [0, 1, 1],
                          "attack_cat": ["Normal", "DoS", "DoS"]})
    threshold = calibrate_threshold(frame["label"], scores, max_fpr=0)["threshold"]
    events = runner._predictions(frame, scores, threshold, "validation", "boundary", "synthetic")
    assert events["predicted_label"].tolist() == [0, 0, 1]
    assert evaluate_binary(frame["label"], scores, threshold)["confusion"] == {
        "tn": 1, "fp": 0, "fn": 1, "tp": 1,
    }


@pytest.mark.parametrize("filename", ["model.joblib", "validation_metrics.json", "dataset.json"])
def test_freeze_rejects_changed_run_artifacts(workspace, filename):
    _append_bytes(workspace["run"] / filename)
    with pytest.raises(ValueError, match="artifact changed"):
        runner.freeze_run(workspace["run"])
    assert not (workspace["run"] / "evaluation_plan.json").exists()


@pytest.mark.parametrize("target", ["model", "run_manifest", "dataset_manifest"])
def test_final_evaluation_rejects_changed_frozen_identities(workspace, target):
    run, data = workspace["run"], workspace["data"]
    runner.freeze_run(run)
    if target == "model":
        _append_bytes(run / "model.joblib")
    elif target == "run_manifest":
        manifest = read_json(run / "run.json")
        write_json(run / "run.json", {**manifest, "run_id": "changed-after-freeze"})
    else:
        path = data / "manifest.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(changed|mismatch)"):
        runner.evaluate_frozen(run, data)
    assert not (run / "evaluation" / "completion.json").exists()


def test_final_evaluation_rejects_modified_operating_budget_after_freeze(workspace):
    run = workspace["run"]
    runner.freeze_run(run)
    plan = read_json(run / "evaluation_plan.json")
    write_json(run / "evaluation_plan.json", {**plan, "max_fpr": 0.5})
    with pytest.raises(ValueError, match="(?i)(plan|budget|changed|checksum|mismatch)"):
        runner.evaluate_frozen(run, workspace["data"])
    assert not (run / "evaluation" / "completion.json").exists()


@pytest.mark.parametrize("split", ["calibration", "test"])
def test_corrupted_reserved_partition_cannot_produce_completed_evaluation(workspace, split):
    run = workspace["run"]
    runner.freeze_run(run)
    _append_bytes(workspace["data"] / f"{split}.parquet")
    with pytest.raises(ValueError, match="checksum"):
        runner.evaluate_frozen(run, workspace["data"])
    assert not (run / "evaluation" / "completion.json").exists()


def test_source_changes_block_final_evaluation_before_reserved_data(workspace, monkeypatch):
    runner.freeze_run(workspace["run"])
    monkeypatch.setattr(runner, "code_fingerprint", lambda: _fingerprint("b" * 64))

    def forbidden(*_args, **_kwargs):
        pytest.fail("source identity must be checked before loading reserved data")

    monkeypatch.setattr(runner, "load_partition", forbidden)
    with pytest.raises(ValueError, match="(?i)source"):
        runner.evaluate_frozen(workspace["run"], workspace["data"])


def test_evaluation_cannot_overwrite_existing_results(workspace):
    run = workspace["run"]
    runner.freeze_run(run)
    runner.evaluate_frozen(run, workspace["data"])
    completion_hash = sha256_file(run / "evaluation" / "completion.json")
    with pytest.raises(FileExistsError, match="already"):
        runner.evaluate_frozen(run, workspace["data"])
    assert sha256_file(run / "evaluation" / "completion.json") == completion_hash


def test_evaluation_verification_detects_modified_result(workspace):
    run = workspace["run"]
    runner.freeze_run(run)
    runner.evaluate_frozen(run, workspace["data"])
    _append_bytes(run / "evaluation" / "test_predictions.parquet")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.verify_evaluation(run)


def test_resume_of_unchanged_suite_skips_training(workspace, monkeypatch):
    original_hash = sha256_file(workspace["run"] / "model.joblib")

    def forbidden(*_args, **_kwargs):
        pytest.fail("unchanged completed runs must not be trained again")

    monkeypatch.setattr(runner, "train_model", forbidden)
    runs = runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)
    assert runs == [workspace["run"]]
    assert sha256_file(workspace["run"] / "model.joblib") == original_hash


def test_resume_requires_explicit_flag_and_unchanged_suite(workspace):
    with pytest.raises(FileExistsError, match="resume"):
        runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"])
    changed = copy.deepcopy(workspace["suite"])
    changed["max_fpr"] = 0.1
    with pytest.raises(ValueError, match="Cannot resume"):
        runner.run_suite(workspace["data"], changed, workspace["output_root"], resume=True)


def test_resume_rejects_source_change(workspace, monkeypatch):
    monkeypatch.setattr(runner, "code_fingerprint", lambda: _fingerprint("c" * 64))
    with pytest.raises(ValueError, match="(?i)source"):
        runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)


def test_resume_rejects_manifest_change(workspace):
    path = workspace["data"] / "manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot resume"):
        runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_resume_rejects_changed_development_partition_even_with_original_manifest(workspace, split):
    _append_bytes(workspace["data"] / f"{split}.parquet")
    with pytest.raises(ValueError, match="(?i)(checksum|changed|mismatch)"):
        runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)


def test_resume_never_loads_calibration_or_test(workspace, monkeypatch):
    def development_load(data_dir, split):
        assert split in {"train", "validation"}, "resume opened a reserved partition"
        return load_partition(data_dir, split)

    monkeypatch.setattr(runner, "load_partition", development_load)
    runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)


def test_resume_preserves_incomplete_attempt_and_retrains_same_configuration(workspace, monkeypatch):
    run = workspace["run"]
    prior_model_hash = sha256_file(run / "model.joblib")
    prior_config = read_json(run / "config.json")
    # Simulate interruption after model/metrics writes but before the completion manifest.
    (run / "run.json").unlink()
    real_run_development = runner.run_development
    calls = []

    def recorded_development(*args, **kwargs):
        calls.append((args, kwargs))
        return real_run_development(*args, **kwargs)

    monkeypatch.setattr(runner, "run_development", recorded_development)
    runs = runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)
    assert runs == [run]
    assert len(calls) == 1
    archived = list((workspace["suite_dir"] / "failed_attempts").glob(f"{run.name}-*"))
    assert len(archived) == 1
    assert archived[0].is_dir()
    assert not (archived[0] / "run.json").exists()
    assert sha256_file(archived[0] / "model.joblib") == prior_model_hash
    assert read_json(archived[0] / "config.json") == prior_config
    current_config = read_json(run / "config.json")
    assert current_config["seed"] == prior_config["seed"]
    assert current_config["model_config"] == prior_config["model_config"]
    assert runner.verify_run(run)["status"] == "complete"


def test_resume_does_not_treat_a_corrupted_completed_run_as_an_interruption(workspace):
    _append_bytes(workspace["run"] / "model.joblib")
    with pytest.raises(ValueError, match="artifact changed"):
        runner.run_suite(workspace["data"], workspace["suite"], workspace["output_root"], resume=True)
    assert (workspace["run"] / "run.json").exists()
    assert not (workspace["suite_dir"] / "failed_attempts").exists()
