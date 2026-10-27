"""CLI integration checks using small, explicitly synthetic research artifacts."""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from ids import benchmark, cli, runner
from ids.common import read_json, write_json
from ids.data import OFFICIAL_FILES, load_unsw_csv
from ids.models import load_bundle

REPO = Path(__file__).resolve().parents[1]


def _fingerprint():
    return {"commit": None, "dirty": None, "source_sha256": "d" * 64, "files": {}}


@pytest.fixture(autouse=True)
def controlled_runtime(monkeypatch):
    monkeypatch.setattr(runner, "code_fingerprint", _fingerprint)
    monkeypatch.setattr(runner, "environment_snapshot", lambda: {"purpose": "cli_test_only"})
    monkeypatch.setattr(cli, "environment_snapshot", lambda: {"purpose": "cli_test_only"})


@pytest.fixture(scope="module")
def cli_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-template")
    raw, prepared = root / "raw with spaces", root / "prepared with spaces"
    config = {
        "suite_name": "cli-fixture",
        "models": [
            {"model": "lr", "features": {"method": "all", "k": None},
             "hyperparameters": {"max_iter": 100}},
            {"model": "rf", "features": {"method": "all", "k": None},
             "hyperparameters": {"n_estimators": 4, "max_depth": 3}},
        ],
        "seeds": [17], "max_fpr": 0.0, "threads": 2, "purpose": "cli_test_only",
    }
    config_path = root / "suite config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "code_fingerprint", _fingerprint)
        patch.setattr(runner, "environment_snapshot", lambda: {"purpose": "cli_test_only"})
        cli.main(["synthetic", "--output", str(raw), "--n-train", "600", "--n-test", "120"])
        cli.main(["prepare", "--train-csv", str(raw / OFFICIAL_FILES["train"][0]),
                  "--test-csv", str(raw / OFFICIAL_FILES["test"][0]), "--output", str(prepared),
                  "--data-kind", "synthetic", "--seed", "17"])
        cli.main(["suite", "--data-dir", str(prepared), "--config", str(config_path),
                  "--output-root", str(root / "runs"), "--device", "cpu"])
    unlabeled = root / "unlabeled input.csv"
    pd.read_csv(raw / OFFICIAL_FILES["test"][0]).drop(columns=["label", "attack_cat"]).to_csv(
        unlabeled, index=False,
    )
    return {"data": prepared, "config": config, "suite_dir": root / "runs" / config["suite_name"],
            "input": unlabeled}


@pytest.fixture
def workspace(cli_template, tmp_path):
    data, suite_dir = tmp_path / "prepared with spaces", tmp_path / "runs" / "cli-fixture"
    shutil.copytree(cli_template["data"], data)
    shutil.copytree(cli_template["suite_dir"], suite_dir)
    source = tmp_path / "unlabeled input.csv"
    shutil.copyfile(cli_template["input"], source)
    config_path = tmp_path / "config with spaces" / "suite config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(cli_template["config"]), encoding="utf-8")
    return {"data": data, "run": suite_dir / "00-lr-s17", "rf_run": suite_dir / "01-rf-s17",
            "input": source, "root": tmp_path, "config": config_path}


def _predict_args(workspace, output):
    return ["predict", "--run-dir", str(workspace["run"]), "--input", str(workspace["input"]),
            "--output", str(output)]


def _finish_evaluation(workspace):
    cli.main(["freeze", "--run-dir", str(workspace["run"])])
    cli.main(["evaluate", "--run-dir", str(workspace["run"]), "--data-dir", str(workspace["data"])])


def test_real_module_cli_predicts_unlabeled_csv_without_inventing_accuracy(workspace):
    output = workspace["root"] / "prediction output.csv"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [sys.executable, "-m", "ids", *_predict_args(workspace, output)],
        cwd=REPO, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "No accuracy inferred" in result.stdout
    frame = pd.read_csv(output)
    assert len(frame) == 120
    assert not {"label", "true_label", "accuracy"}.intersection(frame.columns)
    assert set(frame["threshold_source"]) == {"validation_development_only"}
    assert frame["sample_id"].is_unique
    source = load_unsw_csv(workspace["input"], require_labels=False)
    scores = load_bundle(workspace["run"] / "model.joblib").predict_scores(source)
    threshold = read_json(workspace["run"] / "validation_threshold.json")["threshold"]
    np.testing.assert_allclose(frame["score"], scores, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(frame["predicted_label"], (scores.astype(np.float64) >= threshold).astype(int))


def test_predict_uses_calibration_threshold_only_after_complete_evaluation(workspace):
    _finish_evaluation(workspace)
    output = workspace["root"] / "calibrated predictions.csv"
    cli.main(_predict_args(workspace, output))
    frame = pd.read_csv(output)
    threshold = read_json(workspace["run"] / "evaluation" / "threshold.json")["threshold"]
    assert set(frame["threshold_source"]) == {"calibration"}
    assert frame["threshold"].to_numpy() == pytest.approx(np.full(len(frame), threshold))
    source = load_unsw_csv(workspace["input"], require_labels=False)
    scores = load_bundle(workspace["run"] / "model.joblib").predict_scores(source)
    np.testing.assert_array_equal(frame["predicted_label"], (scores.astype(np.float64) >= threshold).astype(int))


@pytest.mark.parametrize("state", ["modified_validation", "modified_calibration", "incomplete_calibration"])
def test_predict_rejects_invalid_threshold_provenance_without_fallback(workspace, state, capsys):
    run = workspace["run"]
    if state == "modified_validation":
        path = run / "validation_threshold.json"
        value = read_json(path)
        write_json(path, {**value, "threshold": 0.0})
    elif state == "modified_calibration":
        _finish_evaluation(workspace)
        path = run / "evaluation" / "threshold.json"
        value = read_json(path)
        write_json(path, {**value, "threshold": 0.0})
    else:
        # A crash after calibration does not produce a verified final artifact.
        write_json(run / "evaluation" / "threshold.json", {"threshold": 0.0, "source_split": "calibration"})
    output = workspace["root"] / "must not exist.csv"
    with pytest.raises(SystemExit) as error:
        cli.main(_predict_args(workspace, output))
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_predict_refuses_to_overwrite_existing_output(workspace):
    output = workspace["root"] / "existing.csv"
    output.write_text("keep-existing-output\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(_predict_args(workspace, output))
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep-existing-output\n"


@pytest.mark.parametrize("model", ["lr", "rf"])
def test_benchmark_cli_overrides_persisted_training_threads(workspace, model, monkeypatch):
    run = workspace["rf_run"] if model == "rf" else workspace["run"]
    original = load_bundle(run / "model.joblib")
    assert original.num_threads == 2
    if model == "rf":
        assert original.estimator.n_jobs == 2
    real_benchmark = benchmark.benchmark_predict
    observed = []

    def short_checked_benchmark(predict_fn, frame, **kwargs):
        bundle = predict_fn.__self__
        assert bundle.num_threads == 1
        assert kwargs["threads"] == 1
        if model == "rf":
            assert bundle.estimator.n_jobs == 1
        assert frame.attrs["split"] == "validation"
        observed.append(bundle.model_name)
        # Exercise the real predictor and benchmark, shortening only repetitions.
        return real_benchmark(predict_fn, frame, **kwargs, warmup=1, repeats=2, single_rows=3, batch_size=8)

    monkeypatch.setattr(benchmark, "benchmark_predict", short_checked_benchmark)
    output = workspace["root"] / f"{model} benchmark.json"
    cli.main(["benchmark", "--run-dir", str(run), "--data-dir", str(workspace["data"]),
              "--output", str(output), "--threads", "1"])
    result = read_json(output)
    assert observed == [model]
    assert result["threads"] == 1
    assert result["scope"] == "preprocessing_plus_predict_fn"
    assert result["single"]["n_observations"] == 6
    assert result["batch"]["throughput_rows_per_second"] > 0
    # Benchmark overrides are in-memory only; training artifacts stay unchanged.
    assert load_bundle(run / "model.joblib").num_threads == 2
    runner.verify_run(run)


def test_benchmark_cli_rejects_dataset_identity_change(workspace):
    path = workspace["data"] / "manifest.json"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    output = workspace["root"] / "must not exist.json"
    with pytest.raises(SystemExit) as error:
        cli.main(["benchmark", "--run-dir", str(workspace["run"]), "--data-dir", str(workspace["data"]),
                  "--output", str(output), "--threads", "1"])
    assert error.value.code == 2
    assert not output.exists()


def _lab_module():
    spec = importlib.util.spec_from_file_location("tracelight_test_run_lab", REPO / "scripts" / "run_lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lab_dry_run_never_executes_or_freezes_and_preserves_space_paths(workspace, monkeypatch, capsys):
    module = _lab_module()

    def forbidden(*_args, **_kwargs):
        pytest.fail("dry-run must never launch a subprocess")

    monkeypatch.setattr(module.subprocess, "run", forbidden)
    work = workspace["root"] / "lab work with spaces"
    result = module.main(["--data-dir", str(workspace["data"]), "--work-dir", str(work),
                          "--config", str(workspace["config"]), "--resume", "--dry-run"])
    assert result == 0
    assert not work.exists()
    commands = [ast.literal_eval(line.removeprefix("argv: "))
                for line in capsys.readouterr().out.splitlines() if line.startswith("argv: ")]
    assert [command[3] for command in commands] == ["doctor", "suite", "report"]
    assert all("freeze" not in command and "evaluate" not in command for command in commands)
    assert "--require-cuda" in commands[0]
    suite = commands[1]
    assert suite[suite.index("--data-dir") + 1] == str(workspace["data"].resolve())
    assert suite[suite.index("--config") + 1] == str(workspace["config"].resolve())
    assert suite[suite.index("--output-root") + 1] == str(work.resolve() / "runs")
    assert "--resume" in suite


def test_lab_stops_on_failed_suite_instead_of_continuing_to_report(workspace, monkeypatch):
    module = _lab_module()
    calls = []

    def fail_suite(command, *, cwd, env, check):
        calls.append(copy.deepcopy(command))
        assert cwd == REPO
        assert check is True
        assert env["PYTHONHASHSEED"] == "0"
        assert env["OMP_NUM_THREADS"] == "2"
        assert isinstance(command, list)
        if command[3] == "suite":
            raise subprocess.CalledProcessError(7, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fail_suite)
    code = module.main(["--data-dir", str(workspace["data"]), "--work-dir", str(workspace["root"] / "lab work"),
                        "--config", str(workspace["config"]), "--device", "cpu"])
    assert code == 7
    assert [command[3] for command in calls] == ["doctor", "suite"]
