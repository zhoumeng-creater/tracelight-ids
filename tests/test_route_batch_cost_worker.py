"""Fresh-cost entry tests use generated CSVs and synthetic model-loader stubs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file
from ids.context_cache.preprocessing import CSVSchema
from ids.route_batch.cost_worker import run_cost


@pytest.fixture
def deployment(tmp_path):
    schema = CSVSchema.from_dict({"features": ["value", "other"], "numeric": ["value", "other"],
                                  "label_column": "label", "id_column": "sample_id",
                                  "group_column": "group", "family_column": "family"})
    frame = pd.DataFrame({"value": np.arange(640, dtype=float), "other": np.arange(640) % 7,
                          "label": np.arange(640) % 2, "sample_id": [f"e-{i}" for i in range(640)],
                          "group": [f"g-{i}" for i in range(640)], "family": ["a"] * 640})
    source = tmp_path / "evaluation.csv"
    frame.to_csv(source, index=False)
    spec = {"schema": schema.to_dict(), "files": {
        "evaluation": {"path": source.name, "sha256": sha256_file(source), "rows": len(frame)},
        "support": {"path": "MUST-NOT-READ-P.csv"}, "calibration": {"path": "MUST-NOT-READ-C.csv"}},
        "settings": {"threads": 1, "query_batch_size": 128,
                     "benchmark": {"repeats": 1, "single_rows": 2, "warmup": 0}}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    target = tmp_path / "deployment"
    target.mkdir()
    (target / "model.pt").write_bytes(b"synthetic-model")
    (target / "preprocessing.joblib").write_bytes(b"synthetic-preprocessing")
    (target / "threshold.json").write_text("{}", encoding="utf-8")
    descriptor = {"kind": "tabm", "model_file": "model.pt", "preprocessing_file": "preprocessing.joblib",
                  "model_sha256": sha256_file(target / "model.pt"),
                  "preprocessing_sha256": sha256_file(target / "preprocessing.joblib"),
                  "score_contract": "synthetic-sum"}
    (target / "deployment.json").write_text(json.dumps(descriptor), encoding="utf-8")
    pd.DataFrame({"row_id": frame["sample_id"], "label": frame["label"],
                  "score": frame["value"] + frame["other"]}).to_csv(target / "evaluation.csv", index=False)
    return config, target, schema, frame, descriptor


def patch_tabm(monkeypatch, schema):
    calls = []

    class FakeRepresentation:
        def __init__(self):
            self.schema = schema

        def transform(self, frame):
            assert list(frame.columns) == list(schema.features)
            return frame[list(schema.features)].to_numpy(dtype=np.float32)

    class FakeModel:
        def predict_scores(self, matrix, batch_size):
            assert batch_size == 128
            return matrix.sum(axis=1).astype(np.float64)

    def load_preprocessing(path):
        calls.append(("preprocessing", Path(path).name))
        return FakeRepresentation()

    def load_model(path, device):
        calls.append(("model", Path(path).name, device))
        return FakeModel()

    monkeypatch.setattr("ids.route_batch.cost_worker.Representation.load", load_preprocessing)
    monkeypatch.setattr("ids.route_batch.tabm_training.load_predictor", load_model)
    return calls


def test_tabm_reloads_complete_deployment_and_reads_only_e(deployment, monkeypatch):
    config, target, schema, _, descriptor = deployment
    calls = patch_tabm(monkeypatch, schema)
    result = run_cost(config, target)
    assert calls == [("preprocessing", "preprocessing.joblib"), ("model", "model.pt", "cpu")]
    assert result["worker"]["loaded_data_roles"] == ["evaluation"]
    assert result["worker"]["gradient_training_performed"] is False
    assert result["worker"]["registered_evaluation_rows_in_memory"] == 640
    assert result["replay"]["rows"] == 128
    assert result["replay"]["max_absolute_score_error"] == 0
    assert result["csv_to_scores"]["rows"] == 512
    assert len(pd.read_csv(target / "benchmark_features.csv")) == 512
    assert list(pd.read_csv(target / "benchmark_features.csv").columns) == list(schema.features)
    expected = ["model.pt", "preprocessing.joblib", "threshold.json", "deployment.json"]
    assert result["required_deployment_file_bytes"] == sum((target / name).stat().st_size for name in expected)
    assert result["cold_model_load_measured"] is True
    assert "fresh inference-only worker" in result["rss"]["scope"]
    assert json.loads((target / "cost.json").read_text(encoding="utf-8"))["counts"] == result["counts"]


@pytest.mark.parametrize("column", ["row_id", "label", "score"])
def test_replay_mismatch_is_rejected(deployment, monkeypatch, column):
    config, target, schema, *_ = deployment
    patch_tabm(monkeypatch, schema)
    saved = pd.read_csv(target / "evaluation.csv")
    saved.loc[0, column] = {"row_id": "wrong-id", "label": 1, "score": 1000}[column]
    saved.to_csv(target / "evaluation.csv", index=False)
    with pytest.raises(ValueError, match="replay IDs or labels|reproduce saved evaluation scores"):
        run_cost(config, target)
    assert not (target / "cost.json").exists()
    if column == "score":
        diagnostic = json.loads((target / "replay_diagnostic.json").read_text(encoding="utf-8"))
        assert diagnostic["mismatched_rows"] == 1
        assert diagnostic["max_absolute_score_error"] == 1000
        assert diagnostic["timing_allowed"] is False
        assert len(pd.read_csv(target / "replay_scores.csv")) == 128


def test_failed_replay_retains_threshold_decision_flips(deployment, monkeypatch):
    config, target, schema, *_ = deployment
    patch_tabm(monkeypatch, schema)
    saved = pd.read_csv(target / "evaluation.csv")
    saved.loc[0, "score"] = 1000
    saved.to_csv(target / "evaluation.csv", index=False)
    (target / "threshold.json").write_text('{"threshold":500}', encoding="utf-8")
    with pytest.raises(ValueError, match="reproduce saved evaluation scores"):
        run_cost(config, target)
    diagnostic = json.loads((target / "replay_diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["decision_changed_rows"] == 1
    row = pd.read_csv(target / "replay_scores.csv").iloc[0]
    assert row["saved_threshold_decision"] and not row["replayed_threshold_decision"]
    assert not (target / "cost.json").exists()


def test_within_tolerance_threshold_flip_stops_cost(deployment, monkeypatch):
    config, target, schema, *_ = deployment
    patch_tabm(monkeypatch, schema)
    saved = pd.read_csv(target / "evaluation.csv")
    saved.loc[0, "score"] = -2e-5
    saved.to_csv(target / "evaluation.csv", index=False)
    (target / "threshold.json").write_text('{"threshold":-0.00001}', encoding="utf-8")
    with pytest.raises(ValueError, match="reproduce saved evaluation scores"):
        run_cost(config, target)
    report = json.loads((target / "replay_diagnostic.json").read_text(encoding="utf-8"))
    assert report["scores_within_tolerance"] and report["mismatched_rows"] == 0
    assert report["decision_changed_rows"] == 1
    assert report["status"] == "mismatch" and not report["timing_allowed"]
    assert not (target / "cost.json").exists()


@pytest.mark.parametrize("filename", ["model.pt", "preprocessing.joblib"])
def test_hashes_are_checked_before_deserialization(deployment, monkeypatch, filename):
    config, target, schema, *_ = deployment
    calls = patch_tabm(monkeypatch, schema)
    (target / filename).write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_cost(config, target)
    assert calls == []


def test_changed_e_file_fails_before_model_loading(deployment, monkeypatch):
    config, target, schema, *_ = deployment
    calls = patch_tabm(monkeypatch, schema)
    path = config.parent / "evaluation.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="E SHA-256 mismatch"):
        run_cost(config, target)
    assert calls == []


@pytest.mark.parametrize("checkpoint_exists", [True, False])
def test_foundation_counts_real_external_checkpoint_and_rebuilds_on_cpu(deployment, monkeypatch, checkpoint_exists):
    config, target, _, _, descriptor = deployment
    checkpoint = config.parent / "external.ckpt"
    if checkpoint_exists:
        checkpoint.write_bytes(b"synthetic-external-weights")
    descriptor = {"kind": "foundation", "model_file": "model.pt", "model_sha256": descriptor["model_sha256"]}
    (target / "deployment.json").write_text(json.dumps(descriptor), encoding="utf-8")
    calls = []

    class FakeFoundation:
        metadata = {"checkpoint_path": str(checkpoint), "checkpoint_bytes": 999999}

        def predict_scores(self, matrix, batch_size):
            return matrix.sum(axis=1)

    def load_foundation(path, *, device):
        calls.append((Path(path).name, device))
        return FakeFoundation()

    monkeypatch.setattr("ids.route_batch.foundation.load_foundation", load_foundation)
    if not checkpoint_exists:
        with pytest.raises(FileNotFoundError, match="required deployment file"):
            run_cost(config, target)
        assert not (target / "cost.json").exists()
    else:
        result = run_cost(config, target)
        assert calls == [("model.pt", "cpu")]
        expected = sum((target / name).stat().st_size for name in ("model.pt", "threshold.json", "deployment.json"))
        assert result["required_deployment_file_bytes"] == expected + checkpoint.stat().st_size
        assert result["deployment_files"][str(checkpoint.resolve())] != 999999


def test_reference_uses_original_verified_loader_and_original_package_size(deployment, monkeypatch):
    config, target, _, _, _ = deployment
    reference = config.parent / "old-full"
    reference.mkdir()
    for name in ("model.pt", "preprocessor.joblib", "threshold.json", "deployment.json"):
        (reference / name).write_bytes(b"synthetic-old-file")
    (target / "deployment.json").write_text(json.dumps({"kind": "reference", "directory": str(reference)}),
                                           encoding="utf-8")
    calls = []

    class FakeDetector:
        def predict_frame(self, frame):
            return pd.DataFrame({"score": frame["value"] + frame["other"]})

    def load_reference(path):
        calls.append(Path(path))
        return FakeDetector()

    monkeypatch.setattr("ids.context_cache.csv_workflow.CSVDetector.load", load_reference)
    result = run_cost(config, target)
    assert calls == [reference]
    assert result["required_deployment_file_bytes"] == sum(path.stat().st_size for path in reference.iterdir())
