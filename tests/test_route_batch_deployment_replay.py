"""Synthetic full-C/E cold replay; no real model, TabM, fit, or benchmark."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.route_batch import deployment_replay as cold
from ids.route_batch.metrics import decisions, threshold


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    directory = tmp_path / "new-cpu-deployment"
    directory.mkdir()
    recipe = directory / "model.joblib"
    recipe.write_bytes(b"synthetic recipe placeholder; loader is mocked")
    descriptor = {"kind": "foundation", "rebuild_device": "cpu", "model_file": "model.joblib",
                  "model_sha256": sha256_file(recipe), "metadata": {"kind": "tabicl", "build_device": "cpu"},
                  "score_contract": "synthetic test score"}
    write_json(directory / "deployment.json", descriptor)
    schema = {"features": ["x"], "numeric": ["x"], "categorical": [], "label_column": "label",
              "id_column": "sample_id", "group_column": "communication_group", "family_column": "attack_family"}
    frames, files = {}, {}
    for role, n in (("calibration", 100), ("evaluation", 130)):
        y = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        x = np.linspace(-1, 1, n) if role == "calibration" else np.tile([.5, 2.], n // 2)
        if role == "calibration":
            x[-2] = .99999
        frame = pd.DataFrame({"x": x, "label": y,
                              "sample_id": [f"{role}-{i}" for i in range(n)],
                              "communication_group": [f"{role}-group-{i}" for i in range(n)],
                              "attack_family": np.where(y, "Scanning", "Benign")})
        path = tmp_path / f"registered-{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "rows": n, "sha256": sha256_file(path)}
        frames[role] = frame
    # This file intentionally does not exist. Replay must not load registered P.
    files["support"] = {"path": str(tmp_path / "MUST-NOT-READ-P.csv"), "rows": 8, "sha256": "a" * 64}
    config = tmp_path / "config.json"
    spec = {"schema": schema, "files": files, "settings": {"alpha": .01, "threads": 1, "query_batch_size": 128}}
    write_json(config, spec)
    registration = threshold(frames["calibration"].x)
    registration.update(model=descriptor, calibration_sha256=files["calibration"]["sha256"])
    write_json(directory / "threshold.json", registration)
    for role, frame in frames.items():
        p = 1 / (1 + np.exp(-frame.x.to_numpy()))
        pd.DataFrame({"row_id": frame.sample_id, "label": frame.label, "attack_family": frame.attack_family,
                      "score": frame.x, "predicted_attack": decisions(frame.x, registration).astype(int),
                      "raw_probability_normal": 1 - p, "raw_probability_attack": p}).to_csv(directory / f"{role}.csv", index=False)
    calls = {"loads": [], "predict_rows": [], "transform": None, "fail_e": False}
    class Predictor:
        metadata = {"kind": "tabicl", "build_device": "cpu", "score_device": "cpu"}
        def predict_outputs(self, x, *, batch_size, diagnostics):
            assert batch_size == 128 and diagnostics is True
            calls["predict_rows"].append(len(x))
            role = "calibration" if len(x) == 100 else "evaluation"
            if calls["fail_e"] and role == "evaluation":
                raise RuntimeError("synthetic E score failure")
            score = x[:, 0].copy()
            if calls["transform"]:
                score = calls["transform"](role, score)
            p = 1 / (1 + np.exp(-score))
            return {"scores": score, "probabilities": np.column_stack([1-p, p])}
    def load(path, **kwargs):
        assert path == recipe and kwargs == {"device": "cpu", "diagnostics": True}
        calls["loads"].append(path)
        return Predictor()
    monkeypatch.setattr(cold, "load_foundation", load)
    return SimpleNamespace(directory=directory, config=config, spec=spec, frames=frames,
                           descriptor=descriptor, calls=calls, recipe=recipe)


def _read_report(fixture):
    return read_json(fixture.directory / "cold_replay_report.json")


def _set_saved_e_score(fixture, row, score):
    path = fixture.directory / "evaluation.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame.loc[row, "score"] = score
    reg = read_json(fixture.directory / "threshold.json")
    frame["predicted_attack"] = decisions(frame.score, reg).astype(int)
    frame.to_csv(path, index=False)
    registered = fixture.frames["evaluation"].copy()
    registered.loc[row, "x"] = score
    input_path = fixture.spec["files"]["evaluation"]["path"]
    registered.to_csv(input_path, index=False)
    fixture.spec["files"]["evaluation"]["sha256"] = sha256_file(input_path)
    write_json(fixture.config, fixture.spec)


@pytest.mark.parametrize("probability_sidecar", [False, True])
def test_full_c_then_e_passes_without_touching_saved_deployment(deployment, probability_sidecar):
    f = deployment
    if probability_sidecar:
        for role in ("calibration", "evaluation"):
            path = f.directory / f"{role}.csv"
            data = pd.read_csv(path, float_precision="round_trip")
            data[["row_id", "raw_probability_normal", "raw_probability_attack"]].rename(columns={
                "raw_probability_normal": "probability_normal", "raw_probability_attack": "probability_attack"
            }).to_csv(f.directory / f"{role}_probabilities.csv", index=False)
            data.drop(columns=["raw_probability_normal", "raw_probability_attack"]).to_csv(path, index=False)
    before = {path: path.read_bytes() for path in f.directory.iterdir() if path.is_file()}
    report = cold.run(f.config, f.directory)
    assert report["status"] == "matched" and report["timing_allowed"]
    assert all(report["gates"].values())
    assert f.calls["predict_rows"] == [100, 130] and len(f.calls["loads"]) == 1
    assert all(path.read_bytes() == value for path, value in before.items())
    assert report["loaded_data_roles"] == ["calibration", "evaluation"]
    assert report["bootstrap_performed"] is False and report["benchmark_performed"] is False
    for role, rows in (("calibration", 100), ("evaluation", 130)):
        evidence = pd.read_csv(f.directory / f"cold_replay_{role}.csv")
        assert len(evidence) == rows
        assert {"saved_score", "replayed_score", "raw_probability_attack", "saved_raw_probability_attack",
                "replayed_saved_threshold_decision", "replayed_recalibrated_decision"}.issubset(evidence)
    assert report["evaluation_saved_probability_source"]["source"] == (
        "saved probability sidecar" if probability_sidecar else "embedded saved score CSV")


def test_drift_after_first_128_e_rows_is_saved_and_blocks_cost(deployment):
    def transform(role, score):
        if role == "evaluation":
            score[129] += .1
        return score
    deployment.calls["transform"] = transform
    with pytest.raises(ValueError, match="fixed score/decision gates"):
        cold.run(deployment.config, deployment.directory)
    report = _read_report(deployment)
    assert report["status"] == "mismatch" and not report["timing_allowed"]
    assert report["roles"]["evaluation"]["mismatched_score_rows"] == 1
    data = pd.read_csv(deployment.directory / "cold_replay_evaluation.csv")
    assert data.index[~data.within_tolerance].tolist() == [129]
    assert report["gates"]["E_decisions_unchanged_at_saved_threshold"]


def test_tolerated_score_change_that_flips_e_at_saved_threshold_blocks(deployment):
    _set_saved_e_score(deployment, 0, 1.)
    def transform(role, score):
        if role == "evaluation":
            score[0] += 1e-5
        return score
    deployment.calls["transform"] = transform
    with pytest.raises(ValueError, match="fixed score/decision gates"):
        cold.run(deployment.config, deployment.directory)
    report = _read_report(deployment)
    assert report["gates"]["full_C_and_E_scores_within_original_tolerance"]
    assert not report["gates"]["E_decisions_unchanged_at_saved_threshold"]
    assert report["roles"]["evaluation"]["saved_threshold_decision_changed_rows"] == 1


def test_c_recalibration_only_e_flip_blocks_even_when_saved_threshold_is_stable(deployment):
    _set_saved_e_score(deployment, 0, 1.00001)
    def transform(role, score):
        if role == "calibration":
            score[-1] += 2e-5
        return score
    deployment.calls["transform"] = transform
    with pytest.raises(ValueError, match="fixed score/decision gates"):
        cold.run(deployment.config, deployment.directory)
    report = _read_report(deployment)
    assert report["gates"]["full_C_and_E_scores_within_original_tolerance"]
    assert report["gates"]["E_decisions_unchanged_at_saved_threshold"]
    assert not report["gates"]["E_decisions_unchanged_after_C_recalibration"]
    assert report["roles"]["evaluation"]["recalibrated_decision_changed_rows"] == 1


def test_c_alert_changes_are_reported_without_inventing_extra_gate(deployment):
    def transform(role, score):
        if role == "calibration":
            score[-2] += 2e-5
        return score
    deployment.calls["transform"] = transform
    report = cold.run(deployment.config, deployment.directory)
    assert report["timing_allowed"]
    assert report["roles"]["calibration"]["saved_threshold_decision_changed_rows"] == 1
    assert report["calibration_decision_changes_are_reported_not_an_additional_gate"]


@pytest.mark.parametrize("kind,device", [("tabm", "cpu"), ("foundation", "cuda:0"), ("foundation", None)])
def test_wrong_model_or_build_device_cannot_load(deployment, kind, device):
    descriptor = deployment.descriptor.copy()
    descriptor.update(kind=kind, rebuild_device=device)
    write_json(deployment.directory / "deployment.json", descriptor)
    with pytest.raises(ValueError, match="explicit CPU foundation"):
        cold.run(deployment.config, deployment.directory)
    assert not deployment.calls["loads"]
    assert _read_report(deployment)["failed_stage"] == "validate_deployment"


def test_model_hash_failure_is_recorded_before_loading(deployment):
    deployment.recipe.write_bytes(b"changed recipe")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        cold.run(deployment.config, deployment.directory)
    assert not deployment.calls["loads"]
    assert not _read_report(deployment)["timing_allowed"]


def test_saved_score_ids_checked_before_loading(deployment):
    path = deployment.directory / "evaluation.csv"
    saved = pd.read_csv(path)
    saved.loc[129, "row_id"] = "incorrect-late-id"
    saved.to_csv(path, index=False)
    with pytest.raises(ValueError, match="IDs/labels/families"):
        cold.run(deployment.config, deployment.directory)
    assert not deployment.calls["loads"]
    assert _read_report(deployment)["failed_stage"] == "validate_evaluation"


def test_e_failure_keeps_completed_c_evidence_and_failure_report(deployment):
    deployment.calls["fail_e"] = True
    with pytest.raises(RuntimeError, match="synthetic E"):
        cold.run(deployment.config, deployment.directory)
    report = _read_report(deployment)
    assert report["failed_stage"] == "replay_evaluation"
    assert report["roles"]["calibration"]["rows"] == 100
    assert (deployment.directory / "cold_replay_calibration.csv").is_file()
    assert not (deployment.directory / "cold_replay_evaluation.csv").exists()
    assert not report["timing_allowed"]
