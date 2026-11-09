"""Synthetic replay diagnosis; no pretrained checkpoint or real input is opened."""
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.route_batch import foundation as f
from ids.route_batch import replay
from ids.route_batch.data import load_inputs
from ids.route_batch.preprocessing import raw_numeric
from ids.route_batch.protocol import default_settings, make_tasks
from ids.route_batch.runner import _write_scores


class FakeTabICL:
    """Device-dependent construction gives deterministic, distinguishable paths."""
    def __init__(self, **parameters):
        self.parameters = parameters
        self.model_ = SimpleNamespace(to=lambda device: None)

    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.ensemble_generator_ = SimpleNamespace(class_shuffles_={"none": [[0, 1]] * self.parameters["n_estimators"]})
        return self

    def _resolve_device(self):
        self.device_ = self.device

    def _build_inference_config(self):
        pass

    def _move_cache_to_device(self):
        pass

    def predict_proba(self, x):
        offset = .2 if self.parameters["device"].startswith("cuda") else 0.
        p = (1 / (1 + np.exp(-x[:, 0] - offset))).astype(np.float32)
        return np.column_stack([1 - p, p])


@pytest.fixture
def original_run(tmp_path, monkeypatch):
    monkeypatch.setattr(f, "_tabicl_class", lambda: (FakeTabICL, {"commit": "synthetic"}))
    root = tmp_path / "old-run"
    root.mkdir()
    checkpoint = tmp_path / "synthetic-NOT-PRETRAINED.ckpt"
    checkpoint.write_bytes(b"synthetic placeholder not a real model")
    schema = {"features": ["x"], "numeric": ["x"], "categorical": [], "label_column": "label",
              "id_column": "sample_id", "group_column": "communication_group", "family_column": "attack_family"}
    files = {}
    for role, n in (("support", 8), ("calibration", 100), ("evaluation", 130)):
        y = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        x = np.linspace(-1, 1 if role == "calibration" else 2, n)
        if role == "evaluation":
            x[0] = 100.  # API probability saturation retained by both paths.
        frame = pd.DataFrame({"x": x, "label": y, "sample_id": [f"{role}-{i}" for i in range(n)],
                              "communication_group": [f"{role}-group-{i}" for i in range(n)],
                              "attack_family": np.where(y, "Scanning", "Benign")})
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "rows": n, "sha256": sha256_file(path)}
    settings = default_settings()
    settings.update({"representations": [], "reference": [], "foundation": [
        {"id": "tabicl_official_1", "kind": "tabicl", "n_estimators": 1},
        {"id": "tabicl_official_8", "kind": "tabicl", "n_estimators": 8}]})
    spec = {"protocol": "tracelight-route-batch-v1", "data_kind": "synthetic", "settings": settings,
            "schema": schema, "files": files, "artifacts": {}, "tasks": make_tasks(settings),
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}}
    write_json(root / "protocol.json", spec)
    snapshot_files = {"src/ids/synthetic.py": "a" * 64}
    snapshot = {"files": snapshot_files, "source_sha256": canonical_hash(snapshot_files)}
    write_json(root / "source_snapshot.json", snapshot)
    write_json(root / "preflight.json", {"source": snapshot, "protocol_sha256": canonical_hash(spec)})
    inputs = load_inputs(root / "protocol.json")
    for task in spec["tasks"]:
        attempt = root / "tasks" / task["id"] / "attempt-001"
        attempt.mkdir(parents=True)
        model = f.fit_foundation("tabicl", raw_numeric(inputs["frames"]["support"], inputs["schema"]),
                                 inputs["frames"]["support"].label.to_numpy(), checkpoint_path=checkpoint,
                                 seed=settings["foundation_seed"], device=settings["device"], n_estimators=task["n_estimators"])
        model.save(attempt / "model.joblib")
        descriptor = {"kind": "foundation", "model_file": "model.joblib", "metadata": model.metadata,
                      "model_sha256": sha256_file(attempt / "model.joblib"), "score_contract": "registered log odds"}
        metrics = _write_scores(attempt, inputs, {role: model.predict_scores(raw_numeric(inputs["frames"][role], inputs["schema"]))
                                                for role in ("calibration", "evaluation")}, descriptor)
        write_json(attempt / "result.json", {"status": "measurement_failed", "task": task, "model": descriptor,
                                            "metrics": metrics, "cost": {"status": "failed"}})
    return SimpleNamespace(root=root, checkpoint=checkpoint, spec=spec, inputs=inputs)


def _attempt(original, task="tabicl_official_1"):
    return original.root / "tasks" / task / "attempt-001"


def _mock_subprocess(monkeypatch, commands):
    def run(command, **kwargs):
        commands.append(command)
        def value(name):
            return command[command.index(name) + 1]
        replay.replay_path_worker(value("--original-run"), value("--task"), value("--output"),
                                  build_path=value("--build-path"), checkpoint_path=value("--checkpoint"))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(replay.subprocess, "run", run)


def test_predict_outputs_preserves_probability_rounding_and_original_score_rule():
    probabilities = np.array([[1, 0], [0, 1], [.3, .7]], dtype=np.float32)
    model = f.FoundationModel("tabicl", SimpleNamespace(predict_proba=lambda x: probabilities), {},
                              np.ones((3, 1)), np.array([0, 1, 1]))
    out = model.predict_outputs(np.ones((3, 1)), diagnostics=True)
    np.testing.assert_array_equal(out["probabilities"], probabilities.astype(np.float64))
    p = np.clip(probabilities[:, 1].astype(np.float64), f.SCORE_EPSILON, 1 - f.SCORE_EPSILON)
    np.testing.assert_array_equal(out["scores"], np.log(p) - np.log1p(-p))
    np.testing.assert_array_equal(model.predict_scores(np.ones((3, 1))), out["scores"])
    assert out["diagnostics"]["api_probability_dtypes"] == ["float32"]
    assert out["diagnostics"]["raw_attack_probability_one"] == 1
    assert out["diagnostics"]["raw_attack_probability_zero"] == 1


def test_load_preserves_original_build_only_when_explicit(original_run):
    recipe = _attempt(original_run) / "model.joblib"
    default = f.load_foundation(recipe)
    preserved = f.load_foundation(recipe, preserve_build_device=True, diagnostics=True)
    assert default.metadata["build_device"] == "cpu"
    assert "rebuild_diagnostic" not in default.metadata
    assert preserved.metadata["build_device"] == "cuda:0"
    assert preserved.metadata["rebuild_diagnostic"]["preserve_build_device"]
    assert preserved.metadata["score_device"] == "cpu"
    with pytest.raises(ValueError, match="boolean"):
        f.load_foundation(recipe, preserve_build_device="yes")


@pytest.mark.parametrize("task", ["tabicl_official_1", "tabicl_official_8"])
def test_two_isolated_paths_retain_original_and_recalibrate_separately(original_run, tmp_path, monkeypatch, task):
    original_files = {p: p.read_bytes() for p in original_run.root.rglob("*") if p.is_file()}
    commands = []
    _mock_subprocess(monkeypatch, commands)
    report = replay.run_replay_diagnostic(original_run.root, task, tmp_path / "diagnostic")
    assert len(commands) == 2
    assert [c[c.index("--build-path") + 1] for c in commands] == ["original", "cpu"]
    assert report["status"] == "complete"
    assert report["paths"]["original"]["diagnostic_matches_original"]
    assert not report["paths"]["cpu"]["diagnostic_matches_original"]
    assert report["paths"]["original"]["report"]["probe"]["same_object_repeat"]["bitwise_equal"]
    assert report["paths"]["cpu"]["report"]["probe"]["same_object_repeat"]["bitwise_equal"]
    assert report["cross_build_path"]["evaluation"]["scores_cpu_minus_original_path"]["outside_original_tolerance"] > 0
    cpu = report["paths"]["cpu"]
    assert cpu["descriptor"]["rebuild_device"] == "cpu"
    assert cpu["report"]["roles"]["evaluation"]["old_threshold_decision_flips"] > 0
    probe = pd.read_csv(tmp_path / "diagnostic" / "cpu" / "probe.csv")
    assert len(probe) == 128
    assert "original_implied_clipped_attack_probability" in probe
    assert "repeat_raw_probability_attack" in probe
    score = pd.read_csv(tmp_path / "diagnostic" / "cpu" / "evaluation.csv")
    assert len(score) == 130 and "raw_probability_attack" in score
    assert all(p.read_bytes() == old for p, old in original_files.items())
    assert report["original_scores_replaced"] is False and report["tolerance_relaxed"] is False


def test_invalid_task_and_output_cannot_touch_old_run(original_run, tmp_path):
    with pytest.raises(ValueError, match="restricted"):
        replay.validate_replay_inputs(original_run.root, "tabm_protected_v2_w128_lr0.002_s17")
    with pytest.raises(ValueError, match="separate"):
        replay.run_replay_diagnostic(original_run.root, "tabicl_official_1", original_run.root / "diagnostic")
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        replay.run_replay_diagnostic(original_run.root, "tabicl_official_1", output)


def test_source_and_preflight_tampering_rejected_before_model_build(original_run, monkeypatch):
    monkeypatch.setattr(replay, "load_foundation", lambda *args, **kwargs: pytest.fail("model build attempted"))
    source = read_json(original_run.root / "source_snapshot.json")
    source["files"]["src/ids/synthetic.py"] = "b" * 64
    write_json(original_run.root / "source_snapshot.json", source)
    with pytest.raises(ValueError, match="internally inconsistent"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")
    source["source_sha256"] = canonical_hash(source["files"])
    write_json(original_run.root / "source_snapshot.json", source)
    with pytest.raises(ValueError, match="preflight"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")


def test_development_requires_exact_frozen_v1_source(original_run):
    spec = read_json(original_run.root / "protocol.json")
    spec["data_kind"] = "development"
    spec["settings"] = default_settings()
    spec["tasks"] = make_tasks(spec["settings"])
    write_json(original_run.root / "protocol.json", spec)
    with pytest.raises(ValueError, match="frozen round-17 v1 source"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")


def test_recipe_hash_checked_before_deserialization(original_run, monkeypatch):
    recipe = _attempt(original_run) / "model.joblib"
    recipe.write_bytes(b"changed-untrusted-joblib")
    monkeypatch.setattr(replay.joblib, "load", lambda *args: pytest.fail("unverified joblib read"))
    with pytest.raises(ValueError, match="SHA-256"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")


def test_recipe_support_checked_even_when_digest_is_registered(original_run):
    attempt = _attempt(original_run)
    recipe = attempt / "model.joblib"
    payload = joblib.load(recipe)
    payload["support_x"][0, 0] += 1
    joblib.dump(payload, recipe)
    result = read_json(attempt / "result.json")
    result["model"]["model_sha256"] = sha256_file(recipe)
    write_json(attempt / "result.json", result)
    with pytest.raises(ValueError, match="support arrays"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")


def test_checkpoint_may_move_but_must_keep_same_digest(original_run, tmp_path):
    relocated = tmp_path / "relocated.ckpt"
    relocated.write_bytes(original_run.checkpoint.read_bytes())
    context = replay.validate_replay_inputs(original_run.root, "tabicl_official_1", checkpoint_path=relocated)
    assert context["checkpoint"] == relocated
    relocated.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA-256"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1", checkpoint_path=relocated)


def test_score_identity_and_threshold_mismatches_rejected(original_run):
    path = _attempt(original_run) / "evaluation.csv"
    score = pd.read_csv(path, float_precision="round_trip")
    score.loc[0, "row_id"] = "wrong-row"
    score.to_csv(path, index=False)
    with pytest.raises(ValueError, match="score IDs"):
        replay.validate_replay_inputs(original_run.root, "tabicl_official_1")


def test_one_build_path_failure_does_not_hide_other_path(original_run, tmp_path, monkeypatch):
    original_loader = replay.load_foundation
    def loader(*args, **kwargs):
        if kwargs["preserve_build_device"]:
            raise RuntimeError("synthetic original-device failure")
        return original_loader(*args, **kwargs)
    monkeypatch.setattr(replay, "load_foundation", loader)
    _mock_subprocess(monkeypatch, [])
    report = replay.run_replay_diagnostic(original_run.root, "tabicl_official_1", tmp_path / "diagnostic")
    assert report["status"] == "partial"
    assert report["paths"]["original"]["status"] == "failed"
    assert report["paths"]["cpu"]["status"] == "complete"
    assert report["cross_build_path"] == {}


def test_same_model_repeat_drift_is_reported_without_relaxing_tolerance(original_run, tmp_path, monkeypatch):
    original_loader = replay.load_foundation
    def loader(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        original_predict = model.predict_outputs
        calls = 0
        def predict(*args, **kwargs):
            nonlocal calls
            calls += 1
            output = original_predict(*args, **kwargs)
            if calls == 4:
                # Synthetic scorer fault in just the repeated first-128 call.
                output["scores"] = output["scores"] + .01
            return output
        model.predict_outputs = predict
        return model
    monkeypatch.setattr(replay, "load_foundation", loader)
    result = replay.replay_path_worker(original_run.root, "tabicl_official_1", tmp_path / "cpu", build_path="cpu")
    assert result["status"] == "complete"
    report = read_json(tmp_path / "cpu" / "path_report.json")
    assert report["probe"]["same_object_repeat"]["outside_original_tolerance"] == 128
    assert report["probe"]["same_object_repeat"]["atol"] == 3e-5


def test_primary_C_then_E_precedes_state_dependent_probes_without_second_fit(original_run, tmp_path, monkeypatch):
    original_loader = replay.load_foundation
    loads, sizes = [], []
    def loader(*args, **kwargs):
        loads.append(kwargs)
        model = original_loader(*args, **kwargs)
        original_predict = model.predict_outputs
        def predict(x, **kwargs):
            sizes.append(len(x))
            outputs = original_predict(x, **kwargs)
            if len(sizes) > 2:
                # After initial C/E, calls visibly affect scores. Probes must
                # not contaminate the primary comparison with original C/E.
                outputs["scores"] = outputs["scores"] + .01 * len(sizes)
            return outputs
        model.predict_outputs = predict
        return model
    monkeypatch.setattr(replay, "load_foundation", loader)
    result = replay.replay_path_worker(original_run.root, "tabicl_official_1", tmp_path / "original", build_path="original")
    assert len(loads) == 1 and sizes == [100, 130, 128, 128]
    assert result["status"] == "complete" and result["diagnostic_matches_original"]
    report = read_json(tmp_path / "original" / "path_report.json")
    assert report["call_order"] == ["calibration", "evaluation", "evaluation_first128_probe1", "evaluation_first128_probe2"]
    assert report["probe"]["full_E_vs_later_probe"]["outside_original_tolerance"] == 128
    assert report["probe"]["same_object_repeat"]["outside_original_tolerance"] == 128
    probe = pd.read_csv(tmp_path / "original" / "probe.csv")
    np.testing.assert_allclose(probe.probe_minus_initial_E_score, .03)


def test_original_match_flag_requires_E_decisions_beyond_score_tolerance(original_run, tmp_path, monkeypatch):
    attempt = _attempt(original_run)
    saved = pd.read_csv(attempt / "evaluation.csv", float_precision="round_trip")
    registered = read_json(attempt / "threshold.json")
    at_threshold = np.flatnonzero(saved.score.to_numpy() == registered["threshold"])
    assert len(at_threshold)
    original_loader = replay.load_foundation
    def loader(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        original_predict = model.predict_outputs
        calls = 0
        def predict(x, **kwargs):
            nonlocal calls
            calls += 1
            outputs = original_predict(x, **kwargs)
            if calls == 2:
                outputs["scores"][at_threshold[0]] += 1e-5
            return outputs
        model.predict_outputs = predict
        return model
    monkeypatch.setattr(replay, "load_foundation", loader)
    result = replay.replay_path_worker(original_run.root, "tabicl_official_1", tmp_path / "original", build_path="original")
    assert result["status"] == "complete" and result["scores_match_original_within_tolerance"]
    assert not result["old_threshold_E_decisions_match"]
    assert not result["recalibrated_E_decisions_match"]
    assert not result["diagnostic_matches_original"]


def test_replay_tolerance_is_fixed_and_asymmetric_reference_side():
    ref = np.array([0., 100.])
    actual = np.array([3.1e-5, 100.00304])
    report = replay.score_difference(ref, actual)
    assert report["outside_original_tolerance"] == 2
    assert report["atol"] == 3e-5 and report["rtol"] == 3e-5
