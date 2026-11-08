"""Synthetic-only registration, read-only preflight and archive integrity checks."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.calibration import apply_threshold, register_threshold
from ids.context_cache.csv_workflow import V2_POLICIES
from ids.context_cache.scanning_protocol import (
    frozen_template, load_scanning_protocol, pack_scanning_results,
    preflight_scanning, register_scanning_protocol, verify_scanning_code_freeze, _verify_runtime,
)


@pytest.fixture
def synthetic_protocol(tmp_path, monkeypatch):
    import ids.context_cache.scanning_protocol as module
    monkeypatch.setattr(module, "require_pinned_preprocessing", lambda: (None, {"synthetic_test": True}))
    monkeypatch.setattr(module, "environment_snapshot", lambda: {"torch": {"cuda_available": False}})
    spec = frozen_template()
    spec.update(data_kind="synthetic", build_device="cpu", center=0.0, scale=1.0, seed=847,
                query_batch_size=8, baseline_model_sha256="4" * 64)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    frames = {}
    for role, n in (("support", 12), ("calibration", 99), ("evaluation", 24)):
        rng = np.random.default_rng(n)
        frame = pd.DataFrame(rng.normal(size=(n, 48)), columns=spec["schema"]["features"])
        frame["L4_SRC_PORT"] = np.arange(n) + 1024
        frame["L4_DST_PORT"] = 443
        frame["PROTOCOL"] = 6
        frame["L7_PROTO"] = 91
        frame["label"] = 0 if role == "calibration" else np.arange(n) % 2
        frame["sample_id"] = [f"{role}-{i}" for i in range(n)]
        frame["communication_group"] = [f"{role}-group-{i}" for i in range(n)]
        frame["attack_family"] = np.where(frame["label"], "Scanning", "Benign")
        path = prepared / f"{role}.csv"
        frame.to_csv(path, index=False)
        spec["files"][role] = {"path": str(path), "sha256": sha256_file(path), "rows": n}
        frames[role] = frame
    for role in ("checkpoint", "preprocessor"):
        path = tmp_path / f"{role}.bin"
        path.write_bytes(f"synthetic non-loadable {role}".encode())
        target = spec if role == "checkpoint" else spec["artifacts"]
        target[role] = {"path": str(path), "sha256": sha256_file(path)}
    receipt = {"format": "tracelight-nf-ton-v3-preparation-v2", "data_kind": "synthetic",
               "schema": spec["schema"], "seed": 847, "files": copy.deepcopy(spec["files"]),
               **V2_POLICIES, "source": {"official_release_verified": False}}
    path = prepared / "preparation.json"
    write_json(path, receipt)
    spec["artifacts"]["preparation"] = {"path": str(path), "sha256": sha256_file(path)}
    c_scores = np.linspace(-1, 1, 99)
    registration = register_threshold(c_scores, .01, model_sha256=spec["baseline_model_sha256"],
                                     preprocessing_sha256=spec["artifacts"]["preprocessor"]["sha256"],
                                     calibration_sha256=spec["files"]["calibration"]["sha256"])
    path = tmp_path / "threshold.json"
    write_json(path, registration)
    spec["artifacts"]["baseline_threshold"] = {"path": str(path), "sha256": sha256_file(path)}
    c = pd.DataFrame({"row_id": frames["calibration"]["sample_id"], "label": 0,
                      "score": c_scores, "predicted_attack": apply_threshold(c_scores, registration)})
    e_scores = np.linspace(-2, 2, 24)
    e = pd.DataFrame({"row_id": frames["evaluation"]["sample_id"],
                      "label": frames["evaluation"]["label"], "full_score": e_scores,
                      "full_predicted": apply_threshold(e_scores, registration),
                      "legacy_full_score": e_scores + 1, "legacy_full_predicted": e_scores + 1 > 1,
                      "attack_family": frames["evaluation"]["attack_family"]})
    for role, frame in (("baseline_calibration", c), ("baseline_evaluation", e)):
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        spec["artifacts"][role] = {"path": str(path), "sha256": sha256_file(path)}
    config = tmp_path / "protocol.json"
    write_json(config, spec)
    return config, spec


def rebind(config, spec, role):
    spec["artifacts"][role]["sha256"] = sha256_file(spec["artifacts"][role]["path"])
    write_json(config, spec)


def test_preflight_uses_no_model_or_pickle_loading(synthetic_protocol, monkeypatch):
    from ids.context_cache.preprocessing import CSVPreprocessor
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError("model/preprocessing loading or fitting is forbidden in preflight")
    monkeypatch.setattr(CSVPreprocessor, "load", forbidden)
    monkeypatch.setattr(CSVPreprocessor, "fit", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    config, _ = synthetic_protocol
    spec, schema, paths, frames, audit = preflight_scanning(config)
    assert spec["data_kind"] == "synthetic" and len(schema.features) == 48
    assert len(paths) == 9 and len(frames["calibration"]) == 99
    assert audit["baseline_row_order_labels_predictions_threshold_verified"]
    assert audit["model_loads"] == audit["preprocessor_loads"] == audit["preprocessor_fits"] == 0


@pytest.mark.parametrize("field,value", [("mapping_seed", 1), ("alpha", .02), ("width", 47),
                                         ("score_atol", 1), ("arms", ["baseline"]), ("extra", True)])
def test_frozen_intervention_parameters_rejected(synthetic_protocol, field, value):
    config, spec = synthetic_protocol
    spec[field] = value
    write_json(config, spec)
    with pytest.raises(ValueError):
        load_scanning_protocol(config)


def test_development_cannot_use_synthetic_hashes(synthetic_protocol):
    config, spec = synthetic_protocol
    spec.update(data_kind="development", build_device="cuda", seed=20261107,
                query_batch_size=128, center=35989.91845703125, scale=21268.044830665465)
    write_json(config, spec)
    with pytest.raises(ValueError, match="development"):
        load_scanning_protocol(config)


def test_registered_file_tamper_rejected(synthetic_protocol):
    config, spec = synthetic_protocol
    with Path(spec["files"]["support"]["path"]).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch for support"):
        preflight_scanning(config)


def test_archived_order_mismatch_rejected(synthetic_protocol):
    config, spec = synthetic_protocol
    path = Path(spec["artifacts"]["baseline_evaluation"]["path"])
    frame = pd.read_csv(path)
    frame.iloc[::-1].to_csv(path, index=False)
    rebind(config, spec, "baseline_evaluation")
    with pytest.raises(ValueError, match="row order"):
        preflight_scanning(config)


def test_threshold_binding_mismatch_rejected(synthetic_protocol):
    config, spec = synthetic_protocol
    path = Path(spec["artifacts"]["baseline_threshold"]["path"])
    registration = read_json(path)
    registration["binding"]["preprocessing_sha256"] = "f" * 64
    write_json(path, registration)
    rebind(config, spec, "baseline_threshold")
    with pytest.raises(ValueError, match="threshold rank, scores hash or bindings"):
        preflight_scanning(config)


def test_decision_mismatch_rejected(synthetic_protocol):
    config, spec = synthetic_protocol
    path = Path(spec["artifacts"]["baseline_calibration"]["path"])
    frame = pd.read_csv(path)
    frame.loc[0, "predicted_attack"] = True
    frame.to_csv(path, index=False)
    rebind(config, spec, "baseline_calibration")
    with pytest.raises(ValueError, match="decisions differ"):
        preflight_scanning(config)


def test_preparation_hash_binding_rejected(synthetic_protocol):
    config, spec = synthetic_protocol
    path = Path(spec["artifacts"]["preparation"]["path"])
    receipt = read_json(path)
    receipt["files"]["support"]["sha256"] = "f" * 64
    write_json(path, receipt)
    rebind(config, spec, "preparation")
    with pytest.raises(ValueError, match="preparation binding"):
        preflight_scanning(config)


def test_register_cannot_accept_different_data(tmp_path):
    (tmp_path / "prepared").mkdir()
    (tmp_path / "prepared" / "support.csv").write_text("synthetic\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        register_scanning_protocol(round14_root=tmp_path, checkpoint=tmp_path / "weights.ckpt",
                                   output=tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_pack_failed_run_roundtrip(tmp_path):
    root = tmp_path / "run"
    write_json(root / "comparison" / "state.json", {"status": "failed", "error": "synthetic failure"})
    (root / "log.txt").write_text("preserve failure evidence\n")
    archive = tmp_path / "result.zip"
    report = pack_scanning_results(root, archive)
    assert report["readback_verified"] and report["run_status"]["status"] == "failed"
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("run/archive_manifest.json"))
        assert manifest["files"]["log.txt"]["sha256"] == sha256_file(root / "log.txt")
        assert z.read("run/log.txt") == (root / "log.txt").read_bytes()


def test_pack_completed_run_verifies_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "run"
    write_json(root / "comparison" / "state.json", {"status": "complete"})
    calls = []
    module = types.ModuleType("ids.context_cache.scanning_workflow")
    module.verify_scanning_result = lambda path: calls.append(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pack_scanning_results(root, tmp_path / "result.zip")
    assert calls == [root / "comparison"]


def test_pack_detects_change_after_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    payload = root / "log.txt"
    payload.write_text("original evidence")
    original_write = zipfile.ZipFile.write
    def changing_write(self, filename, *args, **kwargs):
        if Path(filename) == payload:
            payload.write_text("modified after manifest")
        return original_write(self, filename, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "write", changing_write)
    with pytest.raises(ValueError, match="readback hash differs"):
        pack_scanning_results(root, tmp_path / "result.zip")


@pytest.mark.parametrize("name", ["weights.ckpt", "nested.zip", "NF-ToN-IoT-v3.csv"])
def test_pack_rejects_duplicate_raw_or_checkpoint(tmp_path, name):
    root = tmp_path / "run"
    root.mkdir()
    (root / name).write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="raw/checkpoint"):
        pack_scanning_results(root, tmp_path / "result.zip")


def test_pack_rejects_output_inside_run(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        pack_scanning_results(tmp_path, tmp_path / "nested.zip")


def test_checked_in_template_matches_frozen_constants():
    path = Path(__file__).resolve().parents[1] / "configs/context-scanning-mechanism-v1.json"
    assert read_json(path) == frozen_template()


@pytest.fixture
def frozen_code_tree(tmp_path):
    names = ["src/ids/context_cache/scanning_protocol.py", "src/ids/common.py",
             "configs/context-scanning-mechanism-v1.json", "pyproject.toml", "requirements-lock.txt",
             "scripts/RUN_SCANNING_MECHANISM.ps1"]
    files = {}
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic release\r\n")
        files[name] = hashlib.sha256(b"synthetic release\n").hexdigest()
    write_json(tmp_path / "configs/scanning-mechanism-code-freeze-v1.json",
               {"format": "tracelight-scanning-code-freeze-v1", "files": files})
    return tmp_path


def test_code_freeze_accepts_crlf_and_lf(frozen_code_tree):
    report = verify_scanning_code_freeze(frozen_code_tree)
    (frozen_code_tree / "src/ids/common.py").write_bytes(b"synthetic release\n")
    assert verify_scanning_code_freeze(frozen_code_tree) == report


def test_code_freeze_rejects_changed_source(frozen_code_tree):
    (frozen_code_tree / "src/ids/common.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="source SHA mismatch"):
        verify_scanning_code_freeze(frozen_code_tree)


def test_code_freeze_rejects_unregistered_module(frozen_code_tree):
    (frozen_code_tree / "src/ids/context_cache/unregistered.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="omits required files"):
        verify_scanning_code_freeze(frozen_code_tree)


def test_code_freeze_rejects_traversal(frozen_code_tree):
    path = frozen_code_tree / "configs/scanning-mechanism-code-freeze-v1.json"
    manifest = read_json(path)
    manifest["files"]["../external.py"] = "0" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="external or self-referential"):
        verify_scanning_code_freeze(frozen_code_tree)


@pytest.mark.parametrize("change", [None, "python", "torch", "numpy", "pandas", "scikit-learn", "joblib", "cuda"])
def test_development_runtime_frozen(change):
    environment = {"python": "3.12.3", "packages": {"torch": "2.6.0+cu126", "numpy": "1.26.4",
                   "pandas": "2.2.2", "scikit-learn": "1.5.1", "joblib": "1.4.2"},
                   "torch": {"cuda_runtime": "12.6", "cuda_available": True}}
    if change is None:
        _verify_runtime(environment)
        return
    if change == "python":
        environment["python"] = "3.11.9"
    elif change == "cuda":
        environment["torch"]["cuda_runtime"] = "12.8"
    else:
        environment["packages"][change] = "different"
    with pytest.raises(RuntimeError, match="frozen"):
        _verify_runtime(environment)
