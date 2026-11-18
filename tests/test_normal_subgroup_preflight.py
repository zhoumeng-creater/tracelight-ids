"""Synthetic I/O backends only; no historical model is loaded or evaluated."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ids.normal_subgroup_training import preflight


class Preprocessor:
    def to_dict(self):
        return {"protocol_policy": "constant_unknown", "synthetic": True}

    def transform(self, frame):
        numeric = np.repeat(frame.row_id.to_numpy(dtype=np.float32)[:, None], 5, axis=1)
        categorical = np.zeros((len(frame), 6), dtype=np.int64)
        categorical[:, 0] = 5
        return numeric, categorical


def predictor(*, shift=0):
    return SimpleNamespace(preprocessor=Preprocessor(),
                           score=lambda frame: frame.row_id.to_numpy(dtype=np.float64) + shift)


@pytest.fixture
def frames():
    return {role: pd.DataFrame({"row_id": np.arange(n)}) for role, n in (("F", 5), ("V", 3))}


def test_cold_replay_loads_twice_and_preserves_original_bytes(tmp_path, frames):
    model = tmp_path / "original.pt"
    model.write_bytes(b"synthetic model bytes")
    calls = []

    def loader(path):
        calls.append(path)
        return predictor()

    scores, numeric, categorical, metadata = preflight.cold_replay(model, frames, loader=loader)
    assert len(calls) == 2
    assert model.read_bytes() == b"synthetic model bytes"
    np.testing.assert_array_equal(scores["F"], np.arange(5))
    assert numeric.dtype == np.float32 and categorical.shape == (5, 6)
    assert metadata["protocol_policy"] == "constant_unknown"


def test_replay_rejects_one_bit_score_change(tmp_path, frames):
    model = tmp_path / "original.pt"
    model.write_bytes(b"synthetic")
    count = 0

    def loader(path):
        nonlocal count
        count += 1
        p = predictor()
        if count == 2:
            p.score = lambda frame: np.nextafter(frame.row_id.to_numpy(dtype=np.float64), np.inf)
        return p

    with pytest.raises(ValueError, match="cold replay differs"):
        preflight.cold_replay(model, frames, loader=loader)


def test_replay_rejects_C_E_without_loading(tmp_path, frames):
    with pytest.raises(ValueError, match="F and V only"):
        preflight.cold_replay(tmp_path / "absent.pt", {**frames, "C": frames["F"]})


def test_replay_rejects_modified_original_artifact(tmp_path, frames):
    model = tmp_path / "original.pt"
    model.write_bytes(b"original")

    def loader(path):
        Path(path).write_bytes(b"changed")
        return predictor()

    with pytest.raises(ValueError, match="artifact changed"):
        preflight.cold_replay(model, frames, loader=loader)


def test_replay_rejects_nonfinite_scores(tmp_path, frames):
    model = tmp_path / "original.pt"
    model.write_bytes(b"original")
    p = predictor()
    p.score = lambda frame: np.full(len(frame), np.nan)
    with pytest.raises(ValueError, match="Invalid F endpoint"):
        preflight.cold_replay(model, frames, loader=lambda path: p)


def test_preflight_rejects_existing_output_before_any_input_access(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "status.json").write_text("incomplete")
    monkeypatch.setattr(preflight, "load_bound_inputs", lambda *a, **k: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="must be empty"):
        preflight.run(tmp_path / "archive", output)


def test_module_requires_an_explicit_mode(monkeypatch):
    from ids.normal_subgroup_training.__main__ import main
    monkeypatch.setattr("sys.argv", ["program", "--r30-archive", "unused", "--output", "unused"])
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 2


@pytest.fixture
def bound_synthetic_run(tmp_path, monkeypatch):
    frames = {}
    for role in ("F", "V"):
        n = 1040
        normal = np.tile(np.r_[np.zeros(512, dtype=int), np.ones(8, dtype=int)], 2)
        frame = pd.DataFrame({"source_id": [role] * n, "row_id": np.arange(n),
                              "label": normal, "profile": ["A"] * 520 + ["B"] * 520})
        for field in ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"):
            frame[field] = np.arange(n)
        frames[role] = frame
    model_dir = tmp_path / "archive"
    model_dir.mkdir()
    models = {}
    for seed in (17, 29, 43):
        path = model_dir / f"model{seed}.pt"
        path.write_bytes(b"synthetic model placeholder")
        models[f"protocol_blind_s{seed}"] = {"path": path.name}
    monkeypatch.setattr(preflight, "load_bound_inputs", lambda *a, **k: (frames, {"models": models}, {}))
    monkeypatch.setattr(preflight, "_runtime", lambda: {"kind": "synthetic_stub"})
    monkeypatch.setattr(preflight, "_load", lambda path: predictor())
    return model_dir, tmp_path / "output", frames


def test_preflight_writes_frozen_targets_without_a_training_ready_state(bound_synthetic_run):
    """Exercise actual binning/targets/serialization with synthetic model outputs."""
    model_dir, output, frames = bound_synthetic_run
    result = preflight.run(model_dir, output)
    assert result["status"] == "zero_update_preflight_complete"
    assert result["new_fits"] == result["optimizer_updates"] == 0
    assert result["training_ready"] is False and result["C_E_access"] is False
    for seed in (17, 29, 43):
        for filename in ("targets.npz", "reference_scores.npz"):
            with np.load(output / f"protocol_blind_s{seed}" / filename, allow_pickle=False) as arrays:
                payload = {key: arrays[key] for key in arrays.files}
                assert all(value.dtype.kind != "O" for value in payload.values())
                if filename == "targets.npz":
                    assert payload["transport_target"].shape == (1040,)
                    assert payload["group_ids"].dtype.kind == "U"
    assert json.loads((output / "status.json").read_text())["training_ready"] is False
    manifest_path = output / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"groups.json", "identities.json", "input_receipt.json"}
    expected.update(f"protocol_blind_s{seed}/{name}"
                    for seed in (17, 29, 43)
                    for name in ("targets.npz", "reference_scores.npz",
                                 "target_metadata.json", "preprocessing.json"))
    assert set(manifest["files"]) == expected
    assert set(manifest["excluded"]) == {"status.json", "output_manifest.json"}
    assert expected | set(manifest["excluded"]) == {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    for relative, entry in manifest["files"].items():
        contents = (output / relative).read_bytes()
        assert entry == {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}
    manifest_bytes = manifest_path.read_bytes()
    assert result["output_manifest"] == {
        "path": "output_manifest.json", "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "bytes": len(manifest_bytes)}
    assert json.loads((output / "status.json").read_text())["output_manifest"] == result["output_manifest"]


def test_formal_preflight_rejects_repeatable_but_inconsistent_same_input_anchors(
        bound_synthetic_run, monkeypatch):
    """Two cold loads can agree while duplicate inputs have conflicting anchors."""
    model_dir, output, frames = bound_synthetic_run

    class DuplicateInputs(Preprocessor):
        def transform(self, frame):
            numeric, categorical = super().transform(frame)
            numeric[1] = numeric[0]  # Two normal rows; no opposite-label collision.
            return numeric, categorical

    calls = []

    def loader(path):
        calls.append(path)
        result = predictor()
        result.preprocessor = DuplicateInputs()
        return result

    artifact = model_dir / "model17.pt"
    scores, numeric, _, _ = preflight.cold_replay(artifact, frames, loader=loader)
    assert len(calls) == 2  # Exact whole-endpoint replay itself succeeds.
    np.testing.assert_array_equal(numeric[0], numeric[1])
    assert scores["F"][0] != scores["F"][1]
    calls.clear()
    monkeypatch.setattr(preflight, "_load", loader)
    with pytest.raises(ValueError, match="inconsistent reference scores for identical model inputs"):
        preflight.run(model_dir, output)
    assert len(calls) == 2
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed" and status["training_ready"] is False
    assert "classes=1" in status["error"]
    assert not (output / "protocol_blind_s17").exists()
    assert not (output / "output_manifest.json").exists()


def test_runtime_mismatch_rejected_before_model_loading(monkeypatch):
    monkeypatch.setattr(preflight, "version", lambda package: "unexpected")
    with pytest.raises(ValueError, match="registered R30 runtime"):
        preflight._runtime()
