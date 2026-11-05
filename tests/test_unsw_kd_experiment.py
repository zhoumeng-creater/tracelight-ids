"""Synthetic end-to-end execution and train-only input-boundary checks."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import ids.budget_transfer as transfer
import ids.unsw_kd_experiment as implementation
from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.models import FeaturePreprocessor, ModelBundle, _network, save_bundle
from ids.risk_research import _state, _state_hash
from ids.unsw_kd import ARMS, array_sha256, build_targets


SCHEMA = {"name": "unsw-kd-integration-synthetic",
          "features": ["x", "z", "cat"], "categorical": ["cat"], "discrete": ["cat"]}


def synthetic_context():
    rng = np.random.default_rng(27)
    frames = {}
    for role, n in (("fit", 24), ("reference", 16), ("constraint", 20)):
        y = np.tile([0, 1], n // 2)
        frames[role] = pd.DataFrame({
            "x": rng.normal(size=n), "z": rng.normal(size=n),
            "cat": np.tile(["left", "right"], n // 2), "label": y,
            "attack_cat": np.where(y == 1, "Attack", "Normal"),
            "sample_id": [f"{role}-{i}" for i in range(n)],
            "feature_hash": [f"{role}-pattern-{i}" for i in range(n)],
        })
    prep = FeaturePreprocessor.fit(frames["fit"], SCHEMA["features"], SCHEMA)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        net = _network(prep.encoded_dim, (7, 3))
    anchor = ModelBundle("mlp", prep, prep.selected_features, prep.encoded_dim, [],
                         network_state=_state(net), hidden_dims=(7, 3),
                         feature_schema=SCHEMA, seed=17)
    logits = np.linspace(-3, 3, len(frames["fit"]), dtype=np.float64)
    targets, audit, arrays = build_targets(frames["fit"], logits)
    protocol = copy.deepcopy(read_json(implementation.PROTOCOL))
    # Only private in-memory test contexts may alter the frozen production sizes.
    protocol["data"]["raw_fields"] = len(SCHEMA["features"])
    protocol["data"]["encoded_features"] = prep.encoded_dim
    protocol["data"]["preprocessing_semantic_sha256"] = canonical_hash(
        implementation.preprocessing_state(prep))
    protocol["data"]["roles"] = {
        role: {"rows": len(frame), "normal": int((frame.label == 0).sum()),
               "attack": int((frame.label == 1).sum())}
        for role, frame in frames.items()
    }
    protocol["student"].update({
        "hidden": [7, 3], "input_features": prep.encoded_dim,
        "parameters": sum(p.numel() for p in net.parameters()),
        "anchor_state_sha256": _state_hash(anchor.network_state),
    })
    n, steps, chunk = len(frames["fit"]), 2, 7
    rows_rc = sum(len(frames[r]) for r in ("reference", "constraint"))
    protocol["training"].update({
        "macro_steps": steps, "chunk_size": chunk, "endpoint": steps,
        "total_optimizer_updates": 4*steps,
        "total_training_row_forwards": 4*steps*n,
        "total_backward_calls": 4*steps*math.ceil(n/chunk),
        "total_endpoint_R_C_score_rows": 4*rows_rc,
        "total_shared_step0_plus_endpoints_R_C_score_rows": 5*rows_rc,
    })
    return {
        "protocol": protocol, "frames": frames, "anchor": anchor,
        "targets": targets, "arrays": arrays,
        "receipt": {"status": "synthetic_test_context", "target_audit": audit},
        "teacher_logits": logits,
    }


def test_synthetic_four_arm_execute_has_fixed_score_queries_and_complete_inventory(tmp_path, monkeypatch):
    context = synthetic_context()
    original_scores = implementation._scores
    queries = []

    def score_spy(bundle, frames, chunk):
        assert set(frames) == {"reference", "constraint"}
        queries.append(sum(len(frame) for frame in frames.values()))
        return original_scores(bundle, frames, chunk)

    def unexpected(*args, **kwargs):
        raise AssertionError("Executor must not read any additional data partition or benchmark")

    monkeypatch.setattr(implementation, "_scores", score_spy)
    monkeypatch.setattr(implementation, "_training_frames", unexpected)
    monkeypatch.setattr(implementation, "verify_development_inputs", unexpected)
    monkeypatch.setattr(implementation, "load_bundle", unexpected)
    monkeypatch.setattr(implementation, "benchmark_pair", unexpected)
    output = tmp_path / "complete"
    result = implementation._execute(
        context, output, device="cpu", measure_cpu=False, source={"files": {}})
    assert result["status"] == "unsw_kd_complete"
    assert result["student_fits"] == 4
    assert result["teacher_fits"] == result["warmup_fits"] == 0
    assert result["budget_passed"] and result["structure_passed"]
    assert not result["acceptance_passed"]
    assert not result["cost"]["measured"] and not result["cost"]["passed"]
    assert result["reserved_partitions_read"] == [] and not result["validation_loaded"]
    assert len(queries) == 5  # common anchor once, then one endpoint per arm
    assert sum(queries) == result["budget"]["R_C_scoring_rows"]
    assert not (output / "FAILURE.json").exists()
    for arm in ARMS:
        folder = output / "arms" / arm
        assert (folder / "model.joblib").is_file()
        history = read_json(folder / "history.json")
        streamed = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert history == streamed and [r["step"] for r in history] == [1, 2]
        report = read_json(folder / "run.json")
        assert report["endpoint_step"] == 2
        with np.load(folder / "scores.npz", allow_pickle=False) as scores:
            assert set(scores.files) == {"reference", "constraint"}
            assert all(scores[role].dtype == np.float64 for role in scores.files)
    assert len(list((output / "paired").glob("*.csv"))) == 7
    done = read_json(output / implementation.COMPLETE)
    actual = {p.relative_to(output).as_posix()
              for p in output.rglob("*") if p.is_file() and p.name != implementation.COMPLETE}
    assert set(done["files"]) == actual
    assert "targets.npz" in done["files"] and "step0/scores.npz" in done["files"]
    for name, digest in done["files"].items():
        assert sha256_file(output / name) == digest
    groups = pd.read_csv(output / "paired/K-H_constraint_feature_groups.csv")
    assert "net_true_positives" in groups


def test_failed_arm_keeps_streamed_history_and_previous_arm_without_complete(tmp_path, monkeypatch):
    context = synthetic_context()
    original_fit = implementation.fit_arm
    started = []

    def fail_K(*args, **kwargs):
        started.append(kwargs["arm"])
        if kwargs["arm"] == "K":
            kwargs["on_step"]({"step": 1, "objective_bce": .5, "partial_test_record": True})
            raise FloatingPointError("synthetic injected failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(implementation, "fit_arm", fail_K)
    output = tmp_path / "partial"
    with pytest.raises(FloatingPointError, match="synthetic injected"):
        implementation._execute(
            context, output, device="cpu", measure_cpu=False, source={"files": {}})
    assert started == ["H", "K"]
    failure = read_json(output / "FAILURE.json")
    assert failure["status"] == "technical_failure"
    assert failure["completed_arms"] == ["H"]
    assert failure["reserved_partitions_read"] == []
    assert (output / "arms/H/model.joblib").is_file()
    assert (output / "arms/H/history.json").is_file()
    partial = [json.loads(line) for line in
               (output / "arms/K/history.jsonl").read_text().splitlines()]
    assert partial == [{"step": 1, "objective_bce": .5, "partial_test_record": True}]
    assert not (output / implementation.COMPLETE).exists()
    assert not (output / "summary.json").exists()
    assert not (output / "arms/S").exists()


def test_existing_output_is_preserved_without_any_training(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")

    def unexpected(*args, **kwargs):
        raise AssertionError("Existing output must be rejected before training")

    monkeypatch.setattr(implementation, "fit_arm", unexpected)
    with pytest.raises(FileExistsError):
        implementation._execute(
            synthetic_context(), output, device="cpu", measure_cpu=False, source={"files": {}})
    assert marker.read_text() == "preserve"
    assert set(p.name for p in output.iterdir()) == {"keep.txt"}


def test_bound_inputs_decodes_only_train_roles_and_F_teacher_key(tmp_path, monkeypatch):
    context = synthetic_context()
    p, frames, anchor = context["protocol"], context["frames"], context["anchor"]
    screen, data = tmp_path / "screen", tmp_path / "prepared"
    screen.mkdir()
    data.mkdir()
    manifest = {"data_kind": "unit_test_fixture", "marker": "train-only boundary"}
    p["data"]["source_screen_commit"] = "synthetic-source"
    write_json(screen / "plan.json", {"binding": {
        "source": {"commit": p["data"]["source_screen_commit"]}, "data": manifest}})
    for role, frame in frames.items():
        frame.loc[:, ["sample_id", "feature_hash", "label", "attack_cat"]].to_parquet(
            screen / f"{role}-identity.parquet", index=False)
    anchor_path = screen / p["student"]["anchor"]
    save_bundle(anchor, anchor_path)
    scores_path = screen / p["teacher"]["scores"]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    # A poison object member would fail allow_pickle=False if it were decoded.
    np.savez_compressed(scores_path, fit=context["teacher_logits"],
                        validation=np.asarray([object()], dtype=object))
    complete = screen / "TRANSFER_SCREEN_COMPLETE.json"
    write_json(complete, {"binding_sha256": "fixture-bound"})
    p["data"]["screen_complete_sha256"] = sha256_file(complete)
    audit = context["receipt"]["target_audit"]
    p["teacher"]["q_le_f8_sha256"] = audit["teacher_q_sha256"]
    p["permutation"]["q_perm_le_f8_sha256"] = audit["q_perm_sha256"]
    p["permutation"]["donor_le_i8_sha256"] = audit["donor_i8_sha256"]
    binding = {
        "screen_binding_sha256": "fixture-bound",
        "files_sha256": {f.relative_to(screen).as_posix(): sha256_file(f)
                         for f in screen.rglob("*") if f.is_file()},
        "identities": {role: {"ordered_label_i8_sha256": array_sha256(f.label, "<i8")}
                       for role, f in frames.items()},
        "score_sha256_little_endian_f8": {"fit": array_sha256(context["teacher_logits"])},
    }
    binding_path = tmp_path / "binding.json"
    write_json(binding_path, binding)
    p["data"]["binding"] = str(binding_path)
    protocol_path = tmp_path / "synthetic-private-protocol.json"
    write_json(protocol_path, p)
    monkeypatch.setattr(implementation, "_protocol", lambda path=None: (p, protocol_path))
    monkeypatch.setattr(implementation, "verify_transfer_screen",
                        lambda path: {"binding_sha256": "fixture-bound"})
    verified, loaded, keys = [], [], []

    def verify_spy(path, *, splits):
        verified.append(splits)
        assert splits == ("train",)
        return manifest

    def partition_spy(path, split):
        loaded.append(split)
        assert split == "train"
        return pd.concat(list(frames.values()), ignore_index=True)

    real_np_load = np.load

    class FOnlyArchive:
        def __enter__(self):
            self.archive = real_np_load(scores_path, allow_pickle=False)
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            keys.append(key)
            assert key == "fit"
            return self.archive[key]

    def np_load_spy(path, **kwargs):
        assert Path(path) == scores_path
        assert kwargs == {"allow_pickle": False}
        return FOnlyArchive()

    monkeypatch.setattr(implementation, "verify_development_inputs", verify_spy)
    monkeypatch.setattr(transfer, "load_partition", partition_spy)
    monkeypatch.setattr(implementation.np, "load", np_load_spy)
    result = implementation._bound_inputs(screen, data)
    assert verified == [("train",)] and loaded == ["train"] and keys == ["fit"]
    assert set(result["frames"]) == {"fit", "reference", "constraint"}
    assert result["receipt"]["new_inference_rows"] == 0
    assert result["receipt"]["reserved_partitions_read"] == []
    assert result["receipt"]["target_audit"]["teacher_q_sha256"] == audit["teacher_q_sha256"]
