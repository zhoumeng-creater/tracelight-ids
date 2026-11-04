"""Synthetic end-to-end evidence for the rebuilt budgeted transfer study.

The V3 source is actually produced and verified, not replaced by mocked identity
metadata. All fitting here uses generated toy CSVs, never real prepared data.
"""
from __future__ import annotations

import copy
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

import ids.budget_transfer as transfer
from ids.common import read_json, sha256_file
from ids.data import make_synthetic_dataset, prepare_dataset
from ids.models import load_bundle
from ids.risk_research import run_risk_research, verify_risk_research

REPO = Path(__file__).resolve().parents[1]
TEST_SOURCE = {"source_sha256": "b" * 64, "files": {}}


def test_paired_report_keeps_recoveries_losses_and_false_alarm_changes_separate(tmp_path):
    identity = pd.DataFrame({"sample_id": [str(i) for i in range(6)], "feature_hash": [str(i) for i in range(6)],
        "label": [0, 0, 1, 1, 1, 1], "attack_cat": ["Normal", "Normal", "A", "A", "B", "B"]})
    runs = []
    for method, alarm in (("direct_rank_tail", [1, 0, 1, 0, 1, 0]), ("direct_pauc", [0, 1, 0, 1, 1, 0])):
        folder = tmp_path / method
        folder.mkdir()
        frame = identity.copy()
        frame["alarm"] = np.asarray(alarm, dtype=bool)
        frame.to_parquet(folder / "validation-predictions.parquet", index=False)
        runs.append({"seed": 17, "budget": "all", "method": method, "relative_dir": method,
            "V_fpr_budget_pass": False, "validation": {"tpr": .5},
            "validation_rank": {"partial_auc_standardized": .5}})
    row, = transfer._paired_evidence(tmp_path, runs)
    assert row["attack_recovered_tp"] == row["attack_new_lost_tp"] == 1
    assert row["normal_new_fp"] == row["normal_repaired_fp"] == 1
    assert row["delta_tp"] == row["delta_fp"] == 0
    assert row["families"]["A"] == {"support": 2, "recovered_tp": 1, "new_lost_tp": 1}
    assert row["candidate_fpr_pass"] is False and row["control_fpr_pass"] is False


def config():
    return yaml.safe_load((REPO / "configs/budget-transfer-smoke.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_template(tmp_path_factory):
    """Real tiny V3 producer; stable test fingerprint while source is edited."""
    import ids.risk_research as risk
    base = tmp_path_factory.mktemp("budget-transfer-source")
    raw, data, v3 = base / "raw", base / "data", base / "v3"
    make_synthetic_dataset(raw, n_train=900, n_test=120)
    prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv", data,
                    data_kind="synthetic")
    source_config = yaml.safe_load((REPO / "configs/risk-v3-smoke.yaml").read_text(encoding="utf-8"))
    source_config.update(conditions=["hard"], max_fpr=.25)
    source_config["anchor"]["hyperparameters"].update(hidden_sizes=[2, 2], epochs=1, patience=1)
    source_config["teacher"]["hyperparameters"].update(max_iter=2, max_leaf_nodes=3)
    source_config["optimization"].update(macro_steps=1, attack_loss_limit=1.)
    with patch.object(risk, "code_fingerprint", lambda: copy.deepcopy(TEST_SOURCE)):
        run_risk_research(data, source_config, v3, device="cpu")
    verify_risk_research(v3)
    # Reserved inputs are deliberately unavailable throughout the new study.
    (data / "calibration.parquet").unlink()
    (data / "test.parquet").unlink()
    return data, v3


@pytest.fixture
def source(source_template, tmp_path, monkeypatch):
    data, v3 = tmp_path / "data", tmp_path / "v3"
    shutil.copytree(source_template[0], data)
    shutil.copytree(source_template[1], v3)
    monkeypatch.setattr(transfer, "code_fingerprint", lambda: copy.deepcopy(TEST_SOURCE))
    return data, v3


def _gate(monkeypatch, qualified):
    original = transfer.qualification

    def forced(*args):
        result = original(*args)
        return {**result, "qualified": qualified, "synthetic_test_forced_gate": True}

    monkeypatch.setattr(transfer, "qualification", forced)


def _source_hashes(root):
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def test_all_methods_fit_with_frozen_V_boundary_matched_teacher_and_common_initial_state(source, tmp_path, monkeypatch):
    import ids.tail_rank_kd as rank
    data, v3 = source
    output = tmp_path / "transfer"
    cfg = config()
    cfg["macro_steps"] = 2  # Check that the null teacher is fixed across updates.
    _gate(monkeypatch, True)
    source_before = _source_hashes(v3)
    fit_ids = pd.read_parquet(v3 / "fit-identity.parquet").sample_id.tolist()
    all_features = read_json(data / "manifest.json").get("feature_schema")
    if all_features is None:
        from ids.data import FEATURES
        all_features = {"features": list(FEATURES)}
    reads, teacher_calls, losses, choices = [], [], {}, {}
    load = transfer.load_partition
    train = transfer.train_model
    fit = transfer._fit_method
    select = transfer._selection
    raw_kl = transfer.raw_bernoulli_kl_rows
    channel = rank.binary_channel_rank_rows
    sample_rank = rank.sample_same_class_pairs
    pauc = transfer.pairwise_pauc_rows
    counts = {"raw_kl": 0, "channel": 0, "pauc": 0}

    def guarded_load(path, split):
        reads.append(split)
        assert split in {"train", "validation"}, "reserved input accessed"
        if split == "validation":
            frozen = read_json(output / "FROZEN_SELECTION.json")
            assert len(frozen["runs"]) == 2 * len(transfer.METHODS) - 1
            assert not frozen["validation_used"]
            assert all("validation" not in row for row in frozen["runs"])
            assert len(frozen["auxiliary"]) == 6
            assert all("validation" not in row for row in frozen["auxiliary"])
        return load(path, split)

    def teacher_fit(training, validation, model_config, seed, **kwargs):
        assert training is validation
        assert training.sample_id.tolist() == fit_ids
        schema = kwargs["feature_schema"]
        teacher_calls.append(copy.deepcopy(schema))
        assert model_config["hyperparameters"]["early_stopping"] is False
        return train(training, validation, model_config, seed, **kwargs)

    def checked_selection(bundle, frames, setting):
        assert "validation" not in frames
        assert not (output / "FROZEN_SELECTION.json").exists()
        return select(bundle, frames, setting)

    def tracked_fit(anchor, frames, setting, method, teacher, seed, device, folder):
        budget = folder.name.split("-", 1)[0]
        active_fit["key"] = (budget, method)
        choices[(budget, method)] = id(teacher)
        result = fit(anchor, frames, setting, method, teacher, seed, device, folder)
        losses[(budget, method)] = read_json(folder / "history.json")[1]["losses"]
        return result

    def tracked_kl(*args, **kwargs):
        counts["raw_kl"] += 1
        return raw_kl(*args, **kwargs)

    def tracked_channel(*args, **kwargs):
        counts["channel"] += 1
        return channel(*args, **kwargs)

    def tracked_pauc(*args, **kwargs):
        counts["pauc"] += 1
        return pauc(*args, **kwargs)

    rank_calls, rank_teachers, active_fit = [], {}, {}

    def tracked_rank(*args, **kwargs):
        rank_calls.append((kwargs["local"], kwargs["shuffle_within_class"]))
        rank_teachers.setdefault(active_fit["key"], []).append(np.asarray(args[0]).copy())
        return sample_rank(*args, **kwargs)

    monkeypatch.setattr(transfer, "load_partition", guarded_load)
    monkeypatch.setattr(transfer, "train_model", teacher_fit)
    monkeypatch.setattr(transfer, "_selection", checked_selection)
    monkeypatch.setattr(transfer, "_fit_method", tracked_fit)
    monkeypatch.setattr(transfer, "raw_bernoulli_kl_rows", tracked_kl)
    monkeypatch.setattr(transfer, "pairwise_pauc_rows", tracked_pauc)
    monkeypatch.setattr(rank, "binary_channel_rank_rows", tracked_channel)
    monkeypatch.setattr(rank, "sample_same_class_pairs", tracked_rank)
    done = transfer.run_transfer_research(v3, data, cfg, output, device="cpu")
    assert reads == ["train", "validation"]
    assert done["n_teacher_fits"] == 2 and done["n_warmup_fits"] == 4
    assert done["n_student_fits"] == 2 * len(transfer.METHODS) - 1 == 17
    assert done["reserved_partitions_read"] == []
    assert _source_hashes(v3) == source_before
    assert not (data / "calibration.parquet").exists() and not (data / "test.parquet").exists()
    assert len(teacher_calls) == 2
    assert teacher_calls[0]["features"] == all_features["features"]
    selected = read_json(output / "feature-selection.json")["selected_features"]
    assert len(selected) == cfg["reduced_k"]
    assert teacher_calls[1]["features"] == selected
    summary = read_json(output / "summary.json")
    assert summary["status"] == "development_only_no_winner_promoted"
    assert summary["independent_confirmation"] is False
    assert {row["method"] for row in summary["runs"]} == set(transfer.METHODS)
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["method"] == "direct_rank_matched"
    frozen_hash = sha256_file(output / "FROZEN_SELECTION.json")
    auxiliary = read_json(output / "auxiliary-evaluations.json")
    assert len(auxiliary) == 6
    assert {row["kind"] for row in auxiliary} == {"teacher", "hard_warmup"}
    frozen_aux = {row["relative_dir"]: row for row in read_json(output / "FROZEN_SELECTION.json")["auxiliary"]}
    for row in auxiliary:
        assert row["selection"] == frozen_aux[row["relative_dir"]]["selection"]
        assert "validation" in row and "validation_rank" in row
        pred = pd.read_parquet(output / row["relative_dir"] / "validation-predictions.parquet")
        np.testing.assert_array_equal(pred.alarm.to_numpy(),
            pred.raw_score.to_numpy(dtype=np.float64) >= row["selection"]["working_point"]["threshold"])
    for budget in ("all", "reduced"):
        runs = [r for r in summary["runs"] if r["budget"] == budget]
        warm = load_bundle(output / f"warmups/{budget}-small-s17/model.joblib")
        assert {r["common_initial_state_sha256"] for r in runs} == {transfer._state_hash(warm.network_state)}
        assert all(r["frozen_selection_sha256"] == frozen_hash for r in runs)
        assert all(r["validation_used_for_selection"] is False for r in runs)
        for r in runs:
            folder = output / r["relative_dir"]
            prediction = pd.read_parquet(folder / "validation-predictions.parquet")
            np.testing.assert_array_equal(prediction.alarm.to_numpy(),
                prediction.raw_score.to_numpy(dtype=np.float64) >= r["selection"]["working_point"]["threshold"])
            bundle = load_bundle(folder / "model.joblib")
            assert bundle.selected_features == (all_features["features"] if budget == "all" else selected)
            if r["method"].startswith("direct_rank_"):
                assert "rank_sampling" in losses[(budget, r["method"])]
            if r["method"].startswith("direct"):
                assert losses[(budget, r["method"])]["pauc"] > 0
            assert "gradients" in losses[(budget, r["method"])]
    assert counts["raw_kl"] > 0 and counts["channel"] > 0 and counts["pauc"] > 0
    assert (False, False) in rank_calls and (True, False) in rank_calls
    assert not any(shuffle for _, shuffle in rank_calls), "null mapping must be frozen outside the step sampler"
    with np.load(output / "teachers/all-s17/training-scores.npz") as arrays:
        original_teacher = arrays["fit"].copy()
    y = transfer._training_frames(v3, data)["fit"].label.to_numpy()
    for (budget, method), teachers in rank_teachers.items():
        assert len(teachers) == 2
        np.testing.assert_array_equal(teachers[0], teachers[1])
        if method == "direct_rank_shuffled":
            assert not np.array_equal(teachers[0], original_teacher)
            for label in (0, 1):
                np.testing.assert_array_equal(np.sort(teachers[0][y == label]), np.sort(original_teacher[y == label]))
        elif method != "direct_rank_matched":
            np.testing.assert_array_equal(teachers[0], original_teacher)
    nulls = [row for row in summary["runs"] if row["method"] == "direct_rank_shuffled"]
    assert len({row["fixed_teacher_permutation_sha256"] for row in nulls}) == 1
    assert nulls[0]["fixed_teacher_permutation_sha256"] is not None
    assert choices[("reduced", "direct_rank_matched")] != choices[("reduced", "direct_rank_tail")]
    assert transfer.verify_transfer(output) == done


def test_negative_qualification_preserves_all_registered_paired_comparisons(source, tmp_path, monkeypatch):
    data, v3 = source
    output = tmp_path / "negative"
    _gate(monkeypatch, False)
    fit, seen = transfer._fit_method, []

    def track(*args):
        method = args[3]
        assert method in transfer.METHODS
        seen.append(method)
        return fit(*args)

    monkeypatch.setattr(transfer, "_fit_method", track)
    done = transfer.run_transfer_research(v3, data, config(), output, device="cpu")
    assert done["n_student_fits"] == 17
    assert set(seen) == set(transfer.METHODS)
    summary = read_json(output / "summary.json")
    assert len(summary["skipped"]) == 1
    assert summary["skipped"][0]["method"] == "direct_rank_matched"
    assert not any(q["qualified"] for q in summary["qualification"])
    assert len(summary["runs"]) == 17


def test_complete_only_resume_binds_config_source_data_device_and_detects_tampering(source, tmp_path, monkeypatch):
    data, v3 = source
    output = tmp_path / "bound"
    cfg = config()
    _gate(monkeypatch, False)
    result = transfer.run_transfer_research(v3, data, cfg, output, device="cpu")
    monkeypatch.setattr(transfer, "_training_frames", lambda *args: pytest.fail("resume reloaded training rows"))
    assert transfer.run_transfer_research(v3, data, cfg, output, device="cpu", resume=True) == result
    with pytest.raises(FileExistsError, match="existing work"):
        transfer.run_transfer_research(v3, data, cfg, output, device="cpu")
    changed = copy.deepcopy(cfg)
    changed["learning_rate"] *= 2
    with pytest.raises(ValueError, match="differs"):
        transfer.run_transfer_research(v3, data, changed, output, device="cpu", resume=True)
    with pytest.raises(ValueError, match="differs"):
        transfer.run_transfer_research(v3, data, cfg, output, device="cuda", resume=True)
    monkeypatch.setattr(transfer, "code_fingerprint", lambda: {**TEST_SOURCE, "source_sha256": "c" * 64})
    with pytest.raises(ValueError, match="differs"):
        transfer.run_transfer_research(v3, data, cfg, output, device="cpu", resume=True)
    monkeypatch.setattr(transfer, "code_fingerprint", lambda: copy.deepcopy(TEST_SOURCE))
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        transfer.verify_transfer(output)


def test_partial_output_cannot_resume_and_source_change_is_rejected(source, tmp_path):
    data, v3 = source
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(FileNotFoundError):
        transfer.run_transfer_research(v3, data, config(), partial, device="cpu", resume=True)
    role = v3 / "fit-identity.parquet"
    with role.open("ab") as stream:
        stream.write(b"source changed")
    with pytest.raises(ValueError, match="artifact mismatch"):
        transfer.run_transfer_research(v3, data, config(), tmp_path / "no-new", device="cpu")
    assert not (tmp_path / "no-new").exists()


@pytest.mark.parametrize("group_key", ["feature_hash", "group_id"])
def test_frozen_training_roles_must_not_share_input_or_session_groups(source, monkeypatch, group_key):
    data, v3 = source
    original = transfer.load_partition
    frame = original(data, "train")
    role_ids = {r: pd.read_parquet(v3 / f"{r}-identity.parquet").sample_id.tolist()
                for r in ("fit", "reference", "constraint")}
    if group_key == "group_id":
        frame["group_id"] = "group:" + frame.feature_hash
    shared = "shared-cross-role-group"
    for role in ("fit", "reference"):
        frame.loc[frame.sample_id == role_ids[role][0], group_key] = shared
    for role, ids in role_ids.items():
        old = pd.read_parquet(v3 / f"{role}-identity.parquet")
        cols = list(old.columns)
        if group_key not in cols:
            cols.append(group_key)
        frame.set_index("sample_id", drop=False).loc[ids, cols].reset_index(drop=True).to_parquet(
            v3 / f"{role}-identity.parquet", index=False)
    monkeypatch.setattr(transfer, "load_partition", lambda *args: frame.copy())
    with pytest.raises(ValueError, match="share a feature/session group"):
        transfer._training_frames(v3, data)


def test_registered_headroom_gate_uses_C_only_and_reports_declined_expansion():
    cfg = config()
    cfg.update(qualification_min_pauc_gain=.02, qualification_min_tpr_gain=.01)
    baseline = {"C_rank": {"partial_auc_standardized": .7}, "C": {"tpr": .8}}
    barely = {"C_rank": {"partial_auc_standardized": .719}, "C": {"tpr": .809}}
    accepted = {"C_rank": {"partial_auc_standardized": .73}, "C": {"tpr": .799}}
    assert not transfer.qualification(barely, baseline, cfg)["qualified"]
    assert transfer.qualification(accepted, baseline, cfg)["qualified"]


def test_configuration_and_real_gpu_rules_fail_before_output(source, tmp_path, monkeypatch):
    data, v3 = source
    cfg = config()
    cfg["tail_multiplier"] = 1 / cfg["max_fpr"]
    with pytest.raises(ValueError, match="interest window"):
        transfer.run_transfer_research(v3, data, cfg, tmp_path / "bad-config", device="cpu")
    # Synthetic fixture remains genuine; only the declared hardware branch is
    # exercised with its verified metadata proxy and source-matching plan.
    manifest = read_json(data / "manifest.json")
    manifest["data_kind"] = "real-proxy-no-fit"
    plan = read_json(v3 / "plan.json")
    plan["data"] = manifest
    monkeypatch.setattr(transfer, "verify_development_inputs", lambda *args: manifest)
    original_read = transfer.read_json
    monkeypatch.setattr(transfer, "read_json", lambda p: plan if Path(p) == v3 / "plan.json" else original_read(p))
    with pytest.raises(ValueError, match="4090D"):
        transfer.run_transfer_research(v3, data, config(), tmp_path / "bad-hardware", device="cpu")
    assert not (tmp_path / "bad-hardware").exists()


def test_cpu_pipeline_measurement_runs_for_both_saved_HGB_and_MLP(source):
    data, v3 = source
    frame = transfer.load_partition(data, "validation").iloc[:8]
    for file in (v3 / "teachers/s17/reference.joblib", v3 / "anchors/s17.joblib"):
        result = transfer._benchmark(load_bundle(file), frame, 0., repeats=3)
        assert set(result["batches"]) == {"1", "8"}
        assert "extraction" in result["excludes"]
        for row in result["batches"].values():
            assert row["repeats"] == 3
            assert row["p95_ms"] >= row["p50_ms"] > 0
            assert row["mean_rows_per_second"] > 0
