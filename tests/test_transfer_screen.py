"""Actual synthetic screen: sealed V3 cohorts, native trees, no prepared V/test."""
import copy
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

import ids.budget_transfer as transfer
import ids.risk_research as risk
import ids.transfer_screen as screen
from ids.common import read_json, sha256_file
from ids.data import make_synthetic_dataset, prepare_dataset
from ids.native_trees import load_native_tree

REPO = Path(__file__).resolve().parents[1]
SOURCE = {"source_sha256": "b"*64, "files": {}}


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    pytest.importorskip("catboost")
    root = tmp_path_factory.mktemp("transfer-screen")
    raw, data, v3, output = (root / name for name in ("raw", "data", "v3", "screen"))
    make_synthetic_dataset(raw, n_train=900, n_test=120)
    prepare_dataset(raw / "UNSW_NB15_training-set.csv", raw / "UNSW_NB15_testing-set.csv", data, data_kind="synthetic")
    source_cfg = yaml.safe_load((REPO / "configs/risk-v3-smoke.yaml").read_text())
    source_cfg.update(conditions=["hard"], max_fpr=.25)
    source_cfg["anchor"]["hyperparameters"].update(hidden_sizes=[2, 2], epochs=1, patience=1)
    source_cfg["teacher"]["hyperparameters"].update(max_iter=2, max_leaf_nodes=3)
    source_cfg["optimization"].update(macro_steps=1, attack_loss_limit=1.)
    with patch.object(risk, "code_fingerprint", lambda: copy.deepcopy(SOURCE)):
        risk.run_risk_research(data, source_cfg, v3, device="cpu")
    for name in ("validation", "calibration", "test"):
        (data / (name + ".parquet")).unlink()
    cfg = yaml.safe_load((REPO / "configs/budget-transfer-smoke.yaml").read_text())
    cfg["measure_cpu"] = True
    before = {p.relative_to(v3).as_posix(): sha256_file(p) for p in v3.rglob("*") if p.is_file()}
    loads, parquet_reads = [], []
    load, parquet = transfer.load_partition, pd.read_parquet
    def only_train(folder, split):
        loads.append(split)
        assert split == "train"
        return load(folder, split)
    def no_v_identity(path, *args, **kwargs):
        parquet_reads.append(str(path))
        assert "validation" not in Path(path).name and "test" not in Path(path).name
        return parquet(path, *args, **kwargs)
    with patch.object(screen, "code_fingerprint", lambda: copy.deepcopy(SOURCE)), \
            patch.object(transfer, "load_partition", only_train), patch.object(pd, "read_parquet", no_v_identity):
        done = screen.run_transfer_screen(v3, data, cfg, output, device="cpu")
    assert loads == ["train"] and parquet_reads
    assert before == {p.relative_to(v3).as_posix(): sha256_file(p) for p in v3.rglob("*") if p.is_file()}
    return output, data, v3, cfg, done


def test_screen_real_synthetic_models_have_twelve_fits_and_no_v_or_kd(completed):
    output, data, _, cfg, done = completed
    assert [done[k] for k in ("n_teacher_fits", "n_warmup_fits", "n_student_fits", "n_native_tree_fits")] == [2, 4, 4, 2]
    assert done["status"] == "screen_complete" and not done["validation_loaded"]
    assert not (output / "validation-identity.parquet").exists()
    assert not (data / "validation.parquet").exists()
    summary = read_json(output / "summary.json")
    assert not summary["automatic_expansion"] and not summary["independent_confirmation"]
    assert {r["method"] for r in summary["runs"]} == {"hard", "direct_pauc"}
    assert len(summary["auxiliary"]) == 8
    for budget in ("all", "reduced"):
        rows = [r for r in summary["runs"] if r["budget"] == budget]
        assert len({r["common_initial_state_sha256"] for r in rows}) == 1
        warm = next(r for r in summary["auxiliary"] if r["budget"] == budget and r.get("capacity") == "small")
        count = warm["fit_rows"]
        assert warm["compute"]["optimizer_updates"] == cfg["warmup_epochs"] * ((count + cfg["warmup_batch_size"] - 1) // cfg["warmup_batch_size"])
        assert warm["compute"]["gradient_forward_rows"] == count * cfg["warmup_epochs"]
        assert all(r["warmup_selected_optimizer_updates"] == warm["compute"]["selected_checkpoint_optimizer_updates"] for r in rows)
    for row in summary["runs"] + summary["auxiliary"]:
        assert row["selection"]["C"]["fpr"] <= cfg["max_fpr"]
        c = row["selection"]["C"]
        assert c["decision_rule"] == "float64(raw_score)>=frozen_training_R+C_threshold"
        assert set(row["cpu_pipeline"]["batches"]) == {"1", str(min(256, c["n_normal"] + c["n_attack"]))}
        assert row["cpu_pipeline"]["measurement_rows"].startswith("fixed first C")
        with np.load(output / row["relative_dir"] / "training-scores.npz") as scores:
            assert "validation" not in scores
    trees = [r for r in summary["auxiliary"] if r["kind"] == "native_tree"]
    assert {r["training_statistics"]["tree_count"] for r in trees} == {64, 256}
    for row in trees:
        tree = load_native_tree(output / row["relative_dir"])
        assert tree.training_statistics["target"] == "hard"
        assert row["deployment"]["deployment_bytes"] > row["deployment"]["model_bytes"]
        assert row["fit_and_query_seconds"] >= row["training_statistics"]["fit_seconds"]
    for budget in ("all", "reduced"):
        diagnostic = read_json(output / f"diagnostic-{budget}-s17.json")
        teacher = next(r for r in summary["auxiliary"] if r["kind"] == "teacher" and r["budget"] == "all")
        student = next(r for r in summary["runs"] if r["budget"] == budget and r["method"] == "direct_pauc")
        selected = diagnostic["selected_working_point"]
        assert selected["teacher_threshold"] == teacher["selection"]["working_point"]["threshold"]
        assert selected["student_threshold"] == student["selection"]["working_point"]["threshold"]
        assert "R_only_diagnostic" in diagnostic
    assert screen.verify_transfer_screen(output) == done


def test_screen_resume_binds_config_source_device_without_loading_data(completed, monkeypatch):
    output, data, v3, cfg, done = completed
    monkeypatch.setattr(screen, "code_fingerprint", lambda: copy.deepcopy(SOURCE))
    monkeypatch.setattr(screen, "_training_frames", lambda *a: pytest.fail("completed resume loaded rows"))
    assert screen.run_transfer_screen(v3, data, cfg, output, device="cpu", resume=True) == done
    with pytest.raises(FileExistsError):
        screen.run_transfer_screen(v3, data, cfg, output, device="cpu")
    altered = {**cfg, "learning_rate": .02}
    with pytest.raises(ValueError, match="differs"):
        screen.run_transfer_screen(v3, data, altered, output, device="cpu", resume=True)
    with pytest.raises(ValueError, match="differs"):
        screen.run_transfer_screen(v3, data, cfg, output, device="cuda", resume=True)


def test_screen_preserves_partial_output_and_rejects_changed_artifacts(completed, tmp_path, monkeypatch):
    output, data, v3, cfg, _ = completed
    monkeypatch.setattr(screen, "code_fingerprint", lambda: copy.deepcopy(SOURCE))
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "keep.txt").write_text("keep")
    with pytest.raises(FileNotFoundError):
        screen.run_transfer_screen(v3, data, cfg, partial, device="cpu", resume=True)
    assert (partial / "keep.txt").read_text() == "keep"
    bad = tmp_path / "tampered"
    shutil.copytree(output, bad)
    (bad / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        screen.verify_transfer_screen(bad)


def test_target_content_exposes_hard_label_redundancy():
    result = screen._target_content([-1000., 1000.], [0, 1])
    assert result["mean_absolute_soft_minus_hard"] == 0
    ambiguous = screen._target_content([0., 0.], [0, 1])
    assert ambiguous["mean_absolute_soft_minus_hard"] == .5


def test_zero_fit_input_audit_uses_sealed_screen_and_no_reserved_data(completed, tmp_path, monkeypatch):
    import ids.input_extrapolation as extrapolation
    output, data, _, _, _ = completed
    target = tmp_path / "extrapolation"
    # The fixture has physically removed prepared validation/calibration/test.
    monkeypatch.setattr(transfer, "_warmup", lambda *a, **kw: pytest.fail("audit tried to train"))
    monkeypatch.setattr(transfer, "_fit_method", lambda *a, **kw: pytest.fail("audit tried to continue training"))
    done = extrapolation.run_input_extrapolation(output, data, target)
    assert done["new_model_fits"] == 0 and done["n_models"] == 8
    assert done["validation_loaded"] is False and done["reserved_partitions_read"] == []
    assert done["automatic_expansion"] is False
    summary = read_json(target / "summary.json")
    assert all(r["fitting_encoding_unchanged"] for r in summary["models"])
    assert all(not any(r["archived_decision_mismatches"].values()) for r in summary["models"])
    with pytest.raises(FileExistsError):
        extrapolation.run_input_extrapolation(output, data, target)


def test_selected_complementarity_uses_registered_joint_thresholds_not_r_only():
    frame = pd.DataFrame({"label": [0, 0, 1, 1], "attack_cat": ["Normal", "Normal", "A", "A"]})
    def selection(t):
        return {"working_point": {"threshold": t}, "C": {"threshold": t}}
    teacher, student = [0., 2., 1., 3.], [0., 1., 2., 3.]
    joint = screen._selected_complementarity(frame, teacher, selection(2.5), student, selection(1.5))
    r_only = screen._selected_complementarity(frame, teacher, selection(.5), student, selection(1.5))
    assert joint["paired"]["teacher_loses_student_tp"] == 1
    assert joint["paired"]["teacher_adds_student_fp"] == 0
    assert r_only["paired"]["teacher_loses_student_tp"] == 0
    assert r_only["paired"]["teacher_adds_student_fp"] == 1


def test_input_audit_rejects_different_data_and_changed_frozen_decisions(completed, tmp_path, monkeypatch):
    import ids.input_extrapolation as extrapolation
    output, data, _, _, _ = completed
    with monkeypatch.context() as patcher:
        patcher.setattr(extrapolation, "verify_development_inputs", lambda *a, **kw: {"different": True})
        with pytest.raises(ValueError, match="manifest differs"):
            extrapolation.run_input_extrapolation(output, data, tmp_path / "different-data")
    assert not (tmp_path / "different-data").exists()
    score = extrapolation._scores
    monkeypatch.setattr(extrapolation, "_scores",
                        lambda *a, **kw: {k: v + 1e6 for k, v in score(*a, **kw).items()})
    target = tmp_path / "replay-mismatch"
    with pytest.raises(ValueError, match="replay changed decisions"):
        extrapolation.run_input_extrapolation(output, data, target)
    assert (target / "REPLAY_MISMATCH.json").exists()
    assert not (target / extrapolation.COMPLETE).exists()
