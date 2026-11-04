"""No-fit teacher capability, score precision and paired-error contracts."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from ids.common import read_json, sha256_file, write_json
from test_upgrade_threshold_lab import lab


class _IdentityPreprocessor:
    def transform(self, frame):
        return frame[["z"]].to_numpy(dtype=np.float64)


class _PublicBinaryTeacher:
    classes_ = np.array([0, 1])

    def decision_function(self, matrix):
        return matrix[:, 0].copy()

    def predict_proba(self, matrix):
        p = expit(matrix[:, 0])
        return np.column_stack((1 - p, p))


def _teacher():
    return SimpleNamespace(model_name="hgb", preprocessor=_IdentityPreprocessor(), estimator=_PublicBinaryTeacher())


def test_native_public_scores_preserve_information_erased_by_exact_old_clip():
    frame = pd.DataFrame({"z": [-1000., -100., -20., -14., 20.]})
    scores, checks = lab.teacher_score_variants(_teacher(), frame)
    np.testing.assert_array_equal(scores["native"], frame.z.to_numpy())
    p32 = expit(frame.z.to_numpy()).astype(np.float32)
    expected = logit(np.clip(p32, 1e-6, 1 - 1e-6)).astype(np.float64)
    np.testing.assert_array_equal(scores["legacy_clipped"], expected)
    assert len(np.unique(scores["native"][:4])) == 4
    assert len(np.unique(scores["legacy_clipped"][:4])) == 1
    assert checks["legacy_lower_clip_fraction"] == .8
    assert checks["native_link_max_abs_probability_difference"] == 0
    # Float64 nextafter rejects all normal plateau rows; no f32 threshold recast.
    threshold = np.nextafter(scores["legacy_clipped"][0], np.inf)
    metrics = lab._fixed_metrics(pd.DataFrame({"label": [0] * 4, "attack_cat": ["Normal"] * 4}),
                                 scores["legacy_clipped"][:4], threshold)
    assert metrics["false_positives"] == 0


def test_paired_teacher_errors_are_four_separate_events_and_family_counts():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    student = np.array([-1., 1., -1., 1., 1., -1., 1., -1.])
    teacher = np.array([1., -1., -1., 1., -1., 1., 1., -1.])
    family = ["A", "A", "B", "B"] + ["Normal"] * 4
    report = lab.teacher_student_complementarity(y, family, teacher, 0., student, 0.)
    assert report["teacher_corrects_student_fn"] == 1
    assert report["teacher_loses_student_tp"] == 1
    assert report["teacher_repairs_student_fp"] == 1
    assert report["teacher_adds_student_fp"] == 1
    assert report["teacher_minus_student_tp"] == report["teacher_minus_student_fp"] == 0
    assert report["per_attack_family"]["A"] == {"support": 2, "teacher_corrects_student_fn": 1, "teacher_loses_student_tp": 1}
    assert report["per_attack_family"]["B"]["teacher_corrects_student_fn"] == 0


def test_grouped_oof_missing_family_is_not_missing_binary_attack_support():
    # Family A is wholly held out of fold 0's teacher, which still saw both
    # normal and OTHER attacks. Low raw probability can still rank above a
    # much lower, separately frozen working-point threshold.
    fit = pd.DataFrame({"label": [1, 1, 1, 0], "attack_cat": ["A", "A", "B", "Normal"]})
    targets = pd.DataFrame({"label": fit.label, "fold": [0, 0, 1, 1],
        "teacher_logit": logit(np.array([.1, .2, .9, .01])), "teacher_margin": [1., 2., -1., -2.]})
    metadata = {"folds": [
        {"fold": 0, "fit": {"families": {"B": 20, "Normal": 10}, "labels": {"0": 10, "1": 20}}},
        {"fold": 1, "fit": {"families": {"A": 2, "B": 7, "Normal": 8}, "labels": {"0": 8, "1": 9}}}]}
    report = lab._oof_family_support(targets, fit, metadata)
    a, b = report["A"], report["B"]
    assert a["n"] == a["n_rows_with_family_absent_in_teacher_fit"] == 2
    assert a["teacher_fit_family_counts_by_target_fold"] == {"0": 0}
    assert a["teacher_fit_binary_counts_by_target_fold"]["0"] == {"0": 10, "1": 20}
    assert a["raw_probability_median"] == pytest.approx(.15)
    assert a["raw0_5_tpr"] == 0 and a["workingpoint_tpr"] == 1
    assert not a["family_is_binary_training_target"]
    assert b["n"] == 1 and b["n_rows_with_family_absent_in_teacher_fit"] == 0
    assert b["teacher_fit_family_counts_by_target_fold"] == {"1": 7}
    assert b["raw0_5_tpr"] == 1 and b["workingpoint_tpr"] == 0
    del metadata["folds"][0]["fit"]["families"]
    with pytest.raises(ValueError, match="absence is unknown"):
        lab._oof_family_support(targets, fit, metadata)


def test_teacher_R_thresholds_fixed_before_V_and_existing_student_scores_reused(tmp_path, monkeypatch):
    v3, output = tmp_path / "v3", tmp_path / "audit"
    teacher_path = v3 / "teachers/s17/reference.joblib"
    teacher_path.parent.mkdir(parents=True)
    teacher_path.write_bytes(b"trusted-synthetic-teacher-interface")
    frames = {}
    for role, values in (("fit", [-2., 2., 3.]), ("reference", [-20., -15., 2.]),
                         ("constraint", [-21., -16., 3.]), ("validation", [-18., -17., 1.])):
        frames[role] = pd.DataFrame({"z": values, "label": [0, 0, 1] if role != "fit" else [0, 1, 1],
            "attack_cat": ["Normal", "Normal", "Attack"] if role != "fit" else ["Normal", "Attack", "Attack"],
            "sample_id": [f"{role}-{i}" for i in range(3)]})
        frames[role].attrs["role"] = role
    legacy, _ = lab.teacher_score_variants(_teacher(), frames["fit"])
    targets = frames["fit"][["sample_id", "label"]].assign(teacher_logit=legacy["legacy_clipped"],
        teacher_margin=[-10., 2., 3.], fold=[0, 1, 0])
    target_path = teacher_path.parent / "targets.parquet"
    targets.to_parquet(target_path, index=False)
    oof_path = teacher_path.parent / "oof.json"
    write_json(oof_path, {"folds": [
        {"fit": {"n_rows": 2, "families": {"Normal": 1, "Attack": 1}, "labels": {"0": 1, "1": 1}}},
        {"fit": {"n_rows": 1, "families": {"Attack": 1}, "labels": {"0": 0, "1": 1}}}]})
    folds = [teacher_path.parent / f"fold-{i}.joblib" for i in (0, 1)]
    for path in folds:
        path.write_bytes(b"synthetic-archived-fold")
    write_json(v3 / "RISK_RESEARCH_COMPLETE.json", {"files": {
        p.relative_to(v3).as_posix(): sha256_file(p) for p in (teacher_path, target_path, oof_path, *folds)}})
    student_dir = output / "models/student-s17"
    student_dir.mkdir(parents=True)
    np.savez_compressed(student_dir / "scores.npz", validation=[-2., -1., 2.])
    model_results = [{"model": {"name": "student-s17"}, "regimes": {"R_only": {"threshold": 0.}}}]
    events, original_score, original_freeze = [], lab.teacher_score_variants, lab.freeze_margin_threshold

    def score(bundle, frame):
        events.append(frame.attrs["role"])
        if frame.attrs["role"] == "validation":
            assert events.count("threshold") == 2
        return original_score(bundle, frame)

    def freeze(*args, **kwargs):
        events.append("threshold")
        return original_freeze(*args, **kwargs)

    monkeypatch.setattr(lab, "load_bundle", lambda *a, **k: _teacher())
    monkeypatch.setattr(lab, "teacher_score_variants", score)
    monkeypatch.setattr(lab, "freeze_margin_threshold", freeze)
    monkeypatch.setattr(lab, "raw_cpu_logits", lambda *a: pytest.fail("MLP scores were forwarded again"))
    row = lab.audit_teacher_value(v3, 17, frames, model_results, output, .01, 1)
    assert events == ["reference", "constraint", "threshold", "threshold", "validation"] + [
        "reference", "threshold", "threshold", "fit", "constraint"] * 2
    assert row["fits"] == 0 and row["teacher_training_roles"] == ["fit"]
    assert row["student_hard_training_roles"] == ["fit", "constraint"]
    assert row["actual_oof_targets"]["native_fold_predictions_queried"] is True
    assert row["actual_oof_targets"]["fold_fit_rows"] == [2, 1]
    assert set(row["paired_with_mlp_R_only"]) == {"native", "legacy_clipped"}
    assert row["evaluations"]["legacy_clipped"]["validation"]["false_positives"] == 0
    assert row["evaluations"]["native"]["partial_auc_standardized_mcclish"] == 1.


@pytest.mark.parametrize("kind", ["class_order", "not_hgb", "nonfinite"])
def test_invalid_teacher_scoring_contract_rejected(kind):
    bundle = _teacher()
    frame = pd.DataFrame({"z": [-1., 1.]})
    if kind == "class_order":
        bundle.estimator.classes_ = np.array([1, 0])
    elif kind == "not_hgb":
        bundle.model_name = "rf"
    else:
        frame.loc[0, "z"] = np.inf
    with pytest.raises(ValueError):
        lab.teacher_score_variants(bundle, frame)


def test_oof_queries_only_original_holdouts_and_rejects_changed_cached_scores(tmp_path, monkeypatch):
    frames = {}
    for role, z in (("fit", [-20., 1., -15., 2.]), ("reference", [-25., -24., 3., 4.]),
                    ("constraint", [-23., -22., 5., 6.])):
        frames[role] = pd.DataFrame({"sample_id": [f"{role}-{i}" for i in range(4)], "z": z,
            "label": [0, 1, 0, 1], "attack_cat": ["Normal", "Attack", "Normal", "Attack"]})
    scores, _ = lab.teacher_score_variants(_teacher(), frames["fit"])
    targets = frames["fit"][["sample_id", "label"]].assign(fold=[0, 0, 1, 1],
        teacher_logit=scores["legacy_clipped"], teacher_margin=[-1., 1., -1., 1.])
    metadata, inventory = {"folds": []}, {}
    for fold in (0, 1):
        relative = f"teachers/s17/fold-{fold}.joblib"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-sealed-fold")
        inventory[relative] = sha256_file(path)
        metadata["folds"].append({"fold": fold, "fit": {"n_rows": 2, "families": {"Normal": 1, "Attack": 1}}})
    queried, original = [], lab.teacher_score_variants

    def score(bundle, frame):
        queried.append(frame.sample_id.tolist())
        return original(bundle, frame)

    monkeypatch.setattr(lab, "load_bundle", lambda *a, **k: _teacher())
    monkeypatch.setattr(lab, "teacher_score_variants", score)
    report, arrays = lab.audit_oof_teacher_value(tmp_path, 17, frames, targets, metadata, inventory, .01, 1)
    heldout_queries = [ids for ids in queried if ids[0].startswith("fit-")]
    assert heldout_queries == [["fit-0", "fit-1"], ["fit-2", "fit-3"]]
    np.testing.assert_array_equal(arrays["native_fit_oof"], frames["fit"].z)
    assert report["pooled_score_auc"] is None and report["fits"] == 0
    targets.loc[0, "teacher_logit"] += 1.
    with pytest.raises(ValueError, match="exact legacy targets"):
        lab.audit_oof_teacher_value(tmp_path, 17, frames, targets, metadata, inventory, .01, 1)


def test_real_synthetic_teacher_value_scores_and_source_binding(tmp_path, monkeypatch):
    """Actual fitted toy HGB/MLP artifacts; audit itself cannot call fit/step."""
    import importlib.util
    import torch
    import ids.models as models
    import ids.upgrade_research as upgrade
    from sklearn.ensemble import HistGradientBoostingClassifier

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("teacher_fixture", repo / "tests/test_upgrade_research.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    data, v3 = fixture.source.__wrapped__(tmp_path, monkeypatch)
    cfg = fixture.config()
    cfg.update(max_fpr=.25, conditions=["hard"])
    cfg["optimization"].update(attack_loss_limit=1., macro_steps=1)
    v4 = tmp_path / "v4"
    upgrade.run_upgrade_research(v3, data, cfg, v4, device="cpu")
    record = lab._metadata_record(v3, v4, data)
    binding = {"source": lab.code_fingerprint(), "purpose": "synthetic_teacher_value_audit", "threads": 1}
    monkeypatch.setattr(models, "train_model", lambda *a, **k: pytest.fail("teacher was refitted"))
    monkeypatch.setattr(HistGradientBoostingClassifier, "fit", lambda *a, **k: pytest.fail("HGB fit called during audit"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("student update called during audit"))
    output = tmp_path / "audit"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        complete = lab._audit_environment("synthetic", record, output, binding, 1)
    finally:
        torch.set_num_threads(previous)
    assert complete["artifact_schema_version"] == 3
    assert complete["n_models"] == 4 and complete["n_teacher_value_audits"] == 1
    summary = read_json(output / "summary.json")
    teacher = summary["teacher_value_audits"][0]
    assert teacher["source_model_sha256"] == sha256_file(v3 / "teachers/s17/reference.joblib")
    assert set(teacher["paired_with_mlp_R_only"]["native"]) == {r["model"]["name"] for r in summary["results"]}
    with np.load(output / "teacher-value-s17/scores.npz", allow_pickle=False) as scores:
        labels = pd.read_parquet(output / "validation-identity.parquet").label.to_numpy()
        for variant in ("native", "legacy_clipped"):
            wp = teacher["evaluations"][variant]["working_point"]
            expected = lab._fixed_metrics(pd.read_parquet(output / "validation-identity.parquet"), scores[variant + "_validation"], wp["threshold"])
            assert expected == teacher["evaluations"][variant]["validation"]
            assert len(scores[variant + "_validation"]) == len(labels)
    assert lab.verify_audit(output) == complete
    oof = teacher["actual_oof_targets"]
    assert oof["query_roles"] == ["F_original_fold_holdout", "R", "C"]
    assert oof["pooled_score_auc"] is None
    with np.load(output / "teacher-value-s17/oof-scores.npz", allow_pickle=False) as scores:
        identities = pd.read_parquet(output / "fit-identity.parquet")
        cached = pd.read_parquet(v3 / "teachers/s17/targets.parquet")
        np.testing.assert_array_equal(scores["legacy_clipped_fit_oof"], cached.teacher_logit)
        np.testing.assert_array_equal(scores["fold"], cached.fold)
        for variant in ("native", "legacy_clipped"):
            decision = scores[f"{variant}_fit_oof_decision"]
            expected = lab._fixed_metrics(identities, decision, .5)
            report = oof["pooled_oof_fixed_decisions"][variant]
            assert report["true_positives"] == expected["true_positives"]
            assert report["false_positives"] == expected["false_positives"]
        for fold in oof["folds"]:
            assert fold["legacy_targets_exactly_reproduced"] is True
            assert fold["source_model_sha256"] == sha256_file(v3 / fold["source_model_relative_path"])
            ix = cached.fold.to_numpy() == fold["fold"]
            for variant in ("native", "legacy_clipped"):
                threshold = fold["evaluations"][variant]["working_point"]["threshold"]
                np.testing.assert_array_equal(scores[f"{variant}_fit_oof_decision"][ix],
                    scores[f"{variant}_fit_oof"][ix] >= threshold)

    # Resealing changed JSON bytes cannot detach the teacher from the archived
    # v3 inventory. This checks the semantic source binding beyond payload hashes.
    summary["teacher_value_audits"][0]["source_model_sha256"] = "a" * 64
    write_json(output / "summary.json", summary)
    write_json(output / "teacher-value-s17/audit.json", summary["teacher_value_audits"][0])
    complete["files"] = {p.relative_to(output).as_posix(): sha256_file(p)
                         for p in output.rglob("*") if p.is_file() and p.name != lab.COMPLETE}
    write_json(output / lab.COMPLETE, complete)
    with pytest.raises(ValueError, match="teacher-value"):
        lab.verify_audit(output)
