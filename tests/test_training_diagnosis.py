import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.support_diagnostic import training_diagnosis as audit
from ids.threshold_validation.observations import RAW_FEATURES


def frame():
    rows = pd.DataFrame({"source_id": ["S"] * 4 + ["T"] * 4, "row_id": [1, 2, 3, 4] * 2,
                         "family": ["Benign", "Benign", "Scanning", "C&C"] + ["Benign", "Benign", "ExternalC2", "ExternalC2"],
                         "domain": ["source"] * 4 + ["target"] * 4,
                         "label": [0, 0, 1, 1] * 2, "weight": [1.0] * 8})
    for i, feature in enumerate(RAW_FEATURES):
        rows[feature] = np.arange(8, dtype=float) + i
    return rows


def test_bce_extreme_logits_stays_finite_and_not_probability_clipped():
    loss = audit.bce(np.array([-1000, 1000, 1000, -1000.0]), np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(loss, [0, 0, 1000, 1000])
    with pytest.raises(ValueError):
        audit.bce([0], [2])


def test_weighted_conflict_infimum_and_pure_group():
    result = audit.collision_floor([[1], [1], [2]], [0, 1, 1], [1, 3, 6])
    expected = -(np.log(.25) + 3 * np.log(.75)) / 10
    assert result["weighted_bce_infimum"] == pytest.approx(expected)
    assert result["weighted_error_infimum"] == .1
    assert result["conflicting_inputs"] == 1
    assert result["rows_in_conflicting_inputs"] == 2


def test_collision_checks_effective_fp32_not_only_original_float64():
    x = np.array([[1.0], [1.0 + 1e-9]])
    assert audit.collision_floor(x, [0, 1], [1, 1])["conflicting_inputs"] == 0
    assert audit.collision_floor(x.astype(np.float32), [0, 1], [1, 1])["weighted_bce_infimum"] == pytest.approx(np.log(2))


def test_own_target_objective_excludes_foreign_source_labels():
    rows = frame()
    mask, weights = audit.objective(rows, "target-ports_zero8-mlp17")
    assert mask.tolist() == [False] * 4 + [True] * 4
    assert len(weights) == 4
    assert weights.sum() == 4


def test_identity_and_labels_must_match_before_scoring():
    rows = frame()
    identities = rows[audit.IDENTITY].assign(support="mixed")
    pd.testing.assert_frame_equal(audit.validate_frame(rows, identities), rows)
    with pytest.raises(AssertionError):
        audit.validate_frame(rows.iloc[::-1], identities)
    changed = rows.copy()
    changed.loc[2, "label"] = 0
    with pytest.raises(AssertionError):
        audit.validate_frame(changed, identities)


def test_strict_ties_do_not_split_normal_group_for_one_percent_budget():
    rows = pd.concat([frame().iloc[[0]]] * 100 + [frame().iloc[[2]]] * 2, ignore_index=True)
    scores = np.array([0.0] * 98 + [1, 1] + [1, 2])
    result = audit.metrics(rows, scores, .5, arm="test", scope="cell", value="x", split="synthetic", seen=True)
    assert result["fp_budget"] == 1
    assert result["frozen_fp"] == 2
    assert result["frozen_qualified"] is False
    assert result["oracle_threshold"] == 1
    assert result["oracle_fp"] == 0
    assert result["oracle_tp"] == 1


def test_family_uses_parent_domain_normal_threshold_and_preserves_rare_control():
    rows = frame()
    scores = np.array([-1, -2, 2, -3, -1, -2, 2, 3.0])
    results, objective = audit.summarize_training(rows, scores, "mixed-ports_zero8-mlp17", {"threshold": 0})
    control = next(r for r in results if r["scope"] == "domain_family" and r["value"] == "source/C&C")
    assert control["oracle_tp"] is None  # No normal denominator inside a pure attack family.
    assert control["domain_oracle_normal_n"] == 2
    assert control["domain_oracle_family_alerts"] == 0
    assert len(objective["objective_contributions"]) == 5
    assert sum(r["contribution_to_objective"] for r in objective["objective_contributions"]) == pytest.approx(objective["final_weighted_training_bce"])


def test_complete_pipeline_resume_and_cached_BD_only(tmp_path, monkeypatch):
    rows = frame()
    r26, out = tmp_path / "r26", tmp_path / "diagnosis"
    (r26 / "local/training").mkdir(parents=True)
    (r26 / "evidence/replay").mkdir(parents=True)
    rows.to_parquet(r26 / "local/training/mixed.parquet", index=False)
    rows[audit.IDENTITY].assign(support="mixed").to_parquet(r26 / "evidence/replay/training-identities.parquet", index=False)
    arms = ["target-ports_zero8-mlp17", "mixed-ports_zero8-mlp17"]
    for arm in arms:
        write_json(r26 / "evidence/calibration" / (arm + ".json"), {"threshold": 0, "ready": True})
    cached = rows.iloc[4:].copy()
    for arm in arms:
        cached[arm] = [-1, -2, 2, 3]
    cached.to_parquet(r26 / "evidence/replay/TQH-B_synthetic.parquet", index=False)
    replay_name = "evidence/replay/TQH-B_synthetic.parquet"
    reg = {"arms": arms, "source_artifacts": {replay_name: sha256_file(r26 / replay_name)}, "training_rows_per_arm": 8,
           "mixed_training_sha256": sha256_file(r26 / "local/training/mixed.parquet")}
    registry = tmp_path / "registration.json"
    write_json(registry, reg)
    monkeypatch.setattr(audit, "REGISTRY", registry)
    monkeypatch.setattr(audit, "effective_inputs", lambda f, arm, _: (f[list(RAW_FEATURES)].to_numpy(), "synthetic_effective"))
    from ids import common
    from ids.support_diagnostic import runner
    monkeypatch.setattr(common, "environment_snapshot", lambda **_: {"python": "3.12", "packages": {}, "torch": {"cuda_available": True}})
    monkeypatch.setattr(runner, "verify_runtime", lambda _: None)
    calls = []

    def synthetic_score(r26, output, f, arm, threads):
        calls.append(arm)
        path = output / "evidence/training_scores" / (arm + ".parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        f[audit.IDENTITY].assign(score=np.where(f.label, 3.0, -3.0)).to_parquet(path, index=False)
        return [path]

    monkeypatch.setattr(audit, "score_one", synthetic_score)
    audit.run(r26, out)
    assert calls == arms
    assert read_json(out / "outcome.json")["scored_rows"] == 16
    audit.run(r26, out)
    assert calls == arms  # Resume consumes existing scores rather than scoring again.
    assert len(pd.read_csv(out / "evidence/cached_BD.csv")) == 2
    cached.to_parquet(r26 / "evidence/replay/TQH-D_unregistered.parquet", index=False)
    audit.run(r26, out, inputs_only=True)
    assert read_json(out / "outcome.json")["status"] == "complete"
    assert len(pd.read_csv(out / "evidence/cached_BD.csv")) == 2
    before = (out / "evidence/input_conflicts.json").read_bytes()
    with pytest.raises(ValueError, match="binding changed"):
        audit.run(r26, out, threads=5)
    assert (out / "evidence/input_conflicts.json").read_bytes() == before


def test_input_only_never_calls_model_loader_or_scorer(tmp_path, monkeypatch):
    rows = frame()
    r26, out = tmp_path / "r26", tmp_path / "out"
    r26.mkdir()
    monkeypatch.setattr(audit, "verify_artifacts", lambda *_: None)
    source = tmp_path / "inputs.txt"
    source.write_text("synthetic")
    monkeypatch.setattr(audit, "load_frame", lambda *_: (rows, source))
    monkeypatch.setattr(audit, "input_analysis", lambda *_: [])
    monkeypatch.setattr(audit, "cached_evaluation", lambda *_: [])
    monkeypatch.setattr(audit, "score_one", lambda *_: pytest.fail("Unexpected model scoring"))
    reg = tmp_path / "reg.json"
    write_json(reg, {"training_rows_per_arm": 8, "arms": ["fixed"]})
    monkeypatch.setattr(audit, "REGISTRY", reg)
    audit.run(r26, out, inputs_only=True)
    assert read_json(out / "outcome.json")["new_model_forward_calls"] == 0


def test_output_cannot_modify_frozen_run(tmp_path):
    with pytest.raises(ValueError, match="separate output"):
        audit.run(tmp_path, tmp_path / "child", inputs_only=True)
