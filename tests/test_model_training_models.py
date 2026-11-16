"""Only synthetic frames and randomly initialized tiny official CPU models."""
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ids.model_training import models
from ids.model_training.preprocessing import CONTINUOUS


def synthetic(rows=80, seed=7):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.uniform(0, 10, (rows, 5)), columns=CONTINUOUS)
    frame["ip_protocol"] = np.resize([6, 17, 58], rows)
    frame["label"] = (frame.orig_pkts > 5).astype(int)
    frame.loc[::11, "duration_seconds"] = -1
    return frame


@pytest.fixture
def design():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "configs/model-design-20261116/design.json").read_text(encoding="utf-8"))
    value["training"]["hgb"].update(max_iter=4, max_leaf_nodes=3)
    value["training"]["tabm"].update(max_epochs=3, patience=2, batch=32, k=2, n_blocks=1,
                                     d_block=8, embedding_dim=4, max_bins=4, dropout=0.1)
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_selection_is_pooled_standardized_pauc_then_nll_then_earliest():
    labels = np.array([0, 1, 0, 1], dtype=float)
    scores = np.array([-3, 2, 1, 4], dtype=float)
    metric = models.selection_metrics(scores, labels)
    assert metric["pauc"] == roc_auc_score(labels, scores, max_fpr=0.01)
    assert models.selection_key({"pauc": 0.9, "nll": 2}, 4) > models.selection_key({"pauc": 0.8, "nll": 0.1}, 1)
    assert models.selection_key({"pauc": 0.9, "nll": 1}, 4) > models.selection_key({"pauc": 0.9, "nll": 2}, 1)
    assert models.selection_key(metric, 1) > models.selection_key(metric, 2)
    assert np.isfinite(models.selection_metrics(np.array([-1000, 1000, -1000, 1000]), labels)["nll"])
    confident_loss = models._nll(np.array([-40.0, 40.0]), np.array([0, 1]))
    assert confident_loss > 0
    assert confident_loss == pytest.approx(np.log1p(np.exp(-40.0)), rel=1e-14)


def test_hgb_one_fit_staged_selection_and_final_refit_use_own_partition(tmp_path, design, monkeypatch):
    f, v = synthetic(), synthetic(50, 19)
    calls = []
    original = models.HistGradientBoostingClassifier.fit

    def tracked(self, x, y, *args, **kwargs):
        calls.append((len(x), self.max_iter, kwargs))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(models.HistGradientBoostingClassifier, "fit", tracked)
    result = models.fit_trajectory(f, v, tmp_path / "trajectory", arm="hgb", seed=20261111,
                                  threads=1, device="cpu", design=design)
    assert result["status"] == "complete", result
    assert len(calls) == 1 and calls[0][0:2] == (80, 4) and calls[0][2] == {}
    history = read(result["history_path"])
    selected = max(history, key=lambda row: (row["validation_pauc"], -row["validation_ensemble_bce"], -row["stage"]))
    assert result["selected_stage"] == selected["stage"]
    predictor = models.load_predictor(result["artifact_path"], threads=1)
    saved = np.load(result["train_scores_path"])
    np.testing.assert_array_equal(predictor.score(v), saved["validation_scores"])
    p = pd.concat([f, v], ignore_index=True)
    final = models.fit_final(p, tmp_path / "final", arm="hgb", seed=20261111,
        stage=result["selected_stage"], threads=1, device="cpu", design=design)
    assert final["status"] == "complete", final
    assert len(calls) == 2 and calls[-1][0:2] == (130, result["selected_stage"])
    state = joblib.load(final["artifact_path"])
    assert state["preprocessing"]["fit_rows"] == len(p)
    assert final["selection"] is None
    assert not any("validation" in key for row in read(final["history_path"]) for key in row)
    np.testing.assert_array_equal(models.load_predictor(final["artifact_path"]).score(p),
                                  np.load(final["train_scores_path"])["train_scores"])


def test_nonempty_directory_rejects_repeat_and_failures_keep_evidence(tmp_path, design):
    f = synthetic()
    bad = models.fit_trajectory(f, f.assign(label=0), tmp_path / "bad", arm="hgb", seed=17,
                               threads=1, device="cpu", design=design)
    assert bad["status"] == "failed" and bad["retry_allowed"] is False
    assert read(tmp_path / "bad/failure.json")["error_type"] == "ValueError"
    with pytest.raises(ValueError, match="empty"):
        models.fit_trajectory(f, f, tmp_path / "bad", arm="hgb", seed=17,
                              threads=1, device="cpu", design=design)


def test_weight_column_is_not_used_and_selected_stage_cannot_exceed_budget(tmp_path, design):
    f = synthetic()
    a = models.fit_final(f, tmp_path / "a", arm="hgb", seed=17, stage=2, threads=1, device="cpu", design=design)
    b = models.fit_final(f.assign(weight=np.where(f.label, 100, 0.1)), tmp_path / "b",
                         arm="hgb", seed=17, stage=2, threads=1, device="cpu", design=design)
    np.testing.assert_array_equal(models.load_predictor(a["artifact_path"]).score(f),
                                  models.load_predictor(b["artifact_path"]).score(f))
    with pytest.raises(ValueError, match="budget"):
        models.fit_final(f, tmp_path / "over", arm="hgb", seed=17, stage=5,
                         threads=1, device="cpu", design=design)


@pytest.mark.parametrize("arm", ["tabm_ple", "tabm_no_ple"])
def test_official_tiny_tabm_roundtrip_final_stage_and_typed_categories(tmp_path, design, arm):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tabm")
    pytest.importorskip("rtdl_num_embeddings")
    f, v = synthetic(48), synthetic(32, 19)
    f["orig_ip_bytes"] = -1  # All-missing numerical column exercises fixed PLE edges.
    result = models.fit_trajectory(f, v, tmp_path / "trajectory", arm=arm, seed=17,
                                  threads=1, device="cpu", design=design)
    assert result["status"] == "complete", result
    state = torch.load(result["artifact_path"], weights_only=True)
    assert state["preprocessing"]["categorical_cardinalities"] == [6, 2, 2, 2, 2, 2]
    assert state["preprocessing"]["fit_rows"] == len(f)
    if arm == "tabm_ple":
        assert len(state["numeric_bins"]) == 5 and state["numeric_bins"][3] == [-1, 1]
    else:
        assert state["numeric_bins"] == []
    predictor = models.load_predictor(result["artifact_path"], device="cpu", threads=1)
    scores = predictor.score(v)
    assert scores.dtype == np.float64 and np.isfinite(scores).all()
    np.testing.assert_array_equal(scores, np.load(result["train_scores_path"])["validation_scores"])
    np.testing.assert_array_equal(scores, predictor.score(v.assign(orig_port=1, resp_port=2, source_id="altered")))
    history = read(result["history_path"])
    assert all(row["path_loss_is_endpoint_loss"] is False for row in history)
    assert history[-1]["updates"] == 2 * result["completed_stages"]
    final = models.fit_final(pd.concat([f, v]), tmp_path / "final", arm=arm, seed=17,
        stage=result["selected_stage"], threads=1, device="cpu", design=design)
    assert final["status"] == "complete", final
    assert final["completed_stages"] == result["selected_stage"] and final["selection"] is None
    final_state = torch.load(final["artifact_path"], weights_only=True)
    assert final_state["preprocessing"]["fit_rows"] == 80
    assert not any("validation" in key for row in read(final["history_path"]) for key in row)
    loaded = models.load_predictor(final["artifact_path"], threads=1)
    np.testing.assert_array_equal(loaded.score(pd.concat([f, v])), np.load(final["train_scores_path"])["train_scores"])


def test_tabm_patience_uses_complete_lexicographic_rule(tmp_path, design, monkeypatch):
    pytest.importorskip("tabm")
    design = copy.deepcopy(design)
    design["training"]["tabm"].update(max_epochs=8, patience=2)
    monkeypatch.setattr(models, "selection_metrics", lambda *args, **kwargs: {"pauc": 0.6, "nll": 0.2})
    result = models.fit_trajectory(synthetic(32), synthetic(24, 19), tmp_path / "early", arm="tabm_no_ple",
                                   seed=17, threads=1, device="cpu", design=design)
    assert result["status"] == "complete", result
    assert result["selected_stage"] == 1 and result["completed_stages"] == 3 and result["stopped_early"]
