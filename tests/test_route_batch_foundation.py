"""Foundation API contracts and random tiny official model; no pretrained weights."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ids.route_batch import foundation as f


class FakeTabICL:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.batches = []
        self.model_ = SimpleNamespace(to=lambda device: None)

    def fit(self, x, y):
        self.received_x = x.copy()
        self.classes_ = np.unique(y)
        self.ensemble_generator_ = SimpleNamespace(
            class_shuffles_={"none": [[0, 1]] * self.parameters["n_estimators"]})
        return self

    def _resolve_device(self):
        self.device_ = self.device

    def _build_inference_config(self):
        pass

    def _move_cache_to_device(self):
        pass

    def predict_proba(self, x):
        self.batches.append(len(x))
        p = 1 / (1 + np.exp(-x[:, 0]))
        return np.column_stack([1 - p, p])


class FakeTabPFN(FakeTabICL):
    @classmethod
    def create_default_for_version(cls, version, **parameters):
        assert version == "v2.5"
        return cls(**parameters)

    def fit(self, x, y):
        import os
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TABPFN_NO_BROWSER"] == "1"
        self.n_estimators_ = 8
        return super().fit(x, y)

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def data():
    x = np.arange(8 * 48, dtype=np.float64).reshape(8, 48) / 100
    x[:, 1] = [65530, 65531, 65532, 65533, 65534, 65535, 0, 1]
    return x, np.arange(8) % 2


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "random-placeholder-not-pretrained.ckpt"
    path.write_bytes(b"synthetic checkpoint placeholder")
    return path


@pytest.fixture
def mock_tabicl(monkeypatch):
    monkeypatch.setattr(f, "_tabicl_class", lambda: (FakeTabICL, {"commit": "synthetic"}))


@pytest.mark.parametrize("views", [1, 8])
def test_tabicl_receives_raw_numeric_and_registered_system_parameters(data, checkpoint, mock_tabicl, views):
    x, y = data
    model = f.fit_foundation("tabicl", x, y, checkpoint_path=checkpoint, seed=17,
                             device="cpu", n_estimators=views)
    np.testing.assert_array_equal(model.estimator.received_x, x)
    settings = model.estimator.parameters
    assert settings["allow_auto_download"] is False
    assert settings["n_jobs"] == 1 and settings["use_amp"] is False
    assert settings["norm_methods"] == ["none", "power"]
    assert settings["feat_shuffle_method"] == "latin"
    assert settings["class_shuffle_method"] == "shift"
    assert settings["outlier_threshold"] == 4.0 and settings["kv_cache"] is True
    assert model.estimator.device == "cpu"
    assert model.metadata["actual_estimators"] == views
    assert "complete-pipeline" in model.metadata["comparison"]
    query = np.tile(x[:1], (257, 1))
    scores = model.predict_scores(query)
    assert model.estimator.batches == [128, 128, 1]
    np.testing.assert_allclose(scores, query[:, 0], atol=1e-14)
    assert scores.dtype == np.float64
    with pytest.raises(ValueError, match="batch_size=128"):
        model.predict_scores(query, batch_size=127)


def test_probability_log_odds_clipping_retains_saturation_ties(data):
    x, y = data
    estimator = SimpleNamespace(predict_proba=lambda _: np.array([[1., 0.], [0., 1.], [0., 1.], [.5, .5]]))
    model = f.FoundationModel("tabicl", estimator, {}, x, y)
    scores = model.predict_scores(x[:4])
    bound = np.log(1 - f.SCORE_EPSILON) - np.log(f.SCORE_EPSILON)
    np.testing.assert_array_equal(scores, [-bound, bound, bound, 0])
    assert model.predict_scores(x[:0]).shape == (0,)


def test_missing_checkpoint_does_not_import_model(data, tmp_path, monkeypatch):
    monkeypatch.setattr(f, "_tabicl_class", lambda: pytest.fail("model import attempted"))
    with pytest.raises(f.FoundationUnavailable, match="no download attempted"):
        f.fit_foundation("tabicl", *data, checkpoint_path=tmp_path / "missing", seed=1,
                         device="cpu", n_estimators=1)


def test_invalid_input_and_wrong_weight_are_errors_not_unavailable(data, checkpoint):
    x, y = data
    with pytest.raises(ValueError, match="raw numeric"):
        f.fit_foundation("tabicl", x.astype(str), y, checkpoint_path=checkpoint, seed=1,
                         device="cpu", n_estimators=1)
    with pytest.raises(ValueError, match="both 0"):
        f.fit_foundation("tabicl", x, np.ones(len(x)), checkpoint_path=checkpoint, seed=1,
                         device="cpu", n_estimators=1)
    with pytest.raises(ValueError, match="SHA-256"):
        f.fit_foundation("tabpfn25", x, y, checkpoint_path=checkpoint, seed=1,
                         device="cpu", n_estimators=8)


def test_tabpfn_version_selection_local_path_offline_and_cpu(data, tmp_path, monkeypatch):
    import os
    import torch
    checkpoint = tmp_path / f.TABPFN_CHECKPOINT
    checkpoint.write_bytes(b"synthetic TabPFN API test; not weights")
    monkeypatch.setattr(f, "TABPFN_CHECKPOINT_SHA256", f._sha256(checkpoint))
    settings = SimpleNamespace(tabpfn=SimpleNamespace(allow_cpu_large_dataset=False))
    loading = SimpleNamespace(download_model=lambda: pytest.fail("download attempted"))
    monkeypatch.setattr(f, "_tabpfn_class", lambda: (FakeTabPFN, "v2.5", loading, settings))
    previous = os.environ.get("HF_HUB_OFFLINE")
    threads = torch.get_num_threads()
    model = f.fit_foundation("tabpfn25", *data, checkpoint_path=checkpoint, seed=11,
                             device="cuda", n_estimators=8)
    parameters = model.estimator.parameters
    assert parameters["model_path"] == str(checkpoint.resolve())
    assert parameters["fit_mode"] == "fit_with_cache"
    assert parameters["inference_precision"] == torch.float32
    assert parameters["average_before_softmax"] is False
    assert parameters["categorical_features_indices"] == []
    assert parameters["ignore_pretraining_limits"] is False
    assert parameters["kv_cache_precision"] == "auto"
    assert model.estimator.device == "cpu"
    assert settings.tabpfn.allow_cpu_large_dataset is False
    assert os.environ.get("HF_HUB_OFFLINE") == previous
    assert torch.get_num_threads() == threads


def test_recipe_rebuild_roundtrip_and_external_weight_binding(data, checkpoint, mock_tabicl, tmp_path):
    model = f.fit_foundation("tabicl", *data, checkpoint_path=checkpoint, seed=19,
                             device="cpu", n_estimators=8)
    path = tmp_path / "recipe.joblib"
    report = model.save(path)
    assert report["requires_rebuild_on_load"] and not report["contains_weights"]
    restored = f.load_foundation(path)
    np.testing.assert_array_equal(restored.predict_scores(data[0]), model.predict_scores(data[0]))
    assert restored.metadata["load_including_rebuild_seconds"] > 0
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        f.load_foundation(path)


def test_tabpfn_wrong_package_version_is_explicit_unavailability(monkeypatch):
    monkeypatch.setattr(f, "version", lambda name: "8.0.0")
    with pytest.raises(f.FoundationUnavailable, match="9.2.0 required; found 8.0.0"):
        f._tabpfn_class()


def test_random_tiny_official_tabicl_fit_predict_and_rebuild(monkeypatch, tmp_path):
    import torch
    checkout = Path(__file__).resolve().parents[1] / "local" / "vendor" / "tabicl"
    if not (checkout / "src" / "tabicl" / "_model" / "tabicl.py").is_file():
        pytest.skip("pinned optional official source absent; no download")
    monkeypatch.syspath_prepend(str(checkout / "src"))
    from tabicl._model.tabicl import TabICL
    config = {
        "max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.0,
        "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
        "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise",
        "icl_ssmax": "qassmax-mlp-elementwise",
    }
    checkpoint = tmp_path / "random-tiny-NOT-PRETRAINED.ckpt"
    with torch.random.fork_rng():
        torch.manual_seed(919)
        network = TabICL(**config).eval()
    torch.save({"config": config, "state_dict": network.state_dict()}, checkpoint)
    x = np.random.default_rng(919).normal(size=(16, 5))
    y = np.arange(len(x)) % 2
    model = f.fit_foundation("tabicl", x, y, checkpoint_path=checkpoint, seed=919,
                             device="cpu", n_estimators=8)
    score = model.predict_scores(x[:3])
    assert score.shape == (3,) and np.isfinite(score).all()
    assert model.metadata["actual_estimators"] == 8
    artifact = tmp_path / "official-recipe.joblib"
    model.save(artifact)
    restored = f.load_foundation(artifact)
    np.testing.assert_array_equal(score, restored.predict_scores(x[:3]))
