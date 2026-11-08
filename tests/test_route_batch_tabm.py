"""Small synthetic numerical and lifecycle checks for the batch TabM trainer."""

import json

import numpy as np
import pytest

from ids.route_batch import tabm_training as training


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(24, 3)).astype(np.float32)
    y = np.tile([0, 1], 12).astype(np.float32)
    return x[:16], y[:16], x[16:], y[16:]


@pytest.fixture(autouse=True)
def single_thread_torch():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def settings(path, **changes):
    return {"width": 8, "learning_rate": 2e-3, "seed": 17, "output_dir": path,
            "device": "cpu", "epochs": 2, "batch_size": 8, "k": 2,
            "n_blocks": 1, "dropout": 0.0, "max_seconds": 60, **changes}


def test_stable_scores_match_ensemble_and_preserve_saturation():
    import torch

    logits = torch.tensor([[[2.], [-3.]], [[100.], [100.]], [[-100.], [-100.]]])
    score = training.stable_scores(logits)
    probability = logits[0].double().sigmoid().mean()
    assert score.dtype == torch.float64
    assert score[0].item() == pytest.approx((probability.log() - torch.log1p(-probability)).item())
    assert score[1:].tolist() == pytest.approx([100., -100.])
    assert not torch.isclose(score[0], logits[0].double().mean())


def test_complete_trajectory_has_real_endpoints_and_loadable_scores(tiny_data, tmp_path):
    import torch

    x, y, v, vy = tiny_data
    result = training.train_trajectory(x, y, v, vy, checkpoints=(1, 2), **settings(tmp_path / "run"))
    assert result["status"] == "complete"
    assert result["epochs_completed"] == 2 and result["updates"] == 4
    assert set(result["checkpoints"]) == {"1", "2"}
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 2 and all(not item["path_loss_is_endpoint_loss"] for item in history)
    assert result["best_epoch"] == min(history, key=lambda item: item["validation_ensemble_bce"])["epoch"]
    for entry in [*result["checkpoints"].values(), result["best_checkpoint"]]:
        predictor = training.load_predictor(entry["model_path"])
        with np.load(entry["scores_path"]) as saved:
            np.testing.assert_array_equal(predictor.predict_scores(v, batch_size=8), saved["validation_scores"])
            predicted = predictor.predict_scores(x, batch_size=8)
            np.testing.assert_array_equal(predicted, saved["train_scores"])
        manual_loss = np.mean(np.logaddexp(0., predicted) - y * predicted)
        assert entry["train_ensemble_bce"] == pytest.approx(manual_loss, abs=1e-12)
        loaded = torch.load(entry["model_path"], weights_only=True)
        assert loaded["config"]["positive_weight"] == 1.0


def test_refit_is_reinitialized_and_requires_no_validation(tiny_data, tmp_path):
    x, y, v, vy = tiny_data
    result = training.train_fixed(x, y, **settings(tmp_path / "fixed"))
    duplicate = training.train_fixed(x, y, **settings(tmp_path / "repeat"))
    assert result["status"] == "complete" and result["best_epoch"] is None
    assert result["config"]["validation_rows"] == 0
    assert result["model_path"].endswith("model.pt")
    left = training.load_predictor(result["model_path"]).predict_scores(v)
    right = training.load_predictor(duplicate["model_path"]).predict_scores(v)
    np.testing.assert_array_equal(left, right)
    assert training.load_predictor(result["model_path"]).predict_scores(v[:0]).shape == (0,)


def test_timeout_is_not_reported_as_complete(tiny_data, tmp_path):
    x, y, v, vy = tiny_data
    result = training.train_trajectory(x, y, v, vy, **settings(tmp_path / "timeout", max_seconds=1e-12))
    assert result["status"] == "timeout" and result["epochs_completed"] == 0
    assert result["best_epoch"] is None and result["checkpoints"] == {}
    with pytest.raises(ValueError, match="incomplete"):
        training.load_predictor(result["interrupted_model_path"])


def test_invalid_arrays_and_nonempty_output_fail_before_training(tiny_data, tmp_path):
    x, y, v, vy = tiny_data
    with pytest.raises(ValueError, match="float32"):
        training.train_trajectory(x.astype(np.float64), y, v, vy, **settings(tmp_path / "bad"))
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "result.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        training.train_fixed(x, y, **settings(occupied))
    assert (occupied / "result.txt").read_text() == "keep"
