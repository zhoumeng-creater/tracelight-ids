"""Real official CatBoost on synthetic rows; never benchmark data."""
import numpy as np
import pandas as pd
import pytest

from ids.native_trees import load_native_tree, mixed_targets, native_frame, train_native_tree


SCHEMA = {"name": "native-tree-toy", "features": ["x", "constant", "protocol"],
          "categorical": ["protocol"], "discrete": ["protocol"]}
CONFIG = {"iterations": 12, "depth": 2, "learning_rate": .1, "l2_leaf_reg": 3., "border_count": 16}


@pytest.fixture
def frame():
    rng = np.random.default_rng(91)
    x = rng.normal(size=96)
    protocol = np.where(np.arange(96) % 3 == 0, "udp", "tcp").astype(object)
    protocol[1] = None
    result = pd.DataFrame({"x": x, "constant": 1., "protocol": protocol,
                           "label": (x > 0).astype(int), "attack_cat": "toy",
                           "sample_id": [f"s{i}" for i in range(len(x))]})
    result.loc[2, "x"] = np.nan
    return result


@pytest.fixture
def official_package():
    package = pytest.importorskip("catboost")
    assert package.__version__ == "1.2.10"


def test_mixture_gradient_identity():
    y, q = np.array([0, 1, 0, 1]), np.array([.1, .9, .4, .6])
    p, weight = np.array([.2, .8, .3, .7]), .3
    target = mixed_targets(y, q, weight)
    np.testing.assert_allclose((p - y) + weight * (p - q), (1 + weight) * (p - target))
    np.testing.assert_array_equal(mixed_targets(y, y, weight), y)


@pytest.mark.parametrize("q", [[.2], [0., 1., -1., .2], [0., 1., np.nan, .2]])
def test_bad_soft_targets_rejected(q):
    with pytest.raises(ValueError):
        mixed_targets([0, 1, 0, 1], q, .3)


def test_native_transform_excludes_metadata_and_keeps_missing_distinct(frame):
    probe = frame.iloc[:3].copy()
    probe["protocol"] = [None, "m:", "<MISSING>"]
    values = native_frame(probe, SCHEMA["features"], SCHEMA["categorical"])
    assert list(values) == SCHEMA["features"]
    assert values.protocol.tolist() == ["m:", "v:m:", "v:<MISSING>"]
    assert np.isnan(values.x.iloc[2])


def test_real_fit_reload_and_unseen_categories(frame, official_package, tmp_path):
    bundle = train_native_tree(frame, CONFIG, 17, 1, SCHEMA)
    probe = frame.iloc[:8].drop(columns=["label", "attack_cat", "sample_id"]).copy()
    probe["protocol"] = "new-protocol"
    raw = bundle.predict_raw(probe)
    assert raw.dtype == np.float64 and np.isfinite(raw).all()
    assert bundle.predict_raw(probe.iloc[:0]).size == 0
    sizes = bundle.save(tmp_path / "model")
    assert sizes["deployment_bytes"] > sizes["model_bytes"] > 0
    restored = load_native_tree(tmp_path / "model")
    np.testing.assert_array_equal(raw, restored.predict_raw(probe))
    assert restored.training_statistics["fit_rows"] == len(frame)
    assert restored.training_statistics["tree_count"] == CONFIG["iterations"]
    assert restored.training_statistics["selection_used"] is False
    with pytest.raises(FileExistsError):
        bundle.save(tmp_path / "model")


def test_hard_and_identical_teacher_are_same_fit(frame, official_package):
    a = train_native_tree(frame, CONFIG, 17, 1, SCHEMA)
    b = train_native_tree(frame, CONFIG, 17, 1, SCHEMA,
                          teacher_probabilities=frame.label.to_numpy(), kd_weight=.3)
    np.testing.assert_array_equal(a.predict_raw(frame), b.predict_raw(frame))
    assert a.training_statistics["weight_source"] == b.training_statistics["weight_source"]


def test_soft_targets_remain_continuous_and_change_fit(frame, official_package):
    a = train_native_tree(frame, CONFIG, 17, 1, SCHEMA)
    q = .1 + .8 * frame.label.to_numpy()
    b = train_native_tree(frame, CONFIG, 17, 1, SCHEMA, teacher_probabilities=q, kd_weight=1.)
    assert b.training_statistics["parameters"]["loss_function"] == "CrossEntropy"
    assert b.training_statistics["target_mean_abs_difference_from_hard"] == pytest.approx(.05)
    assert not np.allclose(a.predict_raw(frame), b.predict_raw(frame))


def test_feature_subset_and_no_label_dependency_in_predict(frame, official_package):
    model = train_native_tree(frame, CONFIG, 17, 1, SCHEMA, selected_features=["x"])
    np.testing.assert_array_equal(model.predict_raw(frame), model.predict_raw(frame[["x"]]))
    with pytest.raises(ValueError):
        train_native_tree(frame, CONFIG, 17, 1, SCHEMA, selected_features=["label"])


def test_saved_model_damage_is_detected(frame, official_package, tmp_path):
    model = train_native_tree(frame, CONFIG, 17, 1, SCHEMA)
    model.save(tmp_path)
    with (tmp_path / "model.cbm").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_native_tree(tmp_path)
