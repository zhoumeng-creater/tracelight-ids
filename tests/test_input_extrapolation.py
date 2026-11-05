"""A frozen input intervention must not change F, categories, or weights."""
import copy

import numpy as np
import pandas as pd
import pytest

from ids.input_extrapolation import FRangeClipper, benchmark_pair, evaluate_intervention, paired_events
from ids.models import FeaturePreprocessor, ModelBundle, _network, load_bundle, save_bundle
from ids.risk_research import _state

SCHEMA = {"name": "range-synthetic", "features": ["x", "constant", "cat"],
          "categorical": ["cat"], "discrete": ["cat"]}


def frame(xs, prefix="F"):
    n = len(xs)
    return pd.DataFrame({"x": xs, "constant": np.ones(n), "cat": ["a", "b"] * (n // 2),
        "label": [0, 1] * (n // 2), "attack_cat": ["Normal", "Attack"] * (n // 2),
        "sample_id": [prefix + str(i) for i in range(n)], "feature_hash": [prefix + str(i) for i in range(n)]})


def test_bounds_use_f_inputs_only_and_preserve_categories_nan_and_constant():
    fitting = frame([0., 1., 2., np.nan])
    base = FeaturePreprocessor.fit(fitting, SCHEMA["features"], SCHEMA)
    bounded = FRangeClipper.from_fitting_inputs(base, fitting)
    np.testing.assert_array_equal(bounded.transform(fitting), base.transform(fitting))
    changed_labels = fitting.assign(label=1-fitting.label)
    other = FRangeClipper.from_fitting_inputs(base, changed_labels)
    np.testing.assert_array_equal(bounded.lower, other.lower)
    np.testing.assert_array_equal(bounded.upper, other.upper)
    heldout = frame([-100., 100., np.nan, 1.], "C").assign(constant=[-7., 99., np.nan, 1.])
    heldout.loc[0, "cat"] = "unseen"
    original, clipped = base.transform(heldout), bounded.transform(heldout)
    np.testing.assert_array_equal(original[:, 2:], clipped[:, 2:])
    np.testing.assert_array_equal(clipped[:, 1], np.zeros(4))
    assert clipped[0, 0] == bounded.lower[0] and clipped[1, 0] == bounded.upper[0]
    np.testing.assert_array_equal(clipped[2:, 0], original[2:, 0])
    assert bounded.range_report(heldout)["outside_any_numeric_range"] == 2
    assert bounded.transform(heldout.iloc[:0]).shape == (0, base.encoded_dim)
    np.testing.assert_array_equal(base.transform(fitting), bounded.base.transform(fitting))


def test_cannot_fit_bounds_without_f_or_from_already_bounded_transform():
    fitting = frame([0., 1., 2., 3.])
    base = FeaturePreprocessor.fit(fitting, SCHEMA["features"], SCHEMA)
    with pytest.raises(ValueError, match="nonempty"):
        FRangeClipper.from_fitting_inputs(base, fitting.iloc[:0])
    with pytest.raises(ValueError, match="original"):
        FRangeClipper.from_fitting_inputs(FRangeClipper.from_fitting_inputs(base, fitting), fitting)


def test_paired_counts_keep_gains_losses_and_input_pattern_concentration():
    f = frame([0., 1., 2., 3.]).assign(group_id=["g0", "g1", "g0", "g1"])
    result = paired_events(f, [0., 0., 1., 1.], [1., 1., 0., 0.], .5, .5)
    assert all(result["all"][k] == 1 for k in ("recovered_fn", "new_fn", "repaired_fp", "new_fp"))
    assert result["families"]["Normal"]["new_fp"] == 1
    assert result["all"]["recovered_fn_feature_patterns"] == 1
    assert result["group_column"] == "group_id"
    assert result["all"]["new_fp_group_patterns"] == [
        {"group_id": "g0", "feature_hash": "F0", "attack_cat": "Normal", "rows": 1}]


def test_frozen_intervention_two_thresholds_and_bundle_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    fitting = frame([0., 1., 2., 3., 4., 5.])
    base = FeaturePreprocessor.fit(fitting, SCHEMA["features"], SCHEMA)
    with torch.random.fork_rng():
        torch.manual_seed(17)
        net = _network(base.encoded_dim, (3, 2))
    bundle = ModelBundle("mlp", base, SCHEMA["features"], base.encoded_dim, [],
                         network_state=_state(net), hidden_dims=(3, 2), feature_schema=SCHEMA)
    weights = copy.deepcopy(bundle.network_state)
    frames = {"fit": fitting, "reference": frame([-30., 40., .5, 1.5], "R"),
              "constraint": frame([-50., 70., 2.5, 3.5], "C")}
    candidate, report, scores = evaluate_intervention(bundle, frames, .1, .25, 2)
    assert report["fitting_encoding_unchanged"]
    assert set(report["results"]) == {"frozen_original_threshold", "registered_R_C_threshold"}
    for role in ("reference", "constraint"):
        assert report["results"]["registered_R_C_threshold"][role]["metrics"]["fpr"] <= .25
        assert len(scores["original_" + role]) == len(frames[role])
    for key in weights:
        torch.testing.assert_close(weights[key], candidate.network_state[key], rtol=0, atol=0)
        torch.testing.assert_close(weights[key], bundle.network_state[key], rtol=0, atol=0)
    path = tmp_path / "diagnostic.joblib"
    save_bundle(candidate, path)
    restored = load_bundle(path)
    np.testing.assert_array_equal(restored.predict_scores(frames["constraint"]), candidate.predict_scores(frames["constraint"]))
    timing = benchmark_pair(bundle, candidate, frames["constraint"], repeats=2, blocks=2)
    for batch in timing["batches"].values():
        assert len(batch["sample_ids"]) == len(batch["paired_duration_ratios"]) == 4
        assert all(sorted(order) == [0, 1] for order in batch["execution_order"])
        assert len(batch["original"]["blocks"]) == len(batch["clipped"]["blocks"]) == 2
        assert len(batch["original"]["durations_ms"]) == len(batch["clipped"]["durations_ms"]) == 4
