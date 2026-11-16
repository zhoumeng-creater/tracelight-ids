"""Synthetic checks of missingness, identity exclusion and fit-only transforms."""
import json

import numpy as np
import pandas as pd
import pytest

from ids.model_training.preprocessing import CONTINUOUS, TypedPreprocessor


def frame(rows=8):
    result = pd.DataFrame({name: np.arange(rows, dtype=float) for name in CONTINUOUS})
    result["ip_protocol"] = np.resize([1, 6, 17, 58, 132, -1, 2, 255], rows)
    return result


def test_fixed_protocol_and_missing_categories_keep_unseen_known_codes():
    train = frame(4).assign(ip_protocol=6)
    processor = TypedPreprocessor.fit(train)
    numeric, categories = processor.transform(frame())
    assert numeric.shape == (8, 5)
    assert categories.shape == (8, 6)
    assert categories[:, 0].tolist() == [0, 1, 2, 3, 4, 5, 5, 5]
    assert processor.cat_cardinalities == [6, 2, 2, 2, 2, 2]
    assert processor.hgb_matrix(frame()).shape == (8, 21)


def test_missing_sentinel_and_nan_have_masks_but_real_zero_does_not():
    train = frame(5)
    train.loc[:, "duration_seconds"] = [-1, np.nan, 0, 2, 4]
    processor = TypedPreprocessor.fit(train)
    values, cats = processor.transform(train)
    assert processor.median[0] == 2
    assert cats[:, 1].tolist() == [1, 1, 0, 0, 0]
    assert values[0, 0] == values[1, 0] == values[3, 0]
    assert values[0, 0] != values[2, 0]


def test_all_missing_and_constant_columns_preserve_width_and_extrapolate():
    train = frame().assign(duration_seconds=-1, orig_pkts=0.1)
    processor = TypedPreprocessor.fit(train)
    assert processor.all_missing.tolist() == [True, False, False, False, False]
    assert processor.median[0] == 0 and processor.scale[0] == processor.scale[1] == 1
    values, _ = processor.transform(train)
    np.testing.assert_array_equal(values[:, :2], 0)
    future = train.iloc[:1].assign(duration_seconds=1000, orig_pkts=1000)
    numeric, categories = processor.transform(future)
    assert numeric[0, 0] > 6 and numeric[0, 1] > 6
    assert categories[0, 1] == 0


def test_future_data_do_not_change_fit_statistics_and_ports_are_not_inputs():
    train = frame()
    processor = TypedPreprocessor.fit(train)
    before = json.dumps(processor.to_dict(), sort_keys=True)
    future = train.assign(orig_port=99999, resp_port=-100, source_id="future", label=1, weight=400)
    numeric, cats = processor.transform(future)
    expected_numeric, expected_cats = processor.transform(train)
    np.testing.assert_array_equal(numeric, expected_numeric)
    np.testing.assert_array_equal(cats, expected_cats)
    extreme = train.assign(orig_ip_bytes=1e20)
    assert processor.audit(extreme)["outside_fit_range_by_feature"]["orig_ip_bytes"] == len(train)
    processor.transform(extreme)
    assert json.dumps(processor.to_dict(), sort_keys=True) == before


@pytest.mark.parametrize("value", [-2, -0.5, np.inf, -np.inf])
def test_invalid_continuous_values_rejected(value):
    with pytest.raises(ValueError, match="Continuous"):
        TypedPreprocessor.fit(frame().assign(resp_pkts=value))


@pytest.mark.parametrize("value", [-2, 6.5, 256, np.inf])
def test_invalid_protocol_codes_rejected(value):
    with pytest.raises(ValueError, match="Protocol"):
        TypedPreprocessor.fit(frame().assign(ip_protocol=value))


def test_json_roundtrip_and_empty_scoring_shape():
    train = frame()
    processor = TypedPreprocessor.from_dict(json.loads(json.dumps(TypedPreprocessor.fit(train).to_dict())))
    assert processor.transform(train.iloc[:0])[0].shape == (0, 5)
    assert processor.hgb_matrix(train.iloc[:0]).shape == (0, 21)
    np.testing.assert_array_equal(processor.transform(train)[0], TypedPreprocessor.fit(train).transform(train)[0])
    state = processor.to_dict()
    state["scale"][0] = 0
    with pytest.raises(ValueError, match="Invalid stored"):
        TypedPreprocessor.from_dict(state)


def test_audit_reports_fp32_input_merges_without_dropping_rows():
    processor = TypedPreprocessor.fit(frame())
    future = frame(2).assign(ip_protocol=6)
    future.loc[1, list(CONTINUOUS)] = 1e8 + 1
    future.loc[0, list(CONTINUOUS)] = 1e8
    audit = processor.audit(future)
    assert audit["rows"] == 2 and audit["unique_typed_inputs"] == 2
    assert audit["unique_standardized_float64_inputs"] == 2
    assert audit["unique_standardized_float32_inputs"] == 1
