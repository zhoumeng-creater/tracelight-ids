"""Generated NF-style fields exercise route representations without real data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.context_cache.preprocessing import CSVSchema, PROTECTED_ID_DOMAINS
from ids.route_batch.preprocessing import Representation


@pytest.fixture
def schema():
    identifiers = list(PROTECTED_ID_DOMAINS)
    return CSVSchema.from_dict({
        "features": [*identifiers, "measurement", "constant"],
        "categorical": identifiers, "numeric": ["measurement", "constant"],
        "label_column": "label", "id_column": "sample_id",
        "group_column": "communication_group", "family_column": "attack_family",
    })


@pytest.fixture
def frame():
    return pd.DataFrame({
        "L4_SRC_PORT": [100] * 12,
        "L4_DST_PORT": [0, 1, 53, 80, 443, 65535] * 2,
        "PROTOCOL": [0, 1, 6, 17, 41, 255] * 2,
        "L7_PROTO": [0, 1, 5, 6, 255, 65535] * 2,
        "measurement": [-100., -10., -1., 0., 1., 10., 100., -5., 5., -2., 2., 0.],
        "constant": [3.] * 12,
        "label": [0, 1] * 6,
        "sample_id": [f"synthetic-{index}" for index in range(12)],
        "communication_group": [f"synthetic-group-{index // 2}" for index in range(12)],
        "attack_family": ["Benign", "Scanning"] * 6,
    })


@pytest.fixture(params=["protected_v2", "log1p_standard"])
def kind(request, monkeypatch):
    if request.param == "protected_v2":
        source = Path(__file__).resolve().parents[1] / "local/vendor/tabicl/src"
        if not (source / "tabicl/_model/tabicl.py").is_file():
            pytest.skip("pinned TabICL source absent; this test does not download it")
        monkeypatch.syspath_prepend(str(source))
    return request.param


def test_all_accepted_identifier_values_remain_distinct_and_recoverable(kind, schema, frame):
    representation = Representation(kind, schema).fit(frame)
    domain = np.arange(65536)
    query = pd.DataFrame({"L4_SRC_PORT": domain, "L4_DST_PORT": domain,
                          "PROTOCOL": domain % 256, "L7_PROTO": domain,
                          "measurement": np.zeros(len(domain)), "constant": np.full(len(domain), 3.)})
    transformed = representation.transform(query)
    assert transformed.dtype == np.float32
    assert transformed.flags.c_contiguous
    assert np.isfinite(transformed).all()
    if kind == "protected_v2":
        mean, scale = representation.processor.protected_mean_, representation.processor.protected_scale_
    else:
        mean, scale = representation.mean[:4], representation.scale[:4]
    for index, (name, (lower, upper)) in enumerate(PROTECTED_ID_DOMAINS.items()):
        assert np.unique(transformed[:, index]).size == upper - lower + 1
        reconstructed = np.rint(transformed[:, index].astype(np.float64) * scale[index] + mean[index])
        np.testing.assert_array_equal(reconstructed, query[name])
    # The source port was constant in T; values outside T still have separate coordinates.
    assert transformed[0, 0] != transformed[65535, 0]


def test_transform_does_not_refit_and_metadata_never_changes_features(kind, schema, frame):
    representation = Representation(kind, schema).fit(frame)
    before = representation.transform(frame)
    report = representation.report()
    changed = frame.copy()
    changed["label"] = 1 - changed["label"]
    changed["sample_id"] = "different"
    changed["communication_group"] = "different"
    changed["attack_family"] = "OtherAttack"
    np.testing.assert_array_equal(representation.transform(changed), before)
    query = frame.iloc[:2].copy()
    query["measurement"] = [1e20, -1e20]
    query["L4_SRC_PORT"] = [0, 65535]
    representation.transform(query)
    np.testing.assert_array_equal(representation.transform(frame), before)
    assert representation.report() == report
    assert report["fit_rows"] == len(frame)
    if kind == "log1p_standard":
        raw = frame[list(schema.features)].to_numpy(dtype=np.float64)
        raw[:, 4:] = np.sign(raw[:, 4:]) * np.log1p(np.abs(raw[:, 4:]))
        np.testing.assert_array_equal(representation.mean, raw.mean(axis=0))
        expected_scale = raw.std(axis=0)
        expected_scale[expected_scale == 0] = 1.
        np.testing.assert_array_equal(representation.scale, expected_scale)
    else:
        assert representation.processor.support_rows_ == len(frame)


def test_signed_log_handles_negative_zero_and_unclipped_extrapolation(schema, frame):
    representation = Representation("log1p_standard", schema).fit(frame)
    query = frame.iloc[:5].copy()
    query["measurement"] = [-1e12, -1., 0., 1., 1e12]
    values = representation.transform(query)[:, 4]
    signed_log = np.sign(query["measurement"]) * np.log1p(np.abs(query["measurement"]))
    expected = ((signed_log.to_numpy() - representation.mean[4]) / representation.scale[4]).astype(np.float32)
    np.testing.assert_array_equal(values, expected)
    assert np.all(np.diff(values) > 0)
    np.testing.assert_array_equal(representation.transform(frame)[:, 5], np.zeros(len(frame), dtype=np.float32))


def test_serialization_preserves_scores_and_report(kind, schema, frame, tmp_path):
    representation = Representation(kind, schema).fit(frame)
    path = tmp_path / "synthetic-representation.joblib"
    representation.save(path)
    loaded = Representation.load(path)
    np.testing.assert_array_equal(loaded.transform(frame), representation.transform(frame))
    assert loaded.report() == representation.report()


@pytest.mark.parametrize("invalid", [-1, 65536, .5, np.nan])
def test_out_of_domain_identifiers_are_rejected(kind, schema, frame, invalid):
    representation = Representation(kind, schema).fit(frame)
    query = frame.copy()
    query["L4_SRC_PORT"] = query["L4_SRC_PORT"].astype(float)
    query.loc[0, "L4_SRC_PORT"] = invalid
    with pytest.raises(ValueError, match="protected ID|finite numeric raw fields"):
        representation.transform(query)


def test_identifier_lexical_variants_have_the_same_integer_identity(kind, schema, frame):
    representation = Representation(kind, schema).fit(frame)
    query = frame.iloc[:3].copy()
    query["L4_SRC_PORT"] = ["00100", "100.0", "100"]
    transformed = representation.transform(query)
    assert transformed[0, 0] == transformed[1, 0] == transformed[2, 0]
