"""Synthetic checks for the explicit NF-v3 identifier input contract."""

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from ids.context_cache.preprocessing import (
    CSVPreprocessor, CSVSchema, OFFICIAL_PROFILE, PROTECTED_IDS_PROFILE,
    PROTECTED_ID_DOMAINS, frame_features, require_pinned_preprocessing,
)


@pytest.fixture
def official():
    if importlib.util.find_spec("tabicl") is None:
        pytest.skip("pinned optional TabICL source required; no automatic installation")
    return require_pinned_preprocessing()[0]


@pytest.fixture
def schema():
    identifiers = list(PROTECTED_ID_DOMAINS)
    return CSVSchema.from_dict({"features": identifiers + ["bytes", "constant"],
                                "categorical": identifiers})


@pytest.fixture
def support():
    return pd.DataFrame({
        "L4_SRC_PORT": ["0", "1", "2", "80", "443", "1", "2", "80"],
        "L4_DST_PORT": ["80", "80", "443", "443", "80", "443", "80", "80"],
        "PROTOCOL": ["6", "17", "6", "17", "6", "17", "6", "17"],
        "L7_PROTO": ["0", "1", "3", "7", "91", "1", "3", "7"],
        "bytes": [0., 1., 2., np.nan, 4., 8., 16., 32.],
        "constant": [9.] * 8,
    })


def test_unseen_identifiers_stay_distinct_without_refitting(official, schema, support, monkeypatch):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    query = pd.concat([support.iloc[[0]]] * 2, ignore_index=True)
    for name in PROTECTED_ID_DOMAINS:
        query[name] = ["200", "201"]
    before = joblib.hash((processor.encoder_, processor.feature_filter_, processor.pipeline_,
                          processor.protected_mean_, processor.protected_scale_))
    monkeypatch.setattr(official.PreprocessingPipeline, "fit", lambda *a, **k: pytest.fail("refit"))
    encoded = processor.transform(query)
    assert np.all(encoded[0, :4] != encoded[1, :4])
    assert encoded.dtype == np.float32 and encoded.flags.c_contiguous
    assert before == joblib.hash((processor.encoder_, processor.feature_filter_, processor.pipeline_,
                                 processor.protected_mean_, processor.protected_scale_))
    np.testing.assert_array_equal(encoded, processor.transform(query.assign(label=[1, 0])))
    assert processor.report()["output_features"] == [*PROTECTED_ID_DOMAINS, "bytes"]


def test_support_constant_ids_are_retained_even_when_all_numeric_columns_are_constant(official, schema, support):
    fixed = pd.concat([support.iloc[[0]]] * 4, ignore_index=True).fillna(0)
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(fixed)
    query = pd.concat([fixed.iloc[[0]]] * 2, ignore_index=True)
    for name, (lower, upper) in PROTECTED_ID_DOMAINS.items():
        query[name] = [str(lower), str(upper)]
    encoded = processor.transform(query)
    assert processor.pipeline_ is None
    assert processor.output_features_ == tuple(PROTECTED_ID_DOMAINS)
    assert processor.protected_support_constant_.all()
    assert np.all(encoded[0] != encoded[1])
    np.testing.assert_array_equal(processor.protected_scale_, np.ones(4))
    assert encoded.shape == (2, 4)


@pytest.mark.parametrize("name", list(PROTECTED_ID_DOMAINS))
def test_every_accepted_integer_has_unique_fp32_encoding_and_exact_rounded_inverse(official, schema, support, name):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    index = processor.schema.categorical.index(name)
    lower, upper = PROTECTED_ID_DOMAINS[name]
    domain = np.arange(lower, upper + 1, dtype=np.float64)
    encoded = ((domain - processor.protected_mean_[index]) / processor.protected_scale_[index]).astype(np.float32)
    assert np.unique(encoded).size == domain.size
    recovered = np.rint(encoded.astype(np.float64) * processor.protected_scale_[index] + processor.protected_mean_[index])
    np.testing.assert_array_equal(recovered, domain)
    evidence = processor.report()["protected_ids"]["domain_verification"][name]
    assert evidence["domain_size"] == upper - lower + 1
    assert evidence["fp32_unique"] and evidence["rounded_inverse_exact"]


@pytest.mark.parametrize("value", [None, np.nan, "", "NaN", "inf", "-1", "65536", "1.5", "1.0000000000000000001"])
@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_invalid_protected_identifiers_fail_at_fit_and_query(official, schema, support, value, stage):
    bad = support.copy()
    bad.loc[0, "L4_SRC_PORT"] = value
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE)
    with pytest.raises(ValueError, match="protected ID L4_SRC_PORT"):
        if stage == "fit":
            processor.fit(bad)
        else:
            processor.fit(support).transform(bad)


def test_protocol_is_byte_but_l7_uses_explicit_project_uint16_domain(official, schema, support):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    query = support.iloc[[0]].copy()
    query["L7_PROTO"] = "65535"
    assert np.isfinite(processor.transform(query)).all()
    query["PROTOCOL"] = "256"
    with pytest.raises(ValueError, match="PROTOCOL"):
        processor.transform(query)


def test_numeric_identifier_lexemes_have_one_canonical_integer_identity(official, schema, support):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    query = pd.concat([support.iloc[[0]]] * 3, ignore_index=True)
    query["L4_SRC_PORT"] = ["001", "1.0", "1e0"]
    encoded = processor.transform(query)
    np.testing.assert_array_equal(encoded[0], encoded[1])
    np.testing.assert_array_equal(encoded[1], encoded[2])


def test_numeric_44_column_branch_matches_original_official_chain(official, support):
    identifiers = list(PROTECTED_ID_DOMAINS)
    numeric = [f"numeric_{i}" for i in range(44)]
    schema = CSVSchema.from_dict({"features": identifiers + numeric, "categorical": identifiers})
    frame = support.loc[:, identifiers].copy()
    for index, name in enumerate(numeric):
        frame[name] = np.arange(len(frame), dtype=float) * index
    frame.loc[2, "numeric_3"] = np.nan
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(frame)
    numeric_frame = frame_features(frame, schema).loc[:, numeric]
    encoder = official.TransformToNumerical(verbose=False).fit(numeric_frame)
    encoded = encoder.transform(numeric_frame)
    feature_filter = official.UniqueFeatureFilter().fit(encoded)
    filtered = feature_filter.transform(encoded)
    pipeline = official.PreprocessingPipeline(normalization_method="none", outlier_threshold=4., random_state=0).fit(filtered)
    query = frame.copy()
    query.loc[0, "numeric_2"] = 1e12
    expected = pipeline.transform(feature_filter.transform(encoder.transform(frame_features(query, schema).loc[:, numeric])))
    np.testing.assert_array_equal(processor.transform(query)[:, 4:], expected.astype(np.float32))
    assert processor.report()["removed_constant_features"] == ["numeric_0"]
    assert not hasattr(processor.pipeline_, "X_transformed_")


def test_saved_v2_state_restores_in_fresh_process(official, schema, support, tmp_path):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    artifact = processor.save(tmp_path / "preprocessor.joblib")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    restored = CSVPreprocessor.load(artifact, expected_sha256=digest)
    assert restored.profile == PROTECTED_IDS_PROFILE
    assert restored.report() == processor.report()
    query = support.iloc[:2].copy()
    query["L4_SRC_PORT"] = ["60000", "60001"]
    np.testing.assert_array_equal(restored.transform(query), processor.transform(query))
    query_path, output = tmp_path / "queries.csv", tmp_path / "encoded.npy"
    query.to_csv(query_path, index=False)
    script = (
        "import sys,numpy as np; from ids.context_cache.preprocessing import CSVPreprocessor,read_csv; "
        "p=CSVPreprocessor.load(sys.argv[1],expected_sha256=sys.argv[4]); "
        "q=read_csv(sys.argv[2],p.schema,require_label=False); np.save(sys.argv[3],p.transform(q))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"), *sys.path])
    result = subprocess.run([sys.executable, "-c", script, str(artifact), str(query_path), str(output), digest],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(output), processor.transform(query))


def test_legacy_v1_state_without_profile_is_still_loadable(official, schema, support, tmp_path):
    processor = CSVPreprocessor(schema).fit(support)
    artifact = processor.save(tmp_path / "legacy.joblib")
    state = joblib.load(artifact)
    assert set(state) == {"format", "schema", "encoder", "feature_filter", "pipeline", "report"}
    state["report"].pop("profile")
    joblib.dump(state, artifact)
    restored = CSVPreprocessor.load(artifact)
    assert restored.profile == OFFICIAL_PROFILE
    np.testing.assert_array_equal(restored.transform(support), processor.transform(support))


def test_v2_profile_and_affine_state_are_bound_on_load(official, schema, support, tmp_path):
    processor = CSVPreprocessor(schema, profile=PROTECTED_IDS_PROFILE).fit(support)
    artifact = processor.save(tmp_path / "bad.joblib")
    state = joblib.load(artifact)
    state["profile"] = OFFICIAL_PROFILE
    joblib.dump(state, artifact)
    with pytest.raises(ValueError, match="format/profile"):
        CSVPreprocessor.load(artifact)
    processor.save(artifact)
    state = joblib.load(artifact)
    state["protected_scale"][0] = 0
    joblib.dump(state, artifact)
    with pytest.raises(ValueError, match="affine state"):
        CSVPreprocessor.load(artifact)


def test_explicit_profile_and_identifier_schema_are_required(schema):
    with pytest.raises(ValueError, match="unknown preprocessing profile"):
        CSVPreprocessor(schema, profile="inferred")
    with pytest.raises(ValueError, match="four NF-v3"):
        CSVPreprocessor({"features": ["x"]}, profile=PROTECTED_IDS_PROFILE)
