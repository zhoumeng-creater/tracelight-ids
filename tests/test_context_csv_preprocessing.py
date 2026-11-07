"""Raw synthetic CSVs through pinned TabICLv2 support-fitted transforms."""

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
    CSVPreprocessor, CSVSchema, frame_features, labels, read_csv,
)


@pytest.fixture
def schema():
    return CSVSchema.from_dict({
        "features": ["duration", "protocol", "constant", "missing_numeric"],
        "categorical": ["protocol"], "label_column": "label", "id_column": "sample_id",
        "group_column": "session", "family_column": "family",
    })


@pytest.fixture
def frames():
    support = pd.DataFrame({
        "duration": [0.0, 1.5, np.nan, 4.0, 5.0, 9.0, 2.0, 3.0],
        "protocol": ["001", "1", "NA", None, "001", "1", "NA", "001"],
        "constant": [3.0] * 8, "missing_numeric": [np.nan] * 8,
        "label": [0, 1] * 4, "sample_id": [f"support-{i}" for i in range(8)],
        "session": [f"support-session-{i // 2}" for i in range(8)],
        "family": ["normal", "synthetic_attack"] * 4,
    })
    query = support.iloc[:3].copy()
    query["duration"] = [1e6, np.nan, -100.0]
    query["protocol"] = ["unseen-only-in-query", None, "001"]
    query["sample_id"] = ["query-001", "query-002", "query-003"]
    query["session"] = ["query-group"] * 3
    return support, query


@pytest.fixture
def official():
    if importlib.util.find_spec("tabicl") is None:
        pytest.skip("pinned optional TabICL source required; no automatic installation")
    from ids.context_cache.preprocessing import require_pinned_preprocessing
    return require_pinned_preprocessing()[0]


def test_matches_official_single_view_support_and_query(official, schema, frames):
    support, query = frames
    encoded = official.TransformToNumerical().fit(frame_features(support, schema))
    reference = official.EnsembleGenerator(
        classification=True, n_estimators=1, norm_methods="none", feat_shuffle_method="none",
        class_shuffle_method="none", outlier_threshold=4.0, random_state=0,
    ).fit(encoded.transform(frame_features(support, schema)), labels(support, schema))
    expected_support = reference.transform(None, mode="train")["none"][0][0].astype(np.float32)
    expected_query = reference.transform(encoded.transform(frame_features(query, schema)), mode="test")["none"][0][0]
    processor = CSVPreprocessor(schema).fit(support)
    np.testing.assert_array_equal(processor.transform(support), expected_support)
    np.testing.assert_array_equal(processor.transform(query), expected_query.astype(np.float32))
    assert processor.report()["output_features"] == ["protocol", "duration"]
    assert processor.report()["removed_constant_features"] == ["constant", "missing_numeric"]
    assert not hasattr(processor.pipeline_, "X_transformed_")
    assert not hasattr(processor, "X_") and not hasattr(processor, "y_")


def test_queries_do_not_refit_or_change_vocab_statistics(official, schema, frames, monkeypatch):
    support, query = frames
    processor = CSVPreprocessor(schema).fit(support)
    snapshot = joblib.hash((processor.encoder_, processor.feature_filter_, processor.pipeline_))
    categories = processor.encoder_.tfm_.named_transformers_["categorical"].categories_[0]
    assert "unseen-only-in-query" not in categories
    monkeypatch.setattr(official.PreprocessingPipeline, "fit", lambda *a, **kw: pytest.fail("query refitted"))
    output = processor.transform(query)
    assert np.isfinite(output).all() and output.dtype == np.float32
    assert snapshot == joblib.hash((processor.encoder_, processor.feature_filter_, processor.pipeline_))
    altered = query.assign(label="irrelevant", sample_id="irrelevant", family="irrelevant", session="irrelevant")
    np.testing.assert_array_equal(processor.transform(altered), output)
    reversed_columns = query.loc[:, list(reversed(query.columns))]
    np.testing.assert_array_equal(processor.transform(reversed_columns), output)


def test_saved_state_is_sufficient_in_fresh_process(official, schema, frames, tmp_path):
    support, query = frames
    schema = CSVSchema.from_dict({**schema.to_dict(), "label_mapping": {"BENIGN": 0, "DDoS": 1}})
    support = support.assign(label=support["label"].map({0: "BENIGN", 1: "DDoS"}))
    query = query.assign(label=query["label"].map({0: "BENIGN", 1: "DDoS"}))
    processor = CSVPreprocessor(schema).fit(support)
    artifact = processor.save(tmp_path / "preprocessing.joblib")
    state = joblib.load(artifact)
    assert set(state) == {"format", "schema", "encoder", "feature_filter", "pipeline", "report"}
    assert not hasattr(state["pipeline"], "X_transformed_")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    restored = CSVPreprocessor.load(artifact, expected_sha256=digest)
    assert restored.schema.label_mapping == {"BENIGN": 0, "DDoS": 1}
    np.testing.assert_array_equal(restored.transform(query), processor.transform(query))
    with pytest.raises(ValueError, match="SHA-256"):
        CSVPreprocessor.load(artifact, expected_sha256="0" * 64)
    query_path, output_path = tmp_path / "query.csv", tmp_path / "transformed.npy"
    query.to_csv(query_path, index=False)
    script = (
        "import sys,numpy as np; from ids.context_cache.preprocessing import CSVPreprocessor,read_csv; "
        "p=CSVPreprocessor.load(sys.argv[1],expected_sha256=sys.argv[4]); "
        "q=read_csv(sys.argv[2],p.schema); np.save(sys.argv[3],p.transform(q))"
    )
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), *sys.path])
    result = subprocess.run([sys.executable, "-c", script, str(artifact), str(query_path), str(output_path), digest],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    np.testing.assert_array_equal(np.load(output_path), processor.transform(query))


def test_csv_preserves_category_and_id_lexemes(schema, frames, tmp_path):
    support, _ = frames
    support.loc[0, "sample_id"] = "001"
    support.loc[1, "sample_id"] = "1"
    path = tmp_path / "raw.csv"
    support.to_csv(path, index=False)
    loaded = read_csv(path, schema)
    assert loaded["sample_id"].iloc[:2].tolist() == ["001", "1"]
    assert loaded["protocol"].iloc[:3].tolist() == ["001", "1", "NA"]
    assert pd.isna(loaded["protocol"].iloc[3])
    assert loaded["duration"].dtype == np.float64
    np.testing.assert_array_equal(labels(loaded, schema), [0, 1] * 4)
    unlabelled = tmp_path / "unlabelled.csv"
    support.drop(columns=["label", "family"]).to_csv(unlabelled, index=False)
    assert "label" not in read_csv(unlabelled, schema, require_label=False)


@pytest.mark.parametrize("text,match", [
    ("duration,duration,sample_id,label\n1,2,a,0\n", "duplicate CSV headers"),
    ("duration,sample_id,label\n1,a,0,extra\n", "wrong number of fields"),
    ("duration,sample_id,label\nnot-a-number,a,0\n", "invalid numeric"),
    ("duration,sample_id,label\ninf,a,0\n", "infinite numeric"),
    ("duration,sample_id,label\n1,a,0\n2,a,1\n", "unique within"),
    ("duration,sample_id,label\n1,,0\n", "ID values must be nonempty"),
    ("duration,sample_id,label\n1,a,0.5\n", "normal=0 and attack=1"),
    ("duration,sample_id,label\n1,a,\n", "normal=0 and attack=1"),
])
def test_csv_rejects_broken_contracts(text, match, tmp_path):
    schema = CSVSchema.from_dict({"features": ["duration"]})
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_csv(path, schema)


@pytest.mark.parametrize("schema,match", [
    ({"features": ["label"]}, "metadata cannot be model features"),
    ({"features": ["x", "x"]}, "duplicate"),
    ({"features": ["x"], "numeric": ["x"], "categorical": ["x"]}, "partition"),
    ({"features": ["x"], "categorical": ["absent"]}, "partition"),
    ({"features": ["x"], "id_column": "label"}, "different columns"),
])
def test_schema_rejects_metadata_and_type_ambiguity(schema, match):
    with pytest.raises(ValueError, match=match):
        CSVSchema.from_dict(schema)


def test_all_constant_retains_official_first_feature(official):
    schema = CSVSchema.from_dict({"features": ["x", "kind"], "categorical": ["kind"]})
    support = pd.DataFrame({"x": [2.0] * 4, "kind": [None] * 4})
    processor = CSVPreprocessor(schema).fit(support)
    output = processor.transform(pd.DataFrame({"x": [100.0], "kind": ["unknown"]}))
    assert output.shape == (1, 1)
    assert processor.output_features_ == ("kind",)
    assert np.isfinite(output).all()


def test_id_can_be_absent_only_under_the_registered_contract(tmp_path):
    path = tmp_path / "without_id.csv"
    path.write_text("x,label\n1,0\n2,1\n", encoding="utf-8")
    no_id = CSVSchema.from_dict({"features": ["x"], "id_column": None})
    assert read_csv(path, no_id).shape == (2, 2)
    declared_id = CSVSchema.from_dict({"features": ["x"]})
    with pytest.raises(ValueError, match="missing registered ID"):
        read_csv(path, declared_id)
    assert read_csv(path, declared_id, require_label=False).shape == (2, 2)

    grouped = CSVSchema.from_dict({"features": ["x"], "group_column": "session", "family_column": "family"})
    path.write_text("x\n1\n2\n", encoding="utf-8")
    assert read_csv(path, grouped, require_label=False).shape == (2, 1)
    path.write_text("x,session\n1,\n2,query-group\n", encoding="utf-8")
    with pytest.raises(ValueError, match="group values must be nonempty"):
        read_csv(path, grouped, require_label=False)
    path.write_text("x,sample_id,label,family\n1,a,0,normal\n2,b,1,attack\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing registered group"):
        read_csv(path, grouped, require_label=True)


def test_raw_text_labels_map_once_and_reject_unknown_or_empty(tmp_path):
    schema = CSVSchema.from_dict({"features": ["x"], "id_column": None,
                                  "label_mapping": {"BENIGN": 0, "DDoS": 1}})
    path = tmp_path / "text_labels.csv"
    path.write_text("x,label\n1,BENIGN\n2,DDoS\n", encoding="utf-8")
    frame = read_csv(path, schema)
    np.testing.assert_array_equal(frame["label"], [0, 1])
    np.testing.assert_array_equal(labels(frame, schema), [0, 1])
    assert CSVSchema.from_dict(schema.to_dict()).label_mapping == schema.label_mapping
    path.write_text("x,label\n1,unknown-attack\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from label_mapping"):
        read_csv(path, schema)
    path.write_text("x,label\n1,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty registered label_mapping"):
        read_csv(path, schema)

    # Distinguish raw numeric-looking strings from already canonical integers.
    reversed_codes = CSVSchema.from_dict({"features": ["x"], "id_column": None,
                                          "label_mapping": {"0": 1, "1": 0}})
    path.write_text("x,label\n1,0\n2,1\n", encoding="utf-8")
    mapped = read_csv(path, reversed_codes)
    np.testing.assert_array_equal(mapped["label"], [1, 0])
    np.testing.assert_array_equal(labels(mapped, reversed_codes), [1, 0])


@pytest.mark.parametrize("mapping", [{"BENIGN": 2}, {"BENIGN": True}, {0: 0}, {}])
def test_label_mapping_has_explicit_binary_targets(mapping):
    with pytest.raises(ValueError, match="label_mapping"):
        CSVSchema.from_dict({"features": ["x"], "label_mapping": mapping})
