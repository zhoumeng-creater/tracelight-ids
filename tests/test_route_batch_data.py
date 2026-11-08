"""Input identity and grouped split checks use only generated rows."""

import json

import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file
from ids.context_cache.preprocessing import CSVSchema
from ids.route_batch.data import load_inputs, split_support


@pytest.fixture
def schema():
    return CSVSchema.from_dict({"features": ["value", "port"], "numeric": ["value"],
                               "categorical": ["port"], "label_column": "label",
                               "id_column": "sample_id", "group_column": "communication_group",
                               "family_column": "attack_family"})


def support_frame():
    return pd.DataFrame({"value": np.arange(160, dtype=float), "port": ["80"] * 160,
                         "label": [0, 1] * 80,
                         "sample_id": [f"p-{i}" for i in range(160)],
                         "communication_group": [f"g-{i // 2}" for i in range(160)],
                         "attack_family": ["Benign", "Scanning"] * 80})


@pytest.fixture
def registered(tmp_path, schema):
    support = support_frame()
    calibration = support.iloc[:8].copy().assign(label=0, attack_family="Benign")
    calibration["sample_id"] = [f"c-{i}" for i in range(len(calibration))]
    calibration["communication_group"] = [f"c-g-{i}" for i in range(len(calibration))]
    evaluation = support.iloc[8:16].copy()
    evaluation["sample_id"] = [f"e-{i}" for i in range(len(evaluation))]
    evaluation["communication_group"] = [f"e-g-{i}" for i in range(len(evaluation))]
    spec = {"protocol": "synthetic-route-test", "schema": schema.to_dict(), "files": {},
            "artifacts": {"preprocessor": {"path": "does-not-exist.joblib", "sha256": "a" * 64}},
            "checkpoint": {"path": "does-not-exist.ckpt", "sha256": "b" * 64}}
    for role, frame in (("support", support), ("calibration", calibration), ("evaluation", evaluation)):
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        spec["files"][role] = {"path": path.name, "sha256": sha256_file(path), "rows": len(frame)}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec), encoding="utf-8")
    return config, spec


def rewrite_role(config, spec, role, change):
    path = config.parent / spec["files"][role]["path"]
    frame = pd.read_csv(path)
    change(frame)
    frame.to_csv(path, index=False)
    spec["files"][role]["sha256"] = sha256_file(path)
    config.write_text(json.dumps(spec), encoding="utf-8")


def test_load_only_registered_csvs_preserves_config_and_reports_equal_inputs(registered):
    config, spec = registered
    before = config.read_bytes()
    result = load_inputs(config)
    assert result["preprocessor"] is None
    assert result["spec"] == spec
    assert config.read_bytes() == before
    assert set(result["frames"]) == {"support", "calibration", "evaluation"}
    assert result["paths"]["checkpoint"].name == "does-not-exist.ckpt"
    assert result["audit"]["cross_role_pairs"]["support/calibration"]["exact_feature_overlap"] == 8


def test_load_rejects_modified_csv_and_wrong_rows(registered):
    config, spec = registered
    path = config.parent / "support.csv"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_inputs(config)
    path.write_bytes(original)
    spec["files"]["support"]["sha256"] = sha256_file(path)
    spec["files"]["support"]["rows"] += 1
    config.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="row count"):
        load_inputs(config)


@pytest.mark.parametrize("kind", ["normal", "group", "id"])
def test_load_rejects_calibration_attacks_and_role_overlap(registered, kind):
    config, spec = registered
    column, value = {"normal": ("label", 1), "group": ("communication_group", "g-0"),
                     "id": ("sample_id", "p-0")}[kind]
    def change(frame):
        frame.loc[0, column] = value

    rewrite_role(config, spec, "calibration", change)
    with pytest.raises(ValueError, match="normal rows only|cross-role overlap"):
        load_inputs(config)


def test_optional_preprocessor_loading_is_explicit_and_hash_bound(registered, monkeypatch):
    config, spec = registered
    captured = {}

    def fake_load(path, *, expected_sha256):
        captured.update(path=path, sha256=expected_sha256)
        return type("Processor", (), {"schema": CSVSchema.from_dict(spec["schema"])})()

    monkeypatch.setattr("ids.route_batch.data.CSVPreprocessor.load", fake_load)
    result = load_inputs(config, load_preprocessor=True)
    assert captured["sha256"] == "a" * 64
    assert result["preprocessor"].schema.to_dict() == spec["schema"]


def test_split_joins_transitive_group_and_feature_components(schema):
    frame = support_frame()
    # g-0 joins rows 0/1; duplicate 1/2 joins g-1; duplicate 3/4 joins g-2.
    frame.loc[2, ["value", "port"]] = frame.loc[1, ["value", "port"]].to_numpy()
    frame.loc[4, ["value", "port"]] = frame.loc[3, ["value", "port"]].to_numpy()
    before = frame.copy(deep=True)
    train, valid, audit = split_support(frame, schema, 20261108)
    assert all(index in train for index in range(6)) or all(index in valid for index in range(6))
    assert audit["component_count"] == 78
    assert audit["largest_component_rows"] == 6
    assert audit["group_overlap"] == audit["exact_feature_overlap"] == 0
    assert set(train) | set(valid) == set(range(len(frame)))
    assert not set(train) & set(valid)
    pd.testing.assert_frame_equal(frame, before)


def test_split_membership_is_order_independent_and_ignores_labels_and_families(schema):
    frame = support_frame()
    _, _, first = split_support(frame, schema, 20261108)
    changed = frame.sample(frac=1, random_state=5).copy()
    changed["label"] = 1 - changed["label"]
    changed["attack_family"] = "irrelevant"
    _, _, second = split_support(changed, schema, 20261108)
    assert first["split_ids_sha256"] == second["split_ids_sha256"]
    assert set(first["train_ids"]) == set(second["train_ids"])
    assert set(first["validation_ids"]) == set(second["validation_ids"])
    assert first["ordered_support_ids_sha256"] != second["ordered_support_ids_sha256"]
    assert first["assignment_uses_labels"] is first["assignment_uses_families"] is False
    assert first["preprocessor_fits"] == 0
    json.dumps(first, allow_nan=False)


def test_single_connected_component_fails_without_retry_or_rebalancing(schema):
    frame = support_frame().assign(communication_group="one-component")
    with pytest.raises(ValueError, match="insufficient class support"):
        split_support(frame, schema, 20261108)


def test_split_requires_both_classes_in_each_partition(schema):
    frame = support_frame().assign(label=0)
    with pytest.raises(ValueError, match="insufficient class support"):
        split_support(frame, schema, 20261108)


def test_split_retains_categorical_lexical_identity_and_numeric_equivalence(schema):
    frame = support_frame()
    frame.loc[2, "value"] = frame.loc[0, "value"]
    frame.loc[2, "port"] = "080"
    _, _, audit = split_support(frame, schema, 20261108)
    assert audit["component_count"] == 80
    frame.loc[2, "port"] = "80"
    _, _, joined = split_support(frame, schema, 20261108)
    assert joined["component_count"] == 79


@pytest.mark.parametrize("keyword,value", [("seed", True), ("seed", -1),
                                           ("val_fraction", 0), ("val_fraction", 1),
                                           ("val_fraction", float("nan")), ("min_class_rows", 0)])
def test_split_rejects_invalid_settings(schema, keyword, value):
    settings = {"seed": 20261108, keyword: value}
    with pytest.raises(ValueError):
        split_support(support_frame(), schema, **settings)
