import json

import numpy as np
import pandas as pd
import pytest

from ids.data import (
    ATTACK_CATEGORIES, CATEGORICAL, FEATURES, load_partition, load_unsw_csv,
    make_synthetic_dataset, prepare_dataset,
)


@pytest.fixture
def sources(tmp_path):
    return make_synthetic_dataset(tmp_path / "raw", n_train=300, n_test=120)


def test_schema_contract_and_synthetic_provenance(sources, tmp_path):
    assert len(FEATURES) == 42
    assert len(CATEGORICAL) == 3
    assert not {"id", "label", "attack_cat", "sample_id", "feature_hash"} & set(FEATURES)
    train, test = sources
    manifest = prepare_dataset(train, test, tmp_path / "prepared", data_kind="synthetic")
    assert manifest["data_kind"] == "synthetic"
    assert manifest["provenance_status"] == "synthetic_fixture"
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["source_files"]["train"]["n_rows"] == 300
    assert manifest["partitions"]["test"]["n_rows"] == 120
    assert manifest["exact_proportions_guaranteed"] is False
    all_training_ids = set()
    for split, target in [("train", .7), ("validation", .15), ("calibration", .15)]:
        frame = load_partition(tmp_path / "prepared", split)
        assert abs(len(frame) / 300 - target) < .04
        assert set(frame.attack_cat) == set(ATTACK_CATEGORIES)
        assert not all_training_ids & set(frame.sample_id)
        all_training_ids.update(frame.sample_id)
    assert all_training_ids == set(load_unsw_csv(train).sample_id)
    pd.testing.assert_frame_equal(load_partition(tmp_path / "prepared", "test"),
                                  load_unsw_csv(test), check_flags=False)


def test_repeated_rows_and_conflicting_labels_stay_together(sources, tmp_path):
    train, test = sources
    frame = pd.read_csv(train, dtype=str, keep_default_na=False)
    base = frame.loc[frame.label == "0"].iloc[[0]].copy()
    conflicting = base.copy()
    conflicting["id"] = "301"
    conflicting["label"] = "1"
    conflicting["attack_cat"] = "DoS"
    pd.concat([frame, conflicting], ignore_index=True).to_csv(train, index=False)
    prepared = tmp_path / "prepared"
    prepare_dataset(train, test, prepared, data_kind="synthetic")
    loaded = {name: load_partition(prepared, name) for name in ("train", "validation", "calibration")}
    hashes = {name: set(part.feature_hash) for name, part in loaded.items()}
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["calibration"]
    assert not hashes["validation"] & hashes["calibration"]
    assert sum(len(part) for part in loaded.values()) == 301
    audit = json.loads((prepared / "audit.json").read_text())
    assert audit["train_source"]["conflicting_label_groups"] == 1
    assert audit["train_source"]["rows_removed"] == 0


def test_train_test_overlap_is_audited_without_deletion(sources, tmp_path):
    train, test = sources
    left = pd.read_csv(train, dtype=str, keep_default_na=False)
    right = pd.read_csv(test, dtype=str, keep_default_na=False)
    right.loc[0, list(FEATURES) + ["attack_cat", "label"]] = left.loc[0, list(FEATURES) + ["attack_cat", "label"]]
    right.to_csv(test, index=False)
    prepared = tmp_path / "prepared"
    manifest = prepare_dataset(train, test, prepared, data_kind="synthetic")
    audit = json.loads((prepared / "audit.json").read_text())
    assert audit["train_test_overlap"]["n_feature_groups"] == 1
    assert audit["train_test_overlap"]["rows_removed"] == 0
    assert manifest["partitions"]["test"]["n_rows"] == 120


def test_split_is_reproducible_but_changes_with_seed(sources, tmp_path):
    first, second, third = [tmp_path / name for name in ("first", "second", "third")]
    prepare_dataset(*sources, first, seed=5, data_kind="synthetic")
    prepare_dataset(*sources, second, seed=5, data_kind="synthetic")
    prepare_dataset(*sources, third, seed=6, data_kind="synthetic")
    one = load_partition(first, "train").sample_id.tolist()
    assert one == load_partition(second, "train").sample_id.tolist()
    assert one != load_partition(third, "train").sample_id.tolist()


def test_missing_values_retained_and_inference_needs_no_labels(sources, tmp_path):
    frame = pd.read_csv(sources[0], dtype=str, keep_default_na=False).loc[:4, list(FEATURES)]
    frame.loc[0, "dur"] = ""
    frame.loc[1, "service"] = ""
    path = tmp_path / "inference.csv"
    frame.to_csv(path, index=False)
    loaded = load_unsw_csv(path, require_labels=False)
    assert np.isnan(loaded.loc[0, "dur"])
    assert pd.isna(loaded.loc[1, "service"])
    assert "label" not in loaded
    assert loaded.sample_id.is_unique
    with pytest.raises(ValueError, match="missing"):
        load_unsw_csv(path)


@pytest.mark.parametrize("mutation,match", [
    ("extra", "unexpected"), ("missing", "missing"), ("duplicate", "Duplicate"),
    ("infinite", "Infinite"), ("text_numeric", "Non-numeric"),
    ("bad_label", "binary"), ("mismatch_label", "disagree"),
    ("unknown_category", "Unknown attack_cat"), ("repeated_id", "unique integer"),
])
def test_malformed_inputs_fail_closed(sources, tmp_path, mutation, match):
    frame = pd.read_csv(sources[0], dtype=str, keep_default_na=False).iloc[:5].copy()
    if mutation == "extra":
        frame["leaking_target"] = 1
    elif mutation == "missing":
        frame = frame.drop(columns="dur")
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame[["dur"]]], axis=1)
    elif mutation == "infinite":
        frame.loc[0, "dur"] = "inf"
    elif mutation == "text_numeric":
        frame.loc[0, "dur"] = "not-a-number"
    elif mutation == "bad_label":
        frame.loc[0, "label"] = "0.5"
    elif mutation == "mismatch_label":
        frame.loc[0, ["label", "attack_cat"]] = ["1", "Normal"]
    elif mutation == "unknown_category":
        frame.loc[0, "attack_cat"] = "UnknownAttack"
    elif mutation == "repeated_id":
        frame.loc[1, "id"] = frame.loc[0, "id"]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_unsw_csv(path)


def test_feature_hash_excludes_metadata_and_source_name(sources, tmp_path):
    source = sources[0]
    same_bytes = tmp_path / "renamed.csv"
    same_bytes.write_bytes(source.read_bytes())
    one, two = load_unsw_csv(source), load_unsw_csv(same_bytes)
    assert one.sample_id.tolist() == two.sample_id.tolist()
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame["id"] = np.arange(10000, 10000 + len(frame))
    frame["label"] = "1"
    frame["attack_cat"] = "DoS"
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False)
    three = load_unsw_csv(changed)
    assert one.feature_hash.tolist() == three.feature_hash.tolist()
    assert one.sample_id.tolist() != three.sample_id.tolist()


def test_official_cannot_accept_synthetic_or_swapped_files(sources, tmp_path):
    with pytest.raises(ValueError, match="Synthetic"):
        prepare_dataset(*sources, tmp_path / "bad", data_kind="official")
    (sources[0].parent / "synthetic_source.json").unlink()
    with pytest.raises(ValueError, match="Do not swap"):
        prepare_dataset(sources[1], sources[0], tmp_path / "swapped", data_kind="official")


@pytest.mark.parametrize("target", ["manifest.json", "train.parquet", "audit.json"])
def test_tampering_is_detected(sources, tmp_path, target):
    prepared = tmp_path / "prepared"
    prepare_dataset(*sources, prepared, data_kind="synthetic")
    with (prepared / target).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_partition(prepared, "train")


def test_prepared_artifacts_cannot_be_accidentally_overwritten(sources, tmp_path):
    prepared = tmp_path / "prepared"
    prepare_dataset(*sources, prepared, data_kind="synthetic")
    with pytest.raises(FileExistsError):
        prepare_dataset(*sources, prepared, data_kind="synthetic")


def test_cross_source_conflicting_labels_are_audited(sources, tmp_path):
    left = pd.read_csv(sources[0], dtype=str, keep_default_na=False)
    right = pd.read_csv(sources[1], dtype=str, keep_default_na=False)
    base = left.loc[left.label == "0"].iloc[0]
    right.loc[0, list(FEATURES)] = base.loc[list(FEATURES)]
    right.loc[0, ["label", "attack_cat"]] = ["1", "DoS"]
    right.to_csv(sources[1], index=False)
    prepared = tmp_path / "prepared"
    prepare_dataset(*sources, prepared, data_kind="synthetic")
    audit = json.loads((prepared / "audit.json").read_text())
    assert audit["train_test_overlap"]["conflicting_label_groups"] == 1


def test_download_provenance_mismatch_is_rejected(sources, tmp_path):
    marker = {"data_kind": "mirror", "files": {sources[0].name: {"sha256": "incorrect"}}}
    (sources[0].parent / "provenance.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="provenance checksum"):
        prepare_dataset(*sources, tmp_path / "prepared", data_kind="synthetic")
