import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from ids.cicids import (
    CIC_CATEGORICAL, CIC_COLUMNS, CIC_DISCRETE, CIC_EXCLUDED, CIC_FEATURES, CIC_FILES,
    CIC_SPLITS, iter_cicids_csv, load_cic_partition, load_cic_prepared_verified,
    make_synthetic_cicids_dataset, prepare_cicids_dataset,
)
from ids.data import sha256_file


@pytest.fixture
def raw(tmp_path):
    return make_synthetic_cicids_dataset(tmp_path / "raw")


def refresh_marker(raw):
    path = raw / "synthetic_source.json"
    marker = json.loads(path.read_text())
    for name in CIC_FILES:
        marker["files"][name] = {"sha256": sha256_file(raw / name), "n_rows": len(pd.read_csv(raw / name))}
    path.write_text(json.dumps(marker), encoding="utf-8")


def test_contract_and_chunked_group_safe_preparation(raw, tmp_path):
    assert len(CIC_COLUMNS) == 91
    assert len(CIC_FEATURES) == 82
    assert not set(CIC_FEATURES) & set(CIC_EXCLUDED)
    assert set(CIC_CATEGORICAL) <= set(CIC_DISCRETE) <= set(CIC_FEATURES)
    output = tmp_path / "prepared"
    m = prepare_cicids_dataset(raw, output, data_kind="synthetic", chunksize=37)
    assert m["data_kind"] == "synthetic"
    assert m["feature_schema"]["features"] == list(CIC_FEATURES)
    assert str(tmp_path) not in json.dumps(m)
    previous = {key: set() for key in ("sample_id", "feature_hash", "session_hash", "group_id")}
    rows = 0
    for split in CIC_SPLITS:
        f = load_cic_partition(output, split)
        assert set(f.label) == {0, 1}
        assert f.groupby("label").size().min() >= 20
        assert f.group_id.equals(f.group_hash)
        assert f.attrs["feature_schema"] == m["feature_schema"]
        for key, seen in previous.items():
            assert not seen.intersection(f[key])
            seen.update(f[key])
        rows += len(f)
    assert rows == 600
    assert load_cic_prepared_verified(output, splits=CIC_SPLITS)["dataset_id"] == m["dataset_id"]


def test_duplicate_and_reverse_session_transitive_union(raw, tmp_path):
    monday = pd.read_csv(raw / "monday.csv")
    tuesday = pd.read_csv(raw / "tuesday.csv")
    tuesday.loc[1, list(CIC_FEATURES)] = monday.loc[0, list(CIC_FEATURES)]
    tuesday.loc[1, "Label"] = "DoS Hulk"  # Conflict retained and audited.
    tuesday.loc[1, ["Src Port", "Dst Port"]] = [60000, 443]
    tuesday.loc[2, ["Src IP", "Dst IP"]] = tuesday.loc[1, ["Dst IP", "Src IP"]].to_numpy()
    tuesday.loc[2, ["Src Port", "Dst Port"]] = [443, 60000]
    tuesday.loc[1, "Timestamp"] = "2017-07-04 09:00:00.000000"
    tuesday.loc[2, "Timestamp"] = "2017-07-04 09:00:01.000000"
    tuesday.to_csv(raw / "tuesday.csv", index=False)
    refresh_marker(raw)
    output = tmp_path / "prepared"
    prepare_cicids_dataset(raw, output, data_kind="synthetic", chunksize=7)
    frame = pd.concat([load_cic_partition(output, s).assign(split=s) for s in CIC_SPLITS], ignore_index=True)
    picked = frame[((frame.source_file == "monday.csv") & (frame.source_row == 0)) |
                   ((frame.source_file == "tuesday.csv") & frame.source_row.isin([1, 2]))]
    assert len(picked) == 3
    assert picked.group_id.nunique() == picked.split.nunique() == 1
    assert picked.feature_hash.nunique() == 2
    assert picked.session_hash.nunique() == 2
    audit = json.loads((output / "audit.json").read_text())
    assert audit["groups"]["conflicting_binary_label_groups"] == 1
    assert audit["rows_removed"] == 0
    assert len((output / "conflicts.jsonl").read_text().splitlines()) == 1


def test_attempted_relabel_and_infinity_audit(raw):
    p = raw / "monday.csv"
    frame = pd.read_csv(p)
    frame.loc[0, "Label"] = "DoS Hulk - Attempted"
    frame.loc[0, "Attempted Category"] = 3
    frame.loc[1, "Flow Bytes/s"] = np.inf
    frame.loc[2, "Fwd Packet Length Mean"] = np.nan
    frame.to_csv(p, index=False)
    chunk, audit = next(iter_cicids_csv(p, chunksize=3))
    assert chunk.loc[0, "label"] == 0
    assert chunk.loc[0, "attack_cat"] == "Normal"
    assert chunk.loc[0, "raw_label"] == "DoS Hulk - Attempted"
    assert chunk.loc[0, "attempted_category"] == 3
    assert audit["attempted_rows"] == 1
    assert np.isnan(chunk.loc[1, "Flow Bytes/s"])
    assert audit["infinity_to_missing"] == {"Flow Bytes/s": 1}
    assert audit["missing_values"]["Fwd Packet Length Mean"] == 1


@pytest.mark.parametrize("column,value,match", [
    ("Flow Bytes/s", "not-a-number", "Non-numeric"),
    ("Protocol", "6.5", "categorical"),
    ("Src IP", "bad-ip", "IPv4"),
    ("Timestamp", "not-a-time", "Timestamp"),
    ("Src Port", "65536", "integer"),
    ("Flow Duration", "-1", "Flow Duration"),
    ("Attempted Category", "7", "integer"),
    ("Label", "", "Missing Label"),
    ("Label", "DoS Hulk - Attempted", "disagrees"),
])
def test_invalid_inputs_fail_closed(raw, column, value, match):
    p = raw / "monday.csv"
    f = pd.read_csv(p, dtype=str, keep_default_na=False)
    f.loc[0, column] = value
    f.to_csv(p, index=False)
    with pytest.raises(ValueError, match=match):
        list(iter_cicids_csv(p, chunksize=7))


def test_header_rejects_extra_or_duplicate_columns(raw):
    p = raw / "monday.csv"
    text = p.read_text()
    p.write_text(text.replace("Flow ID", "id", 1))
    with pytest.raises(ValueError, match="91-column"):
        list(iter_cicids_csv(p))


def test_reproducibility_across_chunk_boundaries_and_verified_reuse(raw, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    first = prepare_cicids_dataset(raw, a, data_kind="synthetic", chunksize=17)
    prepare_cicids_dataset(raw, b, data_kind="synthetic", chunksize=53)
    for split in CIC_SPLITS:
        cols = ["sample_id", "feature_hash", "group_id", "session_hash"]
        pd.testing.assert_frame_equal(load_cic_partition(a, split)[cols], load_cic_partition(b, split)[cols])
    again = prepare_cicids_dataset(raw, a, data_kind="synthetic", reuse=True)
    assert again == first
    with pytest.raises(FileExistsError):
        prepare_cicids_dataset(raw, a, data_kind="synthetic")
    with pytest.raises(ValueError, match="settings"):
        prepare_cicids_dataset(raw, a, data_kind="synthetic", seed=123, reuse=True)


def test_loading_train_does_not_read_sealed_partitions(raw, tmp_path):
    out = tmp_path / "prepared"
    prepare_cicids_dataset(raw, out, data_kind="synthetic")
    (out / "test.parquet").unlink()
    (out / "calibration.parquet").unlink()
    assert len(load_cic_partition(out, "train")) > 0
    assert load_cic_prepared_verified(out, splits=("train", "validation"))
    with pytest.raises(FileNotFoundError):
        load_cic_partition(out, "test")


def test_changed_source_or_partition_refuses_reuse(raw, tmp_path):
    out = tmp_path / "prepared"
    prepare_cicids_dataset(raw, out, data_kind="synthetic")
    with (out / "train.parquet").open("ab") as f:
        f.write(b"altered")
    with pytest.raises(ValueError, match="checksum"):
        load_cic_partition(out, "train")
    with (raw / "monday.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="Source checksum"):
        prepare_cicids_dataset(raw, tmp_path / "other", data_kind="synthetic")


def test_inadequate_binary_support_is_not_repaired_by_row_split(raw, tmp_path):
    for n in CIC_FILES:
        f = pd.read_csv(raw / n)
        f["Label"] = "Benign"
        f.to_csv(raw / n, index=False)
    refresh_marker(raw)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="Insufficient binary class support"):
        prepare_cicids_dataset(raw, out, data_kind="synthetic")
    assert not out.exists()


def test_synthetic_cannot_be_declared_official(raw, tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_cicids_dataset(raw, tmp_path / "bad", data_kind="official_repaired")


def test_zip_safety_rejects_traversal_and_duplicates(tmp_path):
    spec = importlib.util.spec_from_file_location("fetch_cic_test", Path(__file__).parents[1] / "scripts/fetch_cicids.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        for name in CIC_FILES:
            z.writestr(name, "placeholder")
        z.writestr("../escaped.csv", "bad")
    with zipfile.ZipFile(p) as z, pytest.raises(ValueError, match="five known"):
        module.validate_archive_members(z)
