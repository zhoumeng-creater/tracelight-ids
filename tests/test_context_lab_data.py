"""Synthetic-only checks of the fixed NF-ToN-IoT-v3 development sampler."""

import csv
import hashlib
import json

import pytest

from ids.context_cache.lab_data import (
    CATEGORICAL, EXCLUDED_FEATURES, FEATURES, RAW_FEATURES, SCHEMA, SEED,
    _communication_group, _group_role, prepare_lab_data,
)
from ids.context_cache.preprocessing import CSVSchema, read_csv


def endpoint_for(role, offset=0):
    for number in range(1 + offset, 65535):
        src = f"10.8.{number // 256}.{number % 256}"
        group = _communication_group(src, "192.0.2.1", "12345", "443", "6")
        if _group_role(group, SEED) == role:
            return src
    raise AssertionError("no synthetic group found")


def row_for(role, label, signal, *, offset=0):
    row = {name: str(signal) for name in RAW_FEATURES}
    row.update({"IPV4_SRC_ADDR": endpoint_for(role, offset), "IPV4_DST_ADDR": "192.0.2.1",
                "L4_SRC_PORT": "12345", "L4_DST_PORT": "443", "PROTOCOL": "6", "L7_PROTO": "91",
                "FLOW_START_MILLISECONDS": "1700000000000", "FLOW_END_MILLISECONDS": "1700000000001",
                "Label": str(label), "Attack": "DDoS" if label else "Benign"})
    return row


def write_raw(path, rows, header=None):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header or [*RAW_FEATURES, "Label", "Attack"])
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def raw(tmp_path):
    rows = [row_for("support", i % 2, i + 1) for i in range(30)]
    rows += [row_for("calibration", 0, i + 101) for i in range(30)]
    # Calibration-pool attacks are explicitly counted and never silently
    # passed as an allegedly normal calibration file.
    rows += [row_for("calibration", 1, 150)]
    rows += [row_for("evaluation", int(i >= 18), i + 201) for i in range(30)]
    return write_raw(tmp_path / "raw.csv", rows)


def prepare(raw, output, **kwargs):
    return prepare_lab_data(raw, output, data_kind="synthetic", support_per_class=3,
                            calibration_rows=5, evaluation_rows=20, **kwargs)


def test_streamed_preparation_is_repeatable_and_audits_all_roles(raw, tmp_path):
    first = prepare(raw, tmp_path / "first")
    second = prepare(raw, tmp_path / "second")
    assert first["source"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert first["source"]["sha1"] == hashlib.sha1(raw.read_bytes()).hexdigest()
    assert first["source"]["bytes"] == raw.stat().st_size
    assert first["source"]["official_release_verified"] is False
    assert len(RAW_FEATURES) == 53 and len(FEATURES) == 48
    assert set(FEATURES).isdisjoint(EXCLUDED_FEATURES)
    assert {"L4_SRC_PORT", "L4_DST_PORT"} <= set(CATEGORICAL)
    schema = CSVSchema.from_dict(SCHEMA)
    for role, count in (("support", 6), ("calibration", 5), ("evaluation", 20)):
        assert first["files"][role]["sha256"] == second["files"][role]["sha256"]
        frame = read_csv(first["files"][role]["path"], schema)
        assert len(frame) == count
        assert all(_group_role(group, SEED) == role for group in frame.communication_group)
        assert first["scans"][role]["source_rows_scanned"] == 91
    assert first["audit"]["roles"]["support"]["normal"] == 3
    assert first["audit"]["roles"]["support"]["attack"] == 3
    assert first["audit"]["roles"]["calibration"]["attack"] == 0
    assert first["scans"]["calibration"]["non_normal_rows_excluded_by_calibration_contract"] == 1
    assert all(not any(pair.values()) for pair in first["audit"]["cross_role_pairs"].values())
    assert json.loads((tmp_path / "first" / "preparation.json").read_text(encoding="utf-8")) == first
    assert "no label/family stratification" in first["split"]["evaluation"]
    with pytest.raises(FileExistsError):
        prepare(raw, tmp_path / "first")


def test_direction_reversal_and_numeric_code_spelling_keep_one_communication_group():
    forward = _communication_group("192.0.2.1", "198.51.100.2", "123", "443", "6")
    backward = _communication_group("198.51.100.2", "192.0.2.1", "443.0", "123.0", "6.0")
    assert forward == backward
    assert forward != _communication_group("192.0.2.1", "198.51.100.2", "123", "443", "17")


def test_later_roles_exclude_cross_group_exact_inputs_and_refill(tmp_path):
    rows = [row_for("support", 0, 1), row_for("support", 1, 2)]
    rows += [row_for("calibration", 0, 1), row_for("calibration", 0, 3), row_for("calibration", 0, 3)]
    rows += [row_for("evaluation", 1, 1), row_for("evaluation", 0, 3),
             row_for("evaluation", 0, 4), row_for("evaluation", 0, 4),
             row_for("evaluation", 1, 5), row_for("evaluation", 1, 5)]
    # Changing label or IP identity does not hide a duplicate model input.
    raw = write_raw(tmp_path / "duplicate.csv", rows)
    report = prepare_lab_data(raw, tmp_path / "prepared", data_kind="synthetic",
                              support_per_class=1, calibration_rows=2, evaluation_rows=4)
    assert report["scans"]["calibration"]["cross_role_exact_feature_candidates_excluded"] == 1
    assert report["scans"]["evaluation"]["cross_role_exact_feature_candidates_excluded"] == 2
    assert report["files"]["evaluation"]["rows"] == 4
    assert report["audit"]["roles"]["calibration"]["within_role_exact_feature_duplicates"] == 1
    assert report["audit"]["roles"]["evaluation"]["within_role_exact_feature_duplicates"] == 2
    assert all(not any(pair.values()) for pair in report["audit"]["cross_role_pairs"].values())


def test_insufficient_pool_stops_without_seed_or_budget_replacement(tmp_path):
    raw = write_raw(tmp_path / "too-small.csv", [row_for("support", 0, 1)])
    output = tmp_path / "not-written"
    with pytest.raises(ValueError, match="insufficient eligible support"):
        prepare(raw, output)
    assert not output.exists()


def test_development_checks_official_source_and_fixed_budgets_before_csv_parsing(tmp_path):
    raw = tmp_path / "not-a-csv"
    raw.write_bytes(b"synthetic bytes, not official data")
    with pytest.raises(ValueError, match="official NF-ToN-IoT-v3"):
        prepare_lab_data(raw, tmp_path / "wrong-source")
    with pytest.raises(ValueError, match="frozen first-run budgets"):
        prepare_lab_data(raw, tmp_path / "wrong-budget", support_per_class=2)
    with pytest.raises(ValueError, match="final test"):
        prepare_lab_data(raw, tmp_path / "final", data_kind="final_test")


@pytest.mark.parametrize("problem,match", [
    ("header", "header mismatch"), ("family", "label disagreement"), ("nonfinite", "non-finite numeric"),
])
def test_source_contract_errors_are_not_relabelled_or_imputed(tmp_path, problem, match):
    row = row_for("support", 0, 1)
    header = [*RAW_FEATURES, "Label", "Attack"]
    if problem == "header":
        row["label"] = row.pop("Label")
        header[header.index("Label")] = "label"
    elif problem == "family":
        row["Attack"] = "DDoS"
    else:
        row["IN_BYTES"] = "inf"
    raw = write_raw(tmp_path / "bad.csv", [row], header)
    with pytest.raises(ValueError, match=match):
        prepare(raw, tmp_path / "output")
