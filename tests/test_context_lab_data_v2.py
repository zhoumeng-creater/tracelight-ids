"""Synthetic source checks for independent, flow-weighted group pools."""

import csv
import hashlib

import pytest

from ids.context_cache.csv_workflow import audit_roles
from ids.context_cache.lab_data import (
    FLOW_SAMPLING, LEGACY_SAMPLING, RAW_FEATURES, SCHEMA, SEED,
    _communication_group, _group_role, _scan, prepare_lab_data,
)
from ids.context_cache.preprocessing import CSVSchema, read_csv


def endpoint_for(role):
    for number in range(1, 65535):
        src = f"10.8.{number // 256}.{number % 256}"
        group = _communication_group(src, "192.0.2.1", "12345", "443", "6")
        if _group_role(group, SEED) == role:
            return src
    raise AssertionError("no synthetic group found")


def row_for(role, label, signal):
    row = {name: str(signal) for name in RAW_FEATURES}
    row.update({"IPV4_SRC_ADDR": endpoint_for(role), "IPV4_DST_ADDR": "192.0.2.1",
                "L4_SRC_PORT": "12345", "L4_DST_PORT": "443", "PROTOCOL": "6", "L7_PROTO": "91",
                "FLOW_START_MILLISECONDS": "1700000000000", "FLOW_END_MILLISECONDS": "1700000000001",
                "Label": str(label), "Attack": "DDoS" if label else "Benign"})
    return row


def write_raw(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*RAW_FEATURES, "Label", "Attack"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def prepare(path, output, **kwargs):
    return prepare_lab_data(path, output, data_kind="synthetic", sampling_profile=FLOW_SAMPLING,
                            support_per_class=1, calibration_rows=kwargs.pop("calibration_rows", 3),
                            evaluation_rows=4, **kwargs)


@pytest.fixture
def shared_raw(tmp_path):
    rows = [row_for("support", 0, 1), row_for("support", 1, 2)]
    rows += [row_for("calibration", 0, 1) for _ in range(3)]
    rows += [row_for("evaluation", 0, 1), row_for("evaluation", 1, 2),
             row_for("evaluation", 0, 3), row_for("evaluation", 1, 4)]
    return write_raw(tmp_path / "shared.csv", rows)


def test_v2_keeps_cross_group_equal_inputs_and_flow_frequency(shared_raw, tmp_path):
    report = prepare(shared_raw, tmp_path / "prepared")
    assert report["sampling_profile"] == FLOW_SAMPLING
    assert report["risk_unit"] == "flow" and report["exact_feature_policy"] == "report"
    assert report["split"]["role_priority"] is None
    assert report["audit"]["cross_role_pairs"]["support/calibration"]["exact_feature_overlap"] == 1
    assert report["audit"]["cross_role_pairs"]["support/evaluation"]["exact_feature_overlap"] == 2
    assert report["audit"]["cross_role_pairs"]["calibration/evaluation"]["exact_feature_overlap"] == 1
    assert report["selected_concentration"]["calibration"] == {
        "rows": 3, "unique_communication_groups": 1, "largest_group_rows": 3,
        "largest_group_fraction": 1., "top_five_group_rows": 3, "top_five_group_fraction": 1.,
        "scope": "selected flow counts; not independent sample count or inferential effective sample size",
    }
    for scan in report["scans"].values():
        assert scan["cross_role_exact_feature_candidates_excluded"] == 0
        assert scan["identifier_validation"]["rows_validated"] == 9
        assert scan["identifier_validation"]["fields"]["L7_PROTO"]["minimum"] == 91
        assert scan["identifier_validation"]["fields"]["L7_PROTO"]["maximum"] == 91
        assert scan["identifier_validation"]["fields"]["L7_PROTO"]["domain_basis"] == "project input contract"


def test_role_scan_order_and_evaluation_selection_are_independent(shared_raw, tmp_path):
    source_hash = hashlib.sha256(shared_raw.read_bytes()).hexdigest()
    quotas = {"support": {0: 1, 1: 1}, "calibration": {"all": 3}, "evaluation": {"all": 4}}
    def run(roles):
        return {role: _scan(shared_raw, role, quotas[role], set(), source_hash, SEED,
                            validate_identifiers=True)[0] for role in roles}
    forward = run(("support", "calibration", "evaluation"))
    reverse = run(("evaluation", "calibration", "support"))
    assert forward == reverse
    full = prepare(shared_raw, tmp_path / "full")
    changed_calibration = prepare(shared_raw, tmp_path / "fewer-calibration", calibration_rows=2)
    assert full["files"]["evaluation"]["sha256"] == changed_calibration["files"]["evaluation"]["sha256"]
    with open(full["files"]["evaluation"]["path"], encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))[1:]
    assert rows == [item[3] for item in forward["evaluation"]]


@pytest.mark.parametrize("column", ["sample_id", "communication_group"])
def test_report_policy_still_rejects_shared_identity_or_group(shared_raw, tmp_path, column):
    report = prepare(shared_raw, tmp_path / "prepared")
    schema = CSVSchema.from_dict(SCHEMA)
    frames = {role: read_csv(item["path"], schema) for role, item in report["files"].items()}
    frames["evaluation"].loc[0, column] = frames["support"].loc[0, column]
    with pytest.raises(ValueError, match="overlap"):
        audit_roles(frames, schema, exact_feature_policy="report")


@pytest.mark.parametrize("field,value", [
    ("L4_SRC_PORT", "65536"), ("L4_DST_PORT", "-1"), ("PROTOCOL", "256"),
    ("L7_PROTO", ""), ("L7_PROTO", "65536"), ("L7_PROTO", "1.5"),
    ("L7_PROTO", "nan"), ("L7_PROTO", "inf"), ("L7_PROTO", "65535.000000000001"),
])
def test_invalid_identifiers_in_other_role_stop_whole_source_scan(tmp_path, field, value):
    rows = [row_for("support", 0, 1), row_for("support", 1, 2), row_for("evaluation", 0, 3)]
    rows[-1][field] = value
    path = write_raw(tmp_path / "invalid.csv", rows)
    with pytest.raises(ValueError, match=f"invalid identifier {field} at source row 3"):
        _scan(path, "support", {0: 1, 1: 1}, set(), "0" * 64, SEED, validate_identifiers=True)


def test_invalid_identifier_in_noncompetitive_row_is_not_skipped(tmp_path, monkeypatch):
    rows = [row_for("support", 0, 1), row_for("support", 1, 2), row_for("support", 0, 3)]
    rows[-1]["L7_PROTO"] = "missing"
    path = write_raw(tmp_path / "noncompetitive.csv", rows)
    monkeypatch.setattr("ids.context_cache.lab_data._priority", lambda row, *_: row)
    with pytest.raises(ValueError, match="invalid identifier L7_PROTO at source row 3"):
        _scan(path, "support", {0: 1, 1: 1}, set(), "0" * 64, SEED, validate_identifiers=True)


def test_reported_identifier_range_includes_unselected_and_other_role_rows(tmp_path):
    rows = [row_for("support", 0, 1), row_for("support", 1, 2),
            row_for("evaluation", 0, 3), row_for("calibration", 0, 4)]
    rows[2]["L7_PROTO"] = "65535"
    rows[3]["L7_PROTO"] = "0.0"
    path = write_raw(tmp_path / "range.csv", rows)
    _, scan = _scan(path, "support", {0: 1, 1: 1}, set(), "0" * 64, SEED, validate_identifiers=True)
    fields = scan["identifier_validation"]["fields"]
    assert scan["identifier_validation"]["rows_validated"] == 4
    assert (fields["L7_PROTO"]["minimum"], fields["L7_PROTO"]["maximum"]) == (0, 65535)


def test_legacy_default_still_rejects_shared_inputs_by_exclusion(shared_raw, tmp_path):
    kwargs = dict(data_kind="synthetic", support_per_class=1, calibration_rows=3, evaluation_rows=4)
    with pytest.raises(ValueError, match="insufficient eligible calibration"):
        prepare_lab_data(shared_raw, tmp_path / "implicit-legacy", **kwargs)
    with pytest.raises(ValueError, match="insufficient eligible calibration"):
        prepare_lab_data(shared_raw, tmp_path / "explicit-legacy", sampling_profile=LEGACY_SAMPLING, **kwargs)


def test_unknown_sampling_profile_and_modified_development_seed_fail_early(tmp_path):
    nonexistent = tmp_path / "not-read.csv"
    with pytest.raises(ValueError, match="unknown sampling profile"):
        prepare_lab_data(nonexistent, tmp_path / "out", sampling_profile="best_seed")
    with pytest.raises(ValueError, match="frozen first-run budgets and seed"):
        prepare_lab_data(nonexistent, tmp_path / "out", sampling_profile=FLOW_SAMPLING, seed=SEED+1)
