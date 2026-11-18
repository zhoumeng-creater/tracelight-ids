"""Synthetic paired Zeek logs; no runtime, packet download, or model."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "compare_exports", Path(__file__).resolve().parents[1] / "scripts/ait_contract/compare_exports.py")
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def row(ts=1, uid="arbitrary", **changes):
    return {"ts": ts, "id.orig_h": "10.0.0.1", "id.orig_p": 50000, "id.resp_h": "10.0.0.2",
            "id.resp_p": 443, "proto": "tcp", "uid": uid, "duration": 1.5,
            "orig_pkts": 4, "resp_pkts": 5, "orig_ip_bytes": 200, "resp_ip_bytes": 400,
            "conn_state": "SF", "missed_bytes": 0, "contract_export_network_time": 5,
            "contract_after_input_eof": False, "contract_during_shutdown": False, **changes}


def log(directory, rows):
    directory.mkdir()
    (directory / "conn.log").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_exact_keys_ignore_uid_and_report_field_and_eof_changes(tmp_path):
    left, right, out = [tmp_path / name for name in ("strict", "ignore", "out")]
    log(left, [row(1, "left"), row(2), row(3), row(4)])
    log(right, [row(1, "different-uid"), row(2, resp_ip_bytes=800, contract_after_input_eof=True),
                row(3.000001), row(5)])
    result = compare.compare_exports(left, right, out)
    assert result["counts"] == {"strict_only": 2, "ignore_checksums_only": 2,
                                 "common_identical": 1, "common_changed": 1, "ambiguous": 0}
    assert result["field_change_counts"] == {"resp_ip_bytes": 1, "contract_after_input_eof": 1}
    changed = list(csv.DictReader((out / "common_changed.csv").open(encoding="utf-8-sig")))[0]
    assert json.loads(changed["changed_fields"]) == ["resp_ip_bytes", "contract_after_input_eof"]
    assert result["variant_selection"] == "none"


def test_duplicate_key_never_gets_paired_and_missing_is_not_null(tmp_path):
    left, right, out = [tmp_path / name for name in ("strict", "ignore", "out")]
    absent = row(2)
    del absent["duration"]
    log(left, [row(1, "a"), row(1, "b"), absent])
    log(right, [row(1, "a"), row(2, duration=None)])
    result = compare.compare_exports(left, right, out)
    assert result["ambiguous_keys"] == 1 and result["counts"]["ambiguous"] == 3
    assert result["counts"]["common_changed"] == 1
    assert result["counts"]["common_identical"] == 0
    assert result["field_change_counts"] == {"duration": 1}


def test_absent_log_requires_successful_explicit_receipt(tmp_path):
    left, right, out = [tmp_path / name for name in ("strict", "ignore", "out")]
    left.mkdir()
    log(right, [row()])
    with pytest.raises(FileNotFoundError, match="explicit absent-log receipt"):
        compare.compare_exports(left, right, out)
    (left / "export-receipt.json").write_text(json.dumps({"status": "complete", "logs": {"conn": {"exists": False}}}))
    result = compare.compare_exports(left, right, out)
    assert result["counts"]["ignore_checksums_only"] == 1
    assert result["connection_rows"]["strict"] == 0


def test_timestamp_decimal_precision_is_not_collapsed_by_float(tmp_path):
    left, right, out = [tmp_path / name for name in ("strict", "ignore", "out")]
    log(left, [row()])
    log(right, [row()])
    for directory, timestamp in ((left, "1642996762.000000001"), (right, "1642996762.000000002")):
        path = directory / "conn.log"
        path.write_text(path.read_text().replace('"ts": 1,', '"ts": ' + timestamp + ','))
    result = compare.compare_exports(left, right, out)
    assert result["counts"]["common_identical"] == 0
    assert result["counts"]["strict_only"] == result["counts"]["ignore_checksums_only"] == 1
