import csv
import gzip
import hashlib
import io
from zipfile import ZipFile

import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.response_evidence import data, sfc, sources


FIELDS = "ts uid id.orig_h id.resp_h id.orig_p id.resp_p proto duration orig_pkts resp_pkts orig_ip_bytes resp_ip_bytes history conn_state missed_bytes"
ROWS = [
    "100 A a b 54 80 tcp - 1 0 60 0 S S0 0",
    "99 B a b 54 80 tcp - 0 0 0 0 c OTH 0",
    "86401 C b a 80 54 tcp 0.01 1 1 60 60 Sh SF 0",
]


def test_physical_blind_identity_response_and_utc_day(tmp_path):
    def prepare(labels, part):
        lines = ["#fields " + FIELDS + " label det_label"] + [a + " " + b for a, b in zip(ROWS, labels)]
        gz = tmp_path / (part + ".gz")
        gz.write_bytes(gzip.compress("\n".join(lines).encode()))
        blind, truth = tmp_path / (part + ".blind"), tmp_path / (part + ".truth")
        info = data.prepare_log(gz, blind, truth, chunk_rows=2)
        assert info["adjacent_end_time_decreases"] == 1
        return pd.read_parquet(blind), pd.read_parquet(truth)

    a, target = prepare(["Malicious C&C-FileDownload", "Benign -", "Malicious C&C"], "one")
    b, _ = prepare(["Benign -"] * 3, "two")
    pd.testing.assert_frame_equal(a, b)
    assert a.response_evidence_code.tolist() == [0, -1, 1]
    assert a.resp_checksum_uncounted.tolist() == [0, 1, 0]
    assert a.reply_unknown.tolist() == [0, 1, 0]
    assert target.window.tolist() == ["1970-01-01", "1970-01-01", "1970-01-02"]
    assert not {"family", "native_label", "native_detail", "window", "uid"} & set(a)


def test_v7_truth_joins_earlier_backup_and_keeps_unsigned_rows_in_history(tmp_path):
    physical = (
        "#fields\t"
        + FIELDS.replace(" ", "\t")
        + "\n"
        + "\n".join(row.replace(" ", "\t") for row in ROWS)
        + "\n"
    ).encode()
    config = dict(
        name="e",
        filename="e.gz",
        rows=3,
        sensor="sensor",
        uid_order_sha256=hashlib.sha256(b"A\nB\nC\n").hexdigest(),
    )
    archive = tmp_path / "v7.zip"
    buf = io.StringIO()
    cols = [*sfc.IDENTITY, *sfc.LABELS, "sensor_ip", "log_type", "backup_folder"]
    writer = csv.writer(buf)
    writer.writerow(cols)
    writer.writerow(
        [
            "B",
            "99",
            "a",
            "b",
            "54",
            "80",
            "tcp",
            "0",
            "normal",
            "ambiguous_attacker_no_window",
            "",
            "originator",
            "sensor",
            "conn",
            "older",
        ]
    )
    writer.writerow(
        [
            "A",
            "100",
            "a",
            "b",
            "54",
            "80",
            "tcp",
            "1",
            "scanning",
            "attack_window",
            "W4",
            "originator",
            "sensor",
            "conn",
            "older",
        ]
    )
    writer.writerow(
        [
            "C",
            "86401",
            "b",
            "a",
            "80",
            "54",
            "tcp",
            "1",
            "scanning",
            "attack_window",
            "W4",
            "originator",
            "other",
            "conn",
            "later",
        ]
    )
    with ZipFile(archive, "w") as z:
        z.writestr("v7.csv", buf.getvalue())
    sfc.project(
        archive, tmp_path, [config], {"csv_member": "v7.csv"}, raw_reader=lambda *args: physical, verify=False
    )
    a, target = tmp_path / "blind", tmp_path / "truth"
    data.prepare_log(tmp_path / "e.gz", a, target, sfc=True, chunk_rows=2)
    t = pd.read_parquet(target)
    assert t.family.tolist() == ["Scanning", "Unresolved", "Unresolved"]
    assert t.author_window.tolist() == ["W4", "", ""]
    assert t.label_source.tolist()[-1] == "unmatched_v7"
    assert pd.read_parquet(a).row_id.tolist() == [1, 2, 3]


def test_chronology_distinguishes_retrospective_unseen_and_external(tmp_path):
    items = []
    for role, begin in (("P", 1), ("C", 10), ("R", 50), ("H", 20), ("E", 100), ("X", 100)):
        source = dict(name=role, sha256=role, role=role, bytes=9)
        items.append(source)
        write_json(
            data._audit(tmp_path, role),
            dict(bytes=9, source_sha256=role, rows=2, first_flow_end_ms=begin, last_flow_end_ms=begin + 1),
        )
    reg = dict(sources=items, previously_scored=["R"], latest_opened_flow_end_ms=51)
    (path,) = data.check_chronology(tmp_path, reg)
    assert read_json(path)["captured_periods_ms"]["H"] == {"H": [20, 21]}
    reg["previously_scored"].append("H")
    with pytest.raises(ValueError, match="Inspection"):
        data.check_chronology(tmp_path, reg)


def test_resolver_offline_never_connects_and_checks_same_sibling_input(tmp_path, monkeypatch):
    previous = tmp_path / "tracelight-contact-r22/data/raw" / sources.PREVIOUS[0]
    previous.mkdir(parents=True)
    monkeypatch.setattr(sources, "REPO", tmp_path / "tracelight-response-r23")
    blob = b"old"
    spec = dict(name="test", sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
    p = previous / sources.filename(spec)
    p.write_bytes(gzip.compress(blob))
    assert sources.resolve(tmp_path / "offline", spec, {}, cache=tmp_path / "cache") == p
    p.unlink()
    with pytest.raises(FileNotFoundError):
        sources.resolve(tmp_path / "offline", spec, {}, cache=tmp_path / "cache")


def test_heuristic_scanning_is_excluded_from_primary_truth():
    frame = pd.DataFrame(
        {
            "label_source": ["scan_heuristic", "window", "background"],
            "attack": ["1", "1", "0"],
            "type_of_attack": ["scanning", "scanning", "normal"],
        }
    )
    result = data._sfc_inputs(frame)
    assert result.label.tolist() == ["Unresolved", "Malicious", "Benign"]
    assert result.type_of_attack.tolist() == frame.type_of_attack.tolist()
