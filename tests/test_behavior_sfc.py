import gzip
import hashlib
import json
import struct
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ids.behavior_transfer import sfc


def table(rows):
    defaults = dict(
        uid="a",
        ts="1744289628.955",
        orig_h="host",
        orig_p="2345",
        resp_h="other",
        resp_p="80",
        proto="tcp",
        duration="0",
        orig_pkts="1",
        resp_pkts="0",
        orig_ip_bytes="40",
        resp_ip_bytes="0",
        history="S",
        conn_state="S0",
        missed_bytes="0",
    )
    return pa.Table.from_pylist([dict(defaults, **row) for row in rows]).select(sfc.RAW)


def test_duplicate_backups_preserve_physical_first_and_opened_exposure():
    members = [dict(name="first"), dict(name="opened"), dict(name="third")]
    raw, backup, opened, info = sfc.deduplicate(
        [
            table([dict(uid="b"), dict(uid="a")]),
            table([dict(uid="a"), dict(uid="c")]),
            table([dict(uid="b"), dict(uid="d")]),
        ],
        members,
        {"opened"},
    )
    assert raw["uid"].to_pylist() == ["b", "a", "c", "d"]
    assert backup.tolist() == [0, 0, 1, 2]
    assert opened.tolist() == [False, True, True, False]
    assert info["physical_rows"] == 6 and info["repeated_rows"] == 2


@pytest.mark.parametrize("field,value", [("ts", "2"), ("resp_pkts", "3"), ("history", "Sh")])
def test_duplicate_uid_conflict_is_not_silently_merged(field, value):
    with pytest.raises(ValueError, match="different native"):
        sfc.deduplicate([table([{}]), table([{field: value}])], [dict(name="a"), dict(name="b")], set())


def test_clock_is_cumulative_physical_end_not_timestamp_sort():
    raw = table(
        [
            dict(ts="10.000001", duration="2"),
            dict(uid="b", ts="11", duration="0"),
            dict(uid="c", ts="12.000001", duration="0"),
        ]
    )
    start, end, clock = sfc._clock(raw)
    assert start.tolist() == [10000, 11000, 12000]
    assert end.tolist() == [12001, 11000, 12001]
    assert clock.tolist() == [12001, 12001, 12001]


def test_range_member_download_integrity_resume_and_cache(tmp_path, monkeypatch):
    raw = b"native bytes\n" * 100
    compressor = zlib.compressobj(wbits=-15)
    content = compressor.compress(raw) + compressor.flush()
    name = b"sensor/backup/conn.log"
    header = struct.pack(
        "<IHHHHHIIIHH", 0x04034B50, 20, 0, 8, 0, 0, zlib.crc32(raw), len(content), len(raw), len(name), 0
    )
    zip_bytes = header + name + content
    spec = dict(
        name=name.decode(),
        offset=0,
        compressed=len(content),
        size=len(raw),
        crc32=zlib.crc32(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    calls = []

    def read(archive, start, size):
        calls.append((start, size))
        return zip_bytes[start : start + size]

    monkeypatch.setattr(sfc, "_range", read)
    (tmp_path / "members").mkdir()
    (tmp_path / "members/backup.conn.log.deflate.part").write_bytes(content[:4])
    result = sfc.fetch_member(tmp_path, spec, {})
    assert gzip.open(result, "rb").read() == raw
    assert (len(header) + len(name) + 4, len(content) - 4) in calls
    calls.clear()
    assert sfc.fetch_member(tmp_path, spec, {}) == result and not calls
    with pytest.raises(ValueError, match="Cached native member differs"):
        sfc.fetch_member(tmp_path, dict(spec, sha256="0" * 64), {})


def _labels_archive(tmp_path, raw, *, conflict=False):
    frame = raw.select(sfc.IDENTITY).to_pandas()
    frame["attack"] = ["1"] * len(frame)
    frame["type_of_attack"] = "scanning"
    frame["label_source"] = "window"
    frame["window_id"] = "160"
    frame["attacker_role"] = "originator"
    frame["sensor_ip"] = "192.168.0.106"
    frame["log_type"] = "conn"
    if conflict:
        frame.loc[0, "ts"] = "123"
    path = tmp_path / "truth.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("labels.csv", frame.to_csv(index=False))
    return path


def test_truth_join_requires_native_tuple_and_preserves_unmatched(tmp_path):
    raw = table([{}, dict(uid="b")])
    conf = dict(sensor="192.168.0.106", labels_archive=dict(csv_member="labels.csv"))
    labels = sfc.attach_truth(raw, _labels_archive(tmp_path, raw.slice(0, 1)), conf)
    assert labels.iloc[0].window_id == "160"
    assert labels.iloc[1].label_source == "unmatched_v7"
    with pytest.raises(ValueError, match="conflicts with native"):
        sfc.attach_truth(raw, _labels_archive(tmp_path, raw.slice(0, 1), conflict=True), conf)


def test_daily_preparation_keeps_uid_truth_and_blind_separate(tmp_path):
    raw = table([{}, dict(uid="b", ts="1744289629")])
    labels = pd.DataFrame(
        dict(
            attack=["1", "0"],
            type_of_attack=["scanning", "normal"],
            label_source=["window", "background"],
            window_id=["160", ""],
            attacker_role=["originator", ""],
        )
    )
    _, _, clock = sfc._clock(raw)
    args = (
        raw,
        labels,
        clock,
        np.array([True, False]),
        np.array([0, 0]),
        [dict(name="old")],
        "2025-04-10",
        tmp_path,
        "id",
    )
    spec = sfc._prepare_day(args)
    assert spec["status"] == "complete", spec
    blind, truth = pq.read_table(spec["features"]), pq.read_table(spec["truth"])
    assert blind["uid"].to_pylist() == truth["uid"].to_pylist() == ["a", "b"]
    assert "family" not in blind.column_names and "event_id" not in blind.column_names
    assert "flow_end_ms" not in truth.column_names
    assert truth["family"].to_pylist() == ["Scanning", "Benign"]
    assert truth["evidence_level"].to_pylist() == ["author_attack_window", "author_background"]
    assert truth["previously_opened"].to_pylist() == [True, False]
    assert sfc._prepare_day(args) == spec


def test_invalid_day_is_recorded_failed_not_empty_complete(tmp_path):
    raw = table([{}])
    labels = pd.DataFrame(
        dict(
            attack=["0"],
            type_of_attack=["scanning"],
            label_source=["window"],
            window_id=["160"],
            attacker_role=["originator"],
        )
    )
    spec = sfc._prepare_day(
        (
            raw,
            labels,
            np.array([1]),
            np.array([False]),
            np.array([0]),
            [dict(name="new")],
            "2025-04-13",
            tmp_path,
            "id",
        )
    )
    assert spec["status"] == "failed" and "Inconsistent" in spec["error"]
    assert json.loads(spec["audit"].read_text())["status"] == "failed"


def test_empty_backup_preserves_typed_schema(tmp_path):
    path = tmp_path / "input.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("#fields\t" + "\t".join(sfc.RAW) + "\n#close\t2025\n")
    member = dict(name="s/backup/conn.log", sha256="test")
    result = sfc._decode_member((path, member, tmp_path))
    assert pq.read_table(result).schema.names == list(sfc.RAW)
    assert pq.read_table(result).num_rows == 0


def test_provider_full_synthetic_path_and_day_receipts(tmp_path, monkeypatch):
    raw = table([dict(ts="1744289628.955"), dict(uid="b", ts="1744531201")])
    archive = _labels_archive(tmp_path, raw)
    events = tmp_path / "events.json"
    events.write_text("# comment\n[]")
    config = dict(
        sensor="192.168.0.106",
        members=[dict(name="s/backup/conn.log")],
        labels_archive=dict(csv_member="labels.csv", filename="labels.zip"),
        events={},
    )
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps(config))
    monkeypatch.setattr(sfc, "REGISTRY", reg)
    monkeypatch.setattr(
        sfc, "collect_raw", lambda *a, **k: (raw, np.array([0, 0]), np.array([True, False]), {}, config)
    )
    monkeypatch.setattr(
        sfc,
        "archive_file",
        lambda cache, spec: events if spec.get("filename") == "all_attacks_logs.json" else archive,
    )
    specs = sfc.prepare_sources(tmp_path, tmp_path / "out", 2)
    assert len(specs) == 2 and all(x["status"] == "complete" for x in specs)
    assert [x["role"] for x in specs] == ["replay", "forward_unopened"]
    assert all(pq.read_table(x["features"])["row_id"].to_pylist() == [1] for x in specs)
