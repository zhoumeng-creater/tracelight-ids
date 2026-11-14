"""Regression checks for physical order, native truth and resumable member integrity."""

from io import BytesIO
import gzip
import hashlib
import json
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pyarrow.parquet as pq
import pytest

from ids.behavior_transfer import dedale


FIELDS = ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
          "duration", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
          "history", "conn_state", "missed_bytes", "ip_proto"]
SPEC = {"source_id": "DEDALE-green-D15", "name": "DEDALE-green-D15",
        "date": "2025-01-06", "role": "development"}


def _gz(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf8", newline="") as file:
        file.write(value)
    return path


def _inputs(tmp_path):
    rows = [
        ["1736121602", "b", "192.0.2.1", "50000", "192.0.2.2", "80", "tcp",
         "-", "1", "0", "40", "0", "S", "S0", "0", "6"],
        ["1736121601", "a", "192.0.2.1", "50001", "192.0.2.2", "80", "tcp",
         "1", "1", "1", "40", "100", "ShADadf", "SF", "0", "6"],
        ["1736121603", "c", "2001:db8::1", "133", "2001:db8::2", "134", "icmp",
         "-", "1", "0", "56", "0", "-", "OTH", "0", "58"],
    ]
    conn = _gz(tmp_path / "conn.gz", "#fields\t" + "\t".join(FIELDS) + "\n"
               + "\n".join("\t".join(r) for r in rows) + "\n")
    authors = []
    for row, label, comment in zip(rows, ("1", "0", "2"), ("fetch test", "", "fetch test"), strict=True):
        raw = dict(zip(FIELDS, row, strict=True))
        a = {name: "" for name in dedale.AUTHOR_COLUMNS}
        a.update(uid=raw["uid"], ts=raw["ts"], proto=raw["proto"],
                 label=label, step="5" if label != "0" else "0", attack_step="command_and_control",
                 technique="T1105", comments=comment)
        a.update({target: raw[name] for name, target in dedale.AUTHOR_NAMES.items()})
        authors.append(a)
    authors = pd.DataFrame(authors).iloc[[1, 2, 0]]
    labels = tmp_path / "labels.gz"
    authors.to_csv(labels, sep=";", index=False, compression="gzip")
    return conn, labels


def test_physical_order_and_separate_native_truth(tmp_path):
    conn, labels = _inputs(tmp_path)
    features, truth = tmp_path / "f.parquet", tmp_path / "t.parquet"
    registry = {"action_mapping": {"fetch test": {"family": "Download", "native_type": "download_test"}}}
    report = dedale.convert_day(conn, labels, features, truth, SPEC, batch_rows=1, registry=registry)
    blind, target = pq.read_table(features).to_pandas(), pq.read_table(truth).to_pandas()
    assert blind.uid.tolist() == target.uid.tolist() == ["b", "a", "c"]
    assert blind.row_id.tolist() == [1, 2, 3]
    assert blind.ip_protocol.tolist() == [6, 6, 58]
    assert blind.duration_seconds.tolist() == [-1, 1, -1]
    assert blind.response_evidence_code.tolist() == [0, 1, -1]
    assert target.family.tolist() == ["Download", "Benign", "Unresolved"]
    assert target.original_label.tolist() == ["1", "0", "2"]
    assert target.event_id.tolist() == ["DEDALE-green-D15:step:5", "", "DEDALE-green-D15:step:5"]
    assert target.window.tolist() == ["2025-01-06"] * 3
    assert "family" not in blind and "comments" not in blind and "flow_end_ms" not in target
    assert report["rows"] == 3 and report["logical_time_ties"] == 1
    assert blind.flow_end_ms.is_monotonic_increasing


@pytest.mark.parametrize("corruption", ["duplicate", "absent", "extra", "address", "timestamp", "native_label"])
def test_alignment_failures_do_not_publish_partial_outputs(tmp_path, corruption):
    conn, labels = _inputs(tmp_path)
    data = pd.read_csv(labels, sep=";", dtype=str, na_filter=False)
    if corruption == "duplicate":
        data.loc[0, "uid"] = data.uid.iloc[1]
    elif corruption == "absent":
        data = data.iloc[:-1]
    elif corruption == "extra":
        data = pd.concat([data, data.iloc[:1].assign(uid="extra")], ignore_index=True)
    elif corruption == "address":
        data.loc[0, "ip_src"] = "192.0.2.55"
    elif corruption == "timestamp":
        data.loc[0, "ts"] = "1736121610"
    else:
        data.loc[0, "label"] = "7"
    data.to_csv(labels, sep=";", index=False, compression="gzip")
    features, truth = tmp_path / "f.parquet", tmp_path / "t.parquet"
    features.write_bytes(b"existing features")
    truth.write_bytes(b"existing truth")
    with pytest.raises(ValueError):
        dedale.convert_day(conn, labels, features, truth, SPEC,
                           batch_rows=1, registry={"action_mapping": {}})
    assert features.read_bytes() == b"existing features"
    assert truth.read_bytes() == b"existing truth"


def _archive():
    memory = BytesIO()
    with ZipFile(memory, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("daily/conn.log", "test physical flow record\n" * 200)
    data = memory.getvalue()
    i = data.rfind(b"PK\x05\x06")
    size, offset = struct.unpack_from("<II", data, i + 12)
    with ZipFile(BytesIO(data)) as bundle:
        info = bundle.infolist()[0]
    archive = {"filename": "test.zip", "url": "https://invalid.test/data", "bytes": len(data),
               "central_directory_start": offset, "central_directory_bytes": size,
               "central_directory_sha256": hashlib.sha256(data[offset:offset + size]).hexdigest()}
    member = {"archive": "test.zip", "path": info.filename, "offset": info.header_offset,
              "compressed_bytes": info.compress_size, "bytes": info.file_size,
              "crc32": f"{info.CRC:08x}", "method": info.compress_type}
    return data, {"archives": {"test.zip": archive}}, member


def test_download_resumes_and_checks_member(tmp_path, monkeypatch):
    data, registry, member = _archive()
    monkeypatch.setattr(dedale, "BLOCK", 7)
    reads = []

    def partial_failure(archive, start, size):
        reads.append((start, size))
        if len(reads) == 5:
            raise OSError("temporary disconnect")
        return data[start:start + size]

    monkeypatch.setattr(dedale, "_range", partial_failure)
    with pytest.raises(OSError):
        dedale.resolve_member(tmp_path, member, registry)
    part = tmp_path / "test/daily/conn.log.deflate.part"
    assert part.stat().st_size == 7
    reads.clear()
    monkeypatch.setattr(dedale, "_range", lambda archive, start, size: data[start:start + size])
    result = dedale.resolve_member(tmp_path, member, registry)
    assert gzip.open(result, "rt").read() == "test physical flow record\n" * 200
    receipt = json.loads(result.with_suffix(".sha256.json").read_text())
    assert receipt["bytes"] == member["bytes"] and not part.exists()
    # Reusing an intact verified file performs no network request.
    monkeypatch.setattr(dedale, "_range", lambda *args: pytest.fail("Unexpected network read"))
    assert dedale.resolve_member(tmp_path, member, registry) == result


def test_download_rejects_changed_directory(tmp_path, monkeypatch):
    data, registry, member = _archive()
    registry["archives"]["test.zip"]["central_directory_sha256"] = "0" * 64
    monkeypatch.setattr(dedale, "_range", lambda archive, start, size: data[start:start + size])
    with pytest.raises(ValueError, match="directory"):
        dedale.resolve_member(tmp_path, member, registry)


def test_native_two_never_becomes_confirmed_download(tmp_path):
    conn, labels = _inputs(tmp_path)
    features, truth = tmp_path / "f", tmp_path / "t"
    registry = {"action_mapping": {"fetch test": {"family": "Download", "native_type": "download_test"}}}
    dedale.convert_day(conn, labels, features, truth, SPEC, registry=registry)
    target = pq.read_table(truth).to_pandas().set_index("uid")
    assert target.loc["c", "family"] == "Unresolved"
    assert target.loc["c", "evidence_level"] == "author_related_rule"


def test_prepare_failure_preserves_other_day_results(tmp_path, monkeypatch):
    sources = [dict(SPEC, source_id="ok", name="ok"), dict(SPEC, source_id="broken", name="broken")]
    monkeypatch.setattr(dedale, "read_registration", lambda: {"sources": sources})

    def prepare(source_id, cache_dir, output_dir):
        if source_id == "broken":
            raise ValueError("invalid source")
        return {"name": source_id, "source_id": source_id, "role": "development"}

    monkeypatch.setattr(dedale, "prepare_one", prepare)
    results = dedale.prepare_sources(tmp_path / "cache", tmp_path / "out", workers=1)
    assert [r["status"] for r in results] == ["complete", "failed"]
    assert results[1]["error"] == "ValueError: invalid source"


def test_confirmed_uid_with_nonmalicious_author_label_rejected(tmp_path):
    conn, labels = _inputs(tmp_path)
    with pytest.raises(ValueError, match="contradictory native label"):
        dedale.convert_day(conn, labels, tmp_path / "f", tmp_path / "t", SPEC,
                           registry={"action_mapping": {}}, levels={"c": "host_and_network_confirmed"})
