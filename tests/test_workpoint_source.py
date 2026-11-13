import gzip
from hashlib import sha256
import re

import pyarrow.parquet as pq
import pytest

from ids.raw_workpoint import source
from ids.response_evidence.data import prepare_log as old_prepare
from ids.response_evidence.history import enrich_prepared


def test_frozen_source_registration_is_complete_and_historical():
    spec = source.read_spec()
    assert spec["role"] == "H" and spec["name"] == "CTU-IoT-Malware-Capture-43-1"
    assert spec["url"].endswith(spec["name"] + "/labeled/conn.log.labeled")
    assert re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])
    assert spec["bytes"] == 9363261631 and spec["rows"] == 67321809
    assert sum(spec["published_label_counts"].values()) == 67321809
    # Physically starts after the last P/C input, before the previously reported September 2019 cutoff.
    assert 1540548038807 < spec["first_flow_end_ms"] < spec["last_flow_end_ms"] < 1569026415507


def synthetic(tmp_path):
    head = "#fields ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto duration orig_pkts resp_pkts orig_ip_bytes resp_ip_bytes history conn_state missed_bytes label detailed-label"
    times = [86399.2] * 91 + [86400.5] + [86399.4] * 200 + [86401.0, 86461.0]
    rows = [head, "#types fields"]
    labels = [
        "Malicious C&C",
        "Benign -",
        "Malicious C&C-FileDownload",
        "Malicious PartOfAHorizontalPortScan",
        "Malicious FileDownload",
        "Malicious Okiru",
    ]
    types = [
        "- 1 0 60 0 S S0 0",
        "0 0 0 0 0 ^c OTH 0",
        "0.000001 0 0 0 0 C OTH 0",
        "- - - - - - OTH -",
        "0 1 1 40 40 Sh SF 0",
    ]
    for i, t in enumerate(times):
        rows.append(f"{t} u{i} 1.1.1.1 9 2.2.2.{1 + i % 2} 443 tcp {types[i % 5]} {labels[i % 6]}")
    rows.extend(["#close end", ""])
    path = tmp_path / "test.gz"
    path.write_bytes(gzip.compress("\n".join(rows).encode()))
    return path, len(times)


@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_parser_r23_exact_all_columns_and_clocks(tmp_path, workers):
    path, n = synthetic(tmp_path)
    before, target = tmp_path / "old", tmp_path / "old.truth"
    old = old_prepare(path, before, target, chunk_rows=77)
    dest, labels = tmp_path / "new", tmp_path / "new.truth"
    result = source.prepare_log(path, dest, labels, workers=workers, block_rows=4, expected={"rows": n})
    assert result["blocks"] > 20
    assert all(result[k] == old[k] for k in old)
    assert pq.read_table(before).equals(pq.read_table(dest))
    assert pq.read_table(target).equals(pq.read_table(labels))
    assert result["actual_parser_processes"] >= 1
    assert "native_detail" not in pq.read_schema(dest).names
    assert set(pq.read_schema(labels).names) == {
        "row_id",
        "family",
        "native_label",
        "native_detail",
        "window",
    }
    # Exercise unchanged production enrichment using these prepared files.
    obs = tmp_path / "history"
    assert enrich_prepared(dest, obs, workers=2, block_rows=19)["rows"] == n
    assert pq.read_table(obs, columns=["row_id"]).equals(pq.read_table(labels, columns=["row_id"]))


def test_bad_spec_and_schema_do_not_publish_partial_tables(tmp_path):
    path, n = synthetic(tmp_path)
    paths = tmp_path / "saved.blind", tmp_path / "saved.truth"
    for p in paths:
        p.write_bytes(b"before")
    with pytest.raises(ValueError, match="registered rows"):
        source.prepare_log(path, *paths, workers=1, block_rows=9, expected={"rows": n + 1})
    assert all(p.read_bytes() == b"before" for p in paths)
    path.write_bytes(gzip.compress(b"#fields ts uid\n"))
    with pytest.raises(ValueError, match="Zeek fields"):
        source.prepare_log(path, *paths, workers=1)
    assert all(p.read_bytes() == b"before" for p in paths)


def test_offline_source_rejects_wrong_bytes_and_never_calls_network(tmp_path, monkeypatch):
    raw = b"verified data\n"
    path = tmp_path / "Capture.conn.log.labeled.gz"
    path.write_bytes(gzip.compress(raw))
    spec = {"name": "Capture", "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    monkeypatch.setattr(source, "read_spec", lambda: spec)
    assert source.resolve(tmp_path) == path
    path.write_bytes(gzip.compress(raw + b"!"))
    with pytest.raises(ValueError, match="differs"):
        source.resolve(tmp_path)
