import gzip
import hashlib
import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ids.common import read_json, write_json
from ids.contact_evidence import data, download
from ids.contact_evidence.source_history import SourceHistory
from ids.threshold_validation import observations as r21


def test_r21_window_with_expiration_protocols_unknowns_and_cross_batch_ties():
    rng = np.random.default_rng(22)
    size = 600
    clock = np.cumsum(rng.choice([0, 0, 0, 1, 19999, 60000, 60001], size=size))
    clock[305:500] = clock[304]
    keys = rng.integers(0, 8, size=size)
    destination = rng.integers(0, 15, size=size)
    port = rng.integers(-1, 12, size=size)
    responses = rng.integers(-1, 3, size=size)
    addr = ["::1", "192.0.2.1", "2001:db8::1", "198.51.100.2"]
    frame = pa.table(
        {
            "flow_end_ms": clock,
            "source": [addr[k // 2] for k in keys],
            "native_protocol": [["udp", "tcp"][k % 2] for k in keys],
            "destination": [str(d) for d in destination],
            "resp_port": port,
            "resp_pkts": responses,
        }
    )
    expected = np.column_stack(r21._history(clock, keys, destination, port, responses))
    for block_size in (11, 127, 301):
        rolling = SourceHistory()
        actual = np.concatenate(
            [
                np.column_stack(rolling.transform(frame.slice(i, block_size)))
                for i in range(0, size, block_size)
            ]
        )
        np.testing.assert_array_equal(expected, actual)


def test_prepare_keeps_truth_separate_and_never_changes_features_after_relabel(tmp_path):
    fields = "ts id.orig_h id.resp_h id.orig_p id.resp_p proto duration orig_pkts resp_pkts orig_ip_bytes resp_ip_bytes label det_label"
    chunks = [
        "#fields " + fields,
        "#types synthetic",
        "100 s a 954 80 tcp - 1 - 60 0 Malicious PartOfAHorizontalPortScan",
        "99 s a 954 80 tcp - 1 0 60 0 Benign -",
        "100.001 s b 954 80 tcp - 1 2 60 0 Malicious C&C",
        "160.000 s b 954 80 tcp - 1 1 60 0 Malicious C&C-HeartBeat",
        "160.001 s b 954 80 tcp - 1 1 60 0 Benign -",
        "#close stop",
    ]

    def run(lines, path):
        raw = path.with_suffix(".gz")
        raw.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))
        labels = path.with_suffix(".truth")
        receipt = data.prepare_log(raw, path, labels, chunk_rows=2)
        return pd.read_parquet(path), pd.read_parquet(labels), receipt

    a, truth_a, audit = run(chunks, tmp_path / "a.pq")
    b, truth_b, _ = run([s.replace("Malicious C&C", "Benign -") for s in chunks], tmp_path / "b.pq")
    pd.testing.assert_frame_equal(a, b)
    assert not (set(data.TRUTH) - {"row_id"}).intersection(a.columns)
    assert truth_a.native_detail.iloc[3] == "C&C-HeartBeat"
    assert truth_a.family.tolist() != truth_b.family.tolist()
    assert audit["rows"] == 5 and audit["adjacent_end_time_decreases"] == 1
    assert a["hist_flows_60s"].tolist() == [0, 0, 2, 3, 2]


@pytest.mark.parametrize("duplicate,late", [(True, False), (False, True), (False, False)])
def test_chronology_blocks_reuse_of_seen_capture_and_overlap(tmp_path, duplicate, late):
    items = [
        {"name": n, "sha256": n, "bytes": 40, "role": role}
        for n, role in [("train", "P"), ("control", "C"), ("unseen", "E")]
    ]
    if duplicate:
        items[-1]["sha256"] = items[0]["sha256"]
    for index, item in enumerate(items):
        write_json(
            data._audit(tmp_path, item["name"]),
            {
                "first_flow_end_ms": 2 * index,
                "last_flow_end_ms": 2 * index + 1,
                "rows": 7,
                "bytes": 40,
                "source_sha256": item["sha256"],
            },
        )
    reg = {"sources": items, "latest_opened_flow_end_ms": 4 if late else 3}
    if duplicate or late:
        with pytest.raises(ValueError):
            data.check_chronology(tmp_path, reg)
    else:
        (output,) = data.check_chronology(tmp_path, reg)
        assert read_json(output)["captured_periods_ms"]["E"] == {"unseen": [4, 5]}


def test_offline_source_verifies_bytes_and_never_calls_network(tmp_path, monkeypatch):
    payload = b"registered log\n"
    spec = {"name": "capture", "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    previous, current = tmp_path / download.PREVIOUS_DIR, tmp_path / "new"
    previous.mkdir()
    p = previous / "capture.conn.log.labeled.gz"
    p.write_bytes(gzip.compress(payload))
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: pytest.fail("offline access"))
    assert download.resolve_source(current, spec, cache=current) == p
    p.write_bytes(gzip.compress(b"wrong content"))
    with pytest.raises(ValueError):
        download.resolve_source(current, spec, cache=current)
    p.unlink()
    with pytest.raises(FileNotFoundError):
        download.resolve_source(current, spec, cache=current)


def test_range_download_resumes_complete_gzip_members(tmp_path, monkeypatch):
    content = b"abcdefghijklmn"
    ranges = []
    monkeypatch.setattr(download, "BLOCK_BYTES", 5)

    def serve(request, **_):
        lo, hi = map(int, request.get_header("Range")[6:].split("-"))
        ranges.append((lo, hi))
        body = io.BytesIO(content[lo : hi + 1])
        body.status = 206
        body.headers = {"Content-Range": f"bytes {lo}-{hi}/{len(content)}"}
        return body

    monkeypatch.setattr(download, "urlopen", serve)
    spec = {
        "name": "sample",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "url": "https://example.invalid/registered",
    }
    parts = tmp_path / ".partial/sample"
    parts.mkdir(parents=True)
    (parts / "000000000000.gz").write_bytes(gzip.compress(content[:5]))
    actual = download.resolve_source(None, spec, cache=tmp_path, workers=2)
    with gzip.open(actual, "rb") as f:
        assert f.read() == content
    assert sorted(ranges) == [(5, 9), (10, 13)]
    assert not parts.exists()
