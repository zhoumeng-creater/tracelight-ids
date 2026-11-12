import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.response_evidence import tqh


def sample_cell(folder, *, wrong_labels=None):
    """Small standalone author-format cell. Reversed label order exercises the UID join."""
    prefix = "A/A_i30_j0"
    stamp = 1783115001
    base = {
        "ts": stamp,
        "id.orig_h": "s",
        "id.resp_h": "d",
        "id.orig_p": 8001,
        "id.resp_p": 80,
        "proto": "tcp",
        "duration": 1,
        "orig_pkts": 1,
        "resp_pkts": 0,
        "orig_ip_bytes": 70,
        "resp_ip_bytes": 0,
        "history": "S",
        "conn_state": "S0",
        "missed_bytes": 0,
    }
    records = [
        base | {"uid": "u1", "duration": 3},
        base | {"uid": "u2", "ts": stamp + 1},
        base | {"uid": "u3", "ts": stamp + 5, "resp_pkts": 1, "resp_ip_bytes": 70},
    ]
    labels = (
        wrong_labels
        if wrong_labels is not None
        else [
            {"uid": "u3", "label": "malicious_recon"},
            {"uid": "u1", "label": "benign_external"},
            {"uid": "u2", "label": "malicious_c2"},
        ]
    )
    file = Path(folder) / "sample.zip"
    with ZipFile(file, "w") as z:
        for name, values in (("conn.log", records), ("labeled.jsonl", labels)):
            z.writestr(prefix + "/" + name, "\n".join(json.dumps(x) for x in values) + "\n")
        z.writestr(
            prefix + "/manifest.json",
            json.dumps({"start_utc": "2026-07-03T21:00:00Z", "end_utc": "2026-07-03T22:00:00Z"}),
        )
    return file, prefix


def test_parse_author_independent_physical_order_and_offline_identity(tmp_path):
    archive, unit = sample_cell(tmp_path)
    with ZipFile(archive) as z:
        obs, labels, meta = tqh._cell(z, unit)
    assert obs["response_evidence_code"].to_pylist() == [0, 0, 1]
    assert labels["native_detail"].to_pylist() == ["benign_external", "malicious_c2", "malicious_recon"]
    assert labels["family"].to_pylist() == ["Benign", "ExternalC2", "ExternalRecon"]
    assert not {"uid", "label", "native_label", "family", "window"}.intersection(obs.column_names)
    np.testing.assert_array_equal(np.diff(obs["flow_end_ms"].to_numpy()), [0, 3000])
    spec = {
        "profile": "A",
        "unit": "A_i30_j0",
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "name": "TQH-A_i30_j0",
        **meta,
    }
    features, targets, receipt = tqh.prepare_one(tmp_path / "run", tmp_path, spec)
    assert len(pd.read_parquet(features)) == len(pd.read_parquet(targets)) == 3
    assert read_json(receipt)["rows"] == 3
    with pytest.raises(ValueError, match="Offline TQH archive"):
        tqh.prepare_one(tmp_path / "run", tmp_path, spec | {"archive_sha256": "0" * 64})
    with pytest.raises(ValueError, match="labels_sha256"):
        tqh.prepare_one(tmp_path / "run", tmp_path, spec | {"labels_sha256": "0" * 64})


@pytest.mark.parametrize("uids", [["u1", "u2"], ["u1", "u1", "u3"], ["u1", "u2", "other"]])
def test_reject_orphan_and_duplicate_uids(tmp_path, uids):
    path, unit = sample_cell(tmp_path, wrong_labels=[{"uid": uid, "label": "benign"} for uid in uids])
    with ZipFile(path) as z, pytest.raises(ValueError, match="UID"):
        tqh._cell(z, unit)


def test_all_cells_are_distinct_complete_forward_units():
    specs = tqh.read_specs()
    assert len({s["unit"] for s in specs}) == len(specs) == 48
    assert {s["role"] for s in specs} == {"X"}
    assert sum(s["rows"] for s in specs) == 152765
    assert len({s["sha256"] for s in specs}) == len({s["uid_order_sha256"] for s in specs}) == 48
    cutoff = read_json(tqh.REGISTRY.with_name("registration.json"))["latest_opened_flow_end_ms"]
    assert all(s["last_flow_end_ms"] >= s["first_flow_end_ms"] > cutoff for s in specs)


def test_decimal_tqh_clock_never_truncates_submicrosecond_response_duration(tmp_path):
    path, unit = sample_cell(tmp_path)
    with ZipFile(path) as source:
        parts = {name: source.read(name) for name in source.namelist()}
    key = unit + "/conn.log"
    rows = parts[key].splitlines()
    rows[0] = rows[0].replace(b'"duration": 3', b'"duration": 3.00000002')
    parts[key] = b"\n".join(rows) + b"\n"
    with ZipFile(path, "w") as source:
        for name, value in parts.items():
            source.writestr(name, value)
    with ZipFile(path) as source:
        obs, _, _ = tqh._cell(source, unit)
    assert obs["flow_end_ms"][0].as_py() == 1783115004001
    assert obs["flow_end_ms"][1].as_py() == 1783115004001
