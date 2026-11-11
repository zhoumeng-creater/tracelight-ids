from pathlib import Path
import hashlib
import io
import json
import subprocess
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from ids.capture_validation import source
from ids.capture_validation.__main__ import history
from ids.flow_history.execution import pack
from ids.flow_history.features import HISTORY_COLUMNS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_load_verified_source_and_registered_labels(tmp_path, monkeypatch):
    member = b"some synthetic flow rows"
    archive = tmp_path / "official.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("official.csv", member)
    monkeypatch.setattr(source, "CSV", {"bytes": len(member), "sha256": digest(member), "records": 10})
    monkeypatch.setattr(source, "CSV_MEMBER", "official.csv")
    monkeypatch.setattr(source, "ZIP_SHA", digest(archive.read_bytes()))
    monkeypatch.setattr(source, "ZIP_BYTES", archive.stat().st_size)
    path = source.resolve(tmp_path / "raw", archive)
    assert path.read_bytes() == member
    assert source.resolve(tmp_path / "work", path) == path
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        source.resolve(tmp_path / "raw", None)
    labels = tmp_path / "rows.npy"
    np.save(labels, np.array([1, 7, 10], dtype="<u4"))
    entry = {
        "file": labels.name,
        "sha256": digest(labels.read_bytes()),
        "bytes": labels.stat().st_size,
        "rows": 3,
    }
    (tmp_path / "registration.json").write_text(json.dumps({"labels": entry}))
    monkeypatch.setattr(source, "ASSETS", tmp_path)
    assert source.label_index()[0] == labels
    labels.write_bytes(b"wrong")
    with pytest.raises(ValueError):
        source.label_index()


def test_history_verifies_publisher_labels_then_restores_confirmed_truth(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "row_id": ["x", "y", "z"],
            "role": ["P_time", "C_time", "E_time"],
            "label": [1, 0, 1],
            "source_label": [0, 0, 1],
            "attack_family": ["Scanning", "Benign", "Bot"],
            "source_family": ["Benign", "Benign", "Bot"],
        }
    )
    data.to_parquet(tmp_path / "targets.parquet", index=False)
    (tmp_path / "evidence/inputs").mkdir(parents=True)

    def extract(official, original, work, threads, expected):
        assert list(original.attack_family) == list(data.source_family) and list(original.label) == list(
            data.source_label
        )
        work.mkdir()
        (work / "history_audit.json").write_text("{}")
        (work / "benchmark_segment.json").write_text('{"status":"unavailable"}')
        return pd.DataFrame({"row_id": data.row_id, **{c: [0, 1, 2] for c in HISTORY_COLUMNS}})

    monkeypatch.setattr("ids.capture_validation.__main__._extract", extract)
    history(tmp_path, Path("unused"), 2, {})
    assert pd.read_parquet(tmp_path / "inputs/P_time.parquet").attack_family.item() == "Scanning"
    assert pd.read_parquet(tmp_path / "inputs/E_time.parquet").label.item() == 1
    archive = pack(tmp_path, prefix="independent-capture-results")
    assert archive.name.startswith("independent-capture-results-")
    with ZipFile(archive) as z:
        assert "evidence/inputs/registered_history.parquet" in z.namelist()
        assert "targets.parquet" not in z.namelist()


def test_zip_download_without_overwriting_external_files(tmp_path, monkeypatch):
    blob = b"0123456789abcdef"

    class Reply(io.BytesIO):
        def __init__(self, data, status=200, headers=None):
            super().__init__(data)
            self.status = status
            self.headers = headers or {}

    target = tmp_path / "test.zip"
    target.with_suffix(".zip.part").write_bytes(blob[:5])

    def urlopen(url, timeout):
        if isinstance(url, str):
            return Reply(b'{"url":"https://example.org/content"}')
        assert url.headers.get("Range") == "bytes=5-"
        return Reply(blob[5:], 206, {"Content-Range": "bytes 5-15/16"})

    monkeypatch.setattr(source, "urlopen", urlopen)
    monkeypatch.setattr(source, "ZIP_BYTES", len(blob))
    monkeypatch.setattr(source, "ZIP_SHA", digest(blob))
    assert source.download(target).read_bytes() == blob


@pytest.mark.parametrize("shell", [n for n in ("powershell", "pwsh") if shutil.which(n)])
def test_lab_script_parse(shell):
    script = Path(__file__).resolve().parents[1] / "scripts/RUN_INDEPENDENT_CAPTURE.ps1"
    code = (
        "$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('"
        + str(script).replace("'", "''")
        + "',[ref]$null,[ref]$e); if($e.Count){exit 1}"
    )
    subprocess.run([shell, "-NoProfile", "-Command", code], check=True)


def test_external_source_identity_through_real_history_extractor(tmp_path):
    from test_capture_validation_cohort import _prepare, native

    origin = tmp_path / "origin"
    origin.mkdir()
    csv, rows, confirmed = native.__wrapped__(origin)
    targets, coverage = _prepare(csv, rows, confirmed)
    assert coverage["eligible_for_training"]
    root = tmp_path / "replication"
    (root / "evidence/inputs").mkdir(parents=True)
    targets.to_parquet(root / "targets.parquet", index=False)
    spec = {"sha256": digest(csv.read_bytes()), "records": len(rows), "bytes": csv.stat().st_size}
    history(root, csv, 2, spec)
    for role in ("P_time", "C_time", "E_time"):
        frame = pd.read_parquet(root / "inputs" / (role + ".parquet"))
        assert frame.row_id.tolist() == targets.loc[targets.role.eq(role), "row_id"].tolist()
        assert all(name in frame for name in HISTORY_COLUMNS)
    report = json.loads((root / "evidence/inputs/history_audit.json").read_bytes())
    assert report["checked_target_rows"] == len(targets)
