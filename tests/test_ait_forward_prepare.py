from hashlib import sha256
import gzip
import json

import pytest

from scripts.ait_forward import prepare, run


def _fixture(root):
    def write_jsonl(name, rows):
        raw = b"".join(json.dumps(row).encode("utf-8") + b"\n" for row in rows)
        payload = gzip.compress(raw, mtime=0) if name.endswith(".gz") else raw
        (root / name).write_bytes(payload)
        return {"path": name, "sha256": sha256(payload).hexdigest()}

    rows = [
        {
            "flow_id": kind, "stable_flow_id": f"stable-{kind}", "target_kind": kind,
            "label": label, "label_evidence": f"verified review for {kind}" if label is not None else None,
            "observed_end": 1643932801, "replay_export_time": 1643932802,
            "raw8": {field: 0 for field in run.RAW8_FIELDS},
        }
        for kind, label in (("normal", 0), ("control", 1), ("unknown", None))
    ]
    units = [{"unit_id": "application", "flow_ids": ["normal"]}]
    events = [{
        "event_id": "event-control", "kind": "control", "candidate_flow_ids": ["control", "unknown"],
        "association_mode": "one_of", "candidate_set_complete": True, "label_evidence": "event source",
    }]
    evidence = b"independent review"
    (root / "evidence.json").write_bytes(evidence)
    spec = {
        "schema": prepare.INPUT_SCHEMA,
        "source_group": run.GROUP,
        "clock_basis": run.CLOCK,
        "confirmation_window": {"start_utc": "2022-02-04T00:00:00Z", "end_utc": "2022-02-09T00:00:00Z"},
        "normal_evidence": "normal review", "clock_evidence": "clock review",
        "targets": {
            "control": {"status": "qualified", "evidence": "action review"},
            "download": {"status": "unavailable", "evidence": "missing download evidence"},
        },
        "inputs": {
            "rows": write_jsonl("rows.jsonl.gz", rows),
            "normal_units": write_jsonl("units.jsonl", units),
            "events": write_jsonl("events.jsonl", events),
        },
        "evidence_files": [{"path": "evidence.json", "sha256": sha256(evidence).hexdigest()}],
    }
    path = root / "review.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def test_prepare_is_deterministic_and_preserves_unknown_and_partial_scope(tmp_path):
    manifest, spec = _fixture(tmp_path)
    first = prepare.prepare(manifest, tmp_path / "qualified-one")
    second = prepare.prepare(manifest, tmp_path / "qualified-two")
    qualification, _ = run.load_qualification(first)
    assert first.read_bytes() == second.read_bytes()
    assert qualification["lineage"]["inputs"] == {
        "rows": {**spec["inputs"]["rows"], "bytes": (tmp_path / "rows.jsonl.gz").stat().st_size},
        "normal_units": {**spec["inputs"]["normal_units"], "bytes": (tmp_path / "units.jsonl").stat().st_size},
        "events": {**spec["inputs"]["events"], "bytes": (tmp_path / "events.jsonl").stat().st_size},
    }
    for name in run.INPUTS:
        assert (first.parent / name).read_bytes() == (second.parent / name).read_bytes()
        assert sha256((first.parent / name).read_bytes()).hexdigest() == qualification["files"][name]
    rows = [json.loads(line) for line in gzip.decompress((first.parent / "rows.jsonl.gz").read_bytes()).splitlines()]
    assert [row["label"] for row in rows] == [0, 1, None]
    assert all(row["dataset_role"] == "confirmation" for row in rows)
    assert qualification["targets"]["download"]["status"] == "unavailable"


def test_prepare_rejects_tampering_and_invalid_label_membership_before_publication(tmp_path):
    manifest, spec = _fixture(tmp_path)
    output = tmp_path / "qualified"
    unit_path = tmp_path / spec["inputs"]["normal_units"]["path"]
    bad_unit = b'{"unit_id":"bad","flow_ids":["unknown"]}\n'
    unit_path.write_bytes(bad_unit)
    with pytest.raises(ValueError, match="Reviewed source hash differs"):
        prepare.prepare(manifest, output)
    assert not output.exists()

    spec["inputs"]["normal_units"]["sha256"] = sha256(bad_unit).hexdigest()
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="contains non-normal"):
        prepare.prepare(manifest, output)
    assert not output.exists()


def test_diagnostic_writes_no_confirmation_package(tmp_path):
    manifest, spec = _fixture(tmp_path)
    spec["targets"]["control"]["status"] = "unavailable"
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="No target qualifies"):
        prepare.prepare(manifest, tmp_path / "rejected")
    receipt = prepare.prepare(manifest, tmp_path / "diagnostic", diagnostic=True)
    assert list(receipt.parent.iterdir()) == [receipt]
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "diagnostic_only"
    with pytest.raises(ValueError, match="requires a qualified"):
        run.load_qualification(receipt)
