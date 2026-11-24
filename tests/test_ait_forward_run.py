"""Exercise the confirmation gate and a complete paired run on synthetic rows."""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from scripts.ait_forward import run as runner
from scripts.comprehensive_batch.run import read, write
from scripts.coverage_batch.run import report


WINDOW = {"start_utc": "2022-02-04T00:00:00Z", "end_utc": "2022-02-09T00:00:00Z"}
START = datetime.fromisoformat(WINDOW["start_utc"]).timestamp()


@pytest.fixture
def frozen_mock(monkeypatch):
    scored = []
    baseline = SimpleNamespace(
        model_sha256="fake-model", model_bytes=b"synthetic non-executable artifact",
        archive_member_sha256={"model": "fixture"},
        calibration_scores={"c1": 1.0, "c2": 2.0}, calibration_units={"a": ("c1",), "b": ("c2",)},
        pooled_threshold=2.0, last_calibration_time=START - 1,
        known_flow_ids=frozenset({"historic-flow"}), known_stable_flow_ids=frozenset({"historic-stable"}),
    )

    def score(rows):
        scored.append([row["flow_id"] for row in rows])
        return np.array([2.0, 3.0, 4.0, 5.0][:len(rows)])

    baseline.score = score
    monkeypatch.setattr(runner, "load_baseline", lambda _archive: baseline)
    return scored


def make_input(root, *, include_download=True):
    root.mkdir()
    rows = []
    for flow, label in (("normal", 0), ("control", 1), ("download", 1), ("unknown", None)):
        if flow == "download" and not include_download:
            continue
        rows.append({
            "flow_id": flow, "stable_flow_id": f"stable-{flow}", "allocation_group": runner.GROUP,
            "source_run": "harrison", "dataset_role": "confirmation", "clock_basis": runner.CLOCK,
            "label": label, "label_evidence": "synthetic" if label is not None else None,
            "target_kind": flow, "observed_end": START + 1, "replay_export_time": START + 2,
            "raw8": {field: 0 for field in runner.RAW8_FIELDS},
        })
    units = [{"unit_id": "business", "flow_ids": ["normal"]}]
    events = [
        {"event_id": f"event-{kind}", "kind": kind, "candidate_flow_ids": [kind],
         "association_mode": "all_confirmed", "candidate_set_complete": True, "label_evidence": "synthetic"}
        for kind in ("control", "download") if kind != "download" or include_download
    ]
    spec = {
        "schema": runner.CONFIRMATION_SCHEMA, "status": "qualified", "source_group": runner.GROUP,
        "confirmation_window": WINDOW,
        "clock_basis": runner.CLOCK, "normal_evidence": "synthetic", "clock_evidence": "synthetic",
        "targets": {
            "control": {"status": "qualified", "evidence": "synthetic"},
            "download": {"status": "qualified" if include_download else "unavailable", "evidence": "synthetic"},
        },
    }
    return spec, rows, units, events


def write_input(root, spec, rows, units, events):
    for name, records in zip(runner.INPUTS, (rows, units, events), strict=True):
        report.save_rows(root / name, records)
    spec["files"] = {name: sha256((root / name).read_bytes()).hexdigest() for name in runner.INPUTS}
    manifest = root / "qualification.json"
    write(manifest, spec)
    return manifest


def test_freeze_checks_no_confirmation_content(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    receipt = write_input(root, *make_input(root))
    for name in runner.INPUTS:
        (root / name).unlink()
    frozen = runner.freeze("unused", receipt, tmp_path / "output", 1, 1)
    assert frozen["fit_count"] == 0
    assert frozen_mock == []
    with pytest.raises(FileNotFoundError):
        runner.run("unused", receipt, tmp_path / "output")
    assert frozen_mock == []


def test_run_scores_once_and_keeps_all_rows(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    qualification = write_input(root, *make_input(root))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Freeze"):
        runner.run("unused", qualification, output)
    runner.freeze("unused", qualification, output, 1, 1)
    first = runner.run("unused", qualification, output)
    assert runner.run("unused", qualification, output) == first
    assert frozen_mock == [["normal", "control", "download", "unknown"]]
    summary = read(output / "summary.json")
    assert summary["outcomes"] == {"pooled": "criteria_met", "business_max": "criteria_met"}
    assert summary["confirmation_outcomes"] == dict.fromkeys(("pooled", "business_max"), "flow_only_criteria_met")
    assert summary["event_evaluation"]["pooled"]["status"] == "not_registered"
    assert summary["unknown_labels"] == 1 and summary["score_passes"] == 1 and summary["fit_count"] == 0
    assert read(output / "decisions.json")[0]["alerts"] == {"pooled": False, "business_max": False}
    with ZipFile(first) as bundle:
        assert {
            "output/registration.json", "output/model.joblib", "output/decisions.json",
            "output/evaluation_membership.json", "output/summary.json", "output/output_receipt.json",
        } <= set(bundle.namelist())
    assert read(Path(str(first) + ".sha256.json"))["sha256"] == sha256(first.read_bytes()).hexdigest()


def test_rejects_changed_floor_and_changed_input_before_score(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    receipt = write_input(root, *make_input(root))
    output = tmp_path / "output"
    original = runner.freeze("unused", receipt, output, 1, 1)
    registration = read(output / "registration.json")
    registration["minimum_tp"]["control"] = 2
    write(output / "registration.json", registration)
    with pytest.raises(ValueError, match="Frozen registration"):
        runner.run("unused", receipt, output)
    write(output / "registration.json", original)
    (root / runner.INPUTS[0]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="qualified hash"):
        runner.run("unused", receipt, output)
    assert frozen_mock == []


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda row: row.update(stable_flow_id="historic-stable"), "previous-role overlap"),
        (lambda row: row.update(replay_export_time=row["observed_end"] - 1), "Invalid complete-flow"),
        (lambda row: row.update(features8=[None] * 8), "canonical raw8"),
        (lambda row: row.update(dataset_role="external_development"), "whole-run role"),
    ],
)
def test_invalid_confirmation_stops_before_score(tmp_path, frozen_mock, mutate, error):
    root = tmp_path / "qualified"
    spec, rows, units, events = make_input(root)
    mutate(rows[0])
    receipt = write_input(root, spec, rows, units, events)
    output = tmp_path / "output"
    runner.freeze("unused", receipt, output, 1, 1)
    with pytest.raises(ValueError, match=error):
        runner.run("unused", receipt, output)
    assert frozen_mock == []


def test_unqualified_target_reports_partial_scope_only(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    receipt = write_input(root, *make_input(root, include_download=False))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Unavailable"):
        runner.freeze("unused", receipt, output, 1, 1)
    runner.freeze("unused", receipt, output, 1)
    runner.run("unused", receipt, output)
    summary = read(output / "summary.json")
    assert summary["outcomes"]["business_max"] == "partial_scope_criteria_met"
    assert summary["result"]["rules"]["business_max"]["targets"]["download"]["TP"] is None
    assert len(frozen_mock) == 1


def write_protocol(path, *entries):
    write(path, {"schema": runner.EVENT_PROTOCOL_SCHEMA,
                 "required_events": [{"event_id": identity, "kind": kind} for identity, kind in entries]})
    return path


def test_registered_event_requires_guaranteed_hit_even_when_floors_pass(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    spec, rows, units, events = make_input(root)
    events[0]["candidate_set_complete"] = False
    receipt = write_input(root, spec, rows, units, events)
    protocol = write_protocol(
        tmp_path / "protocol.json", ("event-control", "control"), ("event-download", "download"),
    )
    output = tmp_path / "output"
    registration = runner.freeze("unused", receipt, output, 1, 1, event_protocol=protocol)
    assert registration["event_protocol"] == read(protocol)
    assert frozen_mock == []
    runner.run("unused", receipt, output)

    summary = read(output / "summary.json")
    assert summary["outcomes"] == {"pooled": "criteria_met", "business_max": "criteria_met"}
    assert summary["confirmation_outcomes"] == dict.fromkeys(
        ("pooled", "business_max"), "event_criteria_indeterminate",
    )
    required = summary["event_evaluation"]["business_max"]["required_events"]
    assert required[0]["status"] == "indeterminate" and required[0]["reason"] == "unresolved_candidates"
    assert required[0]["ordering_diagnosis"] == "event_evidence_indeterminate"
    assert summary["result"]["score_ranking_event_feasibility"]["status"] == "indeterminate"
    assert required[1]["status"] == "met" and required[1]["reason"] == "guaranteed_hit"
    assert summary["result"]["rule_contrast"]["status"] == "identical_alerts"

    # Release uses the selection stored in the frozen record, not the mutable source file.
    write_protocol(protocol, ("a-different-control", "control"), ("event-download", "download"))
    runner.run("unused", receipt, output)
    assert read(output / "summary.json") == summary
    assert len(frozen_mock) == 1

    registration["event_protocol"]["required_events"][0]["event_id"] = "a-different-control"
    write(output / "registration.json", registration)
    with pytest.raises(ValueError, match="Frozen registration"):
        runner.run("unused", receipt, output)
    assert len(frozen_mock) == 1


def test_incomplete_availability_does_not_create_a_formal_event_pass(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    receipt = write_input(root, *make_input(root, include_download=False))
    output = tmp_path / "output"
    protocol = write_protocol(
        tmp_path / "protocol.json", ("event-control", "control"), ("expected-download", "download"),
    )
    runner.freeze("unused", receipt, output, 1, event_protocol=protocol)
    runner.run("unused", receipt, output)
    summary = read(output / "summary.json")
    assert summary["outcomes"]["business_max"] == "partial_scope_criteria_met"
    assert summary["confirmation_outcomes"]["business_max"] == "event_criteria_indeterminate"
    assert summary["event_evaluation"]["business_max"]["required_events"][1]["reason"] == "target_unavailable"


def test_wrong_required_event_kind_stops_before_scoring(tmp_path, frozen_mock):
    root = tmp_path / "qualified"
    receipt = write_input(root, *make_input(root))
    output = tmp_path / "output"
    protocol = write_protocol(
        tmp_path / "protocol.json", ("event-download", "control"), ("event-control", "download"),
    )
    runner.freeze("unused", receipt, output, 1, 1, event_protocol=protocol)
    with pytest.raises(ValueError, match="different qualified target"):
        runner.run("unused", receipt, output)
    assert frozen_mock == []
