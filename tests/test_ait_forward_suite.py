"""Exercise suite admission, fixed pair coverage, and same-output score reuse."""
from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.ait_forward import comparators, prepare, run, suite
from test_ait_forward_prepare import _fixture


def test_suite_freezes_all_scorers_and_resumes_without_rescoring(tmp_path, monkeypatch):
    manifest, spec = _fixture(tmp_path)
    protocol = tmp_path / "event-protocol.json"
    protocol.write_text(json.dumps({
        "schema": run.EVENT_PROTOCOL_SCHEMA,
        "required_events": [{"event_id": "event-control", "kind": "control"}],
    }), encoding="utf-8")

    source_protocol = protocol.read_bytes()
    spec["evidence_files"].append({"path": protocol.name, "sha256": sha256(source_protocol).hexdigest()})
    manifest.write_text(json.dumps(spec), encoding="utf-8")
    qualification = prepare.prepare(manifest, tmp_path / "qualification")

    output = tmp_path / "result"
    scored = []
    preflight_scored = []
    contents = b"engineering fixture, no serialized estimator"
    digest = sha256(contents).hexdigest()
    baseline = SimpleNamespace(
        model_sha256=digest, archive_member_sha256={"fixture": digest},
        calibration_scores={"ref": 0.0}, calibration_units={"reference": ("ref",)},
        pooled_threshold=0.0, last_calibration_time=1643932799,
        known_flow_ids=frozenset(), known_stable_flow_ids=frozenset(),
    )

    def make_candidate(model_id):
        def score(rows, *, device, threads):
            assert threads == 4
            assert (output / "registration.json").is_file()
            assert (output / "opened_fox_regression.json").is_file()
            assert [row["flow_id"] for row in rows] == ["normal", "control", "unknown"]
            scored.append((model_id, device))
            return np.array([0.0, 1.0, -1.0])

        def fox_regression(rows, rules, **_kwargs):
            assert rows == ["opened-fixture"]
            preflight_scored.append(model_id)
            return {"rules": {rule: {"flipped_decisions": 0} for rule in rules}}

        return SimpleNamespace(
            model_id=model_id, family="hgb" if model_id.startswith("HGB_") else "tabm",
            model_sha256=digest, model_bytes=contents, archive_member_sha256={"fixture": digest},
            calibration_scores=baseline.calibration_scores, pooled_threshold=0.0,
            metadata={}, original_summary={}, benchmark={}, score=score,
            historical_fox_regression=fox_regression,
            training_history_summary={}, primary_training_diagnostics=None,
        )

    models = {model_id: make_candidate(model_id) for model_id in sorted(suite.planned_ids())}
    monkeypatch.setattr(suite, "load_baseline", lambda _: baseline)
    monkeypatch.setattr(comparators, "load_comparators", lambda _, __: models)
    monkeypatch.setattr(suite, "load_opened_fox_rows", lambda _: ["opened-fixture"])
    monkeypatch.setattr(suite, "runtime_environment", lambda _: {"fixture": True})

    result = suite.execute("fixture", qualification, protocol, output, "cuda", control=1)
    assert result.is_file()
    assert len(scored) == 22 and sum(device == "cpu" for _, device in scored) == 4
    assert len(preflight_scored) == 22
    summary = json.loads((output / "summary.json").read_bytes())
    pairs = json.loads((output / "paired_comparisons.json").read_bytes())
    assert summary["successful_score_passes"] == 22 and summary["fit_count"] == 0
    assert len(pairs) == 31 and all(row["status"] == "evaluated" for row in pairs)
    assert suite.execute("fixture", qualification, protocol, output, "cuda", control=1) == result
    assert len(scored) == 22
    assert len(preflight_scored) == 22

    with pytest.raises(ValueError, match="changed"):
        suite.execute("fixture", qualification, protocol, output, "cpu", control=1)
    assert len(scored) == 22
    assert len(preflight_scored) == 22

    protocol.write_bytes(source_protocol + b"\n")
    with pytest.raises(ValueError, match="source-qualified event registry"):
        suite.execute("fixture", qualification, protocol, output, "cuda", control=1)
    assert len(scored) == 22
