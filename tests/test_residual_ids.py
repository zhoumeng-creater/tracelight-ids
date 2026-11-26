from __future__ import annotations

from hashlib import sha256
import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.coverage_batch.materialize import RAW8_FIELDS
from scripts.residual_ids import run
from scripts.residual_ids.models import fit_mlp_pair
from scripts.residual_ids.policy import (
    evaluate_alerts,
    freeze_residual,
    paired_alerts,
    predict_residual,
    rank_diagnostic,
)


def test_residual_budget_uses_total_group_and_preserves_incumbent():
    c = {f"n{i}": float(i == 0) for i in range(200)}
    branches = {fid: 10.0 if fid in {"n1", "n100"} else 0. for fid in c if fid != "n0"}
    groups = {"first": list(c)[:100], "second": list(c)[100:]}
    rule = freeze_residual(c, branches, groups, 0.5)
    assert rule["threshold"] == 10.0 and rule["zero_remaining_FP_units"] == ["first"]
    assert [group["total_FP"] for group in rule["business_units"]] == [1, 0]
    assert predict_residual(
        {"hgb": True, "tie": False, "new": False}, {"tie": 10., "new": 11.}, rule,
    ) == {"hgb": True, "tie": False, "new": True}


def test_over_budget_incumbent_disables_rescue_and_rank_diagnostic():
    base = {"normal": 0.9, "positive": 0.}
    old = {"normal": True, "positive": False}
    units = {"page": ["normal"]}
    rule = freeze_residual({"normal": 0.9}, {}, units, 0.5)
    assert not rule["enabled"] and rule["threshold"] is None
    assert predict_residual(old, {"positive": 3.}, rule) == old
    assert rank_diagnostic(
        base, {"positive": 3.}, units, {"control": ["positive"], "download": None}, .5, old,
    ) == {"status": "incumbent_over_budget", "max_TP": None}


def test_paired_event_retains_unknown_and_unavailable_target():
    scope = {
        "normal_units": {"page": ["normal"]}, "attack_ids": {"control": ["positive"], "download": None},
        "events": [
            {"event_id": "event_a", "kind": "control", "label": 1, "association_mode": "all_confirmed",
             "candidate_set_complete": True, "candidate_flow_ids": ["positive"]},
            {"event_id": "event_b", "kind": "control", "label": 1, "association_mode": "one_of",
             "candidate_set_complete": False, "candidate_flow_ids": ["unknown"]},
        ],
    }
    old = {"normal": False, "positive": False, "unknown": False}
    new = old | {"positive": True, "unknown": True}
    a, b = evaluate_alerts(old, **scope), evaluate_alerts(new, **scope)
    diff = paired_alerts(old, new, normal_units=scope["normal_units"], attack_ids=scope["attack_ids"],
                         before_report=a, after_report=b)
    assert b["unknown_alerts"] == 1 and b["normal_units"][0]["FP"] == 0
    assert b["targets"]["download"]["TP"] is None
    assert diff["targets"]["control"]["gained_flow_ids"] == ["positive"]
    assert diff["events"][0]["lower_hit"] == {"before": False, "after": True, "gained": True, "lost": False}
    assert diff["events"][1]["lower_hit"] == {"before": None, "after": None, "gained": None, "lost": None}


def test_rank_ceiling_can_gain_confirmed_event_without_changing_flow_tp():
    scores = {"normal": 0., "confirmed": 0., "event_only": 0.}
    base = {fid: False for fid in scores}
    event = {
        "event_id": "observed_event", "kind": "control", "label": 1, "association_mode": "one_of",
        "candidate_set_complete": True, "candidate_flow_ids": ["event_only"],
    }
    units = {"site": ["normal"]}
    attacks = {"control": ["confirmed"], "download": None}
    old = evaluate_alerts(base, normal_units=units, attack_ids=attacks, events=[event])
    delta = paired_alerts(base, base, normal_units=units, attack_ids=attacks, before_report=old, after_report=old)
    rank = rank_diagnostic(scores, scores | {"event_only": 2.}, units, attacks, .5, base, [event])
    assert rank["max_TP"]["control"] == old["targets"]["control"]["TP"] == 0
    assert rank["max_definite_events"] == {"control": 1, "download": None}
    assert run.interpret(old, delta, rank, True) == "stop_no_gain_at_frozen_rule"


def test_small_mlp_match_only_differs_on_fixed_exposure_mask(tmp_path):
    rows = [
        {"flow_id": str(i), "label": i % 2,
         "raw8": dict(zip(RAW8_FIELDS, (6, 10 + i, 80, i, 1, 1, None if i % 3 == 0 else i, 500)))}
        for i in range(12)
    ]
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        alerts = {row["flow_id"]: False for row in rows}
        matched, _ = fit_mlp_pair(rows, alerts, seed=17, directory=tmp_path / "all")
        a, b = (matched[kind].score(rows) for kind in ("ordinary", "specialist"))
        assert a.shape == (len(rows),) and np.isfinite(a).all()
        assert np.array_equal(a, b)
        masked, _ = fit_mlp_pair(rows, alerts | {"2": True}, seed=17, directory=tmp_path / "masked")
        assert np.array_equal(a, masked["ordinary"].score(rows))
        assert not np.array_equal(b, masked["specialist"].score(rows))
    finally:
        torch.set_num_threads(prior_threads)


def test_complete_driver_contract_with_stub_models(tmp_path, monkeypatch):
    class StubModel:
        model_bytes = b"test"
        model_sha256 = sha256(model_bytes).hexdigest()

        def __init__(self, field):
            self.field = field

        def score(self, rows):
            return np.asarray([row[self.field] for row in rows], dtype=np.float64)

    baseline = StubModel("hgb")
    baseline.calibration_scores = {"c": 0.}
    tabm = {key: StubModel("tabm") for key in run.TABM}
    for predictor in tabm.values():
        predictor.calibration_scores = {"c": 0.}
    calibration = [{"flow_id": "c", "hgb": 0., "tabm": 0., "mlp": 0.}]
    fit = [{"flow_id": "fit", "hgb": 0., "tabm": 0., "mlp": 0.}]
    rows = [
        {"flow_id": fid, "stable_flow_id": fid, "dataset_role": "V", "label": label,
         "target_kind": kind, "hgb": 0., "tabm": value, "mlp": value}
        for fid, label, kind, value in (
            ("normal", 0, "normal", 0.), ("positive", 1, "control", 1.), ("unknown", None, "unknown", 0.),
        )
    ]
    group = {"normal_units": {"site": ["normal"]},
             "attack_ids": {"control": ["positive"], "download": None}, "events": [],
             "flow_ids": [r["flow_id"] for r in rows]}
    memory_bundle = SimpleNamespace(manifest={"artifacts": {}}, digest="manifest")
    monkeypatch.setattr(run, "Bundle", lambda _: memory_bundle)
    monkeypatch.setattr(run, "prepare", lambda *_: (
        baseline, tabm, {"calibration_units": {"site": ["c"]}}, fit, calibration,
    ))
    monkeypatch.setattr(run, "load_tabm", lambda checkpoint: checkpoint)
    monkeypatch.setattr(run, "evaluation_input", lambda *_: (rows, {"fox": group}))
    monkeypatch.setattr(run, "version", lambda _: "stub")

    def fit_stub(output, *_):
        destination = output / "mlp"
        destination.mkdir()
        models = {}
        for seed in run.SEEDS:
            for kind in ("ordinary", "specialist"):
                key = f"{kind}_s{seed}"
                (destination / f"{key}.pt").write_bytes(b"test")
                models[key] = StubModel("mlp")
        return models

    monkeypatch.setattr(run, "fit_all", fit_stub)
    monkeypatch.setattr(run, "measure", lambda *_: {key: {"p95_ms": 1.} for key in ("1", "256", "1024")})
    output = tmp_path / "run"
    status = run.run(tmp_path / "not-used.zip", tmp_path, output)
    assert status["fit_count"] == 6 and status["eval_row_count"] == 3
    assert set(status["specialist_assessments"]["fox"].values()) == {
        "stop_no_added_detection_over_matched_ordinary",
    }
    receipt = json.loads((output / "output_receipt.json").read_bytes())
    assert "operating_points.json" in receipt and "evaluation.json" in receipt
    for file, expected in receipt.items():
        data = (output / file).read_bytes()
        assert expected == {"bytes": len(data), "sha256": sha256(data).hexdigest()}
