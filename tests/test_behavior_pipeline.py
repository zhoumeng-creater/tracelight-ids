import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer.decisions import decide
from ids.behavior_transfer.evaluation import judge
from ids.common import read_json
from ids.flow_history.execution import stage


def inputs(tmp_path, *, complete=True):
    n = 500
    clock = np.arange(n, dtype=np.int64) * 1000
    if complete:
        clock[400:] += 3_600_000
    frame = pd.DataFrame({"row_id": np.arange(1, n + 1), "flow_end_ms": clock,
                          "hist_dst_ports_60s": np.r_[np.zeros(400), np.full(100, 50)],
                          "hist_dst_ips_60s": np.ones(n), "ip_protocol": 6, "resp_port": 80})
    score = pd.DataFrame({"row_id": frame.row_id, "raw": False, "contact": False,
                          "quality": False, "add_quality": False,
                          "score_raw": -np.ones(n), "score_quality": -np.ones(n)})
    score.loc[410:415, ["raw", "add_quality"]] = True
    score.loc[410:415, "score_raw"] = 5
    proxy = pd.DataFrame({"row_id": frame.row_id, "scan_proxy": False})
    paths = [tmp_path / k for k in ("scores.parquet", "features.parquet", "proxy.parquet")]
    for f, p in zip((score, frame, proxy), paths):
        f.to_parquet(p, index=False)
    cal = pd.DataFrame({"hist_dst_ports_60s": np.zeros(600), "hist_dst_ips_60s": np.ones(600),
                        "score_raw": np.zeros(600), "score_quality": np.zeros(600)})
    return paths, cal


def test_future_values_cannot_change_registered_prefix(tmp_path):
    paths, cal = inputs(tmp_path)
    out, report = tmp_path / "sealed.parquet", tmp_path / "cut.json"
    decide(*paths, cal, out, report)
    original = read_json(report)["registration"]
    bits = pd.read_parquet(out)
    assert bits.loc[400:, "ports_local"].all()
    assert not bits.loc[:399, "ports_local"].any()
    assert (bits["frozen_plus_ports_local"] >= bits["add_quality_old"]).all()
    values = pd.read_parquet(paths[0])
    values.loc[450:, "score_quality"] = 1e8
    values.to_parquet(paths[0], index=False)
    decide(*paths, cal, out, report)
    assert read_json(report)["registration"] == original


def test_incomplete_hour_keeps_original_fallback(tmp_path):
    paths, cal = inputs(tmp_path, complete=False)
    out, report = tmp_path / "sealed.parquet", tmp_path / "cut.json"
    decide(*paths, cal, out, report)
    bits = pd.read_parquet(out)
    assert bits.phase.eq("startup").all()
    assert np.array_equal(bits.quality_counts_old, bits.quality_counts_local)
    assert read_json(report)["registration"]["ports"]["local"]["status"] == "old_calibration_fallback"


def test_joins_require_exact_row_order(tmp_path):
    paths, cal = inputs(tmp_path)
    frame = pd.read_parquet(paths[2]).iloc[::-1]
    frame.to_parquet(paths[2], index=False)
    with pytest.raises(ValueError, match="row order"):
        decide(*paths, cal, tmp_path / "out", tmp_path / "report")


def test_full_decision_to_native_event_report(tmp_path):
    paths, cal = inputs(tmp_path)
    sealed, report = tmp_path / "sealed.parquet", tmp_path / "cut.json"
    decide(*paths, cal, sealed, report)
    truth = pd.DataFrame({"row_id": np.arange(1, 501), "window": "2025-04-10",
                          "family": ["Benign"] * 450 + ["Scanning"] * 49 + ["Unresolved"],
                          "native_detail": ["normal"] * 450 + ["scan"] * 49 + ["related"],
                          "event_id": [""] * 450 + ["one-event"] * 49 + [""],
                          "evidence_level": "author_event", "uid": [str(i) for i in range(500)]})
    truth.loc[300, ["family", "native_detail", "event_id"]] = ["Scanning", "scan", "one-event"]
    truth.to_parquet(tmp_path / "truth.parquet", index=False)
    folder = tmp_path / "report"
    judge(sealed, tmp_path / "truth.parquet", paths[1], folder,
          source={"name": "synthetic", "role": "fixture"}, group="mlp29")
    budget = pd.read_csv(folder / "budget.csv.gz")
    row = budget.loc[budget.phase.eq("after") & budget.policy.eq("ports_local")].iloc[0]
    assert row.normal == 50 and row.fp == 50 and not row.within_budget
    assert row.unresolved_rows == 1 and row.unresolved_alerts == 1
    pairs = pd.read_csv(folder / "paired.csv.gz")
    scan = pairs.loc[pairs.phase.eq("after") & pairs.family.eq("Scanning") &
                     pairs.policy.eq("ports_local") & pairs.baseline.eq("add_quality_old")].iloc[0]
    assert scan.gained == 49 and scan.lost == 0
    event = pd.read_csv(folder / "events.csv.gz")
    assert event.event_id.nunique() == 1
    after = event.loc[event.phase.eq("after") & event.policy.eq("ports_local")].iloc[0]
    assert after.first_record_ms == 300000
    assert after.observed_delay_ms == after.first_alert_ms - 300000
    normal = pd.read_csv(folder / "normal_service_fp.csv.gz")
    assert ((normal.policy == "ports_local") & (normal.phase == "after")).any()
    assert read_json(folder / "summary.json")["events_with_known_attack_labels"] == 1


def test_stage_receipts_detect_changed_outputs(tmp_path):
    output = tmp_path / "data"
    def create():
        output.write_text("first")
        return [output]
    stage(tmp_path, "fixture", "binding", create)
    stage(tmp_path, "fixture", "binding", lambda: pytest.fail("Completed stage reran"))
    output.write_text("changed")
    with pytest.raises(ValueError, match="outputs changed"):
        stage(tmp_path, "fixture", "binding", create)
