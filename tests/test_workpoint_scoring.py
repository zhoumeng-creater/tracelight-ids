"""Regression coverage for frozen three-view workpoint scoring using synthetic records."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from ids.common import write_json
from ids.flow_history.trees import FORMAT
from ids.response_evidence.experiment import ARMS
from ids.route_batch.metrics import threshold
from ids.raw_workpoint import controls, scoring


def test_probe_pairs_reference_and_device_fp_at_strict_cut():
    frozen = pd.DataFrame({k: [True] + [False] * 99 for k in scoring.POLICIES})
    actual = frozen.copy()
    actual.loc[1, "add_contact"] = True
    actual.loc[[0, 1], "quality"] = [False, True]
    paired = controls._paired_policies(frozen, actual)
    assert paired["add_contact"] == {
        "registered_fp": 1,
        "actual_fp": 2,
        "added": 1,
        "lost": 0,
        "changed_alerts": 1,
        "fp_cap": 1,
        "within_budget": False,
    }
    assert paired["quality"]["within_budget"]
    assert paired["quality"]["lost"] == paired["quality"]["added"] == 1
    assert paired["quality"]["changed_alerts"] == 2


def test_residual_budget_ties_and_zero_remaining():
    a = np.array([9.0, 7.0, 7.0, 4.0, 3.0, 2.0])
    raw = np.array([True, False, False, False, False, False])
    rule = scoring.residual_rule(a, raw, 2)
    assert rule["threshold"] == 7 and rule["added_calibration_fp"] == 0
    rule = scoring.residual_rule(a, raw, 1)
    assert rule["remaining_fp"] == 0 and rule["threshold"] == 7 and rule["added_calibration_fp"] == 0
    assert np.array_equal(scoring.decisions([7, 8], rule), [False, True])
    with pytest.raises(ValueError, match="already exceeds"):
        scoring.residual_rule(a, raw, 0)


def test_30s_service_boundary_and_raw_retention():
    x = pd.DataFrame(
        {
            "ip_protocol": [17, 17, 17, 6, 17, 17],
            "orig_port": [99, 443, 1, 443, 1, 1],
            "resp_port": [443, 1234, 443, 80, 40, 443],
            "duration_seconds": [29.999, 30, -1, 100, 120, 30],
            "resp_ip_bytes": [0, 0, 0, 65_536, 0, 65_536],
        }
    )
    values = {k: np.array([9.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for k in scoring.VIEWS}
    values["contact"][4] = 4.0
    frozen = {k: {"threshold": 5.0} for k in scoring.VIEWS}
    supplements = {k: {"threshold": 3.0} for k in ("contact", "quality")}
    out = scoring.make_decisions(values, frozen, supplements, x)
    assert out["add_contact"].tolist() == [True, False, False, False, True, False]
    assert out["add_quic30"].tolist() == [True, True, False, False, False, True]
    assert out["add_web64k"].tolist() == [True, False, False, True, False, False]
    assert np.all(out["add_quic30"] >= out["raw"])
    x.loc[3, "resp_ip_bytes"] = 65_535
    assert not scoring.make_decisions(values, frozen, supplements, x)["add_web64k"][3]
    x.loc[3, ["resp_ip_bytes", "orig_port", "resp_port"]] = [65_536, 80, 1]
    assert not scoring.make_decisions(values, frozen, supplements, x)["add_web64k"][3]


def build_inputs(root):
    gen = np.random.default_rng(13)
    n = 410
    frame = pd.DataFrame({k: gen.normal(size=n) for k in ARMS["quality"]})
    frame["row_id"] = np.arange(1, n + 1)
    frame.loc[405:, ["ip_protocol", "resp_port", "duration_seconds"]] = [17, 443, 30.0]
    targets = (frame.duration_seconds > 0).astype(int)
    cal = pd.DataFrame({"capture": "synthetic-only", "row_id": np.arange(1, 201)})
    cuts = {}
    expected = {}
    with threadpool_limits(1):
        for k in scoring.VIEWS:
            cols = list(ARMS[k])
            matrix = np.ascontiguousarray(frame[cols], dtype=float)
            model = HistGradientBoostingClassifier(max_iter=3, max_leaf_nodes=4, random_state=14).fit(
                matrix[:200], targets.iloc[:200]
            )
            joblib.dump(
                {"format": FORMAT, "features": cols, "estimator": model}, root / ("hgb-" + k + ".joblib")
            )
            cal[k] = model.decision_function(matrix[200:400])
            cuts[k] = threshold(cal[k])
            expected[k] = model.decision_function(matrix[400:])
    write_json(root / "hgb.json", cuts)
    cal.to_parquet(root / "hgb.parquet", index=False)
    held = frame.iloc[400:].copy()
    held["row_id"] = np.arange(1, 11)
    held.to_parquet(root / "features.parquet", row_group_size=4, index=False)
    held[["row_id"]].to_parquet(root / "raw.parquet", row_group_size=3, index=False)
    return held, expected, cuts


def run(root, **kw):
    return scoring.score_group(
        root / "features.parquet",
        root / "raw.parquet",
        root,
        root,
        "hgb",
        root / "out.parquet",
        batch_rows=3,
        threads=1,
        device="cpu",
        **kw,
    )


def test_full_frozen_entry_actual_hgb_and_incomplete_raw_atomic(tmp_path):
    held, expected, cuts = build_inputs(tmp_path)
    report = run(tmp_path)
    result = pd.read_parquet(tmp_path / "out.parquet")
    assert result.row_id.tolist() == held.row_id.tolist()
    for arm in scoring.VIEWS:
        assert np.array_equal(result["score_" + arm], expected[arm])
        assert np.array_equal(result[arm], expected[arm] > cuts[arm]["threshold"])
    for name in scoring.POLICIES:
        assert result[name].dtype == bool and result[name].sum() == report["alerts"][name]
    assert result.add_quic30.iloc[-5:].all()
    original = (tmp_path / "out.parquet").read_bytes()
    pq.write_table(pa.table({"row_id": [1, 3, 2, 4, 5, 6, 7, 8, 9, 10]}), tmp_path / "raw.parquet")
    with pytest.raises(ValueError, match="physical rows"):
        run(tmp_path)
    assert original == (tmp_path / "out.parquet").read_bytes()
    assert not list(tmp_path.glob("out-*.tmp"))


@pytest.mark.parametrize(("group", "device"), [("hgb", "cuda:0"), ("mlp17", "cpu"), ("unknown", "cuda")])
def test_no_silent_device_or_group_substitution(tmp_path, group, device):
    with pytest.raises(ValueError):
        scoring.score_group(
            Path("a"),
            Path("b"),
            tmp_path,
            tmp_path,
            group,
            tmp_path / "o",
            batch_rows=4,
            threads=1,
            device=device,
        )
