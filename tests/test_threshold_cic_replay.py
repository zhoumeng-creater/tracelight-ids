from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.threshold_validation.cic_replay import MODELS, load_capture, replay_observed, summary
from ids.threshold_validation.rules import POLICIES


def sample():
    ticks = np.r_[np.arange(240), [260, 261, 269, 270]]
    family = np.array(["Benign"] * len(ticks), dtype=object)
    family[[135, 138]] = "Scanning"
    family[234:] = "Bot"
    score = np.where(family == "Benign", ticks % 12 - 10.0, 9.0)
    return pd.DataFrame(
        {
            "row_id": np.arange(len(ticks)),
            "flow_end_ms": ticks,
            "label": (family != "Benign").astype(int),
            "attack_family": family,
            "hgb48": score,
            "hgb52_history": score - 0.001 * ticks,
        }
    )


def test_r20_sample_only_warmup_and_frozen_replay():
    observation = sample().sample(frac=1, random_state=612)
    cal = {name: {"threshold": 8.0} for name in MODELS}
    config = {"refresh_ms": 20, "window_ms": 130, "min_reference": 99, "trusted_lag_ms": 15}
    rows, receipts, contamination = replay_observed(observation, cal, config)
    assert rows.flow_end_ms.is_monotonic_increasing
    assert set(receipts.arm) == MODELS.keys()
    for name in MODELS:
        ordered = observation.set_index("row_id").loc[rows.row_id]
        np.testing.assert_array_equal(rows[name + "/fixed"], ordered[name].gt(8.0))
    first = receipts.query("update_ms == 0 and policy != 'fixed'")
    assert first.n_reference.eq(0).all() and first.status.eq("insufficient_reference").all()
    assert contamination.iloc[0].reference_rows == 0
    assert contamination.known_attack.gt(0).any()
    report = summary(rows, receipts, config)
    assert set(report["between_models_same_policy"]) == set(POLICIES)
    assert len(report["adaptive_vs_fixed_same_model"]) == 4
    for policy in POLICIES:
        groups = report["between_models_same_policy"][policy]["groups"]
        assert groups["Scanning"]["rows"] == 2 and groups["Bot"]["rows"] == 10
    for arm in MODELS:
        for policy in POLICIES:
            assert receipts.loc[
                receipts.arm.eq(arm) & receipts.policy.eq(policy), "decision_rows"
            ].sum() == len(rows)


def test_future_truth_changes_never_revise_earlier_alarms():
    obs = sample()
    cal = {name: {"threshold": 8.0} for name in MODELS}
    config = {"refresh_ms": 20, "window_ms": 130, "min_reference": 99, "trusted_lag_ms": 15}
    a, _, _ = replay_observed(obs, cal, config)
    changed = obs.copy()
    changed.loc[changed.flow_end_ms >= 150, "label"] = 0
    changed.loc[changed.flow_end_ms >= 150, "attack_family"] = "Benign"
    b, _, _ = replay_observed(changed, cal, config)
    cols = [arm + "/" + policy for arm in MODELS for policy in POLICIES]
    pd.testing.assert_frame_equal(a.loc[a.flow_end_ms < 150, cols], b.loc[b.flow_end_ms < 150, cols])
    label_free = [arm + "/" + policy for arm in MODELS for policy in ("unlabeled", "fixed")]
    pd.testing.assert_frame_equal(a[label_free], b[label_free])


def test_foreign_upload_refused(tmp_path: Path):
    file = tmp_path / "foreign.zip"
    file.write_bytes(b"not a result archive")
    with pytest.raises(ValueError, match="pinned SHA"):
        load_capture(file)
