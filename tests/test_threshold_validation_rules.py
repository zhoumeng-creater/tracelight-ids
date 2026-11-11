from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ids.threshold_validation.rules import audit_reference_mix, paired_operating_point, replay_thresholds


def stream(times, scores, ids=None):
    return pd.DataFrame(
        {"row_id": range(len(times)) if ids is None else ids, "flow_end_ms": times, "score": scores}
    )


def test_registered_rank_strict_ties_and_short_pool_fallback():
    times = np.r_[np.arange(1000), [1000, 1000, 1010, 2500]]
    scores = np.r_[np.arange(1000), [990.0, 991.0, 100_000.0, 56.0]]
    events = stream(times, scores)
    releases = pd.Series(times, dtype="Int64")
    actual, audit = replay_thresholds(
        events,
        55.0,
        start_ms=1000,
        end_ms=2501,
        window_ms=1000,
        refresh_ms=100,
        trusted_normal_available_ms=releases,
    )
    assert actual.fixed_alert.tolist() == [True, True, True, True]
    assert actual.unlabeled_alert.tolist() == [False, True, True, True]
    assert actual.recent_normal_alert.equals(actual.unlabeled_alert)
    assert actual.unlabeled_threshold.tolist() == [990.0, 990.0, 990.0, 55.0]
    ref = audit.query("update_ms == 1000 and policy == 'unlabeled'").iloc[0]
    assert (ref.n_reference, ref["rank"], ref.max_flow_end_ms) == (1000, 991, 999)
    assert ref.reference_strictly_above == 9
    assert (
        audit.query("update_ms == 2500 and policy == 'unlabeled'").iloc[0].status == "insufficient_reference"
    )


def test_whole_millisecond_future_append_permutation_and_clock_grid():
    times = np.r_[np.arange(100), [100, 100, 110]]
    scores = np.r_[np.arange(100), [1e6, 1e7, 100.0]]
    prefix = stream(times, scores)
    config = {
        "fixed_threshold": 42.0,
        "start_ms": 109,
        "end_ms": 115,
        "window_ms": 100,
        "refresh_ms": 100,
        "min_reference": 99,
    }
    decisions, audit = replay_thresholds(prefix, **config)
    # The u=100 update cannot use the records at exactly 100, even when the caller evaluates from 109.
    assert decisions.unlabeled_threshold.item() == 99.0
    assert audit.query("policy == 'unlabeled'").iloc[0].n_reference == 100
    larger = pd.concat([prefix, stream([111, 115, 199], [1e9] * 3, ids=[120, 121, 122])], ignore_index=True)
    larger = larger.sample(frac=1, random_state=5).assign(label=1, attack_family="anything")
    other, _ = replay_thresholds(larger, **config)
    pd.testing.assert_frame_equal(
        decisions, other.loc[other.row_id.isin(decisions.row_id)].reset_index(drop=True)
    )


def test_confirmation_must_have_arrived_strictly_before_boundary_and_lag_window_shifts():
    times = np.arange(201)
    events = stream(times, times.astype(float))
    releases = pd.Series(times + 10, dtype="Int64")
    releases.iloc[110] = 200  # This old observation's feedback arrives exactly at u.
    releases.iloc[111] = pd.NA
    result, audit = replay_thresholds(
        events,
        600.0,
        start_ms=200,
        end_ms=201,
        refresh_ms=100,
        window_ms=101,
        min_reference=99,
        trusted_lag_ms=10,
        trusted_normal_available_ms=releases,
    )
    # Past scored flows are [89,190) for the confirmed rule and [99,200) for unlabeled.
    refs = audit.set_index("policy")
    assert refs.loc["recent_normal", "n_reference"] == 99
    assert refs.loc["recent_normal", "max_flow_end_ms"] == 189
    assert refs.loc["recent_normal", "max_confirmation_ms"] == 199
    assert refs.loc["unlabeled", "n_reference"] == 101
    assert result.recent_normal_threshold.item() == 189
    assert result.unlabeled_threshold.item() == 199


def test_no_trusted_normals_falls_back_and_posthoc_contamination_never_changes_decisions():
    n = 100
    events = stream(np.arange(n + 1), np.r_[np.zeros(98), [42.0, 42.0, 21.0]])
    settings = dict(start_ms=100, end_ms=101, window_ms=100, refresh_ms=100, min_reference=99)
    result, audit = replay_thresholds(events, 20.0, **settings)
    assert (
        result.fixed_alert.item() and result.recent_normal_alert.item() and not result.unlabeled_alert.item()
    )
    assert audit.query("policy == 'recent_normal'").iloc[0].status == "feedback_unavailable"
    labels = pd.Series(np.zeros(n + 1), dtype="Int64")
    labels.iloc[-4:] = [pd.NA, 1, 1, 1]
    mix = audit_reference_mix(events, audit, labels).iloc[0]
    assert (mix.known_attack, mix.known_normal, mix.unadjudicated) == (2, 97, 1)
    assert mix.attack_fraction_of_labeled == 2 / 99
    result2, audit2 = replay_thresholds(events.assign(label=labels, fake_confirmation=0), 20.0, **settings)
    pd.testing.assert_frame_equal(result, result2)
    pd.testing.assert_frame_equal(audit, audit2)
    feedback = events.flow_end_ms.astype("Int64")
    feedback[labels.ne(0).fillna(True)] = pd.NA
    _, audit3 = replay_thresholds(events, 20.0, trusted_normal_available_ms=feedback, **settings)
    assert audit3.query("policy == 'recent_normal'").iloc[0].status == "insufficient_reference"


def test_paired_operating_point_tracks_crossings_and_empty_families():
    n = 100
    frame = pd.DataFrame(
        {
            "row_id": range(n + 3),
            "label": [0] * n + [1, 1, 1],
            "attack_family": ["Benign"] * n + ["Scanning"] * 3,
            "raw": [True, True] + [False] * (n - 2) + [True, False, True],
            "adaptive": [True, False] + [False] * (n - 2) + [True, True, False],
        }
    )
    info = paired_operating_point(frame, "raw", "adaptive")
    assert info["normal_false_alert_budget"] == 1 and not info["baseline_within_one_percent"]
    assert info["candidate_within_one_percent"]
    assert info["groups"]["Scanning"]["net"] == 0
    assert info["groups"]["Scanning"]["added"] == info["groups"]["Scanning"]["lost"] == 1
    assert info["groups"]["Bot"]["rows"] == 0 and info["groups"]["Bot"]["candidate_rate"] is None
    assert paired_operating_point(frame.tail(3), "raw", "adaptive")["candidate_within_one_percent"] is None


def test_invalid_data_and_empty_output():
    events = stream([5, 6], [0.0, 1.0])
    params = {"start_ms": 10, "end_ms": 11}
    with pytest.raises(ValueError, match="align"):
        replay_thresholds(events, 0, trusted_normal_available_ms=pd.Series([0, 1], index=[20, 30]), **params)
    with pytest.raises(ValueError, match="predates"):
        replay_thresholds(events, 0, trusted_normal_available_ms=pd.Series([4, 50]), **params)
    with pytest.raises(ValueError, match="milliseconds"):
        replay_thresholds(events.assign(flow_end_ms=[5.5, 6.0]), 0, **params)
    with pytest.raises(ValueError, match="row ID"):
        replay_thresholds(events.assign(row_id=0), 0, **params)
    with pytest.raises(ValueError, match="min_reference"):
        replay_thresholds(events, 0, min_reference=12, **params)
    out, updates = replay_thresholds(events, 0, **params)
    assert out.empty and updates.empty
    assert audit_reference_mix(events, updates, pd.Series([pd.NA, pd.NA], dtype="Int64")).empty
