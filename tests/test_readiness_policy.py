from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer import policy
from ids.behavior_transfer.decisions import COMPONENTS, HOUR_MS, decide
from ids.common import read_json
from ids.readiness_replay.readiness import FIXED_POLICIES, MIN_REFERENCE_ROWS, repair_decisions


def inputs(tmp_path, clock, *, constant=False):
    clock = np.asarray(clock, dtype=np.int64)
    n = len(clock)
    frame = pd.DataFrame({"row_id": np.arange(1, n + 1), "flow_end_ms": clock,
                          "hist_dst_ports_60s": np.zeros(n) if constant else np.arange(n) % 13,
                          "hist_dst_ips_60s": np.zeros(n) if constant else np.arange(n) % 5})
    scores = pd.DataFrame({"row_id": frame.row_id})
    for offset, view in enumerate(("raw", "contact", "quality")):
        scores["score_" + view] = np.zeros(n) if constant else (np.arange(n) + offset) % 7 / 7 - .5
        scores[view] = scores["score_" + view] > 0
    scores["add_quality"] = scores.raw | scores.quality
    cal = pd.DataFrame({"hist_dst_ports_60s": np.zeros(600), "hist_dst_ips_60s": np.zeros(600),
                        "score_raw": np.zeros(600), "score_quality": np.zeros(600)})
    source, scored, proxy = [tmp_path / (name + ".parquet") for name in ("features", "scores", "proxy")]
    frame.to_parquet(source, index=False)
    scores.to_parquet(scored, index=False)
    pd.DataFrame({"row_id": frame.row_id, "scan_proxy": np.arange(n) % 41 == 0}).to_parquet(proxy, index=False)
    output, report = tmp_path / "old.parquet", tmp_path / "old.json"
    decide(scored, source, proxy, cal, output, report)
    return frame, scores, pd.read_parquet(output), read_json(report)


@pytest.mark.parametrize("first_hour_count", [1, 2])
def test_sparse_first_hour_waits_then_uses_one_shared_prefix(tmp_path, first_hour_count):
    clocks = np.r_[np.arange(first_hour_count) * 1000, HOUR_MS + np.arange(450) * 1000]
    args = inputs(tmp_path, clocks)
    original = deepcopy(args)
    out, report = repair_decisions(*args)
    assert MIN_REFERENCE_ROWS == 399
    assert report["status"] == "ready" and report["reason"] == "waited_for_samples"
    assert report["prefix_rows"] == 399 and report["cutoff_ms"] == clocks[399]
    assert report["waiting_for_readiness_rows"] == 399 - first_hour_count
    assert out.phase.iloc[:399].eq("startup").all() and out.phase.iloc[399:].eq("after").all()
    blind = args[0].assign(score_raw=args[1].score_raw, score_quality=args[1].score_quality)
    for name in COMPONENTS:
        old = args[3]["registration"][name]
        assert all(rule["threshold"] is None for rule in old["local"]["rules"].values())
        assert np.array_equal(out[name + "_local"].iloc[:399], policy.apply(blind, old["old"])[:399])
        assert report["registration"][name]["local"]["local_rows"] == 399
        assert all(rule["threshold"] is not None
                   for rule in report["registration"][name]["local"]["rules"].values())
    for column in FIXED_POLICIES:
        pd.testing.assert_series_equal(out[column], args[2][column])
    assert np.array_equal(out.frozen_plus_ports_local, out.add_quality_old | out.ports_local)
    for i in range(3):
        pd.testing.assert_frame_equal(args[i], original[i])
    assert args[3] == original[3]


def test_398_399_boundary_excludes_the_complete_switch_clock_group(tmp_path):
    clocks = np.r_[np.arange(398), [HOUR_MS] * 3, HOUR_MS + 1, HOUR_MS + 2]
    args = inputs(tmp_path, clocks)
    out, report = repair_decisions(*args)
    assert report["cutoff_ms"] == HOUR_MS + 1 and report["prefix_rows"] == 401
    assert report["blocked_candidate_clock_group_rows"] == 3
    assert out.phase.iloc[:401].eq("startup").all()
    first = report["registration"]
    args[1].loc[401:, ["score_raw", "score_quality", "score_contact"]] = 1e12
    _, changed = repair_decisions(*args)
    assert changed["registration"] == first
    assert changed["prefix_score_summary"] == report["prefix_score_summary"]


def test_exact_399_predecessors_are_sufficient(tmp_path):
    args = inputs(tmp_path, np.r_[np.arange(398), HOUR_MS, HOUR_MS + 1])
    out, report = repair_decisions(*args)
    assert report["prefix_rows"] == 399 and report["cutoff_ms"] == HOUR_MS + 1
    assert out.phase.iloc[-1] == "after" and out.phase.iloc[-2] == "startup"
    rules = report["registration"]["raw_quality_counts"]["local"]["rules"]
    assert all(rule["rank"] == 399 and rule["n_reference_rows"] == 399 for rule in rules.values())


@pytest.mark.parametrize("clock,reason", [
    (np.r_[np.arange(398), HOUR_MS], "insufficient_strictly_prior_rows"),
    (np.r_[0, np.full(500, HOUR_MS)], "insufficient_strictly_prior_rows"),
    (np.arange(600), "hour_not_elapsed"),
])
def test_never_ready_keeps_older_rules_for_every_row(tmp_path, clock, reason):
    args = inputs(tmp_path, clock)
    out, report = repair_decisions(*args)
    assert report["status"] == "not_ready" and report["reason"] == reason
    assert report["cutoff_ms"] is None and report["prefix_rows"] == 0
    assert out.phase.eq("startup").all()
    blind = args[0].assign(score_raw=args[1].score_raw, score_quality=args[1].score_quality)
    for name in COMPONENTS:
        assert np.array_equal(out[name + "_local"], policy.apply(blind, args[3]["registration"][name]["old"]))


def test_original_adequate_hour_preserves_nonobserved_cutoff_and_exact_r25_output(tmp_path):
    args = inputs(tmp_path, np.r_[np.arange(400), HOUR_MS + 10 + np.arange(200)])
    out, report = repair_decisions(*args)
    pd.testing.assert_frame_equal(out, args[2])
    assert report["reason"] == "original_hour_ready" and report["cutoff_ms"] == HOUR_MS
    assert report["first_after_clock_ms"] == HOUR_MS + 10
    for name in COMPONENTS:
        assert report["registration"][name]["local"]["rules"] == args[3]["registration"][name]["local"]["rules"]


def test_all_tied_finite_scores_are_ready_and_labels_are_irrelevant(tmp_path):
    args = inputs(tmp_path, np.r_[np.arange(399), HOUR_MS, HOUR_MS + 1], constant=True)
    out, report = repair_decisions(*args)
    assert report["status"] == "ready" and report["prefix_rows"] == 399
    assert all(v["all_prefix_equal"] for v in report["prefix_score_summary"].values())
    assert not out.raw_quality_counts_local.any()
    args[0]["family"] = "an unused label"
    args[1]["truth"] = 1
    changed, other = repair_decisions(*args)
    pd.testing.assert_frame_equal(changed, out)
    assert other == report


@pytest.mark.parametrize("bad", [
    "id", "fractional_clock", "clock_backwards", "clock_mismatch", "score_clock", "nan", "old_phase",
])
def test_rejects_misalignment_and_invalid_values(tmp_path, bad):
    args = inputs(tmp_path, np.r_[np.arange(400), HOUR_MS + np.arange(10)])
    if bad == "id":
        args[1].loc[1, "row_id"] = 1
    elif bad == "fractional_clock":
        args[0]["flow_end_ms"] = args[0].flow_end_ms.astype(float) + .5
    elif bad == "clock_backwards":
        args[0].loc[3, "flow_end_ms"] = 1
        args[2].loc[3, "flow_end_ms"] = 1
    elif bad == "clock_mismatch":
        args[2].loc[2, "flow_end_ms"] += 1
    elif bad == "score_clock":
        args[1]["flow_end_ms"] = args[0].flow_end_ms + 1
    elif bad == "nan":
        args[1].loc[2, "score_contact"] = np.nan
    else:
        args[2].loc[0, "phase"] = "after"
    with pytest.raises(ValueError):
        repair_decisions(*args)
