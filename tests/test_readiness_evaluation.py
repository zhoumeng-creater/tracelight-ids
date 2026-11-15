import json

import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer.evaluation import judge
from ids.readiness_replay.evaluation import (
    RARE_FAMILIES,
    evaluate_frames,
    evaluation_scopes,
    prepare_context,
)


POLICIES = (
    "raw_old",
    "contact_old",
    "quality_old",
    "add_quality_old",
    "ports_local",
    "ports_old",
    "targets_local",
    "targets_old",
    "counts_local",
    "counts_old",
    "raw_local",
    "quality_local",
    "quality_counts_local",
    "quality_counts_old",
    "raw_quality_counts_local",
    "frozen_plus_ports_local",
    "frozen_plus_counts_local",
    "no_reply_scan_proxy",
)
SOURCE = {
    "name": "synthetic-source",
    "role": "forward_mixed",
    "provider": "synthetic",
    "source_id": "fixture",
}


def fixture(size=720):
    rng = np.random.default_rng(105)
    ids = np.arange(1, size + 1)
    day = 1736121600000
    clock = day + (np.arange(size) // 3) * 1000
    clock[size // 2 :] += 86400000
    family = np.array(["Benign", "Scanning", "OtherAttack", "Unresolved", "ExternalC2", "Download"])[
        np.arange(size) % 6
    ]
    family[size // 2 :] = np.where(family[size // 2 :] == "Benign", "Unresolved", family[size // 2 :])
    feature = pd.DataFrame(
        dict(
            row_id=ids,
            flow_end_ms=clock,
            uid=[f"U{i}" for i in ids],
            ip_protocol=np.where(ids % 3, 6, 17),
            resp_port=np.array([22, 80, 443, -1])[ids % 4],
            orig_port=10000 + ids,
            duration_seconds=rng.random(size),
            hist_dst_ports_60s=ids % 9,
            hist_dst_ips_60s=ids % 5,
            source="source-host",
            destination="destination-host",
        )
    )
    truth = pd.DataFrame(
        dict(
            row_id=ids,
            uid=feature.uid,
            window=pd.to_datetime(clock, unit="ms", utc=True).strftime("%Y-%m-%d"),
            family=family,
            native_detail="full author explanation",
            native_type=np.where(ids % 4, "short", None),
            event_id=np.where(np.isin(family, ["Scanning", "ExternalC2", "Download"]), "same-stage", ""),
            evidence_level=np.where(ids % 3, "author_window", None),
            exposure=np.where(ids % 2, "new", "opened"),
        )
    )
    scores = pd.DataFrame(
        dict(
            row_id=ids,
            score_raw=rng.random(size),
            score_contact=rng.random(size),
            score_quality=rng.random(size),
        )
    )
    old = pd.DataFrame(dict(row_id=ids, flow_end_ms=clock, phase=np.where(ids <= 130, "startup", "after")))
    for policy in POLICIES:
        old[policy] = rng.random(size) < 0.11
    new = old.copy()
    new["phase"] = np.where(ids <= 399, "startup", "after")
    return feature, scores, truth, old, new


def run(tmp_path, frames, context=None):
    return evaluate_frames(*frames, source=SOURCE, group="hgb", output=tmp_path, context=context)


def normalized(frame, columns):
    frame = frame.loc[:, columns].copy()
    # Deterministic sorting including empty string fields as serialized by pandas.
    keys = frame.astype(str).agg("|".join, axis=1)
    return frame.iloc[np.argsort(keys.to_numpy(), kind="stable")].reset_index(drop=True)


def test_exact_original_judge_parity_for_counts_services_events_and_zero_normal(tmp_path):
    feature, scores, truth, old, new = fixture()
    inputs = tmp_path / "input"
    inputs.mkdir()
    for name, frame in (("features", feature), ("truth", truth), ("scores", scores), ("old", old)):
        frame.to_parquet(inputs / (name + ".parquet"), index=False)
    judge(
        inputs / "old.parquet",
        inputs / "truth.parquet",
        inputs / "features.parquet",
        tmp_path / "legacy",
        source=SOURCE,
        group="hgb",
        scores_file=inputs / "scores.parquet",
    )
    context = prepare_context(feature, truth)
    run(tmp_path / "new", (feature, scores, truth, old, new), context)
    scopes = {"all": "all", "original_startup": "startup", "original_after": "after"}
    for name in ("budget", "paired", "services", "events", "normal_service_fp"):
        expected = pd.read_csv(tmp_path / "legacy" / (name + ".csv.gz"))
        actual = pd.read_csv(tmp_path / "new" / (name + ".csv.gz"))
        actual = actual[actual.phase.isin(scopes)].copy()
        actual["phase"] = actual.phase.map(scopes)
        if "version" in actual:
            actual = actual[actual.version == "original"]
        else:
            actual = actual[actual.comparison.isin(["raw_old", "add_quality_old"])]
        pd.testing.assert_frame_equal(
            normalized(actual, list(expected)),
            normalized(expected, list(expected)),
            check_dtype=False,
            check_exact=True,
        )
    assert json.loads((tmp_path / "new/summary.json").read_text())["scope_rows"] == {
        "all": 720,
        "startup": 399,
        "after": 321,
        "original_startup": 130,
        "original_after": 590,
    }


def test_repaired_pairs_follow_same_policy_and_fixed_original_baselines(tmp_path):
    frames = list(fixture(60))
    original = frames[3].copy(deep=True)
    frames[4].loc[:9, "quality_counts_local"] = ~frames[3].loc[:9, "quality_counts_local"]
    run(tmp_path, frames)
    pairs = pd.read_csv(tmp_path / "paired.csv.gz")
    changed = pairs[
        (pairs.phase == "all")
        & (pairs.policy == "quality_counts_local")
        & (pairs.comparison == "same_policy")
    ]
    assert int((changed.gained + changed.lost).sum()) == 10
    assert (pairs.before + pairs.gained - pairs.lost == pairs.after).all()
    assert set(pairs.comparison) == {"same_policy", "raw_old", "add_quality_old"}
    pd.testing.assert_frame_equal(original, frames[3])


def test_event_start_uses_full_source_across_phases_and_utc_days(tmp_path):
    frames = list(fixture(12))
    frames[2]["family"] = "Scanning"
    frames[2]["event_id"] = "one-stage"
    frames[2]["evidence_level"] = "window"
    for pred in frames[3:]:
        pred["phase"] = ["startup"] * 3 + ["after"] * 9
        pred["quality_counts_local"] = False
    frames[4].loc[7, "quality_counts_local"] = True
    run(tmp_path, frames)
    events = pd.read_csv(tmp_path / "events.csv.gz")
    after = events[
        (events.phase == "after")
        & (events.policy == "quality_counts_local")
        & (events.comparison == "same_policy")
    ]
    assert len(after) == 2
    assert set(after.first_record_ms) == {int(frames[0].flow_end_ms.iloc[0])}
    hit = after[after.after == 1].iloc[0]
    assert hit.observed_delay_ms == int(frames[0].flow_end_ms.iloc[7] - frames[0].flow_end_ms.iloc[0])
    assert hit.observed_delay_ms >= 86400000


@pytest.mark.parametrize(
    "normal,fp,expected", [(99, 0, True), (99, 1, False), (100, 1, True), (199, 2, False), (200, 2, True)]
)
def test_integer_budget_is_inclusive_floor_one_percent(tmp_path, normal, fp, expected):
    frames = list(fixture(normal))
    frames[2]["window"] = "2025-01-06"
    frames[2]["family"] = "Benign"
    frames[2]["event_id"] = ""
    for pred in frames[3:]:
        pred["raw_old"] = False
        pred.loc[: fp - 1, "raw_old"] = True if fp else False
    run(tmp_path, frames)
    b = pd.read_csv(tmp_path / "budget.csv.gz")
    row = b[(b.phase == "all") & (b.version == "repaired") & (b.policy == "raw_old")].iloc[0]
    assert row.normal == normal and row.fp == fp and row.cap == normal // 100
    assert bool(row.within_budget) == expected
    assert pd.read_csv(tmp_path / "events.csv.gz").empty


def test_zero_normal_is_not_a_passing_budget_and_unknown_alerts_are_retained(tmp_path):
    frames = list(fixture(30))
    frames[2]["family"] = "Unresolved"
    frames[3]["raw_old"] = True
    frames[4]["raw_old"] = True
    run(tmp_path, frames)
    budget = pd.read_csv(tmp_path / "budget.csv.gz")
    b = budget[(budget.phase == "all") & (budget.policy == "raw_old")]
    assert (b.normal == 0).all() and b.within_budget.isna().all() and b.fpr.isna().all()
    assert (b.unresolved_rows == b.unresolved_alerts).all()


def test_all_rare_rows_have_uid_values_scores_and_both_versions_without_sampling(tmp_path):
    frames = fixture(120)
    run(tmp_path, frames)
    rare = pd.read_parquet(tmp_path / "rare_rows.parquet")
    expected = frames[2].family.isin(RARE_FAMILIES)
    assert rare.row_id.tolist() == frames[2].loc[expected, "row_id"].tolist()
    assert rare.uid.tolist() == frames[2].loc[expected, "uid"].tolist()
    assert "feature__orig_port" in rare and "score__score_raw" in rare
    assert all(f"{version}__{policy}" in rare for version in ("original", "repaired") for policy in POLICIES)
    assert len(rare) > 3
    examples = pd.read_csv(tmp_path / "examples.csv.gz")
    values = pd.read_parquet(tmp_path / "example_values.parquet")
    assert not examples.family.isin(RARE_FAMILIES).any()
    assert set(examples.row_id) == set(values.row_id)
    assert values.row_id.is_unique
    assert (
        examples.groupby(["window", "family", "native_group", "policy", "comparison", "change"], dropna=False)
        .size()
        .max()
        <= 3
    )


def test_context_reuse_empty_native_values_and_absent_exposure(tmp_path):
    frames = list(fixture(36))
    frames[2] = frames[2].drop(columns=["exposure", "evidence_level", "native_type"])
    frames[2]["native_detail"] = None
    context = prepare_context(frames[0], frames[2])
    run(tmp_path / "one", frames, context)
    run(tmp_path / "two", frames, context)
    rows = pd.read_csv(tmp_path / "one/paired.csv.gz")
    assert rows.native_group.isna().all()
    assert set(rows.exposure) == {SOURCE["role"]}
    assert set(rows.evidence_level) == {"author_label"}
    assert pd.read_csv(tmp_path / "one/budget.csv.gz").equals(pd.read_csv(tmp_path / "two/budget.csv.gz"))
    other = list(frames)
    other[2] = other[2].copy()
    with pytest.raises(ValueError, match="different feature/truth"):
        run(tmp_path / "wrong", other, context)


def test_identity_and_boolean_contracts_are_enforced(tmp_path):
    frames = list(fixture(12))
    frames[4]["raw_old"] = frames[4].raw_old.astype("uint8")
    with pytest.raises(ValueError, match="Boolean"):
        run(tmp_path, frames)
    frames = list(fixture(12))
    frames[1].loc[0, "row_id"] = 99
    with pytest.raises(ValueError, match="scores row"):
        run(tmp_path, frames)
    frames = list(fixture(12))
    frames[4].loc[0, "phase"] = "maybe"
    with pytest.raises(ValueError, match="phase"):
        evaluation_scopes(frames[3], frames[4])
