"""The fixed bounded pool covers train families without changing split groups."""
import numpy as np
import pandas as pd
import pytest

from ids.stage4 import sample_groups, stratified_selection_pool


def sample_frame():
    labels = ["Normal"] * 600 + ["Portscan"] * 300 + ["Other"] * 99 + ["Rare"]
    groups = [f"n{i}" for i in range(600)] + ["giant"] * 300 + [f"o{i}" for i in range(99)] + ["rare"]
    return pd.DataFrame({"sample_id": [f"s{i}" for i in range(1000)], "attack_cat": labels,
                         "label": [0]*600 + [1]*400, "group_hash": groups,
                         "feature_hash": [f"f{i}" for i in range(1000)]})


def test_stratified_pool_covers_giant_family_at_bounded_budget_and_keeps_original_ids():
    frame = sample_frame()
    pool, record = stratified_selection_pool(frame, 100, 20261102)
    assert len(pool) == 100
    assert record["missing_families"] == []
    assert record["family_counts"]["Rare"] == 1
    assert 28 <= record["family_counts"]["Portscan"] <= 31
    assert record["partial_source_groups"] == 1
    assert record["family_group_counts"]["Portscan"] == 1
    assert record["all_source_group_rows_retained"] is False
    assert pool.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(frame.loc[pool.index], pool)
    other, other_record = stratified_selection_pool(frame, 100, 20261102)
    pd.testing.assert_frame_equal(pool, other)
    assert record == other_record
    original_counts = pool.groupby("group_hash").size()
    for seed in range(4):
        outer, _ = sample_groups(pool, seed, fraction=.8)
        assert (outer.groupby("group_hash").size() == original_counts.loc[outer.group_hash.unique()].sort_index()).all()


def test_pool_uses_all_rows_when_within_budget_and_never_duplicates_rows():
    frame = sample_frame()
    pool, record = stratified_selection_pool(frame, 2000, 17)
    pd.testing.assert_frame_equal(frame, pool)
    assert record["partial_source_groups"] == 0
    assert pool.sample_id.is_unique


def test_pool_rejects_missing_metadata_and_unachievable_family_budget():
    frame = sample_frame()
    with pytest.raises(ValueError, match="cover"):
        stratified_selection_pool(frame, 2, 17)
    frame.loc[3, "group_hash"] = np.nan
    with pytest.raises(ValueError, match="nonmissing"):
        stratified_selection_pool(frame, 100, 17)
