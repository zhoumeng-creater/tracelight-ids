"""Frozen and local unlabeled rank policies keep strict ties and the total allocation."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer import policy
from ids.route_batch.metrics import threshold as original_threshold


def test_original_rank_and_ties_are_preserved():
    x = np.r_[np.zeros(199), 10.0]
    assert policy.threshold(x) == original_threshold(x)
    assert np.array_equal(policy.decisions([0, 10, 11], policy.threshold(x)), [False, True, True])
    assert policy.threshold([1, 2, 3])["threshold"] is None
    assert not policy.decisions([10], policy.threshold([1, 2, 3])).any()


def test_thirds_uses_exact_fraction_and_union_allocation():
    # (n+1)*(1-1/300) lies exactly on an integer rank. A binary approximation could move it.
    x = pd.DataFrame({"a": np.arange(599), "b": np.arange(599)[::-1], "c": np.roll(np.arange(599), 200)})
    rules = policy.register(x, ("a", "b", "c"))
    assert all(v["rank"] == 598 for v in rules.values())
    assert all(v == policy.threshold(x[k], Fraction(1, 300)) for k, v in rules.items())
    assert policy.apply(x, rules).sum() == 3 <= len(x) // 100
    z = pd.DataFrame({"a": np.r_[np.zeros(598), 1.0], "b": np.r_[np.zeros(598), 1.0]})
    r = policy.register(z, ("a", "b"))
    assert policy.apply(z, r).sum() == 1
    with pytest.raises(ValueError):
        policy.register(x, ("a", "a"))


def test_local_prefix_fallback_switch_ties_and_future_labels_do_not_select_cut():
    old = policy.register(pd.DataFrame({"a": np.arange(1000)}), ("a",))
    f = pd.DataFrame(
        {
            "flow_end_ms": np.r_[np.arange(1000), [1000, 1000, 1001]],
            "a": np.r_[np.arange(1000) * 2, [-9999, 9999, 10000]],
        }
    )
    partial = policy.prefix_rules(f.iloc[:300], old, ("a",), cutoff_ms=1000)
    assert partial["status"] == "old_calibration_fallback" and partial["rules"] == old
    end = policy.prefix_rules(f.iloc[:1001], old, ("a",), cutoff_ms=1000)
    future = policy.prefix_rules(f.assign(family="Malicious"), old, ("a",), cutoff_ms=1000)
    reference = policy.register(f.iloc[:1000], ("a",))
    reference["a"]["scope"] = "local_unlabeled_alert_rate"
    reference["a"]["n_reference_rows"] = 1000
    assert end["rules"] == future["rules"] == reference
    assert end["local_rows"] == end["fallback_rows"] == 1000
    assert end["rules"]["a"]["threshold"] == 1980
    empty = policy.prefix_rules(f.iloc[:0], old, ("a",), cutoff_ms=1000)
    assert empty["rules"] == old
    with pytest.raises(ValueError):
        policy.prefix_rules(f.iloc[::-1], old, ("a",), cutoff_ms=1000)
