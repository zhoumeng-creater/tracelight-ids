import numpy as np
import pytest

from ids.route_batch.metrics import decisions, evaluate, paired, threshold


def test_strict_rank_ties_and_insufficient_normal():
    r = threshold(np.arange(5000), .01)
    assert r["rank"] == 4951 and r["calibration_fp"] == 49
    assert decisions([4950., 4951.], r).tolist() == [False, True]
    assert threshold([1, 2], .01)["threshold"] is None
    assert not decisions([100.], threshold([1, 2], .01))[0]


def test_metrics_keep_new_errors_and_families():
    y = np.array([0, 0, 1, 1])
    r = {"threshold": .5}
    result = evaluate(y, [.1, .8, .9, .2], r, ["Benign", "Benign", "Scanning", "XSS"])
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["families"]["Scanning"]["tp"] == 1
    assert paired(y, [0, 0, 1, 0], [0, 1, 0, 1]) == {
        "recovered_fn": 1, "new_fn": 1, "repaired_fp": 0, "new_fp": 1}


def test_bad_scores_fail():
    with pytest.raises(ValueError):
        threshold([np.nan])


def test_pauc_does_not_add_area_before_vertical_jump_at_cutoff():
    y = np.r_[np.zeros(100, dtype=int), 1]
    values = np.r_[2., np.zeros(99), 1.]
    result = evaluate(y, values, {"threshold": 2.})
    assert result["normalized_pauc_0_01"] == 0.
