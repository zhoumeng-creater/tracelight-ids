"""Synthetic arrays only: exact tails, identity protection, and fixed-budget selection."""
from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from ids.normal_subgroup_training.selection import (
    QUANTILES, VALIDATION_STEPS, Selector, checkpoint_guards, metrics_to_json,
    normal_tail_metrics, selection_metrics,
)


def fixture_metrics(normal_scores=None, *, positive_scores=(20., 21.), f_scores=(0, 20, 1, 21)):
    f = pd.DataFrame({"source_id": ["F_A", "F_A", "F_B", "F_B"], "row_id": [1, 2, 1, 2],
                      "profile": ["A", "A", "B", "B"], "ip_protocol": [6, 6, 17, 17],
                      "label": [0, 1, 0, 1]})
    v = pd.DataFrame({"source_id": ["V_A"] * 11 + ["V_B"] * 11,
                      "row_id": list(range(11)) * 2, "profile": ["A"] * 11 + ["B"] * 11,
                      "ip_protocol": [6] * 11 + [17] * 11,
                      "label": ([0] * 10 + [1]) * 2})
    normal = np.arange(20, dtype=float) if normal_scores is None else np.asarray(normal_scores)
    scores = np.concatenate((normal[:10], [positive_scores[0]], normal[10:], [positive_scores[1]]))
    groups = ["A_0"] * 10 + ["B_0"] * 10
    return selection_metrics(f_scores, f, scores, v, groups, ["A_0", "B_0"])


def balanced_metrics(**kwargs):
    # Each group's highest observation receives half of the top 10% mass.
    return fixture_metrics(np.r_[np.arange(0, 20, 2), np.arange(1, 20, 2)], **kwargs)


def test_exact_fractional_boundary_ties_conserve_mass_and_hand_computed_excess():
    m = normal_tail_metrics([0, 1, 2, 2], ["a", "a", "b", "b"], ["a", "b"])
    assert m["groups"]["b"]["T"]["0.90"] == Fraction(1, 5)
    assert m["H_q"] == {"0.90": Fraction(1, 10), "0.95": Fraction(1, 20),
                        "0.99": Fraction(1, 100)}
    assert m["max_tie_block"] == 2 and m["tied_pairs"] == 1
    for qname, q in QUANTILES:
        mass = sum(g["normal"] * g["T"][qname] for g in m["groups"].values())
        assert mass == 4 * (1 - q)


def test_tail_is_permutation_and_strict_monotone_invariant():
    scores = np.array([-3., -1., -1., 2., 8., 10.])
    groups = np.array(["a", "b", "a", "b", "a", "b"])
    base = normal_tail_metrics(scores, groups, ["a", "b"])
    order = [5, 3, 1, 0, 4, 2]
    assert normal_tail_metrics(3 * scores[order] - 17, groups[order], ["a", "b"]) == base


def test_sparse_group_not_averaged_away_and_missing_or_unknown_group_rejected():
    m = normal_tail_metrics(np.arange(100), ["large"] * 99 + ["small"], ["large", "small"])
    assert m["H_q"]["0.99"] == Fraction(99, 100)
    with pytest.raises(ValueError, match="support"):
        normal_tail_metrics([0, 1], ["a", "a"], ["a", "b"])
    with pytest.raises(ValueError, match="unknown"):
        normal_tail_metrics([0, 1], ["a", "UNSEEN"], ["a", "b"])


def test_all_constant_normal_can_have_zero_H_but_tie_guards_reject():
    base, collapsed = fixture_metrics(), fixture_metrics(np.zeros(20))
    assert collapsed["tail"]["H"] == 0
    guards = checkpoint_guards(collapsed, base)
    assert not guards["checks"]["normal_max_tie_block_not_increased"]
    assert not guards["checks"]["normal_tied_pairs_not_increased"]
    assert not guards["passed"]


def test_raw_F_score_decrease_reported_but_does_not_reject_preserved_order():
    base = fixture_metrics()
    current = balanced_metrics(f_scores=(0, 19.999999999999996, 1, 22))
    result = checkpoint_guards(current, base)
    assert result["F_lower_positive_ids"] == [("F_A", 2)]
    assert result["F_lower_positive_count"] == 1
    assert result["passed"]


def test_single_F_positive_rank_decrease_rejected_including_exact_tie():
    base = fixture_metrics()
    current = balanced_metrics(f_scores=(0, 1, 1, 22))
    result = checkpoint_guards(current, base)
    assert result["F_worse_positive_ids"] == [("F_A", 2)]
    assert not result["checks"]["F_each_positive_inversions_not_increased"]


def test_V_positive_equal_normal_counts_against_positive_and_panel_guard():
    base = fixture_metrics()
    tied = fixture_metrics(positive_scores=(19., 21.))
    assert tied["V_positive_inversions"] == (1, 0)
    assert checkpoint_guards(tied, base)["V_worse_positive_ids"] == [("V_A", 10)]
    own_panel_tie = fixture_metrics(positive_scores=(9., 21.))
    assert not checkpoint_guards(own_panel_tie, base)["checks"]["V_panel_strict_separation"]


def test_positive_identity_protection_is_not_replaced_by_total_inversion_count():
    base, current = fixture_metrics(), balanced_metrics()
    base["V_positive_inversions"], current["V_positive_inversions"] = (2, 0), (0, 1)
    result = checkpoint_guards(current, base)
    assert sum(current["V_positive_inversions"]) < sum(base["V_positive_inversions"])
    assert result["V_worse_positive_ids"] == [("V_B", 10)]


def test_pooled_and_panel_pauc_are_checked_explicitly():
    base = fixture_metrics()
    for key in ("pooled", "A_TCP", "B_UDP"):
        current = balanced_metrics()
        if key == "pooled":
            current[key]["pauc"] = .99
        else:
            current["panels"][key]["pauc"] = .99
        assert not checkpoint_guards(current, base)["passed"]


def test_lower_overall_H_cannot_hide_worse_one_percent_tail():
    base, current = fixture_metrics(), balanced_metrics()
    current["tail"]["H_q"]["0.99"] = base["tail"]["H_q"]["0.99"] + Fraction(1, 10000)
    assert current["tail"]["H"] < base["tail"]["H"]
    assert not checkpoint_guards(current, base)["checks"]["each_H_q_not_increased"]


def test_pooled_improvement_cannot_hide_source_B_one_percent_tail_regression():
    f = pd.DataFrame({"source_id": ["F_A", "F_A", "F_B", "F_B"], "row_id": [1, 2, 1, 2],
                      "profile": ["A", "A", "B", "B"], "ip_protocol": [6, 6, 17, 17],
                      "label": [0, 1, 0, 1]})
    profiles = np.array(["A"] * 200 + ["B"] * 200 + ["A", "B"])
    v = pd.DataFrame({"source_id": ["V_" + p for p in profiles], "row_id": np.arange(402),
                      "profile": profiles, "ip_protocol": np.where(profiles == "A", 6, 17),
                      "label": [0] * 400 + [1, 1]})
    keys = ["A_0", "A_1", "B_0", "B_1"]
    groups = np.repeat(keys, 100)
    original = np.r_[np.arange(1000, 1100), np.arange(100),
                     np.arange(200, 400, 2), np.arange(201, 400, 2), 2000, 2001]
    changed_parts = [np.arange(offset, 400, 4) for offset in range(4)]
    # Exchange two B assignments within the same pooled 5%/10% bands.
    # Global top 1% becomes A0,A1,B1,B1; B's top 1% becomes B1,B1.
    changed_parts[2][-1], changed_parts[3][-2] = changed_parts[3][-2], changed_parts[2][-1]
    changed = np.r_[np.concatenate(changed_parts), 2000, 2001]
    base = selection_metrics([0, 20, 1, 21], f, original, v, groups, keys)
    current = selection_metrics([0, 20, 1, 21], f, changed, v, groups, keys)
    assert base["tail"]["normal"] == 400
    assert current["source_tails"]["B"]["normal"] == 200
    assert current["source_tails"]["B"]["H_q"]["0.99"] == Fraction(1, 100)
    assert base["source_tails"]["B"]["H_q"]["0.99"] == 0
    assert current["tail"]["H"] < base["tail"]["H"]
    assert all(current["tail"]["H_q"][q] < base["tail"]["H_q"][q] for q, _ in QUANTILES)
    guards = checkpoint_guards(current, base)
    assert all(value for key, value in guards["checks"].items() if key != "each_source_H_q_not_increased")
    assert not guards["source_tail_checks"]["B"]["0.99"]
    assert not guards["passed"]
    assert not Selector(base).update(current, 139)["eligible"]


def test_frozen_identity_metadata_and_group_mapping_cannot_change():
    base = fixture_metrics()
    for key in ("F_contract", "V_contract", "F_positive_ids", "V_positive_ids", "V_normal_group_ids"):
        current = balanced_metrics()
        current[key] = ()
        with pytest.raises(ValueError, match="Frozen selection"):
            checkpoint_guards(current, base)


def test_selector_keeps_earliest_exact_H_tie_and_never_stops_early():
    base, current = fixture_metrics(), balanced_metrics()
    selector = Selector(base)
    for event, step in enumerate(VALIDATION_STEPS, 1):
        result = selector.update(current, step)
        assert result["improved"] == (event == 1)
        assert result["selected_step"] == 139
        assert result["stop"] == (event == 16)
        assert not result["performance_early_stop"]
    assert selector.finish()["candidate_improved"]
    assert selector.finish()["optimizer_steps"] == 2224
    with pytest.raises(ValueError, match="fixed step"):
        selector.update(current, 2363)


def test_no_strict_improvement_falls_back_and_is_never_candidate_success():
    base = fixture_metrics()
    selector = Selector(base)
    with pytest.raises(ValueError, match="sixteen"):
        selector.finish()
    for step in VALIDATION_STEPS:
        assert not selector.update(base, step)["candidate_improved"]
    assert selector.finish()["fallback"]
    assert selector.finish()["selected_step"] == 0


def test_selector_rejects_skipped_or_repeated_event_and_snapshots_baseline():
    base = fixture_metrics()
    selector = Selector(base)
    base["tail"]["H"] = Fraction(0)
    assert selector.best_H == Fraction(1, 10)
    with pytest.raises(ValueError, match="fixed step"):
        selector.update(balanced_metrics(), 278)
    selector.update(balanced_metrics(), 139)
    with pytest.raises(ValueError, match="fixed step"):
        selector.update(balanced_metrics(), 139)


def test_exact_fraction_receipt_is_json_serializable_without_rounding_selection():
    value = {"H": Fraction(1, 18645), "identity": ("A", 1)}
    saved = json.loads(json.dumps(metrics_to_json(value), allow_nan=False))
    assert Fraction(saved["H"]["numerator"], saved["H"]["denominator"]) == value["H"]


def test_absolute_positive_guard_is_not_necessary_for_recalibrated_TP_preservation():
    # Positive strictly increasing transform g(s)=s-7 preserves every rank and decision.
    calibration = np.arange(100, dtype=float)
    old_threshold = calibration[98]
    evaluation = np.array([10., 99., 120.])
    new_threshold = (calibration - 7)[98]
    np.testing.assert_array_equal(evaluation > old_threshold, evaluation - 7 > new_threshold)
    base = fixture_metrics()
    changed = fixture_metrics(f_scores=(-7, 13, -6, 14))
    # This is the old v1 condition, kept only as a counterexample rather than a mode.
    assert not all(s >= old for s, old in zip(changed["F_positive_scores"], base["F_positive_scores"]))
    assert checkpoint_guards(changed, base)["checks"]["F_each_positive_inversions_not_increased"]


def test_F_rank_preservation_alone_does_not_guarantee_C_threshold_detection():
    normal, old_positive, new_positive, calibrated_threshold = 0., 1., .1, .5
    assert int(normal >= old_positive) == int(normal >= new_positive) == 0
    assert old_positive > calibrated_threshold
    assert not new_positive > calibrated_threshold


def test_soft_positive_hinge_has_zero_initial_gradient_and_allows_positive_decrease():
    # Synthetic shared parameter: s_normal=theta, s_positive=theta+2, theta0=0.
    # Four balanced groups and equal members reduce to the average binary BCE below.
    def objective(theta):
        bce = .5 * (np.logaddexp(0, theta) + np.logaddexp(0, -(theta + 2)))
        return bce + (theta + 1) ** 2 + max(-theta, 0) ** 2

    positive_penalty_gradient_at_zero = 0.
    bce_gradient = .5 * (.5 + 1 / (1 + np.exp(-2)) - 1)
    gradient = bce_gradient + 2 + positive_penalty_gradient_at_zero
    finite_difference = (objective(1e-6) - objective(-1e-6)) / 2e-6
    assert finite_difference == pytest.approx(gradient, abs=1e-6)
    theta_after = -.01 * gradient
    assert objective(theta_after) < objective(0)
    assert theta_after + 2 < 2
    assert max(-theta_after, 0) ** 2 > 0
    assert theta_after + 2 > theta_after  # Strict class separation is still retained.


def test_soft_positive_anchor_alone_does_not_guarantee_even_F_rank_guard():
    # A synthetic witness, not the margins or Jacobian of the real R30 model.
    # Normal scores theta-.5 and theta; both normal targets are baseline minus one.
    # Positive score 2*theta+.0001. The baseline is strictly separated.
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    gap = .0001
    bce_gradient = .25 * sigmoid(-.5) + .25 * sigmoid(0) + sigmoid(gap) - 1
    gradient = bce_gradient + 2  # Positive squared-hinge gradient is exactly zero at step0.
    clipped = gradient / max(1., abs(gradient))
    # First bias-corrected AdamW step, theta0=0: weight decay contributes zero.
    theta_after = -.0002 * clipped / (abs(clipped) + 1e-8)
    assert 0 < gap
    assert 2 * theta_after + gap < theta_after
    assert (gap - (2 * theta_after + gap)) ** 2 > 0


@pytest.mark.parametrize("bad", [[0, float("nan")], [0, float("inf")], [[0], [1]]])
def test_invalid_normal_scores_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        normal_tail_metrics(bad, ["a", "b"], ["a", "b"])
