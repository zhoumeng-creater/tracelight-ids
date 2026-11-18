"""Synthetic contracts for F-only groups and immutable transport targets."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from ids.normal_subgroup_training.groups import FIELDS, fit_groups, leaf_nodes, map_groups
from ids.normal_subgroup_training.targets import build_targets, input_equivalence, inverse_ecdf, midranks


def group_frame(n=1024):
    data = {"label": np.zeros(2 * n, dtype=int), "profile": np.repeat(["A", "B"], n)}
    data.update({name: np.tile(np.arange(n), 2).astype(float) for name in FIELDS})
    return data


def target_case(scores=(-4., -3., -2., -1., 10.)):
    n = len(scores)
    return {"reference_scores": np.array(scores), "labels": np.array([0] * (n - 1) + [1]),
            "group_ids": np.array(["A"] * 2 + ["B"] * (n - 3) + [""]),
            "numeric": np.repeat(np.arange(n, dtype=np.float32)[:, None], 5, axis=1),
            "categorical": np.column_stack([np.full(n, 5), np.zeros((n, 5), dtype=int)])}


def test_groups_are_f_only_and_ignore_score_or_positive_values():
    frame = group_frame()
    spec = fit_groups(frame)
    frame["score"] = np.arange(len(frame["label"])) * 100
    assert spec == fit_groups(frame)
    with pytest.raises(ValueError, match="only from role F"):
        fit_groups(frame, role="V")
    other = {k: np.concatenate([v, v[:1]]) for k, v in frame.items()}
    other["label"][-1] = 1
    other["orig_pkts"][-1] = 1e20
    assert spec == fit_groups(other)


def test_groups_are_order_invariant_and_roundtrip_json():
    frame = group_frame()
    spec = fit_groups(frame)
    order = np.random.default_rng(71).permutation(len(frame["label"]))
    shuffled = {k: v[order] for k, v in frame.items()}
    assert spec == fit_groups(shuffled)
    np.testing.assert_array_equal(map_groups(shuffled, spec), map_groups(frame, spec)[order])
    assert spec == json.loads(json.dumps(spec))


def test_balanced_tie_uses_lower_observed_cut_and_never_splits_equal_values():
    frame = group_frame(768)
    frame["orig_pkts"] = np.tile(np.repeat([10., 20., 30.], 256), 2)
    spec = fit_groups(frame)
    assert spec["trees"]["A"]["cut_le"] == 10
    assert spec["trees"]["A"]["left"]["n"] == 256
    assert all(n["n"] >= 256 for n in leaf_nodes(spec))


def test_field_priority_falls_through_constant_feature():
    frame = group_frame()
    frame["orig_pkts"][:] = 5
    spec = fit_groups(frame)
    assert spec["trees"]["A"]["feature"] == "resp_pkts"


def test_depth_and_group_bound_and_unsplittable_leaf():
    spec = fit_groups(group_frame(8192))
    assert len(leaf_nodes(spec)) == 32
    assert max(n["depth"] for n in leaf_nodes(spec)) == 4
    frame = group_frame()
    for field in FIELDS:
        frame[field][:] = 1
    spec = fit_groups(frame)
    assert [n["key"] for n in leaf_nodes(spec)] == ["A_", "B_"]


def test_small_root_unknown_profile_and_bad_counts_fail():
    with pytest.raises(ValueError, match="fewer than 256"):
        fit_groups(group_frame(255))
    frame = group_frame()
    frame["profile"][0] = "X"
    with pytest.raises(ValueError, match="known A/B"):
        fit_groups(frame)
    for bad in (-2, np.inf, -np.inf):
        frame = group_frame()
        frame["orig_pkts"][0] = bad
        with pytest.raises(ValueError, match="Counts require"):
            fit_groups(frame)


def test_missing_canonicalization_and_unseen_pattern_rejection():
    frame = group_frame()
    frame["orig_pkts"][:256] = np.nan
    spec = fit_groups(frame)
    same = copy.deepcopy(frame)
    same["orig_pkts"][:256] = -1
    assert fit_groups(same) == spec
    np.testing.assert_array_equal(map_groups(frame, spec), map_groups(same, spec))
    same["resp_pkts"][0] = -1
    with pytest.raises(ValueError, match="UNSUPPORTED_MISSING"):
        map_groups(same, spec)


def test_frozen_mapping_edges_positive_sentinel_and_missing_group():
    frame = group_frame()
    spec = fit_groups(frame)
    subset = {k: v[:1].copy() for k, v in frame.items()}
    subset["orig_pkts"][0] = 1e20
    mapped = map_groups(subset, spec, require_all=False)
    assert mapped[0].startswith("A_1")
    with pytest.raises(ValueError, match="without V normal support"):
        map_groups(subset, spec)
    subset["label"][0] = 1
    assert map_groups(subset, spec, require_all=False).tolist() == [""]
    with pytest.raises(ValueError, match="never deployment"):
        map_groups(frame, spec, role="E_D")


def test_midrank_and_inverse_ecdf_hand_calculation():
    np.testing.assert_array_equal(midranks([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
    np.testing.assert_array_equal(inverse_ecdf([4, 1, 3, 2], [0, .25, .5, .75, 1]), [1, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="probabilities"):
        inverse_ecdf([1, 2], [-.1])


def test_targets_match_hand_calculation_identity_and_scale():
    case = target_case()
    target = build_targets(**case)
    np.testing.assert_array_equal(target["transport_target"], [-4, -3, -4, -2, 10])
    np.testing.assert_array_equal(target["identity_target"], case["reference_scores"])
    assert target["scale"] == 2
    assert target["group_counts"] == {"A": 2, "B": 2}
    for key in ("transport_target", "identity_target", "reference_scores", "equivalence_ids"):
        assert not target[key].flags.writeable


def test_integer_group_ecdf_ceiling_does_not_advance_an_exact_rank_boundary():
    # rank4 in a 25-row group has u=7/50. FP64 50*u is slightly ABOVE 7.
    # The seventh pooled order statistic (value6), not eighth (value7), is required.
    assert 50 * ((4 - .5) / 25) > 7
    case = target_case(tuple(range(50)) + (100,))
    case["group_ids"] = np.array(["A"] * 25 + ["B"] * 25 + [""])
    target = build_targets(**case)
    assert target["provisional_target"][28] == 6
    assert target["transport_target"][28] == 6


def test_equivalence_min_reconciles_different_group_targets_without_changing_identity():
    case = target_case((-4, -3, -3, -1, 10))
    case["numeric"][2] = case["numeric"][1]
    target = build_targets(**case)
    np.testing.assert_array_equal(target["provisional_target"], [-4, -3, -4, -3, 10])
    np.testing.assert_array_equal(target["transport_target"], [-4, -4, -4, -3, 10])
    assert target["equivalence"]["provisional_conflict_normal_classes"] == 1
    assert target["equivalence"]["rows_lowered_by_equivalence_min"] == 1
    np.testing.assert_array_equal(target["identity_target"], case["reference_scores"])


def test_same_actual_fp32_input_not_raw_precision_and_signed_zero():
    case = target_case()
    case["numeric"][1] = -0.
    equiv = input_equivalence(case["numeric"], case["categorical"])
    assert equiv[0] == equiv[1]
    with pytest.raises(ValueError, match="actual FP32"):
        input_equivalence(case["numeric"].astype(float), case["categorical"])
    case["categorical"][1, 1] = 1
    equiv = input_equivalence(case["numeric"], case["categorical"])
    assert equiv[0] != equiv[1]


def test_opposite_label_equivalence_collision_rejected():
    case = target_case()
    case["numeric"][-1] = case["numeric"][0]
    with pytest.raises(ValueError, match="Opposite-label"):
        build_targets(**case)


def test_gpu_reference_difference_audited_not_silently_changed():
    case = target_case((-4, -3, -3.000001, -1, 10))
    case["numeric"][2] = case["numeric"][1]
    target = build_targets(**case)
    assert target["equivalence"]["reference_inconsistent_classes"] == 1
    assert target["equivalence"]["max_reference_difference"] == pytest.approx(1e-6)
    np.testing.assert_array_equal(target["identity_target"], case["reference_scores"])
    assert target["transport_target"][1] == target["transport_target"][2]


def test_target_order_invariance_and_f_only_role():
    case = target_case()
    target = build_targets(**case)
    order = np.array([3, 0, 4, 1, 2])
    other = build_targets(**{key: value[order] for key, value in case.items()})
    np.testing.assert_array_equal(other["transport_target"], target["transport_target"][order])
    with pytest.raises(ValueError, match="only from role F"):
        build_targets(**case, role="V")


def test_constant_target_invalid_score_and_input_policy_rejected():
    with pytest.raises(ValueError, match="constant"):
        build_targets(**target_case((1, 1, 1, 1, 2)))
    case = target_case()
    case["reference_scores"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_targets(**case)
    case = target_case()
    case["categorical"][0, 0] = 0
    with pytest.raises(ValueError, match="UNKNOWN=5"):
        build_targets(**case)


def test_normal_targets_never_rise_positive_anchors_unchanged_and_ties_initially_equal():
    case = target_case((-4, -4, -2, -1, 10))
    target = build_targets(**case)
    assert target["provisional_target"][0] == target["provisional_target"][1]
    assert np.all(target["transport_target"] <= case["reference_scores"])
    assert target["transport_target"][-1] == 10
