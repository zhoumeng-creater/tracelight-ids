"""Label semantics and whole-input randomization, using only synthetic data."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ids.context_cache.label_inputs import (
    ARMS, attack_union_log_odds, build_support_label_views, canonical_support_group_ids,
    grouped_sham_support_labels, label_sha256, semantic_support_labels,
)


@pytest.fixture
def support():
    patterns = np.array([0] * 4 + [1] * 3 + [2] * 2 + list(range(3, 17)))
    x = np.zeros((len(patterns), 48), dtype=np.float32)
    x[:, 0] = patterns
    y = (patterns != 0).astype(np.int64)
    families = np.where(y == 0, "Benign", np.where(patterns <= 2, "Scanning", "DDoS"))
    return x, y, families


def test_three_views_preserve_binary_task_and_same_class_counts(support):
    x, binary, families = support
    before = x.tobytes(), binary.tobytes(), families.copy()
    views, audit = build_support_label_views(x, binary, families, small_class_rows=5)
    assert tuple(views) == ARMS
    assert audit["class_counts"] == {"baseline": [4, 19], "semantic_family": [4, 5, 14],
                                     "grouped_sham": [4, 5, 14]}
    for values in views.values():
        assert values.dtype == np.int64
        assert np.array_equal(values > 0, binary > 0)
    assert np.array_equal(views["semantic_family"] == 1, families == "Scanning")
    assert x.tobytes() == before[0] and binary.tobytes() == before[1]
    assert np.array_equal(families, before[2])
    assert audit["row_input_group_sha256"] == canonical_support_group_ids(x).tolist()
    assert audit["label_sha256"] == {arm: hashlib.sha256(values.astype("<i8").tobytes()).hexdigest()
                                      for arm, values in views.items()}
    assert np.array(audit["semantic_by_sham_class_counts"]).sum(axis=1).tolist() == [4, 5, 14]
    assert np.array(audit["semantic_by_sham_class_counts"]).sum(axis=0).tolist() == [4, 5, 14]
    json.dumps(audit, allow_nan=False)


def test_same_inputs_never_split_even_with_signed_zero(support):
    x, binary, families = support
    x[4, 5] = -0.0
    original_bits = x.tobytes()
    views, audit = build_support_label_views(x, binary, families)
    ids = canonical_support_group_ids(x)
    assert ids[4] == ids[5] == ids[6]
    for group in set(ids[binary == 1]):
        assert len(set(views["grouped_sham"][(ids == group) & (binary == 1)])) == 1
    assert audit["input_label_conflicts"]["grouped_sham"]["attack_subclass_conflict_groups"] == 0
    assert x.tobytes() == original_bits


def test_assignment_deterministic_and_independent_of_support_row_order(support):
    x, binary, families = support
    first, first_audit = build_support_label_views(x, binary, families)
    repeated, repeated_audit = build_support_label_views(x, binary, families)
    assert first_audit == repeated_audit
    order = np.random.default_rng(17).permutation(len(x))
    shuffled, shuffled_audit = build_support_label_views(x[order], binary[order], families[order])
    for arm in ARMS:
        assert np.array_equal(first[arm], repeated[arm])
        assert np.array_equal(shuffled[arm], first[arm][order])
    assert shuffled_audit["sham"] == first_audit["sham"]


def test_existing_binary_collision_and_new_semantic_collision_are_distinct(support):
    x, binary, families = support
    x[0] = x[4]  # Preserve one normal/Scanning conflict.
    x[9] = x[4]  # OtherAttack shares this input; TF reports the extra conflict.
    views, audit = build_support_label_views(x, binary, families)
    conflicts = audit["input_label_conflicts"]
    assert conflicts["baseline"]["all_label_conflict_groups"] == 1
    assert conflicts["semantic_family"]["attack_subclass_conflict_groups"] == 1
    assert conflicts["grouped_sham"]["attack_subclass_conflict_groups"] == 0
    assert conflicts["grouped_sham"]["all_label_conflict_groups"] == 1
    assert views["grouped_sham"][0] == 0


def test_production_sham_matches_frozen_reference(support):
    path = Path(__file__).resolve().parents[1] / "docs/research_alignment_20261108/scanning_next/label_reference.py"
    spec = importlib.util.spec_from_file_location("frozen_label_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    x, binary, families = support
    expected, _ = module.grouped_sham_support_labels(x, binary, 5)
    actual, _ = grouped_sham_support_labels(x, binary, 5)
    assert np.array_equal(actual, expected)
    assert np.array_equal(semantic_support_labels(binary, families), module.semantic_support_labels(binary, families))


def test_unfillable_group_quota_fails_without_alternative_selection():
    x = np.zeros((9, 48), dtype=np.float32)
    x[1:5, 0], x[5:, 0] = 1, 2
    binary = np.array([0] + [1] * 8)
    with pytest.raises(ValueError, match="did not fill quota"):
        grouped_sham_support_labels(x, binary, 3)


@pytest.mark.parametrize("mutation", ["width", "float64", "nonfinite", "quota", "binary", "family", "missing_class"])
def test_invalid_support_contract_rejected(support, mutation):
    x, binary, families = support
    if mutation == "width":
        x = x[:, :47]
    elif mutation == "float64":
        x = x.astype(np.float64)
    elif mutation == "nonfinite":
        x[0, 0] = np.nan
    elif mutation == "binary":
        binary[0] = 2
    elif mutation == "family":
        families[0] = "Scanning"
    elif mutation == "missing_class":
        families[families == "Scanning"] = "DDoS"
    with pytest.raises(ValueError):
        build_support_label_views(x, binary, families, small_class_rows=6 if mutation == "quota" else None)


def test_label_digest_is_ordered_and_rejects_float_labels():
    assert label_sha256(np.array([0, 1, 2])) != label_sha256(np.array([2, 1, 0]))
    with pytest.raises(ValueError):
        label_sha256(np.array([0., 1., 2.]))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_union_score_keeps_binary_bits_and_sums_attack_probability(dtype):
    binary = np.array([[0, 0], [-2, 7], [1000, -1000]], dtype=dtype)
    actual = attack_union_log_odds(binary)
    assert actual.dtype == dtype
    assert actual.tobytes() == (binary[:, 1] - binary[:, 0]).tobytes()
    three = np.array([[3, 2.5, 2.5], [-3, -4, -5]], dtype=dtype)
    scores = attack_union_log_odds(three)
    assert scores.dtype == dtype
    probabilities = np.exp(three.astype(np.float64) - three.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    expected = np.log(probabilities[:, 1:].sum(axis=1)) - np.log(probabilities[:, 0])
    np.testing.assert_allclose(scores, expected, atol=2e-7, rtol=2e-7)
    assert three[0].argmax() == 0 and scores[0] > 0  # Argmax would change the target.
    assert np.array_equal(scores, attack_union_log_odds(three[:, [0, 2, 1]]))


@pytest.mark.parametrize("values", [np.array([1, 2]), np.array([[0., 1., 2., 3.]]), np.array([[0., np.nan]])])
def test_invalid_logits_rejected(values):
    with pytest.raises(ValueError):
        attack_union_log_odds(values)
