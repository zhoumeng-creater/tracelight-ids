"""Synthetic strict-logit calibration and paired decision semantics."""

import copy
import json

import numpy as np
import pytest

from ids.context_cache.calibration import apply_threshold, compare_decisions, evaluate, register_threshold


BINDING = {"model_sha256": "a" * 64, "preprocessing_sha256": "b" * 64, "calibration_sha256": "c" * 64}


def register(scores, alpha=.01, **binding):
    return register_threshold(scores, alpha, **(BINDING | binding))


@pytest.mark.parametrize("alpha,n,rank", [(.07, 99, 93), (.01, 99, 99),
                                         (.01, 199, 198), (.01, 999, 990)])
def test_rank_uses_exact_decimal_alpha_at_integer_boundaries(alpha, n, rank):
    registration = register(np.arange(n, dtype=float) - 1000, alpha)
    assert registration["rank"] == rank
    assert registration["threshold"] == rank - 1 - 1000
    assert registration["allowed_calibration_fp"] == n - rank
    assert registration["calibration_fp"] == n - rank
    assert registration["boundary_ties"] == 1


def test_ties_use_strict_comparison_without_nextafter_or_probability_clipping():
    normal = np.array([-100.] * 195 + [4.] * 4)
    registration = register(normal)
    assert registration["rank"] == 198
    assert registration["allowed_calibration_fp"] == 1
    assert registration["calibration_fp"] == 0
    assert registration["boundary_ties"] == 4
    assert registration["threshold"] == 4
    scores = [-1e300, -1., 4., np.nextafter(4., np.inf), 1e300]
    assert apply_threshold(scores, registration).tolist() == [False, False, False, True, True]
    assert apply_threshold(scores, registration).dtype == np.bool_


@pytest.mark.parametrize("n", [0, 20, 98])
def test_insufficient_normals_have_json_safe_positive_infinity_endpoint(n):
    registration = register(np.zeros(n))
    assert registration["threshold"] is None
    assert registration["threshold_kind"] == "positive_infinity"
    assert registration["reject_all"] is True
    assert registration["allowed_calibration_fp"] == registration["calibration_fp"] == 0
    assert registration["boundary_ties"] == 0
    loaded = json.loads(json.dumps(registration, allow_nan=False))
    assert loaded == registration
    assert not apply_threshold([-np.finfo(float).max, 0, np.finfo(float).max], loaded).any()
    assert loaded["calibration_fpr"] == (None if n == 0 else 0.)


def test_finite_registration_json_round_trip_and_binding():
    registration = register(np.linspace(-40, -3, 99), model_sha256="D" * 64)
    loaded = json.loads(json.dumps(registration, allow_nan=False))
    assert loaded["binding"] == BINDING | {"model_sha256": "d" * 64}
    assert loaded["scope"] == "development_calibration"
    assert loaded["threshold"] == -3
    assert apply_threshold([-4, -3, -2], loaded).tolist() == [False, False, True]
    shifted = register(np.linspace(-40, -3, 99) + 1)
    assert registration["normal_scores_sha256"] != shifted["normal_scores_sha256"]


def test_each_candidate_must_use_its_own_normal_calibration_scores():
    full_normal = np.linspace(-2, 0, 99)
    fullreg = register(full_normal)
    qcocreg = register(full_normal + 10, model_sha256="d" * 64)
    full_scores, qcoc_scores = np.array([-1., 1.]), np.array([9., 11.])
    assert fullreg["threshold"] == 0 and qcocreg["threshold"] == 10
    assert apply_threshold(qcoc_scores, fullreg).tolist() == [True, True]  # Incorrect threshold reuse.
    assert apply_threshold(qcoc_scores, qcocreg).tolist() == [False, True]
    comparison = compare_decisions([0, 1], full_scores, fullreg, qcoc_scores, qcocreg, ids=["normal", "attack"])
    assert comparison["full"] == comparison["qcoc"]
    assert comparison["full"]["tp"] == 1 and comparison["full"]["fp"] == 0
    assert all(count == 0 for count in comparison["changes"].values())


def test_paired_four_changes_and_stable_quadrants_have_consistent_counts_and_ids():
    fullreg = register(np.zeros(99))
    qcocreg = register(np.zeros(99), model_sha256="d" * 64)
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    full = [-1, 1, -1, 1, -1, 1, -1, 1]
    qcoc = [-1, 1, 1, -1, -1, 1, 1, -1]
    ids = [f"row-{i}" for i in range(8)]
    result = compare_decisions(labels, full, fullreg, qcoc, qcocreg, ids)
    assert result["changes"] == {"repaired_fn": 1, "new_fn": 1, "repaired_fp": 1, "new_fp": 1}
    assert result["prediction_pairs"] == {kind: {"00": 1, "01": 1, "10": 1, "11": 1}
                                           for kind in ("normal", "attack")}
    assert result["net_tp"] == result["net_fp"] == 0
    assert result["full"] == result["qcoc"]
    assert result["full"]["tpr"] == result["full"]["fpr"] == .5
    assert [row["row_id"] for row in result["rows"]] == ids
    assert [row["change"] for row in result["rows"]] == [
        "unchanged_tn", "unchanged_fp", "new_fp", "repaired_fp",
        "unchanged_fn", "unchanged_tp", "repaired_fn", "new_fn",
    ]
    assert result["alignment"] == "caller_aligned_ids"
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    reordered = compare_decisions(labels, full, fullreg, qcoc, qcocreg, ids[::-1])
    assert reordered["ordered_row_ids_sha256"] != result["ordered_row_ids_sha256"]
    positional = compare_decisions(labels, full, fullreg, qcoc, qcocreg)
    assert positional["alignment"] == "position_only"
    with pytest.raises(ValueError, match="unique"):
        compare_decisions(labels, full, fullreg, qcoc, qcocreg, ["same"] * 8)


def test_family_reporting_does_not_change_threshold_or_decisions():
    registration = register(np.zeros(99))
    before = copy.deepcopy(registration)
    labels, scores = [0, 0, 1, 1], [-1, 2, -3, 4]
    first = evaluate(labels, scores, registration, ["normal", "normal", "family-a", "family-b"])
    second = evaluate(labels, scores, registration, ["group-x"] * 4)
    assert {key: value for key, value in first.items() if key != "families"} == {
        key: value for key, value in second.items() if key != "families"}
    assert first["families"]["family-a"]["fn"] == 1
    assert first["families"]["family-b"]["tp"] == 1
    assert first["families"]["normal"]["tpr"] is None
    assert registration == before


@pytest.mark.parametrize("alpha", [0, 1, -0.01, 2, float("nan"), float("inf"), True])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        register([0.], alpha)


@pytest.mark.parametrize("scores", [[0, float("nan")], [float("inf")], [-float("inf")],
                                    [[0., 1.]], ["0.2"], [True]])
def test_nonfinite_or_non_numeric_or_non_vector_scores_rejected(scores):
    with pytest.raises(ValueError):
        register(scores)
    with pytest.raises(ValueError):
        apply_threshold(scores, register(np.zeros(99)))


@pytest.mark.parametrize("labels", [[0, 2], [0, np.nan], [0, "1"], [1], [[0, 1]]])
def test_invalid_labels_rejected(labels):
    registration = register(np.zeros(99))
    with pytest.raises(ValueError, match="labels"):
        evaluate(labels, [-1., 1.], registration)


@pytest.mark.parametrize("changed", [{"calibration_sha256": "d" * 64},
                                    {"preprocessing_sha256": "e" * 64}])
def test_paired_comparison_checks_cohort_and_preprocessing_bindings(changed):
    fullreg, qcocreg = register(np.zeros(99)), register(np.zeros(99), **changed)
    with pytest.raises(ValueError, match="same"):
        compare_decisions([0], [0.], fullreg, [0.], qcocreg)
    with pytest.raises(ValueError, match="same alpha"):
        compare_decisions([0], [0.], fullreg, [0.], register(np.zeros(99), alpha=.02))


def test_malformed_registration_does_not_change_endpoint_or_comparison_silently():
    finite = register(np.zeros(99))
    wrong_rule = copy.deepcopy(finite)
    wrong_rule["score_contract"]["comparison"] = "greater_than_or_equal"
    with pytest.raises(ValueError, match="strict-logit"):
        apply_threshold([0.], wrong_rule)
    insufficient = register(np.zeros(20))
    insufficient["threshold"] = np.finfo(float).max
    with pytest.raises(ValueError, match="positive-infinity"):
        apply_threshold([0.], insufficient)
    with pytest.raises(ValueError, match="SHA-256"):
        register(np.zeros(99), model_sha256="not-a-model-hash")
