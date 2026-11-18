"""Synthetic score records only; no model, fitting or selection process."""
from dataclasses import replace
from fractions import Fraction
import json

import pytest
import torch

from ids.operating_risk_training.evaluation import (
    FrozenRiskManifest,
    evaluate,
    metrics_to_json,
    validate_alignment,
)
from ids.operating_risk_training.operating_point import calibrate


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def fixture(*, normal=(0., 2., 3., 0., 2.), positive=(4., 5.), reference=None):
    q = torch.ones(99, dtype=torch.float64) if reference is None else tensor(reference)
    n, p = tensor(normal), tensor(positive)
    manifest = FrozenRiskManifest(
        reference_ids=[f"q:{i}" for i in range(len(q))], normal_ids=[f"n:{i}" for i in range(len(n))],
        normal_group_ids=["small"] + ["large"] * (len(n)-1), expected_groups=["large", "small"],
        protected_positive_ids=[f"p:{i}" for i in range(len(p))],
    )
    return q, n, p, manifest


def run(q, n, p, manifest):
    return evaluate(calibrate(q, identities=manifest.reference_ids), manifest,
                    reference_scores=q, normal_scores=n, positive_scores=p,
                    reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                    positive_ids=manifest.protected_positive_ids)


def test_group_denominators_exact_risks_and_complete_positive_protection():
    q, n, p, manifest = fixture()
    report = run(q, n, p, manifest)
    assert report["normal"]["groups"] == {
        "large": {"fp_count": 3, "denominator": 4, "risk": Fraction(3, 4)},
        "small": {"fp_count": 0, "denominator": 1, "risk": Fraction(0, 1)},
    }
    assert report["normal"]["risk"] == Fraction(3, 5)
    assert report["protected_positives"]["passed_ids"] == ["p:0", "p:1"]
    assert report["protected_positives"]["lost_ids"] == []
    assert report["protected_positives"]["min_signed_gap"] == 3
    assert report["tp_protection_feasible"] is True
    assert report["operating_point"] == {
        "threshold": 1., "rank": 99, "n_reference": 99, "reference_threshold_ties": 99,
    }


def test_distant_normal_positive_order_swap_does_not_reject_true_tp():
    q, n, p, manifest = fixture(normal=(0., 10., 0., 0., 0.), positive=(9., 11.))
    before = run(q, n, p, manifest)
    n[1], p[0] = 9., 10.
    after = run(q, n, p, manifest)
    assert before["tp_protection_feasible"] is after["tp_protection_feasible"] is True
    assert before["normal"] == after["normal"]
    assert before["protected_positives"]["passed_ids"] == after["protected_positives"]["passed_ids"]


def test_threshold_tied_positive_is_a_lost_identity():
    q, n, p, manifest = fixture(positive=(1., 2.))
    report = run(q, n, p, manifest)
    assert report["tp_protection_feasible"] is False
    assert report["protected_positives"]["lost_ids"] == ["p:0"]
    assert report["protected_positives"]["threshold_tie_count"] == 1
    assert report["protected_positives"]["min_signed_gap"] == 0
    assert report["protected_positives"]["recall"] == Fraction(1, 2)


def test_one_lost_identity_cannot_be_hidden_by_high_average_recall():
    q, n, p, manifest = fixture(positive=[2.] * 999 + [0.])
    report = run(q, n, p, manifest)
    assert report["protected_positives"]["recall"] == Fraction(999, 1000)
    assert report["protected_positives"]["lost_ids"] == ["p:999"]
    assert report["tp_protection_feasible"] is False


def test_normal_risk_or_many_ties_do_not_enter_tp_feasibility():
    q, n, p, manifest = fixture(normal=[100.] * 5, positive=(2., 3.))
    report = run(q, n, p, manifest)
    assert report["normal"]["risk"] == 1
    assert report["operating_point"]["reference_threshold_ties"] == 99
    assert report["tp_protection_feasible"] is True
    assert "H" not in report and "loss" not in report


@pytest.mark.parametrize("transform", [lambda x: x * 5 + 17, lambda x: x**3, torch.exp])
def test_strict_monotone_transform_preserves_all_risks_and_protected_identities(transform):
    q, n, p, manifest = fixture(positive=(1., 2., 0.))
    before = run(q, n, p, manifest)
    after = run(transform(q), transform(n), transform(p), manifest)
    assert before["normal"] == after["normal"]
    for key in ("tp_count", "fn_count", "denominator", "recall", "passed_ids", "lost_ids", "threshold_tie_count"):
        assert before["protected_positives"][key] == after["protected_positives"][key]
    assert before["tp_protection_feasible"] == after["tp_protection_feasible"]


@pytest.mark.parametrize("change", [
    {"normal_ids": ("n:0", "n:0", "n:2", "n:3", "n:4")},
    {"normal_group_ids": ("small", "large")},
    {"normal_group_ids": ("small", "large", "large", "large", "unknown")},
    {"normal_group_ids": ("small", "large", "large", "large", "")},
    {"expected_groups": ("large", "small", "empty")},
    {"expected_groups": ("large", "large", "small")},
    {"protected_positive_ids": ()}, {"normal_ids": ()}, {"reference_ids": ()},
    {"protected_positive_ids": ("p:0", "n:0")},
    {"protected_positive_ids": ("p:0", "q:0")},
    {"normal_ids": ("q:0", "n:1", "n:2", "n:3", "n:4")},
])
def test_manifest_rejects_nonpartition_missing_empty_unknown_duplicate_and_overlapping_roles(change):
    *_, manifest = fixture()
    with pytest.raises(ValueError):
        replace(manifest, **change)


def test_manifest_copies_mutable_lists_to_frozen_tuples():
    q_ids = [f"q:{i}" for i in range(99)]
    n_ids, group_ids, p_ids = ["n"], ["g"], ["p"]
    manifest = FrozenRiskManifest(q_ids, n_ids, group_ids, ["g"], p_ids)
    q_ids[0], n_ids[0], group_ids[0], p_ids[0] = "bad", "bad", "bad", "bad"
    assert manifest.reference_ids[0] == "q:0"
    assert manifest.normal_ids == ("n",) and manifest.normal_group_ids == ("g",)
    assert manifest.protected_positive_ids == ("p",)


@pytest.mark.parametrize("role, replacement", [
    ("normal_ids", ["n:0", "n:1", "n:2", "n:3"]),
    ("normal_ids", ["n:0", "n:1", "n:2", "n:3", "unknown"]),
    ("normal_ids", ["n:0", "n:1", "n:2", "n:3", "n:3"]),
    ("positive_ids", ["p:1", "p:0"]),
    ("reference_ids", [f"q:{i}" for i in reversed(range(99))]),
])
def test_input_scores_require_complete_frozen_identity_order(role, replacement):
    q, n, p, manifest = fixture()
    kwargs = dict(reference_scores=q, normal_scores=n, positive_scores=p,
                  reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                  positive_ids=manifest.protected_positive_ids)
    kwargs[role] = replacement
    with pytest.raises(ValueError, match="identities/order"):
        evaluate(calibrate(q, identities=manifest.reference_ids), manifest, **kwargs)


def test_inconsistent_reference_scores_or_point_metadata_cannot_be_recorded():
    q, n, p, manifest = fixture()
    point = calibrate(q, identities=manifest.reference_ids)
    kwargs = dict(reference_scores=q, normal_scores=n, positive_scores=p,
                  reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                  positive_ids=manifest.protected_positive_ids)
    for changed in [replace(point, rank=98), replace(point, threshold=tensor(0.)),
                    replace(point, threshold_tie_count=98)]:
        with pytest.raises(ValueError):
            evaluate(changed, manifest, **kwargs)
    kwargs["reference_scores"] = q + 1
    with pytest.raises(ValueError, match="do not match"):
        evaluate(point, manifest, **kwargs)
    q[0] = 0.  # Same threshold, different tie count: still not the supplied point.
    kwargs["reference_scores"] = q
    with pytest.raises(ValueError, match="do not match"):
        evaluate(point, manifest, **kwargs)


@pytest.mark.parametrize("role", ["reference_scores", "normal_scores", "positive_scores"])
def test_score_length_must_match_the_entire_declared_role(role):
    q, n, p, manifest = fixture()
    kwargs = dict(reference_scores=q, normal_scores=n, positive_scores=p,
                  reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                  positive_ids=manifest.protected_positive_ids)
    kwargs[role] = kwargs[role][:-1]
    with pytest.raises(ValueError, match="align with all frozen identities"):
        validate_alignment(calibrate(q, identities=manifest.reference_ids), manifest, **kwargs)


def test_shared_validation_preserves_the_threshold_graph_for_risk_losses():
    q, n, p, manifest = fixture()
    q.requires_grad_()
    p.requires_grad_()
    point = calibrate(q, identities=manifest.reference_ids)
    validate_alignment(point, manifest, reference_scores=q, normal_scores=n, positive_scores=p,
                       reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                       positive_ids=manifest.protected_positive_ids)
    margins = point.evaluate(p, identities=manifest.protected_positive_ids).signed_margins
    grad_q, grad_p = torch.autograd.grad(margins.sum(), (q, p))
    torch.testing.assert_close(grad_q, torch.full_like(q, -2/99), rtol=0, atol=0)
    torch.testing.assert_close(grad_p, torch.ones_like(p), rtol=0, atol=0)


def test_json_roundtrip_contains_every_score_and_replays_identical_counts_and_decisions():
    q, n, p, manifest = fixture(positive=(1., 2., 0.))
    original = run(q, n, p, manifest)
    saved = json.loads(json.dumps(metrics_to_json(original), allow_nan=False))
    assert saved["normal"]["risk"] == {"numerator": 3, "denominator": 5}
    restored_manifest = FrozenRiskManifest(**saved["manifest"])
    q2, n2, p2 = [tensor([row["score"] for row in saved["rows"][role]])
                  for role in ("reference", "normal", "protected_positive")]
    assert len(saved["rows"]["reference"]) == 99
    assert torch.equal(q2, q) and torch.equal(n2, n) and torch.equal(p2, p)
    replay = run(q2, n2, p2, restored_manifest)
    assert metrics_to_json(replay) == saved


def test_evaluation_creates_no_gradient_updates_and_consumes_no_rng():
    q, n, p, manifest = fixture()
    for values in (q, n, p):
        values.requires_grad_()
    point = calibrate(q, identities=manifest.reference_ids)
    rng = torch.get_rng_state().clone()
    report = evaluate(point, manifest, reference_scores=q, normal_scores=n, positive_scores=p,
                      reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                      positive_ids=manifest.protected_positive_ids)
    assert all(values.grad is None for values in (q, n, p))
    assert torch.equal(torch.get_rng_state(), rng)
    json.dumps(metrics_to_json(report), allow_nan=False)
