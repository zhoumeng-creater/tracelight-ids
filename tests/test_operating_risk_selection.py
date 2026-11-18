"""Selection behavior for small artificial score populations only."""
from fractions import Fraction
from copy import deepcopy

import pytest

torch = pytest.importorskip("torch")

from ids.operating_risk_training.evaluation import FrozenRiskManifest, evaluate  # noqa: E402
from ids.operating_risk_training.operating_point import calibrate  # noqa: E402
from ids.operating_risk_training.selection import compare_checkpoint  # noqa: E402


def _evaluation(normal=(1.1, 0.2, 0.4, 0.6), positive=(2.0, 3.0), transform=lambda x: x):
    q = transform(torch.linspace(0.0, 1.0, 99, dtype=torch.float64))
    n = transform(torch.tensor(normal, dtype=torch.float64))
    p = transform(torch.tensor(positive, dtype=torch.float64))
    manifest = FrozenRiskManifest(tuple(f"q{i}" for i in range(99)),
                                  ("n1", "n2", "n3", "n4"), ("a", "a", "b", "b"),
                                  ("a", "b"), ("p1", "p2"))
    return evaluate(calibrate(q, identities=manifest.reference_ids), manifest,
                    reference_scores=q, normal_scores=n, positive_scores=p,
                    reference_ids=manifest.reference_ids, normal_ids=manifest.normal_ids,
                    positive_ids=manifest.protected_positive_ids)


def _training_copy(metrics):
    result = deepcopy(metrics)
    for field in ("reference_ids", "normal_ids", "protected_positive_ids"):
        result["manifest"][field] = ["F:" + identity for identity in result["manifest"][field]]
    for role in result["rows"].values():
        for row in role:
            row["identity"] = "F:" + row["identity"]
    for field in ("passed_ids", "lost_ids"):
        result["protected_positives"][field] = ["F:" + identity
                                                for identity in result["protected_positives"][field]]
    return result


def _compare(candidate_f, candidate_v, *, reference_f, reference_v):
    return compare_checkpoint(_training_copy(candidate_f), candidate_v,
                              reference_f=_training_copy(reference_f), reference_v=reference_v)


def test_true_risk_improvement_with_all_identities_is_accepted():
    old = _evaluation()
    new = _evaluation(normal=(0.9, 0.2, 0.4, 0.6))
    result = _compare(new, new, reference_f=old, reference_v=old)
    assert result["replace_reference"]
    assert result["reference_worst_group_risk"] == Fraction(1, 2)
    assert result["candidate_worst_group_risk"] == 0


def test_common_nonlinear_monotone_transform_cannot_be_adopted():
    old = _evaluation()
    new = _evaluation(transform=torch.log1p)
    result = _compare(new, new, reference_f=old, reference_v=old)
    assert not result["replace_reference"]
    assert result["reasons"] == ["no_strict_true_worst_group_risk_improvement"]


@pytest.mark.parametrize("lost_role", ["F", "V"])
def test_even_one_positive_loss_blocks_normal_risk_gain(lost_role):
    old = _evaluation()
    new = _evaluation(normal=(0.9, 0.2, 0.4, 0.6))
    lost = _evaluation(normal=(0.9, 0.2, 0.4, 0.6), positive=(1.0, 3.0))
    result = _compare(lost if lost_role == "F" else new,
                                lost if lost_role == "V" else new,
                                reference_f=old, reference_v=old)
    assert not result["replace_reference"]
    assert result[f"lost_{lost_role.lower()}_ids"] == ["F:p1" if lost_role == "F" else "p1"]


def test_equal_true_risk_retains_reference_even_with_larger_positive_margin():
    old = _evaluation()
    new = _evaluation(normal=(1.001, 0.1, 0.3, 0.5), positive=(20.0, 30.0))
    assert not _compare(new, new, reference_f=old, reference_v=old)["replace_reference"]


def test_more_harmless_normal_ties_do_not_reject_true_improvement():
    old = _evaluation()
    new = _evaluation(normal=(1.0, 1.0, 1.0, 1.0))
    assert new["normal"]["threshold_tie_count"] == 4
    assert _compare(new, new, reference_f=old, reference_v=old)["replace_reference"]


def test_changed_partition_cannot_change_selection_denominator():
    old = _evaluation()
    new = _evaluation()
    new["manifest"]["normal_group_ids"] = ["a", "b", "b", "b"]
    with pytest.raises(ValueError, match="manifest"):
        _compare(new, new, reference_f=old, reference_v=old)


def test_training_scores_cannot_be_reused_as_validation():
    old = _evaluation()
    new = _evaluation(normal=(0.9, 0.2, 0.4, 0.6))
    with pytest.raises(ValueError, match="Training and validation"):
        compare_checkpoint(new, new, reference_f=old, reference_v=old)
