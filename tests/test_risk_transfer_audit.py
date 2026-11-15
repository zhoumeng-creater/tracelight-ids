"""Synthetic checks of historical-only threshold selection and preserved decisions."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from ids.risk_transfer import audit


def test_np_rank_is_smallest_admissible_and_matches_failure_probability():
    rule = audit.order_rule(np.arange(5511), alpha=.01, delta=.05)
    assert rule["rank"] == 5469
    assert rule["threshold"] == 5468
    assert rule["calibration_fp"] == 42
    assert rule["binomial_tail"] == pytest.approx(.03958062748674295)
    assert binom.sf(rule["rank"] - 2, 5511, .99) > .05
    assert binom.cdf(42, 5511, .01) == pytest.approx(rule["binomial_tail"])


def test_strict_ties_are_not_split_to_fill_budget():
    rule = audit.order_rule(np.zeros(1000), alpha=.01, delta=.05)
    assert rule["ready"] and rule["threshold"] == 0
    assert rule["calibration_fp"] == 0
    alerts = audit.decisions(np.array([-1., 0., 1.]), np.zeros(3, bool), rule)
    assert alerts.tolist() == [False, False, True]


def test_np_minimum_sample_boundary():
    insufficient = audit.order_rule(np.arange(298), delta=.05)
    sufficient = audit.order_rule(np.arange(299), delta=.05)
    assert not insufficient["ready"] and insufficient["threshold"] is None
    assert insufficient["rank"] == 299
    assert sufficient["ready"] and sufficient["rank"] == 299
    assert sufficient["binomial_tail"] <= .05


def test_preserved_infinite_atom_cannot_become_zero_alert_success():
    # With n=299, the only admissible order statistic is the maximum.
    # One immutable old alert makes that maximum infinite.
    scores = np.zeros(299)
    old = np.zeros(299, bool)
    old[0] = True
    rule = audit.order_rule(scores, delta=.05, preserve=old)
    assert not rule["ready"] and rule["status"] == "calibration_infeasible"
    assert rule["threshold"] is None
    assert rule["calibration_fp"] == rule["preserved_calibration_fp"] == 1
    alerts = audit.decisions(scores, old, rule)
    assert np.array_equal(alerts, old)
    result = audit.metrics(pd.DataFrame({"family": ["Benign"] * 299}), alerts, old, rule["ready"])
    assert result["fp"] == 1
    assert result["budget"] == 2
    assert not result["qualified"]  # Even though the empirical count fits the budget.


def test_union_calibrates_old_alert_cost_instead_of_two_separate_budgets():
    scores = np.arange(1000, dtype=float)
    old = np.zeros(1000, bool)
    old[:2] = True
    standalone = audit.order_rule(scores, delta=.05)
    union = audit.order_rule(scores, delta=.05, preserve=old)
    assert union["ready"]
    assert union["threshold"] > standalone["threshold"]
    alerts = audit.decisions(scores, old, union)
    assert not np.any(old & ~alerts)
    assert int(alerts.sum()) <= 1000 - union["rank"]
    assert int((old | (scores > standalone["threshold"])).sum()) > int(alerts.sum())


def test_baseline_false_positives_are_a_future_floor_and_budget_is_integer():
    frame = pd.DataFrame({"family": ["Benign"] * 199 + ["ExternalC2", "Unresolved"]})
    old = np.zeros(len(frame), bool)
    old[[0, 1, 199]] = True
    rule = {"ready": True, "preserve_old": True, "threshold": 1.}
    alerts = audit.decisions(np.zeros(len(frame)), old, rule)
    result = audit.metrics(frame, alerts, old, ready=True)
    assert result["budget"] == 1 and result["fp"] == 2
    assert not result["qualified"]
    assert result["tp"] == result["retained"] == 1
    assert result["lost"] == result["fp_lost"] == 0
    assert result["attack_n"] == 1 and result["unresolved_n"] == 1


@pytest.mark.parametrize("scores,kwargs", [
    ([], {}), ([np.inf], {}), ([np.nan], {}), ([[1., 2.]], {}),
    ([1., 2.], {"preserve": [True]}), ([1., 2.], {"alpha": 0}),
    ([1., 2.], {"delta": 1}),
])
def test_invalid_calibration_rejected(scores, kwargs):
    with pytest.raises(ValueError):
        audit.order_rule(scores, **kwargs)


@pytest.fixture
def synthetic_return(tmp_path, monkeypatch):
    root = tmp_path / "return"
    (root / "evidence/replay").mkdir(parents=True)
    (root / "evidence/calibration").mkdir(parents=True)
    candidate, baseline = "mixed-raw8-hgb", "reference-hgb-original-C-or-udp443-30s"
    names = ["TQH-B_synthetic", "TQH-D_synthetic"]
    cal = pd.DataFrame({"family": ["Benign"] * 1000,
                        candidate: np.arange(1000) / 1000,
                        baseline: np.r_[np.ones(2), np.zeros(998)]})
    cal.to_parquet(root / "evidence/replay/calibration.parquet", index=False)
    for n in names:
        frame = pd.DataFrame({"family": ["Benign"] * 199 + ["ExternalC2", "Unresolved"],
                              "flow_end_ms": np.arange(201) + 1700000000000,
                              candidate: np.linspace(-1, 2, 201),
                              baseline: np.r_[1., 1., np.zeros(197), 1., 0.]})
        frame.to_parquet(root / f"evidence/replay/{n}.parquet", index=False)
    audit.write(root / "protocol.json", {"registration": {"roles": {
        "forward_development": names[:1], "transport_external": names[1:]}}})
    audit.write(root / "evidence/calibration/mixed-raw8-hgb.json",
                {"threshold": .98, "identity_sha256": "synthetic-identity"})
    reg = {"candidate": candidate, "baseline": baseline, "alpha": .01, "delta": .05,
           "calibration_normal_rows": 1000, "calibration_identity_sha256": "synthetic-identity",
           "status": "synthetic", "rules": ["old_strong", "mixed_original",
           "old_or_mixed_original", "old_or_mixed_empirical", "mixed_np95", "old_or_mixed_np95"]}
    regpath = tmp_path / "registration.json"
    audit.write(regpath, reg)
    monkeypatch.setattr(audit, "REGISTRATION", regpath)

    def manifest():
        audit.write(root / "archive_manifest.json", {
            p.relative_to(root).as_posix(): {"sha256": audit.digest(p)}
            for p in root.rglob("*") if p.is_file() and p.name != "archive_manifest.json"})

    manifest()
    return root, names, manifest


def test_future_labels_and_scores_do_not_change_frozen_calibration_rules(synthetic_return, tmp_path, monkeypatch):
    root, names, refresh = synthetic_return
    output1, output2 = tmp_path / "first", tmp_path / "second"
    read_parquet = pd.read_parquet
    active_output = [output1]

    def guarded_read(path, *args, **kwargs):
        if any(str(path).endswith(n + ".parquet") for n in names):
            assert (active_output[0] / "calibration_receipt.json").is_file()
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    first = audit.run(root, output1)
    first_receipt = audit.read(output1 / "calibration_receipt.json")
    for n in names:
        p = root / f"evidence/replay/{n}.parquet"
        frame = read_parquet(p)
        frame["mixed-raw8-hgb"] = -frame["mixed-raw8-hgb"] * 1000
        frame.loc[:99, "family"] = "ExternalC2"
        frame.to_parquet(p, index=False)
    refresh()
    active_output[0] = output2
    second = audit.run(root, output2)
    second_receipt = audit.read(output2 / "calibration_receipt.json")
    assert first_receipt["rules"] == second_receipt["rules"]
    assert first_receipt["calibration_decisions"] == second_receipt["calibration_decisions"]
    assert not first.equals(second)
    assert not second[second.rule.str.startswith("old_or_")].lost.any()


def test_non_normal_historical_pool_is_rejected(synthetic_return, tmp_path):
    root, _, refresh = synthetic_return
    p = root / "evidence/replay/calibration.parquet"
    frame = pd.read_parquet(p)
    frame.loc[0, "family"] = "ExternalC2"
    frame.to_parquet(p, index=False)
    refresh()
    with pytest.raises(ValueError, match="normal pool"):
        audit.run(root, tmp_path / "bad-output")
    assert not (tmp_path / "bad-output/calibration_receipt.json").exists()


def test_corrupt_reference_is_rejected_before_calibrating(synthetic_return, tmp_path):
    root, names, _ = synthetic_return
    p = root / f"evidence/replay/{names[0]}.parquet"
    with p.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.run(root, tmp_path / "bad-output")
    assert not (tmp_path / "bad-output/calibration_receipt.json").exists()
