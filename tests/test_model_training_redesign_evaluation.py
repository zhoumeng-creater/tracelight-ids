"""Synthetic evaluation contracts; no archived score or real model execution."""
import copy

import numpy as np
import pandas as pd
import pytest

from ids.model_training_redesign.evaluation import calibrate, compare, evaluate, evaluate_decisions


def _frame(source, attack):
    normal = 100
    n = normal + attack
    tcp = source.startswith("TQH-D")
    return pd.DataFrame({
        "source_id": [source] * n, "row_id": np.arange(n),
        "label": [0] * normal + [1] * attack,
        "family": ["Benign"] * normal + ["ExternalC2"] * attack,
        "flow_end_ms": np.repeat(1700000000000, n),
        "ip_protocol": np.repeat(6 if tcp else 17, n),
        "orig_port": np.repeat(12345, n), "resp_port": np.repeat(8443 if tcp else 443, n),
        "duration_seconds": [0.2] * normal + [31.0] * attack,
    })


@pytest.fixture
def fixture():
    b = [f"TQH-B_i3600_j{j}" for j in (0, 30, 70)]
    d = [f"TQH-D_i{i}_j{j}" for i in (30, 300, 1800, 3600) for j in (0, 30, 70)]
    frame = pd.concat([*[_frame(cell, 24) for cell in b], *[_frame(cell, 2) for cell in d]],
                      ignore_index=True)
    design = {"training": {"seeds": [17, 29, 43]}, "roles": {
        role: {"cells": [{"source_id": cell, "normal": 100, "attack": attacks} for cell in cells],
               "normal": 100 * len(cells), "attack": attacks * len(cells),
               "rows": (100 + attacks) * len(cells)}
        for role, cells, attacks in (("E_B", b, 24), ("E_D", d, 2))}}
    positive = frame.label.eq(1).to_numpy()
    weaker = positive & (frame.source_id.isin(b) | frame.row_id.eq(100)).to_numpy()
    scores = positive.astype(float)
    control_scores = scores.copy()
    control_scores[positive & frame.source_id.isin(d).to_numpy() & frame.row_id.eq(101).to_numpy()] = -1
    decisions = {}
    for model_id in ("legacy_hgb_or_rule", "r26_target_mlp17", "r26_target_mlp29", "r26_target_mlp43"):
        decisions[model_id] = evaluate_decisions(frame, weaker, model_id=model_id)[2]
    for seed in (17, 29, 43):
        for arm, values in (("protocol_blind", scores), ("protocol_aware", control_scores)):
            model_id = f"{arm}_s{seed}"
            decisions[model_id] = evaluate(frame, values, 0.5, model_id=model_id)[2]
        model_id = f"r29_tabm_ple_s{seed}"
        # R29 misses the complete old-reference D TP set; historical alerts only.
        decisions[model_id] = evaluate_decisions(
            frame, positive & frame.source_id.isin(b).to_numpy(), model_id=model_id)[2]
    return frame, design, decisions


def _run(fixture, **kwargs):
    _, design, decisions = fixture
    return compare(decisions, design, implementation_contract_pass=True, **kwargs)


def _rescore(decisions, model_id, selector, score):
    current = decisions[model_id]
    current.loc[selector(current), "score"] = score
    current["alert"] = current.score.gt(float(current.threshold.iloc[0]))


def test_unchanged_fixed_c_rank_strict_greater_and_count():
    result = calibrate(np.arange(5511, dtype=float))
    assert result["rank"] == 5457 and result["threshold"] == 5456
    assert result["calibration_fp"] == 54 and result["ready"]
    frame = _frame("sample", 2)
    unit, _, decision = evaluate(frame, np.zeros(len(frame)), calibrate(np.zeros(5511)), model_id="m")
    assert unit[0]["qualified"] and not decision.alert.any()
    with pytest.raises(ValueError, match="registered"):
        calibrate(np.arange(5510))
    with pytest.raises(ValueError, match="finite"):
        calibrate(np.full(5511, np.nan))


def test_all_registered_conditions_pass_and_old_loss_identity_recovery(fixture):
    result = _run(fixture)
    assert result["all_seed_mechanism_pass"] and result["all_seed_effect_pass"]
    assert result["adoption_pass"] and not result["automatic_deployment_adoption"]
    seed = result["seeds"]["17"]
    assert seed["mechanism_differences"]["D_TP_delta"] == 12
    pair = seed["comparisons"]["legacy_hgb_or_rule"]
    assert pair["strict_D_cell_tp_gain"] and pair["D_net_tp"] == 12
    diagnostic = seed["historical_R29_loss_diagnostics"]["legacy_hgb_or_rule"]
    assert diagnostic["old_lost_n"] == diagnostic["recovered_n"] == 12
    assert diagnostic["still_lost_n"] == diagnostic["newly_lost_from_R29_retained_n"] == 0
    assert diagnostic["old_lost_duration_seconds"] == {"min": 31, "median": 31, "max": 31}
    assert not diagnostic["duration_cutoff_applied"]
    assert len(diagnostic["recovered_identities"]) == 12


def test_identity_order_is_irrelevant_but_truth_alignment_is_not(fixture):
    _, _, decisions = fixture
    decisions["protocol_blind_s17"] = decisions["protocol_blind_s17"].sample(frac=1, random_state=9)
    assert _run(fixture)["adoption_pass"]
    decisions["protocol_blind_s17"].loc[0, "duration_seconds"] += 1
    with pytest.raises(ValueError, match="metadata differ"):
        _run(fixture)


@pytest.mark.parametrize("fault", ["missing_candidate", "missing_history", "missing_old_loss", "duplicate",
                                  "missing_cell", "changed_identity", "count_shift", "nonfinite"])
def test_incomplete_or_inconsistent_results_cannot_pass(fixture, fault):
    _, design, decisions = fixture
    main = "protocol_blind_s17"
    if fault == "missing_candidate":
        decisions.pop(main)
    elif fault == "missing_history":
        decisions.pop("r26_target_mlp43")
    elif fault == "missing_old_loss":
        decisions.pop("r29_tabm_ple_s17")
    elif fault == "duplicate":
        decisions[main] = pd.concat([decisions[main], decisions[main].iloc[[0]]])
    elif fault == "missing_cell":
        decisions[main] = decisions[main].loc[decisions[main].source_id.ne("TQH-D_i30_j0")]
    elif fault == "changed_identity":
        decisions[main].loc[0, "row_id"] = 999
    elif fault == "count_shift":
        design["roles"]["E_D"]["cells"][0]["normal"] += 1
    else:
        decisions[main].loc[0, "score"] = np.inf
    with pytest.raises(ValueError):
        _run(fixture)


def test_one_seed_failing_cannot_be_hidden_by_two_good_seeds(fixture):
    _, _, decisions = fixture
    current = decisions["protocol_blind_s43"]
    control = decisions["protocol_aware_s43"]
    current["score"], current["alert"] = control.score.copy(), control.alert.copy()
    result = _run(fixture)
    assert result["seeds"]["17"]["mechanism_pass"]
    assert not result["seeds"]["43"]["mechanism_pass"]
    assert not result["adoption_pass"]


def test_numerical_effect_pass_does_not_replace_mechanism_or_implementation(fixture):
    frame, design, decisions = fixture
    for seed in (17, 29, 43):
        name = f"protocol_aware_s{seed}"
        # Same perfect ranking, different fixed decision: no ranking mechanism gain.
        decisions[name] = evaluate(frame, frame.label.to_numpy(float), 2.0, model_id=name)[2]
    result = _run(fixture)
    assert result["all_seed_effect_pass"] and not result["all_seed_mechanism_pass"]
    assert not result["adoption_pass"]
    result = compare(decisions, design, implementation_contract_pass=False)
    assert not result["implementation_contract_pass"] and not result["adoption_pass"]
    with pytest.raises(ValueError, match="boolean"):
        compare(decisions, design, implementation_contract_pass="passed")


def test_false_implementation_gate_blocks_otherwise_passing_result(fixture):
    _, design, decisions = fixture
    result = compare(decisions, design, implementation_contract_pass=False)
    assert result["all_seed_effect_pass"] and result["all_seed_mechanism_pass"]
    assert not result["adoption_pass"]


def test_budget_b_capability_and_historical_cell_guards_are_independent(fixture):
    _, _, decisions = fixture
    _rescore(decisions, "protocol_blind_s17", lambda f: f.source_id.eq("TQH-D_i30_j0")
             & f.row_id.isin([0, 1]), 1)
    result = _run(fixture)
    assert not result["seeds"]["17"]["effect_checks"]["all_15_cells_qualified"]
    assert not result["adoption_pass"]


def test_missing_one_b_attack_blocks_adoption(fixture):
    _, _, decisions = fixture
    _rescore(decisions, "protocol_blind_s29", lambda f: f.source_id.eq("TQH-B_i3600_j0")
             & f.row_id.eq(123), -1)
    result = _run(fixture)
    assert not result["seeds"]["29"]["effect_checks"]["B_each_cell_24_TP_total_72"]
    assert not result["adoption_pass"]


def test_qualified_historical_tp_drop_blocks_even_with_overall_gain(fixture):
    _, _, decisions = fixture
    historical = decisions["r26_target_mlp43"]
    cell = "TQH-D_i30_j0"
    historical.loc[historical.source_id.eq(cell) & historical.label.eq(1), "alert"] = True
    _rescore(decisions, "protocol_blind_s17", lambda f: f.source_id.eq(cell) & f.row_id.eq(101), -1)
    result = _run(fixture)
    checks = result["seeds"]["17"]["effect_checks"]
    assert checks["strict_D_total_TP_gain_over_control"]
    assert not checks["no_TP_drop_in_qualified_historical_cells"]
    assert not result["adoption_pass"]


def test_overbudget_reference_is_reported_not_a_qualified_tp_bound(fixture):
    _, _, decisions = fixture
    current = decisions["r26_target_mlp43"]
    cell = current.source_id.eq("TQH-D_i30_j0")
    current.loc[cell, "alert"] = True
    _rescore(decisions, "protocol_blind_s17", lambda f: f.source_id.eq("TQH-D_i30_j0")
             & f.label.eq(1), -1)
    result = _run(fixture)
    pair = result["seeds"]["17"]["comparisons"]["r26_target_mlp43"]
    row = next(row for row in pair["cells"] if row["source_id"] == "TQH-D_i30_j0")
    assert not row["reference_tp_is_required_bound"]
    assert row["lost_tp"] == 2 and row["qualification_repair"]
    assert not row["D_cell_strict_tp_gain"]


def test_equal_tp_counts_still_report_lost_identity_and_old_loss_recovery(fixture):
    _, _, decisions = fixture
    _rescore(decisions, "protocol_blind_s17", lambda f: f.source_id.eq("TQH-D_i30_j0")
             & f.row_id.eq(100), -1)
    result = _run(fixture)
    pair = result["seeds"]["17"]["comparisons"]["legacy_hgb_or_rule"]
    row = next(row for row in pair["cells"] if row["source_id"] == "TQH-D_i30_j0")
    assert row["net_tp"] == 0 and row["added_tp"] == row["lost_tp"] == 1
    assert row["lost_tp_identities"][0]["row_id"] == 100
    diagnostic = result["seeds"]["17"]["historical_R29_loss_diagnostics"]["legacy_hgb_or_rule"]
    assert diagnostic["recovered_n"] == 11 and diagnostic["still_lost_n"] == 1


@pytest.mark.parametrize("fault", ["alert", "multiple_thresholds", "readiness"])
def test_candidate_decisions_must_match_the_shared_score_contract(fixture, fault):
    _, _, decisions = fixture
    current = decisions["protocol_blind_s17"]
    if fault == "alert":
        current.loc[0, "alert"] = True
    elif fault == "multiple_thresholds":
        current.loc[0, "threshold"] = 0.6
    else:
        current.loc[0, "ready"] = False
    with pytest.raises(ValueError):
        _run(fixture)


def test_report_only_historical_frame_cannot_change_adoption(fixture):
    frame, _, decisions = fixture
    original = copy.deepcopy(_run(fixture))
    decisions["extra_historical"] = evaluate_decisions(
        frame, np.ones(len(frame), dtype=bool), model_id="extra_historical")[2]
    result = _run(fixture)
    assert result["adoption_pass"] == original["adoption_pass"]
    assert result["historical_report_only"] == ["extra_historical"]
