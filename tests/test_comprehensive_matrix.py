"""Validate the registered factorial and its fair comparison coverage; no data."""
from collections import Counter, defaultdict
import json

from scripts.comprehensive_batch.matrix import (
    ARMS, CHECKPOINT_STEPS, LEARNING_RATES, PRIMARY_STEP, SEEDS, WEIGHTS,
    build_plan, job_id, snapshot_id,
)


def test_exact_factorial_contains_no_missing_or_duplicate_trajectories():
    plan = build_plan()
    jobs = plan["tabm_jobs"]
    expected = {(arm, seed, lr, weight)
                for arm in ARMS for seed in SEEDS for lr in LEARNING_RATES
                for weight in ((1.,) if arm in ARMS[:3] else WEIGHTS)}
    actual = {(j["arm"], j["seed"], j["learning_rate"], j["auxiliary_weight"]) for j in jobs}
    assert len(jobs) == len(actual) == len(expected) == 72
    assert actual == expected
    assert len({j["job_id"] for j in jobs}) == 72
    assert Counter(j["arm"] for j in jobs) == dict(zip(ARMS, (6, 6, 6, 18, 18, 18)))
    assert len(plan["hgb_jobs"]) == 4
    assert json.loads(json.dumps(plan)) == plan


def test_primary_core_is_fixed_before_sensitivity_and_all_checkpoints_share_paths():
    jobs = build_plan()["tabm_jobs"]
    core = [j for j in jobs if j["primary_configuration"]]
    assert len(core) == 18 and jobs[:18] == core
    assert all(j["learning_rate"] == .002 and j["auxiliary_weight"] == 1. for j in core)
    assert {(j["arm"], j["seed"]) for j in core} == {(arm, seed) for arm in ARMS for seed in SEEDS}
    snapshots = []
    for job in jobs:
        assert job["optimizer_steps"] == 31200
        assert job["checkpoints"] == list(CHECKPOINT_STEPS)
        assert job["same_trajectory_checkpoints"] is True
        assert job["snapshot_ids"] == [snapshot_id(job["job_id"], step) for step in CHECKPOINT_STEPS]
        snapshots.extend(job["snapshot_ids"])
    assert len(snapshots) == len(set(snapshots)) == 288
    assert PRIMARY_STEP == 15600


def test_mechanism_bridges_keep_the_components_they_claim_to_hold_fixed():
    plan = build_plan()
    jobs = {j["job_id"]: j for j in plan["tabm_jobs"]}
    comparisons = [p for p in plan["comparisons"] if p["kind"] in {"unit_component", "matched_auxiliary"}]
    pairs = defaultdict(set)
    for pair in comparisons:
        before, after = jobs[pair["before_job_id"]], jobs[pair["after_job_id"]]
        assert before["seed"] == after["seed"] == pair["seed"]
        assert before["learning_rate"] == after["learning_rate"]
        assert before["auxiliary_weight"] == after["auxiliary_weight"]
        if pair["kind"] == "unit_component":
            assert before["auxiliary_weight"] == 1.
        pairs[pair["pair_name"]].add((pair["seed"], before["learning_rate"], before["auxiliary_weight"]))
    all_settings = {(seed, lr, weight) for seed in SEEDS for lr in LEARNING_RATES for weight in WEIGHTS}
    for name in ("R_vs_G", "S_vs_G", "R_vs_S"):
        assert pairs[name] == all_settings
    assert pairs["R_vs_H"] == pairs["R_vs_D"] == pairs["H_vs_A"] == pairs["D_vs_A"]
    assert len(pairs["R_vs_H"]) == 6


def test_weight_and_learning_rate_sensitivity_have_matched_baselines_for_every_seed():
    plan = build_plan()
    jobs = {j["job_id"]: j for j in plan["tabm_jobs"]}
    weights = [c for c in plan["comparisons"] if c["kind"] == "weight_sensitivity"]
    rates = [c for c in plan["comparisons"] if c["kind"] == "learning_rate_sensitivity"]
    assert len(weights) == len(rates) == 36
    for c in weights + rates:
        a, b = jobs[c["before_job_id"]], jobs[c["after_job_id"]]
        assert a["arm"] == b["arm"] and a["seed"] == b["seed"]
        assert c["primary_configuration"] is False and c["primary_step"] is None
        if c["kind"] == "weight_sensitivity":
            assert a["learning_rate"] == b["learning_rate"]
            assert a["auxiliary_weight"] == 1. and b["auxiliary_weight"] in (.25, 4.)
        else:
            assert a["auxiliary_weight"] == b["auxiliary_weight"]
            assert a["learning_rate"] == .002 and b["learning_rate"] == .0005


def test_budget_pairs_use_same_job_and_all_comparison_groups_are_explicit():
    plan = build_plan()
    assert len(plan["budget_comparisons"]) == 72
    assert {p["job_id"] for p in plan["budget_comparisons"]} == {j["job_id"] for j in plan["tabm_jobs"]}
    assert all((p["before_step"], p["after_step"]) == (15600, 31200) for p in plan["budget_comparisons"])
    groups = defaultdict(set)
    all_ids = []
    for c in plan["comparisons"] + plan["budget_comparisons"]:
        groups[c["group_id"]].add(c["seed"])
        all_ids.append(c["comparison_id"])
    assert len(all_ids) == len(set(all_ids))
    assert all(seeds in ({17, 29, 43}, {20261111}) for seeds in groups.values())
    for field, value in plan["selection"].items():
        if field.startswith("automatic_"):
            assert value is False
    assert plan["selection"]["threshold_searches"] == 0


def test_hgb_factorial_is_fixed_and_not_misrepresented_as_three_new_hgb_seeds():
    plan = build_plan()
    hgb = {j["job_id"]: j for j in plan["hgb_jobs"]}
    assert set(hgb) == {"HGB_narrow_raw8", "HGB_mixed_raw8", "HGB_narrow_ports_zero", "HGB_mixed_ports_zero"}
    assert all(j["seed"] == 20261111 and j["max_iter"] == 200 for j in hgb.values())
    direct = [c for c in plan["comparisons"] if c["kind"] == "hgb_factorial"]
    architecture = [c for c in plan["comparisons"] if c["kind"] == "architecture_descriptive"]
    assert len(direct) == 4 and len(architecture) == 3
    assert all(c["checkpoint_steps"] == [None] for c in direct)
    assert all(c["before_job_id"] == "HGB_mixed_ports_zero" for c in architecture)
    assert {c["after_job_id"] for c in architecture} == {job_id("A_BCE", seed) for seed in SEEDS}


def test_compute_accounting_does_not_count_checkpoints_as_independent_training():
    budget = build_plan()["budget"]
    assert budget["tabm_trajectories"] == 72
    assert budget["tabm_auxiliary_trajectories"] == 66
    assert budget["tabm_bce_only_trajectories"] == 6
    assert budget["tabm_optimizer_updates"] == 2246400
    assert budget["tabm_BCE_sampled_rows"] == 575078400
    assert budget["tabm_auxiliary_discovery_forward_rows"] == 27288518400
    assert budget["tabm_checkpoint_models"] == 288
    assert budget["all_models_to_calibrate"] == 292
    assert budget["auxiliary_discovery_multiple_of_v2"] == 22.


def test_plan_calls_do_not_share_mutable_records():
    first = build_plan()
    first["tabm_jobs"][0]["checkpoints"].clear()
    assert build_plan()["tabm_jobs"][0]["checkpoints"] == list(CHECKPOINT_STEPS)
