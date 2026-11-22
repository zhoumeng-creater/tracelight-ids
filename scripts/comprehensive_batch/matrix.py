"""One prespecified matrix; enumerate trajectories, snapshots and comparisons.

There is no score-dependent expansion or winning-configuration selection here.
All checkpoint budgets share one uninterrupted trajectory for each job.
"""
from __future__ import annotations


ARMS = (
    "A_BCE", "H_reference_attack", "D_reference_alignment", "G_group_tail",
    "R_reference_envelope", "S_alignment_group",
)
SEEDS = (17, 29, 43)
LEARNING_RATES = (.002, .0005)
WEIGHTS = (.25, 1., 4.)
CHECKPOINT_STEPS = (3900, 7800, 15600, 31200)
PRIMARY_STEP = 15600
MAX_STEPS = 31200
FIT_ROWS = 13252
BCE_BATCH_SIZE = 256
HGB_ARMS = ("narrow_raw8", "mixed_raw8", "narrow_ports_zero", "mixed_ports_zero")
PRIMARY_LEARNING_RATE = .002


def _number(value):
    return format(value, "g").replace(".", "p")


def configuration_id(arm, learning_rate, auxiliary_weight):
    return f"{arm}_lr{_number(learning_rate)}_w{_number(auxiliary_weight)}"


def job_id(arm, seed, learning_rate=.002, auxiliary_weight=1.):
    return f"{configuration_id(arm, learning_rate, auxiliary_weight)}_s{seed}"


def snapshot_id(identity, step):
    if step not in CHECKPOINT_STEPS:
        raise ValueError("Snapshot step is outside the registered checkpoint set")
    return f"{identity}_step{step}"


def build_plan():
    """Return fresh JSON-serializable records without reading data or models."""
    jobs = []
    for seed in SEEDS:
        for learning_rate in LEARNING_RATES:
            for arm in ARMS:
                weights = (1.,) if arm in ARMS[:3] else WEIGHTS
                for weight in weights:
                    identity = job_id(arm, seed, learning_rate, weight)
                    primary = learning_rate == PRIMARY_LEARNING_RATE and weight == 1.
                    jobs.append({
                        "job_id": identity, "family": "tabm", "arm": arm,
                        "seed": seed, "learning_rate": learning_rate,
                        "auxiliary_weight": weight, "optimizer_steps": MAX_STEPS,
                        "checkpoints": list(CHECKPOINT_STEPS),
                        "snapshot_ids": [snapshot_id(identity, step) for step in CHECKPOINT_STEPS],
                        "primary_configuration": primary, "fit_cohort": "F_expanded",
                        "same_trajectory_checkpoints": True,
                    })
    # The core runs first, but every registered job runs independently of its outcome.
    jobs.sort(key=lambda item: (not item["primary_configuration"], SEEDS.index(item["seed"]),
                               LEARNING_RATES.index(item["learning_rate"]),
                               ARMS.index(item["arm"]), item["auxiliary_weight"]))
    by_id = {item["job_id"]: item for item in jobs}
    comparisons = []

    def compare(kind, pair, before, after, scope):
        left, right = by_id[before], by_id[after]
        if left["seed"] != right["seed"]:
            raise ValueError("Every paired comparison requires the same seed")
        left_config = configuration_id(left["arm"], left["learning_rate"], left["auxiliary_weight"])
        right_config = configuration_id(right["arm"], right["learning_rate"], right["auxiliary_weight"])
        group = f"{kind}__{right_config}__vs__{left_config}"
        primary = left["primary_configuration"] and right["primary_configuration"]
        comparisons.append({
            "comparison_id": f"{group}_s{left['seed']}", "group_id": group,
            "kind": kind, "pair_name": pair, "seed": left["seed"],
            "before_job_id": before, "after_job_id": after,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_configuration": primary, "primary_step": PRIMARY_STEP if primary else None,
            "claim_scope": scope,
        })

    for seed in SEEDS:
        for lr in LEARNING_RATES:
            unit_pairs = (
                ("H_vs_A", "A_BCE", "H_reference_attack", "Reference attack-margin term at unit auxiliary weight."),
                ("D_vs_A", "A_BCE", "D_reference_alignment", "Normal reference-alignment term with unchanged BCE attack supervision."),
                ("G_vs_A", "A_BCE", "G_group_tail", "Simple business-tail separation beyond ordinary BCE."),
                ("R_vs_A", "A_BCE", "R_reference_envelope", "Whole reference-envelope recipe beyond ordinary BCE."),
                ("S_vs_A", "A_BCE", "S_alignment_group", "Whole alignment plus group-tail recipe beyond ordinary BCE."),
                ("R_vs_H", "H_reference_attack", "R_reference_envelope", "Add D while keeping H and its weight fixed."),
                ("R_vs_D", "D_reference_alignment", "R_reference_envelope", "Add H while keeping D and its weight fixed."),
                ("S_vs_D", "D_reference_alignment", "S_alignment_group", "Add G while keeping D and its weight fixed."),
            )
            for pair, before, after, scope in unit_pairs:
                compare("unit_component", pair, job_id(before, seed, lr), job_id(after, seed, lr), scope)
            for weight in WEIGHTS:
                for pair, before, after, scope in (
                    ("R_vs_G", "G_group_tail", "R_reference_envelope", "Whole auxiliary recipes at matched weight; D and the attack boundary both differ."),
                    ("S_vs_G", "G_group_tail", "S_alignment_group", "Add the weighted D term while retaining exactly the same weighted G term."),
                    ("R_vs_S", "S_alignment_group", "R_reference_envelope", "Change the attack boundary from business G to reference H, retaining the same weighted D."),
                ):
                    compare("matched_auxiliary", pair, job_id(before, seed, lr, weight),
                            job_id(after, seed, lr, weight), scope)
            for arm in ARMS[3:]:
                for weight in (.25, 4.):
                    compare("weight_sensitivity", f"{arm}_w{_number(weight)}_vs_w1",
                            job_id(arm, seed, lr, 1.), job_id(arm, seed, lr, weight),
                            "Scale the entire auxiliary objective; this is not a single-component ablation or a new model-selection rule.")
        for arm in ARMS:
            for weight in ((1.,) if arm in ARMS[:3] else WEIGHTS):
                compare("learning_rate_sensitivity", f"{arm}_lr0p0005_vs_lr0p002",
                        job_id(arm, seed, .002, weight), job_id(arm, seed, .0005, weight),
                        "Matched arm, auxiliary weight, initialization and sampling; only the fixed learning rate differs.")

    budget_comparisons = []
    for item in jobs:
        config = configuration_id(item["arm"], item["learning_rate"], item["auxiliary_weight"])
        group = f"training_budget__{config}__31200_vs_15600"
        budget_comparisons.append({
            "comparison_id": f"{group}_s{item['seed']}", "group_id": group,
            "kind": "training_budget", "seed": item["seed"], "job_id": item["job_id"],
            "before_step": PRIMARY_STEP, "after_step": MAX_STEPS,
            "claim_scope": "Prespecified continuation of one trajectory; both endpoints and earlier checkpoints are reported, never retrospectively selected.",
        })

    hgb_jobs = [{
        "job_id": "HGB_" + arm, "family": "hgb", "arm": arm,
        "seed": 20261111, "max_iter": 200, "snapshot_id": "HGB_" + arm,
        "fit_cohort": "F_base" if arm.startswith("narrow_") else "F_expanded",
        "ports_zero": arm.endswith("ports_zero"),
        "existing_artifact_reuse": "Only verified exact-match frozen artifacts; otherwise one fixed fit.",
    } for arm in HGB_ARMS]
    for pair, before, after in (
        ("mixed_vs_narrow_raw8", "HGB_narrow_raw8", "HGB_mixed_raw8"),
        ("mixed_vs_narrow_ports_zero", "HGB_narrow_ports_zero", "HGB_mixed_ports_zero"),
        ("ports_zero_vs_raw8_narrow", "HGB_narrow_raw8", "HGB_narrow_ports_zero"),
        ("ports_zero_vs_raw8_mixed", "HGB_mixed_raw8", "HGB_mixed_ports_zero"),
    ):
        group = "hgb_factorial__" + pair
        comparisons.append({
            "comparison_id": group, "group_id": group, "kind": "hgb_factorial",
            "pair_name": pair, "seed": 20261111, "before_job_id": before,
            "after_job_id": after, "checkpoint_steps": [None],
            "primary_configuration": False, "primary_step": None,
            "claim_scope": "Frozen HGB coverage/port factorial; one deterministic artifact per arm, not a three-seed TabM experiment.",
        })
    for seed in SEEDS:
        group = "architecture_descriptive__A_BCE_lr0p002_w1__vs__HGB_mixed_ports_zero"
        comparisons.append({
            "comparison_id": f"{group}_s{seed}", "group_id": group,
            "kind": "architecture_descriptive", "pair_name": "A_vs_HGB_mixed_ports_zero",
            "seed": seed, "before_job_id": "HGB_mixed_ports_zero",
            "after_job_id": job_id("A_BCE", seed), "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_configuration": False, "primary_step": None,
            "claim_scope": "Same mixed ports-zero input, different model and training recipe. The one HGB artifact is reused for three TabM seeds; these are not three independent HGB fits or a mechanism ablation.",
        })
    auxiliary_jobs = sum(item["arm"] != "A_BCE" for item in jobs)
    total_updates = sum(item["optimizer_steps"] for item in jobs)
    return {
        "format": "comprehensive-calibration-matrix-v1", "tabm_jobs": jobs,
        "hgb_jobs": hgb_jobs, "comparisons": comparisons,
        "budget_comparisons": budget_comparisons,
        "primary": {"learning_rate": PRIMARY_LEARNING_RATE, "auxiliary_weight": 1.,
                    "checkpoint_step": PRIMARY_STEP, "arms": list(ARMS), "seeds": list(SEEDS)},
        "budget": {
            "tabm_trajectories": len(jobs), "tabm_auxiliary_trajectories": auxiliary_jobs,
            "tabm_bce_only_trajectories": len(jobs) - auxiliary_jobs,
            "tabm_optimizer_updates": total_updates,
            "tabm_BCE_sampled_rows": total_updates * BCE_BATCH_SIZE,
            "tabm_auxiliary_discovery_forward_rows": auxiliary_jobs * MAX_STEPS * FIT_ROWS,
            "tabm_checkpoint_models": len(jobs) * len(CHECKPOINT_STEPS),
            "hgb_fixed_artifacts": len(hgb_jobs), "hgb_new_fits_maximum": len(hgb_jobs),
            "all_models_to_calibrate": len(jobs) * len(CHECKPOINT_STEPS) + len(hgb_jobs),
            "auxiliary_discovery_multiple_of_v2": auxiliary_jobs * MAX_STEPS / (6 * PRIMARY_STEP),
            "active_replay_cost": "Additional; measure actual active blocks, forward/backward rows, wall time and memory.",
            "wall_time_estimate_available": False,
        },
        "selection": {"automatic_model_or_seed_selection": False, "automatic_weight_or_learning_rate_selection": False,
                      "automatic_checkpoint_selection": False, "threshold_searches": 0,
                      "all_registered_results_reported": True, "independent_confirmation_complete": False},
    }


TABM_JOBS = tuple(build_plan()["tabm_jobs"])
RUN_IDS = tuple(job["job_id"] for job in TABM_JOBS)
SNAPSHOT_IDS = tuple(identity for job in TABM_JOBS for identity in job["snapshot_ids"])
