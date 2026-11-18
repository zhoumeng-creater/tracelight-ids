"""Read registered R31 histories/targets; no model import, inference, or fitting."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import hashlib
import json

import numpy as np


ROOT = Path(r"E:\OneDrive - Coventry University\中传\毕业论文\实验归档\20261118_正常子群训练结果\回传")
REPO = Path(r"E:\OneDrive - Coventry University\中传\毕业论文\代码\tracelight-ids\.worktrees\model-training-20261116")
OUT = Path(__file__).parent
INPUTS = {}
QS = ("0.90", "0.95", "0.99")


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def frac(x):
    return Fraction(x["numerator"], x["denominator"])


def dominant(tail):
    return [q for q in QS if frac(tail["H_q"][q]) == frac(tail["H"])]


def main():
    author_reference = [
        {"repository": "yandex-research/tabm", "commit": "28e47ae301c92ec37787dde1ce923a0793f405b4",
         "local_path": "docs/r31_design_review_20261117/author_sources/tabm_README.md",
         "lines": [225, 235], "sha256": "302b2d36c3bf226cc98121923ea1881617287a161e25cf9a765b82b26b9d4969",
         "supports": "Independent member losses; classification probability averaging. Does not specify this project's target or protection guards."},
        {"repository": "khainb/MFSWB", "commit": "fb5c3630841c0a02405a26233d92c5cd49f21668",
         "local_path": "docs/normal_subgroup_design_20261118/author_sources/khainb_MFSWB/SWAE/metrics/wasserstein.py",
         "lines": [[8, 13], [38, 95], [134, 165]],
         "sha256": "eceefd0f9cdca4a902d2d02d38299715798a5dd0376afb606054da9bd77c18b0",
         "supports": "Empirical quantiles, sorted distribution distances, maximum-group aggregation; not frozen one-sided IDS scalar targets or identity guards."},
        {"repository": "khainb/MFSWB", "commit": "fb5c3630841c0a02405a26233d92c5cd49f21668",
         "local_path": "docs/normal_subgroup_design_20261118/author_sources/khainb_MFSWB/SWAE/swae/trainer.py",
         "lines": [[35, 43], [89, 116]],
         "sha256": "a88484df21e56745ae0fd5ed0f95101091bee39f4cee9fdd2ee9dfd888b1d0e0",
         "supports": "Class-conditioned latent distributions plus reconstruction losses; not 1pct IDS operating-point guarantees."},
    ]
    for item in author_reference:
        path = REPO / item["local_path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == item["sha256"]
        INPUTS[str(path)] = actual
        item["hash_verified"] = True
    for name in ("losses.py", "selection.py", "targets.py", "training.py", "evaluation.py"):
        path = ROOT / "source_snapshot/src/ids/normal_subgroup_training" / name
        INPUTS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "scope": "read_only_existing_R31_history_targets_metrics_and_frozen_code",
        "new_model_loads": 0, "new_forwards": 0, "new_fits": 0,
        "new_threshold_searches": 0, "new_candidate_selection": False,
        "raw_C_E_rows_read": False,
        "fixed_author_references": author_reference,
        "interpretation": {
            "experiment_result": "The fixed six-trajectory training and selection scheme failed; all six artifacts fell back to their seed-specific R30 bytes.",
            "mathematical_infeasibility_proven": False,
            "target_completely_inert": False,
            "safe_1pct_operating_point_improvement_proven": False,
            "all_sgd_must_break_ranking_proven": False,
            "F_target_below_zero_implies_D_FP_repair": False,
            "identity_arm_is_frozen_model": False,
            "tie_pair_increase_proves_representation_collapse": False,
        },
        "seeds": {},
    }
    failure_counts = Counter()
    all_dominant = Counter()
    all_guard_pass = []
    event_count = eligible_count = 0
    for seed in (17, 29, 43):
        base = read(ROOT / f"models/transport_target_s{seed}/trajectory/baseline_metrics.json")
        baseline_tail = base["tail"]
        mapping = {tuple(i): (p, z) for i, p, z in zip(
            base["V_contract"]["identities"], base["V_contract"]["profiles"],
            base["V_contract"]["protocols"])}
        seed_result = {
            "baseline_H": baseline_tail["H"],
            "baseline_H_q": baseline_tail["H_q"],
            "baseline_H_dominant_q": dominant(baseline_tail),
            "baseline_tied_pairs": baseline_tail["tied_pairs"],
            "baseline_max_tie_block": baseline_tail["max_tie_block"],
            "baseline_unique_normal_scores": baseline_tail["unique_scores"],
            "V_positive_rows": len(base["V_positive_ids"]),
            "V_normal_rows": baseline_tail["normal"],
            "trajectories": {},
        }
        histories = {}
        for arm in ("transport_target", "identity_target"):
            task_id = f"{arm}_s{seed}"
            path = ROOT / f"models/{task_id}/trajectory"
            summary = read(path / "summary.json")
            history = read(path / "history.json")
            histories[arm] = history
            assert len(history) == 16
            assert [e["step"] for e in history] == list(range(139, 2225, 139))
            events = []
            changes = []
            worse_ids = set()
            worse_profiles = Counter()
            failures = Counter()
            for event in history:
                event_count += 1
                eligible_count += bool(event["selection"]["eligible"])
                m = event["metrics"]
                guard = event["selection"]["guards"]
                failed = [k for k, passed in guard["checks"].items() if not passed]
                failures.update(failed)
                failure_counts.update(failed)
                all_dominant.update(dominant(m["tail"]))
                if guard["passed"]:
                    all_guard_pass.append({"task_id": task_id, "step": event["step"],
                                           "strict_H_improvement": frac(m["tail"]["H"]) < frac(baseline_tail["H"])})
                for identity, current, old in zip(m["V_positive_ids"], m["V_positive_inversions"],
                                                   base["V_positive_inversions"]):
                    if current > old:
                        changes.append(current - old)
                        worse_ids.add(tuple(identity))
                        worse_profiles[str(mapping[tuple(identity)])] += 1
                events.append({
                    "step": event["step"], "guards_passed": guard["passed"],
                    "eligible": event["selection"]["eligible"], "failed_guards": failed,
                    "H": m["tail"]["H"], "H_q": m["tail"]["H_q"],
                    "H_dominant_q": dominant(m["tail"]),
                    "source_H_q": {p: t["H_q"] for p, t in m["source_tails"].items()},
                    "F_absolute_positive_score_lower_count": guard["F_lower_positive_count"],
                    "F_positive_inversion_worse_count": len(guard["F_worse_positive_ids"]),
                    "V_positive_inversion_worse_count": len(guard["V_worse_positive_ids"]),
                    "normal_tied_pairs": m["tail"]["tied_pairs"],
                    "normal_max_tie_block": m["tail"]["max_tie_block"],
                    "normal_unique_scores": m["tail"]["unique_scores"],
                    "window_path_mean_losses_not_endpoint_losses": event["path_mean"],
                    "gradient": event["gradient"],
                    "B1_completed_cycles": event["sampling"]["groups"]["B1"]["completed_cycles"],
                })
            only_v = [e["step"] for e in events if e["failed_guards"] == ["V_each_positive_inversions_not_increased"]
                      and frac(e["H"]) < frac(baseline_tail["H"])]
            seed_result["trajectories"][arm] = {
                "selected_step": summary["selected_step"], "fallback": summary["fallback"],
                "original_selected_artifact_hash_equal": summary["original_artifact_sha256"] == summary["selected_artifact_sha256"],
                "optimizer_steps": summary["optimizer_steps"],
                "guard_failure_counts": dict(failures),
                "H_strictly_improved_events": sum(frac(e["H"]) < frac(baseline_tail["H"]) for e in events),
                "H_q_strictly_improved_events": {q: sum(frac(e["H_q"][q]) < frac(baseline_tail["H_q"][q]) for e in events) for q in QS},
                "H_q_worse_events": {q: sum(frac(e["H_q"][q]) > frac(baseline_tail["H_q"][q]) for e in events) for q in QS},
                "V_worse_positive_unique_identities": len(worse_ids),
                "V_worse_positive_event_profile_protocol_counts": dict(worse_profiles),
                "V_positive_inversion_increase_range": [min(changes), max(changes)],
                "only_V_identity_guard_failed_and_H_improved_events_DESCRIPTIVE_NOT_CANDIDATES": only_v,
                "clipped_steps_total": sum(e["gradient"]["clipped_steps"] for e in events),
                "maximum_preclip_norm": max(e["gradient"]["preclip_max_norm"] for e in events),
                "B1_completed_cycles_at_finish": events[-1]["B1_completed_cycles"],
                "events": events,
            }
        seed_result["matched_step_transport_H_q_below_identity_counts"] = {
            q: sum(frac(t["metrics"]["tail"]["H_q"][q]) < frac(c["metrics"]["tail"]["H_q"][q])
                   for t, c in zip(histories["transport_target"], histories["identity_target"])) for q in QS}
        target_file = ROOT / f"preflight/protocol_blind_s{seed}/targets.npz"
        INPUTS[str(target_file)] = hashlib.sha256(target_file.read_bytes()).hexdigest()
        with np.load(target_file, allow_pickle=False) as arrays:
            normal = arrays["group_ids"] != ""
            original = arrays["reference_scores"][normal]
            target = arrays["transport_target"][normal]
            lowered = target < original
            seed_result["F_target_geometry"] = {
                "normal_rows": int(normal.sum()),
                "normal_original_above_zero": int((original > 0).sum()),
                "normal_original_max_score": float(original.max()),
                "normal_target_max_score": float(target.max()),
                "normal_lowered_rows": int(lowered.sum()),
                "normal_crossed_positive_to_nonpositive": int(((original > 0) & (target <= 0)).sum()),
                "positive_target_changes": int(np.count_nonzero(arrays["transport_target"][~normal] != arrays["reference_scores"][~normal])),
                "mean_lowering_among_lowered": float((original - target)[lowered].mean()),
                "max_lowering": float((original - target).max()),
            }
        seed_result["registered_selected_artifact_calibration_metadata"] = read(
            ROOT / f"evaluation/transport_target_s{seed}/calibration.json")
        result["seeds"][str(seed)] = seed_result
    result["all_events"] = {
        "count": event_count, "eligible_count": eligible_count,
        "H_dominant_q_counts": dict(all_dominant),
        "guard_failure_counts": dict(failure_counts),
        "guard_pass_events": all_guard_pass,
    }
    result["sources_sha256"] = INPUTS
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "design_interpretation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["all_events"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
