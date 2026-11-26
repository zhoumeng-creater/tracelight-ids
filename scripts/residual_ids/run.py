"""Run the fixed CPU four-arm residual-IDS development study."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch

from scripts.ait_forward.baseline import MODEL_ID, REPO, load_baseline
from scripts.ait_forward.comparators import load_comparators
from scripts.ait_forward.decisions import calibrate_rules
from scripts.comprehensive_batch.benchmark import measure
from scripts.comprehensive_batch.matrix import PRIMARY_STEP, job_id, snapshot_id
from scripts.residual_ids.inputs import BENCHMARK_ARTIFACT, Bundle, ensure_disjoint, validate_evaluations
from scripts.residual_ids.models import FIT_SPEC, SEEDS, ResidualPredictor, fit_mlp_pair, load_tabm
from scripts.residual_ids.policy import (
    definite_event_counts,
    evaluate_alerts,
    freeze_residual,
    paired_alerts,
    predict_residual,
    rank_diagnostic,
    standalone_rank_diagnostic,
)


TABM = tuple(snapshot_id(job_id("A_BCE", seed), PRIMARY_STEP) for seed in SEEDS)
F_BASE = REPO / "docs/coverage_basis_20261119/batch/cohorts.json"
F_BASE_SHA256 = "358f8ecc0da25d9be8fc7a553a880838aa8ce39a22f70981a17ab8e346b33277"
OUTPUT_SCHEMA = "tracelight-residual-rescue-v1"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def score(predictor, rows):
    if not rows:
        return {}
    values = np.asarray(predictor.score(rows), dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("Model must return a finite score for every identity")
    return dict(zip((row["flow_id"] for row in rows), values.tolist(), strict=True))


def decision(scores, threshold):
    return {fid: value > threshold for fid, value in scores.items()}


def prepare(bundle, archive):
    if archive.name == "frozen-models.zip" and archive.parent.resolve() == bundle.directory.resolve():
        bundle.content("frozen-models.zip")
    baseline = load_baseline(archive)
    comparators = load_comparators(archive, baseline, model_ids=TABM)
    membership = json.loads(bundle.content("memberships.json"))
    fit = bundle.rows("fit_rows.jsonl.gz", {"fit"})
    calibration = bundle.rows("calibration_rows.jsonl.gz", {"C"})
    ensure_disjoint(fit, calibration)
    if Counter((row["label"], row["target_kind"]) for row in fit) != {
        (1, "control"): 473, (0, "normal"): 12308,
    } or {row["allocation_group"] for row in fit} != {"unraveled:apt_2021_single_campaign"}:
        raise ValueError("Only the fixed Unraveled one-campaign fit cohort is supported")
    f_base = F_BASE.read_bytes()
    if sha256(f_base).hexdigest() != F_BASE_SHA256:
        raise ValueError("Known HGB fitting manifest changed")
    if set(json.loads(f_base)["F_base"]) & {row["flow_id"] for row in fit}:
        raise ValueError("Residual training overlaps the fixed incumbent's original fitting IDs")
    if {r["flow_id"] for r in calibration} != baseline.calibration_scores.keys() or any(
        r["label"] != 0 for r in calibration
    ):
        raise ValueError("C input differs from the fixed normal-only calibration identities")
    groups = membership["calibration_units"]
    if {unit: set(ids) for unit, ids in groups.items()} != {
        unit: set(ids) for unit, ids in baseline.calibration_units.items()
    }:
        raise ValueError("C business allocations differ from the frozen archive")
    return baseline, comparators, membership, fit, calibration


def evaluation_input(bundle, membership, fit, calibration):
    rows = bundle.rows("evaluation_rows.jsonl.gz", {"V", "external_development"})
    ensure_disjoint(fit, calibration, rows)
    evals = membership["evaluations"]
    if evals.keys() != {"fox", "optc"}:
        raise ValueError("Only registered opened Fox and OpTC development cohorts are supported")
    validate_evaluations(rows, evals)
    return rows, evals


class ThresholdPredictor:
    def __init__(self, predictor, threshold):
        self.predictor, self.threshold = predictor, threshold

    def score(self, rows):
        return self.predictor.score(rows) > self.threshold


def interpret(report, changes, rank, enabled):
    if not enabled:
        return "disabled_incumbent_over_C_budget"
    if not report["all_normal_units_within_budget"]:
        return "stop_business_FP_overrun"
    positive = detection_changes(changes)
    gains = positive["flow_gains"] or positive["definite_event_gains"]
    losses = positive["flow_losses"] or positive["definite_event_losses"]
    if losses:
        return "observed_development_tradeoff" if gains else "stop_lost_known_detection"
    if positive["definite_event_gains"]:
        return "observed_definite_event_gain_needs_independent_confirmation"
    if positive["flow_gains"]:
        return "observed_flow_gain_only"
    observed_events = definite_event_counts(report["events"])
    if rank and rank["status"] == "observed_rank_ceiling" and all(
        count is None or count == report["targets"][kind]["TP"] for kind, count in rank["max_TP"].items()
    ) and all(
        count is None or count == observed_events[kind] for kind, count in rank["max_definite_events"].items()
    ):
        return "stop_no_observed_ranking_headroom"
    return "stop_no_gain_at_frozen_rule"


def detection_changes(changes):
    events = changes["events"]
    return {
        "flow_gains": [fid for pair in changes["targets"].values() for fid in pair["gained_flow_ids"] or ()],
        "flow_losses": [fid for pair in changes["targets"].values() for fid in pair["lost_flow_ids"] or ()],
        "definite_event_gains": [e["event_id"] for e in events if e["lower_hit"]["gained"] is True],
        "definite_event_losses": [e["event_id"] for e in events if e["lower_hit"]["lost"] is True],
        "definite_control_event_gains": [
            e["event_id"] for e in events if e["kind"] == "control" and e["lower_hit"]["gained"] is True
        ],
    }


def interpret_specialization(report, changes, rule):
    if not rule["enabled"]:
        return "disabled_incumbent_over_C_budget"
    if not report["all_normal_units_within_budget"]:
        return "stop_business_FP_overrun"
    positive = detection_changes(changes)
    gains = positive["flow_gains"] or positive["definite_event_gains"]
    losses = positive["flow_losses"] or positive["definite_event_losses"]
    if not gains:
        directions = [unit["alert_count_delta"] for unit in changes["normal_units"]]
        if not losses and all(delta <= 0 for delta in directions) and any(delta < 0 for delta in directions):
            return "observed_FP_reduction_only_no_additional_detection"
        return "stop_no_added_detection_over_matched_ordinary"
    if losses:
        return "observed_tradeoff_against_matched_ordinary"
    if positive["definite_event_gains"]:
        return "observed_definite_event_gain_against_ordinary_needs_independent_confirmation"
    return "observed_flow_gain_only_against_ordinary"


def fit_all(output, rows, alerts):
    exposures = {
        key: sum(row["label"] == label and alerts[row["flow_id"]] == selected for row in rows)
        for key, label, selected in (
            ("unchanged_positive_rows", 1, False), ("removed_positive_rows", 1, True),
            ("unchanged_normal_rows", 0, False), ("removed_normal_rows", 0, True),
        )
    }
    if exposures != {
        "unchanged_positive_rows": 473, "removed_positive_rows": 0,
        "unchanged_normal_rows": 12249, "removed_normal_rows": 59,
    }:
        raise ValueError("Incumbent training allocation differs from the verified archived decision trace")
    predictors, histories = {}, {}
    for seed in SEEDS:
        pair, history = fit_mlp_pair(rows, alerts, seed, output / "mlp")
        predictors.update({f"{kind}_s{seed}": model for kind, model in pair.items()})
        histories[str(seed)] = history
    write(output / "training.json", {
        "spec": FIT_SPEC, "seed_pair_histories": histories, "mask_exposure": exposures,
        "masked_fit_flow_ids": sorted(fid for fid, alert in alerts.items() if alert),
        "mechanism_limit": "one_campaign_no_positive_weight_change",
        "ordinary": "class_balanced_BCE_full_F",
        "specialist": "same_class_weights_and_batch_scale_zero_on_frozen_HGB_alerts",
    })
    return predictors


def evaluate(output, predictions, scores, incumbent_threshold, branch_rules, rows, suites):
    summaries = {}
    for name, suite in suites.items():
        ids = suite["flow_ids"]
        scope = {key: suite[key] for key in ("normal_units", "attack_ids", "events")}
        suite_predictions = {m: {fid: p[fid] for fid in ids} for m, p in predictions.items()}
        reports = {m: evaluate_alerts(p, **scope, required_events=suite.get("required_events", ()))
                   for m, p in suite_predictions.items()}
        pairs, status = {}, {}
        ranking = {MODEL_ID: standalone_rank_diagnostic(
            {fid: scores[MODEL_ID][fid] for fid in ids}, scope["normal_units"], scope["attack_ids"], scope["events"],
        )}
        for mid, alerts in suite_predictions.items():
            if mid == MODEL_ID:
                continue
            base = suite_predictions[MODEL_ID]
            pairs[mid] = paired_alerts(
                base, alerts, normal_units=scope["normal_units"], attack_ids=scope["attack_ids"],
                before_report=reports[MODEL_ID], after_report=reports[mid],
            )
            if mid in branch_rules:
                rank = rank_diagnostic(
                    {fid: scores[MODEL_ID][fid] for fid in ids},
                    {fid: scores[mid][fid] for fid in ids if fid in scores[mid]},
                    scope["normal_units"], scope["attack_ids"], incumbent_threshold, base, scope["events"],
                )
            else:
                rank = standalone_rank_diagnostic(
                    {fid: scores[mid][fid] for fid in ids}, scope["normal_units"], scope["attack_ids"], scope["events"],
                )
            ranking[mid] = rank
            status[mid] = interpret(reports[mid], pairs[mid], rank, branch_rules.get(mid, {}).get("enabled", True))
        matched, specialization = {}, {}
        for seed in SEEDS:
            ordinary, specialist = f"ordinary_s{seed}", f"specialist_s{seed}"
            matched[str(seed)] = paired_alerts(
                suite_predictions[ordinary], suite_predictions[specialist],
                normal_units=scope["normal_units"], attack_ids=scope["attack_ids"],
                before_report=reports[ordinary], after_report=reports[specialist],
            )
            specialization[str(seed)] = interpret_specialization(
                reports[specialist], matched[str(seed)], branch_rules[specialist],
            )
        summaries[name] = {
            "outcomes": reports, "relative_to_hgb": pairs, "specialist_vs_ordinary": matched,
            "ranking_diagnostics": ranking, "disposition": status, "specialist_disposition": specialization,
            "positive_changes_vs_hgb": {key: detection_changes(pair) for key, pair in pairs.items()},
            "positive_changes_specialist_vs_ordinary": {
                seed: detection_changes(pair) for seed, pair in matched.items()
            },
        }
    write(output / "evaluation.json", {"scope": "opened_development_only", "cohorts": summaries})
    with gzip.open(output / "decisions_by_flow.jsonl.gz", "xt", encoding="utf-8") as stream:
        for row in rows:
            fid = row["flow_id"]
            item = {key: row[key] for key in ("flow_id", "stable_flow_id", "dataset_role", "label", "target_kind")}
            item["alerts"] = {key: values[fid] for key, values in predictions.items()}
            item["scores"] = {key: values.get(fid) for key, values in scores.items()}
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
    return summaries


def cost_comparison(summaries, predictions, suites, arms):
    result = {}
    for name, summary in summaries.items():
        positives = [fid for ids in suites[name]["attack_ids"].values() if ids for fid in ids]
        performance = {
            mid: (
                {fid for fid in positives if predictions[mid][fid]},
                {e["event_id"] for e in obs["events"] if e["lower_hit"] is True},
                tuple(u["FP"] for u in obs["normal_units"]),
            )
            for mid, obs in summary["outcomes"].items()
        }
        assessments = {}
        for seed in SEEDS:
            model = f"specialist_s{seed}"
            against = (
                summary["positive_changes_vs_hgb"][model],
                summary["positive_changes_specialist_vs_ordinary"][str(seed)],
            )
            reasons = []
            if not summary["outcomes"][model]["all_normal_units_within_budget"]:
                reasons.append("business_FP_budget_exceeded")
            if any(p["flow_losses"] or p["definite_event_losses"] for p in against):
                reasons.append("known_detection_lost")
            if any(not p["definite_control_event_gains"] for p in against):
                reasons.append("no_definite_control_event_gain_over_both_references")
            dominated_by = [tabm for tabm in TABM if dominates(tabm, model, performance, arms)]
            if dominated_by:
                reasons.append("dominated_by_standalone_TabM_on_measured_cost_and_outcomes")
            assessments[str(seed)] = {
                "supported_in_opened_development": not reasons,
                "reasons": reasons, "dominated_by": dominated_by,
                "dimensions": ["per_business_FP", "known_positive_ids", "definite_event_ids",
                               "model_bytes", "CPU_p95_ms_batch_1_256_1024"],
            }
        result[name] = assessments
    return result


def dominates(left, right, performance, costs):
    ai, ae, af = performance[left]
    bi, be, bf = performance[right]
    ac, bc = [
        (costs[key]["model_bytes"], *(costs[key]["batches"][n]["p95_ms"] for n in ("1", "256", "1024")))
        for key in (left, right)
    ]
    return bi <= ai and be <= ae and all(a <= b for a, b in zip(af + ac, bf + bc, strict=True)) and (
        (ai, ae, af, ac) != (bi, be, bf, bc)
    )


def run(archive, bundle_dir, output, *, check_only=False):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    bundle = Bundle(bundle_dir)
    baseline, comparators, membership, fit, calibration = prepare(bundle, archive)
    tabm = {mid: load_tabm(comparator) for mid, comparator in comparators.items()}
    groups = membership["calibration_units"]
    original = calibrate_rules(baseline.calibration_scores, groups)["business_max"]
    threshold = original["threshold"]
    if check_only:
        evaluation_rows, cohorts = evaluation_input(bundle, membership, fit, calibration)
        return {"status": "verified_no_fit_no_forward", "fit_rows": len(fit),
                "calibration_rows": len(calibration), "evaluation_rows": len(evaluation_rows),
                "cohorts": sorted(cohorts), "incumbent_threshold": threshold,
                "tabm": list(tabm), "manifest_sha256": bundle.digest}

    output.mkdir(parents=True, exist_ok=False)
    write(output / "registration.json", {
        "schema": OUTPUT_SCHEMA, "input_manifest_sha256": bundle.digest,
        "input_manifest": bundle.manifest, "fit_spec": FIT_SPEC, "seeds": SEEDS,
        "incumbent": {"id": MODEL_ID, "model_sha256": baseline.model_sha256, "rule": original},
        "tabm": {mid: comparator.model_sha256 for mid, comparator in comparators.items()},
        "code_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "runtime": {name: version(name) for name in ("numpy", "torch", "scikit-learn", "tabm", "rtdl-num-embeddings")},
        "device": "cpu", "threads": 4, "selection": "none_all_registered_seeds_reported",
    })
    c_hgb = score(baseline, calibration)
    if c_hgb != baseline.calibration_scores:
        raise ValueError("Incumbent CPU C features or scores lost exact archive parity")
    hgb_fit_alert = decision(score(baseline, fit), threshold)
    branches = fit_all(output, fit, hgb_fit_alert)
    cal_fallback = [row for row in calibration if c_hgb[row["flow_id"]] <= threshold]
    branch_rules = {key: freeze_residual(c_hgb, score(p, cal_fallback), groups, threshold)
                    for key, p in branches.items()}
    tabm_rules = {}
    for key, predictor in tabm.items():
        current_c = score(predictor, calibration)
        archived_c = comparators[key].calibration_scores
        archived_rule = calibrate_rules(archived_c, groups)["business_max"]
        cpu_rule = calibrate_rules(current_c, groups)["business_max"]
        archived_flags = decision(archived_c, archived_rule["threshold"])
        cpu_flags = decision(current_c, cpu_rule["threshold"])
        tabm_rules[key] = {
            "cpu_rule": cpu_rule, "archived_gpu_rule": archived_rule,
            "max_absolute_cpu_gpu_C_difference": max(abs(v - archived_c[fid]) for fid, v in current_c.items()),
            "C_alert_differences": sum(v != archived_flags[fid] for fid, v in cpu_flags.items()),
        }
    write(output / "operating_points.json", {"hgb": original, "tabm": tabm_rules, "branches": branch_rules})

    rows, suites = evaluation_input(bundle, membership, fit, calibration)
    scores = {MODEL_ID: score(baseline, rows)}
    incumbent_alerts = decision(scores[MODEL_ID], threshold)
    falls_through = [row for row in rows if not incumbent_alerts[row["flow_id"]]]
    predictions = {MODEL_ID: incumbent_alerts}
    for key, predictor in tabm.items():
        scores[key] = score(predictor, rows)
        predictions[key] = decision(scores[key], tabm_rules[key]["cpu_rule"]["threshold"])
    for key, predictor in branches.items():
        scores[key] = score(predictor, falls_through)
        predictions[key] = predict_residual(incumbent_alerts, scores[key], branch_rules[key])
    report = evaluate(output, predictions, scores, threshold, branch_rules, rows, suites)

    benchmark_rows = (bundle.rows(BENCHMARK_ARTIFACT, {"fit"})
                      if BENCHMARK_ARTIFACT in bundle.manifest["artifacts"] else fit)[:1024]
    benchmark_alerts = decision(score(baseline, benchmark_rows), threshold)
    bench = {MODEL_ID: (ThresholdPredictor(baseline, threshold), len(baseline.model_bytes))}
    for key, predictor in tabm.items():
        bench[key] = (ThresholdPredictor(predictor, tabm_rules[key]["cpu_rule"]["threshold"]),
                      len(comparators[key].model_bytes))
    for key, predictor in branches.items():
        bench[key] = (ResidualPredictor(baseline, predictor, branch_rules[key]),
                      len(baseline.model_bytes) + (output / "mlp" / f"{key}.pt").stat().st_size)
    cost_arms = {key: {"model_bytes": size, "batches": measure(predictor, benchmark_rows)}
                 for key, (predictor, size) in bench.items()}
    tradeoffs = cost_comparison(report, predictions, suites, cost_arms)
    write(output / "cpu_cost.json", {
        "device": "cpu", "threads": 4, "scope": "preloaded_canonical_features_to_decision",
        "benchmark_flow_ids": [row["flow_id"] for row in benchmark_rows],
        "arms": cost_arms, "specialist_tradeoffs": tradeoffs,
        "incumbent_nonalert_counts": {
            "C": {"nonalerts": len(cal_fallback), "rows": len(calibration)},
            **{name: {"nonalerts": sum(not incumbent_alerts[fid] for fid in suite["flow_ids"]),
                      "rows": len(suite["flow_ids"])} for name, suite in suites.items()},
            "benchmark": {"nonalerts": sum(not alert for alert in benchmark_alerts.values()),
                          "rows": len(benchmark_rows)},
        },
        "loopback_latency_claimed": False,
    })
    status = {
        "status": "completed_fixed_opened_development_comparison", "independent_confirmation": False,
        "fit_count": len(branches), "eval_row_count": len(rows),
        "assessments": {name: values["disposition"] for name, values in report.items()},
        "specialist_assessments": {name: values["specialist_disposition"] for name, values in report.items()},
        "specialist_cost_tradeoffs": tradeoffs,
        "mechanism_limits": ["single_fit_campaign", "no_positive_weight_difference", "opened_evaluation"],
    }
    write(output / "summary.json", status)
    write(output / "output_receipt.json", {
        path.relative_to(output).as_posix(): {
            "bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*")) if path.is_file()
    })
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.check_only and args.output is None:
        parser.error("--output is required to run the study")
    print(json.dumps(run(args.archive, args.bundle, args.output, check_only=args.check_only)))


if __name__ == "__main__":
    main()
