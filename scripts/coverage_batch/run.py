"""Fixed source-coverage x port-input comparison; real fits run in the laboratory."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_info, threadpool_limits


LEGACY = Path(__file__).resolve().parents[1] / "fit_contract/run_minimal_comparison.py"
spec = importlib.util.spec_from_file_location("fixed_report_helpers", LEGACY)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

PARAMETERS = dict(report.PARAMETERS, class_weight="balanced", categorical_features=None)
ARMS = {
    "narrow_raw8": ("F_base", False),
    "mixed_raw8": ("F_expanded", False),
    "narrow_ports_zero": ("F_base", True),
    "mixed_ports_zero": ("F_expanded", True),
}
COMPARISONS = {
    "coverage_raw8": ("narrow_raw8", "mixed_raw8"),
    "coverage_ports_zero": ("narrow_ports_zero", "mixed_ports_zero"),
    "ports_narrow": ("narrow_raw8", "narrow_ports_zero"),
    "ports_mixed": ("mixed_raw8", "mixed_ports_zero"),
}


def features(rows, zero_ports):
    values = report.matrix(rows, "features8", 8)
    if zero_ports:
        ports = values[:, 1:3]
        ports[np.isfinite(ports)] = 0.0
    return values


def calibration_threshold(values):
    """One fixed empirical 1% operating point, with strict > and no tie splitting."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration requires nonempty finite scalar scores")
    budget = len(values) // 100
    threshold = float(np.sort(values)[-(budget + 1)])
    return {"threshold": threshold, "comparison": ">", "alpha": 0.01,
            "C_count": len(values), "integer_FP_budget": budget,
            "C_empirical_FP": int(np.sum(values > threshold)),
            "population_guarantee": False}


def target_results(scores):
    grouped = defaultdict(list)
    for row in scores.values():
        if row["label"] == 1:
            grouped[row["dataset_role"], row.get("target_kind")].append(row)
    return [{"role": role, "target_kind": kind, "positive_flows": len(group),
             "TP": sum(r["prediction"] for r in group),
             "FN": sum(1 - r["prediction"] for r in group)}
            for (role, kind), group in sorted(grouped.items(), key=lambda item: str(item[0]))]


def inspect(rows, cohorts, events, units):
    index = {r["flow_id"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate flow identity")
    for role in ("F_base", "F_expanded", "C", "V", "external_development"):
        ids = cohorts[role]
        if not ids or len(ids) != len(set(ids)) or any(fid not in index for fid in ids):
            raise ValueError(f"Missing or duplicate {role} identities")
    cohort_sets = {role: set(ids) for role, ids in cohorts.items()}
    if not set(cohorts["F_base"]) < set(cohorts["F_expanded"]):
        raise ValueError("Expanded fitting must strictly include the base fitting identities")
    partitions = [set(cohorts[r]) for r in ("F_expanded", "C", "V", "external_development")]
    if sum(map(len, partitions)) != len(set.union(*partitions)):
        raise ValueError("Fit/calibration/development flow overlap")
    group_roles = {}
    for role in ("F_expanded", "C", "V", "external_development"):
        for fid in cohorts[role]:
            group = index[fid]["allocation_group"]
            group_roles.setdefault(group, set()).add(role)
    if any(len(roles) > 1 for roles in group_roles.values()):
        raise ValueError("A complete allocation group crosses experiment roles")
    score_roles = {"F_base": "F_expanded", "F_expansion": "F_expanded",
                   "C": "C", "C_background": "C", "V": "V", "external_development": "external_development"}
    for row in rows:
        role = row["dataset_role"]
        if role not in score_roles or group_roles.get(row["allocation_group"]) != {score_roles[role]}:
            raise ValueError("Scoring row belongs to an unregistered or reserved role/group")
        if role in {"C", "V", "external_development"} and row["flow_id"] not in cohort_sets[role]:
            raise ValueError("Calibration/development score row is outside its registered cohort")
    for role in ("F_base", "F_expanded"):
        if {index[fid]["label"] for fid in cohorts[role]} != {0, 1}:
            raise ValueError(f"{role} requires explicit normal and target attack labels")
    if any(index[fid]["label"] != 0 for fid in cohorts["C"]):
        raise ValueError("Calibration contains a non-normal label")
    features(rows, False)
    metadata = report.validate_metadata(rows, events, units)
    if not metadata["all_event_references_resolve"]:
        raise ValueError("Event references are unresolved in the registered score universe")
    return {"model_constructions": 0, "fit_calls": 0, "score_calls": 0,
            "threshold_calculations": 0, "unique_rows": len(rows),
            "cohorts": {r: {"rows": len(ids), "labels": dict(Counter(str(index[f]["label"]) for f in ids))}
                        for r, ids in cohorts.items()},
            "allocation_groups": {g: sorted(v) for g, v in group_roles.items()},
            "normal_units": metadata["normal_units"], "events": len(events),
            "planned_fits": len(ARMS), "parameters": PARAMETERS}


def run(rows, cohorts, events, units, output, threads, receipt, estimator_factory=None):
    inspect(rows, cohorts, events, units)
    if threads < 1:
        raise ValueError("threads must be positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing results; automatic retry of a partial fit is disabled")
    output.mkdir(parents=True, exist_ok=True)
    report.save_json(output / "source_receipt.json", receipt)
    index = {r["flow_id"]: r for r in rows}
    event_metadata = {event["event_id"]: event for event in events}
    unit_metadata = {unit["unit_id"]: unit for unit in units}
    factory = estimator_factory or HistGradientBoostingClassifier
    models, costs, thresholds, scores_by_arm = {}, {}, {}, {}
    report.save_json(output / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "sklearn": sklearn.__version__, "numpy": np.__version__, "parameters": PARAMETERS,
        "device": "cpu", "threads": threads, "threadpools_before_limit": threadpool_info()})
    with threadpool_limits(limits=threads):
        # Finish and persist all four fixed fits before computing any C or evaluation score.
        for arm, (role, zero_ports) in ARMS.items():
            fitting = [index[fid] for fid in cohorts[role]]
            estimator = factory(**PARAMETERS)
            started = time.perf_counter()
            estimator.fit(features(fitting, zero_ports), np.asarray([r["label"] for r in fitting]))
            fit_seconds = time.perf_counter() - started
            path = output / f"{arm}.joblib"
            joblib.dump(estimator, path)
            models[arm] = estimator
            costs[arm] = {"fit_seconds": fit_seconds, "fit_rows": len(fitting),
                          "model_bytes": path.stat().st_size, "model_sha256": report.sha(path)}
            report.save_json(output / f"{arm}.fit_receipt.json", costs[arm])
        calibration = [index[fid] for fid in cohorts["C"]]
        c_scores = {}
        for arm, (_, zero_ports) in ARMS.items():
            started = time.perf_counter()
            values = np.asarray(models[arm].decision_function(features(calibration, zero_ports)))
            costs[arm]["calibration_score_seconds"] = time.perf_counter() - started
            thresholds[arm] = calibration_threshold(values)
            c_scores[arm] = dict(zip(cohorts["C"], values.tolist()))
        report.save_json(output / "thresholds.json", thresholds)
        c_ids = set(cohorts["C"])
        remaining = [r for r in rows if r["flow_id"] not in c_ids]
        metrics, targets = {}, {}
        for arm, (_, zero_ports) in ARMS.items():
            started = time.perf_counter()
            values = np.asarray(models[arm].decision_function(features(remaining, zero_ports)))
            costs[arm]["remaining_score_seconds"] = time.perf_counter() - started
            costs[arm]["scored_rows"] = len(rows)
            if values.shape != (len(remaining),) or not np.isfinite(values).all():
                raise ValueError("Nonfinite or non-scalar model score")
            values_by_id = c_scores[arm] | dict(zip((r["flow_id"] for r in remaining), values.tolist()))
            threshold = thresholds[arm]["threshold"]
            scores = {r["flow_id"]: {
                "arm": arm, "flow_id": r["flow_id"], "stable_flow_id": r.get("stable_flow_id"),
                "dataset_role": r["dataset_role"], "allocation_group": r["allocation_group"],
                "label": r["label"], "target_kind": r.get("target_kind"),
                "score": float(values_by_id[r["flow_id"]]), "threshold": threshold,
                "prediction": int(values_by_id[r["flow_id"]] > threshold), "comparison": ">",
            } for r in rows}
            scores_by_arm[arm] = scores
            report.save_rows(output / f"{arm}.scores.jsonl.gz", scores.values())
            event_results = [dict(event_metadata[item["event_id"]], **item)
                             for item in report.event_results(events, scores)]
            unit_results = [dict(unit_metadata[item["unit_id"]], **item)
                            for item in report.normal_unit_results(units, scores)]
            for item in event_results:
                item["source_used_for_fitting_this_arm"] = (
                    item.get("role") == "F_base" or
                    (item.get("role") == "F_expansion" and ARMS[arm][0] == "F_expanded"))
            for item in unit_results:
                item["in_sample_for_this_arm"] = arm in item.get("in_sample_for_arms", [])
            report.save_rows(output / f"{arm}.events.jsonl.gz", event_results)
            report.save_rows(output / f"{arm}.normal_units.jsonl.gz", unit_results)
            metrics[arm] = report.role_results(scores)
            targets[arm] = target_results(scores)
        pair_summaries = {}
        for name, (before, after) in COMPARISONS.items():
            paired, summary = report.paired_results(scores_by_arm[before], scores_by_arm[after])
            report.save_rows(output / f"{name}.paired.jsonl.gz", paired)
            pair_summaries[name] = {"before": before, "after": after, "by_role": summary}
    report.save_json(output / "metrics.json", metrics)
    report.save_json(output / "control_download_metrics.json", targets)
    report.save_json(output / "paired_summary.json", pair_summaries)
    report.save_json(output / "cost.json", {"threads": threads, "device": "cpu", "arms": costs,
        "scope": "Actual fixed fits and batch scoring; not per-alert latency or full extractor cost"})
    summary = {"fit_calls": 4, "arms": list(ARMS), "model_selection": False,
               "early_stopping": False, "seed_search": False, "refits": 0,
               "score_rows_per_arm": len(rows), "thresholds": thresholds,
               "historical_development_only": True, "algorithm_innovation_test": False}
    report.save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    if not args.inputs_only and not registration["execution_enabled"]:
        parser.error("The fixed batch is not materialized and registered for execution")
    if not args.inputs_only and sklearn.__version__ != "1.5.1":
        parser.error("This laboratory batch uses the project lock: scikit-learn==1.5.1")
    loaded, receipt = {}, {"registration_sha256": report.sha(args.registration), "inputs": {}}
    for key, item in registration["inputs"].items():
        path = args.registration.parent / item["path"]
        if report.sha(path) != item["sha256"]:
            raise ValueError(f"Registered input changed: {key}")
        loaded[key] = (json.loads(path.read_text(encoding="utf-8")) if key == "cohorts" else report.read_rows(path))
        receipt["inputs"][key] = dict(item)
    if registration["parameters"] != PARAMETERS or registration["arms"] != {k: list(v) for k, v in ARMS.items()}:
        raise ValueError("The registered parameters/arms differ from this runner")
    receipt["runner_sha256"] = report.sha(Path(__file__))
    receipt["report_helper_sha256"] = report.sha(LEGACY)
    for key, path in {"runner": Path(__file__), "report_helper": LEGACY}.items():
        if key in registration.get("code_sha256", {}) and report.sha(path) != registration["code_sha256"][key]:
            raise ValueError(f"Registered code changed: {key}")
    if args.inputs_only:
        result = inspect(loaded["rows"], loaded["cohorts"], loaded["events"], loaded["normal_units"])
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Preserve previous inputs inspection")
        args.output.mkdir(parents=True, exist_ok=True)
        report.save_json(args.output / "inputs_check.json", result)
        report.save_json(args.output / "source_receipt.json", receipt)
    else:
        result = run(loaded["rows"], loaded["cohorts"], loaded["events"], loaded["normal_units"],
                     args.output, registration["threads"], receipt)
    report.save_json(args.output / "output_receipt.json", {
        p.name: {"sha256": report.sha(p), "bytes": p.stat().st_size}
        for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "output_receipt.json"})
    print(json.dumps({k: v for k, v in result.items() if k not in {"normal_units", "allocation_groups"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
