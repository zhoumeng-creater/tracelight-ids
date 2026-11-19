"""Fixed HGB8/HGB27 laboratory comparison; no model/seed/iteration selection.

F alone fits both estimators. Both artifacts are saved before C is scored.
C25 freezes each maximum-normal-score threshold; strict greater-than alerts.
V is a fixed development replay. Optional external rows remain development.
Use synthetic inputs for local tests; real fitting belongs on the lab workstation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_info, threadpool_limits


PARAMETERS = {
    "max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.1,
    "early_stopping": False, "random_state": 20261111,
}
ARMS = {"hgb8": ("features8", 8), "hgb27": ("features27", 27)}
INFERENCE_ROLES = {"F", "V", "C", "C_background", "external_development"}
ALPHA = 0.01


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    if path is None:
        return []
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def save_rows(path, values):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as stream:
                for row in values:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def matrix(rows, key, width):
    if any(len(row[key]) != width for row in rows):
        raise ValueError(f"{key} must have width {width}")
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    if values.shape != (len(rows), width) or np.isinf(values).any():
        raise ValueError(f"Invalid {key} shape or infinite measurement")
    # JSON null becomes NaN. HGB handles missingness natively; no zero fill.
    return values


def prepare(model_rows, inference_rows):
    model_ids = set()
    for row in model_rows:
        if row["flow_id"] in model_ids:
            raise ValueError("Duplicate supervised model flow_id")
        model_ids.add(row["flow_id"])
        if row["dataset_role"] not in {"F", "V", "C"} or row["label"] not in {0, 1}:
            raise ValueError("Model rows must have explicit F/V/C binary labels")
    fitting = [row for row in model_rows if row["dataset_role"] == "F"]
    calibration = [row for row in model_rows if row["dataset_role"] == "C"]
    if {row["label"] for row in fitting} != {0, 1}:
        raise ValueError("F requires both classes")
    if len(calibration) != 25 or any(row["label"] != 0 for row in calibration):
        raise ValueError("The fixed C cohort is exactly 25 normal model rows")
    if not any(row["dataset_role"] == "V" for row in model_rows):
        raise ValueError("V model rows are required for the fixed replay")
    merged = {row["flow_id"]: dict(row) for row in model_rows}
    seen_inference = set()
    for row in inference_rows:
        fid = row["flow_id"]
        if fid in seen_inference:
            raise ValueError("Duplicate inference flow_id")
        seen_inference.add(fid)
        if row["dataset_role"] not in INFERENCE_ROLES:
            raise ValueError(f"Refuse to score reserved/unregistered role: {row['dataset_role']}")
        if row.get("label") not in {None, 0, 1}:
            raise ValueError("Inference label must be 0/1/null")
        if row["dataset_role"] == "C" and fid not in model_ids:
            raise ValueError("Non-calibration russell observations must use C_background")
        if fid in merged:
            keys = ("dataset_role", "label", "features8", "features27")
            if any(merged[fid].get(key) != row.get(key) for key in keys):
                raise ValueError(f"Model/inference conflict for {fid}")
        else:
            merged[fid] = dict(row)
    all_rows = sorted(merged.values(), key=lambda row: (row["dataset_role"], row["flow_id"]))
    for key, width in ARMS.values():
        matrix(all_rows, key, width)
    for row in all_rows:
        if row["features27"][:8] != row["features8"]:
            raise ValueError(f"Arms do not share current raw8: {row['flow_id']}")
    return fitting, calibration, all_rows


def event_results(events, scores):
    output = []
    seen = set()
    for event in events:
        eid = event["event_id"]
        if eid in seen:
            raise ValueError("Duplicate event_id")
        seen.add(eid)
        candidates = list(dict.fromkeys(event["candidate_flow_ids"]))
        missing = [fid for fid in candidates if fid not in scores]
        mode = event.get("association_mode")
        complete = event.get("candidate_set_complete") is True
        resolved = bool(candidates) and complete and not missing and mode in {"one_of", "all_confirmed"}
        alerts = [bool(scores[fid]["prediction"]) for fid in candidates if fid in scores]
        lower = upper = None
        if resolved:
            lower = all(alerts) if mode == "one_of" else any(alerts)
            upper = any(alerts)
        output.append({"event_id": eid, "label": event.get("label"), "kind": event.get("kind"),
                       "association_mode": mode, "candidate_set_complete": complete,
                       "candidate_flow_ids": candidates, "missing_score_flow_ids": missing,
                       "status": "resolved_bounds" if resolved else "unresolved",
                       "lower_hit": lower, "upper_hit": upper})
    return output


def normal_unit_results(units, scores):
    output = []
    seen = set()
    for unit in units:
        uid = unit["unit_id"]
        if uid in seen:
            raise ValueError("Duplicate normal unit_id")
        seen.add(uid)
        ids = unit["flow_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate normal denominator member in {uid}")
        if any(fid not in scores or scores[fid]["label"] != 0 for fid in ids):
            raise ValueError(f"Normal unit {uid} contains missing or non-normal score rows")
        fp = sum(scores[fid]["prediction"] for fid in ids)
        budget = len(ids) // 100
        output.append({"unit_id": uid, "flow_ids": ids, "normal_count": len(ids), "FP": fp,
                       "FPR": fp / len(ids) if ids else None, "integer_FP_budget": budget,
                       "empirical_budget_met": fp <= budget if ids else None})
    return output


def role_results(scores):
    groups = defaultdict(list)
    for row in scores.values():
        groups[row["dataset_role"]].append(row)
    output = []
    for role, group in sorted(groups.items()):
        counts = Counter((r["label"], r["prediction"]) for r in group)
        tp, fn = counts[1, 1], counts[1, 0]
        fp, tn = counts[0, 1], counts[0, 0]
        output.append({"role": role, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                       "unknown_label_rows": sum(r["label"] is None for r in group),
                       "recall": tp / (tp + fn) if tp + fn else None,
                       "FPR": fp / (fp + tn) if fp + tn else None})
    return output


def paired_results(baseline, context):
    """Compare saved decisions only; no new estimator call or threshold."""
    if baseline.keys() != context.keys():
        raise ValueError("Paired arms must contain the same flow identities")
    output = []
    by_role = defaultdict(Counter)
    for fid in sorted(baseline):
        before, after = baseline[fid], context[fid]
        role, label = before["dataset_role"], before["label"]
        if (role, label) != (after["dataset_role"], after["label"]):
            raise ValueError("Paired role/label mismatch")
        added = before["prediction"] == 0 and after["prediction"] == 1
        removed = before["prediction"] == 1 and after["prediction"] == 0
        count = by_role[role]
        count["rows"] += 1
        if label == 1:
            count["TP_added"] += added
            count["TP_lost"] += removed
        elif label == 0:
            count["FP_added"] += added
            count["FP_removed"] += removed
        else:
            count["unknown_added_alerts"] += added
            count["unknown_removed_alerts"] += removed
        output.append({"flow_id": fid, "stable_flow_id": before.get("stable_flow_id"),
                       "dataset_role": role, "label": label,
                       "baseline_score": before["score"], "context_score": after["score"],
                       "baseline_prediction": before["prediction"], "context_prediction": after["prediction"],
                       "baseline_threshold": before["threshold"], "context_threshold": after["threshold"],
                       "alert_added": added, "alert_removed": removed,
                       "decision_change": "added" if added else "removed" if removed else "unchanged"})
    keys = ("rows", "TP_added", "TP_lost", "FP_added", "FP_removed",
            "unknown_added_alerts", "unknown_removed_alerts")
    summary = [{"role": role, **{key: counts[key] for key in keys}}
               for role, counts in sorted(by_role.items())]
    return output, summary


def validate_metadata(all_rows, events, normal_units):
    """Validate references without creating any scores or predictions."""
    row_index = {row["flow_id"]: row for row in all_rows}
    if len({event["event_id"] for event in events}) != len(events):
        raise ValueError("Duplicate event_id")
    if len({unit["unit_id"] for unit in normal_units}) != len(normal_units):
        raise ValueError("Duplicate normal unit_id")
    for event in events:
        if not isinstance(event["candidate_flow_ids"], list):
            raise ValueError("candidate_flow_ids must be a list")
    for unit in normal_units:
        ids = unit["flow_ids"]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate normal denominator member")
        if any(fid not in row_index or row_index[fid].get("label") != 0 for fid in ids):
            raise ValueError("Normal unit contains missing or non-normal model/inference rows")
    return {
        "event_count": len(events), "normal_unit_count": len(normal_units),
        "event_references": [{"event_id": event["event_id"],
                              "candidate_count": len(set(event["candidate_flow_ids"])),
                              "missing_flow_ids": sorted(set(event["candidate_flow_ids"]) - row_index.keys()),
                              "association_mode": event.get("association_mode"),
                              "candidate_set_complete": event.get("candidate_set_complete") is True}
                             for event in events],
        "normal_units": [{"unit_id": unit["unit_id"], "normal_count": len(unit["flow_ids"]),
                          "integer_FP_budget": len(unit["flow_ids"]) // 100,
                          "all_members_resolve_to_known_normal": True} for unit in normal_units],
        "all_event_references_resolve": all(fid in row_index for event in events
                                            for fid in event["candidate_flow_ids"]),
    }


def inspect_inputs(model_rows, inference_rows, events, normal_units):
    fitting, calibration, all_rows = prepare(model_rows, inference_rows)
    metadata = validate_metadata(all_rows, events, normal_units)
    counts = Counter((row["dataset_role"], row.get("label")) for row in all_rows)
    return {
        "mode": "inputs_only", "model_constructions": 0, "fit_calls": 0, "score_calls": 0,
        "threshold_calculations": 0, "F_rows": len(fitting), "C_rows": len(calibration),
        "model_rows": len(model_rows), "inference_rows": len(inference_rows),
        "unique_merged_flow_ids": len(all_rows),
        "role_label_counts": [{"role": key[0], "label": key[1], "rows": count}
                              for key, count in sorted(counts.items(), key=lambda item: str(item[0]))],
        "feature_widths": {name: width for name, width in ARMS.values()},
        "missing_feature_values": {key: int(np.isnan(matrix(all_rows, key, width)).sum())
                                   for key, width in ARMS.values()},
        "arm_prefixes_equal": True, "reserved_rows_scored": 0,
        "metadata": metadata,
    }


def compare(model_rows, inference_rows, events, normal_units, output, threads, source_receipt=None):
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    fitting, calibration, all_rows = prepare(model_rows, inference_rows)
    validate_metadata(all_rows, events, normal_units)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; preserve the previous run")
    output.mkdir(parents=True, exist_ok=True)
    if source_receipt is not None:
        save_json(output / "source_receipt.json", source_receipt)
    labels = np.asarray([row["label"] for row in fitting], dtype=np.int64)
    estimators, cost, thresholds, calibration_scores = {}, {}, {}, {}
    all_score_rows, all_event_rows, all_unit_rows, metrics = [], [], [], {}
    arm_scores = {}
    fit_ids = [row["flow_id"] for row in fitting]
    save_json(output / "run_config.json", {
        "parameters": PARAMETERS, "arms": ARMS, "threads": threads,
        "F_flow_ids": fit_ids, "C_flow_ids": [r["flow_id"] for r in calibration],
        "model_selection": "none; report both fixed arms", "threshold_selection": "C maximum, strict >",
        "device": "cpu", "intended_machine": "laboratory RTX4090D workstation",
        "sklearn": sklearn.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "V_interpretation": "fixed development replay after C; C is chronologically later than V",
        "external_interpretation": "opened historical external development, not forward deployment",
    })
    with threadpool_limits(limits=threads):
        pools = threadpool_info()
        for arm, (key, width) in ARMS.items():
            estimator = HistGradientBoostingClassifier(**PARAMETERS)
            x = matrix(fitting, key, width)
            start = time.perf_counter()
            estimator.fit(x, labels)
            fit_seconds = time.perf_counter() - start
            path = output / f"{arm}.joblib"
            joblib.dump({"estimator": estimator, "features": key, "width": width,
                         "parameters": PARAMETERS, "F_flow_ids": fit_ids}, path)
            estimators[arm] = estimator
            cost[arm] = {"fit_seconds": fit_seconds, "fit_rows": len(fitting),
                         "iterations_fitted": int(estimator.n_iter_), "model_bytes": path.stat().st_size,
                         "model_sha256": sha(path), "score_seconds": 0.0, "scored_rows": 0}
        # Both model artifacts exist before either C cohort is scored.
        for arm, (key, width) in ARMS.items():
            x = matrix(calibration, key, width)
            start = time.perf_counter()
            values = np.asarray(estimators[arm].decision_function(x), dtype=float)
            cost[arm]["score_seconds"] += time.perf_counter() - start
            cost[arm]["scored_rows"] += len(calibration)
            if values.shape != (25,) or not np.isfinite(values).all():
                raise ValueError("C scores must be 25 finite scalar decisions")
            threshold = float(np.max(values))
            thresholds[arm] = {"threshold": threshold, "comparison": ">", "alpha": ALPHA,
                               "C_count": 25, "C_empirical_FP": 0, "integer_FP_budget": 0,
                               "population_95_percent_guarantee": False}
            calibration_scores[arm] = dict(zip((r["flow_id"] for r in calibration), values.tolist()))
        save_json(output / "thresholds.json", thresholds)
        calibration_ids = {row["flow_id"] for row in calibration}
        remaining = [row for row in all_rows if row["flow_id"] not in calibration_ids]
        for arm, (key, width) in ARMS.items():
            x = matrix(remaining, key, width)
            start = time.perf_counter()
            values = np.asarray(estimators[arm].decision_function(x), dtype=float)
            cost[arm]["score_seconds"] += time.perf_counter() - start
            cost[arm]["scored_rows"] += len(remaining)
            if values.shape != (len(remaining),) or not np.isfinite(values).all():
                raise ValueError("Inference scores must be finite scalar decisions")
            scores_by_id = calibration_scores[arm] | dict(zip((r["flow_id"] for r in remaining), values.tolist()))
            scores = {}
            for row in all_rows:
                score = float(scores_by_id[row["flow_id"]])
                scores[row["flow_id"]] = {
                    "arm": arm, "flow_id": row["flow_id"], "stable_flow_id": row.get("stable_flow_id"),
                    "dataset_role": row["dataset_role"], "label": row.get("label"), "score": score,
                    "prediction": int(score > thresholds[arm]["threshold"]),
                    "threshold": thresholds[arm]["threshold"], "comparison": ">",
                    "clock_basis": row.get("clock_basis"), "context_group": row.get("context_group"),
                    "replay_export_time": row.get("replay_export_time"),
                }
            all_score_rows.extend(scores.values())
            arm_scores[arm] = scores
            all_event_rows.extend(dict(r, arm=arm) for r in event_results(events, scores))
            all_unit_rows.extend(dict(r, arm=arm) for r in normal_unit_results(normal_units, scores))
            metrics[arm] = role_results(scores)
    paired, paired_metrics = paired_results(arm_scores["hgb8"], arm_scores["hgb27"])
    metrics["paired_by_role"] = paired_metrics
    save_rows(output / "scores.jsonl.gz", all_score_rows)
    save_rows(output / "paired_decisions.jsonl.gz", paired)
    save_rows(output / "event_results.jsonl.gz", all_event_rows)
    save_rows(output / "normal_unit_results.jsonl.gz", all_unit_rows)
    save_json(output / "metrics.json", metrics)
    save_json(output / "cost.json", {"device": "cpu", "threads": threads, "native_threadpools": pools,
                                    "measurement": "one actual fit and actual score pass per arm; no benchmark grid",
                                    "arms": cost})
    summary = {"fit_calls": 2, "fit_role": "F", "F_rows": len(fitting), "C_rows": len(calibration),
               "unique_scored_flow_ids_per_arm": len(all_rows), "arms": list(ARMS),
               "V_refits": 0, "model_selection": False, "seed_search": False,
               "normal_units": len(normal_units), "events": len(events), "thresholds": thresholds}
    save_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-rows", type=Path, required=True)
    parser.add_argument("--inference-rows", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--normal-units", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, help="Required for real fitting; not used by --inputs-only")
    parser.add_argument("--inputs-only", action="store_true", help="Inspect real inputs without constructing or running models")
    args = parser.parse_args()
    sources = {name: {"path": str(path.resolve()), "sha256": sha(path)} for name in
               ("model_rows", "inference_rows", "events", "normal_units")
               if (path := getattr(args, name)) is not None}
    source_receipt = {"inputs": sources, "script_sha256": sha(Path(__file__))}
    data = (read_rows(args.model_rows), read_rows(args.inference_rows),
            read_rows(args.events), read_rows(args.normal_units))
    if args.inputs_only:
        result = inspect_inputs(*data)
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError("Output is not empty; preserve the previous inspection")
        args.output.mkdir(parents=True, exist_ok=True)
        save_json(args.output / "source_receipt.json", source_receipt)
        save_json(args.output / "inputs_audit.json", result)
    else:
        if args.threads is None:
            parser.error("--threads is required for real fitting")
        result = compare(*data, args.output, args.threads, source_receipt=source_receipt)
    save_json(args.output / "output_receipt.json", {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "output_receipt.json"})
    display = result
    if args.inputs_only:
        display = dict(result, metadata={key: value for key, value in result["metadata"].items()
                                        if key not in {"event_references", "normal_units"}})
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
