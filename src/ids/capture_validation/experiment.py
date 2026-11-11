"""Run the two fixed HGB arms on the registered external train/calibration/evaluation roles."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from pathlib import Path
import traceback

import pandas as pd
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.flow_history import pipeline
from ids.flow_history.evaluation import paired_comparison
from ids.flow_history.execution import stage
from ids.route_batch.metrics import evaluate


ARMS = ("hgb48", "hgb52_history")
SPLIT = "forward"
METADATA = [
    "row_id",
    "label",
    "attack_family",
    "source_label",
    "source_family",
    "host_key",
    "end_day_utc",
    "communication_group",
    "flow_end_ms",
]


def coverage(root):
    roles = {}
    for role in ("P", "C", "E"):
        file = root / "inputs" / f"{role}_time.parquet"
        columns = METADATA + (["truth_source"] if "truth_source" in pq.read_schema(file).names else [])
        roles[role] = pd.read_parquet(file, columns=columns)
    sizes, reasons = {}, []
    for role, frame in roles.items():
        if frame[METADATA].isna().any().any() or frame.row_id.duplicated().any():
            raise ValueError(f"Missing role metadata or duplicate {role} row IDs")
        if not frame.label.isin([0, 1]).all() or frame.attack_family.eq("Benign").ne(frame.label.eq(0)).any():
            raise ValueError(f"Inconsistent binary/family label in {role}")
        sizes[role] = {
            "rows": len(frame),
            "families": frame.attack_family.value_counts().to_dict(),
            "source_families": frame.source_family.value_counts().to_dict(),
            "distinct_source_hosts": frame.host_key.nunique(),
            "min_end_ms": int(frame.flow_end_ms.min()) if len(frame) else None,
            "max_end_ms": int(frame.flow_end_ms.max()) if len(frame) else None,
        }
    group_overlap = {}
    for left, right in (("P", "C"), ("C", "E"), ("P", "E")):
        group_overlap[f"{left}/{right}"] = len(
            set(roles[left].communication_group) & set(roles[right].communication_group)
        )
        if group_overlap[f"{left}/{right}"]:
            raise ValueError(f"Communication group overlap between {left} and {right}")
        if set(roles[left].row_id) & set(roles[right].row_id):
            raise ValueError(f"Row overlap between {left} and {right}")
        if len(roles[left]) and len(roles[right]) and sizes[left]["max_end_ms"] >= sizes[right]["min_end_ms"]:
            raise ValueError(f"Nonchronological or tied {left}/{right} boundaries")
    families = sizes["P"]["families"]
    if "Benign" not in families or len(set(families) - {"Benign"}) < 2:
        reasons.append("training lacks benign or multiple attack families")
    if set(sizes["C"]["families"]) != {"Benign"} or sizes["C"]["rows"] < 99:
        reasons.append("normal calibration cannot register the strict one percent rank")
    families = sizes["E"]["families"]
    if not {"Benign", "Scanning"}.issubset(families) or not set(families) - {"Benign", "Scanning"}:
        reasons.append("evaluation lacks benign, adjudicated scanning or other attacks")
    return roles, {
        "roles": sizes,
        "communication_group_overlap": group_overlap,
        "not_evaluable_reasons": reasons,
    }


def _train_and_score(root, binding, arm, threads):
    for name, work in (("fit", pipeline.fit), ("score", pipeline.score)):
        stage(root, f"{SPLIT}-{arm}-{name}", binding, partial(work, root, arm, SPLIT, threads, "cpu"))


def _slices(rows, masks, predictions, metrics):
    groups = {}
    for key, selected in masks.items():
        keep = selected & (rows.label.eq(0) | rows.attack_family.eq("Scanning"))
        g = rows.loc[keep]
        slice_result = {}
        for arm, frame in predictions.items():
            measures = evaluate(g.label, frame.loc[keep, "score"], metrics[arm]["threshold"])
            slice_result[arm] = {
                k: measures[k]
                for k in ("normal_rows", "attack_rows", "tp", "fp", "auc", "normalized_pauc_0_01")
            }
        slice_result["scan_and_normal_observed"] = bool(g.label.nunique() == 2)
        groups[key] = slice_result
    return groups


def compare(root, roles):
    directory = root / "evidence" / SPLIT
    predictions = {
        arm: pd.read_parquet(directory / arm / "evaluation.parquet").reset_index(drop=True) for arm in ARMS
    }
    metrics = {arm: read_json(directory / arm / "metrics.json") for arm in ARMS}
    raw, history = (predictions[arm] for arm in ARMS)
    result = paired_comparison(raw, history, metrics[ARMS[0]], metrics[ARMS[1]])
    rows = roles["E"].reset_index(drop=True)
    key = ["row_id", "label", "attack_family", "host_key", "end_day_utc", "communication_group"]
    for frame in predictions.values():
        if not rows[key].equals(frame[key]):
            raise ValueError("Saved evaluation identity differs from the registered E role")
    fields = pd.read_parquet(
        root / "inputs" / "E_time.parquet", columns=["row_id", "PROTOCOL", "TCP_FLAGS", "IN_PKTS", "OUT_PKTS"]
    )
    if not rows.row_id.equals(fields.row_id.reset_index(drop=True)):
        raise ValueError("Unaligned broad SYN input")
    broad = fields.PROTOCOL.eq(6) & fields.TCP_FLAGS.eq(2) & fields.IN_PKTS.eq(1) & fields.OUT_PKTS.eq(0)
    # Fixed before scores. The broader surface is reported separately from the original 47-field signature.
    signatures = _slices(
        rows, {"tcp_single_syn_no_return": broad, "exact_syn47": raw.syn47}, predictions, metrics
    )
    seen = rows.host_key.isin(roles["P"].host_key)
    scan = rows.attack_family.eq("Scanning")
    benign = rows.label.eq(0)
    other = ~benign & ~scan
    counters = pd.DataFrame(
        {"rows": 1, "normal": benign, "scan": scan, "other_attack": other}, index=rows.index
    )
    for name, frame in predictions.items():
        counters[f"{name}_fp"] = frame.alert & benign
        counters[f"{name}_scan_tp"] = frame.alert & scan
        counters[f"{name}_other_tp"] = frame.alert & other
    counters["recovered_attack"] = ~benign & ~raw.alert & history.alert
    counters["lost_attack"] = ~benign & raw.alert & ~history.alert
    counters["repaired_fp"] = benign & raw.alert & ~history.alert
    counters["new_fp"] = benign & ~raw.alert & history.alert
    axes = {
        "family": rows.attack_family,
        "source_family": rows.source_family.astype(str),
        "source_label": rows.source_label.astype(str),
        "day": rows.end_day_utc,
        "source_host": rows.host_key,
        "source_seen_in_train": seen.map({True: "seen", False: "unseen"}),
    }
    if "truth_source" in rows:
        truth = rows.truth_source.fillna("unrecorded").astype(str)
        axes["truth_source"] = truth
        scan_counts = (
            counters.loc[scan, ["rows", "hgb48_scan_tp", "hgb52_history_scan_tp"]]
            .groupby(truth.loc[scan])
            .sum()
        )
        result["scanning_by_truth_source"] = scan_counts.to_dict("index")
    tables = []
    for axis, values in axes.items():
        table = counters.groupby(values).sum().rename_axis("value").reset_index()
        table.insert(0, "by", axis)
        tables.append(table)
    out = root / "evidence" / "pair-strata.parquet"
    pd.concat(tables, ignore_index=True).to_parquet(out, index=False)
    result["other_attack_rows"] = int(other.sum())
    result["delta_other_attack_tp"] = int((history.alert & other).sum() - (raw.alert & other).sum())
    result["non_scanning_attack_tp_not_lower"] = result["delta_other_attack_tp"] >= 0
    result["source_hosts_evaluation"] = rows.host_key.nunique()
    result["unseen_source_hosts_evaluation"] = rows.loc[~seen, "host_key"].nunique()
    labels = rows[["row_id", "source_family", "source_label", "attack_family", "label", "flow_end_ms"]].copy()
    if "truth_source" in rows:
        labels["truth_source"] = rows.truth_source
    labels["broad_syn"] = broad.to_numpy()
    labels["exact_syn47"] = raw.syn47.to_numpy()
    labels["seen_source_in_train"] = seen.to_numpy()
    labels.to_parquet(root / "evidence/pair-labels.parquet", index=False)
    result["row_label_trace"] = "evidence/pair-labels.parquet"
    result["subsets"] = signatures
    result["strata"] = "evidence/pair-strata.parquet"
    result["arms"] = {
        arm: {
            k: stats[k]
            for k in (
                "tp",
                "fp",
                "fn",
                "tn",
                "scan_n",
                "scan_tp",
                "scan_normal_pauc",
                "families",
                "threshold",
            )
        }
        for arm, stats in metrics.items()
    }
    return result


def _cost(root, arm, threads, include_stream):
    directory = root / "evidence" / SPLIT / arm
    bench = read_json(directory / f"cpu-{threads}" / "benchmark.json")
    model = bench["benchmark"]
    row = {
        "cpu_replay": bench["replay"],
        "cpu_registration": bench["registration"],
        "package_with_threshold_bytes": bench["model_and_threshold_bytes"],
        "precomputed_single_p95_ms": model["single"]["p95_ms"],
        "precomputed_batch128_rows_per_s": model["batch"]["throughput_rows_per_second"],
        "precomputed_peak_rss_bytes": model["rss"]["sampled_peak_bytes"],
    }
    if include_stream:
        stream = read_json(directory / f"stream-{threads}" / "stream-benchmark.json")
        row["stream"] = {
            k: stream[k]
            for k in (
                "status",
                "segment",
                "measured_rows",
                "warm_prior_key_rows",
                "service_p95_ms",
                "throughput_rows_per_second",
                "hot_worker_sampled_peak_rss",
                "history_matches_offline",
                "scores_within_tolerance_and_alerts_unchanged",
            )
            if k in stream
        }
    return row


def run_pair(root, binding, threads, monitor):
    """Run two independent CPU fits; benchmark serially after both trainers exit."""
    root = Path(root)
    if threads < 2:
        raise ValueError("Parallel HGB experiment requires at least two available CPU threads")
    roles, population = coverage(root)
    path = root / "evidence" / "pair.json"
    result = {
        "status": "not_evaluable",
        "coverage": population,
        "training_threads_per_arm": threads // 2,
        "arms": list(ARMS),
        "comparison": None,
        "cost": {},
        "failures": {},
    }
    if population["not_evaluable_reasons"]:
        write_json(path, result)
        return result
    result["status"] = "running"
    write_json(path, result)
    failures = result["failures"]
    monitor.activity = "training_and_batch_scoring_two_cpu_hgb"
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_train_and_score, root, binding, arm, threads // 2): arm for arm in ARMS}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failures[futures[future]] = traceback.format_exc()
    if not failures:
        try:
            result["comparison"] = compare(root, roles)
        except Exception:
            failures["paired-evaluation"] = traceback.format_exc()
    for arm in ARMS:
        if arm in failures:
            continue
        for count in (1, threads):
            key = f"{SPLIT}-{arm}-bench-{count}"
            try:
                monitor.activity = key
                stage(root, key, binding, partial(pipeline.benchmark, root, arm, SPLIT, count))
                result["cost"][f"{arm}/cpu-{count}"] = _cost(root, arm, count, False)
                if not (root / "history-work/benchmark_segment.json").is_file():
                    result["cost"][f"{arm}/cpu-{count}"]["stream"] = {
                        "status": "unavailable",
                        "reason": "no registered stream fixture",
                    }
                    continue
                key = f"{SPLIT}-{arm}-stream-{count}"
                monitor.activity = key
                stage(root, key, binding, partial(pipeline.benchmark_stream, root, arm, SPLIT, count))
                result["cost"][f"{arm}/cpu-{count}"] = _cost(root, arm, count, True)
            except Exception:
                failures[key] = traceback.format_exc()
    costs = list(result["cost"].values())
    result["full_cost_replay_available_and_equivalent"] = len(costs) == 4 and all(
        c["cpu_replay"]["equivalent_to_reference"] and c.get("stream", {}).get("status") == "complete"
        for c in costs
    )
    result["status"] = "partial" if failures else "complete"
    write_json(path, result)
    if failures:
        raise RuntimeError("External HGB stage failures: " + ", ".join(failures))
    return result
