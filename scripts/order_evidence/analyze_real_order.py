"""Fixed real-input representation analysis, without estimators or decisions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ordered_pair_neighbor import OrderedPairNeighbor, safe_ratio


FIELDS = ("ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts",
          "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
DIM = 13


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def timestamp(value):
    return datetime.fromisoformat(value).timestamp() if isinstance(value, str) else float(value)


def raw_values(row):
    raw = row["raw8"]
    return [raw[k] for k in FIELDS] if isinstance(raw, dict) else raw


def numeric(raw):
    return np.asarray([np.log1p(v) if v is not None and np.isfinite(v) and v >= 0
                       else np.nan for v in raw[3:]], dtype=float)


def encode(raw, scales):
    values = numeric(raw)
    missing = ~np.isfinite(values)
    protocol = [raw[0] == 6, raw[0] == 17, raw[0] not in (6, 17)]
    return np.concatenate([np.nan_to_num(values) / scales, missing, protocol]).astype(float)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0}
    return dict(zip(("n", "min", "q25", "median", "q75", "max"),
                    [len(values), *np.quantile(values, [0, .25, .5, .75, 1]).tolist()]))


def summarize(records):
    return {
        "views": len(records), "physical_connections": len({r["flow_id"] for r in records}),
        "prefix_count": quantiles([r["prefix_count"] for r in records]),
        "known_attack_prefix_count": quantiles([r["known_attack_prefix_count"] for r in records]),
        "direction_norm": quantiles([np.linalg.norm(r["direction"]) for r in records]),
        "pair_vs_mass_matched_pool_linf": quantiles([r["pool_difference_linf"] for r in records]),
        "known_attack_removal_direction_linf": quantiles([r["oracle_removal_direction_linf"] for r in records]),
        "same_previous_peer": sum(r["same_previous_peer"] for r in records),
        "missing_current_numeric": sum(r["missing_numeric"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    batch = repo / "docs/coverage_basis_20261119/batch"
    paths = {
        "rows": batch / "rows.jsonl.gz",
        "events": batch / "events.jsonl.gz",
        "ait": args.work / "拟合来源与划分落实_20261119/dataset/context_inputs.jsonl.gz",
        "optc": args.work / "统一事件与方法落实_20261119/optc/context_replay_inputs.jsonl.gz",
        "unraveled": args.work / "覆盖与历史依据落实_20261119/attack_coverage/full_capture/connections.jsonl.gz",
    }
    queries = {r["flow_id"]: r for r in load(paths["rows"])
               if r["dataset_role"] not in {"C", "C_background"}}
    fit_values = np.stack([numeric(raw_values(r)) for r in queries.values() if r["fit_expanded_member"]])
    scales = np.asarray([np.median(column[np.isfinite(column) & (column > 0)])
                         for column in fit_values.T])
    assert np.isfinite(scales).all() and (scales > 0).all()
    contexts = []
    for r in load(paths["ait"]):
        if not r["group"].startswith(("ait:santos:", "ait:fox:")):
            continue
        contexts.append({"flow_id": r["measurement_flow_id"], "host": r["host"],
                         "group": r["group"], "time": timestamp(r["available_at"]),
                         "peer": r["peer"], "raw": r["raw8"], "source_family": "AIT"})
    for r in load(paths["optc"]):
        contexts.append({"flow_id": r["physical_flow_id"], "host": r["host"],
                         "group": r["group"], "time": timestamp(r["flow_end_at"]),
                         "peer": r["peer"], "raw": r["raw8"], "source_family": "OpTC"})
    for r in load(paths["unraveled"]):
        if not r["involves_monitored_host"]:
            continue
        host = r["monitored_host"]
        contexts.append({"flow_id": r["flow_id"], "host": host, "group": r["group"],
                         "time": timestamp(r["replay_export_time"]),
                         "peer": r["destination"] if r["source"] == host else r["source"],
                         "raw": raw_values(r), "source_family": "Unraveled"})
    assert all("harrison" not in r["group"] and "meterpreter" not in r["group"].lower()
               and "russell" not in r["group"] for r in contexts)
    contexts.sort(key=lambda r: (r["group"], r["host"], r["time"], r["flow_id"]))
    records, group_counts = [], []
    max_error = 0.0
    ui, uj = np.triu_indices(DIM, 1)
    for (group, host), group_iter in itertools.groupby(contexts, key=lambda r: (r["group"], r["host"])):
        rows = list(group_iter)
        neighbor = OrderedPairNeighbor(DIM, rate=0.0)
        no_known_attack = OrderedPairNeighbor(DIM, rate=0.0)
        p, a, s = np.zeros(DIM), np.zeros((DIM, DIM)), np.zeros((DIM, DIM))
        prefix_count = attack_count = 0
        previous_peers = set()
        previous_ids = []
        before = len(records)
        for time, batch_iter in itertools.groupby(rows, key=lambda r: r["time"]):
            incoming = list(batch_iter)
            embeddings = [encode(r["raw"], scales) for r in incoming]
            neighbor.advance(time)
            no_known_attack.advance(time)
            for row, u in zip(incoming, embeddings):
                meta = queries.get(row["flow_id"])
                if meta is None:
                    continue
                # Verify that metadata joins do not substitute another observation.
                assert np.allclose(np.asarray([np.nan if v is None else v for v in row["raw"]], float),
                                   np.asarray([np.nan if v is None else v for v in raw_values(meta)], float),
                                   equal_nan=True, atol=1e-7, rtol=1e-7)
                result = neighbor.read(u)
                g = result["g"]
                direct = np.concatenate([safe_ratio(a, s) @ g, np.log1p(s @ g), np.log1p(p)])
                error = float(np.max(np.abs(direct - result["candidate"])))
                max_error = max(error, max_error)
                assert np.allclose(direct, result["candidate"], atol=1e-9, rtol=1e-9)
                oracle = no_known_attack.read(u)
                records.append({
                    "flow_id": row["flow_id"], "host": host, "group": group,
                    "time": time, "source_family": row["source_family"], "peer": row["peer"],
                    "role": meta["dataset_role"], "label": meta["label"], "kind": meta["target_kind"],
                    "protocol": row["raw"][0], "response_port": row["raw"][2],
                    "prefix_count": prefix_count, "known_attack_prefix_count": attack_count,
                    "same_previous_peer": bool(row["peer"] in previous_peers),
                    "previous_batch_ids": previous_ids,
                    "missing_numeric": bool(any(v is None or v < 0 for v in row["raw"][3:])),
                    "direction": result["pair_direction"].tolist(),
                    "pooled_direction_matched": result["pooled_mass_matched_direction"].tolist(),
                    "log_pair_support": result["candidate"][DIM:2 * DIM].tolist(),
                    "log_first_support": result["candidate"][2 * DIM:].tolist(),
                    "pool_difference_linf": float(np.max(np.abs(result["pair_direction"] - result["pooled_mass_matched_direction"]))),
                    "oracle_removal_direction_linf": float(np.max(np.abs(result["pair_direction"] - oracle["pair_direction"]))),
                    "equivalence_max_abs": error,
                    "u": u, "g": g, "ratio": result["candidate_ratio_matrix"],
                })
            total = np.sum(embeddings, axis=0)
            cross = np.outer(p, total)
            a += cross - cross.T
            s += cross + cross.T
            np.fill_diagonal(s, 0)
            p += total
            neighbor.update_batch(embeddings)
            kept = [u for row, u in zip(incoming, embeddings)
                    if queries.get(row["flow_id"], {}).get("label") != 1]
            no_known_attack.update_batch(kept)
            prefix_count += len(incoming)
            attack_count += sum(queries.get(row["flow_id"], {}).get("label") == 1 for row in incoming)
            previous_peers = {row["peer"] for row in incoming}
            previous_ids = [row["flow_id"] for row in incoming]
        group_counts.append({"group": group, "host": host, "history_rows": len(rows),
                             "query_views": len(records) - before})
    observed = {r["flow_id"] for r in records}
    missing = sorted(set(queries) - observed)
    assert not missing, f"Missing declared query contexts: {missing[:10]}"

    strata = defaultdict(list)
    populations = defaultdict(list)
    for r in records:
        strata[(r["group"], r["host"], r["protocol"], r["response_port"])].append(r)
        populations[(r["group"], r["host"], r["label"], r["kind"])].append(r)
    matched = []
    matches_by_group = defaultdict(list)
    for key, members in strata.items():
        positives = [r for r in members if r["label"] == 1]
        normals = [r for r in members if r["label"] == 0]
        if not positives:
            continue
        scopes = [("within_group", normals)]
        if key[0].startswith("unraveled:"):
            start, end = min(r["time"] for r in positives), max(r["time"] for r in positives)
            scopes.append(("same_attack_period", [r for r in normals if start <= r["time"] <= end]))
        for (scope, normal_pool), attack in itertools.product(scopes, positives):
            if not normal_pool:
                matched.append({"attack_id": attack["flow_id"], "group": key[0], "host": key[1],
                                "match_scope": scope,
                                "no_same_host_group_protocol_port_normal": True})
                continue
            normal_inputs = np.stack([r["u"] for r in normal_pool])
            distances = np.linalg.norm(normal_inputs - attack["u"], axis=1)
            time_distances = np.asarray([abs(r["time"] - attack["time"]) for r in normal_pool])
            for method, index in (("nearest_time", int(np.argmin(time_distances))),
                                  ("nearest_current_input", int(np.argmin(distances)))):
                normal = normal_pool[index]
                history_part = (attack["ratio"] - normal["ratio"]) @ attack["g"]
                query_part = normal["ratio"] @ (attack["g"] - normal["g"])
                total_delta = np.asarray(attack["direction"]) - np.asarray(normal["direction"])
                assert np.allclose(history_part + query_part, total_delta, atol=1e-12)
                item = {"attack_id": attack["flow_id"], "normal_id": normal["flow_id"],
                        "group": key[0], "host": key[1], "kind": attack["kind"], "method": method,
                        "match_scope": scope,
                        "time_gap_seconds": float(normal["time"] - attack["time"]),
                        "current_input_distance": float(distances[index]),
                        "history_part_l2": float(np.linalg.norm(history_part)),
                        "current_query_part_l2": float(np.linalg.norm(query_part)),
                        "total_direction_difference_l2": float(np.linalg.norm(total_delta)),
                        "history_larger_than_query_part": bool(np.linalg.norm(history_part) > np.linalg.norm(query_part)),
                        "normal_known_attack_prefix_count": normal["known_attack_prefix_count"],
                        "same_peer": attack["peer"] == normal["peer"]}
                matched.append(item)
                matches_by_group[(key[0], key[1], attack["kind"], method, scope)].append(item)
    special = {}
    for group, host in [(g["group"], g["host"]) for g in group_counts]:
        members = [r for r in records if r["group"] == group and r["host"] == host
                   and r["protocol"] == 6 and r["response_port"] == 443]
        positives = [r for r in members if r["label"] == 1]
        if not positives:
            continue
        begin, end = min(r["time"] for r in positives), max(r["time"] for r in positives)
        normal = [r for r in members if r["label"] == 0]
        special[group + "|" + host] = {
            "attack": summarize(positives),
            "normal_before": summarize([r for r in normal if r["time"] < begin]),
            "normal_during": summarize([r for r in normal if begin <= r["time"] <= end]),
            "normal_after": summarize([r for r in normal if r["time"] > end]),
        }
    events = []
    by_id = defaultdict(list)
    for r in records:
        by_id[r["flow_id"]].append(r)
    for event in load(paths["events"]):
        if event.get("role") in {"C", "C_background"} or event.get("label") != 1:
            continue
        if "download" not in event.get("kind", ""):
            continue
        events.append({"event_id": event["event_id"], "kind": event["kind"], "role": event["role"],
                       "association_mode": event.get("association_mode"),
                       "candidate_set_complete": event.get("candidate_set_complete"),
                       "candidate_views": [{k: r[k] for k in ("flow_id", "host", "label", "prefix_count",
                           "known_attack_prefix_count", "direction", "pool_difference_linf", "oracle_removal_direction_linf")}
                           for fid in event["candidate_flow_ids"] for r in by_id[fid]]})

    report = {
        "scope": "fixed descriptive representation analysis; not a classifier comparison",
        "model_calls": 0, "fits": 0, "thresholds": 0, "calibration_used": False,
        "reserved_sources_used": False, "rate": 0.0, "dimension": DIM,
        "scales_positive_log_median_F_expanded": scales.tolist(),
        "numeric_fields": list(FIELDS[3:]), "scale_fit_rows": len(fit_values),
        "input_receipts": {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()},
        "history_views": len(contexts), "query_views": len(records), "query_connections": len(observed),
        "query_classes": dict(Counter(str(r["label"]) for r in records)),
        "missing_query_connections": missing,
        "same_information_reconstruction_max_abs_error": max_error,
        "both_state_payload_float32_plus_clock": 684, "both_readout_dimensions": 39,
        "groups": group_counts,
        "populations": [{"group": k[0], "host": k[1], "label": k[2], "kind": k[3], **summarize(v)}
                        for k, v in populations.items()],
        "same_host_tcp443_by_attack_period": special,
        "paired_summaries": [{"group": k[0], "host": k[1], "kind": k[2], "method": k[3], "match_scope": k[4],
             "pairs": len(v), "different_normal_connections": len({r["normal_id"] for r in v}),
             "history_part_l2": quantiles([r["history_part_l2"] for r in v]),
             "current_query_part_l2": quantiles([r["current_query_part_l2"] for r in v]),
             "time_gap_seconds": quantiles([r["time_gap_seconds"] for r in v]),
             "history_larger_than_query_part": sum(r["history_larger_than_query_part"] for r in v),
             "same_peer_pairs": sum(r["same_peer"] for r in v)} for k, v in matches_by_group.items()],
        "attack_download_events": events,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "real_order_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename, data in (("query_representations.jsonl.gz", records), ("matched_normal_pairs.jsonl.gz", matched)):
        with gzip.open(args.output / filename, "wt", encoding="utf-8") as stream:
            for record in data:
                output = {k: v for k, v in record.items() if k not in {"u", "g", "ratio"}}
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
    print(json.dumps({"history_views": len(contexts), "query_views": len(records),
                      "equivalence_max_abs": max_error, "model_calls": 0, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
