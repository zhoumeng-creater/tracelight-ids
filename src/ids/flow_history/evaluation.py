"""Frozen operating point, historical SYN cohort and within-record comparisons."""
import pandas as pd

from ids.context_cache.lab_data import FEATURES
from ids.route_batch.metrics import decisions, evaluate, paired, threshold


def syn_cohort(frame):
    # The 47-field pattern was identified before this experiment, in round 16.
    fixed = {key: 0 for key in FEATURES if key != "L4_SRC_PORT"}
    fixed.update(L4_DST_PORT=443, PROTOCOL=6, L7_PROTO=91, IN_BYTES=44, IN_PKTS=1,
                 FLOW_DURATION_MILLISECONDS=1, TCP_FLAGS=2, CLIENT_TCP_FLAGS=2,
                 LONGEST_FLOW_PKT=44, SHORTEST_FLOW_PKT=44, MAX_IP_PKT_LEN=44,
                 SRC_TO_DST_AVG_THROUGHPUT=352000, NUM_PKTS_UP_TO_128_BYTES=1, TCP_WIN_MAX_IN=1024)
    return frame[list(fixed)].eq(pd.Series(fixed)).all(axis=1).to_numpy()


def measure(cal, ev, cal_score, score):
    if len(cal) != len(cal_score) or not cal.label.eq(0).all() or not len(cal):
        raise ValueError("Threshold requires aligned normal-only calibration records")
    reg = threshold(cal_score)
    if reg["threshold"] is None:
        raise ValueError("Too few calibration records for a finite registered 1% threshold")
    pred = decisions(score, reg)
    y = ev.label.to_numpy()
    scan = ev.attack_family.eq("Scanning").to_numpy()
    mask = syn_cohort(ev)
    result = evaluate(y, score, reg, ev.attack_family)
    scan_or_normal = scan | (y == 0)
    scan_normal = evaluate(y[scan_or_normal], score[scan_or_normal], reg)
    sub = mask & scan_or_normal
    result.update(threshold=reg, scan_tp=int(pred[scan].sum()), scan_n=int(scan.sum()),
                  scan_normal_pauc=scan_normal["normalized_pauc_0_01"],
                  syn_scan_n=int((mask & scan).sum()), syn_scan_tp=int(pred[mask & scan].sum()),
                  syn_normal_n=int((mask & (y == 0)).sum()), syn_fp=int(pred[mask & (y == 0)].sum()),
                  syn_scan_normal_auc=evaluate(y[sub], score[sub], reg)["auc"])
    rows = ev[["row_id", "label", "attack_family", "communication_group"]].copy()
    rows["score"], rows["alert"], rows["syn47"] = score, pred, mask
    for key in ("host_key", "end_day_utc", "hist_flows_60s"):
        if key in ev:
            rows[key] = ev[key]
    return result, rows


def paired_comparison(left, right, base, candidate):
    identity = ["row_id", "label", "attack_family", "communication_group", "syn47"]
    if not left[identity].reset_index(drop=True).equals(right[identity].reset_index(drop=True)):
        raise ValueError("Paired comparison has different records")
    result = paired(left.label, left.alert, right.alert)
    for name in ("tp", "fp", "scan_tp", "syn_scan_tp", "syn_fp"):
        result["delta_" + name] = candidate[name] - base[name]
    eligible = candidate["scan_n"] > 0 and candidate["normal_rows"] > 0
    result["within_1pct_observed_FPR"] = candidate["fp"] * 100 <= candidate["normal_rows"]
    result["passes_detection_comparison"] = bool(
        eligible and result["within_1pct_observed_FPR"] and candidate["fp"] <= base["fp"]
        and candidate["tp"] >= base["tp"] and candidate["scan_tp"] > base["scan_tp"]
        and candidate["scan_normal_pauc"] > base["scan_normal_pauc"])
    result["syn_subgroup_improved_at_point"] = (
        bool(candidate["syn_scan_tp"] > base["syn_scan_tp"] and candidate["syn_fp"] <= base["syn_fp"])
        if candidate["syn_scan_n"] and candidate["syn_normal_n"] else None)
    return result


def stratify(rows):
    out = []
    scan = rows.attack_family.eq("Scanning")
    for key in ("end_day_utc", "host_key"):
        if key not in rows:
            continue
        for value, group in rows.groupby(key, sort=True):
            n = group.label.eq(0)
            s = scan.loc[group.index]
            out.append({"by": key, "value": str(value), "rows": len(group), "normal": int(n.sum()),
                        "fp": int(group.alert[n].sum()), "tp": int(group.alert[~n].sum()),
                        "scan": int(s.sum()), "scan_tp": int(group.alert[s].sum())})
    return out
