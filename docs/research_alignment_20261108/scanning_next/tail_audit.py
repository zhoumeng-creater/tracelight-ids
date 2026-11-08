"""Read saved round-15 scores and raw CSVs; no model, fit, or new representation.

All target recall boundaries below describe already-opened development data.
They are not registered thresholds and must not be used as deployment choices.
Only the saved source-port mapping is read; no real-data mapping is constructed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARMS = ("baseline", "src_frequency_bijection", "src_mask_zero")
METADATA = {"label", "sample_id", "communication_group", "attack_family"}
SERVICE = ["PROTOCOL", "L7_PROTO", "L4_DST_PORT"]
RAW_PATTERN = SERVICE + ["IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def distribution(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"rows": 0}
    return {
        "rows": int(len(arr)), "min": float(arr.min()), "median": float(np.median(arr)),
        "max": float(arr.max()), "distinct": int(np.unique(arr).size),
    }


def count_dict(values) -> dict:
    return {str(key): int(count) for key, count in Counter(values).most_common()}


def signatures(frame: pd.DataFrame, fields: list[str]) -> list[tuple]:
    # Raw-value equality audit, not model inputs. Decimal equates 0 and 0.0.
    return [tuple(Decimal(str(value)) for value in row) for row in frame[fields].itertuples(index=False, name=None)]


def min_score_only_cost(target_scores, normal_c, normal_e, required: int) -> dict:
    target_scores = np.asarray(target_scores, dtype=np.float64)
    boundary = float(np.sort(target_scores)[::-1][required - 1])
    return {
        "required_targets": int(required), "target_rows": int(len(target_scores)),
        "score_boundary_inclusive": boundary,
        "explanation": "Any strict > threshold detecting at least required_targets must be below this score; all ties are included.",
        "targets_detected_with_ties": int(np.count_nonzero(target_scores >= boundary)),
        "target_boundary_ties": int(np.count_nonzero(target_scores == boundary)),
        "C_normal_at_or_above": int(np.count_nonzero(normal_c >= boundary)),
        "C_normal_rate": float(np.mean(normal_c >= boundary)),
        "E_normal_at_or_above": int(np.count_nonzero(normal_e >= boundary)),
        "E_normal_rate": float(np.mean(normal_e >= boundary)),
    }


def auc_pairwise(scan_scores, normal_scores) -> float:
    # Direct rank comparison, half credit for ties; no estimator or classifier.
    scan = np.asarray(scan_scores, dtype=np.float64)
    normal = np.sort(np.asarray(normal_scores, dtype=np.float64))
    below = np.searchsorted(normal, scan, side="left")
    equal = np.searchsorted(normal, scan, side="right") - below
    return float(np.sum(below + 0.5 * equal) / (len(scan) * len(normal)))


def top_patterns(frame: pd.DataFrame, fields: list[str], limit: int = 8) -> list[dict]:
    counts = frame.groupby(fields, dropna=False, sort=False).size().sort_values(ascending=False)
    return [
        {**{field: scalar(value) for field, value in zip(fields, key)}, "rows": int(count)}
        for key, count in counts.head(limit).items()
    ]


def normal_band(frame: pd.DataFrame, scores: np.ndarray, mask: np.ndarray) -> dict:
    selected = frame.loc[mask]
    groups = selected.communication_group.value_counts()
    return {
        "rows": int(mask.sum()), "score": distribution(scores[mask]),
        "communication_groups": int(len(groups)),
        "largest_group_rows": int(groups.max()) if len(groups) else 0,
        "groups_with_multiple_rows": int((groups > 1).sum()),
        "top_services": top_patterns(selected, SERVICE),
        "top_raw_patterns": top_patterns(selected, RAW_PATTERN),
    }


def synthetic_checks() -> dict:
    # Ties must increase target and normal counts at the target boundary.
    item = min_score_only_cost([0.6, 0.6, 1.0], np.array([0.1, 0.6, 0.6, 0.9]), np.array([0.6]), 2)
    assert item["targets_detected_with_ties"] == 3
    assert item["C_normal_at_or_above"] == 3
    assert item["E_normal_at_or_above"] == 1
    # A strictly increasing common transform preserves the strict comparison,
    # including a score equal to the saved threshold.
    scores = np.array([-1.0, 0.25, 0.5, 0.5, 1.0])
    threshold = 0.5
    assert np.array_equal(scores > threshold, (3 * scores + 7) > (3 * threshold + 7))
    assert auc_pairwise([0.0, 1.0], [0.0, 1.0]) == 0.5
    return {"assertions_passed": 5, "data_kind": "synthetic arrays only"}


def run(project: Path) -> dict:
    comparison = project / "实验归档/20261108_第十五轮Scanning机制/核验材料/context-scanning-v1/comparison"
    prepared = project / "实验归档/20261108_第十四轮输入修复/核验材料/context-input-v2/prepared"
    files = [prepared / name for name in ("support.csv", "normal_calibration.csv", "development_evaluation.csv")]
    files += [comparison / name for name in ("evaluation_scores.csv", "evaluation_cohorts.csv", "mapping_audit.json", "input_collisions.json", "shared/source_port_mapping.npy")]
    files += [comparison / f"{arm}_calibration.csv" for arm in ARMS]
    files += [comparison / arm / "threshold.json" for arm in ARMS]
    inputs = {str(path.relative_to(project)): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files}
    p, c, e = [pd.read_csv(prepared / name, float_precision="round_trip") for name in ("support.csv", "normal_calibration.csv", "development_evaluation.csv")]
    scores = pd.read_csv(comparison / "evaluation_scores.csv", float_precision="round_trip")
    cohorts = pd.read_csv(comparison / "evaluation_cohorts.csv")
    assert e.sample_id.equals(scores.row_id) and e.sample_id.equals(cohorts.row_id)
    assert e.label.equals(scores.label) and e.attack_family.equals(scores.attack_family)
    assert len(p) == 8192 and len(c) == 5000 and len(e) == 20000
    assert c.label.eq(0).all()
    for key in cohorts.columns[1:]:
        assert cohorts[key].dtype == bool
    fields = [field for field in p.columns if field not in METADATA]
    assert len(fields) == 48
    fields47 = [field for field in fields if field != "L4_SRC_PORT"]
    sig_p, sig_c, sig_e = [signatures(frame, fields47) for frame in (p, c, e)]
    dominant = sig_e[int(np.flatnonzero(cohorts.dominant_47_field_pattern)[0])]
    main_p = np.array([sig == dominant for sig in sig_p])
    main_c = np.array([sig == dominant for sig in sig_c])
    main_e = np.array([sig == dominant for sig in sig_e])
    assert np.array_equal(main_e, cohorts.dominant_47_field_pattern)
    scan = e.attack_family.eq("Scanning").to_numpy()
    normal = e.label.eq(0).to_numpy()
    target_service = np.array(cohorts.TCP6_L791_DST443)
    expected_service = e.PROTOCOL.eq(6) & e.L7_PROTO.eq(91) & e.L4_DST_PORT.eq(443)
    assert np.array_equal(target_service, expected_service)

    out = {
        "scope": "opened-development saved-score diagnosis; no model load, fit, new real representation, mapping construction, or threshold registration",
        "score_comparison": "float64 strict >; round_trip CSV parsing; ties included in descriptive required-recall counts",
        "input_files": inputs, "synthetic_checks": synthetic_checks(),
        "rows": {"P": len(p), "C_normal": len(c), "E": len(e), "E_normal": int(normal.sum()), "E_scanning": int(scan.sum())},
        "arms": {},
    }
    c_scores_by_arm = {}
    target_masks = {
        "Scanning": scan, "old_54": np.array(cohorts.old_54_lost),
        "exact_P_positive_15": np.array(cohorts["15_exact_P_positive_matches"]),
        "main47_scanning": main_e & scan,
    }
    assert target_masks["old_54"].sum() == 54 and target_masks["exact_P_positive_15"].sum() == 15
    for arm in ARMS:
        cal = pd.read_csv(comparison / f"{arm}_calibration.csv", float_precision="round_trip")
        threshold = json.loads((comparison / arm / "threshold.json").read_text(encoding="utf-8"))
        assert c.sample_id.equals(cal.row_id) and c.label.equals(cal.label)
        cs = cal.score.to_numpy(dtype=np.float64)
        es = scores[f"{arm}_score"].to_numpy(dtype=np.float64)
        t = threshold["threshold"]
        assert np.array_equal(cs > t, cal.predicted_attack.to_numpy(dtype=bool))
        assert np.array_equal(es > t, scores[f"{arm}_predicted"].to_numpy(dtype=bool))
        c_scores_by_arm[arm] = cs
        cohort_costs = {}
        for key, mask in target_masks.items():
            target = es[mask]
            requested = {"at_least_1": 1, **{f"at_least_{int(rate * 100)}pct": int(np.ceil(rate * mask.sum())) for rate in (0.25, 0.5, 1.0)}}
            cohort_costs[key] = {name: min_score_only_cost(target, cs, es[normal], k) for name, k in requested.items()}
        local = {}
        for key, mask in (("target_service_6_91_443", target_service), ("main47", main_e)):
            ss, ns = es[mask & scan], es[mask & normal]
            cost = {}
            for rate in (0.25, 0.5, 1.0):
                required = int(np.ceil(rate * len(ss)))
                boundary = np.sort(ss)[::-1][required - 1]
                cost[str(rate)] = {"required_scanning": required, "scanning_with_ties": int((ss >= boundary).sum()), "local_normals_at_or_above": int((ns >= boundary).sum()), "local_normal_denominator": len(ns)}
            local[key] = {"Scanning": distribution(ss), "normal": distribution(ns), "Scanning_vs_normal_AUC": auc_pairwise(ss, ns), "score_only_cost": cost}
        out["arms"][arm] = {
            "threshold": t, "strict_C_fp": int((cs > t).sum()),
            "strict_E_fp": int(((es > t) & normal).sum()), "strict_E_tp": int(((es > t) & ~normal).sum()),
            "strict_Scanning_tp": int(((es > t) & scan).sum()),
            "Scanning_vs_all_normal_AUC": auc_pairwise(es[scan], es[normal]),
            "required_recall_normal_cost": cohort_costs, "local_ranking": local,
        }

    # Diagnose the raw-field and communication-group composition of normal tails.
    es = scores.baseline_score.to_numpy(dtype=np.float64)
    cs = c_scores_by_arm["baseline"]
    threshold = out["arms"]["baseline"]["threshold"]
    band_ranges = {
        "current_alerts_strict": (threshold, np.inf, False),
        "normal_between_old54_min_max_inclusive": (float(es[target_masks["old_54"]].min()), float(es[target_masks["old_54"]].max()), True),
        "normal_between_main47_scan_min_max_inclusive": (float(es[main_e & scan].min()), float(es[main_e & scan].max()), True),
    }
    bands = {}
    for name, (low, high, inclusive) in band_ranges.items():
        c_mask = ((cs >= low) if inclusive else (cs > low)) & (cs <= high)
        e_mask = normal & ((es >= low) if inclusive else (es > low)) & (es <= high)
        bands[name] = {"score_lower": low, "score_upper": high if np.isfinite(high) else None, "lower_inclusive": inclusive, "C": normal_band(c, cs, c_mask), "E": normal_band(e, es, e_mask)}
    out["baseline_normal_bands"] = bands

    support_counts = Counter((sig, int(label)) for sig, label in zip(sig_p, p.label))
    support_links = {}
    for role, raw, sigs, vals in (("C", c, sig_c, cs), ("E", e, sig_e, es)):
        mask = raw.label.eq(0).to_numpy() & (vals > threshold)
        n0 = [support_counts[sig, 0] for sig, keep in zip(sigs, mask) if keep]
        n1 = [support_counts[sig, 1] for sig, keep in zip(sigs, mask) if keep]
        support_links[role] = {"alert_normal_rows": int(mask.sum()), "same_raw47_P_normal_count": distribution(n0), "same_raw47_P_attack_count": distribution(n1), "rows_with_P_normal_match": int(np.count_nonzero(np.array(n0) > 0)), "rows_with_P_attack_match": int(np.count_nonzero(np.array(n1) > 0)), "rows_with_both": int(np.count_nonzero((np.array(n0) > 0) & (np.array(n1) > 0)))}
    out["high_normal_raw47_P_overlap"] = support_links

    local_pattern = {field: str(value) for field, value in zip(fields47, dominant) if value != 0}
    main = {"nonzero_raw47_fields": local_pattern, "zero_fields": sum(value == 0 for value in dominant), "roles": {}}
    for role, raw, mask in (("P", p, main_p), ("C", c, main_c), ("E", e, main_e)):
        families = count_dict(raw.loc[mask, "attack_family"])
        labels_per_port = raw.loc[mask].groupby(["L4_SRC_PORT", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        item = {
            "rows": int(mask.sum()), "families": families,
            "Scanning_share": float(raw.loc[mask, "attack_family"].eq("Scanning").mean()),
            "normal_share": float(raw.loc[mask, "label"].eq(0).mean()),
            "communication_groups": int(raw.loc[mask, "communication_group"].nunique()),
            "source_ports_by_family": {str(family): distribution(group.L4_SRC_PORT) for family, group in raw.loc[mask].groupby("attack_family")},
            "full_raw48_mixed_label_patterns_within_main47": int(((labels_per_port[0] > 0) & (labels_per_port[1] > 0)).sum()),
            "full_raw48_total_error_lower_bound_within_main47": int(labels_per_port.min(axis=1).sum()),
        }
        if role == "C":
            item["score_by_arm"] = {arm: distribution(c_scores_by_arm[arm][mask]) for arm in ARMS}
        elif role == "E":
            item["score_by_arm_and_family"] = {arm: {str(family): distribution(scores.loc[group.index, f"{arm}_score"]) for family, group in raw.loc[mask].groupby("attack_family")} for arm in ARMS}
        main["roles"][role] = item
    out["main47_support_and_query"] = main

    # Audit whether a label-granularity probe introduces input contradictions.
    # This reads raw P equality only; it does not construct proposed pseudo labels
    # or materialize a new FP32 model representation.
    full_p_groups = {}
    for sig, family in zip(signatures(p, fields), p.attack_family):
        full_p_groups.setdefault(sig, Counter())[family] += 1
    attack_groups = [Counter({family: count for family, count in group.items() if family != "Benign"}) for group in full_p_groups.values()]
    attack_groups = [group for group in attack_groups if group]
    attack_sizes = Counter(sum(group.values()) for group in attack_groups)
    attack_duplicates = [group for group in attack_groups if sum(group.values()) > 1]
    scan_other_conflicts = [group for group in attack_groups if group.get("Scanning", 0) and sum(group.values()) > group["Scanning"]]
    binary_conflicts = [group for group in full_p_groups.values() if group.get("Benign", 0) and len(group) > 1]
    pure_attack_singletons = sum(sum(group.values()) == 1 and not group.get("Benign", 0) for group in full_p_groups.values())
    collision_audit = json.loads((comparison / "input_collisions.json").read_text(encoding="utf-8"))
    out["P_label_granularity_input_consistency"] = {
        "source": "raw 48-field numeric equality only; no new label map or real FP32 representation",
        "P_unique_raw48_patterns": len(full_p_groups),
        "saved_baseline_P_unique_FP32_patterns": collision_audit["baseline"]["support"]["unique_input_patterns"],
        "T1_counts": {"normal": int(p.label.eq(0).sum()), "Scanning": int(p.attack_family.eq("Scanning").sum()), "other_attack": int((p.label.eq(1) & p.attack_family.ne("Scanning")).sum())},
        "attack_rows": int(p.label.eq(1).sum()), "attack_input_groups": len(attack_groups),
        "attack_rows_per_input_group_histogram": {str(size): count for size, count in sorted(attack_sizes.items())},
        "attack_duplicate_input_groups": len(attack_duplicates),
        "attack_rows_in_duplicate_groups": sum(sum(group.values()) for group in attack_duplicates),
        "attack_duplicate_family_compositions": count_dict(json.dumps(dict(sorted(group.items())), sort_keys=True) for group in attack_duplicates),
        "pure_attack_singleton_groups": int(pure_attack_singletons),
        "T1_Scanning_vs_other_attack_conflict_groups": len(scan_other_conflicts),
        "existing_normal_vs_attack_conflict_groups": [dict(group) for group in binary_conflicts],
        "rowwise_pseudo_split_exposure": "28 identical-input attack pairs could receive different pseudo subclasses if hashing rows independently; no actual pseudo-label allocation was performed",
        "groupwise_512_feasibility": "4039 pure attack singleton groups suffice to achieve a 512-row pseudo class without splitting any attack pair; actual mapping and FP32 group audit remain laboratory work",
    }

    coarse = []
    for key in ((6, 91, 443), (6, 7, 80), (17, 5, 53)):
        item = {"service": dict(zip(SERVICE, key)), "roles": {}}
        for role, raw in (("P", p), ("C", c), ("E", e)):
            mask = np.ones(len(raw), dtype=bool)
            for field, value in zip(SERVICE, key):
                mask &= raw[field].eq(value).to_numpy()
            entry = {"rows": int(mask.sum()), "families": count_dict(raw.loc[mask, "attack_family"]), "communication_groups": int(raw.loc[mask, "communication_group"].nunique())}
            if role != "P":
                vals = cs if role == "C" else es
                normals_here = mask & raw.label.eq(0).to_numpy()
                entry["baseline_normal_alerts"] = int((normals_here & (vals > threshold)).sum())
                entry["normal_rows"] = int(normals_here.sum())
                entry["baseline_local_normal_FPR"] = float((vals[normals_here] > threshold).mean())
            item["roles"][role] = entry
        coarse.append(item)
    out["coarse_service_support_and_query"] = coarse

    mapping = np.load(comparison / "shared/source_port_mapping.npy", allow_pickle=False)
    mapping_audit = json.loads((comparison / "mapping_audit.json").read_text(encoding="utf-8"))
    assert mapping.shape == (65536,) and mapping.dtype == np.uint16
    assert hashlib.sha256(mapping.astype("<u2", copy=False).tobytes()).hexdigest() == mapping_audit["construction"]["mapping_sha256"]
    mapping_response = {"mapping_source": "saved .npy only; no construction", "tolerance": "abs(B1-B0) > 3e-5 + 3e-5*abs(B0)", "roles": {}}
    for role, raw, b0, b1 in (
        ("C", c, cs, c_scores_by_arm["src_frequency_bijection"]),
        ("E", e, es, scores.src_frequency_bijection_score.to_numpy(dtype=np.float64)),
    ):
        ports = raw.L4_SRC_PORT.to_numpy(dtype=np.int64)
        unchanged = mapping[ports] == ports
        changed_score = np.abs(b1 - b0) > (3e-5 + 3e-5 * np.abs(b0))
        groups = {"all": np.ones(len(raw), dtype=bool), "normal": raw.label.eq(0).to_numpy()}
        if role == "E":
            groups.update(target_masks)
        item = {}
        for name, subset in groups.items():
            fixed = subset & unchanged
            item[name] = {
                "rows": int(subset.sum()), "unchanged_query_port_rows": int(fixed.sum()),
                "changed_query_port_rows": int((subset & ~unchanged).sum()),
                "score_changed_beyond_tolerance": int((subset & changed_score).sum()),
                "unchanged_query_port_score_changed": int((fixed & changed_score).sum()),
                "unchanged_query_port_score_unchanged": int((fixed & ~changed_score).sum()),
                "unchanged_query_port_abs_score_delta": distribution(np.abs(b1 - b0)[fixed]),
            }
        mapping_response["roles"][role] = item
    p_ports = p.L4_SRC_PORT.to_numpy(dtype=np.int64)
    mapping_response["P_rows_with_port_change"] = int((mapping[p_ports] != p_ports).sum())
    mapping_response["interpretation"] = "B1 jointly changes support and queries. Changed scores for unchanged full query rows establish context-side or batch-path influence, not an isolated causal effect of the query source port. Frozen preprocessing is unchanged; no claim about which internal support rows caused the response."
    out["saved_B1_mapping_response"] = mapping_response
    out["strict_monotone_calibration"] = {
        "identity": "For strictly increasing f: f(s)>f(t) iff s>t. Transforming calibration order statistic and query scores together therefore preserves all decisions, ranks and ties.",
        "scope": "common scalar strictly increasing transform; does not cover group-dependent transformations or new conditional scores",
        "group_calibration_observation": {"target_service_C_normals": 17, "main47_C_normals": 9, "alpha": 0.01, "target_service_rank": 18, "main47_rank": 10, "finite_threshold_minimum_n_at_alpha_001": 99},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tail_audit.json"))
    args = parser.parse_args()
    result = run(args.project.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "rows": result["rows"], "synthetic_checks": result["synthetic_checks"], "score_only_thresholds_registered": 0, "real_model_runs": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
