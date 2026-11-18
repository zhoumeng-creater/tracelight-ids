"""Independent cached-array/decision audit; never imports a model or training module."""
from pathlib import Path
from fractions import Fraction
from itertools import combinations
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser(description="Audit cached return arrays and decisions; no model execution.")
parser.add_argument("--thesis-root", type=Path,
                    default=Path(r"E:\OneDrive - Coventry University\中传\毕业论文"))
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
arguments = parser.parse_args()
OUT = arguments.output.resolve()
THESIS = arguments.thesis_root.resolve()
NEW = THESIS / "实验归档/20261118_正常子群训练结果/回传"
OLD = THESIS / "实验归档/20261117_第三十轮模型训练重设计/回传"
IDS = ["source_id", "row_id"]
SEEDS = (17, 29, 43)
ARMS = ("transport_target", "identity_target")
LEDGER = {}


def track(path):
    p = Path(path)
    b = p.read_bytes()
    LEDGER[str(p.relative_to(THESIS))] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    return p


def read(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def table(path):
    f = pd.read_parquet(track(path))
    assert not f.duplicated(IDS).any(), str(path)
    return f.set_index(IDS, drop=False).sort_index()


def bitdiff(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta = np.abs(a - b)
    return {"rows": len(a), "bitwise_mismatch_n": int(np.sum(a.view(np.uint64) != b.view(np.uint64))),
            "numeric_mismatch_n": int(np.sum(a != b)), "max_absolute_difference": float(delta.max()),
            "median_absolute_difference": float(np.median(delta))}


def counts(f, alert):
    y = f.label.to_numpy()
    normal, positive = y == 0, y == 1
    return {"normal_n": int(normal.sum()), "attack_n": int(positive.sum()),
            "fp": int((normal & alert).sum()), "tp": int((positive & alert).sum()),
            "fn": int((positive & ~alert).sum()), "budget": int(normal.sum()) // 100}


def exact_tail(scores, groups):
    scores, groups = np.asarray(scores), np.asarray(groups)
    unique = sorted(set(groups))
    result = {}
    for name, q in (("0.90", Fraction(9, 10)), ("0.95", Fraction(19, 20)), ("0.99", Fraction(99, 100))):
        mass = len(scores) * (1 - q)
        boundary = np.sort(scores)[len(scores) - math.ceil(mass)]
        above, tied = scores > boundary, scores == boundary
        share = (mass - int(above.sum())) / int(tied.sum())
        excess = []
        for group in unique:
            mask = groups == group
            group_mass = int((mask & above).sum()) + share * int((mask & tied).sum())
            excess.append(max(Fraction(0), group_mass / int(mask.sum()) - (1 - q)))
        result[name] = max(excess)
    return {"H": max(result.values()), "H_q": result}


def fraction(value):
    return Fraction(value["numerator"], value["denominator"])


def encode(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return value


freeze = read(NEW / "freeze.json")
comparison = read(NEW / "evaluation/comparison.json")
outcome = read(NEW / "outcome.json")
old_protocol = read(OLD / "protocol.json")["design"]
old_freeze = read(OLD / "freeze.json")
E = table(NEW / "evaluation/E.parquet")
C = table(OLD / "inputs/C.parquet")
EB, ED = table(OLD / "inputs/E_B.parquet"), table(OLD / "inputs/E_D.parquet")
expected_E = pd.concat([EB, ED]).sort_index()
assert E.index.equals(expected_E.index)
for column in expected_E.columns:
    assert E[column].equals(expected_E[column]), f"E metadata mismatch: {column}"
assert len(C) == 5511 and C.label.eq(0).all()
cells = {c["source_id"]: (role, c) for role in ("E_B", "E_D")
         for c in old_protocol["roles"][role]["cells"]}
assert len(cells) == 15 and set(E.source_id) == set(cells)
for source, (role, spec) in cells.items():
    f = E.loc[E.source_id.eq(source)]
    assert int(f.label.eq(0).sum()) == spec["normal"]
    assert int(f.label.eq(1).sum()) == spec["attack"]

fv = {role: table(NEW / f"inputs/{role}.parquet").reset_index(drop=True) for role in ("F", "V")}
all_models, cell_rows, breakdown_rows, error_ids, fp_sets, mechanism = {}, [], [], [], {}, {}
for seed in SEEDS:
    old_id = f"protocol_blind_s{seed}"
    old_model = track(OLD / old_freeze["artifacts"][old_id]["path"])
    assert hashlib.sha256(old_model.read_bytes()).hexdigest() == old_freeze["artifacts"][old_id]["sha256"]
    old_C = table(OLD / f"evidence/{old_id}/C-scores.parquet")
    old_E = table(OLD / f"evidence/{old_id}/ED-scores.parquet")
    old_decisions = table(OLD / f"reports/{old_id}-decisions.parquet")
    old_cal = read(OLD / f"evidence/{old_id}/calibration.json")
    assert old_C.index.equals(C.index) and old_E.index.equals(E.index) and old_decisions.index.equals(E.index)
    old_boundary = np.sort(old_C.score.to_numpy())[5456]
    assert old_boundary == old_cal["threshold"]
    old_alert = old_E.score.to_numpy() > old_boundary
    assert np.array_equal(old_alert, old_decisions.alert.to_numpy())
    assert np.array_equal(old_E.score.to_numpy(), old_decisions.score.to_numpy())
    ref_path = track(NEW / f"preflight/{old_id}/reference_scores.npz")
    with np.load(ref_path, allow_pickle=False) as ref:
        reference = {role: ref[role].copy() for role in ("F", "V")}
        vg = ref["V_group_ids"].copy()
    selected_by_arm = {}
    for arm in ARMS:
        name = f"{arm}_s{seed}"
        frozen = freeze["models"][name]
        trajectory = NEW / "models" / name / "trajectory"
        new_model = track(NEW / "models" / frozen["artifact"]["path"])
        model_bytes_equal = new_model.read_bytes() == old_model.read_bytes()
        assert model_bytes_equal and hashlib.sha256(new_model.read_bytes()).hexdigest() == frozen["artifact"]["sha256"]
        assert frozen["selected_step"] == 0
        identities = read(trajectory / "score_identities.json")
        for role in ("F", "V"):
            assert identities[role] == fv[role][IDS].values.tolist()
        with np.load(track(trajectory / "selected_scores.npz"), allow_pickle=False) as saved:
            selected = {role: saved[role].copy() for role in ("F", "V")}
        selected_reference_diff = {role: bitdiff(selected[role], reference[role]) for role in ("F", "V")}
        selected_meta = read(trajectory / "selected_metrics.json")
        baseline_meta = read(trajectory / "baseline_metrics.json")
        assert selected_meta == baseline_meta
        fy, vy = fv["F"].label.to_numpy(), fv["V"].label.to_numpy()
        invariants = {}
        for role, y in (("F", fy), ("V", vy)):
            normal = np.sort(selected[role][y == 0])
            inv = len(normal) - np.searchsorted(normal, selected[role][y == 1], side="left")
            ref_normal = np.sort(reference[role][y == 0])
            ref_inv = len(ref_normal) - np.searchsorted(ref_normal, reference[role][y == 1], side="left")
            invariants[f"{role}_positive_inversion_increased_n"] = int((inv > ref_inv).sum())
            invariants[f"{role}_positive_inversion_total"] = int(inv.sum())
        tails = {"pooled": exact_tail(selected["V"][vy == 0], vg[vy == 0])}
        reference_tails = {"pooled": exact_tail(reference["V"][vy == 0], vg[vy == 0])}
        for profile in ("A", "B"):
            mask = (vy == 0) & fv["V"].profile.eq(profile).to_numpy()
            tails[profile] = exact_tail(selected["V"][mask], vg[mask])
            reference_tails[profile] = exact_tail(reference["V"][mask], vg[mask])
        assert tails["pooled"]["H"] == fraction(selected_meta["tail"]["H"])
        for q in tails["pooled"]["H_q"]:
            assert tails["pooled"]["H_q"][q] == fraction(selected_meta["tail"]["H_q"][q])
            for profile in ("A", "B"):
                assert tails[profile]["H_q"][q] == fraction(selected_meta["source_tails"][profile]["H_q"][q])
        panel_checks = {}
        for panel, profile, protocol in (("A_TCP", "A", 6), ("B_UDP", "B", 17), ("pooled", None, None)):
            mask = np.ones(len(vy), dtype=bool) if profile is None else (
                fv["V"].profile.eq(profile) & fv["V"].ip_protocol.eq(protocol)).to_numpy()
            y, s = vy[mask], selected["V"][mask]
            panel_checks[panel] = {"pauc": float(roc_auc_score(y, s, max_fpr=.01)),
                                  "reference_pauc": float(roc_auc_score(y, reference["V"][mask], max_fpr=.01)),
                                  "strict_separation": bool(s[y == 1].min() > s[y == 0].max())}
        new_tie_n = np.unique(selected["V"][vy == 0], return_counts=True)[1]
        ref_tie_n = np.unique(reference["V"][vy == 0], return_counts=True)[1]
        guard_checks = {
            "F_each_positive_inversions_not_increased": invariants["F_positive_inversion_increased_n"] == 0,
            "V_each_positive_inversions_not_increased": invariants["V_positive_inversion_increased_n"] == 0,
            "V_panel_strict_separation": all(panel_checks[p]["strict_separation"] for p in ("A_TCP", "B_UDP")),
            "V_pooled_pauc_not_lower": panel_checks["pooled"]["pauc"] >= panel_checks["pooled"]["reference_pauc"],
            "V_panel_pauc_not_lower": all(panel_checks[p]["pauc"] >= panel_checks[p]["reference_pauc"] for p in ("A_TCP", "B_UDP")),
            "normal_max_tie_block_not_increased": int(new_tie_n.max()) <= int(ref_tie_n.max()),
            "normal_tied_pairs_not_increased": int((new_tie_n*(new_tie_n-1)//2).sum()) <= int((ref_tie_n*(ref_tie_n-1)//2).sum()),
            "each_H_q_not_increased": all(tails["pooled"]["H_q"][q] <= reference_tails["pooled"]["H_q"][q] for q in tails["pooled"]["H_q"]),
            "each_source_H_q_not_increased": all(tails[p]["H_q"][q] <= reference_tails[p]["H_q"][q] for p in ("A", "B") for q in tails[p]["H_q"])}
        new_C = table(NEW / f"evaluation/{name}/C-scores.parquet")
        new_E = table(NEW / f"evaluation/{name}/E-scores.parquet")
        decisions = table(NEW / f"evaluation/{name}/decisions.parquet")
        calibration = read(NEW / f"evaluation/{name}/calibration.json")
        assert new_C.index.equals(C.index) and new_E.index.equals(E.index) and decisions.index.equals(E.index)
        for column in ("label", "family", "profile", "ip_protocol", "duration_seconds", "flow_end_ms"):
            assert np.array_equal(decisions[column], E[column]), (name, column)
        rank = (99 * (len(new_C) + 1) + 99) // 100
        assert rank == 5457
        threshold = float(np.sort(new_C.score.to_numpy())[rank - 1])
        assert threshold == calibration["threshold"] and calibration["rank"] == rank
        assert calibration["comparison"] == ">" and decisions.threshold.eq(threshold).all()
        assert decisions.ready.all() and calibration["ready"]
        assert calibration["n_normal"] == 5511 and calibration["alpha"] == .01
        assert calibration["calibration_fp"] == int((new_C.score > threshold).sum())
        alert = new_E.score.to_numpy() > threshold
        assert np.array_equal(alert, decisions.alert.to_numpy())
        assert np.array_equal(new_E.score, decisions.score)
        y = E.label.to_numpy()
        lost = (y == 1) & old_alert & ~alert
        gained = (y == 1) & ~old_alert & alert
        model_cells = []
        for source, (role, spec) in cells.items():
            mask = E.source_id.eq(source).to_numpy()
            c = counts(E.loc[mask], alert[mask])
            row = {"model_id": name, "seed": seed, "arm": arm, "role": role, "source_id": source,
                   **c, "within_budget": c["fp"] <= c["budget"], "excess_X": max(c["fp"] - c["budget"], 0)}
            model_cells.append(row)
            cell_rows.append(row)
        role_counts = {role: counts(E.loc[E.source_id.isin([s for s, (r, _) in cells.items() if r == role])],
                                    alert[E.source_id.isin([s for s, (r, _) in cells.items() if r == role]).to_numpy()])
                       for role in ("E_B", "E_D")}
        report = read(NEW / f"evaluation/{name}/report.json")
        total = counts(E, alert)
        excess = sum(c["excess_X"] for c in model_cells)
        qualified = sum(c["within_budget"] for c in model_cells)
        total["pooled_floor_diagnostic_not_adoption_budget"] = total.pop("budget")
        total["cell_budget_sum"] = sum(c["budget"] for c in model_cells)
        for role, summary in role_counts.items():
            summary["pooled_floor_diagnostic_not_adoption_budget"] = summary.pop("budget")
            summary["cell_budget_sum"] = sum(c["budget"] for c in model_cells if c["role"] == role)
        assert report["excess_X"] == excess
        for row in report["cells"]:
            ours = next(c for c in model_cells if c["source_id"] == row["source_id"])
            for key in ("normal_n", "attack_n", "tp", "fp", "fn"):
                assert row[key] == ours[key], (name, key)
            assert row["qualified"] == ours["within_budget"]
        fp = (y == 0) & alert
        if arm == "transport_target":
            fp_sets[seed] = set(E.index[fp])
            for (source, protocol, family), part in decisions.reset_index(drop=True).groupby(["source_id", "ip_protocol", "family"], sort=True):
                c = counts(part, part.alert.to_numpy())
                breakdown_rows.append({"seed": seed, "source_id": source, "role": cells[source][0],
                                       "ip_protocol": int(protocol), "family": family, **c})
        for identity in E.index[fp]:
            row = E.loc[identity]
            error_ids.append({"model_id": name, "seed": seed, "arm": arm, **{k: row[k] for k in
                (*IDS, "profile", "ip_protocol", "family", "duration_seconds", "orig_pkts", "resp_pkts",
                 "orig_ip_bytes", "resp_ip_bytes")}, "score": float(new_E.loc[identity, "score"]),
                 "threshold": threshold, "inherited_R30_FP": bool(old_alert[E.index.get_loc(identity)])})
        all_models[name] = {"model_bytes_equal_R30": model_bytes_equal,
            "model_sha256": hashlib.sha256(new_model.read_bytes()).hexdigest(), "selected_step": 0,
            "C_score_comparison_R30": bitdiff(new_C.score, old_C.score),
            "E_score_comparison_R30": bitdiff(new_E.score, old_E.score),
            "E_decision_mismatch_R30": int((alert != old_alert).sum()),
            "calibration": {"normal_n": len(new_C), "rank": rank, "threshold": threshold,
                            "R30_threshold": old_boundary, "C_FP": int((new_C.score > threshold).sum()),
                            "strict_decision_mismatch_n": int((alert != decisions.alert).sum())},
            "total": total, "roles": role_counts, "qualified_cells": qualified, "excess_X": excess,
            "R30_TP_lost": int(lost.sum()), "R30_TP_added": int(gained.sum()),
            "R30_FP_added": int((fp & ~old_alert).sum()), "R30_FP_removed": int(((y == 0) & old_alert & ~alert).sum()),
            "selected_FV_reference_bitdiff": selected_reference_diff,
            "selected_equals_baseline_metrics": selected_meta == baseline_meta,
            "recomputed_V_tails": tails, "inversions": invariants, "panels": panel_checks,
            "recomputed_guard_checks": guard_checks}
        selected_by_arm[arm] = {"alert": alert, "C": new_C.score.to_numpy(), "E": new_E.score.to_numpy(),
                               "tails": tails, "reference_tails": reference_tails, "guard_checks": guard_checks,
                               "selected_step": frozen["selected_step"], "excess": excess,
                               "development_pass": qualified == 15 and not lost.any()
                               and role_counts["E_B"]["tp"] == 72 and role_counts["E_D"]["tp"] == 1964}
    a, b = selected_by_arm["transport_target"], selected_by_arm["identity_target"]
    recomputed_checks = {"nonfallback_transport": a["selected_step"] > 0,
        "transport_guards_pass": all(a["guard_checks"].values()),
        "control_guards_pass": all(b["guard_checks"].values()),
        "H_strictly_below_R30": a["tails"]["pooled"]["H"] < a["reference_tails"]["pooled"]["H"],
        "H_strictly_below_control": a["tails"]["pooled"]["H"] < b["tails"]["pooled"]["H"],
        "each_H_q_not_above_control": all(a["tails"]["pooled"]["H_q"][q] <= b["tails"]["pooled"]["H_q"][q]
                                         for q in a["tails"]["pooled"]["H_q"]),
        "each_source_H_q_not_above_control": all(a["tails"][p]["H_q"][q] <= b["tails"][p]["H_q"][q]
                                              for p in ("A", "B") for q in a["tails"][p]["H_q"])}
    assert recomputed_checks == comparison["seeds"][str(seed)]["mechanism"]["checks"]
    passed = all(recomputed_checks.values())
    strict_excess = a["excess"] == 0 and a["excess"] < b["excess"]
    adopted = passed and a["development_pass"] and strict_excess
    control_valid = a["excess"] == b["excess"] == 0 and b["development_pass"]
    declared = comparison["seeds"][str(seed)]
    assert declared["mechanism"]["passed"] == passed
    assert (declared["X_transport"], declared["X_control"]) == (a["excess"], b["excess"])
    assert declared["strict_excess_advantage"] == strict_excess
    assert declared["transport_development_pass"] == a["development_pass"]
    assert (declared["transport_adoption_pass"], declared["identity_control_valid_development_result"]) == (adopted, control_valid)
    mechanism[str(seed)] = {"checks": recomputed_checks, "mechanism_pass": passed,
        "X_transport": a["excess"], "X_control": b["excess"], "strict_excess_advantage": strict_excess,
        "adoption_pass": adopted, "identity_control_valid_development_result": control_valid,
        "transport_control_alert_mismatch_n": int((a["alert"] != b["alert"]).sum()),
        "transport_control_C_score_diff": bitdiff(a["C"], b["C"]),
        "transport_control_E_score_diff": bitdiff(a["E"], b["E"])}

intersection = set.intersection(*fp_sets.values())
union = set.union(*fp_sets.values())
cross_rows = []
for identity in sorted(union):
    row = E.loc[identity]
    membership = [seed for seed in SEEDS if identity in fp_sets[seed]]
    cross_rows.append({**{k: row[k] for k in (*IDS, "profile", "ip_protocol", "family", "duration_seconds",
                                          "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")},
                      "seed_count": len(membership), "seeds": ",".join(map(str, membership)),
                      **{f"FP_seed{seed}": identity in fp_sets[seed] for seed in SEEDS}})
cross = {"common_all_three": len(intersection), "union": len(union),
         "unique_by_seed": {str(seed): len(fp_sets[seed] - set.union(*(fp_sets[s] for s in SEEDS if s != seed)))
                            for seed in SEEDS},
         "pairwise_intersection": {f"{a}_{b}": len(fp_sets[a] & fp_sets[b]) for a, b in combinations(SEEDS, 2)}}
assert comparison["adoption_pass"] is False and outcome["adoption_pass"] is False
assert comparison["all_seed_mechanism_pass"] == all(row["mechanism_pass"] for row in mechanism.values())
assert comparison["all_seed_identity_control_valid"] == all(row["identity_control_valid_development_result"] for row in mechanism.values())
assert comparison["retained_development_arm"] == outcome["retained_development_arm"] == "R30"
assert outcome["effect_comparison"] == comparison
report = {"status": "cached_effect_independently_verified", "new_model_loads": 0, "new_model_forwards": 0,
          "new_fits": 0, "new_threshold_searches": 0, "fixed_calibration_recomputation_only": True,
          "new_E_metadata_equals_original_E": True, "models": all_models, "seeds": mechanism,
          "cross_seed_inherited_false_positives": cross,
          "comparison_outcome_recomputed_match": True, "adoption_pass": False,
          "retained_development_arm": "R30", "control_final_output_increment": False,
          "original_package_claim_errors_found": [],
          "actual_adoption_cell_budget_sum": {"E_B": 54, "E_D": 89, "all_15_cells": 143},
          "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "interpretation": "All six final artifacts are byte-identical R30 fallbacks. Final TP/FP are inherited capability and errors, not newly learned improvements.",
          "input_ledger": LEDGER}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "effect_audit.json").write_text(json.dumps(encode(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")
pd.DataFrame(cell_rows).to_csv(OUT / "effect_audit.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(breakdown_rows).to_csv(OUT / "effect_audit_protocol_family.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(error_ids).to_csv(OUT / "effect_audit_fp_identities.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cross_rows).to_csv(OUT / "effect_audit_cross_seed_fp.csv", index=False, encoding="utf-8-sig")
print(json.dumps(encode({"models": {k: {j: v[j] for j in ("total", "roles", "qualified_cells", "excess_X",
        "C_score_comparison_R30", "E_score_comparison_R30", "E_decision_mismatch_R30")} for k,v in all_models.items()},
        "cross_seed":cross}), ensure_ascii=True))
