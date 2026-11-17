"""Audit frozen R30 logs and cached F/V scores. Never imports or executes models.

Run: python training_audit.py [--project PROJECT_ROOT] [--output RESULT_JSON]
Default output: training_audit.json next to this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_suffix(".json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(frame, identities):
    index = pd.MultiIndex.from_tuples([tuple(row) for row in identities], names=["source_id", "row_id"])
    source = frame.set_index(["source_id", "row_id"])
    assert source.index.is_unique and index.is_unique
    assert len(index) == len(source) and set(index) == set(source.index)
    return source.loc[index].reset_index()


def metric(scores, labels):
    return {
        "normal": int(np.sum(labels == 0)), "positive": int(np.sum(labels == 1)),
        "pauc": float(roc_auc_score(labels, scores, max_fpr=0.01)),
        "nll": float(np.logaddexp(0, np.where(labels == 1, -scores, scores)).mean()),
        "strictly_separated": bool(scores[labels == 1].min() > scores[labels == 0].max()),
    }


def metrics(scores, frame):
    panels = {}
    for name, profile, protocol in (("A_TCP", "A", 6), ("B_UDP", "B", 17)):
        mask = ((frame.profile == profile) & (frame.ip_protocol == protocol)).to_numpy()
        panels[name] = metric(scores[mask], frame.label.to_numpy()[mask])
    return {"panels": panels, "pooled": metric(scores, frame.label.to_numpy())}


def score_groups(scores, frame):
    data = frame.assign(score=scores)
    groups = {}
    for (profile, protocol, label), block in data.groupby(["profile", "ip_protocol", "label"]):
        values = block.score.to_numpy()
        groups[f"{profile}_p{protocol}_y{label}"] = {
            "rows": len(block), "min": float(values.min()), "q10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "q90": float(np.quantile(values, .9)),
            "max": float(values.max()),
            "ensemble_bce": float(np.logaddexp(0, -values if label else values).mean()),
        }
    overlaps = []
    normal = data[data.label == 0]
    positive = data[data.label == 1]
    for profile, pos in positive.groupby("profile"):
        positives = pos.score.to_numpy()
        for (normal_profile, protocol), neg in normal.groupby(["profile", "ip_protocol"]):
            negatives = neg.score.to_numpy()
            inversions = np.searchsorted(np.sort(negatives), positives, side="right")
            ties_left = np.searchsorted(np.sort(negatives), positives, side="left")
            overlaps.append({
                "positive_profile": profile, "normal_profile": normal_profile, "normal_protocol": int(protocol),
                "positives": len(positives), "normals": len(negatives),
                "min_positive_minus_max_normal": float(positives.min() - negatives.max()),
                "normal_strictly_above_min_positive": int((negatives > positives.min()).sum()),
                "positive_at_or_below_max_normal": int((positives <= negatives.max()).sum()),
                "inverted_pairs": int((len(negatives) - inversions).sum()),
                "tied_pairs": int((inversions - ties_left).sum()),
            })
    max_normal = normal.score.max()
    low = positive[positive.score <= max_normal]
    return {"groups": groups, "pairwise_ordering": overlaps,
            "positive_at_or_below_any_normal": {
                "rows": len(low), "counts_by_source": low.source_id.value_counts().sort_index().to_dict(),
                "identities": low[["source_id", "row_id"]].values.tolist(),
            }}


def replay_selection(history, patience):
    best_key = None
    best_step = None
    stale = 0
    first_strict = None
    decisions = []
    for record in history:
        v, step = record["validation"], record["step"]
        a, b = (v["panels"][x]["pauc"] for x in ("A_TCP", "B_UDP"))
        key = (min(a, b), v["pooled"]["pauc"], (a+b)/2, -step)
        separated = bool(v["strictly_separated"])
        first = separated and first_strict is None
        improved = first or best_key is None or key > best_key
        if improved:
            best_key, best_step, stale = key, step, 0
        else:
            stale += 1
        if first:
            first_strict = step
        reason = "strict_full_validation_separation" if separated else "ranking_patience" if stale >= patience else None
        expected = {"improved": improved, "stop": reason is not None, "stop_reason": reason,
                    "selected_step": best_step, "stale": stale, "key": list(key),
                    "replacement_reason": "first_strict_full_separation" if first else
                    "ranking_improved" if improved else "ranking_not_improved"}
        assert expected == record["selection"]
        decisions.append(expected)
    assert not any(d["stop"] for d in decisions[:-1]) and decisions[-1]["stop"]
    return {"all_events_match": True, "selected_step": best_step, "final_stale": stale,
            "first_strict_step": first_strict, "stop_reason": decisions[-1]["stop_reason"]}


def compact_cost(cost):
    timing, gpu = cost["timing"], cost.get("gpu_throughput", {})
    result = {"cold_load_s": cost["cold_artifact_load_seconds"],
              "artifact_bytes": cost["model_artifact_bytes"],
              "cpu_single_p50_ms": timing["single"]["p50_ms"],
              "cpu_single_p95_ms": timing["single"]["p95_ms"],
              "cpu_batch128_rows_per_s": timing["batch"]["throughput_rows_per_second"],
              "whole_worker_rss_peak_bytes": timing["rss"]["peak_bytes"],
              "cold_scores_exact": cost["cold_scores_exact"],
              "pretransformed_scores_exact": cost.get("pretransformed_scores_exact"),
              "cpu_single_max_abs_delta": cost["single_query_max_abs_score_delta"],
              "cpu_single_changed_decision_indices": cost["single_query_changed_decision_indices"],
              "gpu_rows_per_s": gpu.get("rows_per_second"),
              "gpu_max_abs_delta": gpu.get("max_abs_delta_from_registered_cpu"),
              "gpu_changed_decision_indices": gpu.get("changed_decision_indices_at_cpu_threshold")}
    for prefix in ("preprocessing_only", "model_only"):
        if prefix+"_timing" in cost:
            result[prefix+"_p50_ms"] = cost[prefix+"_timing"]["single"]["p50_ms"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="Project root containing the experiment archive directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSON file path")
    args = parser.parse_args()
    PROJECT = args.project.resolve()
    ROOT = PROJECT / "实验归档/20261117_第三十轮模型训练重设计/回传"
    R29 = PROJECT / "实验归档/20261117_第二十九轮模型训练/回传"
    R26 = PROJECT / "实验归档/20261115_第二十六轮历史监督/回传"
    OUT = args.output.resolve()
    frames = {role: pd.read_parquet(ROOT / "inputs" / f"{role}.parquet") for role in ("F", "V")}
    freeze, protocol = read(ROOT/"freeze.json"), read(ROOT/"protocol.json")
    patience = protocol["design"]["selection"]["patience_validation_events"]
    result = {"scope": "Read frozen JSON, parquet, NPZ and byte hashes only. No model loads, forward, fit, new threshold or decisions.",
              "source": str(ROOT), "source_hashes": {f: sha(ROOT/f) for f in ("freeze.json", "protocol.json")},
              "groups": {}, "models": {}, "paired": {}, "R29_cost": {}}
    pairing = {}
    for role, frame in frames.items():
        result["groups"][role] = {str(p)+str(y): len(block) for (p,y),block in frame.groupby(["profile","label"])}
    for directory in sorted((ROOT / "models").iterdir()):
        mid = directory.name
        trajectory = directory / "trajectory"
        history, summary = read(trajectory/"history.json"), read(trajectory/"summary.json")
        pairing[mid] = read(trajectory/"pairing.json")
        identities = read(trajectory/"score_identities.json")
        selected = next(row for row in history if row["step"] == summary["selected_step"])
        selection = replay_selection(history, patience)
        assert selection["selected_step"] == summary["selected_step"]
        assert selection["stop_reason"] == summary["stop_reason"]
        actual_sha, best_sha = sha(trajectory/"model.pt"), sha(trajectory/"best.pt")
        assert actual_sha == best_sha == summary["selected_artifact_sha256"] == freeze["artifacts"][mid]["sha256"]
        with np.load(trajectory/"selected_scores.npz", allow_pickle=False) as cache:
            checks, descriptions = {}, {}
            for role, field, group_key in (("F", "train_scores", "training_groups"), ("V", "validation_scores", "validation_groups")):
                frame = aligned(frames[role], identities[role])
                scores = cache[field].astype(np.float64)
                assert len(scores) == len(frame) and np.isfinite(scores).all()
                groups = {}
                for (p,y), block in frame.groupby(["profile", "label"]):
                    mask = ((frame.profile == p) & (frame.label == y)).to_numpy()
                    bce = float(np.logaddexp(0, -scores[mask] if y else scores[mask]).mean())
                    logged = selected[group_key][str(p)+str(y)]["ensemble_bce"]
                    assert np.isclose(bce, logged, atol=1e-12, rtol=1e-12)
                    groups[str(p)+str(y)] = bce
                checks[role+"_ensemble_bce"] = groups
                descriptions[role] = score_groups(scores, frame)
                if role == "V":
                    current = metrics(scores, frame)
                    for key in ("pooled",):
                        assert current[key] == selected["validation"][key]
                    assert current["panels"] == selected["validation"]["panels"]
                    checks["V_ranking"] = current
                else:
                    fs = {p: metric(scores[(frame.profile==p).to_numpy()], frame[frame.profile==p].label.to_numpy())["strictly_separated"] for p in ("A","B")}
                    assert fs == selected["F_profile_strict_separation"]
                    checks["F_profile_strict_separation"] = fs
        assert all(h["sampling"]["groups"][g]["draws"] == h["step"]*64 for h in history for g in ("A0","A1","B0","B1"))
        assert all(h["sampling"]["groups"][g]["unique"] == result["groups"]["F"][g] for h in history for g in ("A0","A1","B0","B1"))
        assert sum(h["gradient"]["window_steps"] for h in history) == summary["optimizer_steps"]
        assert np.isclose(sum(h["fit_seconds"] for h in history), summary["fit_seconds"])
        model = {
            "summary": {k:v for k,v in summary.items() if not k.endswith("_path")},
            "selection_replay": selection,
            "artifact_checks": {"model_sha256": actual_sha, "best_sha256": best_sha,
                "model_best_summary_freeze_identical": True, "selected_scores_sha256": sha(trajectory/"selected_scores.npz"),
                "history_sha256": sha(trajectory/"history.json")},
            "selected": {"step": selected["step"], "training_groups": selected["training_groups"],
                         "validation_groups": selected["validation_groups"], "sampling": selected["sampling"]["groups"],
                         "validation": selected["validation"]},
            "cache_recomputed": checks, "score_distributions": descriptions,
            "trajectory": {"validation_events": len(history),
                "V_B1_ensemble_bce_first": history[0]["validation_groups"]["B1"]["ensemble_bce"],
                "V_B1_ensemble_bce_last": history[-1]["validation_groups"]["B1"]["ensemble_bce"],
                "V_B1_ensemble_bce_min_and_step": min((h["validation_groups"]["B1"]["ensemble_bce"], h["step"]) for h in history),
                "first_F_strict_step": {p:next((h["step"] for h in history if h["F_profile_strict_separation"][p]),None) for p in ("A","B")},
                "clipped_steps_total": sum(h["gradient"]["clipped_steps"] for h in history),
                "preclip_max_norm": max(h["gradient"]["preclip_max_norm"] for h in history),
                "preclip_weighted_mean_norm": sum(h["gradient"]["preclip_mean_norm"]*h["gradient"]["window_steps"] for h in history)/summary["optimizer_steps"],
                "final_sampling": history[-1]["sampling"]["groups"],
                "event_log": [{"step":h["step"], "V_pauc":h["validation"]["pooled"]["pauc"],
                    "panels_pauc":{p:m["pauc"] for p,m in h["validation"]["panels"].items()},
                    "F_B1_ensemble_bce":h["training_groups"]["B1"]["ensemble_bce"],
                    "V_B1_ensemble_bce":h["validation_groups"]["B1"]["ensemble_bce"],
                    "F_group_head_bce":{g:v["head_bce"] for g,v in h["training_groups"].items()},
                    "V_group_head_bce":{g:v["head_bce"] for g,v in h["validation_groups"].items()},
                    "clipped_steps":h["gradient"]["clipped_steps"],
                    "selected_step":h["selection"]["selected_step"], "stale":h["selection"]["stale"]} for h in history]},
            "cost": compact_cost(read(ROOT/"cost"/f"{mid}.json")),
        }
        result["models"][mid] = model
    for seed in (17,29,43):
        aware, blind = [pairing[f"protocol_{arm}_s{seed}"] for arm in ("aware", "blind")]
        base_keys = set(aware) - {"draw_chain_by_validation_step", "dropout_rng_by_validation_step"}
        equality = {k: aware[k] == blind[k] for k in sorted(base_keys)}
        common = sorted(set(aware["draw_chain_by_validation_step"]) & set(blind["draw_chain_by_validation_step"]),key=int)
        streams = {name:all(aware[name][s] == blind[name][s] for s in common) for name in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step")}
        assert all(equality.values()) and all(streams.values())
        result["paired"][str(seed)] = {"initial_fields_equal":equality,"common_steps":list(map(int,common)),"streams_equal":streams}
    for p in sorted((R29/"cost").glob("tabm_ple_s*.json")):
        result["R29_cost"][p.stem] = compact_cost(read(p))
    overlap_ids = [set(map(tuple,result["models"][f"protocol_blind_s{s}"]["score_distributions"]["V"]["positive_at_or_below_any_normal"]["identities"])) for s in (17,29,43)]
    result["V_blind_overlap_identity_agreement"] = {"intersection": len(set.intersection(*overlap_ids)), "union":len(set.union(*overlap_ids))}
    r29_protocol = read(R29/"protocol.json")
    r26_support = read(R26/"evidence/training-support.json")
    result["historical_comparability"] = {
        "R29_protocol_sha256": sha(R29/"protocol.json"),
        "R29_training_contract": r29_protocol["design"]["training"],
        "R30_training_contract": protocol["design"]["training"],
        "R26_training_support_sha256": sha(R26/"evidence/training-support.json"),
        "R26_training_support": r26_support,
        "interpretation": "R29 to R30 changes support, weighting/sampling, selection and final-refit together. R26 target has 180 positives vs R30 F B1=90 and A1=5123. No isolated architecture/protocol causal claim across rounds. R29 CPU cost is descriptive same-lab; R26 comparable deployment timing was not supplied in this return.",
    }
    result["all_checks_pass"] = True
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(OUT),"all_checks_pass":True,"models":len(result["models"])},ensure_ascii=False))


if __name__ == "__main__":
    main()
