"""Read-only audit of returned JSON/bytes. No model import, forward, fit or selection."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics


DEFAULT_ARCHIVE = Path(r"E:\OneDrive - Coventry University\中传\毕业论文\实验归档\20261118_正常子群训练结果\回传")
DEFAULT_OUTPUT = Path(__file__).resolve().parent
Q = ("0.90", "0.95", "0.99")
VG = "V_each_positive_inversions_not_increased"
EXPECTED_STEPS = list(range(139, 2225, 139))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(value):
    if isinstance(value, dict):
        if {"numerator", "denominator"} <= value.keys():
            return Fraction(value["numerator"], value["denominator"])
        return {k: restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore(v) for v in value]
    return value


def encode(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def describe(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "sum": 0, "min": None, "max": None, "median": None, "p90": None, "p95": None}
    def quantile(p):
        pos = (len(values) - 1) * p
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)
    return {"n": len(values), "sum": sum(values), "min": values[0], "max": values[-1],
            "mean": statistics.mean(values), "median": statistics.median(values),
            "p90": quantile(.9), "p95": quantile(.95)}


def inverse_change(current, baseline, role):
    ids = current[f"{role}_positive_ids"]
    assert ids == baseline[f"{role}_positive_ids"]
    old, new = baseline[f"{role}_positive_inversions"], current[f"{role}_positive_inversions"]
    contract = baseline[f"{role}_contract"]
    profiles = {tuple(i): p for i, p in zip(contract["identities"], contract["profiles"])}
    delta = [n - o for n, o in zip(new, old)]
    worse = [{"identity": i, "profile": profiles[tuple(i)], "baseline": o, "current": n, "delta": n - o}
             for i, o, n in zip(ids, old, new) if n > o]
    bins = {"1": 0, "2_5": 0, "6_10": 0, "11_100": 0, "over_100": 0}
    for row in worse:
        v = row["delta"]
        bins["1" if v == 1 else "2_5" if v <= 5 else "6_10" if v <= 10 else "11_100" if v <= 100 else "over_100"] += 1
    return {"positive_rows": len(ids), "worse_count": len(worse), "improved_count": sum(v < 0 for v in delta),
            "unchanged_count": sum(v == 0 for v in delta), "current_nonzero_count": sum(v > 0 for v in new),
            "delta_all": describe(delta), "worsening_delta": describe([r["delta"] for r in worse]),
            "worsening_delta_bins": bins, "worse_by_profile": dict(Counter(r["profile"] for r in worse)),
            "baseline_inversions": describe(old), "current_inversions": describe(new), "worse_rows": worse}


def verify_tail(tail):
    for q in Q:
        group_h = []
        for group in tail["groups"].values():
            h = max(group["T"][q] - (1 - Fraction(q)), Fraction(0))
            assert h == group["h"][q]
            group_h.append(h)
        assert max(group_h) == tail["H_q"][q]
    assert max(tail["H_q"].values()) == tail["H"]


def recompute_checks(m, b, changes):
    for role in ("F", "V"):
        assert m[f"{role}_contract"] == b[f"{role}_contract"]
    verify_tail(m["tail"])
    for p in ("A", "B"):
        verify_tail(m["source_tails"][p])
    return {
        "F_each_positive_inversions_not_increased": changes["F"]["worse_count"] == 0,
        VG: changes["V"]["worse_count"] == 0,
        "V_panel_strict_separation": all(v["strictly_separated"] for v in m["panels"].values()),
        "V_pooled_pauc_not_lower": m["pooled"]["pauc"] >= b["pooled"]["pauc"],
        "V_panel_pauc_not_lower": all(m["panels"][p]["pauc"] >= b["panels"][p]["pauc"] for p in ("A_TCP", "B_UDP")),
        "normal_max_tie_block_not_increased": m["tail"]["max_tie_block"] <= b["tail"]["max_tie_block"],
        "normal_tied_pairs_not_increased": m["tail"]["tied_pairs"] <= b["tail"]["tied_pairs"],
        "each_H_q_not_increased": all(m["tail"]["H_q"][q] <= b["tail"]["H_q"][q] for q in Q),
        "each_source_H_q_not_increased": all(m["source_tails"][p]["H_q"][q] <= b["source_tails"][p]["H_q"][q]
                                                 for p in ("A", "B") for q in Q)}


def audit(archive, output):
    manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))["files"]
    evidence = {}
    def verify(path):
        rel = path.relative_to(archive).as_posix()
        actual = sha(path)
        assert manifest[rel] == actual, f"Changed return payload: {rel}"
        evidence[rel] = {"sha256": actual, "bytes": path.stat().st_size}
        return actual
    def read(path):
        verify(path)
        return restore(json.loads(path.read_text(encoding="utf-8")))
    result = {"scope": "descriptive_existing_event_audit_no_reselection_no_forward_no_training_no_C_E_read",
              "archive": str(archive), "archive_manifest_sha256": sha(archive / "archive_manifest.json"),
              "quantile_summary_method": "linear_interpolation_at_(n-1)*p", "trajectories": {}}
    all_events, pairs = [], {}
    for folder in sorted((archive / "models").glob("*/trajectory")):
        name = folder.parent.name
        summary, history, baseline, selected, pairing = [read(folder / file) for file in
            ("summary.json", "history.json", "baseline_metrics.json", "selected_metrics.json", "pairing.json")]
        assert [r["step"] for r in history] == EXPECTED_STEPS
        assert summary["optimizer_steps"] == 2224 and summary["completed_validation_events"] == 16
        assert summary["fit_calls"] == 1 and summary["final_refits"] == 0 and summary["status"] == "complete"
        assert summary["selected_step"] == 0 and summary["fallback"] and not summary["candidate_improved"]
        assert verify(folder / "model.pt") == verify(folder / "best.pt") == summary["original_artifact_sha256"]
        assert summary["selected_artifact_sha256"] == summary["original_artifact_sha256"] and selected == baseline
        assert pairing["validation_steps"] == EXPECTED_STEPS
        assert set(pairing["selection_rng_unchanged_by_step"]) == set(map(str, EXPECTED_STEPS))
        assert all(v is True for v in pairing["selection_rng_unchanged_by_step"].values())
        pairs[name] = pairing
        events = []
        for row in history:
            m = row["metrics"]
            changes = {role: inverse_change(m, baseline, role) for role in ("F", "V")}
            checks = recompute_checks(m, baseline, changes)
            assert checks == row["selection"]["guards"]["checks"]
            passed = all(checks.values())
            assert passed == row["selection"]["guards"]["passed"]
            h_better = m["tail"]["H"] < baseline["tail"]["H"]
            assert row["selection"]["eligible"] == (passed and h_better)
            for role in ("F", "V"):
                assert [r["identity"] for r in changes[role]["worse_rows"]] == row["selection"]["guards"][f"{role}_worse_positive_ids"]
            failures = [k for k, v in checks.items() if not v]
            source_bad = {p: [q for q in Q if m["source_tails"][p]["H_q"][q] > baseline["source_tails"][p]["H_q"][q]] for p in ("A", "B")}
            event = {"trajectory": name, "step": row["step"], "path_mean": row["path_mean"], "gradient": row["gradient"],
                     "H": m["tail"]["H"], "H_delta": m["tail"]["H"] - baseline["tail"]["H"],
                     "H_strictly_below_baseline": h_better, "H_q": m["tail"]["H_q"],
                     "source_H_q": {p: m["source_tails"][p]["H_q"] for p in ("A", "B")},
                     "H_q_worse": [q for q in Q if m["tail"]["H_q"][q] > baseline["tail"]["H_q"][q]],
                     "source_H_q_worse": source_bad, "pooled_pauc": m["pooled"]["pauc"],
                     "pooled_pauc_delta": m["pooled"]["pauc"] - baseline["pooled"]["pauc"],
                     "panels": m["panels"],
                     "ties": {k: m["tail"][k] for k in ("unique_scores", "max_tie_block", "tied_pairs")},
                     "ties_delta": {k: m["tail"][k] - baseline["tail"][k] for k in ("unique_scores", "max_tie_block", "tied_pairs")},
                     "F_positive_absolute_score_lower_count": row["selection"]["guards"]["F_lower_positive_count"],
                     "positive_inversion_change": changes, "guard_checks": checks,
                     "all_guards_pass": passed, "eligible": passed and h_better, "failed_guards": failures,
                     "remaining_failed_guards_without_V_each_positive": [k for k in failures if k != VG],
                     "logical_pass_without_V_each_positive": h_better and all(v for k, v in checks.items() if k != VG)}
            events.append(event)
        failures = Counter(k for event in events for k in event["failed_guards"])
        combos = Counter(tuple(e["failed_guards"]) for e in events)
        fewer = [e for e in events if e["H_strictly_below_baseline"]]
        minimum = min((len(e["remaining_failed_guards_without_V_each_positive"]) for e in fewer), default=None)
        unions = {}
        for role in ("F", "V"):
            union = {}
            for event in events:
                for row in event["positive_inversion_change"][role]["worse_rows"]:
                    key = tuple(row["identity"])
                    if key not in union:
                        union[key] = {"identity": row["identity"], "profile": row["profile"], "events": 0, "max_delta": 0, "steps": []}
                    item = union[key]
                    item["events"] += 1
                    item["max_delta"] = max(item["max_delta"], row["delta"])
                    item["steps"].append(event["step"])
            unions[role] = {"distinct_worsened_positive_rows": len(union),
                            "profiles": dict(Counter(r["profile"] for r in union.values())), "rows": list(union.values()),
                            "event_worse_count_distribution": describe([e["positive_inversion_change"][role]["worse_count"] for e in events]),
                            "all_worsening_delta_occurrences": describe([r["delta"] for e in events for r in e["positive_inversion_change"][role]["worse_rows"]])}
        weighted = {k: sum(r["path_mean"][k] * r["gradient"]["window_steps"] for r in history) / 2224
                    for k in ("member_bce", "normal", "positive", "total")}
        result["trajectories"][name] = {
            "execution": {k: summary[k] for k in ("arm", "seed", "status", "optimizer_steps", "completed_validation_events", "fit_calls", "final_refits", "selected_step", "fallback", "candidate_improved", "fit_seconds", "total_seconds")},
            "original_and_selected_artifact_sha256": summary["original_artifact_sha256"],
            "baseline": {"H": baseline["tail"]["H"], "H_q": baseline["tail"]["H_q"],
                         "source_H_q": {p: baseline["source_tails"][p]["H_q"] for p in ("A", "B")},
                         "pooled_pauc": baseline["pooled"]["pauc"], "panels": baseline["panels"],
                         "ties": {k: baseline["tail"][k] for k in ("unique_scores", "max_tie_block", "tied_pairs")}},
            "guard_pass_steps": [e["step"] for e in events if e["all_guards_pass"]], "eligible_steps": [e["step"] for e in events if e["eligible"]],
            "H_improved_steps": [e["step"] for e in fewer],
            "H_min": min(e["H"] for e in events), "H_min_steps": [e["step"] for e in events if e["H"] == min(x["H"] for x in events)],
            "guard_failure_counts": dict(failures),
            "sole_failed_guard_events": [{"step": e["step"], "guard": e["failed_guards"][0], "H_better": e["H_strictly_below_baseline"]} for e in events if len(e["failed_guards"]) == 1],
            "joint_failure_sets": [{"guards": list(k), "events": v} for k, v in combos.items()],
            "logical_ablation": {"removed_guard": VG, "is_adoption_or_reselection": False,
                "remaining_pass_and_H_improved_steps": [e["step"] for e in events if e["logical_pass_without_V_each_positive"]],
                "minimum_remaining_failure_count_among_H_improved": minimum,
                "events_with_that_minimum": [{"step": e["step"], "remaining_failed_guards": e["remaining_failed_guards_without_V_each_positive"]} for e in fewer if len(e["remaining_failed_guards_without_V_each_positive"]) == minimum]},
            "loss": {"kind": "path_means_not_endpoint_loss", "first_window": history[0]["path_mean"],
                     "last_window": history[-1]["path_mean"], "all_step_weighted_mean": weighted},
            "gradient": {"clipped_steps": sum(r["gradient"]["clipped_steps"] for r in history),
                         "max_preclip_norm": max(r["gradient"]["preclip_max_norm"] for r in history),
                         "clipped_windows": [r["step"] for r in history if r["gradient"]["clipped_steps"]]},
            "sampling": {"first_event": history[0]["sampling"], "last_event": history[-1]["sampling"]},
            "positive_changes": unions, "events": events}
        all_events.extend(events)
    assert len(result["trajectories"]) == 6 and len(all_events) == 96
    pairing_fields = ("original_artifact_sha256", "preflight_manifest_sha256", "preprocessing_sha256", "numeric_bins_sha256",
                      "F_identity_sha256", "V_identity_sha256", "initial_state_sha256", "dropout_seed", "sampler_seeds",
                      "initial_dropout_rng_sha256", "initial_sampler_rng_sha256", "optimizer_initial_state_entries",
                      "cpu_score_batch_size", "validation_steps", "draw_chain_by_validation_step", "dropout_rng_by_validation_step")
    for seed in (17, 29, 43):
        a, b = (pairs[f"{arm}_s{seed}"] for arm in ("transport_target", "identity_target"))
        assert all(a[k] == b[k] for k in pairing_fields)
        for field in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
            assert set(a[field]) == set(map(str, EXPECTED_STEPS))
    result["aggregate"] = {"events": 96, "new_model_loads": 0, "new_model_forwards": 0, "new_fits": 0,
        "new_calibrations": 0, "optimizer_updates_recorded": 13344,
        "all_guard_pass_events": [{"trajectory": e["trajectory"], "step": e["step"], "H_strictly_below_baseline": e["H_strictly_below_baseline"]} for e in all_events if e["all_guards_pass"]],
        "eligible_events": sum(e["eligible"] for e in all_events),
        "guard_failure_counts": dict(Counter(k for e in all_events for k in e["failed_guards"])),
        "H_improved_events_by_arm": dict(Counter(e["trajectory"].rsplit("_s", 1)[0] for e in all_events if e["H_strictly_below_baseline"])),
        "logical_pass_without_V_positive_guard": [{"trajectory": e["trajectory"], "step": e["step"]} for e in all_events if e["logical_pass_without_V_each_positive"]],
        "clipped_steps": sum(t["gradient"]["clipped_steps"] for t in result["trajectories"].values()),
        "paired_all_16_event_streams_equal": True}
    for guard in all_events[0]["guard_checks"]:
        result["aggregate"]["guard_failure_counts"].setdefault(guard, 0)
    result["aggregate"]["paired_same_step_transport_H_lower_counts"] = {
        str(seed): sum(t["H"] < c["H"] for t, c in zip(
            result["trajectories"][f"transport_target_s{seed}"]["events"],
            result["trajectories"][f"identity_target_s{seed}"]["events"])) for seed in (17, 29, 43)}
    report_path = archive.parent / "来源/NORMAL-SUBGROUP-RESULTS-2026-11-18.md"
    result["reviewed_return_report"] = {"path": str(report_path), "sha256": sha(report_path)}
    result["limitations"] = [
        "Only sixteen recorded endpoints per trajectory were available; no statements about unrecorded updates or unvisited parameter states.",
        "Endpoint raw F/V score vectors are not retained for rejected candidates; guard and H recomputation uses saved endpoint metrics and inversion arrays, not new scores.",
        "Logical guard deletion only rechecks recorded booleans and H; it does not change selection, adopt a model, or predict C/E effects.",
        "The identity arm still optimizes member BCE, two-sided normal anchoring and a one-sided positive anchor; it is not a zero-update score-preservation control.",
        "Loss values are 139-step path means; they do not prove endpoint loss descent or deployment improvement."]
    result["verified_source_files"] = evidence
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_audit.json").write_text(json.dumps(encode(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# 六轨迹训练与 96 个事件审计", "", "本次固定训练与选轮合同下，96 个记录候选中没有合格改进；这不能推出合同在参数空间中不可满足。48 个 transport 事件的 H 均低于各自 baseline，说明正常子群尾部指标被改变；现有证据没有证明这些变化可在保护原检出能力的同时带来 C/E 改善。", "", "## 执行与拒绝", "", "六轨迹均完成 2,224 次更新和 16 次选轮，总计 13,344 次更新；无 P 重拟合，六份交付工件均逐字节回退原同 seed R30。配对的初始化、预处理、F/V 身份、采样及 dropout 流在全部 16 个事件一致。", "", "| 轨迹 | 全守卫通过 | H 严格下降 | H0 → 记录最小H（步） | 去掉V逐正例守卫后其余条件通过 |", "|---|---:|---:|---|---|"]
    for name, t in result["trajectories"].items():
        lines.append(f"| {name} | {len(t['guard_pass_steps'])}/16 | {len(t['H_improved_steps'])}/16 | {float(t['baseline']['H']):.9f} → {float(t['H_min']):.9f}（{','.join(map(str,t['H_min_steps']))}） | {','.join(map(str,t['logical_ablation']['remaining_pass_and_H_improved_steps'])) or '无'} |")
    lines += ["", "transport 在全部 48 个同 seed、同步数的配对事件中，H 也均低于 identity 对照。唯一全守卫通过事件为 identity s29 的 step 417；其 H 未严格下降，因此没有入选。原报告标题“守卫从未通过”与正文矛盾。", "", "仅删除 V 逐正例逆序保护后，剩余条件和 H 严格改善同时成立的事件为 transport s17 的 1807、2085，以及 s43 的 139，共 3/96。s29 仍为 0/16；该轨迹所有事件的正常并列对数均超过 baseline。因此不能把失败归结为一条守卫，也不能据此推出放宽该守卫会让三 seed 同时可用。这里仅做历史事件的逻辑复核，不重新选轮、采用候选或评分 C/E。", "", "| 守卫 | 失败事件数/96 |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in result["aggregate"]["guard_failure_counts"].items()]
    lines += ["", "## 每条轨迹的联合障碍与逆序幅度", ""]
    for name, t in result["trajectories"].items():
        lines += [f"### {name}", "", f"最少剩余失败项数（仅限 H 已下降的事件、删除 V 逐正例守卫后）：{t['logical_ablation']['minimum_remaining_failure_count_among_H_improved']}。对应事件：" + "; ".join(f"{e['step']}: {','.join(e['remaining_failed_guards']) or '无'}" for e in t["logical_ablation"]["events_with_that_minimum"]) + "。", ""]
        for role in ("F", "V"):
            u = t["positive_changes"][role]
            d = u["event_worse_count_distribution"]
            magnitude = u["all_worsening_delta_occurrences"]
            if not u["distinct_worsened_positive_rows"]:
                lines.append(f"{role}：16 个事件均未出现逐正例逆序数增加。")
            else:
                profiles_text = "、".join(f"{p} {n}条" for p, n in u["profiles"].items())
                lines.append(f"{role}：全程曾变差的不同正例 {u['distinct_worsened_positive_rows']} 条（{profiles_text}）；每事件变差人数范围 {d['min']}–{d['max']}，中位数 {d['median']:g}。变差正例-事件的逆序增量中位数 {magnitude['median']:g}，p95 {magnitude['p95']:g}，最大值 {magnitude['max']:g}。这些增量是排在该正例前方或同分的正常样本数量变化，不是实际漏报人数。")
        lines += ["", "联合失败组合：" + "; ".join(f"{x['events']}次[{','.join(x['guards']) or '全部通过'}]" for x in t["joint_failure_sets"]) + "。", ""]
    lines += ["## 高尾分位与并列变化", "", "transport 的 48 个事件均通过 A/B 来源内各 Hq 不增检查。s17 的池化 H 虽全部下降，池化 top1% 的 Hq 却在前 12 个记录事件超过 baseline；这正是只看最大 H 会漏掉的局部退化。s29、s43 的池化三个 Hq 均未超过各自 baseline。以下向量依次为 top10%、top5%、top1%。", "", "| transport轨迹 | 来源 | Hq baseline → 最后事件2224 |", "|---|---|---|"]
    for name, t in result["trajectories"].items():
        if name.startswith("transport"):
            for p in ("A", "B"):
                before = ", ".join(f"{float(t['baseline']['source_H_q'][p][q]):.6f}" for q in Q)
                after = ", ".join(f"{float(t['events'][-1]['source_H_q'][p][q]):.6f}" for q in Q)
                lines.append(f"| {name} | {p} | ({before}) → ({after}) |")
    lines += ["", "transport s29 的正常并列对数相对 baseline 4020 增加 7–17，最大并列块始终为 84。step278 仅失败并列对数保护：4020→4034，其余守卫通过，H 从 0.708647654 降至 0.516375345。这说明存在不同的拒绝路径，不能把所有失败归入逐正例保护，也不能把少量并列增加直接解释成分数塌缩。冻结合同仍然拒绝该事件，本审计不据结果修改规则。", "", "删除 V 逐正例守卫后的三个逻辑通过事件，其原失败规模为：s17 step1807 有12个B正例各增1–3个正常逆序，step2085 有9个B正例各增1–2；s43 step139 有3个B正例各增1。全部六轨迹中，V逆序变差的身份均来自B。", ""]
    lines += ["## 损失、梯度与采样", "", "以下损失是首个与最后一个 139 步窗口的路径均值，不是 step0 或终点损失。两臂 normal loss 的目标不同，不能直接把绝对数值大小当作方法优劣。", "", "| 轨迹 | BCE 首→末 | normal 首→末 | positive 首→末 | 全程裁剪步数/最大裁剪前范数 |", "|---|---|---|---|---|"]
    for name, t in result["trajectories"].items():
        first, last = t["loss"]["first_window"], t["loss"]["last_window"]
        cells = [f"{first[k]:.6g} → {last[k]:.6g}" for k in ("member_bce", "normal", "positive")]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {t['gradient']['clipped_steps']} / {t['gradient']['max_preclip_norm']:.9f} |")
    final_sampling = next(iter(result["trajectories"].values()))["sampling"]["last_event"]["groups"]
    lines += ["", "六条轨迹每组均抽取 142,336 行次，最终累计完整循环：" + "、".join(f"{g}={v['completed_cycles']}" for g, v in final_sampling.items()) + "。B1 的 98 个完整循环对应首个 139 步事件，不能作为全程轮数。transport s43 的首个窗口有 1 步发生裁剪，最大裁剪前范数 1.2984406948；原报告“从未裁剪”不准确。裁剪次数本身不构成优化失稳证据。", "", "## 对随包解释的判断", "", "- “合同不可满足”超出证据：只能确认这六条轨迹的 96 个记录点没有同时满足保护与 H 改善。baseline 已满足保护，identity s29 step417 也满足保护；没有证明满足全部改进条件的参数状态不存在。", "- “目标不是无效”应拆开：transport 的 H 有实际下降，不能说它完全没有作用；其完整方法未产生合格候选，不能据 loss/H 下降证明目标能够解决误报且保留检出。", "- identity 不是严格保持原输出的无更新对照：它仍有 BCE、正常分数 MSE、正例单侧软锚和 AdamW 更新。其失败不足以证明任何微调都会破坏保护条件。", "- “交集出现1次”不准确：1次仅指全守卫通过；全守卫通过且 H 严格改善的交集是0。", "- transport s17 首事件 F 上15个正例绝对分数下降是报告项；真正逆序变差是5个A正例各增加1个正常逆序，二者不能混用。", "- 拒绝事件提供了新增训练过程证据。最终 C/E 因回退只评价原工件，不能据该评价推出被拒绝候选在 C/E 上的实际效果。", "", "## 可复核范围", "", "审计只读取回传 JSON 和工件字节，对所读回传文件逐项核对 archive_manifest 的 SHA-256。H/Hq 与守卫由已保存的 Fraction、逆序数组和 pAUC 记录重新计算，未加载模型。被拒绝事件没有完整原始分数向量，故不能独立重建其 ROC、排序或部署判决。", "", "training_audit.json 保存全部 96 事件的 H/Hq/来源 Hq、各守卫、联合失败、F/V 逐正例变差身份和增量分布；training_audit.py 可按同路径重放，生成同一结果。"]
    (output / "training_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(encode(result["aggregate"]), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    audit(arguments.archive.resolve(), arguments.output.resolve())
