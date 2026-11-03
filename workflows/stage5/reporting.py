"""Summarize the frozen test matrix and package only its verified evidence."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import shutil
from statistics import mean, stdev
import tempfile
import zipfile

from ids.common import canonical_hash, read_json, sha256_file


DATASET_LABELS = {"unsw": "UNSW-NB15", "cicids": "Improved CICIDS2017"}
METRICS = ("attack_recall", "normal_fpr", "precision", "macro_f1", "average_precision", "roc_auc")
PACKAGE = "TraceLight-Stage5.zip"


def _encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _publish(path, data):
    """Write atomically. An existing artifact must match the deterministic rebuild."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing report/checksum differs from sealed evidence: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stats(values):
    # Expose the denominator whenever a metric is undefined for one or more seeds.
    observed = [v for v in values if v is not None]
    return {"n_defined": len(observed), "n_planned": len(values),
            "mean": mean(observed) if len(observed) == len(values) and observed else None,
            "sample_sd": stdev(observed) if len(observed) == len(values) and len(observed) > 1 else None}


def _paired(candidate, reference):
    """Positive detection delta means additional attacks detected at the frozen thresholds."""
    if candidate is None or reference is None:
        if candidate is not reference:
            raise ValueError("Paired runs disagree on empty sensitivity subset")
        return None
    c, r = candidate["confusion"], reference["confusion"]
    if (candidate["n_samples"] != reference["n_samples"]
            or c["tp"] + c["fn"] != r["tp"] + r["fn"] or c["tn"] + c["fp"] != r["tn"] + r["fp"]):
        raise ValueError("Paired test denominators differ")
    families = {f["category"]: f for f in reference["attack_categories"]}
    if set(families) != {f["category"] for f in candidate["attack_categories"]}:
        raise ValueError("Paired family coverage differs")
    differences = []
    for f in candidate["attack_categories"]:
        base = families[f["category"]]
        if f["n_attacks"] != base["n_attacks"]:
            raise ValueError("Paired family denominators differ")
        differences.append({"family": f["category"], "n_attacks": f["n_attacks"],
                            "candidate_detected": f["detected"], "reference_detected": base["detected"],
                            "delta_detected": f["detected"] - base["detected"],
                            "delta_recall_pp": 100 * (f["recall"] - base["recall"])})
    deltas = {key: (None if candidate[key] is None or reference[key] is None
                    else 100 * (candidate[key] - reference[key])) for key in METRICS}
    return {"n_samples": candidate["n_samples"], "delta_pp": deltas,
            "delta_tp": c["tp"] - r["tp"], "delta_fp": c["fp"] - r["fp"], "families": differences}


def _fmt(value, decimals=2, scale=1, signed=False):
    if value is None:
        return "—"
    return format(value * scale, ("+" if signed else "") + f",.{decimals}f")


def _over_budget(metrics, target):
    if metrics is None or metrics["normal_fpr"] is None:
        return None
    return metrics["normal_fpr"] > target


def _table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    return ('<div class="table"><table><thead><tr>' + cells("th", headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + cells("td", r) + "</tr>" for r in rows)
            + "</tbody></table></div>")


def _verify_zip(path, inventory, manifest):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if (len(names) != len(inventory) + 1 or set(names) != set(inventory) | {"MANIFEST.sha256"}
                or z.read("MANIFEST.sha256") != manifest):
            raise ValueError("ZIP inventory differs from the completed evaluation")
        for name, digest in inventory.items():
            with z.open(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"ZIP member differs from the sealed result: {name}")


def _archive(output, inventory):
    path = output / PACKAGE
    manifest = "".join(f"{digest}  {name}\n" for name, digest in sorted(inventory.items())).encode("utf-8")
    if path.exists():
        _verify_zip(path, inventory, manifest)
    else:
        with tempfile.NamedTemporaryFile(prefix=".stage5-", suffix=".partial", dir=output, delete=False) as f:
            tmp = Path(f.name)
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
                for name in sorted(inventory):
                    member = zipfile.ZipInfo(name)  # fixed timestamp, including across resume
                    member.compress_type = zipfile.ZIP_DEFLATED
                    with (output / name).open("rb") as src, z.open(member, "w") as dest:
                        shutil.copyfileobj(src, dest, 1024 * 1024)
                member = zipfile.ZipInfo("MANIFEST.sha256")
                z.writestr(member, manifest)
            _verify_zip(tmp, inventory, manifest)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    _publish(path.with_suffix(".zip.sha256"), (sha256_file(path) + "\n").encode("ascii"))
    return path


def _summary(output, plan, profile):
    rows, pairings, groups = [], [], []
    for item in plan["candidates"]:
        ds, rid = item["dataset"], item["run_id"]
        metrics = read_json(output / "tests" / ds / rid / "metrics.json")
        calibration = read_json(output / "calibration" / ds / f"{rid}.json")
        if (metrics["dataset"] != ds or metrics["run_id"] != rid or metrics["seed"] != item["seed"]
                or calibration["threshold"] != metrics["threshold"]
                or metrics["full"]["threshold"] != metrics["threshold"]):
            raise ValueError("Result does not match its frozen run and threshold")
        rows.append({"dataset": ds, "run_id": rid, "seed": item["seed"], "model": item["model"],
                     "role": item["role"], "condition": item["condition"], "threshold": metrics["threshold"],
                     "deployment": metrics["deployment"], "calibration": calibration,
                     "test_fpr_above_target": {s: _over_budget(metrics[s], plan["max_fpr"])
                                               for s in ("full", "nonoverlap")},
                     "n_overlap": metrics["n_overlap"], "n_nonoverlap": metrics["n_nonoverlap"],
                     "nonoverlap_status": metrics["nonoverlap_status"],
                     "full": metrics["full"], "nonoverlap": metrics["nonoverlap"]})
    for ds in DATASET_LABELS:
        sample = [r for r in rows if r["dataset"] == ds]
        seeds = sorted({r["seed"] for r in sample})
        if not seeds or len(sample) != 2 * len(seeds):
            raise ValueError("Missing frozen candidate/reference pairs")
        for scope in (("full", "nonoverlap") if ds == "unsw" else ("full",)):
            values = []
            for seed in seeds:
                pair = [r for r in sample if r["seed"] == seed]
                cand, ref = _roles(pair)
                delta = _paired(cand[scope], ref[scope])
                deployment = {key: cand["deployment"][key] / ref["deployment"][key]
                              for key in ("raw_fields", "model_file_bytes", "cpu_p95_ms", "cpu_rows_per_s")}
                values.append({"seed": seed, "candidate": cand["run_id"], "reference": ref["run_id"],
                               "difference": delta, "deployment_ratio_candidate_over_reference": deployment})
            aggregate = {key: _stats([v["difference"]["delta_pp"][key] if v["difference"] else None
                                      for v in values]) for key in METRICS}
            pairings.append({"dataset": ds, "scope": scope, "per_seed": values, "delta_pp": aggregate})
            for role in sorted({r["role"] for r in sample}):
                selected = sorted((r for r in sample if r["role"] == role), key=lambda r: r["seed"])
                groups.append({"dataset": ds, "scope": scope, "role": role,
                               "run_ids": [r["run_id"] for r in selected],
                               "metrics": {k: _stats([r[scope][k] if r[scope] else None for r in selected])
                                           for k in METRICS}})
    return {"schema_version": 1, "profile": profile, "phase": "frozen_test_evaluation",
            "source_completion_sha256": sha256_file(output / "EVALUATION_COMPLETE.json"),
            "target_calibration_fpr": plan["max_fpr"], "runs": rows, "groups": groups, "comparisons": pairings,
            "family_universe": plan["known_attack_families"],
            "statistics": "candidate minus reference at each seed; arithmetic mean and sample SD of seed "
                          "differences; fixed splits; no significance test; undefined values stay null"}


def _roles(pair):
    roles = {run["role"]: run for run in pair}
    if len(pair) != 2 or set(roles) != {"candidate", "reference"}:
        raise ValueError("Invalid candidate/reference roles")
    return roles["candidate"], roles["reference"]


def _html(summary):
    real = summary["profile"] == "research"
    mode = "正式封存测试" if real else "合成数据 · 工程验证"
    parts = [f"<h1>TraceLight · {mode}</h1>",
             "<p>两个环境分别拟合。每个模型在独立校准集上固定阈值，整批阈值锁定后才评估测试集。"
             f"校准集正常流量的经验误报预算为 {_fmt(summary['target_calibration_fpr'], 2, 100)}%。"
             "表中保留测试集实际误报数和分母。没有可评估样本的指标以横线表示。</p>"]
    if not real:
        parts.append('<p class="notice">本页使用合成样本，只验证评估程序，数值不可用作论文成绩。</p>')
    for ds, label in DATASET_LABELS.items():
        parts.append(f"<section><h2>{label}</h2>")
        parts.append("<p>" + ("42 字段小型 MLP 对比 42 字段 RF。另给出排除与原始训练源完整字段重复的子集。"
                               if ds == "unsw" else "普通 MI 精简小型 MLP 对比全字段同款 MLP。") + "</p>")
        runs = [r for r in summary["runs"] if r["dataset"] == ds]
        for comp in (c for c in summary["comparisons"] if c["dataset"] == ds):
            scope = comp["scope"]
            parts.append("<h3>" + ("完整测试集" if scope == "full" else "UNSW 无精确重叠子集 · 沿用原阈值") + "</h3>")
            data = []
            for row in runs:
                m = row[scope]
                if m is None:
                    data.append([row["run_id"], "0", "—", "—", "—", "—", "—", "—", "—"])
                    continue
                cf = m["confusion"]
                data.append([row["run_id"], f'{cf["tp"]:,}/{cf["tp"] + cf["fn"]:,}',
                             _fmt(m["attack_recall"], 3, 100), f'{cf["fp"]:,}/{cf["fp"] + cf["tn"]:,}',
                             _fmt(m["normal_fpr"], 3, 100),
                             {True: "超出", False: "预算内", None: "缺测"}[row["test_fpr_above_target"][scope]],
                             _fmt(m["precision"], 3, 100),
                             _fmt(m["macro_f1"], 4), format(row["threshold"], ".17g")])
            parts.append(_table(["模型 / 种子", "检出 / 攻击", "召回 %", "误报 / 正常", "误报率 %", "误报预算",
                                 "精确率 %", "Macro-F1", "校准阈值"], data))
            per = []
            for rec in comp["per_seed"]:
                d = rec["difference"]
                if d is None:
                    per.append([rec["seed"], rec["candidate"], rec["reference"], "—", "—", "—", "—"])
                    continue
                per.append([rec["seed"], rec["candidate"], rec["reference"],
                            f'{d["delta_tp"]:+,}', f'{d["delta_fp"]:+,}',
                            _fmt(d["delta_pp"]["attack_recall"], 4, signed=True),
                            _fmt(d["delta_pp"]["normal_fpr"], 4, signed=True)])
            parts.append(_table(["种子", "候选", "参照", "净增检出", "净增误报", "召回差 pp", "误报率差 pp"], per))
            recall = comp["delta_pp"]["attack_recall"]
            parts.append(f"<p>同种子召回差均值 {_fmt(recall['mean'], 4, signed=True)} pp，"
                         f"种子间样本标准差 {_fmt(recall['sample_sd'], 4)} pp。各训练种子共享数据划分。</p>")
            families = []
            for rec in comp["per_seed"]:
                d = rec["difference"]
                if d is None:
                    continue
                observed = set()
                for f in d["families"]:
                    observed.add(f["family"])
                    families.append([rec["seed"], f["family"], f["n_attacks"], f["candidate_detected"],
                                     f["reference_detected"], _fmt(f["delta_recall_pp"], 3, signed=True)])
                for missing in sorted(set(summary["family_universe"].get(ds, [])) - observed):
                    families.append([rec["seed"], missing, 0, "—", "—", "—"])
            parts.append("<details><summary>各攻击家族的测试分母与检出数</summary>"
                         + _table(["种子", "家族", "攻击数", "候选检出", "参照检出", "召回差 pp"], families)
                         + "</details>")
        dep = [row["deployment"] for row in runs]
        parts.append("<h3>同一模型的部署成本</h3><p>模型大小与单线程 CPU 计时复用第四轮同工件。"
                     "每个种子的 p95 和吞吐分别取既有轮次的中位数；计时范围为预处理加推理。</p>")
        parts.append(_table(["模型 / 种子", "原始字段", "部署字节", "单条 p95 ms", "批吞吐 行/s"],
                            [[r["run_id"], d["raw_fields"], _fmt(d["model_file_bytes"], 0),
                              _fmt(d["cpu_p95_ms"], 3), _fmt(d["cpu_rows_per_s"], 0)]
                             for r, d in zip(runs, dep)]))
        parts.append("</section>")
    parts.append('<footer><a href="summary.json">完整 JSON（含逐家族计数、AP 与 ROC-AUC）</a>'
                 ' · <a href="../STAGE5_PLAN.json">冻结清单</a> · '
                 '<a href="../THRESHOLDS_LOCK.json">全矩阵阈值记录</a></footer>')
    return ("""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><title>TraceLight · 冻结评估</title>
<style>*{box-sizing:border-box}body{margin:0 auto;max-width:76rem;padding:1.4rem;color:#152a35;background:#f5f8fa;
font:16px/1.65 system-ui,'Microsoft YaHei',sans-serif}h1,h2{line-height:1.25}h2{border-bottom:2px solid #28665f;
padding-bottom:.5rem}h3{margin:1.7rem 0 .5rem}p{max-width:67rem}section{margin:2.4rem 0}.table{max-width:100%;
overflow-x:auto;margin:.8rem 0}table{border-collapse:collapse;background:white;font-variant-numeric:tabular-nums;
min-width:36rem;width:100%;font-size:.91rem}td,th{text-align:right;padding:.45rem .65rem;white-space:nowrap;
border-bottom:1px solid #d5e1e4}td:first-child,th:first-child{text-align:left}th{color:#fff;background:#285654}
details{padding:.5rem 0}summary{cursor:pointer;color:#175652;font-weight:650}.notice{border-left:.3rem solid #ab5d14;
padding:.6rem 1rem;background:#ffefd6}a{color:#145956}footer{padding:1rem 0}</style><main>"""
            + "".join(parts) + "</main></html>\n").encode("utf-8")


def finalize(output, config):
    """Validate the completed matrix, rebuild its report and read back every ZIP member."""
    if __package__:
        from .core import verify_core
    else:
        from core import verify_core
    output = Path(output).resolve()
    complete = verify_core(output)
    plan = read_json(output / "STAGE5_PLAN.json")
    if plan["config_sha256"] != canonical_hash(config) or plan["config"] != config:
        raise ValueError("Report configuration differs from the frozen plan")
    summary = _summary(output, plan, config["profile"])
    reports = {"reports/summary.json": _encoded(summary), "reports/index.html": _html(summary)}
    inventory = dict(complete["files"])
    inventory["EVALUATION_COMPLETE.json"] = sha256_file(output / "EVALUATION_COMPLETE.json")
    inventory["EVALUATION_COMPLETE.sha256"] = sha256_file(output / "EVALUATION_COMPLETE.sha256")
    for name, content in reports.items():
        _publish(output / name, content)
        inventory[name] = hashlib.sha256(content).hexdigest()
    return _archive(output, inventory)
