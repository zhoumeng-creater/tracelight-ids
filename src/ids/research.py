"""Research v2: controlled development studies, never a second final test.

This module deliberately loads only train and validation. Historical test
results are already known; no result produced here is independent confirmation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import html
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
import yaml

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import load_partition
from .evaluation import calibrate_threshold, evaluate_binary
from .models import FeaturePreprocessor, ModelBundle, _network, _resolve_device, save_bundle, train_model
from .research_kd import ResearchKDConfig, research_kd_loss
from .runner import verify_development_inputs
from .selection import normalize_feature_schema, select_features


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_config(config):
    """Reject missing controls and ambiguous comparisons before creating output."""
    required = {"schema_version", "purpose", "cohort_policy", "seeds", "max_fpr", "threads", "folds", "fold_seed",
                "max_train_rows", "max_validation_rows", "sampling_seed", "teacher", "models"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Research config keys/version do not match the fixed development contract")
    if config["purpose"] != "research_v2_development_only":
        raise ValueError("Research v2 is development only")
    if config["cohort_policy"] not in {"complete_development", "bounded_pilot"}:
        raise ValueError("cohort_policy must explicitly declare full development or bounded pilot")
    if config["cohort_policy"] == "complete_development" and any(
            config[key] is not None for key in ("max_train_rows", "max_validation_rows")):
        raise ValueError("Complete development must retain all groups and family support; row caps are pilot only")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique list")
    for seed in seeds:
        _integer(seed, "seed", 0)
    for key in ("threads", "folds", "fold_seed", "sampling_seed"):
        _integer(config[key], key, 2 if key == "folds" else 1)
    for key in ("max_train_rows", "max_validation_rows"):
        if config[key] is not None:
            _integer(config[key], key)
    if isinstance(config["max_fpr"], bool) or not 0 < config["max_fpr"] < 1:
        raise ValueError("max_fpr must be between 0 and 1")
    teacher = config["teacher"]
    if teacher.get("model") != "hgb" or teacher.get("features", {}).get("method") != "all":
        raise ValueError("The current OOF teacher must be full-field HGB")
    if teacher.get("hyperparameters", {}).get("early_stopping") is not False:
        raise ValueError("OOF teacher must explicitly disable internal early stopping")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a nonempty list")
    variants = []
    for model in models:
        if set(model) - {"variant", "model", "features", "hyperparameters", "loss"}:
            raise ValueError("Unknown research model keys")
        variant = model.get("variant", "")
        if not variant or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in variant):
            raise ValueError("variant must be a safe lowercase path component")
        variants.append(variant)
        if model.get("model") not in {"mlp", "tabm"}:
            raise ValueError("Research students support mlp and optional official tabm")
        if model["model"] == "mlp":
            ResearchKDConfig(**model.get("loss", {}))
        elif model.get("loss"):
            raise ValueError("TabM is an independent hard-label reference, not an implicit KD variant")
    if len(set(variants)) != len(variants):
        raise ValueError("Research variants must be unique")


def _group_column(frame):
    # Repaired CIC uses group_id for the union of feature/session relations.
    # Falling back directly to feature_hash would discard session isolation.
    for name in ("group_id", "group_hash", "feature_hash"):
        if name in frame:
            return name
    raise ValueError("A verified feature/session group identity is required")


def _identity(frame):
    columns = ["sample_id", "feature_hash", "label"]
    columns.extend(name for name in ("group_id", "group_hash") if name in frame)
    if not set(columns).issubset(frame) or frame["sample_id"].duplicated().any():
        raise ValueError("Development data needs unique stable IDs and group/label identity")
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Development labels must be binary")
    for column in columns:
        if frame[column].isna().any():
            raise ValueError("Development identity contains missing values")
    return frame.loc[:, columns].reset_index(drop=True)


def group_budget(frame, max_rows, seed):
    """Select whole groups in a seeded order; never split a group to meet a cap.

    This is a fixed compute-budget sample, not a representative-family promise.
    Actual family support and selected identities are saved for every run.
    """
    _identity(frame)
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True).copy()
    _integer(max_rows, "max_rows")
    sizes = frame.groupby(_group_column(frame), sort=True).size()
    order = np.random.default_rng(seed).permutation(len(sizes))
    chosen, used = [], 0
    for index in order:
        size = int(sizes.iloc[index])
        if used + size <= max_rows:
            chosen.append(sizes.index[index])
            used += size
    subset = frame.loc[frame[_group_column(frame)].isin(chosen)].reset_index(drop=True).copy()
    if set(subset["label"]) != {0, 1}:
        raise ValueError("Whole-group budget lacks both classes; increase budget, do not silently resample")
    return subset


def _sample_record(frame):
    return {"n_rows": len(frame), "n_groups": frame[_group_column(frame)].nunique(),
            "identity_sha256": canonical_hash(_identity(frame).to_dict(orient="records")),
            "labels": {str(k): int(v) for k, v in frame["label"].value_counts().items()},
            "families": {str(k): int(v) for k, v in frame["attack_cat"].value_counts().items()}}


def make_oof_teacher(train, validation, teacher_config, seed, folds, fold_seed, schema, threads, output, max_fpr=.01):
    """Fit each fold's preprocessing/teacher solely on the other train groups."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    labels = train["label"].to_numpy()
    groups = train[_group_column(train)].to_numpy()
    if len(set(groups)) < folds:
        raise ValueError("Too few training groups for OOF")
    scores = np.full(len(train), np.nan, dtype=np.float64)
    assignment = np.full(len(train), -1, dtype=np.int64)
    records = []
    config = copy.deepcopy(teacher_config)
    config["hyperparameters"]["num_threads"] = threads
    begin = time.perf_counter()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=fold_seed)
    with threadpool_limits(limits=threads):
        for fold, (fit_index, heldout_index) in enumerate(splitter.split(np.zeros(len(train)), labels, groups)):
            fit, heldout = train.iloc[fit_index], train.iloc[heldout_index]
            if set(fit["label"]) != {0, 1}:
                raise ValueError("An OOF fitting fold is missing a binary class")
            for key in {_group_column(train), "feature_hash"}:
                if set(fit[key]) & set(heldout[key]):
                    raise ValueError("OOF group boundary leaked")
            if (assignment[heldout_index] >= 0).any():
                raise ValueError("OOF heldout coverage is not unique")
            start = time.perf_counter()
            bundle = train_model(fit, fit, config, seed, feature_schema=schema)
            scores[heldout_index] = bundle.predict_scores(heldout)
            assignment[heldout_index] = fold
            save_bundle(bundle, root / f"fold-{fold}.joblib")
            records.append({"fold": fold, "fit": _sample_record(fit), "heldout": _sample_record(heldout),
                            "fit_and_predict_seconds": time.perf_counter() - start})
        if not np.isfinite(scores).all() or (assignment < 0).any():
            raise ValueError("OOF scores do not cover every selected train row")
        # Full-train reference is diagnostic and never supplies its in-sample targets.
        reference = train_model(train, train, config, seed, feature_schema=schema)
        reference_scores = reference.predict_scores(validation)
    save_bundle(reference, root / "reference.joblib")
    targets = _identity(train).assign(fold=assignment, teacher_score=scores)
    targets.to_parquet(root / "targets.parquet", index=False)
    diagnostics = {"scope": "training-only OOF targets; full teacher evaluated on development validation",
                   "fold_seed": fold_seed, "teacher_seed": seed, "folds": records,
                   "group_column": _group_column(train), "generation_seconds": time.perf_counter() - begin,
                   "oof": evaluate_binary(labels, scores, .5, train["attack_cat"]),
                   "correct_at_score_0_5_fraction": float(np.mean((scores >= .5) == labels)),
                   "targets_sha256": sha256_file(root / "targets.parquet"),
                   "full_teacher_validation": evaluate_binary(validation["label"], reference_scores,
                       calibrate_threshold(validation["label"], reference_scores, max_fpr)["threshold"], validation["attack_cat"])}
    write_json(root / "oof.json", diagnostics)
    return scores, diagnostics


def train_research_mlp(train, validation, config, seed, *, schema, teacher_scores=None, device="cpu", threads=1):
    """Matched initialization, batch order and hard-BCE stopping for all losses."""
    import torch

    schema = normalize_feature_schema(schema)
    selection = select_features(train, config.get("features", {"method": "all"}), seed,
                                feature_schema=schema)
    selected = selection["selected_features"]
    preprocessor = FeaturePreprocessor.fit(train, selected, schema)
    x_train, x_val = preprocessor.transform(train), preprocessor.transform(validation)
    labels = train["label"].to_numpy(dtype=np.float32)
    val_labels = validation["label"].to_numpy(dtype=np.float32)
    if set(labels) != {0, 1} or set(val_labels) != {0, 1}:
        raise ValueError("Train and validation must each contain both labels")
    hp = copy.deepcopy(config["hyperparameters"])
    hidden = tuple(hp.pop("hidden_sizes", [32, 16]))
    if not hidden or any(type(n) is not int or n < 1 for n in hidden):
        raise ValueError("hidden_sizes must contain positive integers")
    epochs, batch_size, patience = (hp.pop("epochs", 120), hp.pop("batch_size", 1024), hp.pop("patience", 12))
    for key, value in (("epochs", epochs), ("batch_size", batch_size), ("patience", patience)):
        _integer(value, key)
    lr, decay, delta = hp.pop("learning_rate", .001), hp.pop("weight_decay", .0001), hp.pop("min_delta", .00001)
    if hp or not np.isfinite([lr, decay, delta]).all() or lr <= 0 or min(decay, delta) < 0:
        raise ValueError("Invalid or unknown MLP training hyperparameters")
    positive_weight = float((labels == 0).sum() / (labels == 1).sum())
    loss_parameters = copy.deepcopy(config.get("loss", {}))
    if "positive_weight" in loss_parameters:
        raise ValueError("positive_weight is computed exclusively from selected train labels")
    loss_config = ResearchKDConfig(positive_weight=positive_weight, **loss_parameters)
    if loss_config.mode in {"kd", "gated_kd", "gated_matched_hard"}:
        if teacher_scores is None or len(teacher_scores) != len(train):
            raise ValueError("Verified row-aligned training OOF scores are required")
    soft = np.asarray(teacher_scores if teacher_scores is not None else labels, dtype=np.float32)
    group_ids = pd.factorize(train[_group_column(train)], sort=True)[0]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(device)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    begin = time.perf_counter()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        network = _network(x_train.shape[1], hidden).to(device)
        initial_hash = canonical_hash({k: v.detach().cpu().numpy().tolist() for k, v in network.state_dict().items()})
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=decay)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(labels),
                    torch.from_numpy(soft), torch.from_numpy(group_ids.astype(np.int64)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                            generator=torch.Generator().manual_seed(seed))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
        best_loss, best_state, best_epoch, stale, history = float("inf"), None, None, 0, []
        for epoch in range(1, epochs + 1):
            network.train()
            totals = defaultdict(float)
            counts = defaultdict(float)
            gate_max = 0.0
            for inputs, targets, teacher, group in loader:
                inputs, targets, teacher, group = (v.to(device) for v in (inputs, targets, teacher, group))
                optimizer.zero_grad(set_to_none=True)
                loss = research_kd_loss(network(inputs).squeeze(1), targets, loss_config,
                                        teacher_scores=teacher, group_ids=group, partition="train")
                if not torch.isfinite(loss.total):
                    raise ValueError("Research training loss is non-finite")
                loss.total.backward()
                optimizer.step()
                for key, value in loss.components.items():
                    totals[key] += float(value.detach().item()) * len(inputs)
                for key, value in loss.statistics.items():
                    if key == "gate_mean":
                        totals[key] += float(value) * len(inputs)
                    elif key == "gate_max":
                        gate_max = max(gate_max, float(value))
                    else:
                        counts[key] += float(value)
            network.eval()
            val_loss = 0.0
            val_score_chunks = []
            with torch.inference_mode():
                for start in range(0, len(x_val), batch_size):
                    inputs = torch.from_numpy(x_val[start:start + batch_size]).to(device)
                    targets = torch.from_numpy(val_labels[start:start + batch_size]).to(device)
                    logits = network(inputs).squeeze(1)
                    val_loss += criterion(logits, targets).item() * len(inputs)
                    val_score_chunks.append(torch.sigmoid(logits).cpu().numpy())
            val_loss /= len(x_val)
            if not np.isfinite(val_loss):
                raise ValueError("Validation stopping loss is not finite")
            val_scores = np.concatenate(val_score_chunks)
            # Descriptive curve only: stopping still uses the same hard BCE.
            working_point = calibrate_threshold(val_labels, val_scores, .01)
            val_detected = np.count_nonzero(val_scores[val_labels == 1] >= working_point["threshold"])
            history.append({"epoch": epoch, "val_hard_bce": val_loss,
                            "validation_curve_device": device,
                            "val_recall_at_1pct_development_fpr": float(val_detected / np.count_nonzero(val_labels == 1)),
                            "val_actual_development_fpr": working_point["achieved_fpr"],
                            "train_batch_weighted_means": {k: v / len(train) for k, v in totals.items()},
                            "train_summed_batch_counts": dict(counts), "gate_max": gate_max,
                            "pairwise_aggregation": "sample-weighted mean of batch-local group-balanced pair losses; counts summed"})
            if val_loss < best_loss - delta:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        bundle = ModelBundle("mlp", preprocessor, selected, x_train.shape[1], history,
            network_state=best_state, hidden_dims=hidden, device="cpu", batch_size=batch_size,
            num_threads=threads, feature_schema=schema, best_epoch=best_epoch, seed=seed)
        bundle.training_statistics = {"actual_device": device, "parameter_count": sum(p.numel() for p in network.parameters()),
            "best_epoch": best_epoch, "epochs_completed": len(history), "best_validation_loss": best_loss,
            "initial_weights_sha256": initial_hash, "positive_weight": positive_weight,
            "loss": asdict(loss_config), "training_seconds": time.perf_counter() - begin,
            "stopping_rule": "same validation weighted hard BCE, min_delta, patience for every MLP objective",
            "validation_curve_device": device, "final_scoring_device": "cpu",
            "selection": selection["metadata"], "status": "development_only"}
        return bundle
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:]


def verify_research(output):
    root = Path(output).resolve()
    complete = read_json(root / "RESEARCH_COMPLETE.json")
    if complete.get("status") != "development_complete" or complete.get("reserved_partitions_read") != []:
        raise ValueError("Not a completed development-only research artifact")
    if not isinstance(complete.get("files"), dict) or not complete["files"]:
        raise ValueError("Research completion file inventory is empty")
    for name, digest in complete["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Research artifact mismatch: {name}")
    plan = read_json(root / "plan.json")
    config = plan["binding"]["config"]
    validate_config(config)
    expected_runs = {(model["variant"], seed) for model in config["models"] for seed in config["seeds"]}
    summary = read_json(root / "summary.json")
    actual_runs = {(item["variant"], item["seed"]) for item in summary["runs"]}
    if (actual_runs != expected_runs or len(summary["runs"]) != len(expected_runs)
            or complete["n_students"] != len(expected_runs)
            or complete["n_teacher_fits"] != (config["folds"] + 1) * len(config["seeds"])):
        raise ValueError("Research candidate/seed coverage is incomplete")
    required = {"plan.json", "summary.json", "report.html", "train-identity.parquet", "validation-identity.parquet"}
    required.update("source/" + name for name in plan["code"]["files"])
    for variant, seed in expected_runs:
        required.update(f"students/{variant}-s{seed}/{name}" for name in
                        ("model.joblib", "validation-predictions.parquet", "run.json", "history.json"))
    for seed in config["seeds"]:
        required.update(f"teachers/s{seed}/{name}" for name in ("targets.parquet", "oof.json", "reference.joblib"))
        required.update(f"teachers/s{seed}/fold-{fold}.joblib" for fold in range(config["folds"]))
    inventory = {p.relative_to(root).as_posix() for p in root.rglob("*")
                 if p.is_file() and p.name != "RESEARCH_COMPLETE.json"}
    if not required.issubset(complete["files"]) or set(complete["files"]) != inventory:
        raise ValueError("Research completion file inventory is not closed")
    return complete


def paired_comparisons(root, records):
    """Publish both recovered and newly missed attacks, with matched controls."""
    pairs = [("matched_hard", "hard"), ("kd", "matched_hard"), ("gated_kd", "kd"),
             ("gated_kd", "gated_matched_hard"), ("gated_correct_only", "gated_kd"),
             ("gated_confidence_only", "gated_kd"), ("hard_rank", "hard"),
             ("matched_hard_rank", "matched_hard"), ("kd_rank", "kd"),
             ("gated_kd_rank", "gated_kd"), ("hard", "large_hard")]
    lookup = {(row["variant"], row["seed"]): row for row in records}
    result = []
    for candidate, reference in pairs:
        for seed in sorted({row["seed"] for row in records}):
            if (candidate, seed) not in lookup or (reference, seed) not in lookup:
                continue
            left, right = lookup[(candidate, seed)], lookup[(reference, seed)]
            capacity_comparison = reference == "large_hard"
            if not capacity_comparison:
                for key in ("features", "hyperparameters"):
                    if left["model_config"].get(key) != right["model_config"].get(key):
                        raise ValueError("Mechanism comparison has mismatched fields/training budget")
                if left["statistics"]["initial_weights_sha256"] != right["statistics"]["initial_weights_sha256"]:
                    raise ValueError("Mechanism comparison has different initial weights")
            a = pd.read_parquet(root / "students" / f"{candidate}-s{seed}" / "validation-predictions.parquet")
            b = pd.read_parquet(root / "students" / f"{reference}-s{seed}" / "validation-predictions.parquet")
            for column in ("sample_id", "label", "feature_hash", "attack_cat"):
                if not a[column].equals(b[column]):
                    raise ValueError("Paired validation identity differs")
            attack = a.label.to_numpy() == 1
            recovered = attack & (a.decision.to_numpy() == 1) & (b.decision.to_numpy() == 0)
            newly_missed = attack & (a.decision.to_numpy() == 0) & (b.decision.to_numpy() == 1)
            family = []
            for name in sorted(set(a.loc[attack, "attack_cat"])):
                mask = attack & (a.attack_cat.to_numpy() == name)
                family.append({"family": name, "n_attacks": int(mask.sum()),
                    "recovered": int((recovered & mask).sum()), "newly_missed": int((newly_missed & mask).sum()),
                    "net_detected": int(((recovered.astype(int) - newly_missed.astype(int)) * mask).sum())})
            result.append({"candidate": candidate, "reference": reference, "seed": seed,
                "comparison_kind": "capacity" if capacity_comparison else "matched_training_mechanism",
                "recall_delta_pp": 100 * (left["metrics"]["attack_recall"] - right["metrics"]["attack_recall"]),
                "fpr_delta_pp": 100 * (left["metrics"]["normal_fpr"] - right["metrics"]["normal_fpr"]),
                "recovered": int(recovered.sum()), "newly_missed": int(newly_missed.sum()), "families": family,
                "example_recovered_ids": a.loc[recovered, "sample_id"].head(5).tolist(),
                "example_newly_missed_ids": a.loc[newly_missed, "sample_id"].head(5).tolist()})
    return result


def _report(root, summary):
    rows = []
    for item in summary["runs"]:
        metric = item["metrics"]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
            (item["variant"], item["seed"], f'{metric["attack_recall"] * 100:.3f}',
             f'{metric["normal_fpr"] * 100:.3f}', item["model_bytes"],
             item["statistics"].get("parameter_count"), item["statistics"].get("best_epoch"))) + "</tr>")
    body = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>TraceLight · Research v2</title><style>body{font:16px/1.7 system-ui;background:#eef3f6;color:#193446;
max-width:1180px;margin:36px auto;padding:0 20px}header{background:#12374b;color:white;border-radius:16px;padding:30px}
.notice{border-left:4px solid #d69b32;padding:16px;background:#fff7df}table{border-collapse:collapse;width:100%;background:white}
th,td{padding:10px;text-align:left;border-bottom:1px solid #dae3e8}th{background:#dcebe9}.scroll{overflow-x:auto}
code{overflow-wrap:anywhere}a{color:#147b78}</style><header><p>TRACELIGHT / RESEARCH V2</p>
<h1>蒸馏机制与低误报检测</h1><p>可追溯的受控开发实验</p></header>
<p class="notice">所有指标来自已使用过的训练／验证来源。阈值也取自本次验证样本；这是开发比较，不能引用为独立最终测试。
当前样本预算与训练预算见 plan.json。近期模型的固定预算适配不等于完整复现论文最优成绩。
表中最终分数来自CPU重载推理；逐epoch曲线来自训练设备，两种数值口径分别记录，不强制边界样本逐条相同。</p>
<p><a href="summary.json">完整指标、家族与训练统计</a> · <a href="plan.json">运行前计划</a></p>
<p id="support"></p><div class="scroll"><table><thead><tr><th>方案</th><th>种子</th><th>攻击召回 %</th><th>正常误报 %</th>
<th>部署文件 B</th><th>参数量</th><th>最佳 epoch</th></tr></thead><tbody>"""
    paired_rows = []
    for row in summary["comparisons"]:
        families = "".join(f"<tr><td>{html.escape(item['family'])}</td><td>{item['n_attacks']}</td>"
            f"<td>+{item['recovered']}</td><td>-{item['newly_missed']}</td></tr>" for item in row["families"])
        paired_rows.append(f"<details><summary>{html.escape(row['candidate'])} / {html.escape(row['reference'])} · "
            f"seed {row['seed']} · 召回差 {row['recall_delta_pp']:+.3f} pp · 恢复 {row['recovered']} / 新增漏检 {row['newly_missed']}"
            "</summary><table><tr><th>攻击家族</th><th>支持数</th><th>恢复检出</th><th>新增漏检</th></tr>"
            + families + "</table></details>")
    kind = summary["data_kind"]
    label = "合成工程数据 · 不可作为论文性能" if kind == "synthetic" else "真实来源开发数据 · 非独立最终成绩"
    body = body.replace("可追溯的受控开发实验", html.escape(label))
    support = summary["samples"]
    normal_count = int(support["validation"]["labels"].get("0", 0))
    support_text = (f"来源标记：{kind}；采样政策：{summary['cohort_policy']}；train={support['train']['n_rows']}，"
                    f"validation={support['validation']['n_rows']}，正常支持={normal_count}。"
                    "各工作点的实际误报数与阈值保存在完整JSON；小样本不能证明部署误报受控。")
    body = body.replace('<p id="support"></p>', '<p id="support">' + html.escape(support_text) + '</p>')
    tail = "</tbody></table></div><h2>同种子机制对照</h2><p>点击展开全部家族。恢复检出和新增漏检同时保留；"
    tail += "差值属于本次开发工作点，不代表统计显著性或独立泛化保证。</p>" + "".join(paired_rows) + "</html>"
    (root / "report.html").write_text(body + "".join(rows) + tail, encoding="utf-8")


def run_research(data_dir, config, output, *, device="cpu", resume=False, dry_run=False):
    validate_config(config)
    manifest = verify_development_inputs(data_dir)
    schema = normalize_feature_schema(manifest.get("feature_schema"))
    code = code_fingerprint()
    binding = {"config": config, "dataset_manifest_sha256": sha256_file(Path(data_dir) / "manifest.json"),
               "source_sha256": code["source_sha256"], "device": device}
    if dry_run:
        return {"status": "plan_only", "seeds": config["seeds"], "n_students": len(config["models"]) * len(config["seeds"]),
                "n_teacher_fits": (config["folds"] + 1) * len(config["seeds"]), "reserved_partitions_read": [],
                "max_train_rows": config["max_train_rows"], "max_validation_rows": config["max_validation_rows"],
                "dependency": "tabm==0.0.3 required only when a TabM reference is configured"}
    root = Path(output)
    if root.exists():
        if not resume:
            raise FileExistsError("Research output already exists; use a new directory or verify with --resume")
        result = verify_research(root)
        if read_json(root / "plan.json")["binding"] != binding:
            raise ValueError("Cannot resume a different config/source/data/device")
        return result
    recent_versions = None
    if any(model["model"] == "tabm" for model in config["models"]):
        from .recent_baselines import recent_dependency_versions
        recent_versions = recent_dependency_versions()
    # Only two partition loader calls occur here; tests assert the boundary.
    full_train, full_validation = load_partition(data_dir, "train"), load_partition(data_dir, "validation")
    train = group_budget(full_train, config["max_train_rows"], config["sampling_seed"])
    validation = group_budget(full_validation, config["max_validation_rows"], config["sampling_seed"] + 1)
    for key in {"feature_hash", _group_column(train)}:
        if set(train[key]) & set(validation[key]):
            raise ValueError("Train/validation group boundary leaked")
    root.mkdir(parents=True)
    plan = {"created_at": utc_now(), "scope": "research_v2_development_only", "binding": binding,
            "data_kind": manifest["data_kind"], "schema": schema, "code": code,
            "environment": {**environment_snapshot(), "recent_dependencies": recent_versions},
            "samples": {"train": _sample_record(train),
            "validation": _sample_record(validation)},
            "full_development_support": {"train": _sample_record(full_train), "validation": _sample_record(full_validation)},
            "families_absent_after_budget": {"train": sorted(set(full_train.attack_cat) - set(train.attack_cat)),
                                             "validation": sorted(set(full_validation.attack_cat) - set(validation.attack_cat))},
            "reserved_partitions_read": []}
    write_json(root / "plan.json", plan)
    project_root = Path(__file__).resolve().parents[2]
    # Retain exact working bytes, including uncommitted code and line endings.
    for name, digest in code["files"].items():
        source, destination = project_root / name, root / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if sha256_file(destination) != digest:
            raise ValueError("Source changed before its run snapshot was captured")
    _identity(train).to_parquet(root / "train-identity.parquet", index=False)
    _identity(validation).to_parquet(root / "validation-identity.parquet", index=False)
    records = []
    for seed in config["seeds"]:
        teacher_scores, teacher_diagnostics = make_oof_teacher(train, validation, config["teacher"], seed,
            config["folds"], config["fold_seed"], schema, config["threads"], root / "teachers" / f"s{seed}", config["max_fpr"])
        for model in config["models"]:
            variant = model["variant"]
            print(f"Research development: {variant}, seed={seed}", flush=True)
            run_dir = root / "students" / f"{variant}-s{seed}"
            run_dir.mkdir(parents=True)
            start = time.perf_counter()
            if model["model"] == "tabm":
                from .recent_baselines import train_tabm
                selection = select_features(train, model.get("features", {"method": "all"}), seed, feature_schema=schema)
                bundle = train_tabm(train, validation, model, seed, device=device, num_threads=config["threads"],
                                    feature_schema=schema, selected_features=selection["selected_features"])
            else:
                bundle = train_research_mlp(train, validation, model, seed, schema=schema,
                    teacher_scores=teacher_scores, device=device, threads=config["threads"])
            training_seconds = time.perf_counter() - start
            path = run_dir / "model.joblib"
            joblib.dump(bundle, path)
            with threadpool_limits(limits=config["threads"]):
                scores = np.asarray(bundle.predict_scores(validation), dtype=np.float64)
                restored = joblib.load(path)
                reloaded_scores = np.asarray(restored.predict_scores(validation), dtype=np.float64)
            if not np.array_equal(scores, reloaded_scores):
                raise ValueError("Model save/reload changed validation scores")
            threshold = calibrate_threshold(validation["label"], scores, config["max_fpr"])
            metrics = evaluate_binary(validation["label"], scores, threshold["threshold"], validation["attack_cat"])
            predictions = _identity(validation).assign(attack_cat=validation["attack_cat"].to_numpy(), score=scores,
                decision=(scores >= threshold["threshold"]).astype(int), threshold=threshold["threshold"])
            predictions.to_parquet(run_dir / "validation-predictions.parquet", index=False)
            record = {"variant": variant, "seed": seed, "model_config": model, "metrics": metrics,
                      "development_threshold": threshold, "statistics": bundle.training_statistics,
                      "training_seconds": training_seconds, "model_bytes": path.stat().st_size,
                      "oof_targets_sha256": teacher_diagnostics["targets_sha256"],
                      "prediction_sha256": sha256_file(run_dir / "validation-predictions.parquet")}
            write_json(run_dir / "run.json", record)
            write_json(run_dir / "history.json", bundle.history)
            records.append(record)
    if code_fingerprint()["source_sha256"] != code["source_sha256"]:
        raise ValueError("Source changed during research; output remains incomplete")
    summary = {"scope": "development_only_not_independent_test", "runs": records,
               "data_kind": manifest["data_kind"], "cohort_policy": config["cohort_policy"],
               "comparisons": paired_comparisons(root, records),
               "reserved_partitions_read": [], "samples": plan["samples"]}
    write_json(root / "summary.json", summary)
    _report(root, summary)
    complete = {"status": "development_complete", "completed_at": utc_now(), "n_students": len(records),
                "n_teacher_fits": (config["folds"] + 1) * len(config["seeds"]), "reserved_partitions_read": [],
                "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}}
    write_json(root / "RESEARCH_COMPLETE.json", complete)
    return verify_research(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_research(args.data_dir, config, args.output, device=args.device,
                          resume=args.resume, dry_run=args.dry_run)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
