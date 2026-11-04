"""First-stage teacher/input/capacity screen on F/R/C; no KD or V evaluation.

Twelve fits per seed: two HGB teachers, four hard warmups, four small-student
continuations and two native categorical trees. Source artifacts are checksum-verified as bytes; only
their plan and F/R/C identities are decoded. No V identity or prediction is
loaded. Screening results do not automatically trigger a larger experiment.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .budget_transfer import (
    _benchmark, _fit_method, _no_links, _scores, _selection, _training_frames, _warmup, _working_point,
    qualification, validate_config,
)
from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .models import FeaturePreprocessor, load_bundle, save_bundle, train_model
from .native_trees import CATBOOST_VERSION, load_native_tree, train_native_tree
from .risk_research import _fixed_metrics
from .runner import verify_development_inputs
from .selection import normalize_feature_schema, select_features
from .transfer_diagnostics import binary_rank_metrics, transfer_diagnostics

COMPLETE = "TRANSFER_SCREEN_COMPLETE.json"
METHODS = ("hard", "direct_pauc")
ROLES = ("fit", "reference", "constraint")
NATIVE_TREE_SPECS = {name: {"iterations": iterations, "depth": depth, "learning_rate": .05,
    "l2_leaf_reg": 3., "border_count": 128} for name, iterations, depth in
    (("catboost-small", 64, 4), ("catboost-reference", 256, 6))}


def _benchmark_native(bundle, frame, threshold, repeats=25):
    result = {}
    for size in (1, min(256, len(frame))):
        part = frame.iloc[:size]
        for _ in range(5):
            bundle.predict_raw(part) >= threshold
        times = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            bundle.predict_raw(part) >= threshold
            times.append((time.perf_counter_ns() - start) / 1e6)
        result[str(size)] = {"p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "mean_rows_per_second": size * 1000 / float(np.mean(times)), "repeats": repeats}
    return {"batches": result, "includes": "existing CSV fields -> native conversion/CTRs -> CPU model -> raw threshold",
        "excludes": "CSV disk read, network capture and raw packet/flow field extraction",
        "scope": "laboratory CPU measurement on fixed C rows; not laptop deployment acceptance"}


def _source(v3):
    """Verify the sealed byte inventory without decoding old V outputs."""
    done = read_json(v3 / "RISK_RESEARCH_COMPLETE.json")
    if done.get("status") != "development_complete" or done.get("reserved_partitions_read") != []:
        raise ValueError("screen requires completed development-only V3")
    files = done.get("files", {})
    inventory = {p.relative_to(v3).as_posix() for p in v3.rglob("*")
                 if p.is_file() and p.name != "RISK_RESEARCH_COMPLETE.json"}
    required = {"plan.json", *(f"{role}-identity.parquet" for role in ROLES)}
    if not files or set(files) != inventory or not required.issubset(files):
        raise ValueError("screen V3 source inventory mismatch")
    for name, digest in files.items():
        path = _no_links(v3 / name)
        if not path.is_relative_to(v3) or sha256_file(path) != digest:
            raise ValueError(f"screen V3 artifact mismatch: {name}")
    plan = read_json(v3 / "plan.json")
    if (plan.get("plan_id") != canonical_hash({k: v for k, v in plan.items() if k != "plan_id"})
            or done.get("plan_id") != plan["plan_id"]
            or plan.get("reserved_partitions_read") != []
            or plan["config"].get("cohort_policy") != "complete_development"):
        raise ValueError("screen requires the complete frozen V3 plan")
    return done, plan


def _target_content(scores, labels):
    """Describe T=1 teacher softness; neither correctness nor transfer gain."""
    q = expit(np.asarray(scores, dtype=np.float64))
    y = np.asarray(labels)
    return {"temperature": 1., "source": "in_sample_F_teacher",
        "mean_absolute_soft_minus_hard": float(np.abs(q - y).mean()),
        "per_class": {str(c): {"n": int((y == c).sum()),
            "mean_absolute_soft_minus_hard": float(np.abs(q[y == c] - c).mean()),
            "probability_quantiles": np.quantile(q[y == c], [0, .01, .5, .99, 1]).tolist()}
            for c in (0, 1)},
        "interpretation": "At T=1, identically weighted hard+lambda*KL has the gradient of (1+lambda)*BCE with target (y+lambda*q)/(1+lambda)."}


def _selected_complementarity(frame, teacher_scores, teacher_selection, student_scores, student_selection):
    """Paired errors at the SAME R+C working points used by qualification."""
    y = frame.label.to_numpy()
    tt = teacher_selection["working_point"]["threshold"]
    st = student_selection["working_point"]["threshold"]
    td = np.asarray(teacher_scores, dtype=np.float64) >= tt
    sd = np.asarray(student_scores, dtype=np.float64) >= st
    def counts(mask):
        return {"support": int(mask.sum()),
            "teacher_corrects_student_fn": int((mask & (y == 1) & td & ~sd).sum()),
            "teacher_loses_student_tp": int((mask & (y == 1) & ~td & sd).sum()),
            "teacher_repairs_student_fp": int((mask & (y == 0) & ~td & sd).sum()),
            "teacher_adds_student_fp": int((mask & (y == 0) & td & ~sd).sum())}
    return {"threshold_rule": "each model's selected max(R-normal,C-normal) threshold; same as qualification",
        "teacher_threshold": tt, "student_threshold": st,
        "teacher_C": teacher_selection["C"], "student_C": student_selection["C"],
        "paired": counts(np.ones(len(frame), dtype=bool)),
        "per_attack_family": {str(f): counts(frame.attack_cat.to_numpy() == f)
                              for f in sorted(frame.loc[y == 1, "attack_cat"].unique())},
        "scope": "training C selection diagnostic; not independent transfer evidence"}


def _report(root, summary):
    lines = ["# 第一阶段：教师、字段与容量筛查", "",
        "只使用 F/R/C。C 同时参与阈值与 checkpoint 选择；C≤1%（或配置预算）由训练阈值构造，不是独立准入证据。", 
        "本阶段不加载 V，不自动启动 KD。", "",
        "| 类型 | 字段 | 容量/方法 | seed | C TP/攻击 | C FP/正常 | C 标准化 pAUC |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["auxiliary"] + summary["runs"]:
        c = row["selection"]["C"]
        lines.append(f"| {row.get('kind', 'continuation')} | {row['budget']} | {row.get('capacity', row.get('method', 'HGB'))} | "
            f"{row['seed']} | {c['true_positives']}/{c['n_attack']} | {c['false_positives']}/{c['n_normal']} | "
            f"{row['selection']['C_rank']['partial_auc_standardized']:.6f} |")
    lines.extend(["", "四组 warmup 只在相同训练阶段内比较容量/字段；续训 hard 与 direct_pauc 共享对应 small warmup 权重。",
        "teacher headroom 是开发诊断，不能据此宣称可迁移，也不自动删除失败环境或 seed。",
        "两组原生类别 CatBoost 检查学生函数类：64棵深度4树与256棵深度6树；固定配置仍不能代替充分调参比较。",
        "各模型获得的 C 选择机会不同：warmup 每轮选点，续训从已选 warmup 再逐步选点，固定树仅评估终态；这是容量诊断，不是等调参预算排行榜。",
        "warmup 的 minibatch 更新数与续训的全 F 更新数分别保存在 compute 字段，40 epochs 不等于 40 steps。", ""])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def verify_transfer_screen(output):
    root = _no_links(output)
    done = read_json(root / COMPLETE)
    if (done.get("schema_version") != 1 or done.get("status") != "screen_complete"
            or done.get("reserved_partitions_read") != [] or done.get("validation_loaded") is not False):
        raise ValueError("invalid screen completion")
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*")
             if p.is_file() and p.name != COMPLETE}
    if files != done.get("files"):
        raise ValueError("screen artifact inventory mismatch")
    required = {"plan.json", "summary.json", "REPORT.md", "feature-selection.json", "FROZEN_SCREEN.json",
                *(f"{r}-identity.parquet" for r in ROLES)}
    if not required.issubset(files) or "validation-identity.parquet" in files:
        raise ValueError("screen role/evidence inventory mismatch")
    plan, summary = read_json(root / "plan.json"), read_json(root / "summary.json")
    binding = plan["binding"]
    validate_config(binding["config"])
    if canonical_hash(binding) != done.get("binding_sha256"):
        raise ValueError("screen binding mismatch")
    n = len(binding["config"]["seeds"])
    for key, amount in (("teacher", 2*n), ("warmup", 4*n), ("student", 4*n)):
        if done.get(f"n_{key}_fits") != amount or summary.get(f"{key}_fits") != amount:
            raise ValueError("screen fit count mismatch")
    if done.get("n_native_tree_fits") != 2*n or summary.get("native_tree_fits") != 2*n:
        raise ValueError("screen native-tree fit count mismatch")
    if binding.get("native_tree_specs") != NATIVE_TREE_SPECS or binding.get("catboost_version") != CATBOOST_VERSION:
        raise ValueError("screen native-tree protocol changed")
    if (summary.get("validation_loaded") is not False or summary.get("automatic_expansion") is not False
            or summary.get("independent_confirmation") is not False):
        raise ValueError("screen scope changed")
    expected = {(s, b, m) for s in binding["config"]["seeds"] for b in ("all", "reduced") for m in METHODS}
    observed = [(r["seed"], r["budget"], r["method"]) for r in summary["runs"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("screen condition inventory mismatch")
    auxiliary = summary["auxiliary"]
    expected_aux = {(s, b, k, c) for s in binding["config"]["seeds"] for b in ("all", "reduced")
                    for k, c in (("teacher", None), ("hard_warmup", "small"), ("hard_warmup", "wide"))}
    expected_aux |= {(s, "all", "native_tree", name) for s in binding["config"]["seeds"] for name in NATIVE_TREE_SPECS}
    observed_aux = [(r["seed"], r["budget"], r["kind"], r.get("capacity")) for r in auxiliary]
    if len(set(observed_aux)) != len(observed_aux) or set(observed_aux) != expected_aux:
        raise ValueError("screen teacher/warmup inventory mismatch")
    if read_json(root / "FROZEN_SCREEN.json") != {"runs": summary["runs"], "auxiliary": auxiliary}:
        raise ValueError("screen frozen selections changed")
    for row in summary["runs"] + auxiliary:
        folder = root / row["relative_dir"]
        model_files = ("model.cbm", "metadata.json") if row.get("kind") == "native_tree" else ("model.joblib",)
        if (read_json(folder / "run.json") != row
                or any(not (folder / name).is_file() for name in (*model_files, "training-scores.npz"))):
            raise ValueError("screen model/selection evidence missing")
    for name, digest in binding["source"]["files"].items():
        if files.get("source/" + name) != digest:
            raise ValueError("screen source snapshot mismatch")
    return done


def run_transfer_screen(v3_dir, data_dir, config, output, *, device="cuda", resume=False):
    import torch
    config = copy.deepcopy(config)
    validate_config(config)
    root, v3, data = _no_links(output), _no_links(v3_dir), _no_links(data_dir)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((root, v3), (root, data))):
        raise ValueError("screen work and immutable inputs must be separate")
    source_done, old_plan = _source(v3)
    manifest = verify_development_inputs(data, splits=("train",))
    if manifest != old_plan["data"]:
        raise ValueError("screen prepared data differs from frozen V3")
    binding = {"stage": "screen", "config": config, "source": code_fingerprint(), "device": device,
        "v3_dir": str(v3), "data_dir": str(data), "data": manifest,
        "native_tree_specs": NATIVE_TREE_SPECS, "catboost_version": CATBOOST_VERSION,
        "v3_complete_sha256": sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json"), "v3_plan_id": source_done["plan_id"]}
    if root.exists():
        if not resume:
            raise FileExistsError("existing screen is preserved; use a new output or complete-only resume")
        done = verify_transfer_screen(root)
        if read_json(root / "plan.json")["binding"] != binding:
            raise ValueError("screen resume source/config/data/device differs")
        return done
    if manifest.get("data_kind") != "synthetic" and (device not in {"cuda", "cuda:0"}
            or not torch.cuda.is_available() or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper()):
        raise ValueError("real screen fits require laboratory CUDA0 RTX4090D")
    frames = _training_frames(v3, data)
    schema = normalize_feature_schema(manifest.get("feature_schema"))
    if config["reduced_k"] >= len(schema["features"]):
        raise ValueError("screen reduced budget must remove an original field")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {"binding": binding, "loaded_data_partitions": ["train"],
        "reserved_partitions_read": [], "validation_loaded": False, "automatic_expansion": False,
        "source_verification": "complete sealed V3 byte hashes; no decoding V identities/predictions"})
    write_json(root / "environment.json", environment_snapshot())
    for name, digest in binding["source"]["files"].items():
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[2] / name, target)
        if sha256_file(target) != digest:
            raise ValueError("screen source changed during snapshot")
    for role in ROLES:
        shutil.copyfile(v3 / f"{role}-identity.parquet", root / f"{role}-identity.parquet")
    selection = select_features(frames["fit"], {"method": "mutual_info", "k": config["reduced_k"], "selection_seed": 20261104},
                                20261104, feature_schema=schema)
    write_json(root / "feature-selection.json", selection)
    features = {"all": schema["features"], "reduced": selection["selected_features"]}
    preprocessors = {b: FeaturePreprocessor.fit(frames["fit"], fs, schema) for b, fs in features.items()}
    runs, auxiliary, diagnostics = [], [], []
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=config["threads"]):
            for seed in config["seeds"]:
                teachers, teacher_rows = {}, {}
                for budget, fields in features.items():
                    relative = f"teachers/{budget}-s{seed}"
                    folder = root / relative
                    folder.mkdir(parents=True)
                    tschema = {**schema, "name": schema["name"] + "-" + budget, "features": fields,
                        "categorical": [f for f in schema["categorical"] if f in fields],
                        "discrete": [f for f in schema["discrete"] if f in fields]}
                    started = time.perf_counter()
                    print(f"[screen] teacher {budget} seed={seed}; F only", flush=True)
                    teacher = train_model(frames["fit"], frames["fit"], {"model": "hgb", "features": {"method": "all"},
                        "hyperparameters": {**config["teacher_hyperparameters"], "num_threads": config["threads"]}},
                        seed, feature_schema=tschema)
                    save_bundle(teacher, folder / "model.joblib")
                    values = _scores(teacher, frames, config["chunk_size"])
                    trecord, _ = _selection(teacher, frames, config)
                    row = {"kind": "teacher", "budget": budget, "seed": seed, "relative_dir": relative,
                        "selection": trecord, "fit_rows": len(frames["fit"]), "validation_used": False,
                        "selection_evaluations": 1, "selection_stage": "fixed final HGB state; R+C threshold registration only",
                        "fit_and_query_seconds": time.perf_counter() - started,
                        "native_query_rows": len(frames["fit"]) + 2 * (len(frames["reference"]) + len(frames["constraint"])),
                        "target_source": "in_sample_F_native_margin_not_OOF", "target_content": _target_content(values["fit"], frames["fit"].label),
                        "deployment": {"raw_fields": len(fields), "encoded_dim": teacher.encoded_dim,
                            "model_package_bytes": (folder / "model.joblib").stat().st_size}}
                    write_json(folder / "run.json", row)
                    np.savez_compressed(folder / "training-scores.npz", **values)
                    auxiliary.append(row)
                    teachers[budget], teacher_rows[budget] = values, row
                for capacity, spec in NATIVE_TREE_SPECS.items():
                    relative = f"native-trees/{capacity}-s{seed}"
                    folder = root / relative
                    started = time.perf_counter()
                    tree = train_native_tree(frames["fit"], spec, seed, config["threads"], schema)
                    deployment = tree.save(folder)
                    values = {r: tree.predict_raw(frames[r]) for r in ("reference", "constraint")}
                    wp = _working_point(values, frames, config["max_fpr"])
                    record = {"working_point": wp,
                        "C": _fixed_metrics(frames["constraint"], values["constraint"], wp["threshold"]),
                        "C_rank": binary_rank_metrics(frames["constraint"].label.to_numpy(), values["constraint"], config["max_fpr"])}
                    row = {"kind": "native_tree", "budget": "all", "capacity": capacity, "seed": seed,
                        "relative_dir": relative, "selection": record, "training_statistics": tree.training_statistics,
                        "fit_and_query_seconds": time.perf_counter() - started,
                        "fit_and_query_scope": "F native conversion + fitting + model save + R/C queries/threshold/metrics; fit_seconds inside training_statistics covers estimator.fit only",
                        "selection_evaluations": 1, "selection_stage": "fixed final CatBoost state; R+C threshold registration only",
                        "deployment": {**deployment, "raw_fields": len(schema["features"]),
                            "tree_count": tree.training_statistics["tree_count"], "leaf_count": tree.training_statistics["leaf_count"]},
                        "validation_used": False, "supervision": "F hard labels only"}
                    write_json(folder / "run.json", row)
                    np.savez_compressed(folder / "training-scores.npz", **values)
                    auxiliary.append(row)
                for budget in features:
                    warm = {}
                    for capacity, hidden in (("small", config["small_hidden"]), ("wide", config["wide_hidden"])):
                        relative = f"warmups/{budget}-{capacity}-s{seed}"
                        folder = root / relative
                        warm[capacity] = _warmup(preprocessors[budget], schema, hidden, seed, frames, config, device, folder)
                        row = read_json(folder / "run.json")
                        row.update(budget=budget, capacity=capacity, relative_dir=relative)
                        write_json(folder / "run.json", row)
                        np.savez_compressed(folder / "training-scores.npz", **_scores(warm[capacity],
                            {r: frames[r] for r in ("reference", "constraint")}, config["chunk_size"]))
                        auxiliary.append(row)
                    for method in METHODS:
                        relative = f"students/{budget}-{method}-s{seed}"
                        student, row = _fit_method(warm["small"], frames, config, method, None, seed, device, root / relative)
                        warm_record = read_json(root / f"warmups/{budget}-small-s{seed}/run.json")
                        row.update(budget=budget, relative_dir=relative,
                            warmup_selected_epoch=warm_record["selected_epoch"],
                            warmup_selected_optimizer_updates=warm_record["compute"]["selected_checkpoint_optimizer_updates"],
                            selection_evaluations=config["macro_steps"] + 1,
                            warmup_selection_evaluations=warm_record["selection_evaluations"],
                            selection_stage="C selection over continuation steps after C-selected warmup")
                        write_json(root / relative / "run.json", row)
                        runs.append(row)
                    ds = _scores(student, {r: frames[r] for r in ("reference", "constraint")}, config["chunk_size"])
                    x = student.preprocessor.transform(frames["constraint"])
                    report = transfer_diagnostics(calibration_labels=frames["reference"].label.to_numpy(),
                        calibration_teacher_scores=teachers["all"]["reference"], calibration_student_scores=ds["reference"],
                        probe_labels=frames["constraint"].label.to_numpy(), probe_teacher_scores=teachers["all"]["constraint"],
                        probe_student_scores=ds["constraint"], calibration_ids=frames["reference"].sample_id.astype(str).tolist(),
                        probe_ids=frames["constraint"].sample_id.astype(str).tolist(),
                        probe_families=frames["constraint"].attack_cat.astype(str).tolist(),
                        student_input_groups=[hashlib.sha256(r.tobytes()).hexdigest() for r in x],
                        teacher_fit_family_counts=frames["fit"].attack_cat.value_counts().to_dict(), max_fpr=config["max_fpr"])
                    report = {"seed": seed, "budget": budget,
                        "qualification": qualification(teacher_rows["all"]["selection"], row["selection"], config),
                        "selected_working_point": _selected_complementarity(frames["constraint"],
                            teachers["all"]["constraint"], teacher_rows["all"]["selection"], ds["constraint"], row["selection"]),
                        "R_only_diagnostic": report,
                        "R_only_scope": "additional R-only score/support/ambiguity diagnostic; these paired counts do not define qualification",
                        "matched_teacher_selection": teacher_rows[budget]["selection"]}
                    write_json(root / f"diagnostic-{budget}-s{seed}.json", report)
                    diagnostics.append({"seed": seed, "budget": budget, **report["qualification"]})
            if config["measure_cpu"]:
                for row in runs + auxiliary:
                    folder = root / row["relative_dir"]
                    tau = row["selection"]["working_point"]["threshold"]
                    row["cpu_pipeline"] = (_benchmark_native(load_native_tree(folder), frames["constraint"], tau)
                        if row.get("kind") == "native_tree" else _benchmark(load_bundle(folder / "model.joblib"), frames["constraint"], tau))
                    row["cpu_pipeline"]["measurement_rows"] = "fixed first C rows, same order for all models"
                    write_json(folder / "run.json", row)
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
    n = len(config["seeds"])
    summary = {"status": "screen_complete", "runs": runs, "auxiliary": auxiliary, "qualification": diagnostics,
        "teacher_fits": 2*n, "warmup_fits": 4*n, "student_fits": 4*n, "native_tree_fits": 2*n, "validation_loaded": False,
        "independent_confirmation": False, "automatic_expansion": False,
        "comparison_scope": "fixed-capacity diagnostic; cross-family C selection opportunities differ, not an equally tuned ranking",
        "cpu_benchmark_performed": config["measure_cpu"], "next_action": "review F/R/C evidence before registering any KD continuation"}
    write_json(root / "FROZEN_SCREEN.json", {"runs": runs, "auxiliary": auxiliary})
    write_json(root / "summary.json", summary)
    _report(root, summary)
    write_json(root / COMPLETE, {"schema_version": 1, "status": "screen_complete", "created_at": utc_now(),
        "binding_sha256": canonical_hash(binding), "reserved_partitions_read": [], "validation_loaded": False,
        "n_teacher_fits": 2*n, "n_warmup_fits": 4*n, "n_student_fits": 4*n, "n_native_tree_fits": 2*n,
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}})
    return verify_transfer_screen(root)
