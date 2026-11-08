"""Sequential isolated workers; failed cells do not suppress independent cells."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, write_json
from ids.context_cache.preprocessing import labels
from .data import load_inputs, split_support
from .metrics import decisions, evaluate, paired, threshold
from .preprocessing import Representation, collision_report, raw_numeric
from .protocol import REPO, source_snapshot, validate


def _verified_artifact(inputs, name):
    path = inputs["paths"][name]
    if sha256_file(path) != inputs["spec"]["artifacts"][name]["sha256"]:
        raise ValueError(f"artifact SHA mismatch: {name}")
    return path


def _archived_scores(inputs, name):
    schema, frames = inputs["schema"], inputs["frames"]
    result = {}
    for role, artifact, column in (("calibration", f"{name}_calibration" if name == "qcoc"
                                   else "baseline_calibration", "score"),
                                  ("evaluation", "baseline_evaluation", f"{name}_score")):
        saved = pd.read_csv(_verified_artifact(inputs, artifact), dtype={"row_id": str},
                            float_precision="round_trip")
        frame = frames[role]
        if (saved["row_id"].tolist() != frame[schema.id_column].astype(str).tolist()
                or not np.array_equal(saved["label"], labels(frame, schema))):
            raise ValueError("archived score identities differ")
        result[role] = saved[column].to_numpy(dtype=np.float64)
    return result


def preflight(config):
    inputs = load_inputs(config)
    spec, schema = inputs["spec"], inputs["schema"]
    validate(spec)
    _, _, split = split_support(inputs["frames"]["support"], schema,
                                spec["settings"]["split_seed"], spec["settings"]["validation_fraction"])
    availability = []
    for name in spec["settings"].get("reference", []):
        archived = _archived_scores(inputs, name)
        registration = threshold(archived["calibration"], .01)
        availability.append({"id": f"reference_{name}", "saved_scores": "verified",
                             "calibration_fp": registration["calibration_fp"]})
    if spec["data_kind"] == "development":
        import torch
        device = torch.device(spec["settings"]["device"])
        if (device.type != "cuda" or not torch.cuda.is_available()
                or (device.index or 0) >= torch.cuda.device_count()):
            raise RuntimeError("the frozen laboratory batch requires an available CUDA device")
        if sha256_file(inputs["paths"]["checkpoint"]) != spec["checkpoint"]["sha256"]:
            raise ValueError("TabICLv2 checkpoint hash mismatch")
        from ids.context_cache.preprocessing import require_pinned_preprocessing
        from ids.recent_baselines import recent_dependency_versions
        require_pinned_preprocessing()
        recent_dependency_versions()
    return {"protocol_sha256": canonical_hash(spec), "data": inputs["audit"], "split": split,
            "tasks": spec["tasks"], "reference": availability,
            "source": source_snapshot(), "environment": environment_snapshot(),
            "real_model_loads": 0, "real_preprocessor_fits": 0,
            "scope": "development E reused; no old final test is loaded"}


def _write_scores(output, inputs, score_arrays, descriptor):
    schema, frames, spec = inputs["schema"], inputs["frames"], inputs["spec"]
    output.mkdir(parents=True, exist_ok=True)
    registration = threshold(score_arrays["calibration"], spec["settings"]["alpha"])
    registration.update({"score_contract": descriptor["score_contract"],
                         "model": descriptor, "calibration_sha256": spec["files"]["calibration"]["sha256"]})
    write_json(output / "threshold.json", registration)
    for role in ("calibration", "evaluation"):
        frame = frames[role]
        pd.DataFrame({"row_id": frame[schema.id_column].astype(str), "label": labels(frame, schema),
                      "attack_family": frame[schema.family_column].astype(str),
                      "score": score_arrays[role],
                      "predicted_attack": decisions(score_arrays[role], registration).astype(np.uint8)
                      }).to_csv(output / f"{role}.csv", index=False)
    truth = labels(frames["evaluation"], schema)
    families = frames["evaluation"][schema.family_column].to_numpy()
    result = evaluate(truth, score_arrays["evaluation"], registration, families)
    result["threshold"] = registration
    result["sensitivity"] = {
        str(a): evaluate(truth, score_arrays["evaluation"], threshold(score_arrays["calibration"], a), families)
        for a in spec["settings"]["sensitivity_alphas"]}
    if "full" in spec["settings"].get("reference", []):
        old = _archived_scores(inputs, "full")
        result["paired_vs_full"] = paired(truth, decisions(old["evaluation"], threshold(old["calibration"])),
                                          decisions(score_arrays["evaluation"], registration))
    write_json(output / "metrics.json", result)
    return result


def _train_options(task, settings):
    return {"width": task["width"], "learning_rate": task["learning_rate"], "seed": task["seed"],
            "device": settings["device"], "batch_size": settings["batch_size"], "k": settings["k"],
            "n_blocks": settings["n_blocks"], "weight_decay": settings["weight_decay"],
            "dropout": settings["dropout"]}


def _isolated_cost(output, inputs, descriptor):
    write_json(output / "deployment.json", descriptor)
    with (output / "cost-worker.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-m", "ids.route_batch", "_cost-worker",
            "--config", str(inputs["config_path"]), "--output", str(output)],
            stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode or not (output / "cost.json").is_file():
        return {"status": "failed", "error": "isolated CPU cost worker failed",
                "exit_code": completed.returncode, "log": "cost-worker.log"}
    return read_json(output / "cost.json")


def _tabm_task(task, inputs, output):
    from .tabm_training import SCORE_CONTRACT, load_predictor, train_fixed, train_trajectory
    frames, schema, settings = inputs["frames"], inputs["schema"], inputs["spec"]["settings"]
    support = frames["support"]
    tr, va, split = split_support(support, schema, settings["split_seed"], settings["validation_fraction"])
    write_json(output / "split.json", split)
    inner = Representation(task["representation"], schema).fit(support.iloc[tr])
    inner.save(output / "inner_preprocessing.joblib")
    x_train, x_val = inner.transform(support.iloc[tr]), inner.transform(support.iloc[va])
    truth = labels(support, schema)
    write_json(output / "inner_representation.json", {"preprocessing": inner.report(),
        "train": collision_report(x_train, truth[tr]), "validation": collision_report(x_val, truth[va])})
    trajectory = train_trajectory(x_train, truth[tr], x_val, truth[va],
        output_dir=output / "search", epochs=settings["epochs"], checkpoints=settings["checkpoints"],
        **_train_options(task, settings))
    if trajectory["status"] != "complete":
        return {"status": "training_incomplete", "search": trajectory}
    # Freeze the selected epoch BEFORE any C/E model scoring. C and E never select an epoch.
    selection = {"epoch": trajectory["best_epoch"], "criterion": "minimum validation ensemble BCE",
                 "tie_break": "earliest epoch", "split_ids_sha256": split["split_ids_sha256"],
                 "selection_roles": ["support/train", "support/validation"]}
    write_json(output / "selection.json", selection)
    final = Representation(task["representation"], schema).fit(support)
    final.save(output / "preprocessing.joblib")
    x_support = final.transform(support)
    fitted = train_fixed(x_support, truth, output_dir=output / "refit", epochs=selection["epoch"],
                         **_train_options(task, settings))
    if fitted["status"] != "complete":
        return {"status": "training_incomplete", "search": trajectory, "refit": fitted}
    audits = {role: collision_report(final.transform(frame), labels(frame, schema))
              for role, frame in frames.items()}
    write_json(output / "representation.json", {"preprocessing": final.report(), "roles": audits})
    descriptor = {"kind": "tabm", "model_file": "refit/model.pt", "preprocessing_file": "preprocessing.joblib",
                  "model_sha256": sha256_file(fitted["model_path"]),
                  "preprocessing_sha256": sha256_file(output / "preprocessing.joblib"),
                  "score_contract": SCORE_CONTRACT}
    # The actual fixed-training endpoint name is model.pt by the trainer contract.
    descriptor["model_file"] = Path(fitted["model_path"]).relative_to(output).as_posix()
    model = load_predictor(fitted["model_path"], device="cpu")
    def predict(frame, _model=model):
        return _model.predict_scores(final.transform(frame), batch_size=settings["query_batch_size"])
    arrays = {role: predict(frames[role]) for role in ("calibration", "evaluation")}
    result = _write_scores(output, inputs, arrays, descriptor)
    del predict, model
    gc.collect()
    cost = _isolated_cost(output, inputs, descriptor)
    # Endpoint diagnostics are all disclosed, never used to replace the V-selected refit.
    endpoint_results = {}
    endpoints = {"best": trajectory["best_checkpoint"], **trajectory["checkpoints"]}
    for name, checkpoint in endpoints.items():
        endpoint = load_predictor(checkpoint["model_path"], device="cpu")
        endpoint_arrays = {role: endpoint.predict_scores(inner.transform(frames[role]),
                             batch_size=settings["query_batch_size"]) for role in ("calibration", "evaluation")}
        endpoint_results[name] = _write_scores(output / "endpoints" / name, inputs, endpoint_arrays,
            {"kind": "tabm_inner_diagnostic", "epoch": checkpoint["epoch"],
             "fit_role": "support/train", "score_contract": SCORE_CONTRACT})
        del endpoint
    return {"status": "measurement_failed" if cost.get("status") == "failed" else "complete",
            "task": task, "metrics": result, "cost": cost,
            "model": descriptor, "selection": selection, "parameter_count": fitted["parameter_count"],
            "search_seconds": trajectory.get("elapsed_seconds"), "refit_seconds": fitted.get("elapsed_seconds"),
            "endpoint_metrics": endpoint_results}


def _foundation_task(task, inputs, output):
    from .foundation import fit_foundation
    settings, schema, spec = inputs["spec"]["settings"], inputs["schema"], inputs["spec"]
    checkpoint = (spec["tabpfn_checkpoint"]["path"] if task["model_kind"] == "tabpfn25"
                  else inputs["paths"]["checkpoint"])
    start = time.perf_counter()
    model = fit_foundation(task["model_kind"], raw_numeric(inputs["frames"]["support"], schema),
        labels(inputs["frames"]["support"], schema), checkpoint_path=checkpoint,
        seed=settings["foundation_seed"], device=settings["device"], n_estimators=task["n_estimators"])
    build_seconds = time.perf_counter() - start
    model.save(output / "model.joblib")
    descriptor = {"kind": "foundation", "model_file": "model.joblib", "metadata": model.metadata,
                  "model_sha256": sha256_file(output / "model.joblib"),
                  "score_contract": "float64 log-odds of official averaged probabilities; adapter clipping recorded"}
    def predict(frame, _model=model):
        return _model.predict_scores(raw_numeric(frame, schema), batch_size=settings["query_batch_size"])
    arrays = {role: predict(inputs["frames"][role]) for role in ("calibration", "evaluation")}
    result = _write_scores(output, inputs, arrays, descriptor)
    deployment_bytes = (output / "model.joblib").stat().st_size + Path(checkpoint).stat().st_size
    del predict, model
    gc.collect()
    cost = _isolated_cost(output, inputs, descriptor)
    return {"status": "measurement_failed" if cost.get("status") == "failed" else "complete",
            "task": task, "metrics": result, "cost": cost, "model": descriptor,
            "initial_build_seconds": build_seconds, "model_and_checkpoint_bytes": deployment_bytes}


def _reference_task(task, inputs, output):
    from ids.context_cache.csv_workflow import CSVDetector
    name = task["name"]
    arrays = _archived_scores(inputs, name)
    model_root = Path(inputs["spec"]["round14_root"]) / "comparison" / name
    descriptor = {"kind": "reference", "directory": str(model_root), "score_contract": "FP32 attack-normal logit"}
    result = _write_scores(output, inputs, arrays, descriptor)
    if not (model_root / "model.pt").is_file():
        return {"status": "scores_only", "task": task, "metrics": result, "model": descriptor,
                "cost_unavailable": "original deployment model.pt missing; no cost is imputed"}
    start = time.perf_counter()
    model = CSVDetector.load(model_root)
    load_seconds = time.perf_counter()-start
    def predict(frame, _model=model):
        return _model.predict_frame(frame)["score"].to_numpy(dtype=np.float64)
    # Replay a fixed batch, not the entire already-replayed dataset.
    replay = predict(inputs["frames"]["evaluation"].iloc[:128])
    if not np.allclose(replay, arrays["evaluation"][:128], atol=3e-5, rtol=3e-5):
        raise ValueError("reference deployment does not reproduce saved scores")
    deployment_bytes = sum(p.stat().st_size for p in model_root.iterdir() if p.is_file())
    del predict, model
    gc.collect()
    cost = _isolated_cost(output, inputs, descriptor)
    return {"status": "measurement_failed" if cost.get("status") == "failed" else "complete",
            "task": task, "metrics": result, "cost": cost, "model": descriptor,
            "initial_load_seconds": load_seconds, "original_deployment_bytes": deployment_bytes}


def worker(config, task_id, output_dir):
    import torch
    from threadpoolctl import threadpool_limits
    from .foundation import FoundationUnavailable
    inputs = load_inputs(config)
    inputs["config_path"] = Path(config).resolve()
    spec = validate(inputs["spec"])
    task = next(item for item in spec["tasks"] if item["id"] == task_id)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    write_json(output / "state.json", {"status": "running", "task": task})
    write_json(output / "environment.json", environment_snapshot())
    try:
        with threadpool_limits(limits=1):
            result = {"tabm": _tabm_task, "foundation": _foundation_task,
                      "reference": _reference_task}[task["kind"]](task, inputs, output)
    except FoundationUnavailable as exc:
        result = {"status": "unavailable", "task": task, "reason": str(exc)}
    except Exception as exc:
        result = {"status": "failed", "task": task, "error_type": type(exc).__name__,
                  "error": str(exc), "traceback": traceback.format_exc()}
    write_json(output / "result.json", result)
    write_json(output / "state.json", {"status": result["status"], "task": task})
    return result


def _task_states(root, tasks):
    result = []
    for task in tasks:
        folder = root / "tasks" / task["id"]
        attempts = sorted(p for p in folder.glob("attempt-*") if p.is_dir()) if folder.exists() else []
        current = attempts[-1] if attempts else None
        record = read_json(current / "result.json") if current and (current / "result.json").is_file() else None
        result.append({"task": task, "attempt": current, "result": record})
    return result


def summarize(root):
    root = Path(root)
    spec = read_json(root / "protocol.json")
    states = _task_states(root, spec["tasks"])
    table, counts = [], {}
    for state in states:
        record = state["result"] or {"status": "pending_or_interrupted"}
        status = record["status"]
        counts[status] = counts.get(status, 0)+1
        item = {**state["task"], "status": status}
        if "metrics" in record:
            metric, cost = record["metrics"], record.get("cost", {})
            item.update({k: metric[k] for k in ("tp", "fp", "fn", "tpr", "fpr", "auc", "normalized_pauc_0_01")})
            item["scanning_tp"] = metric.get("families", {}).get("Scanning", {}).get("tp")
            item.update({"parameter_count": record.get("parameter_count"),
                         "deployment_bytes": cost.get("required_deployment_file_bytes"),
                         "p95_ms": cost.get("single", {}).get("p95_ms"),
                         "peak_rss_bytes": cost.get("rss", {}).get("sampled_peak_bytes"),
                         "selected_epoch": record.get("selection", {}).get("epoch")})
        table.append(item)
    pd.DataFrame(table).to_csv(root / "summary.csv", index=False)
    confirmation_path = root / "confirmation" / "summary.json"
    confirmation = read_json(confirmation_path) if confirmation_path.is_file() else {"status": "not_run"}
    summary = {"scope": spec["scope"], "counts": counts, "planned_tasks": len(states),
               "all_planned_tasks_complete": all(row["status"] == "complete" for row in table),
               "results": table, "unseen_group_confirmation": confirmation}
    write_json(root / "summary.json", summary)
    lines = ["# 一次批量路线对照", "", "当前结果使用已参与开发的 E；不作为独立确认。", "",
             "| 对照 | 状态 | TP | FP | Scanning TP | CPU p95/ms | 部署字节 |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append("| " + " | ".join(str(row.get(k, "—")) for k in
                     ("id", "status", "tp", "fp", "scanning_tp", "p95_ms", "deployment_bytes")) + " |")
    lines += ["", "同一数据上的三个训练种子表示训练随机性，不是三个独立数据验证。",
              "各臂独立正常校准；模型、容量、学习率与输入表示的作用按配对结果解释。",
              "训练错误、依赖不可用、未运行与没有检测收益分别报告。"]
    if confirmation_path.is_file():
        lines += ["", f"通信组留出确认状态：{confirmation['status']}。",
                  "确认结果见 [confirmation/summary.csv](confirmation/summary.csv)；新旧评价总体分开报告。"]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run(config, output_dir, *, resume=False, retry_unavailable=False):
    config, root = Path(config).resolve(), Path(output_dir).resolve()
    spec = validate(read_json(config))
    snapshot = source_snapshot()
    if root.exists():
        if not resume:
            raise FileExistsError("output exists; use --resume for the unchanged batch")
        if (canonical_hash(read_json(root / "protocol.json")) != canonical_hash(spec)
                or read_json(root / "source_snapshot.json")["source_sha256"] != snapshot["source_sha256"]):
            raise ValueError("resume requires identical protocol and source; preserve the old run")
    else:
        audit = preflight(config)
        root.mkdir(parents=True)
        write_json(root / "protocol.json", spec)
        write_json(root / "source_snapshot.json", snapshot)
        write_json(root / "preflight.json", audit)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")})
    for state in _task_states(root, spec["tasks"]):
        task, old = state["task"], state["result"]
        if old is not None and not (retry_unavailable and old["status"] == "unavailable"):
            continue
        folder = root / "tasks" / task["id"]
        folder.mkdir(parents=True, exist_ok=True)
        attempt_count = sum(p.is_dir() for p in folder.glob("attempt-*"))
        attempt = folder / f"attempt-{attempt_count+1:03d}"
        python = (spec.get("tabpfn_python") if task.get("model_kind") == "tabpfn25" else None) or sys.executable
        print(f"Starting {task['id']} -> {attempt.name}", flush=True)
        with (folder / f"{attempt.name}.log").open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run([python, "-m", "ids.route_batch", "_worker", "--config", str(config),
                    "--task", task["id"], "--output", str(attempt)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False)
                if not (attempt / "result.json").is_file():
                    attempt.mkdir(exist_ok=True)
                    write_json(attempt / "result.json", {"status": "failed", "task": task,
                        "error": "worker exited without a result", "exit_code": completed.returncode})
            except FileNotFoundError as exc:
                attempt.mkdir(exist_ok=True)
                write_json(attempt / "result.json", {"status": "unavailable", "task": task, "reason": str(exc)})
        summarize(root)
    return summarize(root)


def pack(root, target, *, include_models=False):
    """Analysis return ZIP by default; models remain at the laboratory for confirmation."""
    root, target = Path(root).resolve(), Path(target).resolve()
    if target.exists() or target.is_relative_to(root):
        raise ValueError("new archive must be outside the run directory")
    summarize(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file()
             and (include_models or p.suffix not in {".pt", ".joblib"})]
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError("archive contains an external file")
    manifest = {p.relative_to(root).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for p in files:
            archive.write(p, p.relative_to(root).as_posix())
        archive.writestr("archive_manifest.json", json.dumps({"include_models": include_models, "files": manifest},
                                                           ensure_ascii=False, indent=2))
    import hashlib
    with zipfile.ZipFile(target) as archive:
        for name, identity in manifest.items():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != identity["sha256"]:
                raise ValueError(f"archive verification failed: {name}")
    result = {"archive": str(target), "sha256": sha256_file(target), "files": len(files),
              "models_included": include_models}
    write_json(target.with_suffix(target.suffix+".sha256.json"), result)
    return result
