"""Three engineering follow-up tasks; original TabM and confirmation stay read-only."""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

import pandas as pd

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, write_json
from ids.context_cache.preprocessing import labels
from .data import load_inputs
from .foundation import TABPFN_CHECKPOINT_SHA256, TABPFN_CHECKPOINT, FoundationUnavailable
from .protocol import REPO, source_snapshot, validate

FORMAT = "tracelight-route-followup-v1"
ORIGINAL_SOURCE = "67f931d0da7ac59bbbdb6de8dc04fdeb29b800a4a814ece9c3eb55cee4c7b047"
TASKS = ("tabicl_official_1", "tabicl_official_8", "tabpfn25_official_8")


def _separate(original, output):
    if output.is_relative_to(original) or original.is_relative_to(output):
        raise ValueError("follow-up output must be separate from the original run")


def _read_original(original):
    original = Path(original).resolve()
    spec = validate(read_json(original / "protocol.json"))
    source = read_json(original / "source_snapshot.json")
    if canonical_hash(source["files"]) != source["source_sha256"]:
        raise ValueError("original source snapshot is internally inconsistent")
    if spec["data_kind"] == "development" and source["source_sha256"] != ORIGINAL_SOURCE:
        raise ValueError("follow-up requires the registered route-batch-lab-v1 source")
    preflight = read_json(original / "preflight.json")
    if (preflight["protocol_sha256"] != canonical_hash(spec)
            or preflight.get("source") != source):
        raise ValueError("original protocol or source differs from its preflight identity")
    return spec


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("original run must not contain external links")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {"files": files, "identity": canonical_hash(files)}


def _preparation(report_path):
    if report_path is None or not Path(report_path).is_file():
        return {"status": "unavailable", "stage": "preparation_report", "reason": "preparation report is missing"}
    path = Path(report_path).resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("expected an object")
    except (OSError, UnicodeError, ValueError) as exc:
        return {"status": "unavailable", "stage": "preparation_report", "reason": f"cannot read preparation report: {exc}"}
    return {"status": "ready" if value.get("status") == "ready" else "unavailable",
            "stage": value.get("failed_stage", value.get("stage", "preparation")),
            "reason": value.get("error", value.get("reason", value.get("status"))),
            "report": {"path": str(path), "sha256": sha256_file(path)}, "details": value}


def _tabpfn_readiness(python, checkpoint, preparation_report):
    prepared = _preparation(preparation_report)
    if prepared["status"] != "ready":
        return prepared
    if not Path(python).is_file():
        return {"status": "unavailable", "stage": "interpreter", "reason": str(python)}
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        return {"status": "unavailable", "stage": "checkpoint", "reason": str(checkpoint)}
    if checkpoint.name != TABPFN_CHECKPOINT or sha256_file(checkpoint) != TABPFN_CHECKPOINT_SHA256:
        return {"status": "unavailable", "stage": "checkpoint_identity", "reason": "registered filename/SHA differs"}
    return prepared


def preflight(original_run, output, *, tabpfn_python, tabpfn_checkpoint, preparation_report=None):
    original, root = Path(original_run).resolve(), Path(output).resolve()
    _separate(original, root)
    for name in (tabpfn_python, tabpfn_checkpoint):
        if Path(name).resolve().is_relative_to(original):
            raise ValueError("TabPFN environment and checkpoint must stay outside the original run")
    spec = _read_original(original)
    inputs = load_inputs(original / "protocol.json")
    if spec["data_kind"] == "development":
        import torch
        from ids.context_cache.preprocessing import require_pinned_preprocessing
        require_pinned_preprocessing()
        if not torch.cuda.is_available():
            raise ValueError("original TabICL build-path diagnostics require laboratory CUDA")
    return {"original_run": str(original), "output": str(root), "tasks": list(TASKS),
            "original_protocol_sha256": sha256_file(original / "protocol.json"),
            "original_source_sha256": read_json(original / "source_snapshot.json")["source_sha256"],
            "data_audit": inputs["audit"], "real_model_loads": 0,
            "tabpfn": _tabpfn_readiness(tabpfn_python, tabpfn_checkpoint, preparation_report),
            "scope": "old P/C/development E only; no TabM fitting or confirmation evaluation"}


def _environment():
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    return env


def _latest(root, task):
    folder = root / "tasks" / task
    attempts = sorted(p for p in folder.glob("attempt-*") if p.is_dir()) if folder.exists() else []
    current = attempts[-1] if attempts else None
    result = read_json(current / "result.json") if current and (current / "result.json").is_file() else None
    return folder, current, result


def _cpu_cost(config, directory):
    # Full C/E replay and timing run in separate fresh processes so the cost
    # worker's memory excludes calibration frames and diagnostic arrays.
    report = directory / "cold_replay_report.json"
    with (directory / "full-replay-worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.run([sys.executable, "-m", "ids.route_batch", "_followup-full-replay",
            "--config", str(config), "--output", str(directory)], env=_environment(),
            stdout=log, stderr=subprocess.STDOUT, check=False)
    full_replay = read_json(report) if report.is_file() else None
    if child.returncode or not full_replay or full_replay.get("status") != "matched":
        return {"status": "replay_failed", "exit_code": child.returncode,
                "full_replay": full_replay, "log": str(directory / "full-replay-worker.log")}
    with (directory / "cost-worker.log").open("w", encoding="utf-8") as log:
        child = subprocess.run([sys.executable, "-m", "ids.route_batch", "_cost-worker",
            "--config", str(config), "--output", str(directory)], env=_environment(),
            stdout=log, stderr=subprocess.STDOUT, check=False)
    if child.returncode or not (directory / "cost.json").is_file():
        return {"status": "failed", "exit_code": child.returncode, "full_replay": full_replay,
                "replay": read_json(directory / "replay_diagnostic.json")
                if (directory / "replay_diagnostic.json").is_file() else None,
                "log": str(directory / "cost-worker.log")}
    return {"status": "complete", "full_replay": full_replay,
            "measurements": read_json(directory / "cost.json")}


def _tabpfn_task(binding, output):
    from .foundation import fit_foundation
    from .preprocessing import raw_numeric
    from .runner import _write_scores
    config = Path(binding["original_run"]) / "protocol.json"
    inputs = load_inputs(config)
    schema, frames = inputs["schema"], inputs["frames"]
    model = fit_foundation("tabpfn25", raw_numeric(frames["support"], schema),
        labels(frames["support"], schema), checkpoint_path=binding["tabpfn_checkpoint"],
        seed=inputs["spec"]["settings"]["foundation_seed"], device="cpu", n_estimators=8)
    model.save(output / "model.joblib")
    descriptor = {"kind": "foundation", "model_file": "model.joblib", "rebuild_device": "cpu",
        "model_sha256": sha256_file(output / "model.joblib"), "metadata": model.metadata,
        "score_contract": model.metadata["score"],
        "scope": "follow-up CPU-build/CPU-score system on opened development E; not original confirmation"}
    predictions = {role: model.predict_outputs(raw_numeric(frames[role], schema))
                   for role in ("calibration", "evaluation")}
    metrics = _write_scores(output, inputs, {role: value["scores"] for role, value in predictions.items()}, descriptor)
    for role, values in predictions.items():
        pd.DataFrame({"row_id": frames[role][schema.id_column].astype(str),
            "probability_normal": values["probabilities"][:, 0],
            "probability_attack": values["probabilities"][:, 1]}).to_csv(output / f"{role}_probabilities.csv", index=False)
    write_json(output / "deployment.json", descriptor)
    del model
    gc.collect()
    cost = _cpu_cost(config, output)
    return {"status": "complete" if cost["status"] == "complete" else "measurement_failed",
            "metrics": metrics, "cost": cost, "model": descriptor,
            "confirmation_evaluated": False, "build_device": "cpu", "score_device": "cpu"}


def worker(run_dir, task_id, output):
    if task_id not in TASKS:
        raise ValueError("only the two TabICL diagnostics and TabPFN are allowed")
    root, output = Path(run_dir).resolve(), Path(output).resolve()
    binding = read_json(root / "binding.json")
    if source_snapshot()["source_sha256"] != binding["followup_source_sha256"]:
        raise ValueError("follow-up source changed after registration")
    _separate(Path(binding["original_run"]), output)
    if not output.is_relative_to(root / "tasks" / task_id):
        raise ValueError("worker output must be in the registered task directory")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "environment.json", environment_snapshot())
    try:
        if task_id.startswith("tabicl_"):
            from .replay import run_replay_diagnostic
            diagnostic = run_replay_diagnostic(binding["original_run"], task_id, output / "diagnostic")
            cpu = output / "diagnostic" / "cpu"
            cpu_ready = (cpu / "deployment.json").is_file() and (cpu / "evaluation.csv").is_file()
            cost = (_cpu_cost(Path(binding["original_run"]) / "protocol.json", cpu) if cpu_ready
                    else {"status": "unavailable", "reason": "CPU diagnostic path did not produce a deployment"})
            paths_complete = all(diagnostic.get("paths", {}).get(path, {}).get("status") == "complete"
                                 for path in ("original", "cpu"))
            result = {"status": "complete" if paths_complete and cost["status"] == "complete" else "diagnostic_failed",
                      "diagnostic": diagnostic, "cpu_cost": cost,
                      "scope": "replay differences remain visible; CPU cost uses its own same-path scores"}
        else:
            ready = _tabpfn_readiness(binding["tabpfn_python"], binding["tabpfn_checkpoint"],
                                      root / "preparation" / "current.json")
            result = _tabpfn_task(binding, output) if ready["status"] == "ready" else ready
    except FoundationUnavailable as exc:
        result = {"status": "unavailable", "error": str(exc), "stage": "foundation_dependency"}
    except Exception as exc:
        result = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__,
                  "traceback": traceback.format_exc()}
    result.update(task=task_id, tabm_training_calls=0, confirmation_evaluated=False)
    write_json(output / "result.json", result)
    return result


def _followup_binding(root):
    path = root / "binding.json"
    if not path.is_file():
        raise ValueError("not a registered follow-up run: binding.json is missing")
    binding = read_json(path)
    if binding.get("format") != FORMAT or binding.get("tasks") != list(TASKS):
        raise ValueError("not a registered follow-up run")
    _separate(Path(binding["original_run"]).resolve(), root)
    return binding


def summarize(root):
    root = Path(root).resolve()
    _followup_binding(root)
    rows = []
    for task in TASKS:
        _, current, result = _latest(root, task)
        rows.append({"task": task, "attempt": str(current) if current else None,
                     "status": result.get("status") if result else "pending_or_interrupted"})
    preservation = read_json(root / "original_preservation.json") if (root / "original_preservation.json").is_file() else {}
    complete = all(row["status"] == "complete" for row in rows) and preservation.get("unchanged") is True
    result = {"format": FORMAT, "all_tasks_complete": complete, "tasks": rows,
              "original_preservation": preservation, "tabm_training_calls": 0,
              "confirmation_evaluated": False,
              "scope": "engineering follow-up and TabPFN old-development-E evaluation; no new confirmation"}
    write_json(root / "summary.json", result)
    pd.DataFrame(rows).to_csv(root / "summary.csv", index=False)
    return result


def run(original_run, output, *, tabpfn_python, tabpfn_checkpoint, preparation_report=None, resume=False):
    original, root = Path(original_run).resolve(), Path(output).resolve()
    audit = preflight(original, root, tabpfn_python=tabpfn_python, tabpfn_checkpoint=tabpfn_checkpoint,
                      preparation_report=preparation_report)
    identity = {"format": FORMAT, "original_run": str(original),
        "original_protocol_sha256": sha256_file(original / "protocol.json"),
        "original_source_sha256": read_json(original / "source_snapshot.json")["source_sha256"],
        "followup_source_sha256": source_snapshot()["source_sha256"], "tasks": list(TASKS),
        "tabpfn_python": str(Path(tabpfn_python).resolve()), "tabpfn_checkpoint": str(Path(tabpfn_checkpoint).resolve()),
        "tabpfn_build_device": "cpu", "tabpfn_checkpoint_sha256": TABPFN_CHECKPOINT_SHA256,
        "scope": "same old P/C/E; original TabM and opened confirmation are never rerun"}
    before = _inventory(original)
    if root.exists():
        if not resume:
            raise FileExistsError("follow-up output exists; use --resume for the same binding")
        if read_json(root / "binding.json") != identity:
            raise ValueError("follow-up resume requires identical original run, paths and repair source")
        if read_json(root / "original_files.json") != before:
            raise ValueError("original run changed since follow-up registration")
    else:
        root.mkdir(parents=True)
        write_json(root / "binding.json", identity)
        write_json(root / "original_files.json", before)
        write_json(root / "source_snapshot.json", source_snapshot())
    prior = list((root / "preparation").glob("attempt-*.json"))
    prepared = _preparation(preparation_report)
    details = prepared.get("details", prepared)
    write_json(root / "preparation" / f"attempt-{len(prior)+1:03d}.json", details)
    write_json(root / "preparation" / "current.json", details)
    write_json(root / "preflight.json", audit)
    try:
        for task in TASKS:
            folder, _, old = _latest(root, task)
            if old and old["status"] == "complete":
                continue
            folder.mkdir(parents=True, exist_ok=True)
            number = sum(p.is_dir() for p in folder.glob("attempt-*")) + 1
            attempt = folder / f"attempt-{number:03d}"
            print(f"Follow-up {task} -> {attempt.name}; no TabM task is dispatched", flush=True)
            readiness = (_tabpfn_readiness(tabpfn_python, tabpfn_checkpoint, root / "preparation" / "current.json")
                         if task == "tabpfn25_official_8" else None)
            if task == "tabpfn25_official_8" and readiness["status"] != "ready":
                attempt.mkdir()
                write_json(attempt / "result.json", {**readiness, "task": task})
                continue
            python = str(tabpfn_python) if task == "tabpfn25_official_8" else sys.executable
            with (folder / f"{attempt.name}.log").open("w", encoding="utf-8") as log:
                try:
                    child = subprocess.run([python, "-m", "ids.route_batch", "_followup-worker",
                        "--run-dir", str(root), "--task", task, "--output", str(attempt)], env=_environment(),
                        stdout=log, stderr=subprocess.STDOUT, check=False)
                    failure = {"status": "failed", "task": task, "exit_code": child.returncode,
                               "error": "worker returned without a result"}
                    if child.returncode and (attempt / "result.json").is_file():
                        worker_result = read_json(attempt / "result.json")
                        write_json(attempt / "worker-result-before-exit-check.json", worker_result)
                        write_json(attempt / "result.json", {"status": "failed", "task": task,
                            "exit_code": child.returncode, "error": "worker exited with a nonzero status",
                            "worker_result": worker_result})
                except OSError as exc:
                    failure = {"status": "unavailable", "task": task, "stage": "interpreter_start", "error": str(exc)}
            if not (attempt / "result.json").is_file():
                attempt.mkdir(exist_ok=True)
                write_json(attempt / "result.json", failure)
            summarize(root)
    finally:
        after = _inventory(original)
        write_json(root / "original_preservation.json", {"unchanged": before == after,
            "before_identity": before["identity"], "after_identity": after["identity"],
            "original_files": len(before["files"]), "scope": "all original run files, including TabM and confirmation"})
    result = summarize(root)
    if not result["original_preservation"]["unchanged"]:
        raise ValueError("original run changed while follow-up was running; inspect preservation receipt")
    return result


def pack(root, target):
    root, target = Path(root).resolve(), Path(target).resolve()
    binding = _followup_binding(root)
    if target.is_relative_to(Path(binding["original_run"]).resolve()):
        raise ValueError("result archive must stay outside the original run")
    if target.exists() or target.is_relative_to(root):
        raise ValueError("new result archive must be outside follow-up run")
    summarize(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() not in
                   {".pt", ".pth", ".ckpt", ".safetensors", ".joblib"})
    manifest = {}
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("external file in follow-up results")
            name = path.relative_to(root).as_posix()
            manifest[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            archive.write(path, name)
        archive.writestr("archive_manifest.json", json.dumps({"files": manifest, "models_included": False}, indent=2))
    import hashlib
    with zipfile.ZipFile(target) as archive:
        for name, item in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != item["sha256"]:
                raise ValueError(f"archive member mismatch: {name}")
    receipt = {"archive": str(target), "sha256": sha256_file(target), "files": len(files),
               "models_included": False, "tabm_rerun": False}
    write_json(target.with_suffix(target.suffix + ".sha256.json"), receipt)
    return receipt
