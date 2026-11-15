"""Resumable laboratory-only supervision diagnosis on already opened TQH cells."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
import hashlib
from itertools import combinations
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
from threadpoolctl import threadpool_limits

from ids.common import canonical_hash, environment_snapshot, read_json, sha256_file, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, pack, run_lock, stage
from ids.route_batch.metrics import threshold
from ids.route_batch.protocol import REPO, source_snapshot
from ids.threshold_validation.observations import RAW_FEATURES
from . import data
from .evaluation import evaluate_cell, oracle_cell, paired_cell


REFERENCE = REPO / "configs/response-evidence-20261112/r22-model-reference.json"
LEGACY_CALIBRATION = REPO / "configs/support-diagnostic-20261115/legacy-hgb-calibration.json"
RULE_ID = "reference-udp443-30s"
B_OR_ID = "reference-hgb-B-calibration-or-udp443-30s"
LEGACY_ID = "reference-hgb-original-C"
STRONG_ID = "reference-hgb-original-C-or-udp443-30s"


def tasks(reg):
    return [{"id": f"{support}-{view}-{group}", "support": support, "view": view, "group": group,
             "reuse": support == "source" and view == "raw8"}
            for support in reg["supports"] for view in reg["views"] for group in reg["groups"]]


def reference_tasks():
    return [{"id": RULE_ID, "support": "reference", "view": "observed8", "group": "rule", "reuse": True},
            {"id": B_OR_ID, "support": "reference", "view": "observed8", "group": "hgb_B_or_rule", "reuse": True},
            {"id": LEGACY_ID, "support": "reference", "view": "observed8", "group": "hgb_old_C", "reuse": True},
            {"id": STRONG_ID, "support": "reference", "view": "observed8", "group": "hgb_old_C_or_rule", "reuse": True}]


def plan(reg):
    arms = tasks(reg)
    return {"release": reg["version"], "status": reg["status"], "roles": reg["roles"],
            "arms": arms, "new_fits": sum(not a["reuse"] for a in arms),
            "reused_models": sum(a["reuse"] for a in arms),
            "fixed_references": reference_tasks(), "scoring_schemes": len(arms) + len(reference_tasks()),
            "evaluation_cells": len(reg["roles"]["forward_development"] + reg["roles"]["transport_external"]),
            "calibration": "same historical B normal identities for every arm; D uses frozen B rule",
            "label_availability": reg["label_availability"], "new_blind_confirmation": False}


def completed(root, name, binding):
    path = Path(root) / "receipts" / (name + ".json")
    if not path.is_file():
        return False
    receipt = read_json(path)
    if receipt.get("binding") != binding or not receipt.get("outputs"):
        raise ValueError(f"Stage registration changed: {name}")
    for relative, digest in receipt["outputs"].items():
        target = (Path(root) / relative).resolve()
        if not target.is_relative_to(Path(root).resolve()):
            raise ValueError("Receipt output escapes experiment root")
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Completed output changed: {name}/{relative}")
    return True


def inspect_reference(reference_run, groups):
    registered = read_json(REFERENCE)["artifacts"]
    found = {}
    for group in groups:
        members = registered[group + "-raw"]
        names = [name for name in members if name.startswith("models/")]
        if len(names) != 1:
            raise ValueError(f"Expected one frozen model for {group}")
        relative = names[0]
        path = Path(reference_run) / relative
        if not path.is_file() or sha256_file(path) != members[relative]:
            raise ValueError(f"Frozen R22/R23 raw model missing or changed: {path}")
        found[group] = {"path": str(path.resolve()), "relative": relative,
                        "sha256": members[relative], "bytes": path.stat().st_size}
    return found


def verify_runtime(environment):
    expected = read_json(REFERENCE)["runtime"]["packages"]
    differences = {name: {"expected": version, "actual": environment["packages"].get(name)}
                   for name, version in expected.items() if environment["packages"].get(name) != version}
    version = environment["python"].split(".")
    if version[:2] not in (["3", "11"], ["3", "12"]):
        differences["python"] = {"expected": "3.11.x or 3.12.x", "actual": environment["python"]}
    if differences:
        raise ValueError(f"Frozen model runtime differs: {differences}")


def resolve_inputs(raw_dir, selected):
    """Resolve each unique archive once before the independent parser workers start."""
    from ids.response_evidence import sources, tqh
    seen, result = set(), {}
    for spec in selected.values():
        identity = spec.get("archive", spec["name"])
        if identity in seen:
            continue
        seen.add(identity)
        path = (tqh._archive(spec, raw_dir) if spec.get("format") == "tqh"
                else sources.resolve(raw_dir, spec, data.legacy_data.read_registration()))
        result[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def code_snapshot():
    source = source_snapshot()
    configs = {}
    for path in sorted((REPO / "configs").rglob("*.json")):
        configs[path.relative_to(REPO).as_posix()] = hashlib.sha256(
            path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"source": source, "config_files": configs, "config_sha256": canonical_hash(configs)}


def verify_live_code(root, expected):
    if code_snapshot() != expected:
        write_json(Path(root) / "invalidated-run.json", {
            "reason": "Source/config changed during execution; worker results may mix implementations",
            "action": "Use a new output root after restoring a fixed checkout", "recorded_at": utc_now()})
        raise ValueError("Source/config changed during execution; this output cannot be resumed")


def checks(root):
    files = sorted((REPO / "tests").glob("test_support_diagnostic_*.py"))
    if not files:
        raise ValueError("No support-diagnostic synthetic checks found")
    log = Path(root) / "evidence/check.log"
    xml = Path(root) / "evidence/check.xml"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as out:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "--junitxml=" + str(xml), *map(str, files)], cwd=REPO,
                       stdout=out, stderr=subprocess.STDOUT, check=True)
    return [log, xml]


def _threads(threads):
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(max(1, min(4, threads)))


def prepare_worker(root, raw_dir, spec, threads, binding):
    _threads(threads)
    with threadpool_limits(limits=threads):
        stage(Path(root), "prepare-" + spec["name"], binding,
              partial(data.prepare, Path(root), raw_dir, spec))


def model_path(root, task):
    extension = ".joblib" if task["group"] == "hgb" else ".pt"
    return Path(root) / "models" / task["id"] / (task["group"] + extension)


def score_path(root, arm_id, source):
    return Path(root) / "local/scores" / arm_id / (source + ".parquet")


def _write_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return path


def _fit(root, task, refs, threads, device):
    from . import model
    path = model_path(root, task)
    if task["reuse"]:
        source = Path(refs[task["group"]]["path"])
        if sha256_file(source) != refs[task["group"]]["sha256"]:
            raise ValueError("Frozen reference changed after preflight")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)
        report = {"group": task["group"], "view": task["view"], "support": task["support"],
                  "status": "reused_frozen_model", "new_fits": 0,
                  "model_sha256": sha256_file(path), "reference_relative": refs[task["group"]]["relative"]}
        write_json(path.with_suffix(".fit.json"), report)
    else:
        frame = pd.read_parquet(Path(root) / "local/training" / (task["support"] + ".parquet"))
        model.fit(frame, task["view"], path, group=task["group"], weights=frame.weight.to_numpy(),
                  threads=threads, device=device)
        report = read_json(path.with_suffix(".fit.json"))
        report.update(support=task["support"], new_fits=1)
    evidence = Path(root) / "evidence/fits" / (task["id"] + ".json")
    write_json(evidence, report)
    return [path, path.with_suffix(".fit.json"), evidence]


def _blind(frame):
    return frame[["row_id", "source_id", "flow_end_ms", *RAW_FEATURES]]


def _calibrate(root, reg, task, fitted):
    from . import model
    frame = data.calibration_frame(root, reg)
    scores = model.score(fitted, _blind(frame), task["view"], task["group"])
    rule = threshold(scores, reg["alpha"])
    rule.update(ready=rule["threshold"] is not None, scope="historical_B_confirmed_normal_only",
                label_availability=reg["label_availability"], source_ids=reg["roles"]["normal_calibration"],
                latest_calibration_time_ms=int(frame.flow_end_ms.max()),
                identity_sha256=hashlib.sha256(frame[["source_id", "row_id"]].to_csv(
                    index=False, lineterminator="\n").encode()).hexdigest())
    out = Path(root) / "evidence/calibration" / (task["id"] + ".json")
    score_file = Path(root) / "local/calibration" / (task["id"] + ".parquet")
    _write_frame(score_file, frame[["source_id", "row_id"]].assign(score=scores))
    write_json(out, rule)
    return [out, score_file]


def _score(root, task, source, fitted):
    from . import model
    frame = data.frame(root, source)
    scores = np.asarray(model.score(fitted, _blind(frame), task["view"], task["group"]), dtype=np.float64)
    if scores.shape != (len(frame),) or not np.isfinite(scores).all():
        raise ValueError("Model scoring did not return finite row-aligned scores")
    return [_write_frame(score_path(root, task["id"], source), frame[["row_id"]].assign(score=scores))]


def _scopes(frame):
    yield "cell", "all", np.arange(len(frame))
    days = pd.to_datetime(frame.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    for day in sorted(days.unique()):
        yield "utc_day_within_cell", day, np.flatnonzero(days.eq(day))


def _services(frame, scores, rule):
    alerts = np.asarray(scores) > rule["threshold"] if rule.get("ready") else np.zeros(len(frame), bool)
    rows = []
    for (protocol, port), indices in frame.groupby(["ip_protocol", "resp_port"], sort=True).indices.items():
        subset = frame.iloc[indices]
        normal = subset.family.eq("Benign").to_numpy()
        c2 = subset.family.eq("ExternalC2").to_numpy()
        rows.append({"ip_protocol": int(protocol), "resp_port": int(port),
                     "normal_support": int(normal.sum()), "fp": int((alerts[indices] & normal).sum()),
                     "c2_support": int(c2.sum()), "c2_tp": int((alerts[indices] & c2).sum()),
                     "has_same_service_normal_and_c2": bool(normal.any() and c2.any())})
    return rows


def _read_scores(root, arm_id, source, frame):
    cached = pd.read_parquet(score_path(root, arm_id, source))
    if not np.array_equal(cached.row_id, frame.row_id):
        raise ValueError("Cached scores and truth have different physical row identities")
    return cached.score.to_numpy()


def _evaluate(root, task, source):
    frame = data.frame(root, source)
    scores = _read_scores(root, task["id"], source, frame)
    rule = read_json(Path(root) / "evidence/calibration" / (task["id"] + ".json"))
    scopes = []
    for scope, window, indices in _scopes(frame):
        part = frame.iloc[indices]
        values = scores[indices]
        scopes.append({"scope": scope, "window": window,
                       "registered": evaluate_cell(part, values, rule),
                       "oracle": oracle_cell(part, values),
                       "services": _services(part, values, rule)})
    path = Path(root) / "evidence/evaluation" / task["id"] / (source + ".json")
    write_json(path, {"task": task, "source_id": source, "scopes": scopes})
    return [path]


def _proxy(frame):
    """Previously developed observation rule; no labels, fitting or threshold search."""
    return (frame.ip_protocol.eq(17) & (frame.orig_port.eq(443) | frame.resp_port.eq(443))
            & frame.duration_seconds.ge(30)).to_numpy()


def _reference_alerts(frame, task, base_scores, base_rule):
    proxy = _proxy(_blind(frame))
    if task["id"] == RULE_ID:
        return proxy
    if task["id"] in {LEGACY_ID, STRONG_ID}:
        cutoff = read_json(LEGACY_CALIBRATION)["threshold"]
    else:
        if not base_rule["ready"]:
            raise ValueError("B-calibrated HGB reference is not ready")
        cutoff = base_rule["threshold"]
    base = np.asarray(base_scores) > cutoff
    return base if task["id"] == LEGACY_ID else base | proxy


def _reference_calibration(root, reg, task):
    frame = data.calibration_frame(root, reg)
    values, base = None, None
    if task["id"] != RULE_ID:
        values = pd.read_parquet(Path(root) / "local/calibration/source-raw8-hgb.parquet")
        if not values[["source_id", "row_id"]].equals(frame[["source_id", "row_id"]]):
            raise ValueError("Reference calibration identities differ")
        base = read_json(Path(root) / "evidence/calibration/source-raw8-hgb.json")
    alerts = _reference_alerts(frame, task, None if values is None else values.score.to_numpy(), base)
    rule = {"threshold": 0.5, "ready": True, "score_kind": "frozen_boolean_decision",
            "scope": "previously_developed_fixed_rule_not_quantile_fitted",
            "definition": "ip_protocol == 17 and (orig_port == 443 or resp_port == 443) and duration_seconds >= 30",
            "hgb_component": ("source-raw8-hgb with original frozen C threshold"
                              if task["id"] in {LEGACY_ID, STRONG_ID} else
                              "source-raw8-hgb with frozen B normal threshold" if task["id"] == B_OR_ID else None),
            "original_C_rule": read_json(LEGACY_CALIBRATION) if task["id"] in {LEGACY_ID, STRONG_ID} else None,
            "label_availability": reg["label_availability"], "n_normal": len(frame),
            "calibration_fp": int(alerts.sum())}
    path = Path(root) / "evidence/calibration" / (task["id"] + ".json")
    score_file = Path(root) / "local/calibration" / (task["id"] + ".parquet")
    _write_frame(score_file, frame[["source_id", "row_id"]].assign(score=alerts.astype(float)))
    write_json(path, rule)
    return [path, score_file]


def _reference_score(root, task, source):
    frame = data.frame(root, source)
    values, base = None, None
    if task["id"] != RULE_ID:
        values = _read_scores(root, "source-raw8-hgb", source, frame)
        base = read_json(Path(root) / "evidence/calibration/source-raw8-hgb.json")
    alerts = _reference_alerts(frame, task, values, base)
    return [_write_frame(score_path(root, task["id"], source), frame[["row_id"]].assign(score=alerts.astype(float)))]


def make_references(root, reg, binding):
    failures = []
    for task in reference_tasks():
        try:
            if task["id"] != RULE_ID and not completed(root, "calibrate-source-raw8-hgb", binding):
                raise ValueError("Strong reference awaits source HGB calibration")
            stage(root, "calibrate-" + task["id"], binding, partial(_reference_calibration, root, reg, task))
        except Exception:
            failures.append({"arm": task["id"], "phase": "reference_calibration", "trace": traceback.format_exc()})
            continue
        for source in reg["roles"]["forward_development"] + reg["roles"]["transport_external"]:
            try:
                if task["id"] != RULE_ID and not completed(root, "score-source-raw8-hgb-" + source, binding):
                    raise ValueError("Strong reference awaits source HGB scores")
                stage(root, "score-" + task["id"] + "-" + source, binding,
                      partial(_reference_score, root, task, source))
                stage(root, "evaluate-" + task["id"] + "-" + source, binding,
                      partial(_evaluate, root, task, source))
            except Exception:
                failures.append({"arm": task["id"], "source": source, "phase": "reference_score",
                                 "trace": traceback.format_exc()})
    for i, failure in enumerate(failures):
        write_json(Path(root) / "evidence/failures" / ("reference-" + str(i) + ".json"), failure)
    return failures


def arm_worker(root, reg, task, refs, threads, device, binding):
    """One load per arm, but separate receipts for fit, calibration and each cell."""
    from . import model
    root = Path(root)
    _threads(threads)
    started, cpu = time.perf_counter(), time.process_time()
    failures, fitted = [], None
    sources = [name for name in reg["roles"]["forward_development"] + reg["roles"]["transport_external"]
               if completed(root, "prepare-" + name, binding)]
    with threadpool_limits(limits=threads):
        try:
            stage(root, "fit-" + task["id"], binding, partial(_fit, root, task, refs, threads, device))
            needs_forward = not completed(root, "calibrate-" + task["id"], binding) or any(
                not completed(root, "score-" + task["id"] + "-" + source, binding) for source in sources)
            if needs_forward:
                fitted = model.load(model_path(root, task), group=task["group"], threads=threads, device=device)
            stage(root, "calibrate-" + task["id"], binding, partial(_calibrate, root, reg, task, fitted))
        except Exception:
            failure = {"arm": task["id"], "phase": "fit_or_calibration", "trace": traceback.format_exc()}
            write_json(root / "evidence/failures" / (task["id"] + "-fit.json"), failure)
            return [failure]
        for source in sources:
            try:
                stage(root, "score-" + task["id"] + "-" + source, binding,
                      partial(_score, root, task, source, fitted))
                stage(root, "evaluate-" + task["id"] + "-" + source, binding,
                      partial(_evaluate, root, task, source))
            except Exception:
                failure = {"arm": task["id"], "source": source, "phase": "score_or_evaluation",
                           "trace": traceback.format_exc()}
                write_json(root / "evidence/failures" / (task["id"] + "-" + source + ".json"), failure)
                failures.append(failure)
    write_json(root / "evidence/performance" / (task["id"] + ".json"), {
        "arm": task["id"], "pid": os.getpid(), "threads": threads, "device": device,
        "seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
        "rss_after_bytes": psutil.Process().memory_info().rss, "failures": len(failures),
        "measurement_scope": "concurrent laboratory throughput, not isolated deployment latency"})
    return failures


def comparison_pairs(arms):
    result = []
    learned = [arm for arm in arms if arm["support"] != "reference"]
    for left, right in combinations(learned, 2):
        same_support_comparison = left["group"] == right["group"] and left["view"] == right["view"]
        same_view_comparison = left["group"] == right["group"] and left["support"] == right["support"]
        if same_support_comparison or same_view_comparison:
            result.append((left, right))
    references = [arm for arm in arms if arm["support"] == "reference"]
    result.extend((reference, candidate) for reference in references for candidate in learned)
    result.extend(combinations(references, 2))
    return result


def _metric_row(task, source, entry, name):
    metrics = entry[name]
    row = {"arm": task["id"], "support": task["support"], "view": task["view"], "group": task["group"],
           "source_id": source, "scope": entry["scope"], "window": entry["window"],
           **{k: v for k, v in metrics.items() if k != "by_family"}}
    for family, value in metrics["by_family"].items():
        for field in ("n", "tp", "fn", "recall"):
            row[family + "_" + field] = value[field]
    return row


def collate(root, reg, binding, failures):
    """Produce compact self-contained replay inputs, all available comparisons and tables."""
    root = Path(root)
    arms = tasks(reg) + reference_tasks()
    sources = reg["roles"]["forward_development"] + reg["roles"]["transport_external"]
    metrics, oracles, pairs_table, services = [], [], [], []
    complete_evaluations = 0
    for source in sources:
        if not completed(root, "prepare-" + source, binding):
            continue
        frame = data.frame(root, source)
        replay = frame[["source_id", "row_id", "flow_end_ms", "family", *RAW_FEATURES]].copy()
        available, rules, scores = {}, {}, {}
        for task in arms:
            arm = task["id"]
            if not completed(root, "score-" + arm + "-" + source, binding):
                continue
            scores[arm] = _read_scores(root, arm, source, frame)
            replay[arm] = scores[arm]
            rules[arm] = read_json(root / "evidence/calibration" / (arm + ".json"))
            available[arm] = task
            if completed(root, "evaluate-" + arm + "-" + source, binding):
                complete_evaluations += 1
                report = read_json(root / "evidence/evaluation" / arm / (source + ".json"))
                for entry in report["scopes"]:
                    metrics.append(_metric_row(task, source, entry, "registered"))
                    oracles.append(_metric_row(task, source, entry, "oracle"))
                    services.extend({"arm": arm, "source_id": source, "scope": entry["scope"],
                                     "window": entry["window"], **service} for service in entry["services"])
        _write_frame(root / "evidence/replay" / (source + ".parquet"), replay)
        comparisons = []
        for left, right in comparison_pairs(list(available.values())):
            for scope, window, indices in _scopes(frame):
                pair = paired_cell(frame.iloc[indices], scores[left["id"]][indices], rules[left["id"]],
                                   scores[right["id"]][indices], rules[right["id"]])
                header = {"source_id": source, "scope": scope, "window": window,
                          "left_arm": left["id"], "right_arm": right["id"]}
                comparisons.append({**header, **pair})
                for family, change in {"all_attacks": pair["attacks"], "normal": pair["normal"],
                                       "unresolved": pair["unresolved"], **pair["by_family"]}.items():
                    pairs_table.append({**header, "family": family, **change,
                                        "left_qualified": pair["left_qualified"],
                                        "right_qualified": pair["right_qualified"],
                                        "both_qualified": pair["both_qualified"]})
        write_json(root / "evidence/paired" / (source + ".json"), {
            "available_arms": list(available), "expected_arms": len(arms), "comparisons": comparisons})
    calibration = data.calibration_frame(root, reg)
    replay_cal = calibration[["source_id", "row_id", "flow_end_ms", "family", *RAW_FEATURES]].copy()
    for task in arms:
        if not completed(root, "calibrate-" + task["id"], binding):
            continue
        values = pd.read_parquet(root / "local/calibration" / (task["id"] + ".parquet"))
        if not values[["source_id", "row_id"]].equals(calibration[["source_id", "row_id"]]):
            raise ValueError("Calibration cache identity differs from registered normal set")
        replay_cal[task["id"]] = values.score.to_numpy()
    _write_frame(root / "evidence/replay/calibration.parquet", replay_cal)
    identities = []
    for support in reg["supports"]:
        rows = pd.read_parquet(root / "local/training" / (support + ".parquet"),
                               columns=["source_id", "row_id", "family", "domain", "label", "weight"])
        identities.append(rows.assign(support=support))
    _write_frame(root / "evidence/replay/training-identities.parquet", pd.concat(identities, ignore_index=True))
    for name, rows in (("metrics", metrics), ("oracles", oracles), ("paired", pairs_table), ("services", services)):
        pd.DataFrame(rows).to_csv(root / "evidence" / (name + ".csv.gz"), index=False, compression="gzip")
    expected = len(arms) * len(sources)
    success = complete_evaluations == expected and not failures
    verify_live_code(root, read_json(root / "protocol.json")["code"])
    write_json(root / "outcome.json", {
        "status": "complete" if success else "incomplete", "release": reg["version"],
        "research_status": "already_opened_development_diagnostic", "expected_evaluations": expected,
        "completed_evaluations": complete_evaluations, "failures": failures,
        "planned_new_fits": sum(not t["reuse"] for t in tasks(reg)),
        "reused_reference_models": sum(t["reuse"] for t in tasks(reg)),
        "fixed_rule_references": len(reference_tasks()), "common_strong_reference": STRONG_ID,
        "new_blind_confirmations": 0, "label_availability": reg["label_availability"],
        "compact_replay": "evidence/replay retains raw numeric inputs, identities, author family and all available scores; no IPs",
        "interpretation": ["Completion means execution completeness, not detection success.",
                           "D is later Merlin transport transfer, not an independent attack mechanism.",
                           "Missing Scanning/Download support is not a successful endpoint.",
                           "Oracle uses evaluation normal labels and is diagnostic only.",
                           "Report every cell and UTC-day scope separately; never sum overlapping scopes."]})
    return success


def _pool(workers):
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _run(root, reference_run, raw_dir, threads, cpu_workers, gpu_workers, prep_workers,
         preflight_only, synthetic, pool_factory):
    if (root / "invalidated-run.json").is_file():
        raise ValueError("This run was invalidated by an execution-time source/config change; use a new output root")
    reg = data.registration()
    if synthetic and reg.get("status") != "synthetic":
        raise ValueError("CPU synthetic mode requires an explicitly synthetic registration, never the real TQH batch")
    selected = data.specs(reg)
    env = environment_snapshot(include_torch=True)
    if not synthetic and not env.get("torch", {}).get("cuda_available"):
        raise RuntimeError("Formal laboratory execution requires CUDA; CPU is reserved for explicit synthetic checks")
    if not synthetic:
        verify_runtime(env)
    if min(threads, cpu_workers, gpu_workers, prep_workers) < 1 or threads < cpu_workers + gpu_workers:
        raise ValueError("Positive worker counts and at least one CPU thread per concurrent worker are required")
    gpu_threads = max(1, min(2, threads // (cpu_workers + gpu_workers)))
    cpu_threads = max(1, (threads - gpu_threads * gpu_workers) // cpu_workers)
    prep_workers = min(prep_workers, threads, len(selected))
    refs = inspect_reference(reference_run, reg["groups"])
    if not synthetic and read_json(LEGACY_CALIBRATION)["model_sha256"] != refs["hgb"]["sha256"]:
        raise ValueError("Original C rule is not bound to the supplied frozen HGB")
    inputs = resolve_inputs(raw_dir, selected)
    preflight = {"status": "verified", "reference_models": {
        group: {k: v for k, v in info.items() if k != "path"} for group, info in refs.items()},
        "raw_inputs": inputs, "source_count": len(selected), "cuda_required": not synthetic}
    if preflight_only:
        stamp = time.time_ns()
        write_json(root / "evidence/attempts" / (f"preflight-{stamp}.json"), {
            "status": "preflight_only", "preflight": preflight, "environment": env, "plan": plan(reg),
            "fits": 0, "model_forward_calls": 0})
        if not (root / "protocol.json").exists():
            write_json(root / "evidence/preflight.json", preflight)
            write_json(root / "environment.json", env)
            write_json(root / "plan.json", plan(reg))
            write_json(root / "outcome.json", {"status": "preflight_only", "fits": 0, "model_forward_calls": 0,
                                               "source_files_verified": len(inputs)})
        return True
    protocol = {"registration": reg, "source_specs": selected, "code": code_snapshot(),
                "input_binding": preflight, "runtime": {k: env[k] for k in ("python", "packages")},
                "scheduling": {"cpu_workers": cpu_workers, "gpu_workers": gpu_workers,
                               "cpu_threads": cpu_threads, "gpu_threads": gpu_threads,
                               "prep_workers": prep_workers, "prep_threads": max(1, threads // prep_workers)},
                "synthetic": synthetic}
    binding = canonical_hash(protocol)
    if (root / "protocol.json").exists() and canonical_hash(read_json(root / "protocol.json")) != binding:
        raise ValueError("Resume requires identical source/config, inputs, runtime and scheduling; use a new output root")
    write_json(root / "protocol.json", protocol)
    write_json(root / "evidence/preflight.json", preflight)
    write_json(root / "environment.json", env)
    write_json(root / "plan.json", plan(reg))
    failures = []
    with ResourceLog(root) as monitor:
        if not synthetic:
            monitor.activity = "synthetic_checks"
            stage(root, "checks", binding, partial(checks, root))
        monitor.activity = "parallel_input_preparation"
        verify_live_code(root, protocol["code"])
        pending = [spec for spec in selected.values() if not completed(root, "prepare-" + spec["name"], binding)]
        if pending:
            with pool_factory(prep_workers) as pool:
                jobs = {pool.submit(prepare_worker, root, raw_dir, spec, max(1, threads // prep_workers), binding):
                        spec["name"] for spec in pending}
                for future in as_completed(jobs):
                    try:
                        future.result()
                    except Exception:
                        failure = {"source": jobs[future], "phase": "prepare", "trace": traceback.format_exc()}
                        failures.append(failure)
                        write_json(root / "evidence/failures" / ("prepare-" + jobs[future] + ".json"), failure)
        required = reg["source_train"] + reg["roles"]["target_train"] + reg["roles"]["normal_calibration"]
        if any(not completed(root, "prepare-" + name, binding) for name in required):
            write_json(root / "outcome.json", {"status": "incomplete", "phase": "prepare", "failures": failures,
                                               "fits": 0, "model_forward_calls": 0})
            return False
        stage(root, "training-support", binding, partial(data.training_sets, root, reg))
        monitor.activity = "parallel_CPU_GPU_fit_calibration_and_score"
        verify_live_code(root, protocol["code"])
        cells = [name for name in reg["roles"]["forward_development"] + reg["roles"]["transport_external"]
                 if completed(root, "prepare-" + name, binding)]
        pending = []
        for task in tasks(reg):
            stages = ["fit-" + task["id"], "calibrate-" + task["id"]]
            stages += [kind + "-" + task["id"] + "-" + name for name in cells for kind in ("score", "evaluate")]
            statuses = [completed(root, name, binding) for name in stages]
            if not all(statuses):
                pending.append(task)
        with ExitStack() as stack:
            cpu_jobs = [task for task in pending if task["group"] == "hgb"]
            gpu_jobs = [task for task in pending if task["group"] != "hgb"]
            queues = []
            if cpu_jobs:
                queues.append((stack.enter_context(pool_factory(cpu_workers)), cpu_jobs, cpu_threads, "cpu"))
            if gpu_jobs:
                queues.append((stack.enter_context(pool_factory(gpu_workers)), gpu_jobs, gpu_threads,
                               "cpu" if synthetic else "cuda:0"))
            jobs = {pool.submit(arm_worker, root, reg, task, refs, nthreads, device, binding): task["id"]
                    for pool, queue_tasks, nthreads, device in queues for task in queue_tasks}
            for future in as_completed(jobs):
                try:
                    failures.extend(future.result())
                except Exception:
                    failure = {"arm": jobs[future], "phase": "worker", "trace": traceback.format_exc()}
                    failures.append(failure)
                    write_json(root / "evidence/failures" / (jobs[future] + "-worker.json"), failure)
        monitor.activity = "paired_evaluation_and_compact_replay"
        verify_live_code(root, protocol["code"])
        failures.extend(make_references(root, reg, binding))
        return collate(root, reg, binding, failures)


def run(output, reference_run, *, raw_dir=None, threads=8, cpu_workers=2, gpu_workers=2,
        prep_workers=4, preflight_only=False, synthetic=False, pool_factory=_pool):
    root, reference = Path(output).resolve(), Path(reference_run).resolve()
    if root == reference or root.is_relative_to(reference):
        raise ValueError("New output must not modify or nest inside the frozen reference run")
    root.mkdir(parents=True, exist_ok=True)
    with run_lock(root):
        try:
            success = _run(root, reference, raw_dir, threads, cpu_workers, gpu_workers, prep_workers,
                           preflight_only, synthetic, pool_factory)
            write_json(root / "last_attempt.json", {"status": "preflight_only" if preflight_only else
                                                    "complete" if success else "incomplete",
                                                    "recorded_at": utc_now()})
            return success
        except Exception as exc:
            failure = {"status": "failed", "trace": traceback.format_exc(), "recorded_at": utc_now(),
                       "preserved_prior_outcome": (root / "outcome.json").exists()}
            write_json(root / "last_attempt.json", failure)
            write_json(root / "evidence/attempts" / (f"failure-{time.time_ns()}.json"), failure)
            if not (root / "outcome.json").exists():
                write_json(root / "outcome.json", failure)
            print(f"[failed] {exc}; details: {root / 'last_attempt.json'}", file=sys.stderr, flush=True)
            return False
        finally:
            pack(root, prefix="support-diagnostic-results")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "local/support-diagnostic-r26")
    parser.add_argument("--threads", type=int, default=max(4, psutil.cpu_count(logical=True) or 4))
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--gpu-workers", type=int, default=2)
    parser.add_argument("--prep-workers", type=int, default=4)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.plan:
        print(json.dumps(plan(data.registration()), indent=2, ensure_ascii=False))
        return 0
    if args.check:
        checks(args.output)
        print(f"Synthetic checks passed: {args.output / 'evidence/check.xml'}")
        return 0
    if args.reference_run is None:
        parser.error("--reference-run is required for preflight or laboratory execution")
    own_windows_workers()
    return 0 if run(args.output, args.reference_run, raw_dir=args.raw_dir, threads=args.threads,
                   cpu_workers=args.cpu_workers, gpu_workers=args.gpu_workers, prep_workers=args.prep_workers,
                   preflight_only=args.preflight_only) else 1
