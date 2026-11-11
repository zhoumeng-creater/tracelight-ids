"""Prepare both cohorts and evaluate the registered arms without live model selection."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from .cohort import prepare_cohorts, resolve_source
from .evaluation import measure, paired_comparison, stratify, syn_cohort
from .execution import stage
from .features import HISTORY_COLUMNS, extract_history


ARMS = {"mlp48": "mlp", "mlp52_zero": "mlp", "mlp52_history": "mlp",
        "hgb48": "hgb", "hgb52_history": "hgb"}
ROLES = ("P", "C", "E", "P_time", "C_time", "E_time")
REFERENCE_HASHES = {
    "reference_full": ("93ab29743d116231e8202244512f5c0d42206c5f666a1377f7d9c8211efd5675",
                       "d019bb61e3150f79e008839e1cd27894fbfc08182f91a4bd32c08c70f143037a"),
    "reference_qcoc": ("a15268e46492f517d63a99555ae3770e08cf3925db648ed5cdcedabc793fc823",
                       "46602bd4cd69fbf2ab78a4aade349d01be4cd04c0c64ce369dbdff03d40675c2"),
}


def evidence(root, name):
    dest = root / "evidence" / name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def inputs(path, arm, roles):
    features = list(FEATURES)
    if "52" in arm:
        features.extend(HISTORY_COLUMNS)
    frames = [pd.read_parquet(path / (role + ".parquet")) for role in roles]
    if arm.endswith("_zero"):
        for frame in frames:
            frame[list(HISTORY_COLUMNS)] = 0.0
    return features, frames


def prepare(root, original, source_option, threads):
    source = resolve_source(original, source_option, work=root / "source")
    work = root / "cohort"
    data = prepare_cohorts(original, source, work, threads)
    out = evidence(root, "inputs")
    shutil.copyfile(work / "cohort_manifest.json", out / "cohort_manifest.json")
    concat = pd.concat([data[k].assign(role=k) for k in ROLES if len(data[k])], ignore_index=True)
    if concat.row_id.duplicated().any():
        raise ValueError("A target occurs in two registered roles")
    concat.to_parquet(root / "targets.parquet", index=False)
    write_json(root / "resolved_source.json", {"path": str(source)})
    return [root / "targets.parquet", root / "resolved_source.json", out / "cohort_manifest.json"]


def augment(root, threads):
    from .signal import diagnose
    concat = pd.read_parquet(root / "targets.parquet")
    temp = root / "history-work"
    obs = extract_history(Path(read_json(root / "resolved_source.json")["path"]), concat, temp, threads)
    meta = pd.read_parquet(temp / "target_metadata.parquet")
    if not np.array_equal(obs.row_id, concat.row_id):
        raise ValueError("Historical feature rows do not match the registered targets")
    merged = concat.merge(obs, on="row_id", validate="one_to_one").merge(meta, on="row_id", validate="one_to_one")
    if not np.array_equal(merged.row_id, concat.row_id):
        raise ValueError("Metadata dropped or reordered some targets")
    hosts = {v: hashlib.sha256(v.encode()).hexdigest()[:16] for v in merged.source_host.unique()}
    merged["host_key"] = merged.source_host.map(hosts)
    merged["end_day_utc"] = pd.to_datetime(merged.end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    merged = merged.drop(columns=["source_host"])
    output = evidence(root, "inputs")
    features_out = output / "registered_history.parquet"
    merged[["row_id", "role", "host_key", "end_ms", "end_day_utc", *HISTORY_COLUMNS]].to_parquet(
        features_out, index=False)
    shutil.copyfile(temp / "history_audit.json", output / "history_audit.json")
    shutil.copyfile(temp / "benchmark_segment.json", output / "benchmark_segment.json")
    paths = [features_out, output / "history_audit.json", output / "benchmark_segment.json",
             temp / "benchmark_segment.json"]
    if read_json(temp / "benchmark_segment.json")["status"] != "unavailable":
        paths.extend([temp / "benchmark_stream.parquet", temp / "benchmark_warmup.parquet"])
    (root / "inputs").mkdir(exist_ok=True)
    signals = {}
    for role in ROLES:
        path = root / "inputs" / (role + ".parquet")
        data = merged.loc[merged.role.eq(role)].reset_index(drop=True)
        data.to_parquet(path, index=False)
        signals[role] = diagnose(data)
        paths.append(path)
    write_json(output / "history_signal.json", signals)
    paths.append(output / "history_signal.json")
    return paths


def library(arm):
    if ARMS[arm] == "mlp":
        from . import model
        return model
    from . import trees
    return trees


def artifact(root, arm, split):
    suffix = ".pt" if ARMS[arm] == "mlp" else ".joblib"
    return root / "models" / split / (arm + suffix)


def role_names(split):
    return tuple(role + ("_time" if split == "forward" else "") for role in ("P", "C", "E"))


def fit(root, arm, split, threads, device):
    feat, (p,) = inputs(root / "inputs", arm, role_names(split)[:1])
    path = artifact(root, arm, split)
    extra = {"device": device} if ARMS[arm] == "mlp" else {}
    library(arm).fit_arm(p, feat, path, threads=threads, **extra)
    log = evidence(root, split + "/" + arm) / "fit.json"
    shutil.copyfile(path.with_suffix(".fit.json"), log)
    return [path, log]


def score(root, arm, split, threads, device):
    feat, (c, e) = inputs(root / "inputs", arm, role_names(split)[1:])
    lib = library(arm)
    extra = {"device": device} if ARMS[arm] == "mlp" else {}
    model = lib.load_arm(artifact(root, arm, split), threads=threads, **extra)
    times, values = {}, []
    for name, frame in [("C", c), ("E", e)]:
        start = time.perf_counter()
        values.append(lib.score_arm(model, frame, feat))
        elapsed = time.perf_counter() - start
        times[name] = {"seconds": elapsed, "rows_per_second": len(frame) / elapsed}
    return save_scores(root, arm, split, c, e, *values, timing=times)


def save_scores(root, arm, split, c, e, cal_score, score, *, timing):
    result, rows = measure(c, e, cal_score, score)
    result["threshold"]["scope"] = split + "_normal_only_calibration"
    result["score_timing"] = timing
    out = evidence(root, split + "/" + arm)
    write_json(out / "metrics.json", result)
    write_json(out / "strata.json", stratify(rows))
    c[["row_id", "label"]].assign(score=cal_score).to_parquet(out / "calibration.parquet", index=False)
    rows.to_parquet(out / "evaluation.parquet", index=False)
    return [out / file for file in ("metrics.json", "strata.json", "calibration.parquet", "evaluation.parquet")]


def old_references(root, original):
    _, (c, e) = inputs(root / "inputs", "mlp48", ("C", "E"))
    pattern = syn_cohort(e)
    if ((pattern & e.attack_family.eq("Scanning")).sum(), (pattern & e.label.eq(0)).sum()) != (217, 33):
        raise ValueError("Development E differs from the previously identified 47-field group")
    output = []
    for name, hashes in REFERENCE_HASHES.items():
        arrays = []
        for role, expected, frame in zip(("calibration", "evaluation"), hashes, (c, e)):
            source = original / "tasks" / name / "attempt-001" / (role + ".csv")
            if sha256_file(source) != expected:
                raise ValueError(f"Round 17 reference scores changed: {name}/{role}")
            ref = pd.read_csv(source, float_precision="round_trip")
            if not ref[["row_id", "label", "attack_family"]].equals(frame[["row_id", "label", "attack_family"]]):
                raise ValueError("Historical reference row identity differs")
            arrays.append(ref.score.to_numpy())
        output.extend(save_scores(root, name, "development", c, e, *arrays, timing="archived round 17 scores"))
    return output


def benchmark(root, arm, split, threads):
    features, (c, e) = inputs(root / "inputs", arm, role_names(split)[1:])
    out = evidence(root, split + "/" + arm)
    ref_c = pd.read_parquet(out / "calibration.parquet").score.to_numpy()
    ref_e = pd.read_parquet(out / "evaluation.parquet").score.to_numpy()
    dest = out / f"cpu-{threads}"
    library(arm).bench_arm(artifact(root, arm, split), c, e, features, threads,
                           reference_c=ref_c, reference_e=ref_e, output_dir=dest)
    return sorted(p for p in dest.iterdir() if p.is_file())


def benchmark_stream(root, arm, split, threads):
    from .streaming import bench_stream_arm
    features = list(FEATURES) + (list(HISTORY_COLUMNS) if arm.endswith("_history") else [])
    directory = evidence(root, split + "/" + arm)
    registration = read_json(directory / f"cpu-{threads}" / "benchmark.json")["registration"]
    out = directory / f"stream-{threads}"
    bench_stream_arm(artifact(root, arm, split), root / "history-work" / "benchmark_stream.parquet",
                     features, threads, threshold_value=registration, output_dir=out)
    return sorted(p for p in out.iterdir() if p.is_file())


def lane(root, binding, splits, family, threads, device):
    # Independent GPU and CPU lanes. All model parameters were fixed before data loading.
    failures = {}
    for split in splits:
        for arm, kind in ARMS.items():
            if kind != family:
                continue
            key = f"{split}-{arm}"
            try:
                stage(root, key + "-fit", binding, partial(fit, root, arm, split, threads, device))
                stage(root, key + "-score", binding, partial(score, root, arm, split, threads, device))
            except Exception:
                failures[key] = traceback.format_exc()
    return failures


def summary(root, splits, failures):
    result = {"status": "partial" if failures else "complete", "errors": failures, "models": {}, "comparisons": {}}
    forward = read_json(root / "evidence/inputs/cohort_manifest.json")["forward"]
    result["forward_cohort"] = forward
    if not failures and not forward["binary_evaluable"]:
        result["status"] = "development_complete_forward_not_evaluable"
    result["history_cost_file"] = "evidence/inputs/history_audit.json"
    for split in splits:
        predictions = {}
        aux = list(REFERENCE_HASHES) if split == "development" else []
        for name in [*ARMS, *aux]:
            directory = root / "evidence" / split / name
            completed = ("anchors" if name in aux else f"{split}-{name}-score") + ".json"
            if (not (root / "receipts" / completed).is_file() or f"{split}-{name}" in failures
                    or ARMS.get(name) in failures or (name in aux and "anchors" in failures)):
                continue
            metric = read_json(directory / "metrics.json")
            predictions[name] = pd.read_parquet(directory / "evaluation.parquet"), metric
            costs = {}
            for path in sorted(directory.glob("cpu-*/benchmark.json")):
                n = path.parent.name.removeprefix("cpu-")
                bench_key = f"{split}-{name}-bench-{n}"
                if bench_key in failures or not (root / "receipts" / (bench_key + ".json")).is_file():
                    continue
                cost = read_json(path)
                costs[n] = {"single_p95_ms": cost["benchmark"]["single"]["p95_ms"],
                            "batch128_per_s": cost["benchmark"]["batch"]["throughput_rows_per_second"],
                            "hot_process_rss_bytes": cost["benchmark"]["rss"]["sampled_peak_bytes"],
                            "file_and_threshold_bytes": cost["model_and_threshold_bytes"],
                            "equivalent": cost["replay"]["equivalent_to_reference"],
                            "cpu_own_registration_fp": cost["metrics"]["fp"],
                            "cpu_own_registration_tp": cost["metrics"]["tp"]}
                stream_key = f"{split}-{name}-stream-{n}"
                if stream_key not in failures and (root / "receipts" / (stream_key + ".json")).exists():
                    realtime = read_json(directory / f"stream-{n}/stream-benchmark.json")
                    costs[n]["stream"] = {key: realtime.get(key) for key in (
                        "status", "service_p95_ms", "throughput_rows_per_second", "hot_worker_sampled_peak_rss",
                        "history_matches_offline", "scores_within_tolerance_and_alerts_unchanged")}
            result["models"][f"{split}/{name}"] = {
                **{k: metric[k] for k in ("tp", "fp", "fpr", "tpr", "scan_n", "scan_tp", "syn_scan_tp",
                                           "syn_fp", "syn_scan_normal_auc", "scan_normal_pauc")}, "cpu_cost": costs}
        pairs = [("mlp48", "mlp52_history"), ("mlp52_zero", "mlp52_history"), ("hgb48", "hgb52_history")]
        pairs.extend((a, b) for a in aux for b in ("mlp52_history", "hgb52_history"))
        for a, b in pairs:
            if a in predictions and b in predictions:
                left, base = predictions[a]
                right, candidate = predictions[b]
                result["comparisons"][f"{split}/{a}_to_{b}"] = paired_comparison(left, right, base, candidate)
                if split == "forward" and not forward["scanning_evaluable"]:
                    result["comparisons"][f"{split}/{a}_to_{b}"]["passes_detection_comparison"] = None
    result["forward_detection_checks"] = {}
    for candidate, controls in (("mlp52_history", ["mlp48", "mlp52_zero"]), ("hgb52_history", ["hgb48"])):
        comparisons = [result["comparisons"].get(f"forward/{base}_to_{candidate}", {}) for base in controls]
        checks = [row.get("passes_detection_comparison") for row in comparisons]
        result["forward_detection_checks"][candidate] = None if None in checks else all(checks)
    result["interpretation"] = ("Detection checks compare the finite registered arms and the observed 1% point. "
                                "Review target-group ranking, family losses, host/day concentration, CPU replay, "
                                "stream costs and full-source state before a deployment decision.")
    write_json(root / "summary.json", result)


def synthetic_checks(root):
    repo = Path(__file__).resolve().parents[3]
    tests = sorted((repo / "tests").glob("test_flow_history*.py"))
    if not tests:
        raise RuntimeError("Flow-history release tests are missing")
    out = evidence(root, "preflight")
    junit = out / "synthetic-junit.xml"
    log = out / "synthetic-tests.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}", *map(str, tests)],
                                cwd=repo, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Lab synthetic regression failed; see {log}")
    return [junit, log]


def run(root, original, source_option, threads, monitor, *, device="cuda:0"):
    from ids.route_batch.protocol import source_snapshot
    from ids.common import environment_snapshot

    snapshot = source_snapshot()
    env = environment_snapshot()
    spec = {"format": "flow-history-20261111-v1", "source_code": snapshot["source_sha256"],
            "original_run": str(original), "source_option": str(source_option or "original-protocol"),
            "threads": threads, "gpu": device, "alpha": 0.01, "arms": ARMS, "history": list(HISTORY_COLUMNS),
            "budget_rule": "strict >; per-arm normal-only ceil((n+1)*(1-alpha))",
            "runtime": {"python": env["python"], "packages": env["packages"], "torch": env.get("torch")},
            "no_test_selection": True, "old_tabm_training_calls": 0}
    binding = canonical_hash(spec)
    if (root / "protocol.json").exists() and read_json(root / "protocol.json") != spec:
        raise ValueError("The existing experiment was registered with another source, device, path or thread count")
    write_json(root / "protocol.json", spec)
    write_json(root / "environment.json", env)
    monitor.activity = "synthetic_preflight"
    stage(root, "synthetic", binding, partial(synthetic_checks, root))
    monitor.activity = "cohorts"
    stage(root, "cohorts", binding, partial(prepare, root, original, source_option, threads))
    monitor.activity = "60s_history"
    stage(root, "history", binding, partial(augment, root, threads))
    failures = {}
    monitor.activity = "read_archived_reference"
    try:
        stage(root, "anchors", binding, partial(old_references, root, original))
    except Exception:
        failures["anchors"] = traceback.format_exc()
    splits = ["development"]
    if read_json(root / "evidence/inputs/cohort_manifest.json")["forward"]["binary_evaluable"]:
        splits.append("forward")
    monitor.activity = "training_and_batch_scoring_mlp_gpu_hgb_cpu"
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        mlp_threads = min(4, max(1, threads // 2))
        futures = {kind: pool.submit(lane, root, binding, splits, kind,
                                     mlp_threads if kind == "mlp" else threads - mlp_threads,
                                     device) for kind in ("mlp", "hgb")}
        for kind, future in futures.items():
            try:
                failures.update(future.result())
            except Exception:
                failures[kind] = traceback.format_exc()
    for split in splits:
        for arm in ARMS:
            if (f"{split}-{arm}" in failures or ARMS[arm] in failures or
                    not (root / "receipts" / f"{split}-{arm}-score.json").exists()):
                continue
            for n in (1, threads):
                key = f"{split}-{arm}-bench-{n}"
                monitor.activity = key
                try:
                    stage(root, key, binding, partial(benchmark, root, arm, split, n))
                except Exception:
                    failures[key] = traceback.format_exc()
                    continue
                if split == splits[-1] and arm != "mlp52_zero":
                    key = f"{split}-{arm}-stream-{n}"
                    monitor.activity = key
                    try:
                        stage(root, key, binding, partial(benchmark_stream, root, arm, split, n))
                    except Exception:
                        failures[key] = traceback.format_exc()
    summary(root, splits, failures)
    if failures:
        raise RuntimeError(f"Failed stages: {', '.join(failures)}; inspect summary.json and resume")
