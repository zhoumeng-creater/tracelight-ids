"""Frozen lab CPU cost diagnostic: three saved models, zero fitting.

The caller verifies artifact/data bindings and the lab host before this module.
RSS workers deliberately load only one model in each fresh Python process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .common import read_json, write_json

MODEL_KINDS = {"H40": "mlp", "HGB": "hgb", "CatBoost_reference": "catboost"}
SEED = 20261105
THREADS = 4


@contextmanager
def _cpu_threads(use_torch):
    previous = None
    if use_torch:
        import torch
        previous = torch.get_num_threads()
        torch.set_num_threads(THREADS)
    try:
        with threadpool_limits(limits=THREADS):
            yield
    finally:
        if previous is not None:
            torch.set_num_threads(previous)


def _import_backend(kind):
    # Import before the RSS "after import" checkpoint and load timer.
    if kind == "catboost":
        from .native_trees import _catboost
        _catboost()
    else:
        from .models import load_bundle  # noqa: F401
        if kind == "mlp":
            import torch  # noqa: F401


def _decisions(raw_scores, rows, threshold):
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.shape != (rows,) or not np.isfinite(scores).all():
        raise ValueError("Frozen CPU pipeline returned nonfinite or mis-shaped raw margins")
    return scores >= threshold


def _load_pipeline(spec):
    """Construct a ready CPU raw-margin -> frozen-threshold callable; no query."""
    threshold = spec["threshold"]
    if spec["kind"] == "catboost":
        from .native_trees import load_native_tree
        bundle = load_native_tree(spec["path"])
        if bundle.num_threads != THREADS:
            raise ValueError("Frozen native-tree artifact must use four threads")

        def call(frame):
            return _decisions(bundle.predict_raw(frame), len(frame), threshold)
        return call

    from .models import _network, load_bundle
    bundle = load_bundle(spec["path"], device="cpu")
    if bundle.model_name != spec["kind"]:
        raise ValueError("Diagnostic model kind does not match saved artifact")
    if bundle.model_name == "hgb":
        def call(frame):
            scores = bundle.estimator.decision_function(bundle.preprocessor.transform(frame))
            return _decisions(scores, len(frame), threshold)
        return call

    import torch
    with torch.random.fork_rng(devices=[]):
        net = _network(bundle.encoded_dim, bundle.hidden_dims)
    net.load_state_dict(bundle.network_state)
    net.cpu().eval()

    def call(frame):
        matrix = bundle.preprocessor.transform(frame)
        with torch.inference_mode():
            scores = net(torch.from_numpy(matrix)).squeeze(1).cpu().numpy()
        return _decisions(scores, len(frame), threshold)
    return call


def _specs(model_specs):
    if set(model_specs) != set(MODEL_KINDS):
        raise ValueError("Exactly H40, HGB and CatBoost_reference are required")
    result = {}
    for label, kind in MODEL_KINDS.items():
        value = model_specs[label]
        if value.get("kind") != kind:
            raise ValueError(f"Unexpected model kind for {label}")
        threshold = value.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("A finite frozen raw threshold is required")
        if not np.isfinite(threshold):
            raise ValueError("A finite frozen raw threshold is required")
        path = Path(value["path"]).resolve()
        files = [path / n for n in ("model.cbm", "metadata.json")] if kind == "catboost" else [path]
        if any(not p.is_file() for p in files):
            raise FileNotFoundError(f"Missing frozen model files for {label}")
        result[label] = {
            "kind": kind, "path": str(path), "threshold": float(threshold),
            "package_bytes": sum(p.stat().st_size for p in files),
        }
    return result


def _workload(frame, timing_batches):
    if set(timing_batches) != {"1", "256"} or "sample_id" not in frame:
        raise ValueError("Frozen batch1/batch256 identities are required")
    if frame.sample_id.isna().any():
        raise ValueError("Missing constraint sample identity")
    ids = frame.sample_id.astype(str)
    if ids.duplicated().any():
        raise ValueError("Constraint identities must be unique")
    positions = {s: i for i, s in enumerate(ids)}
    result = {}
    for size in (1, 256):
        samples = timing_batches[str(size)].get("sample_ids")
        if not isinstance(samples, list) or len(samples) != 75:
            raise ValueError("Each frozen size must contain exactly 75 inputs")
        result[str(size)] = []
        for sample in samples:
            if not isinstance(sample, list) or len(sample) != size:
                raise ValueError("Frozen timing input has incorrect batch size")
            if any(not isinstance(s, str) or s not in positions for s in sample):
                raise ValueError("Frozen timing input is absent from constraint identities")
            result[str(size)].append([positions[s] for s in sample])
    return result


def _statistics(values, size):
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Timing samples must be positive and finite")
    return {
        "p50_ms": float(np.percentile(values, 50, method="linear")),
        "p95_ms": float(np.percentile(values, 95, method="linear")),
        "mean_ms": float(values.mean()),
        "rows_per_second": float(size * 1000 / values.mean()),
    }


def _rss_worker(request):
    label, spec = request["model"], request["spec"]
    if MODEL_KINDS.get(label) != spec["kind"]:
        raise ValueError("Invalid RSS model")
    _import_backend(spec["kind"])
    with _cpu_threads(spec["kind"] == "mlp"):
        process = psutil.Process()
        rss_import = process.memory_info().rss
        frame = pd.read_parquet(request["input_path"])
        if len(frame) != 256:
            raise ValueError("RSS probe must contain exactly 256 rows")
        rss_input = process.memory_info().rss
        start = time.perf_counter_ns()
        pipeline = _load_pipeline(spec)
        load_ms = (time.perf_counter_ns() - start) / 1e6
        rss_load = process.memory_info().rss
        # The sole worker inference; no hidden warmup and no retry.
        decisions = pipeline(frame)
        if np.asarray(decisions).shape != (256,):
            raise ValueError("RSS pipeline returned invalid decisions")
        rss_probe = process.memory_info().rss
    result = {
        "model": label, "pid": os.getpid(), "threads": THREADS,
        "rss_after_import_bytes": rss_import, "rss_after_input_bytes": rss_input,
        "rss_after_load_bytes": rss_load, "rss_after_probe_bytes": rss_probe,
        "model_load_ms": load_ms, "forward_calls": 1, "forward_rows": len(frame),
        "sample_ids": frame.sample_id.astype(str).tolist(),
        "scope": "one fresh process; resident snapshots, not peak RSS",
    }
    write_json(request["output_path"], result)
    return result


def _run_rss_probe(label, spec, input_path, root):
    request_path = root / f"rss-{label}-request.json"
    result_path = root / f"rss-{label}.json"
    write_json(request_path, {
        "model": label, "spec": spec, "input_path": str(input_path.resolve()),
        "output_path": str(result_path.resolve()),
    })
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    previous_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root + (os.pathsep + previous_pythonpath if previous_pythonpath else "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(THREADS)
    with (root / f"rss-{label}-stdout.txt").open("w", encoding="utf-8") as stdout, (
        root / f"rss-{label}-stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        subprocess.run(
            [sys.executable, "-m", "ids.unsw_diagnostic_cost", "--rss-worker", str(request_path)],
            check=True, stdout=stdout, stderr=stderr, env=env,
        )
    result = read_json(result_path)
    if result.get("model") != label or result.get("forward_calls") != 1 or result.get("forward_rows") != 256:
        raise ValueError("RSS worker did not honor the one-call budget")
    return result


def measure_cost(model_specs, constraint_frame, timing_batches, output):
    """Measure the frozen three-model workload; output must be a new directory.

    Specs: {H40: {kind: "mlp", path: model.joblib, threshold: ...}, HGB: {...},
            CatBoost_reference: {kind: "catboost", path: directory, threshold: ...}}.
    timing_batches is round10 summary["cost"]["timing"]["batches"] unchanged.
    """
    specs = _specs(model_specs)
    workload = _workload(constraint_frame, timing_batches)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    labels = list(MODEL_KINDS)
    # A separate generator from the original two-model/input generator.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3]))
    budget = {
        "formal_calls": 0, "formal_rows": 0, "warmup_calls": 0, "warmup_rows": 0,
        "rss_processes": 0, "rss_forward_calls": 0, "rss_forward_rows": 0,
        "new_fits": 0, "optimizer_updates": 0,
    }
    result = {
        "status": "running", "model_specs": specs, "threads": THREADS,
        "seed": SEED, "order_seed_sequence": [SEED, 3],
        "input_source": "unchanged round10 cost.timing.batches sample_ids",
        "includes": "in-memory fields -> preprocessing/native conversion -> CPU raw margin -> frozen threshold",
        "excludes": "disk CSV read, capture and raw flow feature extraction",
        "percentile_method": "numpy.percentile(method='linear')",
        "batches": {}, "rss": {}, "budget": budget,
        "acceptance": "descriptive cross-family tradeoff; no new 1.25 ratio gate",
    }
    write_json(root / "cost-plan.json", {
        **result, "input_samples": {s: timing_batches[s]["sample_ids"] for s in workload},
    })
    try:
        with (root / "timings.jsonl").open("x", encoding="utf-8") as log, (
            root / "warmup.jsonl"
        ).open("x", encoding="utf-8") as warm_log:
            # threadpoolctl discovers loaded libraries when its context opens.
            for spec in specs.values():
                _import_backend(spec["kind"])
            with _cpu_threads(True):
                pipelines = {label: _load_pipeline(specs[label]) for label in labels}
                for size_key, indices in workload.items():
                    size = int(size_key)
                    first = constraint_frame.iloc[indices[0]]
                    for label in labels:
                        for repeat in range(5):
                            budget["warmup_calls"] += 1
                            budget["warmup_rows"] += size
                            row = {"model": label, "size": size, "repeat": repeat,
                                   "sample_ids": first.sample_id.astype(str).tolist(), "status": "started"}
                            try:
                                pipelines[label](first)
                                row["status"] = "complete"
                            except Exception as exc:
                                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                                raise
                            finally:
                                warm_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                                warm_log.flush()
                    durations = {label: [] for label in labels}
                    orders = []
                    for group, ix in enumerate(indices):
                        part = constraint_frame.iloc[ix]
                        order = [labels[i] for i in rng.permutation(3)]
                        orders.append(order)
                        for position, label in enumerate(order):
                            row = {"model": label, "size": size, "group": group,
                                   "block": group // 25, "repeat_in_block": group % 25,
                                   "position": position, "execution_order": order,
                                   "sample_ids": part.sample_id.astype(str).tolist(), "status": "started"}
                            budget["formal_calls"] += 1
                            budget["formal_rows"] += size
                            start = time.perf_counter_ns()
                            try:
                                decisions = pipelines[label](part)
                                elapsed = (time.perf_counter_ns() - start) / 1e6
                                if np.asarray(decisions).shape != (size,) or elapsed <= 0 or not np.isfinite(elapsed):
                                    raise ValueError("Invalid pipeline output or duration")
                                durations[label].append(elapsed)
                                row.update(status="complete", duration_ms=elapsed)
                            except Exception as exc:
                                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                                raise
                            finally:
                                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                                log.flush()
                    result["batches"][size_key] = {
                        "sample_ids": timing_batches[size_key]["sample_ids"], "execution_order": orders,
                        **{label: {"durations_ms": values, **_statistics(values, size),
                                  "blocks": [_statistics(values[j*25:(j+1)*25], size) for j in range(3)]}
                           for label, values in durations.items()},
                    }
        probe = constraint_frame.iloc[workload["256"][0]].copy()
        input_path = root / "rss-input.parquet"
        probe.to_parquet(input_path, index=False)
        expected_ids = probe.sample_id.astype(str).tolist()
        for label in labels:
            budget["rss_processes"] += 1
            rss = _run_rss_probe(label, specs[label], input_path, root)
            if rss.get("sample_ids") != expected_ids:
                raise ValueError("RSS worker input identities changed")
            result["rss"][label] = rss
            budget["rss_forward_calls"] += rss["forward_calls"]
            budget["rss_forward_rows"] += rss["forward_rows"]
        expected = (450, 57825, 30, 3855, 3, 3, 768)
        actual = tuple(budget[k] for k in (
            "formal_calls", "formal_rows", "warmup_calls", "warmup_rows",
            "rss_processes", "rss_forward_calls", "rss_forward_rows",
        ))
        if actual != expected:
            raise ValueError("Frozen cost budget mismatch")
        result["status"] = "complete"
        write_json(root / "cost-summary.json", result)
        return result
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(root / "cost-failure.json", result)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rss-worker", required=True, type=Path)
    args = parser.parse_args(argv)
    _rss_worker(read_json(args.rss_worker))


if __name__ == "__main__":
    main()
