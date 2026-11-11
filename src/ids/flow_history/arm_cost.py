"""Isolated CPU cost timing and cross-device replay shared by Round 19 arms."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.lab_data import EXCLUDED_FEATURES
from ids.route_batch.benchmark import benchmark_predictor
from ids.route_batch.metrics import decisions, evaluate, threshold


def checked_features(features):
    names = tuple(features)
    excluded = {"row_id", "sample_id", "communication_group", "label", "attack_family", *EXCLUDED_FEATURES}
    if not names or len(set(names)) != len(names) or excluded.intersection(names):
        raise ValueError("model requires distinct numerical input columns excluding labels and identifiers")
    return names


def _replay(frame, saved, cpu, original_threshold, cpu_threshold, output):
    allowed = 3e-5 + 3e-5*np.abs(saved)
    delta = np.abs(cpu - saved)
    reference = decisions(saved, original_threshold)
    same_cut = decisions(cpu, original_threshold)
    recal = decisions(cpu, cpu_threshold)
    row = next((name for name in ("row_id", "sample_id") if name in frame), None)
    evidence = pd.DataFrame({
        "row_id": frame[row].to_numpy() if row else np.arange(len(frame)),
        "label": frame.label.to_numpy(), "reference_score": saved, "cpu_score": cpu,
        "reference_alert": reference, "cpu_at_reference_threshold": same_cut, "cpu_at_own_threshold": recal,
    })
    evidence.to_csv(output, index=False)
    return {"rows": len(cpu), "max_abs_error": float(delta.max()), "outside_tolerance": int((delta > allowed).sum()),
            "decision_changes_reference_threshold": int(np.sum(reference != same_cut)),
            "decision_changes_recalibrated": int(np.sum(reference != recal)), "evidence": output.name}


def _bench_worker(request, output):
    req = read_json(request)
    model_path, threads = Path(req["model"]), req["threads"]
    if req["kind"] == "mlp":
        from .model import load_arm, score_arm
        kwargs = {"device": "cpu", "threads": threads}
    elif req["kind"] == "hgb":
        from .trees import load_arm, score_arm
        kwargs = {"threads": threads}
    else:
        raise ValueError("unknown benchmark model")
    with threadpool_limits(limits=threads):
        start = time.perf_counter()
        model = load_arm(model_path, **kwargs)
        load_seconds = time.perf_counter() - start
        c, e = [pd.read_parquet(Path(request).parent / f"{role}.parquet") for role in ("c", "e")]
        with np.load(Path(request).parent / "reference.npz", allow_pickle=False) as expected:
            ref_c, ref_e = expected["c"], expected["e"]
        if not c.label.eq(0).all():
            raise ValueError("calibration must contain only normal rows")
        cpu_c = score_arm(model, c, req["features"])
        cpu_e = score_arm(model, e, req["features"])
        cpu_threshold, original = threshold(cpu_c), threshold(ref_c)
        cpu_threshold["scope"] = original["scope"] = "normal-only calibration of supplied role"
        out_dir = Path(output).parent
        replay = {"reference_registration": original, "atol": 3e-5, "rtol": 3e-5}
        for role, frame, ref, scores in [("calibration", c, ref_c, cpu_c), ("evaluation", e, ref_e, cpu_e)]:
            replay[role] = _replay(frame, ref, scores, original, cpu_threshold, out_dir / f"{role}-replay.csv.gz")
        replay["equivalent_to_reference"] = (
            replay["calibration"]["outside_tolerance"] + replay["evaluation"]["outside_tolerance"] == 0
            and replay["evaluation"]["decision_changes_reference_threshold"] == 0
            and replay["evaluation"]["decision_changes_recalibrated"] == 0
        )
        measured = benchmark_predictor(lambda frame: score_arm(model, frame, req["features"]), e)
        measured["thread_contract"] = f"fresh CPU worker, native thread pools set to {threads}"
        measured["scope"] = "selected feature columns already available; model input conversion and inference included"
        measured["cold_model_load_measured"] = True
        measured["cold_model_load_seconds"] = load_seconds
        model_size = model_path.stat().st_size
        write_json(out_dir / "threshold.json", cpu_threshold)
        report = {"threads": threads, "native_threadpools": threadpool_info(), "device": "cpu",
                  "kind": req["kind"], "pid": os.getpid(),
                  "registration": cpu_threshold, "metrics": evaluate(e.label, cpu_e, cpu_threshold, e.get("attack_family")),
                  "replay": replay, "benchmark": measured, "model_artifact_bytes": model_size,
                  "model_sha256": sha256_file(model_path),
                  "model_and_threshold_bytes": model_size + (out_dir / "threshold.json").stat().st_size,
                  "feature_engine_cost_included": False}
        write_json(output, report)


def benchmark_arm(kind, path, C, E, features, threads, *, reference_c, reference_e, output_dir):
    """Isolate one CPU thread setting; callers schedule the two settings serially."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for role, frame, scores in [("c", C, reference_c), ("e", E, reference_e)]:
        values = np.asarray(scores, dtype=np.float64)
        if values.shape != (len(frame),) or not len(frame) or not np.isfinite(values).all():
            raise ValueError(f"invalid reference scores for {role}")
        arrays[role] = values
    output = output_dir / "benchmark.json"
    with TemporaryDirectory(dir=output_dir, prefix="inputs-") as tmp:
        work = Path(tmp)
        for role, frame in [("c", C), ("e", E)]:
            meta = [col for col in ("row_id", "sample_id", "label", "attack_family") if col in frame]
            frame.loc[:, [*meta, *features]].to_parquet(work / f"{role}.parquet", index=False)
        np.savez(work / "reference.npz", **arrays)
        request = work / "request.json"
        write_json(request, {"kind": kind, "model": str(Path(path).resolve()), "features": list(features), "threads": threads})
        env = dict(os.environ)
        env.update({name: str(threads) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH"))))
        subprocess.run([sys.executable, "-m", "ids.flow_history.arm_cost", str(request), str(output)],
                       check=True, env=env)
    return read_json(output)


if __name__ == "__main__":
    _bench_worker(*sys.argv[1:])
