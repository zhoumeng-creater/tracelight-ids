"""One fixed laboratory batch: freeze nine fits, then nine C rules, then development."""
from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.calibration_transfer import data  # noqa: E402
from scripts.calibration_transfer.evaluation import RUNS, evaluate_rows, report  # noqa: E402

REGISTRATION = Path("docs/calibration_transfer_batch_20261120/registration.json")
THREADS = 4
DEVICE = "cuda:0"
TRAINING = {
    "optimizer_steps": 15600, "batch_size": 256, "draws_per_class": 128,
    "score_batch_size": 1024, "report_every": 100, "learning_rate": .002,
    "weight_decay": .0003, "gradient_clip_norm": 1., "transfer_weight": 1.,
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def text_digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_registration(repo=REPO):
    registration = json.loads((repo / REGISTRATION).read_text(encoding="utf8"))
    if (registration["runs"] != list(RUNS) or registration["training"] != TRAINING
            or registration["threads"] != THREADS or registration["device"] != DEVICE
            or registration["selection"] != "terminal_step_15600_no_refit"
            or not registration["execution_enabled"]):
        raise ValueError("Registration differs from the fixed runner")
    for name, expected in registration["code_sha256_lf"].items():
        if text_digest(repo / name) != expected:
            raise ValueError(f"Registered code changed: {name}")
    receipt_path = repo / REGISTRATION.parent / "inputs_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    manifest_path = receipt_path.parent / receipt["manifest_path"]
    if data.file_digest(manifest_path) != receipt["manifest_file_sha256"]:
        raise ValueError("Frozen episode manifest changed")
    if receipt["manifest_canonical_sha256"] != registration["inputs_manifest_canonical_sha256"]:
        raise ValueError("Input receipt differs from registration")
    return registration, receipt


def prepare(repo=REPO):
    registration, receipt = verify_registration(repo)
    rows, preprocessor, numeric, categorical, manifest = data.prepare_inputs(repo)
    if data.json_digest(manifest) != receipt["manifest_canonical_sha256"]:
        raise ValueError("F input/episode reconstruction differs from frozen manifest")
    return registration, receipt, rows, preprocessor


def environment():
    """Hardware/package check only; no model construction or input scoring."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    required = {"numpy": "1.26.4", "pandas": "2.2.2", "scikit-learn": "1.5.1",
                "torch": "2.6.0", "tabm": "0.0.3", "rtdl_num_embeddings": "0.0.12"}
    versions = {key: importlib.metadata.version(key) for key in required}
    if any(versions[key].split("+")[0] != value for key, value in required.items()):
        raise RuntimeError(f"Use the existing locked laboratory environment: {versions}")
    if not torch.cuda.is_available() or "4090D" not in torch.cuda.get_device_name(0).replace(" ", ""):
        raise RuntimeError("Real fits require the registered laboratory RTX4090D at cuda:0")
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Project Python contract is 3.11 or 3.12")
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "versions": versions, "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "threads": THREADS,
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())}


def calibration_threshold(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("C requires finite nonempty scalar scores")
    budget = len(values) // 100
    threshold = float(np.sort(values, kind="stable")[-(budget + 1)])
    return {"threshold": threshold, "comparison": ">", "alpha": .01,
            "C_count": len(values), "integer_FP_budget": budget,
            "C_empirical_FP": int((values > threshold).sum()), "population_guarantee": False}


def checked_scores(predictor, rows):
    import numpy as np
    values = np.asarray(predictor.score(rows), dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("Scoring must return one finite scalar for every input row")
    return values


def load_calibration(repo):
    """Called only after all model artifacts are frozen; no V decoding here."""
    cohorts = json.loads((repo / data.BATCH / "cohorts.json").read_text(encoding="utf8"))
    selected = {}
    role_filter = re.compile(r'"dataset_role"\s*:\s*"C"')
    with gzip.open(repo / data.BATCH / "rows.jsonl.gz", "rt", encoding="utf8") as stream:
        for line in stream:
            if role_filter.search(line):
                row = json.loads(line)
                if row["dataset_role"] != "C" or row["label"] != 0:
                    raise ValueError("C must contain only registered normal labels")
                selected[row["flow_id"]] = row
    if set(selected) != set(cohorts["C"]) or len(selected) != 3475:
        raise ValueError("Calibration identities differ from the fixed whole-run cohort")
    return [selected[fid] for fid in cohorts["C"]]


def load_evaluation(repo):
    """Decode opened development only after every C threshold is persisted."""
    from scripts.coverage_batch.run import inspect
    rows = report.read_rows(repo / data.BATCH / "rows.jsonl.gz")
    cohorts = json.loads((repo / data.BATCH / "cohorts.json").read_text(encoding="utf8"))
    events = report.read_rows(repo / data.BATCH / "events.jsonl.gz")
    units = report.read_rows(repo / data.BATCH / "normal_units.jsonl.gz")
    inspect(rows, cohorts, events, units)
    return rows, cohorts, events, units


def execute(output, rows, preprocessor, registration, *, repo=REPO,
            fit_fn=None, predictor_loader=None, calibration_loader=None,
            evaluation_loader=None, evaluation_writer=None):
    """Injected functions support synthetic phase-order checks without real fitting."""
    if fit_fn is None or predictor_loader is None:
        from scripts.calibration_transfer.training import fit, load_predictor
        fit_fn, predictor_loader = fit, load_predictor
    calibration_loader = calibration_loader or load_calibration
    evaluation_loader = evaluation_loader or load_evaluation
    evaluation_writer = evaluation_writer or evaluate_rows
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries, frozen = {}, {}
    for run in RUNS:
        arm, seed = run.rsplit("_s", 1)
        directory = output / "fits" / run
        print(json.dumps({"phase": "fit", "run": run}, ensure_ascii=False), flush=True)
        summaries[run] = fit_fn(rows, preprocessor, directory, arm=arm, seed=int(seed),
                                training_spec=registration["training"], device=DEVICE, threads=THREADS)
        artifact = directory / "model.pt"
        if not artifact.is_file():
            raise RuntimeError(f"Missing terminal artifact: {run}")
        frozen[run] = {"path": artifact.relative_to(output).as_posix(),
                       "sha256": data.file_digest(artifact), "bytes": artifact.stat().st_size}
    for seed in (17, 29, 43):
        for key in ("initial_state_sha256", "sampler_draw_sha256"):
            if len({summaries[f"{arm}_s{seed}"][key] for arm in ("A_BCE", "B_permuted", "C_real")}) != 1:
                raise ValueError(f"Paired initialization or BCE sampling differs: seed {seed}, {key}")
    # This barrier is persisted before opening or transforming calibration rows.
    write(output / "models_frozen.json", {"runs": frozen, "selection": registration["selection"],
                                          "new_fits": 9, "refits": 0})
    calibration = calibration_loader(repo)
    thresholds, c_scores, score_cost = {}, {}, {}
    for run in RUNS:
        artifact = output / frozen[run]["path"]
        if data.file_digest(artifact) != frozen[run]["sha256"]:
            raise ValueError("Frozen model changed before calibration")
        predictor = predictor_loader(artifact, device=DEVICE, threads=THREADS)
        started = time.perf_counter()
        values = checked_scores(predictor, calibration)
        score_cost[run] = {"C_score_seconds": time.perf_counter() - started}
        thresholds[run] = calibration_threshold(values)
        c_scores[run] = dict(zip((r["flow_id"] for r in calibration), map(float, values)))
        del predictor
    write(output / "thresholds.json", thresholds)
    write(output / "C_scores.json", c_scores)
    write(output / "thresholds_frozen.json", {"sha256": data.file_digest(output / "thresholds.json"),
                                              "C_scores_sha256": data.file_digest(output / "C_scores.json"),
                                              "threshold_calculations": 9, "threshold_searches": 0})
    all_rows, cohorts, events, units = evaluation_loader(repo)
    c_ids = set(cohorts["C"])
    remaining = [row for row in all_rows if row["flow_id"] not in c_ids]
    scores = {}
    for run in RUNS:
        artifact = output / frozen[run]["path"]
        if data.file_digest(artifact) != frozen[run]["sha256"]:
            raise ValueError("Frozen model changed before development")
        predictor = predictor_loader(artifact, device=DEVICE, threads=THREADS)
        started = time.perf_counter()
        values = checked_scores(predictor, remaining)
        score_cost[run]["remaining_score_seconds"] = time.perf_counter() - started
        scores[run] = c_scores[run] | dict(zip((r["flow_id"] for r in remaining), map(float, values)))
        del predictor
    result = evaluation_writer(all_rows, cohorts, events, units, scores, thresholds, output / "evaluation")
    write(output / "cost.json", {"fits": summaries, "batch_scoring": score_cost,
                                 "resource_scope": "Fixed fits and complete-flow batch scoring; no extractor or per-alert latency claim"})
    summary = {"status": "complete", "runs": list(RUNS), "fit_calls": 9, "refits": 0,
               "threshold_calculations": 9, "threshold_searches": 0,
               "model_selection": False, "reserved_data_scored": False,
               "historical_development_only": True, "evaluation": result}
    write(output / "summary.json", summary)
    return summary


def output_receipt(output):
    write(output / "output_receipt.json", {
        p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": data.file_digest(p)}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "output_receipt.json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; preserve prior attempts, no automatic refit")
    registration, receipt, rows, preprocessor = prepare()
    env = None if args.inputs_only else environment()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "source_receipt.json", {"registration_sha256_lf": text_digest(REPO / REGISTRATION),
                                           "registration": registration, "inputs": receipt})
    try:
        if args.inputs_only:
            write(output / "inputs_check.json", receipt)
            print(json.dumps({"status": "inputs_only_pass", "F_rows": len(rows),
                              "model_constructions": 0, "fit_calls": 0, "score_calls": 0,
                              "threshold_calculations": 0}), flush=True)
        else:
            write(output / "environment.json", env)
            execute(output, rows, preprocessor, registration)
    except Exception as error:
        write(output / "failure.json", {"type": type(error).__name__, "error": str(error),
                                         "traceback": traceback.format_exc(), "automatic_retry": False})
        raise
    finally:
        output_receipt(output)


if __name__ == "__main__":
    main()
