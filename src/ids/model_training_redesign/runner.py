"""Six bounded F/V trajectories, same-checkpoint delivery and fixed evaluation."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, run_lock
from ids.model_training.runner import contained, files, verify_receipt, checked_stage
from . import VERSION, data, evaluation
from .protocol import REPO, code_snapshot, specifications, tasks, verify_live_code, verify_runtime


def snapshot_sources(root, protocol):
    for rel in protocol["code"]:
        dest = root / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / rel).read_bytes().replace(b"\r\n", b"\n"))
    return files(root / "source_snapshot")


def initialize(root, r29, r26, raw_dir, threads, device):
    if not 1 <= threads <= 64:
        raise ValueError("Use between 1 and 64 CPU threads")
    for source in (r29, r26, raw_dir):
        if source is not None and (root.is_relative_to(source) or source.is_relative_to(root)):
            raise ValueError("New output must be separate from frozen input directories")
    registration, design = specifications()
    roles, binding, references = data.load_inputs(r29, r26, raw_dir, root, design, registration)
    protocol = {"version": VERSION, "registration": registration, "design": design,
                "tasks": tasks(design), "inputs": binding, "code": code_snapshot(),
                "threads": threads, "device": device, "score_device": "cpu", "score_threads": 1,
                "new_blind_confirmation": False}
    destination = root / "protocol.json"
    if destination.exists():
        if read_json(destination) != protocol:
            raise ValueError("Input, code, design or execution identity changed; preserve the existing run")
    else:
        if any(p.name != "run.lock" for p in root.iterdir()):
            raise ValueError("Unregistered output directory is not empty")
        write_json(destination, protocol)
    key = canonical_hash(protocol)
    checked_stage(root, "inputs", key, lambda: data.persist_inputs(root, roles, binding, references))
    checked_stage(root, "sources", key, lambda: snapshot_sources(root, protocol))
    return protocol


def verify_inputs(root, protocol):
    verify_receipt(root, root / "receipts/inputs.json", canonical_hash(protocol))


def train_task(root, task_id, backend=None):
    """Only F and V enter this worker; no P, C or E is loaded for fitting."""
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_inputs(root, protocol)
    if backend is None:
        verify_runtime(protocol["registration"], protocol["device"])
        from . import training as backend
    task = next(t for t in protocol["tasks"] if t["id"] == task_id)
    folder = root / "models" / task_id / "trajectory"

    def execute():
        f, v = (pd.read_parquet(root / f"inputs/{role}.parquet") for role in ("F", "V"))
        result = backend.fit_trajectory(f, v, folder, arm=task["arm"], seed=task["seed"],
            threads=protocol["threads"], device=protocol["device"], design=protocol["design"])
        if result["status"] != "complete":
            raise RuntimeError(f"Incomplete trajectory {task_id}; preserve its failure evidence")
        verify_live_code(protocol)
        return files(folder)

    checked_stage(root, task_id + "-trajectory", {"protocol": canonical_hash(protocol), "task": task}, execute)


def launch(root, mode, task_id):
    path = root / "logs" / f"{mode}-{task_id}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    count = "1" if mode == "cost-worker" else str(read_json(root / "protocol.json")["threads"])
    env.update({key: count for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    with path.open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "ids.model_training_redesign", "--" + mode, task_id,
                        "--output-dir", str(root)], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def train_all(root, protocol):
    for task in protocol["tasks"]:
        launch(root, "worker", task["id"])


def check_pairing(left, right):
    """Compare paired initial conditions and random streams only at shared steps."""
    fields = ("preprocessing_sha256", "numeric_bins_sha256", "initial_state_sha256", "F_identity_sha256",
              "initialization_seed", "dropout_seed", "sampler_seeds",
              "initial_dropout_rng_sha256", "initial_sampler_rng_sha256")
    if any(left[field] != right[field] for field in fields):
        raise ValueError("Paired initialization, preprocessing, bins or random seeds differ")
    common = None
    for name in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
        a, b = left[name], right[name]
        shared = sorted(set(a) & set(b), key=int)
        if not shared or any(a[s] != b[s] for s in shared):
            raise ValueError(f"Paired random stream differs: {name}")
        if common is not None and common != shared:
            raise ValueError("Paired draw and dropout audit steps differ")
        common = shared
    return {"initial_conditions_equal": True, "common_validation_steps": [int(x) for x in common],
            "draw_stream_equal": True, "dropout_stream_equal": True}


def freeze(root, protocol):
    artifacts, pairing = {}, {}
    max_steps = protocol["design"]["training"]["max_optimizer_steps"]
    for task in protocol["tasks"]:
        name = task["id"]
        verify_receipt(root, root / f"receipts/{name}-trajectory.json",
                       {"protocol": canonical_hash(protocol), "task": task})
        folder = root / "models" / name / "trajectory"
        summary = read_json(folder / "summary.json")
        if (summary["status"] != "complete" or summary["fit_calls"] != 1 or summary["final_refits"] != 0
                or not 1 <= summary["selected_step"] <= summary["optimizer_steps"] <= max_steps):
            raise ValueError(f"Invalid fit count or update budget: {name}")
        if sha256_file(folder / "best.pt") != sha256_file(folder / "model.pt"):
            raise ValueError(f"Selected checkpoint was replaced before delivery: {name}")
        artifacts[name] = {"path": (folder / "model.pt").relative_to(root).as_posix(),
                           "sha256": sha256_file(folder / "model.pt"),
                           "selected_step": summary["selected_step"],
                           "optimizer_steps": summary["optimizer_steps"]}
        pairing[name] = read_json(folder / "pairing.json")
    pairs = {str(seed): check_pairing(pairing[f"protocol_blind_s{seed}"],
                                     pairing[f"protocol_aware_s{seed}"])
             for seed in protocol["design"]["training"]["seeds"]}
    if len(artifacts) != 6:
        raise ValueError("Exactly six selected artifacts must be frozen")
    write_json(root / "freeze.json", {"created": utc_now(), "protocol": canonical_hash(protocol),
        "actual_fit_calls": 6, "final_refits": 0, "artifacts": artifacts,
        "paired_contracts": pairs, "implementation_contract_pass": True,
        "calibration_and_evaluation_scoring_started": False})
    return [root / "freeze.json"]


def verify_freeze(root, protocol):
    verify_receipt(root, root / "receipts/freeze.json", canonical_hash(protocol))
    frozen = read_json(root / "freeze.json")
    if (set(frozen["artifacts"]) != {t["id"] for t in protocol["tasks"]}
            or frozen["actual_fit_calls"] != 6 or frozen["final_refits"] != 0
            or not frozen["implementation_contract_pass"]):
        raise ValueError("All six verified selected models are required before C/E scoring")
    for item in frozen["artifacts"].values():
        if sha256_file(contained(root, item["path"])) != item["sha256"]:
            raise ValueError("A frozen selected artifact changed")
    return frozen


def score_task(root, protocol, task, backend=None):
    if backend is None:
        from . import training as backend
    frozen = verify_freeze(root, protocol)
    predictor = backend.load_predictor(contained(root, frozen["artifacts"][task["id"]]["path"]),
                                       threads=1, device="cpu")
    folder = root / "evidence" / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    for role in ("C", "ED"):
        frame = pd.read_parquet(root / f"inputs/{role}.parquet")
        scores = np.asarray(predictor.score(data.blind(frame)), dtype=np.float64)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid CPU scores: {task['id']}/{role}")
        saved = frame[data.IDENTITY].copy()
        saved["score"] = scores
        saved.to_parquet(folder / f"{role}-scores.parquet", index=False)
        if role == "C":
            write_json(folder / "calibration.json", evaluation.calibrate(
                scores, expected_normal_rows=protocol["design"]["roles"]["C"]["normal"]))
    verify_live_code(protocol)
    return files(folder)


def aligned_scores(root, task_id, role, frame):
    saved = pd.read_parquet(root / f"evidence/{task_id}/{role}-scores.parquet")
    pd.testing.assert_frame_equal(saved[data.IDENTITY], frame[data.IDENTITY], check_dtype=False)
    return saved.score.to_numpy(dtype=np.float64)


def report(root, protocol):
    frozen = verify_freeze(root, protocol)
    frame = pd.read_parquet(root / "inputs/ED.parquet")
    old = pd.read_parquet(root / "inputs/reference_decisions.parquet")
    decisions = {name: group.reset_index(drop=True) for name, group in old.groupby("model_id", sort=False)}
    units, days = [], []
    folder = root / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    for task in protocol["tasks"]:
        name = task["id"]
        unit, day, decision = evaluation.evaluate(frame, aligned_scores(root, name, "ED", frame),
            read_json(root / f"evidence/{name}/calibration.json"), model_id=name)
        units.extend(unit)
        days.extend(day)
        decisions[name] = decision
        decision.to_parquet(folder / f"{name}-decisions.parquet", index=False)
    write_json(folder / "units.json", units)
    write_json(folder / "utc_days.json", days)
    for name, rows in (("units", units), ("utc_days", days)):
        pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))}
                      for row in rows]).to_csv(folder / f"{name}.csv", index=False)
    write_json(folder / "comparison.json", evaluation.compare(decisions, protocol["design"],
        implementation_contract_pass=frozen["implementation_contract_pass"]))
    return files(folder)


def cost_worker(root, task_id, backend=None):
    from .costs import measure
    return measure(root, task_id, backend=backend)


def manifest(root):
    selected = [p for p in files(root) if p.name not in ("run.lock", "archive_manifest.json")
                and not p.name.endswith((".zip", ".zip.sha256", ".tmp"))]
    write_json(root / "archive_manifest.json", {"version": VERSION, "created": utc_now(),
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in selected}})


def verify(root):
    record = read_json(root / "archive_manifest.json")
    for rel, digest in record["files"].items():
        path = contained(root, rel)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Run artifact changed: {rel}")
    for receipt in (root / "receipts").glob("*.json"):
        verify_receipt(root, receipt)
    protocol, status = read_json(root / "protocol.json"), read_json(root / "status.json")
    if status["status"] == "complete":
        verify_freeze(root, protocol)
        expected = {"inputs", "sources", "freeze", "report"}
        expected |= {t["id"] + "-" + phase for t in protocol["tasks"]
                     for phase in ("trajectory", "scores", "cost")}
        if not expected <= {p.stem for p in (root / "receipts").glob("*.json")}:
            raise ValueError("Complete run is missing required receipts")
    return {"status": status["status"], "verified_files": len(record["files"]), "model_forwards": 0,
            "verification": "offline hashes and required stage completeness"}


def write_archive(root, validation, suffix="results"):
    output = root / f"{VERSION}-{suffix}.zip"
    record = read_json(root / "archive_manifest.json")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for rel in [*record["files"], "archive_manifest.json"]:
            archive.write(contained(root, rel), rel)
    output.with_suffix(".zip.sha256").write_text(sha256_file(output) + "  " + output.name + "\n", encoding="utf-8")
    return {**validation, "archive": str(output), "sha256": sha256_file(output)}


def pack(root):
    return write_archive(root, verify(root))


def archive_failure(root, error):
    broken = []
    for receipt in (root / "receipts").glob("*.json"):
        try:
            verify_receipt(root, receipt)
        except Exception as exc:
            broken.append({"receipt": receipt.relative_to(root).as_posix(), "error": str(exc)})
    result = {"status": "failed_evidence_only", "original_error_type": type(error).__name__,
              "original_error": str(error), "broken_receipts": broken,
              "completed_experiment_verified": False, "model_forwards": 0}
    write_json(root / "failure_integrity.json", result)
    manifest(root)
    return write_archive(root, result, "failed-evidence")


def run(root, *, r29, raw_dir, r26=None, threads=4, device="cuda:0", inputs_only=False):
    root, r29, raw_dir = Path(root).resolve(), Path(r29).resolve(), Path(raw_dir).resolve()
    r26 = None if r26 is None else Path(r26).resolve()
    with run_lock(root):
        if (root / "archive_manifest.json").exists():
            verify(root)
        protocol = initialize(root, r29, r26, raw_dir, threads, device)
        if inputs_only:
            if not (root / "status.json").exists():
                write_json(root / "status.json", {"status": "inputs_ready", "fits": 0, "model_forwards": 0})
            manifest(root)
            return verify(root)
        if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "complete":
            return pack(root)
        try:
            runtime = verify_runtime(protocol["registration"], device)
            if (root / "runtime.json").exists() and read_json(root / "runtime.json") != runtime:
                raise ValueError("Laboratory runtime changed across continuation")
            write_json(root / "runtime.json", runtime)
            own_windows_workers()
            with ResourceLog(root) as resource:
                resource.activity = "six paired F/V trajectories; no refits"
                train_all(root, protocol)
                verify_live_code(protocol)
                key = canonical_hash(protocol)
                checked_stage(root, "freeze", key, lambda: freeze(root, protocol))
                resource.activity = "all six selected checkpoints frozen; CPU C/E scoring"
                for task in protocol["tasks"]:
                    checked_stage(root, task["id"] + "-scores", key,
                                  lambda task=task: score_task(root, protocol, task))
                checked_stage(root, "report", key, lambda: report(root, protocol))
                resource.activity = "serial independent CPU cost workers; GPU throughput separately"
                for task in protocol["tasks"]:
                    def measure(task=task):
                        launch(root, "cost-worker", task["id"])
                        return [root / f"cost/{task['id']}.json"]
                    checked_stage(root, task["id"] + "-cost", key, measure)
            verify_live_code(protocol)
            write_json(root / "status.json", {"status": "complete", "fits": 6, "final_refits": 0,
                "finished": utc_now(), "comparison": "reports/comparison.json", "new_blind_confirmation": False})
        except BaseException as error:
            write_json(root / "status.json", {"status": "failed", "finished": utc_now(),
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "automatic_fit_retry": False})
            try:
                print(archive_failure(root, error), flush=True)
            except Exception as archive_error:
                print(f"Failure archive failed: {archive_error}; original exception follows", file=sys.stderr)
            raise
        manifest(root)
        return pack(root)
