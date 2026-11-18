"""Serial registered fits, global freeze, one evaluation and isolated costs."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import traceback
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from ids.common import canonical_hash, read_json, utc_now, write_json
from ids.flow_history.execution import ResourceLog, own_windows_workers, run_lock
from ids.model_training.runner import checked_stage, contained, files, verify_receipt
from .bindings import SEEDS, load_bound_inputs, sha256
from .protocol import (ARMS, REPO, VERSION, code_snapshot, specifications, tasks,
                       scoring_host, verify_live_code, verify_preflight, verify_runtime)
from .selection import VALIDATION_STEPS


def snapshot_sources(root, protocol):
    for rel in protocol["code"]:
        dest = root / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / rel).read_bytes().replace(b"\r\n", b"\n"))
    for field in ("registration", "source_manifest"):
        relative = ("configs/normal-subgroup-lab-20261118/registration.json" if field == "registration"
                    else protocol["registration"]["source_manifest_file"])
        dest = root / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((REPO / relative).read_bytes())
    return files(root / "source_snapshot")


def initialize(root, archive, threads, device):
    if isinstance(threads, bool) or threads != 1:
        raise ValueError("This registered GPU-training/CPU-selection worker requires threads=1")
    if root.is_relative_to(archive) or archive.is_relative_to(root):
        raise ValueError("Output must be separate from the original archive")
    registration, design = specifications()
    runtime = verify_runtime(registration)
    bindings_path = REPO / registration["input_bindings_file"]
    frames, bindings, checked = load_bound_inputs(archive, bindings_path=bindings_path)
    protocol = {"version": VERSION, "registration": registration, "design": design,
                "code": code_snapshot(), "tasks": tasks(design), "archive": str(archive),
                "threads": threads, "device": device, "runtime": runtime,
                "scoring_host": scoring_host(),
                "score_device": "cpu", "score_threads": 1, "score_batch_size": 256,
                "new_blind_confirmation": False}
    path = root / "protocol.json"
    if path.exists():
        if read_json(path) != protocol:
            raise ValueError("Existing run has different input, source or runtime identity")
    else:
        if any(p.name != "run.lock" for p in root.iterdir()):
            raise ValueError("Unregistered output directory is not empty")
        write_json(path, protocol)
    key = canonical_hash(protocol)
    checked_stage(root, "sources", key, lambda: snapshot_sources(root, protocol))

    def prepare():
        from .preflight import run as preflight
        preflight(archive, root / "preflight", bindings_path=bindings_path)
        audit = verify_preflight(root / "preflight", frames, bindings, design)
        write_json(root / "preflight_readiness.json", {
            **audit, "protocol_sha256": key, "design_sha256": registration["design_sha256"],
            "source_manifest_sha256": registration["source_manifest_sha256"],
            "inputs": checked, "scoring_host": protocol["scoring_host"],
            "generated_on_this_scoring_host": True})
        for role, frame in frames.items():
            folder = root / "inputs"
            folder.mkdir(exist_ok=True)
            frame.to_parquet(folder / f"{role}.parquet", index=False)
        return files(root / "preflight") + files(root / "inputs") + [root / "preflight_readiness.json"]

    checked_stage(root, "preflight", key, prepare)
    verify_preflight(root / "preflight", frames, bindings, design)
    return protocol


def train_task(root, task_id):
    from .training import fit_trajectory
    if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "failed":
        raise ValueError("Failed attempts cannot start or retry a trajectory")
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_runtime(protocol["registration"], protocol["device"])
    key = canonical_hash(protocol)
    verify_receipt(root, root / "receipts/preflight.json", key)
    task = next(t for t in protocol["tasks"] if t["id"] == task_id)
    frames = {role: pd.read_parquet(root / f"inputs/{role}.parquet") for role in ("F", "V")}
    bindings = read_json(REPO / protocol["registration"]["input_bindings_file"])
    original = Path(protocol["archive"]) / bindings["models"][f"protocol_blind_s{task['seed']}"]["path"]
    folder = root / "models" / task_id / "trajectory"

    def execute():
        result = fit_trajectory(frames, root / "preflight", original, folder,
            arm=task["arm"], seed=task["seed"], device=protocol["device"],
            threads=protocol["threads"], design=protocol["design"])
        if result["status"] != "complete":
            raise RuntimeError(f"Incomplete trajectory {task_id}; preserve failure evidence")
        verify_live_code(protocol)
        return files(folder)

    checked_stage(root, task_id + "-trajectory", {"protocol": key, "task": task}, execute)


def launch(root, mode, task_id):
    folder = root / "logs"
    folder.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    count = "1" if mode == "cost-worker" else str(read_json(root / "protocol.json")["threads"])
    env.update({key: count for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    with (folder / f"{mode}-{task_id}.log").open("a", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-m", "ids.normal_subgroup_training", "--" + mode, task_id,
                        "--output", str(root)], cwd=REPO, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def check_pairing(left, right):
    """Warmstart-specific full-budget pairing, with no shared-prefix shortcut."""
    if (left["target_arm"], right["target_arm"]) != ARMS or left["seed"] != right["seed"]:
        raise ValueError("Expected the same-seed transport/identity target pair")
    fields = ("original_artifact_sha256", "preflight_manifest_sha256", "preprocessing_sha256",
              "numeric_bins_sha256", "F_identity_sha256", "V_identity_sha256",
              "initial_state_sha256", "dropout_seed", "sampler_seeds",
              "initial_dropout_rng_sha256", "initial_sampler_rng_sha256",
              "optimizer_initial_state_entries", "cpu_score_batch_size", "validation_steps")
    if any(left[field] != right[field] for field in fields):
        raise ValueError("Paired warmstart, F/V anchors, optimizer or random-stream contract differs")
    if (left["optimizer_initial_state_entries"] != 0 or left["cpu_score_batch_size"] != 256
            or left["validation_steps"] != list(VALIDATION_STEPS)):
        raise ValueError("Pair requires fresh optimizer, CPU batch256 and all sixteen validation events")
    if ("initialization_seed" in left or "initialization_seed" in right) and (
            left.get("initialization_seed") != right.get("initialization_seed")):
        raise ValueError("Paired initialization metadata differs")
    expected = {str(step) for step in VALIDATION_STEPS}
    for field in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
        if set(left[field]) != expected or set(right[field]) != expected or left[field] != right[field]:
            raise ValueError(f"Paired complete sixteen-event stream differs: {field}")
    for pair in (left, right):
        checks = pair["selection_rng_unchanged_by_step"]
        if set(checks) != expected or not all(value is True for value in checks.values()):
            raise ValueError("Every CPU selection event must preserve the training RNG")
    return {"passed": True, "initial_conditions_equal": True,
            "common_validation_steps": list(VALIDATION_STEPS),
            "draw_stream_equal": True, "dropout_stream_equal": True,
            "F_V_and_preflight_equal": True, "optimizer_initial_state_entries": 0,
            "all_selection_events_preserve_rng": True}


def freeze_models(root, protocol):
    models, pairing = {}, {}
    key = canonical_hash(protocol)
    for task in protocol["tasks"]:
        name = task["id"]
        verify_receipt(root, root / f"receipts/{name}-trajectory.json", {"protocol": key, "task": task})
        folder = root / "models" / name / "trajectory"
        summary = read_json(folder / "summary.json")
        if (summary["status"] != "complete" or summary["arm"] != task["arm"]
                or summary["seed"] != task["seed"] or summary["fit_calls"] != 1
                or summary["final_refits"] != 0 or summary["optimizer_steps"] != 2224
                or summary["selected_step"] not in (0, *VALIDATION_STEPS)
                or summary["completed_validation_events"] != 16):
            raise ValueError(f"Invalid frozen trajectory budget: {name}")
        if sha256(folder / "best.pt") != sha256(folder / "model.pt"):
            raise ValueError("Selected artifact was replaced before delivery")
        if [row["step"] for row in read_json(folder / "history.json")] != list(VALIDATION_STEPS):
            raise ValueError("All sixteen validation events are required")
        item = {field: {"path": (folder / filename).relative_to(root / "models").as_posix(),
                        "sha256": sha256(folder / filename)} for field, filename in
                (("artifact", "model.pt"), ("selected_metrics", "selected_metrics.json"),
                 ("baseline_metrics", "baseline_metrics.json"))}
        models[name] = {**item, "arm": task["arm"], "seed": task["seed"],
                        "selected_step": summary["selected_step"], "optimizer_steps": 2224}
        pairing[name] = read_json(folder / "pairing.json")
    pairs = {}
    expected = {str(step) for step in VALIDATION_STEPS}
    for seed in SEEDS:
        a, b = (pairing[f"{arm}_s{seed}"] for arm in ARMS)
        for field in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
            if set(a[field]) != expected or set(b[field]) != expected:
                raise ValueError("Paired audit must cover all sixteen events, not a shared subset")
        pairs[str(seed)] = check_pairing(a, b)
    frozen = {"status": "all_six_artifacts_frozen", "models": models, "paired_contracts": pairs,
              "implementation_contract_pass": True, "actual_fit_calls": 6, "final_refits": 0,
              "total_optimizer_steps": 13344, "protocol_sha256": key,
              "design_sha256": protocol["registration"]["design_sha256"],
              "bindings_sha256": protocol["registration"]["input_bindings_sha256"],
              "preflight_manifest_sha256": sha256(root / "preflight/output_manifest.json"),
              "calibration_and_evaluation_scoring_started": False}
    write_json(root / "freeze.json", frozen)
    return [root / "freeze.json"]


def verify_freeze(root, protocol):
    verify_receipt(root, root / "receipts/freeze.json", canonical_hash(protocol))
    frozen = read_json(root / "freeze.json")
    if (frozen["status"] != "all_six_artifacts_frozen"
            or set(frozen["models"]) != {t["id"] for t in protocol["tasks"]}
            or frozen["actual_fit_calls"] != 6 or frozen["total_optimizer_steps"] != 13344
            or not frozen["implementation_contract_pass"]):
        raise ValueError("Incomplete global freeze")
    if (set(frozen["paired_contracts"]) != {str(seed) for seed in SEEDS}
            or not all(pair["passed"] is True for pair in frozen["paired_contracts"].values())):
        raise ValueError("Global freeze requires all three complete paired contracts")
    for item in frozen["models"].values():
        for field in ("artifact", "selected_metrics", "baseline_metrics"):
            if sha256(contained(root / "models", item[field]["path"])) != item[field]["sha256"]:
                raise ValueError("Frozen model or selection metrics changed")
    return frozen


def cost_worker(root, task_id):
    from .costs import measure_artifact
    if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "failed":
        raise ValueError("Failed attempts cannot start or retry cost measurement")
    protocol = read_json(root / "protocol.json")
    verify_live_code(protocol)
    verify_runtime(protocol["registration"], protocol["device"])
    frozen = verify_freeze(root, protocol)
    verify_receipt(root, root / "receipts/evaluation.json", canonical_hash(protocol))
    item = frozen["models"][task_id]["artifact"]
    folder = root / "evaluation" / task_id
    return measure_artifact(task_id=task_id, artifact=contained(root / "models", item["path"]),
        artifact_sha256=item["sha256"], frame=pd.read_parquet(root / "evaluation/E.parquet"),
        registered_scores=pd.read_parquet(folder / "E-scores.parquet"),
        threshold=read_json(folder / "calibration.json")["threshold"], gpu_device=protocol["device"],
        output_path=root / "cost" / f"{task_id}.json")


def manifest(root):
    selected = [p for p in files(root) if p.name not in ("run.lock", "archive_manifest.json")
                and not p.name.endswith((".zip", ".zip.sha256"))]
    write_json(root / "archive_manifest.json", {"version": VERSION, "created": utc_now(),
        "files": {p.relative_to(root).as_posix(): sha256(p) for p in selected}})


def verify(root):
    record = read_json(root / "archive_manifest.json")
    for relative, digest in record["files"].items():
        if sha256(contained(root, relative)) != digest:
            raise ValueError(f"Run artifact changed: {relative}")
    for receipt in (root / "receipts").glob("*.json"):
        verify_receipt(root, receipt)
    status = read_json(root / "status.json")
    if status["status"] == "complete":
        protocol = read_json(root / "protocol.json")
        verify_freeze(root, protocol)
        expected = {"sources", "preflight", "freeze", "evaluation"}
        expected |= {t["id"] + suffix for t in protocol["tasks"] for suffix in ("-trajectory", "-cost")}
        if not expected <= {p.stem for p in (root / "receipts").glob("*.json")}:
            raise ValueError("Complete run is missing required stage receipts")
    return {"status": status["status"], "verified_files": len(record["files"]), "new_model_forwards": 0}


def pack(root):
    validation = verify(root)
    suffix = "results" if validation["status"] == "complete" else "evidence"
    output = root / f"{VERSION}-{suffix}.zip"
    record = read_json(root / "archive_manifest.json")
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for rel in [*record["files"], "archive_manifest.json"]:
            archive.write(contained(root, rel), rel)
    output.with_suffix(".zip.sha256").write_text(sha256(output) + "  " + output.name + "\n", encoding="utf-8")
    return {**validation, "archive": str(output), "sha256": sha256(output)}


def final_outcome(comparison, cost_reports):
    """Apply deployment checks after the unchanged effect comparison."""
    expected = {f"{arm}_s{seed}" for seed in SEEDS for arm in ARMS}
    if set(cost_reports) != expected:
        raise ValueError("Final outcome requires all six deployment cost reports")
    deployment = {name: item["deployment_contract_pass"] is True for name, item in cost_reports.items()}
    all_deployment_pass = all(deployment.values())
    adoption_pass = bool(comparison["adoption_pass"] and all_deployment_pass)
    selected = comparison["retained_development_arm"]
    if selected == "transport_target":
        selected_deployment_pass = all_deployment_pass
        retained = selected if adoption_pass else "R30"
    elif selected == "identity_target":
        selected_deployment_pass = all(deployment[f"identity_target_s{seed}"] for seed in SEEDS)
        retained = selected if selected_deployment_pass else "R30"
    elif selected == "R30":
        selected_deployment_pass = None
        retained = "R30"
    else:
        raise ValueError("Unknown effect-comparison retained arm")
    return {"effect_comparison": comparison,
            "effect_comparison_scope": "effect criteria before deployment cost checks",
            "deployment_contract_pass": all_deployment_pass,
            "deployment_contract_pass_by_model": deployment,
            "selected_arm_deployment_contract_pass": selected_deployment_pass,
            "adoption_pass": adoption_pass, "retained_development_arm": retained,
            "automatic_deployment": False, "new_blind_confirmation": False}


def run(root, *, archive, threads=1, device="cuda:0", prepare_only=False):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    with run_lock(root):
        if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "failed":
            raise ValueError("Preserve the failed attempt; incomplete fits are not automatically retried")
        if (root / "archive_manifest.json").exists():
            verify(root)
        registration, _ = specifications()
        if not prepare_only:
            verify_runtime(registration, device)
        try:
            protocol = initialize(root, archive, threads, device)
            key = canonical_hash(protocol)
            if (root / "status.json").exists() and read_json(root / "status.json")["status"] == "complete":
                return pack(root)
            if prepare_only:
                write_json(root / "status.json", {"status": "preflight_ready", "fits": 0,
                    "optimizer_updates": 0, "C_E_access": False, "training_ready": True})
                manifest(root)
                return verify(root)
            own_windows_workers()
            write_json(root / "runtime.json", verify_runtime(registration, device))
            with ResourceLog(root) as resource:
                resource.activity = "six paired F/V trajectories; fixed 2224 steps each"
                for task in protocol["tasks"]:
                    launch(root, "worker", task["id"])
                verify_live_code(protocol)
                checked_stage(root, "freeze", key, lambda: freeze_models(root, protocol))
                resource.activity = "six models frozen; one C calibration and complete E regression"

                def evaluate():
                    from .evaluation import evaluate_frozen
                    verify_freeze(root, protocol)
                    evaluate_frozen(archive, root / "preflight", root / "models", root / "freeze.json",
                        root / "evaluation", design_path=REPO / registration["design_file"],
                        bindings_path=REPO / registration["input_bindings_file"])
                    verify_live_code(protocol)
                    return files(root / "evaluation")

                checked_stage(root, "evaluation", key, evaluate)
                resource.activity = "isolated serial CPU deployment costs; GPU throughput separately"
                for task in protocol["tasks"]:
                    def cost(task=task):
                        launch(root, "cost-worker", task["id"])
                        return [root / "cost" / f"{task['id']}.json"]
                    checked_stage(root, task["id"] + "-cost", key, cost)
            comparison = read_json(root / "evaluation/comparison.json")
            costs = {t["id"]: read_json(root / "cost" / f"{t['id']}.json") for t in protocol["tasks"]}
            write_json(root / "outcome.json", final_outcome(comparison, costs))
            verify_live_code(protocol)
            write_json(root / "status.json", {"status": "complete", "fits": 6, "final_refits": 0,
                "optimizer_updates": 13344, "finished": utc_now(), "outcome": "outcome.json"})
        except BaseException as error:
            write_json(root / "status.json", {"status": "failed", "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(), "automatic_fit_retry": False})
            manifest(root)
            try:
                print(pack(root), flush=True)
            except Exception as packing_error:
                print(f"Failure package could not be verified: {packing_error}", file=sys.stderr)
            raise
        manifest(root)
        return pack(root)
