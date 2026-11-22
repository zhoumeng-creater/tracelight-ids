"""Fixed-budget trajectories with exact recovery and predeclared F snapshots.

Only F enters this module. Recovery continues an interrupted trajectory at an
optimizer boundary; no learning rate, batch size, seed or objective is changed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ids.flow_history.model import _device, _sync
from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network, tabm_head_loss
from scripts.calibration_repair.data import Preprocessor, build_groups, file_digest, json_digest
from scripts.calibration_repair.training import (
    ClassSampler, FIXED_MODEL, Predictor, scoring_arithmetic, state_digest,
    _derived_seed, _seed, _rng_digest,
)
from scripts.comprehensive_batch.objective import ARMS, backward_objective, score_full, score_gradient

FORMAT = "comprehensive-fixed-snapshot-v1"
RECOVERY_FORMAT = "comprehensive-exact-recovery-v1"
FIXED_TRAINING = {
    "batch_size": 256, "draws_per_class": 128, "score_batch_size": 1024,
    "report_every": 100, "recovery_every": 100, "weight_decay": .0003,
    "gradient_clip_norm": 1.,
}


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    os.replace(temp, path)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    torch.save(value, temp)
    os.replace(temp, path)


def _require_initialization_only(root):
    allowed = {"job.json", "preprocessing.json", "groups.json"}
    # The first atomic recovery write can stop before replace(). No optimizer
    # update runs before that first successful commit, so this lone temporary
    # checkpoint is safe to recreate; other training artifacts are not.
    if any(not p.is_file() or
           (p.name != "recovery.pt.part" and p.name.removesuffix(".part") not in allowed)
           for p in Path(root).iterdir()):
        raise ValueError("Missing recovery point after the initialization-only phase")


def _rng(device):
    result = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def _restore_rng(value, device):
    torch.set_rng_state(value["cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(value["cuda"].cpu(), device)


def _sampler_state(sampler):
    return {"audit": sampler.audit(),
            "queues": {str(k): v.tolist() for k, v in sampler.queues.items()},
            "cursors": {str(k): v for k, v in sampler.cursors.items()},
            "cycles": {str(k): v for k, v in sampler.cycles.items()},
            "PCG64": {str(k): v.bit_generator.state for k, v in sampler.rngs.items()}}


def _restore_sampler(labels, seed, per_class, steps, expected):
    # hashlib's running state is not serializable. Replaying ONLY the CPU draws
    # reconstructs its state, both queues and PCG streams without any model work.
    sampler = ClassSampler(labels, seed=seed, per_class=per_class)
    exposure = np.zeros(len(labels), dtype=np.int64)
    for _ in range(steps):
        np.add.at(exposure, sampler.next(), 1)
    if _sampler_state(sampler) != expected:
        raise ValueError("Recovery sampler queues, RNG or draw digest differ")
    return sampler, exposure


def _vector(network):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1).cpu()
                      for p in network.parameters()]).double()


def _cosine(left, right):
    denominator = float(left.norm() * right.norm())
    return None if denominator == 0. else float(torch.dot(left, right)) / denominator


def gradient_probe(network, numeric, categorical, labels, groups, job, *, device,
                   batch_size, score_pass):
    """Full-F balanced member BCE and auxiliary gradients, eval/dropout off.

    This deterministic diagnostic is not the stochastic minibatch training loss.
    Existing gradients, training mode and random streams are restored exactly.
    """
    random = _rng(device)
    mode = network.training
    old_grads = [None if p.grad is None else p.grad.detach().clone() for p in network.parameters()]
    count = {label: int((labels == label).sum()) for label in (0, 1)}
    bce_value = 0.
    network.eval()
    try:
        network.zero_grad(set_to_none=True)
        for start, stop in score_pass.blocks:
            x = torch.from_numpy(numeric[start:stop]).to(device)
            cat = torch.from_numpy(categorical[start:stop]).to(device)
            y = torch.from_numpy(labels[start:stop]).to(device)
            logits = network(x, cat).squeeze(-1)
            per_row = F.binary_cross_entropy_with_logits(
                logits, y[:, None].expand_as(logits), reduction="none").mean(dim=1)
            weights = torch.where(y == 0, .5 / count[0], .5 / count[1])
            loss = (per_row * weights).sum()
            bce_value += float(loss.detach())
            loss.backward()
        bce = _vector(network)
        network.zero_grad(set_to_none=True)
        aux = backward_objective(network, numeric, categorical, groups, arm=job["arm"],
                                 device=device, batch_size=batch_size, score_pass=score_pass,
                                 weight=job["auxiliary_weight"])
        auxiliary = _vector(network)
        components, vectors = {}, {}
        for name, arm in (("D", "D_reference_alignment"), ("H", "H_reference_attack"),
                          ("G", "G_group_tail")):
            network.zero_grad(set_to_none=True)
            term = backward_objective(network, numeric, categorical, groups, arm=arm,
                                      device=device, batch_size=batch_size, score_pass=score_pass, weight=1.)
            vector = _vector(network)
            vectors[name] = vector
            components[name] = {"unweighted_penalty": term["penalty"],
                                "gradient_norm": float(vector.norm()),
                                "BCE_cosine": _cosine(bce, vector), "cost": term["cost"]}
        bn, an = float(bce.norm()), float(auxiliary.norm())
        dot = float(torch.dot(bce, auxiliary))
        return {"probe_definition": "full F class-balanced member BCE and weighted auxiliary; eval mode",
                "BCE": bce_value, "BCE_gradient_norm": bn,
                "weighted_auxiliary_gradient_norm": an,
                "BCE_auxiliary_gradient_dot": dot,
                "BCE_auxiliary_cosine": None if bn == 0 or an == 0 else dot / (bn * an),
                "combined_gradient_norm": float((bce + auxiliary).norm()),
                "unweighted_components_at_same_parameters": components,
                "component_cosines": {"D_H": _cosine(vectors["D"], vectors["H"]),
                                      "D_G": _cosine(vectors["D"], vectors["G"])},
                "auxiliary_weight": job["auxiliary_weight"],
                "BCE_forward_rows": len(labels), "BCE_backward_rows": len(labels),
                "auxiliary_cost": aux["cost"]}
    finally:
        network.train(mode)
        for p, grad in zip(network.parameters(), old_grads):
            p.grad = grad
        _restore_rng(random, device)


def _score_diagnostics(values, labels, rows, groups):
    scores = np.asarray(values, dtype=np.float64)
    per_row = np.logaddexp(0., scores) - labels * scores
    buckets = {"all": list(range(len(labels)))}
    if rows is not None:
        for i, row in enumerate(rows):
            for key in ("source:" + row.get("source_group", "unknown"),
                        "target:" + row.get("target_kind", "unknown"),
                        "source_target:" + row.get("source_group", "unknown") + ":" + row.get("target_kind", "unknown")):
                buckets.setdefault(key, []).append(i)
    distributions = {key: {"rows": len(indices), "ensemble_BCE": float(per_row[indices].mean()),
                           "positive_rows": int(labels[indices].sum()),
                           "score_min": float(scores[indices].min()),
                           "score_max": float(scores[indices].max())}
                     for key, indices in buckets.items()}
    objectives = {arm: score_gradient(scores, groups, arm)[1] for arm in ARMS}
    return {"deployed_score_groups": distributions, "objectives": objectives,
            "score_definition": "stable FP64 log-odds of mean member probability"}


def _exposure_report(exposure, rows, labels):
    result = {}
    for i, n in enumerate(exposure):
        row = None if rows is None else rows[i]
        key = f"label:{int(labels[i])}" if row is None else f"{row.get('source_group')}|{row.get('target_kind')}"
        result.setdefault(key, []).append(int(n))
    return {key: {"distinct_rows": len(values), "draws": sum(values),
                  "minimum_draws_per_row": min(values), "maximum_draws_per_row": max(values),
                  "unique_drawn": sum(n > 0 for n in values)} for key, values in result.items()}


def _fingerprint(numeric, categorical, labels, groups, job, spec, metadata):
    return json_digest({"job": job, "spec": spec, "groups": groups, "metadata": metadata,
                        "arrays": [hashlib.sha256(v.tobytes()).hexdigest()
                                   for v in (numeric, categorical, labels)]})


def train_loop(network, numeric, categorical, labels, groups, *, job, device,
               output_dir, training_spec=None, rows=None, artifact_metadata=None,
               resume=False, interrupt_after_step=None):
    """Synthetic-testable core; production fit alone enforces the frozen matrix."""
    spec = dict(FIXED_TRAINING if training_spec is None else training_spec)
    if job["arm"] not in ARMS or not 0 < job["learning_rate"] or not 0 < job["auxiliary_weight"]:
        raise ValueError("Invalid registered objective or optimization setting")
    checkpoints = list(job["checkpoints"])
    if (checkpoints != sorted(set(checkpoints)) or not checkpoints or checkpoints[0] < 1
            or checkpoints[-1] != job["optimizer_steps"]
            or spec["batch_size"] != 2 * spec["draws_per_class"]
            or min(spec["report_every"], spec["recovery_every"], spec["score_batch_size"]) < 1):
        raise ValueError("Expected ordered fixed checkpoints and balanced positive batch")
    device = torch.device(device)
    numeric, categorical = np.asarray(numeric, dtype=np.float32), np.asarray(categorical, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(artifact_metadata or {})
    fingerprint = _fingerprint(numeric, categorical, labels, groups, job, spec, metadata)
    recovery_path = root / "recovery.pt"
    initialization_recovered = bool(resume and not recovery_path.is_file())
    if initialization_recovered:
        _require_initialization_only(root)
        resume = False
    if not resume and recovery_path.exists():
        raise FileExistsError("A recovery trajectory already exists; explicitly resume it")
    optimizer = torch.optim.AdamW(network.parameters(), lr=job["learning_rate"], weight_decay=spec["weight_decay"])
    sampler = ClassSampler(labels, seed=job["seed"], per_class=spec["draws_per_class"])
    exposure = np.zeros(len(labels), dtype=np.int64)
    dropout_seed = _derived_seed(job["seed"], "dropout")
    _seed(dropout_seed, device)
    initial_rng = _rng_digest(device)
    initial = state_digest(network.state_dict())
    completed = 0
    snapshots, history, diagnostic_points = [], [], []
    costs = {"bce_forward_rows": 0, "bce_backward_rows": 0,
             "transfer_discovery_forward_rows": 0, "transfer_replay_forward_rows": 0,
             "transfer_replay_backward_rows": 0, "transfer_replay_blocks": 0,
             "bce_seconds": 0., "transfer_seconds": 0., "optimizer_seconds": 0.,
             "diagnostic_seconds": 0., "diagnostic_score_rows": 0,
             "probe_BCE_forward_rows": 0, "probe_BCE_backward_rows": 0,
             "probe_auxiliary_replay_rows": 0, "probe_component_replay_rows": 0,
             "active_optimizer_seconds": 0.}
    window = {"bce": 0., "auxiliary": 0., "gradient_norm": 0., "max_gradient_norm": 0.,
              "update_norm": 0., "clipped_steps": 0, "steps": 0}
    if resume:
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
        if recovery.get("format") != RECOVERY_FORMAT or recovery.get("fingerprint") != fingerprint:
            raise ValueError("Recovery job, input, groups, preprocessing or recipe changed")
        if state_digest(recovery["model_state"]) != recovery["model_state_sha256"]:
            raise ValueError("Recovery model state binding changed")
        network.load_state_dict(recovery["model_state"])
        optimizer.load_state_dict(recovery["optimizer_state"])
        completed = recovery["completed_step"]
        sampler, exposure = _restore_sampler(labels, job["seed"], spec["draws_per_class"], completed, recovery["sampler"])
        initial, initial_rng = recovery["initial_state_sha256"], recovery["initial_dropout_rng_sha256"]
        snapshots, history = recovery["snapshots"], recovery["history"]
        diagnostic_points, costs, window = recovery["diagnostic_points"], recovery["cost"], recovery["window"]
        _restore_rng(recovery["random_states"], device)
        for item in snapshots:
            if file_digest(root / item["path"]) != item["sha256"]:
                raise ValueError("Completed budget snapshot changed before recovery")
    x, cats, target = (torch.from_numpy(value).to(device) for value in (numeric, categorical, labels))

    def checkpoint():
        state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        _save(recovery_path, {"format": RECOVERY_FORMAT, "fingerprint": fingerprint,
                             "completed_step": completed, "model_state": state,
                             "model_state_sha256": state_digest(state), "optimizer_state": optimizer.state_dict(),
                             "random_states": _rng(device), "sampler": _sampler_state(sampler),
                             "initial_state_sha256": initial, "initial_dropout_rng_sha256": initial_rng,
                             "snapshots": snapshots, "history": history, "diagnostic_points": diagnostic_points,
                             "cost": costs, "window": window})

    def persist_cost(step):
        record = {"job_id": job["job_id"], "step": step, "cost": deepcopy(costs)}
        _write(root / "cost.json", record)
        return record

    def record_snapshot(step):
        _sync(device)
        begun = time.perf_counter()
        score_pass = score_full(network, numeric, categorical, device=device, batch_size=spec["score_batch_size"])
        diagnostics = _score_diagnostics(score_pass.scores.numpy(), labels, rows, groups)
        probe = gradient_probe(network, numeric, categorical, labels, groups, job, device=device,
                               batch_size=spec["score_batch_size"], score_pass=score_pass)
        diagnostics.update(step=step, gradient_probe=probe, sampling=sampler.audit(),
                           exposure=_exposure_report(exposure, rows, labels))
        directory = root / "snapshots" / f"step{step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        model_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        model = {"format": FORMAT, "job": job, "job_sha256": json_digest(job),
                 "step": step, "evaluation_checkpoint": step in checkpoints,
                 "training_spec": spec, "state_dict": model_state,
                 "state_sha256": state_digest(model_state), "initial_state_sha256": initial,
                 "score_batch_size": spec["score_batch_size"], **metadata}
        artifact = directory / "model.pt"
        if artifact.exists():
            existing = torch.load(artifact, map_location="cpu", weights_only=True)
            if (existing.get("state_sha256") != model["state_sha256"]
                    or existing.get("job_sha256") != model["job_sha256"] or existing.get("step") != step):
                raise ValueError("Recovery replay disagrees with an existing budget snapshot")
        else:
            _save(artifact, model)
        # These files are regenerated only when replaying uncommitted post-crash
        # work. Frozen model bytes are verified and never overwritten.
        score_path = directory / "F_scores.npz"
        with score_path.with_name(score_path.name + ".part").open("wb") as handle:
            np.savez_compressed(handle, scores=score_pass.scores.numpy(), exposure=exposure)
        os.replace(score_path.with_name(score_path.name + ".part"), score_path)
        _write(directory / "F_diagnostics.json", diagnostics)
        identity_rows = [str(i) for i in range(len(labels))] if rows is None else [r["flow_id"] for r in rows]
        _write(root / "F_score_identities.json", identity_rows)
        item = {"step": step, "path": artifact.relative_to(root).as_posix(),
                "sha256": file_digest(artifact), "bytes": artifact.stat().st_size,
                "state_sha256": model["state_sha256"], "sampler_draw_sha256": sampler.audit()["draw_sha256"],
                "diagnostics": (directory / "F_diagnostics.json").relative_to(root).as_posix(),
                "diagnostics_sha256": file_digest(directory / "F_diagnostics.json"),
                "F_scores": score_path.relative_to(root).as_posix(), "F_scores_sha256": file_digest(score_path)}
        diagnostic_points.append(step)
        costs["diagnostic_score_rows"] += len(labels)
        costs["probe_BCE_forward_rows"] += probe["BCE_forward_rows"]
        costs["probe_BCE_backward_rows"] += probe["BCE_backward_rows"]
        costs["probe_auxiliary_replay_rows"] += probe["auxiliary_cost"]["replay_backward_rows"]
        costs["probe_component_replay_rows"] += sum(
            value["cost"]["replay_backward_rows"]
            for value in probe["unweighted_components_at_same_parameters"].values())
        _sync(device)
        costs["diagnostic_seconds"] += time.perf_counter() - begun
        cost_record = persist_cost(step)
        _write(directory / "cost.json", cost_record)
        item.update(cost=deepcopy(costs),
                    cost_record=(directory / "cost.json").relative_to(root).as_posix(),
                    cost_record_sha256=file_digest(directory / "cost.json"))
        if step:
            snapshots.append(item)
            _write(root / "snapshots.json", snapshots)

    if not resume:
        # Commit step zero before the expensive diagnostic so an interrupted
        # initial probe can resume without restarting an optimizer trajectory.
        checkpoint()
    if completed == 0 and 0 not in diagnostic_points:
        record_snapshot(0)
        checkpoint()
    heartbeat = time.perf_counter()
    for step in range(completed + 1, job["optimizer_steps"] + 1):
        started = time.perf_counter()
        network.train()
        draw = sampler.next()
        np.add.at(exposure, draw, 1)
        indices = torch.from_numpy(draw).to(device)
        optimizer.zero_grad(set_to_none=True)
        segment = time.perf_counter()
        logits = network(x[indices], cats[indices])
        if logits.dtype != torch.float32:
            raise ValueError("Expected FP32 member logits")
        bce = tabm_head_loss(logits, target[indices])
        if not bool(torch.isfinite(bce)):
            raise FloatingPointError("Nonfinite member BCE")
        bce.backward()
        _sync(device)
        costs["bce_seconds"] += time.perf_counter() - segment
        costs["bce_forward_rows"] += len(draw)
        costs["bce_backward_rows"] += len(draw)
        auxiliary = None
        if job["arm"] != "A_BCE":
            segment = time.perf_counter()
            auxiliary = backward_objective(network, numeric, categorical, groups, arm=job["arm"],
                                           device=device, batch_size=spec["score_batch_size"],
                                           weight=job["auxiliary_weight"])
            _sync(device)
            costs["transfer_seconds"] += time.perf_counter() - segment
            extra = auxiliary["cost"]
            for destination, source in (("transfer_discovery_forward_rows", "discovery_forward_rows_this_call"),
                                        ("transfer_replay_forward_rows", "replay_forward_rows"),
                                        ("transfer_replay_backward_rows", "replay_backward_rows"),
                                        ("transfer_replay_blocks", "replay_blocks")):
                costs[destination] += extra[source]
        segment = time.perf_counter()
        norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), spec["gradient_clip_norm"], error_if_nonfinite=True))
        # Parameter displacement is measured only at registered milestone updates.
        before = [p.detach().clone() for p in network.parameters()] if step in checkpoints else None
        optimizer.step()
        displacement = None if before is None else float(torch.sqrt(sum(
            (p.detach().double() - old.double()).square().sum() for p, old in zip(network.parameters(), before))))
        _sync(device)
        costs["optimizer_seconds"] += time.perf_counter() - segment
        costs["active_optimizer_seconds"] += time.perf_counter() - started
        completed = step
        window["bce"] += float(bce.detach())
        window["auxiliary"] += 0. if auxiliary is None else auxiliary["penalty"]
        window["gradient_norm"] += norm
        window["max_gradient_norm"] = max(window["max_gradient_norm"], norm)
        window["clipped_steps"] += int(norm > spec["gradient_clip_norm"])
        window["steps"] += 1
        report_due = step % spec["report_every"] == 0 or step in checkpoints
        if report_due:
            history.append({"step": step, "window_steps": window["steps"],
                            "mean_path_member_BCE": window["bce"] / window["steps"],
                            "mean_path_auxiliary": window["auxiliary"] / window["steps"],
                            "preclip_mean_gradient_norm": window["gradient_norm"] / window["steps"],
                            "preclip_max_gradient_norm": window["max_gradient_norm"],
                            "clipped_steps": window["clipped_steps"], "milestone_parameter_update_norm": displacement,
                            "last_pre_update_auxiliary": auxiliary, "sampling": sampler.audit(),
                            "dropout_rng_sha256": _rng_digest(device), "cumulative_cost": deepcopy(costs)})
            _write(root / "history.json", history)
            persist_cost(step)
            window = {key: 0 if key in ("steps", "clipped_steps") else 0. for key in window}
        if step in checkpoints:
            record_snapshot(step)
            # The report at a milestone includes its completed diagnostics.
            history[-1]["cumulative_cost"] = deepcopy(costs)
            _write(root / "history.json", history)
        if step % spec["recovery_every"] == 0 or step in checkpoints:
            checkpoint()
        if report_due or time.perf_counter() - heartbeat >= 30. or step == 1:
            print(json.dumps({"job_id": job["job_id"], "step": step, "optimizer_steps": job["optimizer_steps"],
                              "active_optimizer_seconds": costs["active_optimizer_seconds"],
                              "estimated_remaining_optimizer_seconds": costs["active_optimizer_seconds"]
                              / step * (job["optimizer_steps"] - step)}), flush=True)
            heartbeat = time.perf_counter()
        if interrupt_after_step == step:
            # Only synthetic tests use this injected operational interruption.
            checkpoint()
            raise InterruptedError("Synthetic interruption after committed optimizer step")
    return {"status": "complete", "optimizer_steps": completed, "snapshots": snapshots,
            "initial_state_sha256": initial, "initial_dropout_rng_sha256": initial_rng,
            "final_dropout_rng_sha256": _rng_digest(device), "sampler_draw_sha256": sampler.audit()["draw_sha256"],
            "sampling": sampler.audit(), "exposure": _exposure_report(exposure, rows, labels),
            "cost": costs, "diagnostic_points": diagnostic_points, "history": history,
            "state_sha256": state_digest(network.state_dict()),
            "resumed_this_invocation": bool(resume or initialization_recovered),
            "recovery_checkpoint": "recovery.pt"}


def _registered_job(job):
    from scripts.comprehensive_batch.matrix import build_plan
    plan = build_plan()
    jobs = plan["tabm_jobs"] if isinstance(plan, dict) else plan
    candidates = [item for item in jobs if item.get("job_id") == job.get("job_id")]
    if len(candidates) != 1 or job != candidates[0]:
        raise ValueError("Production job differs from the registered comprehensive matrix")


def fit(rows, preprocessor, output_dir, *, job, device="cuda:0", threads=4, resume=False):
    _registered_job(job)
    if (len(rows) != 13252 or [r["flow_id"] for r in rows] != sorted(r["flow_id"] for r in rows)
            or any(r["dataset_role"] not in ("F_base", "F_expansion") for r in rows)):
        raise ValueError("Production fitting requires canonical registered F only")
    labels = np.asarray([r["label"] for r in rows], dtype=np.float32)
    if (int((labels == 0).sum()), int((labels == 1).sum())) != (12747, 505):
        raise ValueError("Production F class counts differ")
    root = Path(output_dir)
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError("Preserve existing job output; use exact resume")
    root.mkdir(parents=True, exist_ok=True)
    initialization_recovered = resume and not (root / "recovery.pt").is_file()
    if initialization_recovered:
        # Prior to train_loop's first recovery commit, fit writes only these
        # bindings and performs initialization; no optimizer step can execute.
        _require_initialization_only(root)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    actual = _device(device, threads)
    if actual.type != "cuda":
        raise ValueError("Production fitting requires laboratory CUDA")
    numeric, categorical = preprocessor.transform(rows)
    if numeric.shape != (13252, 5) or categorical.shape != (13252, 8):
        raise ValueError("Production encoding differs from the frozen contract")
    parameters = _parameters(FIXED_MODEL)
    bins, constants = _numeric_bins(numeric, parameters)
    groups = build_groups(rows, Path(__file__).resolve().parents[2])
    preprocessing = preprocessor.to_dict()
    metadata = {"parameters": parameters, "preprocessing": preprocessing, "numeric_bins": bins,
                "constant_bin_columns": constants, "preprocessing_sha256": json_digest(preprocessing),
                "numeric_bins_sha256": json_digest(bins),
                "F_identity_sha256": json_digest([r["flow_id"] for r in rows])}
    if initialization_recovered:
        for name, expected in (("job.json", job), ("preprocessing.json", preprocessing), ("groups.json", groups)):
            path = root / name
            if path.exists() and json.loads(path.read_text(encoding="utf8")) != expected:
                raise ValueError("Interrupted initialization binding changed")
    _write(root / "job.json", job)
    _write(root / "preprocessing.json", preprocessing)
    _write(root / "groups.json", groups)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(actual)
    with scoring_arithmetic(), torch.random.fork_rng(devices=[actual.index or 0]):
        _seed(job["seed"], actual)
        network = make_tabm_network(5, preprocessor.cat_cardinalities, parameters, numeric_bins=bins).to(actual)
        result = train_loop(network, numeric, categorical, labels, groups, job=job, device=actual,
                            output_dir=root, rows=rows, artifact_metadata=metadata,
                            resume=bool(resume and not initialization_recovered))
        parameters_count = sum(p.numel() for p in network.parameters())
        result["cost"].update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(actual),
                               cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(actual))
    result.pop("history")
    summary = {"job": job, "fit_trajectories": 1, "final_refits": 0, "fit_rows": len(rows),
               "initialization_recovered_without_optimizer_updates": initialization_recovered,
               "parameter_count": parameters_count, "invocation_seconds": time.perf_counter() - started,
               **metadata, **result}
    _write(root / "summary.json", summary)
    return summary


def load_predictor(artifact, *, device="cuda:0", threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    job = state.get("job", {})
    _registered_job(job)
    if (state.get("format") != FORMAT or state.get("job_sha256") != json_digest(job)
            or state.get("step") not in job["checkpoints"] or not state.get("evaluation_checkpoint")
            or state.get("training_spec") != FIXED_TRAINING
            or state.get("parameters") != _parameters(FIXED_MODEL)):
        raise ValueError("Unregistered or diagnostic-only comprehensive snapshot")
    if (state_digest(state["state_dict"]) != state["state_sha256"]
            or json_digest(state["preprocessing"]) != state["preprocessing_sha256"]
            or json_digest(state["numeric_bins"]) != state["numeric_bins_sha256"]
            or state["score_batch_size"] != FIXED_TRAINING["score_batch_size"]):
        raise ValueError("Snapshot state or fitted preprocessing changed")
    preprocessor = Preprocessor.from_dict(state["preprocessing"])
    actual = _device(device, threads)
    with torch.random.fork_rng(devices=[]):
        network = make_tabm_network(5, preprocessor.cat_cardinalities, state["parameters"], numeric_bins=state["numeric_bins"])
    network.load_state_dict(state["state_dict"])
    return Predictor(network.to(actual).eval(), preprocessor, actual, state["score_batch_size"])
