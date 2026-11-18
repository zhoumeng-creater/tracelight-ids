"""One fixed-budget F-only adaptation, selected through deterministic CPU F/V scores."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from ids.model_training_redesign import training as original
from ids.model_training_redesign.sampling import (
    GroupSampler, IDENTITY, canonical_frame, derived_seed, group_indices,
)
from .bindings import DEFAULT_BINDINGS, SEEDS, sha256
from .groups import map_groups
from .losses import BatchTargets, training_step
from .selection import Selector, metrics_to_json, selection_metrics


REPO = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN = REPO / "configs/normal-subgroup-refinement-20261118/design.json"
ARMS = ("transport_target", "identity_target")
STEPS = tuple(range(139, 2225, 139))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(metrics_to_json(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _validate_production(frames, artifact, *, arm, seed, threads, design):
    if set(frames) != {"F", "V"}:
        raise ValueError("Training accepts exactly F and V; C/E are forbidden")
    if (arm not in ARMS or isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
            or seed not in SEEDS or isinstance(threads, bool) or threads != 1):
        raise ValueError("Expected a registered target arm/seed and CPU selection threads=1")
    frozen = _read(DEFAULT_DESIGN)
    for key in ("version", "model", "roles", "groups", "targets", "loss", "training", "selection"):
        if design.get(key) != frozen[key]:
            raise ValueError(f"Production design differs from the frozen v2 contract: {key}")
    t = design["training"]
    if (t["optimizer_steps_per_trajectory"] != 2224 or t["validation_every_steps"] != 139
            or t["validation_events"] != 16 or t["sampler"]["draws_per_group"] != 64
            or t["performance_early_stop"] or t["final_refits"] != 0
            or t["optimizer"] != {"name": "AdamW", "lr": .0002, "weight_decay": .0003,
                                  "betas": [.9, .999], "eps": 1e-8, "state": "reset_same_for_pair"}
            or design["loss"]["normal_weight"] != 1 or design["loss"]["positive_weight"] != 1):
        raise ValueError("Production updates, optimizer and loss weights are fixed")
    for role, frame in frames.items():
        counts = (len(frame), int(frame.label.eq(0).sum()), int(frame.label.eq(1).sum()))
        expected = design["roles"][role]
        if counts != (expected["rows"], expected["normal"], expected["positive"]):
            raise ValueError(f"Unexpected production {role} row counts")
    if {key: len(value) for key, value in group_indices(frames["F"]).items()} != t["sampler"]["groups"]:
        raise ValueError("Production F source/class counts differ from the registered sampler")
    binding = _read(DEFAULT_BINDINGS)["models"][f"protocol_blind_s{seed}"]
    if sha256(artifact) != binding["sha256"]:
        raise ValueError("Original artifact differs from the registered same-seed R30 model")


def _verified_preflight(root, frames, seed, artifact):
    """Verify frozen payload bytes and exact canonical row/metadata alignment."""
    root = Path(root).resolve()
    status = _read(root / "status.json")
    if (status.get("status") != "zero_update_preflight_complete"
            or status.get("optimizer_updates") != 0 or status.get("new_fits") != 0
            or status.get("C_E_access") is not False):
        raise ValueError("Training requires a complete zero-update F/V preflight")
    manifest_path = root / "output_manifest.json"
    expected_manifest = status["output_manifest"]
    if (expected_manifest["path"] != manifest_path.name
            or expected_manifest["sha256"] != sha256(manifest_path)
            or expected_manifest["bytes"] != manifest_path.stat().st_size):
        raise ValueError("Preflight output manifest binding changed")
    manifest = _read(manifest_path)["files"]
    key = f"protocol_blind_s{seed}"
    required = {"groups.json", "identities.json", "input_receipt.json"}
    required.update(f"{key}/{name}" for name in (
        "targets.npz", "reference_scores.npz", "target_metadata.json", "preprocessing.json"))
    if not required.issubset(manifest):
        raise ValueError("Preflight manifest is missing a required training payload")
    for relative, item in manifest.items():
        path = (root / relative).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]):
            raise ValueError(f"Preflight payload binding changed: {relative}")
    if status["models"][key]["artifact_sha256"] != sha256(artifact):
        raise ValueError("Preflight reference belongs to a different original artifact")
    identities = _read(root / "identities.json")
    for role, frame in frames.items():
        if frame[IDENTITY].values.tolist() != identities[role]:
            raise ValueError(f"Preflight {role} physical identity order changed")
    groups = _read(root / "groups.json")
    with np.load(root / key / "targets.npz", allow_pickle=False) as values:
        targets = {name: values[name].copy() for name in values.files}
    with np.load(root / key / "reference_scores.npz", allow_pickle=False) as values:
        reference = {name: values[name].copy() for name in values.files}
    metadata = _read(root / key / "target_metadata.json")
    if metadata["equivalence"]["reference_inconsistent_classes"]:
        raise ValueError("Formal reference scores disagree for identical inputs")
    for role, saved in (("F", targets["group_ids"]), ("V", reference["V_group_ids"])):
        actual = map_groups(frames[role], groups, role=role, require_all=True)
        if not np.array_equal(actual, saved):
            raise ValueError(f"Preflight {role} group assignment/metadata changed")
    if not np.array_equal(targets["reference_scores"], reference["F"]):
        raise ValueError("F target anchors and reference score file disagree")
    if not np.array_equal(targets["identity_target"], reference["F"]):
        raise ValueError("Identity control must retain the original F scores")
    for role in ("F", "V"):
        if reference[role].shape != (len(frames[role]),) or not np.isfinite(reference[role]).all():
            raise ValueError(f"Invalid frozen {role} reference scores")
    return targets, reference, metadata, _read(root / key / "preprocessing.json"), expected_manifest


def _cpu_scores(predictor, network, arrays, actual):
    """Use a separate CPU model; neither moves training parameters nor advances RNG."""
    before = original._rng_state(actual)
    predictor.network.load_state_dict({name: value.detach().cpu()
                                       for name, value in network.state_dict().items()})
    predictor.network.eval()
    result = {role: predictor.score_transformed(*values) for role, values in arrays.items()}
    after = original._rng_state(actual)
    if original._state_sha256(before) != original._state_sha256(after):
        raise RuntimeError("CPU selection consumed training RNG")
    if any(not np.isfinite(scores).all() for scores in result.values()):
        raise FloatingPointError("Nonfinite CPU selection score")
    if any(p.device.type != "cpu" for p in predictor.network.parameters()):
        raise AssertionError("Selection mirror must remain on CPU")
    return result


def _optimizer_steps(optimizer):
    return max((int(state["step"]) for state in optimizer.state.values() if "step" in state), default=0)


def _save_adapted(path, network, template, *, arm, seed, step, sampler, actual, original_hash):
    import torch

    state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
    payload = dict(template)
    payload.update(
        arm="protocol_blind", target_arm=arm, seed=seed, selected_step=step,
        original_artifact_sha256=original_hash, adaptation="normal-subgroup-v2",
        state_dict=state, state_sha256=original._state_sha256(state), final_refits=0,
        dropout_rng_state=original._rng_state(actual), rng_step=step, sampling_state=sampler.state())
    torch.save(payload, path)


def _fit(frames, preflight_dir, artifact, root, *, arm, seed, device, threads, design,
         progress, validation_steps=STEPS):
    """Private core; tiny tests inject a schedule, the public entry never exposes it."""
    import torch
    from ids.flow_history.model import _device, _sync

    started = time.perf_counter()
    frames = {role: canonical_frame(frame) for role, frame in frames.items()}
    if set(frames) != {"F", "V"}:
        raise ValueError("Training accepts exactly F and V")
    targets, reference, metadata, preprocessing, manifest = _verified_preflight(
        preflight_dir, frames, seed, artifact)
    original_hash = sha256(artifact)
    template = torch.load(artifact, map_location="cpu", weights_only=True)
    if template["arm"] != "protocol_blind" or template["seed"] != seed:
        raise ValueError("Expected the same-seed original blind artifact")
    predictor = original.load_predictor(artifact, threads=1, device="cpu")
    if predictor.preprocessor.to_dict() != preprocessing:
        raise ValueError("Original preprocessing differs from preflight")
    for field in ("k", "n_blocks", "d_block", "dropout"):
        if template["parameters"][field] != design["model"][field]:
            raise ValueError(f"Original architecture differs: {field}")
    arrays = {role: predictor.preprocessor.transform(frame) for role, frame in frames.items()}
    labels = frames["F"].label.to_numpy(dtype=np.float32)
    v_normal = frames["V"].label.to_numpy() == 0
    expected_groups = tuple(sorted(metadata["group_counts"]))
    if metadata["group_counts"] != {g: int(np.sum(targets["group_ids"] == g)) for g in expected_groups}:
        raise ValueError("Frozen F group sizes changed")
    normal = labels == 0
    chosen = targets[arm]
    if chosen.shape != labels.shape or not np.isfinite(chosen).all():
        raise ValueError("Invalid row-aligned normal target table")
    if np.any(chosen[normal] > reference["F"][normal]) or not np.array_equal(
            chosen[~normal], reference["F"][~normal]):
        raise ValueError("Frozen target raised a normal or changed a positive anchor")
    group_sizes = np.array([metadata["group_counts"].get(g, 0) for g in targets["group_ids"]])
    b = design["training"]["sampler"]["draws_per_group"]
    sampler = GroupSampler(frames["F"], seed=seed, group_batch_size=b)
    profile_counts = tuple(len(sampler.groups[p + "0"]) for p in ("A", "B"))
    dropout_seed = derived_seed(seed, "normal-subgroup-dropout-v2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous_threads = torch.get_num_threads()
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32)
    actual = _device(device, threads)
    history = []
    pairing = {
        "target_arm": arm, "seed": seed, "original_artifact_sha256": original_hash,
        "preflight_manifest_sha256": manifest["sha256"],
        "preprocessing_sha256": template["preprocessing_sha256"],
        "numeric_bins_sha256": template["numeric_bins_sha256"],
        "F_identity_sha256": original._hash_json(frames["F"][IDENTITY].values.tolist()),
        "V_identity_sha256": original._hash_json(frames["V"][IDENTITY].values.tolist()),
        "normal_target_sha256": _array_hash(chosen[normal]),
        "initial_state_sha256": template["state_sha256"],
        "initialization_seed": seed,
        "dropout_seed": dropout_seed, "sampler_seeds": sampler.seeds,
        "initial_sampler_rng_sha256": sampler.rng_sha256(),
        "optimizer_initial_state_entries": 0,
        "validation_steps": list(validation_steps), "cpu_score_batch_size": predictor.batch_size,
        "draw_chain_by_validation_step": {}, "dropout_rng_by_validation_step": {},
        "selection_rng_unchanged_by_step": {},
    }
    _write(root / "preprocessing.json", preprocessing)
    _write(root / "numeric_bins.json", template["numeric_bins"])
    _write(root / "score_identities.json", {r: f[IDENTITY].values.tolist() for r, f in frames.items()})
    update_seconds = 0.
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=[] if actual.type == "cpu" else [actual.index]):
            network = deepcopy(predictor.network).to(actual)
            original._seed_torch(dropout_seed, actual)
            pairing["initial_dropout_rng_sha256"] = original._state_sha256(original._rng_state(actual))
            current = _cpu_scores(predictor, network, arrays, actual)
            if any(current[r].tobytes() != reference[r].tobytes() for r in ("F", "V")):
                raise ValueError("Step0 CPU outputs differ from frozen preflight reference")
            baseline = selection_metrics(current["F"], frames["F"], current["V"], frames["V"],
                                         reference["V_group_ids"][v_normal], expected_groups)
            selector = Selector(baseline)
            _write(root / "baseline_metrics.json", baseline)
            _write(root / "selected_metrics.json", baseline)
            shutil.copyfile(artifact, root / "best.pt")
            selected_scores = current
            opt = design["training"]["optimizer"]
            optimizer = torch.optim.AdamW(network.parameters(), lr=opt["lr"],
                                          weight_decay=opt["weight_decay"], betas=tuple(opt["betas"]),
                                          eps=opt["eps"])
            if len(optimizer.state):
                raise AssertionError("Adaptation optimizer must start without inherited moments")
            _write(root / "pairing.json", pairing)
            f_num, f_cat = (torch.from_numpy(value).to(actual) for value in arrays["F"])
            y = torch.from_numpy(labels).to(actual)
            window = []
            progress["fit_calls"] = 1
            for step in range(1, validation_steps[-1] + 1):
                draws = sampler.next()
                normal_rows = np.concatenate((draws[:b], draws[2 * b:3 * b]))
                positive_rows = np.concatenate((draws[b:2 * b], draws[3 * b:]))
                batch = BatchTargets(
                    normal_targets=chosen[normal_rows],
                    positive_reference_scores=reference["F"][positive_rows],
                    normal_group_sizes=group_sizes[normal_rows],
                    profile_normal_counts=profile_counts, normal_group_count=len(expected_groups),
                    scale=metadata["scale"], draws_per_group=b)
                indices = torch.from_numpy(draws).to(actual)
                progress["attempted_update"] = step
                update_started = time.perf_counter()
                try:
                    report = training_step(network, optimizer, f_num[indices], f_cat[indices],
                                           y[indices], batch)
                finally:
                    progress["optimizer_steps"] = _optimizer_steps(optimizer)
                _sync(actual)
                update_seconds += time.perf_counter() - update_started
                if progress["optimizer_steps"] != step:
                    raise RuntimeError("Expected exactly one AdamW update per four-group batch")
                window.append(report)
                if step not in validation_steps:
                    continue
                current = _cpu_scores(predictor, network, arrays, actual)
                metrics = selection_metrics(current["F"], frames["F"], current["V"], frames["V"],
                                            reference["V_group_ids"][v_normal], expected_groups)
                decision = selector.update(metrics, step)
                audit = sampler.audit(reset_window=True)
                pairing["draw_chain_by_validation_step"][str(step)] = audit["draw_sha256"]
                pairing["dropout_rng_by_validation_step"][str(step)] = original._state_sha256(
                    original._rng_state(actual))
                pairing["selection_rng_unchanged_by_step"][str(step)] = True
                record = {
                    "step": step, "optimizer_steps": progress["optimizer_steps"],
                    "path_loss_is_endpoint_loss": False,
                    "path_mean": {name: float(np.mean([r[name] for r in window]))
                                  for name in ("member_bce", "normal", "positive", "total")},
                    "gradient": {"preclip_mean_norm": float(np.mean(
                        [r["gradient_norm_before_clip"] for r in window])),
                        "preclip_max_norm": max(r["gradient_norm_before_clip"] for r in window),
                        "clipped_steps": sum(r["gradient_norm_before_clip"] > 1 for r in window),
                        "window_steps": len(window)},
                    "sampling": audit, "metrics": metrics, "selection": decision}
                history.append(record)
                if decision["improved"]:
                    _save_adapted(root / "best.pt", network, template, arm=arm, seed=seed,
                                  step=step, sampler=sampler, actual=actual, original_hash=original_hash)
                    selected_scores = {r: values.copy() for r, values in current.items()}
                    _write(root / "selected_metrics.json", metrics)
                _write(root / "history.json", history)
                _write(root / "pairing.json", pairing)
                _write(root / "sampling.json", sampler.state())
                torch.save({"step": step, "dropout_rng_state": original._rng_state(actual)},
                           root / "rng_state.pt")
                # Never stop for separation, guard failure or an early selected checkpoint.
                window.clear()
            final = selector.finish()
            if final["optimizer_steps"] != validation_steps[-1] or len(history) != len(validation_steps):
                raise RuntimeError("Trajectory did not complete its entire fixed validation schedule")
            shutil.copyfile(root / "best.pt", root / "model.pt")
            np.savez_compressed(root / "selected_scores.npz", **selected_scores,
                                train_scores=selected_scores["F"], validation_scores=selected_scores["V"])
            parameters = sum(p.numel() for p in network.parameters())
            if final["fallback"] and sha256(root / "model.pt") != original_hash:
                raise AssertionError("Step0 fallback must preserve the original artifact bytes")
            if sha256(artifact) != original_hash:
                raise RuntimeError("Original R30 artifact changed during adaptation")
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark = previous[1]
        torch.backends.cudnn.deterministic = previous[2]
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.set_num_threads(previous_threads)
    return {
        "status": "complete", "arm": arm, "protocol_arm": "protocol_blind", "seed": seed,
        "fit_calls": 1, "final_refits": 0, "selected_step": final["selected_step"],
        "candidate_improved": final["candidate_improved"], "fallback": final["fallback"],
        "optimizer_steps": progress["optimizer_steps"], "completed_validation_events": len(history),
        "performance_early_stop": False, "stop_reason": "fixed_budget_complete",
        "artifact_path": str(root / "model.pt"), "best_path": str(root / "best.pt"),
        "selected_scores_path": str(root / "selected_scores.npz"),
        "train_scores_path": str(root / "selected_scores.npz"),
        "score_identities_path": str(root / "score_identities.json"),
        "baseline_metrics_path": str(root / "baseline_metrics.json"),
        "selected_metrics_path": str(root / "selected_metrics.json"),
        "history_path": str(root / "history.json"), "pairing_path": str(root / "pairing.json"),
        "rng_audit_path": str(root / "rng_state.pt"), "parameters": parameters,
        "original_artifact_sha256": original_hash,
        "selected_artifact_sha256": sha256(root / "model.pt"),
        "model_artifact_bytes": (root / "model.pt").stat().st_size,
        "selected_preprocessing_sha256": template["preprocessing_sha256"],
        "selected_numeric_bins_sha256": template["numeric_bins_sha256"],
        "preflight_manifest_sha256": manifest["sha256"],
        "cpu_score_batch_size": predictor.batch_size, "selection_device": "cpu",
        "training_device": str(actual), "fit_seconds": update_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def fit_trajectory(frames, preflight_dir, artifact, output_dir, *, arm, seed,
                   device="cuda", threads=1, design=None):
    """Registered production entry: exactly 2224 updates and sixteen CPU checks.

    Formal registration/approval belongs to the outer runner. This function
    rejects any alteration of the frozen design or historical F/V/model bindings.
    It never reads C/E, refits preprocessing or resumes a nonempty directory.
    """
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Training output directory must be empty; incomplete fits are not retried")
    design = _read(DEFAULT_DESIGN) if design is None else deepcopy(design)
    _validate_production(frames, artifact, arm=arm, seed=seed, threads=threads, design=design)
    seed = int(seed)
    root.mkdir(parents=True, exist_ok=True)
    progress = {"fit_calls": 0, "optimizer_steps": 0, "attempted_update": 0}
    _write(root / "attempt.json", {
        "status": "started", "arm": arm, "seed": seed, "design": design,
        "original_artifact_sha256": sha256(artifact), "final_refits": 0})
    try:
        result = _fit(frames, preflight_dir, artifact, root, arm=arm, seed=seed,
                      device=device, threads=threads, design=design, progress=progress,
                      validation_steps=STEPS)
    except Exception as error:
        result = {"status": "failed", "arm": arm, "seed": seed, **progress, "final_refits": 0,
                  "error_type": type(error).__name__, "error": str(error), "retry_allowed": False}
        _write(root / "failure.json", {**result, "traceback": traceback.format_exc()})
    _write(root / "summary.json", result)
    return result


# Both adapted and byte-identical fallback artifacts use the original R30 format.
load_predictor = original.load_predictor
