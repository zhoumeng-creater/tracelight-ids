"""Frozen six-trajectory registration and zero-update readiness checks."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np

from ids.common import read_json
from ids.model_training.runner import contained
from .bindings import REPO, SEEDS, sha256
from .groups import fit_groups, map_groups
from .selection import Selector, metrics_to_json, selection_metrics


VERSION = "normal-subgroup-r31-v1"
REGISTRATION = REPO / "configs/normal-subgroup-lab-20261118/registration.json"
ARMS = ("transport_target", "identity_target")


def scoring_host():
    """Bind CPU anchors and live scoring to one host, not merely package names."""
    return {"hostname": platform.node(), "machine": platform.machine(),
            "processor": platform.processor(), "platform": platform.platform(),
            "python": platform.python_version()}


def tasks(design):
    if (design["seeds"] != list(SEEDS) or design["arms"] != list(ARMS)
            or design["training"]["planned_future_fit_calls"] != 6
            or design["training"]["optimizer_steps_per_trajectory"] != 2224
            or design["training"]["final_refits"] != 0):
        raise ValueError("Only the frozen six trajectories, 2224 steps each and zero refits are supported")
    return [{"id": f"{arm}_s{seed}", "arm": arm, "seed": seed}
            for seed in SEEDS for arm in ARMS]


def source_paths():
    paths = list((REPO / "src/ids").rglob("*.py"))
    paths += [REPO / rel for rel in (
        "configs/normal-subgroup-refinement-20261118/design.json",
        "configs/normal-subgroup-training-20261118/input-bindings.json",
        "configs/normal-subgroup-lab-20261118/input-bindings.json",
        "scripts/RUN_NORMAL_SUBGROUP.ps1", "pyproject.toml")]
    return sorted(paths)


def code_snapshot():
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in source_paths()}


def specifications():
    registration = read_json(REGISTRATION)
    if registration["version"] != VERSION or not registration["execution_enabled"]:
        raise ValueError("This entry requires its own enabled registration")
    for field in ("design", "input_bindings", "source_manifest"):
        if sha256(REPO / registration[field + "_file"]) != registration[field + "_sha256"]:
            raise ValueError(f"Frozen {field} differs from registration")
    manifest = read_json(REPO / registration["source_manifest_file"])
    if code_snapshot() != manifest["files"]:
        raise ValueError("Source differs from the laboratory release")
    design = read_json(REPO / registration["design_file"])
    tasks(design)
    return registration, design


def verify_live_code(protocol):
    if protocol.get("scoring_host") != scoring_host():
        raise ValueError("CPU scoring host differs; generate a new preflight on this host")
    registration, design = specifications()
    if (registration != protocol["registration"] or design != protocol["design"]
            or code_snapshot() != protocol["code"]):
        raise ValueError("Code or registration changed during this run")


def verify_runtime(registration, device=None):
    actual = {name: version(name) for name in registration["runtime_packages"]}
    if actual != registration["runtime_packages"]:
        raise RuntimeError("Registered package versions differ; do not upgrade during a run")
    result = {"packages": actual}
    if device is not None:
        import torch
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Formal trajectories require the registered laboratory CUDA device")
        name = torch.cuda.get_device_name(torch.device(device))
        if name != registration["training_device_name"]:
            raise RuntimeError(f"Expected {registration['training_device_name']}; got {name}")
        result.update(device=device, device_name=name, cuda=torch.version.cuda)
    return result


def verify_preflight(root, frames, bindings, design):
    """Verify saved payload bytes and all baseline guards before any update."""
    root = Path(root)
    status = read_json(root / "status.json")
    if (status["status"] != "zero_update_preflight_complete" or status["new_fits"] != 0
            or status["optimizer_updates"] != 0 or status["C_E_access"]):
        raise ValueError("A complete zero-update F/V preflight is required")
    binding = status["output_manifest"]
    manifest_path = contained(root, binding["path"])
    if sha256(manifest_path) != binding["sha256"] or manifest_path.stat().st_size != binding["bytes"]:
        raise ValueError("Preflight manifest changed")
    manifest = read_json(manifest_path)
    for rel, item in manifest["files"].items():
        path = contained(root, rel)
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Preflight payload changed: {rel}")
    required = {"groups.json", "identities.json", "input_receipt.json"}
    required |= {f"protocol_blind_s{seed}/{name}" for seed in SEEDS for name in
                 ("targets.npz", "reference_scores.npz", "target_metadata.json", "preprocessing.json")}
    if set(manifest["files"]) != required:
        raise ValueError("Preflight payload set differs")
    identities = read_json(root / "identities.json")
    for role in ("F", "V"):
        if identities[role] != frames[role][["source_id", "row_id"]].values.tolist():
            raise ValueError("Preflight row identities differ")
    groups = read_json(root / "groups.json")
    if groups != fit_groups(frames["F"], role="F"):
        raise ValueError("Frozen groups differ from the F-only rule")
    v_groups = map_groups(frames["V"], groups, role="V", require_all=True)
    results = {}
    for seed in SEEDS:
        key = f"protocol_blind_s{seed}"
        if status["models"][key]["artifact_sha256"] != bindings["models"][key]["sha256"]:
            raise ValueError("Preflight used a different original model")
        metadata = read_json(root / key / "target_metadata.json")
        if metadata["equivalence"]["reference_inconsistent_classes"] != 0:
            raise ValueError("Identical model inputs have conflicting formal anchors")
        with np.load(root / key / "reference_scores.npz", allow_pickle=False) as scores:
            if not np.array_equal(scores["V_group_ids"], v_groups):
                raise ValueError("Frozen V group mapping differs")
            metrics = selection_metrics(scores["F"], frames["F"], scores["V"], frames["V"],
                v_groups[frames["V"].label.to_numpy() == 0], groups["group_keys"])
        Selector(metrics)  # Includes full F separation and all V/source/tie guards.
        results[str(seed)] = metrics_to_json(metrics)
    return {"status": "all_three_baselines_pass", "output_manifest_sha256": binding["sha256"],
            "score_device": "cpu", "score_threads": 1, "score_batch_size": 256,
            "real_fit_calls": 0, "baseline_metrics": results,
            "design_version": design["version"]}
