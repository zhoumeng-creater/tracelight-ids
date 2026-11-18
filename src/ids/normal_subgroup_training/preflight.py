"""CPU cold replay and frozen targets, with no fit, optimizer, C, or E access."""
from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .bindings import DEFAULT_BINDINGS, SEEDS, load_bound_inputs, sha256
from .groups import fit_groups, map_groups
from .targets import build_targets


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def _load(artifact):
    from ids.model_training_redesign.training import load_predictor
    return load_predictor(artifact, threads=1, device="cpu")


def _runtime():
    expected = {"numpy": "1.26.4", "torch": "2.6.0+cu126", "tabm": "0.0.3",
                "rtdl_num_embeddings": "0.0.12"}
    actual = {package: version(package) for package in expected}
    if actual != expected:
        raise ValueError("Zero-update replay requires the registered R30 runtime versions")
    return actual


def _freeze_outputs(root):
    """Hash all payloads; final status binds this manifest without a hash cycle."""
    excluded = ("output_manifest.json", "status.json")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            files[path.relative_to(root).as_posix()] = {
                "sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = root / "output_manifest.json"
    _write(manifest, {
        "version": "normal-subgroup-preflight-output-v1",
        "excluded": list(excluded),
        "status_binding": "Final status.json records this manifest's hash and size; neither hashes itself",
        "files": files,
    })
    return {"path": manifest.name, "sha256": sha256(manifest), "bytes": manifest.stat().st_size}


def cold_replay(artifact, frames, *, loader=None):
    """Load twice from the same bytes and require exact F/V endpoint replay."""
    if set(frames) != {"F", "V"}:
        raise ValueError("Cold replay accepts F and V only")
    loader = _load if loader is None else loader
    before = sha256(artifact)
    first = loader(artifact)
    reference = {role: np.asarray(first.score(frame), dtype=np.float64)
                 for role, frame in frames.items()}
    numeric, categorical = first.preprocessor.transform(frames["F"])
    preprocessing = first.preprocessor.to_dict()
    if preprocessing.get("protocol_policy") != "constant_unknown":
        raise ValueError("Expected original protocol-blind preprocessing")
    del first
    second = loader(artifact)
    if second.preprocessor.to_dict() != preprocessing:
        raise ValueError("Cold reload changed preprocessing")
    for role, frame in frames.items():
        scores = np.asarray(second.score(frame), dtype=np.float64)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise ValueError(f"Invalid {role} endpoint")
        if scores.tobytes() != reference[role].tobytes():
            raise ValueError(f"Zero-update cold replay differs for {role}")
    del second
    if sha256(artifact) != before:
        raise ValueError("Baseline artifact changed during preflight")
    return reference, numeric, categorical, preprocessing


def run(archive, output_dir, *, bindings_path=DEFAULT_BINDINGS):
    """Explicit zero-update entry, never a training registration or fit runner."""
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Preflight output directory must be empty")
    runtime = _runtime()
    frames, bindings, checked = load_bound_inputs(archive, bindings_path=bindings_path)
    groups = fit_groups(frames["F"], role="F")
    assignments = {role: map_groups(frame, groups, role=role, require_all=True)
                   for role, frame in frames.items()}
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "groups.json", groups)
    _write(root / "input_receipt.json", checked)
    _write(root / "identities.json", {role: frame[["source_id", "row_id"]].values.tolist()
                                       for role, frame in frames.items()})
    receipt = {"status": "started", "new_fits": 0, "optimizer_updates": 0,
               "C_E_access": False, "training_ready": False, "runtime": runtime, "models": {}}
    _write(root / "status.json", receipt)
    try:
        for seed in SEEDS:
            key = f"protocol_blind_s{seed}"
            artifact = Path(archive) / bindings["models"][key]["path"]
            scores, numeric, categorical, preprocessing = cold_replay(artifact, frames)
            targets = build_targets(scores["F"], frames["F"].label.to_numpy(), assignments["F"],
                                    numeric, categorical, role="F")
            consistency = targets["equivalence"]
            if consistency["reference_inconsistent_classes"]:
                raise ValueError(
                    "Formal CPU anchors have inconsistent reference scores for identical model inputs: "
                    f"classes={consistency['reference_inconsistent_classes']}, "
                    f"max_difference={consistency['max_reference_difference']}")
            dest = root / key
            dest.mkdir()
            arrays = {name: value for name, value in targets.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(dest / "targets.npz", **arrays, group_ids=assignments["F"])
            np.savez_compressed(dest / "reference_scores.npz", F=scores["F"], V=scores["V"],
                                V_group_ids=assignments["V"])
            _write(dest / "target_metadata.json", {name: value for name, value in targets.items()
                                                    if not isinstance(value, np.ndarray)})
            _write(dest / "preprocessing.json", preprocessing)
            receipt["models"][key] = {"artifact_sha256": sha256(artifact), "cold_loads": 2,
                                      "exact_F_V_replay": True, "optimizer_updates": 0}
            _write(root / "status.json", receipt)
        receipt["output_manifest"] = _freeze_outputs(root)
        receipt["status"] = "zero_update_preflight_complete"
        receipt["training_ready"] = False
        receipt["next"] = "review_target_effect_and_finish_training_implementation_before_registration"
        _write(root / "status.json", receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
        _write(root / "status.json", receipt)
        raise
