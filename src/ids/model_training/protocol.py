"""Portable code, design, task and laboratory runtime contracts."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from ids.common import canonical_hash, read_json

REPO = Path(__file__).resolve().parents[3]
REGISTRATION = REPO / "configs/model-training-20261116/registration.json"


def specifications():
    registration = read_json(REGISTRATION)
    design = read_json(REPO / registration["design_file"])
    if canonical_hash(design) != registration["design_sha256"]:
        raise ValueError("Reviewed design changed without a new registration")
    return registration, design


def tasks(design):
    result = [{"id": "hgb", "arm": "hgb", "seed": design["training"]["hgb"]["random_state"]}]
    result += [{"id": f"{arm}_s{seed}", "arm": arm, "seed": seed}
               for arm in ("tabm_ple", "tabm_no_ple") for seed in design["training"]["neural_seeds"]]
    if len(result) != 7 or design["training"]["max_fits"] != 14:
        raise ValueError("This implementation requires the reviewed seven-task budget")
    return result


def code_snapshot():
    paths = list((REPO / "src/ids").rglob("*.py"))
    paths += [REGISTRATION, REPO / "configs/model-design-20261116/design.json",
              REPO / "configs/training-diagnosis-20261116/registration.json",
              REPO / "configs/support-diagnostic-20261115/legacy-hgb-calibration.json",
              REPO / "scripts/recover_r26_training_rows.py", REPO / "scripts/RUN_MODEL_TRAINING.ps1",
              REPO / "pyproject.toml"]
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(paths)}


def verify_live_code(protocol):
    if code_snapshot() != protocol["code"]:
        raise ValueError("Live source changed during this registered run")


def verify_runtime(registration, device):
    versions = {name: importlib.metadata.version(name) for name in registration["runtime_packages"]}
    differences = {name: {"expected": version, "actual": versions[name]}
                   for name, version in registration["runtime_packages"].items() if versions[name] != version}
    if differences:
        raise RuntimeError(f"Laboratory runtime differs: {differences}")
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Formal R29 training requires the registered laboratory CUDA device")
    selected = torch.device(device)
    name = torch.cuda.get_device_name(selected)
    if "4090" not in name:
        raise RuntimeError(f"Expected the laboratory RTX 4090D, got {name}")
    return {"packages": versions, "device": device, "device_name": name,
            "cuda": torch.version.cuda, "training": "CUDA TabM; CPU HGB",
            "deployment_scoring": "CPU, one thread, all seven final artifacts frozen first"}
