"""The reviewed design, six tasks, portable source snapshot and lab runtime."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

from ids.common import canonical_hash, read_json
from . import VERSION

REPO = Path(__file__).resolve().parents[3]
REGISTRATION = REPO / "configs/model-training-redesign-20261117/registration.json"


def specifications():
    registration = read_json(REGISTRATION)
    design = read_json(REPO / registration["design_file"])
    if registration["version"] != VERSION or not registration["execution_enabled"]:
        raise ValueError("This entry requires its own enabled laboratory registration")
    if canonical_hash(design) != registration["design_sha256"]:
        raise ValueError("Reviewed design differs from the frozen registration")
    bindings = read_json(REPO / registration["input_bindings_file"])
    if canonical_hash(bindings) != registration["input_bindings_sha256"]:
        raise ValueError("Input artifact bindings differ from the registration")
    for name, value in bindings.items():
        if name in registration and registration[name] != value:
            raise ValueError(f"Conflicting input binding: {name}")
        registration[name] = value
    tasks(design)
    return registration, design


def tasks(design):
    if ([x["id"] for x in design["arms"]] != ["protocol_blind", "protocol_aware"]
            or design["training"]["seeds"] != [17, 29, 43]
            or design["training"]["max_fits"] != 6
            or design["training"]["final_refits"] != 0):
        raise ValueError("Only the registered three paired seeds and zero refits are supported")
    return [{"id": f"{arm}_s{seed}", "arm": arm, "seed": seed}
            for seed in (17, 29, 43) for arm in ("protocol_blind", "protocol_aware")]


def code_snapshot():
    paths = list((REPO / "src/ids").rglob("*.py"))
    paths += list((REPO / "configs/model-training-redesign-20261117").glob("*.json"))
    paths += [REPO / "configs/response-evidence-20261112/tqh.json",
              REPO / "scripts/RUN_MODEL_TRAINING_REDESIGN.ps1", REPO / "pyproject.toml"]
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in sorted(paths)}


def verify_live_code(protocol):
    if code_snapshot() != protocol["code"]:
        raise ValueError("Registered source changed during this run")


def verify_runtime(registration, device):
    versions = {name: importlib.metadata.version(name) for name in registration["runtime_packages"]}
    differences = {name: {"expected": expected, "actual": versions[name]}
                   for name, expected in registration["runtime_packages"].items()
                   if versions[name] != expected}
    if differences:
        raise RuntimeError(f"Laboratory package versions differ: {differences}")
    import torch
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Formal training requires the registered laboratory CUDA device")
    selected = torch.device(device)
    name = torch.cuda.get_device_name(selected)
    if "4090" not in name:
        raise RuntimeError(f"Expected the laboratory RTX 4090D; got {name}")
    return {"packages": versions, "device": device, "device_name": name,
            "cuda": torch.version.cuda, "training": "six serial CUDA trajectories; zero refits",
            "deployment_scoring": "CPU one thread, all six selected artifacts frozen first"}
