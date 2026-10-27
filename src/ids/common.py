"""Portable artifact helpers. Never collect environment secrets or usernames."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     allow_nan=False).encode()).hexdigest()


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_fingerprint() -> dict:
    root = Path(__file__).resolve().parents[2]
    source = {}
    for base in (root / "src", root / "configs", root / "scripts"):
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml", ".json", ".sh", ".ps1"}:
                source[p.relative_to(root).as_posix()] = sha256_file(p)
    for name in ("pyproject.toml", "requirements-lock.txt"):
        if (root / name).is_file():
            source[name] = sha256_file(root / name)
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"commit": commit, "dirty": dirty, "source_sha256": canonical_hash(source), "files": source}


def environment_snapshot(include_torch: bool = True) -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "PyYAML", "joblib",
                 "threadpoolctl", "psutil", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "os": platform.system(),
              "os_release": platform.release(), "machine": platform.machine(),
              "processor": platform.processor(), "cpu_physical": psutil.cpu_count(logical=False),
              "cpu_logical": psutil.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
              "packages": packages, "recorded_at": utc_now()}
    if include_torch and packages["torch"]:
        import torch
        result["torch"] = {"cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                           "cudnn": torch.backends.cudnn.version(), "gpus": []}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                result["torch"]["gpus"].append({"index": i, "name": p.name,
                                               "total_memory_bytes": p.total_memory})
    try:
        value = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                         "--format=csv,noheader"], stderr=subprocess.DEVNULL,
                                        text=True, timeout=10)
        result["nvidia_smi"] = value.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result
