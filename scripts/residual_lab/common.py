"""Shared manifest and wheel checks for the Windows laboratory bundle."""
from __future__ import annotations

import csv
from hashlib import file_digest
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WHEEL_LOCK = HERE / "wheels-win-cp312.tsv"
BUNDLE_MANIFEST = "lab-bundle.json"
BUNDLE_SCHEMA = "tracelight-residual-lab-windows-v1"


def fingerprint(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": file_digest(stream, "sha256").hexdigest()}


def verify_file(path, expected):
    if not path.is_file() or fingerprint(path) != {
        "bytes": int(expected["bytes"]), "sha256": expected["sha256"],
    }:
        raise ValueError(f"Missing or mismatched file: {path}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wheel_entries():
    with WHEEL_LOCK.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def checked_member(root, name):
    relative = Path(name)
    if relative.is_absolute() or relative.drive or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid manifest path: {name}")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes the bundle: {name}")
    return path
