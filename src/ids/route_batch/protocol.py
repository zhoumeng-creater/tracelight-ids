"""Freeze a finite, complete matrix before any laboratory model execution."""
from __future__ import annotations

import copy
from pathlib import Path
import sys

from ids.common import canonical_hash, read_json, sha256_file, write_json

FORMAT = "tracelight-route-batch-v1"
REPO = Path(__file__).resolve().parents[3]


def default_settings():
    return {"representations": ["protected_v2", "log1p_standard"],
            "widths": [128, 512], "learning_rates": [.002, .0003], "seeds": [17, 29, 43],
            "epochs": 200, "checkpoints": [50, 200], "batch_size": 256,
            "k": 32, "n_blocks": 3, "weight_decay": .0003, "dropout": .1,
            "split_seed": 20261108, "validation_fraction": .25,
            "device": "cuda:0", "threads": 1, "query_batch_size": 128,
            "alpha": .01, "sensitivity_alphas": [.001, .005, .02],
            "benchmark": {"repeats": 3, "single_rows": 64, "warmup": 8},
            "foundation": [{"id": "tabicl_official_1", "kind": "tabicl", "n_estimators": 1},
                           {"id": "tabicl_official_8", "kind": "tabicl", "n_estimators": 8},
                           {"id": "tabpfn25_official_8", "kind": "tabpfn25", "n_estimators": 8}],
            "foundation_seed": 20261107, "reference": ["full", "qcoc"]}


def make_tasks(settings):
    """Every factorial cell is declared in advance; no score-dependent retries."""
    tasks = [{"id": f"reference_{name}", "kind": "reference", "name": name}
             for name in settings.get("reference", [])]
    tasks += [{**item, "kind": "foundation", "model_kind": item["kind"]}
              for item in settings.get("foundation", [])]
    for seed in settings["seeds"]:
        for representation in settings["representations"]:
            for width in settings["widths"]:
                for learning_rate in settings["learning_rates"]:
                    tasks.append({"id": f"tabm_{representation}_w{width}_lr{learning_rate:g}_s{seed}",
                                  "kind": "tabm", "representation": representation, "width": width,
                                  "learning_rate": learning_rate, "seed": seed})
    ids = [item["id"] for item in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task identifiers")
    return tasks


def source_snapshot():
    files = sorted((REPO / "src" / "ids").rglob("*.py"))
    import hashlib
    hashes = {p.relative_to(REPO).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in files}
    return {"hash_encoding": "UTF8 source bytes with CRLF normalized to LF", "files": hashes,
            "source_sha256": canonical_hash(hashes)}


def register(round14_root, checkpoint, output, *, tabpfn_checkpoint=None, tabpfn_python=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base = read_json(REPO / "configs" / "context-scanning-label-v1.json")
    spec = {key: copy.deepcopy(base[key]) for key in
            ("schema", "files", "artifacts", "checkpoint", "baseline_model_sha256")}
    root = Path(round14_root).resolve()
    for item in (*spec["files"].values(), *spec["artifacts"].values()):
        relative = item["path"].split("ROUND14/", 1)[1]
        item["path"] = str(root / relative)
    spec["checkpoint"]["path"] = str(Path(checkpoint).resolve())
    # Do not silently register a different data cohort or checkpoint.
    for item in (*spec["files"].values(), *spec["artifacts"].values(), spec["checkpoint"]):
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"frozen input mismatch: {item['path']}")
    spec.update({"protocol": FORMAT, "data_kind": "development", "round14_root": str(root),
                 "settings": default_settings(), "wall_clock_limit": None,
                 "selection": "within each T trajectory: minimum V ensemble BCE, earliest tie; refit all P",
                 "scope": "route comparison on already-used development E; not independent confirmation"})
    for name in ("qcoc_calibration", "comparison_report"):
        path = root / "comparison" / ({"qcoc_calibration": "qcoc_calibration.csv",
                                         "comparison_report": "comparison.json"}[name])
        spec["artifacts"][name] = {"path": str(path), "sha256": sha256_file(path)}
    spec["tabpfn_checkpoint"] = {
        "path": str(Path(tabpfn_checkpoint).resolve()) if tabpfn_checkpoint else str(
            Path(checkpoint).resolve().parent / "tabpfn-v2.5-classifier-v2.5_default.ckpt"),
        "sha256": "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"}
    spec["tabpfn_python"] = str(Path(tabpfn_python).resolve()) if tabpfn_python else str(
        Path(sys.executable).resolve().parents[2] / "tracelight-route-tabpfn" / "Scripts" / "python.exe")
    spec["tasks"] = make_tasks(spec["settings"])
    write_json(output, spec)
    return {"config": str(output), "tasks": len(spec["tasks"]),
            "tabm_trajectories": 24, "tabm_full_support_refits": 24,
            "wall_clock_limit": None}


def validate(spec):
    if spec.get("protocol") != FORMAT or spec.get("data_kind") not in {"development", "synthetic"}:
        raise ValueError("unsupported route-batch protocol")
    settings = spec["settings"]
    if spec["tasks"] != make_tasks(settings):
        raise ValueError("frozen task list differs from the declared factorial settings")
    if settings["threads"] != 1 or settings["alpha"] != .01:
        raise ValueError("CPU single-thread and 1% primary calibration are fixed")
    if spec["data_kind"] == "development" and settings != default_settings():
        raise ValueError("development matrix differs from the frozen complete batch")
    if spec["data_kind"] == "development" and spec.get("wall_clock_limit") is not None:
        raise ValueError("user authorized the complete batch without a wall-clock cutoff")
    return spec
