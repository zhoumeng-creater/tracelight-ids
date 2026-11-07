"""Raw CSV integration fixture: no pretrained weights or external datasets."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from ids.common import read_json, sha256_file, write_json
from .csv_workflow import PROTOCOL, run_comparison
from .upstream import require_pinned_tabicl


def synthetic_csv_check(output):
    import torch

    require_pinned_tabicl()
    from tabicl._model.tabicl import TabICL

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    schema = {"features": ["measure", "flow_bytes", "protocol", "support_constant"],
              "categorical": ["protocol"], "label_column": "label", "id_column": "sample_id",
              "group_column": "session", "family_column": "family",
              "label_mapping": {"BENIGN": 0, "synthetic_attack": 1}}
    rng = np.random.default_rng(20261107)
    files = {}
    for offset, (role, n) in enumerate((("support", 32), ("calibration", 199), ("evaluation", 80))):
        truth = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        categories = ["tcp", "udp" if role == "support" else "unseen", "NA", None]
        frame = pd.DataFrame({
            "sample_id": [f"{role}-{i:05d}" for i in range(n)],
            "session": [f"{role}-session-{i // 4}" for i in range(n)],
            "measure": rng.normal(size=n) + offset,
            "flow_bytes": rng.lognormal(size=n) + np.arange(n) + offset * 1000,
            "protocol": [categories[i % 4] for i in range(n)],
            "support_constant": np.ones(n) * 3 if role == "support" else np.arange(n),
            "label": ["BENIGN" if y == 0 else "synthetic_attack" for y in truth],
            "family": ["normal" if y == 0 else f"synthetic_attack_{i % 3}" for i, y in enumerate(truth)],
        })
        frame.loc[::11, "measure"] = np.nan
        path = inputs / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": path.name, "sha256": sha256_file(path), "rows": n}
    torch.manual_seed(20261107)
    config = {"max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
              "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
              "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.0, "zero_init": False}
    model = TabICL(**config).eval()
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "query_mlp"):
                module.query_mlp[-1].weight.normal_(0, 0.2)
                module.query_mlp[-1].bias.normal_(0, 0.1)
    checkpoint = inputs / "synthetic-random-not-pretrained.ckpt"
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)
    del model
    spec = {"protocol": PROTOCOL, "data_kind": "synthetic", "schema": schema, "files": files,
            "checkpoint": {"path": checkpoint.name, "sha256": sha256_file(checkpoint)},
            "slots": 4, "alpha": 0.01, "build_device": "cpu", "query_batch_size": 37,
            "query_chunk_size": 11, "threads": 1, "seed": 20261107}
    protocol_path = inputs / "protocol.json"
    write_json(protocol_path, spec)
    report = run_comparison(protocol_path, output / "run")
    paired = pd.read_csv(output / "run" / "paired_evaluation.csv", float_precision="round_trip")
    # Retain the original feature text while stripping every metadata column.
    # This exercises a genuine feature-only CSV, not the prepared tensor path.
    raw_query = output / "feature_only_queries.csv"
    with (inputs / "evaluation.csv").open(encoding="utf-8", newline="") as source:
        with raw_query.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=schema["features"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv.DictReader(source))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in sys.path if p and Path(p).is_dir())
    cold_checks = {}
    for name in ("full", "qcoc"):
        cold = output / "cold" / name
        shutil.copytree(output / "run" / name, cold)
        predictions = cold.parent / f"{name}_predictions.csv"
        command = [sys.executable, "-m", "ids.context_cache", "predict-csv", "--deployment", str(cold.resolve()),
                   "--input", str(raw_query.resolve()), "--output", str(predictions.resolve())]
        child = subprocess.run(command, cwd=cold, env=env, capture_output=True, text=True, timeout=60)
        if child.returncode:
            raise RuntimeError(f"Cold CSV prediction failed: {child.stderr[-3000:]}")
        actual = pd.read_csv(predictions, float_precision="round_trip")
        np.testing.assert_array_equal(actual["score"].to_numpy(), paired[f"{name}_score"].to_numpy())
        np.testing.assert_array_equal(actual["predicted_attack"].to_numpy(),
                                      paired[f"{name}_predicted"].to_numpy())
        cold_checks[name] = {"fresh_process_raw_csv_scores_equal": True,
                             "fresh_process_registered_decisions_equal": True,
                             "files_required": sorted(p.name for p in cold.iterdir())}
    summary = {"stage": "raw_csv_synthetic_integration", "data_kind": "synthetic",
               "weights": "random_tiny_official_architecture_not_pretrained",
               "support_rows": 32, "normal_calibration_rows": 199, "evaluation_rows": 80,
               "slots": 4, "alpha": .01, "candidate_count": 2, "cold_checks": cold_checks,
               "raw_text_label_mapping": schema["label_mapping"],
               "removed_support_constant": "support_constant" in report["preprocessing"]["removed_constant_features"],
               "different_model_bindings": (report["registrations"]["full"]["binding"]["model_sha256"]
                                            != report["registrations"]["qcoc"]["binding"]["model_sha256"]),
               "each_threshold_uses_own_scores": {name: read_json(output / "run" / name / "threshold.json")
                                                   ["normal_scores_sha256"] for name in ("full", "qcoc")},
               "nontrivial_thresholds": report["nontrivial_thresholds"],
               "source_snapshot": report["source_snapshot"],
               "scope": "CSV, calibration, package and paired-report engineering verification; no IDS result"}
    assert summary["removed_support_constant"] and summary["different_model_bindings"]
    assert all(summary["nontrivial_thresholds"].values())
    write_json(output / "synthetic_csv_report.json", summary)
    return {"output": str(output.resolve()), "stage": summary["stage"], "cold_checks": cold_checks}
