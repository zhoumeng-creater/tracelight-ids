"""Actual subprocess follow-up with random tiny weights and synthetic C/E only."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.route_batch import foundation as f
from ids.route_batch.data import load_inputs
from ids.route_batch.preprocessing import raw_numeric
from ids.route_batch.protocol import REPO, default_settings, make_tasks, source_snapshot
from ids.route_batch.runner import _write_scores


def test_followup_cli_random_tiny_official_models_preserves_old_run(tmp_path, monkeypatch):
    import torch
    vendor = REPO / "local/vendor/tabicl/src"
    if vendor.is_dir():
        monkeypatch.syspath_prepend(str(vendor))
    try:
        f._tabicl_class()
    except ImportError:
        pytest.skip("pinned optional TabICL source absent; no download")
    from tabicl._model.tabicl import TabICL
    config = {
        "max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.,
        "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
        "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise",
        "icl_ssmax": "qassmax-mlp-elementwise",
    }
    checkpoint = tmp_path / "random-tiny-NOT-PRETRAINED.ckpt"
    with torch.random.fork_rng():
        torch.manual_seed(919)
        network = TabICL(**config).eval()
    torch.save({"config": config, "state_dict": network.state_dict()}, checkpoint)
    root = tmp_path / "old run"
    root.mkdir()
    features = [f"x{i}" for i in range(5)]
    schema = {"features": features, "numeric": features, "categorical": [], "label_column": "label",
              "id_column": "sample_id", "group_column": "communication_group", "family_column": "attack_family"}
    rng, files = np.random.default_rng(919), {}
    for role, n in (("support", 16), ("calibration", 100), ("evaluation", 130)):
        truth = np.zeros(n, dtype=int) if role == "calibration" else np.arange(n) % 2
        frame = pd.DataFrame(rng.normal(size=(n, 5)), columns=features)
        frame["label"] = truth
        frame["sample_id"] = [f"{role}-{i}" for i in range(n)]
        frame["communication_group"] = frame["sample_id"]
        frame["attack_family"] = np.where(truth, "Scanning", "Benign")
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "rows": n, "sha256": sha256_file(path)}
    settings = default_settings()
    settings.update(device="cpu", representations=[], reference=[],
                    benchmark={"repeats": 1, "single_rows": 1, "warmup": 0})
    settings["foundation"] = settings["foundation"][:2]
    spec = {"protocol": "tracelight-route-batch-v1", "data_kind": "synthetic", "settings": settings,
            "schema": schema, "files": files, "artifacts": {}, "tasks": make_tasks(settings),
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}}
    write_json(root / "protocol.json", spec)
    snapshot = source_snapshot()
    write_json(root / "source_snapshot.json", snapshot)
    write_json(root / "preflight.json", {"source": snapshot, "protocol_sha256": canonical_hash(spec)})
    inputs = load_inputs(root / "protocol.json")
    for task in spec["tasks"]:
        attempt = root / "tasks" / task["id"] / "attempt-001"
        attempt.mkdir(parents=True)
        model = f.fit_foundation("tabicl", raw_numeric(inputs["frames"]["support"], inputs["schema"]),
            inputs["frames"]["support"].label.to_numpy(), checkpoint_path=checkpoint,
            seed=settings["foundation_seed"], device="cpu", n_estimators=task["n_estimators"])
        model.save(attempt / "model.joblib")
        descriptor = {"kind": "foundation", "model_file": "model.joblib", "metadata": model.metadata,
                      "model_sha256": sha256_file(attempt / "model.joblib"), "score_contract": model.metadata["score"]}
        scores = {role: model.predict_scores(raw_numeric(inputs["frames"][role], inputs["schema"]))
                  for role in ("calibration", "evaluation")}
        metrics = _write_scores(attempt, inputs, scores, descriptor)
        write_json(attempt / "result.json", {"status": "measurement_failed", "task": task,
                                            "model": descriptor, "metrics": metrics})
        del model
    write_json(root / "summary.json", {"original_summary": "must not change"})
    write_json(root / "tasks/tabm_protected/result.json", {"status": "complete", "do_not_rerun": True})
    write_json(root / "confirmation/frozen.json", {"opened": True, "do_not_rerun": True})
    before = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
    target = tmp_path / "followup run"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(vendor)]),
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    common = ["--original-run", str(root), "--output", str(target),
              "--tabpfn-python", str(tmp_path / "absent-python.exe"),
              "--tabpfn-checkpoint", str(tmp_path / f.TABPFN_CHECKPOINT)]
    def cli(arguments):
        result = subprocess.run([sys.executable, "-m", "ids.route_batch", *arguments],
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    cli(["followup-preflight", *common, "--report", str(tmp_path / "preflight.json")])
    assert not target.exists()
    cli(["followup", *common])
    summary = read_json(target / "summary.json")
    statuses = {row["task"]: row["status"] for row in summary["tasks"]}
    assert statuses == {"tabicl_official_1": "complete", "tabicl_official_8": "complete",
                        "tabpfn25_official_8": "unavailable"}, summary
    assert summary["original_preservation"]["unchanged"]
    assert not summary["all_tasks_complete"]
    for task in ("tabicl_official_1", "tabicl_official_8"):
        folder = target / "tasks" / task / "attempt-001/diagnostic/cpu"
        assert read_json(folder / "cold_replay_report.json")["status"] == "matched"
        cost = read_json(folder / "cost.json")
        assert cost["worker"]["loaded_data_roles"] == ["evaluation"]
        assert cost["replay"]["mismatched_rows"] == 0
    archive = tmp_path / "followup.zip"
    cli(["followup-pack", "--run-dir", str(target), "--output", str(archive)])
    assert archive.is_file()
    assert before == {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}
