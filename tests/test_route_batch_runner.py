import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, sha256_file, write_json
from ids.route_batch.protocol import FORMAT, default_settings, make_tasks
from ids.route_batch import runner


def synthetic_config(tmp_path):
    rng = np.random.default_rng(92)
    features = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "bytes", "duration"]
    schema = {"features": features, "numeric": features[4:], "categorical": features[:4],
              "id_column": "sample_id", "group_column": "communication_group",
              "family_column": "attack_family", "label_column": "label"}
    files = {}
    for role, n in (("support", 128), ("calibration", 100), ("evaluation", 128)):
        x = rng.normal(size=n)
        y = np.zeros(n, dtype=int) if role == "calibration" else (x > 0).astype(int)
        frame = pd.DataFrame({"L4_SRC_PORT": rng.integers(1, 65535, n), "L4_DST_PORT": 443,
            "PROTOCOL": 6, "L7_PROTO": 91, "bytes": np.exp(x), "duration": rng.normal(size=n),
            "label": y, "attack_family": np.where(y, "Scanning", "Benign"),
            "sample_id": [f"{role}-{i}" for i in range(n)],
            "communication_group": [f"{role}-group-{i}" for i in range(n)]})
        path = tmp_path / f"{role}.csv"
        frame.to_csv(path, index=False)
        files[role] = {"path": str(path), "sha256": sha256_file(path), "rows": n}
    settings = default_settings()
    settings.update(representations=["log1p_standard"], widths=[8], learning_rates=[.002], seeds=[17],
                    epochs=2, checkpoints=[1, 2], k=2, n_blocks=1, batch_size=16, device="cpu",
                    foundation=[], reference=[], benchmark={"repeats": 1, "single_rows": 4, "warmup": 1})
    spec = {"protocol": FORMAT, "data_kind": "synthetic", "schema": schema, "files": files,
            "artifacts": {}, "settings": settings, "tasks": make_tasks(settings),
            "scope": "synthetic only", "wall_clock_limit": None}
    path = tmp_path / "config.json"
    write_json(path, spec)
    return path, spec


def test_matrix_has_all_factorial_cells_and_distinct_axes():
    tasks = make_tasks(default_settings())
    assert len(tasks) == 29
    assert sum(t["kind"] == "tabm" for t in tasks) == 24
    assert len({t["id"] for t in tasks}) == 29


def test_synthetic_worker_full_fit_scores_and_pack(tmp_path):
    pytest.importorskip("tabm")
    config, spec = synthetic_config(tmp_path)
    run_root = tmp_path / "run"
    audit = runner.preflight(config)
    assert audit["real_model_loads"] == 0
    result = runner.run(config, run_root)
    assert result["all_planned_tasks_complete"], result
    task_dir = run_root / "tasks" / spec["tasks"][0]["id"] / "attempt-001"
    record = read_json(task_dir / "result.json")
    assert record["selection"]["epoch"] in (1, 2)
    assert set(record["endpoint_metrics"]) == {"best", "1", "2"}
    assert record["cost"]["csv_to_scores"]["rows"] == 128
    assert record["metrics"]["normal_rows"] + record["metrics"]["attack_rows"] == 128
    assert Path(task_dir / record["model"]["model_file"]).is_file()
    resumed = runner.run(config, run_root, resume=True)
    assert resumed["all_planned_tasks_complete"]
    assert len([p for p in task_dir.parent.glob("attempt-*") if p.is_dir()]) == 1
    archive = runner.pack(run_root, tmp_path / "analysis.zip")
    assert not archive["models_included"] and archive["files"] > 10


def test_resume_rejects_changed_protocol(tmp_path):
    config, spec = synthetic_config(tmp_path)
    root = tmp_path / "run"
    root.mkdir()
    write_json(root / "protocol.json", spec)
    write_json(root / "source_snapshot.json", runner.source_snapshot())
    changed = copy.deepcopy(spec)
    changed["scope"] = "different experiment"
    config.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identical protocol"):
        runner.run(config, root, resume=True)


def test_random_official_foundation_in_isolated_score_and_cost_workers(tmp_path, monkeypatch):
    import torch
    checkout = Path(__file__).resolve().parents[1] / "local" / "vendor" / "tabicl" / "src"
    if not checkout.is_dir():
        pytest.skip("pinned TabICL source absent")
    import os
    monkeypatch.setenv("PYTHONPATH", str(checkout) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.syspath_prepend(str(checkout))
    from tabicl._model.tabicl import TabICL
    configuration = {"max_classes": 10, "embed_dim": 8, "col_num_blocks": 1, "col_nhead": 2,
        "col_num_inds": 4, "row_num_blocks": 1, "row_nhead": 2, "row_num_cls": 2,
        "icl_num_blocks": 2, "icl_nhead": 2, "ff_factor": 2, "dropout": 0.,
        "zero_init": False, "col_feature_group": "same", "col_feature_group_size": 3,
        "col_target_aware": True, "col_ssmax": "qassmax-mlp-elementwise", "icl_ssmax": "qassmax-mlp-elementwise"}
    checkpoint = tmp_path / "random-tiny-not-pretrained.ckpt"
    with torch.random.fork_rng():
        torch.manual_seed(5)
        network = TabICL(**configuration)
    torch.save({"config": configuration, "state_dict": network.state_dict()}, checkpoint)
    config, spec = synthetic_config(tmp_path)
    spec["checkpoint"] = {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}
    spec["settings"]["seeds"] = []
    spec["settings"]["foundation"] = [{"id": "tabicl_official_1", "kind": "tabicl", "n_estimators": 1}]
    spec["tasks"] = make_tasks(spec["settings"])
    write_json(config, spec)
    result = runner.run(config, tmp_path / "foundation-run")
    assert result["all_planned_tasks_complete"], result
    state = runner._task_states(tmp_path / "foundation-run", spec["tasks"])[0]["result"]
    assert state["cost"]["worker"]["gradient_training_performed"] is False
    assert state["cost"]["required_deployment_file_bytes"] > checkpoint.stat().st_size
