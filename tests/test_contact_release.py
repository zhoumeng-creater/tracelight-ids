from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.contact_evidence import data, experiment
from ids.contact_evidence.runner import gpu_probe


def test_release_contains_real_registered_inputs_and_reference():
    reg = data.read_registration()
    assert [s["name"].rsplit("-", 2)[-2] for s in reg["sources"] if s["role"] == "E"] == ["52", "60"]
    reference = read_json(data.REGISTRY.with_name("r21-reference.json"))
    assert reference["calibration_rows"] == 1826
    normals = pd.read_parquet(data.REGISTRY.with_name("r21-normal-controls.parquet"))
    assert len(normals) == 1826 and not normals[["capture", "row_id"]].duplicated().any()
    assert np.isfinite(normals[["raw", "history"]]).all().all()


def test_actual_cuda_entry_and_portable_model_on_synthetic_data(tmp_path, monkeypatch):
    import torch
    from ids.flow_history import model

    if not torch.cuda.is_available():
        pytest.skip("No CUDA device for synthetic fit")
    evidence = read_json(gpu_probe(tmp_path)[0])
    assert evidence["device"] == "cuda:0" and evidence["rows"] == 4096
    monkeypatch.setattr(model, "EPOCHS", 1)
    cols = experiment.ARMS["contact"]
    rng = np.random.default_rng(27)
    frame = pd.DataFrame(rng.uniform(-1, 30, (256, len(cols))), columns=cols)
    matrix = tmp_path / "features/train.parquet"
    matrix.parent.mkdir()
    frame.assign(label=np.arange(len(frame)) % 2).to_parquet(matrix, index=False)
    path, build = experiment.fit(tmp_path, "mlp29", "contact", 1)
    assert Path(path).is_file() and read_json(build)["seed"] == 29
    gpu = model.load_arm(path, device="cuda:0", threads=1)
    cpu = model.load_arm(path, device="cpu", threads=1)
    np.testing.assert_allclose(
        model.score_arm(gpu, frame, cols), model.score_arm(cpu, frame, cols), atol=3e-5, rtol=3e-5
    )
