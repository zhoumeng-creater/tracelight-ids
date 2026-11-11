"""Regression checks for the paired lightweight model and isolated deployment cost."""
from importlib import import_module
import os

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.context_cache.lab_data import FEATURES
from ids.route_batch.metrics import threshold

torch = pytest.importorskip("torch")
model = import_module("ids.flow_history.model")


def synthetic():
    rng = np.random.default_rng(433)
    rows = 280
    x = rng.uniform(0, 10, size=(rows, len(FEATURES)))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame.loc[:, list(FEATURES[:4])] = np.rint(frame.loc[:, list(FEATURES[:4])]*11)
    frame["label"] = ((x[:, 4] + x[:, 5]) > 10).astype(int)
    frame["attack_family"] = np.where(frame.label, "Scanning", "Benign")
    frame["sample_id"] = [f"synthetic:{i}" for i in range(rows)]
    frame["history_count"] = rng.poisson(8, rows)
    return frame.iloc[:180].copy(), frame.iloc[180:].copy()


def test_paired_initial_function_and_added_slots():
    rng = torch.Generator().manual_seed(41)
    raw = torch.randn(15, 48, generator=rng)
    history = torch.randn(15, 4, generator=rng)
    regular, extended = model._network(48), model._network(52)
    torch.testing.assert_close(regular(raw), extended(torch.cat([raw, history], dim=1)))
    assert torch.count_nonzero(extended[0].weight[:, 48:]).item() == 0
    extended(torch.cat([raw, history], dim=1)).sum().backward()
    assert torch.count_nonzero(extended[0].weight.grad[:, 48:]).item() > 0


def test_fit_roundtrip_schema_constants_and_p_only(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "EPOCHS", 3)
    p, e = synthetic()
    names = [*FEATURES, "history_count"]
    p["history_count"] = 0
    path = model.fit_arm(p, names, tmp_path / "model.pt", threads=2, device="cpu")
    fitted = model.load_arm(path, device="cpu", threads=2)
    assert fitted.mean[-1] == 0 and fitted.scale[-1] == 1
    assert len(fitted.mean) == len(names)
    assert read_json(path.with_suffix(".fit.json"))["rows"] == len(p)
    first = model.score_arm(fitted, e, names)
    repeated = model.score_arm(model.load_arm(path, device="cpu", threads=2), e, names)
    np.testing.assert_array_equal(first, repeated)
    assert len(first) == len(e) and np.isfinite(first).all()
    with pytest.raises(ValueError, match="feature order"):
        model.score_arm(fitted, e, names[::-1])
    with pytest.raises(ValueError, match="nonfinite"):
        model.score_arm(fitted, e.assign(history_count=np.nan), names)
    with pytest.raises(ValueError, match="labels and identifiers"):
        model.fit_arm(p, ["label"], path, threads=1, device="cpu")


@pytest.mark.parametrize("threads", [1, 2])
def test_isolated_cost_and_score_evidence(tmp_path, monkeypatch, threads):
    monkeypatch.setattr(model, "EPOCHS", 2)
    p, e = synthetic()
    names = [*FEATURES, "history_count"]
    path = model.fit_arm(p, names, tmp_path / "model.pt", threads=1, device="cpu")
    fitted = model.load_arm(path, device="cpu", threads=1)
    c = pd.concat([p, e]).loc[lambda frame: frame.label == 0]
    baseline_c, baseline_e = [model.score_arm(fitted, frame, names) for frame in (c, e)]
    if threads == 2:
        baseline_e += 3.0  # A changed endpoint must produce inspectable mismatched rows.
    work = tmp_path / "cost"
    result = model.bench_arm(path, c, e, names, threads, reference_c=baseline_c,
                             reference_e=baseline_e, output_dir=work)
    assert result["pid"] != os.getpid() and result["threads"] == threads
    assert result["benchmark"]["batch"]["requested_batch_size"] == 128
    assert result["benchmark"]["single"]["n_observations"] == 192
    assert result["benchmark"]["rss"]["sampled_peak_bytes"] > 0
    assert result["model_artifact_bytes"] == path.stat().st_size
    assert result["feature_engine_cost_included"] is False
    assert result["replay"]["equivalent_to_reference"] == (threads == 1)
    evidence = pd.read_csv(work / "evaluation-replay.csv.gz")
    assert evidence.row_id.tolist() == e.sample_id.tolist()
    assert len(list(work.glob("inputs-*"))) == 0
    assert result["replay"]["reference_registration"]["threshold"] == threshold(baseline_c)["threshold"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime unavailable")
def test_synthetic_cuda_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "EPOCHS", 2)
    p, e = synthetic()
    p = pd.concat([p] * 24, ignore_index=True)
    names = [*FEATURES, "history_count"]
    path = model.fit_arm(p, names, tmp_path / "mlp.pt", threads=2)
    gpu = model.load_arm(path, device="cuda:0", threads=2)
    actual = model.score_arm(gpu, e, names)
    cpu = model.load_arm(path, device="cpu", threads=2)
    ref = model.score_arm(cpu, e, names)
    np.testing.assert_allclose(actual, ref, rtol=3e-5, atol=3e-5)
    meta = read_json(path.with_suffix(".fit.json"))
    assert meta["device"] == "cuda:0" and meta["cuda_peak_allocated_bytes"] > 0
    probe = meta["score_batch_probe"]
    assert probe["status"] == "measured_gpu" and probe["rows_per_trial"] == 4096
    candidates = probe["candidates"]
    assert [row["batch"] for row in candidates] == [512, 2048, 4096]
    assert all(len(row["seconds"]) == 5 and row["cuda_peak_allocated_bytes"] > 0 for row in candidates)
    best = max((r for r in candidates if r["within_score_tolerance"]), key=lambda r: r["median_rows_per_second"])
    assert gpu.score_batch == best["batch"] == probe["selected"]
    assert cpu.score_batch == model.SCORE_BATCH
    assert torch.load(path, weights_only=True)["score_batch_probe"] == probe
