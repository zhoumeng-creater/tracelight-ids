"""Official TabM continuation checks on 48 generated rows, never real F/C/V."""
from copy import deepcopy
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabm")
pytest.importorskip("rtdl_num_embeddings")

from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network
from scripts.comprehensive_batch import training


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def device(request):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with training.scoring_arithmetic():
        try:
            yield torch.device(request.param)
        finally:
            torch.set_num_threads(threads)


@pytest.mark.parametrize("arm,lr,weight", [("R_reference_envelope", .002, 1.),
                                           ("S_alignment_group", .0005, .25)])
def test_official_tabm_optimizer_rng_and_snapshot_resume_exact(tmp_path, device, arm, lr, weight):
    rng = np.random.Generator(np.random.PCG64(193))
    numeric = rng.normal(size=(48, 5)).astype(np.float32)
    cardinalities = [6, 2, 2, 2, 2, 2, 2, 2]
    categorical = np.column_stack([rng.integers(0, c, size=48, dtype=np.int64) for c in cardinalities])
    labels = np.array([0] * 32 + [1] * 16, dtype=np.float32)
    groups = {"references": [{"id": "s1", "indices": list(range(16))},
                              {"id": "s2", "indices": list(range(16, 32))}],
              "targets": [{"id": f"b{i}", "indices": list(range(i, i + 8))} for i in range(0, 32, 8)],
              "attacks": list(range(32, 48))}
    parameters = _parameters(training.FIXED_MODEL)
    bins, _ = _numeric_bins(numeric, parameters)
    torch.manual_seed(193)
    prototype = make_tabm_network(5, cardinalities, parameters, numeric_bins=bins).to(device)
    spec = dict(training.FIXED_TRAINING, batch_size=16, draws_per_class=8,
                score_batch_size=16, report_every=1, recovery_every=1)
    job = {"job_id": "official_synthetic", "arm": arm, "seed": 17,
           "learning_rate": lr, "auxiliary_weight": weight, "optimizer_steps": 4,
           "checkpoints": [2, 4]}
    args = dict(job=job, device=device, training_spec=spec)
    uninterrupted = deepcopy(prototype)
    expected = training.train_loop(uninterrupted, numeric, categorical, labels, groups,
                                   output_dir=tmp_path / "full", **args)
    with pytest.raises(InterruptedError):
        training.train_loop(deepcopy(prototype), numeric, categorical, labels, groups,
                            output_dir=tmp_path / "resumed", interrupt_after_step=2, **args)
    restored = deepcopy(prototype)
    actual = training.train_loop(restored, numeric, categorical, labels, groups,
                                 output_dir=tmp_path / "resumed", resume=True, **args)
    assert actual["state_sha256"] == expected["state_sha256"]
    def without_measured_seconds(history):
        history = deepcopy(history)
        for record in history:
            record["cumulative_cost"] = {key: value for key, value in record["cumulative_cost"].items()
                                         if not key.endswith("_seconds")}
        return history
    assert without_measured_seconds(actual["history"]) == without_measured_seconds(expected["history"])
    assert actual["sampling"] == expected["sampling"]
    assert actual["final_dropout_rng_sha256"] == expected["final_dropout_rng_sha256"]
    assert actual["cost"]["transfer_discovery_forward_rows"] == 4 * 48
    for one, two in zip(actual["snapshots"], expected["snapshots"]):
        assert one["state_sha256"] == two["state_sha256"]
        a = np.load(tmp_path / "resumed" / one["F_scores"])
        b = np.load(tmp_path / "full" / two["F_scores"])
        np.testing.assert_array_equal(a["scores"], b["scores"])
    checkpoint = torch.load(tmp_path / "resumed" / actual["snapshots"][-1]["path"],
                            map_location="cpu", weights_only=True)
    cold = make_tabm_network(5, cardinalities, parameters, numeric_bins=bins).to(device)
    cold.load_state_dict(checkpoint["state_dict"])
    score = training.score_full(cold, numeric, categorical, device=device, batch_size=16).scores.numpy()
    np.testing.assert_array_equal(score, a["scores"])
