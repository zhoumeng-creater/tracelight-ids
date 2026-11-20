"""Optional official TabM checks on generated arrays only; never load project F/C/V.

Supply the pinned optional dependencies through the environment/PYTHONPATH.
The production architecture is exercised, but only two synthetic optimizer
updates occur and no artifact is presented as a completed laboratory fit.
"""
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabm")
pytest.importorskip("rtdl_num_embeddings")

from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network  # noqa: E402
from ids.route_batch.tabm_training import stable_scores  # noqa: E402
from scripts.calibration_transfer import training  # noqa: E402
from scripts.calibration_transfer.objective import backward_transfer, score_full  # noqa: E402
from scripts.normal_tail_mechanism import transfer_penalty  # noqa: E402


@pytest.fixture(params=["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable; official CPU path still checked"))])
def device(request, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old_threads = torch.get_num_threads()
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield torch.device(request.param)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[1:3]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[3:5]


def synthetic_inputs():
    rng = np.random.Generator(np.random.PCG64(193))
    numeric = rng.normal(size=(48, 5)).astype(np.float32)
    cardinalities = [6, 2, 2, 2, 2, 2, 2, 2]
    categorical = np.column_stack([rng.integers(0, width, size=48, dtype=np.int64)
                                   for width in cardinalities])
    labels = np.asarray([0] * 32 + [1] * 16, dtype=np.float32)
    episodes = [{"id": "synthetic_day_2", "Q": list(range(8)), "E": list(range(8, 16))},
                {"id": "synthetic_day_3", "Q": list(range(16)), "E": list(range(16, 32))}]
    parameters = _parameters(training.FIXED_MODEL)
    bins, constants = _numeric_bins(numeric, parameters)
    assert parameters["n_bins"] == 48
    assert not constants and len(bins) == 5
    return numeric, categorical, labels, episodes, cardinalities, parameters, bins


def official_model(cardinalities, parameters, bins, device):
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(193)
        return make_tabm_network(5, cardinalities, parameters, numeric_bins=bins).to(device)


def test_official_deployed_score_replay_matches_direct_gradient(device):
    numeric, categorical, labels, episodes, cardinalities, parameters, bins = synthetic_inputs()
    model = official_model(cardinalities, parameters, bins, device)
    direct = deepcopy(model).eval()
    model.train()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    discovery = score_full(model, numeric, categorical, device=device, batch_size=16)
    assert model.training
    direct_scores = torch.cat([
        stable_scores(direct(torch.from_numpy(numeric[start:start + 16]).to(device),
                             torch.from_numpy(categorical[start:start + 16]).to(device)))
        for start in range(0, len(numeric), 16)])
    assert direct_scores.dtype == torch.float64
    assert torch.equal(discovery.scores, direct_scores.detach().cpu())
    attacks = np.flatnonzero(labels == 1).tolist()
    replay = backward_transfer(model, numeric, categorical, episodes, attacks,
                               device=device, batch_size=16, score_pass=discovery)
    penalty = transfer_penalty([(direct_scores[e["Q"]], direct_scores[e["E"]]) for e in episodes],
                               direct_scores[attacks])["penalty"]
    penalty.backward()
    assert replay["penalty"] == float(penalty.detach())
    assert replay["cost"]["replay_backward_rows"] > 0
    assert model.training
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    checked = 0
    for actual, expected in zip(model.parameters(), direct.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=5e-7)
            checked += 1
    assert checked > 0


def test_official_two_update_loops_and_state_roundtrip(device, tmp_path):
    numeric, categorical, labels, episodes, cardinalities, parameters, bins = synthetic_inputs()
    prototype = official_model(cardinalities, parameters, bins, device)
    initial = training.state_digest(prototype.state_dict())
    mapping = np.arange(len(numeric))
    mapping[:32] = np.random.Generator(np.random.PCG64(29)).permutation(32)
    permuted = [{**e, "Q": mapping[e["Q"]].tolist(), "E": mapping[e["E"]].tolist()} for e in episodes]
    spec = dict(training.FIXED_TRAINING, optimizer_steps=2, batch_size=16,
                draws_per_class=8, score_batch_size=16, report_every=1)
    results = {}
    for arm, chosen in (("B_permuted", permuted), ("C_real", episodes)):
        model = deepcopy(prototype)
        assert training.state_digest(model.state_dict()) == initial
        out = tmp_path / arm
        out.mkdir()
        result = training.train_loop(model, numeric, categorical, labels, chosen,
                                     arm=arm, seed=17, training_spec=spec, device=device, output_dir=out)
        assert result["optimizer_steps"] == 2
        assert result["cost"]["transfer_discovery_forward_rows"] == 2 * len(numeric)
        assert result["cost"]["bce_forward_rows"] == 32
        assert training.state_digest(model.state_dict()) != initial
        assert all(torch.isfinite(value).all() for value in model.state_dict().values())
        expected = score_full(model, numeric, categorical, device=device, batch_size=16).scores
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        artifact = out / "synthetic_two_updates.pt"
        torch.save({"synthetic_only": True, "optimizer_steps": 2, "parameters": parameters,
                    "numeric_bins": bins, "state_dict": state}, artifact)
        loaded = torch.load(artifact, map_location="cpu", weights_only=True)
        assert loaded["synthetic_only"] and loaded["optimizer_steps"] == 2
        restored = official_model(cardinalities, loaded["parameters"], loaded["numeric_bins"], device)
        restored.load_state_dict(loaded["state_dict"])
        assert training.state_digest(restored.state_dict()) == training.state_digest(state)
        actual = score_full(restored, numeric, categorical, device=device, batch_size=16).scores
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        results[arm] = result
    assert results["B_permuted"]["sampling"] == results["C_real"]["sampling"]
    assert results["B_permuted"]["final_dropout_rng_sha256"] == results["C_real"]["final_dropout_rng_sha256"]


def test_short_synthetic_checkpoint_is_not_loadable_as_terminal_fit(tmp_path, monkeypatch):
    artifact = tmp_path / "short_synthetic_checkpoint.pt"
    torch.save({"synthetic_only": True, "format": training.FORMAT, "arm": "C_real", "seed": 17,
                "training_spec": dict(training.FIXED_TRAINING, optimizer_steps=2),
                "parameters": _parameters(training.FIXED_MODEL), "selected_step": 2,
                "checkpoint_rule": "terminal_step_only", "final_refits": 0}, artifact)

    def forbidden(*args, **kwargs):
        raise AssertionError("A shortened artifact must be rejected before constructing a model")

    monkeypatch.setattr(training, "make_tabm_network", forbidden)
    with pytest.raises(ValueError, match="nonterminal"):
        training.load_predictor(artifact, device="cpu")
