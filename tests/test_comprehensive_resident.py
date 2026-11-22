"""Resident execution parity on generated arrays only; no F/C/V files are read."""
from copy import deepcopy
import json
import os

import numpy as np
import pytest
import torch

from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network
from scripts.comprehensive_batch import objective, training


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


def generated(rows=39):
    rng = np.random.Generator(np.random.PCG64(204))
    numeric = rng.normal(size=(rows, 5)).astype(np.float32)
    cardinalities = [6, 2, 2, 2, 2, 2, 2, 2]
    categorical = np.column_stack([rng.integers(c, size=rows, dtype=np.int64) for c in cardinalities])
    normal = rows * 2 // 3
    labels = np.array([0] * normal + [1] * (rows - normal), dtype=np.float32)
    split = normal // 2
    edges = (0, split // 2, split, split + (normal - split) // 2, normal)
    groups = {"references": [{"id": "s1", "indices": list(range(split))},
                              {"id": "s2", "indices": list(range(split, normal))}],
              "targets": [{"id": f"g{i}", "indices": list(range(start, stop))}
                          for i, (start, stop) in enumerate(zip(edges[:-1], edges[1:]))],
              "attacks": list(range(normal, rows))}
    return numeric, categorical, labels, groups, cardinalities


class TinyHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, numeric, categorical):
        return (self.linear(self.dropout(numeric)) + categorical[:, :1] * .01).unsqueeze(-1)


def official(numeric, cardinalities, device):
    pytest.importorskip("tabm")
    pytest.importorskip("rtdl_num_embeddings")
    parameters = _parameters(training.FIXED_MODEL)
    bins, _ = _numeric_bins(numeric, parameters)
    torch.manual_seed(214)
    return make_tabm_network(5, cardinalities, parameters, numeric_bins=bins).to(device)


def gradients(network):
    return [None if p.grad is None else p.grad.detach().cpu().clone() for p in network.parameters()]


def assert_gradients_equal(before, after):
    for one, two in zip(before, after):
        assert (one is None and two is None) or (one is not None and two is not None and torch.equal(one, two))


def history_without_timings(history):
    result = deepcopy(history)
    for row in result:
        row["cumulative_cost"] = {k: v for k, v in row["cumulative_cost"].items() if not k.endswith("_seconds")}
    return result


def test_same_thirteen_blocks_reuse_inputs_without_numpy_block_conversion(device, monkeypatch):
    numeric, categorical, _, _, _ = generated(13252)
    torch.manual_seed(11)
    model = TinyHeads().to(device).train()
    x, cats = torch.from_numpy(numeric).to(device), torch.from_numpy(categorical).to(device)
    resident = objective.resident_inputs(numeric, categorical, device=device,
                                        numeric_tensor=x, categorical_tensor=cats)
    assert resident.numeric_tensor.data_ptr() == x.data_ptr()
    assert resident.categorical_tensor.data_ptr() == cats.data_ptr()
    expected = objective.score_full(model, numeric, categorical, device=device, batch_size=1024)
    random = training._rng_digest(device)
    shapes = []
    handle = model.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    monkeypatch.setattr(torch, "from_numpy", lambda *a, **k: pytest.fail("resident block repeated CPU conversion"))
    actual = objective.score_full(model, numeric, categorical, device=device, batch_size=1024, resident=resident)
    handle.remove()
    assert torch.equal(actual.scores, expected.scores)
    assert actual.blocks == expected.blocks and len(actual.blocks) == 13
    assert shapes == [(stop - start, 5) for start, stop in expected.blocks]
    assert actual.scores.dtype == torch.float64 and actual.scores.device.type == "cpu"
    assert random == training._rng_digest(device) and model.training


@pytest.mark.parametrize("arm", objective.ARMS)
def test_official_scores_gradients_and_all_diagnostics_match_legacy(device, arm):
    numeric, categorical, labels, groups, cardinalities = generated()
    prototype = official(numeric, cardinalities, device)
    one, two = deepcopy(prototype), deepcopy(prototype)
    resident = objective.resident_inputs(numeric, categorical, device=device)
    results, saved_gradients, saved_rng = [], [], []
    for model, cache in ((one, None), (two, resident)):
        training._seed(97, device)
        model.train()
        # Existing stochastic BCE gradients must be preserved by either replay.
        logits = model(torch.from_numpy(numeric[:16]).to(device), torch.from_numpy(categorical[:16]).to(device))
        training.tabm_head_loss(logits, torch.from_numpy(labels[:16]).to(device)).backward()
        discovered = objective.score_full(model, numeric, categorical, device=device, batch_size=16, resident=cache)
        result = objective.backward_objective(model, numeric, categorical, groups, arm=arm, device=device,
                                               batch_size=16, score_pass=discovered, weight=.25, resident=cache)
        results.append((discovered, result))
        saved_gradients.append(gradients(model))
        saved_rng.append(training._rng_digest(device))
        assert model.training
    assert torch.equal(results[0][0].scores, results[1][0].scores)
    assert results[0][1] == results[1][1]
    assert saved_rng[0] == saved_rng[1]
    assert_gradients_equal(*saved_gradients)
    for model in (one, two):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003).step()
    assert training.state_digest(one.state_dict()) == training.state_digest(two.state_dict())


def test_official_production_block_shape_and_tail_match(device):
    numeric, categorical, _, groups, cardinalities = generated(2051)
    prototype = official(numeric, cardinalities, device)
    one, two = deepcopy(prototype), deepcopy(prototype)
    resident = objective.resident_inputs(numeric, categorical, device=device)
    saved = []
    for model, cache in ((one, None), (two, resident)):
        discovered = objective.score_full(model, numeric, categorical, device=device, batch_size=1024, resident=cache)
        result = objective.backward_objective(model, numeric, categorical, groups, arm="S_alignment_group",
                                               device=device, batch_size=1024, score_pass=discovered, resident=cache)
        saved.append((discovered, result, gradients(model)))
    assert saved[0][0].blocks == saved[1][0].blocks == ((0, 1024), (1024, 2048), (2048, 2051))
    assert torch.equal(saved[0][0].scores, saved[1][0].scores)
    assert saved[0][1] == saved[1][1]
    assert_gradients_equal(saved[0][2], saved[1][2])


@pytest.mark.parametrize("arm", objective.ARMS)
def test_official_short_trajectory_matches_legacy_state_dropout_and_history(tmp_path, device, arm):
    numeric, categorical, labels, groups, cardinalities = generated()
    prototype = official(numeric, cardinalities, device)
    spec = dict(training.FIXED_TRAINING, batch_size=16, draws_per_class=8,
                score_batch_size=16, report_every=1, recovery_every=1)
    job = {"job_id": "generated_resident_parity", "arm": arm, "seed": 29,
           "learning_rate": .0005, "auxiliary_weight": 4., "optimizer_steps": 2, "checkpoints": [1, 2]}
    result = []
    for enabled, name in ((False, "old"), (True, "resident")):
        result.append(training.train_loop(deepcopy(prototype), numeric, categorical, labels, groups,
                                          job=job, device=device, training_spec=spec, output_dir=tmp_path / name,
                                          use_resident=enabled))
    one, two = result
    for key in ("state_sha256", "initial_state_sha256", "initial_dropout_rng_sha256",
                "final_dropout_rng_sha256", "sampler_draw_sha256", "sampling"):
        assert one[key] == two[key]
    assert history_without_timings(one["history"]) == history_without_timings(two["history"])
    for a, b in zip(one["snapshots"], two["snapshots"]):
        assert a["state_sha256"] == b["state_sha256"]
        np.testing.assert_array_equal(np.load(tmp_path / "old" / a["F_scores"])["scores"],
                                      np.load(tmp_path / "resident" / b["F_scores"])["scores"])
        assert json.loads((tmp_path / "old" / a["diagnostics"]).read_text()) == json.loads(
            (tmp_path / "resident" / b["diagnostics"]).read_text())


def test_cooperative_stop_commits_and_resumes_exactly(tmp_path, monkeypatch):
    numeric, categorical, labels, groups, _ = generated()
    spec = dict(training.FIXED_TRAINING, batch_size=16, draws_per_class=8,
                score_batch_size=16, report_every=2, recovery_every=2)
    job = {"job_id": "generated_stop", "arm": "R_reference_envelope", "seed": 17,
           "learning_rate": .002, "auxiliary_weight": 1., "optimizer_steps": 4, "checkpoints": [2, 4]}
    torch.manual_seed(310)
    prototype = TinyHeads()
    args = dict(job=job, device="cpu", training_spec=spec)
    expected = training.train_loop(deepcopy(prototype), numeric, categorical, labels, groups,
                                   output_dir=tmp_path / "full", **args)
    stop = tmp_path / "execution_stop.request"
    original = training._save

    def signal_after_save(path, value):
        original(path, value)
        if path.name == "recovery.pt" and value["completed_step"] == 2:
            stop.write_text("coordinator requested stop", encoding="utf8")

    monkeypatch.setattr(training, "_save", signal_after_save)
    with pytest.raises(InterruptedError, match="checkpoint saved"):
        training.train_loop(deepcopy(prototype), numeric, categorical, labels, groups,
                            output_dir=tmp_path / "resume", stop_path=stop, **args)
    recovery = torch.load(tmp_path / "resume" / "recovery.pt", weights_only=True)
    assert recovery["completed_step"] == 2
    monkeypatch.setattr(training, "_save", original)
    stop.unlink()
    actual = training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                                 output_dir=tmp_path / "resume", stop_path=stop, resume=True, **args)
    assert actual["state_sha256"] == expected["state_sha256"]
    assert actual["sampler_draw_sha256"] == expected["sampler_draw_sha256"]
    assert actual["final_dropout_rng_sha256"] == expected["final_dropout_rng_sha256"]


def test_existing_stop_request_prevents_first_optimizer_update(tmp_path):
    numeric, categorical, labels, groups, _ = generated()
    stop = tmp_path / "execution_stop.request"
    stop.write_text("stop", encoding="utf8")
    job = {"job_id": "generated_stop_before_start", "arm": "A_BCE", "seed": 17,
           "learning_rate": .002, "auxiliary_weight": 1., "optimizer_steps": 1, "checkpoints": [1]}
    with pytest.raises(InterruptedError, match="checkpoint saved"):
        training.train_loop(TinyHeads(), numeric, categorical, labels, groups,
                            job=job, device="cpu", output_dir=tmp_path / "job", stop_path=stop)
    recovery = torch.load(tmp_path / "job" / "recovery.pt", weights_only=True)
    assert recovery["completed_step"] == 0 and recovery["sampler"]["audit"]["total_draws"] == 0


def test_resident_input_version_change_is_rejected():
    numeric, categorical, _, _, _ = generated()
    cache = objective.resident_inputs(numeric, categorical, device="cpu")
    cache.numeric_tensor.add_(1.)
    with pytest.raises(ValueError, match="binding or tensor version"):
        objective.score_full(TinyHeads(), numeric, categorical, device="cpu", batch_size=16, resident=cache)
