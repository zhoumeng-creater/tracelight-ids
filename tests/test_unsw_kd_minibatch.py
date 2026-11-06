"""Synthetic checks of H/K batching, frozen objectives, and actual work counts."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from ids.models import ModelBundle, _network
from ids.risk_research import _state, _state_hash
from ids.unsw_kd import array_sha256, class_weights
from ids.unsw_kd_minibatch import BATCH_SIZE, LOSS_SCOPE, fit_minibatch_arm


def synthetic_case(n=1025, epochs=2):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314)
        net = _network(5, (7, 3))
    base = ModelBundle("mlp", None, [f"x{i}" for i in range(5)], 5, [],
                       network_state=_state(net), hidden_dims=(7, 3), seed=17)
    rng = np.random.default_rng(62)
    x = rng.normal(size=(n, 5)).astype(np.float32)
    # A singleton minority-class tail exposes batch-local weighting and N/1024 bugs.
    y = np.asarray([0]*(n-1)+[1])
    q = rng.uniform(.05, .95, size=n)
    orders = np.asarray([np.arange(n) if i == 0 else rng.permutation(n)
                         for i in range(epochs)], dtype=np.int64)
    return base, x, y, q, orders


def test_exactly_one_forward_backward_clip_step_per_batch_and_only_final_state(monkeypatch):
    import ids.unsw_kd_minibatch as implementation

    base, x, y, q, orders = synthetic_case()
    original = copy.deepcopy(base.network_state)
    anchor_hash = _state_hash(original)
    forwards, backwards, clips, optimizers = [], [], [], []
    network_factory = implementation._network
    optimizer_factory = torch.optim.AdamW
    backward_method = torch.Tensor.backward
    clip_method = torch.nn.utils.clip_grad_norm_

    def counted_network(*args, **kwargs):
        network = network_factory(*args, **kwargs)
        network.register_forward_hook(lambda m, inputs, out: forwards.append(len(inputs[0])))
        return network

    def counted_optimizer(*args, **kwargs):
        optimizer = optimizer_factory(*args, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer

    def counted_backward(tensor, *args, **kwargs):
        backwards.append(tensor.dtype)
        return backward_method(tensor, *args, **kwargs)

    def counted_clip(*args, **kwargs):
        norm = clip_method(*args, **kwargs)
        clips.append(float(norm))
        return norm

    monkeypatch.setattr(implementation, "_network", counted_network)
    monkeypatch.setattr(torch.optim, "AdamW", counted_optimizer)
    monkeypatch.setattr(torch.Tensor, "backward", counted_backward)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", counted_clip)
    results = {}
    for arm in ("H", "K"):
        callbacks = []

        def callback(record):
            callbacks.append(record["epoch"])
            record["epoch"] = -1

        target = y if arm == "H" else (y+.3*q)/1.3
        candidate, history, report = fit_minibatch_arm(
            base, x, y, target, orders, arm=arm, device="cpu",
            teacher_probabilities=None if arm == "H" else q, on_epoch=callback)
        assert callbacks == [1, 2] and [row["epoch"] for row in history] == [1, 2]
        assert report["optimizer_updates"] == report["backward_calls"] == report["forward_calls"] == 4
        assert report["gradient_forward_rows"] == 2050
        assert report["batches_per_epoch"] == 2 and report["batch_size"] == BATCH_SIZE == 1024
        assert report["extra_F_evaluation_rows"] == report["R_C_evaluation_rows_inside_training"] == 0
        assert report["teacher_queries"] == 0 and not report["checkpoint_selection"]
        assert report["endpoint_epoch"] == candidate.best_epoch == 2
        assert report["common_initial_state_sha256"] == report["anchor_state_after_sha256"] == anchor_hash
        assert report["target_sha256"] == array_sha256(target)
        assert report["weights_sha256"] == array_sha256(class_weights(y))
        assert report["permutations_sha256"] == array_sha256(orders, "<i8")
        assert report["epoch_permutation_sha256"] == [array_sha256(row, "<i8") for row in orders]
        assert report["training_loss_scope"] == LOSS_SCOPE
        assert candidate._cached_network is None and candidate.history == []
        assert candidate.training_statistics == {} and candidate.device == "cpu"
        assert report["cuda_peak_allocated_bytes"] is None
        for epoch, row in enumerate(history, 1):
            assert row["last_batch_rows"] == 1
            assert row["optimizer_updates"] == row["backward_calls"] == 2
            assert row["cumulative_optimizer_updates"] == row["cumulative_backward_calls"] == epoch*2
            assert row["cumulative_gradient_forward_rows"] == epoch*len(y)
            assert row["objective_bce"] == pytest.approx(
                row["normalized_hard_plus_kl"]+row["teacher_entropy_constant"], abs=1e-12)
        results[arm] = report["final_state_sha256"]
    assert forwards == [1024, 1]*4 and backwards == [torch.float64]*8 and len(clips) == 8
    assert len(optimizers) == 2 and optimizers[0] is not optimizers[1]
    for optimizer in optimizers:
        assert all(float(state["step"]) == 4 for state in optimizer.state.values())
        group = optimizer.param_groups[0]
        assert group["lr"] == .001 and group["weight_decay"] == .0001
        assert group["betas"] == (.9, .999) and group["eps"] == 1e-8
        assert group["foreach"] is False and group["fused"] is False and not group["amsgrad"]
    assert results["H"] != results["K"]
    assert _state_hash(base.network_state) == anchor_hash
    for key in original:
        torch.testing.assert_close(base.network_state[key], original[key], rtol=0, atol=0)


def test_singleton_tail_gradient_uses_global_weights_times_actual_batch_inverse(monkeypatch):
    import ids.unsw_kd_minibatch as implementation

    base, x, y, q, orders = synthetic_case(epochs=1)
    r = (y+.3*q)/1.3
    captured = []
    network_factory = implementation._network

    def observe_network(*args, **kwargs):
        net = network_factory(*args, **kwargs)

        def before_forward(module, inputs):
            captured.append({"state": _state(module), "input": inputs[0].detach().clone()})

        net.register_forward_pre_hook(before_forward)
        return net

    original_clip = torch.nn.utils.clip_grad_norm_

    def observe_clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        captured[-1]["gradient"] = [p.grad.clone() for p in parameters]
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(implementation, "_network", observe_network)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", observe_clip)
    fit_minibatch_arm(base, x, y, r, orders, arm="K", device="cpu", teacher_probabilities=q)
    assert [len(row["input"]) for row in captured] == [1024, 1]
    counts = np.bincount(y, minlength=2)
    for batch, row in enumerate(captured):
        start, n_batch = batch*1024, len(row["input"])
        reference = _network(5, (7, 3))
        reference.load_state_dict(row["state"])
        z = reference(row["input"]).squeeze(1).double()
        # Independent class-count formula, including the single-class tail.
        indices = np.arange(start, start+n_batch)
        importance = len(y)/(2*counts[y[indices]])
        row_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            z, torch.from_numpy(r[indices]), reduction="none")
        (row_bce*torch.from_numpy(importance)).mean().backward()
        for expected, actual in zip(reference.parameters(), row["gradient"]):
            torch.testing.assert_close(actual, expected.grad, rtol=1e-6, atol=1e-6)
    assert sum(float(g.square().sum()) for g in captured[-1]["gradient"])**.5 > 10


def test_replay_and_hard_teacher_equivalence_restore_rng_threads_and_autocast():
    base, x, y, _, orders = synthetic_case(n=19)
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    first, h_history, h_report = fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        repeated, repeated_history, _ = fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu")
    kd, k_history, _ = fit_minibatch_arm(
        base, x, y, (y+.3*y)/1.3, orders, arm="K", device="cpu", teacher_probabilities=y)
    assert _state_hash(first.network_state) == _state_hash(repeated.network_state) == _state_hash(kd.network_state)
    assert h_history == repeated_history
    assert [r["objective_bce"] for r in h_history] == [r["objective_bce"] for r in k_history]
    assert h_report["network_dtype"] == "float32" and h_report["loss_dtype"] == "float64"
    assert h_report["amp"] is False and h_report["tf32"] is False
    assert all(value.dtype == torch.float32 for value in first.network_state.values())
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)


def test_callback_failure_restores_runtime_and_original_anchor():
    base, x, y, _, orders = synthetic_case(n=19)
    before_rng = torch.get_rng_state().clone()
    before_threads = torch.get_num_threads()
    before_deterministic = torch.are_deterministic_algorithms_enabled()
    before_hash = _state_hash(base.network_state)

    def fail(record):
        raise RuntimeError("synthetic callback failure")

    with pytest.raises(RuntimeError, match="synthetic callback failure"):
        fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu", on_epoch=fail)
    assert torch.get_num_threads() == before_threads
    assert torch.are_deterministic_algorithms_enabled() == before_deterministic
    assert _state_hash(base.network_state) == before_hash
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)


@pytest.mark.parametrize("change", ["duplicate", "float", "width", "empty", "negative", "rank"])
def test_invalid_permutations_fail_before_any_network_creation(monkeypatch, change):
    import ids.unsw_kd_minibatch as implementation

    base, x, y, _, orders = synthetic_case(n=19)
    if change == "duplicate":
        orders[0, 0] = orders[0, 1]
    elif change == "float":
        orders = orders.astype(float)
    elif change == "width":
        orders = orders[:, :-1]
    elif change == "empty":
        orders = orders[:0]
    elif change == "negative":
        orders[0, 0] = -1
    else:
        orders = orders[0]
    monkeypatch.setattr(implementation, "_network", lambda *a, **kw: pytest.fail("network created"))
    with pytest.raises(ValueError, match="permutations|every F row"):
        fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu")


@pytest.mark.parametrize("arm", ["S", "M", "h", "teacher"])
def test_no_additional_arms(arm):
    base, x, y, _, orders = synthetic_case(n=19)
    with pytest.raises(ValueError, match="Only H/K"):
        fit_minibatch_arm(base, x, y, y, orders, arm=arm, device="cpu")


def test_targets_are_exactly_bound_and_inputs_are_finite():
    base, x, y, q, orders = synthetic_case(n=19)
    r = (y+.3*q)/1.3
    with pytest.raises(ValueError, match="exact frozen"):
        fit_minibatch_arm(base, x, y, r, orders, arm="K", device="cpu", teacher_probabilities=1-q)
    with pytest.raises(ValueError, match="exact frozen"):
        fit_minibatch_arm(base, x, y, r, orders, arm="K", device="cpu")
    with pytest.raises(ValueError, match="hard labels"):
        fit_minibatch_arm(base, x, y, r, orders, arm="H", device="cpu")
    with pytest.raises(ValueError, match="no teacher"):
        fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu", teacher_probabilities=q)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite encoded F"):
        fit_minibatch_arm(base, x, y, y, orders, arm="H", device="cpu")
