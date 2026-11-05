"""Synthetic evidence for the ordinary-KD target and training contracts."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from ids.models import ModelBundle, _network
from ids.risk_research import _state, _state_hash
from ids.unsw_kd import (_backward_full_f, array_sha256, build_targets,
                        class_weights, fit_arm)
from ids.upgrade_kd import raw_bernoulli_kl_rows


def fitting_identity():
    # Includes pure repeated patterns, two mixed-label patterns, and one
    # nonexchangeable count vector. The row order is deliberately unsorted.
    pattern_rows = [
        ("z", [0, 0], -.3), ("d", [1, 1], 1.8),
        ("a", [0, 0], -3.), ("y", [1, 1], .1),
        ("m", [0, 1], -.7), ("b", [1, 0], .9),
        ("single", [1, 1, 1], 3.),
    ]
    rows, logits = [], []
    for pattern, labels, logit in pattern_rows:
        for label in labels:
            rows.append({"sample_id": f"s{len(rows)}", "feature_hash": pattern,
                         "label": label, "attack_cat": "unused"})
            logits.append(logit)
    return pd.DataFrame(rows), np.asarray(logits, dtype=np.float64)


def anchor(width=5):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(314)
        net = _network(width, (7, 3))
    return ModelBundle("mlp", None, [f"x{i}" for i in range(width)], width, [],
                       network_state=_state(net), hidden_dims=(7, 3), seed=17)


def test_pattern_permutation_preserves_row_marginals_and_identical_input_targets():
    frame, logits = fitting_identity()
    targets, audit, arrays = build_targets(frame, logits)
    y = frame.label.to_numpy()
    assert audit["feature_patterns"] == 7
    assert audit["n_strata"] == 4
    assert audit["nonexchangeable_patterns"] == 1
    assert audit["eligible_rows"] == 12
    assert audit["actually_changed_rows"] == 12
    assert audit["class_conditional_q_multisets_exactly_preserved"]
    q, qp = arrays["teacher_q"], arrays["permuted_teacher_q"]
    for c in (0, 1):
        np.testing.assert_array_equal(np.sort(q[y == c]), np.sort(qp[y == c]))
        assert arrays["class_mean_q"][y == c].min() == q[y == c].mean()
        assert arrays["weights"][y == c].sum() == pytest.approx(.5)
    assert pd.DataFrame({"h": frame.feature_hash, "q": qp}).groupby("h").q.nunique().max() == 1
    # Mixed targets use the row's hard label; teacher q itself remains pattern-consistent.
    np.testing.assert_array_equal(targets["H"], y)
    for arm, soft in (("K", q), ("S", qp), ("M", arrays["class_mean_q"])):
        np.testing.assert_array_equal(targets[arm], (y+.3*soft)/1.3)
    again, audit2, arrays2 = build_targets(frame, logits)
    assert audit == audit2
    for key in targets:
        np.testing.assert_array_equal(again[key], targets[key])
    np.testing.assert_array_equal(arrays2["donor_group_index"], arrays["donor_group_index"])
    # Attack family names are not used to construct either control.
    changed = frame.assign(attack_cat=np.arange(len(frame)))
    targets2, _, _ = build_targets(changed, logits)
    for arm in targets:
        np.testing.assert_array_equal(targets[arm], targets2[arm])


def test_frozen_q_is_bound_without_reexpressing_sigmoid():
    frame, logits = fitting_identity()
    q = expit(logits)
    targets, audit, _ = build_targets(frame, logits, frozen_q=q)
    assert audit["teacher_q_sha256"] == array_sha256(q)
    assert audit["target_source"] == "frozen bound teacher_q"
    np.testing.assert_array_equal(targets["K"], (frame.label.to_numpy()+.3*q)/1.3)


def test_inconsistent_pattern_q_is_rejected_before_training():
    frame, logits = fitting_identity()
    logits[1] += .1
    with pytest.raises(ValueError, match="identical feature pattern"):
        build_targets(frame, logits)
    with pytest.raises(ValueError, match="aligned"):
        build_targets(frame, logits[:-1])


def test_actual_float32_network_chunk_gradient_matches_normalized_kl():
    rng = np.random.default_rng(62)
    x = rng.normal(size=(17, 5)).astype(np.float32)
    # Some chunks contain only one class: chunk-local weights would be wrong.
    y = np.asarray([0]*3 + [1]*14)
    t = rng.normal(size=len(y)).astype(np.float64)
    q = expit(t)
    r = (y + .3*q)/1.3
    w = class_weights(y)
    base = anchor()
    whole, chunked, explicit = [_network(5, (7, 3)) for _ in range(3)]
    for net in (whole, chunked, explicit):
        net.load_state_dict(base.network_state)
    whole_log = _backward_full_f(whole, x, y, r, w, chunk_size=len(y), teacher_probabilities=q)
    chunk_log = _backward_full_f(chunked, x, y, r, w, chunk_size=4, teacher_probabilities=q)
    z = explicit(torch.from_numpy(x)).squeeze(1).double()
    hard = torch.nn.functional.binary_cross_entropy_with_logits(
        z, torch.from_numpy(y.astype(np.float64)), reduction="none")
    kl = raw_bernoulli_kl_rows(z, teacher_logits=torch.from_numpy(t), temperature=1.)
    (torch.from_numpy(w)*(hard+.3*kl)/1.3).sum().backward()
    for p_whole, p_chunk, p_explicit in zip(whole.parameters(), chunked.parameters(), explicit.parameters()):
        torch.testing.assert_close(p_chunk.grad, p_whole.grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(p_explicit.grad, p_whole.grad, rtol=1e-5, atol=1e-7)
    assert chunk_log["gradient_forward_rows"] == whole_log["gradient_forward_rows"] == len(y)
    assert chunk_log["backward_calls"] == 5
    assert whole_log["backward_calls"] == 1
    assert chunk_log["objective_bce"] == pytest.approx(
        chunk_log["normalized_hard_plus_kl"] + chunk_log["teacher_entropy_constant"], abs=1e-12)


def test_four_arms_return_only_final_endpoint_and_leave_anchor_unchanged(monkeypatch):
    import ids.unsw_kd as implementation

    frame, logits = fitting_identity()
    targets, _, arrays = build_targets(frame, logits)
    y = frame.label.to_numpy()
    x = np.random.default_rng(5).normal(size=(len(y), 5)).astype(np.float32)
    x_original = x.copy()
    base = anchor()
    state_before = _state_hash(base.network_state)
    initial_state = copy.deepcopy(base.network_state)
    forward_calls = []
    original_network = implementation._network

    def counting_network(*args, **kwargs):
        net = original_network(*args, **kwargs)
        net.register_forward_hook(lambda module, inputs, output: forward_calls.append(len(inputs[0])))
        return net

    monkeypatch.setattr(implementation, "_network", counting_network)
    original_threads = torch.get_num_threads()
    results = {}
    for arm in ("H", "K", "S", "M"):
        callback_steps = []
        q = {"H": None, "K": arrays["teacher_q"],
             "S": arrays["permuted_teacher_q"], "M": arrays["class_mean_q"]}[arm]
        candidate, history, report = fit_arm(
            base, x, y, targets[arm], arm=arm, device="cpu", teacher_probabilities=q,
            macro_steps=3, chunk_size=4, on_step=lambda row: callback_steps.append(row["step"]))
        assert callback_steps == [1, 2, 3]
        assert len(history) == 3
        assert report["endpoint_step"] == candidate.best_epoch == 3
        assert report["optimizer_updates"] == 3
        assert report["gradient_forward_rows"] == 45
        assert report["backward_calls"] == 12
        assert report["R_C_evaluation_rows_inside_training"] == 0
        assert report["checkpoint_selection"] is False
        assert report["common_initial_state_sha256"] == state_before
        assert report["final_state_sha256"] != state_before
        assert candidate._cached_network is None
        assert candidate.training_statistics == {} and candidate.history == []
        assert report["cuda_peak_allocated_bytes"] is None
        results[arm] = report["final_state_sha256"]
    assert sum(forward_calls) == 4 * 3 * len(y)
    assert len(forward_calls) == 4 * 3 * 4
    assert len(set(results.values())) > 1
    assert torch.get_num_threads() == original_threads
    assert _state_hash(base.network_state) == state_before
    for key in initial_state:
        torch.testing.assert_close(initial_state[key], base.network_state[key], rtol=0, atol=0)
    np.testing.assert_array_equal(x, x_original)


def test_fixed_anchor_replay_is_identical_and_soft_q_must_match_target():
    frame, logits = fitting_identity()
    targets, _, arrays = build_targets(frame, logits)
    x = np.random.default_rng(2).normal(size=(len(frame), 5)).astype(np.float32)
    base = anchor()
    kwargs = dict(arm="K", device="cpu", teacher_probabilities=arrays["teacher_q"],
                  macro_steps=2, chunk_size=4)
    _, history1, report1 = fit_arm(base, x, frame.label, targets["K"], **kwargs)
    _, history2, report2 = fit_arm(base, x, frame.label, targets["K"], **kwargs)
    assert history1 == history2
    assert report1["final_state_sha256"] == report2["final_state_sha256"]
    with pytest.raises(ValueError, match="exact frozen q"):
        fit_arm(base, x, frame.label, targets["K"], arm="K", device="cpu",
                teacher_probabilities=arrays["permuted_teacher_q"], macro_steps=1)
    with pytest.raises(ValueError, match="hard labels"):
        fit_arm(base, x, frame.label, targets["K"], arm="H", device="cpu", macro_steps=1)
