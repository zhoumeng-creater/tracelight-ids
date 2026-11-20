"""Exact deployed-score transfer gradient with bounded activation memory.

One call first scores every F row in eval mode at one parameter state. It then
differentiates the finite-episode objective with respect to that score vector,
and replays only the canonical blocks carrying nonzero score derivatives. The
replay uses the same parameters, mode, rows, order and block shape. No optimizer
step may occur between discovery and replay. This is a piecewise exact gradient,
including the tie conventions in ``normal_tail_mechanism``, not stale mining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ids.route_batch.tabm_training import stable_scores
from scripts.normal_tail_mechanism import budget, transfer_penalty


@dataclass(frozen=True)
class ScorePass:
    scores: torch.Tensor
    blocks: tuple[tuple[int, int], ...]
    state_versions: tuple
    batch_size: int
    forward_rows: int


def _versions(network):
    # Optimizer in-place parameter updates increment _version. Buffers are also
    # bound because changing a running statistic changes the deployed function.
    return tuple((kind, name, id(value), value._version)
                 for kind, values in (("parameter", network.named_parameters()),
                                      ("buffer", network.named_buffers()))
                 for name, value in values)


def _arrays(numeric, categorical):
    numeric = np.asarray(numeric)
    if (numeric.dtype != np.float32 or numeric.ndim != 2 or not len(numeric)
            or not np.isfinite(numeric).all()):
        raise ValueError("Expected finite nonempty float32 numeric input")
    if categorical is not None:
        categorical = np.asarray(categorical)
        if (categorical.dtype != np.int64 or categorical.ndim != 2
                or len(categorical) != len(numeric)):
            raise ValueError("Expected aligned int64 categorical input")
    return numeric, categorical


def _block(network, numeric, categorical, start, stop, device):
    numeric_block = torch.from_numpy(numeric[start:stop]).to(device)
    categorical_block = (None if categorical is None else
                         torch.from_numpy(categorical[start:stop]).to(device))
    scores = stable_scores(network(numeric_block, categorical_block))
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite deployed transfer score")
    return scores


def score_full(network, numeric, categorical, *, device, batch_size):
    """Return detached FP64 F scores without consuming training dropout RNG."""
    numeric, categorical = _arrays(numeric, categorical)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    blocks = tuple((start, min(start + batch_size, len(numeric)))
                   for start in range(0, len(numeric), batch_size))
    versions = _versions(network)
    training = network.training
    network.eval()
    try:
        with torch.no_grad():
            scores = torch.cat([_block(network, numeric, categorical, start, stop, device).cpu()
                                for start, stop in blocks])
    finally:
        network.train(training)
    if _versions(network) != versions:
        raise RuntimeError("Model state changed during full-score discovery")
    return ScorePass(scores, blocks, versions, batch_size, len(numeric))


def _indices(values, rows, name):
    result = torch.as_tensor(values, dtype=torch.long)
    if (result.ndim != 1 or not result.numel() or bool((result < 0).any())
            or bool((result >= rows).any()) or len(result.unique()) != len(result)):
        raise ValueError(f"{name} requires unique nonempty indices within F")
    return result


def score_gradient(scores, episodes, attack_indices, *, alpha=.01, margin=1.):
    """Compute the frozen operator and its exact score-space subgradient on CPU."""
    leaf = torch.as_tensor(scores, dtype=torch.float64, device="cpu").detach().clone()
    if leaf.ndim != 1 or not leaf.numel() or not bool(torch.isfinite(leaf).all()):
        raise ValueError("Need finite nonempty F scores")
    leaf.requires_grad_(True)
    attack_indices = _indices(attack_indices, len(leaf), "attack_indices")
    pairs, ids = [], []
    for position, episode in enumerate(episodes):
        if isinstance(episode, dict):
            q, e = episode["Q"], episode["E"]
            ids.append(episode.get("id", str(position)))
        else:
            q, e = episode
            ids.append(str(position))
        q, e = _indices(q, len(leaf), "Q"), _indices(e, len(leaf), "E")
        if set(q.tolist()) & set(e.tolist()):
            raise ValueError("Q and E must be disjoint within an episode")
        if (set(q.tolist()) | set(e.tolist())) & set(attack_indices.tolist()):
            raise ValueError("Normal episode contains a registered F positive")
        pairs.append((leaf[q], leaf[e]))
    result = transfer_penalty(pairs, leaf[attack_indices], alpha=alpha, margin=margin)
    derivative, = torch.autograd.grad(result["penalty"], leaf)
    diagnostics: dict[str, Any] = {
        key: float(result[key].detach())
        for key in ("penalty", "normal_penalty", "attack_penalty")}
    diagnostics["min_attack_score"] = float(leaf[attack_indices].min().detach())
    diagnostics["episodes"] = []
    for identity, pair, terms in zip(ids, pairs, result["episodes"]):
        record = {key: (float(value.detach()) if torch.is_tensor(value) else value)
                  for key, value in terms.items()}
        attack_tp = int((leaf[attack_indices].detach() > terms["past_boundary"].detach()).sum())
        record.update(id=identity, past_count=len(pair[0]), past_budget=budget(len(pair[0]), alpha),
                      past_fp=int((pair[0].detach() > terms["past_boundary"].detach()).sum()),
                      attack_count=len(attack_indices), attack_tp_at_past_boundary=attack_tp,
                      attack_fn_at_past_boundary=len(attack_indices) - attack_tp)
        diagnostics["episodes"].append(record)
    return derivative, diagnostics


def backward_transfer(network, numeric, categorical, episodes, attack_indices, *,
                      device, batch_size, score_pass=None, alpha=.01, margin=1.,
                      weight=1., backward=True):
    """Add transfer gradients to existing BCE gradients; never update weights.

    ``backward=False`` computes exact diagnostics without a second model forward.
    ``weight=0`` likewise has no transfer gradient. A supplied score pass must
    have been produced at the current model state and with this same block size.
    The caller freezes numeric/categorical arrays throughout discovery and replay.
    """
    numeric, categorical = _arrays(numeric, categorical)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    discovered_here = score_pass is None
    if score_pass is None:
        score_pass = score_full(network, numeric, categorical, device=device, batch_size=batch_size)
    if (score_pass.batch_size != batch_size or len(score_pass.scores) != len(numeric)
            or score_pass.state_versions != _versions(network)):
        raise RuntimeError("Stale or incompatible transfer score pass")
    derivative, diagnostics = score_gradient(
        score_pass.scores, episodes, attack_indices, alpha=alpha, margin=margin)
    active = derivative.ne(0)
    replayed_rows = 0
    replayed_blocks = 0
    training = network.training
    network.eval()
    try:
        if backward and weight:
            for start, stop in score_pass.blocks:
                if not bool(active[start:stop].any()):
                    continue
                scores = _block(network, numeric, categorical, start, stop, device)
                if not torch.equal(scores.detach().cpu(), score_pass.scores[start:stop]):
                    raise RuntimeError("Replay score differs from frozen discovery score")
                (scores * derivative[start:stop].to(device) * weight).sum().backward()
                replayed_rows += stop - start
                replayed_blocks += 1
    finally:
        network.train(training)
    if score_pass.state_versions != _versions(network):
        raise RuntimeError("Model state changed during transfer replay")
    diagnostics["cost"] = {
        "discovery_forward_rows_this_call": len(numeric) if discovered_here else 0,
        "discovery_forward_rows_in_score_pass": score_pass.forward_rows,
        "replay_forward_rows": replayed_rows,
        "replay_backward_rows": replayed_rows,
        "replay_blocks": replayed_blocks,
        "nonzero_score_derivatives": int(active.sum()),
    }
    diagnostics["applied_weight"] = float(weight) if backward else 0.
    return diagnostics
