"""Finite fitting-pool, one-way pAUC surrogate and its global-tail sampler.

This is an independent implementation of a standard empirical CVaR objective,
not LibAUC/SOPA, a new loss, or a population false-alarm guarantee. For a
monotone pair loss, the worst beta mass of negatives is the same for every
positive. We minimize the CVaR threshold analytically by sorting GLOBAL fitting
scores, sharing boundary-tie mass equally. Mini-batch top-k is a different loss.

The exact loss is useful for small-pool verification. The lab runner should
score the complete fitting pool at the current parameters, sample a fixed
number of pairs from that global tail, then accumulate all pair gradients
before ONE optimizer step. The caller owns/checkpoints the NumPy RNG state.
Sampling gives an unbiased subgradient at those frozen parameters; changing
parameters between chunks, or reusing a stale tail, loses that interpretation.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


def _fit(partition):
    if partition != "fit":
        raise ValueError("direct pAUC accepts the fitting role only")


def _positive_number(value, name, *, upper=None):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0
            or (upper is not None and value > upper)):
        raise ValueError(f"{name} must be finite and in (0, {upper or 'infinity'}]")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _pool(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if (scores.ndim != 1 or not scores.size or scores.dtype.kind != "f"
            or not np.isfinite(scores).all()):
        raise ValueError("scores must be a nonempty finite floating vector")
    if (labels.ndim != 1 or labels.shape != scores.shape or labels.dtype.kind not in "biuf"
            or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all()
            or not np.any(labels == 0) or not np.any(labels == 1)):
        raise ValueError("labels must be aligned binary values containing both classes")
    return scores.astype(np.float64, copy=False), labels


def global_negative_tail(scores, labels, *, beta=.01, partition="fit"):
    """Return full-row negative sampling weights and global-tail statistics.

    With M negatives, the CVaR mass is beta*M, which need not be an integer.
    All scores strictly above the cutoff receive mass 1; the remaining mass
    is shared equally by ALL scores tied at the cutoff. Returned ``weights``
    sum to one, and positives/below-cutoff negatives have zero weight. This
    symmetric choice is a valid subgradient at ties, independent of row order.
    All selection/weights are detached and must be refreshed each macrostep.
    """
    _fit(partition)
    beta = _positive_number(beta, "beta", upper=1.)
    scores, labels = _pool(scores, labels)
    negative = labels == 0
    negative_scores = scores[negative]
    mass = beta * len(negative_scores)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("beta times the negative count must be representable")
    # ceil rather than round preserves fractional final negative mass.
    rank = min(len(negative_scores), math.ceil(mass))
    cutoff = float(np.partition(negative_scores, len(negative_scores) - rank)[-rank])
    above = negative & (scores > cutoff)
    tied = negative & (scores == cutoff)
    above_count, tied_count = int(above.sum()), int(tied.sum())
    boundary_mass = mass - above_count
    weights = np.zeros(len(scores), dtype=np.float64)
    if above_count:
        weights[above] = 1. / mass
    weights[tied] = (boundary_mass / mass) / tied_count
    if (boundary_mass <= 0 or boundary_mass > tied_count + 1e-10
            or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1., atol=1e-12)):
        raise RuntimeError("global CVaR tail mass calculation failed")
    return {"weights": weights, "statistics": {
        "objective": "finite_pool_one_way_empirical_cvar_logistic_surrogate",
        "selection_pool": "all_fitting_rows", "beta": beta,
        "normal_rows": len(negative_scores), "attack_rows": int((labels == 1).sum()),
        "negative_tail_mass": mass, "cutoff": cutoff,
        "strictly_above_cutoff_rows": above_count, "boundary_tie_rows": tied_count,
        "boundary_mass": boundary_mass, "nonzero_negative_weights": int((weights > 0).sum()),
        "boundary_ties": "equal_fractional_mass", "libauc_reproduction": False}}


def sample_global_tail_pairs(scores, labels, *, rng, n_pairs=65536, beta=.01, partition="fit"):
    """Sample positive-uniform / global-CVaR-negative pairs WITH replacement.

    ``positive_indices`` and ``negative_indices`` address the original full
    fitting row vector. The arithmetic mean of ``pairwise_pauc_rows`` over all
    returned pairs estimates the exact empirical CVaR surrogate. No importance
    weights are needed because negatives are drawn from the target distribution.
    Reproducibility/resumption requires saving ``rng.bit_generator.state``;
    this function neither seeds a global RNG nor maintains hidden state.
    """
    _fit(partition)
    n_pairs = _positive_integer(n_pairs, "n_pairs")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    scores, labels = _pool(scores, labels)
    tail = global_negative_tail(scores, labels, beta=beta, partition=partition)
    positive = np.flatnonzero(labels == 1)
    supported_negative = np.flatnonzero(tail["weights"] > 0)
    probabilities = tail["weights"][supported_negative]
    # choice requires machine-accurate total probability; this only removes
    # floating summation error, not the mathematically defined fractional mass.
    probabilities = probabilities / probabilities.sum()
    pi = rng.choice(positive, size=n_pairs, replace=True)
    ni = rng.choice(supported_negative, size=n_pairs, replace=True, p=probabilities)
    return {"positive_indices": pi, "negative_indices": ni,
            "statistics": {**tail["statistics"], "n_pairs": n_pairs,
                           "positive_sampling": "uniform_with_replacement",
                           "negative_sampling": "global_tail_weights_with_replacement",
                           "pair_reduction": "arithmetic_mean_over_all_sampled_pairs"}}


def _logits(value, name):
    if (not isinstance(value, Tensor) or value.ndim != 1 or not value.numel()
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a nonempty finite floating tensor vector")
    return value


def pairwise_pauc_rows(positive_logits, negative_logits, *, margin=0., temperature=1., partition="fit"):
    """Return [P] T*softplus((margin + negative - positive)/T), without reduction.

    Inputs contain ALIGNED sampled pairs, not a Cartesian product. Margin is
    a fixed nonnegative scalar and T is positive. Neither labels nor teacher
    scores enter this direct target. Float64 arithmetic keeps extreme finite
    logits stable and gradients connected to the original student tensors.
    The runner sums chunk rows divided by the GLOBAL sampled-pair count.
    """
    _fit(partition)
    pos = _logits(positive_logits, "positive_logits")
    neg = _logits(negative_logits, "negative_logits")
    if len(pos) != len(neg) or pos.device != neg.device:
        raise ValueError("pair logits must be aligned and on the same device")
    if (isinstance(margin, bool) or not isinstance(margin, Real)
            or not math.isfinite(margin) or margin < 0):
        raise ValueError("margin must be a finite nonnegative scalar")
    temp = _positive_number(temperature, "temperature")
    delta = (neg.to(torch.float64) - pos.to(torch.float64) + float(margin)) / temp
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("temperature-scaled pair differences must be finite")
    rows = temp * F.softplus(delta)
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("pairwise pAUC surrogate became non-finite")
    return rows


def direct_pauc_loss(scores, labels, *, beta=.01, margin=0., temperature=1.,
                     max_pairs_per_chunk=65536, partition="fit"):
    """Exact differentiable finite-pool CVaR surrogate for small-pool checks.

    Return ``{"loss": Tensor, "statistics": dict}``. Selection uses detached
    scores, then all positives are paired with all nonzero-weight negatives.
    Chunking bounds each temporary pair matrix, NOT the total retained autograd
    graph or quadratic work. Use ``sample_global_tail_pairs`` on lab-scale data.
    Selection at ties uses the symmetric valid subgradient described above.
    """
    _fit(partition)
    scores = _logits(scores, "scores")
    if (not isinstance(labels, Tensor) or labels.ndim != 1 or labels.shape != scores.shape
            or labels.device != scores.device):
        raise ValueError("labels must be an aligned tensor vector on the same device")
    chunk_pairs = _positive_integer(max_pairs_per_chunk, "max_pairs_per_chunk")
    tail = global_negative_tail(scores.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                                beta=beta, partition=partition)
    pos = torch.nonzero(labels == 1).flatten()
    neg_np = np.flatnonzero(tail["weights"] > 0)
    neg = torch.as_tensor(neg_np, dtype=torch.int64, device=scores.device)
    weights = torch.as_tensor(tail["weights"][neg_np], dtype=torch.float64, device=scores.device)
    pos_chunk = min(len(pos), max(1, math.isqrt(chunk_pairs)))
    total = scores[0].to(torch.float64) * 0.
    for p_start in range(0, len(pos), pos_chunk):
        p_ids = pos[p_start:p_start + pos_chunk]
        neg_chunk = max(1, chunk_pairs // len(p_ids))
        for n_start in range(0, len(neg), neg_chunk):
            n_ids = neg[n_start:n_start + neg_chunk]
            p_values = scores[p_ids, None].expand(-1, len(n_ids)).reshape(-1)
            n_values = scores[n_ids][None, :].expand(len(p_ids), -1).reshape(-1)
            rows = pairwise_pauc_rows(p_values, n_values, margin=margin, temperature=temperature,
                                     partition=partition).reshape(len(p_ids), len(n_ids))
            total = total + (rows * weights[n_start:n_start + len(n_ids)]).sum() / len(pos)
    return {"loss": total, "statistics": {**tail["statistics"],
            "pair_evaluations": len(pos) * len(neg), "pair_reduction": "exact_positive_mean_tail_weighted",
            "margin": float(margin), "temperature": float(temperature)}}
