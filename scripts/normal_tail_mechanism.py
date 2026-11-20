"""Exact finite-unit calibration-transfer operators; no real-data fitting CLI.

Order statistics enforce empirical integer budgets, not population guarantees.
Raw logit gaps retain the deployed score's units. No learned normalization or
smooth quantile is substituted. Standard hinge/min/max are not novelty claims.
"""
from __future__ import annotations

import math
import torch


def budget(n: int, alpha: float = .01) -> int:
    if n <= 0 or not 0 <= alpha < 1:
        raise ValueError('Nonempty normal unit and 0 <= alpha < 1 required')
    return math.floor(alpha*n)


def boundary(scores: torch.Tensor, alpha: float = .01) -> torch.Tensor:
    if scores.ndim != 1 or not scores.numel() or not bool(torch.isfinite(scores).all()):
        raise ValueError('Need a finite nonempty score vector')
    # 1-based n-floor(alpha*n); not interpolated quantile(.99).
    return torch.sort(scores, stable=True).values[-(budget(len(scores), alpha)+1)]


def episode_terms(past_normal, next_normal, attacks, alpha=.01, margin=1.):
    """One training-only Q -> E relation and a shared positive margin.

    attacks contains the same F positives for every treatment. This is an offline
    training margin, not a claim that attack labels or normal labels arrive online.
    The hard count uses strict > and is diagnostic; no threshold search exists.
    """
    if attacks.ndim != 1 or not attacks.numel() or not bool(torch.isfinite(attacks).all()):
        raise ValueError('Need finite training positives; collapse must not qualify')
    if margin <= 0:
        raise ValueError('Positive separation margin required')
    tq, te = boundary(past_normal, alpha), boundary(next_normal, alpha)
    gap = te-tq
    attack_gap = tq+margin-attacks.min()
    return {'past_boundary': tq, 'next_boundary': te,
            'normal_gap': gap, 'attack_gap': attack_gap,
            'normal_penalty': gap.relu(), 'attack_penalty': attack_gap.relu(),
            'normal_count': len(next_normal), 'normal_budget': budget(len(next_normal), alpha),
            'normal_fp': int((next_normal.detach() > tq.detach()).sum())}


def transfer_penalty(episodes, attacks, alpha=.01, margin=1.):
    if not episodes:
        raise ValueError('Need actual or explicitly shuffled calibration/evaluation pairs')
    terms = [episode_terms(q, e, attacks, alpha, margin) for q, e in episodes]
    # All episodes remain in diagnostics; a pooled average cannot hide one failure.
    normal = torch.stack([t['normal_penalty'] for t in terms]).max()
    attack = torch.stack([t['attack_penalty'] for t in terms]).max()
    return {'penalty': normal+attack, 'normal_penalty': normal,
            'attack_penalty': attack, 'episodes': terms}
