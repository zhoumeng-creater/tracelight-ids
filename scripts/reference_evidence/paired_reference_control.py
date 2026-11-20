"""Differentiable ordinary whole-reference control without null-gap edges.

This replaces only the path structure in a separate control. It does not modify
the old WholeReferenceModel or establish an innovative detection method. There
is no real-data loader, training loop, threshold selection or scoring CLI.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from scripts.stage_design.reference_model import RelativeCost, Sequence


@dataclass
class PairedStages:
    first_direction: int
    pairs: torch.Tensor
    tail: torch.Tensor | None

    @property
    def boundary_type(self):
        return self.first_direction, self.tail is not None, len(self.pairs) > 0


def pair_stages(sequence: Sequence) -> PairedStages:
    """Tensor view of the lossless pair_runs/unpack rule; retain gradients.

    Values may contain transformed stage features, so the integer-byte validator
    in paired_no_gap.pair_runs is not applied to these feature tensors. The same
    alternating directions, adjacent indices and unchanged odd tail are used.
    """
    if any(a == b for a, b in zip(sequence.directions, sequence.directions[1:])):
        raise ValueError('Expected maximal alternating direction runs')
    n = len(sequence.values) // 2
    pairs = sequence.values[:2*n].reshape(n, 2, sequence.values.shape[-1])
    tail = sequence.values[-1] if len(sequence.values) % 2 else None
    return PairedStages(sequence.directions[0], pairs, tail)


def full_bottleneck_tensor(costs: torch.Tensor):
    """Exact full-path DTW/Fréchet min-max with a piecewise autograd graph.

    No null costs or skipped rows/columns exist. Detached indices select only
    the diagnostic path; the returned score remains a torch scalar connected
    to local costs. Stable predecessor tie order is diagonal, query, reference.
    """
    if costs.ndim != 2 or not all(costs.shape):
        raise ValueError('Need a nonempty two-dimensional cost matrix')
    if not bool(torch.isfinite(costs).all()) or bool((costs < 0).any()):
        raise ValueError('Costs must be finite and nonnegative')
    n, m = costs.shape
    values = [[None for _ in range(m)] for _ in range(n)]
    previous = {}
    values[0][0] = costs[0, 0]
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            predecessors = []
            if i and j:
                predecessors.append((i-1, j-1))
            if i:
                predecessors.append((i-1, j))
            if j:
                predecessors.append((i, j-1))
            best = torch.stack([values[a][b] for a, b in predecessors]).min(dim=0)
            values[i][j] = torch.maximum(costs[i, j], best.values)
            previous[i, j] = predecessors[int(best.indices.detach())]
    path = []
    i, j = n-1, m-1
    while True:
        path.append((i, j))
        if i == j == 0:
            break
        i, j = previous[i, j]
    path.reverse()
    return values[-1][-1], path


class PairedReferenceControl(nn.Module):
    """Same RelativeCost and alpha/beta as the old model, ordinary no-gap DP.

    The caller supplies the already permitted normal-reference bank and handles
    physical-identity exclusions. Unsupported boundary types produce no score.
    Exact identity is zero, but repeated identical pairs can warp at zero cost.
    """
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.cost = RelativeCost(features, hidden)
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.intercept = nn.Parameter(torch.tensor(0.0))

    def forward(self, query: Sequence, references: list[Sequence]):
        q = pair_stages(query)
        scores = []
        traces = []
        indices = []
        for index, reference in enumerate(references):
            r = pair_stages(reference)
            if q.boundary_type != r.boundary_type:
                continue
            trace = []
            if len(q.pairs):
                # Each cell compares two corresponding real directions. Both
                # stage costs enter through max; weights are shared unchanged.
                stage_costs = self.cost(q.pairs[:, None, :, :], r.pairs[None, :, :, :])
                pair_costs = stage_costs.max(dim=-1).values
                score, path = full_bottleneck_tensor(pair_costs)
                trace = [{'operation': 'pair_match',
                          'query_stage_indices': (2*i, 2*i+1),
                          'reference_stage_indices': (2*j, 2*j+1)} for i, j in path]
            else:
                score = query.values.new_zeros(())
            if q.tail is not None:
                tail_cost = self.cost(q.tail, r.tail)
                score = torch.maximum(score, tail_cost)
                trace.append({'operation': 'tail_match',
                              'query_stage_indices': (len(query.values)-1,),
                              'reference_stage_indices': (len(reference.values)-1,)})
            scores.append(score)
            traces.append(trace)
            indices.append(index)
        if not scores:
            raise ValueError('No compatible full reference; representation is unsupported')
        winner = torch.stack(scores).min(dim=0)
        selected = int(winner.indices.detach())
        score = winner.values
        return {'score': score,
                'logit': F.softplus(self.log_scale)*score-self.intercept,
                'reference_index': indices[selected],
                'path': traces[selected]}
