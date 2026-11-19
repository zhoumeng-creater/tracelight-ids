"""Research prototype of a supervised nonnegative whole-reference matcher.

This module has no dataset loader, fitting loop, calibration, or real-data CLI.
All caller-supplied stages must include a final presence coordinate equal to 1.
The null stage has all coordinates 0. Directions are matching constraints only.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Sequence:
    values: torch.Tensor
    directions: tuple[int, ...]

    def __post_init__(self):
        if self.values.ndim != 2 or len(self.values) != len(self.directions) or not len(self.values):
            raise ValueError('A sequence needs nonempty stage rows and matching directions')
        if not torch.all(self.values[:, -1] == 1):
            raise ValueError('Real stages must have presence=1 in their last coordinate')
        if any(d not in (0, 1) for d in self.directions):
            raise ValueError('Directions must use the connection originator/responder convention')


class RelativeCost(nn.Module):
    """Nonnegative, zero at identity, potentially asymmetric/context-dependent.

    c(q,r) = sum_k (q_k-r_k)^2 softplus(h([r,q-r])_k).
    Signed differences in h are deliberate: +/- deviations need not mean the same.
    """
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(2*features, hidden), nn.Tanh(), nn.Linear(hidden, features))

    def forward(self, query, reference):
        query, reference = torch.broadcast_tensors(query, reference)
        delta = query-reference
        weight = F.softplus(self.network(torch.cat((reference, delta), dim=-1)))
        return (delta.square()*weight).sum(dim=-1)


def bottleneck_edit(pair, insertion, deletion, compatible, require_match=True):
    """Exact hard min/max edit DP, with no free gaps and no temperature.

    All query and reference stages are consumed. An extra state records whether
    at least one compatible match occurred. That condition alone does NOT prevent
    matching a shared first stage and putting every other stage into a gap.
    """
    n,m=pair.shape
    inf=pair.new_tensor(float('inf')); zero=pair.new_zeros(())
    dp=[[[inf for _ in range(2)] for _ in range(m+1)] for _ in range(n+1)]
    previous={}
    dp[0][0][0]=zero
    for i in range(n+1):
        for j in range(m+1):
            if not (i or j):continue
            for state in (0,1):
                choices=[]
                # Stable tie preference: match, query insertion, reference deletion.
                if state and i and j and compatible[i-1,j-1]:
                    for oldstate in (0,1):
                        choices.append((torch.maximum(dp[i-1][j-1][oldstate],pair[i-1,j-1]),
                                        (i-1,j-1,oldstate,'match',pair[i-1,j-1])))
                if i:
                    choices.append((torch.maximum(dp[i-1][j][state],insertion[i-1]),
                                    (i-1,j,state,'query_gap',insertion[i-1])))
                if j:
                    choices.append((torch.maximum(dp[i][j-1][state],deletion[j-1]),
                                    (i,j-1,state,'reference_gap',deletion[j-1])))
                if choices:
                    values=torch.stack([x[0] for x in choices])
                    best=values.min(dim=0)
                    dp[i][j][state]=best.values
                    previous[i,j,state]=choices[int(best.indices)][1]
    state=1 if require_match else int(torch.stack(dp[n][m]).argmin())
    score=dp[n][m][state]
    trace=[]
    if torch.isfinite(score):
        i,j=n,m
        while i or j:
            ni,nj,ns,op,cost=previous[i,j,state]
            trace.append({'operation':op,'query_index':i-1 if i>ni else None,
                          'reference_index':j-1 if j>nj else None,'cost':float(cost.detach())})
            i,j,state=ni,nj,ns
        trace.reverse()
    return score,trace


class WholeReferenceModel(nn.Module):
    """A proposal, not a fitted detector. Normal references are explicit inputs."""
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.cost=RelativeCost(features,hidden)
        self.log_scale=nn.Parameter(torch.tensor(0.0))
        self.intercept=nn.Parameter(torch.tensor(0.0))

    def forward(self, query: Sequence, references: list[Sequence]):
        if not references:
            raise ValueError('No reference support; do not turn unavailability into an attack score')
        scores=[];traces=[];independent=[]
        zero=torch.zeros_like(query.values[0])
        insertion=self.cost(query.values,zero)
        for ref in references:
            pair=self.cost(query.values[:,None,:],ref.values[None,:,:])
            deletion=self.cost(zero,ref.values)
            compatible=torch.tensor([[a==b for b in ref.directions] for a in query.directions],device=pair.device)
            score,trace=bottleneck_edit(pair,insertion,deletion,compatible)
            scores.append(score);traces.append(trace)
            independent.append(pair.masked_fill(~compatible,float('inf')).min(dim=1).values)
        stacked=torch.stack(scores)
        winner=stacked.min(dim=0)
        if not torch.isfinite(winner.values):
            raise ValueError('No admissible full reference path')
        # Same local costs and normal bank; relaxed control may switch reference
        # and stage arbitrarily, or use the same query-gap option. It drops the
        # whole-path constraints, so it is a lower bound, not an equivalent model.
        relaxed=torch.stack(independent+[insertion]).min(dim=0).values.max()
        score=winner.values
        return {'score':score,'logit':F.softplus(self.log_scale)*score-self.intercept,
                'relaxed_score':relaxed,'reference_index':int(winner.indices),
                'path':traces[int(winner.indices)]}


def diagonal_quantifier_controls(costs):
    """Fixed equal-length alignments isolate only the reference quantifier.

    costs[reference,stage]. Complete-reference min-max >= per-stage max-min.
    This is an algebra check; it is not the full variable-length comparison.
    """
    return costs.max(dim=1).values.min(),costs.min(dim=0).values.max()
