"""Ordinary no-gap bottleneck matching after lossless direction-run pairing.

This is a structural repair/control, not an original detection algorithm.
No model, fit, threshold or real-data classification is provided here.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PairedRuns:
    first_direction: int
    pairs: tuple[tuple[int, int], ...]
    tail: int | None

    @property
    def boundary_type(self):
        return self.first_direction, self.tail is not None


def pair_runs(runs):
    if not runs:
        raise ValueError('No observed data runs; representation unavailable')
    if any(d not in (0,1) or mass<=0 or int(mass)!=mass for d,mass in runs):
        raise ValueError('Runs need a direction and positive unique-byte mass')
    if any(runs[i][0]==runs[i-1][0] for i in range(1,len(runs))):
        raise ValueError('Expected maximal alternating direction runs')
    paired=tuple((int(runs[i][1]),int(runs[i+1][1])) for i in range(0,len(runs)-1,2))
    return PairedRuns(runs[0][0],paired,int(runs[-1][1]) if len(runs)%2 else None)


def unpack(paired):
    d=paired.first_direction
    runs=[]
    for a,b in paired.pairs:
        runs.extend([[d,a],[1-d,b]])
    if paired.tail is not None:
        runs.append([d,paired.tail])
    return runs


def full_bottleneck(pair_costs):
    """Standard gap-free bottleneck DTW/Fréchet recurrence on full pairs.

    Horizontal/vertical steps repeat a real pair comparison. They do not delete
    an observation or compare it with a learned null token. A path covers every
    row and column, but may ignore repetition multiplicity when repeated pair
    vectors have zero cost. Inputs here are abstract costs, not traffic scores.
    """
    n=len(pair_costs)
    m=len(pair_costs[0]) if n else 0
    if not n or not m or any(len(row)!=m for row in pair_costs):
        raise ValueError('Need a nonempty rectangular cost matrix')
    if any(not math.isfinite(float(c)) or c<0 for row in pair_costs for c in row):
        raise ValueError('Costs must be finite and nonnegative')
    values=[[math.inf]*m for _ in range(n)]
    previous={}
    values[0][0]=float(pair_costs[0][0])
    for i in range(n):
        for j in range(m):
            if i==j==0:continue
            predecessors=[]
            if i and j:predecessors.append((values[i-1][j-1],i-1,j-1))
            if i:predecessors.append((values[i-1][j],i-1,j))
            if j:predecessors.append((values[i][j-1],i,j-1))
            value,pi,pj=min(predecessors,key=lambda x:x[0])
            values[i][j]=max(float(pair_costs[i][j]),value)
            previous[i,j]=(pi,pj)
    path=[]
    i,j=n-1,m-1
    while True:
        path.append((i,j))
        if i==j==0:break
        i,j=previous[i,j]
    path.reverse()
    return values[-1][-1],path


def reference_cost(query: PairedRuns,reference: PairedRuns,pair_costs,tail_cost=None):
    """Combine complete-pair comparisons and an explicitly observed real tail.

    A boundary mismatch is unsupported, never an artificial attack score.
    A caller must provide costs for both coordinate values of each real pair.
    """
    if query.boundary_type!=reference.boundary_type:
        raise ValueError('Different boundary types; no full no-gap explanation')
    if bool(query.pairs)!=bool(reference.pairs):
        raise ValueError('A complete pair cannot be matched to no observed pair')
    if query.pairs:
        if len(pair_costs)!=len(query.pairs) or any(len(row)!=len(reference.pairs) for row in pair_costs):
            raise ValueError('Cost matrix must include every real pair')
        score,path=full_bottleneck(pair_costs)
    else:
        score,path=0.,[]
    if query.tail is not None:
        if tail_cost is None or not math.isfinite(tail_cost) or tail_cost<0:
            raise ValueError('The real tail requires an explicit nonnegative cost')
        score=max(score,float(tail_cost))
    elif tail_cost is not None:
        raise ValueError('There is no observed tail to score')
    return score,path
