"""Synthetic mathematics/autograd checks only; never load real traffic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from reference_model import Sequence,WholeReferenceModel,bottleneck_edit,diagonal_quantifier_controls


def brute_paths(pair,ins,dele,allowed,i=0,j=0,matched=False,score=0.0):
    n,m=pair.shape
    if i==n and j==m:
        return [score] if matched else []
    values=[]
    if i<n and j<m and allowed[i,j]:
        values+=brute_paths(pair,ins,dele,allowed,i+1,j+1,True,max(score,float(pair[i,j])))
    if i<n:values+=brute_paths(pair,ins,dele,allowed,i+1,j,matched,max(score,float(ins[i])))
    if j<m:values+=brute_paths(pair,ins,dele,allowed,i,j+1,matched,max(score,float(dele[j])))
    return values


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(20261119)
    model=WholeReferenceModel(3,8).double()
    normal=Sequence(torch.tensor([[1.,.2,1.],[2.,.8,1.]],dtype=torch.float64),(0,1))
    same=model(normal,[normal])
    assert same['score']==0 and same['relaxed_score']==0
    samples=torch.randn(30,3,dtype=torch.float64)
    nonnegative=model.cost(samples,torch.randn_like(samples))
    assert (nonnegative>=0).all()
    assert (model.cost(samples,samples)==0).all()
    whole,independent=diagonal_quantifier_controls(torch.tensor([[0.,4.],[4.,0.]]))
    assert whole==4 and independent==0
    # A signed identity value is not an operating-point guarantee.
    signed_self=-1.;signed_calibration_threshold=-2.
    assert signed_self>signed_calibration_threshold
    # Exhaustive small path enumeration checks the actual variable-length DP.
    dp_cases=0
    for n in range(1,4):
        for m in range(1,4):
            pair=torch.rand(n,m,dtype=torch.float64)
            ins=torch.rand(n,dtype=torch.float64);dele=torch.rand(m,dtype=torch.float64)
            allowed=torch.tensor([[i%2==j%2 for j in range(m)] for i in range(n)])
            exact=min(brute_paths(pair,ins,dele,allowed))
            result,_=bottleneck_edit(pair,ins,dele,allowed)
            assert abs(float(result)-exact)<1e-12
            dp_cases+=1
    # At least one match can still leave the rest to cheap gaps.
    pair=torch.tensor([[0.,5.],[5.,5.]],dtype=torch.float64)
    gap=torch.tensor([.01,.01],dtype=torch.float64)
    score,path=bottleneck_edit(pair,gap,gap,torch.eye(2,dtype=torch.bool))
    assert abs(float(score)-.01)<1e-12 and sum(e['operation']=='match' for e in path)==1
    query=Sequence(torch.tensor([[1.2,.25,1.],[3.1,.75,1.]],dtype=torch.float64),(0,1))
    result=model(query,[normal])
    assert result['score']>=result['relaxed_score']
    loss=torch.nn.functional.binary_cross_entropy_with_logits(result['logit'],torch.ones((),dtype=torch.float64))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # A unique active path has the expected local finite-difference derivative.
    parameter=model.cost.network[-1].bias
    analytic=float(parameter.grad[0]);old=float(parameter[0]);step=1e-6
    with torch.no_grad():
        parameter[0]=old+step
        plus=float(torch.nn.functional.binary_cross_entropy_with_logits(model(query,[normal])['logit'],torch.ones((),dtype=torch.float64)))
        parameter[0]=old-step
        minus=float(torch.nn.functional.binary_cross_entropy_with_logits(model(query,[normal])['logit'],torch.ones((),dtype=torch.float64)))
        parameter[0]=old
    numerical=(plus-minus)/(2*step)
    assert abs(analytic-numerical)<1e-6,(analytic,numerical)
    report={'input_kind':'synthetic_only','real_model_calls':0,'optimizer_steps':0,
            'exact_normal_anchor':float(same['score']),'nonnegative_local_costs':True,
            'diagonal_chimera_counterexample':{'whole_reference':float(whole),'independent_stages':float(independent)},
            'signed_anchor_false_positive_counterexample':{'self_score':signed_self,'calibration_threshold':signed_calibration_threshold},
            'small_edit_dp_bruteforce_cases':dp_cases,
            'one_match_does_not_prevent_gap_shortcut':{'score':float(score),'path':path},
            'autograd_finite_difference':{'analytic':analytic,'numerical':numerical},
            'prototype_parameters':sum(p.numel() for p in model.parameters()),
            'architecture_status':'synthetic prototype dimensions, not a registered experiment configuration'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(report))


if __name__=='__main__':main()
