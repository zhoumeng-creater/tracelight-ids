"""Synthetic algebra/gradient checks only. No classifier or dataset is loaded."""
import argparse
from itertools import product
import json
from pathlib import Path
import torch
from normal_tail_mechanism import boundary, budget, episode_terms, transfer_penalty


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    dtype=torch.float64
    checked=0
    # Include repeated scores and zero/positive integer budgets.
    for n in range(1,6):
        for values in product((-1.,0.,1.),repeat=n):
            e=torch.tensor(values,dtype=dtype)
            for alpha in (.01,.2,.5):
                te=boundary(e,alpha)
                for threshold in (-2.,-1.,-.5,0.,.5,1.,2.):
                    assert (int((e>threshold).sum())<=budget(n,alpha)) == bool(te<=threshold)
                    checked+=1
    q=torch.arange(50,dtype=dtype)
    assert boundary(q)==49 and torch.quantile(q,.99)<49
    q=torch.tensor([-2.,-1.,0.],dtype=dtype,requires_grad=True)
    e=torch.tensor([-1.5,.25,1.],dtype=dtype,requires_grad=True)
    a=torch.tensor([4.,5.],dtype=dtype,requires_grad=True)
    term=episode_terms(q,e,a);loss=term['normal_penalty']+term['attack_penalty'];loss.backward()
    assert q.grad.tolist()==[0.,0.,-1.] and e.grad.tolist()==[0.,0.,1.]
    # Correct gradient of common translation is zero; detached calibration is wrong.
    shift=torch.tensor(.25,dtype=dtype,requires_grad=True)
    genuine=boundary(e.detach()+shift)-boundary(q.detach()+shift)
    genuine.backward();real_grad=float(shift.grad)
    shift2=torch.tensor(.25,dtype=dtype,requires_grad=True)
    proxy=boundary(e.detach()+shift2)-boundary(q.detach()+shift2).detach()
    proxy.backward();proxy_grad=float(shift2.grad)
    assert real_grad==0 and proxy_grad==1
    # A tail-focused correction can move a future normal component relative to Q.
    w=torch.tensor(.4,dtype=dtype,requires_grad=True)
    fn=lambda z: episode_terms(torch.tensor([-2.,-1.,0.],dtype=dtype),
                               torch.stack((z-2,z-1,z)),torch.tensor([3.,4.],dtype=dtype))['normal_penalty']
    fn(w).backward();eps=1e-6
    numeric=float((fn(w.detach()+eps)-fn(w.detach()-eps))/(2*eps))
    assert abs(float(w.grad)-numeric)<1e-8
    # Raising calibration alone can remove normal gap while destroying attack margin.
    old=episode_terms(torch.tensor([0.],dtype=dtype),torch.tensor([1.],dtype=dtype),torch.tensor([1.5],dtype=dtype))
    high=episode_terms(torch.tensor([1.],dtype=dtype),torch.tensor([1.],dtype=dtype),torch.tensor([1.5],dtype=dtype))
    assert high['normal_penalty']==0 and high['attack_penalty']>old['attack_penalty']
    collapsed=episode_terms(torch.zeros(3,dtype=dtype),torch.zeros(4,dtype=dtype),torch.zeros(2,dtype=dtype))
    assert collapsed['normal_penalty']==0 and collapsed['attack_penalty']==1
    # Raw-gap contraction is not an actual risk improvement.
    scales=[]
    for c in (1.,.1,.01):
        t=episode_terms(torch.tensor([0.],dtype=dtype)*c,torch.tensor([1.],dtype=dtype)*c,torch.tensor([4.],dtype=dtype)*c)
        scales.append({'scale':c,'gap':float(t['normal_gap']),'fp':t['normal_fp'],'attack_penalty':float(t['attack_penalty'])})
    assert all(t['fp']==1 for t in scales)
    # Current-range normalization has a zero-gradient failure at positive violation.
    x=torch.tensor([0.,1.],dtype=dtype,requires_grad=True)
    ratio=(x[1]-x[0])/(x.max()-x.min());ratio.backward();assert not x.grad.any()
    report={'input':'synthetic_only','real_model_calls':0,'optimizer_steps':0,
        'integer_count_equivalence_checks':checked,'n50_integer_boundary_vs_interpolation':[49.,float(torch.quantile(torch.arange(50,dtype=dtype),.99))],
        'normal_component_gradient':{'autograd':float(w.grad),'finite_difference':numeric},
        'common_translation_gradient':real_grad,'detached_threshold_wrong_translation_gradient':proxy_grad,
        'raising_threshold_attack_tradeoff_checked':True,'constant_collapse_attack_penalty':1.,
        'scale_contraction_not_count_improvement':scales,'current_range_normalization_rejected_zero_gradient':True,
        'empirical_identity_not_population_guarantee':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
