"""Synthetic representation checks only: no datasets, models, labels or fitting."""
from __future__ import annotations
import json, math, random
from pathlib import Path

class OrderContrast:
    """Nonnegative increments; strict availability batches; packed upper triangles."""
    def __init__(self, dim: int, rate: float = 0.0):
        self.dim, self.rate, self.time = dim, rate, None
        self.p = [0.0] * dim
        self.pairs = [(i, j) for i in range(dim) for j in range(i+1, dim)]
        self.a = [0.0] * len(self.pairs)
        self.s = [0.0] * len(self.pairs)

    def advance(self, time: float):
        if self.time is not None:
            if time < self.time: raise ValueError('availability must be monotone')
            decay = math.exp(-self.rate * (time-self.time))
            self.p = [v*decay for v in self.p]
            self.a = [v*decay*decay for v in self.a]
            self.s = [v*decay*decay for v in self.s]
        self.time = time

    def read(self, query):
        if len(query) != self.dim or any(v < 0 for v in query):
            raise ValueError('nonnegative query required')
        total = sum(query)
        g = [v/total if total else 0.0 for v in query]
        out = [0.0]*self.dim
        pair_support = [0.0]*self.dim
        ratio = []
        for n, (i,j) in enumerate(self.pairs):
            r = self.a[n]/self.s[n] if self.s[n] > 0.0 else 0.0
            ratio.append(r)
            out[i] += r*g[j]
            out[j] -= r*g[i]
            pair_support[i] += self.s[n]*g[j]
            pair_support[j] += self.s[n]*g[i]
        return {'context':out, 'ratio':ratio,
                'log_pair_support':[math.log1p(v) for v in pair_support],
                'log_first_order_support':[math.log1p(v) for v in self.p],
                'unsupported_pair_count':sum(v == 0 for v in self.s)}

    def batch(self, time, observations, queries=()):
        self.advance(time)
        # All queries see the same strict prefix, before any batch observation.
        reads = [self.read(q) for q in queries]
        if any(len(x)!=self.dim or any(v<0 for v in x) for x in observations):
            raise ValueError('nonnegative observations required')
        u = [sum(x[d] for x in observations) for d in range(self.dim)]
        for n,(i,j) in enumerate(self.pairs):
            self.a[n] += self.p[i]*u[j] - self.p[j]*u[i]
            self.s[n] += self.p[i]*u[j] + self.p[j]*u[i]
        self.p = [p+x for p,x in zip(self.p,u)]
        return reads

def build(events, rate=0.0, dim=3):
    m=OrderContrast(dim,rate)
    for time, xs in events: m.batch(time,xs)
    return m
def seq(xs, rate=0.0):
    return build([(float(t),[x]) for t,x in enumerate(xs)],rate, len(xs[0]))
def close(a,b):
    return len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10) for x,y in zip(a,b))
def means(xs): return [sum(row[d] for row in xs)/len(xs) for d in range(len(xs[0]))]

def checks():
    report={'scope':'synthetic_representation_only','real_model_calls':0,'real_data_reads':0,'tests':{}}
    e1,e2,e3=[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]
    q=[.5,.5,0.]
    left=[e1,e2]+[e3]*7; right=[e2,e1]+[e3]*7
    ml,mr=seq(left),seq(right)
    assert means(left)==means(right) and left[-7:]==right[-7:]
    assert ml.p==mr.p and not close(ml.read(q)['context'],mr.read(q)['context'])
    report['tests']['same_query_mean_endpoint_last7']={'left':ml.read(q),'right':mr.read(q)}

    # Same unweighted mean AND the same exponentially weighted first moment.
    delta=[2.,-3.,1.]
    x=[[1+.2*d,1.,0.] for d in delta]+[e3]*4
    y=[[1-.2*d,1.,0.] for d in delta]+[e3]*4
    mx,my=seq(x,math.log(2)),seq(y,math.log(2))
    assert close(means(x),means(y)) and close(mx.p,my.p)
    assert not close(mx.read(q)['context'],my.read(q)['context'])
    report['tests']['same_mean_and_ema']={'ema':mx.p,'left':mx.read(q),'right':my.read(q)}

    # Closed-form all-pairs calculation independently verifies recurrence/decay/ties.
    rng=random.Random(912)
    events=[(0.,[[rng.random() for _ in range(3)]]),
            (1.,[[rng.random() for _ in range(3)] for _ in range(2)]),
            (3.,[[rng.random() for _ in range(3)]])]
    m=build(events,.3); m.advance(5.)
    rows=[(t,x) for t,xs in events for x in xs]
    direct_a=[];direct_s=[]
    for d,e in m.pairs:
        aa=ss=0.
        for ti,xi in rows:
            for tj,xj in rows:
                if ti>=tj: continue
                w=math.exp(-.3*(10-ti-tj))
                aa+=w*(xi[d]*xj[e]-xi[e]*xj[d])
                ss+=w*(xi[d]*xj[e]+xi[e]*xj[d])
        direct_a.append(aa);direct_s.append(ss)
    assert close(m.a,direct_a) and close(m.s,direct_s)
    assert all(abs(a)<=s+1e-12 for a,s in zip(m.a,m.s))
    report['tests']['exact_strict_pair_expansion']={'A':m.a,'S':m.s,'passed':True}

    scaled=seq([[v*7 for v in row] for row in left])
    assert close(scaled.read(q)['context'],ml.read(q)['context'])
    assert close(scaled.a,[v*49 for v in ml.a])
    report['tests']['global_positive_amplitude_invariance']={'factor':7,'passed':True}

    stale=seq([e1,e2],.1)
    before=stale.read(q)
    stale.advance(101.)
    after=stale.read(q)
    assert close(before['context'],after['context'])
    assert sum(after['log_pair_support'])<sum(before['log_pair_support'])*1e-7
    report['tests']['stale_history_counterexample']={'before':before,'after_100_idle_seconds':after,
        'ratio_does_not_forget':True,'support_channels_decay':True}

    # Deliberate failure: duplicating real multi-channel events is NOT invariant.
    base=seq([e1,[1.,1.,0.]])
    copies=seq([e1,e1,[1.,1.,0.],[1.,1.,0.]])
    assert not close(base.read(q)['context'],copies.read(q)['context'])
    report['tests']['event_duplication_counterexample']={'base':base.read(q),'duplicated':copies.read(q)}

    # Same piecewise-linear geometric path, changed event segmentation.
    unsplit=seq([[1.,1.,0.],e1])
    split=seq([[.5,.5,0.],[.5,.5,0.],e1])
    assert close(unsplit.p,split.p) and close(unsplit.a,split.a)
    assert not close(unsplit.s,split.s)
    assert close(unsplit.a,[-1.,0.,0.]) and close(unsplit.s,[1.,0.,0.])
    assert close(split.s,[1.5,0.,0.])
    report['tests']['same_geometric_signature_different_segmentation']={
        'unsplit':unsplit.read(q),'split':split.read(q),
        'A_equal':True,'P_equal':True,'S_12':[unsplit.s[0],split.s[0]],
        'not_a_higher_signature_order_gain':True}

    cancel1=seq([e1,e2,e2,e1]+[e3]*7)
    cancel2=seq([e2,e1,e1,e2]+[e3]*7)
    assert close(cancel1.a,cancel2.a) and close(cancel1.s,cancel2.s)
    assert close(cancel1.read(q)['context'][:2],[0.,0.])
    report['tests']['order_cancellation_counterexample']={'first':cancel1.read(q),'second':cancel2.read(q)}

    pure=seq([e1]*100)
    assert close(pure.read(q)['context'],[0.,0.,0.])
    report['tests']['pure_repeated_c2_is_invisible']={'read':pure.read(q)}

    # Same observed trajectory can describe legitimate and malicious work.
    normal=seq(left);attack=seq(left)
    assert normal.read(q)==attack.read(q)
    report['tests']['normal_attack_identical_observation_counterexample']={'cannot_distinguish':True}

    more_background=seq([e1,e2]+[e3]*1000)
    overlap_background=seq([e1,e2]+[[1.,1.,0.]]*1000)
    assert close(more_background.read(q)['context'],ml.read(q)['context'])
    assert abs(overlap_background.read(q)['context'][0])<1e-5
    report['tests']['background_scope']={'disjoint':more_background.read(q),'overlapping':overlap_background.read(q)}
    assert not close(ml.read(e3)['context'],[0.,0.,0.])
    assert math.isclose(sum(v*g for v,g in zip(ml.read(q)['context'],q)),0.)
    report['tests']['keep_vector_not_self_bilinear']={'one_hot_read':ml.read(e3),
        'q_dot_Rq':sum(v*g for v,g in zip(ml.read(q)['context'],q)),
        'self_bilinear_is_identically_zero':True}

    tied=OrderContrast(3)
    reads=tied.batch(10.,[e1,e2],[q,q])
    assert reads[0]==reads[1] and close(tied.a,[0.,0.,0.])
    report['tests']['same_batch_no_order_or_cross_read']={'before':reads,'after_A':tied.a}

    available=[(2.,e3),(20.,e1)]
    prefix=seq([e3])
    queried=build([(t,[v]) for t,v in available if t<10.])
    assert queried.read(q)==prefix.read(q)
    report['tests']['late_export_not_recovered']={'query_at':10,'excluded_export_at':20,'read':queried.read(q)}
    report['tests']['state_budget']={'r8_float32_values':8+2*(8*7//2),
        'packed_numeric_bytes_plus_float64_clock':4*(8+2*(8*7//2))+8,
        'python_object_overhead_included':False,
        'shared_host_index_and_extractor_cost_must_be_added':True}
    report['test_count']=len(report['tests'])
    return report

if __name__=='__main__':
    report=checks()
    path=Path(__file__).with_name('temporal_synthetic_results.json')
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'passed':report['test_count'],'real_model_calls':0,'output':str(path)},ensure_ascii=False))
