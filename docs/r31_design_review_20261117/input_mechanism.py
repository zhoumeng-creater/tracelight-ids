"""Audit frozen inputs, PLE mathematics and cached scores. No torch/model operations."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ['duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes']
COUNTS = NUM[1:]
ID = ['source_id','row_id']
SEEDS = (17,29,43)
MAJOR_PATTERNS = [(12.,8.,1371.,3872.),(12.,8.,1371.,3873.),(12.,8.,1371.,3874.),
                  (11.,8.,1285.,4725.),(11.,8.,1285.,5205.)]

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(v):
    a=np.asarray(v,dtype=np.float64)
    if not len(a): return {'n':0}
    return {'n':len(a),**dict(zip(['min','q10','q25','median','q75','q90','max'],map(float,np.quantile(a,[0,.1,.25,.5,.75,.9,1]))))}
def aligned(frame,ids):
    index=pd.MultiIndex.from_tuples([tuple(x) for x in ids],names=ID)
    return frame.set_index(ID).loc[index].reset_index()
def transform(frame,base):
    raw=frame[NUM].to_numpy(dtype=np.float64)
    missing=np.isnan(raw)|(raw==-1)
    filled=np.where(missing,np.array(base['median']),raw)
    numeric=((np.log1p(filled)-np.array(base['mean']))/np.array(base['scale'])).astype(np.float32)
    assert np.isfinite(numeric).all()
    protocol=np.array([dict(zip([1,6,17,58,132],range(5))).get(int(x),5) for x in frame.ip_protocol],dtype=np.int64)
    return numeric,missing,protocol
def ple_basis(numeric,bins):
    # Pure NumPy transcription of the parameter-free author encoding.
    # Not the learned embedding/network; returned raw numeric skip branch is also retained.
    maxn=max(len(b)-1 for b in bins)
    pieces=[]
    for j,b in enumerate(bins):
        b=np.asarray(b,dtype=np.float32);width=np.diff(b)
        w=np.float32(1)/width;bias=-b[:-1]/width
        x=numeric[:,j,None]*w[None,:]+bias[None,:]
        if len(width)>1:
            x[:,0]=np.minimum(x[:,0],1)
            x[:,1:-1]=np.clip(x[:,1:-1],0,1)
            x[:,-1]=np.maximum(x[:,-1],0)
        pieces.append(x)
    return np.column_stack(pieces)
def conflicts(matrix,labels):
    keys=[row.tobytes() for row in np.ascontiguousarray(matrix)]
    groups={}
    for i,k in enumerate(keys): groups.setdefault(k,[]).append(i)
    mixed=[v for v in groups.values() if len(set(labels[v]))>1]
    return {'unique':len(groups),'conflicting_groups':len(mixed),'conflicting_rows':sum(map(len,mixed)),
            'conflicting_positive_rows':sum(int(labels[v].sum()) for v in mixed)}
def pattern_groups(frame):
    return {tuple(float(x) for x in k):g for k,g in frame.groupby(COUNTS,sort=True)}
def support_for(target,pool):
    groups=pattern_groups(pool)
    rows=[]
    for _,r in target.iterrows():
        key=tuple(float(r[c]) for c in COUNTS);g=groups.get(key)
        item={'source_id':r.source_id,'row_id':int(r.row_id),'duration':float(r.duration_seconds),'score':float(r.score)}
        if g is None:
            item['same_count_rows']=0
        else:
            dur=g.duration_seconds.to_numpy();value=float(r.duration_seconds)
            nearest=int(np.argmin(np.abs(np.log1p(dur)-np.log1p(value))))
            nr=g.iloc[nearest]
            item.update({'same_count_rows':len(g),'same_count_duration':stats(dur),
                'inside_same_count_range':bool(dur.min()<=value<=dur.max()),
                'below_same_count_range':bool(value<dur.min()),'above_same_count_range':bool(value>dur.max()),
                'nearest_duration':float(nr.duration_seconds),'nearest_duration_abs_gap':float(abs(value-nr.duration_seconds)),
                'nearest_log_duration_abs_gap':float(abs(np.log1p(value)-np.log1p(nr.duration_seconds))),
                'nearest_source_id':nr.source_id,'nearest_row_id':int(nr.row_id),'nearest_score':float(nr.score),
                'score_minus_nearest':float(r.score-nr.score),
                'nearest_score_abs_gap':float(abs(r.score-nr.score))})
        rows.append(item)
    matched=[x for x in rows if x['same_count_rows']]
    return {'rows':rows,'summary':{'target_rows':len(rows),'matched_rows':len(matched),
        'inside_same_count_duration_range':sum(x['inside_same_count_range'] for x in matched),
        'below_same_count_duration_range':sum(x['below_same_count_range'] for x in matched),
        'above_same_count_duration_range':sum(x['above_same_count_range'] for x in matched),
        'nearest_duration_abs_gap':stats([x['nearest_duration_abs_gap'] for x in matched]),
        'nearest_log_duration_abs_gap':stats([x['nearest_log_duration_abs_gap'] for x in matched]),
        'score_minus_nearest':stats([x['score_minus_nearest'] for x in matched]),
        'nearest_score_abs_gap':stats([x['nearest_score_abs_gap'] for x in matched])}}

def frozen_training_signal(frame,target_patterns):
    result={}
    for label,part in [('all',frame),('A',frame.loc[frame.profile.eq('A')]),('B',frame.loc[frame.profile.eq('B')])]:
        neg=part.loc[part.label.eq(0)].copy();pos=part.loc[part.label.eq(1),'score'].to_numpy()
        negmax=float(neg.score.max());posmin=float(pos.min());gap=posmin-negmax
        tail_n=max(1,int(np.ceil(len(neg)*.01)))
        tail=neg.nlargest(tail_n,'score')
        matched=neg.loc[[tuple(float(r[c]) for c in COUNTS) in target_patterns for _,r in neg.iterrows()]]
        # Algebra on saved ensemble log-odds, not backward or actual per-head gradients.
        diff=tail.score.to_numpy()[:,None]-pos[None,:]
        pair_loss=np.logaddexp(0,diff)
        pair_derivative=np.exp(-np.logaddexp(0,-diff))
        result[label]={'normal_max':negmax,'positive_min':posmin,'min_positive_minus_max_normal':gap,
            'all_pairs_logistic_loss_upper_bound':float(np.logaddexp(0,-gap)),
            'all_pairs_abs_score_derivative_upper_bound':float(np.exp(-np.logaddexp(0,gap))),
            'top_one_percent_normal_rows':tail_n,
            'top_one_percent_normal_profiles':{k:int(v) for k,v in tail.profile.value_counts().items()},
            'top_one_percent_pair_logistic_mean':float(pair_loss.mean()),
            'top_one_percent_pair_abs_score_derivative_mean':float(pair_derivative.mean()),
            'matching_D_FP_count_normal_rows':len(matched),
            'matching_D_FP_count_normal_scores':stats(matched.score),
            'matching_D_FP_count_in_top_one_percent':int(tail.set_index(ID).index.isin(matched.set_index(ID).index).sum()),
            'matching_D_FP_count_pair_derivative_max_bound':float(np.exp(-np.logaddexp(0,posmin-matched.score.max()))) if len(matched) else None}
    return result

def original_alert_composition(calibration,decisions,threshold):
    c=calibration.loc[calibration.label.eq(0)].copy()
    d=decisions.loc[decisions.label.eq(0)&decisions.source_id.str.startswith('TQH-D_')].copy()
    c['alert']=c.score.gt(threshold)  # Replay only the already registered C rule.
    assert int(c.alert.sum())==54
    def groups(frame):
        keys=[tuple(float(r[k]) for k in COUNTS) if int(r.ip_protocol)==6 else None for _,r in frame.iterrows()]
        return np.array([str(MAJOR_PATTERNS.index(k)) if k in MAJOR_PATTERNS else 'other' for k in keys])
    c['pattern_group']=groups(c);d['pattern_group']=groups(d)
    rows=[]
    for role,frame in [('C',c),('D',d)]:
        for source,g in [('pooled',frame),*list(frame.groupby('source_id'))]:
            for group in [*map(str,range(5)),'other']:
                x=g.loc[g.pattern_group.eq(group)]
                rows.append({'role':role,'source_id':source,'group':group,'counts':list(MAJOR_PATTERNS[int(group)]) if group!='other' else None,
                    'normal_denominator':len(g),'group_rows':len(x),'original_fp':int(x.alert.sum()),
                    'normal_frequency':len(x)/len(g),'original_within_group_alert_rate':float(x.alert.mean()) if len(x) else None})
    cg={r['group']:r for r in rows if r['role']=='C' and r['source_id']=='pooled'}
    dg={r['group']:r for r in rows if r['role']=='D' and r['source_id']=='pooled'}
    assert all(x['group_rows']>0 for x in cg.values())
    composition=sum((dg[g]['normal_frequency']-cg[g]['normal_frequency'])*cg[g]['original_within_group_alert_rate'] for g in cg)
    conditional=sum(dg[g]['normal_frequency']*((dg[g]['original_within_group_alert_rate'] or 0)-cg[g]['original_within_group_alert_rate']) for g in cg)
    delta=float(d.alert.mean()-c.alert.mean())
    assert abs(delta-composition-conditional)<1e-12
    return {'groups':rows,'partition':'five previously identified TCP count patterns plus all remaining normal rows',
        'C_original_fpr':float(c.alert.mean()),'D_original_fpr':float(d.alert.mean()),'D_minus_C_fpr':delta,
        'composition_component_using_C_group_alert_rates':composition,'within_group_rate_component_using_D_weights':conditional,
        'D_fp_equivalent_composition':composition*len(d),'D_fp_equivalent_within_group':conditional*len(d),
        'interpretation':'Descriptive exact decomposition of original decisions, not causal attribution, not a candidate decision rule.'}

def main(project,out):
    root=project/'实验归档/20261117_第三十轮模型训练重设计/回传'
    frames={k:pd.read_parquet(root/f'inputs/{k}.parquet') for k in ['F','V','C','ED']}
    traj=root/'models/protocol_blind_s17/trajectory'
    base=read(traj/'preprocessing.json')['base'];bins=read(traj/'numeric_bins.json')['bins']
    nt,missing,proto=transform(frames['F'],base)
    preps=[]
    for arm in ['blind','aware']:
        for seed in SEEDS:
            t=root/f'models/protocol_{arm}_s{seed}/trajectory'
            assert read(t/'preprocessing.json')['base']==base
            assert read(t/'numeric_bins.json')['bins']==bins
            preps.append(str(t.relative_to(root)))
    labels=frames['F'].label.to_numpy()
    basis=ple_basis(nt,bins)
    collision={}
    for arm in ['blind','aware']:
        p=np.zeros(len(proto),dtype=np.int64) if arm=='blind' else proto
        raw=frames['F'][NUM].to_numpy()
        collision[arm]={
            'raw_five_numeric_missing_protocol':conflicts(np.column_stack([raw,missing,p]),labels),
            'preprocessed_fp32_missing_protocol':conflicts(np.column_stack([nt,missing,p]),labels),
            'parameter_free_ple_basis_plus_linear_skip':conflicts(np.column_stack([basis,nt,missing,p]),labels)}
    output={'scope':{'model_loads':0,'network_forwards':0,'new_scores':0,'fits':0,'threshold_searches':0,'new_decisions':0,
            'representation':'Pure NumPy non-trainable preprocessing and PLE-basis arithmetic only; no learned embedding or network'},
        'source':str(root),'bindings':{f'inputs/{k}.parquet':sha(root/f'inputs/{k}.parquet') for k in frames},
        'source_code_url':'https://github.com/yandex-research/rtdl-num-embeddings/blob/7381c5a4ed01253fbef9ea8cfed137f907265552/package/rtdl_num_embeddings.py',
        'bin_counts':dict(zip(NUM,[len(b)-1 for b in bins])),
        'global_F_ranges':{c:{'min':base['minimum'][j],'max':base['maximum'][j]} for j,c in enumerate(NUM)},
        'F_conflicts':collision,'models':{}}
    all_patterns=set()
    for seed in SEEDS:
        dec=pd.read_parquet(root/f'reports/protocol_blind_s{seed}-decisions.parquet')
        fp=dec.loc[dec.source_id.str.startswith('TQH-D_')&dec.label.eq(0)&dec.ip_protocol.eq(6)&dec.alert]
        all_patterns.update(pattern_groups(fp))
    for arm in ['blind','aware']:
      for seed in SEEDS:
        name=f'protocol_{arm}_s{seed}';t=root/f'models/{name}/trajectory'
        cache=np.load(t/'selected_scores.npz');ids=read(t/'score_identities.json')
        roleframes={}
        for role,key in [('F','train_scores'),('V','validation_scores')]:
            d=aligned(frames[role],ids[role]);d['score']=cache[key];roleframes[role]=d
        for role in ['C','ED']:
            s=pd.read_parquet(root/f'evidence/{name}/{role}-scores.parquet')
            d=frames[role].merge(s,on=ID,validate='one_to_one');roleframes[role]=d
        dec=pd.read_parquet(root/f'reports/{name}-decisions.parquet')
        fp=dec.loc[dec.source_id.str.startswith('TQH-D_')&dec.label.eq(0)&dec.ip_protocol.eq(6)&dec.alert]
        nn,mm,_=transform(fp,base)
        outside={c:int(((nn[:,j]<bins[j][0])|(nn[:,j]>bins[j][-1])).sum()) for j,c in enumerate(NUM)}
        support={}
        for role in ['F','V']:
            for profile in ['A','B']:
                d=roleframes[role];pool=d.loc[d.label.eq(0)&d.profile.eq(profile)&d.ip_protocol.eq(6)]
                support[role+'_'+profile]=support_for(fp,pool)
        d=roleframes['C'];support['C_B']=support_for(fp,d.loc[d.label.eq(0)&d.ip_protocol.eq(6)])
        patterns=[]
        for pattern in sorted(all_patterns):
            record={'counts':list(pattern),'roles':{}}
            for role,d in roleframes.items():
                mask=d.label.eq(0)&d.ip_protocol.eq(6)
                for c,v in zip(COUNTS,pattern):mask &= d[c].eq(v)
                for profile,g in d.loc[mask].groupby('profile'):
                    record['roles'][role+'_'+profile]={'duration':stats(g.duration_seconds),'score':stats(g.score)}
            patterns.append(record)
        output['models'][name]={'D_TCP_FP':len(fp),'global_PLE_endpoint_extrapolation':outside,
            'missing_by_feature':dict(zip(NUM,map(int,mm.sum(axis=0)))),'duration_support':support,'pattern_role_distributions':patterns,
            'F_frozen_score_training_signal':frozen_training_signal(roleframes['F'],set(pattern_groups(fp))),
            'original_alert_composition':original_alert_composition(roleframes['C'],dec,read(root/f'evidence/{name}/calibration.json')['threshold']),
            'frozen_C_threshold':read(root/f'evidence/{name}/calibration.json')['threshold']}
    out.mkdir(parents=True,exist_ok=True)
    output['bindings'].update({k:sha(root/k) for k in ['archive_manifest.json','freeze.json',
        'models/protocol_blind_s17/trajectory/preprocessing.json','models/protocol_blind_s17/trajectory/numeric_bins.json']})
    (out/'input_mechanism.json').write_bytes((json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    compact={k:v for k,v in output.items() if k!='models'}
    compact['models']={}
    rate_rows=[]
    for name,m in output['models'].items():
        compact['models'][name]={k:v for k,v in m.items() if k not in ['duration_support','pattern_role_distributions','original_alert_composition']}
        compact['models'][name]['duration_support']={k:v['summary'] for k,v in m['duration_support'].items()}
        compact['models'][name]['original_alert_composition']={k:v for k,v in m['original_alert_composition'].items() if k!='groups'}
        compact['models'][name]['pooled_original_group_rates']=[x for x in m['original_alert_composition']['groups'] if x['source_id']=='pooled']
        rate_rows.extend({'model_id':name,**x} for x in m['original_alert_composition']['groups'])
    (out/'input_mechanism_summary.json').write_bytes((json.dumps(compact,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    pd.DataFrame(rate_rows).to_csv(out/'input_mechanism_original_group_rates.csv',index=False)
    print(json.dumps({'models':len(output['models']),'F_conflicting_groups':collision['blind']['preprocessed_fp32_missing_protocol']['conflicting_groups'],
        'D_TCP_FP_outside_global_bins':{name:sum(m['global_PLE_endpoint_extrapolation'].values()) for name,m in output['models'].items()},
        'scope':output['scope'],'written':str(out)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();main(a.project,a.output_dir)
