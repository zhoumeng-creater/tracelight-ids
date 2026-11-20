"""Fixed raw-input geometry descriptions; no model, labels-to-fit, or threshold."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(l) for l in f]


def signature(row):return tuple(x[0] for x in row['bursts'])


def coordinates(row):
    values=np.array([x[1] for x in row['bursts']],dtype=float)
    directions=np.array(signature(row))
    assert (values>0).all()
    raw=np.log(values)
    shape=raw.copy()
    for d in (0,1):
        chosen=directions==d
        if chosen.any():shape[chosen]-=np.log(values[chosen].sum())
    return raw,shape


def quantiles(values):
    if not values:return {'n':0}
    return dict(zip(('n','min','q25','median','q75','q95','q99','max'),[len(values),*np.quantile(values,[0,.25,.5,.75,.95,.99,1]).tolist()]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rr=load(args.geometry)
    assert not any(any(w in r['source'].lower() for w in ('russell','harrison','meterpreter')) for r in rr)
    bank=[r for r in rr if r['role'].startswith('F') and r['label']==0 and r['geometry_eligible']]
    grouped=defaultdict(list)
    for r in bank:grouped[signature(r)].append(r)
    prepared={k:(v,np.stack([coordinates(r)[0] for r in v]),np.stack([coordinates(r)[1] for r in v])) for k,v in grouped.items()}
    results=[]
    for r in rr:
        if not r['geometry_eligible']:continue
        key=signature(r)
        rec={'flow_id':r['flow_id'],'source':r['source'],'label':r['label'],'target_kind':r['target_kind'],
             'role':r['role'],'stage_count':len(key),'reference_count':0}
        if key in prepared:
            refs,raw,shape=prepared[key]
            mask=np.array([v['stable_flow_id']!=r['stable_flow_id'] for v in refs])
            selected=np.flatnonzero(mask)
            rec['reference_count']=int(mask.sum())
            if mask.any():
                qraw,qshape=coordinates(r)
                for name,q,b in [('bytes',qraw,raw),('direction_normalized',qshape,shape)]:
                    delta=np.abs(b[mask]-q)
                    losses=delta.max(axis=1)
                    index=int(losses.argmin())
                    best=int(selected[index])
                    independent=float(delta.min(axis=0).max())
                    whole=float(losses[index])
                    assert whole+1e-12>=independent
                    rec[name]={'whole_log_change':whole,'independent_log_change':independent,
                               'whole_factor':float(np.exp(whole)),'independent_factor':float(np.exp(independent)),
                               'whole_reference':refs[best]['flow_id'],
                               'whole_reference_source':refs[best]['source'],
                               'bottleneck_stage':int(delta[index].argmax()),
                               'independent_reference_ids':[refs[int(selected[i])]['flow_id'] for i in delta.argmin(axis=0)]}
        results.append(rec)
    groups={}
    for src in sorted({r['source'] for r in results}):
        for kind in ('normal','control','download','unknown','other_attack'):
            chosen=[r for r in results if r['source']==src and r['target_kind']==kind]
            if not chosen:continue
            supported=[r for r in chosen if r['reference_count']]
            groups[src+'/'+kind]={'queries':len(chosen),'structurally_supported':len(supported)}
            for field in ('bytes','direction_normalized'):
                groups[src+'/'+kind][field]={
                    'whole_factor':quantiles([r[field]['whole_factor'] for r in supported]),
                    'independent_factor':quantiles([r[field]['independent_factor'] for r in supported]),
                    'whole_minus_independent_log':quantiles([r[field]['whole_log_change']-r[field]['independent_log_change'] for r in supported])}
    with gzip.open(args.output/'geometry_rows.jsonl.gz','wt',encoding='utf8') as f:
        for r in results:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
    out={'input_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),'normal_bank':len(bank),
         'queries':len(results),'groups':groups,'geometry':'same stage count/directions, raw byte ratio and per-direction-normalized stage ratios',
         'interpretation':'Minimum representation change, no accepted tolerance, predictions, AUC or operating threshold.',
         'model_calls':0,'thresholds_selected':0}
    (args.output/'geometry_summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'normal_bank':len(bank),'queries':len(results),'groups':len(groups)}))


if __name__=='__main__':main()
