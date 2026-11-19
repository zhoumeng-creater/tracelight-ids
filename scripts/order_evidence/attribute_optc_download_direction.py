"""Attribute an already-computed fixed representation to original records."""
from pathlib import Path
from datetime import datetime
import gzip,json,hashlib

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
WORK, OUT = args.work, args.output
OUT.mkdir(parents=True, exist_ok=True)
QPATH=OUT/'representation/query_representations.jsonl.gz'
CONTEXT=WORK/'统一事件与方法落实_20261119/optc/context_replay_inputs.jsonl.gz'
BATCH=WORK/'覆盖与历史依据落实_20261119/batch/rows.jsonl.gz'

def read(p):
    with gzip.open(p,'rt',encoding='utf8') as f:
        yield from (json.loads(s) for s in f)

def main():
    queries=[r for r in read(QPATH) if r['flow_id'].startswith('optc:')]
    q=next(r for r in queries if r['flow_id']=='optc:20190924:CSHrim2uG4Jj6MUi31')
    ctx=list(read(CONTEXT))
    rr=sorted([r for r in ctx if r['group']==q['group']],key=lambda r:(r['flow_end_at'],r['flow_id']))
    pref=[r for r in rr if datetime.fromisoformat(r['flow_end_at']).timestamp()<q['time']]
    labels={r['flow_id']:r.get('label') for r in read(BATCH) if r['flow_id'].startswith('optc:')}
    rare=[]
    for r in pref:
        if r['raw8'][3] is not None and r['raw8'][0]==6:continue
        rare.append({'flow_id':r['physical_flow_id'],'raw8':r['raw8'],'flow_end_at':r['flow_end_at'],
            'previous_strict_prefix_count':sum(x['flow_end_at']<r['flow_end_at'] for x in pref),
            'seconds_before_download':q['time']-datetime.fromisoformat(r['flow_end_at']).timestamp(),
            'label':labels.get(r['physical_flow_id']),'present_in_scored_rows':r['physical_flow_id'] in labels,
            'host_is_originator':r['host_is_originator'],'source':r['source'],'destination':r['destination'],
            'source_path':r['source_path'],'source_line':r['source_line']})
    norm=[r for r in queries if r['label']==0 and r['response_port']==80]
    inds=[5,11,12];ds=q['direction']
    report={'query':q['flow_id'],'prefix_count':len(pref),'rare_prior_records':rare,
        'rare_channels':{'5':'duration_missing','11':'UDP','12':'other_protocol'},'direction':ds,
        'rare_channel_squared_norm_fraction':sum(ds[i]**2 for i in inds)/sum(v*v for v in ds),
        'normal_http_comparator_count':len(norm),
        'normal_http_rare_direction_nonzero_counts':{str(i):sum(r['direction'][i]!=0 for r in norm) for i in inds},
        'normal_http_rare_first_support_nonzero_counts':{str(i):sum(r['log_first_support'][i]!=0 for r in norm) for i in inds},
        'all_normal_background_TCP_duration_present':sum(r['raw8'][0]==6 and r['raw8'][3] is not None for r in ctx if r['group'].startswith('optc:20190919')),
        'rare_channels_present_in_query_itself':False,
        'interpretation':'Three unlabeled protocol/measurement records late in one attack-hour prefix dominate direction. Normal-hour background lacks these channels. Presence already appears in first-order P; no command-to-download edge has been established.',
        'source_receipts':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [QPATH,CONTEXT,BATCH]],
        'model_calls':0,'fitting_calls':0,'threshold_calls':0}
    (OUT/'optc_download_direction_attribution.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    p=OUT/'optc_unraveled_evidence.json'
    evidence=json.loads(p.read_text(encoding='utf8'))
    evidence['OpTC']['download_direction_attribution']=report
    evidence['OpTC']['source_scope_check']=json.loads((OUT/'optc_source_scope_check.json').read_text(encoding='utf8'))
    p.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ['prefix_count','rare_prior_records','rare_channel_squared_norm_fraction','normal_http_comparator_count','normal_http_rare_direction_nonzero_counts','normal_http_rare_first_support_nonzero_counts']},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
