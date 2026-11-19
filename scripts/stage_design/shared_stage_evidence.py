"""Exact stage/prefix overlap in existing geometry; never call a model.

Length identity is an identity of this observed representation, not equality of
payload, protocol semantics, or malicious intent. No distance threshold is used.
"""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    with gzip.open(args.geometry,'rt',encoding='utf8') as f:rr=[json.loads(l) for l in f]
    assert not any(any(x in r['source'].lower() for x in ('russell','harrison','meterpreter')) for r in rr)
    fit=[r for r in rr if r['role'].startswith('F')]
    normals=[r for r in fit if r['label']==0 and r['raw8']['resp_port']==443]
    positives=[r for r in fit if r['target_kind']=='control' and r['raw8']['resp_port']==443]
    vocabulary=defaultdict(list)
    for r in normals:
        for i,token in enumerate(r['bursts']):
            vocabulary[tuple(token)].append((r['flow_id'],i))
    positive_rows=[]
    for r in positives:
        shared=[i for i,x in enumerate(r['bursts']) if tuple(x) in vocabulary]
        positive_rows.append({'flow_id':r['flow_id'],'source':r['source'],
                              'stage_count':len(r['bursts']),'shared_stage_indices':shared,
                              'first_stage_normal_examples':vocabulary.get(tuple(r['bursts'][0]),[])[:3],
                              'all_stages_in_normal_vocabulary':len(shared)==len(r['bursts'])})
    within_source={}
    for source in sorted({r['source'] for r in rr}):
        positive=[r for r in rr if r['source']==source and r['label']==1 and r['raw8']['resp_port']==443]
        normal=[r for r in rr if r['source']==source and r['label']==0 and r['raw8']['resp_port']==443]
        if not positive:continue
        data={'positive_connections':len(positive),'normal_connections':len(normal),'prefixes':{}}
        for length in (1,2,3,4):
            lookup=defaultdict(list)
            for r in normal:
                if len(r['bursts'])>=length:
                    lookup[tuple(map(tuple,r['bursts'][:length]))].append(r['flow_id'])
            matched=[]
            for r in positive:
                key=tuple(map(tuple,r['bursts'][:length]))
                if len(key)==length and key in lookup:
                    matched.append({'flow_id':r['flow_id'],'prefix':key,'normal_flow_ids':lookup[key]})
            data['prefixes'][str(length)]={'positive_matches':len(matched),
                'distinct_normal_connections':len({x for v in matched for x in v['normal_flow_ids']}),'matches':matched}
        within_source[source]=data
    normal_bank=[r for r in fit if r['label']==0 and r['geometry_eligible']]
    shapes=Counter(tuple(map(tuple,r['bursts'])) for r in normal_bank)
    report={'geometry_sha256':hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        'input_contract':'direction + first-observed unique TCP bytes per direction run; no payload semantics',
        'F_normal_https_connections':len(normals),'F_normal_token_types':len(vocabulary),
        'F_https_control_connections':len(positives),
        'F_controls_first_token_in_normal_vocabulary':sum(bool(x['first_stage_normal_examples']) for x in positive_rows),
        'F_controls_all_tokens_in_normal_vocabulary':sum(x['all_stages_in_normal_vocabulary'] for x in positive_rows),
        'F_eligible_normal_TCP_reference_connections':len(normal_bank),
        'F_exact_normal_burst_profiles':len(shapes),
        'F_unique_profile_total_stages':sum(map(len,shapes)),
        'F_unique_profile_max_stages':max(map(len,shapes)),
        'pair_cost_cells_for_query_6_stages':6*sum(map(len,shapes)),
        'pair_cost_cells_for_query_11_stages':11*sum(map(len,shapes)),
        'F_normal_stage_count_distribution':dict(sorted(Counter(len(r['bursts']) for r in normal_bank).items())),
        'positive_stage_overlaps':positive_rows,'within_source_prefixes':within_source,
        'unknown_connections_used_as_normal':0,'model_calls':0,'thresholds_selected':0,
        'interpretation':'Shared local inputs refute inheriting positive flow labels as local truth. Exact all-stage normal stitching was NOT observed in these F controls; that mechanism premise remains unconfirmed.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))}))


if __name__=='__main__':main()
