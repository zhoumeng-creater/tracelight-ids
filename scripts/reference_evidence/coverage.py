"""Descriptive normal-reference support in existing real stage tables.

No learned model, anomaly score, operating threshold, calibration source, or
reserved source is read. Interval envelopes are necessary support descriptions,
not a declaration that every interpolated point is normal.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np


def load(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        return [json.loads(l) for l in f]


def profile(r):return tuple(map(tuple,r['bursts']))
def shape(r):return tuple(d for d,_ in r['bursts'])
def identity(r):return r['stable_flow_id']
def group(source):
    return 'ait:'+source.split('_')[1]+':complete_run' if source.startswith('ait_') else 'unraveled:apt_2021_single_campaign'
def host_role(source):
    return source.split('_',2)[2] if source.startswith('ait_') else None


def normal_prefixes(rows):
    output={}
    for s in sorted({r['source'] for r in rows}):
        rr=[r for r in rows if r['source']==s and r['label']==0 and r['geometry_eligible']]
        counts=Counter(profile(r)[:2] for r in rr)
        output[s]={'normal_eligible':len(rr),'prefixes':[
            {'first_two_stages':key,'connections':count}
            for key,count in counts.most_common()]}
    return output


class Pool:
    def __init__(self,rows):
        self.rows=rows
        self.exact=defaultdict(list);self.shapes=defaultdict(list);self.tokens=defaultdict(set)
        for r in rows:
            self.exact[profile(r)].append(r);self.shapes[shape(r)].append(r)
            for token in profile(r):self.tokens[token].add(identity(r))

    def describe(self,q,earlier=False):
        ident=identity(q)
        def legal(r):
            return identity(r)!=ident and (not earlier or r['replay_export_time']<q['replay_export_time'])
        same_profile=[r for r in self.exact.get(profile(q),[]) if legal(r)]
        same_shape=[r for r in self.shapes.get(shape(q),[]) if legal(r)]
        if earlier:
            available=[r for r in self.rows if legal(r)]
            tokens={token for r in available for token in profile(r)}
            token_covered=[token in tokens for token in profile(q)]
        else:
            available_count=len(self.rows)-sum(identity(r)==ident for r in self.rows)
            token_covered=[bool(self.tokens.get(token,set())-{ident}) for token in profile(q)]
        result={'reference_connections':len(available) if earlier else available_count,
            'exact_whole_reference_count':len(same_profile),
            'exact_whole_reference_examples':[r['flow_id'] for r in same_profile[:3]],
            'same_direction_sequence_reference_count':len(same_shape),
            'every_token_present_somewhere':bool(q['bursts']) and all(token_covered),
            'unsupported_token_positions':[i for i,v in enumerate(token_covered) if not v],
            'all_positions_exactly_supported_in_same_shape':False,
            'within_same_shape_coordinate_envelope':False,
            'within_same_shape_direction_total_envelope':False,
            'same_shape_direction_total_min':None,'same_shape_direction_total_max':None}
        if same_shape:
            arr=np.array([[n for _,n in r['bursts']] for r in same_shape],dtype=np.int64)
            values=np.array([n for _,n in q['bursts']],dtype=np.int64)
            result['all_positions_exactly_supported_in_same_shape']=bool(np.any(arr==values,axis=0).all())
            result['within_same_shape_coordinate_envelope']=bool(((arr.min(axis=0)<=values)&(values<=arr.max(axis=0))).all())
            totals=np.array([r['unique_payload_bytes'] for r in same_shape],dtype=np.int64)
            lo,hi=totals.min(axis=0),totals.max(axis=0)
            result['same_shape_direction_total_min']=lo.tolist();result['same_shape_direction_total_max']=hi.tolist()
            result['within_same_shape_direction_total_envelope']=bool(((lo<=q['unique_payload_bytes'])&(q['unique_payload_bytes']<=hi)).all())
        return result


def summarize(records,scope):
    valid=[r for r in records if r['geometry_eligible'] and scope in r['coverage']]
    entries=[r['coverage'][scope] for r in valid]
    return {'all_connections':len(records),'geometry_eligible':sum(r['geometry_eligible'] for r in records),
        'assessed_connections':len(valid),
        'exact_whole':sum(x['exact_whole_reference_count']>0 for x in entries),
        'same_direction_sequence':sum(x['same_direction_sequence_reference_count']>0 for x in entries),
        'every_token_present_somewhere':sum(x['every_token_present_somewhere'] for x in entries),
        'positionwise_exact_same_shape':sum(x['all_positions_exactly_supported_in_same_shape'] for x in entries),
        'positionwise_exact_without_complete_reference':sum(x['all_positions_exactly_supported_in_same_shape'] and not x['exact_whole_reference_count'] for x in entries),
        'coordinate_envelope_same_shape':sum(x['within_same_shape_coordinate_envelope'] for x in entries),
        'direction_total_envelope_same_shape':sum(x['within_same_shape_direction_total_envelope'] for x in entries)}


def normal_counterexamples(records,rows,pool):
    lookup={r['flow_id']:r for r in rows};selected=Counter();examples=[]
    for r in records:
        if r['label']!=0 or not r['geometry_eligible']:continue
        c=r['coverage']['F_all']
        if not c['all_positions_exactly_supported_in_same_shape'] or c['exact_whole_reference_count']:continue
        if selected[r['source']]>=3:continue
        selected[r['source']]+=1;q=lookup[r['flow_id']]
        candidates=[b for b in pool.shapes[shape(q)] if identity(b)!=identity(q)]
        witnesses=[]
        for i,tok in enumerate(q['bursts']):
            b=next(b for b in candidates if b['bursts'][i]==tok)
            witnesses.append({'position':i,'query_stage':tok,'reference_flow_id':b['flow_id'],'reference_profile':b['bursts']})
        examples.append({'query_flow_id':q['flow_id'],'source':q['source'],'label':q['label'],
            'query_profile':q['bursts'],'no_exact_whole_reference':True,'position_witnesses':witnesses})
    return examples


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    geom=args.repo/'docs/packet_interaction_20261119/connection_geometry.jsonl.gz'
    rr=load(geom)
    assert not any(any(x in r['source'].lower() for x in ('russell','harrison','meterpreter')) for r in rr)
    bank=[r for r in rr if r['role'].startswith('F') and r['label']==0 and r['geometry_eligible']]
    assert len(bank)==6840
    allpool=Pool(bank)
    bysource={s:Pool([r for r in bank if r['source']==s]) for s in {r['source'] for r in rr}}
    crosscapture={s:Pool([r for r in bank if r['source']!=s]) for s in bysource}
    crossgroup={s:Pool([r for r in bank if group(r['source'])!=group(s)]) for s in bysource}
    byrole={s:Pool([r for r in bank if host_role(r['source'])==host_role(s)]) for s in bysource if host_role(s)}
    records=[]
    for q in rr:
        result={k:q[k] for k in ('flow_id','stable_flow_id','source','role','label','target_kind','geometry_eligible','raw8')}
        result['independent_group']=group(q['source'])
        result['stage_count']=len(q['bursts']);result['unique_payload_bytes']=q['unique_payload_bytes']
        result['coverage']={}
        if q['geometry_eligible']:
            s=q['source']
            pools={'F_all':allpool,'F_same_capture':bysource[s],
                   'F_other_capture':crosscapture[s],'F_other_run_campaign':crossgroup[s]}
            if s in byrole:pools['F_same_AIT_server_role']=byrole[s]
            for name,pool in pools.items():result['coverage'][name]=pool.describe(q)
            if q['role'].startswith('F'):
                result['coverage']['F_same_capture_strictly_earlier']=bysource[s].describe(q,earlier=True)
        records.append(result)
    lookup={r['flow_id']:r for r in records}
    summaries={}
    for s in sorted({r['source'] for r in records}):
        for kind in ('normal','control','download','unknown','other_attack'):
            selected=[r for r in records if r['source']==s and r['target_kind']==kind]
            if not selected:continue
            summaries[s+'/'+kind]={scope:summarize(selected,scope) for scope in sorted({k for r in selected for k in r['coverage']})}
    for role in ('F','V'):
        for kind in ('normal','control','download'):
            selected=[r for r in records if r['role'].startswith(role) and r['target_kind']==kind]
            summaries[role+'/'+kind]={scope:summarize(selected,scope) for scope in sorted({k for r in selected for k in r['coverage']})}
            https=[r for r in selected if r['raw8']['resp_port']==443]
            if https:
                summaries[role+'/'+kind+'_https']={scope:summarize(https,scope) for scope in sorted({k for r in https for k in r['coverage']})}
    actions=[]
    # Filter the text before parsing: calibration/reserved activity metadata is not used.
    events=args.repo/'docs/coverage_basis_20261119/batch/events.jsonl.gz'
    with gzip.open(events,'rt',encoding='utf8') as f:
        for line in f:
            if 'ait:santos:normal:' not in line and 'ait:fox:normal:' not in line:continue
            e=json.loads(line)
            if e['kind']!='normal_owncloud_file_download':continue
            assert e.get('role') not in ('C','C_background')
            candidates=[lookup[x] for x in e['candidate_flow_ids'] if x in lookup]
            exact=[];normal=[];unknown=[];usable=[]
            for r in candidates:
                assert r['label'] in (0,None),'A normal action candidate is not silently relabeled here'
                (normal if r['label']==0 else unknown).append(r['flow_id'])
                if r['geometry_eligible']:usable.append(r['flow_id'])
                if r['geometry_eligible'] and r['coverage']['F_all']['exact_whole_reference_count']:
                    exact.append(r['flow_id'])
            actions.append({'event_id':e['event_id'],'association_mode':e['association_mode'],
                'candidate_set_complete':e['candidate_set_complete'],'candidate_flow_ids':e['candidate_flow_ids'],
                'represented_candidates':len(candidates),'normal_candidates':normal,'unknown_candidates':unknown,
                'geometry_eligible_candidates':usable,'exact_reference_candidates':exact,
                'at_least_one_exact_candidate':bool(exact),
                'all_candidates_exactly_covered':len(exact)==len(e['candidate_flow_ids'])})
    action_summary={}
    for run in ('santos','fox'):
        aa=[a for a in actions if a['event_id'].startswith('ait:'+run+':')]
        action_summary[run]={'actions':len(aa),'unique_candidates':len({x for a in aa for x in a['candidate_flow_ids']}),
            'unique_normal_candidates':len({x for a in aa for x in a['normal_candidates']}),
            'unique_unknown_candidates':len({x for a in aa for x in a['unknown_candidates']}),
            'unique_eligible_candidates':len({x for a in aa for x in a['geometry_eligible_candidates']}),
            'unique_exact_candidates':len({x for a in aa for x in a['exact_reference_candidates']}),
            'actions_with_at_least_one_exact_candidate':sum(a['at_least_one_exact_candidate'] for a in aa),
            'actions_with_all_candidates_exact':sum(a['all_candidates_exactly_covered'] for a in aa)}
    recordpath=args.output/'coverage_records.jsonl.gz'
    with recordpath.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz,io.TextIOWrapper(gz,encoding='utf8',newline='\n') as f:
        for r in records:f.write(json.dumps(r,separators=(',',':'))+'\n')
    report={'input_sha256':hashlib.sha256(geom.read_bytes()).hexdigest(),
        'bank_connections':len(bank),'bank_exact_profiles':len(allpool.exact),
        'bank_singleton_profiles':sum(len(x)==1 for x in allpool.exact.values()),
        'bank_repeated_profiles':sum(len(x)>1 for x in allpool.exact.values()),
        'bank_max_profile_multiplicity':max(map(len,allpool.exact.values())),
        'normal_first_two_profiles':normal_prefixes(rr),
        'bank_duplicate_stable_ids':sum(n-1 for n in Counter(identity(r) for r in bank).values()),
        'role_contract':'Static F normal library -> opened fox V; F leave-self-out and same-capture strict export-prefix are descriptive, not independent tests.',
        'clock_contract':'Strict prefix uses replay_export_time < query replay_export_time; retrospective known-normal labels are not proved available then.',
        'coverage_contract':'Exact means full ordered (direction, unique-byte-count) profile; shape/envelopes only describe observed support and never declare synthetic combinations normal.',
        'summaries':summaries,'normal_download_action_summary':action_summary,'normal_download_actions':actions,
        'model_calls':0,'thresholds_selected':0,'new_experiment_registrations':0,
        'records':{'path':recordpath.name,'sha256':hashlib.sha256(recordpath.read_bytes()).hexdigest()}}
    (args.output/'coverage.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8',newline='\n')
    (args.output/'coverage_real_normal_counterexamples.json').write_text(json.dumps(normal_counterexamples(records,rr,allpool),indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'F_normal':summaries['F/normal']['F_all'],'V_normal':summaries['V/normal']['F_all'],'downloads':action_summary}))


if __name__=='__main__':main()
