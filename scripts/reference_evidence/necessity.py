"""Exact combinatorial stage support, without distances, models or thresholds.

Token = (connection direction, first-observed unique TCP bytes in that run).
Identity here is observational, not payload or application-semantic identity.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def read(path):
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:yield json.loads(line)


def is_subsequence(query,reference):
    i=0
    for token in reference:
        if i<len(query) and query[i]==token:i+=1
    return i==len(query)


def intersection(sets):
    if not sets:return set()
    ordered=sorted(sets,key=len)
    result=set(ordered[0])
    for values in ordered[1:]:
        result.intersection_update(values)
        if not result:break
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    geometry=args.repo/'docs/packet_interaction_20261119/connection_geometry.jsonl.gz'
    rr=list(read(geometry))
    assert not any(any(z in r['source'].lower() for z in ('russell','harrison','meterpreter')) for r in rr)
    bank=[r for r in rr if r['role'].startswith('F') and r['label']==0 and r['geometry_eligible']]
    profile_members=defaultdict(list)
    for r in bank:profile_members[tuple(map(tuple,r['bursts']))].append(r)
    profiles=sorted(profile_members)
    members=[sorted(profile_members[k],key=lambda x:x['flow_id']) for k in profiles]
    identities=[{r['stable_flow_id'] for r in v} for v in members]
    exact={k:i for i,k in enumerate(profiles)}
    tokens=defaultdict(set);positions=defaultdict(set);bags=defaultdict(set)
    counters=[]
    for i,k in enumerate(profiles):
        counter=Counter(k);counters.append(counter)
        for token in counter:tokens[token].add(i)
        for j,token in enumerate(k):positions[len(k),j,token].add(i)
        bags[tuple(sorted(counter.items()))].add(i)
    metadata={r['flow_id']:r for r in read(args.repo/'docs/coverage_basis_20261119/batch/rows.jsonl.gz')
              if r['dataset_role'] not in ('C','C_background')}
    events={e['event_id']:e for e in read(args.repo/'docs/coverage_basis_20261119/batch/events.jsonl.gz')
            if e.get('role') not in ('C','C_background') and not any(z in e['event_id'].lower() for z in ('russell','harrison','meterpreter'))}
    def support(ids,identity):
        return {i for i in ids if identities[i]-{identity}}
    def chosen_member(i,identity):
        return next(r for r in members[i] if r['stable_flow_id']!=identity)
    def facts(r):
        m=metadata.get(r['flow_id'],{})
        related=[events[e] for e in m.get('event_ids',[]) if e in events]
        return {'flow_id':r['flow_id'],'stable_flow_id':r['stable_flow_id'],'source':r['source'],
                'role':r['role'],'target_kind':r['target_kind'],'bursts':r['bursts'],
                'source_label_basis':m.get('source_label_basis',[]),
                'label_basis':m.get('label_basis'),
                'author_annotation':m.get('author_annotation'),
                'source_fact_label':m.get('source_fact_label'),
                'event_ids':m.get('event_ids',[]),
                'event_evidence':[{'event_id':e['event_id'],'kind':e['kind'],
                    'association_mode':e.get('association_mode'),'candidate_count':len(e.get('candidate_flow_ids',[])),
                    'candidate_set_complete':e.get('candidate_set_complete')} for e in related]}
    metrics=[];internal={}
    for r in rr:
        if not r['geometry_eligible']:
            continue
        k=tuple(map(tuple,r['bursts']));identity=r['stable_flow_id']
        per_token=[support(tokens[t],identity) for t in k]
        per_position=[support(positions[len(k),j,t],identity) for j,t in enumerate(k)]
        same_set=intersection(per_token)
        qc=Counter(k)
        multiset={i for i in same_set if all(counters[i][t]>=n for t,n in qc.items())}
        subseq={i for i in multiset if is_subsequence(k,profiles[i])}
        exact_ids=support({exact[k]},identity) if k in exact else set()
        bag_equal=support(bags[tuple(sorted(qc.items()))],identity)
        row={'flow_id':r['flow_id'],'source':r['source'],'role':r['role'],'target_kind':r['target_kind'],
             'label':r['label'],'stage_count':len(k),'resp_port':r['raw8']['resp_port'],
             'independent_any_position':all(per_token),'independent_same_position_same_length':all(per_position),
             'one_reference_contains_token_set':bool(same_set),
             'one_reference_contains_token_multiset':bool(multiset),
             'one_reference_contains_query_subsequence':bool(subseq),
             'one_reference_same_full_multiset':bool(bag_equal),'whole_exact_profile':bool(exact_ids)}
        row['any_position_independent_but_not_whole']=row['independent_any_position'] and not row['whole_exact_profile']
        row['same_position_independent_but_not_whole']=row['independent_same_position_same_length'] and not row['whole_exact_profile']
        row['same_bag_but_not_whole']=bool(bag_equal) and not bool(exact_ids)
        row['multiset_contained_but_no_ordered_subsequence']=bool(multiset) and not bool(subseq)
        # Every whole exact sequence provides each of these weaker supports.
        assert not row['whole_exact_profile'] or (row['independent_same_position_same_length'] and row['one_reference_contains_query_subsequence'])
        assert not row['independent_same_position_same_length'] or row['independent_any_position']
        metrics.append(row)
        internal[r['flow_id']]=(r,per_token,per_position,same_set,multiset,subseq,exact_ids,bag_equal)
    bool_fields=[k for k,v in metrics[0].items() if type(v) is bool]
    def summary(values):
        return {'connections':len(values),**{k:sum(v[k] for v in values) for k in bool_fields}}
    groups={}
    for role in ('F_base','F_expansion','V'):
        for kind in ('normal','control','download','unknown','other_attack'):
            selected=[r for r in metrics if r['role']==role and r['target_kind']==kind]
            if selected:groups[f'{role}/{kind}']=summary(selected)
    for source in sorted({r['source'] for r in metrics}):
        for kind in ('normal','control','download'):
            selected=[r for r in metrics if r['source']==source and r['target_kind']==kind]
            if selected:groups[f'{source}/{kind}']=summary(selected)
    for kind in ('normal','control','download'):
        values=[r for r in metrics if r['target_kind']==kind]
        groups['all/'+kind]=summary(values)
        groups['TCP443/'+kind]=summary([r for r in values if r['resp_port']==443])
    examples=[]
    for source in sorted({r['source'] for r in metrics}):
        for condition in ('same_position_independent_but_not_whole','same_bag_but_not_whole','any_position_independent_but_not_whole'):
            chosen=sorted([r for r in metrics if r['source']==source and r['target_kind']=='normal' and r[condition]],
                          key=lambda r:(r['stage_count'],r['flow_id']))[:3]
            for selected in chosen:
                r,per_token,per_position,same_set,multiset,subseq,exact_ids,bag_equal=internal[selected['flow_id']]
                identity=r['stable_flow_id'];providers=[]
                choices=per_position if condition=='same_position_independent_but_not_whole' else per_token
                for j,ps in enumerate(choices):
                    i=min(ps,key=lambda i:chosen_member(i,identity)['flow_id'])
                    provider=chosen_member(i,identity)
                    providers.append({'query_stage':j,'query_token':r['bursts'][j],
                        'reference_stage':j if condition=='same_position_independent_but_not_whole' else profiles[i].index(tuple(r['bursts'][j])),
                        'reference':facts(provider)})
                examples.append({'condition':condition,'query':facts(r),'supports':selected,
                                 'stage_providers':providers,
                                 'same_multiset_reference':facts(chosen_member(min(bag_equal),identity)) if bag_equal else None})
    report={'bank':'F whole-connection normal, geometry_eligible; self/same stable measurement identity excluded per query',
        'bank_rows':len(bank),'bank_profiles':len(profiles),'geometry_sha256':hashlib.sha256(geometry.read_bytes()).hexdigest(),
        'definitions':{'whole_exact_profile':'entire direction/byte token list identical',
            'independent_same_position_same_length':'each ordinal stage explained by some same-length reference; provider may change',
            'independent_any_position':'each token occurs somewhere in some normal reference',
            'one_reference_contains_query_subsequence':'one ordered reference contains query tokens, with reference-only skipping allowed; descriptive relation, not accepted free-gap model',
            'same_bag_but_not_whole':'a reference has exactly the same multiset but no exact full ordered profile exists'},
        'groups':groups,'examples':examples,
        'excluded_geometry_by_source_kind':dict(Counter(r['source']+'/'+r['target_kind'] for r in rr if not r['geometry_eligible'])),
        'real_model_calls':0,'distances_evaluated':0,'thresholds_selected':0,
        'interpretation':'Exact support relations only. Local-to-global support disagreement can be a normal business property; it is not an attack label or model false-positive count.'}
    (args.output/'necessity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    with gzip.open(args.output/'necessity_queries.jsonl.gz','wt',encoding='utf8') as f:
        for r in metrics:f.write(json.dumps(r,separators=(',',':'))+'\n')
    print(json.dumps({'bank_rows':len(bank),'bank_profiles':len(profiles),'groups':{k:v for k,v in groups.items() if k.startswith(('all/','V/','F_base/'))}},ensure_ascii=False))


if __name__=='__main__':main()
