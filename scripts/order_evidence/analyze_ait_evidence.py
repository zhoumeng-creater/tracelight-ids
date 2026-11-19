"""Fixed facts from existing AIT flows and actor events. No predictive score."""
from __future__ import annotations
import bisect, collections, gzip, hashlib, itertools, json
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
WORK, OUT = args.work, args.output
OUT.mkdir(parents=True, exist_ok=True)
DATA = WORK / '拟合来源与划分落实_20261119' / 'dataset'
BATCH = WORK / '覆盖与历史依据落实_20261119' / 'batch'
HISTORY = WORK / '覆盖与历史依据落实_20261119' / 'history_mechanism'

def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)

def run_id(fid): return fid.split(':')[1] if fid.startswith('ait:') else None
def small(r):
    return {'flow_id':r['flow_id'], 'label':r['label'], 'target_kind':r['target_kind'],
            'available_at':r['available_at'], 'host':r['host'], 'peer':r['peer'],
            'resp_port':r['raw8']['resp_port']}

def main():
    paths = [DATA/'context_inputs.jsonl.gz', DATA/'events.jsonl.gz',
             BATCH/'rows.jsonl.gz', HISTORY/'candidate_history_evidence.jsonl.gz']
    inputs = {}
    for r in rows(paths[0]):
        fid=r.get('measurement_flow_id',r['flow_id'])
        if run_id(fid) in ('santos','fox'):
            inputs[fid]=r
    events = [e for e in rows(paths[1]) if e.get('source_run') in ('santos','fox')]
    scored = {}
    for r in rows(paths[2]):
        if r.get('source_run') in ('santos','fox'):
            c=inputs[r['flow_id']]
            scored[r['flow_id']]={**r,'available_at':c['available_at'],'host':c['host'],'peer':c['peer']}
    report={'scope':'existing_AIT_santos_F_and_fox_V_only', 'real_model_calls':0,
            'fitting_calls':0,'threshold_searches':0,'C_or_reserved_material_used':False,
            'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            'runs':{},'dataset_inputs':{}}
    report['dataset_inputs']['background_by_host']=dict(collections.Counter(r['host'] for r in inputs.values()))
    report['dataset_inputs']['clock']='available_at: fixed Zeek offline export network_time, not detector receipt'
    report['dataset_inputs']['join']='context_inputs.measurement_flow_id == batch.rows.flow_id; role from frozen batch'
    report['dataset_inputs']['same_time_rule']='all queries read strict prefix before the complete equal-time batch is inserted'
    history={r['flow_id']:r for r in rows(paths[3]) if r.get('run') in ('santos','fox')}
    for run in ('santos','fox'):
        rr=[r for r in scored.values() if r['source_run']==run]
        ee=[e for e in events if e['source_run']==run]
        host='ait:'+run+':intranet_server'
        # Exact same monitored host and transport service; no size/time matching chosen after labels.
        control=[r for r in rr if r['label']==1 and r['target_kind']=='control' and r['raw8']['resp_port']==443 and r['host']==host]
        normal=[r for r in rr if r['label']==0 and r['raw8']['resp_port']==443 and r['host']==host]
        pos_times=sorted(r['available_at'] for r in control)
        before=[r for r in normal if r['available_at']<min(pos_times)]
        during=[r for r in normal if min(pos_times)<=r['available_at']<=max(pos_times)]
        after=[r for r in normal if r['available_at']>max(pos_times)]
        with_attack=[(r,bisect.bisect_left(pos_times,r['available_at'])) for r in normal]
        inherited=[(r,n) for r,n in with_attack if n>0]
        shared=[]
        for p in control:
            for n in normal:
                if p['available_at']==n['available_at']:
                    shared.append({'attack_flow_id':p['flow_id'],'normal_flow_id':n['flow_id'],
                                   'available_at':p['available_at'], 'same_prefix':True,
                                   'same_raw8':p['raw8']==n['raw8']})
        # Evidence-side command order; all candidate ambiguity retained.
        commands=[e for e in ee if e['kind']=='webshell_command']
        commands.sort(key=lambda e:e['start'])
        counts=collections.Counter(); examples={}
        for a,b in itertools.combinations(commands,2):
            if a['end']>b['start']:continue
            aa=[inputs[f]['available_at'] for f in a['candidate_flow_ids']]
            bb=[inputs[f]['available_at'] for f in b['candidate_flow_ids']]
            if max(aa)<min(bb):cls='all_candidates_preserve_actor_before'
            elif min(aa)>=max(bb):cls='no_candidate_pair_preserves_strict_actor_before'
            else:cls='candidate_dependent_or_tied'
            counts[cls]+=1
            examples.setdefault(cls,{'earlier_action':a['event_id'],'later_action':b['event_id'],
                                     'earlier_exports':aa,'later_exports':bb})
        by_kind=[]
        for kind in sorted({e['kind'] for e in ee}):
            ek=[e for e in ee if e['kind']==kind]
            ids=set(f for e in ek for f in e['candidate_flow_ids'])
            labels=collections.Counter(str(scored.get(f,{}).get('label')) for f in ids)
            incidence=collections.Counter(f for e in ek for f in e['candidate_flow_ids'])
            by_kind.append({'kind':kind,'event_count':len(ek),'unique_candidate_flows':len(ids),
                            'flow_labels':dict(labels),
                            'single_candidate_events':sum(len(e['candidate_flow_ids'])==1 for e in ek),
                            'possible_parent_flows_linked_to_multiple_actions':sum(n>1 for n in incidence.values()),
                            'hosts':sorted({inputs[f]['host'] for f in ids}),
                            'resp_ports':sorted({inputs[f]['raw8'][2] for f in ids})})
        normal_sequences=[]
        for kind in ('normal_intranet_page_retrieval','normal_owncloud_file_download'):
            actors=collections.defaultdict(list)
            for e in ee:
                if e['kind']==kind:actors[e['action_evidence']['host_role']].append(e)
            nc=collections.Counter();ne={}
            for actor,seq in actors.items():
                seq.sort(key=lambda e:e['start'])
                for a,b in zip(seq,seq[1:]):
                    if a['end']>b['start']:continue
                    aa=[inputs[f]['available_at'] for f in a['candidate_flow_ids']]
                    bb=[inputs[f]['available_at'] for f in b['candidate_flow_ids']]
                    if max(aa)<min(bb):cls='all_candidates_preserve_actor_before'
                    elif min(aa)>=max(bb):cls='no_candidate_pair_preserves_strict_actor_before'
                    else:cls='candidate_dependent_or_tied'
                    nc[cls]+=1
                    ne.setdefault(cls,{'actor':actor,'earlier_action':a['event_id'],'later_action':b['event_id'],
                                       'earlier_export_range':[min(aa),max(aa)],'later_export_range':[min(bb),max(bb)],
                                       'overlapping_candidate_flow_count':len(set(a['candidate_flow_ids'])&set(b['candidate_flow_ids']))})
            normal_sequences.append({'kind':kind,'actor_count':len(actors),'consecutive_nonoverlapping_action_pairs':sum(nc.values()),
                                     'counts':dict(nc),'examples':ne})
        dow=[]
        for e in ee:
            if e['kind']!='attack_download':continue
            dow.append({'event_id':e['event_id'],'actor_start':e['start'],'actor_end':e['end'],
                        'candidates':[{'flow_id':f,'available_at':inputs[f]['available_at'],
                            'label':scored[f]['label'],'resp_port':scored[f]['raw8']['resp_port'],
                            'peer':inputs[f]['peer'],
                            'previous_class':history[f]['previous_class'],
                            'prefix_count':history[f]['prefix_count'],
                            'prefix_categories':history[f]['prefix_categories'],
                            'wget_predecessor_available':any(t['available_before_query'] for cmd in history[f]['exact_url_wget_commands'] for t in cmd['candidates'])}
                            for f in e['candidate_flow_ids']]})
        report['runs'][run]={'event_business':by_kind,
            'same_host_HTTPS_control_vs_normal':{'attack_control_flows':len(control),'normal_flows':len(normal),
                'attack_available_interval':[min(pos_times),max(pos_times)],
                'normal_before_first_control':len(before),'normal_during_control_interval':len(during),
                'normal_after_last_control':len(after),
                'normal_queries_with_prior_confirmed_controls':len(inherited),
                'prior_control_count_range_for_these_normals':[min(n for _,n in inherited),max(n for _,n in inherited)] if inherited else None,
                'examples_with_prior_control':[dict(small(r),prior_confirmed_controls=n) for r,n in sorted(inherited,key=lambda t:t[0]['available_at'])[:3]],
                'shared_export_time_pairs':shared,
                'control_peer_addresses':sorted({r['peer'] for r in control}),
                'normal_peer_addresses':sorted({r['peer'] for r in normal})},
            'actor_vs_export_command_pair_order':{'pairs':sum(counts.values()),'counts':dict(counts),'examples':examples},
            'normal_actor_consecutive_event_order':normal_sequences,
            'attack_downloads':dow}
    out=OUT/'ait_evidence.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(out),'run_counts':{k:v['same_host_HTTPS_control_vs_normal'] for k,v in report['runs'].items()}},ensure_ascii=True,indent=2))

if __name__=='__main__': main()
