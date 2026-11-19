"""Check the assembled normal cohorts and preserve historical C25 identities."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from urllib.parse import urlsplit, unquote, parse_qs

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
PRIOR=WORK/'拟合来源与划分落实_20261119'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    candidates=rows(ROOT/'connection_candidates.jsonl.gz');byid={r['flow_id']:r for r in candidates}
    qualified=rows(ROOT/'qualified_connections.jsonl.gz');c=rows(ROOT/'C_candidates.jsonl.gz');v=rows(ROOT/'normal_time_validation.jsonl.gz')
    episodes={r['episode_id']:r for r in rows(ROOT/'business_episodes.jsonl.gz')}
    components={r['dependency_component_id']:r for r in rows(ROOT/'dependency_components.jsonl.gz')}
    requests={r['request_id']:r for r in rows(ROOT/'request_evidence.jsonl.gz')}
    contexts={r['physical_flow_id']:r for r in rows(ROOT/'context_rows.jsonl.gz')}
    inputs=rows(ROOT/'context_inputs.jsonl.gz')
    checks=[]
    def check(name,value):checks.append({'check':name,'passed':bool(value)})
    check('all_candidate_connection_ids_unique',len(byid)==len(candidates))
    check('qualified_partition_complete',{r['flow_id'] for r in qualified}=={r['flow_id'] for r in candidates if r['normal_identity_qualified']} and len(qualified)==len(c)+len(v)+sum(r['role']=='unresolved' for r in qualified))
    check('C_and_later_validation_stable_ids_disjoint',not ({r['stable_flow_id'] for r in c}&{r['stable_flow_id'] for r in v}))
    check('C_and_later_validation_episode_ids_disjoint',not ({e for r in c for e in r['episode_ids']}&{e for r in v for e in r['episode_ids']}))
    check('C_and_later_validation_dependency_components_disjoint',not ({r['dependency_component_id'] for r in c}&{r['dependency_component_id'] for r in v}))
    check('roles_fixed_by_complete_early_and_later_days',all(r['date_utc'] in {'2022-01-21','2022-01-22'} and not r['cross_day_connection'] for r in c) and all(r['date_utc'] in {'2022-01-23','2022-01-24'} and not r['cross_day_connection'] for r in v))
    check('all_consumable_components_complete',all(components[r['dependency_component_id']]['all_episodes_complete'] and not components[r['dependency_component_id']]['cross_day_connection_present'] for r in c+v))
    check('normal_labels_have_whole_connection_evidence',all(r['label']==0 and not r['rejection_reasons'] and r['service_request_ids'] and r['episode_ids'] and all(not requests[q]['author_attack_labels'] for q in r['service_request_ids']) for r in qualified))
    check('all_intranet_request_candidates_supported_by_actors',all(requests[q]['normal_actor_service_supported'] for r in qualified if r['capture_host']=='intranet_server' for q in r['service_request_ids']))
    check('all_cloud_frontend_candidates_supported_by_actors',all(r['possible_frontend_request_ids'] and all(requests[q]['normal_actor_service_supported'] for q in r['possible_frontend_request_ids']) for r in qualified if r['capture_host']=='cloud_share'))
    check('raw8_features_and_full_background_context_references_present',len(contexts)==len(inputs)==25638 and all(r['stable_flow_id'] in contexts and r['features8']==contexts[r['stable_flow_id']]['current_raw8_log1p'] and r['features27']==contexts[r['stable_flow_id']]['model_features'] for r in qualified))
    check('context_strictly_past',all(c['max_history_available_at'] is None or c['max_history_available_at']<c['effective_available_at'] for c in contexts.values()))
    check('cross_day_connections_retained_unresolved',all(r['role']=='unresolved' for r in candidates if r['cross_day_connection']))
    previous=load(PRIOR/'calibration/candidate_connections.json')
    old25={r['model_flow']['flow_id']:r for r in previous['candidates'] if r['eligible']}
    aliases=rows(PRIOR/'dataset/measurement_aliases.jsonl.gz')
    history=[]
    for a in aliases:
        old=old25[a['old_flow_id']];now=byid.get(a['new_flow_id'])
        history.append({'old_flow_id':a['old_flow_id'],'current_flow_id':a['new_flow_id'],'stable_flow_id':a['stable_flow_id'],'historical_C25_qualification_preserved':True,'historical_source_path':str(PRIOR/'calibration/candidate_connections.json'),'current_normal_identity_qualified':None if now is None else now['normal_identity_qualified'],'current_role':None if now is None else now['role'],'current_rejection_reasons':None if now is None else now['rejection_reasons'],'current_dependency_component_id':None if now is None else now['dependency_component_id'],'current_raw8_equal_historical':now is not None and now['raw8']=={k:old['model_flow'][k] for k in now['raw8']},'previously_opened_and_C_eligible':True,'new_temporal_validation_is_not_blind':True})
    check('all_historical_C25_preserved_and_raw8_equal',len(history)==25 and all(r['current_raw8_equal_historical'] for r in history))
    strata=[]
    for role in ('C_candidate','normal_time_validation','unresolved'):
        for host in ('intranet_server','cloud_share'):
            selected=[r for r in qualified if r['role']==role and r['capture_host']==host]
            if not selected:continue
            eids={e for r in selected for e in r['episode_ids']}
            strata.append({'role':role,'capture_host':host,'connections':len(selected),'episodes':len(eids),'dependency_components':len({r['dependency_component_id'] for r in selected}),'actors':sorted({episodes[e]['actor'] for e in eids}),'dates':sorted({r['date_utc'] for r in selected}),'file_transfer_request_connections':sum(r['business_stratum']=='owncloud_file_transfer_request' for r in selected)})
    downloads=[]
    for actor in (f'internal_employee_{i}' for i in range(4)):
        member=f'gather/{actor}/logs/sm.log';starts={}
        for n,line in enumerate((ROOT/'source'/member).open(encoding='utf-8'),1):
            x=json.loads(line);tid=x.get('transition_id')
            if x.get('message')=='Starting menu action':starts[tid]=(n,x)
            if x.get('message')=='Downloaded file' and tid in starts:
                bn,b=starts[tid]
                downloads.append({'action_id':f'ait:russellmitchell:normal_download:{actor}:{tid}','actor':actor,'member':member,'intent_line':bn,'completion_line':n,'start':b['timestamp'],'completion':x['timestamp'],'file':x['file']})
    file_links=defaultdict(list)
    for r in qualified:
        for q in r['file_transfer_request_ids']:file_links[q].append(r)
    file_records=[]
    for request_id,flows in sorted(file_links.items()):
        q=requests[request_id];u=urlsplit(q['target']);query=parse_qs(u.query)
        possible_actors={a for f in flows for a in f['possible_actor_ids']}
        matches=[]
        for action in downloads:
            obj=action['file'];target=obj['path'].rstrip('/')+'/'+obj['name']
            exact=(obj['file_type']=='file' and unquote(u.path)=='/remote.php/webdav'+target) or (obj['file_type']=='dir' and u.path=='/index.php/apps/files/ajax/download.php' and query.get('files')==[obj['name']] and query.get('dir')==[obj['path']])
            if exact and math.floor(action['start'])<=q['timestamp']<=action['completion'] and action['actor'] in possible_actors:matches.append(action)
        file_records.append({'service_request_id':request_id,'request':q,'candidate_flow_ids':[f['flow_id'] for f in flows],'candidate_roles':sorted({f['role'] for f in flows}),'candidate_episode_ids':sorted({e for f in flows for e in f['episode_ids']}),'exact_object_and_time_actor_completion_candidates':matches,'association':'Apache request second and complete connection envelope; all flow candidates retained. SM action uses exact object path/type and intent-to-completion time.','unique_TCP_transaction_claimed':len(flows)==1})
    file_summary=[]
    for role in ('C_candidate','normal_time_validation','unresolved'):
        chosen=[r for r in file_records if role in r['candidate_roles']]
        flow_set={f['flow_id'] for f in qualified if f['role']==role and f['file_transfer_request_ids']}
        episode_set={e for f in qualified if f['flow_id'] in flow_set for e in f['episode_ids']}
        file_summary.append({'role':role,'file_request_candidate_connections':len(flow_set),'unique_service_file_requests':len(chosen),'episode_candidates':len(episode_set),'unique_exact_actor_completed_downloads':len({a['action_id'] for r in chosen for a in r['exact_object_and_time_actor_completion_candidates']}),'file_requests_without_exact_actor_completion':sum(not r['exact_object_and_time_actor_completion_candidates'] for r in chosen)})
    save(ROOT/'file_business_evidence.json',{'business_stratum_semantics':'owncloud_file_transfer_request denotes a connection containing a possible file request within its envelope, not a proven one-to-one file transfer. Use this request/action table for counts.','summary':file_summary,'requests':file_records})
    save(ROOT/'historical_C25_disposition.json',{'original_file_sha256':sha(PRIOR/'calibration/candidate_connections.json'),'current_role_counts':dict(Counter(str(r['current_role']) for r in history)),'rows':history})
    save(ROOT/'assembly_check.json',{'status':'passed' if all(r['passed'] for r in checks) else 'failed','checks':checks,'strata':strata,'candidate_count':len(candidates),'qualified_count':len(qualified),'cross_day_candidates_retained':sum(r['cross_day_connection'] for r in candidates),'historical_C25_disposition_counts':dict(Counter(str(r['current_role']) for r in history)),'model_calls':0,'normalization_reruns':0,'script_sha256':sha(Path(__file__))})
    save(ROOT/'output_receipt.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='output_receipt.json'})
    print(json.dumps({'checks':len(checks),'failed':[r for r in checks if not r['passed']],'strata':strata,'historical_C25':dict(Counter(str(r['current_role']) for r in history))},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
