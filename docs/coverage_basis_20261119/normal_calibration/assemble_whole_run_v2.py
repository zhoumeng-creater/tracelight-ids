"""Separate author-normal evidence from transport quality and source roles.

V1 artifacts remain historical. No new source, parser, model or threshold run.
"""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import math
from urllib.parse import urlsplit,unquote,parse_qs

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'v2'
WORK=ROOT.parents[1]
PRIOR=WORK/'拟合来源与划分落实_20261119'
START,END=1642723200,1643068800

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(p,rs):
    with Path(p).open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as z:
        with io.TextIOWrapper(z,encoding='utf-8',newline='\n') as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def main():
    OUT.mkdir(exist_ok=True)
    candidates=rows(ROOT/'connection_candidates.jsonl.gz')
    requests={r['request_id']:r for r in rows(ROOT/'request_evidence.jsonl.gz')}
    episodes=rows(ROOT/'business_episodes.jsonl.gz');ep={r['episode_id']:r for r in episodes}
    contexts={r['physical_flow_id']:r for r in rows(ROOT/'context_rows.jsonl.gz')}
    physical={}
    for p in (ROOT/'extracted/intranet_server/connections.jsonl.gz',PRIOR/'ait_extracted/russellmitchell/cloud_share/connections.jsonl.gz'):
        physical.update({r['flow_id']:r for r in rows(p)})
    out=[];links=[]
    for old in candidates:
        r=dict(old);f=physical[r['flow_id']]
        whole=[requests[q] for q in r['service_request_ids']]
        front=[requests[q] for q in r['possible_frontend_request_ids']]
        reasons=[]
        if not whole:reasons.append('no_service_request_evidence_for_observation')
        if any(q['author_attack_labels'] for q in whole):reasons.append('author_attack_request_possible_in_observation')
        st,en=r['connection_start'],r['observed_end']
        if en is None or not(START<=st<=en<END):reasons.append('full_observation_interval_not_inside_author_validity')
        actor_side=whole if r['capture_host']=='intranet_server' else front
        if not actor_side:reasons.append('no_actor_side_request_evidence')
        if any(q.get('known_internal_actor') is None for q in actor_side):reasons.append('unknown_actor_among_possible_service_requests')
        if any(q['author_attack_labels'] for q in actor_side):reasons.append('author_attack_label_on_actor_side_request')
        if r['capture_host']=='cloud_share':
            if 'not_expected_cloud_reverse_proxy_endpoint' in old['rejection_reasons']:reasons.append('not_expected_cloud_reverse_proxy_endpoint')
            if {q['request_identity_sha256'] for q in front}!={q['request_identity_sha256'] for q in whole}:reasons.append('frontend_backend_request_identity_sets_not_closed')
        label_ok=not reasons
        missing=[k for k,v in f['raw8'].items() if v is None or v<0]
        context=contexts.get(f['stable_flow_id'])
        input_ready=not missing and context is not None and isinstance(f['replay_export_time'],(int,float)) and math.isfinite(f['replay_export_time'])
        history={'normal_identity_qualified':r.pop('normal_identity_qualified'),'label':r.pop('label'),'role':r.pop('role'),'rejection_reasons':r.pop('rejection_reasons'),'component_role':r.pop('chronological_component_role')}
        incomplete_eids=sorted({eid for q in actor_side for eid in q.get('normal_episode_candidates',[]) if not ep[eid]['complete_valid_day']})
        r.update(schema='whole-run-normal-v2',historical_v1=history,label_evidence_qualified=label_ok,label=0 if label_ok else None,
                 label_evidence_rejection_reasons=sorted(set(reasons)),model_input_ready=input_ready,calibration_candidate=label_ok and input_ready,
                 role='C_candidate_whole_run',source_role='C_candidate_whole_run',source_group='ait:russellmitchell:complete_run',
                 label_basis='B_author_normal_service_requests_known_internal_actors_full_observation_interval' if label_ok else None,
                 label_evidence={'known_actor_ids':sorted({q.get('known_internal_actor') for q in actor_side if q.get('known_internal_actor')}),'unknown_actor_request_ids':[q['request_id'] for q in actor_side if q.get('known_internal_actor') is None],'author_attack_request_ids':[q['request_id'] for q in whole+front if q['author_attack_labels']],'actor_activity_attribution_missing_request_count':sum(not q.get('normal_episode_candidates') for q in actor_side),'incomplete_activity_candidate_ids':incomplete_eids,'activity_closure_used_as_label_gate':False,'SF_used_as_label_gate':False,'condition':'Original complete service logs with author line labels plus actor address ownership. All possible requests in this observation envelope must be normal; proxy frontend/backend identity sets must close. No unique TCP request assignment is inferred.'},
                 input_quality={'raw8_complete':not missing,'missing_fields':missing,'conn_state':f['zeek_record'].get('conn_state'),'history':f.get('history'),'graceful_TCP_close_observed':f['zeek_record'].get('conn_state')=='SF','after_input_eof':f['after_input_eof'],'during_shutdown':f['during_shutdown'],'replay_export_time':f['replay_export_time'],'context_available':context is not None,'measurement_contract':f['measurement_contract'],'capture_provenance':f['capture_provenance'],'quality_flags_not_automatic_label_rejections':True},
                 role_basis='Complete russellmitchell source is the candidate C allocation. No early/late role boundary within this source; unknown labels remain unknown.',
                 context_reference={'path':str(ROOT/'context_rows.jsonl.gz'),'physical_flow_id':f['stable_flow_id']} if context is not None else None,
                 features8=context['current_raw8_log1p'] if context is not None else None,features27=context['model_features'] if context is not None else None,
                 training_or_calibration_executed=False)
        out.append(r)
        for eid in r['episode_ids']:links.append({'episode_id':eid,'event_id':eid,'flow_id':r['flow_id'],'stable_flow_id':r['stable_flow_id'],'relationship':'actor_activity_interval_candidate','activity_complete_valid_day':ep[eid]['complete_valid_day'],'connection_label_propagates_from_episode':False})
    qualified=[r for r in out if r['label_evidence_qualified']];consumable=[r for r in out if r['calibration_candidate']]
    byid={r['flow_id']:r for r in out}

    downloads=[]
    for i in range(4):
        actor=f'internal_employee_{i}';member=f'gather/{actor}/logs/sm.log';starts={}
        for n,line in enumerate((ROOT/'source'/member).open(encoding='utf-8'),1):
            x=json.loads(line);tid=x.get('transition_id')
            if x.get('message')=='Starting menu action':starts[tid]=(n,x)
            if x.get('message')=='Downloaded file' and tid in starts:
                bn,b=starts[tid];downloads.append({'action_id':f'ait:russellmitchell:normal_download:{actor}:{tid}','actor':actor,'member':member,'intent_line':bn,'completion_line':n,'start':b['timestamp'],'completion':x['timestamp'],'file':x['file']})
    request_flows=defaultdict(list)
    for r in consumable:
        for q in r['file_transfer_request_ids']:request_flows[q].append(r)
    files=[]
    for qid,fs in sorted(request_flows.items()):
        q=requests[qid];u=urlsplit(q['target']);query=parse_qs(u.query);actors={a for f in fs for a in f['label_evidence']['known_actor_ids']};matches=[]
        for a in downloads:
            obj=a['file'];target=obj['path'].rstrip('/')+'/'+obj['name']
            exact=(obj['file_type']=='file' and unquote(u.path)=='/remote.php/webdav'+target) or (obj['file_type']=='dir' and u.path=='/index.php/apps/files/ajax/download.php' and query.get('files')==[obj['name']] and query.get('dir')==[obj['path']])
            if exact and math.floor(a['start'])<=q['timestamp']<=a['completion'] and a['actor'] in actors:matches.append(a)
        files.append({'service_request_id':qid,'request':q,'candidate_flow_ids':[f['flow_id'] for f in fs],'exact_object_and_time_actor_completion_candidates':matches,'unique_flow_for_request':len(fs)==1,'multiple_connections_not_independent_downloads':True})

    previous=load(ROOT/'historical_C25_disposition.json');old_details=[]
    for h in previous['rows']:
        r=byid[h['current_flow_id']];f=physical[r['flow_id']];actor_side=[requests[q] for q in r['possible_frontend_request_ids']]
        old_reasons=h['current_rejection_reasons'] or []
        cause='not_downgraded_in_v1' if h['current_normal_identity_qualified'] else 'SF_input_gate_only' if old_reasons==['connection_not_SF_complete'] else 'closed_episode_gate_only' if old_reasons==['some_possible_proxy_requests_have_unknown_or_inactive_actor'] else 'other'
        old_details.append({'old_flow_id':h['old_flow_id'],'current_flow_id':r['flow_id'],'stable_flow_id':r['stable_flow_id'],'v1_normal_identity_qualified':h['current_normal_identity_qualified'],'v1_role':h['current_role'],'v1_rejection_reasons':old_reasons,'v1_exclusion_cause':cause,'conn_state':f['zeek_record'].get('conn_state'),'unknown_actor_requests':sum(q.get('known_internal_actor') is None for q in actor_side),'requests_without_activity_candidates':sum(not q.get('normal_episode_candidates') for q in actor_side),'author_attack_requests':sum(bool(q['author_attack_labels']) for q in actor_side),'incomplete_episode_ids':r['label_evidence']['incomplete_activity_candidate_ids'],'v2_label_evidence_qualified':r['label_evidence_qualified'],'v2_label':r['label'],'v2_model_input_ready':r['model_input_ready'],'v2_calibration_candidate':r['calibration_candidate'],'raw8_equal_historical':h['current_raw8_equal_historical'],'previous_normal_actor_and_request_evidence_preserved':True})
    v1_eligible=[r for r in out if r['historical_v1']['normal_identity_qualified']]
    recovered=[r for r in qualified if not r['historical_v1']['normal_identity_qualified']]
    v1_role_unknown=[r for r in out if r['historical_v1']['normal_identity_qualified'] and r['historical_v1']['role']=='unresolved']
    state_counts=Counter(r['input_quality']['conn_state'] for r in consumable)
    strata=Counter((r['capture_host'],r['date_utc'],r['business_stratum'],r['input_quality']['conn_state']) for r in consumable)
    episodes_used={e for r in consumable for e in r['episode_ids']}
    file_actions={a['action_id'] for r in files for a in r['exact_object_and_time_actor_completion_candidates']}
    summary={'schema':'whole-run-normal-v2','all_service_connection_candidates':len(out),'label_evidence_qualified':len(qualified),'model_input_ready_among_label_qualified':len(consumable),'label_qualified_but_input_pending':len(qualified)-len(consumable),'source_role':'C_candidate_whole_run','v1_qualified_count':len(v1_eligible),'v1_role_unknown_270_still_label_qualified':sum(r['label_evidence_qualified'] for r in v1_role_unknown),'v1_role_unknown_270_now_C_candidates':sum(r['calibration_candidate'] for r in v1_role_unknown),'additional_label_qualified_after_removing_unjustified_gates':len(recovered),'recovered_old_reasons':dict(Counter(reason for r in recovered for reason in r['historical_v1']['rejection_reasons'])),'normal_candidates_by_conn_state':dict(state_counts),'normal_candidates_by_host':dict(Counter(r['capture_host'] for r in consumable)),'normal_candidates_by_business_stratum':dict(Counter(r['business_stratum'] for r in consumable)),'actor_activity_candidates_referenced':len(episodes_used),'normal_connections_without_activity_episode_attribution':sum(not r['episode_ids'] for r in consumable),'normal_connections_with_incomplete_activity_candidates':sum(bool(r['label_evidence']['incomplete_activity_candidate_ids']) for r in consumable),'unique_service_file_requests':len(files),'exact_completed_normal_download_actions':len(file_actions),'file_requests_without_completed_action_match':sum(not r['exact_object_and_time_actor_completion_candidates'] for r in files),'file_request_connection_candidates':sum(bool(r['file_transfer_request_ids']) for r in consumable),'old_C25_v1_exclusion_breakdown':dict(Counter(r['v1_exclusion_cause'] for r in old_details)),'old_C25_v2_label_qualified':sum(r['v2_label_evidence_qualified'] for r in old_details),'old_C25_v2_C_candidates':sum(r['v2_calibration_candidate'] for r in old_details),'qualified_strata':[{'capture_host':h,'date_utc':d,'business_stratum':b,'conn_state':s,'connections':n} for (h,d,b,s),n in sorted(strata.items())],'scope':'Only complete russellmitchell source; normal validation may use fox/OpTC under separate source assignments. No role change to source files is applied here.','independence':'Flows and activity candidates are correlated within one simulation, shared actors and services; no iid population guarantee follows from connection count.','training_calls':0,'score_calls':0,'threshold_calls':0,'source_normalization_runs':0,'source_tables_modified':False}
    policy={'version':'v2','correction':'SF and explicitly closed actor activity were extra v1 cohort conditions, not author normal-label rules or requirements of the fixed raw8 contract. V2 removes both from label qualification.','label_rule':'Whole observation interval inside author validity; original service requests exist; every possible actor-side request belongs to a known author normal internal actor and is author-unlabelled; no candidate attack request; cloud frontend/backend request identity sets close and proxy endpoint is correct.','input_rule':'Keep conn_state, completeness, missingness, EOF and truncation as separate facts. model_input_ready requires complete raw8 plus existing context and finite replay time; it does not require SF or activity closure.','role_rule':'Candidate C allocation covers the whole russellmitchell source. Unknown labels are not normal calibration samples. V1 early/later data remain history.','whole_run_answer':'The 270 v1 role-unresolved rows already met the stricter row-level normal evidence; only linked component closure prevented temporal roles. Whole-source assignment removes that role obstacle, without converting unknown row labels.','official_normal_rule':'https://zenodo.org/records/19483937','NP_limit':'Sample dependence remains; neither recovered connections nor activity components establish iid draws or a 95% population 1% guarantee.'}
    write_rows(OUT/'connection_candidates.jsonl.gz',out);write_rows(OUT/'qualified_connections.jsonl.gz',qualified);write_rows(OUT/'C_candidates.jsonl.gz',consumable)
    write_rows(OUT/'business_episodes.jsonl.gz',[dict(e,historical_v1_role=e['role'],role='C_candidate_whole_run',source_group='ait:russellmitchell:complete_run') for e in episodes])
    write_rows(OUT/'episode_flow_links.jsonl.gz',links)
    save(OUT/'file_business_evidence.json',{'summary':{k:summary[k] for k in ('unique_service_file_requests','exact_completed_normal_download_actions','file_requests_without_completed_action_match','file_request_connection_candidates')},'requests':files})
    save(OUT/'historical_C25_disposition.json',{'v1_history_file':str(ROOT/'historical_C25_disposition.json'),'v1_history_sha256':sha(ROOT/'historical_C25_disposition.json'),'summary':{k:summary[k] for k in ('old_C25_v1_exclusion_breakdown','old_C25_v2_label_qualified','old_C25_v2_C_candidates')},'rows':old_details})
    save(OUT/'summary.json',summary);save(OUT/'policy.json',policy)
    save(OUT/'source_receipt.json',{'inputs':{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in [ROOT/'connection_candidates.jsonl.gz',ROOT/'request_evidence.jsonl.gz',ROOT/'business_episodes.jsonl.gz',ROOT/'context_rows.jsonl.gz',ROOT/'historical_C25_disposition.json',ROOT/'source_receipt.json']},'script_sha256':sha(Path(__file__))})
    assert len({r['flow_id'] for r in out})==16374
    assert all(r['label']==0 and r['label_evidence_qualified'] and r['model_input_ready'] for r in consumable)
    assert len(v1_role_unknown)==270 and all(r['calibration_candidate'] for r in v1_role_unknown)
    assert len(old_details)==25 and all(r['v2_calibration_candidate'] for r in old_details)
    save(OUT/'output_receipt.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_receipt.json'})
    print(json.dumps({k:summary[k] for k in ('label_evidence_qualified','model_input_ready_among_label_qualified','additional_label_qualified_after_removing_unjustified_gates','normal_candidates_by_conn_state','unique_service_file_requests','exact_completed_normal_download_actions','file_request_connection_candidates','old_C25_v2_C_candidates')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
