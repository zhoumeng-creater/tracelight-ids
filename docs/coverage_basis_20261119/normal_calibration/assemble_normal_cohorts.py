"""Assemble full-service normal connection evidence and chronological cohorts.

No fitting, scoring, threshold selection, or held-out source access. Episodes
come from original normal-user state-machine boundaries, never network scores.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right
import gzip
import hashlib
import importlib.util
import io
import json
import math
import re
from urllib.parse import urlsplit, unquote, parse_qsl
import yaml

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
SOURCE=ROOT/'source'
PRIOR=WORK/'拟合来源与划分落实_20261119'
CAPTURES={'intranet_server':ROOT/'extracted/intranet_server','cloud_share':PRIOR/'ait_extracted/russellmitchell/cloud_share'}
RAW8=('ip_protocol','orig_port','resp_port','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes')
EARLY={'2022-01-21','2022-01-22'}
LATE={'2022-01-23','2022-01-24'}
ACCESS=re.compile(r'(?:(\S+:\d+) )?(\S+) - \S+ \[(.*?)\] "(.*?)" (\d+) (\S+) "(.*?)" "(.*?)"')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iso(t):return datetime.fromtimestamp(t,timezone.utc).isoformat() if t is not None else None
def day(t):return datetime.fromtimestamp(t,timezone.utc).date().isoformat()
def read_rows(p):
    p=Path(p)
    with (gzip.open(p,'rt',encoding='utf-8') if p.suffix=='.gz' else p.open(encoding='utf-8')) as f:
        for n,line in enumerate(f,1):
            if line.strip():yield n,json.loads(line)
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_rows(p,records):
    with Path(p).open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='\n') as out:
            for r in records:out.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def parse_service(host,stem):
    out=[];stats=[]
    for rotation in range(1,5):
        member=f'gather/{host}/logs/apache2/{stem}.log.{rotation}';p=SOURCE/member
        labels_path=SOURCE/member.replace('gather/','labels/',1)
        labels={r['line']:r['labels'] for _,r in read_rows(labels_path)} if labels_path.exists() else {}
        count=0;times=[]
        for n,line in enumerate(p.open(encoding='utf-8'),1):
            m=ACCESS.fullmatch(line.rstrip('\r\n'));assert m is not None,(member,n)
            vhost,client,date,request,status,size,referer,agent=m.groups()
            t=datetime.strptime(date,'%d/%b/%Y:%H:%M:%S %z').timestamp()
            parts=request.split(' ',2)
            if len(parts)==3:
                method,target,version=parts;url=urlsplit(target)
                identity=(method,unquote(url.path),tuple(sorted(parse_qsl(url.query,keep_blank_values=True))),version,int(status))
            else:
                method,target,version=None,request,None;identity=(None,request,(),None,int(status))
            r={'request_id':f'russellmitchell:{member}:{n}','member':member,'line':n,'line_sha256':hashlib.sha256(line.rstrip('\r\n').encode()).hexdigest(),'timestamp':t,'utc':iso(t),'client_ip':client,'vhost':vhost,'request':request,'method':method,'target':target,'status':int(status),'logged_bytes':None if size=='-' else int(size),'author_attack_labels':labels.get(n,[]),'identity':identity,'request_identity_sha256':hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()}
            out.append(r);count+=1;times.append(t)
        stats.append({'member':member,'sha256':sha(p),'rows':count,'first':iso(min(times)) if times else None,'last':iso(max(times)) if times else None,'attack_label_rows':len(labels),'complete_archive_member':True})
    return sorted(out,key=lambda x:(x['timestamp'],x['request_id'])),stats

def make_episodes(servers):
    episodes=[];actor_receipts=[]
    for i in range(4):
        actor=f'internal_employee_{i}';member=f'gather/{actor}/logs/sm.log';p=SOURCE/member
        assert 'internal_employee' in servers[actor]['groups']
        current=None;counts=Counter();last_time=None
        for n,x in read_rows(p):
            last_time=x['timestamp'];msg=x.get('message','');state=x.get('current_state','');transition=x.get('transition','');target=x.get('target','')
            if msg.startswith('Executing transition') and state=='selecting_activity':
                service='intranet_browsing' if transition=='wpdiscuz_goto_wordpress' else 'owncloud_file_workflow' if transition=='owncloud_go_to_owncloud' else None
                if service:
                    if current:
                        current['end']=x['timestamp'];current['end_line']=n;current['closure']='interrupted_by_next_activity';episodes.append(current)
                    current={'episode_id':f'ait:russellmitchell:normal_activity:{actor}:{x["run"]}:{x["transition_id"]}','actor':actor,'actor_ip':servers[actor]['default_ipv4_address'],'service':service,'start':x['timestamp'],'end':None,'start_line':n,'end_line':None,'member':member,'run_id':x['run'],'successful_openings':0,'download_completions':[],'closure':'not_closed'}
            if current:
                if msg in ('Opened wordpress','Opened wordpress post','Opened owncloud'):
                    current['successful_openings']+=1
                if msg=='Downloaded file':
                    current['download_completions'].append({'line':n,'timestamp':x['timestamp'],'transition_id':x.get('transition_id'),'file':x.get('file')})
                family='wpdiscuz_' if current['service']=='intranet_browsing' else 'owncloud_'
                if msg=='Moved to new state' and x.get('new_state')=='selecting_activity' and state.startswith(family):
                    current['end']=x['timestamp'];current['end_line']=n;current['closure']='explicit_return_to_activity_selection';episodes.append(current);counts[current['service']]+=1;current=None
        if current:
            current['last_logged_time']=last_time;episodes.append(current)
        actor_receipts.append({'actor':actor,'actor_ip':servers[actor]['default_ipv4_address'],'member':member,'sha256':sha(p),'author_groups':servers[actor]['groups'],'closed_episode_counts':dict(counts)})
    return episodes,actor_receipts

def index_records(records,key):
    grouped=defaultdict(list)
    for r in records:grouped[r[key]].append(r)
    return {k:(sorted(rs,key=lambda r:r['timestamp']),sorted(r['timestamp'] for r in rs)) for k,rs in grouped.items()}

def slice_records(index,key,start,end):
    rs,ts=index.get(key,([],[]))
    return rs[bisect_right(ts,start-1):bisect_right(ts,end)]

def main():
    servers=yaml.safe_load((SOURCE/'processing/config/servers.yaml').read_text(encoding='utf-8'))
    meta=yaml.safe_load((SOURCE/'dataset.yaml').read_text(encoding='utf-8'))
    begin=datetime.fromisoformat(meta['start']).replace(tzinfo=timezone.utc).timestamp()
    end=datetime.fromisoformat(meta['end']).replace(tzinfo=timezone.utc).timestamp()
    assert (day(begin),day(end))==('2022-01-21','2022-01-25')
    policy={'schema':'normal-business-cohorts-v1','source':'russellmitchell only; previously opened development source','early_candidate_C_days_UTC':sorted(EARLY),'later_normal_validation_days_UTC':sorted(LATE),'episode_rule':'Explicit selecting_activity -> target business activity start through explicit return to selecting_activity; entire original activity, not request/flow count.','connection_rule':'Full SF Zeek record, complete raw8, wholly inside simulation, not EOF/shutdown; all possible service requests have author-normal labels plus known internal actor active business episode. Shared ownCloud proxy requires bidirectional request-identity coverage and every possible frontend actor normal.','role_rule':'No cross-day episode or connection is split. Connected components of episode-flow edges stay together; cross-day/incomplete/conflicting components are unresolved.','quality_filter':'No rejection based on truncation, packet-length normalization, packet count, byte count, or model score.','normal_label_scope':'Conditional B-grade service-connection normality given complete published log members and actor evidence; no unique client TCP request attribution is implied by coarse Apache seconds.','services':['intranet_browsing','owncloud_file_workflow'],'strata':'OwnCloud exact file-transfer request flows distinguished from interface/API workflow flows.','independent_sample_claim':False,'population_one_percent_guarantee':False,'threshold_or_model_calls':0,'official_label_rule_source':'https://zenodo.org/records/19483937'}
    save(ROOT/'cohort_policy.json',policy)
    intranet,istat=parse_service('intranet_server','intranet.smith.russellmitchell.com-access')
    backend,bstat=parse_service('cloud_share','other_vhosts_access')
    proxy,pstat=parse_service('webserver','cloud.smith.russellmitchell.com-access')
    episodes,actors=make_episodes(servers);ep_byid={e['episode_id']:e for e in episodes}
    ep_by_actor=defaultdict(list)
    for e in episodes:
        e['start_utc']=iso(e['start']);e['end_utc']=iso(e['end'])
        e['complete_valid_day']=bool(e['end'] is not None and e['closure']=='explicit_return_to_activity_selection' and begin<=e['start']<=e['end']<end and day(e['start'])==day(e['end']) and e['successful_openings']>0)
        e['date_utc']=day(e['start'])
        ep_by_actor[(e['actor_ip'],e['service'])].append(e)
    known={a['actor_ip']:a['actor'] for a in actors}
    def actor_episodes(request,service):
        # Keep incomplete/cross-day alternatives too; they cannot qualify a
        # normal identity but remain part of the source association evidence.
        return [e['episode_id'] for e in ep_by_actor[(request['client_ip'],service)] if e['start']<request['timestamp']+1 and request['timestamp']<=(e['end'] if e['end'] is not None else e.get('last_logged_time',e['start']))]
    for rs,service in [(intranet,'intranet_browsing'),(proxy,'owncloud_file_workflow')]:
        for r in rs:
            r['normal_episode_candidates']=actor_episodes(r,service)
            r['known_internal_actor']=known.get(r['client_ip'])
            r['normal_actor_service_supported']=bool(r['normal_episode_candidates'] and all(ep_byid[e]['complete_valid_day'] for e in r['normal_episode_candidates']) and not r['author_attack_labels'] and begin<=r['timestamp']<end)
    in_index=index_records(intranet,'client_ip');back_index=index_records(backend,'client_ip')
    proxy_index={'all':(proxy,[r['timestamp'] for r in proxy])}
    all_connections={};candidate_rows=[];context_inputs=[];capture_counts=[]
    for host,directory in CAPTURES.items():
        connections=[r for _,r in read_rows(directory/'connections.jsonl.gz')]
        addresses=set(servers[host]['ipv4_addresses']+servers[host]['ipv6_addresses']+['127.0.0.1','::1'])
        server=servers[host]['default_ipv4_address'];host_views=0;service_count=0
        for f in connections:
            fid=f['flow_id'];all_connections[fid]=f
            src,dst=f['source_address'],f['destination_address'];local_src=src in addresses;local_dst=dst in addresses
            if local_src or local_dst:
                host_views+=1
                context_inputs.append({'group':f'ait:russellmitchell:{host}:complete_capture','host':f'ait:russellmitchell:{host}','peer':dst if local_src else src,'flow_id':f['stable_flow_id'],'physical_flow_id':f['stable_flow_id'],'measurement_flow_id':fid,'available_at':f['replay_export_time'],'flow_end_at':f['observed_end'],'raw8':[None if f['raw8'][k] is None or f['raw8'][k]<0 else f['raw8'][k] for k in RAW8]})
            if f['native_protocol']!='tcp':continue
            if dst==server and f['raw8']['resp_port'] in (80,443):client=src
            elif src==server and f['raw8']['orig_port'] in (80,443):client=dst
            else:continue
            service_count+=1;st=f['connection_start'];en=f['observed_end'];reasons=[];epids=set();identities=[];front=[]
            whole=[] if en is None else slice_records(in_index if host=='intranet_server' else back_index,client,st,en)
            if not whole:reasons.append('no_service_request_in_complete_connection_envelope')
            if any(r['author_attack_labels'] for r in whole):reasons.append('author_attack_request_possible_in_connection')
            if host=='intranet_server':
                for r in whole:
                    epids.update(r['normal_episode_candidates'])
                if any(not r['normal_actor_service_supported'] for r in whole):reasons.append('some_service_requests_lack_normal_actor_activity_evidence')
            else:
                if client!=servers['webserver']['default_ipv4_address']:reasons.append('not_expected_cloud_reverse_proxy_endpoint')
                front=[] if en is None else slice_records(proxy_index,'all',st,en)
                if not front:reasons.append('no_frontend_requests_in_complete_connection_envelope')
                if any(not r['normal_actor_service_supported'] for r in front):reasons.append('some_possible_proxy_requests_have_unknown_or_inactive_actor')
                front_keys=defaultdict(list);back_keys=defaultdict(list)
                for r in front:front_keys[r['request_identity_sha256']].append(r);epids.update(r['normal_episode_candidates'])
                for r in whole:back_keys[r['request_identity_sha256']].append(r)
                if set(front_keys)!=set(back_keys):reasons.append('frontend_backend_request_identity_sets_not_closed')
                for key,rs in back_keys.items():
                    fs=front_keys.get(key,[])
                    identities.append({'request_identity_sha256':key,'backend_request_ids':[r['request_id'] for r in rs],'frontend_request_ids':[r['request_id'] for r in fs],'possible_clients':sorted({r['client_ip'] for r in fs})})
            if not epids:reasons.append('no_supported_normal_business_episode')
            if en is None or not(begin<=st<=en<end):reasons.append('whole_connection_not_inside_author_simulation')
            if f['zeek_record'].get('conn_state')!='SF':reasons.append('connection_not_SF_complete')
            if not f['raw8_complete']:reasons.append('raw8_incomplete')
            if f['after_input_eof'] or f['during_shutdown']:reasons.append('completion_depends_on_capture_EOF_or_shutdown')
            file_requests=[r for r in whole if r['method']=='GET' and r['status']==200 and (unquote(urlsplit(r['target']).path).startswith('/remote.php/webdav/') or urlsplit(r['target']).path=='/index.php/apps/files/ajax/download.php')]
            candidate_rows.append({'flow_id':fid,'stable_flow_id':f['stable_flow_id'],'capture_host':host,'service':'intranet_browsing' if host=='intranet_server' else 'owncloud_file_workflow','business_stratum':'intranet_browser_requests' if host=='intranet_server' else 'owncloud_file_transfer_request' if file_requests else 'owncloud_interface_or_API','client_endpoint':client,'connection_start':st,'observed_end':en,'start_utc':iso(st),'end_utc':iso(en),'date_utc':day(st),'cross_day_connection':en is not None and day(st)!=day(en),'normal_identity_qualified':not reasons,'label':0 if not reasons else None,'rejection_reasons':sorted(set(reasons)),'episode_ids':sorted(epids),'service_request_ids':[r['request_id'] for r in whole],'possible_frontend_request_ids':[r['request_id'] for r in front],'file_transfer_request_ids':[r['request_id'] for r in file_requests],'possible_actor_ids':sorted({ep_byid[e]['actor'] for e in epids}),'transaction_identity_candidates':identities,'flow_mapping_cardinality':'whole_connection_all_request_candidates_normal' if not reasons else 'unresolved','raw8':f['raw8'],'raw8_missing_fields':f['raw8_missing_fields'],'capture_provenance':f['capture_provenance'],'replay_export_time':f['replay_export_time'],'after_input_eof':f['after_input_eof'],'during_shutdown':f['during_shutdown'],'source_connections_path':str(directory/'connections.jsonl.gz'),'source_extraction_receipt':str(directory/'extraction_receipt.json')})
        capture_counts.append({'capture_host':host,'all_connections':len(connections),'host_view_connections':host_views,'service_endpoint_candidates':service_count,'connection_source_sha256':sha(directory/'connections.jsonl.gz'),'extraction_receipt_sha256':sha(directory/'extraction_receipt.json'),'earliest_connection_start':iso(min(f['connection_start'] for f in connections)),'latest_observed_end':iso(max(f['observed_end'] for f in connections if f['observed_end'] is not None))})

    # Dependency components join every episode that could share a connection,
    # including unresolved candidates. Only chronology determines role.
    parent={eid:eid for eid in ep_byid}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        aa,bb=find(a),find(b)
        if aa!=bb:parent[max(aa,bb)]=min(aa,bb)
    for r in candidate_rows:
        for eid in r['episode_ids'][1:]:union(r['episode_ids'][0],eid)
    components=defaultdict(list)
    for eid in ep_byid:components[find(eid)].append(eid)
    component_roles={};component_records=[]
    for anchor,eids in components.items():
        comp_id='ait:russellmitchell:normal_dependency:'+hashlib.sha256('\n'.join(sorted(eids)).encode()).hexdigest()
        days={ep_byid[e]['date_utc'] for e in eids};complete=all(ep_byid[e]['complete_valid_day'] for e in eids)
        linked=[r for r in candidate_rows if any(e in eids for e in r['episode_ids'])]
        crossday=any(r['cross_day_connection'] for r in linked)
        role='C_candidate' if complete and not crossday and days<=EARLY else 'normal_time_validation' if complete and not crossday and days<=LATE else 'unresolved'
        component_roles[anchor]=(comp_id,role)
        component_records.append({'dependency_component_id':comp_id,'episode_ids':sorted(eids),'dates_utc':sorted(days),'role':role,'all_episodes_complete':complete,'cross_day_connection_present':crossday,'candidate_flow_ids':[r['flow_id'] for r in linked],'qualified_flow_ids':[r['flow_id'] for r in linked if r['normal_identity_qualified']]})
    for e in episodes:
        cid,role=component_roles[find(e['episode_id'])];e.update(dependency_component_id=cid,role=role)
    for r in candidate_rows:
        if r['episode_ids']:
            cid,role=component_roles[find(r['episode_ids'][0])]
        else:cid,role=None,'unresolved'
        r.update(dependency_component_id=cid,chronological_component_role=role,role=role if r['normal_identity_qualified'] and not r['cross_day_connection'] else 'unresolved')

    module_path=ROOT/'code/docs/unified_events_20261119/split_method/prefix_context.py'
    spec=importlib.util.spec_from_file_location('fixed_normal_context',module_path);module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    result=module.build_prefix_context(context_inputs,time_mode='available_at');context=result.pop('rows')
    by_stable={r['physical_flow_id']:r for r in context}
    qualified=[]
    for r in candidate_rows:
        if not r['normal_identity_qualified']:continue
        c=by_stable[r['stable_flow_id']]
        qualified.append({**r,'features8':c['current_raw8_log1p'],'features27':c['model_features'],'context_reference':{'path':str(ROOT/'context_rows.jsonl.gz'),'physical_flow_id':r['stable_flow_id'],'group':c['group'],'history_flow_count':c['history_flow_count'],'clock':'offline Zeek network_time, not actual detector receipt'}})
    role_ids={role:{r['stable_flow_id'] for r in qualified if r['role']==role} for role in ('C_candidate','normal_time_validation')}
    assert not role_ids['C_candidate']&role_ids['normal_time_validation']
    assert len({r['flow_id'] for r in candidate_rows})==len(candidate_rows)
    assert all(not r['cross_day_connection'] for r in qualified if r['role']!='unresolved')
    strata=Counter((r['role'],r['date_utc'],r['capture_host'],r['business_stratum']) for r in qualified)
    used_ep={eid for r in qualified if r['role']!='unresolved' for eid in r['episode_ids']}
    used_components={r['dependency_component_id'] for r in qualified if r['role']!='unresolved'}
    sums={'schema':'normal-business-cohorts-v1','captures':capture_counts,'source_log_members':istat+bstat+pstat,'actor_sources':actors,'business_episodes_total':len(episodes),'business_episodes_complete_valid_day':sum(e['complete_valid_day'] for e in episodes),'business_episodes_with_consumable_qualified_connections':len(used_ep),'dependency_components_with_consumable_connections':len(used_components),'whole_service_connection_candidates':len(candidate_rows),'normal_identity_qualified_connections':len(qualified),'role_counts':dict(Counter(r['role'] for r in qualified)),'episode_counts_by_role':dict(Counter(e['role'] for e in episodes if e['episode_id'] in used_ep)),'dependency_component_counts_by_role':dict(Counter(c['role'] for c in component_records if c['dependency_component_id'] in used_components)),'qualified_strata':[{'role':role,'date_utc':d,'capture_host':host,'business_stratum':business,'connections':n} for (role,d,host,business),n in sorted(strata.items())],'candidate_rejection_reasons':dict(Counter(reason for r in candidate_rows for reason in r['rejection_reasons'])),'context_full_background_rows':len(context),'cross_role_stable_identity_overlap':0,'cross_role_dependency_component_overlap':0,'prior_C25_preserved_path':str(PRIOR/'calibration/candidate_connections.json'),'model_calls':0,'training_calls':0,'threshold_calculations':0,'population_1pct_95pct_guarantee_claimed':False,'dependence':'Four simulated users, one AIT run, shared services and generation mechanism. Flows within activities and dependency components are correlated; dates and actors are not proven independent population draws.','coverage_limit':'Only evidence-supported internal-actor intranet browsing and ownCloud workflows; unknown/VPN actors and incomplete or mixed envelopes remain unresolved, not normal.','source_receipt_sha256':sha(ROOT/'source_receipt.json'),'code_source_receipt_sha256':sha(ROOT/'code_source_receipt.json'),'script_sha256':sha(Path(__file__))}
    write_rows(ROOT/'business_episodes.jsonl.gz',episodes)
    write_rows(ROOT/'dependency_components.jsonl.gz',component_records)
    write_rows(ROOT/'connection_candidates.jsonl.gz',candidate_rows)
    write_rows(ROOT/'qualified_connections.jsonl.gz',qualified)
    write_rows(ROOT/'C_candidates.jsonl.gz',[r for r in qualified if r['role']=='C_candidate'])
    write_rows(ROOT/'normal_time_validation.jsonl.gz',[r for r in qualified if r['role']=='normal_time_validation'])
    write_rows(ROOT/'request_evidence.jsonl.gz',[{k:v for k,v in r.items() if k!='identity'} for rs in (intranet,backend,proxy) for r in rs])
    write_rows(ROOT/'context_inputs.jsonl.gz',context_inputs);write_rows(ROOT/'context_rows.jsonl.gz',context)
    save(ROOT/'context_audit.json',result);save(ROOT/'summary.json',sums)
    save(ROOT/'output_receipt.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name not in ('output_receipt.json',)})
    print(json.dumps({k:sums[k] for k in ('whole_service_connection_candidates','normal_identity_qualified_connections','role_counts','business_episodes_with_consumable_qualified_connections','dependency_components_with_consumable_connections','context_full_background_rows')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
