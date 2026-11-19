"""Map fixed action evidence onto full-capture Zeek exports; retain ambiguity."""
from __future__ import annotations
import argparse,bisect,collections,datetime as dt,gzip,hashlib,json,math
from pathlib import Path
from urllib.parse import urlsplit
import yaml
from fetch_sources import ROOT,write_json
from assemble_action_candidates import apache

EXTRACTED=ROOT.parent/'ait_extracted'
RAW8=('ip_protocol','orig_port','resp_port','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes')
def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    if not p.exists():return []
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]
def save_rows(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    op=gzip.open if p.suffix=='.gz' else open
    with op(p,'wt',encoding='utf-8',newline='\n') as f:
        for x in data:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def overlap(f,start,end):
    last=f.get('observed_end',f.get('zeek_reported_end'))
    return last is not None and f['connection_start']<=end and last>=start
def service_endpoint(f,ep):
    r=f['raw8'];src=f['source_address'];dst=f['destination_address']
    direct=src==ep['client_ip'] and dst==ep['server_ip'] and r['resp_port']==ep['server_port']
    reverse=dst==ep['client_ip'] and src==ep['server_ip'] and r['orig_port']==ep['server_port']
    return direct or reverse
def pos_http_matches(event,f,http):
    target=urlsplit(event['requested_url']);wanted=target.path+('?' +target.query if target.query else '')
    return [r for r in http.get(f['uid'],[]) if r.get('method')=='GET' and r.get('host','').split(':')[0]==target.hostname and r.get('uri')==wanted]

def map_run(run,parameters):
    base=ROOT/'source'/run;dest=ROOT/'network_mapping'/run
    metadata=yaml.safe_load((base/'dataset.yaml').read_text(encoding='utf-8'))
    start=dt.datetime.fromisoformat(metadata['start']).replace(tzinfo=dt.timezone.utc).timestamp();end=dt.datetime.fromisoformat(metadata['end']).replace(tzinfo=dt.timezone.utc).timestamp()
    originals=read_json(ROOT/f'{run}_action_candidates.json')
    actor_context_path=ROOT/'all_actor_attribution.json'
    actor_context=read_json(actor_context_path).get('events',{}) if actor_context_path.exists() else {}
    original_events={e['episode_id']:e for key in ('control_actions','download_actions','reverse_shell','normal_actions') for e in originals[key]}
    byhost={};http_byhost={};ssl_byhost={};service_byhost={};receipts={};background={};service_provenance={}
    source_receipts=read_json(ROOT/'small_source_receipt.json')
    official_path=ROOT/'metadata/zenodo_record_19483937.json'
    official_rule={'url':'https://zenodo.org/records/19483937','local_metadata':str(official_path),'local_metadata_sha256':hashlib.sha256(official_path.read_bytes()).hexdigest(),'source_field':'metadata.description','rule':'Author explicitly defines gathered log events without a corresponding labels JSON object as normal; apply only to the supplied log/label pair and simulation interval.'}
    for host in ('intranet_server','cloud_share'):
        p=EXTRACTED/run/host
        if not (p/'extraction_receipt.json').exists():continue
        receipts[host]=read_json(p/'extraction_receipt.json')
        fs=rows(p/'connections.jsonl.gz')
        for f in fs:
            assert set(f['raw8'])==set(RAW8)
            assert f['flow_id']==f'ait:{run}:{host}:{f["uid"]}'
            f['source_address']=f.get('source_address',f.get('source'));f['destination_address']=f.get('destination_address',f.get('destination'))
            f['observed_end']=f.get('observed_end',f.get('zeek_reported_end'))
        byhost[host]=fs
        ips=set(next(x['all_local_addresses'] for x in read_json(ROOT/'capture_host_inventory.json')['hosts'] if x['run']==run and x['capture_host']==host))
        background[host]={'normalized_path':str(p/'connections.jsonl.gz'),'full_capture_connections':len(fs),'involving_declared_host':sum(f['source_address'] in ips or f['destination_address'] in ips for f in fs),'not_involving_declared_host':sum(f['source_address'] not in ips and f['destination_address'] not in ips for f in fs)}
        http=collections.defaultdict(list);ssl=collections.defaultdict(list)
        for x in rows(p/'zeek/http.log'):http[x['uid']].append(x)
        for x in rows(p/'zeek/ssl.log'):ssl[x['uid']].append(x)
        http_byhost[host]=http;ssl_byhost[host]=ssl
        service,stats=apache(base,host,start,end)
        assert not stats.get('unparsed',0),stats
        service_provenance[host]={'author_default_normal_rule':official_rule,'simulation_interval':[start,end],'scan_stats':stats,'complete_source_members':[{'member':x['member'],'bytes':x['bytes'],'sha256':x['sha256']} for x in source_receipts if x['archive']==run+'_no-pcaps.zip' and (x['member'].startswith(f'gather/{host}/logs/apache2/') or x['member'].startswith(f'labels/{host}/logs/apache2/'))],'scope':'All original access-log records in the supplied complete members within simulation interval, not only known normal-action rows.','proxy_attribution':'A shared proxy/VPN endpoint does not identify one actor; actor/transaction assignment remains conditional unless additional evidence resolves it.'}
        byclient=collections.defaultdict(list)
        for x in service:byclient[x['client']].append(x)
        service_byhost[host]={k:sorted(v,key=lambda r:r['timestamp']) for k,v in byclient.items()}
    actor_path=base/'gather/attacker_0/logs/ait.aecid.attacker.wpdiscuz/sm.log'
    actor=[(i,json.loads(s)) for i,s in enumerate(actor_path.open(encoding='utf-8'),1)]
    events=[];links=[];candidate_flows={};event_lookup={}
    for param in [x for x in parameters if x['run']==run]:
        eid=param['episode_id'];host=param['capture_host'];kind=param['kind'];original=original_events[eid]
        event={'event_id':eid,'episode_id':eid,'chain_id':param.get('chain_id'),'run':run,'capture_host':host,'kind':kind,'action_label':0 if kind.startswith('normal_') else 1,'candidate_flow_ids':[],'mapping_status':'pending_capture' if host not in byhost else 'no_candidate','action_evidence':original,'mapping_parameters':param,'connection_label_from_action_only':False,'all_normal_actor_attribution':actor_context.get(eid)}
        for f in byhost.get(host,[]):
            if f['raw8']['ip_protocol']!=6:continue
            evidence={};is_candidate=False
            if kind in ('webshell_command','normal_intranet_page_retrieval','normal_owncloud_file_download'):
                if not service_endpoint(f,param['observed_service_endpoint']):continue
                w=param['capture_candidate_overlap_window'];is_candidate=overlap(f,w['start'],w['end'])
                a=param['actor_interval'];rq=param['service_request_time_interval']
                evidence={'endpoint_match':True,'action_overlap':overlap(f,a['start'],a['end']),'request_second_overlap':overlap(f,rq['start_inclusive'],rq['end_exclusive']),'association_basis':'actual observed endpoints + full connection interval overlaps; HTTP client port absent in Apache, all candidates retained'}
            elif kind=='attack_download':
                ep=param['observed_service_endpoint'];a=param['actor_interval']
                if f['source_address']!=ep['client_ip'] or f['raw8']['resp_port']!=ep['server_port']:continue
                if ep['server_ip'] is not None and f['destination_address']!=ep['server_ip']:continue
                is_candidate=overlap(f,a['start'],a['end'])
                http_matches=pos_http_matches(param,f,http_byhost[host]);names=sorted({r.get('server_name') for r in ssl_byhost[host].get(f['uid'],[]) if r.get('server_name')})
                evidence={'endpoint_match':True,'action_overlap':is_candidate,'http_exact_request':http_matches,'http_response_status_observed':[r.get('status_code') for r in http_matches],'ssl_server_names':names,'requested_host_sni':param['initial_server_name'] in names,'association_basis':'victim-originated target service + action interval; exact HTTP and SNI evidence retained separately; no HTTP response status inferred'}
            elif kind=='reverse_shell_channel':
                socket=param['attacker_listen_socket'];listener=socket[0].removeprefix('::ffff:');port=socket[1]
                if f['source_address']!=param['victim_inside_ip'] or f['destination_address']!=listener or f['raw8']['resp_port']!=port:continue
                receipt=param['connection_receipt']['timestamp'];starts=[(n,x) for n,x in actor if x.get('message')=='Starting reverse shell' and x['timestamp']<=receipt]
                if not starts:continue
                n,x=max(starts,key=lambda z:z[1]['timestamp']);is_candidate=overlap(f,x['timestamp'],receipt)
                z=f.get('zeek_record',{});history=z.get('history','');state=z.get('conn_state')
                rejected=state=='REJ' and z.get('orig_bytes')==0 and z.get('resp_bytes')==0
                established=all(c in history for c in 'ShADd') and z.get('orig_bytes',0)>0 and z.get('resp_bytes',0)>0
                evidence={'endpoint_match':True,'action_overlap':is_candidate,'starting_reverse_shell':{'member':'gather/attacker_0/logs/ait.aecid.attacker.wpdiscuz/sm.log','line':n,'timestamp':x['timestamp']},'connected_receipt':param['connection_receipt'],'transport_evidence':{'conn_state':state,'history':history,'orig_application_bytes':z.get('orig_bytes'),'resp_application_bytes':z.get('resp_bytes'),'successful_transport_candidate':established,'excluded_as_successful_transport':rejected,'exclusion_reason':'REJ + zero application bytes in both directions; no established channel' if rejected else None},'association_basis':'victim inside IP to actor exact listening IP/port, starting-to-connected window; no assumption about NAT source-port preservation'}
            if not is_candidate:continue
            fid=f['flow_id'];event['candidate_flow_ids'].append(fid);candidate_flows[fid]=(host,f)
            links.append({'event_id':eid,'flow_id':fid,'run':run,'capture_host':host,'relationship':'candidate','evidence':evidence,'unique_for_event':False})
        if event['candidate_flow_ids']:event['mapping_status']='unique_candidate' if len(event['candidate_flow_ids'])==1 else 'multiple_candidates'
        event['evaluation_candidate_flow_ids']=list(event['candidate_flow_ids'])
        event['candidate_exclusions']=[]
        event['candidate_set_semantics']={'mutually_exclusive':False,'one_of_envelope':'conservative min/max envelope; an action may involve multiple connections, including redirect and transfer legs'}
        if kind=='reverse_shell_channel' and event['candidate_flow_ids']:
            related=[l for l in links if l['event_id']==eid]
            good=[l['flow_id'] for l in related if l['evidence']['transport_evidence']['successful_transport_candidate']]
            rejected=[l['flow_id'] for l in related if l['evidence']['transport_evidence']['excluded_as_successful_transport']]
            event['excluded_successful_transport_flow_ids']=rejected
            event['resolved_successful_transport_flow_ids']=good if len(good)==1 and len(good)+len(rejected)==len(related) else []
            event['evaluation_candidate_flow_ids']=[f for f in event['candidate_flow_ids'] if f not in rejected]
            event['candidate_exclusions']=[{'flow_id':f,'scope':'successful_reverse_shell_channel_evaluation','reason':'Zeek REJ, history without completed handshake, zero application bytes both directions; original candidate/link retained'} for f in rejected]
            event['transport_resolution_basis']='Complete handshake and bidirectional application data + actor connected receipt; every other retained candidate must be an explicitly rejected zero-data attempt.'
        events.append(event);event_lookup[eid]=event
    byflow=collections.defaultdict(list)
    for link in links:
        link['unique_for_event']=len(event_lookup[link['event_id']]['candidate_flow_ids'])==1
        byflow[link['flow_id']].append(link)
    out=[]
    servers=yaml.safe_load((base/'processing/config/servers.yaml').read_text(encoding='utf-8'))
    for fid,(host,f) in candidate_flows.items():
        ls=byflow[fid];es=[event_lookup[x['event_id']] for x in ls];normal=[e for e in es if e['action_label']==0];positive=[e for e in es if e['action_label']==1]
        server=servers[host]['default_ipv4_address'];client=f['source_address'] if f['destination_address']==server else f['destination_address'] if f['source_address']==server else None
        whole=[];st=f['connection_start'];en=f['observed_end']
        if client is not None and en is not None:
            candidates=service_byhost[host].get(client,[]);times=[x['timestamp'] for x in candidates]
            for x in candidates[bisect.bisect_right(times,st-1):bisect.bisect_right(times,en)]:
                if x['timestamp']+1>st and x['timestamp']<=en:whole.append(x)
        labelled=[x for x in whole if x['labels']]
        missing=[k for k,v in f['raw8'].items() if v is None or isinstance(v,(int,float)) and v<0]
        state=f.get('zeek_record',{}).get('conn_state',f.get('conn_state'))
        valid_interval=en is not None and start<=st<=en<end
        normal_actors_resolved=host!='cloud_share' or bool(normal) and all(e.get('all_normal_actor_attribution') and e['all_normal_actor_attribution']['all_normal_actor_logs_read'] and e['all_normal_actor_attribution']['all_candidates_author_normal'] and e['all_normal_actor_attribution']['known_named_actor_retained'] for e in normal)
        label=None;basis=[];grade='unresolved'
        if normal and not positive and whole and not labelled and valid_interval and state=='SF' and not missing and normal_actors_resolved:
            label=0;grade='B_author_normal_full_service_log_with_named_actor_anchor'
            basis=['normal actor completion + exact Apache match anchors candidate','all possible Apache requests over full connection are author-unlabeled normal','no positive action candidate shares connection','SF connection wholly inside simulation; complete raw8']
        if positive and not normal and valid_interval and not missing:
            exact_http=[l for l in ls if l['evidence'].get('http_exact_request')]
            unique_control=[l for l in ls if l['unique_for_event'] and event_lookup[l['event_id']]['kind']=='webshell_command']
            unique_reverse=[l for l in ls if event_lookup[l['event_id']]['kind']=='reverse_shell_channel' and event_lookup[l['event_id']].get('resolved_successful_transport_flow_ids')==[fid]]
            if exact_http and state=='SF':
                label=1;grade='actor_completed_download_plus_exact_http_request_uid';basis=['actor completed-download receipt + victim source + actor interval + exact URL HTTP GET on this UID + SF connection','HTTP response status is not inferred if omitted by Zeek']
            elif unique_control and whole and all(x['labels'] for x in whole):
                label=1;grade='unique_control_candidate_plus_full_connection_attack_service_labels';basis=['unique control interval/endpoint candidate','all potential Apache requests during whole connection have author attack labels']
            elif unique_reverse:
                label=1;grade='actor_confirmed_reverse_shell_successful_transport';basis=['victim-to-attacker exact listener tuple during start-to-connected interval','one completed-handshake bidirectional-data candidate; every other retained candidate is REJ with zero appdata']
        reasons={'normal_anchor_events':len(normal),'positive_candidate_events':len(positive),'full_connection_service_requests':len(whole),'full_connection_author_attack_requests':len(labelled),'valid_simulation_interval':valid_interval,'conn_state':state,'raw8_missing_fields':missing,'whole_connection_requests':whole,'label_is_conditional_on_author_service_log_completeness':label==0,'service_provenance':service_provenance[host],'transaction_identity_not_implied_by_shared_proxy':host=='cloud_share','all_normal_actor_context_checked':normal_actors_resolved,'all_actor_context_file':str(actor_context_path) if host=='cloud_share' else None}
        out.append({'flow_id':fid,'run':run,'capture_host':host,'uid':f['uid'],'flow_key_id':f.get('flow_key_id'),'raw8':f['raw8'],'source_address':f['source_address'],'destination_address':f['destination_address'],'connection_start':st,'observed_end':en,'replay_export_time':f.get('replay_export_time'),'after_input_eof':f.get('after_input_eof'),'during_shutdown':f.get('during_shutdown'),'label':label,'label_basis':basis,'evidence_grade':grade,'event_ids':[x['event_id'] for x in ls],'qualification_reasons':reasons,'source_provenance':{'normalized_path':str(EXTRACTED/run/host/'connections.jsonl.gz'),'extraction_receipt':str(EXTRACTED/run/host/'extraction_receipt.json'),'source_scope':f.get('source_scope'),'native_protocol':f.get('native_protocol'),'zeek_record':f.get('zeek_record')}})
    save_rows(dest/'events.jsonl',events);save_rows(dest/'links.jsonl',links);save_rows(dest/'connections.jsonl.gz',out)
    positive_by_kind=collections.Counter()
    for f in out:
        if f['label']==1:
            for kind in {event_lookup[eid]['kind'] for eid in f['event_ids']}:positive_by_kind[kind]+=1
    summary={'run':run,'events':len(events),'event_status':dict(collections.Counter(e['mapping_status'] for e in events)),'links':len(links),'candidate_connections':len(out),'connection_labels':dict(collections.Counter(str(x['label']) for x in out)),'positive_connections_by_event_kind':dict(positive_by_kind),'full_background_upstream':background,'events_complete_for_fixed_source_set':len(events)==len(original_events),'connections_scope':'All event-candidate connections, including label-null. Merge into upstream full background by flow_id; other background remains label-null.','normal_qualification_scope':'B-grade conditional normal: whole connection potential service requests, official default-normal rule, named normal action anchor. Shared-proxy transaction attribution is not unique. Excludes mixed/positive-linked/unfinished/out-of-simulation and None/negative raw8.','no_training_or_scoring':True}
    write_json(dest/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',choices=('santos','fox','all'),default='all');a=ap.parse_args()
    parameters=read_json(ROOT/'network_mapping_parameters.json')['events_parameters']
    for run in ('santos','fox') if a.run=='all' else (a.run,):map_run(run,parameters)
