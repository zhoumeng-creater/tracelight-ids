"""Bind four existing normal SSH actor logs to existing complete AIT captures."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import re
import yaml

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
PRIOR=WORK/'拟合来源与划分落实_20261119'
SOURCE=PRIOR/'ait_new_runs/source'
TARGETS=[('santos','internal_employee_2'),('santos','remote_employee_0'),('fox','internal_employee_3'),('fox','remote_employee_2')]
DOWNLOAD=re.compile(r'(?<![A-Za-z0-9_])(wget|curl|apt-get|apt|dnf|yum|pip3?|npm|git|scp|sftp|rsync|fetch)(?![A-Za-z0-9_])')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iso(t):return datetime.fromtimestamp(t,timezone.utc).isoformat() if t is not None else None
def save(p,r):Path(p).write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def source_rows(p):
    with Path(p).open(encoding='utf-8') as f:
        for n,s in enumerate(f,1):
            if s.strip():yield n,json.loads(s)
def gzrows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for s in f:
            if s.strip():yield json.loads(s)
def write_rows(p,rs):
    with Path(p).open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as gz:
        with io.TextIOWrapper(gz,encoding='utf-8',newline='\n') as f:
            for r in rs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def ref(member,n,r):
    out={'member':member,'line':n,'timestamp':r['timestamp'],'utc':iso(r['timestamp']),'message':r['message'],'transition_id':r.get('transition_id')}
    if 'outlines' in r:
        raw=json.dumps(r['outlines'],ensure_ascii=False,separators=(',',':')).encode()
        out['shell_output_sha256']=hashlib.sha256(raw).hexdigest();out['shell_output_lines']=len(r['outlines'])
    return out

def main():
    inputs={};sessions=[];commands=[];per_actor=[];settings={}
    for run,actor in TARGETS:
        base=SOURCE/run;server_path=base/'processing/config/servers.yaml';meta_path=base/'dataset.yaml'
        servers=yaml.safe_load(server_path.read_text(encoding='utf-8'));meta=yaml.safe_load(meta_path.read_text(encoding='utf-8'))
        begin=datetime.fromisoformat(meta['start']).replace(tzinfo=timezone.utc).timestamp();end=datetime.fromisoformat(meta['end']).replace(tzinfo=timezone.utc).timestamp()
        settings[run]={'servers':servers,'start':begin,'end':end}
        for p in (server_path,meta_path):inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
        member=f'gather/{actor}/logs/sm.log';path=base/member
        inputs[str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
        host_by_ip={v['default_ipv4_address']:k for k,v in servers.items()}
        assert 'employee' in servers[actor]['groups']
        current=None;by_transition={};counts=Counter();last=None
        for n,r in source_rows(path):
            msg=r.get('message');tid=r.get('transition_id');last=r['timestamp']
            if msg=='Connecting to SSH host':
                h=r['ssh_host'];sid=f'ait:{run}:normal_ssh_session:{actor}:{tid}'
                current={'event_id':sid,'kind':'normal_ssh_session','run':run,'actor':actor,'actor_ip':servers[actor]['default_ipv4_address'],'actor_groups':servers[actor]['groups'],'source_member':member,'target_ip':h['host'],'target_port':h['port'],'target_host_role':host_by_ip.get(h['host']),'proxy_host':h.get('proxy_host'),'actor_connect_attempt':ref(member,n,r),'actor_connected':None,'actor_disconnect_attempt':None,'actor_disconnected':None,'shell_output_evidence':[],'actor_failure_evidence':[],'command_event_ids':[],'last_session_evidence_at':r['timestamp'],'source_registered_role':'F' if run=='santos' else 'V','training_admission':False}
                sessions.append(current);counts['session_connect_attempts']+=1
            if current and r.get('current_state','').startswith('ssh'):
                current['last_session_evidence_at']=r['timestamp']
            if msg=='Connected to SSH host' and current:
                current['actor_connected']=ref(member,n,r);counts['session_connected_receipts']+=1
            if msg=='Disconnecting from SSH Host' and current:current['actor_disconnect_attempt']=ref(member,n,r)
            if msg=='Disconnected from SSH Host' and current:
                current['actor_disconnected']=ref(member,n,r);counts['session_disconnected_receipts']+=1
            if msg=='Executing command' and current:
                obj=r['command'];eid=f'ait:{run}:normal_ssh_command:{actor}:{tid}'
                cmd={'event_id':eid,'kind':'normal_ssh_command','run':run,'actor':actor,'session_event_id':current['event_id'],'target_ip':current['target_ip'],'target_host_role':current['target_host_role'],'command':obj['cmd'],'sudo':bool(obj.get('sudo')),'working_directory':obj.get('chdir'),'actor_execution_attempt':ref(member,n,r),'actor_executed_receipt':None,'shell_output_evidence':[],'download_related_command_tokens':DOWNLOAD.findall(obj['cmd']),'exit_status':None,'exit_status_observed':False,'source_registered_role':current['source_registered_role'],'training_admission':False}
                commands.append(cmd);by_transition[tid]=cmd;current['command_event_ids'].append(eid);counts['command_attempts']+=1
            if msg=='Received shell output' and current:
                evidence=ref(member,n,r)
                if tid in by_transition:by_transition[tid]['shell_output_evidence'].append(evidence)
                else:current['shell_output_evidence'].append(evidence)
            if msg=='Executed command' and tid in by_transition:
                by_transition[tid]['actor_executed_receipt']=ref(member,n,r);counts['command_executed_receipts']+=1
            if current and msg in ('State machine execution failure','Trying to recover statemachine from exception '):current['actor_failure_evidence'].append(ref(member,n,r))
        per_actor.append({'run':run,'actor':actor,'source_path':str(path),'last_actor_timestamp':last,**counts})

    flows={};capture_counts=[]
    for run in ('santos','fox'):
        for host in ('intranet_server','cloud_share'):
            directory=PRIOR/'ait_extracted'/run/host;p=directory/'connections.jsonl.gz'
            inputs[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
            total=0;ssh=0
            for f in gzrows(p):
                total+=1
                if f['native_protocol']=='tcp' and (f['raw8']['orig_port']==22 or f['raw8']['resp_port']==22):flows[f['flow_id']]=f;ssh+=1
            capture_counts.append({'run':run,'capture_host':host,'all_connections_scanned':total,'SSH_endpoint_connections':ssh,'source_path':str(p)})
    links=[];referenced=set();command_byid={r['event_id']:r for r in commands}
    for s in sessions:
        run=s['run'];t0=s['actor_connect_attempt']['timestamp'];t1=s['actor_connected']['timestamp'] if s['actor_connected'] else None
        valid=settings[run]['start']<=t0<settings[run]['end'] and t1 is not None and t1<settings[run]['end']
        s['connect_interval_in_author_simulation']=valid
        s['target_is_one_of_current_monitored_hosts']=s['target_host_role'] in ('intranet_server','cloud_share')
        s['actor_session_closed']=s['actor_disconnected'] is not None
        seeds=[]
        for fid,f in flows.items():
            if f['scenario']!=run or t1 is None:continue
            direct=f['destination_address']==s['target_ip'] and f['raw8']['resp_port']==s['target_port']
            reverse=f['source_address']==s['target_ip'] and f['raw8']['orig_port']==s['target_port']
            if not(direct or reverse):continue
            if t0<=f['connection_start']<=t1:
                seeds.append((fid,f,direct))
        s['candidate_flow_ids']=[fid for fid,_,_ in seeds]
        s['mapping_status']='no_candidate' if not seeds else 'unique_candidate' if len(seeds)==1 else 'multiple_candidates'
        s['same_monitored_host_candidate_flow_ids']=[fid for fid,f,_ in seeds if f['capture_host']==s['target_host_role']]
        s['whole_connection_normal_label_assigned']=False
        for fid,f,direct in seeds:
            peer=f['source_address'] if direct else f['destination_address']
            relationship={'event_id':s['event_id'],'flow_id':fid,'relationship':'SSH_connect_interval_candidate','observed_client_ip':peer,'actor_ip_matches_observed_client':peer==s['actor_ip'],'target_endpoint_exact':True,'connection_start_inside_actor_connect_to_connected_interval':True,'capture_host_is_target_host':f['capture_host']==s['target_host_role'],'source_actor_via_remote_VPN_possible':'remote_employee' in s['actor_groups'],'actor_identity_from_shared_VPN_endpoint_claimed':False,'normal_action_label':0,'normal_whole_connection_label':None,'basis':'Original normal actor SSH Connecting-to-Connected interval and target IP/port. All matching captures and UIDs retained; no tolerance expansion or best-candidate selection.'}
            links.append(relationship);referenced.add(fid)
        for cid in s['command_event_ids']:
            c=command_byid[cid];ct0=c['actor_execution_attempt']['timestamp'];ct1=c['actor_executed_receipt']['timestamp'] if c['actor_executed_receipt'] else ct0
            c['execution_interval_in_author_simulation']=settings[run]['start']<=ct0<=ct1<settings[run]['end']
            c['candidate_flow_ids']=[]
            for fid,f,_ in seeds:
                en=f['observed_end']
                if en is not None and f['connection_start']<=ct1 and en>=ct0:
                    c['candidate_flow_ids'].append(fid)
                    links.append({'event_id':cid,'flow_id':fid,'relationship':'command_within_parent_SSH_candidate','parent_session_event_id':s['event_id'],'command_interval_overlaps_observed_connection':True,'actor_executed_receipt_present':c['actor_executed_receipt'] is not None,'capture_host_is_target_host':f['capture_host']==s['target_host_role'],'normal_action_label':0,'normal_whole_connection_label':None})
            c['mapping_status']='no_candidate' if not c['candidate_flow_ids'] else 'unique_candidate' if len(c['candidate_flow_ids'])==1 else 'multiple_candidates'
            c['same_monitored_host_candidate_flow_ids']=[fid for fid in c['candidate_flow_ids'] if flows[fid]['capture_host']==s['target_host_role']]
    output_flows=[];complete_control_candidates=[]
    for fid in sorted(referenced):
        f=dict(flows[fid]);f['normal_management_event_ids']=sorted({l['event_id'] for l in links if l['flow_id']==fid});f['normal_whole_connection_label']=None
        host_meta=settings[f['scenario']]['servers'][f['capture_host']]
        local_addresses=set(host_meta['ipv4_addresses']+host_meta['ipv6_addresses']+['127.0.0.1','::1'])
        local=f['source_address'] in local_addresses or f['destination_address'] in local_addresses
        f['context_reference']={'path':str(PRIOR/'dataset/context_rows.jsonl.gz'),'flow_id':fid} if local else None
        f['context_unavailable_reason']=None if local else 'Neither endpoint is this capture host; no monitored-host context was constructed.'
        candidate_sessions=[]
        for s in sessions:
            if s['candidate_flow_ids']!=[fid] or not s['connect_interval_in_author_simulation'] or f['capture_host']!=s['target_host_role']:continue
            cs=[command_byid[c] for c in s['command_event_ids']]
            peer=f['source_address'] if f['destination_address']==s['target_ip'] else f['destination_address']
            if (peer==s['actor_ip'] and s['actor_disconnected'] is not None and cs and
                all(c['actor_executed_receipt'] is not None and fid in c['candidate_flow_ids'] for c in cs) and
                settings[s['run']]['start']<=s['actor_disconnected']['timestamp']<settings[s['run']]['end'] and
                f['zeek_record'].get('conn_state')=='SF' and f['raw8_complete'] and not f['after_input_eof'] and not f['during_shutdown']):
                candidate_sessions.append(s['event_id'])
        f['complete_normal_control_candidate_session_ids']=candidate_sessions
        f['normal_management_candidate_label']=0 if candidate_sessions else None
        f['training_admission']=False
        if candidate_sessions:complete_control_candidates.append(f)
        output_flows.append(f)
    same_sessions=[s for s in sessions if s['target_is_one_of_current_monitored_hosts']]
    valid_sessions=[s for s in same_sessions if s['connect_interval_in_author_simulation']]
    valid_commands=[c for c in commands if c['target_host_role'] in ('intranet_server','cloud_share') and c['execution_interval_in_author_simulation']]
    summary={'schema':'normal-SSH-management-source-assembly-v1','sources':per_actor,'capture_counts':capture_counts,'session_attempts':len(sessions),'connected_receipts':sum(s['actor_connected'] is not None for s in sessions),'disconnected_receipts':sum(s['actor_disconnected'] is not None for s in sessions),'command_attempts':len(commands),'command_executed_receipts':sum(c['actor_executed_receipt'] is not None for c in commands),'actual_command_counts':dict(Counter(c['command'] for c in commands)),'download_related_command_attempts':sum(bool(c['download_related_command_tokens']) for c in commands),'target_host_counts':dict(Counter(s['target_host_role'] for s in sessions)),'same_monitored_host_sessions':len(same_sessions),'same_monitored_host_sessions_in_valid_simulation':len(valid_sessions),'valid_same_host_session_mapping_status':dict(Counter(s['mapping_status'] for s in valid_sessions)),'valid_same_host_command_attempts':len(valid_commands),'valid_same_host_command_executed_receipts':sum(c['actor_executed_receipt'] is not None for c in valid_commands),'valid_same_host_command_mapping_status':dict(Counter(c['mapping_status'] for c in valid_commands)),'all_events':len(sessions)+len(commands),'all_links':len(links),'referenced_connections':len(output_flows),'complete_normal_control_connection_candidates':len(complete_control_candidates),'complete_candidate_rule':'Unique same-target-host connection; original actor source IP directly matches; valid-period actor connect and disconnect; every command attempt has its execution receipt on this connection; SF, complete raw8, not EOF/shutdown. This is a proposed normal-identity row, not automatic training admission.','events_by_registered_source_role':dict(Counter(s['source_registered_role'] for s in valid_sessions)),'target_network_scope':'Same monitored host means SSH target equals intranet_server or cloud_share, not merely a foreign connection seen in that capture.','normal_semantics':'Author normal actor action evidence; whole-flow normal label is not automatically propagated from a command or a shared VPN endpoint.','command_completion_semantics':'Executed command is the simulator completion receipt; no process exit code is invented. Blocking top attempts without completion remain attempts.','download_conclusion':'No actual SSH command in these four source logs invokes wget/curl/apt or the other enumerated transfer/package tools. No network download is attributed to unrelated concurrent traffic.','old_tables_modified':False,'new_source_downloads':0,'normalization_or_Zeek_reruns':0,'model_calls':0,'training_calls':0}
    assert len({e['event_id'] for e in sessions+commands})==len(sessions)+len(commands)
    assert all(l['flow_id'] in referenced for l in links)
    write_rows(ROOT/'events.jsonl.gz',sessions+commands);write_rows(ROOT/'event_flow_links.jsonl.gz',links);write_rows(ROOT/'referenced_connections.jsonl.gz',output_flows)
    write_rows(ROOT/'complete_control_candidates.jsonl.gz',complete_control_candidates)
    save(ROOT/'summary.json',summary);save(ROOT/'source_receipt.json',{'inputs':inputs,'script_sha256':sha(Path(__file__))})
    save(ROOT/'same_host_sessions.json',{'sessions':same_sessions,'valid_same_host_commands':valid_commands})
    save(ROOT/'output_receipt.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='output_receipt.json'})
    print(json.dumps({k:summary[k] for k in ('session_attempts','command_attempts','download_related_command_attempts','same_monitored_host_sessions','same_monitored_host_sessions_in_valid_simulation','valid_same_host_session_mapping_status','valid_same_host_command_attempts','valid_same_host_command_executed_receipts','all_links','referenced_connections')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
