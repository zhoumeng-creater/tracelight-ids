"""Assemble source evidence, not training labels or model outputs.

Joins the complete author flow file to all connections in two byte-exact
PCAP slices; uses raw packet totals to test connection completeness.
"""
from __future__ import annotations
import collections,csv,datetime,gzip,hashlib,json,socket,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source/unraveled'
OUT=ROOT/'network_mapping'
CAMPAIGN='unraveled:apt_2021_single_campaign'
IMPLEMENTATION='unraveled:drive_by_implant_tls:framework_unresolved'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def lines(p,rows):
    with p.open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def flow_csv(p):
    widths=collections.Counter();good=[];bad=[]
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);header=next(reader)
        for row in reader:
            widths[len(row)]+=1
            if len(row)!=len(header):bad.append({'source_line':reader.line_num,'column_count':len(row),'raw_fields':row});continue
            good.append({'source_line':reader.line_num,**dict(zip(header,row))})
    return good,bad,{'source_path':str(p),'sha256':sha(p),'header_columns':len(header),'width_counts':dict(widths),'valid_rows':len(good),'malformed_rows':len(bad)}
def tcp_packets(p):
    b=p.read_bytes();i=24;rows=[]
    while i<len(b):
        sec,usec,cap,orig=struct.unpack_from('<IIII',b,i);x=b[i+16:i+16+cap];record_offset=i;i+=16+cap
        if x[12:14]!=b'\x08\x00':continue
        ip=x[14:];ihl=(ip[0]&15)*4
        if len(ip)<ihl+20 or ip[9]!=6 or struct.unpack_from('!H',ip,6)[0]&0x1fff:continue
        sport,dport,seq,ack=struct.unpack_from('!HHII',ip,ihl);flags=ip[ihl+13]
        rows.append({'key':(socket.inet_ntoa(ip[12:16]),sport,socket.inet_ntoa(ip[16:20]),dport,6),'time':sec+usec/1e6,'ip_bytes':struct.unpack_from('!H',ip,2)[0],'frame_bytes':cap,'flags':flags,'record_offset':record_offset})
    return rows
def main():
    author,bad,summary=flow_csv(SRC/'flow_20210621_net1013x.csv')
    _,bad22,summary22=flow_csv(SRC/'flow_20210622_net1013x.csv')
    dump(OUT/'author_csv_validation.json',[summary,summary22]);lines(OUT/'malformed_author_rows.jsonl',[{'source_date':'20210621',**x} for x in bad]+[{'source_date':'20210622',**x} for x in bad22])
    index=collections.defaultdict(list)
    for a in author:
        k=(a['src_ip'],int(a['src_port']),a['dst_ip'],int(a['dst_port']),int(a['protocol']));index[k].append(a)
    connections=[];links=[]
    for scope in ('normal_mail','apt_c2'):
        packets=tcp_packets(SRC/'pcap_slices'/scope/'slice.pcap')
        tls={x['uid']:x for x in map(json.loads,(OUT/scope/'zeek/ssl.log').read_text().splitlines())}
        cs=list(map(json.loads,gzip.open(OUT/scope/'all_connections.jsonl.gz','rt',encoding='utf-8')))
        for c in cs:
            f=c['raw8'];key=(c['source'],f['orig_port'],c['destination'],f['resp_port'],f['ip_protocol'])
            candidates=[a for a in index.get(key,[]) if abs(int(a['bidirectional_first_seen_ms'])/1000-c['connection_start'])<.0011]
            c.update({'implementation_id':IMPLEMENTATION if c['destination']=='10.8.10.84' else None,'author_flow_candidate_ids':[a['id'] for a in candidates],'label':None,'training_eligible':False,'role':'opened_source_qualification_development','qualified_normal_denominator_member':False})
            selected=c['source']=='10.1.3.8' and c['destination'] in ('10.8.10.84','68.65.122.10') and f['resp_port']==443
            if not selected:connections.append(c);continue
            rev=(key[2],key[3],key[0],key[1],key[4])
            ps=[p for p in packets if p['key'] in (key,rev)]
            orig=[p for p in ps if p['key']==key];resp=[p for p in ps if p['key']==rev]
            check={'author_candidate_count':len(candidates),'packet_counts_match':False,'l2_bytes_match':False,'pcap_first_last_match_author_milliseconds':False,'tcp_syn_seen':any(p['flags']&2 and not p['flags']&16 for p in orig),'tcp_fin_both_directions':all(any(p['flags']&1 for p in ds) for ds in (orig,resp)),'zeek_state':c['conn_state'],'zeek_missed_bytes':c['missed_bytes']}
            if len(candidates)==1 and ps:
                a=candidates[0]
                check.update({'packet_counts_match':(len(orig),len(resp))==(int(a['src2dst_packets']),int(a['dst2src_packets'])),'l2_bytes_match':(sum(x['frame_bytes'] for x in orig),sum(x['frame_bytes'] for x in resp))==(int(a['src2dst_bytes']),int(a['dst2src_bytes'])),'pcap_first_last_match_author_milliseconds':abs(min(p['time'] for p in ps)-int(a['bidirectional_first_seen_ms'])/1000)<.0011 and abs(max(p['time'] for p in ps)-int(a['bidirectional_last_seen_ms'])/1000)<.0011})
                c['author_annotation']={k:a[k] for k in ('Activity','Stage','DefenderResponse','Signature')}
                c['author_flow_source']={'path':str(SRC/'flow_20210621_net1013x.csv'),'sha256':summary['sha256'],'line':a['source_line'],'id':a['id']}
            complete=all(check[k] for k in ('packet_counts_match','l2_bytes_match','pcap_first_last_match_author_milliseconds','tcp_syn_seen','tcp_fin_both_directions')) and c['conn_state']=='SF' and c['missed_bytes']==0 and not c['raw8_missing_fields']
            c.update({'full_connection_observed':complete,'completeness_evidence':check,'tls_evidence':tls.get(c['uid']),'raw_packet_first':min(p['time'] for p in ps) if ps else None,'raw_packet_last':max(p['time'] for p in ps) if ps else None})
            is_c2=c['destination']=='10.8.10.84';event='unraveled:20210621:apt:c2_channel_episode' if is_c2 else 'unraveled:20210621:host10.1.3.8:privateemail_business_episode'
            c['event_ids']=[event]
            if is_c2:
                c['source_claim_label']=1 if c.get('author_annotation',{}).get('Signature')=='APT' else None
                c['label']=1 if complete and c.get('author_annotation',{}).get('Activity')=='Exfiltration over C2 channel' and c['source_claim_label']==1 else None
                c['evidence_grade']='B_author_per_flow_c2_annotation_exact_transport' if c['label']==1 else 'incomplete_or_unresolved'
                c['label_basis']='作者逐流 APT/Exfiltration over C2 channel 标注 + 精确网络匹配；没有逐命令真值。'
                c['label_provenance_limits']=['未取得作者赋标代码，不能声称独立复现其真值生成过程。','未取得这几条流对应 implant 的 PID/执行命令；具体框架未知不影响作者逐流 C2 通道标签。']
                c['training_blockers']=['本次为同一攻击链的连续片段，未装配整链且未分配 F/V 角色。']
            else:
                named=(tls.get(c['uid'],{}).get('server_name')=='privateemail.com')
                c['source_claim_label']=0 if complete and named and c.get('author_annotation',{}).get('Activity')=='Normal' else None
                c['label']=c['source_claim_label']
                c['label_basis']='作者正常浏览脚本实际成功登录回执 + 同主机/域名/TLS SNI/时间对应 + 作者 Normal 标注；会话内每个加密事务不可见。'
                c['evidence_grade']='B_actual_browser_business_whole_connection' if c['source_claim_label']==0 else 'incomplete_or_unresolved'
                c['training_blockers']=['仅截取完整浏览 episode 的前段，不构成完整正常 episode/自然正常率分母。','主机无时区日志与 PCAP 的 +7h 对齐来自 URL/时刻匹配，未取得系统时区配置。']
            c['selection_exclusions']=[] if complete else ['片段内未包含作者该流全部包；不得使用截短 raw8 冒充完整连接。']
            links.append({'event_id':event,'flow_id':c['flow_id'],'candidate_retained':True,'evaluation_complete_connection_candidate':complete,'author_flow_candidate_ids':c['author_flow_candidate_ids'],'evidence':check})
            connections.append(c)
    c2=[c for c in connections if c.get('event_ids')==['unraveled:20210621:apt:c2_channel_episode']]
    normal=[c for c in connections if c.get('event_ids')==['unraveled:20210621:host10.1.3.8:privateemail_business_episode']]
    host=SRC/'repo/data/host-logs/filebeat/10_1_3_8-ubuntu_labeled';hs=host.read_text(encoding='utf-8').splitlines()
    hostevidence=[{'source_line':i+1,'text':line} for i,line in enumerate(hs) if line.startswith('2021-06-21') and ('"url": "https://privateemail.com/"' in line or 'Successfully logged in' in line or 'Signed out' in line)]
    events=[{'event_id':'unraveled:20210621:apt:c2_channel_episode','implementation_id':IMPLEMENTATION,'campaign_id':CAMPAIGN,'kind':'author_labeled_c2_channel_activity','individual_command_ground_truth':False,'source_activity':'Exfiltration over C2 channel','candidate_flow_ids':[c['flow_id'] for c in c2],'complete_connection_candidate_flow_ids':[c['flow_id'] for c in c2 if c['full_connection_observed']],'independent_campaign_increment':1,'training_eligible':False,'role':'opened_source_qualification_development'},
      {'event_id':'unraveled:20210621:host10.1.3.8:privateemail_business_episode','campaign_id':None,'kind':'normal_browser_mail_business','source_path':str(host),'source_sha256':sha(host),'host_evidence':hostevidence,'host_log_timezone':'unspecified; +7h to UTC inferred from exact privateemail navigation and original TLS start','host_navigation_naive':'2021-06-21 08:11:58.719','host_login_success_naive':'2021-06-21 08:12:25.321','host_logout_naive':'2021-06-21 08:35:26.720','candidate_flow_ids':[c['flow_id'] for c in normal],'complete_connection_candidate_flow_ids':[c['flow_id'] for c in normal if c['full_connection_observed'] and c['source_claim_label']==0],'full_business_episode_captured':False,'normal_rate_denominator_eligible':False,'training_eligible':False,'role':'opened_source_qualification_development'}]
    lines(OUT/'connections.jsonl',connections);lines(OUT/'events.jsonl',events);lines(OUT/'links.jsonl',links)
    receipt={'all_slice_connections':len(connections),'candidate_connections':len(links),'c2_candidates':len(c2),'c2_complete':sum(c['full_connection_observed'] for c in c2),'normal_candidates':len(normal),'normal_complete_business_connections':sum(c['source_claim_label']==0 for c in normal),'candidate_incomplete':sum(not c['full_connection_observed'] for c in c2+normal),'action_episodes':2,'new_apt_campaigns':1,'verified_precise_framework_names':0,'verified_attack_download_actions':0,'connection_labels':dict(collections.Counter(str(c['label']) for c in connections)),'training_role_assigned':False,'training_eligible_connections':sum(c['training_eligible'] for c in connections),'complete_normal_denominator':False,'actual_available_at_known':False,'source_csv_validation':[summary,summary22],'output_hashes':{p.name:sha(p) for p in (OUT/'connections.jsonl',OUT/'events.jsonl',OUT/'links.jsonl')}}
    assert receipt['c2_complete']==5 and receipt['normal_complete_business_connections']==16
    assert receipt['candidate_incomplete']==4 and len({c['flow_id'] for c in connections})==len(connections)
    dump(OUT/'assembly_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
