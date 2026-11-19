"""Map complete author-flow annotations to full Zeek capture via raw packet IDs.

All tuple/time candidates are preserved. Label consistency is checked across
the whole possible packet set, so splits/reuse need not be forced one-to-one.
This script never computes or reads a model score.
"""
from __future__ import annotations
import argparse,bisect,collections,csv,functools,gzip,hashlib,ipaddress,json,socket,struct,time
from pathlib import Path
from assemble_unraveled import ROOT,SRC,flow_csv,sha,dump
OUT=ROOT/'full_capture'
CAMPAIGN='unraveled:apt_2021_single_campaign'

def readgz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def write_rows(p,rows):
    opener=gzip.open if p.suffix=='.gz' else open
    kwargs={'compresslevel':1} if p.suffix=='.gz' else {}
    with opener(p,'wt',encoding='utf-8',newline='\n',**kwargs) as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
@functools.lru_cache(maxsize=65536)
def address(s):return str(ipaddress.ip_address(s))
def key(proto,src,sp,dst,dp):
    if proto in (1,58):sp=dp=0
    a,b=(address(src),int(sp or 0)),(address(dst),int(dp or 0))
    return (proto,*a,*b) if a<=b else (proto,*b,*a)
def akey(a):return key(int(a['protocol']),a['src_ip'],int(a['src_port']),a['dst_ip'],int(a['dst_port']))
def zkey(c):
    f=c['raw8'];return key(f['ip_protocol'],c['source'],f['orig_port'],c['destination'],f['resp_port'])
def truth(a):
    if a['Activity']=='Normal' and a['Signature']=='None':return 0,'normal'
    if a['Signature'] in ('APT','AA','SH') and a['Activity']!='Normal':
        return 1,'c2_channel' if a['Activity'] in ('Exfiltration over C2 channel','Encrypted Channel: Symmetric Cryptography') else 'other_attack'
    return None,'unknown'
def stratum(a):
    app=a['application_name'];p=int(a['protocol']);port=int(a['dst_port'])
    if app.startswith('TLS'):return 'tls_application'
    if app.startswith('HTTP'):return 'http_application'
    if app.startswith('DNS'):return 'dns'
    if app.startswith('SSH'):return 'ssh'
    if app.startswith('NTP'):return 'ntp'
    if p==6 and port==443:return 'tcp443_other_application'
    if p==6:return 'other_tcp'
    if p==17:return 'other_udp'
    return 'other_protocol'

class Intervals:
    def __init__(self,rows):
        self.items={}
        for k,xs in rows.items():
            xs=sorted(xs);starts=[];prefix=[];hi=-1
            for a,b,i in xs:starts.append(a);hi=max(hi,b);prefix.append(hi)
            self.items[k]=(xs,starts,prefix)
    def query(self,k,t):
        if k not in self.items:return []
        xs,starts,prefix=self.items[k];i=bisect.bisect_right(starts,t)-1;result=[]
        while i>=0 and prefix[i]>=t:
            if xs[i][1]>=t:result.append(xs[i][2])
            i-=1
        return result
    def overlap(self,k,lo,hi):
        if k not in self.items:return []
        xs,starts,prefix=self.items[k];i=bisect.bisect_right(starts,hi)-1;result=[]
        while i>=0 and prefix[i]>=lo:
            if xs[i][1]>=lo:result.append(xs[i][2])
            i-=1
        return result

def stats():return {'orig_pkts':0,'resp_pkts':0,'orig_l2_bytes':0,'resp_l2_bytes':0,'orig_ip_bytes':0,'resp_ip_bytes':0,'first_us':None,'last_us':None,'first_record_index':None,'last_record_index':None,'packet_identity_digest':hashlib.sha256(),'uncovered_packets':0,'ambiguous_membership_packets':0}
def add(s,p,orig,uncovered=False,ambiguous=False):
    side='orig' if orig else 'resp';s[side+'_pkts']+=1;s[side+'_l2_bytes']+=p['cap'];s[side+'_ip_bytes']+=p['ip_bytes']
    s['first_us']=p['us'] if s['first_us'] is None else min(s['first_us'],p['us']);s['last_us']=p['us'] if s['last_us'] is None else max(s['last_us'],p['us'])
    if s['first_record_index'] is None:s['first_record_index']=p['index']
    s['last_record_index']=p['index'];s['packet_identity_digest'].update(struct.pack('<QQI',p['index'],p['offset'],p['cap']))
    s['uncovered_packets']+=uncovered;s['ambiguous_membership_packets']+=ambiguous
def finish(s):return {**s,'packet_identity_digest':s['packet_identity_digest'].hexdigest()}

def packets(path,diagnostics):
    with path.open('rb') as f:
        assert f.read(24)[:4]==bytes.fromhex('d4c3b2a1');index=0
        while True:
            offset=f.tell();h=f.read(16)
            if not h:break
            if len(h)!=16:diagnostics['incomplete_record_header']+=1;break
            sec,usec,cap,orig=struct.unpack('<IIII',h);x=f.read(cap);index+=1
            if len(x)!=cap:diagnostics['incomplete_frame']+=1;break
            diagnostics['all_complete_packet_records']+=1
            if len(x)<14:diagnostics['short_ethernet']+=1;continue
            layer=14;ether=struct.unpack_from('!H',x,12)[0]
            while ether in (0x8100,0x88a8):
                if len(x)<layer+4:break
                ether=struct.unpack_from('!H',x,layer+2)[0];layer+=4
            ip=x[layer:]
            if ether==0x800 and len(ip)>=20:
                ihl=(ip[0]&15)*4;proto=ip[9];length=struct.unpack_from('!H',ip,2)[0]
                src,dst=socket.inet_ntoa(ip[12:16]),socket.inet_ntoa(ip[16:20])
                if struct.unpack_from('!H',ip,6)[0]&0x1fff:diagnostics['nonfirst_ipv4_fragment_unmapped']+=1;continue
            elif ether==0x86dd and len(ip)>=40:
                ihl=40;proto=ip[6];length=40+struct.unpack_from('!H',ip,4)[0];src,dst=str(ipaddress.ip_address(ip[8:24])),str(ipaddress.ip_address(ip[24:40]))
                if proto in (0,43,44,60):diagnostics['ipv6_extension_unmapped']+=1;continue
            else:diagnostics['non_ip_or_short_ip']+=1;continue
            if proto in (6,17) and len(ip)>=ihl+4:sp,dp=struct.unpack_from('!HH',ip,ihl)
            elif proto in (1,58):sp=dp=0
            else:diagnostics['unsupported_transport']+=1;continue
            yield {'index':index,'offset':offset,'us':sec*1000000+usec,'ms':sec*1000+usec//1000,'cap':cap,'ip_bytes':length,'proto':proto,'src':src,'dst':dst,'sp':sp,'dp':dp,'key':key(proto,src,sp,dst,dp)}

def packet_mapping(authors,cs,pcap):
    ai=collections.defaultdict(list);zi=collections.defaultdict(list)
    for i,a in enumerate(authors):ai[akey(a)].append((int(a['bidirectional_first_seen_ms']),int(a['bidirectional_last_seen_ms']),i))
    capture_end=json.loads((OUT/'pcap_quality.json').read_text(encoding='utf-8'))['max_packet_time']
    for i,c in enumerate(cs):
        lo=round(c['connection_start']*1e6)
        hi=round(max(c.get('replay_export_time') or capture_end,c.get('zeek_reported_end') or c['connection_start'])*1e6)
        zi[zkey(c)].append((lo,hi,i))
    author_index,zeek_index=Intervals(ai),Intervals(zi)
    aa=[stats() for _ in authors];zz=[stats() for _ in cs];linkstats={};diag=collections.Counter();start=time.monotonic();last=0
    with gzip.open(OUT/'packet_membership.jsonl.gz','wt',encoding='utf-8',compresslevel=1,newline='\n') as ledger:
        for p in packets(pcap,diag):
            ais=author_index.query(p['key'],p['ms']);zis=zeek_index.query(p['key'],p['us']);diag['parsed_packets']+=1
            diag['packets_without_author_candidate']+=not ais;diag['packets_without_zeek_candidate']+=not zis;diag['packets_multi_author']+=len(ais)>1;diag['packets_multi_zeek']+=len(zis)>1
            ledger.write(json.dumps([p['index'],p['offset'],p['us'],p['cap'],p['ip_bytes'],ais,zis],separators=(',',':'))+'\n')
            for a in ais:add(aa[a],p,p['src']==address(authors[a]['src_ip']) and (p['proto'] in (1,58) or p['sp']==int(authors[a]['src_port'])),uncovered=not zis,ambiguous=len(ais)>1)
            for z in zis:add(zz[z],p,p['src']==address(cs[z]['source']) and (p['proto'] in (1,58) or p['sp']==cs[z]['raw8']['orig_port']),uncovered=not ais,ambiguous=len(zis)>1)
            for a in ais:
                for z in zis:
                    pair=(a,z)
                    if pair not in linkstats:linkstats[pair]={'author_index':a,'zeek_index':z,'packet_count':0,'first_record_index':p['index'],'last_record_index':None,'packet_identity_digest':hashlib.sha256(),'ambiguous_packet_count':0,'basis':'raw_packet_candidate_intersection'}
                    ls=linkstats[pair];ls['packet_count']+=1;ls['last_record_index']=p['index'];ls['packet_identity_digest'].update(struct.pack('<QQI',p['index'],p['offset'],p['cap']));ls['ambiguous_packet_count']+=len(ais)>1 or len(zis)>1
            elapsed=time.monotonic()-start
            if elapsed-last>30:print(json.dumps({'status':'packet_mapping','parsed_packets':diag['parsed_packets'],'source_byte_offset':p['offset'],'elapsed_seconds':round(elapsed)}),flush=True);last=elapsed
    # Keep time-only candidates even if no original packet intersection was verified.
    for a,row in enumerate(authors):
        for z in zeek_index.overlap(akey(row),int(row['bidirectional_first_seen_ms'])*1000,int(row['bidirectional_last_seen_ms'])*1000+999):
            if (a,z) not in linkstats:linkstats[(a,z)]={'author_index':a,'zeek_index':z,'packet_count':0,'first_record_index':None,'last_record_index':None,'packet_identity_digest':None,'ambiguous_packet_count':0,'basis':'tuple_time_candidate_without_packet_intersection'}
    linkrows=[]
    for v in linkstats.values():
        if v['packet_identity_digest'] is not None:v['packet_identity_digest']=v['packet_identity_digest'].hexdigest()
        linkrows.append(v)
    payload={'author_stats':[finish(s) for s in aa],'zeek_stats':[finish(s) for s in zz],'diagnostics':dict(diag)}
    with gzip.open(OUT/'packet_stats.json.gz','wt',encoding='utf-8',compresslevel=1) as f:json.dump(payload,f,separators=(',',':'))
    write_rows(OUT/'packet_links.jsonl.gz',linkrows)
    return payload,linkrows

def assemble(authors,cs,payload,linkrows):
    aa,zz=payload['author_stats'],payload['zeek_stats'];aby=collections.defaultdict(list);zby=collections.defaultdict(list)
    author_sha=sha(SRC/'flow_20210621_net1013x.csv')
    tail_path=SRC/'full_capture/tail_record_evidence.json';tail=json.loads(tail_path.read_text(encoding='utf-8')) if tail_path.exists() else None
    tail_candidates=[]
    if tail:
        tk=key(tail['ip_protocol'],tail['source'],tail['orig_port'],tail['destination'],tail['resp_port'])
        possible=[c for c in cs if zkey(c)==tk and c['connection_start']<=tail['packet_timestamp']]
        if possible:
            latest=max(c['connection_start'] for c in possible);tail_candidates=[c['flow_id'] for c in possible if c['connection_start']==latest]
    for x in linkrows:aby[x['author_index']].append(x);zby[x['zeek_index']].append(x)
    events=[];matches=[]
    for i,a in enumerate(authors):
        s=aa[i];label,kind=truth(a)
        packet_match=(s['orig_pkts'],s['resp_pkts'])==(int(a['src2dst_packets']),int(a['dst2src_packets']))
        bytes_match=(s['orig_l2_bytes'],s['resp_l2_bytes'])==(int(a['src2dst_bytes']),int(a['dst2src_bytes']))
        time_match=s['first_us'] is not None and s['first_us']//1000==int(a['bidirectional_first_seen_ms']) and s['last_us']//1000==int(a['bidirectional_last_seen_ms'])
        row={'event_id':f'unraveled:20210621:net1013x:author_flow:{a["id"]}','kind':'author_annotated_network_flow','semantic_kind':kind,'independent_action_count':0,'individual_command_truth':False,'campaign_id':CAMPAIGN,'group':CAMPAIGN,'role':'opened_source_development','fit_validation_role':None,'author_label':label,'source_flow_id':a['id'],'source_line':a['source_line'],'source_path':str(SRC/'flow_20210621_net1013x.csv'),'source_address':a['src_ip'],'destination_address':a['dst_ip'],'orig_port':int(a['src_port']),'resp_port':int(a['dst_port']),'ip_protocol':int(a['protocol']),'source_start':int(a['bidirectional_first_seen_ms'])/1000,'source_end':int(a['bidirectional_last_seen_ms'])/1000,'involves_monitored_host':'10.1.3.8' in (a['src_ip'],a['dst_ip']),'application_name':a['application_name'],'application_category_name':a['application_category_name'],'requested_server_name':a['requested_server_name'],'normal_business_stratum':stratum(a) if label==0 else None,'annotation':{k:a[k] for k in ('Activity','Stage','DefenderResponse','Signature')},'candidate_flow_ids':[cs[x['zeek_index']]['flow_id'] for x in aby[i]],'packet_supported_candidate_flow_ids':[cs[x['zeek_index']]['flow_id'] for x in aby[i] if x['packet_count']>0],'author_expected_packet_counts':[int(a['src2dst_packets']),int(a['dst2src_packets'])],'author_expected_l2_bytes':[int(a['src2dst_bytes']),int(a['dst2src_bytes'])],'packet_observation':s,'packet_counts_match':packet_match,'l2_bytes_match':bytes_match,'first_last_milliseconds_match':time_match,'exact_author_packet_validation':packet_match and bytes_match and time_match,'match_status':'no_candidate' if not aby[i] else 'tuple_time_only' if not any(x['packet_count'] for x in aby[i]) else 'single_candidate' if len(aby[i])==1 else 'multiple_candidates_preserved'}
        row['unmatched_or_quality_reasons']=[]
        if not aby[i]:row['unmatched_or_quality_reasons'].append('no_tuple_time_Zeek_candidate')
        if not packet_match:row['unmatched_or_quality_reasons'].append('author_packet_count_differs_from_interval_candidate_set')
        if not bytes_match:row['unmatched_or_quality_reasons'].append('author_L2_byte_count_differs_from_interval_candidate_set')
        if not time_match:row['unmatched_or_quality_reasons'].append('author_millisecond_first_or_last_differs')
        if s['ambiguous_membership_packets']:row['unmatched_or_quality_reasons'].append('overlapping_author_windows_preserved')
        events.append(row);matches.append(row)
    for z,c in enumerate(cs):
        s=zz[z];pairs=zby[z];all_a=[x['author_index'] for x in pairs];packet_a=[x['author_index'] for x in pairs if x['packet_count']>0]
        labels={truth(authors[a])[0] for a in all_a};kinds={truth(authors[a])[1] for a in all_a}
        expected=c['raw8'];possible_enough=all(expected[k] is None or s[k]>=expected[k] for k in ('orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes'))
        exact_counts=all(expected[k] is not None and s[k]==expected[k] for k in ('orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes'))
        nonempty=s['orig_pkts']+s['resp_pkts']>0
        closed=nonempty and s['uncovered_packets']==0 and possible_enough and bool(packet_a)
        consistent=bool(labels) and None not in labels and len(labels)==1
        c['label']=next(iter(labels)) if closed and consistent else None
        c.update({'raw8_observation_scope':'complete original PCAP records; incomplete final source record excluded without reconstruction','partial_capture':c['flow_id'] in tail_candidates,'partial_capture_basis':{'tail_evidence_path':str(tail_path),'candidate_flow_ids':tail_candidates,'association':'same five-tuple and latest connection start before truncated final record; no later connection in capture'} if c['flow_id'] in tail_candidates else None})
        target_kind='normal' if c['label']==0 else 'c2' if c['label']==1 and kinds=={'c2_channel'} else 'other_attack' if c['label']==1 and kinds=={'other_attack'} else 'unknown'
        c.update({'author_attack_label':c['label'],'target_kind':target_kind,'target_label':c['label'] if target_kind in ('normal','c2') else None,'author_activity':sorted({authors[a]['Activity'] for a in all_a}),'author_stage':sorted({authors[a]['Stage'] for a in all_a}),'author_signature':sorted({authors[a]['Signature'] for a in all_a}),'source_group':CAMPAIGN,'author_annotation':{'Activity':sorted({authors[a]['Activity'] for a in all_a}),'Stage':sorted({authors[a]['Stage'] for a in all_a}),'Signature':sorted({authors[a]['Signature'] for a in all_a})}})
        c.update({'label_basis':'B_author_explicit_labels_consistent_over_closed_packet_candidate_set' if c['label'] is not None else 'unknown_or_mixed_or_unclosed_packet_coverage','evidence_grade':'B_author_annotation_with_original_packet_linkage' if c['label'] is not None else 'unresolved','semantic_kinds':sorted(kinds),'is_control_channel':c['label']==1 and kinds=={'c2_channel'},'development_label_usable':c['label'] is not None,'fit_validation_role':None,'event_ids':[events[a]['event_id'] for a in all_a],'source_annotation_candidate_count':len(all_a),'source_annotation_packet_supported_count':len(packet_a),'packet_candidate_observation':s,'packet_candidate_coverage_closed':closed,'candidate_packet_totals_equal_Zeek':exact_counts,'candidate_packet_totals_cover_Zeek':possible_enough,'feature_eligible':not c['raw8_missing_fields'],'label_independent_of_conn_state':True,'normal_business_strata':sorted({stratum(authors[a]) for a in all_a if truth(authors[a])[0]==0}),'normal_source_grade':'author_explicit_flow_annotation' if c['label']==0 else None,'qualification_reasons':[],'source_provenance':{'author_flow_path':str(SRC/'flow_20210621_net1013x.csv'),'author_flow_sha256':author_sha,'packet_membership_path':str(OUT/'packet_membership.jsonl.gz'),'packet_identity':'source PCAP SHA256 + global record index + byte offset; digest sequence encoded little-endian <QQI(index,offset,caplen)>','source_pcap_sha256':c['source_pcap_sha256']}})
        if not nonempty:c['qualification_reasons'].append('no_parsed_raw_packet_candidate')
        if s['uncovered_packets']:c['qualification_reasons'].append('raw_packet_candidates_without_valid_author_annotation')
        if not possible_enough:c['qualification_reasons'].append('candidate_packet_totals_do_not_cover_Zeek_observation')
        if len(labels)>1 or None in labels:c['qualification_reasons'].append('mixed_or_unknown_source_labels')
        if not all_a:c['qualification_reasons'].append('no_author_flow_candidate')
        if s['ambiguous_membership_packets']:c['qualification_reasons'].append('overlapping_Zeek_lifecycle_windows_retained_as_candidate_set')
    # An author flow row is evidence, not an independent command/action.
    # All C2 rows in this capture are one campaign activity event.
    aggregate_groups=collections.defaultdict(list);annotation_to_event={}
    for e in events:
        e['annotation_id']=e['event_id']
        focus='host10.1.3.8' if e['involves_monitored_host'] else 'other_capture_hosts'
        aggregate=f'unraveled:20210621:net1013x:{focus}:{e["semantic_kind"]}_observation_set'
        annotation_to_event[e['annotation_id']]=aggregate;e['event_id']=aggregate
        aggregate_groups[aggregate].append(e)
    byid={c['flow_id']:c for c in cs};aggregate_events=[]
    for event_id,es in aggregate_groups.items():
        fids=sorted({fid for e in es for fid in e['candidate_flow_ids']});semantic=es[0]['semantic_kind'];lab=es[0]['author_label']
        all_confirmed=bool(fids) and all(byid[fid]['label']==lab and (semantic!='c2_channel' or byid[fid]['is_control_channel']) for fid in fids)
        source_complete=all(e['exact_author_packet_validation'] and e['packet_supported_candidate_flow_ids'] for e in es)
        aggregate_events.append({'event_id':event_id,'kind':'known_c2_channel_activity' if semantic=='c2_channel' else 'author_declared_'+semantic+'_observation_set','semantic_kind':semantic,'campaign_id':CAMPAIGN,'source_group':CAMPAIGN,'group':CAMPAIGN,'role':'opened_source_development','fit_validation_role':None,'author_label':lab,'target_label':lab if semantic in ('normal','c2_channel') else None,'involves_monitored_host':es[0]['involves_monitored_host'],'source_start':min(e['source_start'] for e in es),'source_end':max(e['source_end'] for e in es),'candidate_flow_ids':fids,'confirmed_flow_ids':[fid for fid in fids if byid[fid]['label']==lab and (semantic!='c2_channel' or byid[fid]['is_control_channel'])],'association_mode':'all_confirmed' if all_confirmed else None,'candidate_set_complete':True if source_complete else 'unknown','candidate_set_scope':'all valid author-annotated rows of this semantic class in this one complete capture; not the whole six-week campaign','individual_command_truth':False,'independent_action_count':0,'author_flow_count':len(es),'source_annotation_ids':[e['annotation_id'] for e in es],'annotation_evidence_path':str(OUT/'author_flow_matches.jsonl.gz')})
    for c in cs:
        c['source_annotation_ids']=c['event_ids'];c['event_ids']=sorted({annotation_to_event[e] for e in c['source_annotation_ids']})
    links=[{'event_id':events[x['author_index']]['event_id'],'source_annotation_id':events[x['author_index']]['annotation_id'],'flow_id':cs[x['zeek_index']]['flow_id'],'candidate_retained':True,**x} for x in linkrows]
    write_rows(OUT/'events.jsonl',aggregate_events);write_rows(OUT/'author_flow_matches.jsonl.gz',matches);write_rows(OUT/'target_host_author_flow_matches.jsonl.gz',[x for x in matches if x['involves_monitored_host']]);write_rows(OUT/'connections.jsonl.gz',cs);write_rows(OUT/'links.jsonl',links)
    # Historical slice records are aliases only; the full capture replaces them.
    bykey=collections.defaultdict(list)
    for c in cs:bykey[zkey(c)].append(c)
    aliases=[]
    oldfile=ROOT/'network_mapping/connections.jsonl'
    if oldfile.exists():
        for old in map(json.loads,oldfile.read_text(encoding='utf-8').splitlines()):
            possible=bykey.get(zkey(old),[]);exact=[c for c in possible if abs(c['connection_start']-old['connection_start'])<.000002]
            aliases.append({'old_flow_id':old['flow_id'],'old_source_scope':old['source_scope'],'full_flow_ids':[c['flow_id'] for c in exact],'status':'exact_start_tuple' if len(exact)==1 else 'ambiguous' if exact else 'unmatched','raw8_equal':old['raw8']==exact[0]['raw8'] if len(exact)==1 else None,'reason':'full capture is authoritative; a historical partial observation may have different raw8 or lack the true start'})
    write_rows(OUT/'old_slice_aliases.jsonl',aliases)
    target=[x for x in events if x['involves_monitored_host']];tc=[x for x in cs if x['involves_monitored_host']]
    strata={};label_by_flow={c['flow_id']:c['label'] for c in cs}
    for name in sorted({x['normal_business_stratum'] for x in target if x['author_label']==0}):
        xs=[x for x in target if x['normal_business_stratum']==name]
        strata[name]={'author_explicit_normal_rows':len(xs),'packet_exact_author_rows':sum(x['exact_author_packet_validation'] for x in xs),'with_packet_supported_candidate':sum(bool(x['packet_supported_candidate_flow_ids']) for x in xs),'candidate_flow_ids':sorted({fid for x in xs for fid in x['candidate_flow_ids']}),'accepted_normal_flow_ids':sorted({fid for x in xs for fid in x['candidate_flow_ids'] if label_by_flow[fid]==0})}
    dump(OUT/'target_host_normal_business_strata.json',{'host':'10.1.3.8','source_grade':'author_explicit_Normal; no per-row process-log requirement','strata_can_share_connections':True,'business_names_are_application_strata_not_proven_individual_actor_actions':True,'strata':strata})
    summary={'status':'complete','source_scope':'one full unchanged net1013x PCAP, all hosts and protocols retained','campaign_id':CAMPAIGN,'role':'opened_source_development','fit_validation_role':None,'no_training_or_scoring':True,'all_connections':len(cs),'all_author_valid_rows':len(authors),'target_host_author_rows':len(target),'target_host_connections':len(tc),'links':len(links),'all_connection_labels':dict(collections.Counter(str(c['label']) for c in cs)),'target_host_connection_labels':dict(collections.Counter(str(c['label']) for c in tc)),'target_host_author_label_counts':dict(collections.Counter((str(x['author_label'])+':'+x['semantic_kind']) for x in target)),'author_match_status':dict(collections.Counter(x['match_status'] for x in events)),'target_author_match_status':dict(collections.Counter(x['match_status'] for x in target)),'exact_author_packet_validation':sum(x['exact_author_packet_validation'] for x in events),'target_exact_author_packet_validation':sum(x['exact_author_packet_validation'] for x in target),'all_unmatched_or_quality_reasons':dict(collections.Counter(r for x in events for r in x['unmatched_or_quality_reasons'])),'target_unmatched_or_quality_reasons':dict(collections.Counter(r for x in target for r in x['unmatched_or_quality_reasons'])),'labels_assigned_to_non_SF_connections':dict(collections.Counter(c['conn_state'] for c in cs if c['label'] is not None and c['conn_state']!='SF')),'normal_missing_raw8_labels_retained':sum(c['label']==0 and bool(c['raw8_missing_fields']) for c in cs),'packet_mapping_diagnostics':payload['diagnostics'],'alias_status':dict(collections.Counter(x['status'] for x in aliases)),'earlier_slice_version_superseded':str(ROOT/'network_mapping'),'stable_flow_key':'SHA256(source entire PCAP SHA + native protocol + original endpoints/ports + start fixed at 6 decimals)','source_pcap_sha256':cs[0]['source_pcap_sha256'] if cs else None,'output_hashes':{p.name:sha(p) for p in (OUT/'connections.jsonl.gz',OUT/'events.jsonl',OUT/'links.jsonl',OUT/'target_host_author_flow_matches.jsonl.gz',OUT/'old_slice_aliases.jsonl')}}
    summary.update({'aggregate_events':len(aggregate_events),'target_host_target_label_counts':dict(collections.Counter(str(c['target_label']) for c in tc)),'target_host_target_kind_counts':dict(collections.Counter(c['target_kind'] for c in tc)),'partial_capture_flow_ids':tail_candidates,'partial_capture_raw8_scope':'complete original records observed; final incomplete record not reconstructed','normal_denominator_scope':'all source-labelled normal observations involving the declared host in this complete-record capture; source normal annotation only, not a claim about unlabelled/malformed background','packet_membership_sha256':sha(OUT/'packet_membership.jsonl.gz')})
    summary['source_scope']='one complete downloaded source object; Zeek processes all complete original packet records, excluding the incomplete final record only'
    assert len(target)==12582 and len({c['flow_id'] for c in cs})==len(cs)
    dump(OUT/'assembly_receipt.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

def main(wait):
    if wait:
        while not (OUT/'extraction_receipt.json').exists():time.sleep(10)
    receipt=json.loads((OUT/'extraction_receipt.json').read_text(encoding='utf-8'));assert receipt['status']=='complete'
    authors,bad,validation=flow_csv(SRC/'flow_20210621_net1013x.csv');dump(OUT/'author_csv_validation.json',validation);write_rows(OUT/'malformed_author_rows.jsonl',bad)
    cs=readgz(OUT/'base_connections.jsonl.gz')
    cache=OUT/'packet_stats.json.gz'
    if cache.exists() and (OUT/'packet_links.jsonl.gz').exists():
        with gzip.open(cache,'rt',encoding='utf-8') as f:payload=json.load(f)
        links=readgz(OUT/'packet_links.jsonl.gz')
    else:payload,links=packet_mapping(authors,cs,Path(json.loads((SRC/'full_capture/download_receipt.json').read_text(encoding='utf-8'))['local_path']))
    assemble(authors,cs,payload,links)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--wait-extraction',action='store_true');main(a.parse_args().wait_extraction)
