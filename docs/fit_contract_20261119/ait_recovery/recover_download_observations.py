"""Associate fixed R27 downloads with an independent capture point.

Parses original PCAPs, checks all NAT/sequence candidates, and exports the entire
selected firewall rotation through the existing pinned Zeek contract. No packet
repair, labels by size, model fitting, or model scoring.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
NOW=BASE/'统一事件与方法落实_20261119'
REPO=BASE.parent/'代码/tracelight-ids/.worktrees/ait-flow-contract-20261118'
TARGETS=(3574,3575,3578)
NAT='192.168.230.4'


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
helper=module('checksum_helpers',NOW/'ait/diagnose_checksum_semantics.py')
reader=module('packet_reader',REPO/'scripts/prepare_ait_packet_evidence.py')


def tuple_key(a,ap,b,bp):return tuple(sorted(((a,int(ap)),(b,int(bp)))))
def envelope_key(p):return tuple_key(p['origin_ip'],p['origin_port'],p['responder_ip'],p['responder_port'])
def coverage(intervals,required):
    intervals=sorted(intervals);merged=[]
    for a,b in intervals:
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    gaps=[];cursor=0
    for a,b in merged:
        if a>cursor:gaps.append([cursor,a])
        cursor=max(cursor,b)
    if required is not None and cursor<required:gaps.append([cursor,required])
    return {'unique_payload_bytes':sum(b-a for a,b in merged),'payload_extent':cursor,'fin_payload_end':required,'gap_count':len(gaps),'first_gaps':gaps[:10],'complete_to_fin':required is not None and not gaps and cursor==required,'merged_interval_count':len(merged)}


def main():
    old_sem_path=NOW/'ait_attack/checksum_semantics.json';old_sem=load(old_sem_path)
    old={r['packet_connection_id']:r for r in old_sem['packet_candidates'] if r['packet_connection_id'] in TARGETS}
    assert set(old)==set(TARGETS) and all(r['model_row_blocked_on_capture_semantics'] for r in old.values())
    receipt=load(ROOT/'source/firewall-source-receipt.json');pcap=Path(receipt['pcap'])
    assert helper.sha(pcap)==receipt['sha256'] and pcap.stat().st_size==receipt['bytes']
    catalogs,packet_count=reader.parse_connections(pcap,start=float('-inf'),stop=float('inf'))
    matches={}
    for cid,r in old.items():
        p=r['packet_envelope'];syn=r['handshake']['syn']
        matches[cid]=[c for c in catalogs if c['origin_ip']==NAT and c['responder_ip']==p['responder_ip'] and int(c['responder_port'])==int(p['responder_port']) and c['syn_sequence']==syn['sequence']]
    save(ROOT/'nat_candidates.json',{'pcap_sha256':receipt['sha256'],'total_firewall_connections':len(catalogs),'pcap_counts':packet_count,'selection':'Exact fixed remote IP/port and client initial TCP sequence; translated source address independently established by capture/SNAT configuration. No candidate selection by flow size.','targets':[{'old_packet_id':cid,'old_tuple':r['packet_envelope'],'firewall_candidates':matches[cid]} for cid,r in old.items()]})
    print(json.dumps({'stage':'nat_candidates','counts':{cid:len(v) for cid,v in matches.items()}},ensure_ascii=False),flush=True)
    if not all(matches.values()):
        save(ROOT/'recovery_summary.json',{'status':'missing_NAT_sequence_candidate','counts':{cid:len(v) for cid,v in matches.items()},'old_records_still_blocked':list(TARGETS)})
        return
    candidates={};by_key=defaultdict(list)
    for cid,ps in matches.items():
        for p in ps:
            key=f"{cid}:{p['connection_id']}";candidates[key]={'old_packet_id':cid,'envelope':p}
            by_key[envelope_key(p)].append(key)
    directions=('origin','responder')
    stats={k:{d:Counter() for d in directions} for k in candidates}
    intervals={k:{d:[] for d in directions} for k in candidates}
    fins={k:{d:[] for d in directions} for k in candidates}
    firsts={k:{d:None for d in directions} for k in candidates}
    sample_errors=defaultdict(list);payload_fingerprints=defaultdict(set)
    # Compare 32 captured payload bytes, not packet checksums/addresses changed by NAT.
    original_pcap=NOW/'ait_attack/source/log.pcap.1642684613'
    original_signatures=defaultdict(dict)
    old_by_key={envelope_key(r['packet_envelope']):cid for cid,r in old.items()}
    for ts,linktype,frame,wire in reader.pcap_records(original_pcap):
        parsed=helper.ipv4(frame,linktype)
        if parsed is None:continue
        ip,tcp,m=parsed;cid=old_by_key.get(tuple_key(m['source_ip'],m['source_port'],m['destination_ip'],m['destination_port']))
        if cid is None:continue
        p=old[cid]['packet_envelope']
        if not float(p['first'])<=ts<=float(p['last']):continue
        d='origin' if m['source_ip']==p['origin_ip'] else 'responder'
        available=max(0,min(len(tcp),m['ip_total_length']-m['ip_header_bytes'])-m['tcp_header_bytes'])
        if available>=32:
            sig=(d,m['sequence'],m['acknowledgment'],hashlib.sha256(tcp[m['tcp_header_bytes']:m['tcp_header_bytes']+32]).hexdigest())
            original_signatures[cid][sig]=ts
    for number,(ts,linktype,frame,wire) in enumerate(reader.pcap_records(pcap),1):
        parsed=helper.ipv4(frame,linktype)
        if parsed is None:continue
        ip,tcp,m=parsed;keys=by_key.get(tuple_key(m['source_ip'],m['source_port'],m['destination_ip'],m['destination_port']),[])
        keys=[k for k in keys if candidates[k]['envelope']['first']<=ts<=candidates[k]['envelope']['last']]
        if not keys:continue
        check=helper.checksum_info(ip,tcp,m)
        d='origin' if m['source_ip']==NAT else 'responder'
        for k in keys:
            s=stats[k][d];s['packets']+=1;s['declared_ip_bytes']+=m['ip_total_length']
            s['tcp_'+check['tcp_checksum']]+=1;s['ip_'+check['ipv4_checksum']]+=1
            s['tcp_verifiable']+=check['tcp_checksum']!='unverifiable'
            s['invalid_length']+=check['ipv4_total_exceeds_captured'] or check['ipv4_total_zero']
            s['pcap_captured_less_than_original']+=len(frame)<wire
            s['bad_checksum_equals_partial_seed']+=check['tcp_checksum']=='incorrect' and check['tcp_checksum_matches_pseudoheader_seed'] is True
            if check['tcp_checksum']=='correct' and check['ipv4_checksum']=='correct':s['checksum_correct_ip_bytes']+=m['ip_total_length']
            if check['tcp_checksum']!='correct' and len(sample_errors[k])<3:sample_errors[k].append({'packet_number':number,'timestamp':ts,**m,**check})
            if m['flags']&2 and firsts[k][d] is None:firsts[k][d]={'sequence':m['sequence'],'acknowledgment':m['acknowledgment'],'flags':m['flags'],'timestamp':ts}
            orig=old[candidates[k]['old_packet_id']]
            initial=orig['handshake']['syn' if d=='origin' else 'syn_ack']['sequence']
            rel=(m['sequence']-initial-1)%2**32
            payload_len=m['ip_total_length']-m['ip_header_bytes']-m['tcp_header_bytes']
            if check['tcp_checksum']!='unverifiable' and payload_len>0 and rel<2**31:
                intervals[k][d].append((rel,rel+payload_len))
                if payload_len>=32:
                    sig=(d,m['sequence'],m['acknowledgment'],hashlib.sha256(tcp[m['tcp_header_bytes']:m['tcp_header_bytes']+32]).hexdigest())
                    if sig in original_signatures[candidates[k]['old_packet_id']]:
                        payload_fingerprints[k].add(sig)
                        delta=ts-original_signatures[candidates[k]['old_packet_id']][sig]
                        s['payload_fingerprint_matches']+=1
                        if 'max_abs_payload_timestamp_delta' not in s or abs(delta)>s['max_abs_payload_timestamp_delta']:s['max_abs_payload_timestamp_delta']=abs(delta)
            if m['flags']&1 and rel<2**31:fins[k][d].append(rel+max(0,payload_len))
            if m['flags']&4:s['reset_packets']+=1
    audited=[]
    for k,c in candidates.items():
        p=c['envelope'];assert stats[k]['origin']['packets']==p['origin_packets'] and stats[k]['responder']['packets']==p['responder_packets']
        both_sequence=all(firsts[k][d] is not None and firsts[k][d]['sequence']==old[c['old_packet_id']]['handshake']['syn' if d=='origin' else 'syn_ack']['sequence'] and firsts[k][d]['acknowledgment']==old[c['old_packet_id']]['handshake']['syn' if d=='origin' else 'syn_ack']['acknowledgment'] for d in directions)
        complete={d:coverage(intervals[k][d],max(fins[k][d]) if fins[k][d] else None) for d in directions}
        for d in directions:
            for key in ('tcp_correct','tcp_incorrect','tcp_unverifiable','ip_correct','ip_incorrect','invalid_length','bad_checksum_equals_partial_seed','checksum_correct_ip_bytes','payload_fingerprint_matches'):stats[k][d].setdefault(key,0)
        audit={'candidate_id':k,**c,'directions':{d:dict(v) for d,v in stats[k].items()},'matching_initial_sequence_numbers_both_directions':both_sequence,'first_handshake_packets':firsts[k],'sequence_coverage':complete,'matching_payload_fingerprints':len(payload_fingerprints[k]),'checksum_samples':sample_errors[k]}
        audit['transport_recovery_complete']=both_sequence and len(payload_fingerprints[k])>0 and all(complete[d]['complete_to_fin'] and not stats[k][d]['tcp_unverifiable'] and not stats[k][d]['ip_incorrect'] for d in directions)
        audit['checksum_semantics_supported']=stats[k]['origin']['tcp_incorrect']==stats[k]['origin']['bad_checksum_equals_partial_seed'] and stats[k]['responder']['tcp_incorrect']==0
        audited.append(audit)
    save(ROOT/'packet_recovery.json',{'source_pcap':receipt,'original_pcap':str(original_pcap),'original_pcap_sha256':helper.sha(original_pcap),'original_checksum_semantics_sha256':helper.sha(old_sem_path),'candidates':audited,'packet_repair':False,'model_calls':0})
    print(json.dumps({'stage':'packet_recovery','complete':[a['candidate_id'] for a in audited if a['transport_recovery_complete']],'blocked':[a['candidate_id'] for a in audited if not a['transport_recovery_complete']]},ensure_ascii=False),flush=True)
    sys.path.insert(0,str(REPO/'scripts/ait_contract'))
    import zeek_runtime,analyze
    runtime=load(NOW/'ait/runtime-receipt.json');assert runtime['version']=='8.0.11'
    def export(mode):
        dest=ROOT/'zeek'/mode;prior=dest/'export-receipt.json'
        if prior.exists():
            r=load(prior);assert r['pcap_sha256']==receipt['sha256'] and r['mode']==mode
            for name,info in r['logs'].items():
                if info['exists']:assert helper.sha(dest/f'{name}.log')==info['sha256']
            return r
        return zeek_runtime.export_pcap(pcap,dest,mode,runtime)
    with ThreadPoolExecutor(max_workers=2) as pool:exports=dict(zip(('strict','ignore_checksums'),pool.map(export,('strict','ignore_checksums'))))
    arm_records={}
    for mode in exports:
        logs=ROOT/'zeek'/mode
        arm_records[mode]={'conn':[analyze.normalize_connection(r,(0,float('inf'))) for r in analyze.records(logs/'conn.log')],**{name:analyze.records(logs/f'{name}.log') for name in ('http','ssl','files')}}
    for a in audited:
        p=a['envelope'];key=envelope_key(p);a['zeek']={}
        for mode,records in arm_records.items():
            cs=[r for r in records['conn'] if tuple_key(r['source'],r['orig_port'],r['destination'],r['resp_port'])==key and p['first']-0.00001<=r['connection_start']<=p['last']+0.00001]
            uids={r['uid'] for r in cs}
            protocols={name:[r for r in records[name] if r.get('uid') in uids or any(u in uids for u in r.get('conn_uids',[]))] for name in ('http','ssl','files')}
            totals={d:{'pkts':sum(r[('orig' if d=='origin' else 'resp')+'_pkts'] for r in cs),'ip_bytes':sum(r[('orig' if d=='origin' else 'resp')+'_ip_bytes'] for r in cs)} for d in directions}
            a['zeek'][mode]={'connections':cs,'protocol_evidence':protocols,'totals':totals}
        a['canonical_variant']='ignore_checksums' if a['transport_recovery_complete'] and a['checksum_semantics_supported'] else None
        a['canonical_reason']='Complete original capture segments and sequence coverage, same TCP handshake/32-byte payload prefixes across independently configured NAT capture points; whole-member pinned Zeek export with explicit checksum semantics.' if a['canonical_variant'] is not None else 'Incomplete transport or unsupported checksum evidence; no canonical row admitted.'
        a['zeek_ignore_counts_match_packets']=all(a['zeek']['ignore_checksums']['totals'][d]['pkts']==a['directions'][d]['packets'] and a['zeek']['ignore_checksums']['totals'][d]['ip_bytes']==a['directions'][d]['declared_ip_bytes'] for d in directions)
        a['zeek_strict_counts_match_checksum_correct_subset']=all(a['zeek']['strict']['totals'][d]['pkts']==a['directions'][d]['tcp_correct'] and a['zeek']['strict']['totals'][d]['ip_bytes']==a['directions'][d]['checksum_correct_ip_bytes'] for d in directions)
        a['model_row_ready_from_capture']=a['canonical_variant'] is not None and a['zeek_ignore_counts_match_packets'] and bool(a['zeek']['ignore_checksums']['connections']) and all(r['missed_bytes']==0 and all(r[k]>=0 for k in analyze.RAW8) for r in a['zeek']['ignore_checksums']['connections'])
    result={'source_pcap':receipt,'original_records_remain_blocked':list(TARGETS),'model_calls':0,'packet_modifications':0,'all_candidates_retained':True,'candidates':audited,'export_receipts':exports,'capture_config_receipt':load(ROOT/'source/capture_config/receipt.json'),'availability_rule':'Use whole-rotation replay export and preserve EOF/shutdown flags; neither flow end nor replay export is actual detector receipt or action-label availability.','event_label_rule':'This recovers alternative observations of existing two actions; it creates no new independent events, no automatic pure-flow label, and no blind split.','script_sha256':helper.sha(Path(__file__))}
    save(ROOT/'recovered_download_observations.json',result)
    summary={'status':'complete' if all(a['model_row_ready_from_capture'] for a in audited) else 'partial_or_blocked','targets':len(TARGETS),'nat_candidates':len(audited),'complete_transport_candidates':sum(a['transport_recovery_complete'] for a in audited),'model_row_capture_ready':sum(a['model_row_ready_from_capture'] for a in audited),'original_records_still_blocked':list(TARGETS),'candidate_ids':[a['candidate_id'] for a in audited],'source_sha256':receipt['sha256'],'output_sha256':helper.sha(ROOT/'recovered_download_observations.json'),'model_calls':0}
    save(ROOT/'recovery_summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
