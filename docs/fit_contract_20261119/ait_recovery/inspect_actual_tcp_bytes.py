"""Inspect real captured TCP bytes of only the three fixed NAT download flows.

Includes captured payload prefixes from packets with invalid IPv4 total length.
Never fills holes, edits packet headers, decrypts TLS, or emits payload bodies.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('recovery',ROOT/'recover_download_observations.py')
recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)


def merge_real_bytes(records,fin_end):
    merged=[];conflicts=0
    for r in sorted(records,key=lambda x:(x['start'],-len(x['data']))):
        start,data=r['start'],r['data']
        if merged and start<=merged[-1][0]+len(merged[-1][1]):
            ms,md=merged[-1];overlap=min(ms+len(md)-start,len(data))
            old=md[start-ms:start-ms+overlap]
            if old!=data[:overlap]:conflicts+=sum(a!=b for a,b in zip(old,data[:overlap]))
            if overlap<len(data):md.extend(data[overlap:])
        else:merged.append((start,bytearray(data)))
    gaps=[];cursor=0
    for start,data in merged:
        if start>cursor:gaps.append([cursor,start])
        cursor=max(cursor,start+len(data))
    if fin_end is not None and cursor<fin_end:gaps.append([cursor,fin_end])
    return merged,{'captured_payload_bytes_with_retransmissions':sum(len(r['data']) for r in records),'captured_unique_payload_bytes':sum(len(data) for start,data in merged),'sequence_extent':cursor,'fin_payload_end':fin_end,'actual_missing_bytes_to_fin':sum(b-a for a,b in gaps),'actual_gap_count':len(gaps),'first_gaps':gaps[:12],'all_gap_ranges':gaps,'conflicting_overlap_bytes':conflicts,'contiguous_intervals':len(merged),'actual_bytes_complete_to_fin':fin_end is not None and not gaps and not conflicts and cursor==fin_end}


def framing(merged,side,port,stats):
    prefix=bytes(merged[0][1]) if merged and merged[0][0]==0 else b''
    if side!='responder':return None
    if port==443:
        offset=0;closed=[];failure=None
        while offset+5<=len(prefix):
            content_type,major,minor,length=struct.unpack('!BBBH',prefix[offset:offset+5])
            if content_type not in (20,21,22,23,24) or major!=3 or length>18432:
                failure={'status':'header_not_plausible_TLS_record','offset':offset,'content_type':content_type,'version':[major,minor],'declared_length':length};break
            end=offset+5+length
            if end>len(prefix):
                missing=[(max(offset,a),min(end,b)) for a,b in stats['all_gap_ranges'] if max(offset,a)<min(end,b)]
                failure={'status':'TLS_record_intersects_actual_capture_gap','record_start':offset,'record_end':end,'payload_length':length,'content_type':content_type,'version':[major,minor],'first_contiguous_bytes_end':len(prefix),'missing_ranges_inside_record':missing,'missing_bytes_inside_record':sum(b-a for a,b in missing)};break
            closed.append({'record_start':offset,'record_end':end,'content_type':content_type,'version':[major,minor]});offset=end
        return {'protocol':'TLS_record_framing_without_decryption','initial_contiguous_payload_bytes':len(prefix),'complete_initial_record_count':len(closed),'first_records':closed[:8],'first_incomplete_record':failure,'whole_stream_TLS_closure_established':stats['actual_bytes_complete_to_fin'] and failure is None and offset==len(prefix)}
    if port==80:
        end=prefix.find(b'\r\n\r\n')
        if end<0:return {'protocol':'HTTP','header_complete_in_captured_prefix':False,'initial_contiguous_payload_bytes':len(prefix)}
        lines=prefix[:end].decode('iso-8859-1').split('\r\n');headers={}
        for line in lines[1:]:
            if ':' in line:
                key,value=line.split(':',1);headers[key.lower()]=value.strip()
        length=int(headers['content-length']) if headers.get('content-length','').isdigit() else None
        return {'protocol':'HTTP','status_line':lines[0],'header_complete_in_captured_prefix':True,'header_bytes':end+4,'content_length':length,'content_type':headers.get('content-type'),'transfer_encoding':headers.get('transfer-encoding'),'expected_payload_end_from_content_length':None if length is None else end+4+length,'declared_content_end_equals_FIN':length is not None and end+4+length==stats['fin_payload_end'],'actual_body_complete':stats['actual_bytes_complete_to_fin'] and length is not None and end+4+length==stats['fin_payload_end'],'initial_contiguous_payload_bytes':len(prefix)}
    return None


def main():
    source_path=ROOT/'recovered_download_observations.json';source=recovery.load(source_path)
    pcap=Path(source['source_pcap']['pcap']);candidates={r['candidate_id']:r for r in source['candidates']}
    by_key=defaultdict(list)
    for k,r in candidates.items():by_key[recovery.envelope_key(r['envelope'])].append(k)
    payload={k:{d:[] for d in ('origin','responder')} for k in candidates}
    fin_positions={k:{d:[] for d in ('origin','responder')} for k in candidates}
    max_ack={k:{d:None for d in ('origin','responder')} for k in candidates}
    for number,(ts,linktype,frame,wire) in enumerate(recovery.reader.pcap_records(pcap),1):
        parsed=recovery.helper.ipv4(frame,linktype)
        if parsed is None:continue
        ip,tcp,m=parsed;keys=by_key.get(recovery.tuple_key(m['source_ip'],m['source_port'],m['destination_ip'],m['destination_port']),[])
        keys=[k for k in keys if candidates[k]['envelope']['first']<=ts<=candidates[k]['envelope']['last']]
        if not keys:continue
        d='origin' if m['source_ip']==recovery.NAT else 'responder';opposite='responder' if d=='origin' else 'origin'
        assert m['tcp_header_bytes']>=20 and m['tcp_header_bytes']<=len(tcp) and not m['fragment_bits']
        declared=m['ip_total_length']-m['ip_header_bytes']-m['tcp_header_bytes']
        # Real byte limit only. Trim possible Ethernet padding for otherwise complete IP packets.
        actual=max(0,min(len(tcp)-m['tcp_header_bytes'],declared))
        for k in keys:
            initial=candidates[k]['first_handshake_packets'][d]['sequence'];rel=(m['sequence']-initial-1)%2**32
            if actual and rel<2**31:
                data=tcp[m['tcp_header_bytes']:m['tcp_header_bytes']+actual]
                payload[k][d].append({'start':rel,'data':data,'timestamp':ts,'packet_number':number,'declared_payload_bytes':declared,'length_anomaly':declared>actual})
            if m['flags']&1 and rel<2**31:fin_positions[k][d].append(rel+actual)
            if m['flags']&16:
                peer_initial=candidates[k]['first_handshake_packets'][opposite]['sequence'];ack=(m['acknowledgment']-peer_initial-1)%2**32
                if ack<2**31 and (max_ack[k][opposite] is None or ack>max_ack[k][opposite]['relative_ack']):max_ack[k][opposite]={'relative_ack':ack,'timestamp':ts,'packet_number':number}
    out=[]
    for k,c in candidates.items():
        item={'candidate_id':k,'old_packet_id':c['old_packet_id'],'firewall_packet_id':c['envelope']['connection_id'],'directions':{}}
        for d in ('origin','responder'):
            records=payload[k][d];fin_end=max(fin_positions[k][d]) if fin_positions[k][d] else None
            merged,stats=merge_real_bytes(records,fin_end)
            assert stats['captured_payload_bytes_with_retransmissions']==c['envelope'][d+'_captured_tcp_payload_bytes']
            starts=sorted(set([r['start'] for r in records]+([] if fin_end is None else [fin_end])))
            progression=Counter();examples=[]
            for r in records:
                if not r['length_anomaly']:continue
                ix=bisect_right(starts,r['start'])
                if ix==len(starts):progression['no_next_sequence_start']+=1;continue
                advance=starts[ix]-r['start'];actual=len(r['data']);declared=r['declared_payload_bytes']
                category='equals_actual_payload' if advance==actual else ('equals_IP_declared_payload' if advance==declared else 'other_including_overlap_or_retransmission')
                progression[category]+=1
                if len(examples)<5:examples.append({'packet_number':r['packet_number'],'timestamp':r['timestamp'],'relative_sequence':r['start'],'actual_payload_bytes':actual,'IP_declared_payload_bytes':declared,'next_distinct_payload_or_FIN_sequence':starts[ix],'advance':advance,'classification':category})
            ack=max_ack[k][d]
            stats['maximum_peer_ACK']=ack
            stats['all_actual_gaps_acknowledged_by_peer']=bool(stats['all_gap_ranges']) and ack is not None and all(b<=ack['relative_ack'] for a,b in stats['all_gap_ranges'])
            stats['length_anomaly_sequence_progression']=dict(progression);stats['length_anomaly_examples']=examples
            stats['framing']=framing(merged,d,int(c['envelope']['responder_port']),stats)
            # Raw payload is not exported. Hash the exact captured contiguous intervals for reproducibility.
            stats['captured_intervals_sha256']=[{'start':start,'end':start+len(data),'sha256':hashlib.sha256(data).hexdigest()} for start,data in merged]
            item['directions'][d]=stats
        item['real_payload_recovery_complete']=all(v['actual_bytes_complete_to_fin'] for v in item['directions'].values())
        item['outcome']='actual_capture_gaps_remain_no_repair_from_existing_bytes' if not item['real_payload_recovery_complete'] else 'actual_bytes_contiguous_requires_separate_header_semantics_decision'
        out.append(item)
    result={'method':'All physically captured TCP payload bytes are included, even when IPv4 total_length exceeds captured bytes. Merge sequence intervals, compare overlapping bytes, check FIN/peer ACK, and parse initial TLS record framing or HTTP headers without decryption. No bytes invented, no packet edited.','source_pcap_sha256':source['source_pcap']['sha256'],'recovery_input_sha256':recovery.helper.sha(source_path),'candidates':out,'script_sha256':recovery.helper.sha(Path(__file__)),'model_calls':0}
    recovery.save(ROOT/'actual_tcp_byte_evidence.json',result)
    print(json.dumps({'candidates':[{ 'id':r['candidate_id'],'real_payload_recovery_complete':r['real_payload_recovery_complete'],'actual_missing_bytes':{d:s['actual_missing_bytes_to_fin'] for d,s in r['directions'].items()},'actual_gap_count':{d:s['actual_gap_count'] for d,s in r['directions'].items()}} for r in out]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
