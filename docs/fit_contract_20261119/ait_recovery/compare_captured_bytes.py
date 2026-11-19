"""Check whether the old capture supplies any bytes absent at the firewall.

Only the same three fixed TCP connections are examined. No packet is changed.
"""
import hashlib,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actual_byte_reader',ROOT/'inspect_actual_tcp_bytes.py')
actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
recovery=actual.recovery


def main():
    final=recovery.load(ROOT/'actual_tcp_byte_evidence.json')
    old=recovery.load(recovery.NOW/'ait_attack/checksum_semantics.json')
    old={r['packet_connection_id']:r for r in old['packet_candidates'] if r['packet_connection_id'] in recovery.TARGETS}
    keys={recovery.envelope_key(r['packet_envelope']):cid for cid,r in old.items()}
    payload={cid:{d:[] for d in ('origin','responder')} for cid in old}
    pcap=recovery.NOW/'ait_attack/source/log.pcap.1642684613'
    for ts,linktype,frame,wire in recovery.reader.pcap_records(pcap):
        parsed=recovery.helper.ipv4(frame,linktype)
        if parsed is None:continue
        ip,tcp,m=parsed;cid=keys.get(recovery.tuple_key(m['source_ip'],m['source_port'],m['destination_ip'],m['destination_port']))
        if cid is None:continue
        env=old[cid]['packet_envelope']
        if not float(env['first'])<=ts<=float(env['last']):continue
        d='origin' if m['source_ip']==env['origin_ip'] else 'responder'
        initial=old[cid]['handshake']['syn' if d=='origin' else 'syn_ack']['sequence']
        rel=(m['sequence']-initial-1)%2**32
        declared=m['ip_total_length']-m['ip_header_bytes']-m['tcp_header_bytes'];length=max(0,min(len(tcp)-m['tcp_header_bytes'],declared))
        if length and rel<2**31:payload[cid][d].append({'start':rel,'data':tcp[m['tcp_header_bytes']:m['tcp_header_bytes']+length]})
    results=[]
    for r in final['candidates']:
        cid=r['old_packet_id'];comp={}
        for d,s in r['directions'].items():
            merged,stats=actual.merge_real_bytes(payload[cid][d],s['fin_payload_end'])
            hashes=[{'start':start,'end':start+len(data),'sha256':hashlib.sha256(data).hexdigest()} for start,data in merged]
            comp[d]={'same_interval_boundaries_and_all_payload_sha256':hashes==s['captured_intervals_sha256'],'old_unique_captured_payload_bytes':stats['captured_unique_payload_bytes'],'old_missing_bytes_to_fin':stats['actual_missing_bytes_to_fin'],'old_gap_count':stats['actual_gap_count'],'old_overlap_conflicts':stats['conflicting_overlap_bytes']}
        results.append({'old_packet_id':cid,'candidate_id':r['candidate_id'],'directions':comp,'two_capture_union_adds_no_payload_bytes':all(s['same_interval_boundaries_and_all_payload_sha256'] for s in comp.values())})
    result={'candidates':results,'old_pcap_sha256':recovery.helper.sha(pcap),'firewall_actual_byte_evidence_sha256':recovery.helper.sha(ROOT/'actual_tcp_byte_evidence.json'),'method':'Merge every physically captured payload byte by TCP sequence separately in each capture, then compare all contiguous interval boundaries and SHA256. Equal inventories prove that combining these two captures adds no observed payload bytes.','script_sha256':recovery.helper.sha(Path(__file__))}
    recovery.save(ROOT/'cross_capture_byte_identity.json',result)
    print(json.dumps({'union_recovers_additional_bytes':any(not r['two_capture_union_adds_no_payload_bytes'] for r in results),'fixed_connections':len(results)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
