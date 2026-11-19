"""Test declared-IP-length truncation metadata with frozen Zeek 8.0.11 -C.

Only PCAP record orig_len can change. Packet timestamps, caplen and all captured
frame bytes remain original. No payload recovery or training admission is claimed.
"""
from collections import Counter,defaultdict
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'declared_length_representation'
spec=importlib.util.spec_from_file_location('recovery',ROOT/'recover_download_observations.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def records(path):
    with path.open('rb') as f:
        header=f.read(24)
        magic=header[:4]
        if magic==bytes.fromhex('d4c3b2a1'):endian='<';factor=1e6
        elif magic==bytes.fromhex('a1b2c3d4'):endian='>';factor=1e6
        elif magic==bytes.fromhex('4d3cb2a1'):endian='<';factor=1e9
        elif magic==bytes.fromhex('a1b23c4d'):endian='>';factor=1e9
        else:raise ValueError('Unsupported PCAP header')
        linktype=struct.unpack(endian+'I',header[20:24])[0]
        yield header,endian,linktype
        number=0
        while True:
            h=f.read(16)
            if not h:break
            assert len(h)==16
            sec,sub,cap,orig=struct.unpack(endian+'IIII',h);data=f.read(cap);assert len(data)==cap
            number+=1;yield number,sec+sub/factor,h,data,orig


def apply_rule(source,group,candidates,fin_evidence,write_full):
    indexed=defaultdict(list)
    for cid,p in candidates.items():indexed[r.envelope_key(p)].append(cid)
    generator=records(source);header,endian,linktype=next(generator)
    destination=OUT/(group+'-declared-length.pcap')
    output=destination.open('wb') if write_full else None
    if output:output.write(header)
    changes=OUT/(group+'-record-changes.jsonl.gz')
    counts={cid:{d:Counter() for d in ('origin','responder')} for cid in candidates}
    preserved_hash=hashlib.sha256();changed=0;total=0
    with gzip.open(changes,'wt',encoding='utf-8',newline='\n') as journal:
        for number,ts,h,frame,orig in generator:
            total+=1;new_orig=orig;parsed=r.helper.ipv4(frame,linktype);matched=[]
            if parsed is not None:
                ip,tcp,m=parsed;key=r.tuple_key(m['source_ip'],m['source_port'],m['destination_ip'],m['destination_port'])
                matched=[cid for cid in indexed.get(key,[]) if float(candidates[cid]['first'])<=ts<=float(candidates[cid]['last'])]
                assert len(matched)<=1
                if matched:
                    cid=matched[0];p=candidates[cid];side='origin' if (m['source_ip'],m['source_port'])==(p['origin_ip'],int(p['origin_port'])) else 'responder'
                    c=counts[cid][side];c['packets']+=1;c['declared_ip_bytes']+=m['ip_total_length']
                    complete_header=m['ip_header_bytes']>=20 and m['tcp_header_bytes']>=20 and m['ip_header_bytes']+m['tcp_header_bytes']<=len(ip) and m['ip_total_length']>=m['ip_header_bytes']+m['tcp_header_bytes']
                    valid_ip=complete_header and r.helper.folded_sum(ip[:m['ip_header_bytes']])==65535 and not m['fragment_bits']
                    assert valid_ip, (cid,number,'IP/TCP header precondition failed')
                    l2=len(frame)-len(ip);declared_frame=l2+m['ip_total_length']
                    if orig<declared_frame:
                        # Existing byte-level evidence, including all physical captured prefixes,
                        # establishes the TCP sequence extent and cumulative peer ACK.
                        evidence=fin_evidence[cid][side]
                        initial=evidence['initial_sequence'];relative=(m['sequence']-initial-1)%2**32
                        declared_payload=m['ip_total_length']-m['ip_header_bytes']-m['tcp_header_bytes']
                        assert relative<2**31 and relative+declared_payload<=evidence['fin_payload_end']
                        assert relative+declared_payload<=evidence['maximum_peer_ACK']['relative_ack']
                        new_orig=declared_frame;changed+=1;c['orig_len_changes']+=1
                        journal.write(json.dumps({'source_record_index_1based':number,'packet_candidate_id':cid,'timestamp':ts,'caplen':len(frame),'old_orig_len':orig,'new_orig_len':new_orig,'L2_offset':l2,'ipv4_total_length':m['ip_total_length'],'ipv4_header_checksum_valid':True,'tcp_header_complete':True,'relative_tcp_sequence':relative,'declared_tcp_payload_end':relative+declared_payload,'peer_ack_limit':evidence['maximum_peer_ACK']['relative_ack']},separators=(',',':'))+'\n')
            if not write_full:assert new_orig==orig,'No-op control group unexpectedly requires output'
            if output:output.write(h[:12]+struct.pack(endian+'I',new_orig));output.write(frame)
            preserved_hash.update(h[:12]);preserved_hash.update(frame)
    if output:output.close()
    for cid,p in candidates.items():
        assert counts[cid]['origin']['packets']==int(p['origin_packets']) and counts[cid]['responder']['packets']==int(p['responder_packets'])
    derived=destination if write_full else source
    # Independently read the output: the only differences permitted are record orig_len.
    compared=0;original=records(source);modified=records(derived);assert next(original)==next(modified)
    for left,right in zip(original,modified,strict=True):
        assert left[:2]==right[:2] and left[2][:12]==right[2][:12] and left[3]==right[3]
        assert right[4]>=left[4]
        compared+=left[4]!=right[4]
    assert compared==changed
    return {'source':str(source),'source_sha256':r.helper.sha(source),'derived':str(derived),'derived_sha256':r.helper.sha(derived),'full_capture_context_retained':True,'derived_is_original_noop':not write_full,'all_records':total,'changed_records':changed,'unchanged_timestamps_caplen_frame_bytes_sha256':preserved_hash.hexdigest(),'change_journal':str(changes),'change_journal_sha256':r.helper.sha(changes),'counts':{str(cid):{d:dict(c) for d,c in ds.items()} for cid,ds in counts.items()},'only_orig_len_changed_verified':True}


def signature(row,raw8):return (row['source'],int(row['orig_port']),row['destination'],int(row['resp_port']),float(row['connection_start']),*(float(row[k]) for k in raw8))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    attack=r.load(r.NOW/'ait_attack/checksum_semantics.json')
    attack_rows={c['packet_connection_id']:c for c in attack['packet_candidates'] if not c['model_row_blocked_on_capture_semantics'] or c['packet_connection_id'] in r.TARGETS}
    assert len(attack_rows)==35
    cloud=r.load(r.NOW/'ait/checksum_semantics.json');cloud_rows={c['packet_connection_id']:c for c in cloud['packet_candidates']};assert len(cloud_rows)==38
    bytes_evidence=r.load(ROOT/'actual_tcp_byte_evidence.json');fin={}
    for c in bytes_evidence['candidates']:
        cid=c['old_packet_id'];fin[cid]={}
        for d,ds in c['directions'].items():fin[cid][d]={**ds,'initial_sequence':attack_rows[cid]['handshake']['syn' if d=='origin' else 'syn_ack']['sequence']}
    intranet=r.NOW/'ait_attack/source/log.pcap.1642684613';cloudpcap=r.NOW/'ait/source/cloud-share-log.pcap.1642684647'
    original_hashes={str(p):r.helper.sha(p) for p in (intranet,cloudpcap)}
    intranet_receipt=apply_rule(intranet,'intranet',{k:v['packet_envelope'] for k,v in attack_rows.items()},fin,True)
    cloud_receipt=apply_rule(cloudpcap,'cloud',{k:v['frozen_checksum_mapping']['packet_envelope'] for k,v in cloud_rows.items()},{},False)
    r.save(OUT/'representation_receipt.json',{'intranet':intranet_receipt,'cloud':cloud_receipt,'rule':'For scoped IPv4/TCP records with complete valid IPv4 and TCP headers and sequence/ACK-supported declared length, orig_len=max(orig_len,L2_offset+IPv4_total_length). Leave caplen, all frame bytes and timestamps unchanged. This represents a partially captured declared IP datagram, not an observed physical Ethernet wire frame length.','scope':'Fixed 3 download, 32 control and 38 cloud packet candidates; source labels never consulted by the byte transformation.','source_pcap_sha256_before':original_hashes})
    print(json.dumps({'stage':'representation','intranet_changed_records':intranet_receipt['changed_records'],'cloud_changed_records':cloud_receipt['changed_records']}),flush=True)
    sys.path.insert(0,str(r.REPO/'scripts/ait_contract'))
    import zeek_runtime,analyze
    runtime=r.load(r.NOW/'ait/runtime-receipt.json');assert runtime['version']=='8.0.11'
    export=zeek_runtime.export_pcap(Path(intranet_receipt['derived']),OUT/'zeek-ignore_checksums','ignore_checksums',runtime)
    new=[analyze.normalize_connection(row,(0,float('inf'))) for row in analyze.records(OUT/'zeek-ignore_checksums/conn.log')]
    comparison=[]
    for cid,old in attack_rows.items():
        p=old['packet_envelope'];key=r.envelope_key(p)
        rows=[row for row in new if r.tuple_key(row['source'],row['orig_port'],row['destination'],row['resp_port'])==key and float(p['first'])-0.00001<=row['connection_start']<=float(p['last'])+0.00001]
        previous=old['zeek_rows']['ignore_checksums']
        old_sig=Counter(signature(row,analyze.RAW8) for row in previous);new_sig=Counter(signature(row,analyze.RAW8) for row in rows)
        totals={d:{'pkts':sum(row[('orig' if d=='origin' else 'resp')+'_pkts'] for row in rows),'ip_bytes':sum(row[('orig' if d=='origin' else 'resp')+'_ip_bytes'] for row in rows)} for d in ('origin','responder')}
        expected=intranet_receipt['counts'][str(cid)]
        all_counted=all(totals[d]['pkts']==expected[d]['packets'] and totals[d]['ip_bytes']==expected[d]['declared_ip_bytes'] for d in ('origin','responder'))
        old_identity=Counter((row['source'],row['orig_port'],row['destination'],row['resp_port'],float(row['connection_start']),float(row['duration_seconds'])) for row in previous)
        new_identity=Counter((row['source'],str(row['orig_port']),row['destination'],str(row['resp_port']),float(row['connection_start']),float(row['duration_seconds'])) for row in rows)
        comparison.append({'packet_candidate_id':cid,'role':'download' if cid in r.TARGETS else 'control','old_connections':previous,'derived_connections':rows,'raw8_multiset_identical':old_sig==new_sig,'endpoint_start_duration_identical':old_identity==new_identity,'declared_packet_and_ip_byte_totals_all_counted':all_counted,'expected_packet_counts_and_declared_bytes':expected,'zeek_totals':totals})
    cloud_compare=[{'packet_candidate_id':cid,'byte_identical_input':True,'raw8_multiset_identical':True,'connections':c['frozen_checksum_mapping']['arms']['ignore_checksums']} for cid,c in cloud_rows.items()]
    assert cloud_receipt['changed_records']==0 and cloud_receipt['derived_sha256']==cloud_receipt['source_sha256']
    before_after={p:h==r.helper.sha(Path(p)) for p,h in original_hashes.items()};assert all(before_after.values())
    result={'status':'header_feature_representation_validation_only','training_admission':False,'model_calls':0,'packet_payload_recovery':False,'representation_receipt':str(OUT/'representation_receipt.json'),'zeek_version':'8.0.11','zeek_mode':'ignore_checksums','new_zeek_export_count':1,'export_receipt':export,'intranet':comparison,'cloud':cloud_compare,'all_originals_unchanged':before_after,'summary':{'changed_download_records':intranet_receipt['changed_records'],'download_candidates':3,'downloads_all_header_statistics_counted':sum(x['role']=='download' and x['declared_packet_and_ip_byte_totals_all_counted'] for x in comparison),'downloads_endpoint_start_duration_unchanged':sum(x['role']=='download' and x['endpoint_start_duration_identical'] for x in comparison),'control_packet_candidates':32,'control_raw8_unchanged':sum(x['role']=='control' and x['raw8_multiset_identical'] for x in comparison),'control_zeek_records':sum(len(x['derived_connections']) for x in comparison if x['role']=='control'),'cloud_packet_candidates':38,'cloud_raw8_unchanged':38,'cloud_zeek_records':sum(len(x['connections']) for x in cloud_compare)},'interpretation':'Packet counts are capture records and IP byte totals are IPv4-declared datagram bytes. Offload aggregation means these are not established physical wire packet/byte counts. Missing payload remains missing. Labels, grouping, and admission need a common input contract decision.','official_sources':['https://raw.githubusercontent.com/zeek/zeek/v8.0.11/src/packet_analysis/protocol/ip/IP.cc','https://raw.githubusercontent.com/zeek/zeek/v8.0.11/src/packet_analysis/protocol/tcp/TCP.cc','https://www.stratosphereips.org/blog/2015/7/17/new-dataset-ctu-13-extended-now-includes-pcap-files-of-normal-traffic'],'script_sha256':r.helper.sha(Path(__file__))}
    r.save(OUT/'validation.json',result);print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
