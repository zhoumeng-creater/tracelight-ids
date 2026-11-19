"""Full unchanged Unraveled PCAP -> fixed Zeek, with stable physical flow IDs."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,json,shutil,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'source/unraveled/full_capture'
OUT=ROOT/'full_capture'
sys.path.insert(0,str(ROOT/'vendor'))
import zeek_runtime,analyze

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def quality(p):
    result={'packet_records':0,'caplen_less_than_orig_len':0,'ipv4_frame_shorter_than_declared_total':0,'min_packet_time':None,'max_packet_time':None,'truncated_tail':None,'caplen_gt_snaplen':0,'adjacent_timestamp_reversals':0}
    with p.open('rb') as f:
        head=f.read(24);assert head[:4]==bytes.fromhex('d4c3b2a1')
        snap,link=struct.unpack_from('<II',head,16);assert link==1
        result['pcap_snaplen']=snap;result['linktype']=link;prev=None
        while True:
            offset=f.tell();h=f.read(16)
            if not h:break
            if len(h)!=16:result['truncated_tail']={'offset':offset,'header_bytes':len(h)};break
            sec,usec,cap,orig=struct.unpack('<IIII',h);assert 0<cap<10485760 and usec<1000000,(offset,cap,usec)
            x=f.read(cap)
            if len(x)!=cap:result['truncated_tail']={'offset':offset,'expected_frame_bytes':cap,'actual_frame_bytes':len(x)};break
            result['packet_records']+=1;result['caplen_less_than_orig_len']+=cap<orig;result['caplen_gt_snaplen']+=cap>snap
            if len(x)>=34 and x[12:14]==b'\x08\x00':result['ipv4_frame_shorter_than_declared_total']+=cap<14+struct.unpack_from('!H',x,16)[0]
            t=sec+usec/1e6;result['min_packet_time']=t if result['min_packet_time'] is None else min(result['min_packet_time'],t);result['max_packet_time']=t if result['max_packet_time'] is None else max(result['max_packet_time'],t)
            result['adjacent_timestamp_reversals']+=prev is not None and t<prev;prev=t
    result['representation_modified']=False
    return result

def main(wait):
    receipt_path=SRC/'download_receipt.json'
    if wait:
        while not receipt_path.exists():time.sleep(10)
    download=json.loads(receipt_path.read_text(encoding='utf-8'));assert download['status']=='complete'
    p=Path(download['local_path']);OUT.mkdir(parents=True,exist_ok=True)
    print('Full original PCAP verified by download receipt; checking record integrity',flush=True)
    if not (OUT/'pcap_quality.json').exists():dump(OUT/'pcap_quality.json',quality(p))
    quality_data=json.loads((OUT/'pcap_quality.json').read_text(encoding='utf-8'))
    processing_pcap=p;processing_sha=download['sha256'];derivation=None
    if quality_data['truncated_tail']:
        # The source object ends inside its last record. Keep the original,
        # and expose its complete-record prefix without rewriting any record.
        stop=quality_data['truncated_tail']['offset']
        derivative=SRC/'net1013x.complete_records.pcap'
        derivation_path=SRC/'complete_records_receipt.json'
        failed=OUT/'zeek_original_truncated_tail'
        if (OUT/'zeek/zeek-stderr.txt').exists() and not (OUT/'zeek/export-receipt.json').exists():
            assert 'truncated dump file' in (OUT/'zeek/zeek-stderr.txt').read_text(encoding='utf-8')
            assert not failed.exists()
            (OUT/'zeek').resolve().relative_to(ROOT.resolve());failed.resolve().relative_to(ROOT.resolve())
            (OUT/'zeek').rename(failed)
        if derivation_path.exists():
            derivation=json.loads(derivation_path.read_text(encoding='utf-8'))
            assert derivative.stat().st_size==stop
        else:
            digest=hashlib.sha256()
            with p.open('rb') as src,derivative.open('wb') as dst:
                left=stop
                while left:
                    b=src.read(min(left,8*1024*1024));assert b;dst.write(b);digest.update(b);left-=len(b)
            derivation={'source_pcap':str(p),'source_sha256':download['sha256'],'processing_pcap':str(derivative),'processing_sha256':digest.hexdigest(),'preserved_source_interval_half_open':[0,stop],'incomplete_trailing_record_bytes_excluded':p.stat().st_size-stop,'complete_packet_records_preserved':quality_data['packet_records'],'all_complete_record_headers_and_frame_bytes_unchanged':True,'original_failure_logs':str(failed),'tail_record':quality_data['truncated_tail'],'reason':'source S3 object ends inside its last record; no caplen/orig_len or frame repair'}
            dump(derivation_path,derivation)
        processing_pcap=derivative;processing_sha=derivation['processing_sha256']
    runtime=json.loads((ROOT/'runtime_receipt.json').read_text(encoding='utf-8'))
    export_path=OUT/'zeek/export-receipt.json'
    if export_path.exists():
        export=json.loads(export_path.read_text(encoding='utf-8'));assert export['pcap_sha256']==processing_sha
    else:
        print('Starting fixed Zeek 8.0.11 -C on full unchanged PCAP',flush=True)
        export=zeek_runtime.export_pcap(processing_pcap,OUT/'zeek','ignore_checksums',runtime)
        assert export['pcap_sha256']==processing_sha
    ids=set();counts=collections.Counter()
    with (OUT/'zeek/conn.log').open(encoding='utf-8') as incoming,gzip.open(OUT/'base_connections.jsonl.gz','wt',encoding='utf-8',newline='\n') as out:
        for line,txt in enumerate(incoming,1):
            raw=json.loads(txt);c=analyze.normalize_connection(raw,(0,float('inf')))
            features={k:c.pop(k) for k in analyze.RAW8};features={k:None if v<0 else v for k,v in features.items()}
            key=[download['sha256'],c['native_protocol'],c['source'],features['orig_port'],c['destination'],features['resp_port'],format(c['connection_start'],'.6f')]
            digest=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()
            flow_id='unraveled:20210621:net1013x:'+digest[:32];assert flow_id not in ids,('duplicate stable flow key',key);ids.add(flow_id)
            c.update({'flow_id':flow_id,'stable_flow_id':flow_id,'stable_flow_key_sha256':digest,'stable_flow_key':key,'source_scope':'full_author_capture_20210621_net1013x','source_address':c['source'],'destination_address':c['destination'],'observed_end':c['zeek_reported_end'],'physical_flow_id':flow_id,'campaign_id':'unraveled:apt_2021_single_campaign','group':'unraveled:apt_2021_single_campaign','monitored_host':'10.1.3.8','involves_monitored_host':'10.1.3.8' in (c['source'],c['destination']),'raw8':features,'raw8_missing_fields':[k for k,v in features.items() if v is None],'label':None,'role':'opened_source_development','fit_validation_role':None,'actual_available_at':None,'available_at':None,'source_path':str(OUT/'zeek/conn.log'),'source_line':line,'source_pcap_sha256':download['sha256'],'zeek_record':raw})
            out.write(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n');counts['all_connections']+=1;counts['involving_monitored_host']+=c['involves_monitored_host'];counts['raw8_missing']+=bool(c['raw8_missing_fields'])
    result={'status':'complete','download_receipt':str(receipt_path),'source_pcap_sha256':download['sha256'],'processing_pcap_sha256':processing_sha,'complete_records_derivation':derivation,'quality':quality_data,'export':export,'counts':dict(counts),'base_connections_sha256':zeek_runtime.sha256(OUT/'base_connections.jsonl.gz'),'role':'opened_source_development','fit_validation_role':None,'training_or_scoring_run':False}
    dump(OUT/'extraction_receipt.json',result);print(json.dumps({'status':'full_export_complete','counts':dict(counts)},ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-download',action='store_true');main(p.parse_args().wait_download)
