"""Run the existing pinned Zeek extractor on unchanged Unraveled byte slices."""
from pathlib import Path
import argparse,gzip,hashlib,json,shutil,struct,sys
ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT.parent/'normal_calibration'
VENDOR=ROOT/'vendor'
FILES=('zeek_runtime.py','analyze.py','export.zeek')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(name):
    VENDOR.mkdir(exist_ok=True)
    vendor_receipt=[]
    for fn in FILES:
        src=UPSTREAM/'code/scripts/fit_contract/vendor_ait/scripts/ait_contract'/fn
        dst=VENDOR/fn
        if not dst.exists():shutil.copyfile(src,dst)
        assert sha(src)==sha(dst)
        vendor_receipt.append({'path':str(src),'snapshot':str(dst),'sha256':sha(dst)})
    sys.path.insert(0,str(VENDOR))
    import zeek_runtime,analyze
    runtime=json.loads((UPSTREAM/'runtime_receipt.json').read_text(encoding='utf-8'))
    runtime['export_script']=str(VENDOR/'export.zeek')
    (ROOT/'runtime_receipt.json').write_text(json.dumps(runtime,ensure_ascii=False,indent=2),encoding='utf-8')
    p=ROOT/'source/unraveled/pcap_slices'/name/'slice.pcap'
    out=ROOT/'network_mapping'/name;out.mkdir(parents=True,exist_ok=True)
    data=p.read_bytes();offset=24;n=short=ipshort=0
    while offset<len(data):
        sec,usec,cap,orig=struct.unpack_from('<IIII',data,offset);frame=data[offset+16:offset+16+cap]
        short+=cap<orig
        if frame[12:14]==b'\x08\x00':ipshort+=len(frame)<14+struct.unpack_from('!H',frame,16)[0]
        n+=1;offset+=16+cap
    assert offset==len(data)
    receipt=zeek_runtime.export_pcap(p,out/'zeek','ignore_checksums',runtime)
    rows=[]
    for line,raw in enumerate(analyze.records(out/'zeek/conn.log'),1):
        c=analyze.normalize_connection(raw,(0,float('inf')))
        raw8={k:c.pop(k) for k in analyze.RAW8};raw8={k:None if v<0 else v for k,v in raw8.items()}
        c.update({'flow_id':f'unraveled:20210621:net1013x:{name}:'+c['uid'],'source_scope':name,'campaign_id':'unraveled:apt_2021_single_campaign','raw8':raw8,'raw8_missing_fields':[k for k,v in raw8.items() if v is None],'label':None,'available_at':None,'source_path':str(out/'zeek/conn.log'),'source_line':line,'zeek_record':raw})
        rows.append(c)
    with gzip.open(out/'all_connections.jsonl.gz','wt',encoding='utf-8',newline='\n') as f:
        for c in rows:f.write(json.dumps(c,ensure_ascii=False,allow_nan=False)+'\n')
    final={'pcap':str(p),'pcap_sha256':sha(p),'packet_records':n,'caplen_less_than_orig_len':short,'ipv4_declared_length_larger_than_frame':ipshort,'packet_representation_modified':False,'connections':len(rows),'missing_raw8':sum(bool(x['raw8_missing_fields']) for x in rows),'export':receipt,'vendor':vendor_receipt,'all_connections_sha256':sha(out/'all_connections.jsonl.gz')}
    (out/'extraction_receipt.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in final.items() if k not in ('export','vendor')},ensure_ascii=False))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('name',choices=['normal_mail','apt_c2']);main(a.parse_args().name)
