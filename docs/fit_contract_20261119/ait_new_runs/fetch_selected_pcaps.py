"""Prepare or explicitly fetch one complete, selected fox/santos PCAP member.

Default invocation only writes a plan. --execute must name one run and host.
Original packet bytes are saved unchanged; this script does no Zeek conversion.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, time, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from fetch_sources import ROOT, index, range_bytes, write_json

RUNS=('santos','fox')
HOSTS=('intranet_server','cloud_share')
CHUNK=8*1024*1024

def selection():
    rows=[]
    for run in RUNS:
        for host in HOSTS:
            hits=[x for x in index(run+'.zip') if x['path'].startswith(f'gather/{host}/logs/suricata/log.pcap.')]
            assert len(hits)==1,(run,host,len(hits))
            rows.append({'run':run,'host':host,'archive':run+'.zip',**hits[0]})
    return rows

def digest(path):
    sha=hashlib.sha256();crc=0;n=0
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            sha.update(b);crc=zlib.crc32(b,crc);n+=len(b)
    return {'bytes':n,'crc32':crc&0xffffffff,'sha256':sha.hexdigest()}

def fetch(row,workers):
    dest=ROOT/'source'/row['run']/row['path']
    receipt=ROOT/'pcap_receipts'/f'{row["run"]}_{row["host"]}.json'
    if dest.exists():
        actual=digest(dest)
        assert actual['bytes']==row['bytes'] and actual['crc32']==row['CRC'],actual
        assert receipt.exists(),'Verified original exists but receipt missing; do not silently replace provenance.'
        old=json.loads(receipt.read_text(encoding='utf-8'));assert old['sha256']==actual['sha256']
        print(json.dumps({'reused_verified_original':str(dest),**actual}),flush=True);return
    cache=ROOT/'pcap_ranges'/row['run']/row['host']
    header=range_bytes(row['archive'],row['offset'],row['offset']+255,cache/'local_header.bin')
    v=struct.unpack('<IHHHHHIIIHH',header[:30]);assert v[0]==0x04034b50 and v[3]==8
    assert header[30:30+v[-2]].decode()==row['path']
    offset=row['offset']+30+v[-2]+v[-1]
    jobs=[(i,start,min(start+CHUNK,offset+row['compressed'])-1) for i,start in enumerate(range(offset,offset+row['compressed'],CHUNK))]
    def one(job):
        i,start,end=job;p=cache/f'{i:04d}.deflate'
        range_bytes(row['archive'],start,end,p)
        return i,p.stat().st_size
    began=time.perf_counter();done=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,j):j for j in jobs}
        for f in as_completed(futures):
            i,n=f.result();done+=n
            print(json.dumps({'run':row['run'],'host':row['host'],'chunk':i,'compressed_completed':done,'compressed_total':row['compressed']}),flush=True)
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.partial');decoder=zlib.decompressobj(-15)
    with tmp.open('wb') as out:
        for i,_,_ in jobs:
            with (cache/f'{i:04d}.deflate').open('rb') as src:
                for b in iter(lambda:src.read(1024*1024),b''):out.write(decoder.decompress(b))
        out.write(decoder.flush())
    assert decoder.eof and not decoder.unused_data
    actual=digest(tmp)
    assert actual['bytes']==row['bytes'] and actual['crc32']==row['CRC'],actual
    tmp.replace(dest)
    result={**row,**actual,'source_record':'https://zenodo.org/records/19483937','source_url':f'https://zenodo.org/api/records/19483937/files/{row["archive"]}/content','data_offset':offset,'local_path':str(dest),'packet_bytes_unchanged':True,'pcap_record_fields_unchanged':True,'elapsed_seconds':time.perf_counter()-began,'chunk_bytes':CHUNK,'workers':workers}
    write_json(receipt,result);print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',choices=RUNS);ap.add_argument('--host',choices=HOSTS)
    ap.add_argument('--execute',action='store_true');ap.add_argument('--workers',type=int,default=4,choices=(1,2,3,4))
    a=ap.parse_args();rows=selection()
    completed=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'pcap_receipts').glob('*.json'))]
    started=a.execute or bool(completed) or any((ROOT/'pcap_ranges').glob('*/*/*.deflate'))
    plan={'members':rows,'total_compressed_bytes':sum(x['compressed'] for x in rows),'total_uncompressed_bytes':sum(x['bytes'] for x in rows),'execution_started':started,'completed_member_count':len(completed),'all_selected_members_completed':len(completed)==len(rows),'completed_receipts':completed,'capture_end_and_action_coverage':'must inspect actual complete original after fetching; filename is only start-time hint','derived_representation':'not created by this downloader; originals immutable'}
    write_json(ROOT/'selected_pcap_download_plan.json',plan)
    if not a.execute:
        print(json.dumps(plan,indent=2));return
    if not a.run or not a.host:ap.error('--execute requires --run and --host')
    fetch(next(x for x in rows if x['run']==a.run and x['host']==a.host),a.workers)

if __name__=='__main__':main()
