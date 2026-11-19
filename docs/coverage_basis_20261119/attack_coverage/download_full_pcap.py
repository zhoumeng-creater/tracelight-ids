"""Resume one fixed public PCAP, validating every byte range and source ETag."""
from __future__ import annotations
import concurrent.futures,hashlib,json,os,time
from pathlib import Path
from range_source import ROOT,get_range,sha

URL='https://dapt2021.s3.amazonaws.com/raw/network-captures/Week5_Day1_06212021/net1013x.pcap'
SIZE=3518820352
ETAG='"f752b111fd79f14d3712f9ccdb046c9e-210"'
OUT=ROOT/'source/unraveled/full_capture'
PCAP=OUT/'net1013x.pcap'
JOURNAL=OUT/'range_journal.jsonl'

def merged(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if result and lo<=result[-1][1]:result[-1][1]=max(hi,result[-1][1])
        else:result.append([lo,hi])
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not PCAP.exists():
        assert not JOURNAL.exists(),'Journal without target file'
        with PCAP.open('wb') as f:f.truncate(SIZE)
    assert PCAP.stat().st_size==SIZE
    done=[];reused=0
    with PCAP.open('r+b') as target:
        if JOURNAL.exists():
            for line in JOURNAL.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);start,end=r['range_start'],r['range_end']
                target.seek(start);b=target.read(end-start+1)
                assert sha(b)==r['sha256'],('resume range mismatch',start,end)
                assert r['etag']==ETAG and r['object_bytes']==SIZE
                done.append([start,end+1])
        if not done:
            for rp in sorted((ROOT/'source/unraveled/pcap_slices').glob('*/ranges/*.json')):
                r=json.loads(rp.read_text(encoding='utf-8'))
                assert r['url']==URL and r['object_bytes']==SIZE and r['etag']==ETAG
                b=rp.with_suffix('.bin').read_bytes();assert sha(b)==r['sha256']
                target.seek(r['range_start']);target.write(b);target.flush();os.fsync(target.fileno())
                r={**r,'retrieval':'reused_verified_prior_range','prior_receipt':str(rp)}
                with JOURNAL.open('a',encoding='utf-8') as log:log.write(json.dumps(r)+'\n');log.flush();os.fsync(log.fileno())
                done.append([r['range_start'],r['range_end']+1])
            reused=sum(hi-lo for lo,hi in merged(done))
        spans=merged(done);done_bytes=sum(hi-lo for lo,hi in spans)
        gaps=[];pos=0
        for lo,hi in spans:
            if pos<lo:gaps.append((pos,lo))
            pos=hi
        if pos<SIZE:gaps.append((pos,SIZE))
        jobs=[(lo,min(hi-1,lo+4*1024*1024-1)) for a,hi in gaps for lo in range(a,hi,4*1024*1024)]
        initial=done_bytes;start_time=time.monotonic();last=0
        print(json.dumps({'status':'start','total_bytes':SIZE,'already_verified_bytes':done_bytes,'prior_slice_bytes_reused':reused,'remaining_ranges':len(jobs),'workers':4}),flush=True)
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            iterator=iter(jobs);active={}
            for job in list(next(iterator,None) for _ in range(4)):
                if job is not None:active[pool.submit(get_range,URL,*job)]=job
            while active:
                ready,_=concurrent.futures.wait(active,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    job=active.pop(future);b,r=future.result()
                    assert r['object_bytes']==SIZE and r['etag']==ETAG
                    target.seek(r['range_start']);target.write(b);target.flush();os.fsync(target.fileno())
                    with JOURNAL.open('a',encoding='utf-8') as log:log.write(json.dumps(r)+'\n');log.flush();os.fsync(log.fileno())
                    done_bytes+=len(b)
                    nxt=next(iterator,None)
                    if nxt is not None:active[pool.submit(get_range,URL,*nxt)]=nxt
                elapsed=time.monotonic()-start_time
                if elapsed-last>=30 or not active:
                    rate=(done_bytes-initial)/max(elapsed,.001);eta=(SIZE-done_bytes)/rate if rate else None
                    status={'status':'transferring','verified_bytes':done_bytes,'total_bytes':SIZE,'percent':round(100*done_bytes/SIZE,2),'MiB_per_second':round(rate/1024**2,2),'eta_seconds':round(eta) if eta is not None else None}
                    (OUT/'progress.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status),flush=True);last=elapsed
    assert done_bytes==SIZE
    digest=hashlib.sha256()
    with PCAP.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    receipt={'status':'complete','source_url':URL,'source_etag':ETAG,'object_bytes':SIZE,'local_path':str(PCAP),'sha256':digest.hexdigest(),'journal_path':str(JOURNAL),'journal_sha256':sha(JOURNAL.read_bytes()),'verified_byte_coverage':[[0,SIZE]],'prior_range_cache_reuse':True,'retrieval':'validated HTTP206 ranges, fixed source ETag, 4 workers; resume ranges locally SHA checked','scope':'one complete author capture, no model selection; earlier slices retained as history'}
    (OUT/'download_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
