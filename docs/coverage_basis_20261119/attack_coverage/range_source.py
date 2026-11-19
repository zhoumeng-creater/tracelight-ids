"""Fetch bounded, byte-exact parts of an author-hosted object with receipts."""
from __future__ import annotations
import concurrent.futures,hashlib,json,re,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
def sha(b): return hashlib.sha256(b).hexdigest()
def get_range(url,start,end):
    for attempt in range(3):
        try:
            request_url=url+('&' if '?' in url else '?')+f'byte_range={start}-{end}-{attempt}'
            r=requests.get(request_url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,45))
            r.raise_for_status()
            assert r.status_code==206,(r.status_code,r.headers.get('Content-Range'))
            m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',r.headers.get('Content-Range',''))
            assert m and tuple(map(int,m.groups()[:2]))==(start,end)
            assert len(r.content)==end-start+1
            return r.content,{'url':url,'range_start':start,'range_end':end,'object_bytes':int(m[3]),'etag':r.headers.get('ETag'),'sha256':sha(r.content),'bytes':len(r.content)}
        except (requests.RequestException,AssertionError):
            if attempt==2:raise
            time.sleep(1+attempt)

def fetch_complete(url,name,size,chunk_bytes=1024*1024):
    p=ROOT/'source'/name
    receipt=ROOT/'receipts'/(name.replace('/','__')+'.json')
    if p.exists() and receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf-8')); assert sha(p.read_bytes())==old['sha256'];return p
    ranges=[(a,min(size-1,a+chunk_bytes-1)) for a in range(0,size,chunk_bytes)]
    blocks=[]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for b,rec in pool.map(lambda se:get_range(url,*se),ranges):
            blocks.append((b,rec));print('range complete',rec['range_end']+1,'/',size,flush=True)
    assert {r['etag'] for _,r in blocks}.__len__()==1
    assert all(r['object_bytes']==size for _,r in blocks)
    data=b''.join(b for b,_ in blocks);assert len(data)==size
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    receipt.write_text(json.dumps({'url':url,'local_path':str(p),'bytes':size,'sha256':sha(data),'complete_object':True,'retrieval':'validated contiguous HTTP ranges','range_receipts':[r for _,r in blocks]},ensure_ascii=False,indent=2),encoding='utf-8')
    return p

if __name__=='__main__':
    fetch_complete('https://dapt2021.s3.amazonaws.com/processed/network-flows/Week5_Day1_06212021/net1013x_Flow_labeled.csv','unraveled/flow_20210621_net1013x.csv',26038650)
