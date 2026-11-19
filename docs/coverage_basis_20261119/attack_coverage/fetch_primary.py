"""Bounded public author-source retrieval; never downloads a dataset wholesale."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
def save_json(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def fetch(url,name,max_bytes=8*1024*1024):
    p=ROOT/'source'/name;receipt=ROOT/'receipts'/(name.replace('/','__')+'.json')
    if p.exists() and receipt.exists():
        b=p.read_bytes();old=json.loads(receipt.read_text(encoding='utf-8'));assert hashlib.sha256(b).hexdigest()==old['sha256'];return b
    with requests.get(url,stream=True,timeout=(15,60)) as r:
        r.raise_for_status();assert r.status_code==200,r.status_code
        declared=r.headers.get('Content-Length')
        if declared:assert int(declared)<=max_bytes,(name,declared)
        blocks=[];n=0
        for block in r.iter_content(256*1024):
            n+=len(block);assert n<=max_bytes,(name,n);blocks.append(block)
        b=b''.join(blocks)
        if declared:assert len(b)==int(declared)
        headers={k:r.headers.get(k) for k in ('Content-Length','ETag','Last-Modified','Content-Type')}
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    save_json(receipt,{'url':url,'local_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'response_headers':headers})
    print(json.dumps({'file':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}),flush=True)
    return b

if __name__=='__main__':
    from concurrent.futures import ThreadPoolExecutor
    jobs=[
      ('https://gitlab.com/api/v4/projects/asu22%2Funraveled/repository/commits/master','unraveled/commit.json'),
      ('https://gitlab.com/api/v4/projects/asu22%2Funraveled/repository/tree?per_page=100','unraveled/root_tree.json'),
      ('https://dapt2021.s3.amazonaws.com/README.txt','unraveled/S3_README.txt'),
      ('https://api.github.com/repos/AmyDylen/IEU-C2D/git/trees/main?recursive=1','ieu_c2d/tree.json'),
      ('https://raw.githubusercontent.com/AmyDylen/IEU-C2D/main/README.md','ieu_c2d/README.md'),
      ('https://dapt2021.s3.amazonaws.com/?list-type=2&prefix=raw/host-logs/10_1_3_8/&max-keys=1000','unraveled/host_10_1_3_8_listing.xml'),
      ('https://dapt2021.s3.amazonaws.com/?list-type=2&prefix=raw/network-captures/Week5_Day2_06222021/&max-keys=1000','unraveled/pcap_day_listing.xml'),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda j:fetch(*j),jobs))
