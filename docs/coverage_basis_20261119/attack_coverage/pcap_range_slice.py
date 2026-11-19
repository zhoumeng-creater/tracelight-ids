"""Extract bounded contiguous PCAP record intervals, preserving captured bytes.

Seeks use validated record chains, never packet-content-dependent selection.
The derivative is a declared contiguous byte interval, not the full day.
"""
from __future__ import annotations
import argparse,concurrent.futures,datetime,hashlib,json,struct
from pathlib import Path
from range_source import ROOT,get_range,sha

URL='https://dapt2021.s3.amazonaws.com/raw/network-captures/Week5_Day1_06212021/net1013x.pcap'
SIZE=3518820352
PROBE=128*1024

def record(data,i):
    if i+16>len(data):return None
    sec,usec,cap,orig=struct.unpack_from('<IIII',data,i)
    if not (1620000000<=sec<1630000000 and usec<1000000 and 14<=cap<=65535 and cap<=orig<=1048576):return None
    if i+16+cap>len(data):return None
    ether=struct.unpack_from('!H',data,i+28)[0]
    if ether not in (0x800,0x806,0x86dd,0x8100,0x88cc,0x8863,0x8864):return None
    return sec+usec/1e6,i+16+cap

def first_chain(data):
    for i in range(len(data)-64):
        a=record(data,i)
        if a:
            b=record(data,a[1]);c=record(data,b[1]) if b else None
            if c and abs(a[0]-b[0])<3600 and abs(b[0]-c[0])<3600:return i,a[0]
    raise ValueError('No validated three-record PCAP chain in probe')

class Remote:
    def __init__(self,out):self.out=out;self.receipts=[];self.cache={}
    def read(self,start,end):
        key=(start,end)
        if key not in self.cache:
            p=self.out/'ranges'/f'{start}_{end}.bin';r=p.with_suffix('.json')
            if p.exists() and r.exists():
                b=p.read_bytes();rec=json.loads(r.read_text());assert sha(b)==rec['sha256']
            else:
                b,rec=get_range(URL,start,end);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);r.write_text(json.dumps(rec,indent=2))
            assert rec['object_bytes']==SIZE
            self.cache[key]=(b,rec);self.receipts.append(rec)
        return self.cache[key][0]
    def probe(self,offset):
        b=self.read(offset,min(SIZE-1,offset+PROBE-1));i,t=first_chain(b);return offset+i,t
    def bracket(self,target):
        low,lt=self.probe(24);high,ht=self.probe(SIZE-PROBE)
        assert lt<target<ht,(target,lt,ht)
        while high-low>PROBE:
            mid=(low+high)//2;off,t=self.probe(mid)
            if t<target:low,lt=off,t
            else:high,ht=off,t
        return {'low_offset':low,'low_time':lt,'high_offset':high,'high_time':ht}

def extract(name,start,end):
    out=ROOT/'source'/'unraveled'/'pcap_slices'/name;out.mkdir(parents=True,exist_ok=True)
    remote=Remote(out);header=remote.read(0,23)
    assert header[:4]==bytes.fromhex('d4c3b2a1') and struct.unpack_from('<I',header,20)[0]==1
    lo=remote.bracket(start);hi=remote.bracket(end)
    first,last=lo['low_offset'],hi['high_offset']
    assert 0<last-first<128*1024*1024,('bounded slice exceeded',last-first)
    chunks=[(x,min(last-1,x+1024*1024-1)) for x in range(first,last,1024*1024)]
    with concurrent.futures.ThreadPoolExecutor(4) as pool:pieces=list(pool.map(lambda se:remote.read(*se),chunks))
    body=b''.join(pieces);i=0;n=0;times=[];prev=None;outoforder=0
    while i<len(body):
        a=record(body,i);assert a is not None,('bad record',first+i)
        t,j=a;times.append(t);outoforder+=int(prev is not None and t<prev);prev=t;i=j;n+=1
    assert i==len(body)
    assert len({r['etag'] for r in remote.receipts})==1
    data=header+body;p=out/'slice.pcap';p.write_bytes(data)
    receipt={'source_url':URL,'source_object_bytes':SIZE,'source_etag':remote.receipts[0]['etag'],'selection':'contiguous source PCAP record interval chosen by declared action/business times; no model scores','requested_time_start':start,'requested_time_end':end,'source_record_byte_interval_half_open':[first,last],'global_header_source_byte_interval_half_open':[0,24],'lower_bracket':lo,'upper_bracket':hi,'output_path':str(p),'output_bytes':len(data),'sha256':sha(data),'packet_records':n,'min_packet_time':min(times),'max_packet_time':max(times),'adjacent_out_of_order_timestamps':outoforder,'preservation':'all record headers, caplen, orig_len and frame bytes copied unchanged','complete_source_day':False,'limitations':['Contiguous source-byte slice only; possible ongoing connections at both boundaries.','Completeness of selected connections must be checked against original flow row and TCP boundaries.','No assumption that endpoint-only or all unlabelled traffic is normal.'],'requests':remote.receipts}
    (out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('output_path','output_bytes','packet_records','min_packet_time','max_packet_time','sha256')}),flush=True)
    return receipt

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('name',choices=['normal_mail','apt_c2']);ns=a.parse_args()
    if ns.name=='normal_mail':extract(ns.name,1624288300,1624288440)
    else:extract(ns.name,1624318560,1624318800)
