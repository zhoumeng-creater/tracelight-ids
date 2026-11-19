"""Fetch named fox/santos ZIP metadata or members; verify every bounded range."""
from __future__ import annotations
import argparse, hashlib, json, struct, zlib
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
META = ROOT / 'metadata'
RECORD = 'https://zenodo.org/api/records/19483937'
ALLOWED = {'fox.zip', 'fox_no-pcaps.zip', 'santos.zip', 'santos_no-pcaps.zip'}

def sha(b): return hashlib.sha256(b).hexdigest()
def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def metadata():
    p = META / 'zenodo_record_19483937.json'
    if not p.exists():
        r = requests.get(RECORD, timeout=(15,45)); r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(r.content)
    return json.loads(p.read_text(encoding='utf-8'))

def range_bytes(archive, start, end, cache):
    info = next(x for x in metadata()['files'] if x['key'] == archive)
    expected = end-start+1
    receipt = cache.with_suffix(cache.suffix+'.json')
    if cache.exists() and receipt.exists():
        b=cache.read_bytes(); old=json.loads(receipt.read_text())
        if len(b)==expected and old['start']==start and old['end']==end and sha(b)==old['sha256']:
            return b
    cache.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        base = info['links']['self'] if attempt != 1 else f'https://zenodo.org/records/19483937/files/{archive}'
        url=f'{base}?download=1&ait_new_range={start}-{end}'
        try:
            with requests.get(url, headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}, stream=True, timeout=(15,45)) as r:
                assert r.status_code==206, r.status_code
                assert r.headers.get('Content-Range')==f'bytes {start}-{end}/{info["size"]}', r.headers.get('Content-Range')
                parts=[]; n=0
                for block in r.iter_content(256*1024):
                    n+=len(block); assert n<=expected; parts.append(block)
                assert n==expected
                b=b''.join(parts)
            cache.write_bytes(b)
            write_json(receipt, {'archive':archive,'url':url,'start':start,'end':end,'bytes':len(b),'sha256':sha(b),'archive_bytes':info['size']})
            return b
        except (requests.RequestException, AssertionError) as e:
            print(json.dumps({'archive':archive,'range':[start,end],'attempt':attempt+1,'error':str(e)}),flush=True)
            last=e
    raise RuntimeError(f'Range failed: {archive} {start}-{end}') from last

def index(archive):
    assert archive in ALLOWED
    out=META/f'{archive}.members.json'
    if out.exists(): return json.loads(out.read_text(encoding='utf-8'))
    size=next(x['size'] for x in metadata()['files'] if x['key']==archive)
    tail=range_bytes(archive,max(0,size-65557),size-1,META/archive/'tail.bin')
    at=tail.rfind(b'PK\x05\x06'); assert at>=0
    end=struct.unpack('<4s4H2IH',tail[at:at+22])
    cdsize,cdoffset=end[5:7]
    if cdsize==0xffffffff or cdoffset==0xffffffff or end[4]==65535:
        loc=struct.unpack('<4sIQI',tail[at-20:at]); assert loc[0]==b'PK\x06\x07'
        z=range_bytes(archive,loc[2],loc[2]+55,META/archive/'zip64_eocd.bin')
        vals=struct.unpack('<4sQ2H2I4Q',z); assert vals[0]==b'PK\x06\x06'
        cdsize,cdoffset=vals[8:10]
    assert cdsize<20*1024*1024,cdsize
    central=range_bytes(archive,cdoffset,cdoffset+cdsize-1,META/archive/'central.bin')
    rows=[]; p=0
    while p<len(central):
        v=struct.unpack('<4s6H3I5H2I',central[p:p+46]); assert v[0]==b'PK\x01\x02'
        n,e,c=v[10:13]; name=central[p+46:p+46+n].decode('utf-8' if v[3]&2048 else 'cp437')
        ext=central[p+46+n:p+46+n+e]; uncomp,comp,offset=v[9],v[8],v[16]
        q=0
        while q<len(ext):
            tag,l=struct.unpack('<HH',ext[q:q+4]); value=ext[q+4:q+4+l];q+=4+l
            if tag==1:
                k=0
                if uncomp==0xffffffff: uncomp=struct.unpack('<Q',value[k:k+8])[0];k+=8
                if comp==0xffffffff: comp=struct.unpack('<Q',value[k:k+8])[0];k+=8
                if offset==0xffffffff: offset=struct.unpack('<Q',value[k:k+8])[0]
        rows.append({'path':name,'bytes':uncomp,'compressed':comp,'CRC':v[7],'offset':offset,'method':v[4],'flags':v[3]})
        p+=46+n+e+c
    assert p==len(central)
    write_json(out,rows)
    print(json.dumps({'index':archive,'members':len(rows),'central_bytes':cdsize}),flush=True)
    return rows

def member(archive,name,max_compressed=30*1024*1024):
    row=next(x for x in index(archive) if x['path']==name)
    assert row['compressed']<=max_compressed, row
    run=archive.split('_')[0].split('.')[0]
    out=ROOT/'source'/run/name
    receipt=ROOT/'receipts'/run/(name.replace('/','__')+'.json')
    if out.exists() and receipt.exists():
        b=out.read_bytes();assert len(b)==row['bytes'] and zlib.crc32(b)&0xffffffff==row['CRC']
        return json.loads(receipt.read_text(encoding='utf-8'))
    key=sha(name.encode())[:16]; cache=ROOT/'ranges'/archive/key
    header=range_bytes(archive,row['offset'],row['offset']+255,cache/'header.bin')
    v=struct.unpack('<IHHHHHIIIHH',header[:30]); assert v[0]==0x04034b50
    if 30+v[-2]>len(header):
        header=range_bytes(archive,row['offset'],row['offset']+30+v[-2]-1,cache/'header_long.bin')
    assert header[30:30+v[-2]].decode()==name
    offset=row['offset']+30+v[-2]+v[-1]
    pieces=[]; chunk=8*1024*1024
    for i,start in enumerate(range(offset,offset+row['compressed'],chunk)):
        pieces.append(range_bytes(archive,start,min(start+chunk,offset+row['compressed'])-1,cache/f'{i:04d}.bin'))
    packed=b''.join(pieces)
    b=zlib.decompress(packed,-15) if row['method']==8 else packed
    assert row['method'] in (0,8) and len(b)==row['bytes'] and zlib.crc32(b)&0xffffffff==row['CRC']
    out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
    result={'archive':archive,'record':RECORD,'member':name,'local_path':str(out),'data_offset':offset,**row,'sha256':sha(b)}
    write_json(receipt,result)
    print(json.dumps({'member':name,'run':run,'bytes':len(b),'sha256':sha(b)}),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('archive',choices=sorted(ALLOWED));ap.add_argument('--member');ap.add_argument('--max-compressed',type=int,default=30*1024*1024)
    a=ap.parse_args()
    if a.member: member(a.archive,a.member,a.max_compressed)
    else: index(a.archive)
