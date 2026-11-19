"""Fetch only fixed, small official configuration members needed for route proof."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,struct,zlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
INDEX=BASE/'独立事件与风险迁移_20261115/候选元数据/russellmitchell_no-pcaps.zip.members.json'
TARGETS=['gather/inet-firewall/facts.json','gather/inet-firewall/configs/shorewall/interfaces','gather/inet-firewall/configs/shorewall/snat','gather/inet-firewall/configs/shorewall/params','gather/inet-firewall/configs/shorewall/rules','gather/inet-firewall/configs/suricata/suricata.yaml']
URL='https://zenodo.org/records/19483937/files/russellmitchell_no-pcaps.zip'

def fetch(entry):
    dest=ROOT/'source/capture_config'/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    receipt=dest.with_name(dest.name+'.receipt.json')
    if dest.exists() and receipt.exists():
        raw=dest.read_bytes();assert len(raw)==entry['bytes'] and zlib.crc32(raw)==entry['CRC'];return json.loads(receipt.read_text(encoding='utf-8'))
    start=entry['offset'];end=start+entry['compressed']+511
    url=URL+f'?download=1&ait_recovery_config_offset={start}'
    response=requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,45))
    assert response.status_code==206,(entry['path'],response.status_code)
    assert response.headers['Content-Range']==f'bytes {start}-{end}/522084364'
    blob=response.content;fields=struct.unpack('<IHHHHHIIIHH',blob[:30]);assert fields[0]==0x04034b50 and fields[3]==8
    assert blob[30:30+fields[-2]].decode()==entry['path']
    offset=30+fields[-2]+fields[-1];raw=zlib.decompress(blob[offset:offset+entry['compressed']],-15)
    assert len(raw)==entry['bytes'] and zlib.crc32(raw)==entry['CRC']
    dest.write_bytes(raw)
    result={**entry,'url':url,'content_range':response.headers['Content-Range'],'sha256':hashlib.sha256(raw).hexdigest(),'local_path':str(dest)}
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return result

if __name__=='__main__':
    index=json.loads(INDEX.read_text(encoding='utf-8'));selected=[r for r in index if r['path'] in TARGETS];assert len(selected)==len(TARGETS)
    with ThreadPoolExecutor(max_workers=3) as pool: results=list(pool.map(fetch,selected))
    (ROOT/'source/capture_config/receipt.json').write_text(json.dumps({'index_sha256':hashlib.sha256(INDEX.read_bytes()).hexdigest(),'members':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'members':len(results),'compressed_bytes':sum(r['compressed'] for r in results)},ensure_ascii=False),flush=True)
