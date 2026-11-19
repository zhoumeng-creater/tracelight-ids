"""Collect only fixed russellmitchell service logs and four normal actor logs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import struct
import zlib
import requests

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
INDEX=WORK/'独立事件与风险迁移_20261115/候选元数据/russellmitchell_no-pcaps.zip.members.json'
REPO=WORK.parent/'代码/tracelight-ids'
OLD=WORK/'独立来源与算法主线_20261118/ait_small_sources'
RAW=REPO/'data/raw/ait-lds/source-audit-20261115'
URL='https://zenodo.org/records/19483937/files/russellmitchell_no-pcaps.zip'
TARGETS=['dataset.yaml','processing/config/servers.yaml','labels/intranet_server/logs/apache2/intranet.smith.russellmitchell.com-access.log.2']
for host,stem in [('intranet_server','intranet.smith.russellmitchell.com-access'),('cloud_share','other_vhosts_access'),('webserver','cloud.smith.russellmitchell.com-access')]:
    TARGETS.extend(f'gather/{host}/logs/apache2/{stem}.log.{rotation}' for rotation in range(1,5))
TARGETS.extend(f'gather/internal_employee_{i}/logs/sm.log' for i in range(4))

def get(entry):
    member=entry['path'];dest=ROOT/'source'/member;dest.parent.mkdir(parents=True,exist_ok=True)
    receipt=dest.with_name(dest.name+'.receipt.json')
    if dest.exists() and receipt.exists():
        raw=dest.read_bytes();assert len(raw)==entry['bytes'] and zlib.crc32(raw)==entry['CRC']
        return json.loads(receipt.read_text(encoding='utf-8'))
    local=next((p for p in [OLD/member,RAW/member] if p.exists()),None)
    if local:
        raw=local.read_bytes();provenance={'reused_local_source':str(local)}
    else:
        start=entry['offset'];end=start+entry['compressed']+511
        url=URL+f'?download=1&normal_calibration_offset={start}'
        response=requests.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(15,60))
        assert response.status_code==206,(member,response.status_code)
        assert response.headers['Content-Range']==f'bytes {start}-{end}/522084364'
        blob=response.content;fields=struct.unpack('<IHHHHHIIIHH',blob[:30]);assert fields[0]==0x04034b50
        assert blob[30:30+fields[-2]].decode()==member
        offset=30+fields[-2]+fields[-1];compressed=blob[offset:offset+entry['compressed']]
        raw=zlib.decompress(compressed,-15) if fields[3]==8 else compressed
        provenance={'url':url,'content_range':response.headers['Content-Range']}
    assert len(raw)==entry['bytes'] and zlib.crc32(raw)==entry['CRC'],member
    dest.write_bytes(raw)
    result={**entry,**provenance,'sha256':hashlib.sha256(raw).hexdigest(),'local_path':str(dest)}
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    entries={r['path']:r for r in json.loads(INDEX.read_text(encoding='utf-8'))}
    with ThreadPoolExecutor(max_workers=3) as pool: receipts=list(pool.map(get,[entries[name] for name in TARGETS]))
    result={'source_record':'https://zenodo.org/records/19483937','archive':'russellmitchell_no-pcaps.zip','index_sha256':hashlib.sha256(INDEX.read_bytes()).hexdigest(),'members':receipts}
    (ROOT/'source_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'members':len(receipts),'new_download_members':sum('url' in r for r in receipts),'new_compressed_bytes':sum(r['compressed'] for r in receipts if 'url' in r)},ensure_ascii=False))
