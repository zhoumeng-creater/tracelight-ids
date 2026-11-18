"""Materialize connector-returned file references; never save signed URLs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import requests

BASE = Path(__file__).resolve().parent
def fetch(s):
    path = BASE / 'sources' / s['local_name']
    if not path.exists() or path.stat().st_size != int(s['size']):
        part = path.with_suffix(path.suffix + '.partial')
        with requests.get(s.pop('download_url'), stream=True, timeout=(30, 120)) as r:
            r.raise_for_status()
            n = 0
            with part.open('wb') as out:
                for chunk in r.iter_content(1024 * 1024):
                    n += len(chunk)
                    if n > int(s['size']):
                        raise ValueError('Oversize response')
                    out.write(chunk)
        if n != int(s['size']):
            raise ValueError('Incomplete response')
        part.replace(path)
    s.pop('download_url', None)
    s['path'] = str(path)
    s['sha256'] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    print('READY ' + s['local_name'], flush=True)
    return s

if __name__ == '__main__':
    specs = json.loads(sys.stdin.read())
    (BASE / 'sources').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(fetch, specs))
    receipt_path = BASE / 'meterpreter_source_receipt.json'
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text(encoding='utf-8'))['sources']
        result = list({s['id']:s for s in old + result}.values())
    (BASE / 'meterpreter_source_receipt.json').write_text(json.dumps({
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'qualification-only; no model or feature distribution selection',
        'initial_network_hour_utc':['2019-09-25T14:00:00Z','2019-09-25T15:00:00Z'],
        'additional_conn_reason':'Exact author PID/host-linked control UID may be exported in a later rotation; check adjacent hour and final available day rotation, without changing the identity query.',
        'group_id':'optc:meterpreter:20190925_complete_branch',
        'selected_action':'Sysclient0051 malicious Notepad++ update at 10:29:42 -04:00',
        'sources':result,'downloaded_bytes':sum(int(s['size']) for s in result)
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
