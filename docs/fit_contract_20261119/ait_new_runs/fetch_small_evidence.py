"""Select and fetch only action, normal-business, service and source metadata."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from fetch_sources import ROOT, index, member, write_json

def selection(run):
    names={
        'dataset.yaml','processing/config/servers.yaml',
        'gather/attacker_0/logs/attacks.log',
        'gather/attacker_0/logs/ait.aecid.attacker.wpdiscuz/sm.log',
        'gather/attacker_0/configs/config.yml',
        'gather/internal_employee_0/logs/sm.log',
        'gather/internal_employee_1/logs/sm.log',
        'gather/internal_employee_0/configs/config.yml',
        'gather/internal_employee_1/configs/config.yml',
        'gather/intranet_server/facts.json','gather/cloud_share/facts.json',
    }
    rows=index(f'{run}_no-pcaps.zip')
    for x in rows:
        n=x['path']
        if any(f'{prefix}/{host}/logs/apache2/' in n for prefix in ('gather','labels') for host in ('intranet_server','cloud_share')) and 'access' in n and x['bytes']:
            names.add(n)
        if n.startswith(('gather/intranet_server/configs/','gather/cloud_share/configs/')) and n.endswith('suricata.yaml'):
            names.add(n)
    actual={x['path']:x for x in rows}
    return [actual[n] for n in sorted(names) if n in actual]

def main():
    jobs=[(run,x) for run in ('fox','santos') for x in selection(run)]
    write_json(ROOT/'small_source_selection.json',{'compressed_bytes':sum(x['compressed'] for _,x in jobs),'members':[{'run':run,**x} for run,x in jobs]})
    receipts=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks={pool.submit(member,f'{run}_no-pcaps.zip',x['path']):(run,x['path']) for run,x in jobs}
        for f in as_completed(tasks): receipts.append(f.result())
    write_json(ROOT/'small_source_receipt.json',receipts)
    print({'completed_members':len(receipts),'uncompressed_bytes':sum(x['bytes'] for x in receipts)})

if __name__=='__main__':main()
