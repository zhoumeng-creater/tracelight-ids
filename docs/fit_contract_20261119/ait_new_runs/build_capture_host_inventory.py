"""Resolve monitored-host addresses from each original facts.json interface."""
import ipaddress,json,re,hashlib
from fetch_sources import ROOT,write_json

rows=[]
for run in ('santos','fox'):
    for host in ('intranet_server','cloud_share'):
        member=f'gather/{host}/facts.json';p=ROOT/'source'/run/member
        text=p.read_text(encoding='utf-8');lines=text.splitlines();d=json.loads(text)['ansible_facts']
        anchor=next(i for i,s in enumerate(lines) if '"ansible_facts": {' in s)
        entries=[]
        for interface in d['interfaces']:
            values=d[interface]
            addresses=[]
            if values.get('ipv4'):addresses.append(values['ipv4']['address'])
            addresses.extend(x['address'] for x in values.get('ipv4_secondaries',[]))
            addresses.extend(x['address'] for x in values.get('ipv6',[]))
            start=next(i for i in range(anchor+1,len(lines)) if re.match(r'^    "'+re.escape(interface)+r'": \{',lines[i]))
            stop=next(i for i in range(start+1,len(lines)) if re.match(r'^    \}',lines[i]))
            for address in addresses:
                found=[i+1 for i in range(start,stop+1) if re.search(r'"address":\s*"'+re.escape(address)+r'"',lines[i])]
                assert len(found)==1,(run,host,interface,address,found)
                ip=ipaddress.ip_address(address)
                entries.append({'address':str(ip),'interface':interface,'ip_version':ip.version,'is_loopback':ip.is_loopback,'is_link_local':ip.is_link_local,'source_member':member,'source_path':str(p),'source_line':found[0],'json_pointer':f'/ansible_facts/{interface}'})
        declared=set(d['all_ipv4_addresses']+d['all_ipv6_addresses'])
        assert declared.issubset({x['address'] for x in entries})
        rows.append({'run':run,'capture_host':host,'host_id':f'ait:{run}:{host}','all_local_addresses':[x['address'] for x in entries],'nonloopback_local_addresses':[x['address'] for x in entries if not x['is_loopback']],'interfaces':entries,'facts_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'host_view_rule':'Include a connection only if either endpoint is one of this capture host\'s original local addresses. Select peer from the opposite endpoint; do not infer host from connection initiator.','nonlocal_capture_connections':'retain in upstream full capture table; no monitored-host view unless independently identified','packet_address_coverage_status':'to be counted from complete normalized capture'})
write_json(ROOT/'capture_host_inventory.json',{'hosts':rows,'address_basis':'original ansible_facts per-interface IPv4/IPv6, cross-checked against all_ipv4_addresses/all_ipv6_addresses'})
print(json.dumps(rows,indent=2))
