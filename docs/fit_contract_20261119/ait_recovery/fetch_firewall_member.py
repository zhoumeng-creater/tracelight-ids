"""Fetch exactly one evidence-selected firewall rotation, using the proven range reader."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[1]
HELPER=BASE/'统一事件与方法落实_20261119/ait/fetch_cloud_member.py'
spec=importlib.util.spec_from_file_location('existing_ait_range_fetcher',HELPER)
fetcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)
fetcher.ROOT=ROOT
fetcher.MEMBER='gather/inet-firewall/logs/suricata/log.pcap.1642957916'
fetcher.OFFSET=1037404189
fetcher.COMPRESSED=694128142
fetcher.SIZE=789200482
fetcher.CRC=2278223961
fetcher.CACHE=ROOT/'source/firewall_member_chunks'
fetcher.DEST=ROOT/'source/firewall-log.pcap.1642957916'

if __name__=='__main__':
    index=BASE/'独立事件与风险迁移_20261115/候选元数据/russellmitchell.zip.members.json'
    entries=json.loads(index.read_text(encoding='utf-8'))
    pcaps=[r for r in entries if r['path'].startswith('gather/') and ('.pcap.' in r['path'] or r['path'].endswith('.pcap'))]
    selected=next(r for r in pcaps if r['path']==fetcher.MEMBER)
    selection={'source_record':'https://zenodo.org/records/19483937','member_index':str(index),'member_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'pcap_members':pcaps,'selected_member':selected,'selection_basis':['Intranet gateway 10.143.0.1 equals firewall ens7.','Firewall ens3 192.168.230.4 is the Suricata interface; official Shorewall MASQUERADE sends 10.143.0.0/17 out ens3.','The three fixed download connections at 1642996762.36-1642996786.20 occur after latest firewall rotation start 1642957916.','NAT association must be established by sequence numbers, packet content and time, with all candidate matches retained.'],'target_packet_ids':[3574,3575,3578],'helper_sha256':hashlib.sha256(HELPER.read_bytes()).hexdigest(),'model_calls':0,'packet_modifications':0}
    (ROOT/'source_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fetcher.main()
    # The shared helper's receipt basename is historical; retain it and expose a descriptive copy.
    historical=ROOT/'source/cloud-source-receipt.json'
    if historical.exists():
        (ROOT/'source/firewall-source-receipt.json').write_bytes(historical.read_bytes())
