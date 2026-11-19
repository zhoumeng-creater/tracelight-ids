"""Summarize source hashes, action scope and capture candidates."""
import collections, datetime as dt, hashlib, json
from pathlib import Path
from fetch_sources import ROOT, index, write_json

def fileinfo(p):
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    actions=[];pcaps=[]
    chosen={'fox':'1642508536','santos':'1642339189'}
    for run in ('fox','santos'):
        d=json.loads((ROOT/f'{run}_action_candidates.json').read_text(encoding='utf-8'))
        for kind in ('control_actions','download_actions','reverse_shell','normal_actions'):
            for x in d[kind]:
                begin=x.get('intent',x.get('connection_receipt'))
                actions.append({'run':run,'chain_id':x.get('chain_id'),'episode_id':x['episode_id'],'kind':x['kind'],'start_utc':begin['utc'],'source_file':str(ROOT/f'{run}_action_candidates.json'),'network_admission':'see_actual_network_mapping' if (ROOT/'network_mapping'/run/'summary.json').exists() else 'pending_not_fit_ready','network_mapping_file':str(ROOT/'network_mapping'/run/'events.jsonl')})
        fs=json.loads((ROOT/'source'/run/'gather/inet-firewall/facts.json').read_text(encoding='utf-8'))['ansible_facts']
        for x in index(run+'.zip'):
            if x['path'].startswith('gather/inet-firewall/logs/suricata/log.pcap.'):
                stamp=x['path'].rsplit('.',1)[1]
                pcaps.append({'run':run,**x,'archive':run+'.zip','filename_start_utc':dt.datetime.fromtimestamp(int(stamp),dt.timezone.utc).isoformat(),'covers_attack_control_download_by_rotation_names':stamp==chosen[run],'packet_time_end_verified':False,'capture_interface':'ens3','capture_ip':fs['ens3']['ipv4']['address'],'lan_gateway':fs['ens7']['ipv4']['address'],'whole_member_downloaded':False,'selection_status':'not selected; full intranet and cloud originals obtained instead'})
    write_json(ROOT/'episode_candidate_registry.json',{'selection_not_based_on_model_scores':True,'allocation_suggestion':'santos whole attack chain F candidate -> fox whole attack chain V candidate; both subject to network admission','roles_not_finalized_here':True,'episodes':actions})
    write_json(ROOT/'firewall_pcap_candidates.json',pcaps)
    files=[ROOT/'metadata/zenodo_record_19483937.json',ROOT/'small_source_receipt.json',ROOT/'firewall_config_receipt.json',ROOT/'action_candidate_summary.json',ROOT/'capture_probe_summary.json',ROOT/'episode_candidate_registry.json',ROOT/'firewall_pcap_candidates.json']
    files += [p for p in (ROOT/'selected_pcap_download_plan.json',ROOT/'network_mapping_parameters.json',ROOT/'capture_host_inventory.json',ROOT/'all_actor_attribution.json',ROOT/'normal_actor_context_receipt.json',ROOT/'network_delivery_validation.json',ROOT/'README.md') if p.exists()]
    files += list((ROOT/'pcap_receipts').glob('*.json'))
    files += list((ROOT/'network_mapping').glob('*/*'))
    files+=list((ROOT/'metadata').glob('*.members.json'))
    files+=[ROOT/f'{run}_action_candidates.json' for run in ('fox','santos')]
    files+=list(ROOT.glob('*.py'))
    small=json.loads((ROOT/'small_source_receipt.json').read_text(encoding='utf-8'))+json.loads((ROOT/'firewall_config_receipt.json').read_text(encoding='utf-8'))
    if (ROOT/'normal_actor_context_receipt.json').exists():small+=json.loads((ROOT/'normal_actor_context_receipt.json').read_text(encoding='utf-8'))
    small=list({(x['archive'],x['member']):x for x in small}.values())
    pcaps_complete=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'pcap_receipts').glob('*.json')]
    mapping=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'network_mapping').glob('*/summary.json')]
    write_json(ROOT/'delivery_receipt.json',{'official_record':'https://zenodo.org/records/19483937','license':'CC-BY-NC-SA-4.0','small_source_members_complete':len(small),'small_source_member_compressed_bytes':sum(x['compressed'] for x in small),'small_source_member_uncompressed_bytes':sum(x['bytes'] for x in small),'each_complete_member_checks':'ZIP central size + CRC32 + local SHA256','probe_compressed_bytes':4*1024*1024,'complete_pcaps_downloaded':len(pcaps_complete),'pcap_uncompressed_bytes':sum(x['bytes'] for x in pcaps_complete),'original_pcaps':pcaps_complete,'episodes':len(actions),'network_mapping_summaries':mapping,'files':[fileinfo(p) for p in files if p.is_file()],'status':'Actual full-capture network mapping exported with all fixed event targets, original/evaluation candidate sets and connection label evidence. No training or scoring performed.'})
    print(json.dumps({'episodes':len(actions),'small_source_members':len(small),'complete_pcaps_downloaded':len(pcaps_complete)}))

if __name__=='__main__':main()
