"""Preserve exact actor/service anchors for later full PCAP association."""
import collections, datetime as dt, json
from urllib.parse import urlsplit
import yaml
from fetch_sources import ROOT,write_json
from fetch_selected_pcaps import selection

def main():
    sources={(x['run'],x['host']):x for x in selection()};out=[]
    for run in ('santos','fox'):
        base=ROOT/'source'/run
        servers=yaml.safe_load((base/'processing/config/servers.yaml').read_text(encoding='utf-8'))
        d=json.loads((ROOT/f'{run}_action_candidates.json').read_text(encoding='utf-8'))
        for key in ('control_actions','normal_actions'):
            for action in d[key]:
                host='cloud_share' if action['kind']=='normal_owncloud_file_download' else 'intranet_server'
                begin=action['intent'];end=action.get('completion',action.get('response'))
                matches=action['apache_matches'];assert len(matches)==1
                service=matches[0]
                if host=='cloud_share':
                    server_port=int(service['vhost'].rsplit(':',1)[1]);port_basis='Apache vhost'
                else:
                    scheme=urlsplit(action['url']).scheme;assert scheme=='https'
                    server_port=443;port_basis='actor HTTPS URL (verify actual connection port)'
                out.append({'episode_id':action['episode_id'],'run':run,'chain_id':action.get('chain_id'),'kind':action['kind'],'pcap_member':sources[run,host]['path'],'capture_host':host,'ip_protocol':6,'observed_service_endpoint':{'client_ip':service['client'],'client_port':None,'server_ip':servers[host]['default_ipv4_address'],'server_port':server_port,'port_basis':port_basis},'actor_endpoint':{'role':action.get('host_role','attacker_0'),'ip':action.get('host_ip',servers['attacker_0']['default_ipv4_address'])},'actor_interval':{'start':begin['timestamp'],'end':end['timestamp'],'start_utc':begin['utc'],'end_utc':end['utc']},'service_request_time_interval':{'start_inclusive':service['timestamp'],'end_exclusive':service['timestamp']+1,'basis':'Apache timestamp resolution one second; retain original UTC'},'capture_candidate_overlap_window':{'start':min(begin['timestamp'],service['timestamp']),'end':max(end['timestamp'],service['timestamp']+1)},'http_request_exact':service['request'],'http_response_status':service['status'],'apache_logged_bytes':service['logged_bytes'],'file_object':action.get('file'),'evidence':{'actor_intent':begin,'actor_completion_or_response':end,'apache':service},'connection_matching_rule':'Keep every same observed client/server/service connection overlapping the request/action interval, including already-open persistent connections. Exact HTTP URI or other connection evidence may refine later. Never choose nearest flow or copy one positive label to all candidates.','raw_field_source':'native capture; actor IP must not replace proxy/VPN IP; URL, labels, IP identity and filename remain evidence-side fields','model_input_admission':'pending actual network extraction; no model run'})
        for action in d['download_actions']:
            url=urlsplit(action['url']);begin=action['intent'];end=action['completion'];known_ip=url.hostname if action['object']=='password_list' else None
            out.append({'episode_id':action['episode_id'],'run':run,'chain_id':action['chain_id'],'kind':action['kind'],'pcap_member':sources[run,'intranet_server']['path'],'capture_host':'intranet_server','ip_protocol':6,'observed_service_endpoint':{'client_ip':servers['intranet_server']['default_ipv4_address'],'client_port':None,'server_ip':known_ip,'server_port':url.port or (443 if url.scheme=='https' else 80)},'actor_interval':{'start':begin['timestamp'],'end':end['timestamp'],'start_utc':begin['utc'],'end_utc':end['utc']},'requested_url':action['url'],'initial_server_name':url.hostname,'destination_resolution':'Use contemporaneous PCAP DNS/TLS/HTTP and redirect evidence; never current DNS or assumed codeload IP','connection_matching_rule':'Collect outbound request/redirect/final-transfer chain with actual network evidence; do not conflate incoming HTTPS command response with downloaded file. Preserve all candidate connections until evidence resolves them.','evidence':{'intent':begin,'completion':end},'model_input_admission':'pending actual network extraction; no model run'})
        for action in d['reverse_shell']:
            out.append({'episode_id':action['episode_id'],'run':run,'chain_id':action['chain_id'],'kind':action['kind'],'pcap_member':sources[run,'intranet_server']['path'],'capture_host':'intranet_server','ip_protocol':6,'victim_inside_ip':servers['intranet_server']['default_ipv4_address'],'attacker_listen_socket':action['listen_socket'],'remote_socket_seen_after_nat':action['remote_socket'],'connection_receipt':action['connection_receipt'],'connection_matching_rule':'Use actual SYN and payload/connection evidence before the connected receipt; preserve NAT viewpoint, do not assume port preservation. This receipt alone gives no lower time bound.','model_input_admission':'pending actual network extraction; no model run'})
    assert len(out)==len({x['episode_id'] for x in out})==254
    summary={'events':len(out),'per_run_kind':dict(collections.Counter(f'{x["run"]}:{x["kind"]}' for x in out)),'timestamp_policy':'Original numeric epoch and UTC retained; original seconds quantization represented explicitly, no clock offset guessed','time_available_at':'unknown; an action receipt does not establish detector availability','derived_pcap_policy':'Any orig_len-only repair must have its own verified transform receipt. Never mutate the source PCAP or change caplen/payload here.','events_parameters':out}
    write_json(ROOT/'network_mapping_parameters.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='events_parameters'},indent=2))

if __name__=='__main__':main()
