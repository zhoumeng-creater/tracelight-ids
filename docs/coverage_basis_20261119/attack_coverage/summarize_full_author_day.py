"""Count already-downloaded author annotations; no traffic scoring or retrieval."""
import collections,csv,datetime,json
from pathlib import Path
from assemble_unraveled import ROOT,SRC,flow_csv,dump,sha

def utc(ms):return datetime.datetime.fromtimestamp(int(ms)/1000,datetime.timezone.utc).isoformat()
def run():
    rows,bad,validation=flow_csv(SRC/'flow_20210621_net1013x.csv')
    target=[x for x in rows if '10.1.3.8' in (x['src_ip'],x['dst_ip'])]
    def c2(x):return x['Signature']=='APT' and x['Activity'] in ('Exfiltration over C2 channel','Encrypted Channel: Symmetric Cryptography')
    def normal(x):return x['Activity']=='Normal' and x['Signature']=='None'
    subsets={'all_target_host':target,'target_tcp_resp_port443':[x for x in target if x['protocol']=='6' and x['dst_port']=='443'],'target_tls_application':[x for x in target if x['application_name']=='TLS' or x['application_name'].startswith('TLS.')], 'target_c2_server':[x for x in target if '10.8.10.84' in (x['src_ip'],x['dst_ip'])]}
    stats={}
    for name,xs in subsets.items():
        stats[name]={'valid_author_rows':len(xs),'explicit_normal':sum(normal(x) for x in xs),'explicit_C2':sum(c2(x) for x in xs),'other_explicit_attack':sum(x['Signature'] in ('APT','AA','SH') and not c2(x) for x in xs),'unresolved_or_other':sum(not normal(x) and not c2(x) and x['Signature'] not in ('APT','AA','SH') for x in xs),'activity_signature_counts':[{'activity':k[0],'stage':k[1],'signature':k[2],'count':v} for k,v in collections.Counter((x['Activity'],x['Stage'],x['Signature']) for x in xs).items()]}
    ref=json.loads((SRC/'pcap_slices/normal_mail/receipt.json').read_text(encoding='utf-8'))
    result={'host':'10.1.3.8','author_day':'Week5_Day1_06212021','source_flow_table':validation,'capture_folder_not_UTC_calendar_day':True,'all_valid_rows_min_first_utc':utc(min(int(x['bidirectional_first_seen_ms']) for x in rows)),'all_valid_rows_max_last_utc':utc(max(int(x['bidirectional_last_seen_ms']) for x in rows)),'host_rows_min_first_utc':utc(min(int(x['bidirectional_first_seen_ms']) for x in target)),'host_rows_max_last_utc':utc(max(int(x['bidirectional_last_seen_ms']) for x in target)),'subsets':stats,'pcap_source_url':ref['source_url'],'pcap_full_bytes':ref['source_object_bytes'],'pcap_GiB':ref['source_object_bytes']/1024**3,'anonymous_range_access_verified':True,'same_object_etag':ref['source_etag'],'full_pcap_downloaded':False,'evidence_against_whole_victim_ip_attack_label':'Same victim IP has both author-explicit Normal and per-flow APT C2 annotations; table counts and source rows retained. This disproves whole-victim-IP=APT labeling, not all possible endpoint/time rules.','ground_truth_generation_code_status':'not obtained; cannot independently reproduce author annotation procedure','normal_policy':'Accept author-explicit Normal as source-graded normal labels when exact flow linkage is complete; no requirement for a process log for every normal row. Unlabelled/Background/malformed rows are not Normal.','flow_granularity_limit':'NFStream rows may split longer physical connections. Full extraction must keep all author candidates and refuse conflicting/mixed whole-connection labels; do not directly compare their raw fields as Zeek raw8.','next_bounded_action':'If parent authorizes, retrieve this one 3518820352-byte net1013x PCAP with validated ranges and extract full capture once; associate all source-grade explicit Normal/C2 by tuple plus intervals, retaining other/unknown records. No other day/source needed for that step.'}
    dump(ROOT/'full_author_day_summary.json',result)
    print(json.dumps({k:result[k] for k in ('subsets','pcap_full_bytes','pcap_GiB','all_valid_rows_min_first_utc','all_valid_rows_max_last_utc')},ensure_ascii=False))
if __name__=='__main__':run()
