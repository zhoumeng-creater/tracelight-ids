"""Join existing activity evidence to packet geometry; no new labels or decisions."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    rr={r['flow_id']:r for r in rows(args.geometry/'connection_geometry.jsonl.gz')}
    groups={}
    for source in sorted({r['source'] for r in rr.values()}):
        for kind in ('control','normal','download'):
            selected=[r for r in rr.values() if r['source']==source and r['target_kind']==kind]
            if not selected:continue
            groups[source+'/'+kind]={
                'n':len(selected),
                'eligible':sum(r['geometry_eligible'] for r in selected),
                'raw8_packet_counts_equal_extracted':sum(r['packet_counts']==[r['raw8']['orig_pkts'],r['raw8']['resp_pkts']] for r in selected),
                'ack_map_reconstructed_by_bursts':sum(r['ack_map_exactly_reconstructed_from_direction_bursts'] for r in selected),
                'ack_segments':dict(sorted(Counter(r['ack_segment_count'] for r in selected).items())),
                'ack_uncovered_mass_sum':sum(1-r['ack_covered_pair_fraction'] for r in selected if r['ack_covered_pair_fraction'] is not None),
            }
    normal_downloads=[]
    for e in rows(repo/'docs/coverage_basis_20261119/batch/events.jsonl.gz'):
        if e.get('role') in ('C','C_background') or e.get('kind')!='normal_owncloud_file_download':continue
        if not e['event_id'].startswith(('ait:santos:','ait:fox:')):continue
        ids=e['candidate_flow_ids']
        normal_downloads.append({'event_id':e['event_id'],'association_mode':e['association_mode'],
            'candidate_set_complete':e['candidate_set_complete'],
            'candidate_count':len(ids),'geometry_present':sum(x in rr for x in ids),
            'whole_connection_normal_candidates':sum(x in rr and rr[x]['label']==0 for x in ids),
            'unknown_candidates':sum(x in rr and rr[x]['label'] is None for x in ids),
            'geometry_eligible_candidates':sum(x in rr and rr[x]['geometry_eligible'] for x in ids),
            'candidate_geometry':[{'flow_id':x,'target_kind':rr[x]['target_kind'],'ack_segments':rr[x]['ack_segment_count'],
                'K':rr[x]['ack_covered_pair_fraction'],'eligible':rr[x]['geometry_eligible']} for x in ids if x in rr]})
    full=args.work/'覆盖与历史依据落实_20261119/attack_coverage/full_capture'
    uid={r['uid']:r['flow_id'] for r in rows(full/'connections.jsonl.gz') if r.get('involves_monitored_host')}
    successful_http=defaultdict(list)
    for h in rows(full/'zeek/http.log'):
        flow=uid.get(h['uid'])
        if flow not in rr or rr[flow]['label']!=0:continue
        if h.get('method')=='GET' and h.get('status_code')==200 and h.get('response_body_len',0)>0:
            successful_http[flow].append({k:h.get(k) for k in ('uid','trans_depth','response_body_len','resp_fuids','resp_mime_types')})
    def two_stage(r):
        a,b=r['unique_payload_bytes']
        return a>0 and b>0 and r['segments']==[[[0,a,0]],[[0,b,a]]]
    examples=[]
    for flow,events in successful_http.items():
        r=rr[flow]
        if r['geometry_eligible'] and two_stage(r):
            examples.append({'flow_id':flow,'unique_payload_bytes':r['unique_payload_bytes'],
                             'raw8':r['raw8'],'http_transactions':events})
    downloads=[r for r in rr.values() if r['target_kind']=='download']
    output={'groups':groups,'normal_owncloud_download_actions':normal_downloads,
        'normal_http_successful_body_connections':len(successful_http),
        'normal_http_eligible_two_stage_connections':len(examples),
        'normal_http_two_stage_examples':examples,
        'confirmed_attack_downloads':[{'flow_id':r['flow_id'],'geometry_eligible':r['geometry_eligible'],
            'two_stage_shape':two_stage(r),'unique_payload_bytes':r['unique_payload_bytes']} for r in downloads],
        'interpretation':'Two-stage normalized ACK topology is identical; byte amounts, timing, business and source need not be identical. This is a real counterexample to topology-alone intent, not a classifier result.',
        'model_calls':0}
    (args.geometry/'relation_evidence.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'normal_download_actions':len(normal_downloads),'normal_http_successful':len(successful_http),
                      'normal_http_two_stage':len(examples),'attack_downloads':len(downloads)}))


if __name__=='__main__':main()
