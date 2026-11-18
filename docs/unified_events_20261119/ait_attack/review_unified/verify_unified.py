"""Independent source-to-unified integrity check; no mutation of unified inputs."""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOW = HERE.parents[1]
PREV = NOW.parent / '独立来源与算法主线_20261118'
UNIFIED = NOW / 'unified'
RAW8 = ('ip_protocol','orig_port','resp_port','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes')


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with (gzip.open(p,'rt',encoding='utf-8-sig') if p.suffix=='.gz' else p.open(encoding='utf-8-sig')) as f:
        return [json.loads(x) for x in f if x.strip()]
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def numeric(v): return None if v in (None,'','-') else float(v)
def timestamp(v):
    if v in (None,'','-'): return None
    try: return float(v)
    except ValueError: return datetime.fromisoformat(v.replace('Z','+00:00')).timestamp()


def main():
    files = [p for p in UNIFIED.iterdir() if p.is_file()]
    before = {p.name:sha(p) for p in files}
    event_rows=rows(UNIFIED/'events.jsonl.gz'); flow_rows=rows(UNIFIED/'observations.jsonl.gz')
    links=rows(UNIFIED/'event_flow_links.jsonl.gz'); assignments=rows(UNIFIED/'split_assignments.jsonl.gz')
    events={r['event_id']:r for r in event_rows}; flows={r['flow_id']:r for r in flow_rows}
    checks={}
    def check(name, condition, detail=None): checks[name]={'pass':bool(condition),'detail':detail}
    check('unique_event_ids',len(events)==len(event_rows))
    check('unique_observation_ids',len(flows)==len(flow_rows))
    check('all_links_resolve',all(x['event_id'] in events and x['flow_id'] in flows for x in links))
    check('no_cross_group_links',all(events[x['event_id']]['group_id']==flows[x['flow_id']]['group_id'] for x in links))
    expected_events={}; expected_flow_labels={}; source_raw={}; expected_links=Counter(); source_counts={}
    def expect_event(eid,label,unit):
        assert eid not in expected_events
        expected_events[eid]=(label,unit)
    def expect_link(eid,fid,packet=None): expected_links[(eid,fid,packet)] += 1
    cloud=load(NOW/'ait/normal_event_connections.json')
    for r in rows(NOW/'ait/referenced_connections.jsonl'):
        source_raw[r['flow_id']]=r
        expected_flow_labels[r['flow_id']]='unknown'
    for e in cloud['events']:
        expect_event(e['event_id'],'normal' if e['business_label']=='normal_file_download' else 'unknown','business_action')
        for c in e['packet_connection_candidates']:
            for z in c['zeek_links']: expect_link(e['event_id'],z['flow_id'],c['packet_connection_id'])
    source_counts['cloud_events']=len(cloud['events'])
    attack_seen={}
    for mode in ('strict','ignore_checksums'):
        stage=next((NOW/f'ait_attack/contract/analysis-{mode}').glob('attempt-*'))
        pairs=csvrows(stage/'action_connection_pairs.csv')
        wanted={r['uid'] for r in pairs if r['uid']}
        for r in csvrows(stage/'raw8.csv'):
            if r['uid'] in wanted:
                fid=f"ait-intranet:{mode}:{r['uid']}";source_raw[fid]=r;expected_flow_labels[fid]='unknown'
        for r in pairs:
            eid='ait:intranet:'+r['action_id']
            expected=('normal' if r['kind']=='normal_business' else 'malicious','business_action')
            if eid not in attack_seen: attack_seen[eid]=expected
            else: assert attack_seen[eid]==expected
            if r['uid']:expect_link(eid,f"ait-intranet:{mode}:{r['uid']}")
    for eid,(label,unit) in attack_seen.items():expect_event(eid,label,unit)
    source_counts['intranet_actions']=len(attack_seen)
    normal_events=rows(NOW/'optc/normal_flow_events.jsonl.gz')
    normal_observations=rows(NOW/'optc/normal_flow_observations.jsonl.gz')
    normal_objects=rows(NOW/'optc/normal_http_file_objects.jsonl.gz')
    for r in normal_observations:source_raw[r['flow_id']]=r;expected_flow_labels[r['flow_id']]='unknown'
    for r in normal_events:
        eid=r['event_id'];fid='optc:20190919:'+r['uid'];expect_event(eid,'normal','network_session');expect_link(eid,fid);expected_flow_labels[fid]='normal'
    for r in normal_objects:expect_event(r['event_id'],'normal','http_file_object');expect_link(r['event_id'],'optc:20190919:'+r['uid'])
    source_counts.update(optc_normal_events=len(normal_events),optc_normal_observations=len(normal_observations),optc_normal_objects=len(normal_objects))
    missing_publisher_durations=[]
    def publisher(r,date):
        fid='optc:'+date+':'+r['uid']
        duration=numeric(r['duration'])
        if duration is None: missing_publisher_durations.append(fid)
        raw=dict(zip(RAW8,[{'tcp':6,'udp':17,'icmp':1}.get(r['proto']),r['id.orig_p'],r['id.resp_p'],duration,r['orig_pkts'],r['resp_pkts'],r['orig_ip_bytes'],r['resp_ip_bytes']]))
        raw.update(connection_start=r['ts'],zeek_reported_end=None if duration is None else float(r['ts'])+duration)
        source_raw[fid]=raw;expected_flow_labels[fid]='unknown'
        return fid
    emp=load(PREV/'evidence/optc/optc_event_slice.json');session={r['uid']:r for r in emp['joined_event']['sessions']}
    for r in emp['candidates']['conn_uid_matches_all']:
        fid=publisher(r,'20190924');eid='optc:empire:download:'+r['uid'];exact=session[r['uid']]['request_matches_reported_second_under_minus04']
        expect_event(eid,'malicious' if exact else 'attack_associated_transfer','http_file_transfer');expect_link(eid,fid)
        if exact:expected_flow_labels[fid]='malicious_download'
    ctrl=load(PREV/'evidence/optc/optc_control_channel_slice.json')
    confirmed=set(ctrl['confirmed_control_uids'])
    check('all_empire_controls_are_source_confirmed',all(s['conn_record']['uid'] in confirmed and not s['unconfirmed_reasons'] for s in ctrl['all_conn_endpoint_candidates_and_joins']))
    for s in ctrl['all_conn_endpoint_candidates_and_joins']:
        r=s['conn_record'];fid=publisher(r,'20190924');eid='optc:empire:control:'+r['uid'];expect_event(eid,'malicious','network_session');expect_link(eid,fid);expected_flow_labels[fid]='known_control_transport'
    meter=load(NOW/'split_method/meterpreter_qualification.json')
    for s in meter['download_events']:
        exact=s['matches_author_host_payload'];expect_event(s['event_id'],'malicious' if exact else 'attack_associated_transfer','http_file_transfer')
        for r in s['conn']:
            fid=publisher(r,'20190925');expect_link(s['event_id'],fid)
            if exact:expected_flow_labels[fid]='malicious_download'
    check('no_meterpreter_control_candidates_dropped',all(len(s['conn'])==1 for s in meter['matched_control_sessions']))
    for s in meter['matched_control_sessions']:
        fid=publisher(s['conn'][0],'20190925');expect_event(s['event_id'],'malicious','network_session');expect_link(s['event_id'],fid);expected_flow_labels[fid]='known_control_transport'
    pe=load(PREV/'evidence/optc/benign_file_business.json')
    for r in pe['candidate_file_rows']:expect_event('optc:20190919:unknown_PE:'+r['fuid'],'unknown','http_file_object')
    check('exact_source_event_set',set(expected_events)==set(events),{'expected':len(expected_events),'actual':len(events)})
    check('exact_source_observation_set',set(source_raw)==set(flows),{'expected':len(source_raw),'actual':len(flows)})
    check('no_label_or_unit_promotion',all((events[k]['label'],events[k]['unit'])==v for k,v in expected_events.items()))
    check('exact_source_flow_labels',all(flows[k]['flow_label']==v for k,v in expected_flow_labels.items()))
    actual_links=Counter((r['event_id'],r['flow_id'],r.get('packet_connection_id')) for r in links)
    check('all_source_links_and_multiplicity_preserved',actual_links==expected_links,{'expected':sum(expected_links.values()),'actual':len(links),'missing':sum((expected_links-actual_links).values()),'extra':sum((actual_links-expected_links).values())})
    link_counts=Counter(r['event_id'] for r in links)
    check('linked_flow_count_preserves_link_rows',all(e['linked_flow_count']==link_counts[k] for k,e in events.items()))
    roles=defaultdict(set)
    for r in event_rows+flow_rows:roles[r['group_id']].add(r['split_role'])
    check('whole_group_single_role',all(len(v)==1 for v in roles.values()),{k:sorted(v) for k,v in roles.items()})
    expected_assignments=Counter([('event',r['event_id'],r['group_id'],r['split_role']) for r in event_rows]+[('observation',r['flow_id'],r['group_id'],r['split_role']) for r in flow_rows])
    check('all_split_assignments_exact',Counter((r['unit'],r['id'],r['group_id'],r['split_role']) for r in assignments)==expected_assignments)
    cloud_sem=load(NOW/'ait/checksum_semantics.json');attack_sem=load(NOW/'ait_attack/checksum_semantics.json')
    canonical_cloud={fid for c in cloud_sem['packet_candidates'] for fid in c['frozen_checksum_mapping']['request_second_flow_ids']['ignore_checksums']}
    all_cleared_attack_uids={'ait-intranet:ignore_checksums:'+r['uid'] for c in attack_sem['packet_candidates'] if not c['model_row_blocked_on_capture_semantics'] for r in c['zeek_rows']['ignore_checksums']}
    canonical_attack=all_cleared_attack_uids & set(source_raw)
    blocked_attack={'ait-intranet:ignore_checksums:'+r['uid'] for c in attack_sem['packet_candidates'] if c['model_row_blocked_on_capture_semantics'] for r in c['zeek_rows']['ignore_checksums']}
    check('canonical_70_exact_audited_Uids',{k for k,v in flows.items() if v['canonical_status']=='capture_qualified_C'}==canonical_cloud|canonical_attack,{'cloud':len(canonical_cloud),'attack':len(canonical_attack)})
    check('four_length_blocked_exact',{k for k,v in flows.items() if v['canonical_status']=='blocked_invalid_captured_IP_lengths'}==blocked_attack,sorted(blocked_attack))
    check('canonical_no_strict_no_label_propagation',all(k in flows and flows[k].get('variant')=='ignore_checksums' and flows[k]['flow_label']=='unknown' for k in canonical_cloud|canonical_attack))
    raw_mismatch=[];time_mismatch=[];missing_flags=[]
    for fid,r in source_raw.items():
        f=flows[fid]
        if f['raw8']!={k:numeric(r.get(k)) for k in RAW8}:raw_mismatch.append(fid)
        for source_key,out_key in [('connection_start','connection_start'),('zeek_reported_end','observed_end'),('replay_export_time','replay_export_time')]:
            if timestamp(r.get(source_key))!=f[out_key]:time_mismatch.append((fid,source_key))
        if r.get('duration_missing') in (True,'True'):missing_flags.append(fid)
    check('raw8_values_equal_source_no_zero_fill',not raw_mismatch,raw_mismatch[:10])
    check('times_equal_source_no_zero_fill',not time_mismatch,time_mismatch[:10])
    check('identity_fields_outside_raw8',all(set(f['raw8'])==set(RAW8) for f in flow_rows))
    check('actual_availability_remains_unknown',all(f['actual_detector_received_time'] is None for f in flow_rows))
    check('publisher_no_missing_duration_exercises_fallback',not missing_publisher_durations,missing_publisher_durations)
    check('missing_duration_sentinels_not_canonical',all(flows[k]['raw8']['duration_seconds']==-1 and flows[k]['observed_end'] is None and flows[k]['canonical_status']!='capture_qualified_C' for k in missing_flags),{'count':len(missing_flags),'statuses':dict(Counter(flows[k]['canonical_status'] for k in missing_flags))})
    metadata_mismatches=[]
    for fid,f in flows.items():
        expected_missing=[k for k in RAW8 if f['raw8'][k] is None or f['raw8'][k]<0]
        if f.get('raw8_missing_fields')!=expected_missing or f.get('duration_missing')!=('duration_seconds' in expected_missing) or f.get('raw8_complete')!=(not expected_missing):
            metadata_mismatches.append(fid)
    check('raw8_missing_metadata_preserved',not metadata_mismatches,metadata_mismatches[:10])
    unknown=[e for e in event_rows if e['label']=='unknown'];repeats=[e for e in event_rows if e['label']=='attack_associated_transfer']
    check('unknown_and_repeat_counts_not_upgraded',len(unknown)==11 and len(repeats)==3,{'unknown':len(unknown),'repeated_transfers':len(repeats)})
    pure_wrong=[l for l in links if l.get('pure_flow_truth') and events[l['event_id']]['source']=='AIT']
    check('AIT_candidate_links_never_pure_flow_truth',not pure_wrong)
    check('frozen_unified_unchanged',before=={p.name:sha(p) for p in files})
    summary=load(UNIFIED/'summary.json')
    check('summary_counts_match_actual',summary['counts']=={'events':len(event_rows),'observations':len(flow_rows),'links':len(links),'split_assignments':len(assignments)})
    check('summary_missing_counts_match_actual',summary['integrity'].get('rows_with_missing_raw8')==len(missing_flags)==441 and summary['integrity'].get('qualified_rows_with_missing_raw8')==sum(not flows[k]['raw8_complete'] for k in canonical_cloud|canonical_attack)==0)
    result={'scope':'One independent source-to-unified review; no downloads, model calls or edits to unified files.','checks':checks,'passed':all(x['pass'] for x in checks.values()),'source_counts':source_counts,'input_sha256':before,'observations':[
        {'severity':'resolved','finding':'The revision preserves raw8_missing_fields, duration_missing and raw8_complete. The metadata and summary assertions verify 441 missing-duration rows and zero incomplete canonical rows.'},
        {'severity':'resolved','finding':'Code review confirms optc_flow and unknown PE event ends now preserve null for missing duration. Actual publisher durations are present and all assembled observation times still match source.'},
        {'severity':'count_unit','finding':'The 32 cleared intranet packet candidates contain 33 whole-episode -C UIDs, of which 32 are referenced by action_connection_pairs. The unlinked tail of packet 3579 is not an omitted action candidate.', 'unlinked_cleared_uids':sorted(all_cleared_attack_uids-set(source_raw))}],
        'script_sha256':sha(Path(__file__))}
    (HERE/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':result['passed'],'checks':len(checks),'failures':{k:v for k,v in checks.items() if not v['pass']},'source_counts':source_counts,'missing_duration_sentinels':len(missing_flags)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
