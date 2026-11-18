"""Assemble evidence, observations and whole-chain roles; never fit or score a model."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RAW8 = ['ip_protocol', 'orig_port', 'resp_port', 'duration_seconds',
        'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
AIT = 'ait:russellmitchell:complete_run'
EMPIRE = 'optc:empire_custom:20190924_continues_20190925'
NORMAL = 'optc:20190919:declared_normal_business_subset'
UNKNOWN = 'optc:20190919:selected_normal_candidates'
METER = 'optc:meterpreter:20190925_complete_branch'
ROLES = {AIT: 'development', EMPIRE: 'external_development',
         NORMAL: 'external_development', UNKNOWN: 'unassigned_unknown',
         METER: 'reserved_historical_confirmation'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def num(v):
    if v in (None, '', '-'): return None
    return float(v)

def epoch(v):
    if v in (None, ''): return None
    if isinstance(v, (int, float)): return v
    try: return float(v)
    except ValueError: return datetime.fromisoformat(v.replace('Z', '+00:00')).timestamp()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def write_rows(path, rows):
    with path.open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as gz:
        with io.TextIOWrapper(gz, encoding='utf-8', newline='\n') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace.resolve()
    now = base/'统一事件与方法落实_20261119'
    prev = base/'独立来源与算法主线_20261118'
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def track(path):
        path = Path(path)
        inputs[str(path.relative_to(base)).replace('\\', '/')] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return path
    def load(path):
        return json.loads(track(path).read_text(encoding='utf-8-sig'))
    def lines(path):
        path = track(path)
        with (gzip.open(path, 'rt', encoding='utf-8-sig') if path.suffix == '.gz' else path.open(encoding='utf-8-sig')) as f:
            yield from (json.loads(line) for line in f if line.strip())
    def csvrows(path):
        with track(path).open(encoding='utf-8-sig', newline='') as f:
            return list(csv.DictReader(f))
    events, flows, links = {}, {}, []
    def event(eid, group, source, kind, unit, label, grade, start, end, evidence, **extra):
        assert eid not in events, eid
        events[eid] = dict(event_id=eid, group_id=group, split_role=ROLES[group], source=source,
                           kind=kind, unit=unit, label=label, evidence_grade=grade,
                           start=epoch(start), end=epoch(end), evidence=evidence,
                           label_propagates_to_all_linked_flows=False, **extra)
    def link(eid, fid, basis, **extra):
        links.append(dict(event_id=eid, flow_id=fid, basis=basis, **extra))
    def flow(fid, group, raw, provenance, canonical, availability_basis, **extra):
        raw_values={k: num(raw.get(k)) for k in RAW8}
        missing_fields=[k for k,v in raw_values.items() if v is None or v<0]
        row = dict(flow_id=fid, group_id=group, split_role=ROLES[group],
                   raw8=raw_values, raw8_missing_fields=missing_fields,
                   duration_missing='duration_seconds' in missing_fields,
                   raw8_complete=len(missing_fields)==0,
                   source_address=raw.get('source'), destination_address=raw.get('destination'),
                   connection_start=epoch(raw.get('connection_start')),
                   observed_end=epoch(raw.get('zeek_reported_end')),
                   replay_export_time=epoch(raw.get('replay_export_time')),
                   actual_detector_received_time=None, availability_basis=availability_basis,
                   canonical_status=canonical, provenance=provenance,
                   flow_label='unknown', **extra)
        if fid in flows:
            assert flows[fid] == row, fid
        else: flows[fid] = row
    def optc_flow(r, date, group, provenance):
        fid = 'optc:'+date+':'+r['uid']
        duration=num(r['duration'])
        raw = dict(ip_protocol={'tcp':6,'udp':17,'icmp':1}.get(r['proto']),
                   orig_port=r['id.orig_p'], resp_port=r['id.resp_p'], duration_seconds=r['duration'],
                   orig_pkts=r['orig_pkts'],resp_pkts=r['resp_pkts'],orig_ip_bytes=r['orig_ip_bytes'],resp_ip_bytes=r['resp_ip_bytes'],
                   source=r['id.orig_h'],destination=r['id.resp_h'],connection_start=r['ts'],
                   zeek_reported_end=None if duration is None else float(r['ts'])+duration)
        flow(fid,group,raw,provenance,'publisher_connection_record','flow_end_lower_bound_only',
             extractor='publisher Bro; original PCAP extraction not reproduced')
        return fid

    # Normal ownCloud action -> all real packet/Zeek candidates, not an invented pure-flow label.
    cloud = load(now/'ait/normal_event_connections.json')
    cloud_sem = load(now/'ait/checksum_semantics.json')
    cloud_clear = set(cloud_sem['semantic_decision']['candidate_ids'])
    canonical_cloud_ids = set()
    for e in cloud['events']:
        canonical_cloud_ids.update(e['request_second_payload_zeek_ids_by_mode'].get('ignore_checksums', []))
    for r in lines(now/'ait/referenced_connections.jsonl'):
        canonical = 'diagnostic_strict_checksum_filtered' if r['variant']=='strict' else 'not_qualified_outside_audited_packet_candidates'
        if r['flow_id'] in canonical_cloud_ids and set(r['raw_packet_candidate_ids']) <= cloud_clear:
            canonical = 'capture_qualified_C'
        flow(r['flow_id'],AIT,r,dict(file='ait/referenced_connections.jsonl',capture_sha256=r['capture_sha256'],uid=r['uid']),
             canonical,'Zeek_replay_export_not_production_ingestion',variant=r['variant'],
             packet_connection_ids=r['raw_packet_candidate_ids'])
    for e in cloud['events']:
        eid=e['event_id']; normal=e['business_label']=='normal_file_download'
        event(eid,AIT,'AIT','normal_file_download' if normal else 'download_unresolved','business_action',
              'normal' if normal else 'unknown','action_server_proxy' if normal else 'server_proxy_only',
              *e['action_window_used'],dict(file='ait/normal_event_connections.json',event_id=eid),
              association=e['request_second_proxy_payload_candidate_class'],
              selected_packet_candidate_ids=e['request_second_proxy_payload_packet_ids'],
              captured_network_leg='proxy_to_cloud_backend')
        for c in e['packet_connection_candidates']:
            for z in c['zeek_links']:
                link(eid,z['flow_id'],'time_endpoint_packet_candidate',packet_connection_id=c['packet_connection_id'],
                     proxy_payload_in_request_second=c['packet_connection_id'] in e['request_second_proxy_payload_packet_ids'],
                     record_payload_in_request_second=z.get('packet_payload_active_in_request_second',False),
                     shared_with_event_ids=c['shared_with_event_ids'],pure_flow_truth=False)

    # Actual intranet extraction; broken declared packet lengths never become valid raw8 by using -C.
    attack_sem=load(now/'ait_attack/checksum_semantics.json')
    attack_quality={}
    for c in attack_sem['packet_candidates']:
        for variant,rows in c['zeek_rows'].items():
            for r in rows:
                key=(variant,r['uid'])
                state=attack_quality.setdefault(key,dict(ids=[],blocked=False))
                state['ids'].append(c['packet_connection_id'])
                state['blocked'] |= c['model_row_blocked_on_capture_semantics']
    for variant in ['strict','ignore_checksums']:
        stage=next((now/'ait_attack/contract'/('analysis-'+variant)).glob('attempt-*'))
        pairs=csvrows(stage/'action_connection_pairs.csv')
        wanted={r['uid'] for r in pairs if r['uid']}
        for r in csvrows(stage/'raw8.csv'):
            if r['uid'] not in wanted: continue
            state=attack_quality.get((variant,r['uid']))
            status='diagnostic_strict_checksum_filtered' if variant=='strict' else 'capture_semantics_not_qualified'
            if state and variant=='ignore_checksums':
                status='blocked_invalid_captured_IP_lengths' if state['blocked'] else 'capture_qualified_C'
            flow('ait-intranet:'+variant+':'+r['uid'],AIT,r,
                 dict(file=str((stage/'raw8.csv').relative_to(now)).replace('\\','/'),uid=r['uid']),
                 status,'Zeek_replay_export_not_production_ingestion',variant=variant,
                 packet_connection_ids=state['ids'] if state else [])
        for r in pairs:
            eid='ait:intranet:'+r['action_id']
            if eid not in events:
                label='normal' if r['kind']=='normal_business' else 'malicious'
                event(eid,AIT,'AIT',r['kind'],'business_action',label,'author_action_or_server_request',num(r['start']),num(r['end']),
                      dict(file=str((stage/'action_connection_pairs.csv').relative_to(now)).replace('\\','/'),action_id=r['action_id'],
                           evidence_member=r['evidence_member'],evidence_line=r['evidence_line']),
                      business_success_time=num(r['success_time']),association='all_extractor_candidates_retained')
            if r['uid']:
                link(eid,'ait-intranet:'+variant+':'+r['uid'],r['support'],variant=variant,
                     candidate_count_in_variant=int(r['candidate_count']),pure_flow_truth=False,
                     matches_known_command_tuple=r['matches_known_command_tuple']=='True',
                     matches_prior_upload_tuple=r['matches_prior_upload_tuple']=='True')

    # OpTC normal flows and normal file objects are separate units. Mixed HTTP flows stay unknown.
    normal_rows=list(lines(now/'optc/normal_flow_events.jsonl.gz'))
    normal_ids={r['uid'] for r in normal_rows}
    for r in lines(now/'optc/normal_flow_observations.jsonl.gz'):
        raw=dict(r)
        if 'connection_start' not in raw: raw['connection_start']=epoch(r.get('start_utc'))
        if 'zeek_reported_end' not in raw: raw['zeek_reported_end']=epoch(r.get('flow_end_utc'))
        flow(r['flow_id'],NORMAL,raw,dict(file='optc/normal_flow_observations.jsonl.gz',uid=r['uid'],source_line=r.get('source_line')),
             'publisher_connection_record','flow_end_lower_bound_only',
             extractor='publisher Bro; original PCAP extraction not reproduced')
    for r in normal_rows:
        eid=r['event_id']; fid='optc:20190919:'+r['uid']
        event(eid,NORMAL,'OpTC',r['business_type'],'network_session','normal',r['evidence_grade'],r['start_utc'],r['flow_end_utc'],
              dict(file='optc/normal_flow_events.jsonl.gz',event_id=eid,source_rows=r['source_rows']),
              business_episode_group=r['business_episode_group'])
        link(eid,fid,'author_background_browser_uid_five_tuple_plus_business_evidence',pure_flow_truth=True,
             evidence_grade=r['evidence_grade'])
        flows[fid]['flow_label']='normal';flows[fid]['flow_label_grade']=r['evidence_grade']
    for r in lines(now/'optc/normal_http_file_objects.jsonl.gz'):
        event(r['event_id'],NORMAL,'OpTC',r['business_type'],'http_file_object','normal',r['evidence_grade'],
              r['request_utc'],r['file_observed_end_utc'],dict(file='optc/normal_http_file_objects.jsonl.gz',event_id=r['event_id'],source_rows=r['source_rows']),
              object_bytes=r['object_bytes'],fuid=r['fuid'])
        link(r['event_id'],'optc:20190919:'+r['uid'],'HTTP_FUID_UID_and_parent_page',pure_flow_truth=False)

    # Empire complete chain, retaining repeated transfer identity without fabricating a second official action.
    emp=load(prev/'evidence/optc/optc_event_slice.json')
    sessions={r['uid']:r for r in emp['joined_event']['sessions']}
    for r in emp['candidates']['conn_uid_matches_all']:
        fid=optc_flow(r,'20190924',EMPIRE,dict(file='previous/optc_event_slice.json',uid=r['uid'],source_line=r.get('_source_line')))
        s=sessions[r['uid']];eid='optc:empire:download:'+r['uid']
        exact=s['request_matches_reported_second_under_minus04']
        event(eid,EMPIRE,'OpTC','malicious_tool_download' if exact else 'repeated_tool_transfer','http_file_transfer',
              'malicious' if exact else 'attack_associated_transfer',
              'official_action_host_HTTP_file_conn' if exact else 'same_file_host_network_repeat',
              s['http_request_utc'][0],s['flow_observed_end_utc'][0],dict(file='previous/optc_event_slice.json',uid=r['uid']),
              matches_documented_action_second=exact)
        link(eid,fid,'host_UID_five_tuple_HTTP_FUID',pure_flow_truth=exact)
        if exact:flows[fid]['flow_label']='malicious_download'
    ctrl=load(prev/'evidence/optc/optc_control_channel_slice.json')
    for s in ctrl['all_conn_endpoint_candidates_and_joins']:
        r=s['conn_record']; fid=optc_flow(r,'20190924',EMPIRE,dict(file='previous/optc_control_channel_slice.json',uid=r['uid'],source_line=r.get('_source_line')))
        eid='optc:empire:control:'+r['uid']
        event(eid,EMPIRE,'OpTC','known_control_transport','network_session','malicious','official_agent_PID_UID_tuple_active_window',
              s['conn_start_utc'],s['flow_observed_end_utc'],dict(file='previous/optc_control_channel_slice.json',uid=r['uid']))
        link(eid,fid,'official_agent_host_UID_five_tuple',pure_flow_truth=True)
        flows[fid]['flow_label']='known_control_transport'

    # Reservation is for the entire Meterpreter branch, not only the retrieved slices.
    meter=load(now/'split_method/meterpreter_qualification.json')
    for s in meter['download_events']:
        exact=s['matches_author_host_payload'];eid=s['event_id']
        event(eid,METER,'OpTC','malicious_upgrade_download' if exact else 'additional_payload_transfer','http_file_transfer',
              'malicious' if exact else 'attack_associated_transfer',s['evidence_grade'],num(s['http']['ts']),
              s['available_at_lower_bound_utc'],dict(file='split_method/meterpreter_qualification.json',event_id=eid),
              matches_author_host_payload=exact,action_id=s['action_id'])
        for r in s['conn']:
            fid=optc_flow(r,'20190925',METER,dict(file='split_method/meterpreter_qualification.json',uid=r['uid'],source_line=r.get('_source_line')))
            link(eid,fid,'host_UID_five_tuple_HTTP_FUID',pure_flow_truth=exact)
            if exact: flows[fid]['flow_label']='malicious_download'
    for s in meter['matched_control_sessions']:
        r=s['conn'][0];eid=s['event_id']
        fid=optc_flow(r,'20190925',METER,dict(file='split_method/meterpreter_qualification.json',uid=r['uid'],source_line=r.get('_source_line')))
        event(eid,METER,'OpTC','known_control_transport','network_session','malicious','official_PID_UID_five_tuple',
              num(r['ts']),s['available_at_lower_bound_utc'],dict(file='split_method/meterpreter_qualification.json',event_id=eid))
        link(eid,fid,'official_PID_UID_five_tuple',pure_flow_truth=True)
        flows[fid]['flow_label']='known_control_transport'
    pe=load(prev/'evidence/optc/benign_file_business.json')
    for r in pe['candidate_file_rows']:
        eid='optc:20190919:unknown_PE:'+r['fuid']
        event(eid,UNKNOWN,'OpTC','unexplained_PE_download','http_file_object','unknown','benign_directory_conflicts_with_specific_object_evidence',
              num(r['ts']),None if num(r['duration']) is None else float(r['ts'])+num(r['duration']),dict(file='previous/benign_file_business.json',fuid=r['fuid']),
              unresolved_uid=r['conn_uids'])

    registry=load(now/'split_method/split_registry.json')
    # Exact row-role assignment is derived from complete groups, never random per flow.
    counts=Counter(l['event_id'] for l in links)
    for row in events.values(): row['linked_flow_count']=counts[row['event_id']]
    assert len(events)==len(set(events)) and len(flows)==len(set(flows))
    assert all(l['event_id'] in events and l['flow_id'] in flows for l in links)
    assert all(events[l['event_id']]['group_id']==flows[l['flow_id']]['group_id'] for l in links)
    assert all(len(f['raw8'])==8 and all(v is not None for v in f['raw8'].values()) for f in flows.values())
    assert all(f['raw8_complete'] for f in flows.values() if f['canonical_status'] in ['capture_qualified_C','publisher_connection_record'])
    assert len(normal_rows)==41630
    assert sum(r['label']=='normal' and r['kind']=='normal_file_download' for r in events.values())==31
    assert sum(r['unit']=='http_file_object' and r['label']=='normal' for r in events.values())==5026
    assert sum(r['group_id']==METER for r in events.values())==5
    # No qualified model record is selected from strict, and no length-damaged record is silently admitted.
    assert all(f.get('variant')=='ignore_checksums' for f in flows.values() if f['canonical_status']=='capture_qualified_C')
    event_rows=sorted(events.values(),key=lambda x:x['event_id'])
    flow_rows=sorted(flows.values(),key=lambda x:x['flow_id'])
    links.sort(key=lambda x:(x['event_id'],x['flow_id'],x.get('packet_connection_id',0)))
    write_rows(out/'events.jsonl.gz',event_rows)
    write_rows(out/'observations.jsonl.gz',flow_rows)
    write_rows(out/'event_flow_links.jsonl.gz',links)
    assignments=[dict(unit='event',id=r['event_id'],group_id=r['group_id'],split_role=r['split_role']) for r in event_rows]
    assignments += [dict(unit='observation',id=r['flow_id'],group_id=r['group_id'],split_role=r['split_role']) for r in flow_rows]
    write_rows(out/'split_assignments.jsonl.gz',assignments)
    group_rows=[]
    for g,role in ROLES.items():
        group_rows.append(dict(group_id=g,split_role=role,events=sum(r['group_id']==g for r in event_rows),
                               observations=sum(r['group_id']==g for r in flow_rows),
                               whole_group_reserved=g==METER,assembled_extent='selected_evidence_slices_not_complete_campaign'))
    write_json(out/'groups.json',dict(schema='unified-groups-v1',groups=group_rows,
               wider_source_registry='split_method/split_registry.json',wider_registered_groups=len(registry['source_chain_roles']),
               new_fit_validation_calibration_assignment=None,
               reason='Independent whole-chain roles completed; no new trained detector or threshold calibration is registered.'))
    # All declared normal connections belong to exactly one semantic evaluation unit.
    normal_units=defaultdict(list)
    for r in normal_rows:
        normal_units[(r['hostname'],r['evidence_grade'],r['business_type'])].append('optc:20190919:'+r['uid'])
    evaluation_units=[]
    for (host,grade,business),members in sorted(normal_units.items()):
        key=host+'|'+grade+'|'+business
        evaluation_units.append(dict(unit_id='optc_normal:'+hashlib.sha256(key.encode()).hexdigest()[:16],
                                     group_id=NORMAL,host_for_evaluation_only=host,evidence_grade=grade,business_type=business,
                                     denominator=len(members),fp_budget_at_one_percent=len(members)//100,
                                     flow_ids=sorted(members),role='external_development',
                                     denominator_scope='declared_business_subset_not_all_host_traffic'))
    assert sum(u['denominator'] for u in evaluation_units)==41630
    assert len({fid for u in evaluation_units for fid in u['flow_ids']})==41630
    write_rows(out/'evaluation_units.jsonl.gz',evaluation_units)
    summary=dict(schema='unified-evidence-v1',model_fits=0,model_forwards=0,threshold_searches=0,
                 counts=dict(events=len(event_rows),observations=len(flow_rows),links=len(links),split_assignments=len(assignments)),
                 event_units=dict(Counter(r['unit'] for r in event_rows)),
                 event_labels=dict(Counter(r['label'] for r in event_rows)),
                 flow_labels=dict(Counter(r['flow_label'] for r in flow_rows)),
                 observation_status=dict(Counter(r['canonical_status'] for r in flow_rows)),
                 evaluation_units=dict(declared_normal_business_units=len(evaluation_units),normal_connections=41630,
                                       units_requiring_zero_FP=sum(u['fp_budget_at_one_percent']==0 for u in evaluation_units)),
                 integrity=dict(all_links_resolve=True,cross_group_links=0,whole_group_role_conflicts=0,
                                rows_with_missing_raw8=sum(not f['raw8_complete'] for f in flow_rows),
                                qualified_rows_with_missing_raw8=0,strict_rows_used_as_canonical=0,
                                original_source_files_preserved=True),
                 limits=['Counts mix objects, actions and flows; never sum as independent attacks.',
                         'AIT action-to-flow candidates are not pure supervised labels.',
                         'OpTC B1/B2 business subset is not a whole-network false-positive denominator.',
                         'Observation table is evidence-linked, not the complete background stream required for history.',
                         'flow_end is a lower bound, not actual online availability.',
                         'Meterpreter whole branch reserved, retrieved slices not the full branch.'])
    write_json(out/'summary.json',summary)
    inputs['build_unified.py']={'sha256':sha(Path(__file__)),'bytes':Path(__file__).stat().st_size}
    write_json(out/'source_receipt.json',dict(schema='unified-input-receipt-v1',workspace=str(base),inputs=inputs))
    names=['events.jsonl.gz','observations.jsonl.gz','event_flow_links.jsonl.gz','split_assignments.jsonl.gz','evaluation_units.jsonl.gz','groups.json','summary.json','source_receipt.json']
    outputs={name:{'sha256':sha(out/name),'bytes':(out/name).stat().st_size} for name in names}
    write_json(out/'output_receipt.json',dict(schema='unified-output-receipt-v1',outputs=outputs))
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
