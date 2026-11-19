"""Qualify pre-existing ownCloud observations for a normal-only C cohort.

Uses complete physical TCP envelopes and all service log requests in them.
No scores, packet edits, new downloads, or assumption that one normal object
makes every candidate connection normal.
"""
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent.parent
OLD=WORK/'统一事件与方法落实_20261119/ait'
PRIOR=WORK/'独立来源与算法主线_20261118'
KNOWN={'10.143.2.25':'internal_employee_2','10.143.3.65':'internal_employee_3'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')

def main():
    parser_path=PRIOR/'assemble_ait_normal_manifest.py'
    spec=importlib.util.spec_from_file_location('frozen_ait_access_parser',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    inputs={str(parser_path):sha(parser_path)}
    logs={};proxies={};backends={};coverage={}
    for rotation in ['1','4']:
        for host,stem in [('webserver','cloud.smith.russellmitchell.com-access'),('cloud_share','other_vhosts_access')]:
            member=f'gather/{host}/logs/apache2/{stem}.log.{rotation}'
            path=PRIOR/'ait_small_sources'/member;inputs[str(path)]=sha(path)
            assert inputs[str(path)]==parser.HASHES[member]
            rows=[]
            for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
                r=parser.parse_access(line,member,i);assert r is not None,(member,i)
                rows.append(r)
            logs[member]=rows
            if host=='webserver':proxies[rotation]=rows
            else:backends[rotation]=[r for r in rows if r['client_ip']=='172.19.130.68']
        days={int(r['timestamp'])//86400 for r in proxies[rotation]+backends[rotation]}
        assert max(days)-min(days)<2,(rotation,days)
        # Rotation .4 contains a small pre-validity tail from Jan20; the
        # publisher-valid run starts Jan21. Never admit that partial day.
        coverage[rotation]=(max(min(days)*86400,parser.START),min((max(days)+1)*86400,parser.END))
    semantics=load(OLD/'checksum_semantics.json')
    event_manifest=load(OLD/'normal_event_connections.json')
    for name in ['checksum_semantics.json','normal_event_connections.json','referenced_connections.jsonl']:
        inputs[str(OLD/name)]=sha(OLD/name)
    flow_rows={r['flow_id']:r for r in (json.loads(s) for s in (OLD/'referenced_connections.jsonl').read_text(encoding='utf-8').splitlines())}
    e_by_id={e['event_id']:e for e in event_manifest['events']}
    candidates=[]
    for packet in semantics['packet_candidates']:
        mapping=packet['frozen_checksum_mapping'];env=mapping['packet_envelope'];pid=packet['packet_connection_id']
        start,end=env['first'],env['last'];rotation=next((k for k,(a,b) in coverage.items() if a<=start and end<b),None)
        reasons=[];associations=[];proxy_all=[];backend_all=[]
        if rotation is None:reasons.append('whole_TCP_envelope_not_inside_one_complete_available_log_rotation')
        else:
            # Include the entire timestamp second at each boundary. This is a
            # deliberately broad candidate set, not a nearest-request join.
            lower,upper=math.floor(start),math.floor(end)+1
            proxy_all=[r for r in proxies[rotation] if lower<=r['timestamp']<upper]
            backend_all=[r for r in backends[rotation] if lower<=r['timestamp']<upper]
            if not proxy_all or not backend_all:reasons.append('no_complete_two_sided_transaction_evidence')
            if any(r['client_ip'] not in KNOWN for r in proxy_all):reasons.append('whole_envelope_contains_other_or_VPN_proxy_actor_candidates')
            for backend in backend_all:
                # Request identity includes method, decoded URI/query, version
                # and status. Keep every same-identity request in the envelope.
                matches=[r for r in proxy_all if r['key']==backend['key']]
                associations.append(dict(backend_line=backend['line'],backend_member=backend['member'],
                    proxy_lines=[r['line'] for r in matches],possible_clients=sorted({r['client_ip'] for r in matches}),
                    identity_sha256=backend['request_identity_sha256']))
                if not matches:reasons.append('backend_transaction_without_proxy_identity_match')
                elif any(r['client_ip'] not in KNOWN for r in matches):reasons.append('backend_transaction_has_unqualified_actor_explanation')
            for proxy in proxy_all:
                if not any(r['key']==proxy['key'] for r in backend_all):
                    reasons.append('proxy_transaction_without_backend_identity_match')
        event_ids=mapping['event_ids_with_request_second_proxy_payload']
        if any(e_by_id[e]['business_label']!='normal_file_download' for e in event_ids):reasons.append('unresolved_download_action')
        flow_ids=mapping['request_second_flow_ids']['ignore_checksums']
        if len(flow_ids)!=1:reasons.append('request_bearing_Zeek_record_not_unique')
        f=flow_rows[flow_ids[0]] if len(flow_ids)==1 else None
        if f and (f['duration_missing'] or f['after_input_eof'] or f['during_shutdown']):reasons.append('record_completion_not_qualified')
        candidates.append(dict(packet_connection_id=pid,flow_ids=flow_ids,event_ids=event_ids,
            complete_TCP_start=start,complete_TCP_end=end,complete_log_rotation=rotation,
            proxy_request_count=len(proxy_all),backend_request_count=len(backend_all),
            possible_actor_counts=dict(Counter(r['client_ip'] for r in proxy_all)),
            transaction_identity_candidates=associations,
            eligible=not reasons,rejection_reasons=sorted(set(reasons)),
            label='normal' if not reasons else 'unknown',
            evidence_grade='B_actor_service_complete_TCP_envelope_all_request_candidates_normal' if not reasons else None,
            model_flow=f if not reasons else None))
    selected=[c for c in candidates if c['eligible']]
    ids=[fid for c in selected for fid in c['flow_ids']]
    assert len(ids)==len(set(ids))
    summary=dict(schema='ait-normal-calibration-candidates-v1',score_calls=0,training_calls=0,
        base_packet_candidates=len(candidates),qualified_physical_connections=len(selected),qualified_model_rows=len(ids),
        rejection_reason_counts=dict(Counter(reason for c in candidates for reason in c['rejection_reasons'])),
        group='ait:russellmitchell:complete_run',proposed_role='normal_calibration_C',
        cohort='previously proven normal download request-bearing records with all full-envelope service-request candidates qualified',
        not_whole_network_normal_population=True,unique_physical_IDs=True,
        high_probability_one_percent_guarantee_claimed=False,
        known_benign_actors=KNOWN,inputs_sha256=inputs,script_sha256=sha(Path(__file__)))
    save(ROOT/'candidate_connections.json',dict(summary=summary,candidates=candidates))
    save(ROOT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
