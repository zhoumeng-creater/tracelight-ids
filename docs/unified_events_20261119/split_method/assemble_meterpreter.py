"""Qualification of the preselected first Meterpreter action, without model use."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote

BASE = Path(__file__).resolve().parent
WORK = BASE.parents[1]
helper_path = WORK / '独立来源与算法主线_20261118/scripts/assemble_optc_event_slice.py'
spec = importlib.util.spec_from_file_location('optc_prior', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ACTION = datetime.fromisoformat('2019-09-25T10:29:42-04:00').timestamp()
TARGETS = {
    'sysclient0051.systemia.com':dict(ip='142.20.56.52',action='2019-09-25T10:29:42-04:00',pids=(2712,568),payload='ckfgw.exe'),
    'sysclient0351.systemia.com':dict(ip='142.20.57.96',action='2019-09-25T11:23:31-04:00',pids=(1932,1256),payload='f.exe')}
SERVER = '53.192.68.50'

def main():
    receipt_path = BASE / 'meterpreter_source_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    sources = {s['local_name']:s for s in receipt['sources']}
    stats, ecar = {}, []
    for name, s in sources.items():
        if 'ecarbro' not in name:
            continue
        count = 0
        with gzip.open(s['path'],'rt',encoding='utf-8') as f:
            for lineno,line in enumerate(f,1):
                row = json.loads(line)
                count += 1
                if row['hostname'].casefold() in TARGETS:
                    ecar.append({**row,'_source_file':name,'_source_line':lineno})
        stats[name]={'rows':count}
    http = []
    for hn in [name for name in sources if '_http.' in name]:
        for row in helper.bro_rows(Path(sources[hn]['path']), stats.setdefault(hn,{})):
            if row.get('id.orig_h') in {t['ip'] for t in TARGETS.values()} and row.get('id.resp_h') == SERVER:
                http.append({**row,'_source_file':hn})
    downloads = [r for r in http if unquote(urlsplit(r.get('uri','')).path).casefold().split('/')[-1]
                 in ['update.exe','ckfgw.exe','f.exe']]
    download_uids = {r['uid'] for r in downloads}
    control_host = [r for r in ecar if r['properties'].get('dest_ip') == SERVER
                    and r['properties'].get('dest_port') == '8080'
                    and r.get('pid') in TARGETS[r['hostname'].casefold()]['pids']]
    all_uids = download_uids | {r['properties']['bro_uid'] for r in control_host}
    files, conn = [], []
    for name in sources:
        prefix = 'files' if '_files.' in name else 'conn' if '_conn.' in name else None
        if prefix is None:
            continue
        target = files if prefix == 'files' else conn
        for row in helper.bro_rows(Path(sources[name]['path']), stats.setdefault(name,{})):
            if (prefix=='conn' and row['uid'] in all_uids) or (prefix=='files' and helper.value_set(row.get('conn_uids')) & download_uids):
                target.append({**row,'_source_file':name})
    download_events = []
    for h in downloads:
        uid = h['uid']
        host = next(host for host,t in TARGETS.items() if t['ip']==h['id.orig_h'])
        selected_target = TARGETS[host]
        action_epoch = datetime.fromisoformat(selected_target['action']).timestamp()
        matches_payload = unquote(urlsplit(h['uri']).path).casefold().split('/')[-1] == selected_target['payload']
        er = [r for r in ecar if r['properties'].get('bro_uid')==uid]
        fr = [r for r in files if uid in helper.value_set(r.get('conn_uids'))]
        cr = [r for r in conn if r['uid']==uid]
        complete = bool(fr) and all(r.get('missing_bytes')=='0' and r.get('overflow_bytes')=='0'
            and r.get('timedout')=='F' and r.get('seen_bytes')==r.get('total_bytes') for r in fr)
        checks = [{'uid':uid,'all_five_tuple_fields_match':all([
            r['properties'].get('src_ip')==h.get('id.orig_h'),
            r['properties'].get('src_port')==h.get('id.orig_p'),
            r['properties'].get('dest_ip')==h.get('id.resp_h'),
            r['properties'].get('dest_port')==h.get('id.resp_p'),
            r['properties'].get('l4protocol')=='6'])} for r in er]
        ends = [float(r['ts'])+float(r['duration']) for r in cr+fr if r.get('duration','-')!='-']
        download_events.append({
            'event_id':f'optc:meterpreter:20190925:download:{uid}',
            'group_id':'optc:meterpreter:20190925_complete_branch',
            'host':host,'matches_author_host_payload':matches_payload,
            'action_id':f'optc:meterpreter:20190925:{host.split(".")[0]}:malicious_upgrade' if matches_payload else None,
            'http':h,'files':fr,'conn':cr,'host_records':er,'tuple_checks':checks,
            'file_complete':complete,'http_minus_author_action_seconds':float(h['ts'])-action_epoch,
            'matches_documented_second':int(float(h['ts']))==int(action_epoch),
            'evidence_grade':'host_network_complete_file' if complete and er and cr and h['status_code']=='200' else 'incomplete',
            'available_at_lower_bound_utc':helper.utc(max(ends)) if ends else None,
            'available_at_actual_utc':None,
            'causal_predecessor_label':None})
    joined_controls = []
    for uid in sorted({r['properties']['bro_uid'] for r in control_host}):
        er=[r for r in control_host if r['properties']['bro_uid']==uid]
        cr=[r for r in conn if r['uid']==uid]
        if cr:
            checks = [all([r['properties'].get('src_ip')==c.get('id.orig_h'),
                r['properties'].get('src_port')==c.get('id.orig_p'),
                r['properties'].get('dest_ip')==c.get('id.resp_h'),
                r['properties'].get('dest_port')==c.get('id.resp_p'),
                r['properties'].get('l4protocol')=='6']) for r in er for c in cr]
            joined_controls.append({'event_id':f'optc:meterpreter:20190925:control:{uid}',
                'group_id':'optc:meterpreter:20190925_complete_branch','uid':uid,
                'host_records':er,'conn':cr,
                'all_five_tuple_fields_match':all(checks),'host':er[0]['hostname'].casefold(),
                'available_at_lower_bound_utc':helper.utc(max(float(c['ts'])+float(c['duration']) for c in cr if c['duration']!='-')),
                'label':'known_meterpreter_transport_not_per_command',
                'all_host_path_labels':[r['properties'].get('image_path') for r in ecar if r['properties'].get('bro_uid')==uid],
                'available_at_actual_utc':None})
    report={
        'schema':'optc-meterpreter-qualification-v1','model_calls':0,
        'split_role':'reserved_external_historical_confirmation_no_fit_or_selection',
        'full_group':'Both 0051 and 0351, migrations and reruns remain one branch. These slices qualify the first official upgrade on each host; they are not the complete group.',
        'selection_rule':'Two officially named upgrade hosts/actions determine the hours; adjacent/final conn rotation locates fixed author PID-linked UID. No score or feature distribution selects slices.',
        'ground_truth':{'path':str(WORK/'独立来源与算法主线_20261118/evidence/optc/OpTCRedTeamGroundTruth.pdf'),
            'pages_one_based':[6,7],'first_action_local':'2019-09-25T10:29:42-04:00',
            'host':'Sysclient0051','expected_original_pid':2712,'expected_original_image':'cKfGW.exe',
            'migration_pid':568,'migration_process':'LSASS','server':SERVER,'callback_port':8080,
            'second_host':'Sysclient0351','second_action_local':'2019-09-25T11:23:31-04:00',
            'second_original_pid':1932,'second_original_image':'f.exe','second_migration_pid':1256},
        'sources_receipt':str(receipt_path),'source_receipt_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        'full_scan_counts':stats,'download_events':download_events,
        'matched_control_sessions':joined_controls,
        'unjoined_selected_host_control_uids':sorted({r['properties']['bro_uid'] for r in control_host}-{r['uid'] for r in conn}),
        'retained_host_server_http_candidates':http,
        'summary':{'download_candidates':len(download_events),
            'documented_second_matches':sum(r['matches_documented_second'] for r in download_events),
            'complete_host_network_downloads':sum(r['evidence_grade']=='host_network_complete_file' for r in download_events),
            'complete_downloads_matching_author_host_payload':sum(r['evidence_grade']=='host_network_complete_file' and r['matches_author_host_payload'] for r in download_events),
            'matched_control_sessions':len(joined_controls),
            'hosts_with_complete_download_and_exact_control':sorted({r['host'] for r in download_events if r['evidence_grade']=='host_network_complete_file' and r['matches_author_host_payload']} & {r['host'] for r in joined_controls if r['all_five_tuple_fields_match']}),
            'normal_denominator_assigned':False,'full_branch_assembled':False,
            'ready_for_model_selection':False},
        'limits':['No byte hash proves exact malicious payload identity; author action plus update path and process association provide truth.',
            'Host image labels for the same PID/UID may conflict and are all retained.',
            'Bro observed end is a lower bound, not measured export/ingestion time.',
            'No residual same-hour sessions are assigned normal.',
            'Qualification records must not determine algorithm, window, threshold or features.'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (BASE/'meterpreter_qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False))

if __name__ == '__main__':
    main()
