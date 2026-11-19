"""Real flow/event facts only; no classifier, score, threshold, or parameter search."""
from __future__ import annotations
import bisect, collections, gzip, hashlib, json, statistics
from datetime import datetime
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--work", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
WORK, OUT = args.work, args.output
OUT.mkdir(parents=True, exist_ok=True)
NOW = WORK / '覆盖与历史依据落实_20261119'
OPTC = WORK / '统一事件与方法落实_20261119' / 'optc'
PATHS = [NOW/'batch/rows.jsonl.gz', OPTC/'context_replay_inputs.jsonl.gz',
         OPTC/'normal_flow_events.jsonl.gz', OPTC/'normal_http_file_objects.jsonl.gz',
         NOW/'attack_coverage/full_capture/connections.jsonl.gz']

def rows(p):
    with gzip.open(p, 'rt', encoding='utf8') as f:
        yield from (json.loads(s) for s in f)

def epoch(v):
    return datetime.fromisoformat(v).timestamp() if isinstance(v,str) else v

def stats(xs):
    return {'n':len(xs),'min':min(xs) if xs else None,
            'median':statistics.median(xs) if xs else None,'max':max(xs) if xs else None}

def annotate(sequence):
    groups=collections.defaultdict(list)
    for r in sequence: groups[(r['group'],r['host'])].append(r)
    details={}
    for key, rr in groups.items():
        rr.sort(key=lambda r:(r['time'],r['id']))
        times=[r['time'] for r in rr]
        by_peer=collections.defaultdict(list)
        cumulative=collections.Counter()
        cursor=0
        for r in rr:
            while cursor<len(rr) and rr[cursor]['time']<r['time']:
                p=rr[cursor];by_peer[p['peer']].append(p)
                cumulative[p['label']]+=1
                cumulative['tls443']+=int(p['raw'][0]==6 and p['raw'][2]==443)
                cumulative['normal_tls443']+=int(p['raw'][0]==6 and p['raw'][2]==443 and p['label']==0)
                cumulative['normal_http80']+=int(p['raw'][0]==6 and p['raw'][2]==80 and p['label']==0)
                cursor+=1
            n=bisect.bisect_left(times,r['time'])
            prev=rr[n-1] if n else None
            last_batch=rr[bisect.bisect_left(times,prev['time']):n] if prev else []
            same=by_peer[r['peer']]
            d={**r,'prefix_count':n,'prefix_labels':{str(k):cumulative[k] for k in (0,1,None)},
               'prior_tls443':cumulative['tls443'],'prior_normal_tls443':cumulative['normal_tls443'],
               'prior_normal_http80':cumulative['normal_http80'],
               'same_peer_prefix':len(same),'previous_same_peer':bool(prev and prev['peer']==r['peer']),
               'last_prior_batch_count':len(last_batch),
               'last_prior_batch_any_same_peer':any(p['peer']==r['peer'] for p in last_batch),
               'last_prior_batch_all_same_peer':bool(last_batch) and all(p['peer']==r['peer'] for p in last_batch),
               'previous_id':prev['id'] if prev else None,
               'previous_label':prev['label'] if prev else None,
               'previous_service':prev['service'] if prev else None,
               'previous_gap_seconds':r['time']-prev['time'] if prev else None,
               'same_peer_latest_gap_seconds':r['time']-same[-1]['time'] if same else None,
               'previous_same_non_duration_shape':bool(prev and (r['raw'][:1]+r['raw'][4:])==(prev['raw'][:1]+prev['raw'][4:])),
               'response_dominant':None if r['raw'][6] is None or r['raw'][7] is None else r['raw'][7]>r['raw'][6]}
            details[r['id']]=d
    return details

def summary(rr):
    return {'flows':len(rr),'hosts':len({r['host'] for r in rr}),
            'groups':len({r['group'] for r in rr}),
            'same_peer_predecessor':sum(r['previous_same_peer'] for r in rr),
            'last_prior_batch_any_same_peer':sum(r['last_prior_batch_any_same_peer'] for r in rr),
            'last_prior_batch_all_same_peer':sum(r['last_prior_batch_all_same_peer'] for r in rr),
            'nonunique_last_prior_batch':sum(r['last_prior_batch_count']>1 for r in rr),
            'any_prior_same_peer':sum(r['same_peer_prefix']>0 for r in rr),
            'previous_same_non_duration_shape':sum(r['previous_same_non_duration_shape'] for r in rr),
            'any_prior_tls443':sum(r['prior_tls443']>0 for r in rr),
            'any_prior_confirmed_normal_tls443':sum(r['prior_normal_tls443']>0 for r in rr),
            'any_prior_confirmed_attack':sum(r['prefix_labels']['1']>0 for r in rr),
            'prior_confirmed_attack_count':stats([r['prefix_labels']['1'] for r in rr]),
            'response_dominant':sum(r['response_dominant'] is True for r in rr),
            'prefix_count':stats([r['prefix_count'] for r in rr]),
            'same_peer_latest_gap_seconds':stats([r['same_peer_latest_gap_seconds'] for r in rr if r['same_peer_latest_gap_seconds'] is not None]),
            'previous_gap_seconds':stats([r['previous_gap_seconds'] for r in rr if r['previous_gap_seconds'] is not None])}

def main():
    batch={r['flow_id']:r for r in rows(PATHS[0]) if r['flow_id'].startswith(('optc:','unraveled:'))}
    normal_events={'optc:20190919:'+r['uid']:r for r in rows(PATHS[2])}
    sequence=[]
    for r in rows(PATHS[1]):
        b=batch.get(r['physical_flow_id'],{})
        sequence.append({'id':r['physical_flow_id'],'host':r['host'],'peer':r['peer'],'group':r['group'],
            'time':epoch(r['flow_end_at']),'raw':r['raw8'],'label':b.get('label'),
            'kind':b.get('target_kind'),'service':'TLS443' if r['raw8'][2]==443 else 'HTTP80' if r['raw8'][2]==80 else 'other',
            'business':normal_events.get(r['physical_flow_id'],{}).get('business_type'),
            'host_is_originator':r['host_is_originator']})
    od=annotate(sequence)
    normal=[r for r in od.values() if r['label']==0]
    controls=[r for r in od.values() if r['label']==1 and r['kind']=='control']
    downloads=[r for r in od.values() if r['label']==1 and r['kind']=='download']
    optc_groups={'control':controls,'attack_download':downloads,
        'normal_tls':[r for r in normal if r['business']=='normal_tls_browser_background'],
        'normal_http':[r for r in normal if r['business']!='normal_tls_browser_background'],
        'same_host_normal_tls':[r for r in normal if r['host']=='142.20.57.246' and r['business']=='normal_tls_browser_background'],
        'same_host_normal_http':[r for r in normal if r['host']=='142.20.57.246' and r['business']!='normal_tls_browser_background']}
    objects=list(rows(PATHS[3]));parent_status=collections.Counter();seen_flows=collections.defaultdict(set)
    for obj in objects:
        fid='optc:20190919:'+obj['uid'];pid='optc:20190919:'+obj['parent_document_uid']
        if pid==fid:s='same_connection_not_prior_completed_flow'
        elif pid not in od:s='parent_not_in_existing_context'
        elif od[pid]['host']!=od[fid]['host']:s='different_monitored_host'
        elif od[pid]['time']<od[fid]['time']:s='separate_parent_flow_available'
        elif od[pid]['time']==od[fid]['time']:s='separate_parent_tied'
        else:s='separate_parent_not_yet_ended'
        parent_status[s]+=1;seen_flows[s].add(fid)
    fields=['ip_protocol','orig_port','resp_port','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes']
    us=[]
    for r in rows(PATHS[4]):
        if not r['involves_monitored_host']:continue
        b=batch.get(r['flow_id'],{})
        assert b, r['flow_id']
        us.append({'id':r['flow_id'],'host':'10.1.3.8','peer':r['destination'] if r['source']=='10.1.3.8' else r['source'],
            'group':r['source_group'],'time':r['replay_export_time'],'raw':[r['raw8'][k] for k in fields],
            'label':b.get('label'),'kind':b.get('target_kind'),
            'service':'|'.join(r.get('normal_business_strata',[])) if b.get('label')==0 else r.get('service'),
            'business':r.get('normal_business_strata',[]),'host_is_originator':r['source']=='10.1.3.8',
            'connection_start':r['connection_start'],'observed_end':r['observed_end']})
    ud=annotate(us)
    ucontrols=[r for r in ud.values() if r['label']==1]
    cfirst=min(r['time'] for r in ucontrols);clast=max(r['time'] for r in ucontrols)
    ugroups={'control':ucontrols,'normal':[r for r in ud.values() if r['label']==0],
        'normal_tls443':[r for r in ud.values() if r['label']==0 and r['raw'][0]==6 and r['raw'][2]==443],
        'normal_ssh':[r for r in ud.values() if r['label']==0 and 'ssh' in r['business']],
        'normal_in_control_export_span':[r for r in ud.values() if r['label']==0 and cfirst<=r['time']<=clast],
        'normal_tls443_in_control_export_span':[r for r in ud.values() if r['label']==0 and r['raw'][0]==6 and r['raw'][2]==443 and cfirst<=r['time']<=clast]}
    controlled_peers={r['peer'] for r in ucontrols}
    report={'scope':'existing_real_event_and_connection_facts_only','fitting_calls':0,'model_calls':0,'threshold_calls':0,
        'OpTC':{'context_records':len(sequence),'groups':{k:summary(v) for k,v in optc_groups.items()},
          'attack_download_details':downloads,
          'controls_non_duration_shapes':dict(collections.Counter(str(r['raw'][:1]+r['raw'][4:]) for r in controls)),
          'normal_objects_parent_completion':dict(parent_status),'normal_objects_parent_completion_unique_connections':{k:len(v) for k,v in seen_flows.items()},
          'same_host_normal_examples':sorted(optc_groups['same_host_normal_tls'],key=lambda r:r['id'])[:3],
          'normal_http_examples':sorted(optc_groups['normal_http'],key=lambda r:abs(r['raw'][7]-downloads[0]['raw'][7]))[:3]},
        'Unraveled':{'context_records':len(us),'groups':{k:summary(v) for k,v in ugroups.items()},
          'control_export_range':[cfirst,clast],'control_peers':sorted(controlled_peers),
          'normal_connections_to_control_peers':sum(r['label']==0 and r['peer'] in controlled_peers for r in ud.values()),
          'control_services':dict(collections.Counter((r['raw'][0],r['raw'][2]) for r in ucontrols)),
          'control_non_duration_unique_shapes':len({tuple(r['raw'][:1]+r['raw'][4:]) for r in ucontrols}),
          'control_observation_time_counts':len({r['time'] for r in ucontrols}),
          'control_raw5_shapes_most_common':collections.Counter(str(r['raw'][:1]+r['raw'][4:]) for r in ucontrols).most_common(5),
          'normal_examples_during_control':sorted(ugroups['normal_tls443_in_control_export_span'],key=lambda r:r['time'])[:3]},
        'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in PATHS]}
    report['Unraveled']['control_services']={str(k):v for k,v in report['Unraveled']['control_services'].items()}
    (OUT/'optc_unraveled_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'OpTC':report['OpTC']['groups'],'Unraveled':report['Unraveled']['groups'],
                      'parent_completion':dict(parent_status)},ensure_ascii=True,indent=2))

if __name__=='__main__':main()
