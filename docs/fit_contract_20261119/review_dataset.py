"""One independent, read-only materialized-dataset check; no parser or model runs."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import groupby
import gzip
import hashlib
import json
import math

NOW = Path(__file__).resolve().parent
OLD = NOW.parent / '统一事件与方法落实_20261119'
DATA = NOW / 'dataset'
RAW8 = ('ip_protocol', 'orig_port', 'resp_port', 'duration_seconds', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes')
CAPTURES = [('santos', 'intranet_server'), ('santos', 'cloud_share'), ('fox', 'intranet_server'), ('fox', 'cloud_share'), ('russellmitchell', 'cloud_share')]
IMMUTABLE = ('uid', 'source', 'destination', 'source_address', 'destination_address', 'native_protocol', 'raw8', 'raw8_missing_fields', 'raw8_complete', 'duration_missing', 'connection_start', 'observed_end', 'replay_export_time', 'after_input_eof', 'during_shutdown', 'flow_key', 'flow_key_id', 'stable_flow_id', 'capture_sha256_original', 'capture_sha256_declared', 'capture_provenance', 'zeek_record')
checks, failures = [], []

def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def rows(p):
    p = Path(p)
    with (gzip.open(p, 'rt', encoding='utf-8') if p.suffix == '.gz' else p.open(encoding='utf-8-sig')) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def check(name, ok, detail=None):
    result = {'check': name, 'passed': bool(ok)}
    if detail is not None:
        result['detail'] = detail
    checks.append(result)
    if not ok:
        failures.append(result)

def projection(r):
    return digest({k: r.get(k) for k in IMMUTABLE})

def close(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)

summary = load(DATA / 'summary.json')
observations, fingerprints, all_ids, old_ait_ids = {}, {}, set(), []
for r in rows(DATA / 'observations.jsonl.gz'):
    fid = r['flow_id']
    if fid in all_ids:
        old_ait_ids.append('duplicate:' + fid)
    all_ids.add(fid)
    if fid.startswith('ait:'):
        fingerprints[fid] = projection(r)
        observations[fid] = {k: r.get(k) for k in ('flow_id', 'stable_flow_id', 'scenario', 'capture_host', 'source_address', 'destination_address', 'dataset_role', 'split_role', 'supervised', 'label', 'raw8', 'raw8_missing_fields', 'raw8_complete', 'capture_provenance', 'replay_export_time', 'observed_end', 'after_input_eof', 'during_shutdown', 'missed_bytes', 'flow_key', 'capture_sha256_original', 'event_ids')}
    elif not fid.startswith('optc:'):
        old_ait_ids.append(fid)
check('v2_no_old_AIT_measurement_ids_or_duplicate_ids', not old_ait_ids, old_ait_ids[:10])

inventory = load(NOW / 'ait_new_runs/capture_host_inventory.json')
hosts = {(r['run'], r['capture_host']): set(r['all_local_addresses']) for r in inventory['hosts']}
facts = load(OLD / 'ait/source/capture_config/gather/cloud_share/facts.json')['ansible_facts']
hosts[('russellmitchell', 'cloud_share')] = set(facts['all_ipv4_addresses'] + facts['all_ipv6_addresses'] + ['127.0.0.1', '::1'])
expected_ids, host_view_ids, nonlocal_ids, capture_results, differences = set(), set(), set(), [], []
for run, host in CAPTURES:
    counts = Counter()
    for r in rows(NOW / 'ait_extracted' / run / host / 'connections.jsonl.gz'):
        fid = r['flow_id']
        expected_ids.add(fid)
        counts['full_connections'] += 1
        if fid not in observations or projection(r) != fingerprints[fid]:
            differences.append(fid)
        src = r['source_address'] in hosts[(run, host)]
        dst = r['destination_address'] in hosts[(run, host)]
        direction = 'both_local' if src and dst else 'local_source' if src else 'local_destination' if dst else 'neither_local'
        counts[direction] += 1
        (host_view_ids if src or dst else nonlocal_ids).add(fid)
    capture_results.append({'run': run, 'capture_host': host, 'addresses': sorted(hosts[(run, host)]), **counts})
check('all_109512_input_connections_retained_once', len(expected_ids) == 109512 and expected_ids == set(observations), {'input': len(expected_ids), 'output': len(observations), 'missing': len(expected_ids-set(observations)), 'extra': len(set(observations)-expected_ids)})
check('input_network_fields_raw8_missing_flags_and_time_unchanged', not differences, differences[:10])
excluded = load(DATA / 'excluded_host_views.json')
check('nonlocal_records_retained_but_excluded_only_from_host_views', {r['flow_id'] for r in excluded} == nonlocal_ids and len(excluded) == len(nonlocal_ids), {'host_views': len(host_view_ids), 'nonlocal_records': len(nonlocal_ids)})

model_rows = list(rows(DATA / 'model_rows.jsonl.gz'))
model_ids = {r['flow_id'] for r in model_rows}
roles = defaultdict(set)
for r in model_rows:
    roles[r['dataset_role']].add(r['stable_flow_id'])
overlap = {a+'_'+b: len(roles[a]&roles[b]) for a,b in [('F','V'),('F','C'),('V','C')]}
check('F_V_C_stable_identity_disjoint', not any(overlap.values()), overlap)
check('supervision_exactly_qualified_rows', model_ids == {fid for fid,r in observations.items() if r['supervised']} and len(model_ids) == len(model_rows) and all(observations[r['flow_id']]['label'] == r['label'] in (0,1) for r in model_rows))
check('complete_run_roles_preserved', all(r['dataset_role'] == {'santos':'F','fox':'V','russellmitchell':'C'}[r['scenario']] for r in model_rows))

aliases = list(rows(DATA / 'measurement_aliases.jsonl.gz'))
selected = {r['model_flow']['flow_id']: r for r in load(NOW / 'calibration/candidate_connections.json')['candidates'] if r['eligible']}
alias_errors = []
for a in aliases:
    old = selected[a['old_flow_id']]['model_flow']
    new = observations[a['new_flow_id']]
    key = new['flow_key']
    if (new['raw8'] != {k: old[k] for k in RAW8} or
        tuple(key[k] for k in ('native_protocol','source','orig_port','destination','resp_port','connection_start')) != tuple(old[k] for k in ('native_protocol','source','orig_port','destination','resp_port','connection_start')) or
        new['capture_sha256_original'] != 'e7039504139d6cadab64ad5c1772233a04705c06757fde3251c11beefb491bac' or new['dataset_role'] != 'C' or new['label'] != 0 or a['stable_flow_id'] != new['stable_flow_id']):
        alias_errors.append(a)
check('C25_same_capture_exact_tuple_start_and_raw8_alias', len(aliases)==25 and set(selected)=={r['old_flow_id'] for r in aliases} and len({r['new_flow_id'] for r in aliases})==25 and not alias_errors, {'aliases':len(aliases),'mismatches':alias_errors[:3]})

events = {r['event_id']:r for r in rows(DATA/'events.jsonl.gz')}
links = list(rows(DATA/'event_flow_links.jsonl.gz'))
link_pairs = Counter((r['event_id'],r['flow_id']) for r in links)
check('all_event_links_have_existing_endpoints', all(r['event_id'] in events and r['flow_id'] in all_ids for r in links), {'links':len(links)})
mapping_errors, mapped_ids, expected_new_events, new_pair_counts = [], set(), set(), Counter()
for run in ('santos','fox'):
    directory = NOW/'ait_new_runs/network_mapping'/run
    for r in rows(directory/'connections.jsonl.gz'):
        mapped_ids.add(r['flow_id'])
        if observations[r['flow_id']]['label'] != r['label']:
            mapping_errors.append('flow_label:'+r['flow_id'])
    for e in rows(directory/'events.jsonl'):
        eid=e['event_id']; expected_new_events.add(eid)
        if eid not in events or any(events[eid].get(k)!=e.get(k) for k in ('mapping_status','candidate_flow_ids','evaluation_candidate_flow_ids','action_label')):
            mapping_errors.append('event:'+eid)
    new_pair_counts.update((r['event_id'],r['flow_id']) for r in rows(directory/'links.jsonl'))
check('all_254_events_and_all_candidates_preserved_without_label_upgrade', len(expected_new_events)==254 and not mapping_errors and all(link_pairs[k]==n for k,n in new_pair_counts.items()), {'events':len(expected_new_events),'candidate_links':sum(new_pair_counts.values()),'mismatches':mapping_errors[:10]})
check('only_explicit_mapped_or_C_alias_rows_supervised', model_ids <= mapped_ids|{a['new_flow_id'] for a in aliases})
current_counts=Counter(r['event_id'] for r in links)
check('event_link_counts_current_not_historical', all(e['linked_flow_count']==current_counts[eid] for eid,e in events.items()))

context_inputs=list(rows(DATA/'context_inputs.jsonl.gz'))
context_rows=list(rows(DATA/'context_rows.jsonl.gz'))
context_by_id={r['flow_id']:r for r in context_rows}
input_errors=[]
for r in context_inputs:
    o=observations[r['measurement_flow_id']]
    expected_raw=[None if o['raw8'][k] is not None and o['raw8'][k]<0 else o['raw8'][k] for k in RAW8]
    addresses=hosts[(o['scenario'],o['capture_host'])]
    peer=o['destination_address'] if o['source_address'] in addresses else o['source_address']
    if r['flow_id'] != o['stable_flow_id'] or r['available_at'] != o['replay_export_time'] or r['raw8'] != expected_raw or r['peer'] != peer or r['host'] != f"ait:{o['scenario']}:{o['capture_host']}":
        input_errors.append(r['measurement_flow_id'])
check('context_exact_complete_host_views_with_null_missing_values', {r['measurement_flow_id'] for r in context_inputs} == host_view_ids == set(context_by_id) and len(context_inputs)==len(context_rows)==len(host_view_ids) and not input_errors, {'rows':len(context_rows),'mismatches':input_errors[:10]})

ordered=sorted(context_inputs,key=lambda r:(r['group'],r['available_at'],r['flow_id']))
states={}; context_errors=[]; tie_batches=0
for (group,t),batchiter in groupby(ordered,key=lambda r:(r['group'],r['available_at'])):
    batch=list(batchiter)
    if len(batch)>1: tie_batches+=1
    for inp in batch:
        state=states.setdefault((group,inp['host']),{'count':0,'means':[0.0]*8,'valid':[0]*8,'previous':None})
        actual=context_by_id[inp['measurement_flow_id']]
        values=[None if v is None else math.log1p(v) for v in inp['raw8']]
        prev=state['previous']; empty=prev is None
        prev_values=[0.0]*8 if empty else prev['values']
        means=[0.0]*8 if empty else [v if n else None for v,n in zip(state['means'],state['valid'])]
        previous_id=None if empty else prev['measurement_flow_id']
        maximum=None if empty else prev['available_at']
        elapsed=0.0 if empty else math.log1p(t-maximum)
        same_peer=0 if empty else int(prev['peer']==inp['peer'])
        expected_features=values+prev_values+means+[elapsed,same_peer,int(empty)]
        if not (actual['ordering_flow_id']==inp['flow_id'] and actual['physical_flow_id']==inp['flow_id'] and actual['previous_flow_id']==previous_id and actual['history_flow_count']==state['count'] and actual['max_history_available_at']==maximum and (maximum is None or maximum<t) and actual['mean_valid_counts']==state['valid'] and close(actual['model_features'],expected_features)):
            context_errors.append(inp['measurement_flow_id'])
    for inp in batch:
        state=states[(group,inp['host'])]
        values=[None if v is None else math.log1p(v) for v in inp['raw8']]
        state['count']+=1
        for i,v in enumerate(values):
            if v is not None:
                state['valid'][i]+=1
                state['means'][i]+=(v-state['means'][i])/state['valid'][i]
        state['previous']=dict(inp,values=values)
check('context_independent_strict_past_means_previous_and_stable_ties', not context_errors, {'checked_rows':len(context_rows),'equal_time_batches':tie_batches,'mismatches':context_errors[:10]})
check('context_row_order_uses_stable_flowkey_not_random_UID', [(r['group'],r['effective_available_at'],r['ordering_flow_id']) for r in context_rows] == [(r['group'],r['available_at'],r['flow_id']) for r in ordered])

quality_tables=defaultdict(Counter)
quality_flags=('truncated','orig_len_adjusted','zeek_missed_bytes','raw8_missing','after_input_eof')
for m in model_rows:
    o=observations[m['flow_id']]; p=o['capture_provenance']
    q=(p['capture_truncated_record_candidates']>0,p['orig_len_adjusted_record_candidates']>0,(o['missed_bytes'] or 0)>0,not o['raw8_complete'],bool(o['after_input_eof']))
    for scope in ('all',m['dataset_role'],m['dataset_role']+':'+m['capture_host']):
        quality_tables[scope][(m['label'],q)]+=1
quality=[]; perfect=[]
for scope,table in sorted(quality_tables.items()):
    labels={label for label,q in table}
    tests={}
    for i,name in enumerate(quality_flags):
        cells=defaultdict(set)
        for (label,q),n in table.items():cells[q[i]].add(label)
        tests[name]=len(labels)==2 and all(len(x)==1 for x in cells.values())
    joint=defaultdict(set)
    for label,q in table:joint[q].add(label)
    tests['joint_flags']=len(labels)==2 and all(len(x)==1 for x in joint.values())
    perfect.extend({'scope':scope,'flag':k} for k,v in tests.items() if v)
    quality.append({'scope':scope,'rows':[{'label':label,**dict(zip(quality_flags,q)),'rows':n} for (label,q),n in sorted(table.items())],'perfect_classification_by_quality_flag':tests})
check('quality_flags_have_no_perfect_class_relationship', not perfect, perfect)
check('summary_counts_match_actual_artifacts', summary['new_observations']==len(observations) and summary['unified_observations']==len(all_ids) and summary['complete_background_host_views']==len(context_rows) and summary['supervised_rows']==len(model_rows) and summary['unified_events']==len(events) and summary['unified_links']==len(links))

result={'schema':'independent-dataset-review-v1','status':'passed' if not failures else 'findings','dataset':str(DATA),'checks':checks,'failures':failures,'capture_host_counts':capture_results,'quality_by_class':quality,'quality_perfect_relationships':perfect,'model_role_class_counts':[{'role':role,'label':label,'rows':n} for (role,label),n in sorted(Counter((r['dataset_role'],r['label']) for r in model_rows).items())],'input_receipt_sha256':hashlib.sha256((DATA/'output_receipt.json').read_bytes()).hexdigest(),'review_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'normalization_runs':0,'model_calls':0,'training_calls':0}
(NOW/'dataset_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':len(checks),'failed':len(failures),'failures':failures,'quality_perfect_relationships':perfect},ensure_ascii=False))
