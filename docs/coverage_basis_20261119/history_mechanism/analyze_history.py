"""Read frozen observations/context; describe retained evidence. No model operations."""
from __future__ import annotations
import bisect
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = WORK / '拟合来源与划分落实_20261119' / 'dataset'
OLD = WORK / '统一事件与方法落实_20261119'

def rows(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def write_rows(name, value):
    with gzip.open(OUT / name, 'wt', encoding='utf-8') as stream:
        for row in value:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')

def main():
    paths = [DATA / n for n in ('events.jsonl.gz', 'event_flow_links.jsonl.gz',
            'observations.jsonl.gz', 'context_rows.jsonl.gz', 'context_inputs.jsonl.gz')]
    paths += [OLD / 'split_method/context_rows.jsonl.gz', OLD / 'optc/context_replay_inputs.jsonl.gz']
    paths += [DATA / 'inference_rows.jsonl.gz']
    events = {}
    for e in rows(paths[0]):
        if e.get('source_run') in ('santos', 'fox') or e.get('group_id') == 'optc:empire_custom:20190924_continues_20190925' or e.get('kind') in ('webpage_image_download', 'webpage_stylesheet_download'):
            events[e['event_id']] = e
    candidates = collections.defaultdict(set)
    linked = collections.defaultdict(list)
    for link in rows(paths[1]):
        eid = link['event_id']
        if eid in events:
            candidates[eid].add(link['flow_id'])
    for eid, e in events.items():
        # Use every original mapper candidate; evaluation exclusions do not select evidence.
        candidates[eid].update(e.get('candidate_flow_ids', []))
        for fid in candidates[eid]:
            linked[fid].append(eid)
    obs = {}
    for r in rows(paths[2]):
        if r.get('split_role') == 'reserved_historical_confirmation':
            continue
        label = r.get('label')
        if label not in (0, 1):
            label = {'normal': 0, 'malicious': 1}.get(r.get('flow_label'))
        obs[r['flow_id']] = {k: r.get(k) for k in ('connection_start', 'observed_end', 'replay_export_time', 'source_address', 'destination_address')}
        obs[r['flow_id']]['label'] = label
    # The frozen inference table owns external binary flow labels; no score fields exist.
    for r in rows(paths[7]):
        if r['flow_id'] in obs and r.get('label') in (0, 1):
            obs[r['flow_id']]['label'] = r['label']
    inputs = {}
    for p in (paths[4], paths[6]):
        for r in rows(p):
            inputs[r.get('measurement_flow_id', r['flow_id'])] = {'peer': r['peer'], 'host': r['host']}
    contexts = {}
    by_physical = collections.defaultdict(list)
    groups = collections.defaultdict(list)
    for p in (paths[3], paths[5]):
        for r in rows(p):
            fid = r['flow_id']
            physical = fid if fid.startswith('ait:') else r['physical_flow_id']
            context = {k: r[k] for k in ('group', 'host', 'time_mode', 'effective_available_at', 'previous_flow_id', 'history_flow_count', 'same_peer_as_previous')}
            context.update(flow_id=fid, physical_id=physical, peer=inputs[fid]['peer'])
            contexts[fid] = context
            by_physical[physical].append(context)
            groups[(r['group'], r['host'])].append(context)
    def category(fid):
        if fid is None:
            return 'empty'
        f = contexts[fid]['physical_id'] if fid in contexts else fid
        es = [events[x] for x in linked[f]]
        chain = any(e.get('action_label') == 1 or e.get('kind') in ('known_control_transport', 'malicious_tool_download', 'repeated_tool_transfer') for e in es)
        label = obs.get(f, {}).get('label')
        if chain and label == 1:
            return 'confirmed_attack_chain_flow'
        if chain:
            return 'possible_attack_chain_candidate'
        if label == 0:
            return 'qualified_normal_flow'
        if any(e.get('action_label') == 0 or e['kind'] in ('webpage_image_download', 'webpage_stylesheet_download') for e in es):
            return 'possible_normal_business_candidate'
        return 'unknown_flow'
    cats = ('confirmed_attack_chain_flow', 'possible_attack_chain_candidate', 'qualified_normal_flow', 'possible_normal_business_candidate', 'unknown_flow')
    indexes = {}
    for key, rr in groups.items():
        rr.sort(key=lambda x: (x['effective_available_at'], x['flow_id']))
        cum = {c: [0] for c in cats}
        for r in rr:
            cls = category(r['flow_id'])
            for c in cats:
                cum[c].append(cum[c][-1] + int(c == cls))
        indexes[key] = (rr, [r['effective_available_at'] for r in rr], cum)
    def event_chain(e):
        return e.get('chain_id') or (e.get('group_id') if e.get('group_id') == 'optc:empire_custom:20190924_continues_20190925' else None)
    def action_start(e):
        return e.get('start', e.get('action_evidence', {}).get('intent', {}).get('timestamp'))
    def action_end(e):
        ae = e.get('action_evidence', {})
        return e.get('end', ae.get('completion', ae.get('response', {})).get('timestamp'))
    result = []
    candidate_rows = []
    for eid, e in sorted(events.items()):
        chain = event_chain(e)
        start = action_start(e)
        prior_actions = [a for a in events.values() if chain and event_chain(a) == chain and action_end(a) is not None and start is not None and action_end(a) <= start and a['event_id'] != eid]
        issuing_actions = []
        if e['kind'] == 'attack_download':
            requested_url = e['action_evidence']['url']
            issuing_actions = [a for a in events.values() if event_chain(a) == chain and a['kind'] == 'webshell_command'
                and requested_url in a.get('action_evidence', {}).get('command', [])]
        detail = []
        for fid in sorted(candidates[eid]):
            for c in by_physical[fid]:
                q = c['effective_available_at']
                rr, ts, cum = indexes[(c['group'], c['host'])]
                n = bisect.bisect_left(ts, q)
                assert n == c['history_flow_count'], (fid, n, c['history_flow_count'])
                prev = contexts.get(c['previous_flow_id'])
                if prev:
                    assert prev['effective_available_at'] < q
                    assert prev['host'] == c['host'] and prev['group'] == c['group']
                    assert int(prev['peer'] == c['peer']) == c['same_peer_as_previous']
                prior_status = []
                for a in prior_actions:
                    statuses = []
                    for afid in sorted(candidates[a['event_id']]):
                        cc = [x for x in by_physical[afid] if x['host'] == c['host'] and x['group'] == c['group']]
                        statuses.append({'flow_id': afid, 'same_as_query_connection': afid == fid,
                            'available_before_query': any(x['effective_available_at'] < q for x in cc),
                            'retained_in_last_slot': bool(prev and prev['physical_id'] == afid),
                            'export_times': [x['effective_available_at'] for x in cc],
                            'flow_label': obs.get(afid, {}).get('label')})
                    prior_status.append({'event_id': a['event_id'], 'kind': a['kind'],
                        'action_start': action_start(a), 'action_end': action_end(a), 'candidates': statuses,
                        'any_available': any(x['available_before_query'] for x in statuses),
                        'all_available': bool(statuses) and all(x['available_before_query'] for x in statuses),
                        'any_in_last_slot': any(x['retained_in_last_slot'] for x in statuses)})
                row = {'event_id': eid, 'kind': e['kind'], 'run': e.get('source_run', 'OpTC'),
                    'flow_id': fid, 'context_view_id': c['flow_id'], 'group': c['group'], 'host': c['host'],
                    'query_time_mode': c['time_mode'], 'query_available_at': q,
                    'action_start': start, 'action_end': action_end(e),
                    'query_minus_action_end_seconds': None if action_end(e) is None else q - action_end(e),
                    'flow_label': obs.get(fid, {}).get('label'), 'peer': c['peer'],
                    'previous_flow_id': None if not prev else prev['physical_id'],
                    'previous_context_view_id': c['previous_flow_id'], 'previous_class': category(c['previous_flow_id']),
                    'previous_linked_event_ids': [] if not prev else linked[prev['physical_id']],
                    'previous_available_at': None if not prev else prev['effective_available_at'],
                    'previous_peer': None if not prev else prev['peer'],
                    'same_peer_as_previous': c['same_peer_as_previous'],
                    'prefix_count': n, 'prefix_first_available_at': ts[0] if n else None,
                    'prefix_span_seconds': q - ts[0] if n else 0,
                    'prefix_categories': {cat: cum[cat][n] for cat in cats},
                    'prior_completed_same_workflow_actions': prior_status,
                    'exact_url_wget_commands': [{'event_id': a['event_id'], 'command': a['action_evidence'].get('command'),
                        'intent': a['action_evidence']['intent'], 'response': a['action_evidence']['response'],
                        'candidates': [{'flow_id': afid, 'same_as_query_connection': afid == fid,
                            'export_times': [x['effective_available_at'] for x in by_physical[afid]],
                            'available_before_query': any(x['effective_available_at'] < q for x in by_physical[afid]),
                            'retained_in_last_slot': bool(prev and prev['physical_id'] == afid)}
                            for afid in sorted(candidates[a['event_id']])]} for a in issuing_actions],
                    'observation': obs.get(fid)}
                detail.append(row)
                candidate_rows.append(row)
        result.append({'event_id': eid, 'run': e.get('source_run', 'OpTC'), 'kind': e['kind'],
            'unit': e.get('unit'), 'chain_id': chain, 'action_start': start, 'action_end': action_end(e),
            'action_evidence': e.get('action_evidence', e.get('evidence')), 'candidate_flow_ids': sorted(candidates[eid]),
            'candidate_count': len(candidates[eid]), 'context_view_count': len(detail),
            'all_candidates_have_context': all(by_physical[f] for f in candidates[eid]) and bool(candidates[eid]),
            'candidate_details': detail})
    write_rows('event_history_evidence.jsonl.gz', result)
    write_rows('candidate_history_evidence.jsonl.gz', candidate_rows)
    table_fields = ['event_id', 'kind', 'run', 'flow_id', 'context_view_id', 'query_time_mode',
        'query_available_at', 'action_start', 'action_end', 'query_minus_action_end_seconds',
        'flow_label', 'peer', 'previous_flow_id', 'previous_class', 'previous_available_at',
        'previous_peer', 'same_peer_as_previous', 'prefix_count', 'prefix_first_available_at', 'prefix_span_seconds']
    with (OUT / 'all_event_candidate_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=table_fields + list(cats))
        writer.writeheader()
        for c in candidate_rows:
            writer.writerow({**{k: c[k] for k in table_fields}, **c['prefix_categories']})
    def summarize(rr):
        cc = [c for r in rr for c in r['candidate_details']]
        unique_cc = {c['context_view_id']: c for c in cc}.values()
        return {'events': len(rr), 'unique_candidate_flows': len({f for r in rr for f in r['candidate_flow_ids']}),
            'event_candidate_context_rows': len(cc), 'all_candidates_have_context_events': sum(r['all_candidates_have_context'] for r in rr),
            'previous_class_rows': dict(collections.Counter(c['previous_class'] for c in cc)),
            'previous_class_unique_views': dict(collections.Counter(c['previous_class'] for c in unique_cc)),
            'same_peer_unique_views': sum(c['same_peer_as_previous'] for c in unique_cc),
            'same_peer_rows': sum(c['same_peer_as_previous'] for c in cc),
            'events_all_candidates_samepeer': sum(bool(r['candidate_details']) and all(c['same_peer_as_previous'] for c in r['candidate_details']) for r in rr),
            'events_any_candidate_samepeer': sum(any(c['same_peer_as_previous'] for c in r['candidate_details']) for r in rr),
            'prefix_count_range': [min((c['prefix_count'] for c in cc), default=None), max((c['prefix_count'] for c in cc), default=None)],
            'prefix_span_seconds_range': [min((c['prefix_span_seconds'] for c in cc), default=None), max((c['prefix_span_seconds'] for c in cc), default=None)]}
    summary = {'model_fit_calls': 0, 'model_score_calls': 0, 'threshold_calls': 0,
        'event_count': len(result), 'candidate_context_rows': len(candidate_rows),
        'groups': {f'{run}/{kind}': summarize([r for r in result if r['run'] == run and r['kind'] == kind])
            for run, kind in sorted({(r['run'], r['kind']) for r in result})},
        'optc_normal_objects_deduplicated_parent_summary': summarize([r for r in result if r['kind'] in ('webpage_image_download', 'webpage_stylesheet_download')]),
        'interpretation': ['AIT query uses offline Zeek export network_time, not real ingestion.',
            'OpTC query is offline flow-end replay, not observed exporter availability.',
            'Prior actor-completed workflow events are not causal edges.',
            'Unknown prefix flows cannot be called benign or unrelated.',
            'Candidate rows are not independent events; all candidate bounds retained.']}
    dump('summary.json', summary)
    dump('attack_download_details.json', [r for r in result if r['kind'] in ('attack_download', 'malicious_tool_download', 'repeated_tool_transfer')])
    dump('source_receipt.json', {'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'output_events': len(result),
        'no_model_imports': True})
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
