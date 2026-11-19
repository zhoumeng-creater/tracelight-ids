"""Bind complete capture backgrounds, qualified flow labels and fixed F/V/C roles.

This program exports data and deterministic context only. It never imports a
model, fits, scores, or selects a threshold. Historical v1 rows remain unchanged;
new measurement rows and explicit aliases extend that table.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'docs/unified_events_20261119/split_method'))
from prefix_context import RAW8_FIELDS, build_prefix_context  # noqa: E402

CAPTURES = [('santos', 'intranet_server'), ('santos', 'cloud_share'),
            ('fox', 'intranet_server'), ('fox', 'cloud_share'),
            ('russellmitchell', 'cloud_share')]
ROLES = {'santos': 'F', 'fox': 'V', 'russellmitchell': 'C_background'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8-sig')) as f:
        yield from (json.loads(line) for line in f if line.strip())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def write_rows(path, values):
    with Path(path).open('wb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as z:
        with io.TextIOWrapper(z, encoding='utf-8', newline='\n') as f:
            for row in values:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def old_flow_key(row):
    return (row['native_protocol'], row['source'], row['orig_port'],
            row['destination'], row['resp_port'], row['connection_start'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    work = args.workspace.resolve()
    now = work / '拟合来源与划分落实_20261119'
    old = work / '统一事件与方法落实_20261119'
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def track(path):
        path = Path(path)
        inputs[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return path

    inventory = load(track(now / 'ait_new_runs/capture_host_inventory.json'))
    hosts = {(r['run'], r['capture_host']): r for r in inventory['hosts']}
    c_facts_path = old / 'ait/source/capture_config/gather/cloud_share/facts.json'
    c_facts = load(track(c_facts_path))['ansible_facts']
    c_candidates = load(track(now / 'calibration/candidate_connections.json'))
    selected_c = [r for r in c_candidates['candidates'] if r['eligible']]
    assert len(selected_c) == 25
    c_addresses = {r['model_flow']['destination'] for r in selected_c}
    assert c_addresses == {'172.19.130.106'}
    assert c_addresses <= set(c_facts['all_ipv4_addresses'])
    hosts[('russellmitchell', 'cloud_share')] = {
        'host_id': 'ait:russellmitchell:cloud_share',
        'all_local_addresses': sorted(set(c_facts['all_ipv4_addresses'] +
                                         c_facts['all_ipv6_addresses'] + ['127.0.0.1', '::1'])),
        'address_basis': 'original cloud facts ansible_facts local interface addresses',
    }

    mapped = {}
    new_events, new_links = [], []
    for run in ['santos', 'fox']:
        directory = now / 'ait_new_runs/network_mapping' / run
        for row in rows(track(directory / 'connections.jsonl.gz')):
            assert row['flow_id'] not in mapped
            assert row['label'] in (0, 1, None)
            mapped[row['flow_id']] = row
        for row in rows(track(directory / 'events.jsonl')):
            row = dict(row, dataset_role=ROLES[run], source_run=run,
                       group_id=f'ait:{run}:complete_run', split_role=ROLES[run], source='AIT',
                       label='malicious' if row['action_label'] == 1 else 'normal', unit='business_action')
            interval = row['mapping_parameters'].get('actor_interval', {})
            row.update(start=interval.get('start'), end=interval.get('end'),
                       evidence_grade='actor_action_and_published_service_evidence',
                       label_propagates_to_all_linked_flows=False)
            new_events.append(row)
        new_links.extend(rows(track(directory / 'links.jsonl')))
    assert len(new_events) == 254
    assert len({r['event_id'] for r in new_events}) == len(new_events)

    observations, context_inputs, capture_counts, excluded = [], [], [], []
    observations_by_id = {}
    c_index = defaultdict(list)
    for run, sensor in CAPTURES:
        directory = now / 'ait_extracted' / run / sensor
        receipt = load(track(directory / 'extraction_receipt.json'))
        path = track(directory / 'connections.jsonl.gz')
        assert sha(path) == receipt['outputs']['connections.jsonl.gz']['sha256']
        host = hosts[(run, sensor)]
        addresses = set(host['all_local_addresses'])
        count = Counter()
        for original in rows(path):
            count['all_capture_records'] += 1
            row = dict(original)
            fid = row['flow_id']
            assert fid not in observations_by_id
            source, destination = row['source_address'], row['destination_address']
            local_source, local_destination = source in addresses, destination in addresses
            row.update(dataset_role=ROLES[run], label=None, label_basis=None,
                       supervised=False, monitored_host=host['host_id'], dataset_source='AIT',
                       group_id=f'ait:{run}:complete_run', split_role=ROLES[run])
            if fid in mapped:
                overlay = mapped[fid]
                assert overlay['uid'] == row['uid']
                if 'raw8' in overlay:
                    assert overlay['raw8'] == row['raw8']
                row.update({key: overlay.get(key) for key in
                            ('label', 'label_basis', 'evidence_grade', 'event_ids', 'qualification_reasons')})
            if run == 'russellmitchell':
                c_index[old_flow_key(row)].append(row)
            if not (local_source or local_destination):
                count['nonlocal_records_without_host_view'] += 1
                excluded.append({'flow_id': fid, 'reason': 'neither_endpoint_is_declared_capture_host_address'})
            else:
                count['host_view_records'] += 1
                raw8 = [row['raw8'][key] for key in RAW8_FIELDS]
                # Zeek's absent duration sentinel is a missing observation, not
                # a negative measurement or an imputed zero.
                null_fields = [key for key, value in zip(RAW8_FIELDS, raw8)
                               if value is not None and value < 0]
                raw8 = [None if value is not None and value < 0 else value for value in raw8]
                context_inputs.append({
                    'group': f'ait:{run}:{sensor}:complete_capture', 'host': host['host_id'],
                    'peer': destination if local_source else source,
                    'flow_id': row['stable_flow_id'], 'physical_flow_id': row['stable_flow_id'],
                    'measurement_flow_id': fid,
                    'available_at': row['replay_export_time'], 'flow_end_at': row['observed_end'],
                    'raw8': raw8, 'original_negative_missing_fields': null_fields,
                    'clock_basis': 'fixed Zeek offline export network_time; not actual detector ingestion',
                })
                row['supervised'] = row['label'] in (0, 1)
            observations.append(row)
            observations_by_id[fid] = row
        capture_counts.append({'run': run, 'capture_host': sensor, **dict(count)})
    assert set(mapped) <= set(observations_by_id)

    aliases, c_events = [], set()
    for candidate in selected_c:
        original = candidate['model_flow']
        matches = c_index[old_flow_key(original)]
        assert len(matches) == 1, (original['flow_id'], len(matches))
        current = matches[0]
        assert current['raw8'] == {key: original[key] for key in RAW8_FIELDS}
        current.update(dataset_role='C', label=0, supervised=True,
                       split_role='C',
                       label_basis=candidate['evidence_grade'], evidence_grade=candidate['evidence_grade'],
                       event_ids=candidate['event_ids'], qualification_reasons=[])
        c_events.update(candidate['event_ids'])
        aliases.append({'old_flow_id': original['flow_id'], 'new_flow_id': current['flow_id'],
                        'stable_flow_id': current['stable_flow_id'],
                        'basis': 'same original capture SHA, ordered five-tuple and exact start; equal raw8',
                        'packet_connection_id': candidate['packet_connection_id'],
                        'event_ids': candidate['event_ids']})
        for eid in candidate['event_ids']:
            new_links.append({'event_id': eid, 'flow_id': current['flow_id'],
                              'basis': 'existing action candidates plus full-envelope normal actor qualification',
                              'normal_whole_flow_qualified': True, 'unique_action_mapping_claimed': False})

    context_start = time.perf_counter()
    context = build_prefix_context(context_inputs, time_mode='available_at')
    context_seconds = time.perf_counter() - context_start
    assert context['audit']['all_supplied_rows_accounted_for']
    assert context['audit']['strictly_past_history']
    context_rows = context.pop('rows')
    stable_to_measurement = {r['stable_flow_id']: r['flow_id'] for r in observations}
    assert len(stable_to_measurement) == len(observations)
    # Zeek UIDs can change on re-export. Same-clock ordering is therefore based
    # on source SHA + tuple + start, and only then rebound to displayed UIDs.
    for view in context_rows:
        view['ordering_flow_id'] = view['flow_id']
        view['flow_id'] = stable_to_measurement[view['flow_id']]
        if view['previous_flow_id'] is not None:
            view['previous_flow_id'] = stable_to_measurement[view['previous_flow_id']]
    by_context_id = {r['flow_id']: r for r in context_rows}
    inference_rows, model_rows = [], []
    requested_ids = set(mapped) | {r['new_flow_id'] for r in aliases}
    for row in observations:
        if row['flow_id'] not in requested_ids:
            continue
        assert row['flow_id'] in by_context_id
        view = by_context_id[row['flow_id']]
        assert row['dataset_role'] in ('F', 'V', 'C')
        prepared = {key: row.get(key) for key in
                           ('flow_id', 'stable_flow_id', 'flow_key_id', 'scenario', 'capture_host',
                            'dataset_role', 'label', 'label_basis', 'evidence_grade', 'event_ids',
                            'measurement_contract', 'capture_provenance')} | {
            'context_group': view['group'], 'monitored_host': view['host'],
            'features8': view['current_raw8_log1p'], 'features27': view['model_features'],
            'replay_export_time': view['effective_available_at'],
            'context_previous_flow_id': view['previous_flow_id'],
            'history_flow_count': view['history_flow_count'],
        }
        inference_rows.append(prepared)
        if row['supervised']:
            model_rows.append(prepared)
    role_classes = Counter((r['dataset_role'], r['label']) for r in model_rows)
    assert role_classes[('C', 0)] == 25 and role_classes[('C', 1)] == 0
    for role in ['F', 'V']:
        assert role_classes[(role, 0)] > 0 and role_classes[(role, 1)] > 0, (role, role_classes)
    identities = defaultdict(set)
    for row in model_rows:
        identities[row['dataset_role']].add(row['stable_flow_id'])
    assert not (identities['F'] & identities['V'] or identities['F'] & identities['C'] or identities['V'] & identities['C'])

    historical_events = list(rows(track(old / 'unified/events.jsonl.gz')))
    historical_observations = list(rows(track(old / 'unified/observations.jsonl.gz')))
    historical_links = list(rows(track(old / 'unified/event_flow_links.jsonl.gz')))
    # Keep one measurement in the current table. The complete historical AIT
    # table remains at the frozen v1 path, rather than coexisting under new UIDs.
    old_events = [r for r in historical_events if r['source'] == 'OpTC' or r['event_id'] in c_events]
    old_events = [dict(r, dataset_role='C_evidence', split_role='C_evidence',
                      historical_v1_event_id=r['event_id'],
                      historical_v1_linked_flow_count=r.get('linked_flow_count'))
                  if r['event_id'] in c_events else r for r in old_events]
    old_observations = [r for r in historical_observations if r['flow_id'].startswith('optc:')]
    old_links = [r for r in historical_links if r['flow_id'].startswith('optc:')]
    old_ids = {r['event_id'] for r in old_events}
    assert c_events <= old_ids
    assert not (old_ids & {r['event_id'] for r in new_events})
    all_events = old_events + new_events
    all_observations = old_observations + observations
    all_links = old_links + new_links
    current_link_counts = Counter(r['event_id'] for r in all_links)
    for event in all_events:
        event['linked_flow_count'] = current_link_counts[event['event_id']]
    event_ids = {r['event_id'] for r in all_events}
    flow_ids = {r['flow_id'] for r in all_observations}
    assert len(flow_ids) == len(all_observations)
    assert all(r['event_id'] in event_ids and r['flow_id'] in flow_ids for r in all_links)

    external_context = {r['physical_flow_id']: r for r in rows(track(old / 'split_method/context_rows.jsonl.gz'))}
    external_linked = defaultdict(list)
    for link in old_links:
        external_linked[link['flow_id']].append(link['event_id'])
    for row in old_observations:
        fid = row['flow_id']
        if fid not in external_context:
            continue  # Includes the reserved Meterpreter branch: never score it here.
        view = external_context[fid]
        label = row.get('flow_label')
        binary = 0 if label == 'normal' else 1 if label in ('malicious_download', 'known_control_transport') else None
        inference_rows.append({
            'flow_id': fid, 'stable_flow_id': fid, 'scenario': row['group_id'],
            'capture_host': view['host'], 'dataset_role': 'external_development',
            'label': binary, 'label_basis': row.get('flow_label_grade', label),
            'event_ids': external_linked[fid], 'context_group': view['group'],
            'monitored_host': view['host'], 'features8': view['current_raw8_log1p'],
            'features27': view['model_features'], 'replay_export_time': view['effective_available_at'],
            'time_mode': view['time_mode'], 'history_flow_count': view['history_flow_count'],
            'context_previous_flow_id': view['previous_flow_id'],
        })
    assert len({r['flow_id'] for r in inference_rows}) == len(inference_rows)
    event_metadata = []
    for event in new_events:
        event_metadata.append({
            'event_id': event['event_id'], 'label': event['action_label'], 'kind': event['kind'],
            'role': event['dataset_role'], 'chain_id': event['chain_id'],
            'candidate_flow_ids': event.get('evaluation_candidate_flow_ids', event['candidate_flow_ids']),
            'association_mode': 'one_of',
            'candidate_set_complete': event['mapping_status'] in ('unique_candidate', 'multiple_candidates'),
        })
    external_events = {r['event_id']: r for r in old_events if r['source'] == 'OpTC'}
    external_candidates = defaultdict(set)
    for fid, eids in external_linked.items():
        if fid in external_context:
            for eid in eids:
                external_candidates[eid].add(fid)
    for eid, fids in sorted(external_candidates.items()):
        event = external_events[eid]
        if event['label'] not in ('normal', 'malicious'):
            continue
        event_metadata.append({
            'event_id': eid, 'label': 0 if event['label'] == 'normal' else 1 if event['label'] == 'malicious' else None,
            'kind': event['kind'], 'role': 'external_development', 'chain_id': event['group_id'],
            'candidate_flow_ids': sorted(fids), 'association_mode': 'all_confirmed',
            'candidate_set_complete': True,
        })
    normal_units = []
    for role in ['F', 'V', 'C']:
        for sensor in ['intranet_server', 'cloud_share']:
            ids = sorted(r['flow_id'] for r in model_rows if r['dataset_role'] == role and
                         r['label'] == 0 and r['capture_host'] == sensor)
            if ids:
                normal_units.append({'unit_id': f'{role}:{sensor}:qualified_normal_business',
                    'role': role, 'flow_ids': ids, 'normal_count': len(ids),
                    'one_percent_integer_budget': len(ids) // 100})
    for unit in rows(track(old / 'unified/evaluation_units.jsonl.gz')):
        normal_units.append(dict(unit, role='external_development', normal_count=unit['denominator'],
                                 one_percent_integer_budget=unit['fp_budget_at_one_percent']))
    write_rows(out / 'events.jsonl.gz', all_events)
    write_rows(out / 'observations.jsonl.gz', all_observations)
    write_rows(out / 'event_flow_links.jsonl.gz', all_links)
    write_rows(out / 'context_inputs.jsonl.gz', context_inputs)
    write_rows(out / 'context_rows.jsonl.gz', context_rows)
    write_rows(out / 'model_rows.jsonl.gz', model_rows)
    write_rows(out / 'inference_rows.jsonl.gz', inference_rows)
    write_rows(out / 'event_metadata.jsonl.gz', event_metadata)
    write_rows(out / 'normal_units.jsonl.gz', normal_units)
    write_rows(out / 'measurement_aliases.jsonl.gz', aliases)
    assignments = [{'unit': 'flow', 'id': r['flow_id'], 'role': r['split_role'], 'group_id': r['group_id']}
                   for r in old_observations]
    assignments += [{'unit': 'event', 'id': r['event_id'], 'role': r['split_role'], 'group_id': r['group_id']}
                    for r in old_events if r['event_id'] not in c_events]
    assignments += [{'unit': 'flow', 'id': r['flow_id'], 'stable_id': r['stable_flow_id'],
                    'role': r['dataset_role'], 'supervised': r['supervised'],
                    'source_run': r['scenario'], 'complete_capture': f"{r['scenario']}:{r['capture_host']}"}
                   for r in observations]
    assignments += [{'unit': 'event', 'id': r['event_id'], 'role': r['dataset_role'],
                     'source_run': r['source_run'], 'chain_id': r['chain_id']} for r in new_events]
    assignments += [{'unit': 'event', 'id': eid, 'role': 'C_evidence',
                     'source_run': 'russellmitchell'} for eid in sorted(c_events)]
    write_rows(out / 'split_assignments.jsonl.gz', assignments)
    save(out / 'context_audit.json', context)
    save(out / 'excluded_host_views.json', excluded)
    quality = Counter((r['scenario'], r['capture_host'], r['dataset_role'], r['label'],
                       bool(r['capture_provenance']['capture_truncated_record_candidates'])) for r in observations)
    save(out / 'quality_by_class.json', [{'run': k[0], 'capture_host': k[1], 'role': k[2],
         'label': k[3], 'truncated_record_candidate_present': k[4], 'rows': n}
         for k, n in sorted(quality.items(), key=lambda item: str(item[0]))])
    summary = {'schema': 'fit-dataset-v1', 'model_calls': 0, 'training_calls': 0,
               'role_class_counts': [{'role': k[0], 'label': k[1], 'rows': n}
                                     for k, n in sorted(role_classes.items())],
               'new_events': len(new_events), 'new_observations': len(observations),
               'unified_events': len(all_events), 'unified_observations': len(all_observations),
               'unified_links': len(all_links), 'captures': capture_counts,
               'complete_background_host_views': len(context_rows),
               'context_build_wall_seconds': context_seconds,
               'context_cost_scope': 'one offline build including sorting; not an online latency benchmark',
               'supervised_rows': len(model_rows), 'C_unique_whole_flows': len(aliases),
               'old_table_preserved_at_frozen_v1_path': True, 'F_V_C_stable_identity_overlap': 0,
               'old_AIT_measurements_excluded_from_current_table': True,
               'inference_rows': len(inference_rows), 'event_evaluation_rows': len(event_metadata),
               'normal_evaluation_units': len(normal_units),
               'AIT_clock': 'offline Zeek replay export network_time',
               'actual_online_availability_claimed': False,
               'OpTC_clock': 'existing explicit offline completed-flow replay; not silently substituted',
               'held_out_sources_opened_for_this_assembly': [],
               'C_population_one_percent_guarantee_claimed': False,
               'historical_external_development_directory': str(old / 'unified'),
               'C_empirical_one_percent_integer_budget': 0}
    save(out / 'summary.json', summary)
    save(out / 'source_receipt.json', {'inputs': inputs, 'script_sha256': sha(Path(__file__)),
          'prefix_module_sha256': sha(REPO / 'docs/unified_events_20261119/split_method/prefix_context.py')})
    save(out / 'output_receipt.json', {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                                      for p in sorted(out.iterdir()) if p.is_file() and p.name != 'output_receipt.json'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
