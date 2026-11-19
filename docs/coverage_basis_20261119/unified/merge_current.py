"""Materialize current evidence tables; no training/scoring or history features."""
from __future__ import annotations
import argparse
import collections
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
NOW = OUT.parent
WORK = NOW.parent
BASE = WORK / '拟合来源与划分落实_20261119/dataset'
CAL = NOW / 'normal_calibration'
MGMT = NOW / 'normal_management'
SLICE = NOW / 'attack_coverage/network_mapping'
FIELDS = ('ip_protocol', 'orig_port', 'resp_port', 'duration_seconds', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes')
REMOVED = {'features27', 'history_features', 'model_features', 'history19', 'previous_raw8_log1p', 'prefix_mean_log1p'}

def rows(path):
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', encoding='utf-8') as stream:
        for n, line in enumerate(stream, 1):
            yield n, json.loads(line)

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def stripped(value):
    if isinstance(value, dict):
        return {k: stripped(v) for k, v in value.items() if k not in REMOVED}
    if isinstance(value, list):
        return [stripped(v) for v in value]
    return value

def write(name, data):
    path = OUT / name
    # Fixed gzip timestamp permits byte-identical re-runs of the frozen inputs.
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        for item in data:
            zipped.write((json.dumps(stripped(item), ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode('utf-8'))

def document(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')

def source_ref(path, line=None):
    return {'path': str(path), **({'line': line} if line is not None else {})}

def feature8(raw):
    return [None if raw.get(k) is None or raw[k] < 0 else math.log1p(raw[k]) for k in FIELDS]

def role_of(row):
    return row.get('current_role', row.get('role', row.get('dataset_role', row.get('split_role', 'opened_source_development'))))

def source_scope(fid, row):
    if fid.startswith('ait:'):
        run = fid.split(':')[1]
        return {'source_family': 'AIT', 'source_run': run, 'source_group': f'ait:{run}:complete_run'}
    if fid.startswith('optc:'):
        return {'source_family': 'OpTC', 'source_run': fid.split(':')[1],
                'source_group': row.get('group_id'),
                'campaign_id': row.get('group_id') if any(s in row.get('group_id', '') for s in ('empire', 'meterpreter')) else None}
    if fid.startswith('unraveled:'):
        return {'source_family': 'Unraveled', 'source_run': '20210621', 'source_group': row['campaign_id'], 'campaign_id': row['campaign_id']}
    raise ValueError(f'Unknown source identity: {fid}')

def main(attack_source=None, normal_source=None):
    OUT.mkdir(parents=True, exist_ok=True)
    normal_source = Path(normal_source) if normal_source is not None else CAL / 'v2'
    used = set()
    def read(path):
        used.add(path)
        return rows(path)
    base_roles = {(r['unit'], r['id']): r for _, r in read(BASE / 'split_assignments.jsonl.gz')}
    base_labels = {r['flow_id']: {'label': r.get('label'), 'label_basis': r.get('label_basis')}
                   for _, r in read(BASE / 'inference_rows.jsonl.gz')}
    old_c = {r['flow_id']: r for _, r in read(BASE / 'model_rows.jsonl.gz') if r['dataset_role'] == 'C'}
    assert len(old_c) == 25
    observations, stable_index, aliases = {}, {}, {}
    events, links = {}, {}
    merge_counts = collections.Counter()

    def add_observation(r, path, line, origin):
        fid = r['flow_id']
        sid = r.get('stable_flow_id', fid)
        target = stable_index.get(sid, fid)
        if fid in observations:
            assert observations[fid]['stable_flow_id'] == sid, ('identity_conflict', fid)
            target = fid
        aliases[fid] = target
        raw = r['raw8']
        assert isinstance(raw, dict) and set(FIELDS).issubset(raw), ('raw8_schema', fid)
        if target in observations:
            prev = observations[target]
            assert prev['raw8'] == raw, ('raw8_identity_conflict', fid, target)
            prev['source_rows'].append(source_ref(path, line))
            merge_counts[origin + '_reused'] += 1
            return target
        stable_index[sid] = target
        old_assignment = base_roles.get(('flow', fid))
        old_role = old_assignment['role'] if old_assignment else None
        label = base_labels.get(fid, {}).get('label', r.get('label'))
        label = label if label in (0, 1) else None
        current_role = old_role or 'opened_source_development'
        if current_role in ('C', 'C_background'):
            current_role = 'opened_source_development'
        row = stripped(copy.deepcopy(r))
        historical_state = {k: row.pop(k) for k in ('role', 'dataset_role', 'split_role', 'flow_label', 'supervised', 'training_eligible', 'training_admission') if k in row}
        historical_state['registered_assignment'] = old_assignment
        historical_state['label'] = label
        historical_state['label_basis'] = base_labels.get(fid, {}).get('label_basis', r.get('label_basis'))
        row.update(flow_id=target, stable_flow_id=sid, label=label, features8=feature8(raw),
                   current_role=current_role, historical_roles=[] if old_role is None else [old_role],
                   historical_state=historical_state, source_rows=[source_ref(path, line)],
                   new_batch_registered=False, training_admission=False, calibration_admission=False,
                   source_dataset=origin)
        row['source_address'] = r.get('source_address', r.get('source'))
        row['destination_address'] = r.get('destination_address', r.get('destination'))
        row['observed_end'] = r.get('observed_end', r.get('zeek_reported_end'))
        row.update(source_scope(fid, r))
        row['actual_available_at'] = r.get('actual_detector_received_time', r.get('detector_received_time'))
        row['clock_basis'] = r.get('availability_basis') or ('offline_Zeek_export_network_time_not_detector_receipt' if r.get('replay_export_time') is not None else 'publisher_record_actual_export_time_unknown')
        observations[target] = row
        merge_counts[origin + '_added'] += 1
        return target

    for line, r in read(BASE / 'observations.jsonl.gz'):
        add_observation(r, BASE / 'observations.jsonl.gz', line, 'base')
    for line, r in read(CAL / 'extracted/intranet_server/connections.jsonl.gz'):
        add_observation(r, CAL / 'extracted/intranet_server/connections.jsonl.gz', line, 'normal_calibration_capture')

    def event_role(r):
        prior = base_roles.get(('event', r['event_id']))
        current = prior['role'] if prior else role_of(r)
        return 'historical_C25_evidence' if current == 'C' else current

    def add_event(r, path, line, current_role=None):
        eid = r['event_id']
        assert eid not in events, ('duplicate_event', eid)
        row = stripped(copy.deepcopy(r))
        prior = base_roles.get(('event', eid))
        row['historical_roles'] = [] if prior is None else [prior['role']]
        row['historical_role_fields'] = {k: row.pop(k) for k in ('role', 'dataset_role', 'split_role') if k in row}
        row['current_role'] = current_role or event_role(r)
        row['new_batch_registered'] = False
        row['training_admission'] = False
        row['source_rows'] = [source_ref(path, line)]
        events[eid] = row

    def add_link(r, path, line):
        row = stripped(copy.deepcopy(r))
        row['flow_id'] = aliases.get(row['flow_id'], row['flow_id'])
        key = (row['event_id'], row['flow_id'])
        if key in links:
            links[key]['additional_source_rows'].append(source_ref(path, line))
            links[key]['additional_relationship_evidence'].append(row)
            merge_counts['duplicate_link_evidence_collapsed'] += 1
        else:
            row['source_rows'] = [source_ref(path, line)]
            row['additional_source_rows'] = []
            row['additional_relationship_evidence'] = []
            links[key] = row

    for line, r in read(BASE / 'events.jsonl.gz'):
        add_event(r, BASE / 'events.jsonl.gz', line)
    for line, r in read(BASE / 'event_flow_links.jsonl.gz'):
        add_link(r, BASE / 'event_flow_links.jsonl.gz', line)

    for line, r in read(normal_source / 'business_episodes.jsonl.gz'):
        row = {**r, 'event_id': r['episode_id'], 'kind': 'normal_business_episode', 'unit': 'closed_actor_business_episode',
               'group_id': 'ait:russellmitchell:complete_run', 'source_claim_label': 0,
               'label': r.get('label'), 'label_propagates_to_all_linked_flows': False}
        add_event(row, normal_source / 'business_episodes.jsonl.gz', line, r['role'])
    service_candidates = []
    qualified_ids = set()
    for line, r in read(normal_source / 'connection_candidates.jsonl.gz'):
        fid = aliases.get(r['flow_id'], r['flow_id'])
        assert fid in observations, ('missing_calibration_observation', fid)
        o = observations[fid]
        assert o['stable_flow_id'] == r['stable_flow_id'] and o['raw8'] == r['raw8']
        qualified = r['label_evidence_qualified']
        assert qualified == (r['label'] == 0)
        o['label'] = 0 if qualified else None
        o['current_role'] = r['role']
        o['normal_label_evidence_qualified'] = qualified
        o['normal_input_quality'] = r['input_quality']
        o['normal_model_input_ready'] = r['model_input_ready']
        o['normal_calibration_candidate'] = r['calibration_candidate']
        o['normal_identity_evidence'] = {k: r.get(k) for k in ('business_stratum', 'episode_ids', 'service_request_ids', 'possible_frontend_request_ids', 'possible_actor_ids', 'dependency_component_id', 'label_evidence', 'label_evidence_rejection_reasons', 'label_basis', 'role_basis', 'flow_mapping_cardinality')}
        o['label_basis'] = 'normal_calibration_v2_whole_observation_interval_normal_requests' if qualified else None
        o['source_rows'].append(source_ref(normal_source / 'connection_candidates.jsonl.gz', line))
        if qualified:
            qualified_ids.add(fid)
        candidate = stripped(copy.deepcopy(r))
        candidate['flow_id'] = fid
        candidate['current_role'] = candidate.pop('role')
        candidate['new_batch_registered'] = False
        service_candidates.append(candidate)
    normal_link_count = 0
    for line, r in read(normal_source / 'episode_flow_links.jsonl.gz'):
        add_link(r, normal_source / 'episode_flow_links.jsonl.gz', line)
        normal_link_count += 1
    assert len(service_candidates) == 16374

    # Original C25 is an historical registration, never a second current sample set.
    old_c_history = []
    for fid, original in sorted(old_c.items()):
        o = observations[aliases[fid]]
        o['historical_C25_member'] = True
        old_c_history.append({'flow_id': o['flow_id'], 'stable_flow_id': o['stable_flow_id'],
            'historical_role': 'C', 'historical_label': 0, 'current_role': o['current_role'],
            'current_label': o['label'], 'current_normal_label_evidence_qualified': o.get('normal_label_evidence_qualified', False),
            'source_reference': source_ref(BASE / 'model_rows.jsonl.gz'), 'current_sample_copy': False})

    management_rows = []
    for line, r in read(MGMT / 'referenced_connections.jsonl.gz'):
        fid = add_observation(r, MGMT / 'referenced_connections.jsonl.gz', line, 'normal_management')
        o = observations[fid]
        o['normal_management_evidence'] = {k: r.get(k) for k in ('normal_management_event_ids', 'normal_management_candidate_label', 'complete_normal_control_candidate_session_ids')}
        o['current_role'] = 'opened_source_development'
        if r.get('normal_management_candidate_label') == 0:
            assert o['label'] in (0, None)
            o['label'] = 0
            o['label_basis'] = 'normal_actor_direct_SSH_complete_connection_candidate'
            o['normal_management_whole_connection_candidate'] = True
        # Other six candidate connections acquire no normal label from command links.
        management_rows.append({'flow_id': fid, 'stable_flow_id': o['stable_flow_id'],
            'source_registered_roles': o['historical_roles'], 'current_role': o['current_role'],
            'candidate_normal_label': r.get('normal_management_candidate_label'), 'current_label': o['label'],
            'training_admission': False, 'source_rows': o['source_rows'][-1:]})
    assert len(management_rows) == 7 and sum(r['candidate_normal_label'] == 0 for r in management_rows) == 1
    for line, r in read(MGMT / 'events.jsonl.gz'):
        add_event(r, MGMT / 'events.jsonl.gz', line, 'opened_source_development')
    for line, r in read(MGMT / 'event_flow_links.jsonl.gz'):
        add_link(r, MGMT / 'event_flow_links.jsonl.gz', line)

    # Full-capture intake is independent from old slice history: never concatenate both.
    attack_info = {'state': 'pending_complete_capture_intake', 'canonical_slice_rows_added': 0}
    slice_rows = [stripped(r) for _, r in read(SLICE / 'connections.jsonl')]
    slice_aliases = []
    if attack_source is not None:
        used.add(Path(attack_source) / 'assembly_receipt.json')
        attack_info = ingest_full_attack(Path(attack_source), read, add_observation, add_event, add_link, observations, slice_rows, slice_aliases)
    else:
        slice_aliases = [{'old_slice_flow_id': r['flow_id'], 'current_flow_id': None,
            'status': 'pending_full_capture', 'historical_label': r.get('label'),
            'included_as_current_observation': False} for r in slice_rows]

    # Build explicit current role assignments for all main-table entities.
    assignments = []
    for unit, rr, id_key in (('flow', observations, 'flow_id'), ('event', events, 'event_id')):
        for rid, row in rr.items():
            assignments.append({'unit': unit, 'id': rid, 'current_role': row['current_role'],
                'historical_roles': row['historical_roles'],
                'group_id': row.get('campaign_id', row.get('group_id', row.get('scenario'))),
                'new_batch_registered': False})
    missing_events = [eid for eid, fid in links if eid not in events]
    missing_flows = [fid for eid, fid in links if fid not in observations]
    assert not missing_events and not missing_flows, ('dangling_links', missing_events[:3], missing_flows[:3])
    assert len(stable_index) == len(observations)
    qualified_roles = collections.Counter(observations[f]['current_role'] for f in qualified_ids)
    assert all(observations[r['flow_id']]['label'] is None for r in service_candidates if not r['label_evidence_qualified'])
    assert all(r['current_label'] == 0 for r in old_c_history), 'Old C25 normal evidence must not be downgraded by SF/episode-closure gates.'
    assert all(row['current_role'] != 'C' for row in observations.values())
    assert all(not row['new_batch_registered'] for row in observations.values())
    assert not any(len(row['features8']) != 8 for row in observations.values())

    write('observations.jsonl.gz', (observations[k] for k in sorted(observations)))
    write('events.jsonl.gz', (events[k] for k in sorted(events)))
    write('event_flow_links.jsonl.gz', (links[k] for k in sorted(links)))
    write('split_assignments.jsonl.gz', sorted(assignments, key=lambda r: (r['unit'], r['id'])))
    write('flow_rows.jsonl.gz', ({k: r.get(k) for k in ('flow_id', 'stable_flow_id', 'raw8', 'features8',
        'label', 'label_basis', 'current_role', 'historical_roles', 'source_dataset', 'scenario', 'capture_host',
        'group_id', 'campaign_id', 'source_family', 'source_run', 'source_group',
        'connection_start', 'observed_end', 'replay_export_time', 'actual_available_at',
        'clock_basis', 'new_batch_registered', 'training_admission', 'calibration_admission')}
        for r in (observations[k] for k in sorted(observations))))
    write('service_connection_candidates.jsonl.gz', sorted(service_candidates, key=lambda r: r['flow_id']))
    write('normal_identity_qualified.jsonl.gz', ({'flow_id': f, 'stable_flow_id': observations[f]['stable_flow_id'],
        'current_role': observations[f]['current_role'], 'label': 0, 'evidence': observations[f]['normal_identity_evidence'],
        'input_quality': observations[f]['normal_input_quality'],
        'model_input_ready': observations[f]['normal_model_input_ready'],
        'calibration_candidate': observations[f]['normal_calibration_candidate'],
        'new_batch_registered': False} for f in sorted(qualified_ids)))
    write('historical_normal_dependency_components.jsonl.gz', (r for _, r in read(CAL / 'dependency_components.jsonl.gz')))
    write('historical_normal_v1_assignments.jsonl.gz', ({'flow_id': r['flow_id'], 'stable_flow_id': r['stable_flow_id'],
        'historical_role': r['role'], 'historical_label': r['label'], 'historical_source': str(CAL),
        'current_sample_copy': False} for _, r in read(CAL / 'qualified_connections.jsonl.gz')))
    write('historical_C25.jsonl.gz', old_c_history)
    write('normal_management_candidates.jsonl.gz', management_rows)
    write('historical_attack_slice_records.jsonl.gz', slice_rows)
    write('historical_attack_slice_aliases.jsonl.gz', slice_aliases)
    summary = {'schema': 'current-evidence-unified-20261119-v1', 'events': len(events), 'observations': len(observations),
        'event_flow_links': len(links), 'split_assignments': len(assignments), 'flow_rows': len(observations),
        'normal_business_episodes_added': 255, 'service_connection_candidates': len(service_candidates),
        'service_candidates_without_episode': sum(not r['episode_ids'] for r in service_candidates),
        'normal_episode_flow_links_added': normal_link_count,
        'normal_label_evidence_qualified': len(qualified_ids), 'qualified_role_counts': dict(qualified_roles),
        'normal_model_input_ready': sum(observations[f]['normal_model_input_ready'] for f in qualified_ids),
        'normal_calibration_candidates': sum(observations[f]['normal_calibration_candidate'] for f in qualified_ids),
        'normal_management_events_added': 58, 'normal_management_candidate_connections_reused': 7,
        'normal_management_complete_normal_candidates': 1,
        'historical_C25_current_states': dict(collections.Counter(f"{r['current_role']}|normal_label_evidence={r['current_normal_label_evidence_qualified']}" for r in old_c_history)),
        'attack_intake': attack_info, 'merge_counts': dict(merge_counts),
        'current_role_counts': dict(collections.Counter(r['current_role'] for r in observations.values())),
        'checks': {'all_links_resolve': True, 'unique_stable_observations': True,
            'features8_width': 8, 'features27_exported': False, 'unqualified_normal_candidates_remain_null': True,
            'old_C25_not_duplicated': True, 'current_adopted_C_rows': 0, 'new_training_batch_registered': False},
        'training_calls': 0, 'model_score_calls': 0, 'threshold_calls': 0}
    document('summary.json', summary)
    (OUT / 'README.md').write_text(
        '# 当前统一证据表\n\n'
        f"已实际合并 {len(events):,} 个事件、{len(observations):,} 个唯一观察身份和 {len(links):,} 条事件—流关系。"
        '本表保存当前事实与候选分配，不注册新的训练、选模或校准批次。\n\n'
        '主表为 `events.jsonl.gz`、`observations.jsonl.gz`、`event_flow_links.jsonl.gz`、`split_assignments.jsonl.gz`。'
        '`flow_rows.jsonl.gz` 是 observations 的同身份精简视图，只提供 raw8/features8；两表不能相加计数。'
        '没有导出 features27 或历史摘要。\n\n'
        f"正常业务保留全部 255 个 episode、{len(service_candidates):,} 个服务连接候选；"
        f"其中 {normal_link_count:,} 条实际 episode—流边进入主 links。"
        '没有 episode 的候选仍在 `service_connection_candidates.jsonl.gz`，没有虚构事件。\n\n'
        f"正常标签证据成立的连接为 {len(qualified_ids):,} 条；观察质量、model_input_ready、calibration_candidate 与 label 分列。"
        'SF 状态和 episode 显式闭合不是正常标签的必要条件。无正常标签证据的候选保持 null；正常事件不会自动给所有关联流贴标签。\n\n'
        '旧 C25 的原角色和标签保存在 `historical_C25.jsonl.gz`，当前观察不复制。'
        '旧 v1 的 365/618/270 分配保存在 `historical_normal_v1_assignments.jsonl.gz`；当前正常分配取 v2。'
        'current_role 与 historical_roles 分开；C_candidate 类角色是候选身份，不等于已采用校准批次。\n\n'
        'SSH 接入 58 个事件、18 条关系，7 个引用连接全部复用基底身份；仅 1 条新增完整正常控制候选标签，来源为 fox。'
        '其余命令尝试、未结束会话与候选关系保留，未扩展正常整流标签。新 SSH 材料当前为 opened_source_development，保留原来源 F/V 关联。\n\n'
        f"Unraveled 当前接入状态：`{attack_info['state']}`。"
        '旧两片段只保存在 historical_attack_slice_records/aliases 和 exposure 引用中，不与整场观察相加。'
        '整场接入时整个 campaign 使用 opened_source_development，不能拆成 F/V。\n\n'
        '所有主观察标记 new_batch_registered=false、training_admission=false、calibration_admission=false；'
        '这表示本次合并没有创建实验批次，不否认已成立的开发标签。AIT/Unraveled 的 replay_export_time 是离线 Zeek 时钟；'
        '实际 detector 接收时刻没有被补造。\n\n'
        '重建基底及正常/SSH：`python merge_current.py`。完整攻击来源冻结后：'
        '`python merge_current.py --attack-source ../attack_coverage/full_capture`。'
        '脚本按源稳定身份核对 raw8、合并证据和引用；SHA-256 见 source_receipt/output_receipt。拟合、评分、阈值调用均为 0。\n',
        encoding='utf-8')
    source_hashes = [{'path': str(p), 'sha256': sha(p)} for p in sorted(used)]
    document('source_receipt.json', {'script_sha256': sha(Path(__file__)), 'sources': source_hashes,
        'argv_attack_source': None if attack_source is None else str(attack_source), 'normal_source': str(normal_source)})
    document('historical_references.json', {'former_history19_tables_not_consumed': [str(BASE / 'context_rows.jsonl.gz'), str(CAL / 'context_rows.jsonl.gz')],
        'old_fit_dataset_preserved': str(BASE), 'historical_attack_slice_source': str(SLICE),
        'old_slice_exposure': {'campaign_id': 'unraveled:apt_2021_single_campaign', 'observations': len(slice_rows),
            'label_counts': dict(collections.Counter(str(r.get('label')) for r in slice_rows)),
            'normal_denominator_complete': False, 'current_samples': False}})
    document('output_receipt.json', {'outputs': [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
        for p in sorted(OUT.glob('*.jsonl.gz'))]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def ingest_full_attack(directory, read, add_observation, add_event, add_link, observations, slice_rows, slice_aliases):
    """One complete campaign in opened development; input full source must be frozen."""
    receipt_path = directory / 'assembly_receipt.json'
    assert receipt_path.exists(), 'Full-capture assembly receipt is required; never ingest in-progress output.'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    source = directory / 'connections.jsonl.gz'
    count = 0
    for line, original in read(source):
        r = copy.deepcopy(original)
        assert r.get('stable_flow_id'), 'Full-source stable_flow_id required; do not replace it with random Zeek UID.'
        fid = add_observation(r, source, line, 'unraveled_full_capture')
        o = observations[fid]
        o['current_role'] = 'opened_source_development'
        o['label'] = r.get('label')
        o['campaign_id'] = r['campaign_id']
        o['historical_roles'] = []
        o['label_basis'] = r.get('label_basis')
        count += 1
    for filename, add in (('events.jsonl', add_event), ('links.jsonl', add_link)):
        for line, r in read(directory / filename):
            if add is add_event:
                add(r, directory / filename, line, 'opened_source_development')
            else:
                add(r, directory / filename, line)
    old_labels = {r['flow_id']: r.get('label') for r in slice_rows}
    for _, relation in read(directory / 'old_slice_aliases.jsonl'):
        candidates = relation['full_flow_ids']
        assert all(fid in observations for fid in candidates)
        slice_aliases.append({**relation, 'old_slice_flow_id': relation['old_flow_id'],
            'current_flow_id': candidates[0] if len(candidates) == 1 else None,
            'historical_label': old_labels[relation['old_flow_id']],
            'included_as_second_current_observation': False})
    assert len(slice_aliases) == len(slice_rows)
    assert len({o['campaign_id'] for o in observations.values() if o['source_dataset'] == 'unraveled_full_capture'}) == 1
    return {'state': 'full_capture_ingested', 'directory': str(directory), 'connections': count,
        'campaign_role': 'opened_source_development', 'canonical_slice_rows_added': 0,
        'historical_slice_alias_status': dict(collections.Counter(r['status'] for r in slice_aliases)),
        'source_receipt': receipt}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack-source', type=Path, default=None,
        help='Frozen complete-capture directory, never old network_mapping slices')
    parser.add_argument('--normal-source', type=Path, default=CAL / 'v2')
    args = parser.parse_args()
    main(args.attack_source, args.normal_source)
