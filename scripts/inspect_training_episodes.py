"""Describe real F-only time slices; no fitted models, scores, or thresholds."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path


def load_f(path):
    result = []
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        for line in stream:
            # Do not deserialize calibration, development or reserved rows.
            if '"F_base"' not in line and '"F_expansion"' not in line:
                continue
            row = json.loads(line)
            role = row.get('dataset_role', row.get('role'))
            assert role in ('F_base', 'F_expansion')
            result.append(row)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def bounds(rows):
    if not rows:
        return None
    result = {}
    for key in ('connection_start', 'observed_end', 'replay_export_time'):
        vv = [r[key] for r in rows if r[key] is not None]
        result[key] = {'present': len(vv), 'missing': len(rows)-len(vv),
                       'min': min(vv) if vv else None, 'max': max(vv) if vv else None,
                       'min_utc': iso(min(vv)) if vv else None, 'max_utc': iso(max(vv)) if vv else None}
    return result


def budget(n):
    rank = math.ceil(99*(n+1)/100)
    return {'normal_count': n, 'one_percent_integer_budget': n//100,
            'one_observation_fpr': 1/n if n else None,
            'empirical_rank_one_based': rank,
            'empirical_rank_is_observed': rank <= n,
            'n_sufficient_for_alpha_01_delta_05_max_order_statistic': n >= 299}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    batch = args.repo/'docs/coverage_basis_20261119/batch'
    all_f = load_f(batch/'rows.jsonl.gz')
    rows = [r for r in all_f if r['fit_expanded_member']]
    units = load_f(batch/'normal_units.jsonl.gz')
    events = load_f(batch/'events.jsonl.gz')
    assert len(rows) == 13252
    assert Counter(r['target_kind'] for r in rows) == {'normal': 12747, 'control': 504, 'download': 1}
    assert all(r['target_label'] in (0, 1) and r['actual_available_at'] is None for r in rows)
    lookup = {r['flow_id']: r for r in rows}
    all_lookup = {r['flow_id']: r for r in all_f}
    assert len(lookup) == len(rows)
    assert len({r['stable_flow_id'] for r in rows}) == len(rows)
    normal_ids = {r['flow_id'] for r in rows if r['target_label'] == 0}
    memberships = defaultdict(list)
    unit_sets = {u['unit_id']: set(u['flow_ids']) for u in units}
    for unit in units:
        for fid in unit['flow_ids']:
            assert fid in normal_ids
            memberships[fid].append(unit['unit_id'])

    # Connected normal-action candidates must not be counted as independent
    # examples across a boundary. Unknown candidates can connect normal actions.
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    normal_actions = [e for e in events if e['role'] == 'F_base' and e['label'] == 0]
    for event in normal_actions:
        candidates = event['candidate_flow_ids']
        for fid in candidates:
            join(candidates[0], fid)
    components = defaultdict(set)
    for fid in parent:
        components[find(fid)].add(fid)
    comp_members = {}
    comp_all_members = {}
    comp_by_flow = {}
    for i, (_, members) in enumerate(sorted(components.items())):
        available = sorted(members & normal_ids)
        if not available:
            continue
        key = 'ait_normal_action_component_'+str(i)
        comp_members[key] = available
        assert all(fid in all_lookup for fid in members)
        comp_all_members[key] = sorted(members)
        for fid in available:
            comp_by_flow[fid] = key
    component_details = []
    for key, members in comp_members.items():
        aa = [e for e in normal_actions if set(e['candidate_flow_ids']) & set(members)]
        component_details.append({'component_id': key, 'fit_normal_count': len(members),
            'normal_flow_ids': members, 'event_ids': [e['event_id'] for e in aa],
            'normal_feature_time': bounds([lookup[f] for f in members]),
            'all_candidates_feature_time': bounds([all_lookup[f] for f in comp_all_members[key]]),
            'all_candidate_count': len(comp_all_members[key]),
            'one_of_candidate_relation_retained': True})

    def group_key(row):
        return comp_by_flow.get(row['flow_id'], row['stable_flow_id'])
    group_ranges = {}
    for row in rows:
        key = group_key(row); t = row['replay_export_time']
        if key not in group_ranges:
            group_ranges[key] = [t, t]
        else:
            group_ranges[key] = [min(group_ranges[key][0], t), max(group_ranges[key][1], t)]

    for key, members in comp_all_members.items():
        times = [all_lookup[f]['replay_export_time'] for f in members]
        group_ranges[key] = [min(times),max(times)]

    def composition(selected):
        # The existing Unraveled strata overlap. Their full membership tuple
        # makes a disjoint bookkeeping partition without inventing new labels.
        return Counter('|'.join(sorted(memberships[r['flow_id']])) or 'no_existing_business_stratum'
                       for r in selected if r['target_label'] == 0)
    def total_variation(a, b):
        if not sum(a.values()) or not sum(b.values()):
            return None
        return .5*sum(abs(a[k]/sum(a.values())-b[k]/sum(b.values())) for k in a.keys()|b.keys())

    sources = {}
    for source in sorted({r['source_group'] for r in rows}):
        rr = sorted([r for r in rows if r['source_group'] == source], key=lambda r:(r['replay_export_time'],r['flow_id']))
        source_normals = [r for r in rr if r['target_label'] == 0]
        source_ids = {r['flow_id'] for r in rr}
        source_units = [u for u in units if unit_sets[u['unit_id']] & source_ids]
        source_report = {'fit_rows': len(rr), 'normal_count': len(source_normals),
            'target_counts': dict(Counter(r['target_kind'] for r in rr)),
            'clock_basis': dict(Counter(r['clock_basis'] for r in rr)),
            'observed_feature_times_by_target': {kind: bounds([r for r in rr if r['target_kind'] == kind])
                for kind in sorted({r['target_kind'] for r in rr})},
            'allocation_groups': sorted({r['allocation_group'] for r in rr}),
            'all_label_and_detector_receipt_times_unavailable': True,
            'normal_business_memberships': dict(composition(source_normals)),
            'normal_rows_with_AIT_action_component': sum(r['flow_id'] in comp_by_flow for r in source_normals),
            'normal_action_partition_fully_available': all(r['flow_id'] in comp_by_flow for r in source_normals),
            'grids': {}}
        for name, width in [('utc_day', 86400), ('utc_hour', 3600)]:
            first = math.floor(rr[0]['replay_export_time']/width)
            last = math.floor(rr[-1]['replay_export_time']/width)
            bins = []
            previous = []
            prior = []
            for bin_id in range(first, last+1):
                lo, hi = bin_id*width, (bin_id+1)*width
                current = [r for r in rr if lo <= r['replay_export_time'] < hi]
                normals = [r for r in current if r['target_label'] == 0]
                prior_normals = [r for r in prior if r['target_label'] == 0]
                previous_normals = [r for r in previous if r['target_label'] == 0]
                full_q = [r for r in prior_normals if group_ranges[group_key(r)][1] < lo]
                full_e = [r for r in normals if group_ranges[group_key(r)][0] >= lo and group_ranges[group_key(r)][1] < hi]
                both = {group_key(r) for r in normals}&{group_key(r) for r in prior_normals}
                item = {'start_utc': iso(lo), 'end_utc_exclusive': iso(hi),
                    'source_clock_bin_start': lo, 'source_clock_bin_end': hi,
                    'target_counts': dict(Counter(r['target_kind'] for r in current)),
                    'flow_ids_sha256': digest(sorted(r['flow_id'] for r in current)),
                    'E': budget(len(normals)), 'Q_previous_bin': budget(len(previous_normals)),
                    'Q_all_strictly_prior': budget(len(prior_normals)),
                    'Q_prior_after_known_action_crossing_exclusion': budget(len(full_q)),
                    'E_after_known_action_crossing_exclusion': budget(len(full_e)),
                    'known_normal_action_components_shared_Q_E': len(both),
                    'normal_business_memberships': dict(composition(normals)),
                    'composition_total_variation_vs_previous_bin': total_variation(composition(previous_normals),composition(normals)),
                    'composition_total_variation_vs_prior_prefix': total_variation(composition(prior_normals),composition(normals)),
                    'business_units': {u['unit_id']: budget(sum(r['flow_id'] in unit_sets[u['unit_id']] for r in normals))
                       for u in source_units}}
                bins.append(item)
                prior.extend(current); previous = current
            signatures = defaultdict(list)
            for r in source_normals:
                raw = r['raw8']
                key = tuple(raw[k] for k in ('ip_protocol','duration_seconds','orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes'))
                signatures[key].append(r)
            repeated_cross = [v for v in signatures.values() if len({math.floor(r['replay_export_time']/width) for r in v})>1]
            shared_actions = [v for v in comp_members.values() if v and lookup[v[0]]['source_group']==source
                and len({math.floor(lookup[f]['replay_export_time']/width) for f in v})>1]
            source_report['grids'][name] = {'all_calendar_bins': len(bins),
                'nonempty_fit_bins': sum(bool(b['target_counts']) for b in bins),
                'normal_bins': sum(b['E']['normal_count']>0 for b in bins),
                'normal_bins_with_previous_bin_observed_99percent_rank': sum(b['E']['normal_count']>0 and b['Q_previous_bin']['empirical_rank_is_observed'] for b in bins),
                'normal_bins_with_prior_prefix_observed_99percent_rank': sum(b['E']['normal_count']>0 and b['Q_all_strictly_prior']['empirical_rank_is_observed'] for b in bins),
                'normal_bins_with_known_action_filter_Q_E_and_observed_99percent_rank': sum(b['E_after_known_action_crossing_exclusion']['normal_count']>0 and b['Q_prior_after_known_action_crossing_exclusion']['empirical_rank_is_observed'] for b in bins),
                'normal_bins_with_one_percent_budget_at_least_one': sum(b['E']['one_percent_integer_budget']>=1 for b in bins),
                'attack_bins_with_prior_prefix_observed_99percent_rank': sum((b['target_counts'].get('control',0)+b['target_counts'].get('download',0))>0 and b['Q_all_strictly_prior']['empirical_rank_is_observed'] for b in bins),
                'nonzero_adjacent_composition_changes': sum(b['composition_total_variation_vs_previous_bin'] is not None and b['composition_total_variation_vs_previous_bin']>0 for b in bins),
                'normal_exact_primary_input_signatures_crossing_bins': len(repeated_cross),
                'normal_rows_in_cross_bin_input_signatures': sum(map(len,repeated_cross)),
                'known_normal_action_components_crossing_bins': len(shared_actions),
                'known_normal_rows_in_cross_bin_action_components': sum(map(len,shared_actions)),
                'bins': bins}
        sources[source] = source_report

    output = {'created_on': '2026-11-20', 'scope': 'F_base/F_expansion only; fit_expanded_member true for training slices',
        'source_filtered_records': len(all_f), 'fit_records':len(rows),
        'excluded_not_fit_counts': dict(Counter(r['target_kind'] for r in all_f if not r['fit_expanded_member'])),
        'fit_base_rows': sum(r['fit_base_member'] for r in rows),
        'fit_flow_ids_sha256': digest(sorted(lookup)),
        'raw_input_files_sha256': {name: hashlib.sha256((batch/name).read_bytes()).hexdigest()
            for name in ('rows.jsonl.gz','normal_units.jsonl.gz','events.jsonl.gz')},
        'parsed_roles': ['F_base','F_expansion'], 'model_calls':0,'thresholds_selected':0,
        'definition': {'time': 'replay_export_time; offline network export clock, not actual receipt or label time',
            'bins': 'complete calendar UTC hour/day intervals; including empty intervals; no outcome-based window selection',
            'Q': 'normal rows in immediately previous interval or all strictly earlier intervals, separately described',
            'E': 'all fit rows whose feature export time lies in the interval; normal denominator separately retained',
            'component_filter': 'AIT normal action candidate graph components, using unknown candidates as graph bridges and all candidates for time bounds; exclusion counts apply only where action links are known. Unraveled rows have no per-action partition and are not certified independent by this filter.',
            'composition': 'disjoint tuple of existing normal-unit memberships; observed total variation, not proof of population drift',
            'dependency': 'all slices remain inside their original complete run/campaign training role, not independent evaluation episodes',
            'rank_resolution': 'ceil(.99*(n+1)) <= n requires n>=99; floor(n/100)>=1 requires n>=100; alpha=.01 delta=.05 NP max-order feasibility requires n>=299'},
        'existing_normal_units': [{k:v for k,v in u.items() if k!='flow_ids'} for u in units],
        'sources':sources,
        'normal_action_components':component_details,
        'normal_events':len(normal_actions),
        'normal_event_component_count':len(comp_members),
        'unraveled_business_activity_independence': 'No per-action normal activity partition is present in frozen rows; author_declared_normal_observation_set is an aggregate, not independent actor events.',
        'actual_online_label_availability_established':False,
        'not_a_training_registration':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({s:{name:{k:v for k,v in grid.items() if k!='bins'} for name,grid in info['grids'].items()} for s,info in sources.items()},ensure_ascii=False))


if __name__ == '__main__':
    main()
