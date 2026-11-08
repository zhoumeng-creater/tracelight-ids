"""Describe round-14 input coverage without fitting or loading any model.

All projections below are descriptive views of stored raw CSV columns. They do
not replace the registered 48-feature model input, and no threshold is selected.
"""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
DATA = REPO.parents[1] / '实验归档/20261108_第十四轮输入修复/核验材料/context-input-v2'
IDS = ('L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO')
SERVICE = ('PROTOCOL', 'L7_PROTO', 'L4_DST_PORT')
ARMS = ('legacy_full', 'full', 'qcoc')


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def signature(row, columns):
    values = tuple(float(row[c]) for c in columns)
    assert all(math.isfinite(x) for x in values)
    return values


def subgroup(row):
    return signature(row, SERVICE) == (6, 91, 443)


def distribution(values):
    counts = Counter(values)
    return {'n': len(values), 'unique': len(counts), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def summary(rows, projections):
    groups = Counter(r['communication_group'] for r in rows)
    return {'rows': len(rows), 'unique_communication_groups': len(groups),
            'largest_group_rows': max(groups.values(), default=0),
            'unique_patterns': {name: len({signature(r, cols) for r in rows}) for name, cols in projections.items()},
            'source_port': distribution([float(r['L4_SRC_PORT']) for r in rows]),
            'destination_port': distribution([float(r['L4_DST_PORT']) for r in rows]),
            'source_equals_destination_port_rows': sum(r['L4_SRC_PORT'] == r['L4_DST_PORT'] for r in rows),
            'destination_port_numerically_smaller_rows': sum(float(r['L4_DST_PORT']) < float(r['L4_SRC_PORT']) for r in rows)}


def projection_coverage(queries, references, projections):
    result = {}
    for name, columns in projections.items():
        q = [signature(r, columns) for r in queries]
        ref_sets = {label: {signature(r, columns) for r in rows} for label, rows in references.items()}
        result[name] = {label: sum(value in found for value in q) for label, found in ref_sets.items()}
        result[name]['in_both_P_scanning_and_P_normal'] = sum(
            value in ref_sets['P_scanning'] and value in ref_sets['P_normal'] for value in q)
        result[name]['in_neither_P_scanning_nor_P_normal'] = sum(
            value not in ref_sets['P_scanning'] and value not in ref_sets['P_normal'] for value in q)
    return result


def collision_summary(scan, normal, columns):
    sc = Counter(signature(r, columns) for r in scan)
    nc = Counter(signature(r, columns) for r in normal)
    shared = sc.keys() & nc.keys()
    return {'shared_patterns': len(shared), 'scanning_rows_in_shared_patterns': sum(sc[k] for k in shared),
            'normal_rows_in_shared_patterns': sum(nc[k] for k in shared),
            'total_error_lower_bound_if_only_this_projection_is_observable': sum(min(sc[k], nc[k]) for k in shared)}


def main():
    protocol = json.loads((DATA / 'protocol.json').read_text(encoding='utf-8'))
    features = protocol['schema']['features']
    non_id = [c for c in features if c not in IDS]
    projections = {'all48': features, 'without_source_port47': [c for c in features if c != 'L4_SRC_PORT'],
                   'non_ID44': non_id, 'all_four_IDs': IDS, 'protocol_L7_destination_port': SERVICE}
    raw = {role: read_csv(DATA / 'prepared' / name) for role, name in {
        'P': 'support.csv', 'C': 'normal_calibration.csv', 'E': 'development_evaluation.csv'}.items()}
    assert [len(raw[x]) for x in ('P', 'C', 'E')] == [8192, 5000, 20000]
    scores = read_csv(DATA / 'comparison/evaluation_scores.csv')
    assert [r['sample_id'] for r in raw['E']] == [r['row_id'] for r in scores]
    assert [r['attack_family'] for r in raw['E']] == [r['attack_family'] for r in scores]
    thresholds = {arm: json.loads((DATA / f'comparison/{arm}/threshold.json').read_text(encoding='utf-8'))['threshold'] for arm in ARMS}
    predictions = {arm: [float(r[arm + '_score']) > thresholds[arm] for r in scores] for arm in ARMS}
    for arm in ARMS:
        assert predictions[arm] == [r[arm + '_predicted'] == 'True' for r in scores]
    lost_indexes = [i for i, r in enumerate(raw['E']) if r['attack_family'] == 'Scanning'
                    and predictions['legacy_full'][i] and not predictions['full'][i]]
    gained_indexes = [i for i, r in enumerate(raw['E']) if r['attack_family'] == 'Scanning'
                      and not predictions['legacy_full'][i] and predictions['full'][i]]
    assert len(lost_indexes) == 54 and len(gained_indexes) == 3
    populations = {'P_scanning': [r for r in raw['P'] if r['attack_family'] == 'Scanning'],
                   'P_normal': [r for r in raw['P'] if r['label'] == '0'],
                   'C_normal': raw['C'],
                   'E_scanning': [r for r in raw['E'] if r['attack_family'] == 'Scanning'],
                   'E_normal': [r for r in raw['E'] if r['label'] == '0'],
                   'lost54': [raw['E'][i] for i in lost_indexes],
                   'gained3': [raw['E'][i] for i in gained_indexes]}
    assert len(populations['P_scanning']) == 512 and len(populations['E_scanning']) == 1078
    references = {k: v for k, v in populations.items() if k in ('P_scanning', 'P_normal', 'C_normal', 'E_normal')}
    coverage = {name: projection_coverage(populations[name], references, projections) for name in ('E_scanning', 'lost54', 'gained3')}
    individual = {}
    for qname in ('E_scanning', 'lost54', 'gained3'):
        individual[qname] = {}
        for rname, rr in [('P_all', raw['P']), ('P_scanning', populations['P_scanning']), ('P_normal', populations['P_normal'])]:
            values = {c: {float(r[c]) for r in rr} for c in IDS}
            tuples = {signature(r, IDS) for r in rr}
            individual[qname][rname] = {'each_ID_seen': {c: sum(float(r[c]) in values[c] for r in populations[qname]) for c in IDS},
                'all_four_IDs_individually_seen': sum(all(float(r[c]) in values[c] for c in IDS) for r in populations[qname]),
                'exact_four_ID_tuple_seen': sum(signature(r, IDS) in tuples for r in populations[qname])}
    service_keys = Counter(signature(r, SERVICE) for r in populations['E_scanning'])
    services = []
    for service, count in service_keys.most_common(20):
        row = {'PROTOCOL_L7_DST': service, 'E_scanning_rows': count}
        for name in ('P_scanning', 'P_normal', 'C_normal', 'E_normal'):
            row[name] = sum(signature(r, SERVICE) == service for r in populations[name])
        mask = [i for i, r in enumerate(raw['E']) if signature(r, SERVICE) == service]
        row['registered_E_TP'] = {a: sum(predictions[a][i] and raw['E'][i]['attack_family'] == 'Scanning' for i in mask) for a in ARMS}
        row['registered_E_FP'] = {a: sum(predictions[a][i] and raw['E'][i]['label'] == '0' for i in mask) for a in ARMS}
        services.append(row)
    subpop = {k: [r for r in v if subgroup(r)] for k, v in populations.items()}
    dominant, dominant_count = Counter(signature(r, non_id) for r in populations['lost54']).most_common(1)[0]
    dominant_populations = {k: [r for r in v if subgroup(r) and signature(r, non_id) == dominant] for k, v in populations.items()}
    port_overlaps = {}
    port_groups = ('P_scanning', 'P_normal', 'C_normal', 'E_scanning', 'E_normal')
    for i, left in enumerate(port_groups):
        for right in port_groups[i + 1:]:
            a = [float(r['L4_SRC_PORT']) for r in dominant_populations[left]]
            b = [float(r['L4_SRC_PORT']) for r in dominant_populations[right]]
            shared = set(a) & set(b)
            port_overlaps[left + '/' + right] = {'shared_source_port_values': len(shared),
                'left_rows': sum(x in shared for x in a), 'right_rows': sum(x in shared for x in b)}
    top_patterns = []
    for pattern, count in Counter(signature(r, non_id) for r in populations['E_scanning']).most_common(10):
        item = {'E_scanning_rows': count, 'non_ID44_values': dict(zip(non_id, pattern))}
        for name, rr in populations.items():
            item[name + '_matching_rows'] = sum(signature(r, non_id) == pattern for r in rr)
        top_patterns.append(item)
    field_ranges = {name: {c: distribution([float(r[c]) for r in rr]) for c in non_id}
                    for name, rr in subpop.items() if name in ('P_scanning', 'P_normal', 'C_normal', 'E_scanning', 'E_normal', 'lost54')}
    cross_groups = {f'{a}/{b}': len({r['communication_group'] for r in raw[a]} & {r['communication_group'] for r in raw[b]})
                    for a, b in [('P', 'C'), ('P', 'E'), ('C', 'E')]}
    assert all(v == 0 for v in cross_groups.values())
    report = {'scope': 'descriptive stored raw CSV and fixed registered predictions only; zero model load/forward/fit, zero threshold selection',
        'projections': {k: list(v) for k, v in projections.items()}, 'registered_thresholds': thresholds,
        'projection_warning': 'non-ID44 and without-source-port47 are diagnostic projections, not current deployed inputs; their collision bounds do not apply to all48',
        'populations': {k: summary(v, projections) for k, v in populations.items()},
        'coverage_by_projection': coverage, 'individual_ID_coverage': individual,
        'E_scanning_normal_collisions': {k: collision_summary(populations['E_scanning'], populations['E_normal'], cols) for k, cols in projections.items()},
        'cross_role_communication_group_intersections': cross_groups,
        'top20_E_scanning_services': services,
        'TCP6_L7_91_DST443': {k: summary(v, projections) for k, v in subpop.items()},
        'dominant_lost54_non_ID_pattern': {'lost54_rows': dominant_count, 'non_ID44_values': dict(zip(non_id, dominant)),
            'same_service_populations': {k: summary(v, projections) for k, v in dominant_populations.items()},
            'same_service_source_port_overlaps': port_overlaps},
        'top10_E_scanning_non_ID_patterns': top_patterns, 'subgroup_non_ID_field_ranges': field_ranges,
        'metadata_available': ['sample_id', 'communication_group', 'label', 'attack_family'],
        'metadata_not_in_prepared_CSV': ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'],
        'sources': {str(p.relative_to(DATA)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [DATA / 'protocol.json', DATA / 'comparison/evaluation_scores.csv'] + [DATA / f'comparison/{a}/threshold.json' for a in ARMS] + [DATA / 'prepared' / f for f in
                     ['support.csv', 'normal_calibration.csv', 'development_evaluation.csv']]}}
    output = OUT / 'artifact_coverage.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'E_scan_coverage': coverage['E_scanning'], 'lost54_coverage': coverage['lost54'],
                      'dominant_lost54_rows': dominant_count, 'communication_group_overlap': cross_groups}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
