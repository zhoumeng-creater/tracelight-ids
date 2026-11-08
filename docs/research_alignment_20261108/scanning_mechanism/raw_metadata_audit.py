"""Read-only original-record audit; streams official ZIP, never fits a model.

The prepared sample_id suffix is a 1-based CSV data-record number (header excluded).
Run with Python stdlib only. No extracted CSV or row-level metadata is written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import zipfile


class HashingReader(io.RawIOBase):
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.byte_count = 0

    def readable(self):
        return True

    def readinto(self, target):
        block = self.source.read(len(target))
        target[:len(block)] = block
        self.digest.update(block)
        self.byte_count += len(block)
        return len(block)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc(milliseconds):
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def top(counter, n=10):
    return [{'value': list(k) if isinstance(k, tuple) else k, 'count': v}
            for k, v in sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))[:n]]


def describe(rows):
    if not rows:
        return {'rows': 0}
    src = Counter(r['src'] for r in rows)
    dst = Counter(r['dst'] for r in rows)
    pairs = Counter((r['src'], r['dst']) for r in rows)
    starts = sorted(r['start'] for r in rows)
    ends = sorted(r['end'] for r in rows)
    minutes = Counter(r['start'] // 60000 for r in rows)
    days = Counter(utc(r['start'])[:10] for r in rows)
    return {
        'rows': len(rows), 'labels': dict(Counter(r['label'] for r in rows)),
        'families': dict(Counter(r['family'] for r in rows)),
        'source_hosts': len(src), 'destination_hosts': len(dst),
        'any_hosts': len(set(src) | set(dst)), 'directed_ip_pairs': len(pairs),
        'top_source_hosts': top(src), 'top_destination_hosts': top(dst),
        'top_directed_ip_pairs': top(pairs),
        'start_min_utc': utc(starts[0]), 'start_max_utc': utc(starts[-1]),
        'end_min_utc': utc(ends[0]), 'end_max_utc': utc(ends[-1]),
        'start_span_seconds': (starts[-1] - starts[0]) / 1000,
        'start_quantiles_utc': {str(q): utc(starts[round(q * (len(starts) - 1))])
                                for q in (0, .25, .5, .75, 1)},
        'end_before_start_rows': sum(r['end'] < r['start'] for r in rows),
        'unique_start_milliseconds': len(set(starts)),
        'active_start_minutes': len(minutes),
        'largest_start_minute': {'utc': utc(max(minutes, key=minutes.get) * 60000),
                                 'count': max(minutes.values())},
        'top_start_minutes': [{'utc': utc(v['value'] * 60000), 'count': v['count']}
                              for v in top(minutes)],
        'start_utc_day_counts': dict(sorted(days.items())),
    }


def keys(rows, kind):
    if kind == 'source_host':
        return [r['src'] for r in rows]
    if kind == 'destination_host':
        return [r['dst'] for r in rows]
    if kind == 'any_host':
        return [host for r in rows for host in (r['src'], r['dst'])]
    if kind == 'directed_ip_pair':
        return [(r['src'], r['dst']) for r in rows]
    if kind == 'directed_transport_five_tuple':
        return [(r['src'], r['src_port'], r['dst'], r['dst_port'], r['protocol'])
                for r in rows]
    raise ValueError(kind)


def overlap(a, b):
    out = {}
    for kind in ('source_host', 'destination_host', 'any_host', 'directed_ip_pair',
                 'directed_transport_five_tuple'):
        ka, kb = keys(a, kind), keys(b, kind)
        sa, sb = set(ka), set(kb)
        out[kind] = {'a_unique': len(sa), 'b_unique': len(sb),
                     'shared_unique': len(sa & sb),
                     'a_occurrences_seen_in_b': sum(k in sb for k in ka),
                     'b_occurrences_seen_in_a': sum(k in sa for k in kb),
                     'a_occurrences': len(ka), 'b_occurrences': len(kb)}
    if a and b:
        amin, amax = min(r['start'] for r in a), max(r['start'] for r in a)
        bmin, bmax = min(r['start'] for r in b), max(r['start'] for r in b)
        da = {utc(r['start'])[:10] for r in a}
        db = {utc(r['start'])[:10] for r in b}
        out['time'] = {
            'overlap_start_envelope_seconds': max(0, min(amax, bmax) - max(amin, bmin)) / 1000,
            'a_start_rows_inside_b_envelope': sum(bmin <= r['start'] <= bmax for r in a),
            'b_start_rows_inside_a_envelope': sum(amin <= r['start'] <= amax for r in b),
            'a_all_flows_end_before_b_starts': max(r['end'] for r in a) < bmin,
            'b_all_flows_end_before_a_starts': max(r['end'] for r in b) < amin,
            'shared_start_utc_days': sorted(da & db),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[3]
    parser.add_argument('--archive', type=Path, default=repo / 'data/raw/nf-ton-iot-v3/NF-ToN-IoT-v3-official.zip')
    parser.add_argument('--round-dir', type=Path, default=repo.parents[1] / '实验归档/20261108_第十四轮输入修复/核验材料/context-input-v2')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    prepared, wanted, provenance = {}, {}, {}
    feature_names = None
    for role, stem in [('P', 'support'), ('C', 'normal_calibration'), ('E', 'development_evaluation')]:
        path = args.round_dir / 'prepared' / f'{stem}.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        prepared[role] = rows
        provenance[role] = {'file': str(path), 'sha256': sha256(path), 'rows': len(rows)}
        current = [k for k in rows[0] if k not in ('label', 'sample_id', 'communication_group', 'attack_family')]
        if feature_names is None:
            feature_names = current
        assert feature_names == current and len(current) == 48
        for row in rows:
            prefix, suffix = row['sample_id'].rsplit(':', 1)
            record = int(suffix)
            assert record > 0 and record not in wanted
            wanted[record] = (role, row, prefix)
    scores_path = args.round_dir / 'comparison/evaluation_scores.csv'
    with scores_path.open(encoding='utf-8-sig', newline='') as stream:
        scores = {r['row_id']: r for r in csv.DictReader(stream)}
    provenance['scores'] = {'file': str(scores_path), 'sha256': sha256(scores_path)}
    lost_ids = {r['sample_id'] for r in prepared['E']
                if r['attack_family'] == 'Scanning'
                and scores[r['sample_id']]['legacy_full_predicted'] == 'True'
                and scores[r['sample_id']]['full_predicted'] == 'False'}
    f47 = [f for f in feature_names if f != 'L4_SRC_PORT']
    def key(row, features):
        return tuple(Decimal(row[f]) for f in features)
    p_scan48 = {key(r, feature_names) for r in prepared['P'] if r['attack_family'] == 'Scanning'}
    lost_rows = [r for r in prepared['E'] if r['sample_id'] in lost_ids]
    main47 = Counter(key(r, f47) for r in lost_rows).most_common(1)[0][0]
    ppositive_ids = {r['sample_id'] for r in lost_rows if key(r, feature_names) in p_scan48}
    main47_ids = {r['sample_id'] for rows in prepared.values() for r in rows if key(r, f47) == main47}
    assert len(wanted) == 33192 and len(lost_ids) == 54 and len(ppositive_ids) == 15
    metadata = {role: [] for role in prepared}
    prefixes = {p for _, _, p in wanted.values()}
    assert len(prefixes) == 1
    field_comparisons = label_comparisons = family_comparisons = 0
    family_normalizations = Counter()
    print(json.dumps({'event': 'start', 'selected_records': len(wanted), 'max_selected_record': max(wanted)}, ensure_ascii=False), flush=True)
    with zipfile.ZipFile(args.archive) as archive:
        entries = [i for i in archive.infolist() if i.filename.endswith('/data/NF-ToN-IoT-v3.csv')]
        assert len(entries) == 1
        info = entries[0]
        with archive.open(info) as member:
            hashing = HashingReader(member)
            with io.TextIOWrapper(io.BufferedReader(hashing, buffer_size=1024 * 1024), encoding='utf-8-sig', newline='') as text:
                reader = csv.reader(text, strict=True)
                header = next(reader)
                indexes = {name: index for index, name in enumerate(header)}
                assert len(indexes) == len(header) == 55
                for record, raw in enumerate(reader, start=1):
                    assert len(raw) == len(header), (record, len(raw))
                    if record in wanted:
                        role, row, _ = wanted[record]
                        for f in feature_names:
                            actual = raw[indexes[f]]
                            assert actual == row[f] or Decimal(actual) == Decimal(row[f]), (record, f, actual, row[f])
                            field_comparisons += 1
                        assert Decimal(raw[indexes['Label']]) == Decimal(row['label']), record
                        label_comparisons += 1
                        raw_family = raw[indexes['Attack']]
                        assert raw_family.strip().casefold() == row['attack_family'].casefold(), record
                        family_normalizations[(raw_family, row['attack_family'])] += 1
                        family_comparisons += 1
                        metadata[role].append({
                            'id': row['sample_id'], 'src': raw[indexes['IPV4_SRC_ADDR']],
                            'dst': raw[indexes['IPV4_DST_ADDR']],
                            'start': int(raw[indexes['FLOW_START_MILLISECONDS']]),
                            'end': int(raw[indexes['FLOW_END_MILLISECONDS']]),
                            'src_port': int(Decimal(row['L4_SRC_PORT'])),
                            'dst_port': int(Decimal(row['L4_DST_PORT'])),
                            'protocol': int(Decimal(row['PROTOCOL'])),
                            'label': int(row['label']), 'family': row['attack_family'],
                        })
                    if record % 1000000 == 0:
                        print(json.dumps({'event': 'progress', 'records': record,
                                          'selected_aligned': label_comparisons,
                                          'elapsed_seconds': round(time.perf_counter() - started, 1)}), flush=True)
                physical_lines = reader.line_num
        # ZipExtFile reads through EOF and validates its central-directory CRC.
        assert hashing.byte_count == info.file_size
        assert hashing.digest.hexdigest() == next(iter(prefixes)), hashing.digest.hexdigest()
    assert label_comparisons == len(wanted)
    subsets = dict(metadata)
    for role, rows in metadata.items():
        subsets[f'{role}_Scanning'] = [r for r in rows if r['family'] == 'Scanning']
        subsets[f'{role}_main47_all'] = [r for r in rows if r['id'] in main47_ids]
        subsets[f'{role}_main47_Scanning'] = [r for r in rows if r['id'] in main47_ids and r['family'] == 'Scanning']
        subsets[f'{role}_main47_normal'] = [r for r in rows if r['id'] in main47_ids and r['label'] == 0]
    subsets['E_lost54'] = [r for r in metadata['E'] if r['id'] in lost_ids]
    subsets['E_lost_exact_P_Scanning15'] = [r for r in metadata['E'] if r['id'] in ppositive_ids]
    pair_names = list(itertools.combinations(['P', 'C', 'E'], 2)) + [
        ('P_Scanning', 'E_Scanning'), ('P', 'E_lost54'), ('P_Scanning', 'E_lost54'),
        ('P_main47_Scanning', 'E_main47_Scanning'),
        ('P_main47_normal', 'E_main47_Scanning'),
        ('E_main47_normal', 'E_main47_Scanning'),
        ('P_Scanning', 'E_lost_exact_P_Scanning15'),
    ]
    output = {
        'schema_version': 1, 'audit_date': '2026-11-08',
        'method': 'Read-only zip stream with csv.reader(strict=True), actual CSV records, all 48 numeric values and labels aligned; no fit, no model call, no feature generation.',
        'source': {'archive': str(args.archive), 'archive_sha256': sha256(args.archive),
                   'member': info.filename, 'uncompressed_bytes': info.file_size,
                   'observed_uncompressed_bytes': hashing.byte_count,
                   'csv_sha256': hashing.digest.hexdigest(), 'sample_id_prefix_matches_csv_sha256': True,
                   'zip_crc32_hex': f'{info.CRC:08x}', 'zip_read_to_eof_crc_validated': True,
                   'csv_data_records': record, 'physical_lines_including_header': physical_lines,
                   'multiline_csv_records_observed': physical_lines != record + 1},
        'provenance': provenance,
        'alignment': {'selected_records': len(wanted), 'aligned_records': label_comparisons,
                      'numeric_field_comparisons': field_comparisons,
                      'label_comparisons': label_comparisons, 'family_comparisons': family_comparisons,
                      'family_comparison_rule': 'original Attack.strip().casefold() equals prepared attack_family.casefold(), matching lab_data.py FAMILIES normalization',
                      'family_normalizations': top(family_normalizations, 100),
                      'mismatches': 0, 'feature_order': feature_names},
        'definitions': {
            'P': 'support', 'C': 'normal calibration', 'E': 'opened development evaluation',
            'main47': 'most frequent all-but-L4_SRC_PORT pattern among 54 legacy-to-v2 lost Scanning rows; fixed from prior opened artifacts',
            'directed_ip_pair': '(IPV4_SRC_ADDR, IPV4_DST_ADDR), ports excluded',
            'directed_transport_five_tuple': '(source IP, source port, destination IP, destination port, IP protocol)',
            'any_host_occurrences': 'two endpoint occurrences per row, not row count',
            'time': 'original Unix millisecond flow timestamps, rendered UTC; start-time envelope overlap is not event identity',
            'interpretation': 'Host/time overlap bounds the split claim; IP/time are absent from model inputs, so overlap alone is not feature leakage.',
        },
        'subsets': {name: describe(rows) for name, rows in subsets.items()},
        'overlaps': {f'{a}__{b}': overlap(subsets[a], subsets[b]) for a, b in pair_names},
        'elapsed_seconds': round(time.perf_counter() - started, 3),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'complete', 'records': record, 'selected_aligned': label_comparisons,
                      'csv_sha256': hashing.digest.hexdigest(), 'output': str(args.output),
                      'elapsed_seconds': output['elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
