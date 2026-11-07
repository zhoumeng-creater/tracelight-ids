"""Describe actual FP32 model-input collisions without loading a model.

Input overlap is reported independently of row/group identity. Equal inputs
from distinct communications are not automatically treated as train leakage.
Labels are used only for diagnostics and never enter a feature signature.
"""

from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json

import numpy as np
import pandas as pd

from .preprocessing import labels


ROLES = ('support', 'calibration', 'evaluation')


def _raw_keys(frame, schema):
    numeric = set(schema.numeric)
    result = []
    for row in frame.loc[:, list(schema.features)].itertuples(index=False, name=None):
        values = []
        for name, value in zip(schema.features, row):
            if pd.isna(value):
                values.append(None)
            elif name in numeric:
                number = float(value)
                values.append((0.0 if number == 0 else number).hex())
            else:
                values.append(str(value))
        result.append(hashlib.sha256(json.dumps(values, ensure_ascii=False,
                                               separators=(',', ':')).encode()).hexdigest())
    return result


def _summary(keys, raw_keys, truth, families):
    counts = Counter(keys)
    by_key = defaultdict(lambda: {'normal': 0, 'attack': 0, 'families': Counter(), 'raw': set()})
    for key, raw, y, family in zip(keys, raw_keys, truth, families):
        item = by_key[key]
        item['normal' if y == 0 else 'attack'] += 1
        item['families'][family] += 1
        item['raw'].add(raw)
    mixed = [v for v in by_key.values() if v['normal'] and v['attack']]
    merged = [v for v in by_key.values() if len(v['raw']) > 1]
    mixed_families = Counter()
    for item in mixed:
        mixed_families.update(item['families'])
    return {
        'rows': len(keys), 'unique_input_patterns': len(counts),
        'duplicate_excess_rows': len(keys) - len(counts),
        'largest_pattern_rows': max(counts.values()),
        'mixed_label_patterns': len(mixed),
        'mixed_label_normal_rows': sum(v['normal'] for v in mixed),
        'mixed_label_attack_rows': sum(v['attack'] for v in mixed),
        'mixed_label_family_rows': dict(sorted(mixed_families.items())),
        'minority_row_count_for_identical_inputs': sum(min(v['normal'], v['attack']) for v in mixed),
        'merged_distinct_raw_patterns': len(merged),
        'rows_in_merged_patterns': sum(v['normal'] + v['attack'] for v in merged),
        'raw_patterns_in_merged_patterns': sum(len(v['raw']) for v in merged),
    }


def audit_final_inputs(preprocessor, frames, schema):
    """Transform declared roles using an already support-fitted preprocessor.

    Audits the final contiguous FP32 vectors including later scaling/clipping,
    not just ordinal IDs. It never fits, predicts, selects, or changes a split.
    """
    if set(frames) != set(ROLES):
        raise ValueError('representation audit requires all three roles')
    raw, encoded, role_reports = {}, {}, {}
    for role in ROLES:
        frame = frames[role]
        truth = labels(frame, schema)
        values = np.asarray(preprocessor.transform(frame))
        if (values.dtype != np.float32 or values.ndim != 2 or values.shape[0] != len(frame)
                or values.shape[1] < 1 or not np.isfinite(values).all()):
            raise ValueError('representation audit requires finite FP32 [rows, features] inputs')
        values = np.array(values, dtype='<f4', order='C', copy=True)
        values[values == 0] = 0.0  # -0 and +0 have identical numerical meaning.
        raw[role] = _raw_keys(frame, schema)
        encoded[role] = [hashlib.sha256(row.tobytes()).hexdigest() for row in values]
        families = (frame[schema.family_column].astype(str).tolist() if schema.family_column
                    else ['attack' if y else 'normal' for y in truth])
        groups = Counter(frame[schema.group_column].astype(str)) if schema.group_column else None
        role_reports[role] = {
            'raw': _summary(raw[role], raw[role], truth, families),
            'final_fp32': _summary(encoded[role], raw[role], truth, families),
            'output_features': values.shape[1],
            'ordered_final_input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
            'communication_groups': None if groups is None else {
                'unique_groups': len(groups), 'largest_group_rows': max(groups.values()),
                'largest_group_fraction': max(groups.values()) / len(frame),
                'top_five_group_rows': sum(n for _, n in groups.most_common(5)),
            },
        }
    overlaps = {}
    for left, right in combinations(ROLES, 2):
        entry = {}
        for kind, keys in [('raw', raw), ('final_fp32', encoded)]:
            common = set(keys[left]) & set(keys[right])
            entry[kind] = {'shared_patterns': len(common),
                           'left_rows': sum(k in common for k in keys[left]),
                           'right_rows': sum(k in common for k in keys[right])}
        overlaps[f'{left}/{right}'] = entry
    return {'format': 'tracelight-final-input-audit-v1',
            'scope': 'descriptive_only_no_model_loading_fitting_or_selection',
            'signature': 'ordered little-endian finite float32; signed zero normalized; labels excluded',
            'preprocessing': preprocessor.report(), 'roles': role_reports,
            'cross_role_pairs': overlaps,
            'interpretation': 'Input collisions are distinct from identity leakage. Minority counts describe '
                              'deterministic identical-input constraints, not unavoidable false negatives. '
                              'Group counts are descriptive and not an effective independent sample size.'}
