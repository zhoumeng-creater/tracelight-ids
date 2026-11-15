"""Pinned historical supervision and whole-cell time separation; no score-based selection."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from ids.common import read_json, write_json
from ids.contact_evidence.data import prepared, truth, _audit
from ids.response_evidence import data as legacy_data
from ids.threshold_validation.observations import RAW_FEATURES

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / 'configs/support-diagnostic-20261115/registration.json'


def registration():
    return read_json(REGISTRY)


def specs(reg):
    all_specs = {s['name']: s for s in legacy_data.read_registration()['sources']}
    names = reg['source_train'] + [n for rows in reg['roles'].values() for n in rows]
    if len(names) != len(set(names)):
        raise ValueError('A source cannot appear in multiple roles')
    selected = {n: dict(all_specs[n]) for n in names}
    missing = [n for n in names if not {'first_flow_end_ms', 'last_flow_end_ms'} <= selected[n].keys()]
    if missing:
        coverage = read_json(ROOT / 'docs/forward_threshold_20261111/iot23-coverage.json')
        captures = {row['capture']: row for row in coverage['captures']}
        for name in missing:
            record = captures[name]
            if record['source_sha256'] != selected[name]['sha256']:
                raise ValueError('Historical source chronology is not bound to its registered SHA256')
            # Conservative millisecond bounds for phase-order checking only.
            # The old audit used microseconds, while the replay uses ceil-ms;
            # do not pass these bounds as exact legacy preparation timestamps.
            selected[name]['chronology_bounds_ms'] = {
                'first': record['first_completed_us'] // 1000,
                'last': (record['last_completed_us'] + 999) // 1000,
            }

    def boundary(name, side):
        spec = selected[name]
        if 'chronology_bounds_ms' in spec:
            return spec['chronology_bounds_ms'][side]
        return spec[side + '_flow_end_ms']

    roles = reg['roles']
    sequence = [reg['source_train'], roles['target_train'], roles['normal_calibration'],
                roles['forward_development'], roles['transport_external']]
    for left, right in zip(sequence, sequence[1:]):
        if max(boundary(n, 'last') for n in left) >= min(
            boundary(n, 'first') for n in right
        ):
            raise ValueError('Source/target/calibration/evaluation chronology overlaps')
    target_start = min(boundary(n, 'first') for n in roles['target_train'])
    for name in reg['source_train']:
        selected[name]['before_target_start_ms'] = target_start
    return selected


def prepare(root, raw_dir, spec):
    outputs = legacy_data.prepare_one(Path(root), raw_dir, spec)
    if 'before_target_start_ms' in spec:
        receipt = read_json(_audit(root, spec['name']))
        if receipt['last_flow_end_ms'] >= spec['before_target_start_ms']:
            raise ValueError('Prepared historical source reaches target training period')
    return outputs


def frame(root, name):
    obs = pd.read_parquet(prepared(root, name), columns=['row_id', 'flow_end_ms', *RAW_FEATURES])
    labels = pd.read_parquet(truth(root, name), columns=['row_id', 'family'])
    if not np.array_equal(obs.row_id, labels.row_id) or not obs.row_id.is_unique:
        raise ValueError('Observation/truth physical identities differ')
    return obs.assign(family=labels.family.to_numpy(), source_id=name)


def mass_weights(frame, support):
    """Normalize once over the complete allowed training set, never within a minibatch."""
    if support not in {'source', 'target', 'mixed'}:
        raise ValueError('Unknown supervision support')
    if frame.empty or set(frame.label.unique()) != {0, 1}:
        raise ValueError('Training needs both classes')
    if support == 'source':
        if frame.label.value_counts().nunique() != 1:
            raise ValueError('Original source support is not class balanced')
        return np.ones(len(frame), dtype=np.float64)
    keys = ['domain', 'label'] if support == 'mixed' else ['label']
    if support == 'mixed' and set(frame.domain.unique()) != {'source', 'target'}:
        raise ValueError('Mixed arm requires exactly source and target domains')
    counts = frame.groupby(keys).label.transform('size').to_numpy(dtype=np.float64)
    expected = 4 if support == 'mixed' else 2
    if len(frame.groupby(keys)) != expected:
        raise ValueError('Every registered domain/class cell needs support')
    weights = 1 / counts
    return weights / weights.mean()


def training_sets(root, reg):
    pools = pd.concat([frame(root, n) for n in reg['source_train']], ignore_index=True)
    candidates = pools.loc[pools.family.ne('Unresolved')].reset_index(drop=True)
    rng = np.random.default_rng(20261111)
    chosen = []
    for normal in (True, False):
        ix = np.flatnonzero(candidates.family.eq('Benign').to_numpy() == normal)
        chosen.extend(rng.choice(ix, size=4096, replace=False))
    source = candidates.iloc[chosen].sort_values(['source_id', 'row_id']).reset_index(drop=True)
    identity = source[['source_id', 'row_id', 'family']].rename(columns={'source_id': 'capture'})
    digest = hashlib.sha256(identity.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
    expected = read_json(ROOT / 'configs/contact-evidence-20261112/r21-reference.json')
    if digest != expected['train_selected_sha256_csv_lf']:
        raise ValueError('Reconstructed old training identities differ from frozen R21/R23')
    target = pd.concat([frame(root, n) for n in reg['roles']['target_train']], ignore_index=True)
    unresolved = int(target.family.eq('Unresolved').sum())
    target = target.loc[target.family.ne('Unresolved')].reset_index(drop=True)
    source = source.assign(domain='source', label=source.family.ne('Benign').astype('int8'))
    target = target.assign(domain='target', label=target.family.ne('Benign').astype('int8'))
    if int(target.family.eq('ExternalC2').sum()) != 180:
        raise ValueError('Registered B training C2 count differs from metadata review')
    sets = {'source': source, 'target': target,
            'mixed': pd.concat([source, target], ignore_index=True)}
    outputs, counts = [], {}
    for name, rows in sets.items():
        rows = rows.assign(weight=mass_weights(rows, name))
        dest = Path(root) / 'local/training' / (name + '.parquet')
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(dest, index=False)
        outputs.append(dest)
        counts[name] = {
            'rows': len(rows), 'family_rows': rows.family.value_counts().to_dict(),
            'domain_class_mass': [dict(domain=d, label=int(y), rows=len(g), mass=float(g.weight.sum()))
                                  for (d, y), g in rows.groupby(['domain', 'label'])],
        }
    report = Path(root) / 'evidence/training-support.json'
    write_json(report, {'source_identity_sha256': digest, 'target_unresolved_excluded': unresolved,
                        'supports': counts, 'independent_attack_implementations': 'not inferred from row count',
                        'target_labels_identical_in_target_and_mixed': True})
    return outputs + [report]


def calibration_frame(root, reg):
    rows = pd.concat([frame(root, n) for n in reg['roles']['normal_calibration']], ignore_index=True)
    normal = rows.loc[rows.family.eq('Benign')].reset_index(drop=True)
    if len(normal) < 99:
        raise ValueError('Historical normal calibration cannot produce a finite 1% rank')
    return normal
