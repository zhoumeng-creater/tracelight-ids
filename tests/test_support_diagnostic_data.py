"""Synthetic identity, chronological split and supervision-mass checks."""
import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.contact_evidence.data import _audit, prepared, truth
from ids.route_batch.metrics import threshold
from ids.support_diagnostic import data
from ids.threshold_validation.observations import RAW_FEATURES


def tiny_registry():
    return {
        'source_train': ['S'],
        'roles': {'target_train': ['T'], 'normal_calibration': ['C'],
                  'forward_development': ['E'], 'transport_external': ['X']},
    }


def synthetic_specs():
    return [{'name': name, 'sha256': name, 'first_flow_end_ms': index * 100 + 1,
             'last_flow_end_ms': (index + 1) * 100}
            for index, name in enumerate(['S', 'T', 'C', 'E', 'X'])]


def test_actual_registration_metadata_has_bound_source_chronology():
    # Reads only the shipped JSON contracts, never a flow or a model score.
    reg = data.registration()
    selected = data.specs(reg)
    for name in reg['source_train']:
        spec = selected[name]
        assert 'chronology_bounds_ms' in spec
        assert 'first_flow_end_ms' not in spec
        assert spec['chronology_bounds_ms']['last'] < spec['before_target_start_ms']
    assert len(selected) == 2 + 6 + 3 + 3 + 12


def test_source_cannot_be_reused_across_roles(monkeypatch):
    reg = tiny_registry()
    reg['roles']['normal_calibration'] = ['T']
    monkeypatch.setattr(data.legacy_data, 'read_registration', lambda: {'sources': synthetic_specs()})
    with pytest.raises(ValueError, match='multiple roles'):
        data.specs(reg)


@pytest.mark.parametrize('left,right', [('S', 'T'), ('T', 'C'), ('C', 'E'), ('E', 'X')])
def test_cross_phase_time_overlap_or_reversal_fails(monkeypatch, left, right):
    specs = synthetic_specs()
    by_name = {spec['name']: spec for spec in specs}
    by_name[left]['last_flow_end_ms'] = by_name[right]['first_flow_end_ms']
    monkeypatch.setattr(data.legacy_data, 'read_registration', lambda: {'sources': specs})
    with pytest.raises(ValueError, match='chronology overlaps'):
        data.specs(tiny_registry())


def test_historical_coverage_must_match_source_digest(monkeypatch):
    specs = synthetic_specs()
    specs[0].pop('first_flow_end_ms')
    specs[0].pop('last_flow_end_ms')
    monkeypatch.setattr(data.legacy_data, 'read_registration', lambda: {'sources': specs})
    monkeypatch.setattr(data, 'read_json', lambda path: {'captures': [
        {'capture': 'S', 'source_sha256': 'wrong', 'first_completed_us': 1000, 'last_completed_us': 100000}
    ]})
    with pytest.raises(ValueError, match='registered SHA256'):
        data.specs(tiny_registry())


def test_actual_prepared_source_end_checked_before_target(tmp_path, monkeypatch):
    spec = {'name': 'S', 'before_target_start_ms': 101}
    report = _audit(tmp_path, 'S')

    def prepare(_root, _raw, _spec):
        write_json(report, {'last_flow_end_ms': 101})
        return [report]

    monkeypatch.setattr(data.legacy_data, 'prepare_one', prepare)
    with pytest.raises(ValueError, match='reaches target'):
        data.prepare(tmp_path, tmp_path, spec)


def test_calibration_needs_99_normal_rows_and_excludes_attacks(tmp_path, monkeypatch):
    reg = tiny_registry()
    frame = pd.DataFrame({'family': ['Benign'] * 98 + ['ExternalC2'] * 100})
    monkeypatch.setattr(data, 'frame', lambda root, name: frame)
    with pytest.raises(ValueError, match='finite 1% rank'):
        data.calibration_frame(tmp_path, reg)
    frame = pd.concat([frame, pd.DataFrame({'family': ['Benign']})], ignore_index=True)
    normal = data.calibration_frame(tmp_path, reg)
    assert len(normal) == 99 and normal.family.eq('Benign').all()
    finite = threshold(np.arange(len(normal)), alpha=0.01)
    assert finite['rank'] == 99 and finite['threshold'] == 98
    assert threshold(np.arange(98), alpha=0.01)['threshold'] is None


def test_mixed_each_domain_and_class_has_one_quarter_mass():
    rows = pd.DataFrame({'domain': ['source'] * 8 + ['target'] * 12,
                         'label': [0] * 4 + [1] * 4 + [0] * 10 + [1] * 2})
    weights = data.mass_weights(rows, 'mixed')
    assert weights.mean() == pytest.approx(1)
    masses = rows.assign(weight=weights).groupby(['domain', 'label']).weight.sum()
    np.testing.assert_allclose(masses / weights.sum(), np.full(4, 0.25))
    target = rows.loc[rows.domain.eq('target')]
    target_masses = target.assign(weight=data.mass_weights(target, 'target')).groupby('label').weight.sum()
    np.testing.assert_allclose(target_masses / target_masses.sum(), [0.5, 0.5])


def synthetic_frame(name, normal, attack, *, unresolved=0):
    count = normal + attack + unresolved
    family = ['Benign'] * normal + ['ExternalC2'] * attack + ['Unresolved'] * unresolved
    values = {column: np.arange(count, dtype=float) + index for index, column in enumerate(RAW_FEATURES)}
    return pd.DataFrame(values).assign(row_id=np.arange(count), flow_end_ms=np.arange(count),
                                        family=family, source_id=name)


def synthetic_training_sources():
    # More than 4096 candidates in both classes, so the frozen sampling matters.
    return {'S': synthetic_frame('S', 4200, 4200),
            'T': synthetic_frame('T', 200, 180, unresolved=3)}


def expected_s_identity(sources):
    # Independent replay of the old registry's row-selection operation.
    candidates = sources['S'].loc[lambda x: x.family.ne('Unresolved')]
    rng = np.random.default_rng(20261111)
    normal = candidates.loc[candidates.family.eq('Benign')]
    attack = candidates.loc[candidates.family.ne('Benign')]
    selected = pd.concat([normal.iloc[rng.choice(len(normal), 4096, replace=False)],
                          attack.iloc[rng.choice(len(attack), 4096, replace=False)]])
    identity = selected.sort_values(['source_id', 'row_id'])[['source_id', 'row_id', 'family']]
    identity = identity.rename(columns={'source_id': 'capture'})
    return hashlib.sha256(identity.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def test_training_target_rows_identical_in_target_and_mixed_and_s_binding(tmp_path, monkeypatch):
    sources = synthetic_training_sources()
    digest = expected_s_identity(sources)
    monkeypatch.setattr(data, 'frame', lambda root, name: sources[name].copy())
    monkeypatch.setattr(data, 'read_json', lambda path: {'train_selected_sha256_csv_lf': digest})
    outputs = data.training_sets(tmp_path, tiny_registry())
    source, target, mixed = [pd.read_parquet(path) for path in outputs[:3]]
    assert len(source) == 8192 and source.label.value_counts().to_dict() == {0: 4096, 1: 4096}
    assert not source.duplicated(['source_id', 'row_id']).any()
    pd.testing.assert_frame_equal(target.drop(columns='weight'),
                                  mixed.loc[mixed.domain.eq('target')].drop(columns='weight').reset_index(drop=True))
    pd.testing.assert_frame_equal(source.drop(columns='weight'),
                                  mixed.loc[mixed.domain.eq('source')].drop(columns='weight').reset_index(drop=True))
    report = read_json(outputs[-1])
    assert report['source_identity_sha256'] == digest
    assert report['target_unresolved_excluded'] == 3
    # A changed selected identity must fail before replacing the frozen selection.
    changed = copy.deepcopy(sources)
    selected_id = source.iloc[0].row_id
    changed['S'].loc[changed['S'].row_id.eq(selected_id), 'row_id'] += 100000
    monkeypatch.setattr(data, 'frame', lambda root, name: changed[name].copy())
    with pytest.raises(ValueError, match='frozen R21/R23'):
        data.training_sets(tmp_path / 'changed', tiny_registry())


def test_prepared_truth_identity_mismatch_rejected(tmp_path):
    obs = synthetic_frame('T', 2, 2)
    obs_path, truth_path = prepared(tmp_path, 'T'), truth(tmp_path, 'T')
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    obs[['row_id', 'flow_end_ms', *RAW_FEATURES]].to_parquet(obs_path, index=False)
    obs[['row_id', 'family']].iloc[::-1].to_parquet(truth_path, index=False)
    with pytest.raises(ValueError, match='physical identities'):
        data.frame(tmp_path, 'T')
