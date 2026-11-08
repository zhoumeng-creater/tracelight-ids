"""Synthetic source-port interventions; no real CSV, weights, or model calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ids.context_cache.scanning_inputs import (
    ARMS, DOMAIN_SIZE, apply_intervention, audit_input_collisions, audit_mapping,
    build_frequency_bijection, mapping_sha256, scaled_source_ports,
)


@pytest.fixture(scope='module')
def fixture():
    support = np.asarray(list(range(8)) + list(range(10, 18)) * 2 + list(range(100, 108)) * 3
                         + [443] * 4 + [65000] * 5, dtype=np.int64)
    mapping, report = build_frequency_bijection(support)
    center, scale = float(support.mean()), float(support.std())
    ports = np.array([0, 0, 7, 11, 107, 443, 65000, 65535, 55555, 55555])
    matrix = np.arange(len(ports) * 48, dtype=np.float32).reshape(len(ports), 48) / 37
    matrix[:, 0] = scaled_source_ports(ports, baseline_center=center, baseline_scale=scale)
    return support, mapping, report, center, scale, ports, matrix


def test_frozen_reference_and_mapping_digest(fixture):
    support, mapping, report, *_ = fixture
    path = Path(__file__).parents[1] / 'docs/research_alignment_20261108/scanning_mechanism/mechanism_reference.py'
    spec = importlib.util.spec_from_file_location('scanning_reference', path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    expected, expected_report = reference.build_frequency_bijection(support)
    assert mapping.tobytes() == expected.tobytes()
    assert report == expected_report
    assert mapping_sha256(mapping) == '7bed797abfa8356256276fd0267c0b414e19b0b8719d63d70e7cf24a5224c0d0'
    assert report['mapping_sha256'] == hashlib.sha256(mapping.astype('<u2').tobytes()).hexdigest()
    assert not mapping.flags.writeable


def test_frequency_identity_and_unseen_invariants(fixture):
    support, mapping, _, center, scale, ports, _ = fixture
    domain = np.arange(DOMAIN_SIZE)
    counts = np.bincount(support, minlength=DOMAIN_SIZE)
    assert np.array_equal(np.sort(mapping), domain)
    assert np.array_equal(counts, counts[mapping])
    assert np.array_equal(np.sort(support), np.sort(mapping[support]))
    assert np.array_equal(np.isin(ports, support), np.isin(mapping[ports], support))
    assert np.array_equal(ports[:, None] == ports, mapping[ports, None] == mapping[ports])
    old = scaled_source_ports(support, baseline_center=center, baseline_scale=scale)
    new = scaled_source_ports(mapping[support], baseline_center=center, baseline_scale=scale)
    assert np.sort(old).tobytes() == np.sort(new).tobytes()
    assert mapping[443] == 443 and mapping[65000] == 65000


def test_deterministic_mapping_independent_of_support_order(fixture):
    support, mapping, report, *_ = fixture
    repeated, repeated_report = build_frequency_bijection(support[::-1])
    assert np.array_equal(mapping, repeated)
    assert report == repeated_report


@pytest.mark.parametrize('arm', ARMS)
def test_frozen_other_columns_and_no_mutation(fixture, arm):
    _, mapping, _, center, scale, ports, matrix = fixture
    original = matrix.tobytes()
    result = apply_intervention(arm, matrix, ports, mapping, center, scale)
    assert result.shape == matrix.shape and result.dtype == np.float32
    assert result[:, 1:].tobytes() == matrix[:, 1:].tobytes()
    assert matrix.tobytes() == original
    assert not np.shares_memory(result, matrix)
    if arm == 'baseline':
        assert result.tobytes() == original
    elif arm == 'src_mask_zero':
        assert np.all(result[:, 0].view(np.uint32) == 0)
    else:
        expected = scaled_source_ports(mapping[ports], baseline_center=center, baseline_scale=scale)
        assert result[:, 0].tobytes() == expected.tobytes()


def test_mapping_movement_cohorts_and_cross_column_equalities(fixture):
    support, mapping, _, _, _, ports, _ = fixture
    destinations = ports.copy()
    destinations[0] = mapping[ports[0]]
    masks = {'target': {'role': 'E', 'mask': np.arange(len(ports)) < 3},
             'empty': {'role': 'E', 'mask': np.zeros(len(ports), dtype=bool)}}
    result = audit_mapping(mapping, support, {'P': support, 'E': ports}, masks, {'E': destinations})
    moved = mapping[ports] != ports
    assert result['roles']['E']['changed_rows'] == int(moved.sum())
    assert result['cohorts']['target']['changed_rows'] == int(moved[:3].sum())
    assert result['cohorts']['empty']['changed_fraction'] is None
    equal = result['roles']['E']['source_destination_equality']
    assert equal['before_rows'] == int((ports == destinations).sum())
    assert equal['after_rows'] == int((mapping[ports] == destinations).sum())
    assert sum(sum(row) for row in result['roles']['E']['port_band_transition_counts']) == len(ports)
    json.dumps(result, allow_nan=False)


def test_mask_creates_actual_collision_lower_bound(fixture):
    _, mapping, _, center, scale, _, _ = fixture
    ports = np.array([0, 0, 7, 7, 7])
    matrix = np.full((len(ports), 48), .25, dtype=np.float32)
    matrix[:, 0] = scaled_source_ports(ports, baseline_center=center, baseline_scale=scale)
    labels = np.array([0, 0, 1, 1, 1])
    before = audit_input_collisions(matrix, labels)
    b1 = apply_intervention(ARMS[1], matrix, ports, mapping, center, scale)
    b2 = apply_intervention(ARMS[2], matrix, ports, None, center, scale)
    assert before['mixed_label_patterns'] == 0
    assert audit_input_collisions(b1, labels)['unique_input_patterns'] == 2
    after = audit_input_collisions(b2, labels)
    assert after['unique_input_patterns'] == 1
    assert after['mixed_label_patterns'] == 1
    assert after['minority_row_count_for_identical_inputs'] == 2
    assert after['mixed_label_normal_rows'] == 2 and after['mixed_label_attack_rows'] == 3


def test_collision_normalizes_signed_zero_and_handles_empty():
    matrix = np.zeros((2, 48), dtype=np.float32)
    matrix[1, 4] = np.float32(-0.0)
    assert audit_input_collisions(matrix, [0, 1])['mixed_label_patterns'] == 1
    empty = audit_input_collisions(np.empty((0, 48), dtype=np.float32), [])
    assert empty['unique_input_patterns'] == empty['largest_pattern_rows'] == 0


@pytest.mark.parametrize('invalid', [[], [-1], [65536], [0.5], [np.nan], [True], [[1, 2]], ['443']])
def test_invalid_port_inputs_rejected(invalid):
    with pytest.raises(ValueError):
        build_frequency_bijection(invalid)


def test_nonbijection_and_frequency_violation_rejected(fixture):
    support, mapping, *_ = fixture
    bad = mapping.copy()
    bad[0] = bad[1]
    with pytest.raises(ValueError, match='bijection'):
        mapping_sha256(bad)
    bad = np.arange(DOMAIN_SIZE, dtype=np.uint16)
    bad[0], bad[443] = bad[443], bad[0]
    with pytest.raises(ValueError, match='frequency'):
        audit_mapping(bad, support, {'P': support})


@pytest.mark.parametrize('arm', ARMS)
def test_misaligned_frozen_slot_rejected(fixture, arm):
    _, mapping, _, center, scale, ports, matrix = fixture
    bad = matrix.copy()
    bad[0, 0] += 1
    with pytest.raises(ValueError, match='Slot 0'):
        apply_intervention(arm, bad, ports, mapping, center, scale)


@pytest.mark.parametrize('bad', [np.zeros((2, 47), dtype=np.float32),
                               np.zeros((2, 48), dtype=np.float64),
                               np.full((2, 48), np.nan, dtype=np.float32)])
def test_bad_matrix_rejected(fixture, bad):
    _, mapping, _, center, scale, _, _ = fixture
    with pytest.raises(ValueError):
        apply_intervention('baseline', bad, [0, 7], mapping, center, scale)


def test_bad_cohort_and_destination_alignment_rejected(fixture):
    support, mapping, _, _, _, ports, _ = fixture
    with pytest.raises(ValueError, match='boolean'):
        audit_mapping(mapping, support, {'E': ports}, {'x': {'role': 'E', 'mask': [1]}})
    with pytest.raises(ValueError, match='align'):
        audit_mapping(mapping, support, {'E': ports}, role_destination_ports={'E': [443]})


@pytest.mark.parametrize('labels', [[0], [0, 2], [0, np.nan], ['0', '1']])
def test_collision_bad_labels_rejected(labels):
    with pytest.raises(ValueError):
        audit_input_collisions(np.zeros((2, 48), dtype=np.float32), labels)
