"""Frozen input interventions for the scanning mechanism contrast.

These are diagnostic interventions, not a new learning algorithm. This module
does not load CSVs, estimate scaling, use labels, or call a model. The laboratory
caller must supply its frozen baseline FP32 inputs and baseline P-only scaling.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

DOMAIN_SIZE = 65536
SEED = 20261108
KEY_VERSION = 'src-port-frequency-bijection-v1'
N_FEATURES = 48
SOURCE_PORT_COLUMN = 0
ARMS = ('baseline', 'src_frequency_bijection', 'src_mask_zero')


def _ports(values, *, allow_empty=False):
    values = np.asarray(values)
    if values.ndim != 1 or (not allow_empty and not values.size):
        raise ValueError('Source ports must be a nonempty one-dimensional array')
    if values.dtype.kind not in 'iuf' or values.dtype.kind == 'b':
        raise ValueError('Source ports must contain numeric integers')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values >= DOMAIN_SIZE):
        raise ValueError('Source ports must lie in 0..65535')
    if np.any(values != np.floor(values)):
        raise ValueError('Source ports must be integers')
    return values.astype(np.int64, copy=False)


def _key(count, value):
    payload = [KEY_VERSION, SEED, int(count), int(value)]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).digest()


def mapping_sha256(table):
    """Hash table[0],...,table[65535] as exactly 131072 little-endian uint16 bytes."""
    table = _ports(table)
    if table.size != DOMAIN_SIZE or not np.array_equal(np.sort(table), np.arange(DOMAIN_SIZE)):
        raise ValueError('Mapping must be a full-domain bijection')
    return hashlib.sha256(table.astype('<u2').tobytes(order='C')).hexdigest()


def build_frequency_bijection(support_source_ports):
    """Construct only from baseline P source ports, with the fixed seed/version.

    For each count c, ascending values in {v:c_P(v)=c} map to the same values
    sorted by SHA-256(compact JSON [version, seed, c, v]), then by v for ties.
    The zero-count stratum includes every P-unseen domain value. Singleton
    strata are necessarily fixed. No C/E values or labels are accepted.
    """
    support = _ports(support_source_ports)
    counts = np.bincount(support, minlength=DOMAIN_SIZE)
    table = np.empty(DOMAIN_SIZE, dtype=np.uint16)
    strata = []
    for count in sorted(set(counts.tolist())):
        values = np.flatnonzero(counts == count)
        ordered = sorted(values.tolist(), key=lambda value: (_key(count, value), value))
        table[values] = np.asarray(ordered, dtype=np.uint16)
        fixed = int(np.count_nonzero(table[values] == values))
        strata.append({'P_frequency': int(count), 'domain_values': int(values.size),
                       'fixed_values': fixed, 'changed_values': int(values.size) - fixed,
                       'support_rows_in_stratum': int(count * values.size)})
    mapped_counts = np.bincount(table[support].astype(np.int64), minlength=DOMAIN_SIZE)
    if not np.array_equal(counts, mapped_counts):
        raise AssertionError('P source-port multiset changed')
    changed = table.astype(np.int64) != np.arange(DOMAIN_SIZE)
    report = {
        'intervention': KEY_VERSION, 'seed': SEED, 'domain': [0, DOMAIN_SIZE - 1],
        'source_port_column': SOURCE_PORT_COLUMN,
        'key_encoding': 'UTF-8 compact JSON [version,seed,P_frequency,value]; SHA-256 byte ordering; value tie-break',
        'mapping_hash_encoding': 'domain order 0..65535, little-endian uint16, 131072 bytes',
        'mapping_sha256': mapping_sha256(table),
        'support_rows': int(support.size), 'distinct_support_ports': int(np.count_nonzero(counts)),
        'unseen_domain_values': int(np.count_nonzero(counts == 0)),
        'changed_supported_values': int(np.count_nonzero(changed & (counts > 0))),
        'changed_supported_rows': int(counts[changed].sum()),
        'changed_unseen_values': int(np.count_nonzero(changed & (counts == 0))),
        'singleton_frequency_strata': [s for s in strata if s['domain_values'] == 1],
        'frequency_strata': strata,
        'support_count_vector_sha256': hashlib.sha256(counts.astype('<u8').tobytes()).hexdigest(),
        'P_multiset_preserved': True,
        'scaling': 'caller-provided baseline P center/scale remain frozen; no refit',
        'labels_calibration_or_evaluation_used_to_construct_map': False,
    }
    table.setflags(write=False)
    return table, report


def _frozen_matrix(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != N_FEATURES or values.dtype != np.dtype('float32'):
        raise ValueError('Expected an already frozen N-by-48 FP32 matrix')
    if not np.isfinite(values).all():
        raise ValueError('Frozen model inputs must be finite')
    return values


def scaled_source_ports(source_ports, *, baseline_center, baseline_scale):
    """Apply supplied baseline constants; this function never estimates them."""
    ports = _ports(source_ports)
    center, scale = float(baseline_center), float(baseline_scale)
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Baseline center and positive scale must be finite')
    result = ((ports.astype(np.float64) - center) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Provided scaling produces nonfinite FP32 values')
    return result


def b1_frozen_source_port_view(frozen_fp32, source_ports, table, *, baseline_center, baseline_scale):
    """Replace slot 0 by bijected IDs under frozen baseline scaling.

    Other 47 slots and row order are copied byte-for-byte. Both the matrix and
    source-port vector must be aligned to the baseline. The caller retains the
    original matrix; this function returns a new array.
    """
    baseline = _frozen_matrix(frozen_fp32)
    ports = _ports(source_ports)
    if len(ports) != len(baseline):
        raise ValueError('Source-port rows must align with the frozen matrix')
    mapping_sha256(table)
    table = np.asarray(table)
    expected = scaled_source_ports(ports, baseline_center=baseline_center, baseline_scale=baseline_scale)
    if not np.array_equal(baseline[:, SOURCE_PORT_COLUMN].view(np.uint32), expected.view(np.uint32)):
        raise ValueError('Slot 0 does not match the supplied baseline source ports and scaling')
    output = baseline.copy()
    output[:, SOURCE_PORT_COLUMN] = scaled_source_ports(table[ports], baseline_center=baseline_center,
                                                       baseline_scale=baseline_scale)
    return output


def b2_zero_source_port_view(frozen_fp32):
    """Delete slot-0 information after frozen preprocessing; keep 48 columns."""
    output = _frozen_matrix(frozen_fp32).copy()
    output[:, SOURCE_PORT_COLUMN] = np.float32(0)
    return output


def apply_intervention(arm, frozen_x, raw_source_ports, mapping, center, scale):
    """Return one aligned arm from frozen inputs, without fitting or mutation.

    ``mapping`` is required for B1 and may be None for B0/B2. Every arm checks
    that the raw port vector and frozen scaling describe the baseline slot.
    """
    if arm not in ARMS:
        raise ValueError(f'Unknown scanning arm: {arm}')
    baseline = _frozen_matrix(frozen_x)
    ports = _ports(raw_source_ports)
    if len(ports) != len(baseline):
        raise ValueError('Source-port rows must align with the frozen matrix')
    expected = scaled_source_ports(ports, baseline_center=center, baseline_scale=scale)
    if not np.array_equal(baseline[:, 0].view(np.uint32), expected.view(np.uint32)):
        raise ValueError('Slot 0 does not match the supplied baseline source ports and scaling')
    if arm == 'src_frequency_bijection':
        return b1_frozen_source_port_view(baseline, ports, mapping,
                                         baseline_center=center, baseline_scale=scale)
    if arm == 'src_mask_zero':
        return b2_zero_source_port_view(baseline)
    return baseline.copy()


def _port_band(ports):
    # Descriptive numeric intervals only; no application/service label inferred.
    return np.where(ports <= 1023, 0, np.where(ports <= 49151, 1, 2))


def _movement_report(ports, table, destination_ports=None):
    ports = _ports(ports, allow_empty=True)
    changed = table[ports].astype(np.int64)
    movement = changed - ports
    magnitude = np.abs(movement)
    old_band, new_band = _port_band(ports), _port_band(changed)
    transitions = np.zeros((3, 3), dtype=np.int64)
    np.add.at(transitions, (old_band, new_band), 1)
    result = {
        'rows': int(len(ports)),
        'changed_rows': int(np.count_nonzero(movement)),
        'changed_fraction': float(np.count_nonzero(movement) / len(ports)) if len(ports) else None,
        'distinct_source_ports': int(np.unique(ports).size),
        'signed_displacement_mean': float(movement.mean()) if len(ports) else None,
        'absolute_displacement_mean': float(magnitude.mean()) if len(ports) else None,
        'absolute_displacement_median': float(np.median(magnitude)) if len(ports) else None,
        'absolute_displacement_max': int(magnitude.max()) if len(ports) else None,
        'port_band_intervals': [[0, 1023], [1024, 49151], [49152, 65535]],
        'port_band_transition_counts': transitions.tolist(),
        'port_band_changed_rows': int(np.count_nonzero(old_band != new_band)),
    }
    if destination_ports is not None:
        destinations = _ports(destination_ports, allow_empty=True)
        if destinations.shape != ports.shape:
            raise ValueError('Destination ports must align with source-port rows')
        old_equal, new_equal = ports == destinations, changed == destinations
        result['source_destination_equality'] = {
            'before_rows': int(old_equal.sum()), 'after_rows': int(new_equal.sum()),
            'lost_rows': int(np.count_nonzero(old_equal & ~new_equal)),
            'gained_rows': int(np.count_nonzero(~old_equal & new_equal)),
            'unchanged_rows': int(np.count_nonzero(old_equal == new_equal)),
        }
    return result


def audit_mapping(mapping, support_source_ports, role_ports, cohort_masks=None,
                  role_destination_ports=None):
    """Report a supplied P-only mapping; labels never enter the construction.

    ``role_ports`` maps role names to aligned port arrays. ``cohort_masks`` maps
    cohort names to dictionaries containing ``role`` and a same-length boolean
    ``mask``. The cohorts are diagnostic reports, never map-selection inputs.
    Optional destination arrays use the same role names.
    """
    digest = mapping_sha256(mapping)
    table = _ports(mapping)
    support = _ports(support_source_ports)
    counts = np.bincount(support, minlength=DOMAIN_SIZE)
    if not np.array_equal(counts, counts[table]):
        raise ValueError('Mapping must preserve each P-frequency stratum')
    if not isinstance(role_ports, dict) or not role_ports:
        raise ValueError('role_ports must be a nonempty role-to-array dictionary')
    role_arrays = {role: _ports(ports, allow_empty=True) for role, ports in role_ports.items()}
    destinations = {} if role_destination_ports is None else role_destination_ports
    if not isinstance(destinations, dict) or not set(destinations).issubset(role_arrays):
        raise ValueError('Destination-port roles must occur in role_ports')
    destinations = {role: _ports(ports, allow_empty=True) for role, ports in destinations.items()}
    for role, ports in destinations.items():
        if ports.shape != role_arrays[role].shape:
            raise ValueError('Destination ports must align with source-port rows')
    strata = []
    for frequency in np.unique(counts):
        values = np.flatnonzero(counts == frequency)
        fixed = int(np.count_nonzero(table[values] == values))
        strata.append({'P_frequency': int(frequency), 'domain_values': int(values.size),
                       'fixed_values': fixed, 'changed_values': int(values.size) - fixed,
                       'support_rows_in_stratum': int(frequency * values.size)})
    roles = {role: _movement_report(ports, table, destinations.get(role))
             for role, ports in role_arrays.items()}
    cohorts = {}
    for name, cohort in (cohort_masks or {}).items():
        if not isinstance(cohort, dict) or set(cohort) != {'role', 'mask'}:
            raise ValueError('Each cohort must contain exactly role and mask')
        role = cohort['role']
        if role not in role_arrays:
            raise ValueError('Cohort role must occur in role_ports')
        mask = np.asarray(cohort['mask'])
        if mask.dtype != np.dtype(bool) or mask.shape != role_arrays[role].shape:
            raise ValueError('Cohort mask must be a same-length boolean vector')
        dest = destinations[role][mask] if role in destinations else None
        cohorts[name] = {'role': role, **_movement_report(role_arrays[role][mask], table, dest)}
    return {
        'mapping_sha256': digest, 'mapping_version': KEY_VERSION, 'mapping_seed': SEED,
        'mapping_hash_encoding': 'domain order 0..65535, little-endian uint16, 131072 bytes',
        'frequency_strata': strata, 'P_multiset_preserved': True,
        'P_seen_unseen_preserved': True, 'source_port_equality_preserved': True,
        'roles': roles, 'cohorts': cohorts,
        'interpretation': 'Numbering sensitivity is not proof of spurious correlation or semantic '
                          'invariance. Port intervals and cross-column equalities may change.',
    }


def audit_input_collisions(x, labels):
    """Describe finite FP32 equal-input constraints, not unavoidable misses.

    Positive and negative zero are normalized for numerical input equality.
    The minority count is a total-error lower bound for a deterministic row
    classifier on these exact inputs, not an estimate of model error.
    """
    matrix = _frozen_matrix(x)
    truth = np.asarray(labels)
    if truth.ndim != 1 or len(truth) != len(matrix) or truth.dtype.kind not in 'biuf':
        raise ValueError('Binary labels must align with the FP32 matrix')
    if not np.isfinite(truth).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError('Labels must be binary 0/1')
    canonical = np.array(matrix, dtype='<f4', order='C', copy=True)
    canonical[canonical == 0] = 0.0
    pattern_counts = {}
    for row, label in zip(canonical, truth):
        counts = pattern_counts.setdefault(row.tobytes(), [0, 0])
        counts[int(label)] += 1
    mixed = [counts for counts in pattern_counts.values() if counts[0] and counts[1]]
    return {
        'rows': int(len(matrix)), 'unique_input_patterns': len(pattern_counts),
        'duplicate_excess_rows': len(matrix) - len(pattern_counts),
        'largest_pattern_rows': max((sum(c) for c in pattern_counts.values()), default=0),
        'mixed_label_patterns': len(mixed),
        'mixed_label_normal_rows': sum(c[0] for c in mixed),
        'mixed_label_attack_rows': sum(c[1] for c in mixed),
        'minority_row_count_for_identical_inputs': sum(min(c) for c in mixed),
        'signature': 'ordered little-endian finite float32; signed zero normalized; labels excluded',
        'ordered_final_input_sha256': hashlib.sha256(canonical.tobytes()).hexdigest(),
        'interpretation': 'Minority counts are total-error lower bounds for deterministic same-input '
                          'predictions; they are not unavoidable FN counts or model-response evidence.',
    }
