"""Reference input interventions for the proposed scanning mechanism contrast.

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
