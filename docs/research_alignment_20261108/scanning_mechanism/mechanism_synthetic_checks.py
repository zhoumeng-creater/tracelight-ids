"""Pure synthetic invariants for source-port interventions; no real data I/O."""
import hashlib
import json
from pathlib import Path

import numpy as np

from mechanism_reference import (
    DOMAIN_SIZE,
    b1_frozen_source_port_view,
    b2_zero_source_port_view,
    build_frequency_bijection,
    mapping_sha256,
    scaled_source_ports,
)


def main():
    checks = []

    def check(name, condition, detail=None):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(name)

    # Different positive-frequency strata, two singleton strata, and a zero stratum.
    support = np.asarray(list(range(8)) + list(range(10, 18)) * 2 + list(range(100, 108)) * 3
                         + [443] * 4 + [65000] * 5, dtype=np.int64)
    original_support = support.copy()
    table, metadata = build_frequency_bijection(support)
    domain = np.arange(DOMAIN_SIZE, dtype=np.int64)
    check('full_domain_bijection', np.array_equal(np.sort(table.astype(np.int64)), domain), DOMAIN_SIZE)
    support_counts = np.bincount(support, minlength=DOMAIN_SIZE)
    check('each_value_stays_in_its_P_frequency_stratum',
          np.array_equal(support_counts, support_counts[table.astype(np.int64)]))
    check('P_multiset_preserved', np.array_equal(np.sort(support), np.sort(table[support].astype(np.int64))))
    check('positive_singleton_strata_are_fixed', int(table[443]) == 443 and int(table[65000]) == 65000)
    repeat_table, repeat_metadata = build_frequency_bijection(support[::-1].tolist())
    check('deterministic_and_P_row_order_invariant', np.array_equal(table, repeat_table)
          and metadata['mapping_sha256'] == repeat_metadata['mapping_sha256'])
    check('declared_mapping_hash_matches_explicit_little_endian_bytes',
          metadata['mapping_sha256'] == hashlib.sha256(table.astype('<u2').tobytes()).hexdigest())

    # These moments belong only to this synthetic fixture. Production must pass frozen baseline moments.
    center, scale = float(support.mean()), float(support.std())
    old_scaled = scaled_source_ports(support, baseline_center=center, baseline_scale=scale)
    new_scaled = scaled_source_ports(table[support], baseline_center=center, baseline_scale=scale)
    check('standardized_P_marginal_bitwise_preserved',
          np.sort(old_scaled).tobytes() == np.sort(new_scaled).tobytes())
    query = np.asarray([0, 0, 7, 11, 107, 443, 65000, 65535, 55555, 55555], dtype=np.int64)
    mapped_query = table[query].astype(np.int64)
    check('seen_unseen_membership_preserved', np.array_equal(np.isin(query, support), np.isin(mapped_query, support)))
    check('pairwise_ID_equality_preserved', np.array_equal(query[:, None] == query[None, :],
                                                         mapped_query[:, None] == mapped_query[None, :]))
    scaled_domain = scaled_source_ports(table[domain], baseline_center=center, baseline_scale=scale)
    check('synthetic_scaling_retains_full_domain_FP32_identity', np.unique(scaled_domain).size == DOMAIN_SIZE)

    matrix = (np.arange(len(query) * 48, dtype=np.float32).reshape(len(query), 48) / np.float32(37))
    matrix[:, 0] = scaled_source_ports(query, baseline_center=center, baseline_scale=scale)
    original_matrix_bytes = matrix.tobytes()
    b1 = b1_frozen_source_port_view(matrix, query, table, baseline_center=center, baseline_scale=scale)
    b2 = b2_zero_source_port_view(matrix)
    check('B1_other47_slots_byte_identical', matrix[:, 1:].tobytes() == b1[:, 1:].tobytes())
    check('B2_other47_slots_byte_identical', matrix[:, 1:].tobytes() == b2[:, 1:].tobytes())
    check('B1_only_supplied_mapping_and_frozen_scaling_used', np.array_equal(b1[:, 0],
          scaled_source_ports(mapped_query, baseline_center=center, baseline_scale=scale)))
    check('B2_exact_positive_zero_slot_with_shape_preserved', b2.shape == matrix.shape
          and b2.dtype == np.float32 and np.all(b2[:, 0].view(np.uint32) == 0))
    check('input_arrays_unmodified', original_matrix_bytes == matrix.tobytes()
          and np.array_equal(support, original_support))

    pair_ports = np.asarray([0, 7])
    pair = np.full((2, 48), np.float32(0.25), dtype=np.float32)
    pair[:, 0] = scaled_source_ports(pair_ports, baseline_center=center, baseline_scale=scale)
    pair_b1 = b1_frozen_source_port_view(pair, pair_ports, table, baseline_center=center, baseline_scale=scale)
    pair_b2 = b2_zero_source_port_view(pair)
    check('B1_preserves_input_distinction_while_B2_creates_projection_collision',
          pair[0].tobytes() != pair[1].tobytes() and pair_b1[0].tobytes() != pair_b1[1].tobytes()
          and pair_b2[0].tobytes() == pair_b2[1].tobytes())

    constant_support = np.asarray([443, 443, 443, 443])
    constant_table, _ = build_frequency_bijection(constant_support)
    check('constant_P_singleton_and_zero_frequency_domain_handled',
          int(constant_table[443]) == 443 and np.array_equal(np.sort(constant_table.astype(np.int64)), domain)
          and np.all(scaled_source_ports(constant_table[constant_support], baseline_center=443, baseline_scale=1) == 0))
    rejected = []
    for invalid in ([], [-1], [65536], [0.5], [np.nan], [True], [[1, 2]]):
        try:
            build_frequency_bijection(invalid)
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    check('invalid_source_port_domains_rejected', all(rejected), len(rejected))
    bad_mapping = table.copy()
    bad_mapping[0] = bad_mapping[1]
    try:
        mapping_sha256(bad_mapping)
    except ValueError:
        mapping_rejected = True
    else:
        mapping_rejected = False
    check('nonbijective_mapping_rejected', mapping_rejected)
    bad_matrix = matrix.copy()
    bad_matrix[0, 0] += np.float32(1)
    try:
        b1_frozen_source_port_view(bad_matrix, query, table, baseline_center=center, baseline_scale=scale)
    except ValueError:
        mismatched_baseline_rejected = True
    else:
        mismatched_baseline_rejected = False
    check('source_port_matrix_alignment_mismatch_rejected', mismatched_baseline_rejected)
    report = {'scope': 'pure synthetic fixtures only; no CSV, model, labels, calibration or evaluation used to construct mapping',
              'checks_passed': sum(c['passed'] for c in checks), 'checks_failed': sum(not c['passed'] for c in checks),
              'checks': checks, 'synthetic_support_ports': support.tolist(),
              'synthetic_baseline_center': center, 'synthetic_baseline_scale': scale,
              'synthetic_mapping': metadata, 'numpy_version': np.__version__,
              'real_CSV_reads': 0, 'real_support_maps_constructed': 0, 'real_preprocessing_fits': 0,
              'model_loads_or_forwards': 0, 'algorithm_effect_claim': False,
              'reference_sha256': hashlib.sha256(Path(__file__).with_name('mechanism_reference.py').read_bytes()).hexdigest()}
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'checks_passed': report['checks_passed'], 'checks_failed': report['checks_failed'],
                      'mapping_sha256': metadata['mapping_sha256'], 'output': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
