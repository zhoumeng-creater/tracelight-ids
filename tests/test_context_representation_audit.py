"""Synthetic representation collisions, kept separate from identity leakage."""

import numpy as np
import pandas as pd
import pytest

from ids.context_cache.preprocessing import CSVSchema
from ids.context_cache.representation_audit import audit_final_inputs


SCHEMA = CSVSchema.from_dict({
    'features': ['port', 'signal'], 'categorical': ['port'], 'numeric': ['signal'],
    'label_column': 'label', 'id_column': 'id', 'group_column': 'group', 'family_column': 'family'})


class FrozenUnknownEncoder:
    def transform(self, frame):
        return np.array([[0.0 if p == '80' else -1.0, float(s)]
                         for p, s in zip(frame.port, frame.signal)], dtype=np.float32)

    def report(self):
        return {'kind': 'synthetic_known_80_other_unknown'}


def frames():
    result = {}
    for role, ports, ys in [('support', ['80', '80'], [0, 1]),
                            ('calibration', ['9000', '9001'], [0, 0]),
                            ('evaluation', ['9002', '9003'], [0, 1])]:
        result[role] = pd.DataFrame({'port': ports, 'signal': [0.0, 0.0], 'label': ys,
            'id': [role+'0', role+'1'], 'group': [role, role],
            'family': ['attack' if y else 'normal' for y in ys]})
    return result


def test_actual_float32_unknown_merge_is_not_hidden_by_raw_overlap_check():
    result = audit_final_inputs(FrozenUnknownEncoder(), frames(), SCHEMA)
    overlap = result['cross_role_pairs']['calibration/evaluation']
    assert overlap['raw'] == {'shared_patterns': 0, 'left_rows': 0, 'right_rows': 0}
    assert overlap['final_fp32'] == {'shared_patterns': 1, 'left_rows': 2, 'right_rows': 2}
    e = result['roles']['evaluation']
    assert e['raw']['mixed_label_patterns'] == 0
    assert e['final_fp32']['mixed_label_patterns'] == 1
    assert e['final_fp32']['minority_row_count_for_identical_inputs'] == 1
    assert e['final_fp32']['merged_distinct_raw_patterns'] == 1
    assert e['final_fp32']['rows_in_merged_patterns'] == 2
    assert e['communication_groups']['largest_group_fraction'] == 1.0


def test_labels_ids_and_family_do_not_change_input_signatures():
    original = frames()
    first = audit_final_inputs(FrozenUnknownEncoder(), original, SCHEMA)
    changed = {k: v.copy() for k,v in original.items()}
    changed['evaluation']['label'] = [1, 1]
    changed['evaluation']['id'] = ['relabel0', 'relabel1']
    changed['evaluation']['family'] = ['changed', 'changed']
    second = audit_final_inputs(FrozenUnknownEncoder(), changed, SCHEMA)
    assert first['cross_role_pairs'] == second['cross_role_pairs']
    assert first['roles']['evaluation']['ordered_final_input_sha256'] == second['roles']['evaluation']['ordered_final_input_sha256']
    assert second['roles']['evaluation']['final_fp32']['mixed_label_patterns'] == 0


def test_signed_zero_is_one_input_and_nonfinite_output_is_rejected():
    class SignedZero(FrozenUnknownEncoder):
        def transform(self, frame):
            return np.array([[0.0], [-0.0]], dtype=np.float32)
    result = audit_final_inputs(SignedZero(), frames(), SCHEMA)
    assert result['roles']['evaluation']['final_fp32']['unique_input_patterns'] == 1
    class Nonfinite(SignedZero):
        def transform(self, frame):
            return np.array([[np.inf], [0]], dtype=np.float32)
    with pytest.raises(ValueError, match='finite FP32'):
        audit_final_inputs(Nonfinite(), frames(), SCHEMA)
