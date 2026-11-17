"""Contract tests for identity-sensitive cached decisions, using toy values only."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PATH = Path(__file__).resolve().parents[1]/'scripts/audit_retained_increment.py'
SPEC = importlib.util.spec_from_file_location('cached_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_alignment_uses_both_keys_and_rejects_changed_truth():
    truth = pd.DataFrame({'source_id': ['a', 'b', 'a'], 'row_id': [0, 0, 1], 'label': [0, 1, 1]})
    saved = truth.assign(score=[0.2, 1.0, -0.5]).iloc[[2, 0, 1]]
    np.testing.assert_array_equal(audit.align(truth, saved).score, [0.2, 1.0, -0.5])
    for bad in [saved.iloc[:2], pd.concat([saved, saved.iloc[[0]]]),
                saved.assign(row_id=[9, 0, 0]), saved.assign(label=[0, 0, 1])]:
        with pytest.raises((ValueError, AssertionError)):
            audit.align(truth, bad)


def test_strict_native_boundary_and_set_decomposition():
    base = np.array([True, False, False, False, False])
    policy = audit.fixed_policy(base, [-10, -2, -1, 0, 0.1], -2)
    np.testing.assert_array_equal(policy['original'], [False, False, True, True, True])
    np.testing.assert_array_equal(policy['candidate'], [True, False, False, False, True])
    np.testing.assert_array_equal(policy['increment'], [False, False, False, False, True])
    assert not (policy['candidate'] & ~policy['direct']).any()
    assert not (base & policy['increment']).any()
    # A positive historical threshold can only tighten the native boundary.
    tight = audit.fixed_policy(np.zeros(3, dtype=bool), [0, 0.2, 0.21], 0.2)
    np.testing.assert_array_equal(tight['candidate'], [False, False, True])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_does_not_silently_become_no_alert(bad):
    with pytest.raises(ValueError):
        audit.fixed_policy([False], [bad], -1)
    with pytest.raises(ValueError):
        audit.fixed_policy([False], [0], bad)


def test_integer_budget_and_reference_protection_boundary():
    labels = np.array([0]*199+[1]*2)
    flags = np.zeros(len(labels), dtype=bool)
    flags[[0, 199, 200]] = True
    reference = audit.counts(labels, flags)
    assert reference['fp_budget'] == 1 and reference['qualified']
    candidate = dict(reference, tp=1)
    assert audit.reference_regression(candidate, reference)
    flags[1] = True
    over_budget = audit.counts(labels, flags)
    assert not over_budget['qualified']
    assert not audit.reference_regression(candidate, over_budget)


def test_every_gate_and_all_seed_requirement():
    args = dict(all_cells_qualified=True, b_tp=72, base_tp_lost=0,
                reference_regressions=0, d_tp=1695, base_d_tp=1694)
    assert audit.pass_conditions(**args)[1]
    for key, bad in [('b_tp', 71), ('d_tp', 1694), ('base_tp_lost', 1),
                     ('reference_regressions', 1), ('all_cells_qualified', False)]:
        changed = dict(args, **{key: bad})
        assert not audit.pass_conditions(**changed)[1]


def test_frozen_design_rejects_floor_and_paired_reference_relaxation():
    path = PATH.parents[1]/'configs/retained-increment-design-20261117/design.json'
    design = audit.read_json(path)
    audit.validate_contract(design)
    design['increment_models']['tabm_ple_s17']['increment_floor'] = -0.01
    with pytest.raises(ValueError):
        audit.validate_contract(design)
    design = audit.read_json(path)
    design['acceptance']['same_seed_original_PLE_qualified_cell_TP_count_decrease_allowed'] = True
    with pytest.raises(ValueError):
        audit.validate_contract(design)
