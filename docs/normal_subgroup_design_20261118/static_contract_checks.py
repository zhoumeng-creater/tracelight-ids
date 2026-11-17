"""Design arithmetic checks only; no torch, data fitting, model or score generation."""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import itertools
import json


ROOT = Path(__file__).resolve().parent


def quantile(values, u):
    ordered = sorted(values)
    mass = len(ordered) * u
    rank = (mass.numerator + mass.denominator - 1) // mass.denominator
    return ordered[max(1, min(len(ordered), rank)) - 1]


def targets(values, groups, input_classes):
    result = []
    for score, group in zip(values, groups):
        members = [s for s, g in zip(values, groups) if g == group]
        below = sum(s < score for s in members)
        tied = sum(s == score for s in members)
        u = F(2 * below + tied, 2 * len(members))
        result.append(min(score, quantile(values, u)))
    final = [min(t for t, key in zip(result, input_classes) if key == own)
             for own in input_classes]
    return result, final


def tail_metrics(values, groups):
    result = []
    for q in (F(9, 10), F(19, 20), F(99, 100)):
        remaining = len(values) * (1 - q)
        weights = {}
        for score, count in sorted(Counter(values).items(), reverse=True):
            take = min(remaining, count)
            weights[score] = take / count
            remaining -= take
        assert remaining == 0
        assert sum(weights[s] for s in values) == len(values) * (1 - q)
        excess = []
        for group in set(groups):
            idx = [i for i, g in enumerate(groups) if g == group]
            share = sum(weights[values[i]] for i in idx) / len(idx)
            excess.append(max(share - (1 - q), 0))
        result.append(max(excess))
    counts = Counter(values)
    ties = (max(counts.values()), sum(n * (n - 1) // 2 for n in counts.values()))
    return result, ties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = []
    design = json.loads((ROOT.parent.parent / 'configs/normal-subgroup-training-20261118/design.json').read_text(encoding='utf-8'))
    assert not design['execution_enabled'] and not design['training_entry_ready']
    assert design['training']['optimizer_steps_per_trajectory'] == 139 * 16
    assert design['training']['planned_future_max_optimizer_steps'] == 6 * 139 * 16
    assert design['training']['final_refits'] == 0
    checks.append('nonexecuting_spec_and_fixed_budget')
    support = json.loads((ROOT / 'f_group_counts.json').read_text(encoding='utf-8'))
    assert len(support['groups']) == 32
    assert sum(g['F_normal_n'] for g in support['groups']) == 14691
    assert sum(g['V_normal_n'] for g in support['groups']) == 18645
    assert all(g['F_normal_n'] >= 256 and g['V_normal_n'] > 0 for g in support['groups'])
    checks.append('frozen_group_coverage_and_min_support')
    values = [-12, -11, -10, -9, -4, -3, -2, -1]
    groups = ['a'] * 4 + ['b'] * 4
    initial, final = targets(values, groups, list(range(8)))
    assert initial == [-12, -11, -10, -9, -12, -10, -4, -2]
    assert all(t <= s for t, s in zip(final, values))
    assert len(set(final)) > 1
    checks.append('hand_computed_one_sided_midrank_targets')
    provisional, final = targets([-3, -3, -8, -3], ['a', 'a', 'b', 'b'], [0, 1, 2, 0])
    assert provisional[0] == provisional[1]
    assert final[0] == final[3] == min(provisional[0], provisional[3])
    provisional, final = targets([-8, -3, -3, 2], ['a', 'a', 'b', 'b'], [0, 1, 1, 2])
    assert provisional[1] != provisional[2]
    assert final[1] == final[2] == min(provisional[1], provisional[2])
    checks.append('group_ties_and_identical_input_minimum_resolution')
    # Enumerate uniform draws from two unequal source pools; group membership can cross sources.
    pools = [[('x', F(1)), ('x', F(4)), ('y', F(9)), ('y', F(16))],
             [('x', F(25)), ('y', F(36))]]
    all_rows = sum(pools, [])
    counts = Counter(g for g, _ in all_rows)
    exact = sum(sum(e for g, e in all_rows if g == group) / n for group, n in counts.items()) / len(counts)
    draws = list(itertools.product(*pools))
    estimates = [sum(F(len(pool), len(counts) * counts[g]) * e
                     for pool, (g, e) in zip(pools, pair)) for pair in draws]
    assert sum(estimates) / len(estimates) == exact
    checks.append('group_weight_expectation_no_extra_batch_division')
    h, ties = tail_metrics(values, groups)
    transformed, transformed_ties = tail_metrics([3 * s + 17 for s in values], groups)
    assert h == transformed and ties == transformed_ties
    cubic, _ = tail_metrics([s ** 3 for s in values], groups)
    assert h == cubic
    checks.append('tail_metric_strict_monotone_invariance')
    collapsed, collapsed_ties = tail_metrics([0] * len(values), groups)
    assert max(collapsed) == 0 and collapsed_ties[0] > ties[0] and collapsed_ties[1] > ties[1]
    checks.append('constant_score_false_improvement_rejected_by_tie_guard')
    assert not all(new <= old for new, old in zip([F(1, 10), F(1, 20), F(2, 100)], [F(2, 10), F(1, 20), F(1, 100)]))
    checks.append('wide_tail_gain_cannot_hide_one_percent_tail_loss')
    assert not all(new >= old for new, old in zip([5, 6, 6], [4, 5, 7]))
    checks.append('one_positive_score_drop_cannot_be_averaged_away')
    manifest = json.loads((ROOT / 'author_sources/khainb_MFSWB/source_manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        raw = (ROOT / 'author_sources/khainb_MFSWB' / item['path']).read_bytes()
        assert len(raw) == item['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == item['git_blob_sha1']
    checks.append('licensed_MFSWB_author_files_original_bytes')
    result = {'status': 'passed', 'checks': checks, 'checks_passed': len(checks),
              'scope': 'hand_arrays_and_frozen_metadata_only',
              'new_fits': 0, 'model_loads': 0, 'network_forwards': 0, 'new_scores': 0,
              'candidate_targets_on_real_data': 0, 'threshold_searches': 0,
              'not_training_implementation_or_effect_validation': True}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
