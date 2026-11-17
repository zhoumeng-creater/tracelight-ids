"""Evaluate one registered Boolean policy using frozen R29 caches only.

No model loading, fitting, forward, score generation or threshold search.
Original files are read-only; every run writes a new output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
KEY = ['source_id', 'row_id']
TRUTH = [*KEY, 'label', 'family', 'flow_end_ms', 'ip_protocol', 'orig_port', 'resp_port',
         'duration_seconds', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes', 'resp_ip_bytes']
MODELS = ['tabm_ple_s17', 'tabm_ple_s29', 'tabm_ple_s43']
REFERENCES = ['legacy_hgb_or_rule', 'r26_target_hgb', 'r26_target_mlp17',
              'r26_target_mlp29', 'r26_target_mlp43', 'hgb']
BASE = 'legacy_hgb_or_rule'
FORMULA = 'base OR ((NOT base) AND score > max(0, original_C_threshold))'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path, *, text=False):
    path = Path(path)
    data = path.read_text(encoding='utf-8-sig').encode('utf-8') if text else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                          encoding='utf-8', newline='\n')


def validate_identity(frame):
    if not set(KEY).issubset(frame) or frame.empty:
        raise ValueError('Missing identity columns or empty frame')
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('Null or duplicate physical identity')


def align(truth, saved):
    """Exact one-to-one identity alignment; never trust saved row order."""
    validate_identity(truth)
    validate_identity(saved)
    wanted = pd.MultiIndex.from_frame(truth[KEY])
    indexed = saved.set_index(KEY)
    if len(indexed) != len(wanted) or len(wanted.difference(indexed.index)) or len(indexed.index.difference(wanted)):
        raise ValueError('Missing or extra cached identities')
    matched = indexed.reindex(wanted).reset_index()
    for col in TRUTH:
        if col in saved and col in truth:
            pd.testing.assert_series_equal(truth[col].reset_index(drop=True), matched[col],
                                           check_dtype=False, check_names=False)
    return matched


def booleans(values, *, cached_binary=False):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError('Expected one-dimensional flags')
    if values.dtype.kind == 'b':
        return values.copy()
    if cached_binary and np.isin(values, [0, 1]).all():
        return values.astype(bool)
    raise ValueError('Expected Boolean flags')


def fixed_policy(base, scores, original_threshold):
    base = booleans(base)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != base.shape or not np.isfinite(scores).all() or not np.isfinite(original_threshold):
        raise ValueError('Nonfinite or misaligned scores/threshold')
    boundary = max(0.0, float(original_threshold))
    original = scores > original_threshold
    increment = ~base & (scores > boundary)
    candidate = base | increment
    direct = base | original
    if (base & increment).any() or (base & ~candidate).any() or (candidate & ~direct).any():
        raise AssertionError('Policy set containment violated')
    return {'threshold': boundary, 'original': original, 'increment': increment,
            'candidate': candidate, 'direct': direct}


def counts(labels, alerts):
    labels, alerts = np.asarray(labels), booleans(alerts)
    if labels.shape != alerts.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError('Invalid binary truth/alert alignment')
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    fp, tp = int(((labels == 0) & alerts).sum()), int(((labels == 1) & alerts).sum())
    return {'normal': n0, 'positive': n1, 'fp': fp, 'tp': tp, 'fn': n1-tp,
            'fp_budget': n0//100, 'qualified': bool(n0 and fp <= n0//100)}


def reference_regression(candidate_counts, reference_counts):
    return bool(reference_counts['qualified'] and candidate_counts['tp'] < reference_counts['tp'])


def pass_conditions(*, all_cells_qualified, b_tp, base_tp_lost, reference_regressions, d_tp, base_d_tp):
    gates = {'all_15_cells_qualified': bool(all_cells_qualified), 'B_72': b_tp == 72,
             'base_TP_identities_retained': base_tp_lost == 0,
             'all_qualified_reference_cells_TP_protected': reference_regressions == 0,
             'strict_D_gain_over_base': d_tp > base_d_tp}
    return gates, all(gates.values())


def validate_contract(design):
    if list(design['increment_models']) != MODELS or design['acceptance']['references'] != REFERENCES:
        raise ValueError('Wrong fixed model/reference set')
    if design['decision']['formula'] != FORMULA:
        raise ValueError('Formula changed')
    for model in MODELS:
        if design['increment_models'][model]['increment_floor'] != 0:
            raise ValueError('Floor must be the native constant zero')
    expected_true = ['all_three_seeds_must_pass', 'all_15_cells', 'also_compare_same_seed_original_PLE',
                     'D_TP_must_strictly_exceed_base_1694', 'full_added_lost_identity_reports']
    a = design['acceptance']
    if not all(a[k] is True for k in expected_true) or a['B_TP_required'] != 72:
        raise ValueError('Acceptance changed')
    if a['base_TP_identity_loss_required'] != 0 or a['qualified_reference_cell_TP_count_decrease_allowed']:
        raise ValueError('Reference protection changed')
    if a['same_seed_original_PLE_qualified_cell_TP_count_decrease_allowed']:
        raise ValueError('Paired original PLE protection changed')


def preflight(run, registration_path):
    reg = read_json(registration_path)
    if reg['version'] != 'retained-increment-cached-audit-v1' or reg['scope'] != 'cached_decisions_only':
        raise ValueError('Wrong executable audit registration')
    if reg['script_sha256_lf'] != digest(__file__, text=True):
        raise ValueError('Audit source differs from registration')
    for item in ['design', 'design_document']:
        if digest(REPO/reg[item]['path'], text=True) != reg[item]['sha256_lf']:
            raise ValueError(f'{item} differs from registration')
    design = read_json(REPO/reg['design']['path'])
    validate_contract(design)
    verified = {}
    for rel, expected in design['required_artifacts'].items():
        path = run/rel
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError(f'Original artifact differs: {rel}')
        verified[rel] = expected['sha256']
    return reg, design, verified


def audit(run, output, registration_path):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents or output in run.parents:
        raise ValueError('Output must be separate from original input tree')
    if output.exists():
        raise FileExistsError('Use an absent output directory; never overwrite a prior result')
    reg, design, verified = preflight(run, registration_path)
    cells = design['roles']['ED']['cells']
    truth = pd.read_parquet(run/'inputs/ED.parquet')[TRUTH].reset_index(drop=True)
    validate_identity(truth)
    if truth[TRUTH].isna().any().any() or not truth.label.isin([0, 1]).all():
        raise ValueError('Incomplete evaluation truth')
    if set(truth.source_id) != set(cells) or len(truth) != 17059:
        raise ValueError('Evaluation cell/row coverage changed')
    if int(truth.label.eq(0).sum()) != 15023 or int(truth.label.eq(1).sum()) != 2036:
        raise ValueError('Evaluation denominators changed')
    if not np.array_equal(truth.label.to_numpy(), truth.family.eq('ExternalC2').astype(int).to_numpy()):
        raise ValueError('Family and label disagree')
    normal, positive = truth.label.eq(0).to_numpy(), truth.label.eq(1).to_numpy()
    region_masks = {prefix: truth.source_id.str.startswith(f'TQH-{prefix}_').to_numpy() for prefix in ['B','D']}
    masks = {cell: truth.source_id.eq(cell).to_numpy() for cell in cells}
    units = {(u['model_id'], u['source_id']): u for u in read_json(run/'reports/units.json')}
    cached = {}
    reference_counts = {}
    for model in [*REFERENCES, *MODELS]:
        frame = align(truth, pd.read_parquet(run/f'reports/{model}-decisions.parquet'))
        if not booleans(frame.ready).all():
            raise ValueError('Original reference was not ready')
        cached[model] = booleans(frame.alert)
        reference_counts[model] = {}
        for cell, mask in masks.items():
            current = counts(truth.label.to_numpy()[mask], cached[model][mask])
            old = units[(model, cell)]
            if (current['normal'], current['positive'], current['fp'], current['tp'], current['qualified']) != (
                    old['normal_n'], old['attack_n'], old['fp'], old['tp'], old['qualified']):
                raise ValueError('Cached reference does not reproduce original unit report')
            reference_counts[model][cell] = current
    base = cached[BASE]
    c_frame = pd.read_parquet(run/'inputs/C.parquet')
    validate_identity(c_frame)
    if len(c_frame) != 5511 or c_frame.label.ne(0).any():
        raise ValueError('C must remain the 5511 original normal rows')
    if len(pd.MultiIndex.from_frame(truth[KEY]).intersection(pd.MultiIndex.from_frame(c_frame[KEY]))):
        raise ValueError('C and ED identities overlap')
    c_base = booleans(c_frame['reference-hgb-original-C-or-udp443-30s'], cached_binary=True)
    base_regions = {region: counts(truth.label.to_numpy()[mask], base[mask]) for region, mask in region_masks.items()}
    if (base_regions['B']['tp'], base_regions['B']['fp'], base_regions['D']['tp'], base_regions['D']['fp']) != (66,12,1694,15):
        raise ValueError('Frozen base differs from design evidence')
    per_cell, comparisons, changes, decisions, summaries, c_reports = [], [], [], {}, {}, {}
    for model in MODELS:
        saved = align(truth, pd.read_parquet(run/f'evidence/{model}/ED-scores.parquet'))
        score = saved.score.to_numpy(dtype=np.float64)
        cal = read_json(run/f'evidence/{model}/calibration.json')
        original_threshold = design['increment_models'][model]['original_C_threshold']
        if cal['threshold'] != original_threshold or cal['comparison'] != '>':
            raise ValueError('Frozen threshold differs from design')
        policy = fixed_policy(base, score, original_threshold)
        if not np.array_equal(policy['original'], cached[model]):
            raise ValueError('Frozen scores do not reproduce original PLE decisions')
        c_saved = align(c_frame[TRUTH], pd.read_parquet(run/f'evidence/{model}/C-scores.parquet'))
        c_policy = fixed_policy(c_base, c_saved.score, original_threshold)
        if int(c_policy['original'].sum()) != cal['calibration_fp']:
            raise ValueError('Original C decision count changed')
        c_reports[model] = {'normal': 5511, 'base_fp': int(c_base.sum()),
            'original_PLE_fp': int(c_policy['original'].sum()),
            'increment_fp': int(c_policy['increment'].sum()),
            'combined_fp': int(c_policy['candidate'].sum()), 'recalibration': False}
        frame = truth.copy()
        frame['score'] = score
        frame['base_alert'] = base
        for key in ['original','increment','candidate','direct']:
            frame[key+'_alert'] = policy[key]
        frame['candidate_threshold'] = policy['threshold']
        frame['original_threshold'] = original_threshold
        decisions[model] = frame
        all_qualified, regressions = True, []
        for cell, mask in masks.items():
            label = truth.label.to_numpy()[mask]
            totals = {name: counts(label, values[mask]) for name, values in
                      [('base', base), ('increment', policy['increment']),
                       ('candidate', policy['candidate']), ('direct', policy['direct'])]}
            for key in ['tp', 'fp']:
                if totals['base'][key] + totals['increment'][key] != totals['candidate'][key]:
                    raise AssertionError('Disjoint channel count sum failed')
            all_qualified &= totals['candidate']['qualified']
            per_cell.append({'model': model, 'source_id': cell, 'normal': totals['candidate']['normal'],
                'positive': totals['candidate']['positive'], 'fp_budget': totals['candidate']['fp_budget'],
                **{f'{name}_{key}': value[key] for name,value in totals.items() for key in ['tp','fp']},
                'qualified': totals['candidate']['qualified']})
            for ref in [*REFERENCES, model]:
                rc = reference_counts[ref][cell]
                regression = reference_regression(totals['candidate'], rc)
                row = {'model': model, 'reference': ref, 'source_id': cell,
                    'reference_qualified': rc['qualified'], 'reference_tp': rc['tp'],
                    'candidate_tp': totals['candidate']['tp'], 'tp_difference': totals['candidate']['tp']-rc['tp'],
                    'protected_TP_regression': regression}
                comparisons.append(row)
                if regression:
                    regressions.append(row)
        for ref, alerts in [(r, cached[r]) for r in [*REFERENCES, model]] + [('direct_original_OR', policy['direct'])]:
            for label, kind in [(1, 'TP'), (0, 'FP')]:
                for direction, flag in [('added', policy['candidate'] & ~alerts),
                                        ('lost' if label else 'removed', alerts & ~policy['candidate'])]:
                    part = truth.loc[flag & truth.label.eq(label).to_numpy()].copy()
                    part['model'], part['reference'], part['change'] = model, ref, kind+'_'+direction
                    changes.append(part)
        regions = {}
        for region, mask in region_masks.items():
            regions[region] = {name: counts(truth.label.to_numpy()[mask], values[mask]) for name, values in
                [('base',base),('increment',policy['increment']),('candidate',policy['candidate']),('direct',policy['direct'])]}
            regions[region]['qualified_cells'] = sum(row['qualified'] for row in per_cell
                if row['model'] == model and row['source_id'].startswith(f'TQH-{region}_'))
        tp_lost = int((positive & base & ~policy['candidate']).sum())
        gates, passed = pass_conditions(all_cells_qualified=all_qualified, b_tp=regions['B']['candidate']['tp'],
            base_tp_lost=tp_lost, reference_regressions=len(regressions), d_tp=regions['D']['candidate']['tp'],
            base_d_tp=base_regions['D']['tp'])
        summaries[model] = {'gates': gates, 'passed': passed, 'regions': regions,
            'reference_regressions': regressions, 'base_TP_identities_lost': tp_lost,
            'base_alerts_lost': int((base & ~policy['candidate']).sum()),
            'floor_removed_from_direct_OR': {
                'TP': int((positive & policy['direct'] & ~policy['candidate']).sum()),
                'FP': int((normal & policy['direct'] & ~policy['candidate']).sum())},
            'D_UDP_FP_original_PLE': int((region_masks['D'] & normal & truth.ip_protocol.eq(17).to_numpy() & cached[model]).sum()),
            'D_UDP_FP_candidate': int((region_masks['D'] & normal & truth.ip_protocol.eq(17).to_numpy() & policy['candidate']).sum())}
    all_pass = all(summaries[m]['passed'] for m in MODELS)
    result = {'schema': 'retained-increment-audit-result-v1', 'status': 'complete',
        'candidate_status': 'eligible_for_artifact_and_cost_verification' if all_pass else 'rejected_fixed_candidate',
        'all_three_passed': all_pass, 'models': summaries, 'C_descriptive_only': c_reports,
        'registered_version': reg['version'], 'registration_sha256_lf': digest(registration_path, text=True),
        'execution': {'new_fits': 0, 'model_loads': 0, 'model_forwards': 0, 'new_model_scores': 0,
                      'threshold_searches': 0, 'recalibrations': 0, 'new_cached_policy_evaluations': 3},
        'scope': 'already-opened development caches; not independent confirmation',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'created_utc': datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'registration.json', reg)
    write_json(output/'input_verification.json', {'artifacts': verified, 'all_original_hashes_match': True,
        'identity_join': 'exact one-to-one (source_id,row_id)', 'original_PLE_decisions_reproduced': True,
        'all_original_reference_unit_counts_reproduced': True, 'ED_rows': len(truth), 'C_rows': len(c_frame)})
    for model, frame in decisions.items():
        frame.to_parquet(output/f'{model}-decisions.parquet', index=False)
    pd.DataFrame(per_cell).to_csv(output/'cells.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(comparisons).to_csv(output/'reference_comparisons.csv', index=False, encoding='utf-8-sig')
    pd.concat(changes, ignore_index=True).to_parquet(output/'identity_changes.parquet', index=False)
    write_json(output/'summary.json', result)
    after = {rel: digest(run/rel) for rel in verified}
    if after != verified:
        raise ValueError('Original inputs changed during read-only audit')
    write_json(output/'manifest.json', {'complete': True, 'original_artifacts_unchanged': True,
        'files': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration', type=Path,
        default=REPO/'configs/retained-increment-audit-20261117/registration.json')
    args = parser.parse_args()
    result = audit(args.run, args.output, args.registration)
    print(json.dumps({'candidate_status': result['candidate_status'], 'models': {
        name: {'passed': item['passed'], 'B': item['regions']['B']['candidate'],
               'D': item['regions']['D']['candidate'], 'gates': item['gates']}
        for name,item in result['models'].items()}}, ensure_ascii=True))


if __name__ == '__main__':
    main()
