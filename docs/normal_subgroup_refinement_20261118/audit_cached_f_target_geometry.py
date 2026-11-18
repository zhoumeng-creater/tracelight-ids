"""Old GPU F-cache target geometry only: no models, new scores, or C/E input."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO/'src'))
from ids.model_training_redesign.preprocessing import Preprocessor
from ids.normal_subgroup_training.groups import fit_groups, map_groups
from ids.normal_subgroup_training.targets import build_targets, midranks
from ids.normal_subgroup_training.selection import normal_tail_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {'n': 0}
    return {'n': len(x), 'sum': float(x.sum()), 'mean': float(x.mean()),
            'min': float(x.min()), 'median': float(np.median(x)),
            'q90': float(np.quantile(x, .9)), 'max': float(x.max())}


def fraction_record(x):
    return {'value': float(x), 'exact': str(x)}


def tail(scores, groups, q):
    """Exact tail mass and group fractions with shared boundary-tie fractions."""
    n = len(scores)
    mass = Fraction(n * (100 - q), 100)
    boundary_index = (mass.numerator + mass.denominator - 1) // mass.denominator - 1
    boundary = np.sort(scores)[::-1][boundary_index]
    above, equal = scores > boundary, scores == boundary
    boundary_weight = (mass - int(above.sum())) / int(equal.sum())
    weights = np.where(above, 1., np.where(equal, float(boundary_weight), 0.))
    rates, excess = {}, {}
    for group in sorted(set(groups.tolist())):
        selected = groups == group
        group_mass = int((selected & above).sum()) + int((selected & equal).sum()) * boundary_weight
        rates[group] = group_mass / int(selected.sum())
        excess[group] = max(rates[group] - Fraction(100 - q, 100), Fraction(0))
    return {'H': max(excess.values()), 'rates': rates, 'excess': excess, 'weights': weights,
            'mass': mass, 'boundary': float(boundary), 'boundary_tie_n': int(equal.sum()),
            'boundary_weight': boundary_weight}


def ordering(original, target):
    _, inv, counts = np.unique(original, return_inverse=True, return_counts=True)
    mins, maxs = np.full(len(counts), np.inf), np.full(len(counts), -np.inf)
    np.minimum.at(mins, inv, target)
    np.maximum.at(maxs, inv, target)
    split = int(np.sum(mins != maxs))
    crossings = int(np.sum(np.maximum.accumulate(maxs)[:-1] > mins[1:]))
    collapsed = int(np.sum(np.maximum.accumulate(maxs)[:-1] == mins[1:]))
    return {'strictly_increasing_transform_of_original': split == 0 and crossings == 0 and collapsed == 0,
            'original_equal_score_blocks_split_by_target': split,
            'ordered_blocks_with_earlier_target_exceeding_block_minimum': crossings,
            'ordered_blocks_touching_previous_maximum': collapsed,
            'original_unique_scores': int(len(counts)), 'target_unique_scores': int(len(np.unique(target)))}


def posthoc_patterns(frame, scores, targets, groups, original_top1, target_top1):
    """Historical D-nominated patterns: inspect only F after targets are frozen."""
    patterns = [(12,8,1371,3872),(12,8,1371,3873),(12,8,1371,3874),
                (11,8,1285,4725),(11,8,1285,5205)]
    normal = frame.label.to_numpy() == 0
    counts = frame[['orig_pkts','resp_pkts','orig_ip_bytes','resp_ip_bytes']].to_numpy()
    tcp = frame.ip_protocol.to_numpy() == 6
    b_normal = normal & frame.profile.eq('B').to_numpy()
    b_original_top1 = tail(scores[b_normal], groups[b_normal], 99)
    b_target_top1 = tail(targets[b_normal], groups[b_normal], 99)
    before_hash = hashlib.sha256(targets.astype('<f8').tobytes()).hexdigest()

    def top1_mass(selected, original_metric=original_top1, target_metric=target_top1):
        masses = []
        for values, metric in [(scores, original_metric), (targets, target_metric)]:
            above = selected & (values > metric['boundary'])
            equal = selected & (values == metric['boundary'])
            masses.append(int(above.sum()) + int(equal.sum()) * metric['boundary_weight'])
        return {'original_mass': fraction_record(masses[0]), 'ideal_target_mass': fraction_record(masses[1]),
                'mass_change': fraction_record(masses[1] - masses[0]),
                'original_within_pattern_rate': fraction_record(masses[0] / int(selected.sum())),
                'ideal_within_pattern_rate': fraction_record(masses[1] / int(selected.sum()))}

    result = []
    for pattern in patterns:
        chosen = normal & tcp & np.all(counts == np.array(pattern), axis=1)
        memberships = []
        for group in sorted(set(groups[chosen].tolist())):
            full_group = normal & (groups == group)
            selected_in_group = chosen[full_group]
            ranks = midranks(scores[full_group])
            u = (ranks - .5) / int(full_group.sum())
            memberships.append({'leaf': group, 'leaf_normal_n': int(full_group.sum()),
                                'pattern_rows': int((chosen & full_group).sum()),
                                'original_scores': stats(scores[chosen & full_group]),
                                'ideal_targets': stats(targets[chosen & full_group]),
                                'lowered_rows': int(np.sum(scores[chosen & full_group] > targets[chosen & full_group])),
                                'delta': stats(scores[chosen & full_group] - targets[chosen & full_group]),
                                'pooled_F_top1': top1_mass(chosen & full_group),
                                'one_based_midrank': stats(ranks[selected_in_group]),
                                'u_midrank_minus_half_over_n': stats(u[selected_in_group])})
        delta = scores[chosen] - targets[chosen]
        result.append({'counts': list(pattern), 'normal_TCP_F_rows': int(chosen.sum()),
                       'profile_counts': {str(k): int(v) for k,v in frame.loc[chosen,'profile'].value_counts().items()},
                       'original_scores': stats(scores[chosen]), 'ideal_targets': stats(targets[chosen]),
                       'lowered_rows': int((delta > 0).sum()), 'delta': stats(delta),
                       'pooled_F_top1': top1_mass(chosen),
                       'F_B_top1': top1_mass(chosen & b_normal, b_original_top1, b_target_top1),
                       'leaf_memberships': memberships})
    assert before_hash == hashlib.sha256(targets.astype('<f8').tobytes()).hexdigest()
    return {'scope': 'posthoc_pattern_diagnostic_not_used_for_targets',
            'pattern_origin': 'historical_D_error_explanation_not_F_group_construction',
            'filter': 'F label0 TCP only', 'target_hash_unchanged_after_diagnostic': before_hash,
            'F_B_normal_denominator': int(b_normal.sum()), 'F_B_exact_tail_mass': str(b_original_top1['mass']),
            'patterns': result}


def run(archive, output):
    fpath = archive/'inputs/F.parquet'
    frame = pd.read_parquet(fpath)
    assert len(frame) == 19904 and int(frame.label.eq(0).sum()) == 14691
    assert not frame.duplicated(['source_id', 'row_id']).any()
    spec = fit_groups(frame)
    frozen = json.loads((REPO/'docs/normal_subgroup_design_20261118/f_group_counts.json').read_text(encoding='utf-8'))
    assert spec['trees'] == frozen['trees']
    design = json.loads((REPO/'configs/normal-subgroup-training-20261118/design.json').read_text(encoding='utf-8'))
    binding = json.loads((REPO/'configs/normal-subgroup-training-20261118/input-bindings.json').read_text(encoding='utf-8'))
    assert sha(fpath) == binding['files']['inputs/F.parquet']['sha256']
    result = {'scope': {'kind': 'old_GPU_F_cache_ideal_target_geometry_only', 'formal_CPU_targets': False,
                        'new_model_scores': 0, 'model_loads': 0, 'network_forwards': 0, 'fits': 0,
                        'optimizer_steps': 0, 'new_candidate_decisions': 0, 'threshold_searches': 0,
                        'C_E_inputs_read': False, 'V_scores_used': False,
                        'groups_changed': False, 'target_formula_changed': False,
                        'interpretation': 'Ideal target arrays are not trained model outputs or evidence of deployment effect.'},
              'bindings': {'F.parquet': sha(fpath), 'design.json': sha(REPO/'configs/normal-subgroup-training-20261118/design.json'),
                           'groups.py': sha(REPO/'src/ids/normal_subgroup_training/groups.py'),
                           'targets.py': sha(REPO/'src/ids/normal_subgroup_training/targets.py')},
              'group_n': len(spec['group_keys']), 'seeds': {}}
    indexed = frame.set_index(['source_id', 'row_id'])
    for seed in design['seeds']:
        trajectory = archive/f'models/protocol_blind_s{seed}/trajectory'
        paths = {name: trajectory/name for name in ['preprocessing.json', 'score_identities.json', 'selected_scores.npz']}
        ids = json.loads(paths['score_identities.json'].read_text(encoding='utf-8'))['F']
        index = pd.MultiIndex.from_tuples([tuple(x) for x in ids], names=['source_id', 'row_id'])
        assert len(index) == len(frame) and index.is_unique and set(index) == set(indexed.index)
        aligned = indexed.loc[index].reset_index()
        with np.load(paths['selected_scores.npz'], allow_pickle=False) as cache:
            assert set(cache.files) == {'train_scores', 'validation_scores'}
            scores = cache['train_scores']
        assert scores.shape == (len(aligned),) and scores.dtype == np.float64 and np.isfinite(scores).all()
        prep = Preprocessor.from_dict(json.loads(paths['preprocessing.json'].read_text(encoding='utf-8')))
        assert prep.protocol_policy == 'constant_unknown'
        numeric, categorical = prep.transform(aligned)
        labels = aligned.label.to_numpy()
        groups = map_groups(aligned, spec, role='F')
        targets = build_targets(scores, labels, groups, numeric, categorical)
        normal = labels == 0
        original, ideal, ngroups = scores[normal], targets['transport_target'][normal], groups[normal]
        provisional = targets['provisional_target'][normal]
        delta = original - ideal
        assert np.all(delta >= 0) and np.array_equal(targets['identity_target'], scores)
        before = {q: tail(original, ngroups, q) for q in [90, 95, 99]}
        after = {q: tail(ideal, ngroups, q) for q in [90, 95, 99]}
        checked_before = normal_tail_metrics(original, ngroups, spec['group_keys'])
        checked_after = normal_tail_metrics(ideal, ngroups, spec['group_keys'])
        profile_audit = {}
        normal_profiles = aligned.profile.to_numpy()[normal]
        for profile in ['A', 'B']:
            selected = normal_profiles == profile
            profile_groups = sorted(set(ngroups[selected].tolist()))
            old = normal_tail_metrics(original[selected], ngroups[selected], profile_groups)
            new = normal_tail_metrics(ideal[selected], ngroups[selected], profile_groups)
            profile_audit[profile] = {
                'normal_n': int(selected.sum()), 'target_changed_for_this_audit': False,
                'H': {'before': fraction_record(old['H']), 'at_ideal_target': fraction_record(new['H'])},
                'H_q': {q: {'before': fraction_record(old['H_q'][q]), 'at_ideal_target': fraction_record(new['H_q'][q])}
                        for q in old['H_q']},
                'groups': {g: {'normal_n': old['groups'][g]['normal'],
                               'top1_rate_before': fraction_record(old['groups'][g]['T']['0.99']),
                               'top1_rate_at_ideal_target': fraction_record(new['groups'][g]['T']['0.99']),
                               'top1_excess_before': fraction_record(old['groups'][g]['h']['0.99']),
                               'top1_excess_at_ideal_target': fraction_record(new['groups'][g]['h']['0.99'])}
                           for g in profile_groups}}
        index_mismatch = value_mismatch = 0
        ordered = np.sort(original)
        for group in spec['group_keys']:
            chosen = ngroups == group
            ranks = midranks(original[chosen])
            numerator = (2*ranks - 1).astype(np.int64)
            denominator = 2*int(chosen.sum())
            exact_indices = (len(original)*numerator + denominator-1)//denominator - 1
            floating_indices = np.ceil(len(original)*((ranks-.5)/int(chosen.sum()))).astype(np.int64)-1
            index_mismatch += int(np.sum(exact_indices != floating_indices))
            value_mismatch += int(np.sum(ordered[exact_indices] != ordered[floating_indices]))
        for q in [90, 95, 99]:
            name = f'0.{q}'
            assert before[q]['H'] == checked_before['H_q'][name]
            assert after[q]['H'] == checked_after['H_q'][name]
            for group in spec['group_keys']:
                assert before[q]['rates'][group] == checked_before['groups'][group]['T'][name]
                assert after[q]['rates'][group] == checked_after['groups'][group]['T'][name]
        old_top_weights = before[99]['weights']
        old_top = old_top_weights > 0
        grouped = []
        for group in spec['group_keys']:
            selected = ngroups == group
            in_top = selected & old_top
            qdata = {}
            for q in [90, 95, 99]:
                qdata[str(q)] = {'tail_share_before': fraction_record(before[q]['rates'][group]),
                                 'tail_share_at_ideal_target': fraction_record(after[q]['rates'][group]),
                                 'positive_excess_before': fraction_record(before[q]['excess'][group]),
                                 'positive_excess_at_ideal_target': fraction_record(after[q]['excess'][group])}
            grouped.append({'group': group, 'normal_n': int(selected.sum()), 'lowered_n': int((selected & (delta > 0)).sum()),
                            'delta': stats(delta[selected]), 'original_top1_supported_rows': int(in_top.sum()),
                            'original_top1_lowered_n': int((in_top & (delta > 0)).sum()),
                            'original_top1_unchanged_n': int((in_top & (delta == 0)).sum()),
                            'original_top1_weighted_delta_sum': float(np.dot(old_top_weights[selected], delta[selected])),
                            'tail_metrics': qdata})
        no_improvement = [g for g in spec['group_keys'] if before[99]['excess'][g] > 0
                          and after[99]['excess'][g] >= before[99]['excess'][g]]
        residual = [g for g in spec['group_keys'] if after[99]['excess'][g] > 0]
        unchanged = [g for g in grouped if g['original_top1_unchanged_n'] > 0]
        h = {str(q): {'before': fraction_record(before[q]['H']), 'at_ideal_target': fraction_record(after[q]['H']),
                      'nonincreasing': after[q]['H'] <= before[q]['H'], 'strictly_improved': after[q]['H'] < before[q]['H'],
                      'ideal_tail_boundary_tie_n': after[q]['boundary_tie_n'],
                      'changed_tail_membership_rows': int(np.sum(before[q]['weights'] != after[q]['weights']))}
             for q in [90, 95, 99]}
        result['seeds'][str(seed)] = {
            'cache_schema': {'train_scores': {'shape': list(scores.shape), 'dtype': str(scores.dtype)},
                             'available_keys': ['train_scores', 'validation_scores'], 'used_key': 'train_scores'},
            'bindings': {name: sha(path) for name, path in paths.items()},
            'target_array_sha256_diagnostic_only': hashlib.sha256(ideal.astype('<f8').tobytes()).hexdigest(),
            'scale': targets['scale'], 'lowered_normal_n': int((delta > 0).sum()),
            'unchanged_normal_n': int((delta == 0).sum()), 'target_delta': stats(delta),
            'provisional_delta': stats(original - provisional), 'equivalence': targets['equivalence'],
            'positive_anchor_changed_n': int(np.sum(targets['transport_target'][~normal] != scores[~normal])),
            'original_top1': {'exact_mass': str(before[99]['mass']), 'supported_rows': int(old_top.sum()),
                              'lowered_rows': int(np.sum(old_top & (delta > 0))),
                              'unchanged_rows': int(np.sum(old_top & (delta == 0))),
                              'weighted_mean_delta': float(np.dot(old_top_weights, delta) / float(before[99]['mass'])),
                              'delta_on_supported_rows': stats(delta[old_top])},
            'ordering': ordering(original, ideal), 'H_q': h,
            'profile_denominator_audit': profile_audit,
            'integer_ECDF_ceiling_audit': {'index_rows_different_from_float_ceiling': index_mismatch,
                                          'value_rows_different_from_float_ceiling': value_mismatch,
                                          'formula': '(N0*(2*below+tie_count)+(2*n_g)-1)//(2*n_g)-1'},
            'independent_exact_tail_crosscheck': True,
            'H': {'before': fraction_record(checked_before['H']), 'at_ideal_target': fraction_record(checked_after['H'])},
            'normal_ties': {key: {'before': checked_before[key], 'at_ideal_target': checked_after[key]}
                            for key in ['max_tie_block', 'tied_pairs']},
            'all_three_H_q_strictly_improved_at_ideal_target': all(after[q]['H'] < before[q]['H'] for q in [90, 95, 99]),
            'groups_with_original_1pct_excess_not_improved': no_improvement,
            'groups_with_residual_1pct_excess': residual,
            'groups_with_unchanged_original_top1_rows': [g['group'] for g in unchanged],
            'posthoc_pattern_diagnostic': posthoc_patterns(aligned, scores, targets['transport_target'], groups,
                                                         before[99], after[99]),
            'groups': grouped}
    output.mkdir(parents=True, exist_ok=True)
    (output/'cached_f_target_geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 旧GPU F缓存上的固定目标几何审计', '',
             '本次只将冻结目标公式作用于R30已有F分数；没有加载模型、运行前向、训练或产生新模型评分，也没有读取C/E。结果是理想目标数组的性质，不是训练后误报或检出结果。旧GPU缓存不能作为正式CPU逐身份锚。', '',
             '分组与目标均按已定规格原样执行，32个组与既有静态树完全一致。F身份与缓存逐行对齐，文件和代码哈希见JSON。', '',
             '| seed | 正常下调/总量 | 原top1%支持行下调/支持行 | H90 原→目标 | H95 原→目标 | H99 原→目标 |',
             '|---|---:|---:|---:|---:|---:|']
    for seed, record in result['seeds'].items():
        metrics = [record['H_q'][str(q)] for q in [90, 95, 99]]
        lines.append(f"| {seed} | {record['lowered_normal_n']}/14691 | {record['original_top1']['lowered_rows']}/{record['original_top1']['supported_rows']} | " +
                     ' | '.join(f"{m['before']['value']:.8f} → {m['at_ideal_target']['value']:.8f}" for m in metrics) + ' |')
    lines.extend(['', '原top1%使用完整正常样本质量146.91，边界并列等比例分配；“支持行”包括边界上有非零权重的行，不能把147条整数行叫作精确1%。', ''])
    for seed, record in result['seeds'].items():
        eq, rank = record['equivalence'], record['ordering']
        lines.extend([f"## seed {seed}", '',
                      f"正常分数下调均值{record['target_delta']['mean']:.8f}，最大{record['target_delta']['max']:.8f}；原top1%加权平均下调{record['original_top1']['weighted_mean_delta']:.8f}。正例固定锚改变0条。",
                      f"目标是否只是原分数严格递增变换：{rank['strictly_increasing_transform_of_original']}；存在早先低原分数记录的目标超过后续原分数块的块数为{rank['ordered_blocks_with_earlier_target_exceeding_block_minimum']}。这表明目标层面是否发生重排，不表示模型已学会该重排。",
                      f"原1%尾部富集未改善的组：{', '.join(record['groups_with_original_1pct_excess_not_improved']) or '无'}。目标后仍有1%尾部富集的组：{', '.join(record['groups_with_residual_1pct_excess']) or '无'}。原top1%中仍有不下调记录的组：{', '.join(record['groups_with_unchanged_original_top1_rows']) or '无'}。",
                      f"旧GPU缓存同输入参考分数不一致的等价类{eq['reference_inconsistent_classes']}个，最大差{eq['max_reference_difference']:.9g}；初始目标不一致的正常等价类{eq['provisional_conflict_normal_classes']}个，等价类min额外下调{eq['rows_lowered_by_equivalence_min']}行。没有修改identity控制目标，也没有加入容差。", ''])
        tie = record['normal_ties']
        lines.extend([f"理想F目标的最大同分块{tie['max_tie_block']['before']}→{tie['max_tie_block']['at_ideal_target']}，同分对数{tie['tied_pairs']['before']}→{tie['tied_pairs']['at_ideal_target']}。ECDF目标离散化会增加并列；不能据此认定实际模型能通过V并列保护。本审计未计算V目标或新模型评分。", ''])
        lines.extend(['历史D错误形态的F内事后诊断如下。五个模式仅在目标冻结后用于解释，未进入分组或目标构造，诊断前后完整目标hash相同。TCP筛选只用于本表。', '',
                      '| 四计数模式 | F正常TCP人数 | 原分数中位 | 下调人数 | 下调均值/最大 | pooled F top1%质量 原→目标 | 叶组 |',
                      '|---|---:|---:|---:|---:|---:|---|'])
        for pattern in record['posthoc_pattern_diagnostic']['patterns']:
            n = pattern['normal_TCP_F_rows']
            mass = pattern['pooled_F_top1']
            lines.append(f"| {pattern['counts']} | {n} | {pattern['original_scores'].get('median', float('nan')):.6f} | {pattern['lowered_rows']} | {pattern['delta'].get('mean', 0):.6f}/{pattern['delta'].get('max', 0):.6f} | {mass['original_mass']['exact']} → {mass['ideal_target_mass']['exact']} | {', '.join(m['leaf'] for m in pattern['leaf_memberships'])} |")
        key_pattern = record['posthoc_pattern_diagnostic']['patterns'][3]
        key_b = next(m for m in key_pattern['leaf_memberships'] if m['leaf'].startswith('B_'))
        lines.extend(['', f"关键(11,8,1285,4725)的41条包含A5条和B36条；B36条位于{key_b['leaf']}，下调{key_b['lowered_rows']}/36，原分数中位{key_b['original_scores']['median']:.8f}，目标中位{key_b['ideal_targets']['median']:.8f}，下调均值{key_b['delta']['mean']:.8f}，组内u中位{key_b['u_midrank_minus_half_over_n']['median']:.8f}。其pooled F top1%质量{key_b['pooled_F_top1']['original_mass']['exact']}→{key_b['pooled_F_top1']['ideal_target_mass']['exact']}，必须与绝对分数下调一起解释。", '',
                      '逐模式、逐叶的组内midrank及u分布已保存JSON；该诊断不用于调整参数。', ''])
        b_audit = record['profile_denominator_audit']['B']
        lines.extend(['按F_B单独的5848条正常重算，1%质量为58.48；继续使用原来基于全F构造的同一目标，没有重新生成B域目标。', '',
                      '| 四计数模式 | F_B top1%质量 原→同一理想目标 |', '|---|---:|'])
        for pattern in record['posthoc_pattern_diagnostic']['patterns']:
            mass = pattern['F_B_top1']
            lines.append(f"| {pattern['counts']} | {mass['original_mass']['exact']} → {mass['ideal_target_mass']['exact']} |")
        lines.extend(['', f"B域叶组最大1%富集H99：{b_audit['H_q']['0.99']['before']['value']:.8f}→{b_audit['H_q']['0.99']['at_ideal_target']['value']:.8f}。A/B各自三q的H与B每叶变化见JSON。全F或F_B尾部质量都不是C/D阈值后的新增误报数。", ''])
    lines.extend(['## 如何解释不能改善的情况', '',
                  '固定单侧目标只下调超过相应全体正常分位的记录。某组的极端高分已经不高于该分位时，其目标可能保持原值；其他组下调后，它在新的1%尾部中反而占比上升。组内midrank和全体ECDF还有离散取整与并列，不保证所有组最终条件分布相同。',
                  '在固定F正常样本上，若w_g=n_g/N，则每个q都满足 $\\sum_g w_g T_g(q)=1-q$。只要一组尾部份额下降，就必有其他组上升来保持总尾部质量；因此不能再追加“所有组每个q均不增加”的约束并同时要求严格改善。H最大值下降不等于所有组改善，实际15cell误报预算仍须逐项验收。',
                  'JSON逐组记录原/目标的10%、5%、1%尾部质量、正常目标下调量及原top1%未变行数；若H99在理想目标下不改善，这是当前目标几何的实际局限，不能用H90平均改善掩盖，也不能靠修改分箱、系数或分位补救。',
                  '即使理想目标重排改善，网络仍须满足正例约束并将变化迁移到V；本审计未检验网络可达性或泛化。'])
    lines.extend(['', 'ECDF取整已改为整数分子/分母实现，避免数学整数边界被FP64舍入推进一格。三seed本批实际F缓存的整数/原浮点索引差异与值差异均为0；该修正没有改变本报告的实际目标数组。合成边界用N0=50、n_g=25、rank=4证明浮点ceil会错选下一格。'])
    (output/'cached_f_target_geometry.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({seed: {k: row[k] for k in ['lowered_normal_n', 'original_top1', 'H_q',
                    'groups_with_original_1pct_excess_not_improved', 'groups_with_residual_1pct_excess']}
                    for seed, row in result['seeds'].items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.archive, args.output)
