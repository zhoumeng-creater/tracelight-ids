"""Synthetic single-head checks for calibration-sensitive KV geometry.

No dataset, checkpoint, network, training loop, or external package is used.
This is a mathematical reference, not a TabICLv2 implementation.
"""

from itertools import combinations
import json
import math
from pathlib import Path


def attention(query, keys, values, masses):
    logits = [sum(a * b for a, b in zip(query, key)) for key in keys]
    offset = max(logits)
    raw = [mass * math.exp(logit - offset) for mass, logit in zip(masses, logits)]
    weights = [value / sum(raw) for value in raw]
    return sum(p * v for p, v in zip(weights, values)), weights


def source_gradients(query, keys, values, masses, scales):
    """Gradient w.r.t. each *individual* normalized source state.

    Repeated states are grouped for arithmetic only. Dividing by mass recovers
    the derivative for a single original state, and later sums use multiplicity.
    """
    score, weights = attention(query, keys, values, masses)
    return [
        [(p / mass) * (value - score) * q * scales[0] for q in query]
        + [(p / mass) * scales[1]]
        for p, value, mass in zip(weights, values, masses)
    ]


def scalar_scales(keys, values, masses):
    all_keys = [number for key, mass in zip(keys, masses) for _ in range(mass) for number in key]
    all_values = [value for value, mass in zip(values, masses) for _ in range(mass)]
    def std(items):
        avg = sum(items) / len(items)
        return math.sqrt(sum((v - avg) ** 2 for v in items) / len(items))
    return std(all_keys) or 1.0, std(all_values) or 1.0


def merge_pair(keys, values, masses, pair):
    first, second = pair
    remaining = [i for i in range(len(masses)) if i not in pair]
    total = masses[first] + masses[second]
    key = [(masses[first] * a + masses[second] * b) / total
           for a, b in zip(keys[first], keys[second])]
    value = (masses[first] * values[first] + masses[second] * values[second]) / total
    return ([keys[i] for i in remaining] + [key],
            [values[i] for i in remaining] + [value],
            [masses[i] for i in remaining] + [total])


def partition_distortion(keys, values, masses, pair, metric, scales):
    i, j = pair
    zi = [v / scales[0] for v in keys[i]] + [values[i] / scales[1]]
    zj = [v / scales[0] for v in keys[j]] + [values[j] / scales[1]]
    return (masses[i] * masses[j] / (masses[i] + masses[j])) * sum(
        g * (a - b) ** 2 for g, a, b in zip(metric, zi, zj)
    )


def state_sensitive_merge(keys, values, masses, pair, state_metrics, scales):
    """Solve the two-state diagonal distortion centroid, preserving true mass."""
    i, j = pair
    z = [[v / scales[0] for v in key] + [value / scales[1]]
         for key, value in zip(keys, values)]
    total = masses[i] + masses[j]
    centroid = []
    for d in range(len(z[0])):
        wi, wj = masses[i] * state_metrics[i][d], masses[j] * state_metrics[j][d]
        centroid.append((wi * z[i][d] + wj * z[j][d]) / (wi + wj) if wi + wj > 0
                        else (masses[i] * z[i][d] + masses[j] * z[j][d]) / total)
    distortion = sum(masses[t] * sum(g * (a - b) ** 2
                                    for g, a, b in zip(state_metrics[t], z[t], centroid))
                     for t in pair)
    remaining = [t for t in range(len(masses)) if t not in pair]
    model = ([keys[t] for t in remaining] + [[c * scales[0] for c in centroid[:-1]]],
             [values[t] for t in remaining] + [centroid[-1] * scales[1]],
             [masses[t] for t in remaining] + [total])
    return distortion, model


def rank_threshold(scores, alpha):
    """Strict > rule with split-conformal-style normal rank.

    This deterministic helper does not assert exchangeability for the toy data.
    """
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return math.inf if rank > len(scores) else sorted(scores)[rank - 1]


def evaluate(keys, values, masses):
    common = attention((0.0, -1.0), keys, values, masses)[0]
    rare = attention((1.0, 1.0), keys, values, masses)[0]
    attack = attention((-1.0, 1.0), keys, values, masses)[0]
    normals = [common] * 9800 + [rare] * 200
    threshold = rank_threshold(normals, 0.01)
    return dict(common_score=common, rare_score=rare, attack_score=attack,
                threshold=threshold, fp=sum(s > threshold for s in normals),
                tp=1000 if attack > threshold else 0)


def run():
    keys = [(5.0, 0.0), (-5.0, 0.0), (0.0, -8.0), (0.0, -12.0)]
    values = [0.0, 1.0, -100.0, -120.0]
    masses = [1, 1, 49, 49]
    scales = scalar_scales(keys, values, masses)
    attack_grad = source_gradients((-1.0, 1.0), keys, values, masses, scales)
    boundary_grad = source_gradients((1.0, 1.0), keys, values, masses, scales)
    metric = [sum(mass * (a[d] - b[d]) ** 2
                  for mass, a, b in zip(masses, attack_grad, boundary_grad)) / sum(masses)
              for d in range(3)]
    peak = max(metric)
    metric = [g / peak for g in metric]
    pairs = list(combinations(range(4), 2))
    plain = min(pairs, key=lambda pair: partition_distortion(keys, values, masses, pair, [1, 1, 1], scales))
    shared = min(pairs, key=lambda pair: partition_distortion(keys, values, masses, pair, metric, scales))
    state_metrics = [[(a - b) ** 2 for a, b in zip(ga, gb)]
                     for ga, gb in zip(attack_grad, boundary_grad)]
    sensitive = min(pairs, key=lambda pair: state_sensitive_merge(keys, values, masses, pair, state_metrics, scales)[0])
    plain_eval = evaluate(*merge_pair(keys, values, masses, plain))
    sensitive_eval = evaluate(*state_sensitive_merge(keys, values, masses, sensitive, state_metrics, scales)[1])
    assert plain == (0, 1), plain
    assert shared == (0, 1), shared
    assert sensitive == (2, 3), sensitive
    assert plain_eval['tp'] == 0 and sensitive_eval['tp'] == 1000
    assert plain_eval['fp'] == sensitive_eval['fp'] == 0

    # Any values fail to distinguish these two queries once all keys have x=0.
    bad_keys, bad_values, bad_masses = merge_pair(keys, values, masses, (0, 1))
    _, normal_weights = attention((1.0, 1.0), bad_keys, bad_values, bad_masses)
    _, attack_weights = attention((-1.0, 1.0), bad_keys, bad_values, bad_masses)
    assert normal_weights == attack_weights

    # Check a single original source-state derivative by central difference.
    eps = 1e-5
    base_key = keys[0]
    plus_keys = [(base_key[0] + eps * scales[0], base_key[1])] + keys[1:]
    minus_keys = [(base_key[0] - eps * scales[0], base_key[1])] + keys[1:]
    finite_difference = (attention((-1, 1), plus_keys, values, masses)[0]
                         - attention((-1, 1), minus_keys, values, masses)[0]) / (2 * eps)
    assert math.isclose(finite_difference, attack_grad[0][0], rel_tol=1e-5, abs_tol=1e-8)

    # A common shift leaves calibrated decisions unchanged, despite score MSE.
    base_normal = [0.0] * 9800 + [0.4] * 200
    base_attack = [0.6] * 1000
    shifted_normal = [s + 7.0 for s in base_normal]
    shifted_attack = [s + 7.0 for s in base_attack]
    t0, t1 = rank_threshold(base_normal, .01), rank_threshold(shifted_normal, .01)
    assert [s > t0 for s in base_attack] == [s > t1 for s in shifted_attack]
    tail_normal = [0.0] * 9800 + [0.65] * 200
    tail_threshold = rank_threshold(tail_normal, .01)
    tail_mse = sum((a - b) ** 2 for a, b in zip(base_normal, tail_normal)) / len(base_normal)
    assert sum(s > tail_threshold for s in base_attack) == 0
    assert math.isclose(tail_mse, .00125)

    # Correct mass, even before any learned geometry, is necessary.
    weighted = attention((1,), [(0,), (0,)], [0, 1], [99, 1])[0]
    unweighted = attention((1,), [(0,), (0,)], [0, 1], [1, 1])[0]
    assert weighted == .01 and unweighted == .5

    # Rank threshold is conservative at ties and with insufficient normal n.
    assert math.isinf(rank_threshold([0.0] * 20, .01))
    assert rank_threshold([0.0] * 200, .01) == 0.0
    return {
        'scope': 'constructed mathematical example; no IDS data, TabICLv2 weights, or runtime benchmark',
        'source_states': {'keys': keys, 'values': values, 'masses': masses},
        'normalized_diagonal_metric': metric,
        'per_source_state_diagonal_metrics': state_metrics,
        'same_three_slot_comparison': {
            'full': evaluate(keys, values, masses),
            'ordinary_geometry_merge': list(plain), 'ordinary_geometry_result': plain_eval,
            'rejected_shared_head_metric_merge': list(shared),
            'calibration_geometry_merge': list(sensitive), 'calibration_geometry_result': sensitive_eval,
            'ordinary_distortions': {str(p): partition_distortion(keys, values, masses, p, [1, 1, 1], scales) for p in pairs},
            'shared_metric_distortions': {str(p): partition_distortion(keys, values, masses, p, metric, scales) for p in pairs},
            'state_metric_distortions': {str(p): state_sensitive_merge(keys, values, masses, p, state_metrics, scales)[0] for p in pairs},
        },
        'value_only_cannot_recover': normal_weights == attack_weights,
        'gradient_check': {'analytic': attack_grad[0][0], 'finite_difference': finite_difference},
        'tail_mse_counterexample': {'normal_mse': tail_mse, 'before_tp': 1000, 'after_tp': 0},
        'common_shift_preserves_decisions': True,
        'multiplicity': {'correct': weighted, 'omitted': unweighted},
        'ties_and_insufficient_calibration_checked': True,
    }


if __name__ == '__main__':
    results = run()
    target = Path(__file__).with_name('geometry_reference_results.json')
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False))
