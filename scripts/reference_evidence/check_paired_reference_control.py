"""Synthetic-only checks; run from the repository using python -m.

python -m scripts.reference_evidence.check_paired_reference_control --output result.json
"""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from scripts.stage_design.reference_model import Sequence, WholeReferenceModel
from .paired_no_gap import pair_runs, unpack, full_bottleneck
from .paired_reference_control import PairedReferenceControl, pair_stages, full_bottleneck_tensor


def sequence(values, first=0):
    x = torch.tensor([[float(v), 1.] for v in values], dtype=torch.float64)
    return Sequence(x, tuple((first+i) % 2 for i in range(len(values))))


def finite_difference(model, query, reference, parameter, index, step=1e-6):
    model.zero_grad(set_to_none=True)
    target = torch.ones((), dtype=torch.float64)
    loss = F.binary_cross_entropy_with_logits(model(query, [reference])['logit'], target)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    analytic = float(parameter.grad[index])
    old = float(parameter[index].detach())
    with torch.no_grad():
        parameter[index] = old+step
        plus = float(F.binary_cross_entropy_with_logits(model(query, [reference])['logit'], target))
        parameter[index] = old-step
        minus = float(F.binary_cross_entropy_with_logits(model(query, [reference])['logit'], target))
        parameter[index] = old
    numerical = (plus-minus)/(2*step)
    assert abs(analytic-numerical) < 1e-6, (analytic, numerical)
    assert abs(analytic) > 1e-8, 'The checked parameter must be active'
    return {'analytic': analytic, 'numerical': numerical}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20261119)
    model = PairedReferenceControl(2, 5).double()
    old = WholeReferenceModel(2, 5).double()
    assert list(model.state_dict()) == list(old.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in old.parameters())
    # Views obey the already checked integer-run rule for either initial
    # direction, even/odd lengths and the one-real-tail boundary case.
    identity_cases = 0
    for first in (0, 1):
        for length in range(1, 8):
            seq = sequence(range(1, length+1), first)
            runs = [[d, i+1] for i, d in enumerate(seq.directions)]
            raw = pair_runs(runs)
            paired = pair_stages(seq)
            assert unpack(raw) == runs
            assert paired.pairs[:, :, 0].tolist() == [list(p) for p in raw.pairs]
            assert (None if paired.tail is None else int(paired.tail[0])) == raw.tail
            result = model(seq, [seq])
            assert result['score'].item() == 0.
            assert {i for edge in result['path'] for i in edge['query_stage_indices']} == set(range(length))
            assert {i for edge in result['path'] for i in edge['reference_stage_indices']} == set(range(length))
            identity_cases += 1
    # Every tiny binary cost matrix agrees with the already brute-force checked
    # scalar operator. The tensor score keeps a gradient-bearing leaf path.
    dp_cases = 0
    for n in range(1, 4):
        for m in range(1, 4):
            for entries in product((0., 1.), repeat=n*m):
                costs = torch.tensor(entries, dtype=torch.float64).reshape(n, m).requires_grad_()
                score, path = full_bottleneck_tensor(costs)
                expected, _ = full_bottleneck(costs.detach().tolist())
                assert score.item() == expected
                assert {i for i, j in path} == set(range(n))
                assert {j for i, j in path} == set(range(m))
                assert score.requires_grad
                dp_cases += 1
    # Constant positive weights make the impossible cheap-gap route explicit:
    # one exact first pair, then a distinct real pair, against one reference pair.
    constant = PairedReferenceControl(2, 5).double()
    with torch.no_grad():
        for parameter in constant.cost.parameters():
            parameter.zero_()
    reference = sequence([1., 2.])
    later_query = sequence([1., 2., 1., 5.])
    later = constant(later_query, [reference])
    expected_later = 9.*F.softplus(torch.tensor(0., dtype=torch.float64))
    assert torch.allclose(later['score'], expected_later)
    assert {i for edge in later['path'] for i in edge['query_stage_indices']} == {0, 1, 2, 3}
    assert all(edge['operation'] != 'gap' for edge in later['path'])
    later_grad = finite_difference(constant, later_query, reference, constant.cost.network[-1].bias, 0)
    # The full pair is identical. Only the final observed real tail differs,
    # so its cost must be the bottleneck and must retain parameter gradients.
    tail_query = sequence([1., 2., 3.4])
    tail_reference = sequence([1., 2., 3.])
    tail_result = model(tail_query, [tail_reference])
    assert torch.equal(tail_result['score'], model.cost(tail_query.values[-1], tail_reference.values[-1]))
    assert tail_result['path'][-1]['operation'] == 'tail_match'
    tail_grad = finite_difference(model, tail_query, tail_reference, model.cost.network[-1].bias, 0)
    assert torch.equal(tail_result['logit'], F.softplus(model.log_scale)*tail_result['score']-model.intercept)
    try:
        model(tail_query, [reference])
    except ValueError:
        pass
    else:
        raise AssertionError('The tail cannot be paired to an absent/null tail')
    # Incompatible references are ignored without changing the original bank ID.
    selected = model(tail_query, [reference, tail_reference])
    assert selected['reference_index'] == 1
    report = {
        'input_kind': 'synthetic_only', 'real_model_calls': 0, 'optimizer_steps': 0,
        'ordinary_control_not_new_candidate': True,
        'same_RelativeCost_alpha_beta_state_names_and_parameter_count': True,
        'parameters_in_this_synthetic_configuration': sum(p.numel() for p in model.parameters()),
        'pair_rule_roundtrip_and_self_zero_cases': identity_cases,
        'tiny_binary_matrices_equal_scalar_operator_and_full_coverage': dp_cases,
        'later_pair_cannot_be_skipped': {'score': float(later['score'].detach()),
                                       'expected': float(expected_later), 'path': later['path'],
                                       'autograd_finite_difference': later_grad},
        'observed_tail': {'score': float(tail_result['score'].detach()),
                          'path': tail_result['path'], 'autograd_finite_difference': tail_grad,
                          'missing_reference_tail_rejected': True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
