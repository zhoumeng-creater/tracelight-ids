"""Post-AdamW acceptance on a caller-defined finite empirical population.

This is a backtracking/rejection control, not a new optimizer or a population
safety certificate. The caller evaluates its complete predeclared constraint
population (and, if applicable, the candidate's reference-derived threshold).
Validation/test data must not enter that callback. No losses, thresholds or
constraint definitions are changed here to make an update pass.

The accepted parameter delta may be shortened, while AdamW's moments and step
counter are those of the FULL proposal. That explicit convention is not the
same as recomputing AdamW at a smaller learning rate. A rejected transaction
restores model values, optimizer state, gradients and Torch RNG exactly.

Design basis and limitations: docs/research_alignment_20261103/
risk_v4_literature_audit.md, especially the post-step acceptance control.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Real

import torch


_ACCEPTED_STATE = "full_proposal_adamw_moments_and_step_with_shortened_parameter_delta"
_REJECTED_STATE = "original_parameters_optimizer_gradients_and_torch_rng_restored"


def _rng_state():
    # CPU-only checks need not initialize CUDA just to preserve its RNG. A
    # network already resident on CUDA has initialized that runtime beforehand.
    return (torch.random.get_rng_state().clone(),
            [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None)


def _restore_rng(state):
    torch.random.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state_all(state[1])


def _values(tensors):
    return [value.detach().clone() for value in tensors]


def _gradients(parameters):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]


def _restore_values(tensors, values):
    with torch.no_grad():
        for tensor, value in zip(tensors, values):
            tensor.copy_(value)


def _restore_gradients(parameters, gradients):
    for parameter, value in zip(parameters, gradients):
        parameter.grad = None if value is None else value.detach().clone()


def _same_gradients(parameters, gradients):
    return all((parameter.grad is None and value is None) or
               (parameter.grad is not None and value is not None and torch.equal(parameter.grad, value))
               for parameter, value in zip(parameters, gradients))


def _delta_norm(parameters, previous):
    return math.sqrt(sum(float((parameter.detach().to(torch.float64) - value.to(torch.float64)).square().sum())
                         for parameter, value in zip(parameters, previous)))


def _same_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(_same_nested(left[key], right[key]) for key in left))
    if isinstance(left, (tuple, list)):
        return (type(right) is type(left) and len(left) == len(right)
                and all(_same_nested(one, two) for one, two in zip(left, right)))
    return type(left) is type(right) and left == right


def guarded_adamw_step(network, optimizer, *, evaluate, max_backtracks=8, shrink=.5):
    """Try one complete AdamW proposal, accepting the first feasible candidate.

    ``network`` must be a module and ``optimizer`` an AdamW over its parameters.
    Gradients have already been accumulated/clipped by the caller. ``evaluate``
    takes no arguments and returns a JSON-like dict with a Python-bool
    ``feasible`` and a dict ``metrics``. Additional JSON-like fields are kept.
    It must deterministically evaluate the current network without changing
    parameters, buffers, optimizer or gradients; changing train/eval mode is
    allowed and is restored. Its Torch RNG consumption is also restored.

    The initial callback must return feasible=True. Otherwise this function
    raises and leaves the transaction unchanged; finding a feasible start is a
    separate Phase-I task. ``max_backtracks=8`` means up to nine candidates at
    fractions 1, shrink, ..., shrink**8. The full proposal is evaluated exactly,
    without reconstructing it from a rounded subtraction/addition.

    If no candidate passes, the original state is restored and ``accepted`` is
    False. Any exception also restores it before propagating. No positive
    fraction or objective improvement is guaranteed. Acceptance protects only
    the fixed finite population actually checked by the callback.
    """
    if not isinstance(network, torch.nn.Module) or not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("guarded_adamw_step requires a module and an AdamW optimizer")
    if not callable(evaluate):
        raise TypeError("evaluate must be callable")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer")
    if isinstance(shrink, bool) or not isinstance(shrink, Real) or not math.isfinite(shrink) or not 0 < shrink < 1:
        raise ValueError("shrink must be a finite number strictly between zero and one")
    parameters = tuple(network.parameters())
    if not parameters:
        raise ValueError("network has no parameters")
    parameter_ids = {id(parameter) for parameter in parameters}
    optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if (not optimized or len({id(parameter) for parameter in optimized}) != len(optimized)
            or any(id(parameter) not in parameter_ids for parameter in optimized)):
        raise ValueError("AdamW parameters must be unique members of the guarded network")
    if not any(parameter.grad is not None for parameter in optimized):
        raise ValueError("guarded step requires already accumulated gradients")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise ValueError("network parameters must be finite")
    if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise ValueError("accumulated gradients must be finite")

    buffers, modules = tuple(network.buffers()), tuple(network.modules())
    original = _values(parameters)
    original_buffers = _values(buffers)
    original_gradients = _gradients(parameters)
    original_modes = [module.training for module in modules]
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_rng = _rng_state()

    def restore_original():
        _restore_values(parameters, original)
        _restore_values(buffers, original_buffers)
        _restore_gradients(parameters, original_gradients)
        optimizer.load_state_dict(copy.deepcopy(original_optimizer))
        for module, mode in zip(modules, original_modes):
            module.training = mode
        _restore_rng(original_rng)

    def evaluate_current():
        candidate = _values(parameters)
        candidate_buffers = _values(buffers)
        candidate_gradients = _gradients(parameters)
        candidate_modes = [module.training for module in modules]
        candidate_optimizer = copy.deepcopy(optimizer.state_dict())
        candidate_rng = _rng_state()
        try:
            with torch.no_grad():
                result = evaluate()
            if (not isinstance(result, dict) or type(result.get("feasible")) is not bool
                    or not isinstance(result.get("metrics"), dict)):
                raise ValueError("evaluate must return a dict with bool feasible and dict metrics")
            try:
                json.dumps(result, allow_nan=False)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("evaluation report must be finite JSON-like data") from error
            if (any(not torch.equal(parameter, value) for parameter, value in zip(parameters, candidate))
                    or any(not torch.equal(buffer, value) for buffer, value in zip(buffers, candidate_buffers))
                    or not _same_gradients(parameters, candidate_gradients)
                    or not _same_nested(optimizer.state_dict(), candidate_optimizer)):
                raise ValueError("evaluate must not mutate parameters, buffers, gradients or optimizer state")
            return copy.deepcopy(result)
        finally:
            # Also run this on invalid reports, errors or mutating callbacks;
            # the enclosing transaction then restores optimizer state as well.
            _restore_values(parameters, candidate)
            _restore_values(buffers, candidate_buffers)
            _restore_gradients(parameters, candidate_gradients)
            for module, mode in zip(modules, candidate_modes):
                module.training = mode
            _restore_rng(candidate_rng)

    try:
        before = evaluate_current()
        if not before["feasible"]:
            raise ValueError("guarded AdamW requires a verified feasible starting model")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
            raise ValueError("AdamW produced non-finite parameters")
        proposed = _values(parameters)
        proposal_rng = _rng_state()
        proposal_report = None
        for backtracks in range(max_backtracks + 1):
            fraction = float(shrink ** backtracks)
            if backtracks:
                with torch.no_grad():
                    for parameter, start, end in zip(parameters, original, proposed):
                        parameter.copy_(start + fraction * (end - start))
            report = evaluate_current()
            if backtracks == 0:
                proposal_report = report
            if report["feasible"]:
                _restore_gradients(parameters, original_gradients)
                _restore_rng(proposal_rng)
                norm = _delta_norm(parameters, original)
                return {"accepted": True, "fraction": fraction, "backtracks": backtracks,
                        "evaluations": backtracks + 2, "before": before, "after": report,
                        "proposal": proposal_report, "parameter_delta_l2": norm, "moved": norm > 0,
                        "state_semantics": _ACCEPTED_STATE,
                        "scope": "caller_evaluated_finite_empirical_population_only"}
        restore_original()
        return {"accepted": False, "fraction": 0., "backtracks": max_backtracks,
                "evaluations": max_backtracks + 2, "before": before, "after": copy.deepcopy(before),
                "proposal": proposal_report, "parameter_delta_l2": 0., "moved": False,
                "state_semantics": _REJECTED_STATE,
                "scope": "caller_evaluated_finite_empirical_population_only"}
    except BaseException:
        restore_original()
        raise
