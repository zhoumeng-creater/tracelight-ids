"""Six fixed objectives, with the frozen score arithmetic and exact replay.

For the registered source/business partitions, qmax <= emax. D is the normal
reference gap, H separates attacks at source-reference tails, G separates them
at business tails, R=D+H and S=D+G. Weighting is performed only by the caller:
weight lambda multiplies the ENTIRE selected auxiliary objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ids.route_batch.tabm_training import stable_scores
from scripts.calibration_repair.objective import objective_terms as _base_terms
from scripts.calibration_transfer.objective import (
    ScorePass, _arrays, _block, _versions, score_full as _legacy_score_full,
)
from scripts.normal_tail_mechanism import budget
from scripts.comprehensive_batch.matrix import ARMS


@dataclass(frozen=True)
class ResidentInputs:
    """One immutable task input on its execution device; no activations cached."""

    numeric: np.ndarray
    categorical: np.ndarray | None
    numeric_tensor: torch.Tensor
    categorical_tensor: torch.Tensor | None
    tensor_versions: tuple


def resident_inputs(numeric, categorical, *, device, numeric_tensor=None, categorical_tensor=None):
    """Upload once, or reuse already resident BCE inputs after one binding check.

    The caller keeps both CPU arrays and device tensors fixed for the lifetime
    of this object, exactly as required by the original discovery/replay path.
    """
    numeric, categorical = _arrays(numeric, categorical)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    def prepare(array, tensor):
        if array is None:
            if tensor is not None:
                raise ValueError("Resident categories have no matching CPU input")
            return None
        source = torch.from_numpy(array)
        if tensor is None:
            return source.to(device)
        if (tensor.device != device or tensor.dtype != source.dtype or tensor.shape != source.shape
                or tensor.requires_grad or not tensor.is_contiguous()
                or not torch.equal(tensor.detach().cpu(), source)):
            raise ValueError("Resident tensor differs from the bound CPU input")
        return tensor

    x, cats = prepare(numeric, numeric_tensor), prepare(categorical, categorical_tensor)
    return ResidentInputs(numeric, categorical, x, cats,
                          (x._version, None if cats is None else cats._version))


def _check_resident(resident, numeric, categorical, device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if (resident.numeric is not numeric or resident.categorical is not categorical
            or resident.numeric_tensor.device != device
            or resident.tensor_versions != (resident.numeric_tensor._version,
                None if resident.categorical_tensor is None else resident.categorical_tensor._version)):
        raise ValueError("Resident input binding or tensor version changed")


def _resident_block(network, resident, start, stop, *, check_finite=True):
    cats = None if resident.categorical_tensor is None else resident.categorical_tensor[start:stop]
    scores = stable_scores(network(resident.numeric_tensor[start:stop], cats))
    if check_finite and not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Nonfinite deployed transfer score")
    return scores


def score_full(network, numeric, categorical, *, device, batch_size, resident=None):
    """Same canonical forwards; resident discovery copies the score vector once."""
    if resident is None:
        return _legacy_score_full(network, numeric, categorical, device=device, batch_size=batch_size)
    numeric, categorical = _arrays(numeric, categorical)
    _check_resident(resident, numeric, categorical, device)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    blocks = tuple((start, min(start + batch_size, len(numeric)))
                   for start in range(0, len(numeric), batch_size))
    versions = _versions(network)
    training = network.training
    network.eval()
    try:
        with torch.no_grad():
            # Concatenation only copies values. Every network call retains the
            # original block shape, FP32 logits and FP64 stable-score arithmetic.
            scores = torch.cat([_resident_block(network, resident, start, stop, check_finite=False)
                                for start, stop in blocks]).cpu()
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("Nonfinite deployed transfer score")
    finally:
        network.train(training)
    if _versions(network) != versions:
        raise RuntimeError("Model state changed during full-score discovery")
    return ScorePass(scores, blocks, versions, batch_size, len(numeric))


def objective_terms(scores, groups, arm, *, alpha=.01, margin=1.):
    """Keep existing G/R exactly, and expose their components and bridge S."""
    if arm not in ARMS:
        raise ValueError("Unknown comprehensive-batch arm")
    result = _base_terms(scores, groups, "R_reference_envelope", alpha=alpha, margin=margin)
    d = result["reference_transfer_penalty"]
    h = result["reference_attack_penalty"]
    g = result["group_penalty"]
    zero = scores.sum() * 0.
    normal, attack = {
        "A_BCE": (zero, zero),
        "H_reference_attack": (zero, h),
        "D_reference_alignment": (d, zero),
        "G_group_tail": (zero, g),
        "R_reference_envelope": (d, h),
        "S_alignment_group": (d, g),
    }[arm]
    result.update(penalty=normal + attack, normal_penalty=normal,
                  attack_penalty=attack, alignment_group_penalty=d + g)
    return result


def score_gradient(scores, groups, arm, *, alpha=.01, margin=1.):
    """Return exact CPU FP64 subgradient and strict integer-budget diagnostics."""
    leaf = torch.as_tensor(scores, dtype=torch.float64, device="cpu").detach().clone()
    leaf.requires_grad_(True)
    result = objective_terms(leaf, groups, arm, alpha=alpha, margin=margin)
    derivative, = torch.autograd.grad(result["penalty"], leaf)
    scalar_keys = ("penalty", "normal_penalty", "attack_penalty", "group_penalty",
                   "reference_envelope_penalty", "reference_transfer_penalty",
                   "reference_attack_penalty", "min_reference_boundary",
                   "max_reference_boundary", "max_target_boundary", "min_attack_score",
                   "attack_limit", "alignment_group_penalty")
    diagnostics = {key: float(result[key].detach()) for key in scalar_keys}
    diagnostics.update(arm=arm, alpha=float(alpha), attack_margin=float(margin),
                       comparison="strict >", score_tolerance=0.,
                       selected_objective_exactly_zero=diagnostics["penalty"] == 0.,
                       reference_envelope_exactly_zero=diagnostics["reference_envelope_penalty"] == 0.,
                       tie_gradient="stable canonical kth row; equal global extrema share gradient")
    values = leaf.detach()
    members = result["memberships"]
    attacks = values[members["attacks"]]
    diagnostics["references"], diagnostics["targets"], diagnostics["reference_target_pairs"] = [], [], []
    for name, boundaries in (("references", result["reference_boundaries"]),
                             ("targets", result["target_boundaries"])):
        for (item, indices), threshold in zip(members[name], boundaries.detach()):
            scores_here = values[indices]
            row = {"id": item["id"], "source": item.get("source"),
                   "normal_count": len(indices), "integer_budget": budget(len(indices), alpha),
                   "boundary": float(threshold), "fp_at_own_boundary": int((scores_here > threshold).sum()),
                   "boundary_tie_count": int((scores_here == threshold).sum()),
                   "rank_from_largest": budget(len(indices), alpha) + 1,
                   "attack_count": len(attacks), "attack_tp": int((attacks > threshold).sum()),
                   "attack_fn": int((attacks <= threshold).sum()),
                   "minimum_attack_gap": float(attacks.min() - threshold),
                   "attack_margin_satisfied": bool(threshold <= result["attack_limit"].detach())}
            diagnostics[name].append(row)
    for (reference, _), threshold in zip(members["references"], result["reference_boundaries"].detach()):
        for target, indices in members["targets"]:
            fp = int((values[indices] > threshold).sum())
            limit = budget(len(indices), alpha)
            diagnostics["reference_target_pairs"].append({
                "reference_id": reference["id"], "target_id": target["id"],
                "reference_source": reference.get("source"), "target_source": target.get("source"),
                "normal_count": len(indices), "integer_budget": limit, "normal_fp": fp,
                "budget_excess": max(0, fp - limit), "budget_satisfied": fp <= limit,
                "reference_boundary": float(threshold)})
    diagnostics["all_reference_target_budgets_satisfied"] = all(
        pair["budget_satisfied"] for pair in diagnostics["reference_target_pairs"])
    diagnostics["all_reference_attack_margins_satisfied"] = all(
        row["attack_margin_satisfied"] for row in diagnostics["references"])
    diagnostics["objective_components"] = {name: float(result[key].detach()) for name, key in
        (("D", "reference_transfer_penalty"), ("H", "reference_attack_penalty"),
         ("G", "group_penalty"), ("R", "reference_envelope_penalty"),
         ("S", "alignment_group_penalty"))}
    diagnostics["all_scores_identical"] = bool(values.min() == values.max())
    diagnostics["all_normal_scores_identical"] = bool(
        values[torch.unique(torch.cat([ix for _, ix in members["references"]]))].unique().numel() == 1)
    diagnostics["normal_unique_scores"] = int(
        values[torch.unique(torch.cat([ix for _, ix in members["references"]]))].unique().numel())
    return derivative, diagnostics


def backward_objective(network, numeric, categorical, groups, *, arm, device,
                       batch_size, score_pass=None, alpha=.01, margin=1.,
                       weight=1., backward=True, resident=None):
    """Add the selected auxiliary derivatives to existing BCE gradients using exact block replay.

    The caller keeps input arrays fixed throughout discovery and replay. A reused
    score pass is checked against parameter/buffer versions and block size. Every
    active block is recomputed with the original shape and must match bitwise.
    """
    numeric, categorical = _arrays(numeric, categorical)
    if resident is not None:
        _check_resident(resident, numeric, categorical, device)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    discovered_here = score_pass is None
    if score_pass is None:
        score_pass = score_full(network, numeric, categorical, device=device, batch_size=batch_size,
                                resident=resident)
    if (score_pass.batch_size != batch_size or len(score_pass.scores) != len(numeric)
            or score_pass.state_versions != _versions(network)):
        raise RuntimeError("Stale or incompatible reference score pass")
    derivative, diagnostics = score_gradient(score_pass.scores, groups, arm, alpha=alpha, margin=margin)
    active = derivative.ne(0)
    replayed_rows = replayed_blocks = 0
    training = network.training
    network.eval()
    try:
        if backward and weight:
            for start, stop in score_pass.blocks:
                if not bool(active[start:stop].any()):
                    continue
                scores = (_block(network, numeric, categorical, start, stop, device) if resident is None
                          else _resident_block(network, resident, start, stop))
                if not torch.equal(scores.detach().cpu(), score_pass.scores[start:stop]):
                    raise RuntimeError("Replay score differs from frozen discovery score")
                (scores * derivative[start:stop].to(device) * weight).sum().backward()
                replayed_rows += stop - start
                replayed_blocks += 1
    finally:
        network.train(training)
    if score_pass.state_versions != _versions(network):
        raise RuntimeError("Model state changed during reference replay")
    diagnostics["cost"] = {
        "discovery_forward_rows_this_call": len(numeric) if discovered_here else 0,
        "discovery_forward_rows_in_score_pass": score_pass.forward_rows,
        "replay_forward_rows": replayed_rows, "replay_backward_rows": replayed_rows,
        "replay_blocks": replayed_blocks, "nonzero_score_derivatives": int(active.sum())}
    diagnostics["applied_weight"] = float(weight) if backward else 0.
    return diagnostics
