"""Frozen-target loss and one update; no data loading or trajectory runner.

Rows are ordered A0, A1, B0, B1, with the same number of draws per block.
The auxiliary arrays have normal order A0/B0 and positive order A1/B1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Any

from ..model_training_redesign.training import four_group_loss
from ..route_batch.tabm_training import stable_scores


MEMBERS = 32


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _frozen_vector(value, size, name, *, integral=False):
    import torch

    # Construct in FP64 directly: Python float lists must not first round
    # through PyTorch's FP32 default before becoming frozen score anchors.
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.requires_grad:
        raise ValueError(f"{name} is a frozen target and must not require gradients")
    if result.shape != (size,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be an aligned finite vector")
    if integral and ((result < 1).any() or (result != result.round()).any()):
        raise ValueError(f"{name} must contain positive integer counts")
    return result.to(dtype=torch.int64 if integral else torch.float64).clone()


@dataclass(frozen=True)
class BatchTargets:
    """Copied fixed anchors for one four-group batch.

    normal_group_sizes are full-F leaf counts, not batch counts.
    profile_normal_counts are full-F A0/B0 counts. No auxiliary weight is
    passed to BCE. Only draws_per_group is reduced in synthetic tests.
    """

    normal_targets: Any
    positive_reference_scores: Any
    normal_group_sizes: Any
    profile_normal_counts: tuple[int, int]
    normal_group_count: int
    scale: float
    draws_per_group: int = 64

    def __post_init__(self):
        b = _positive_integer(self.draws_per_group, "draws_per_group")
        groups = _positive_integer(self.normal_group_count, "normal_group_count")
        if len(self.profile_normal_counts) != 2:
            raise ValueError("Expected full-F normal counts for A and B")
        counts = tuple(_positive_integer(n, "profile_normal_counts") for n in self.profile_normal_counts)
        # Converting a differentiable tensor to float would silently remove the
        # requirement that Delta is a frozen, external scalar.
        if getattr(self.scale, "requires_grad", False) or isinstance(self.scale, bool):
            raise ValueError("scale must be a frozen scalar at least 1")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("scale must be a finite frozen scalar at least 1")
        object.__setattr__(self, "draws_per_group", b)
        object.__setattr__(self, "normal_group_count", groups)
        object.__setattr__(self, "profile_normal_counts", counts)
        object.__setattr__(self, "scale", scale)
        for name in ("normal_targets", "positive_reference_scores", "normal_group_sizes"):
            values = _frozen_vector(getattr(self, name), 2 * b, name, integral=name == "normal_group_sizes")
            object.__setattr__(self, name, values)
        if (self.normal_group_sizes > sum(counts)).any():
            raise ValueError("A normal group cannot exceed the full-F normal population")


def _validate_logits(logits, draws_per_group):
    import torch

    b = _positive_integer(draws_per_group, "draws_per_group")
    if (logits.shape != (4 * b, MEMBERS, 1) or not logits.is_floating_point()
            or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite floating logits (4*draws_per_group,32,1)")
    return b


def member_bce(logits, labels, *, draws_per_group=64):
    """Original independent-member BCE; validate the fixed source/class order."""
    import torch

    b = _validate_logits(logits, draws_per_group)
    expected = torch.tensor([0, 1, 0, 1], device=logits.device, dtype=logits.dtype).repeat_interleave(b)
    if labels.shape != (4 * b,) or labels.device != logits.device or not torch.equal(labels, expected):
        raise ValueError("Labels must follow the equally sized A0/A1/B0/B1 blocks")
    return four_group_loss(logits, labels, b)[0]


@dataclass
class AuxiliaryLosses:
    normal: Any
    positive: Any

    @property
    def total(self):
        return self.normal + self.positive


def auxiliary_losses(logits, targets: BatchTargets):
    """Actual mean-probability score; both auxiliary coefficients are exactly 1."""
    import torch

    b = _validate_logits(logits, targets.draws_per_group)
    scores = stable_scores(logits)  # FP64 logsigmoid/logsumexp, still connected to logits.
    normal_scores = torch.cat((scores[:b], scores[2 * b:3 * b]))
    positive_scores = torch.cat((scores[b:2 * b], scores[3 * b:]))
    normal_targets = targets.normal_targets.to(device=logits.device)
    positive_reference = targets.positive_reference_scores.to(device=logits.device)
    group_sizes = targets.normal_group_sizes.to(device=logits.device, dtype=torch.float64)
    profile_counts = torch.tensor(targets.profile_normal_counts, device=logits.device,
                                  dtype=torch.float64).repeat_interleave(b)
    weight = profile_counts / (b * targets.normal_group_count * group_sizes)
    # SUM is essential: weight already includes 1/b. No /128 or /256 follows.
    normal = (weight * ((normal_scores - normal_targets) / targets.scale).square()).sum()
    # Equal A1/B1 block sizes make this mean exactly 0.5*mean_A1 + 0.5*mean_B1.
    positive = torch.relu((positive_reference - positive_scores) / targets.scale).square().mean()
    return AuxiliaryLosses(normal=normal, positive=positive)


def _rng_state(parameters):
    import torch

    cuda_devices = sorted({p.device.index for p in parameters if p.device.type == "cuda"})
    return (torch.get_rng_state().clone(),
            {device: torch.cuda.get_rng_state(device).clone() for device in cuda_devices})


def _restore_rng(state):
    import torch

    torch.set_rng_state(state[0])
    for device, value in state[1].items():
        torch.cuda.set_rng_state(value, device)


def _same_rng(left, right):
    import torch

    return (torch.equal(left[0], right[0]) and left[1].keys() == right[1].keys()
            and all(torch.equal(left[1][device], right[1][device]) for device in left[1]))


def training_step(network, optimizer, numeric, categorical, labels, targets: BatchTargets):
    """One train-BCE backward + eval-aux backward, then one clip and one step.

    No historical weights or data are read here. The caller owns the immutable
    anchors, sampler, optimizer configuration, trajectory budget and selection.
    Original module modes are restored on success or failure. RNG consumed by
    BCE remains consumed; the auxiliary path must leave that RNG state intact.
    """
    import torch

    parameters = list(network.parameters())
    modes = [(module, module.training) for module in network.modules()]
    try:
        optimizer.zero_grad(set_to_none=True)
        network.train()
        bce = member_bce(network(numeric, categorical), labels,
                         draws_per_group=targets.draws_per_group)
        if not torch.isfinite(bce):
            raise FloatingPointError("Nonfinite member BCE")
        bce.backward()

        # The BCE graph is released before allocating the auxiliary graph.
        # Do not zero gradients here: both contributions belong to one update.
        auxiliary_rng = _rng_state(parameters)
        try:
            network.eval()
            auxiliary = auxiliary_losses(network(numeric, categorical), targets)
            if not torch.isfinite(auxiliary.total):
                raise FloatingPointError("Nonfinite auxiliary loss")
            auxiliary.total.backward()
        finally:
            unchanged = _same_rng(auxiliary_rng, _rng_state(parameters))
            _restore_rng(auxiliary_rng)
        if not unchanged:
            raise RuntimeError("Eval-mode auxiliary forward/backward consumed training RNG")

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError("Nonfinite parameters after the optimizer update")
        return {
            "member_bce": float(bce.detach()),
            "normal": float(auxiliary.normal.detach()),
            "positive": float(auxiliary.positive.detach()),
            "total": float(bce.detach() + auxiliary.total.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "auxiliary_rng_unchanged": True,
            "optimizer_steps": 1,
        }
    finally:
        for module, original_mode in modes:
            module.training = original_mode
