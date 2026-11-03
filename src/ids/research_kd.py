"""Train-batch objectives for prospective KD research, not efficacy claims.

No function loads a partition or accesses validation/test. ``partition='train'``
is an explicit misuse guard, not proof of provenance: callers must first verify
sample identities and OOF teacher isolation. Teacher values are attack scores,
not calibrated probabilities. Temperature rescales their *score log odds*.
Existing published model/protocol implementations are deliberately unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


_MODES = frozenset({"hard", "matched_hard", "kd", "gated_kd", "gated_matched_hard"})
_GATE_POLICIES = frozenset({"correct_confidence", "correct_only", "confidence_only"})


def _number(value: Any, name: str, *, lower: float, upper: float | None = None,
            strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if (value <= lower if strict else value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} is outside its allowed range")
    return value


@dataclass(frozen=True)
class ResearchKDConfig:
    """All weights are explicit; reductions are batch means, never renormalized.

    ``matched_hard`` substitutes q=y into exactly the ordinary KD formula.
    At T=1 its effective attack weight is (1-alpha)*positive_weight+alpha,
    and its normal weight is 1. At other T it also preserves the KD logit
    scaling and T**2 multiplier, so there is no equivalent single BCE weight.

    Default gating uses correct teacher score side of 0.5 and confidence
    |2q-1|. Scores on the boundary are inactive. Active rows have g=|2q-1|;
    rows below ``gate_min_confidence`` have g=0. ``correct_only`` keeps this
    eligibility but assigns g=1. The explicit ``confidence_only`` ablation
    assigns g=|2q-1| above the confidence threshold even to wrong teachers.
    It does NOT protect students from incorrect teachers. The hard/soft
    coefficients are 1-alpha*g and alpha*g; rejected teachers restore the
    original hard objective rather than downweighting or removing that row.

    ``gated_matched_hard`` obtains exactly the same g from real teacher scores
    as ``gated_kd``, then substitutes y for the soft target. It controls both
    the gate selection and the gate-dependent, per-row effective weights.
    Both gated modes therefore require the real OOF teacher scores.

    Pair ranking selects the highest detached-score normals and lowest-score
    attacks within the supplied train batch. Each selected attack group and
    normal group gets equal aggregate mass when integer ``group_ids`` are
    supplied; this is selected-batch group balancing, not population weighting.
    """

    mode: str = "hard"
    alpha: float = 0.5
    positive_weight: float = 1.0
    temperature: float = 1.0
    gate_policy: str = "correct_confidence"
    gate_min_confidence: float = 0.5
    pairwise_weight: float = 0.0
    normal_fraction: float = 0.1
    max_normals: int = 32
    max_attacks: int = 64
    max_pairs: int = 2048
    pairwise_margin: float = 0.5
    pairwise_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}")
        if not isinstance(self.gate_policy, str) or self.gate_policy not in _GATE_POLICIES:
            raise ValueError(f"gate_policy must be one of {sorted(_GATE_POLICIES)}")
        _number(self.alpha, "alpha", lower=0, upper=1)
        _number(self.positive_weight, "positive_weight", lower=0, strict=True)
        temperature = _number(self.temperature, "temperature", lower=0, strict=True)
        if (not math.isfinite(temperature * temperature) or temperature * temperature == 0
                or not math.isfinite(1 / temperature)):
            raise ValueError("temperature or its squared/inverse scale is not representable")
        _number(self.gate_min_confidence, "gate_min_confidence", lower=0, upper=1)
        _number(self.pairwise_weight, "pairwise_weight", lower=0)
        _number(self.normal_fraction, "normal_fraction", lower=0, upper=1, strict=True)
        _number(self.pairwise_margin, "pairwise_margin", lower=0)
        _number(self.pairwise_temperature, "pairwise_temperature", lower=0, strict=True)
        for name in ("max_normals", "max_attacks", "max_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class LossBreakdown:
    """Differentiable scalar components and detached, JSON-safe batch counts."""

    total: Tensor
    components: dict[str, Tensor]
    statistics: dict[str, int | float]

    def detached_metrics(self) -> dict[str, int | float]:
        """Batch summaries; the caller must declare its epoch aggregation rule."""
        return {**{key: float(value.detach().item()) for key, value in self.components.items()},
                **self.statistics}


def _vector(value: Tensor, name: str, logits: Tensor) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.shape != logits.shape:
        raise ValueError(f"{name} must be a one-dimensional tensor aligned with logits")
    if value.device != logits.device:
        raise ValueError(f"{name} must be on the same device as logits")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _group_weights(indices: Tensor, group_ids: Tensor | None, logits: Tensor) -> tuple[Tensor, int]:
    if group_ids is None:
        return logits.new_full((len(indices),), 1 / len(indices)), len(indices)
    _, inverse, counts = torch.unique(group_ids[indices], return_inverse=True, return_counts=True)
    return 1 / (len(counts) * counts[inverse].to(logits.dtype)), len(counts)


def _pairwise(logits: Tensor, labels: Tensor, settings: ResearchKDConfig,
              group_ids: Tensor | None) -> tuple[Tensor, dict[str, int | float]]:
    zero = logits.sum() * 0
    stats: dict[str, int | float] = {
        "pairwise_pairs": 0, "pairwise_selected_normals": 0, "pairwise_selected_attacks": 0,
        "pairwise_normal_groups": 0, "pairwise_attack_groups": 0,
    }
    if settings.pairwise_weight == 0:
        return zero, stats
    normal = torch.nonzero(labels == 0, as_tuple=True)[0]
    attack = torch.nonzero(labels == 1, as_tuple=True)[0]
    if not len(normal) or not len(attack):
        return zero, stats
    detached = logits.detach()
    n_normal = min(math.ceil(settings.normal_fraction * len(normal)),
                   settings.max_normals, settings.max_pairs)
    n_attack = min(len(attack), settings.max_attacks, settings.max_pairs // n_normal)
    normal = normal[torch.argsort(detached[normal], descending=True, stable=True)[:n_normal]]
    attack = attack[torch.argsort(detached[attack], descending=False, stable=True)[:n_attack]]
    normal_weights, normal_groups = _group_weights(normal, group_ids, logits)
    attack_weights, attack_groups = _group_weights(attack, group_ids, logits)
    difference = logits[attack, None] - logits[normal][None, :]
    scale = settings.pairwise_temperature
    margins = F.softplus((settings.pairwise_margin - difference) / scale) * scale
    loss = (margins * attack_weights[:, None] * normal_weights[None, :]).sum()
    stats.update({"pairwise_pairs": n_normal * n_attack,
                  "pairwise_selected_normals": n_normal, "pairwise_selected_attacks": n_attack,
                  "pairwise_normal_groups": normal_groups, "pairwise_attack_groups": attack_groups})
    return loss, stats


def research_kd_loss(student_logits: Tensor, train_labels: Tensor, config: ResearchKDConfig,
                     *, teacher_scores: Tensor | None = None, group_ids: Tensor | None = None,
                     partition: str = "train") -> LossBreakdown:
    """Compute train-only hard/KD plus optional capped pairwise smooth margin.

    No teacher gradient is propagated. ``teacher_scores`` must already be
    verified OOF scores aligned to these training rows. A tensor cannot prove
    that contract; identity verification belongs to the orchestration layer.
    Supplying a reserved-partition name fails before inspecting its tensors.

    Components (scalars): ``hard``, ``soft``, ``hard_contribution``,
    ``soft_contribution``, ``base``, ``pairwise``, ``pairwise_contribution``,
    ``total``. ``soft`` includes temperature-squared scaling. Gated terms use
    the whole-batch denominator, with no active-gate normalization. Pairwise
    difficulty mining is local to this batch; it does not guarantee a target
    FPR, and is not final threshold calibration. Gate statistics are zero for
    non-gated modes. ``gate_wrong_mass`` is the sum of gate coefficients on
    wrong-side teachers, before multiplying by alpha; it is not a probability.
    """
    if partition != "train":
        raise ValueError("Research losses and hard-negative mining accept train batches only")
    if not isinstance(config, ResearchKDConfig):
        raise TypeError("config must be ResearchKDConfig")
    logits = student_logits
    if (not isinstance(logits, Tensor) or logits.ndim != 1 or not logits.numel()
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())):
        raise ValueError("student_logits must be a nonempty finite one-dimensional floating tensor")
    labels = _vector(train_labels, "train_labels", logits)
    if labels.requires_grad or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("train_labels must be fixed binary 0/1 values")
    labels = labels.to(logits.dtype)
    if group_ids is not None:
        _vector(group_ids, "group_ids", logits)
        if group_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("group_ids must use an integer tensor dtype")
    teacher = None
    if teacher_scores is not None:
        teacher = _vector(teacher_scores, "teacher_scores", logits)
        if not teacher.is_floating_point() or not bool(((teacher >= 0) & (teacher <= 1)).all()):
            raise ValueError("teacher_scores must be floating attack scores in [0,1]")
        teacher = teacher.detach().to(logits.dtype)
    if config.mode in {"kd", "gated_kd", "gated_matched_hard"} and teacher is None:
        raise ValueError("KD modes require aligned OOF teacher_scores")
    positive_weight = logits.new_tensor(config.positive_weight)
    if not bool(torch.isfinite(positive_weight)) or not bool(positive_weight > 0):
        raise ValueError("positive_weight must remain positive and finite in the logits dtype")
    if config.mode != "hard":
        temperature_scale = logits.new_tensor(config.temperature ** 2)
        if not bool(torch.isfinite(temperature_scale)) or not bool(temperature_scale > 0):
            raise ValueError("temperature squared must remain positive and finite in the logits dtype")
    hard_rows = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight,
                                                  reduction="none")
    hard = hard_rows.mean()
    soft = logits.sum() * 0
    gate = torch.zeros_like(logits)
    correct = torch.zeros_like(labels, dtype=torch.bool)
    if config.mode == "hard":
        hard_contribution, soft_contribution = hard, soft
    else:
        targets = labels if config.mode in {"matched_hard", "gated_matched_hard"} else teacher
        temperature = config.temperature
        # logit(0/1) is +/-inf, whose sigmoid is exactly 0/1: no hidden clipping.
        tempered = (targets if temperature == 1 else torch.sigmoid(torch.logit(targets) / temperature))
        soft_rows = F.binary_cross_entropy_with_logits(logits / temperature, tempered,
                                                      reduction="none") * temperature ** 2
        soft = soft_rows.mean()
        if config.mode in {"gated_kd", "gated_matched_hard"}:
            confidence = (2 * teacher - 1).abs()
            correct = ((teacher > .5) & (labels == 1)) | ((teacher < .5) & (labels == 0))
            eligible = confidence >= config.gate_min_confidence
            if config.gate_policy != "confidence_only":
                eligible = eligible & correct
            magnitude = (torch.ones_like(confidence) if config.gate_policy == "correct_only"
                         else confidence)
            gate = torch.where(eligible, magnitude, torch.zeros_like(confidence))
        else:
            gate = torch.ones_like(logits)
        soft_coefficient = config.alpha * gate
        hard_contribution = ((1 - soft_coefficient) * hard_rows).mean()
        soft_contribution = (soft_coefficient * soft_rows).mean()
    base = hard_contribution + soft_contribution
    pairwise, pair_stats = _pairwise(logits, labels, config, group_ids)
    pairwise_contribution = config.pairwise_weight * pairwise
    total = base + pairwise_contribution
    components = {"hard": hard, "soft": soft, "hard_contribution": hard_contribution,
                  "soft_contribution": soft_contribution, "base": base, "pairwise": pairwise,
                  "pairwise_contribution": pairwise_contribution, "total": total}
    if not all(bool(torch.isfinite(value)) for value in components.values()):
        raise ValueError("Research loss became non-finite; inspect scales and logits")
    if config.mode not in {"gated_kd", "gated_matched_hard"}:
        gate = torch.zeros_like(gate)
    statistics = {"n_samples": len(labels), "n_normals": int((labels == 0).sum().item()),
                  "n_attacks": int((labels == 1).sum().item()),
                  "gate_correct_count": int(correct.sum().item()),
                  "gate_active_count": int((gate > 0).sum().item()),
                  "gate_mean": float(gate.mean().item()),
                  "gate_max": float(gate.max().item()),
                  "gate_total_mass": float(gate.sum().item()),
                  "gate_correct_mass": float(gate[correct].sum().item()),
                  "gate_wrong_mass": float(gate[~correct].sum().item()),
                  "gate_wrong_active_count": int(((gate > 0) & ~correct).sum().item()), **pair_stats}
    return LossBreakdown(total, components, statistics)
