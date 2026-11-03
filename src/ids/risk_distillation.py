"""V3 design prototype: empirical anchor-relative risk constraints for KD.

This module never loads data or calibrates a deployment threshold. Its bounds
are pointwise statements about supplied, frozen margin decisions (>= 0), not
population, distribution-shift, optimizer-convergence or exported-score claims.
The constraint pool participates in training and is NOT an independent audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def _number(value, name, *, minimum=0.0, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if (value <= minimum if positive else value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _vector(value, name, *, length=None, device=None, floating=False):
    if (not isinstance(value, Tensor) or value.ndim != 1 or not value.numel()
            or (length is not None and len(value) != length)
            or (device is not None and value.device != device)):
        raise ValueError(f"{name} must be a nonempty aligned one-dimensional tensor on the same device")
    if floating and not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _ids(values, length):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != length:
        raise ValueError("sample_ids must identify every row in order")
    result = tuple(values)
    if any(not isinstance(x, str) or not x.strip() for x in result) or len(set(result)) != length:
        raise ValueError("sample_ids must be unique nonempty strings")
    return result


def working_margins(logits, *, threshold, scale=1.):
    """Subtract a frozen raw-logit threshold in FLOAT64, retaining autograd.

    A Python float compared/subtracted directly with float32 logits may round
    a nextafter threshold back onto a tie. This helper prevents that conversion.
    It is still not a proof of equivalence to exported sigmoid probabilities.
    """
    z = _vector(logits, "logits", floating=True)
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold):
        raise ValueError("threshold must be a frozen finite raw-logit scalar")
    scale = _number(scale, "scale", positive=True)
    if not math.isfinite(1 / scale):
        raise ValueError("scale inverse is not finite")
    margins = (z.to(torch.float64) - threshold) / scale
    if not bool(torch.isfinite(margins).all()):
        raise ValueError("working margin became non-finite")
    return margins


def freeze_margin_threshold(logits, labels, max_fpr=.01, *, partition="reference"):
    """Finite raw-logit threshold allowing floor(alpha*n_normal) reference FPs.

    Ties at the next disallowed normal logit are rejected together. This controls
    only the supplied reference cohort, not C, validation or a new population.
    """
    if partition != "reference":
        raise ValueError("working-point threshold requires the training reference role")
    z = _vector(logits, "logits", floating=True)
    y = _vector(labels, "labels", length=len(z), device=z.device)
    if y.requires_grad or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("labels must be fixed binary 0/1")
    alpha = _number(max_fpr, "max_fpr", maximum=1)
    if alpha == 1:
        raise ValueError("finite working-point calibration requires max_fpr < 1")
    normal = z.detach().to(torch.float64)[y == 0]
    if not len(normal):
        raise ValueError("working-point calibration requires normal reference rows")
    allowed = math.floor(alpha * len(normal))
    boundary = normal.sort(descending=True).values[allowed]
    threshold = torch.nextafter(boundary, boundary.new_tensor(float("inf")))
    if not bool(torch.isfinite(threshold)):
        raise ValueError("working-point threshold is not finite; reject this score representation")
    actual = int((normal >= threshold).sum())
    return {"threshold": float(threshold), "max_fpr": alpha, "n_normal": len(normal),
            "allowed_false_positives": allowed, "false_positives": actual,
            "achieved_fpr": actual / len(normal),
            "decision_rule": "float64(raw_logit) >= frozen_raw_logit_threshold",
            "scope": "empirical_training_reference_only_not_population_guarantee"}


@dataclass(frozen=True)
class RiskConstraint:
    """normal_fpr is an absolute FPR ceiling; attack_loss is lost-TP mass.

    A group denotes a declared normal stratum or attack family. Duplicate hash
    groups are isolation units, NOT automatically appropriate risk strata.
    Both rates divide by ALL normals/attacks in the specified population.
    """

    name: str
    kind: str
    limit: float
    group: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or any(
                x not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for x in self.name):
            raise ValueError("constraint name must be a nonempty safe lowercase identifier")
        if self.kind not in {"normal_fpr", "attack_loss"}:
            raise ValueError("kind must be normal_fpr or attack_loss")
        _number(self.limit, "limit", maximum=1)
        if self.group is not None and (type(self.group) is not int or self.group < 0):
            raise ValueError("group must be None or a nonnegative integer")


@dataclass(frozen=True)
class RiskReference:
    constraints: tuple[RiskConstraint, ...]
    sample_ids: tuple[str, ...]
    anchor_margins: Tensor
    labels: Tensor
    members: tuple[Tensor, ...]
    margin_floor: float
    majorizer: str
    normal_repair_credit: bool
    signature: str


def _reference_digest(constraints, identities, anchor, labels, members, floor, majorizer, credit):
    digest = hashlib.sha256()
    metadata = {"constraints": [asdict(x) for x in constraints], "sample_ids": identities,
                "margin_floor": floor, "majorizer": majorizer, "normal_repair_credit": credit}
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    for value in (anchor, labels, *members):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _signature(value):
    if value is not None and (not isinstance(value, str) or len(value) != 64
                              or any(x not in "0123456789abcdef" for x in value)):
        raise ValueError("reference signature must be None or a lowercase SHA-256 string")
    return value


def build_risk_reference(anchor_margins, labels, constraints, *, sample_ids,
                         normal_strata=None, attack_families=None, margin_floor=.05,
                         majorizer="adaptive", normal_repair_credit=True,
                         partition="constraint_train", min_support=1):
    """Freeze a complete constraint-training population and its row identity.

    No random split, per-batch quantile or silent rare-group omission occurs.
    The caller must verify actual source/role isolation; names alone do not
    establish provenance. Near-boundary anchor slack is retained, never subtracted.
    """
    if partition != "constraint_train":
        raise ValueError("risk reference requires constraint_train, not a reserved or certificate pool")
    anchor = _vector(anchor_margins, "anchor_margins", floating=True)
    labels = _vector(labels, "labels", length=len(anchor), device=anchor.device)
    if labels.requires_grad or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError("labels must be fixed binary 0/1")
    identities = _ids(sample_ids, len(anchor))
    floor = _number(margin_floor, "margin_floor", positive=True)
    if majorizer not in {"adaptive", "fixed"}:
        raise ValueError("majorizer must be adaptive or fixed")
    if type(normal_repair_credit) is not bool:
        raise ValueError("normal_repair_credit must be boolean")
    if type(min_support) is not int or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    constraints = tuple(constraints)
    if not constraints or any(not isinstance(x, RiskConstraint) for x in constraints):
        raise ValueError("a nonempty sequence of RiskConstraint objects is required")
    if len({x.name for x in constraints}) != len(constraints):
        raise ValueError("constraint names must be unique")
    groups = {}
    for name, values in (("normal_fpr", normal_strata), ("attack_loss", attack_families)):
        if values is not None:
            _vector(values, name, length=len(anchor), device=anchor.device)
            if values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                raise ValueError("strata and families must use integer tensors")
            groups[name] = values.detach().clone()
    labels = labels.detach().clone().to(torch.int64)
    members = []
    for spec in constraints:
        mask = labels == (0 if spec.kind == "normal_fpr" else 1)
        if spec.group is not None:
            if spec.kind not in groups:
                raise ValueError("group-specific constraints require an explicit strata/family vector")
            mask = mask & (groups[spec.kind] == spec.group)
        if int(mask.sum()) < min_support:
            raise ValueError(f"constraint {spec.name} has insufficient support; do not silently omit it")
        members.append(mask)
    anchor = anchor.detach().clone().to(torch.float64)
    if not math.isfinite(1 / floor):
        raise ValueError("margin_floor inverse is not finite")
    signature = _reference_digest(constraints, identities, anchor, labels, members,
                                  floor, majorizer, normal_repair_credit)
    return RiskReference(constraints, identities, anchor, labels, tuple(members), floor,
                         majorizer, normal_repair_credit, signature)


@dataclass
class ConstraintReport:
    names: tuple[str, ...]
    values: Tensor
    violations: Tensor
    actual_values: Tensor
    baseline_slack: Tensor
    reference_signature: str
    constraints: tuple[RiskConstraint, ...]
    diagnostics: dict[str, dict[str, float | int]]

    def detached_metrics(self):
        return {name: {**self.diagnostics[name], "surrogate": float(self.values[i].detach()),
                       "violation": float(self.violations[i].detach()),
                       "actual": float(self.actual_values[i]),
                       "anchor_slack": float(self.baseline_slack[i])}
                for i, name in enumerate(self.names)}


def constraint_risks(student_margins, reference, *, sample_ids, partition="constraint_train"):
    """Compute full-population empirical bounds; no minibatch denominators.

    normal: FPR_B + mean((1-B) relu(1+m/d) - B clip(-m/k,0,1)).
    attack: mean(B relu(1-m/d)), with NO credit for newly rescued attacks.
    Thus actual FPR <= normal value and lost baseline-TP mass <= attack value.
    These remain valid for supplied margins regardless of training adaptation.
    They do not establish population bounds or survive a changed decision rule.
    """
    if partition != "constraint_train":
        raise ValueError("constraint gradients accept constraint_train only")
    if not isinstance(reference, RiskReference):
        raise TypeError("reference must be RiskReference")
    # A frozen dataclass does not freeze tensor contents, and .data mutations
    # can bypass PyTorch version counters. Rehash the COMPLETE reference here.
    # This O(n) CPU-copy cost is explicit prototype overhead, not a free check.
    if reference.signature != _reference_digest(
            reference.constraints, reference.sample_ids, reference.anchor_margins,
            reference.labels, reference.members, reference.margin_floor,
            reference.majorizer, reference.normal_repair_credit):
        raise ValueError("frozen risk reference was modified after its signature was recorded")
    margins = _vector(student_margins, "student_margins", length=len(reference.sample_ids), floating=True)
    if _ids(sample_ids, len(margins)) != reference.sample_ids:
        raise ValueError("sample_ids or their order differ from the frozen risk population")
    # Float64 reductions protect small-rate budgets; autograd returns gradients
    # through this cast to the student's original dtype/device.
    m = margins.to(torch.float64)
    a = reference.anchor_margins.to(m.device)
    baseline = a >= 0
    candidate = m.detach() >= 0
    d = (a.abs().clamp_min(reference.margin_floor) if reference.majorizer == "adaptive"
         else torch.full_like(a, reference.margin_floor))
    new_upper = (~baseline) * F.relu(1 + m / d)
    repair_lower = baseline * ((-m / reference.margin_floor).clamp(0, 1)
                               if reference.normal_repair_credit else torch.zeros_like(m))
    loss_upper = baseline * F.relu(1 - m / d)
    anchor_new = (~baseline) * F.relu(1 + a / d)
    anchor_lost = baseline * F.relu(1 - a / d)
    values, actuals, slacks, diagnostics = [], [], [], {}
    for spec, mask in zip(reference.constraints, reference.members):
        mask = mask.to(m.device)
        count = int(mask.sum())
        b = baseline[mask]
        s = candidate[mask]
        new = float(((~b) & s).to(torch.float64).mean())
        repaired = float((b & (~s)).to(torch.float64).mean())
        base_rate = float(b.to(torch.float64).mean())
        if spec.kind == "normal_fpr":
            # Reduce the pointwise NONNEGATIVE envelope, avoiding cancellation
            # between two global means at a strict small-rate boundary.
            value = (b.to(torch.float64) - repair_lower[mask] + new_upper[mask]).mean()
            actual = s.to(torch.float64).mean()
            slack = anchor_new[mask].mean()
            diagnostics[spec.name] = {"support": count, "baseline_fpr": base_rate,
                "new_fp_mass": new, "repaired_fp_mass": repaired,
                "new_fp_upper": float(new_upper[mask].detach().mean()),
                "repair_credit_lower": float(repair_lower[mask].detach().mean()),
                "weak_anchor_mass": float(((a[mask].abs() < reference.margin_floor) & (~b))
                                          .to(torch.float64).mean())}
        else:
            value = loss_upper[mask].mean()
            actual = (b & (~s)).to(torch.float64).mean()
            slack = anchor_lost[mask].mean()
            diagnostics[spec.name] = {"support": count, "baseline_tpr": base_rate,
                "baseline_tp_count": int(b.sum()), "no_anchor_tp_to_preserve": int(not bool(b.any())),
                "lost_tp_mass": repaired, "rescued_tp_mass": new,
                "student_tpr": float(s.to(torch.float64).mean()),
                "tpr_lower_from_surrogate": base_rate - float(value.detach()),
                "weak_anchor_mass": float(((a[mask].abs() < reference.margin_floor) & b)
                                          .to(torch.float64).mean())}
        values.append(value)
        actuals.append(actual)
        slacks.append(slack)
    values = torch.stack(values)
    actuals = torch.stack(actuals).detach()
    slacks = torch.stack(slacks).detach()
    limits = values.new_tensor([x.limit for x in reference.constraints])
    if not bool(torch.isfinite(values).all()):
        raise ValueError("risk surrogate overflowed; inspect margin scales and floor")
    return ConstraintReport(tuple(x.name for x in reference.constraints), values, values - limits,
                            actuals, slacks, reference.signature, reference.constraints, diagnostics)


def risk_lagrangian(report, dual_weights):
    """Detached nonnegative multipliers times raw constraint violations.

    Slack constraints may contribute a negative constant/value. This is not
    positive-only penalty training. It does not certify optimizer convergence.
    """
    if not isinstance(report, ConstraintReport):
        raise TypeError("report must be ConstraintReport")
    _vector(dual_weights, "dual_weights", length=len(report.names), floating=True)
    if not bool((dual_weights >= 0).all()):
        raise ValueError("dual_weights must be nonnegative")
    value = (dual_weights.detach().to(report.violations) * report.violations).sum()
    if not bool(torch.isfinite(value)):
        raise ValueError("risk Lagrangian became non-finite; inspect multipliers and margin scales")
    return value


class ProjectedDual:
    """Bounded projected ascent state; clipping is practical, not feasibility.

    Updates require a full reference report, not an arbitrary risk minibatch.
    The state binds the reference on first update. A runner must additionally
    checkpoint model/optimizer/RNG/sampler/source/data/threshold identities.
    """

    def __init__(self, constraints, *, step_size=.1, maximum=100., reference_signature=None):
        self.constraints = tuple(constraints)
        if (not self.constraints or any(not isinstance(x, RiskConstraint) for x in self.constraints)
                or len({x.name for x in self.constraints}) != len(self.constraints)):
            raise ValueError("unique nonempty constraints are required")
        self.step_size = _number(step_size, "step_size", positive=True)
        self.maximum = _number(maximum, "maximum", positive=True)
        self.reference_signature = _signature(reference_signature)
        self.weights = torch.zeros(len(self.constraints), dtype=torch.float64)
        self.updates = 0

    def update(self, report):
        if not isinstance(report, ConstraintReport) or report.constraints != self.constraints:
            raise ValueError("dual update must match the frozen constraint definitions")
        if self.reference_signature is not None and report.reference_signature != self.reference_signature:
            raise ValueError("dual update uses another frozen risk reference")
        values = report.violations.detach().to(device="cpu", dtype=torch.float64)
        if values.shape != self.weights.shape or not bool(torch.isfinite(values).all()):
            raise ValueError("invalid dual violations")
        updated = (self.weights + self.step_size * values).clamp(0, self.maximum)
        if not bool(torch.isfinite(updated).all()):
            raise ValueError("dual update became non-finite")
        self.weights = updated
        self.reference_signature = report.reference_signature
        self.updates += 1
        return self.weights.clone()

    def state_dict(self):
        return {"schema_version": 1, "constraints": [asdict(x) for x in self.constraints],
                "step_size": self.step_size, "maximum": self.maximum,
                "weights": self.weights.tolist(), "updates": self.updates,
                "reference_signature": self.reference_signature}

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if (not isinstance(state, dict) or set(state) != expected
                or type(state["schema_version"]) is not int or state["schema_version"] != 1):
            raise ValueError("dual checkpoint schema does not match")
        if state["constraints"] != [asdict(x) for x in self.constraints]:
            raise ValueError("dual checkpoint constraints or budgets differ")
        if state["step_size"] != self.step_size or state["maximum"] != self.maximum:
            raise ValueError("dual checkpoint optimizer settings differ")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("dual checkpoint update count is invalid")
        signature = _signature(state["reference_signature"])
        if state["updates"] > 0 and signature is None:
            raise ValueError("an updated dual checkpoint must remain bound to its reference")
        if self.reference_signature is not None and signature != self.reference_signature:
            raise ValueError("dual checkpoint belongs to another risk reference")
        try:
            weights = torch.tensor(state["weights"], dtype=torch.float64)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("dual checkpoint weights are invalid") from exc
        if (weights.shape != self.weights.shape or not bool(torch.isfinite(weights).all())
                or not bool(((weights >= 0) & (weights <= self.maximum)).all())):
            raise ValueError("dual checkpoint weights are invalid")
        if signature is None and bool((weights != 0).any()):
            raise ValueError("nonzero dual weights require a bound reference")
        self.weights, self.updates, self.reference_signature = weights, state["updates"], signature


@dataclass(frozen=True)
class BoundaryKDConfig:
    mode: str = "kd"
    kd_weight: float = .3
    positive_weight: float = 1.
    temperature: float = 1.

    def __post_init__(self):
        if self.mode not in {"hard", "kd", "matched_hard"}:
            raise ValueError("mode must be hard, kd or matched_hard")
        _number(self.kd_weight, "kd_weight")
        _number(self.positive_weight, "positive_weight", positive=True)
        temp = _number(self.temperature, "temperature", positive=True)
        if not math.isfinite(temp * temp) or temp * temp == 0 or not math.isfinite(1 / temp):
            raise ValueError("temperature scales are not representable")


def boundary_kd_objective(student_logits, labels, config, *, student_threshold,
                           student_scale=1., teacher_margins=None, partition="fit"):
    """Fixed hard term + additive working-point-margin KD on fitting data.

    Teacher margins are detached (logit score minus the teacher's own frozen
    logit threshold, divided by its frozen reference scale); their sigmoid is a soft ALARM
    target, not a calibrated attack posterior. OOF/context/reference provenance
    is a caller contract. matched_hard keeps the same coefficient/temperature
    but replaces the teacher alarm target with the true label.
    """
    if partition != "fit":
        raise ValueError("boundary KD objective accepts the fitting role only")
    if not isinstance(config, BoundaryKDConfig):
        raise TypeError("config must be BoundaryKDConfig")
    z = _vector(student_logits, "student_logits", floating=True)
    y = _vector(labels, "labels", length=len(z), device=z.device)
    if y.requires_grad or not bool(((y == 0) | (y == 1)).all()):
        raise ValueError("labels must be fixed binary values")
    student_margin = working_margins(z, threshold=student_threshold, scale=student_scale)
    y = y.to(z.dtype)
    weight = z.new_tensor(config.positive_weight)
    if not bool(torch.isfinite(weight)) or not bool(weight > 0):
        raise ValueError("positive_weight is not representable in student dtype")
    hard = F.binary_cross_entropy_with_logits(z, y, pos_weight=weight)
    soft = z.sum() * 0
    if config.mode != "hard" and config.kd_weight:
        if config.mode == "kd":
            teacher = _vector(teacher_margins, "teacher_margins", length=len(z), device=z.device, floating=True)
            target = torch.sigmoid(teacher.detach().to(torch.float64) / config.temperature)
        else:
            target = y.to(torch.float64)
        soft = config.temperature ** 2 * F.binary_cross_entropy_with_logits(
            student_margin / config.temperature, target)
    total = hard + config.kd_weight * soft
    if not all(bool(torch.isfinite(x)) for x in (hard, soft, total)):
        raise ValueError("boundary KD objective became non-finite")
    return {"hard": hard, "soft": soft, "total": total}
