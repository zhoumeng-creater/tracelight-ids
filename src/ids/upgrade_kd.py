"""Standard raw-logit binary KD controls; no new distillation mechanism.

The teacher's RAW prediction logits are matched to the student's RAW logits.
No reference/alarm threshold is subtracted. This is ordinary forward Bernoulli
KL with an explicit T**2 factor, not calibrated-posterior or low-FPR assurance.
Hard supervision, teacher fitting, class-prior corrections and optimizers are
caller responsibilities. See local/risk-v3-return-audit-20261103/kd_review.md.

Class-balanced reduction is .5*(mean_normal + mean_attack), NOT their sum.
It changes the empirical mixture and scale relative to ordinary row mean.
Neither reduction silently applies the hard objective's positive-class weight.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


def _vector(value, name, *, length=None, device=None):
    if (not isinstance(value, Tensor) or value.ndim != 1 or not value.numel()
            or not value.is_floating_point()
            or (length is not None and len(value) != length)
            or (device is not None and value.device != device)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be an aligned nonempty finite floating vector on the same device")
    return value


def _scalar(value, name, *, nonnegative=False, positive=False):
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(f"{name} must be a finite frozen scalar")
        value = float(value.detach())
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite frozen scalar")
    result = float(value)
    if (nonnegative and result < 0) or (positive and result <= 0):
        raise ValueError(f"{name} has an invalid sign")
    return result


def _temperature(value):
    result = _scalar(value, "temperature", positive=True)
    square = result * result
    if not math.isfinite(square) or square == 0 or not math.isfinite(1 / result):
        raise ValueError("temperature scales must be representable")
    return result


def _labels(labels, student):
    if (not isinstance(labels, Tensor) or labels.ndim != 1 or len(labels) != len(student)
            or labels.device != student.device or labels.dtype == torch.bool or labels.is_complex()
            or not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all())):
        raise ValueError("labels must be an aligned finite binary numeric vector on the same device")
    return labels.detach().to(torch.int64)


def raw_bernoulli_kl_rows(student_logits, *, teacher_logits, temperature=1.,
                          teacher_logit_shift=0., partition="fit"):
    """Return [N] T**2*KL(teacher || student), WITHOUT beta or any reduction.

    Teacher, temperature and additive teacher-logit shift are fixed/detached.
    Float64 logsigmoid arithmetic preserves gradients to the original student.
    Shift is applied BEFORE division by T; its default is zero. No probability
    clipping or hidden offset is used. For chunked reductions the caller uses
    full-population denominators, not chunk means or per-fold class proportions.
    """
    if partition != "fit":
        raise ValueError("raw KD accepts the fitting role only")
    student = _vector(student_logits, "student_logits")
    teacher = _vector(teacher_logits, "teacher_logits", length=len(student), device=student.device)
    temp = _temperature(temperature)
    shift = _scalar(teacher_logit_shift, "teacher_logit_shift")
    s = student.to(torch.float64) / temp
    t = (teacher.detach().to(torch.float64) + shift) / temp
    if not bool(torch.isfinite(s).all() and torch.isfinite(t).all()):
        raise ValueError("temperature-scaled logits must be finite")
    log_s = torch.stack((F.logsigmoid(-s), F.logsigmoid(s)), dim=1)
    log_t = torch.stack((F.logsigmoid(-t), F.logsigmoid(t)), dim=1)
    rows = temp ** 2 * (log_t.exp() * (log_t - log_s)).sum(dim=1)
    if not bool(torch.isfinite(rows).all()):
        raise ValueError("raw binary KL became non-finite")
    return rows


def raw_bernoulli_kl(student_logits, labels, *, teacher_logits, class_balanced=False,
                     coefficient=1., temperature=1., teacher_logit_shift=0., partition="fit"):
    """Return additive soft scalar plus detached reduction/provenance statistics.

    False: coefficient/N * sum_i row_KL_i.
    True: coefficient/2 * (sum_normal/N0 + sum_attack/N1).
    Balanced mode requires BOTH classes; an absent class is an error rather than
    an implicit change of estimand. Use the rows API with global counts when a
    chunk lacks one class. Labels are validated and detached, never soft targets.
    """
    if type(class_balanced) is not bool:
        raise ValueError("class_balanced must be boolean")
    rows = raw_bernoulli_kl_rows(student_logits, teacher_logits=teacher_logits, temperature=temperature,
                                teacher_logit_shift=teacher_logit_shift, partition=partition)
    y = _labels(labels, student_logits)
    factor = _scalar(coefficient, "coefficient", nonnegative=True)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if class_balanced and (not n0 or not n1):
        raise ValueError("class-balanced reduction requires both classes; use rows with global counts for chunks")
    mean0 = rows[y == 0].mean() if n0 else None
    mean1 = rows[y == 1].mean() if n1 else None
    raw = .5 * (mean0 + mean1) if class_balanced else rows.mean()
    soft = factor * raw
    if not bool(torch.isfinite(soft)):
        raise ValueError("reduced raw KL became non-finite")
    per_row0 = .5 / n0 if class_balanced else 1 / len(y)
    per_row1 = .5 / n1 if class_balanced else 1 / len(y)
    stats = {"method": "standard_raw_bernoulli_forward_kl", "kl_direction": "teacher_to_student",
             "probability_units": "raw_prediction_logits_not_alarm_margins", "rows": len(y),
             "normal_rows": n0, "attack_rows": n1, "class_balanced": class_balanced,
             "reduction": "half_sum_of_class_means" if class_balanced else "all_row_mean",
             "normal_total_weight_before_coefficient": n0 * per_row0,
             "attack_total_weight_before_coefficient": n1 * per_row1,
             "normal_row_weight_before_coefficient": per_row0 if n0 else None,
             "attack_row_weight_before_coefficient": per_row1 if n1 else None,
             "temperature": _temperature(temperature), "temperature_squared_factor": True,
             "teacher_logit_shift": _scalar(teacher_logit_shift, "teacher_logit_shift"),
             "coefficient": factor, "hard_positive_weight_applied": False,
             "normal_mean_t2_kl": float(mean0.detach()) if mean0 is not None else None,
             "attack_mean_t2_kl": float(mean1.detach()) if mean1 is not None else None,
             "soft": float(soft.detach())}
    return {"soft": soft, "stats": stats}
