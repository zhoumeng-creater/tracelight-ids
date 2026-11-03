"""Clean-room scalar-logit controls grounded in PCT, MPT and EKD equations.

These functions return ONLY additive soft regularizers and detached statistics.
The caller supplies identical hard-label F/C supervision, frozen reference
outputs/thresholds and training budgets for every control. No loading, fitting,
threshold estimation, class-incremental bias or optimizer is hidden here.

Sources and adaptations (author code was read, not copied):
* PCT, CVPR 2021, section 4.2: https://arxiv.org/html/2011.09161v3
  The paper's weighted-row intent is used, with positive KL(anchor || student)
  and T**2 normalization, not the [N,1]*[N] broadcasting behavior in the authors'
  later unified code at fb2268c67706834632ec1427945fcf30a7218227.
* MPT, AAAI 2026: https://arxiv.org/html/2511.08322v2
  DOUBLE-SOURCE BINARY ADAPTATION ONLY, without new-class logit bias or restricted
  old-class heads. Both sources use full-row correctness masks. T**2 is explicit;
  author loss.py at 2b2d7b5661c58f2da3d6c34cbb7a4a1c19f4ed9e omits it.
* EKD, CVPR 2022, equations 10-11: https://arxiv.org/html/2206.02325
  Single-row binary adaptation of sigmoid rank-COUNT MSE. Author TFace code at
  171ec1434aca0972e111663ac646c9c1e3e44b5b uses face pairs, running thresholds
  and hard-negative mining. Here thresholds must be frozen beforehand and no
  mining occurs. This is NOT a reproduction of the original face-recognition task.

See docs/research_alignment_20261103/pct_mpt_code_audit.md for source differences.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor
from torch.nn import functional as F


def _fit_role(partition):
    if partition != "fit":
        raise ValueError("baseline soft objectives accept the fitting role only")


def _vector(value, name, *, length=None, device=None):
    if (not isinstance(value, Tensor) or value.ndim != 1 or not value.numel()
            or not value.is_floating_point()
            or (length is not None and len(value) != length)
            or (device is not None and value.device != device)
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be an aligned nonempty finite floating vector on the same device")
    return value


def _labels(value, length, device):
    if (not isinstance(value, Tensor) or value.ndim != 1 or len(value) != length
            or value.device != device or value.dtype == torch.bool or value.is_complex()
            or not bool(torch.isfinite(value).all()) or not bool(((value == 0) | (value == 1)).all())):
        raise ValueError("labels must be an aligned finite binary numeric vector")
    return value.detach().to(torch.int64)


def _constant(value, name, like):
    """Nonnegative scalar coefficient, explicitly detached even if supplied as Tensor."""
    if isinstance(value, Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(f"{name} must be a finite nonnegative scalar")
        result = value.detach().to(device=like.device, dtype=torch.float64)
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    else:
        result = like.new_tensor(float(value), dtype=torch.float64)
    if not bool(torch.isfinite(result)) or not bool(result >= 0):
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return result


def _temperature(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("temperature must be a finite positive frozen scalar")
    value = float(value)
    if not math.isfinite(value * value) or value * value == 0 or not math.isfinite(1 / value):
        raise ValueError("temperature scales must be representable")
    return value


def _threshold(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite frozen raw-logit scalar")
    return float(value)


def _correctness(logits, labels, threshold, tie_positive):
    if type(tie_positive) is not bool:
        raise ValueError("correctness_tie_positive must be boolean")
    # Cast BEFORE comparison so nextafter thresholds are not rounded to float32.
    score = logits.detach().to(torch.float64)
    decision = score >= threshold if tie_positive else score > threshold
    return (decision == labels).detach()


def _forward_kl(student, teacher, temperature):
    """Rowwise KL(Bernoulli(sigmoid(teacher/T)) || Bernoulli(sigmoid(student/T)))."""
    s, t = student.to(torch.float64) / temperature, teacher.detach().to(torch.float64) / temperature
    if not bool(torch.isfinite(s).all() and torch.isfinite(t).all()):
        raise ValueError("temperature-scaled logits are non-finite")
    # Stable two-class log probabilities; [-z/2,z/2] has exactly this distribution.
    log_s = torch.stack((F.logsigmoid(-s), F.logsigmoid(s)), dim=1)
    log_t = torch.stack((F.logsigmoid(-t), F.logsigmoid(t)), dim=1)
    row_kl = (log_t.exp() * (log_t - log_s)).sum(dim=1)
    if not bool(torch.isfinite(row_kl).all()):
        raise ValueError("binary KL is non-finite; inspect score scales")
    return row_kl


def _focal_term(student, labels, source, *, threshold, alpha, beta, temperature, coefficient, tie_positive):
    correct = _correctness(source, labels, threshold, tie_positive)
    a, b = _constant(alpha, "alpha", student), _constant(beta, "beta", student)
    weight = a + b * correct.to(torch.float64)
    factor = _constant(coefficient, "coefficient", student)
    row_kl = _forward_kl(student, source, temperature)
    # Both are [N], never [N,1] times [N]. Divide by ALL rows, not mask support or sum(weights).
    raw = (weight * row_kl).mean()
    soft = factor * temperature ** 2 * raw
    stats = {"rows": len(student), "correct_rows": int(correct.sum()),
             "correct_fraction": float(correct.to(torch.float64).mean()),
             "alpha": float(a), "beta": float(b), "coefficient": float(factor),
             "mean_row_weight": float(weight.mean()), "unweighted_kl_mean": float(row_kl.detach().mean()),
             "weighted_kl_mean_before_t2": float(raw.detach()), "soft": float(soft.detach()),
             "threshold": threshold, "denominator": "all_input_rows",
             "correctness_rule": "logit>=threshold" if tie_positive else "logit>threshold"}
    return soft, stats


def _result(soft, stats):
    if not bool(torch.isfinite(soft)):
        raise ValueError("baseline soft objective is non-finite; inspect coefficients and scores")
    return {"soft": soft, "stats": stats}


def pct_focal_kl(student_logits, labels, *, anchor_logits, anchor_threshold=0., alpha=1., beta=5.,
                 temperature=1., coefficient=1., correctness_tie_positive=True, partition="fit"):
    """PCT-intent binary FD: coefficient*T**2*mean_N[(alpha+beta*old_correct)*KL(B||S)].

    KL uses RAW logits, not logits minus the correctness threshold. The frozen
    old working point controls ONLY the old-correct mask. threshold=0 and
    correctness_tie_positive=False exactly match argmax([-z/2,z/2]), including
    its tie-to-class-0 convention. Defaults instead follow this repo's >= rule.
    Labels, anchor, alpha/beta and coefficient are detached. Hard loss is absent.
    """
    _fit_role(partition)
    student = _vector(student_logits, "student_logits")
    anchor = _vector(anchor_logits, "anchor_logits", length=len(student), device=student.device)
    y = _labels(labels, len(student), student.device)
    temp = _temperature(temperature)
    threshold = _threshold(anchor_threshold, "anchor_threshold")
    soft, source_stats = _focal_term(student, y, anchor, threshold=threshold, alpha=alpha, beta=beta,
                                   temperature=temp, coefficient=coefficient,
                                   tie_positive=correctness_tie_positive)
    return _result(soft, {"method": "pct_paper_intent_binary_fdkl", "temperature": temp,
                          "temperature_squared": temp ** 2, "kl_direction": "anchor_to_student",
                          "probability_units": "raw_logits", "anchor": source_stats})


def mpt_style_focal_kl(student_logits, labels, *, anchor_logits, teacher_logits,
                      anchor_threshold=0., teacher_threshold=0., alpha=1., beta=5., temperature=1.,
                      anchor_coefficient=1., teacher_coefficient=1., correctness_tie_positive=True,
                      partition="fit"):
    """MPT-STYLE BINARY adaptation: sum of independently masked anchor/teacher FD.

    Both sources use alpha+beta*source_correct over ALL input rows. There are no
    old/new class blocks or training-only class bias, hence this is NOT original
    MPT reproduction. Each term has T**2: to match the author's unscaled KL code
    at temperature T and external weight w, set the corresponding coefficient to
    w/T**2. Caller must record this choice and keep the hard-label term unchanged.
    """
    _fit_role(partition)
    student = _vector(student_logits, "student_logits")
    anchor = _vector(anchor_logits, "anchor_logits", length=len(student), device=student.device)
    teacher = _vector(teacher_logits, "teacher_logits", length=len(student), device=student.device)
    y = _labels(labels, len(student), student.device)
    temp = _temperature(temperature)
    old, old_stats = _focal_term(
        student, y, anchor, threshold=_threshold(anchor_threshold, "anchor_threshold"),
        alpha=alpha, beta=beta, temperature=temp, coefficient=anchor_coefficient,
        tie_positive=correctness_tie_positive)
    new, new_stats = _focal_term(
        student, y, teacher, threshold=_threshold(teacher_threshold, "teacher_threshold"),
        alpha=alpha, beta=beta, temperature=temp, coefficient=teacher_coefficient,
        tie_positive=correctness_tie_positive)
    return _result(old + new, {"method": "mpt_style_double_source_binary_fdkl_no_class_bias",
                               "temperature": temp, "temperature_squared": temp ** 2,
                               "kl_direction": "each_source_to_student", "probability_units": "raw_logits",
                               "class_incremental_bias": False, "anchor": old_stats, "teacher": new_stats})


def _threshold_array(value, name, student):
    if not isinstance(value, Tensor):
        if not isinstance(value, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, Real) for item in value):
            raise ValueError(f"{name} must be a fixed finite one-dimensional threshold array")
        value = torch.tensor(value, device=student.device, dtype=torch.float64)
    _vector(value, name)
    return value.detach().to(device=student.device, dtype=torch.float64)


def adapted_ekd_rank_count(student_logits, labels, *, teacher_logits, student_thresholds, teacher_thresholds,
                           temperature=.1, normal_coefficient=1., attack_coefficient=1., coefficient=1.,
                           partition="fit"):
    """Single-row binary EKD adaptation with PRE-FROZEN working-point arrays.

    r_S(i)=sum_k sigmoid((z_S(i)-t_S(k))/temperature), with teacher r_T similarly.
    soft=coefficient*(normal_coefficient*mean_{y=0}[(r_S-r_T)**2]
                      +attack_coefficient*mean_{y=1}[(r_S-r_T)**2]).
    No K or K**2 normalization, KL T**2 factor, dynamic quantile/momentum, pair
    construction, critical-only selection or negative mining. An absent class
    contributes zero and has explicit support/absence statistics. Threshold and
    teacher tensors are detached; the caller must not recompute them per batch.
    """
    _fit_role(partition)
    student = _vector(student_logits, "student_logits")
    teacher = _vector(teacher_logits, "teacher_logits", length=len(student), device=student.device)
    y = _labels(labels, len(student), student.device)
    st = _threshold_array(student_thresholds, "student_thresholds", student)
    tt = _threshold_array(teacher_thresholds, "teacher_thresholds", student)
    if len(st) != len(tt):
        raise ValueError("teacher and student must have the same predeclared number of working points")
    temp = _temperature(temperature)
    s_distance = (student.to(torch.float64)[:, None] - st[None, :]) / temp
    t_distance = (teacher.detach().to(torch.float64)[:, None] - tt[None, :]) / temp
    if not bool(torch.isfinite(s_distance).all() and torch.isfinite(t_distance).all()):
        raise ValueError("EKD relative margins are non-finite; inspect frozen scales")
    s_count, t_count = torch.sigmoid(s_distance).sum(dim=1), torch.sigmoid(t_distance).sum(dim=1)
    row_squared = (s_count - t_count).square()
    zero = student.to(torch.float64).sum() * 0
    terms, counts = [], []
    for label in (0, 1):
        mask = y == label
        counts.append(int(mask.sum()))
        terms.append(row_squared[mask].mean() if bool(mask.any()) else zero)
    normal = _constant(normal_coefficient, "normal_coefficient", student)
    attack = _constant(attack_coefficient, "attack_coefficient", student)
    factor = _constant(coefficient, "coefficient", student)
    soft = factor * (normal * terms[0] + attack * terms[1])
    return _result(soft, {"method": "adapted_ekd_single_row_binary_rank_count_mse", "rows": len(student),
                          "n_thresholds": len(st), "temperature": temp, "temperature_squared_factor": False,
                          "anchor_count_normalization": "sum_not_mean", "denominator": "each_present_class",
                          "normal_rows": counts[0], "attack_rows": counts[1],
                          "normal_absent": counts[0] == 0, "attack_absent": counts[1] == 0,
                          "normal_mse": float(terms[0].detach()), "attack_mse": float(terms[1].detach()),
                          "normal_coefficient": float(normal), "attack_coefficient": float(attack),
                          "coefficient": float(factor), "thresholds_frozen_caller_contract": True,
                          "student_thresholds": st.tolist(), "teacher_thresholds": tt.tolist()})
