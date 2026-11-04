"""Same-label, cross-example preference transfer for the rebuilt research study.

This is a falsifiable task adaptation of relational/rank distillation, not a new
ranking theorem. Labels supply class separation; the teacher supplies only the
relative ordering of two DIFFERENT examples having the same true label. Native
teacher margins are never converted to clipped class probabilities.

Pairs are drawn once from the full fitting pool, with one uniformly drawn
teacher-tail anchor and one uniformly drawn other example from its class. The
two classes each carry weight 1/2. A class with no eligible anchor contributes
zero, with its absent coverage reported explicitly. Chunking must retain these
global pair weights. R normals define an interest region, NOT an FPR certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class TailRankPairs:
    left: np.ndarray
    right: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    n_rows: int
    stats: dict


def _array(value, name):
    result = np.asarray(value)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "fiu"
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    return result


def _positive(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def sample_same_class_pairs(teacher_scores, labels, *, reference_normal_scores=None,
                           pairs_per_class=4096, max_fpr=.01, tail_multiplier=5.,
                           local=True, shuffle_within_class=False, seed=17):
    """Build a reproducible full-pool pair population with frozen order targets.

    For local=True, active(i) = [t_i >= quantile_higher(R_normal, 1-m*alpha)].
    The fixed teacher's R scores and fitting scores must use the SAME native
    coordinate. Fold teachers cannot be pooled in this API: call it separately
    per fold, or use a single teacher as the rebuilt protocol does.

    Each class samples pairs_per_class pairs WITH replacement. Self pairs are
    excluded without rejection sampling; duplicate ordered pairs can occur.
    Left endpoints are uniform among active members, right endpoints uniform
    among all other members of the same class. Thus a pair has at least one
    active endpoint; this is an anchored sampling distribution, not a uniform
    distribution over all eligible unordered pairs.

    Shuffling permutes teacher scores independently within each true class
    BEFORE both interest-mask construction and target construction. It preserves
    class score distributions and tail support counts but destroys score/row
    correspondence. The RNG stream for pair selection is separate from the
    shuffle stream. Ties have target .5 and are reported, not broken by row ID.
    """
    teacher = _array(teacher_scores, "teacher_scores").astype(np.float64, copy=True)
    y = _array(labels, "labels")
    if len(y) != len(teacher) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be aligned binary labels")
    if isinstance(pairs_per_class, bool) or not isinstance(pairs_per_class, Integral) or pairs_per_class < 1:
        raise ValueError("pairs_per_class must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(local) is not bool or type(shuffle_within_class) is not bool:
        raise ValueError("local and shuffle_within_class must be booleans")
    alpha, multiplier = _positive(max_fpr, "max_fpr"), _positive(tail_multiplier, "tail_multiplier")
    interest_fraction = alpha * multiplier
    if not 0 < alpha < 1 or not 0 < interest_fraction < 1:
        raise ValueError("max_fpr and max_fpr*tail_multiplier must lie strictly between zero and one")
    shuffle_seed, pair_seed = np.random.SeedSequence(int(seed)).spawn(2)
    shuffle_rng, pair_rng = np.random.default_rng(shuffle_seed), np.random.default_rng(pair_seed)
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        if shuffle_within_class:
            teacher[pool] = teacher[shuffle_rng.permutation(pool)]
    threshold = None
    if local:
        reference = _array(reference_normal_scores, "reference_normal_scores").astype(np.float64)
        # An order statistic commutes with strictly increasing transformations;
        # interpolated quantiles would not preserve this exact property.
        threshold = float(np.quantile(reference, 1-interest_fraction, method="higher"))
    active = np.ones(len(teacher), dtype=bool) if not local else teacher >= threshold
    left, right, target, weight, class_stats = [], [], [], [], {}
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        anchors = np.flatnonzero(active[pool])
        n_pairs = int(pairs_per_class) if len(pool) >= 2 and len(anchors) else 0
        summary = {"pool_rows": len(pool), "active_rows": len(anchors), "pairs": n_pairs,
                   "target_zero": 0, "target_half": 0, "target_one": 0,
                   "weight_total": .5 if n_pairs else 0.}
        class_stats[str(label)] = summary
        if not n_pairs:
            continue
        left_position = pair_rng.choice(anchors, size=n_pairs, replace=True)
        other_position = pair_rng.integers(0, len(pool)-1, size=n_pairs)
        other_position += other_position >= left_position
        i, j = pool[left_position], pool[other_position]
        q = np.where(teacher[i] > teacher[j], 1., np.where(teacher[i] < teacher[j], 0., .5))
        left.append(i)
        right.append(j)
        target.append(q)
        weight.append(np.full(n_pairs, .5/n_pairs, dtype=np.float64))
        summary.update({"target_zero": int((q == 0).sum()), "target_half": int((q == .5).sum()),
                        "target_one": int((q == 1).sum())})
    result = TailRankPairs(
        left=np.concatenate(left) if left else np.empty(0, dtype=np.int64),
        right=np.concatenate(right) if right else np.empty(0, dtype=np.int64),
        targets=np.concatenate(target) if target else np.empty(0, dtype=np.float64),
        weights=np.concatenate(weight) if weight else np.empty(0, dtype=np.float64),
        n_rows=len(teacher),
        stats={"method": "same_class_cross_example_order_transfer", "local": local,
               "teacher_score_coordinate": "caller_supplied_native_margin_same_teacher_for_F_and_R",
               "tail_threshold": threshold, "tail_interest_fraction": interest_fraction if local else None,
               "max_fpr": alpha, "tail_multiplier": multiplier,
               "tail_threshold_is_fpr_certificate": False,
               "sampling": "uniform_active_anchor_then_uniform_other_same_class_with_replacement",
               "shuffle_within_class": shuffle_within_class, "seed": int(seed),
               "class_reduction": "half_each_class_zero_if_no_eligible_pairs", "classes": class_stats})
    for array in (result.left, result.right, result.targets, result.weights):
        array.flags.writeable = False
    return result


def _tensor(value, name, *, length=None, device=None, allow_empty=False):
    if (not isinstance(value, Tensor) or value.ndim != 1 or not value.is_floating_point()
            or (not allow_empty and not value.numel()) or not bool(torch.isfinite(value).all())
            or (length is not None and len(value) != length)
            or (device is not None and value.device != device)):
        raise ValueError(f"{name} must be an aligned finite floating vector on the same device")
    return value


def tail_rank_kd_rows(student_left, student_right, targets, *, temperature=1.):
    """Return pair BCE((s_left-s_right)/T, fixed teacher order), WITHOUT weights.

    This target is a pair preference, not an attack probability. There is no
    teacher entropy weighting and no T**2 factor. For a tied pair BCE has a
    constant log(2) minimum; a large logged loss is not evidence of a large
    student gradient. Targets are detached; gradients reach BOTH endpoints.
    """
    left = _tensor(student_left, "student_left", allow_empty=True)
    right = _tensor(student_right, "student_right", length=len(left), device=left.device, allow_empty=True)
    q = _tensor(targets, "targets", length=len(left), device=left.device, allow_empty=True).detach()
    if not bool(((q == 0) | (q == .5) | (q == 1)).all()):
        raise ValueError("targets must be teacher orders in {0,.5,1}")
    temp = _positive(temperature, "temperature")
    difference = (left.to(torch.float64)-right.to(torch.float64))/temp
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("scaled student differences must be finite")
    return F.binary_cross_entropy_with_logits(difference, q.to(torch.float64), reduction="none")


def tail_rank_kd(student_logits, pairs: TailRankPairs, *, temperature=1.):
    """Convenience full-population reduction; rows API supports bounded chunks."""
    student = _tensor(student_logits, "student_logits", length=pairs.n_rows)
    left = torch.tensor(pairs.left.copy(), dtype=torch.int64, device=student.device)
    right = torch.tensor(pairs.right.copy(), dtype=torch.int64, device=student.device)
    targets = torch.tensor(pairs.targets.copy(), dtype=torch.float64, device=student.device)
    weights = torch.tensor(pairs.weights.copy(), dtype=torch.float64, device=student.device)
    return (weights*tail_rank_kd_rows(student[left], student[right], targets, temperature=temperature)).sum()


def binary_channel_rank_rows(student_logits, *, teacher_logits, beta=1., standardize=True):
    """ICML2025 RankLoss, explicit TWO-channel adaptation of the author defaults.

    Author source: RankingLoss-KD@50a61305fd068dbbd14c04ad070ea9fb9b76c4d0,
    mdistiller/distillers/rankloss.py, RankLoss / normalize. The public source
    hardcodes 1000 classes and CUDA, so it cannot directly consume binary IDS.
    This adaptation uses channels [-s/2,+s/2], the same unbiased row std +1e-7,
    beta logistic soft-sign, and author default output multiplier 1/.0125=80.
    A two-class row has ONE channel pair. With row standardization it collapses
    almost to sign(margin); it provides no cross-example relational knowledge.
    Returned rows are negative correlation, not a nonnegative divergence.
    """
    student = _tensor(student_logits, "student_logits")
    teacher = _tensor(teacher_logits, "teacher_logits", length=len(student), device=student.device).detach()
    sharpness = _positive(beta, "beta")
    if type(standardize) is not bool:
        raise ValueError("standardize must be boolean")
    s, t = student.to(torch.float64), teacher.to(torch.float64)
    if standardize:
        s = s/(s.abs()/math.sqrt(2)+1e-7)
        t = t/(t.abs()/math.sqrt(2)+1e-7)
    return -80*torch.tanh(.5*sharpness*s)*torch.tanh(.5*sharpness*t)
