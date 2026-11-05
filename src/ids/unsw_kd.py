"""Frozen UNSW ordinary-KD targets and single-pass continuation.

This module accepts only in-memory fitting inputs. It never reads a data
partition, evaluates R/C, selects checkpoints, or expands an experiment grid.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time

import numpy as np
from scipy.special import expit

from .models import _network
from .risk_research import _state, _state_hash

ARMS = ("H", "K", "S", "M")
KD_WEIGHT = 0.3
SEED = 17


def array_sha256(values, dtype="<f8"):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _labels(values):
    result = np.asarray(values)
    if (result.ndim != 1 or not len(result) or not np.isfinite(result).all()
            or set(np.unique(result)) != {0, 1}):
        raise ValueError("F labels must be a finite binary vector with both classes")
    return result.astype(np.int64)


def _probabilities(values, n):
    result = np.asarray(values, dtype=np.float64)
    if (result.shape != (n,) or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError("teacher probabilities must be aligned and in [0,1]")
    return result.copy()


def class_weights(labels):
    y = _labels(labels)
    return np.where(y == 0, .5 / np.count_nonzero(y == 0),
                    .5 / np.count_nonzero(y == 1)).astype(np.float64)


def build_targets(fit_identity, teacher_fit_logits, *, frozen_q=None):
    """Return (H/K/S/M targets, JSON audit, numeric permutation/target arrays).

    frozen_q is an optional already-bound copy of the original scipy.expit
    output, not an alternate teacher. The orchestrator verifies its frozen SHA.
    """
    import pandas as pd

    if "feature_hash" not in fit_identity or "label" not in fit_identity:
        raise ValueError("F identity requires feature_hash and label")
    y = _labels(fit_identity.label.to_numpy())
    z = np.asarray(teacher_fit_logits, dtype=np.float64)
    if z.shape != y.shape or not np.isfinite(z).all():
        raise ValueError("teacher F native margins must be finite and aligned")
    if fit_identity.feature_hash.isna().any():
        raise ValueError("F feature hashes must not be missing")
    hashes = fit_identity.feature_hash.astype(str).to_numpy()
    q = expit(z) if frozen_q is None else _probabilities(frozen_q, len(y))
    frame = pd.DataFrame({"feature_hash": hashes, "label": y, "q": q})
    group = frame.groupby("feature_hash", sort=True).agg(
        n=("label", "size"), n1=("label", "sum"), q_min=("q", "min"),
        q_max=("q", "max"), q=("q", "first"))
    group["n0"] = group.n - group.n1
    if not (group.q_min == group.q_max).all():
        raise ValueError("teacher q differs within an identical feature pattern")
    group["stratum_size"] = group.groupby(["n0", "n1"], sort=True).q.transform("size")
    eligible = group.stratum_size.to_numpy() >= 2
    row_group = group.index.get_indexer(hashes).astype(np.int64)
    row_eligible = eligible[row_group]
    donor = np.arange(len(group), dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 353]))
    for _, indices in group.groupby(["n0", "n1"], sort=True).indices.items():
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) >= 2:
            order = rng.permutation(indices)
            donor[order] = np.roll(order, 1)
    if not np.all(donor[eligible] != np.flatnonzero(eligible)):
        raise ValueError("exchangeable patterns must have no self-donors")
    q_per_group = group.q.to_numpy()
    q_perm = q_per_group[donor][row_group]
    if any(not np.array_equal(np.sort(q[y == c]), np.sort(q_perm[y == c])) for c in (0, 1)):
        raise ValueError("pattern permutation changed class-conditional q multisets")
    means = np.asarray([q[y == c].mean() for c in (0, 1)], dtype=np.float64)
    q_mean = means[y]
    soft = {"K": q, "S": q_perm, "M": q_mean}
    targets = {"H": y.astype(np.float64),
               **{arm: (y + KD_WEIGHT * value) / (1 + KD_WEIGHT)
                  for arm, value in soft.items()}}
    weights = class_weights(y)
    displacement = np.abs(q - y)

    def coverage(mask):
        total_mass = displacement[mask].sum()
        return {
            "rows": int(mask.sum()),
            "eligible_rows": int((mask & row_eligible).sum()),
            "eligible_row_fraction": float(row_eligible[mask].mean()),
            "eligible_abs_q_minus_y_mass_fraction":
                float(displacement[mask & row_eligible].sum() / total_mass) if total_mass else None,
            "actually_changed_rows": int(((q != q_perm) & mask).sum()),
            "actually_changed_row_fraction": float((q != q_perm)[mask].mean()),
        }

    arrays = {
        "weights": weights, "labels": y,
        "teacher_q": q, "permuted_teacher_q": q_perm, "class_mean_q": q_mean,
        "feature_hash": group.index.to_numpy(dtype=str),
        "n0": group.n0.to_numpy(dtype=np.int64), "n1": group.n1.to_numpy(dtype=np.int64),
        "donor_group_index": donor, "row_group_index": row_group,
        **{"target_" + arm: target for arm, target in targets.items()},
    }
    audit = {
        "fit_rows": len(y), "lambda_soft_to_hard": KD_WEIGHT, "temperature": 1,
        "target_source": "frozen bound teacher_q" if frozen_q is not None else "scipy.special.expit",
        "teacher_q_sha256": array_sha256(q),
        "q_perm_sha256": array_sha256(q_perm),
        "donor_i8_sha256": array_sha256(donor, "<i8"),
        "weights_sha256": array_sha256(weights),
        "labels_i8_sha256": array_sha256(y, "<i8"),
        "targets_sha256": {arm: array_sha256(value) for arm, value in targets.items()},
        "class_mean_q": {str(c): float(means[c]) for c in (0, 1)},
        "class_weight_sum": {str(c): float(weights[y == c].sum()) for c in (0, 1)},
        "feature_patterns": len(group),
        "n_strata": int(group.groupby(["n0", "n1"], sort=True).ngroups),
        "eligible_patterns": int(eligible.sum()),
        "nonexchangeable_patterns": int((~eligible).sum()),
        **coverage(np.ones(len(y), dtype=bool)),
        "by_class": {str(c): coverage(y == c) for c in (0, 1)},
        "class_balanced_mean_abs_q_minus_qperm": float(np.sum(weights * np.abs(q-q_perm))),
        "class_balanced_mean_abs_target_K_minus_S":
            float(np.sum(weights * np.abs(targets["K"]-targets["S"]))),
        "class_conditional_q_multisets_exactly_preserved": True,
        "full_feature_pattern_consistency_preserved": True,
        "permutation_seed_sequence": [SEED, 353], "mapping_count": 1,
    }
    return targets, audit, arrays


def _backward_full_f(net, encoded_f, labels, target, weights, *,
                     chunk_size, teacher_probabilities=None):
    """One complete F gradient, with a global class denominator in every chunk."""
    import torch
    from torch.nn import functional as F

    device = next(net.parameters()).device
    totals = {"objective_bce": 0., "hard_bce": 0., "teacher_kl": 0.,
              "teacher_entropy_constant": 0., "gradient_forward_rows": 0, "backward_calls": 0}
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        x = torch.from_numpy(encoded_f[start:end]).to(device)
        z = net(x).squeeze(1).to(torch.float64)
        r = torch.from_numpy(target[start:end]).to(device)
        w = torch.from_numpy(weights[start:end]).to(device)
        loss = (w * F.binary_cross_entropy_with_logits(z, r, reduction="none")).sum()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite fitting loss; no update performed")
        loss.backward()
        with torch.no_grad():
            hard = torch.from_numpy(labels[start:end].astype(np.float64)).to(device)
            totals["objective_bce"] += float(loss)
            totals["hard_bce"] += float((w * F.binary_cross_entropy_with_logits(
                z.detach(), hard, reduction="none")).sum())
            if teacher_probabilities is not None:
                q = torch.from_numpy(teacher_probabilities[start:end]).to(device)
                entropy = -(torch.special.xlogy(q, q) + torch.special.xlogy(1-q, 1-q))
                kl = F.binary_cross_entropy_with_logits(z.detach(), q, reduction="none") - entropy
                totals["teacher_kl"] += float((w * kl).sum())
                totals["teacher_entropy_constant"] += float(
                    (w * entropy * KD_WEIGHT / (1+KD_WEIGHT)).sum())
        totals["gradient_forward_rows"] += end - start
        totals["backward_calls"] += 1
    totals["normalized_hard_plus_kl"] = (
        (totals["hard_bce"] + KD_WEIGHT * totals["teacher_kl"]) / (1+KD_WEIGHT)
        if teacher_probabilities is not None else totals["hard_bce"])
    return totals


def fit_arm(anchor, encoded_F, labels, target, *, arm, device,
            teacher_probabilities=None, macro_steps=40, chunk_size=2048, on_step=None):
    """Return (endpoint bundle, F-only history, budget/runtime report).

    The laboratory orchestrator locks 40/2048 from the protocol; smaller values
    are supported solely for synthetic execution checks. No checkpoint other
    than the final step is returned. The anchor and encoded arrays are unchanged.
    """
    import torch

    if arm not in ARMS:
        raise ValueError("unknown arm")
    if any(type(v) is not int or v < 1 for v in (macro_steps, chunk_size)):
        raise ValueError("positive integer macro_steps and chunk_size required")
    if anchor.model_name != "mlp" or anchor.network_state is None:
        raise ValueError("a frozen MLP anchor is required")
    y = _labels(labels)
    x = np.asarray(encoded_F, dtype=np.float32)
    if x.shape != (len(y), anchor.encoded_dim) or not np.isfinite(x).all():
        raise ValueError("finite encoded F must match rows and anchor width")
    r = _probabilities(target, len(y))
    q = None if teacher_probabilities is None else _probabilities(teacher_probabilities, len(y))
    if arm == "H":
        if q is not None or not np.array_equal(r, y):
            raise ValueError("hard arm requires exactly hard labels and no teacher target")
    elif q is None or not np.array_equal(r, (y+KD_WEIGHT*q)/(1+KD_WEIGHT)):
        raise ValueError("soft arm requires its exact frozen q and mixed target")
    weights = class_weights(y)
    selected_device = torch.device(device)
    if selected_device.type not in {"cpu", "cuda"}:
        raise ValueError("CPU synthetic or CUDA laboratory execution required")
    use_cuda = selected_device.type == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn = torch.backends.cudnn.deterministic
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cuda_devices = [selected_device.index if selected_device.index is not None
                    else torch.cuda.current_device()] if use_cuda else []
    history = []
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(SEED)
            net = _network(anchor.encoded_dim, tuple(anchor.hidden_dims)).to(
                device=selected_device, dtype=torch.float32)
            net.load_state_dict(anchor.network_state)
            net.train()
            initial = _state_hash(_state(net))
            optimizer = torch.optim.AdamW(
                net.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999),
                eps=1e-8, amsgrad=False, foreach=False, fused=False)
            if use_cuda:
                torch.cuda.synchronize(selected_device)
                torch.cuda.reset_peak_memory_stats(selected_device)
            for step in range(1, macro_steps+1):
                optimizer.zero_grad(set_to_none=True)
                losses = _backward_full_f(net, x, y, r, weights, chunk_size=chunk_size,
                                         teacher_probabilities=q)
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                record = {
                    "step": step, **losses, "preclip_gradient_norm": float(norm),
                    "gradient_clip_triggered": bool(float(norm) > 10.),
                    "cumulative_optimizer_updates": step,
                    "cumulative_gradient_forward_rows": step * len(y),
                    "cumulative_backward_calls": step * math.ceil(len(y)/chunk_size),
                }
                history.append(record)
                if on_step is not None:
                    on_step(copy.deepcopy(record))
            if any(not bool(torch.isfinite(p).all()) for p in net.parameters()):
                raise FloatingPointError("non-finite final parameters")
            if use_cuda:
                torch.cuda.synchronize(selected_device)
            bundle = copy.copy(anchor)
            bundle.network_state = _state(net)
            bundle._cached_network = None
            bundle.history = []
            bundle.training_statistics = {}
            bundle.best_epoch = macro_steps
            bundle.seed = SEED
            bundle.device = "cpu"
            bundle.batch_size = chunk_size
            bundle.num_threads = 4
            report = {
                "arm": arm, "status": "trained", "endpoint_step": macro_steps,
                "checkpoint_selection": False, "common_initial_state_sha256": initial,
                "final_state_sha256": _state_hash(bundle.network_state),
                "target_sha256": array_sha256(r), "weights_sha256": array_sha256(weights),
                "training_seconds": time.perf_counter()-started,
                "fit_rows": len(y), "optimizer_updates": macro_steps,
                "gradient_forward_rows": macro_steps * len(y),
                "backward_calls": macro_steps * math.ceil(len(y)/chunk_size),
                "chunks_per_step": math.ceil(len(y)/chunk_size),
                "R_C_evaluation_rows_inside_training": 0,
                "parameters": sum(p.numel() for p in net.parameters()),
                "optimizer": {"name": "AdamW", "learning_rate": .001,
                              "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8,
                              "amsgrad": False, "foreach": False, "fused": False,
                              "state_reset": True},
                "network_dtype": "float32", "loss_dtype": "float64", "amp": False,
                "deterministic_algorithms": True, "tf32": False, "threads": 4,
                "seed": SEED, "device": str(selected_device),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected_device))
                    if use_cuda else None,
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected_device))
                    if use_cuda else None,
            }
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    return bundle, history, report
