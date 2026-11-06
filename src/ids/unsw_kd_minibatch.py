"""Frozen H/K mini-batch continuation; only in-memory F inputs are accepted.

The lab orchestrator binds the original anchor/targets and forty permutations.
This module performs no data loading, teacher query, endpoint evaluation,
checkpoint selection, extra loss pass, or experiment-matrix expansion.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .models import _network
from .risk_research import _state, _state_hash
from .unsw_kd import KD_WEIGHT, SEED, _labels, _probabilities, array_sha256, class_weights

ARMS = ("H", "K")
BATCH_SIZE = 1024
LOSS_SCOPE = (
    "Global F-weighted sum of per-row losses at each batch's pre-update parameters; "
    "parameters change within the epoch. Not a fixed-model or endpoint F evaluation."
)


def _orders(permutations, n):
    values = np.asarray(permutations)
    if (values.ndim != 2 or not len(values) or values.shape[1] != n
            or values.dtype.kind not in "iu"):
        raise ValueError("A nonempty integer matrix of complete F permutations is required")
    expected = np.arange(n, dtype=np.int64)
    for row in values:
        if not np.array_equal(np.sort(row), expected):
            raise ValueError("Each epoch must contain every F row exactly once")
    return np.ascontiguousarray(values, dtype=np.int64)


def fit_minibatch_arm(anchor, encoded_F, labels, target, permutations, *, arm, device,
                      teacher_probabilities=None, on_epoch=None):
    """Return (fixed final bundle, F-only path history, actual budget report).

    Batch size is always 1024. Production accepts exactly forty bound orders;
    shorter in-memory orders are useful for synthetic tests of this primitive.
    Each batch has one forward, backward, clipping operation, and AdamW step.
    The fixed full-F class weights are multiplied by N / actual batch size.
    """
    import torch
    from torch.nn import functional as F

    if arm not in ARMS:
        raise ValueError("Only H/K mini-batch arms are implemented")
    if anchor.model_name != "mlp" or anchor.network_state is None or anchor.estimator is not None:
        raise ValueError("A frozen MLP anchor is required")
    y = _labels(labels)
    x = np.asarray(encoded_F, dtype=np.float32)
    if x.shape != (len(y), anchor.encoded_dim) or not np.isfinite(x).all():
        raise ValueError("Finite encoded F must match the frozen anchor width")
    r = _probabilities(target, len(y))
    q = None if teacher_probabilities is None else _probabilities(teacher_probabilities, len(y))
    if arm == "H":
        if q is not None or not np.array_equal(r, y):
            raise ValueError("H requires exactly hard labels and no teacher probabilities")
    elif q is None or not np.array_equal(r, (y + KD_WEIGHT*q)/(1 + KD_WEIGHT)):
        raise ValueError("K requires the exact frozen teacher probabilities and mixed target")
    orders = _orders(permutations, len(y))
    weights = class_weights(y)
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"} or (selected.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("CPU synthetic or CUDA laboratory execution required")
    use_cuda = selected.type == "cuda"
    cuda_devices = [selected.index if selected.index is not None else torch.cuda.current_device()] if use_cuda else []
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
           torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32)
    anchor_before = _state_hash(anchor.network_state)
    history = []
    forward_rows = forward_calls = backward_calls = updates = 0
    started = time.perf_counter()
    order_hashes = [array_sha256(row, "<i8") for row in orders]
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices), torch.autocast(selected.type, enabled=False):
            torch.manual_seed(SEED)
            net = _network(anchor.encoded_dim, tuple(anchor.hidden_dims)).to(device=selected, dtype=torch.float32)
            net.load_state_dict(anchor.network_state)
            net.train()
            initial = _state_hash(_state(net))
            if initial != anchor_before:
                raise ValueError("Loaded network differs from the common anchor")
            optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001,
                betas=(.9, .999), eps=1e-8, amsgrad=False, foreach=False, fused=False)
            if use_cuda:
                torch.cuda.synchronize(selected)
                torch.cuda.reset_peak_memory_stats(selected)
            for epoch, order in enumerate(orders, 1):
                totals = {"objective_bce": 0., "hard_bce": 0., "teacher_kl": 0.,
                          "teacher_entropy_constant": 0.}
                norms = []
                clipping = 0
                for start in range(0, len(y), BATCH_SIZE):
                    ix = order[start:start+BATCH_SIZE]
                    actual_size = len(ix)
                    optimizer.zero_grad(set_to_none=True)
                    z = net(torch.from_numpy(x[ix]).to(selected)).squeeze(1).to(torch.float64)
                    forward_rows += actual_size
                    forward_calls += 1
                    wb = torch.from_numpy(weights[ix]).to(selected)
                    rb = torch.from_numpy(r[ix]).to(selected)
                    rows = F.binary_cross_entropy_with_logits(z, rb, reduction="none")
                    full_weight_batch_loss = (wb*rows).sum()
                    loss = full_weight_batch_loss * (len(y)/actual_size)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite mini-batch loss; no update for this batch")
                    loss.backward()
                    backward_calls += 1
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    norms.append(float(norm))
                    clipping += int(float(norm) > 10.)
                    # Reuse detached logits; do not query the network for logging.
                    with torch.no_grad():
                        hard = torch.from_numpy(y[ix].astype(np.float64)).to(selected)
                        totals["objective_bce"] += float(full_weight_batch_loss.detach())
                        totals["hard_bce"] += float((wb*F.binary_cross_entropy_with_logits(
                            z.detach(), hard, reduction="none")).sum())
                        if q is not None:
                            qb = torch.from_numpy(q[ix]).to(selected)
                            entropy = -(torch.special.xlogy(qb, qb) + torch.special.xlogy(1-qb, 1-qb))
                            kl = F.binary_cross_entropy_with_logits(z.detach(), qb, reduction="none") - entropy
                            totals["teacher_kl"] += float((wb*kl).sum())
                            totals["teacher_entropy_constant"] += float(
                                (wb*entropy * KD_WEIGHT/(1+KD_WEIGHT)).sum())
                record = {
                    "epoch": epoch, **totals, "loss_scope": LOSS_SCOPE,
                    "normalized_hard_plus_kl": (totals["hard_bce"] + KD_WEIGHT*totals["teacher_kl"])
                        / (1+KD_WEIGHT) if q is not None else totals["hard_bce"],
                    "permutation_sha256": order_hashes[epoch-1],
                    "optimizer_updates": len(norms), "gradient_forward_rows": len(y),
                    "backward_calls": len(norms), "last_batch_rows": len(y)-(len(norms)-1)*BATCH_SIZE,
                    "preclip_gradient_norm_mean": float(np.mean(norms)),
                    "preclip_gradient_norm_max": float(np.max(norms)),
                    "gradient_clip_count": clipping,
                    "cumulative_optimizer_updates": updates,
                    "cumulative_gradient_forward_rows": forward_rows,
                    "cumulative_forward_calls": forward_calls, "cumulative_backward_calls": backward_calls,
                }
                history.append(record)
                if on_epoch is not None:
                    on_epoch(copy.deepcopy(record))
            if any(not bool(torch.isfinite(parameter).all()) for parameter in net.parameters()):
                raise FloatingPointError("Nonfinite final mini-batch parameters")
            if _state_hash(anchor.network_state) != anchor_before:
                raise ValueError("Common anchor changed during continuation")
            if use_cuda:
                torch.cuda.synchronize(selected)
            bundle = copy.copy(anchor)
            bundle.network_state = _state(net)
            bundle._cached_network = None
            bundle.history = []
            bundle.training_statistics = {}
            bundle.best_epoch = len(orders)
            bundle.seed = SEED
            bundle.device = "cpu"
            bundle.batch_size = BATCH_SIZE
            bundle.num_threads = 4
            report = {
                "arm": arm, "status": "trained", "endpoint_epoch": len(orders),
                "checkpoint_selection": False, "common_initial_state_sha256": initial,
                "anchor_state_after_sha256": _state_hash(anchor.network_state),
                "final_state_sha256": _state_hash(bundle.network_state),
                "target_sha256": array_sha256(r), "weights_sha256": array_sha256(weights),
                "permutations_sha256": array_sha256(orders, "<i8"),
                "epoch_permutation_sha256": order_hashes,
                "training_seconds": time.perf_counter()-started,
                "fit_rows": len(y), "epochs": len(orders), "batch_size": BATCH_SIZE,
                "batches_per_epoch": math.ceil(len(y)/BATCH_SIZE),
                "optimizer_updates": updates, "gradient_forward_rows": forward_rows,
                "forward_calls": forward_calls, "backward_calls": backward_calls,
                "R_C_evaluation_rows_inside_training": 0, "extra_F_evaluation_rows": 0,
                "teacher_queries": 0, "parameters": sum(p.numel() for p in net.parameters()),
                "weighting": "fixed full-F class weights multiplied by N / actual batch rows",
                "training_loss_scope": LOSS_SCOPE,
                "optimizer": {"name": "AdamW", "learning_rate": .001, "weight_decay": .0001,
                    "betas": [.9, .999], "eps": 1e-8, "amsgrad": False, "foreach": False,
                    "fused": False, "state_reset": True, "clip_norm": 10., "clip_per_batch": True},
                "network_dtype": "float32", "loss_dtype": "float64", "amp": False,
                "deterministic_algorithms": True, "tf32": False, "threads": 4,
                "seed": SEED, "device": str(selected),
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)) if use_cuda else None,
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)) if use_cuda else None,
            }
    finally:
        torch.set_num_threads(old[0])
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[5:7]
    return bundle, history, report
