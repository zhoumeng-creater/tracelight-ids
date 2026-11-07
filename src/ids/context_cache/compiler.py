"""QCOC (2026) joint-KV prototypes and query-calibrated value fitting.

Independent implementation of https://arxiv.org/html/2610.11784v1, section 3
and appendix C. Input queries are Phi_N(Q): QASSMax and the attention scale
have ALREADY been applied. This module never reads datasets or model weights.
"""

from __future__ import annotations

import math

import torch

from .cache import WeightedKVEntry


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _distances(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # The direct distance kernel avoids large-coordinate cancellation in
    # ||x||^2 + ||c||^2 - 2 x.c and does not allocate [N, M, 2d].
    return torch.cdist(points, centers, compute_mode="donot_use_mm_for_euclid_dist").square()


def _nonempty_assignment(distances: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Nearest prototype, then farthest eligible point for each empty slot.

    torch.argmin/argmax choose the first index on exact ties. A donor must
    retain at least one point. This rule also defines the all-identical case.
    The repair changes assignments only, not the current prototype values.
    """
    assignments = distances.argmin(dim=1)
    count = torch.bincount(assignments, minlength=distances.shape[1])
    repairs = 0
    for empty in (count == 0).nonzero().flatten().tolist():
        eligible = count[assignments] > 1
        losses = distances.gather(1, assignments[:, None]).squeeze(1)
        losses = losses.masked_fill(~eligible, -torch.inf)
        donor = int(losses.argmax().item())
        old = int(assignments[donor].item())
        count[old] -= 1
        count[empty] += 1
        assignments[donor] = empty
        repairs += 1
    return assignments, repairs


def _joint_kv_prototypes(
    key: torch.Tensor, value: torch.Tensor, slots: int, iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """One full, unit-mass [N,d] head, calculated in float64.

    C determines the last Lloyd means. QCOC then reassigns to those means to
    obtain D for the mass, without a hidden extra centroid update. Empty D
    groups are repaired by the same explicit farthest-donor rule.
    """
    scale_k = key.std(correction=0)
    scale_v = value.std(correction=0)
    zero_k, zero_v = bool(scale_k == 0), bool(scale_v == 0)
    if zero_k:
        scale_k = key.new_tensor(1.0)
    if zero_v:
        scale_v = value.new_tensor(1.0)
    points = torch.cat((key / scale_k, value / scale_v), dim=1)

    seeds = [0]
    nearest = _distances(points, points[:1]).squeeze(1)
    for _ in range(1, slots):
        candidate_distances = nearest.clone()
        candidate_distances[seeds] = -torch.inf
        selected = int(candidate_distances.argmax().item())
        seeds.append(selected)
        nearest = torch.minimum(nearest, _distances(points, points[selected:selected + 1]).squeeze(1))
    centers = points[seeds].clone()
    previous = None
    repairs = 0
    for step in range(iterations):
        assignments, changed = _nonempty_assignment(_distances(points, centers))
        repairs += changed
        # Explicit per-cluster reductions retain source order, including CUDA.
        centers = torch.stack([points[assignments == j].mean(dim=0) for j in range(slots)])
        stable = previous is not None and torch.equal(assignments, previous)
        previous = assignments.clone()
        if stable:
            break

    count_assignments, final_repairs = _nonempty_assignment(_distances(points, centers))
    counts = torch.bincount(count_assignments, minlength=slots)
    d = key.shape[1]
    info = {
        "lloyd_iterations": step + 1,
        "lloyd_empty_repairs": repairs,
        "final_count_empty_repairs": final_repairs,
        "final_count_assignment_changes": int((assignments != count_assignments).sum().item()),
        "seed_source_ids": seeds,
        "key_scale": float(scale_k.item()),
        "value_scale": float(scale_v.item()),
        "zero_key_scale": zero_k,
        "zero_value_scale": zero_v,
        "min_mass": int(counts.min().item()),
        "max_mass": int(counts.max().item()),
    }
    return centers[:, :d] * scale_k, centers[:, d:] * scale_v, counts, info


def _fit_values(
    key: torch.Tensor, value: torch.Tensor, queries: torch.Tensor,
    compact_key: torch.Tensor, anchor: torch.Tensor, log_mass: torch.Tensor,
    *, ridge: float, query_chunk_size: int, output_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict]:
    """Accumulate the regression in query chunks; no full [Q,N] attention."""
    slots, width = anchor.shape
    gram = key.new_zeros((slots, slots))
    rhs = key.new_zeros((slots, width))
    anchor_sse = key.new_zeros(())
    chunks = range(0, queries.shape[0], query_chunk_size)
    for start in chunks:
        q = queries[start:start + query_chunk_size]
        target = (q @ key.T).softmax(dim=-1) @ value
        p = (q @ compact_key.T + log_mass).softmax(dim=-1)
        gram += p.T @ p
        rhs += p.T @ target
        anchor_sse += (p @ anchor - target).square().sum()

    gamma = torch.maximum(gram.diagonal().sum() / slots, gram.new_tensor(torch.finfo(gram.dtype).eps))
    coefficient = ridge * gamma
    system = gram + coefficient * torch.eye(slots, dtype=gram.dtype, device=gram.device)
    fitted = torch.linalg.solve(system, rhs + coefficient * anchor)
    stored = fitted.to(dtype=output_dtype)
    if not torch.isfinite(stored).all():
        raise ValueError("Fitted values overflow the output dtype; use a higher-precision cache")

    # Report errors both before and after export rounding. Recompute targets
    # in chunks rather than retaining all source-query outputs for every head.
    fitted_sse = key.new_zeros(())
    stored_sse = key.new_zeros(())
    stored64 = stored.to(dtype=torch.float64)
    for start in chunks:
        q = queries[start:start + query_chunk_size]
        target = (q @ key.T).softmax(dim=-1) @ value
        p = (q @ compact_key.T + log_mass).softmax(dim=-1)
        fitted_sse += (p @ fitted - target).square().sum()
        stored_sse += (p @ stored64 - target).square().sum()
    penalty = coefficient * (fitted - anchor).square().sum()
    stored_penalty = coefficient * (stored64 - anchor).square().sum()
    return stored, {
        "gamma": float(gamma.item()),
        "ridge_coefficient": float(coefficient.item()),
        "anchor_sse": float(anchor_sse.item()),
        "fitted_sse_float64": float(fitted_sse.item()),
        "stored_value_sse": float(stored_sse.item()),
        "fitted_objective_float64": float((fitted_sse + penalty).item()),
        "stored_value_objective": float((stored_sse + stored_penalty).item()),
    }


@torch.no_grad()
def compile_entry(
    entry: WeightedKVEntry, queries_phi: torch.Tensor, *, slots: int,
    iterations: int = 15, ridge: float = 1e-3, query_chunk_size: int = 256,
) -> tuple[WeightedKVEntry, dict]:
    """Compile full unit-mass KV to exactly ``slots`` per batch and head.

    ``queries_phi`` is [B,H,Q,d] after Phi_N, including attention scaling.
    The TabICLv2 bridge supplies all full-support queries, Q=N; accepting Q
    independently also allows small standalone operator checks. The caller
    owns query provenance. No query labels or risk-calibration data are used.

    Computation/solve use float64 on the input device. Returned K/V and masses
    use their input dtypes. Keys and log masses are rounded to the deployment
    dtype *before* value fitting. Reports include value-rounding error.
    """
    slots = _positive_int(slots, "slots")
    iterations = _positive_int(iterations, "iterations")
    query_chunk_size = _positive_int(query_chunk_size, "query_chunk_size")
    if iterations > 15:
        raise ValueError("QCOC compilation is limited to at most 15 Lloyd iterations")
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be a finite positive number")
    if entry.source_count != entry.key.shape[-2] or not bool((entry.log_mass == 0).all()):
        raise ValueError("QCOC compilation requires a full unit-mass cache, not an already compressed cache")
    if slots > entry.source_count:
        raise ValueError("slots cannot exceed the full source count")
    if (not isinstance(queries_phi, torch.Tensor) or queries_phi.ndim != 4
            or queries_phi.shape[:2] != entry.key.shape[:2]
            or queries_phi.shape[-1] != entry.key.shape[-1] or queries_phi.shape[-2] < 1):
        raise ValueError("queries_phi must have shape [B,H,Q,d] matching the full cache, with Q > 0")
    if not queries_phi.is_floating_point() or not bool(torch.isfinite(queries_phi).all()):
        raise ValueError("queries_phi must contain finite floating-point values")
    if queries_phi.device != entry.key.device:
        raise ValueError("queries_phi and cache must be on the same device")

    report = {
        "method": "qcoc_2026",
        "implementation": "independent_from_paper",
        "source": "https://arxiv.org/html/2610.11784v1",
        "source_count": entry.source_count,
        "slots": slots,
        "query_count": queries_phi.shape[-2],
        "queries_already_scaled": True,
        "max_lloyd_iterations": iterations,
        "ridge": float(ridge),
        "query_chunk_size": query_chunk_size,
        "compute_dtype": "float64",
        "output_dtype": str(entry.key.dtype),
        "input_cache_tensor_bytes": entry.tensor_bytes,
        "prototype_count_assignment": "C_means_then_D_reassignment_with_farthest_empty_repair",
        "identity": slots == entry.source_count,
        "heads": [],
    }
    if slots == entry.source_count:
        identity = WeightedKVEntry(entry.key.detach().clone(), entry.value.detach().clone(),
                                   entry.log_mass.detach().clone(), entry.source_count)
        report["target_attention_passes"] = 0
        report["output_cache_tensor_bytes"] = identity.tensor_bytes
        return identity, report

    output_keys, output_values, output_masses = [], [], []
    for batch in range(entry.key.shape[0]):
        batch_keys, batch_values, batch_masses = [], [], []
        for head in range(entry.key.shape[1]):
            key = entry.key[batch, head].to(dtype=torch.float64)
            value = entry.value[batch, head].to(dtype=torch.float64)
            phi = queries_phi[batch, head].to(dtype=torch.float64)
            compact_key, anchor, counts, info = _joint_kv_prototypes(key, value, slots, iterations)
            stored_key = compact_key.to(dtype=entry.key.dtype)
            stored_anchor = anchor.to(dtype=entry.value.dtype)
            stored_mass = counts.to(dtype=torch.float64).log().to(dtype=entry.log_mass.dtype)
            fitted, fit_info = _fit_values(
                key, value, phi, stored_key.to(dtype=torch.float64),
                stored_anchor.to(dtype=torch.float64), stored_mass.to(dtype=torch.float64),
                ridge=float(ridge), query_chunk_size=query_chunk_size, output_dtype=entry.value.dtype,
            )
            batch_keys.append(stored_key)
            batch_values.append(fitted)
            batch_masses.append(stored_mass.unsqueeze(0))
            report["heads"].append({"batch": batch, "head": head, **info, **fit_info})
        output_keys.append(torch.stack(batch_keys))
        output_values.append(torch.stack(batch_values))
        output_masses.append(torch.stack(batch_masses))
    compiled = WeightedKVEntry(torch.stack(output_keys), torch.stack(output_values),
                               torch.stack(output_masses), entry.source_count)
    report["target_attention_passes"] = 2
    report["output_cache_tensor_bytes"] = compiled.tensor_bytes
    return compiled, report
