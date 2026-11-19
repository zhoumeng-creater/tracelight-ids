"""Synthetic algebra only: no datasets, checkpoints or model calls.

Run: python check_distillation_algebra.py
Writes check_distillation_algebra.json beside this file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def threshold(normal, beta):
    values = np.asarray(normal, dtype=np.float64)
    budget = int(np.floor(beta * len(values)))
    assert 0 <= budget < len(values)
    return float(np.sort(values)[len(values) - budget - 1]), budget


def interval_bits(scores, normal, betas):
    anchors = np.array([threshold(normal, b)[0] for b in betas])
    return np.asarray(scores)[:, None] > anchors[None, :]


def ordinal_loss(scores, anchors, bits, tau=0.7):
    sign = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    margins = (np.asarray(scores)[:, None] - np.asarray(anchors)[None, :]) / tau
    return float(np.logaddexp(0.0, -sign * margins).mean())


def ordinal_score_gradient(scores, anchors, bits, tau=0.7):
    """Thresholds are constants in this derivative (stop-gradient)."""
    sign = 2.0 * np.asarray(bits, dtype=np.float64) - 1.0
    margins = (np.asarray(scores)[:, None] - np.asarray(anchors)[None, :]) / tau
    return (-sign * sigmoid(-sign * margins) / tau).mean(axis=1) / len(scores)


def bernoulli_kl(teacher, student_logits):
    q = np.asarray(teacher)
    log_p = -np.logaddexp(0.0, -np.asarray(student_logits))
    log_one_minus_p = -np.logaddexp(0.0, np.asarray(student_logits))
    return float((q * (np.log(q) - log_p)
                  + (1.0 - q) * (np.log1p(-q) - log_one_minus_p)).mean())


def finite_difference(fn, values, h=1e-6):
    x = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = h
        result[i] = (fn(x + delta) - fn(x - delta)) / (2.0 * h)
    return result


def check_budget_and_ties():
    tied = np.array([0.1, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.8, 0.9, 0.9])
    cases = []
    for name, values, beta, expected_fp in [
        ("ties_budget_two", tied, 0.2, 2),
        ("ties_budget_one_is_not_spent", tied, 0.1, 0),
        ("zero_budget", tied, 0.0, 0),
        ("small_n_1_percent", np.linspace(0, 1, 17), 0.01, 0),
        ("all_equal", np.full(200, 0.4), 0.01, 0),
    ]:
        anchor, budget = threshold(values, beta)
        fp = int(np.sum(values > anchor))
        assert fp == expected_fp and fp <= budget
        assert not (anchor > anchor)
        assert np.nextafter(anchor, np.inf) > anchor
        cases.append(dict(name=name, n=len(values), beta=beta, budget=budget,
                          threshold=anchor, strict_fp=fp, ties=int(np.sum(values == anchor))))
    return dict(passed=True, cases=cases)


def check_magnitude_invariance_and_coarsening():
    normal = np.linspace(0.01, 0.99, 400)
    betas = [0.0025, 0.005, 0.01]
    teacher = np.array([0.2, 0.3, 0.984, 0.987, 0.9895, 0.999])
    transform = lambda q: sigmoid(np.log(q / (1.0 - q)) / 2.0)
    transformed = transform(teacher)
    bits = interval_bits(teacher, normal, betas)
    other_bits = interval_bits(transformed, transform(normal), betas)
    assert np.array_equal(bits, other_bits)

    student = np.array([-0.3, 0.2, 0.6, 0.8, 0.9, 1.3])
    student_normal = np.linspace(-1.1, 1.0, 400)
    anchors = [threshold(student_normal, b)[0] for b in betas]
    loss = ordinal_loss(student, anchors, bits)
    other_loss = ordinal_loss(student, anchors, other_bits)
    grad = ordinal_score_gradient(student, anchors, bits)
    other_grad = ordinal_score_gradient(student, anchors, other_bits)
    kl_grad = (sigmoid(student) - teacher) / len(teacher)
    transformed_kl_grad = (sigmoid(student) - transformed) / len(teacher)
    assert loss == other_loss and np.array_equal(grad, other_grad)
    assert not np.allclose(kl_grad, transformed_kl_grad)

    # A finite interval grid does not preserve the order inside an interval.
    reordered = teacher.copy()
    reordered[:2] = reordered[:2][::-1]
    assert teacher[0] < teacher[1] and reordered[0] > reordered[1]
    assert np.array_equal(bits, interval_bits(reordered, normal, betas))
    assert np.unique(bits, axis=0).shape[0] > 1  # Not a constant-target fixture.
    return dict(
        passed=True, betas=betas, teacher=teacher.tolist(),
        transformed_teacher=transformed.tolist(), interval_bits=bits.astype(int).tolist(),
        ordinal_losses=[loss, other_loss],
        ordinal_gradient_max_difference=float(np.max(np.abs(grad - other_grad))),
        kl_losses=[bernoulli_kl(teacher, student), bernoulli_kl(transformed, student)],
        kl_gradient_max_difference=float(np.max(np.abs(kl_grad - transformed_kl_grad))),
        within_interval_order_reversal_keeps_targets=True,
    )


def check_projection_equivalence():
    # Deliberately rank-deficient, with a non-zero unreachable residual.
    j = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2], [2, -1, 1], [-1, 2, 1]], dtype=float)
    theta0 = np.array([0.2, -0.1, 0.4])
    offset = np.array([0.1, -0.2, 0.3, 0.0, 0.4])
    target = np.array([0.3, 0.8, -0.4, 1.7, -0.5])
    s0 = offset + j @ theta0
    residual = target - s0
    projection = j @ np.linalg.pinv(j)
    projected_target = s0 + projection @ residual
    gradient = -j.T @ residual
    projected_gradient = -j.T @ projection @ residual
    np.testing.assert_allclose(gradient, projected_gradient, atol=1e-12, rtol=1e-12)
    unreachable_norm = float(np.linalg.norm(residual - projection @ residual))
    assert unreachable_norm > 0.1
    theta, projected_theta = theta0.copy(), theta0.copy()
    trajectory_errors, loss_constant_errors = [], []
    expected_constant = 0.5 * unreachable_norm ** 2
    for _ in range(10):
        score = offset + j @ theta
        projected_score = offset + j @ projected_theta
        raw_loss = 0.5 * np.sum((score - target) ** 2)
        proj_loss = 0.5 * np.sum((score - projected_target) ** 2)
        loss_constant_errors.append(abs(raw_loss - proj_loss - expected_constant))
        theta -= 0.02 * j.T @ (score - target)
        projected_theta -= 0.02 * j.T @ (projected_score - projected_target)
        trajectory_errors.append(float(np.max(np.abs(theta - projected_theta))))
    assert max(trajectory_errors) < 1e-12
    assert max(loss_constant_errors) < 1e-12
    return dict(passed=True, jacobian_shape=list(j.shape), rank=int(np.linalg.matrix_rank(j)),
                unreachable_residual_norm=unreachable_norm,
                initial_gradient_max_difference=float(np.max(np.abs(gradient - projected_gradient))),
                synthetic_sgd_steps=10, trajectory_max_difference=max(trajectory_errors),
                objective_constant_difference=expected_constant,
                constant_difference_max_error=max(loss_constant_errors))


def check_stop_gradient_and_quantile_ties():
    scores = np.array([-1.2, -0.4, 0.1, 0.9, 1.3])
    normal_indices = np.array([0, 1, 2, 3])
    beta, tau = 0.25, 0.7
    bits = np.array([[0], [0], [1], [1], [1]], dtype=bool)
    anchor = threshold(scores[normal_indices], beta)[0]
    frozen_fn = lambda z: ordinal_loss(z, [anchor], bits, tau)
    recomputed_fn = lambda z: ordinal_loss(
        z, [threshold(z[normal_indices], beta)[0]], bits, tau)
    frozen_gradient = ordinal_score_gradient(scores, [anchor], bits, tau)
    frozen_fd = finite_difference(frozen_fn, scores)
    # The selected normal order statistic is unique here. Its derivative is e_k.
    anchor_index = int(normal_indices[np.flatnonzero(scores[normal_indices] == anchor)[0]])
    full_gradient = frozen_gradient.copy()
    full_gradient[anchor_index] -= frozen_gradient.sum()
    full_fd = finite_difference(recomputed_fn, scores)
    np.testing.assert_allclose(frozen_gradient, frozen_fd, atol=1e-9, rtol=1e-7)
    np.testing.assert_allclose(full_gradient, full_fd, atol=1e-9, rtol=1e-7)
    assert np.max(np.abs(frozen_gradient - full_gradient)) > 1e-3
    assert abs(float(full_gradient.sum())) < 1e-12
    shift = np.full_like(scores, 0.123)
    assert abs(recomputed_fn(scores + shift) - recomputed_fn(scores)) < 1e-12
    assert abs(frozen_fn(scores + shift) - frozen_fn(scores)) > 1e-4

    # At tied quantiles, the composed objective need not have a unique derivative.
    tied_scores = np.array([-0.5, 0.0, 0.0, 1.0])
    tied_bits = np.array([[0], [1], [0], [1]], dtype=bool)
    tied_fn = lambda z: ordinal_loss(z, [threshold(z, 0.5)[0]], tied_bits, 1.0)
    h = 1e-7
    delta = np.array([0, h, 0, 0])
    right = (tied_fn(tied_scores + delta) - tied_fn(tied_scores)) / h
    left = (tied_fn(tied_scores) - tied_fn(tied_scores - delta)) / h
    assert abs(right - left) > 1e-3
    return dict(
        passed=True, unique_anchor=anchor, anchor_index=anchor_index,
        frozen_gradient=frozen_gradient.tolist(), full_quantile_gradient=full_gradient.tolist(),
        frozen_finite_difference_max_error=float(np.max(np.abs(frozen_gradient - frozen_fd))),
        full_finite_difference_max_error=float(np.max(np.abs(full_gradient - full_fd))),
        stopped_vs_full_max_difference=float(np.max(np.abs(frozen_gradient - full_gradient))),
        common_shift_invariant_when_quantile_recomputed=True,
        common_shift_invariant_when_quantile_frozen=False,
        tied_quantile_left_derivative=float(left), tied_quantile_right_derivative=float(right),
        tied_quantile_unique_derivative=False,
    )


def main():
    path = Path(__file__).resolve()
    checks = {
        "strict_budget_and_ties": check_budget_and_ties(),
        "teacher_magnitude_and_finite_intervals": check_magnitude_invariance_and_coarsening(),
        "jacobian_projection_equivalence": check_projection_equivalence(),
        "stop_gradient_is_not_full_quantile_derivative": check_stop_gradient_and_quantile_ties(),
    }
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope="synthetic_algebra_only", numpy_version=np.__version__,
        script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        real_model_training_calls=0, real_model_forward_calls=0,
        real_data_reads=0, checks=checks,
        all_passed=all(c["passed"] for c in checks.values()),
    )
    destination = path.with_suffix(".json")
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": result["all_passed"], "checks": len(checks),
                      "result": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
