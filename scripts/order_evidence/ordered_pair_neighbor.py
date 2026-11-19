"""Exact, label-free same-state neighbor for the directional memory proposal.

No datasets or estimators are loaded. The caller owns roles, host grouping and
availability batches. Call advance(t), read every query, then update_batch().
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def safe_ratio(numerator, denominator):
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


class OrderedPairNeighbor:
    """P plus directed off-diagonal B: exactly dimension**2 state scalars."""

    def __init__(self, dimension: int, rate: float = 0.0):
        self.dimension = dimension
        self.rate = rate
        self.time = None
        self.p = np.zeros(dimension, dtype=float)
        self.row, self.col = np.where(~np.eye(dimension, dtype=bool))
        self.b = np.zeros(dimension * (dimension - 1), dtype=float)

    def advance(self, time: float):
        if self.time is not None:
            if time < self.time:
                raise ValueError("availability must be monotone")
            rho = np.exp(-self.rate * (time - self.time))
            self.p *= rho
            self.b *= rho * rho
        self.time = time

    def update_batch(self, observations):
        values = np.asarray(observations, dtype=float)
        if values.size == 0:
            return
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("observations must have shape (n, dimension)")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("finite nonnegative encodings required")
        total = values.sum(axis=0)
        self.b += self.p[self.row] * total[self.col]
        self.p += total

    def matrices(self):
        b = np.zeros((self.dimension, self.dimension))
        b[self.row, self.col] = self.b
        return b, b - b.T, b + b.T

    def read(self, query):
        query = np.asarray(query, dtype=float)
        if query.shape != self.p.shape or np.any(query < 0) or not np.all(np.isfinite(query)):
            raise ValueError("query must be finite, nonnegative and match dimension")
        g = query / query.sum() if query.sum() > 0 else np.zeros_like(query)
        b, a, s = self.matrices()
        support = s @ g
        candidate_direction = safe_ratio(a, s) @ g
        pooled_direction = safe_ratio(a @ g, support)
        supported_query_mass = (s > 0) @ g
        pooled_mass_matched_direction = supported_query_mass * pooled_direction
        common_support = np.log1p(support)
        first_order = np.log1p(self.p)
        return {
            "candidate": np.concatenate([candidate_direction, common_support, first_order]),
            "pool_then_normalize": np.concatenate([pooled_direction, common_support, first_order]),
            "pool_then_normalize_matched": np.concatenate([pooled_mass_matched_direction, common_support, first_order]),
            "pair_direction": candidate_direction,
            "pooled_direction": pooled_direction,
            "pooled_mass_matched_direction": pooled_mass_matched_direction,
            "supported_query_mass": supported_query_mass,
            "support": support,
            "g": g,
            "state": np.concatenate([self.p, self.b]),
            "candidate_ratio_matrix": safe_ratio(a, s),
        }

    def batch(self, time, observations, queries=()):
        self.advance(time)
        readouts = [self.read(q) for q in queries]
        self.update_batch(observations)
        return readouts


def checks():
    rng = np.random.default_rng(20261119)
    r = 13
    n = OrderedPairNeighbor(r, 0.03)
    p = np.zeros(r)
    a = np.zeros((r, r))
    s = np.zeros((r, r))
    last_time = None
    errors = []
    for step in range(40):
        time = float(step * step + step)
        rho = np.exp(-0.03 * (time - last_time)) if last_time is not None else 1.0
        p *= rho
        a *= rho * rho
        s *= rho * rho
        n.advance(time)
        query = rng.random(r)
        g = query / query.sum()
        expected = np.concatenate([safe_ratio(a, s) @ g, np.log1p(s @ g), np.log1p(p)])
        actual = n.read(query)
        errors.append(float(np.max(np.abs(actual["candidate"] - expected))))
        assert np.allclose(actual["candidate"], expected, atol=1e-12, rtol=1e-12)
        _, aa, ss = n.matrices()
        assert np.allclose(aa, a) and np.allclose(ss, s)
        # The inverse state transform is exact up to floating-point rounding.
        reconstructed = (s + a) / 2
        assert np.allclose(reconstructed[n.row, n.col], n.b)
        observations = rng.random((1 + step % 3, r))
        total = observations.sum(axis=0)
        cross = np.outer(p, total)
        a += cross - cross.T
        s += cross + cross.T
        np.fill_diagonal(s, 0)
        p += total
        n.update_batch(observations)
        last_time = time

    def build(sequence):
        state = OrderedPairNeighbor(3)
        for t, token in enumerate(sequence):
            state.batch(float(t), [np.eye(3)[token]])
        return state

    left, right = build([0, 1, 2, 0, 1, 2]), build([1, 0, 0, 2, 2, 1])
    l, rr = left.read(np.ones(3)), right.read(np.ones(3))
    assert not np.array_equal(left.b, right.b)
    assert np.allclose(l["candidate"], rr["candidate"])
    assert np.allclose(left.p, right.p)
    assert np.allclose(left.matrices()[2], right.matrices()[2])

    # Opposite sparse and common relations demonstrate noncommuting pooling.
    mix = OrderedPairNeighbor(3)
    for i, j, value in [(0, 1, 1.0), (2, 0, 100.0)]:
        mix.b[(mix.row == i) & (mix.col == j)] = value
    # This algebraic state is only a local normalization example, not asserted
    # to be the state of a physical sequence with the arbitrary zero P here.
    v = mix.read(np.array([0.0, 0.5, 0.5]))
    assert np.isclose(v["pair_direction"][0], 0.0)
    assert np.isclose(v["pooled_direction"][0], -99 / 101)

    return {
        "scope": "algebra_and_synthetic_only; module reusable for real prefixes",
        "real_model_calls": 0,
        "full_state_and_all_readout_equivalence_max_absolute_error": max(errors),
        "equivalence_batches": 40,
        "state_scalar_count": int(n.p.size + n.b.size),
        "r13_float32_payload_plus_float64_clock": 4 * r * r + 8,
        "both_readout_dimensions": [len(v["candidate"]), len(v["pool_then_normalize"])],
        "query_compression_collision": {
            "left_tokens": [0, 1, 2, 0, 1, 2],
            "right_tokens": [1, 0, 0, 2, 2, 1],
            "g": l["g"].tolist(),
            "left_A_upper": left.matrices()[1][np.triu_indices(3, 1)].tolist(),
            "right_A_upper": right.matrices()[1][np.triu_indices(3, 1)].tolist(),
            "same_all_candidate_inputs": True,
            "candidate_readout": l["candidate"].tolist(),
        },
        "normalization_order_example": {
            "pair_first_direction": v["pair_direction"].tolist(),
            "pool_first_direction": v["pooled_direction"].tolist(),
            "pool_first_mass_matched_direction": v["pooled_mass_matched_direction"].tolist(),
            "supported_query_mass": v["supported_query_mass"].tolist(),
            "shared_support": v["support"].tolist(),
            "physical_sequence_claim": False,
        },
        "not_a_detection_result": True,
    }


if __name__ == "__main__":
    result = checks()
    destination = Path(__file__).with_name("ordered_pair_neighbor_checks.json")
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks_passed": True, "result": str(destination)}))
