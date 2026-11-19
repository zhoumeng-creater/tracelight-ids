"""Fixed algebraic counterexamples and F-only input support; no model calls."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "coverage_basis_20261119" / "batch" / "rows.jsonl.gz"


def phi(u: float) -> np.ndarray:
    # The interior of a two-bin PLE with knots 0, 1, 2.
    return np.array([np.clip(u, 0.0, 1.0), np.clip(u - 1.0, 0.0, 1.0)])


def residual_projector(secants: np.ndarray) -> np.ndarray:
    return np.eye(secants.shape[0]) - secants @ np.linalg.pinv(secants)


def algebra() -> dict:
    endpoints = np.column_stack([phi(2.0) - phi(0.0)])
    p = residual_projector(endpoints)
    mid = p @ (phi(1.0) - phi(0.0))
    assert np.allclose(p @ endpoints, 0.0)
    assert np.allclose(mid, [0.5, -0.5])
    full = residual_projector(np.column_stack([phi(1.0), phi(2.0)]))
    assert np.allclose(full, 0.0)
    # Identical endpoint labels admit both benign and malicious middle points.
    assert np.isclose(float(mid @ mid), 0.5)
    # Group-centred projection that preserves group identity leaves group offset free.
    normal_group_offsets = [-10.0, -5.0]
    assert normal_group_offsets[1] > normal_group_offsets[0]
    # Collapsing all affine-span directions also collapses off-manifold attacks.
    circle_normals = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    pc = residual_projector((circle_normals[1:] - circle_normals[0]).T)
    assert np.allclose(pc, 0.0)
    # Units are arbitrary: measuring the first PLE coordinate 2x changes the metric.
    scale = np.diag([2.0, 1.0])
    ps = residual_projector(scale @ endpoints)
    scaled_mid = ps @ scale @ phi(1.0)
    assert np.isclose(float(scaled_mid @ scaled_mid), 0.8)
    return {
        "checks_passed": 6,
        "endpoint_secants_annihilated": True,
        "middle_residual": mid.tolist(),
        "middle_squared_norm": float(mid @ mid),
        "middle_label_not_identifiable_from_two_normal_endpoints": True,
        "normal_middle_included_then_all_two_bin_directions_collapsed": True,
        "within_component_invariance_does_not_fix_between_component_score_offsets": normal_group_offsets,
        "curved_normal_manifold_affine_span_can_erase_off_manifold_attack": True,
        "arbitrary_coordinate_scaling_changes_middle_squared_norm_to": float(scaled_mid @ scaled_mid),
    }


def input_support() -> dict:
    with gzip.open(SOURCE, "rt", encoding="utf-8") as stream:
        # Do not inspect V/C/external labels or construct any candidate prediction.
        rows = [r for line in stream if (r := json.loads(line))["dataset_role"] in {"F_base", "F_expansion"}
                and r["target_label"] in {0, 1}]
    result = {}
    fields = ("ip_protocol", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
    for cohort in ("F_base", "F_expanded"):
        selected = [r for r in rows if cohort == "F_expanded" or r["dataset_role"] == "F_base"]
        normals = [r for r in selected if r["target_label"] == 0]
        groups = defaultdict(list)
        for r in normals:
            groups[tuple(r["raw8"][k] for k in fields)].append(r)
        varying = {k: v for k, v in groups.items()
                   if len({r["raw8"]["duration_seconds"] for r in v if r["raw8"]["duration_seconds"] is not None}) >= 2}
        cross = {k: v for k, v in varying.items() if len({r["allocation_group"] for r in v}) >= 2}
        attacks = [r for r in selected if r["target_label"] == 1]
        result[cohort] = {
            "normal_rows": len(normals), "attack_rows": len(attacks),
            "attack_target_kinds": dict(Counter(r["target_kind"] for r in attacks)),
            "normal_condition_groups": len(groups),
            "singleton_normal_condition_groups": sum(len(v) == 1 for v in groups.values()),
            "normal_groups_with_two_observed_distinct_durations": len(varying),
            "normal_rows_in_these_groups": sum(map(len, varying.values())),
            "cross_allocation_group_distinct_duration_normal_groups": len(cross),
            "attack_rows_with_any_same_condition_normal_support": sum(tuple(r["raw8"][k] for k in fields) in groups for r in attacks),
            "attack_rows_with_same_condition_normal_duration_pair": sum(tuple(r["raw8"][k] for k in fields) in varying for r in attacks),
        }
    return {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "condition_fields": list(fields), "cohorts": result,
        "interpretation": "Observed normal pairs are label-consistent examples, not interventions or counterfactual nuisance pairs.",
    }


if __name__ == "__main__":
    report = {"real_model_calls": 0, "fitting_calls": 0, "threshold_computations": 0,
              "algebra": algebra(), "f_only_support": input_support()}
    dest = HERE / "representation_mechanism_checks.json"
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
