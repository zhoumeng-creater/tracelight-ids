"""Check initialization and 15-round Lloyd execution on the existing toy.

This imports only the pure-stdlib mathematical reference. No real data, model,
training, or V-fit is used. Running this file writes only its adjacent JSON.
"""

import json
import math
from pathlib import Path

import geometry_reference as reference


def distance(point, center, metric):
    return sum(g * (x - c) ** 2 for x, c, g in zip(point, center, metric))


def source_initialization(points, masses, metrics, count, mode):
    """Start at source ID 0; break every farthest-point tie by source ID."""
    selected = [0]
    steps = []
    while len(selected) < count:
        priorities = {}
        for i, point in enumerate(points):
            if i in selected:
                continue
            if mode == "euclidean":
                priority = min(distance(point, points[j], [1.0] * len(point))
                               for j in selected)
            elif mode == "source_weighted":
                priority = masses[i] * min(distance(point, points[j], metrics[i])
                                           for j in selected)
            else:
                raise ValueError(mode)
            priorities[i] = priority
        chosen = min(priorities, key=lambda i: (-priorities[i], i))
        steps.append({"selected_before": list(selected),
                      "candidate_priorities": priorities, "chosen": chosen})
        selected.append(chosen)
    return selected, steps


def assignments_for(points, centers, metrics):
    return [min(range(len(centers)),
                key=lambda j: (distance(point, centers[j], metrics[i]),
                               distance(point, centers[j], [1.0] * len(point)), j))
            for i, point in enumerate(points)]


def distortion(points, centers, assignments, masses, metrics):
    return sum(masses[i] * distance(point, centers[assignments[i]], metrics[i])
               for i, point in enumerate(points))


def update_centers(points, masses, metrics, assignments, count):
    centers = []
    for cluster in range(count):
        members = [i for i, label in enumerate(assignments) if label == cluster]
        # All tested initializations have nonempty clusters. Do not conceal an
        # empty cluster by inventing a new initialization policy in this audit.
        if not members:
            raise RuntimeError(f"Empty cluster {cluster} in this toy execution")
        center = []
        for d in range(len(points[0])):
            denominator = sum(masses[i] * metrics[i][d] for i in members)
            if denominator > 0.0:
                coordinate = sum(masses[i] * metrics[i][d] * points[i][d]
                                 for i in members) / denominator
            else:
                coordinate = (sum(masses[i] * points[i][d] for i in members)
                              / sum(masses[i] for i in members))
            center.append(coordinate)
        centers.append(center)
    return centers


def lloyd(points, masses, metrics, seeds, rounds=15):
    centers = [list(points[i]) for i in seeds]
    history = []
    previous_loss = math.inf
    for iteration in range(rounds):
        assignments = assignments_for(points, centers, metrics)
        before = distortion(points, centers, assignments, masses, metrics)
        centers = update_centers(points, masses, metrics, assignments, len(seeds))
        after = distortion(points, centers, assignments, masses, metrics)
        if after > before + 1e-12 or after > previous_loss + 1e-12:
            raise AssertionError("Lloyd objective increased")
        previous_loss = after
        history.append({"round": iteration + 1, "assignments": assignments,
                        "before_centroid": before, "after_centroid": after})
    final_assignments = assignments_for(points, centers, metrics)
    # A matching final assignment makes reported centers and source masses
    # consistent without an unreported 16th centroid update.
    stable = final_assignments == history[-1]["assignments"]
    return {"seeds": seeds, "centers": centers,
            "assignments": final_assignments, "stable": stable,
            "history": history,
            "objective": distortion(points, centers, final_assignments, masses, metrics)}


def run():
    source = reference.run()
    keys = source["source_states"]["keys"]
    values = source["source_states"]["values"]
    masses = source["source_states"]["masses"]
    scales = reference.scalar_scales(keys, values, masses)
    points = [[x / scales[0] for x in key] + [v / scales[1]]
              for key, v in zip(keys, values)]
    ordinary_metrics = [[1.0] * len(points[0]) for _ in points]
    state_metrics = source["per_source_state_diagonal_metrics"]
    common_seeds, common_steps = source_initialization(
        points, masses, ordinary_metrics, 3, "euclidean")
    sensitive_seeds, sensitive_steps = source_initialization(
        points, masses, state_metrics, 3, "source_weighted")
    cases = {}
    for name, metrics, seeds in [
        ("ordinary_euclidean_init", ordinary_metrics, common_seeds),
        ("state_metric_euclidean_init", state_metrics, common_seeds),
        ("state_metric_source_weighted_init", state_metrics, sensitive_seeds),
    ]:
        result = lloyd(points, masses, metrics, seeds, rounds=15)
        groups = [[i for i, label in enumerate(result["assignments"]) if label == j]
                  for j in range(3)]
        output_keys = [[z * scales[0] for z in center[:-1]] for center in result["centers"]]
        output_values = [center[-1] * scales[1] for center in result["centers"]]
        output_masses = [sum(masses[i] for i in group) for group in groups]
        result.update({"source_groups": groups,
                       "output_keys": output_keys, "output_values": output_values,
                       "output_masses": output_masses,
                       "evaluation_before_V_fit": reference.evaluate(
                           output_keys, output_values, output_masses)})
        cases[name] = result
    return {
        "scope": "Pure synthetic 15-round Lloyd solver audit; no V-fit or real model.",
        "source_states": source["source_states"], "scales": scales,
        "initialization": {"euclidean": common_steps, "source_weighted": sensitive_steps},
        "cases": cases,
        "criterion_exhaustive_best_merge": source["same_three_slot_comparison"]["calibration_geometry_merge"],
        "important_limit": "Source-weighted initialization is an explicit algorithm change. This toy does not establish global convergence, real-model utility, or utility after anchored V-fit.",
    }


if __name__ == "__main__":
    result = run()
    Path(__file__).with_name("solver_reference_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({name: {"seeds": case["seeds"],
                             "source_groups": case["source_groups"],
                             "stable": case["stable"],
                             "objective": case["objective"],
                             "evaluation_before_V_fit": case["evaluation_before_V_fit"]}
                      for name, case in result["cases"].items()}, indent=2))
