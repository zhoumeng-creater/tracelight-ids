"""Frozen-model F-range intervention: no training, no V, no automatic promotion.

The sole policy clips numeric encoded coordinates to their F extrema. It does
not select percentiles, features, rows or attack families using C outcomes.
Both old-threshold and re-registered R+C results are retained.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .budget_transfer import _cpu_network, _no_links, _scores, _state_hash, _training_frames, _working_point
from .common import code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .models import FeaturePreprocessor, load_bundle, save_bundle
from .research import _group_column
from .risk_research import _fixed_metrics, _infer
from .runner import verify_development_inputs
from .transfer_diagnostics import binary_rank_metrics
from .transfer_screen import verify_transfer_screen

COMPLETE = "INPUT_EXTRAPOLATION_COMPLETE.json"


@dataclass
class FRangeClipper:
    """Persistable wrapper around an unchanged, already-fitted preprocessor."""

    base: FeaturePreprocessor
    lower: np.ndarray
    upper: np.ndarray

    @property
    def selected_features(self):
        return self.base.selected_features

    @property
    def numeric_columns(self):
        return self.base.numeric_columns

    @property
    def categorical_columns(self):
        return self.base.categorical_columns

    @property
    def encoded_dim(self):
        return self.base.encoded_dim

    @classmethod
    def from_fitting_inputs(cls, base, fitting):
        if not isinstance(base, FeaturePreprocessor) or fitting.empty:
            raise ValueError("F range requires original fitted preprocessing and nonempty F")
        x = base.transform(fitting)
        k = len(base.numeric_columns)
        result = cls(copy.deepcopy(base), x[:, :k].min(axis=0), x[:, :k].max(axis=0))
        if not np.array_equal(result.transform(fitting), x):
            raise ValueError("F-range intervention changed fitting inputs")
        return result

    def transform(self, frame):
        x = self.base.transform(frame)
        k = len(self.numeric_columns)
        if self.lower.shape != (k,) or self.upper.shape != (k,) or np.any(self.lower > self.upper):
            raise ValueError("Invalid numeric F bounds")
        x[:, :k] = np.clip(x[:, :k], self.lower, self.upper)
        return x

    def range_report(self, frame):
        x = self.base.transform(frame)[:, :len(self.numeric_columns)]
        outside = (x < self.lower) | (x > self.upper)
        return {
            "rows": len(frame), "outside_any_numeric_range": int(outside.any(axis=1).sum()),
            "columns": [{"name": name, "lower_encoded": float(self.lower[j]),
                         "upper_encoded": float(self.upper[j]), "outside_rows": int(outside[:, j].sum())}
                        for j, name in enumerate(self.numeric_columns)],
        }


def paired_events(frame, old_scores, new_scores, old_threshold, new_threshold):
    """Count gains AND losses, including concentration in original input groups."""
    y = frame.label.to_numpy()
    old = np.asarray(old_scores, dtype=np.float64) >= old_threshold
    new = np.asarray(new_scores, dtype=np.float64) >= new_threshold
    group_column = _group_column(frame)

    def count(mask, include_groups=False):
        result = {"rows": int(mask.sum())}
        events = {"recovered_fn": mask & (y == 1) & ~old & new,
                  "new_fn": mask & (y == 1) & old & ~new,
                  "repaired_fp": mask & (y == 0) & old & ~new,
                  "new_fp": mask & (y == 0) & ~old & new}
        for name, event in events.items():
            result[name] = int(event.sum())
            result[name + "_feature_patterns"] = int(frame.loc[event, "feature_hash"].nunique())
            result[name + "_groups"] = int(frame.loc[event, group_column].nunique())
            if include_groups:
                columns = list(dict.fromkeys([group_column, "feature_hash", "attack_cat"]))
                result[name + "_group_patterns"] = (
                    frame.loc[event].groupby(columns, sort=True).size().rename("rows").reset_index().to_dict("records"))
        return result

    return {"group_column": group_column, "all": count(np.ones(len(frame), dtype=bool), True),
            "families": {str(f): count(frame.attack_cat.to_numpy() == f)
                         for f in sorted(frame.attack_cat.unique())}}


def evaluate_intervention(bundle, frames, original_threshold, max_fpr=.01, chunk_size=2048):
    """Artifact analysis only. Reuse sealed weights and estimate F-only bounds."""
    candidate = copy.deepcopy(bundle)
    candidate.preprocessor = FRangeClipper.from_fitting_inputs(bundle.preprocessor, frames["fit"])
    before = _state_hash(bundle.network_state)
    if _state_hash(candidate.network_state) != before:
        raise ValueError("Range intervention changed network weights")
    query = {r: frames[r] for r in ("reference", "constraint")}
    old, new = _scores(bundle, query, chunk_size), _scores(candidate, query, chunk_size)
    identity_working_point = _working_point(old, frames, max_fpr)
    working_point = _working_point(new, frames, max_fpr)
    identity = {"working_point": identity_working_point, "roles": {}}
    for role, frame in query.items():
        identity["roles"][role] = {"rank": binary_rank_metrics(frame.label.to_numpy(), old[role], max_fpr)}
        for policy, threshold in (("frozen_original_threshold", original_threshold),
                                  ("registered_R_C_threshold", identity_working_point["threshold"])):
            metric = _fixed_metrics(frame, old[role], threshold)
            metric["decision_rule"] = "float64(raw_score)>=explicit_" + policy
            identity["roles"][role][policy] = metric
    records = {}
    for policy, threshold in (("frozen_original_threshold", original_threshold),
                              ("registered_R_C_threshold", working_point["threshold"])):
        records[policy] = {}
        for role, frame in query.items():
            metric = _fixed_metrics(frame, new[role], threshold)
            metric["decision_rule"] = "float64(raw_score)>=explicit_" + policy
            records[policy][role] = {
                "metrics": metric, "rank": binary_rank_metrics(frame.label.to_numpy(), new[role], max_fpr),
                "paired": paired_events(frame, old[role], new[role], original_threshold, threshold),
                "paired_baseline": "same-CPU identity scores at the sealed original threshold",
                "paired_registered_identity": paired_events(frame, old[role], new[role],
                    identity_working_point["threshold"], threshold),
            }
    return candidate, {"network_state_sha256": before, "fitting_encoding_unchanged": True,
        "policy": "clip all numeric encoded coordinates to F min/max; categorical coordinates unchanged",
        "boundary_source": "F inputs only; no labels or C quantiles", "working_point": working_point,
        "identity_baseline": identity,
        "original_threshold": original_threshold,
        "range": {r: candidate.preprocessor.range_report(f) for r, f in frames.items()},
        "results": records}, {**{"original_" + r: v for r, v in old.items()},
                              **{"clipped_" + r: v for r, v in new.items()}}


def benchmark_pair(original, candidate, frame, *, repeats=25, blocks=3, seed=20261105):
    """Lab-only caller: mixed common inputs, paired order, raw timings retained."""
    rng = np.random.default_rng(seed)
    # Equal class representation is a measurement workload, not a deployment prevalence claim.
    pools = [np.flatnonzero(frame.label.to_numpy() == c) for c in (0, 1)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("Timing workload requires both classes")
    samples = {}
    for size in (1, 256):
        indices = []
        for rep in range(repeats * blocks):
            if size == 1:
                ix = rng.choice(pools[rep % 2], size=1)
            else:
                ix = np.concatenate([rng.choice(p, size=size // 2, replace=len(p) < size // 2) for p in pools])
                rng.shuffle(ix)
            indices.append(ix)
        samples[size] = indices
    nets = [_cpu_network(b) for b in (original, candidate)]
    bundles = (original, candidate)

    def call(i, part):
        return _infer(nets[i], bundles[i].preprocessor.transform(part), "cpu", max(1, len(part)))

    result = {"seed": seed, "workload": "alternating single-row classes; balanced batch256",
              "timing_blocks": blocks, "repeats_per_block": repeats,
              "scope": "laboratory mixed F/R/C development workload; not independent laptop acceptance",
              "includes": "DataFrame -> preprocessing -> network raw scores",
              "excludes": "disk CSV read, packet/flow feature extraction, threshold comparison",
              "batches": {}}
    for size, indices in samples.items():
        times = [[], []]
        orders = []
        for i in range(2):
            for _ in range(5):
                call(i, frame.iloc[indices[0]])
        for ix in indices:
            part = frame.iloc[ix]
            order = rng.permutation(2).tolist()
            orders.append(order)
            for i in order:
                start = time.perf_counter_ns()
                call(i, part)
                times[i].append((time.perf_counter_ns() - start) / 1e6)
        def timing_summary(values):
            return {"p50_ms": float(np.percentile(values, 50)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "rows_per_second": float(size * 1000 / np.mean(values))}

        result["batches"][str(size)] = {
            "sample_ids": [frame.iloc[ix].sample_id.astype(str).tolist() for ix in indices],
            "execution_order": orders, "order_labels": ["original", "clipped"],
            "paired_duration_ratios": (np.asarray(times[1]) / times[0]).tolist(),
            **{name: {"durations_ms": times[i], **timing_summary(times[i]),
                       "blocks": [timing_summary(times[i][j * repeats:(j + 1) * repeats])
                                  for j in range(blocks)]}
               for i, name in enumerate(("original", "clipped"))},
        }
    return result


def run_input_extrapolation(screen_dir, data_dir, output, *, measure_cpu=False):
    """Strictly bound lab audit; source byte identity is never relaxed."""
    screen, data, root = map(_no_links, (screen_dir, data_dir, output))
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in (screen, data)):
        raise ValueError("Audit output must be separate from immutable inputs")
    if root.exists():
        raise FileExistsError("Preserve existing audit output; use a new directory")
    done = verify_transfer_screen(screen)
    plan = read_json(screen / "plan.json")
    manifest = verify_development_inputs(data, splits=("train",))
    if manifest != plan["binding"]["data"]:
        raise ValueError("Prepared manifest differs from sealed screen; no semantic resume bypass")
    if measure_cpu and manifest.get("data_kind") != "synthetic":
        from .models import _torch
        torch = _torch()
        if not torch.cuda.is_available() or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper():
            raise ValueError("Formal real-data timing requires laboratory RTX4090D host")
    frames = _training_frames(screen, data)
    cfg = plan["binding"]["config"]
    summary = read_json(screen / "summary.json")
    models = [r for r in summary["auxiliary"] if r["kind"] == "hard_warmup"] + summary["runs"]
    expected = {f"warmups/{budget}-{capacity}-s17" for budget in ("all", "reduced")
                for capacity in ("small", "wide")} | {
                    f"students/{budget}-{method}-s17" for budget in ("all", "reduced")
                    for method in ("hard", "direct_pauc")}
    if len(models) != 8 or {r["relative_dir"] for r in models} != expected:
        raise ValueError("This fixed audit requires all eight seed-17 screen MLP artifacts")
    if manifest.get("data_kind") != "synthetic" and cfg["max_fpr"] != .01:
        raise ValueError("The fixed real-data audit requires the sealed 1% development budget")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {"source_screen_complete_sha256": sha256_file(screen / "TRANSFER_SCREEN_COMPLETE.json"),
        "source_screen_binding": done["binding_sha256"], "source": code_fingerprint(),
        "data": manifest, "environment": environment_snapshot(), "threads": cfg["threads"],
        "models": [r["relative_dir"] for r in models],
        "policies": ["identity", "F_numeric_minmax"], "new_model_fits": 0,
        "reserved_partitions_read": [], "validation_loaded": False, "automatic_expansion": False,
        "candidate_promotion": False, "measure_cpu": measure_cpu})
    records = []
    import torch
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(cfg["threads"])
    try:
        with threadpool_limits(limits=cfg["threads"]):
            for row in models:
                relative = row["relative_dir"]
                print(f"[input-audit] frozen {relative}; model fits=0", flush=True)
                bundle = load_bundle(screen / relative / "model.joblib")
                candidate, report, scores = evaluate_intervention(bundle, frames,
                    row["selection"]["working_point"]["threshold"], cfg["max_fpr"], cfg["chunk_size"])
                # Fresh CPU BLAS can differ slightly from sealed CPU/GPU score batches.
                # Refuse changes in frozen working-point decisions rather than silently rebasing.
                with np.load(screen / relative / "training-scores.npz") as archived:
                    tau = row["selection"]["working_point"]["threshold"]
                    mismatches = {r: int(((scores["original_" + r] >= tau) != (archived[r] >= tau)).sum())
                                  for r in ("reference", "constraint")}
                    score_delta = {r: float(np.max(np.abs(scores["original_" + r] - archived[r])))
                                   for r in ("reference", "constraint")}
                report["archived_decision_mismatches"] = mismatches
                report["archive_to_cpu_max_abs_score_delta"] = score_delta
                if any(mismatches.values()):
                    write_json(root / "REPLAY_MISMATCH.json", {"model": relative, "counts": mismatches})
                    raise ValueError("Frozen score replay changed decisions; investigate before comparison")
                folder = root / relative
                folder.mkdir(parents=True)
                save_bundle(candidate, folder / "diagnostic-candidate.joblib")
                report.update(relative_dir=relative, original_model_sha256=sha256_file(screen / relative / "model.joblib"),
                    original_package_bytes=(screen / relative / "model.joblib").stat().st_size,
                    diagnostic_package_bytes=(folder / "diagnostic-candidate.joblib").stat().st_size)
                if measure_cpu:
                    report["cpu_pipeline"] = benchmark_pair(bundle, candidate, frames["constraint"])
                np.savez_compressed(folder / "scores.npz", **scores)
                write_json(folder / "report.json", report)
                records.append(report)
    finally:
        torch.set_num_threads(previous_threads)
    write_json(root / "summary.json", {"models": records, "new_model_fits": 0,
        "automatic_expansion": False, "independent_confirmation": False})
    write_json(root / COMPLETE, {"status": "input_extrapolation_audit_complete", "created_at": utc_now(),
        "new_model_fits": 0, "n_models": len(records), "validation_loaded": False,
        "reserved_partitions_read": [], "automatic_expansion": False,
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}})
    return read_json(root / COMPLETE)
