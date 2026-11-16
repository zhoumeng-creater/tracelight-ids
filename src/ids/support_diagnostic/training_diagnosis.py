"""Frozen R26 training diagnosis: shared rows, original objectives, no fitting."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.flow_history.execution import pack, run_lock, stage
from ids.threshold_validation.observations import RAW_FEATURES
from . import data
from .evaluation import oracle_cell

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "configs/training-diagnosis-20261116/registration.json"
IDENTITY = ["source_id", "row_id", "family", "domain", "label", "weight"]


def bce(logits, labels):
    """Stable Bernoulli loss from the saved raw logits, not clipped probabilities."""
    scores, y = np.asarray(logits, dtype=np.float64), np.asarray(labels)
    if scores.shape != y.shape or scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Expected finite row-aligned logits")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Binary labels are required")
    return np.logaddexp(0.0, np.where(y == 1, -scores, scores))


def collision_floor(values, labels, weights):
    """Empirical BCE infimum over arbitrary functions of exactly these inputs."""
    x, y, w = np.asarray(values), np.asarray(labels), np.asarray(weights, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or y.shape != w.shape or not len(x):
        raise ValueError("Input, label, and weight shapes differ")
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Finite inputs and positive weights are required")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Binary labels are required")
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=w)
    positive = np.bincount(inverse, weights=w * y)
    negative = mass - positive
    mixed = (positive > 0) & (negative > 0)
    p = positive[mixed] / mass[mixed]
    entropy = -(p * np.log(p) + (1 - p) * np.log1p(-p))
    return {
        "rows": len(x), "unique_inputs": len(mass),
        "conflicting_inputs": int(mixed.sum()),
        "rows_in_conflicting_inputs": int(mixed[inverse].sum()),
        "weighted_bce_infimum": float(np.sum(mass[mixed] * entropy) / w.sum()),
        "weighted_error_infimum": float(np.minimum(positive, negative).sum() / w.sum()),
        "meaning": "in-sample unrestricted-function lower bound, not a model optimum or generalization bound",
    }


def validate_frame(frame, identities):
    """Bind reconstructed/original numeric rows to the frozen physical identities."""
    required = set(IDENTITY) | set(RAW_FEATURES)
    if not required <= set(frame):
        raise ValueError(f"Training frame missing columns: {sorted(required - set(frame))}")
    if frame.duplicated(["source_id", "row_id"]).any():
        raise ValueError("Duplicate physical training identity")
    expected = identities.loc[identities.support.eq("mixed"), IDENTITY].reset_index(drop=True)
    actual = frame[IDENTITY].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    if not np.array_equal(frame.label, frame.family.ne("Benign").astype(int)):
        raise ValueError("Family and binary label disagree")
    if not np.isfinite(frame[list(RAW_FEATURES)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite training input")
    return frame.reset_index(drop=True)


def verify_artifacts(r26, registration):
    for rel, expected in registration["source_artifacts"].items():
        path = r26 / rel
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"R26 frozen artifact missing or changed: {path}")


def load_frame(r26, recovered=None):
    reg = read_json(REGISTRY)
    original = r26 / "local/training/mixed.parquet"
    path = Path(recovered) if recovered else original
    if not path.is_file():
        raise FileNotFoundError(
            f"Training numeric rows are missing: {path}. Use the retained R26 laboratory run, "
            "or recover registered rows with scripts/recover_r26_training_rows.py; do not rerun R26.")
    if recovered is None and sha256_file(path) != reg["mixed_training_sha256"]:
        raise ValueError("Original R26 mixed training parquet digest differs")
    if recovered is not None:
        receipt = read_json(path.with_suffix(".json"))
        bindings = {"output_sha256": sha256_file(path), "protocol_sha256": sha256_file(r26 / "protocol.json"),
                    "training_identity_sha256": sha256_file(r26 / "evidence/replay/training-identities.parquet")}
        if any(receipt.get(key) != digest for key, digest in bindings.items()):
            raise ValueError("Recovered inputs differ from their original-source recovery receipt")
    frame = pd.read_parquet(path)
    identities = pd.read_parquet(r26 / "evidence/replay/training-identities.parquet")
    return validate_frame(frame, identities), path


def objective(frame, arm):
    support = arm.split("-", 1)[0]
    mask = frame.domain.eq("target").to_numpy() if support == "target" else np.ones(len(frame), bool)
    weights = data.mass_weights(frame.loc[mask], support)
    return mask, weights


def effective_inputs(frame, arm, r26):
    """Inspect stored preprocessing tensors only; never construct/evaluate a network."""
    from ids.flow_history import model as legacy
    from .model import view_frame
    masked = view_frame(frame, "ports_zero8")
    group = arm.rsplit("-", 1)[1]
    if group == "hgb":
        return masked[list(RAW_FEATURES)].to_numpy(dtype=np.float64), "raw_float64_before_HGB_bins"
    import torch
    state = torch.load(r26 / "models" / arm / (group + ".pt"), map_location="cpu", weights_only=True)
    values = legacy._transform(masked, RAW_FEATURES)
    # Match legacy _tensor's arithmetic and final FP32 cast without a model forward.
    result = np.ascontiguousarray((values - state["mean"].numpy()) / state["scale"].numpy(), dtype=np.float32)
    mask, _ = objective(frame, arm)
    # Recreate the original per-arm array layout before reduction: boolean slicing
    # a transformed union changes NumPy's summation order at ~1e-13.
    own_values = legacy._transform(masked.loc[mask], RAW_FEATURES)
    mean, scale = own_values.mean(axis=0), own_values.std(axis=0)
    scale[scale == 0] = 1
    if not np.array_equal(mean, state["mean"].numpy()) or not np.array_equal(scale, state["scale"].numpy()):
        raise ValueError(f"Recovered inputs do not reproduce frozen training normalization: {arm}")
    return result, "model_standardized_float32"


def input_analysis(frame, r26, arms):
    from .model import view_frame
    common = view_frame(frame, "ports_zero8")[list(RAW_FEATURES)].to_numpy(dtype=np.float64)
    rows = []
    source = frame.domain.eq("source").to_numpy()
    for view, values in (("source_raw8", frame[list(RAW_FEATURES)].to_numpy(dtype=float)),
                         ("source_ports_zero8", common)):
        rows.append({"arm": "source-statistics-only", "representation": view,
                     "scope": "source_uniform_objective",
                     **collision_floor(values[source], frame.label.to_numpy()[source], np.ones(source.sum()))})
    for arm in arms:
        mask, weights = objective(frame, arm)
        effective, kind = effective_inputs(frame, arm, r26)
        for name, values in (("common_raw_ports_zero8", common), (kind, effective)):
            result = collision_floor(values[mask], frame.label.to_numpy()[mask], weights)
            rows.append({"arm": arm, "representation": name, "scope": "own_training_objective", **result})
    return rows


def metrics(frame, scores, threshold, *, arm, scope, value, split, seen):
    losses = bce(scores, frame.label.to_numpy())
    normal = frame.label.eq(0).to_numpy()
    attack = ~normal
    alerts = np.asarray(scores) > threshold
    normal_n, attack_n = int(normal.sum()), int(attack.sum())
    result = {
        "arm": arm, "split": split, "scope": scope, "value": value, "seen_in_fit": seen,
        "rows": len(frame), "normal_n": normal_n, "attack_n": attack_n,
        "bce_mean": float(losses.mean()),
        "bce_normal": float(losses[normal].mean()) if normal_n else None,
        "bce_attack": float(losses[attack].mean()) if attack_n else None,
        "frozen_threshold": threshold, "frozen_fp": int((alerts & normal).sum()),
        "frozen_tp": int((alerts & attack).sum()), "fp_budget": normal_n // 100,
    }
    result["frozen_fpr"] = result["frozen_fp"] / normal_n if normal_n else None
    result["frozen_qualified"] = result["frozen_fp"] <= normal_n // 100 if normal_n else None
    # Use the existing strict > / unsplittable ties diagnostic, including pooled descriptive scopes.
    proxy = frame[["row_id", "family"]].reset_index(drop=True).assign(
        row_id=np.arange(len(frame)), source_id="diagnostic_scope")
    oracle = oracle_cell(proxy, scores)
    result.update(oracle_tp=oracle["tp"] if normal_n else None,
                  oracle_fp=oracle["fp"] if normal_n else None,
                  oracle_threshold=oracle["threshold"], oracle_diagnostic_only=True)
    return result


def summarize_training(frame, scores, arm, rule):
    threshold = rule["threshold"]
    results = []
    domain_oracles = {}
    for domain, index in frame.groupby("domain", sort=True).indices.items():
        part = frame.iloc[index][["row_id", "family"]].reset_index(drop=True).assign(
            row_id=np.arange(len(index)), source_id="domain_diagnostic")
        domain_oracles[domain] = oracle_cell(part, scores[index])
    for scope, columns in (("domain", ["domain"]), ("source", ["source_id"]),
                           ("domain_family", ["domain", "family"])):
        for key, index in frame.groupby(columns, sort=True).indices.items():
            part = frame.iloc[index]
            value = "/".join(map(str, key if isinstance(key, tuple) else (key,)))
            seen = arm.startswith("mixed-") or part.domain.eq("target").all()
            row = metrics(part, scores[index], threshold, arm=arm, scope=scope,
                          value=value, split="training_union", seen=bool(seen))
            if scope == "domain_family":
                reference = domain_oracles[part.domain.iloc[0]]
                cutoff = reference["threshold"]
                row.update(domain_oracle_threshold=cutoff, domain_oracle_normal_n=reference["normal_n"],
                           domain_oracle_family_alerts=int((scores[index] > cutoff).sum()) if cutoff is not None else None)
            results.append(row)
    mask, weights = objective(frame, arm)
    losses = bce(scores[mask], frame.label.to_numpy()[mask])
    total_weight = float(weights.sum())
    contributions = []
    own = frame.loc[mask].reset_index(drop=True)
    for (domain, family), index in own.groupby(["domain", "family"], sort=True).indices.items():
        contributions.append({"arm": arm, "domain": domain, "family": family, "rows": len(index),
                              "objective_mass": float(weights[index].sum() / total_weight),
                              "bce_within_group": float(losses[index].mean()),
                              "contribution_to_objective": float(np.sum(losses[index] * weights[index]) / total_weight)})
    return results, {"arm": arm, "rows": int(mask.sum()),
                     "final_weighted_training_bce": float(np.average(losses, weights=weights)),
                     "objective_contributions": contributions}


def cached_evaluation(r26, arms, registration):
    results = []
    names = sorted(name for name in registration["source_artifacts"]
                   if name.startswith("evidence/replay/TQH-") and name.endswith(".parquet"))
    for name in names:
        path = r26 / name
        frame = pd.read_parquet(path)
        frame = frame.loc[frame.family.ne("Unresolved")].copy()
        frame["label"] = frame.family.ne("Benign").astype(int)
        for arm in arms:
            rule = read_json(r26 / "evidence/calibration" / (arm + ".json"))
            results.append(metrics(frame, frame[arm].to_numpy(), rule["threshold"], arm=arm,
                                   scope="cell", value=path.stem, split="cached_BD_development", seen=False))
    return results


def score_one(r26, output, frame, arm, threads):
    from . import model
    group = arm.rsplit("-", 1)[1]
    suffix = ".joblib" if group == "hgb" else ".pt"
    fitted = model.load(r26 / "models" / arm / (group + suffix), group=group,
                        threads=threads, device="cuda:0")
    started = time.perf_counter()
    scores = np.asarray(model.score(fitted, frame[list(RAW_FEATURES)], "ports_zero8", group), dtype=float)
    # Both existing backends return raw logits/decision_function, not probabilities.
    bce(scores, frame.label.to_numpy())
    path = output / "evidence/training_scores" / (arm + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[IDENTITY].assign(score=scores).to_parquet(path, index=False, compression="zstd")
    receipt = output / "evidence/scoring" / (arm + ".json")
    write_json(receipt, {"arm": arm, "new_fits": 0, "scored_rows": len(frame),
                         "model_device": "cpu" if group == "hgb" else "cuda:0",
                         "wall_seconds": time.perf_counter() - started})
    return [path, receipt]


def run(r26, output, *, recovered=None, inputs_only=False, threads=4):
    r26, output = Path(r26).resolve(), Path(output).resolve()
    if output == r26 or output.is_relative_to(r26) or r26.is_relative_to(output):
        raise ValueError("Use a separate output directory, outside the frozen R26 run")
    reg = read_json(REGISTRY)
    verify_artifacts(r26, reg)
    frame, frame_path = load_frame(r26, recovered)
    if len(frame) != reg["training_rows_per_arm"]:
        raise ValueError("Training union size differs from registration")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        binding = {"registration": canonical_hash(reg), "training_frame": sha256_file(frame_path),
                   "diagnosis_code": sha256_file(Path(__file__)), "threads": threads}
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            old_binding = read_json(protocol_path)
            if any(old_binding.get(key) != value for key, value in binding.items()):
                raise ValueError("Diagnostic binding changed; use a fresh output directory")
        if not inputs_only:
            from .runner import verify_runtime
            from ids.common import environment_snapshot
            runtime = environment_snapshot(include_torch=True)
            verify_runtime(runtime)
            if not runtime.get("torch", {}).get("cuda_available"):
                raise RuntimeError("Frozen model scoring belongs in the laboratory CUDA environment; use --inputs-only locally")
            binding.update(runtime=runtime["packages"], python=runtime["python"])
            if protocol_path.exists() and read_json(protocol_path) != binding:
                raise ValueError("Diagnostic runtime changed; use a fresh output directory")
        inputs = input_analysis(frame, r26, reg["arms"])
        write_json(output / "evidence/input_conflicts.json", {"new_model_forward_calls": 0, "results": inputs})
        pd.DataFrame(cached_evaluation(r26, reg["arms"], reg)).to_csv(output / "evidence/cached_BD.csv", index=False)
        if inputs_only:
            status = {"status": "inputs_and_cached_scores_complete", "new_fits": 0,
                      "new_model_forward_calls": 0, "training_diagnosis_complete": False}
            write_json(output / "input_analysis_status.json", status)
            if not (output / "outcome.json").exists():
                write_json(output / "outcome.json", status)
            return
        binding_hash = canonical_hash(binding)
        write_json(protocol_path, binding)
        write_json(output / "environment.json", runtime)
        results, objectives = [], []
        for arm in reg["arms"]:
            stage(output, "score-" + arm, binding_hash, partial(score_one, r26, output, frame, arm, threads))
            cached = pd.read_parquet(output / "evidence/training_scores" / (arm + ".parquet"))
            pd.testing.assert_frame_equal(cached[IDENTITY], frame[IDENTITY], check_dtype=False, check_exact=True)
            rule = read_json(r26 / "evidence/calibration" / (arm + ".json"))
            metrics_rows, loss = summarize_training(frame, cached.score.to_numpy(), arm, rule)
            results.extend(metrics_rows)
            floor = next(x for x in inputs if x["arm"] == arm and x["representation"] != "common_raw_ports_zero8")
            loss["effective_input_bce_infimum"] = floor["weighted_bce_infimum"]
            loss["gap_above_unrestricted_empirical_infimum"] = loss["final_weighted_training_bce"] - floor["weighted_bce_infimum"]
            objectives.append(loss)
        pd.DataFrame(results).to_csv(output / "evidence/training_metrics.csv", index=False)
        write_json(output / "evidence/final_training_objectives.json", objectives)
        verify_artifacts(r26, reg)
        write_json(output / "outcome.json", {"status": "complete", "new_fits": 0,
                   "models_scored": len(reg["arms"]), "scored_rows": len(reg["arms"]) * len(frame),
                   "new_BD_model_forward_calls": 0, "independent_confirmation": False,
                   "interpretation": "Training fit, exact-input conflicts, and held-out transport errors are separate; no automatic model/threshold selection."})
        pack(output, prefix="training-diagnosis-results")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r26-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovered-training", type=Path)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        run(args.r26_run, args.output, recovered=args.recovered_training,
            inputs_only=args.inputs_only, threads=args.threads)
    except Exception:
        # Existing per-arm receipts preserve completed scores. Keep the traceback
        # visible and never report a partial diagnosis as a successful result.
        import traceback
        if args.output.exists() and args.output.resolve() != args.r26_run.resolve() and not (
            args.output.resolve().is_relative_to(args.r26_run.resolve())
            or args.r26_run.resolve().is_relative_to(args.output.resolve())
        ):
            write_json(args.output / "last_failure.json", {"status": "failed", "trace": traceback.format_exc()})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
