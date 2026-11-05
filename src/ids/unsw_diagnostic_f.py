"""F-only measurements on frozen models; no fitting or optimizer construction."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy

from .common import write_json
from .models import _network
from .risk_research import _state, _state_hash
from .unsw_kd import KD_WEIGHT, class_weights

MODELS = ("anchor", "H40", "K40")
TERMS = ("hard_bce", "teacher_kl", "class_mean_kl", "mixed_bce")
BATCH_SIZE = 1024
IDENTITY = ("sample_id", "feature_hash", "label", "attack_cat")


def _entropy(p):
    return -xlogy(p, p) - xlogy(1-p, 1-p)


def _bce_prob(r, p):
    return -xlogy(r, p) - xlogy(1-r, 1-p)


def _cosine(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else None


def _checked_inputs(bundles, frame, targets):
    if set(bundles) != set(MODELS):
        raise ValueError("Exactly anchor/H40/K40 frozen bundles are required")
    if not isinstance(frame, pd.DataFrame) or not set(IDENTITY).issubset(frame.columns):
        raise ValueError("F requires sample_id/feature_hash/label/attack_cat")
    if frame.loc[:, IDENTITY].isna().any().any() or frame.sample_id.duplicated().any():
        raise ValueError("F identity must be complete with unique sample_id")
    for name in ("sample_id", "feature_hash", "attack_cat"):
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("F identity strings must be nonempty")
    y = frame.label.to_numpy(dtype=np.int64)
    w = class_weights(frame.label.to_numpy())
    if not np.array_equal(y, targets["labels"]) or not np.array_equal(w, targets["weights"]):
        raise ValueError("Frozen labels/weights do not match F")
    q = np.asarray(targets["teacher_q"], dtype=np.float64)
    qm = np.asarray(targets["class_mean_q"], dtype=np.float64)
    for values in (q, qm):
        if values.shape != y.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Aligned finite teacher probabilities in [0,1] required")
    means = np.asarray([q[y == c].mean() for c in (0, 1)])
    if not np.array_equal(qm, means[y]):
        raise ValueError("Class mean targets differ from frozen F teacher probabilities")
    r = (y + KD_WEIGHT*q)/(1+KD_WEIGHT)
    if not np.array_equal(targets["target_K"], r) or not np.array_equal(targets["target_H"], y):
        raise ValueError("Frozen H/K targets do not match their definitions")
    hashes = frame.feature_hash.to_numpy(dtype=str)
    unique, group = np.unique(hashes, return_inverse=True)
    if (not np.array_equal(unique, targets["feature_hash"])
            or not np.array_equal(group, targets["row_group_index"])):
        raise ValueError("Frozen feature patterns are not aligned with F")
    qc = pd.DataFrame({"group": group, "q": q}).groupby("group").q
    if not np.array_equal(qc.min().to_numpy(), qc.max().to_numpy()):
        raise ValueError("Teacher q must be constant within identical input patterns")
    anchor = bundles["anchor"]
    x = None
    for name in MODELS:
        bundle = bundles[name]
        if (bundle.model_name != "mlp" or bundle.network_state is None
                or bundle.encoded_dim != anchor.encoded_dim
                or tuple(bundle.hidden_dims) != tuple(anchor.hidden_dims)
                or bundle.selected_features != anchor.selected_features):
            raise ValueError("Three models must use the frozen common MLP structure")
        encoded = np.asarray(bundle.preprocessor.transform(frame), dtype=np.float32)
        if encoded.shape != (len(y), bundle.encoded_dim) or not np.isfinite(encoded).all():
            raise ValueError("F encoding must be finite with the expected dimensions")
        if x is None:
            x = encoded
        elif not np.array_equal(x, encoded):
            raise ValueError("Frozen model preprocessing differs on F")
    return y, w, q, qm, r, unique, group, x


def _pattern_bounds(y, w, q, qm, r, unique, group):
    mass = np.bincount(group, weights=w, minlength=len(unique))
    rows = {}
    lower_rows = {}
    for name, target in (("H", y), ("K", r), ("M", (y+KD_WEIGHT*qm)/(1+KD_WEIGHT))):
        optimal = np.bincount(group, weights=w*target, minlength=len(unique)) / mass
        loss = _bce_prob(target, optimal[group])
        lower_rows[name] = loss
        entropy = float(np.sum(w*_entropy(target)))
        minimum = float(np.sum(w*loss))
        rows[name] = {
            "minimum_bce": minimum, "row_target_entropy": entropy,
            "same_input_conflict": minimum-entropy,
        }
    teacher_hard = float(np.sum(w*_bce_prob(y, q)))
    teacher_mixed = float(np.sum(w*_bce_prob(r, q)))
    reference = {
        "hard_bce": teacher_hard if np.isfinite(teacher_hard) else None,
        "mixed_bce": teacher_mixed if np.isfinite(teacher_mixed) else None,
        "teacher_kl": 0.,
        "finite": bool(np.isfinite(teacher_hard) and np.isfinite(teacher_mixed)),
        "scope": "Saved teacher q_F, not an assumed member of the student function class",
    }
    return rows, lower_rows, reference


def _geometry(gradients, batch_rows, layout):
    h, q, m, k = (gradients[t] for t in TERMS)
    expected = (h+KD_WEIGHT*q)/(1+KD_WEIGHT)
    residual = float(np.linalg.norm(k-expected))
    tolerance = 1e-7 + 1e-5*max(float(np.linalg.norm(k)), float(np.linalg.norm(expected)))
    correction, instance = k-h, q-m
    h2 = float(np.dot(h, h))
    result = {
        "scope": "Anchor only; no endpoint gradient or convergence claim",
        "parameter_layout": layout,
        "gradient_norms": {t: float(np.linalg.norm(gradients[t])) for t in TERMS},
        "hard_teacher_cosine": _cosine(h, q),
        "hard_teacher_dot": float(np.dot(h, q)),
        "correction_norm": float(np.linalg.norm(correction)),
        "correction_hard_cosine": _cosine(correction, h),
        "correction_projection_on_hard": float(np.dot(correction, h)/h2) if h2 else None,
        "instance_gradient_norm": float(np.linalg.norm(instance)),
        "instance_hard_cosine": _cosine(instance, h),
        "teacher_dot_mixed": float(np.dot(q, k)),
        "batch_cancellation": {},
    }
    for term in TERMS:
        sum_norms = sum(b[term+"_norm"] for b in batch_rows)
        result["batch_cancellation"][term] = {
            "sum_batch_norms": sum_norms,
            "norm_of_sum": float(np.linalg.norm(gradients[term])),
            "norm_of_sum_divided_by_sum_norms":
                float(np.linalg.norm(gradients[term])/sum_norms) if sum_norms else None,
        }
    identity = {"residual_l2": residual, "tolerance": tolerance,
                "valid": bool(residual <= tolerance)}
    return result, identity


def _group_residuals(name, identity, weights, rows, group, lower_k):
    base = identity.copy()
    base["group"] = group
    base["weight_mass"] = weights
    for metric, values in rows.items():
        base[metric+"_weighted_sum"] = weights*values
    base["mixed_excess_weighted_sum"] = weights*(rows["mixed_bce"]-lower_k)
    value_cols = ["weight_mass"] + [c for c in base if c.endswith("_weighted_sum")]

    def aggregate(key):
        result = base.groupby(key, sort=True, observed=True)[value_cols].sum()
        result["rows"] = base.groupby(key, sort=True, observed=True).size()
        for metric in rows:
            result[metric+"_weighted_mean"] = result[metric+"_weighted_sum"]/result.weight_mass
        result["model"] = name
        return result.reset_index()

    return base, aggregate("label"), aggregate("attack_cat"), aggregate("feature_hash")


def diagnose_f(bundles, fit_frame, targets, output: Path, device="cuda"):
    """Measure frozen anchor/H40/K40 once over F, reusing each anchor graph.

    The caller binds files and F identities before invoking this in the lab.
    CPU execution is for synthetic tests. No optimizer is created. Each anchor
    batch uses four backwards on one forward; the endpoints use inference only.
    """
    import torch
    from torch.nn import functional as F

    y, w, q, qm, r, unique, group, x = _checked_inputs(bundles, fit_frame, targets)
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"} or (selected.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("CUDA laboratory device or CPU synthetic device required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bounds, lower_rows, teacher_reference = _pattern_bounds(y, w, q, qm, r, unique, group)
    old = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(), torch.backends.cudnn.benchmark,
           torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
           torch.backends.cudnn.allow_tf32)
    cuda_devices = [selected.index if selected.index is not None else torch.cuda.current_device()] \
        if selected.type == "cuda" else []
    before = {name: _state_hash(bundles[name].network_state) for name in MODELS}
    logits, states, batch_rows, gradients, layout = {}, {}, [], {}, []
    forward_rows = forward_calls = backward_calls = 0
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices):
            for name in MODELS:
                bundle = bundles[name]
                net = _network(bundle.encoded_dim, tuple(bundle.hidden_dims)).to(
                    device=selected, dtype=torch.float32)
                net.load_state_dict(bundle.network_state)
                net.eval()
                predictions = []
                if name == "anchor":
                    offset = 0
                    for key, parameter in net.named_parameters():
                        layout.append({"name": key, "shape": list(parameter.shape),
                                       "start": offset, "stop": offset+parameter.numel()})
                        offset += parameter.numel()
                    gradients = {term: np.zeros(offset, dtype=np.float64) for term in TERMS}
                for start in range(0, len(y), BATCH_SIZE):
                    stop = min(start+BATCH_SIZE, len(y))
                    xb = torch.from_numpy(x[start:stop]).to(selected)
                    with torch.set_grad_enabled(name == "anchor"):
                        z = net(xb).squeeze(1).to(torch.float64)
                    forward_rows += stop-start
                    forward_calls += 1
                    if not bool(torch.isfinite(z).all()):
                        raise FloatingPointError("Non-finite frozen F logits")
                    predictions.append(z.detach().cpu().numpy().copy())
                    if name == "anchor":
                        wb = torch.from_numpy(w[start:stop]).to(selected)
                        yy = torch.from_numpy(y[start:stop].astype(np.float64)).to(selected)
                        qq = torch.from_numpy(q[start:stop]).to(selected)
                        mm = torch.from_numpy(qm[start:stop]).to(selected)
                        rr = torch.from_numpy(r[start:stop]).to(selected)
                        eq = -(torch.special.xlogy(qq, qq)+torch.special.xlogy(1-qq, 1-qq))
                        em = -(torch.special.xlogy(mm, mm)+torch.special.xlogy(1-mm, 1-mm))
                        losses = (
                            (wb*F.binary_cross_entropy_with_logits(z, yy, reduction="none")).sum(),
                            (wb*(F.binary_cross_entropy_with_logits(z, qq, reduction="none")-eq)).sum(),
                            (wb*(F.binary_cross_entropy_with_logits(z, mm, reduction="none")-em)).sum(),
                            (wb*F.binary_cross_entropy_with_logits(z, rr, reduction="none")).sum(),
                        )
                        batch = {"batch_index": len(batch_rows), "start": start,
                                 "stop": stop, "rows": stop-start, "weight_mass": float(wb.sum())}
                        vectors = {}
                        for index, (term, loss) in enumerate(zip(TERMS, losses)):
                            if not bool(torch.isfinite(loss)):
                                raise FloatingPointError("Non-finite anchor diagnostic loss")
                            net.zero_grad(set_to_none=True)
                            loss.backward(retain_graph=index < len(TERMS)-1)
                            backward_calls += 1
                            vector = torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()])
                            vector = vector.cpu().to(torch.float64).numpy()
                            if not np.isfinite(vector).all():
                                raise FloatingPointError("Non-finite anchor diagnostic gradient")
                            vectors[term] = vector
                            gradients[term] += vector
                            batch[term+"_norm"] = float(np.linalg.norm(vector))
                            batch[term+"_loss"] = float(loss.detach())
                        batch["hard_teacher_cosine"] = _cosine(vectors["hard_bce"], vectors["teacher_kl"])
                        batch_rows.append(batch)
                logits[name] = np.concatenate(predictions).astype(np.float64, copy=False)
                states[name] = {"state_before": before[name], "state_after": _state_hash(_state(net)),
                                "bundle_state_after": _state_hash(bundle.network_state)}
                states[name]["unchanged"] = bool(states[name]["state_after"] == before[name]
                                                  == states[name]["bundle_state_after"])
                if not states[name]["unchanged"]:
                    raise ValueError("Frozen model state changed during diagnostic")
    finally:
        torch.set_num_threads(old[0])
        torch.use_deterministic_algorithms(old[1], warn_only=old[2])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[3:5]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[5:7]

    geometry, gradient_identity = _geometry(gradients, batch_rows, layout)
    identity = fit_frame.loc[:, IDENTITY].copy()
    scores = identity.copy()
    summaries = {}
    classes, families, patterns, quantile_frames = [], [], [], []
    quantile_edges = {}
    drivers = {"teacher_q": q, "target_displacement": np.abs(r-y), "signed_target_displacement": r-y}
    for driver, values in drivers.items():
        quantile_edges[driver] = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
    for name, z in logits.items():
        scores[name+"_logit"] = z
        softplus = np.logaddexp(0, z)
        rows = {"hard_bce": softplus-y*z, "teacher_kl": softplus-q*z-_entropy(q),
                "mixed_bce": softplus-r*z, "absolute_q_residual": np.abs(expit(z)-q),
                "absolute_mixed_target_residual": np.abs(expit(z)-r)}
        losses = {metric: float(np.sum(w*value)) for metric, value in rows.items()}
        bounds_valid = bool(losses["hard_bce"] >= bounds["H"]["minimum_bce"]-1e-12
                            and losses["mixed_bce"] >= bounds["K"]["minimum_bce"]-1e-12)
        finite = bool(np.isfinite(z).all() and all(np.isfinite(v) for v in losses.values()))
        summaries[name] = {
            "losses": losses, "finite": finite, "bounds_valid": bounds_valid,
            "state_checks": states[name],
            "mixed_excess_above_pattern_bound": losses["mixed_bce"]-bounds["K"]["minimum_bce"],
            "mixed_excess_above_teacher_function": losses["mixed_bce"]-teacher_reference["mixed_bce"]
                if teacher_reference["mixed_bce"] is not None else None,
        }
        base, c, f, p = _group_residuals(name, identity, w, rows, group, lower_rows["K"])
        classes.append(c)
        families.append(f)
        patterns.append(p)
        value_cols = ["weight_mass"]+[col for col in base if col.endswith("_weighted_sum")]
        for driver, values in drivers.items():
            edges = np.asarray(quantile_edges[driver])
            base["quantile_bin"] = np.searchsorted(edges[1:-1], values, side="right")
            agg = base.groupby("quantile_bin", sort=True)[value_cols].sum()
            agg["rows"] = base.groupby("quantile_bin", sort=True).size()
            agg["driver"] = driver
            agg["model"] = name
            quantile_frames.append(agg.reset_index())

    expected_rows = len(y)*len(MODELS)
    expected_backward = math.ceil(len(y)/BATCH_SIZE)*len(TERMS)
    budget = {"forward_rows": forward_rows, "forward_calls": forward_calls,
              "backward_calls": backward_calls, "optimizer_updates": 0, "new_fits": 0,
              "new_teacher_queries": 0, "V_calibration_test_rows": 0,
              "batch_size": BATCH_SIZE, "expected_forward_rows": expected_rows,
              "expected_backward_calls": expected_backward,
              "valid": forward_rows == expected_rows and backward_calls == expected_backward}
    summary = {
        "scope": "Frozen F-only observations; parameter gradients belong to anchor only",
        "fit_rows": len(y), "feature_patterns": len(unique), "models": summaries,
        "pattern_bounds": bounds, "teacher_function_reference": teacher_reference,
        "gradient_geometry": geometry, "gradient_identity": gradient_identity,
        "quantile_edges": quantile_edges,
        "quantile_definitions": {"teacher_q": "q", "target_displacement": "abs(r-y)",
                                 "signed_target_displacement": "r-y",
                                 "binning": "unweighted deciles; duplicate boundaries collapsed"},
        "budget": budget,
        "valid": bool(budget["valid"] and gradient_identity["valid"]
                      and all(s["finite"] and s["bounds_valid"] and s["state_checks"]["unchanged"]
                              for s in summaries.values())),
    }
    scores.to_parquet(output/"fit-scores.parquet", index=False)
    np.savez_compressed(output/"anchor-gradients.npz", **gradients)
    pd.DataFrame(batch_rows).to_csv(output/"gradient-batches.csv", index=False)
    pd.concat(classes, ignore_index=True).to_csv(output/"residuals-by-class.csv", index=False)
    pd.concat(families, ignore_index=True).to_csv(output/"residuals-by-family.csv", index=False)
    pd.concat(patterns, ignore_index=True).to_parquet(output/"residuals-by-pattern.parquet", index=False)
    pd.concat(quantile_frames, ignore_index=True).to_csv(output/"residuals-by-quantile.csv", index=False)
    write_json(output/"summary.json", summary)
    return summary
