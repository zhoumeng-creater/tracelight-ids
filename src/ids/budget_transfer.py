"""Budgeted low-FPR transfer study, using only the frozen development cohorts.

F fits every model; R registers normal thresholds; C selects every model under
the SAME finite R+C calibration rule. V is loaded only after every selection is
sealed. No old-model retention/phase-I/guard. All results remain development.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import canonical_hash, code_fingerprint, environment_snapshot, read_json, sha256_file, utc_now, write_json
from .data import load_partition
from .direct_pauc import pairwise_pauc_rows, sample_global_tail_pairs
from .models import FeaturePreprocessor, ModelBundle, _network, load_bundle, save_bundle, train_model
from .research import _group_column
from .risk_research import _fixed_metrics, _infer, _state, _state_hash, verify_risk_research
from .runner import verify_development_inputs
from .selection import normalize_feature_schema, select_features
from .transfer_diagnostics import binary_rank_metrics, training_threshold, transfer_diagnostics
from .upgrade_kd import raw_bernoulli_kl_rows

COMPLETE = "BUDGET_TRANSFER_COMPLETE.json"
METHODS = ("hard", "direct_pauc", "raw_kd", "direct_raw_kd", "direct_channel_rank",
           "direct_rank_uniform", "direct_rank_tail", "direct_rank_shuffled", "direct_rank_matched")


def validate_config(config):
    keys = {"schema_version", "purpose", "seeds", "threads", "max_fpr", "reduced_k", "small_hidden", "wide_hidden",
            "warmup_epochs", "warmup_batch_size", "macro_steps", "chunk_size", "learning_rate", "weight_decay",
            "gradient_clip", "pauc_pairs", "rank_pairs_per_class", "pauc_weight", "kd_weight", "tail_multiplier",
            "teacher_hyperparameters", "qualification_min_pauc_gain", "qualification_min_tpr_gain", "measure_cpu"}
    if not isinstance(config, dict) or set(config) != keys or config["schema_version"] != 1:
        raise ValueError("budget-transfer configuration schema mismatch")
    if config["purpose"] != "budget_transfer_development_only":
        raise ValueError("budget transfer is development only")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("distinct nonnegative seeds required")
    for k in ("threads", "reduced_k", "warmup_epochs", "warmup_batch_size", "macro_steps", "chunk_size", "pauc_pairs", "rank_pairs_per_class"):
        if type(config[k]) is not int or config[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    for k in ("small_hidden", "wide_hidden"):
        if not isinstance(config[k], list) or not config[k] or any(type(n) is not int or n < 1 for n in config[k]):
            raise ValueError("hidden widths must be positive integers")
    for k in ("max_fpr", "learning_rate", "weight_decay", "gradient_clip", "pauc_weight", "kd_weight", "tail_multiplier",
              "qualification_min_pauc_gain", "qualification_min_tpr_gain"):
        v = config[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"invalid {k}")
    if (not 0 < config["max_fpr"] < .5 or min(config[k] for k in
            ("learning_rate", "gradient_clip", "pauc_weight", "kd_weight", "tail_multiplier")) <= 0
            or config["max_fpr"] * config["tail_multiplier"] >= 1):
        raise ValueError("invalid objective scales/interest window")
    hp = config["teacher_hyperparameters"]
    if (not isinstance(hp, dict) or set(hp) != {"max_iter", "max_leaf_nodes", "learning_rate", "early_stopping"}
            or hp["early_stopping"] is not False):
        raise ValueError("explicit fixed HGB hyperparameters without internal stopping required")
    if type(config["measure_cpu"]) is not bool:
        raise ValueError("measure_cpu must be boolean")


def _no_links(path):
    from .upgrade_research import _no_links as check
    return check(path)


def _training_frames(v3, data):
    """Do not load V while any teacher/student selection remains outstanding."""
    train = load_partition(data, "train")
    if train.sample_id.duplicated().any():
        raise ValueError("training IDs must be unique")
    indexed, frames = train.set_index("sample_id", drop=False), {}
    for role in ("fit", "reference", "constraint"):
        identity = pd.read_parquet(v3 / f"{role}-identity.parquet")
        frame = indexed.loc[identity.sample_id.tolist()].reset_index(drop=True)
        if not frame.loc[:, identity.columns].equals(identity.reset_index(drop=True)):
            raise ValueError("frozen role identity changed")
        if set(frame.label) != {0, 1}:
            raise ValueError("every training role needs both classes; no redraw")
        frames[role] = frame
    ids = [str(x) for f in frames.values() for x in f.sample_id]
    if len(ids) != len(set(ids)) or set(ids) != set(train.sample_id.astype(str)):
        raise ValueError("frozen roles must partition all training rows")
    for key in {_group_column(train), "feature_hash"}:
        sets = [set(f[key]) for f in frames.values()]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("frozen training roles share a feature/session group")
    return frames


def _cpu_network(bundle, state=None):
    import torch
    with torch.random.fork_rng(devices=[]):
        net = _network(bundle.encoded_dim, bundle.hidden_dims)
    net.load_state_dict(bundle.network_state if state is None else state)
    return net


def _scores(bundle, frames, chunk):
    if bundle.model_name == "hgb":
        return {r: np.asarray(bundle.estimator.decision_function(bundle.preprocessor.transform(f)), dtype=np.float64)
                for r, f in frames.items()}
    net = _cpu_network(bundle)
    return {r: _infer(net, bundle.preprocessor.transform(f), "cpu", chunk) for r, f in frames.items()}


def _working_point(scores, frames, alpha):
    r = training_threshold(scores["reference"], frames["reference"].label.to_numpy(), alpha)
    c = training_threshold(scores["constraint"], frames["constraint"].label.to_numpy(), alpha)
    tau = max(r["threshold"], c["threshold"])
    return {"threshold": tau, "reference_only": r, "constraint_normal": c,
            "source": "maximum of conservative training R-normal and C-normal thresholds; C is used for training selection",
            "population_guarantee": False}


def _selection(bundle, frames, config):
    scores = _scores(bundle, {r: frames[r] for r in ("reference", "constraint")}, config["chunk_size"])
    wp = _working_point(scores, frames, config["max_fpr"])
    c = _fixed_metrics(frames["constraint"], scores["constraint"], wp["threshold"])
    c["decision_rule"] = "float64(raw_score)>=frozen_training_R+C_threshold"
    rank = binary_rank_metrics(frames["constraint"].label.to_numpy(), scores["constraint"], config["max_fpr"])
    return {"working_point": wp, "C": c, "C_rank": rank}, scores


def _key(record):
    return (record["C"]["tpr"], record["C_rank"]["partial_auc_standardized"], -record["C"]["fpr"])


def _bundle(preprocessor, schema, hidden, seed, config, state):
    return ModelBundle("mlp", preprocessor, preprocessor.selected_features, preprocessor.encoded_dim, [],
        network_state=state, hidden_dims=tuple(hidden), seed=seed, feature_schema=schema,
        batch_size=config["chunk_size"], num_threads=config["threads"])


def _warmup(preprocessor, schema, hidden, seed, frames, config, device, output):
    """Shared hard-only fit; no teacher/no V, same C selection for every budget."""
    import torch
    import torch.nn.functional as F
    started = time.perf_counter()
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    net = _network(preprocessor.encoded_dim, tuple(hidden)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    x, y = preprocessor.transform(frames["fit"]), frames["fit"].label.to_numpy(dtype=np.float32)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    row_weights = np.where(y == 0, len(y) / (2 * n0), len(y) / (2 * n1)).astype(np.float32)
    rng, history, best, best_record = np.random.default_rng(seed), [], None, None
    for epoch in range(config["warmup_epochs"] + 1):
        if epoch:
            net.train()
            order = rng.permutation(len(y))
            for start in range(0, len(y), config["warmup_batch_size"]):
                ix = order[start:start + config["warmup_batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                z = net(torch.from_numpy(x[ix]).to(device)).squeeze(1)
                loss = (F.binary_cross_entropy_with_logits(z, torch.from_numpy(y[ix]).to(device), reduction="none")
                        * torch.from_numpy(row_weights[ix]).to(device)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
        candidate = _bundle(preprocessor, schema, hidden, seed, config, _state(net))
        record, _ = _selection(candidate, frames, config)
        history.append({"epoch": epoch, **record})
        if epoch in {0, 1, config["warmup_epochs"]} or epoch % 10 == 0:
            print(f"[transfer] warmup {output.name} epoch={epoch} C_TPR={record['C']['tpr']:.6f}", flush=True)
        if best is None or _key(record) > _key(best_record):
            best, best_record, best_epoch = candidate, record, epoch
    best.best_epoch = best_epoch
    output.mkdir(parents=True, exist_ok=False)
    save_bundle(best, output / "model.joblib")
    write_json(output / "history.json", history)
    write_json(output / "run.json", {"seed": seed, "kind": "hard_warmup", "selection": best_record,
        "selected_epoch": best_epoch, "training_seconds": time.perf_counter() - started,
        "selection_evaluations": config["warmup_epochs"] + 1,
        "selection_stage": "C selection over initialization and every warmup epoch",
        "compute": {"optimizer_updates": config["warmup_epochs"] * ((len(y) + config["warmup_batch_size"] - 1) // config["warmup_batch_size"]),
            "selected_checkpoint_optimizer_updates": best_epoch * ((len(y) + config["warmup_batch_size"] - 1) // config["warmup_batch_size"]),
            "gradient_forward_rows": config["warmup_epochs"] * len(y),
            "CPU_R_C_selection_forward_rows": (config["warmup_epochs"] + 1) * (len(frames["reference"]) + len(frames["constraint"])),
            "training_unit": "minibatch epochs; continuation uses full-F macrosteps"},
        "deployment": {"raw_fields": len(best.selected_features), "encoded_dim": best.encoded_dim,
            "hidden_sizes": list(best.hidden_dims), "parameters": sum(p.numel() for p in net.parameters()),
            "model_package_bytes": (output / "model.joblib").stat().st_size},
        "fit_rows": len(y), "C_is_selection": True, "validation_used": False})
    return best


def _backward(net, xs, y, teacher, config, method, seed, step):
    import torch
    import torch.nn.functional as F
    from .tail_rank_kd import binary_channel_rank_rows, sample_same_class_pairs, tail_rank_kd_rows
    device, chunk = next(net.parameters()).device, config["chunk_size"]
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    totals, gradients = {"hard": 0., "pauc": 0., "kd": 0.,
        "gradient_forward_rows": len(y), "detached_global_tail_score_rows": 0}, {}
    def grad_snapshot():
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten().cpu().double()
                          for p in net.parameters()])
    for start in range(0, len(y), chunk):
        ix = np.arange(start, min(start + chunk, len(y)))
        z = net(torch.from_numpy(xs[ix]).to(device)).squeeze(1)
        target = torch.from_numpy(y[ix].astype(np.float32)).to(device)
        weights = torch.tensor(np.where(y[ix] == 0, .5 / n0, .5 / n1), device=device, dtype=z.dtype)
        hard = (F.binary_cross_entropy_with_logits(z, target, reduction="none") * weights).sum()
        hard.backward()
        totals["hard"] += float(hard.detach())
    if step == 1:
        gradients["hard"] = grad_snapshot()
    if method.startswith("direct"):
        # Whole F inference at exactly the parameters used by ALL chunk gradients.
        scores = _infer(net, xs, str(device), chunk)
        net.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 617]))
        pairs = sample_global_tail_pairs(scores, y, rng=rng, n_pairs=config["pauc_pairs"], beta=config["max_fpr"])
        pi, ni = pairs["positive_indices"], pairs["negative_indices"]
        for start in range(0, len(pi), chunk):
            p, n = pi[start:start + chunk], ni[start:start + chunk]
            z = net(torch.from_numpy(xs[np.concatenate([p, n])]).to(device)).squeeze(1)
            loss = config["pauc_weight"] * pairwise_pauc_rows(z[:len(p)], z[len(p):]).sum() / len(pi)
            loss.backward()
            totals["pauc"] += float(loss.detach())
        totals["pauc_sampling"] = pairs["statistics"]
        totals["gradient_forward_rows"] += 2 * len(pi)
        totals["detached_global_tail_score_rows"] = len(y)
    if step == 1:
        gradients["after_pauc"] = grad_snapshot()
    if method in {"raw_kd", "direct_raw_kd", "direct_channel_rank"}:
        totals["gradient_forward_rows"] += len(y)
        for start in range(0, len(y), chunk):
            ix = np.arange(start, min(start + chunk, len(y)))
            z = net(torch.from_numpy(xs[ix]).to(device)).squeeze(1)
            t = torch.tensor(teacher["fit"][ix], device=device, dtype=torch.float64)
            rows = (binary_channel_rank_rows(z, teacher_logits=t) if method == "direct_channel_rank"
                    else raw_bernoulli_kl_rows(z, teacher_logits=t, temperature=1.))
            weights = torch.tensor(np.where(y[ix] == 0, .5 / n0, .5 / n1), device=device, dtype=rows.dtype)
            loss = config["kd_weight"] * (rows * weights).sum()
            loss.backward()
            totals["kd"] += float(loss.detach())
    elif method.startswith("direct_rank_"):
        pairs = sample_same_class_pairs(teacher["fit"], y,
            reference_normal_scores=teacher["reference_normal"], pairs_per_class=config["rank_pairs_per_class"],
            max_fpr=config["max_fpr"], tail_multiplier=config["tail_multiplier"],
            local=method != "direct_rank_uniform", shuffle_within_class=False,
            seed=int(np.random.SeedSequence([seed, step, 907]).generate_state(1)[0]))
        for start in range(0, len(pairs.left), chunk):
            left, right = pairs.left[start:start + chunk], pairs.right[start:start + chunk]
            z = net(torch.from_numpy(xs[np.concatenate([left, right])]).to(device)).squeeze(1)
            rows = tail_rank_kd_rows(z[:len(left)], z[len(left):],
                torch.tensor(pairs.targets[start:start + chunk], device=device, dtype=z.dtype), temperature=1.)
            w = torch.tensor(pairs.weights[start:start + chunk], device=device, dtype=rows.dtype)
            loss = config["kd_weight"] * (rows * w).sum()
            loss.backward()
            totals["kd"] += float(loss.detach())
        totals["rank_sampling"] = pairs.stats
        totals["gradient_forward_rows"] += 2 * len(pairs.left)
    if step == 1:
        end = grad_snapshot()
        kd = end - gradients["after_pauc"]
        h, direct = gradients["hard"], gradients["after_pauc"] - gradients["hard"]
        def cosine(a, b):
            return float(a.dot(b) / (a.norm() * b.norm())) if a.norm() and b.norm() else None
        totals["gradients"] = {"hard_norm": float(h.norm()), "pauc_norm": float(direct.norm()), "kd_norm": float(kd.norm()),
            "hard_kd_cosine": cosine(h, kd), "pauc_kd_cosine": cosine(direct, kd),
            "scope": "global objectives before gradient clipping and AdamW, not scalar-loss ratios"}
    return totals


def _fit_method(anchor, frames, config, method, teacher, seed, device, output):
    import torch
    started = time.perf_counter()
    net = _cpu_network(anchor).to(device)
    initial = _state_hash(_state(net))
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    xs, y = anchor.preprocessor.transform(frames["fit"]), frames["fit"].label.to_numpy(dtype=np.int64)
    shuffle_hash = None
    if method == "direct_rank_shuffled":
        # Fix the null teacher once; moving supervision would add a confound.
        teacher = {**teacher, "fit": teacher["fit"].copy()}
        rng = np.random.default_rng(np.random.SeedSequence([seed, 353]))
        permutation = np.arange(len(y))
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            permutation[indices] = rng.permutation(indices)
        teacher["fit"] = teacher["fit"][permutation]
        shuffle_hash = hashlib.sha256(permutation.astype("<i8").tobytes()).hexdigest()
    best, best_record, best_step, history = None, None, 0, []
    for step in range(config["macro_steps"] + 1):
        losses = None
        if step:
            net.train()
            optimizer.zero_grad(set_to_none=True)
            losses = _backward(net, xs, y, teacher, config, method, seed, step)
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), config["gradient_clip"], error_if_nonfinite=True)
            losses["preclip_gradient_norm"] = float(grad_norm)
            optimizer.step()
        candidate = copy.copy(anchor)
        candidate.network_state, candidate._cached_network = _state(net), None
        record, _ = _selection(candidate, frames, config)
        history.append({"step": step, "losses": losses, **record})
        if best is None or _key(record) > _key(best_record):
            best, best_record, best_step = candidate, record, step
        if step in {0, 1, config["macro_steps"]} or step % 10 == 0:
            print(f"[transfer] seed={seed} {output.name} {method} step={step} C_TPR={record['C']['tpr']:.6f}", flush=True)
    best.best_epoch = best_step
    output.mkdir(parents=True, exist_ok=False)
    save_bundle(best, output / "model.joblib")
    write_json(output / "history.json", history)
    scores = _scores(best, {r: frames[r] for r in ("reference", "constraint")}, config["chunk_size"])
    np.savez_compressed(output / "training-scores.npz", reference=scores["reference"], constraint=scores["constraint"])
    run = {"seed": seed, "method": method, "status": "trained", "selected_step": best_step,
        "common_initial_state_sha256": initial, "selection": best_record,
        "training_seconds": time.perf_counter() - started, "fit_rows": len(y),
        "compute": {"optimizer_updates": config["macro_steps"],
            "gradient_forward_rows": sum(h["losses"]["gradient_forward_rows"] for h in history[1:]),
            "detached_global_tail_score_rows": sum(h["losses"]["detached_global_tail_score_rows"] for h in history[1:]),
            "CPU_R_C_selection_forward_rows": (config["macro_steps"] + 1) * (len(frames["reference"]) + len(frames["constraint"])),
            "equal_training_FLOPs_claim": False},
        "fixed_teacher_permutation_sha256": shuffle_hash,
        "deployment": {"raw_fields": len(best.selected_features), "encoded_dim": best.encoded_dim,
            "hidden_sizes": list(best.hidden_dims), "parameters": sum(p.numel() for p in net.parameters()),
            "model_package_bytes": (output / "model.joblib").stat().st_size}, "validation_used_for_selection": False}
    write_json(output / "run.json", run)
    return best, run


def qualification(teacher_selection, direct_selection, config):
    """Registered headroom diagnostic; never delete a seed from comparisons."""
    dg = teacher_selection["C_rank"]["partial_auc_standardized"] - direct_selection["C_rank"]["partial_auc_standardized"]
    tg = teacher_selection["C"]["tpr"] - direct_selection["C"]["tpr"]
    accepted = dg >= config["qualification_min_pauc_gain"] or tg >= config["qualification_min_tpr_gain"]
    return {"qualified": bool(accepted), "C_teacher_minus_direct_pauc": dg, "C_teacher_minus_direct_tpr": tg,
        "rule": "C pAUC gain >= registered margin OR C joint-calibrated TPR gain >= registered margin",
        "scope": "development triage, not proof of transferability or teacher uselessness",
        "failure_action": "retain negative diagnosis and all registered comparisons; no selective seed deletion or teacher search"}


def _benchmark(bundle, frame, threshold, repeats=25):
    """Lab CPU pipeline timing, independent of teacher; no packet extraction claim."""
    net = _cpu_network(bundle) if bundle.model_name == "mlp" else None
    def call(part):
        x = bundle.preprocessor.transform(part)
        z = (_infer(net, x, "cpu", max(1, len(x))) if net is not None
             else np.asarray(bundle.estimator.decision_function(x), dtype=np.float64))
        return z >= threshold
    result = {}
    for size in (1, min(256, len(frame))):
        part = frame.iloc[:size]
        for _ in range(5):
            call(part)
        timings = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            call(part)
            timings.append((time.perf_counter_ns() - start) / 1e6)
        result[str(size)] = {"p50_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
            "mean_rows_per_second": size * 1000 / float(np.mean(timings)), "repeats": repeats}
    return {"batches": result, "includes": "existing CSV fields -> preprocessing -> CPU model -> raw threshold",
            "excludes": "CSV disk read, network capture and raw packet/flow field extraction",
            "scope": "laboratory CPU measurement; not laptop deployment acceptance"}


def _paired_evidence(root, runs):
    """Existing late-V predictions only; do not fit, choose thresholds or rank."""
    by_key = {(r["seed"], r["budget"], r["method"]): r for r in runs}
    result = []
    for candidate in runs:
        if candidate["method"] != "direct_rank_tail":
            continue
        cp = pd.read_parquet(root / candidate["relative_dir"] / "validation-predictions.parquet")
        columns = ["sample_id", "feature_hash", "label", "attack_cat"]
        y, c = cp.label.to_numpy(), cp.alarm.to_numpy(dtype=bool)
        for method in METHODS:
            baseline = by_key.get((candidate["seed"], candidate["budget"], method))
            if baseline is None or method == "direct_rank_tail":
                continue
            bp = pd.read_parquet(root / baseline["relative_dir"] / "validation-predictions.parquet")
            if not cp[columns].equals(bp[columns]):
                raise ValueError("paired prediction identity differs")
            b = bp.alarm.to_numpy(dtype=bool)
            row = {"seed": candidate["seed"], "budget": candidate["budget"], "candidate": candidate["method"], "control": method,
                "candidate_fpr_pass": candidate["V_fpr_budget_pass"], "control_fpr_pass": baseline["V_fpr_budget_pass"],
                "attack_recovered_tp": int((c & ~b & (y == 1)).sum()), "attack_new_lost_tp": int((~c & b & (y == 1)).sum()),
                "normal_new_fp": int((c & ~b & (y == 0)).sum()), "normal_repaired_fp": int((~c & b & (y == 0)).sum()),
                "delta_tp": int(c[y == 1].sum() - b[y == 1].sum()), "delta_fp": int(c[y == 0].sum() - b[y == 0].sum()),
                "delta_tpr": candidate["validation"]["tpr"] - baseline["validation"]["tpr"],
                "delta_standardized_pauc": candidate["validation_rank"]["partial_auc_standardized"] - baseline["validation_rank"]["partial_auc_standardized"],
                "thresholds": "each model's own frozen training R+C threshold; no V matching/tuning", "families": {}}
            for family in sorted(cp.loc[y == 1, "attack_cat"].unique()):
                mask = (y == 1) & (cp.attack_cat.to_numpy() == family)
                row["families"][str(family)] = {"support": int(mask.sum()),
                    "recovered_tp": int((c & ~b & mask).sum()), "new_lost_tp": int((~c & b & mask).sum())}
            result.append(row)
    return result


def _write_report(root, summary, auxiliary):
    lines = ["# Budgeted transfer: complete development results", "",
        "No model is promoted by this report. V has been used in past development; this is not independent confirmation.",
        "All thresholds/checkpoints were fixed in FROZEN_SELECTION.json before the first V read.", "",
        "| Budget | Seed | Method | V TP / attack | V FP / normal | V TPR % | V FPR % | FPR pass | Step |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|"]
    for r in summary["runs"]:
        v = r["validation"]
        lines.append(f"| {r['budget']} | {r['seed']} | {r['method']} | {v['true_positives']}/{v['n_attack']} | "
            f"{v['false_positives']}/{v['n_normal']} | {100*v['tpr']:.4f} | {100*v['fpr']:.4f} | "
            f"{r['V_fpr_budget_pass']} | {r['selected_step']} |")
    lines.extend(["", "Teacher and capacity probes (compare the four warmups ONLY within their matched training stage):", "",
        "| Kind | Budget | Capacity | Seed | V TPR % | V FPR % |",
        "|---|---|---|---:|---:|---:|"])
    for r in auxiliary:
        lines.append(f"| {r['kind']} | {r['budget']} | {r.get('capacity', 'HGB')} | {r['seed']} | "
                     f"{100*r['validation']['tpr']:.4f} | {100*r['validation']['fpr']:.4f} |")
    lines.extend(["", "Read paired-comparisons.json before interpreting a net TP gain: recovered TP, newly lost TP and FP changes are separate.",
        "Native KL / direct pAUC / uniform / fixed-shuffle / matched-teacher controls falsify different explanations.",
        "A failed FPR budget cannot be hidden by high recall. C headroom flags never delete a seed.",
        "Same optimizer steps do not imply equal compute. See each run's compute and deployment fields and teacher query costs.", ""])
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def verify_transfer(output):
    root = _no_links(output)
    done = read_json(root / COMPLETE)
    if done.get("schema_version") != 1 or done.get("status") != "development_complete" or done.get("reserved_partitions_read") != []:
        raise ValueError("invalid transfer completion")
    files = {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file() and p.name != COMPLETE}
    if files != done.get("files"):
        raise ValueError("transfer artifact inventory mismatch")
    plan = read_json(root / "plan.json")
    if canonical_hash(plan["binding"]) != done.get("binding_sha256"):
        raise ValueError("transfer binding mismatch")
    summary = read_json(root / "summary.json")
    if not all(name in files for name in ("REPORT.md", "paired-comparisons.json", "FROZEN_SELECTION.json", "auxiliary-evaluations.json")):
        raise ValueError("transfer required result/paired evidence missing")
    config = plan["binding"]["config"]
    validate_config(config)
    expected = {(s, b, m) for s in config["seeds"] for b in ("all", "reduced") for m in METHODS
                if not (b == "all" and m == "direct_rank_matched")}
    observed = [(r["seed"], r["budget"], r["method"]) for r in summary["runs"]]
    aliases = {(s, "all", "direct_rank_matched") for s in config["seeds"]}
    skipped = [(r["seed"], r["budget"], r["method"]) for r in summary["skipped"]]
    if (len(set(observed)) != len(observed) or set(observed) != expected
            or len(skipped) != len(aliases) or set(skipped) != aliases):
        raise ValueError("transfer planned condition/seed inventory mismatch")
    n = len(config["seeds"])
    for key, amount in (("teacher", 2 * n), ("warmup", 4 * n), ("student", 17 * n)):
        if done.get(f"n_{key}_fits") != amount or summary.get(f"{key}_fits") != amount:
            raise ValueError("transfer fit count mismatch")
    frozen = read_json(root / "FROZEN_SELECTION.json")
    frozen_hash = sha256_file(root / "FROZEN_SELECTION.json")
    frozen_runs = {r["relative_dir"]: r for r in frozen["runs"]}
    if set(frozen_runs) != {r["relative_dir"] for r in summary["runs"]}:
        raise ValueError("frozen selection inventory changed")
    for name, digest in plan["binding"]["source"]["files"].items():
        if files.get("source/" + name) != digest:
            raise ValueError("transfer source snapshot differs from binding")
    for run in summary["runs"]:
        prefix = root / run["relative_dir"]
        if read_json(prefix / "run.json") != run:
            raise ValueError("transfer summary/run mismatch")
        if (run.get("frozen_selection_sha256") != frozen_hash or
                any(run.get(k) != v for k, v in frozen_runs[run["relative_dir"]].items())):
            raise ValueError("transfer selection changed after validation")
        required = ("model.joblib", "history.json", "training-scores.npz", "validation-predictions.parquet")
        if any(not (prefix / f).is_file() for f in required):
            raise ValueError("transfer required student evidence missing")
    auxiliary = read_json(root / "auxiliary-evaluations.json")
    aux_frozen = {r["relative_dir"]: r for r in frozen["auxiliary"]}
    if len(auxiliary) != 6 * n or len(aux_frozen) != 6 * n:
        raise ValueError("transfer teacher/capacity probe inventory missing")
    for item in auxiliary:
        if (item["relative_dir"] not in aux_frozen or
                any(item.get(k) != v for k, v in aux_frozen[item["relative_dir"]].items())):
            raise ValueError("teacher/capacity probe selection changed")
        if not all((root / item["relative_dir"] / name).is_file() for name in ("model.joblib", "run.json", "validation-predictions.parquet")):
            raise ValueError("teacher/capacity probe predictions missing")
    return done


def run_transfer_research(v3_dir, data_dir, config, output, *, device="cuda", resume=False):
    import torch
    config = copy.deepcopy(config)
    validate_config(config)
    root, v3, data = _no_links(output), _no_links(v3_dir), _no_links(data_dir)
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((root, v3), (root, data))):
        raise ValueError("new work and prepared/source directories must be separate")
    source_done = verify_risk_research(v3)
    manifest = verify_development_inputs(data)
    old_plan = read_json(v3 / "plan.json")
    if manifest != old_plan["data"] or old_plan["config"]["cohort_policy"] != "complete_development":
        raise ValueError("complete frozen V3 cohorts with identical prepared data required")
    binding = {"config": config, "source": code_fingerprint(), "v3_dir": str(v3), "data_dir": str(data),
        "v3_complete_sha256": sha256_file(v3 / "RISK_RESEARCH_COMPLETE.json"), "v3_plan_id": source_done["plan_id"],
        "data": manifest, "device": device}
    if root.exists():
        if not resume:
            raise FileExistsError("existing work is preserved; use a new output or complete-only resume")
        done = verify_transfer(root)
        if read_json(root / "plan.json")["binding"] != binding:
            raise ValueError("resume source/config/data/device differs")
        return done
    if manifest.get("data_kind") != "synthetic":
        if (device not in {"cuda", "cuda:0"} or not torch.cuda.is_available()
                or "4090D" not in torch.cuda.get_device_name(0).replace(" ", "").upper()):
            raise ValueError("real fits require laboratory CUDA0 RTX4090D")
    frames = _training_frames(v3, data)
    schema = normalize_feature_schema(manifest.get("feature_schema"))
    if config["reduced_k"] >= len(schema["features"]):
        raise ValueError("reduced input must remove at least one original field")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "plan.json", {"binding": binding, "reserved_partitions_read": [],
        "scope": "F fitting, R+C shared training calibration/selection, V once after ALL selections; development only"})
    write_json(root / "environment.json", environment_snapshot())
    for name, digest in binding["source"]["files"].items():
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[2] / name, target)
        if sha256_file(target) != digest:
            raise ValueError("source changed during snapshot")
    for role in (*frames, "validation"):
        shutil.copyfile(v3 / f"{role}-identity.parquet", root / f"{role}-identity.parquet")
    print(f"[transfer] MI selection on all {len(frames['fit'])} F rows only; CPU phase", flush=True)
    selection = select_features(frames["fit"], {"method": "mutual_info", "k": config["reduced_k"], "selection_seed": 20261104},
                                20261104, feature_schema=schema)
    print(f"[transfer] MI selection complete in {selection['metadata']['seconds']:.1f}s", flush=True)
    write_json(root / "feature-selection.json", selection)
    features = {"all": schema["features"], "reduced": selection["selected_features"]}
    preprocessors = {k: FeaturePreprocessor.fit(frames["fit"], fs, schema) for k, fs in features.items()}
    records, skipped, diagnostics, auxiliary, teacher_fits, warmup_fits = [], [], [], [], 0, 0
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=config["threads"]):
            for seed in config["seeds"]:
                teacher_scores, teacher_selections = {}, {}
                for budget in features:
                    out = root / "teachers" / f"{budget}-s{seed}"
                    out.mkdir(parents=True, exist_ok=False)
                    tc = {"model": "hgb", "features": {"method": "all"},
                          "hyperparameters": {**config["teacher_hyperparameters"], "num_threads": config["threads"]}}
                    # A restricted schema uses exactly the student's original fields.
                    tschema = {"name": schema["name"] + "-" + budget, "features": features[budget],
                               "categorical": [f for f in schema["categorical"] if f in features[budget]],
                               "discrete": [f for f in schema["discrete"] if f in features[budget]]}
                    started = time.perf_counter()
                    print(f"[transfer] teacher {budget} seed={seed} fitting F only", flush=True)
                    teacher = train_model(frames["fit"], frames["fit"], tc, seed, feature_schema=tschema)
                    save_bundle(teacher, out / "model.joblib")
                    values = _scores(teacher, frames, config["chunk_size"])
                    teacher_scores[budget] = {**values, "reference_normal": values["reference"][frames["reference"].label.to_numpy() == 0]}
                    trecord, _ = _selection(teacher, frames, config)
                    teacher_selections[budget] = trecord
                    write_json(out / "run.json", {"seed": seed, "information_budget": budget, "selection": trecord,
                        "fit_rows": len(frames["fit"]), "target_source": "in-sample F, native float64 HGB decision_function, NOT OOF",
                        "fit_and_query_seconds": time.perf_counter() - started, "model_package_bytes": (out / "model.joblib").stat().st_size,
                        "native_query_rows": len(frames["fit"]) + 2 * (len(frames["reference"]) + len(frames["constraint"])),
                        "fit_families": frames["fit"].attack_cat.value_counts().to_dict(), "validation_used": False})
                    np.savez_compressed(out / "training-scores.npz", **values)
                    auxiliary.append({"relative_dir": out.relative_to(root).as_posix(), "kind": "teacher",
                        "seed": seed, "budget": budget, "selection": trecord,
                        "deployment": {"raw_fields": len(teacher.selected_features), "encoded_dim": teacher.encoded_dim,
                            "model_package_bytes": (out / "model.joblib").stat().st_size}})
                    teacher_fits += 1
                for budget in features:
                    warm = {}
                    for capacity, hidden in (("small", config["small_hidden"]), ("wide", config["wide_hidden"])):
                        warm[capacity] = _warmup(preprocessors[budget], schema, hidden, seed, frames, config, device,
                                                root / "warmups" / f"{budget}-{capacity}-s{seed}")
                        warmup_fits += 1
                        auxiliary.append({"relative_dir": f"warmups/{budget}-{capacity}-s{seed}", "kind": "hard_warmup",
                            "seed": seed, "budget": budget, "capacity": capacity,
                            "deployment": read_json(root / "warmups" / f"{budget}-{capacity}-s{seed}" / "run.json")["deployment"],
                            "selection": read_json(root / "warmups" / f"{budget}-{capacity}-s{seed}" / "run.json")["selection"]})
                    base = warm["small"]
                    for method in ("hard", "direct_pauc"):
                        relative = f"students/{budget}-{method}-s{seed}"
                        _, run = _fit_method(base, frames, config, method, teacher_scores["all"], seed, device, root / relative)
                        run.update(budget=budget, relative_dir=relative)
                        write_json(root / relative / "run.json", run)
                        records.append(run)
                    direct_run = records[-1]
                    direct = load_bundle(root / direct_run["relative_dir"] / "model.joblib")
                    ds = _scores(direct, {r: frames[r] for r in ("reference", "constraint")}, config["chunk_size"])
                    x = direct.preprocessor.transform(frames["constraint"])
                    # Group exact encoded bytes, not labels, family IDs or sessions.
                    row_groups = [hashlib.sha256(row.tobytes()).hexdigest() for row in x]
                    report = transfer_diagnostics(calibration_labels=frames["reference"].label.to_numpy(),
                        calibration_teacher_scores=teacher_scores["all"]["reference"], calibration_student_scores=ds["reference"],
                        probe_labels=frames["constraint"].label.to_numpy(), probe_teacher_scores=teacher_scores["all"]["constraint"],
                        probe_student_scores=ds["constraint"], calibration_ids=frames["reference"].sample_id.astype(str).tolist(),
                        probe_ids=frames["constraint"].sample_id.astype(str).tolist(), probe_families=frames["constraint"].attack_cat.astype(str).tolist(),
                        student_input_groups=row_groups, teacher_fit_family_counts=frames["fit"].attack_cat.value_counts().to_dict(),
                        max_fpr=config["max_fpr"])
                    gate = qualification(teacher_selections["all"], direct_run["selection"], config)
                    report.update(seed=seed, budget=budget, qualification=gate,
                        capacity_control={cap: read_json(root / "warmups" / f"{budget}-{cap}-s{seed}" / "run.json")["selection"]
                                          for cap in warm},
                        matched_teacher_selection=teacher_selections[budget])
                    write_json(root / f"qualification-{budget}-s{seed}.json", report)
                    diagnostics.append({"seed": seed, "budget": budget, **gate})
                    for method in METHODS[2:]:
                        if method == "direct_rank_matched" and budget == "all":
                            skipped.append({"seed": seed, "budget": budget, "method": method,
                                "reason": "identical full-input teacher; exact alias of direct_rank_tail, no duplicate fit"})
                            continue
                        # Keep every paired seed even when its teacher is weak.
                        relative = f"students/{budget}-{method}-s{seed}"
                        chosen = teacher_scores[budget] if method == "direct_rank_matched" else teacher_scores["all"]
                        _, run = _fit_method(base, frames, config, method, chosen, seed, device, root / relative)
                        run.update(budget=budget, relative_dir=relative)
                        write_json(root / relative / "run.json", run)
                        records.append(run)
            # Written BEFORE the first validation feature/label is loaded.
            frozen = {"runs": records, "skipped": skipped, "qualification": diagnostics, "auxiliary": auxiliary,
                      "teacher_fits": teacher_fits, "warmup_fits": warmup_fits, "validation_used": False}
            write_json(root / "FROZEN_SELECTION.json", frozen)
            frozen_hash = sha256_file(root / "FROZEN_SELECTION.json")
            validation = load_partition(data, "validation")
            identity = pd.read_parquet(root / "validation-identity.parquet")
            validation = validation.set_index("sample_id", drop=False).loc[identity.sample_id.tolist()].reset_index(drop=True)
            if not validation.loc[:, identity.columns].equals(identity.reset_index(drop=True)):
                raise ValueError("validation identity changed")
            if set(validation.sample_id.astype(str)) & set().union(*(set(f.sample_id.astype(str)) for f in frames.values())):
                raise ValueError("training/validation overlap")
            for run in records:
                folder = root / run["relative_dir"]
                bundle = load_bundle(folder / "model.joblib")
                values = _scores(bundle, {"validation": validation}, config["chunk_size"])["validation"]
                tau = run["selection"]["working_point"]["threshold"]
                metrics = _fixed_metrics(validation, values, tau)
                metrics["decision_rule"] = "float64(raw_score)>=frozen_training_R+C_threshold"
                run.update(validation=metrics, validation_rank=binary_rank_metrics(validation.label.to_numpy(), values, config["max_fpr"]),
                    V_fpr_budget_pass=bool(metrics["false_positives"] <= np.floor(config["max_fpr"] * metrics["n_normal"])),
                    frozen_selection_sha256=frozen_hash)
                if config["measure_cpu"]:
                    run["cpu_pipeline"] = _benchmark(bundle, validation, tau)
                pred = validation.loc[:, ["sample_id", "feature_hash", "label", "attack_cat"]].copy()
                pred["raw_score"], pred["alarm"] = values, values >= tau
                pred.to_parquet(folder / "validation-predictions.parquet", index=False)
                write_json(folder / "run.json", run)
            for item in auxiliary:
                folder = root / item["relative_dir"]
                bundle = load_bundle(folder / "model.joblib")
                z = _scores(bundle, {"validation": validation}, config["chunk_size"])["validation"]
                tau = item["selection"]["working_point"]["threshold"]
                metrics = _fixed_metrics(validation, z, tau)
                metrics["decision_rule"] = "float64(raw_score)>=frozen_training_R+C_threshold"
                item.update(validation=metrics, validation_rank=binary_rank_metrics(validation.label.to_numpy(), z, config["max_fpr"]))
                if config["measure_cpu"]:
                    item["cpu_pipeline"] = _benchmark(bundle, validation, tau)
                pred = validation.loc[:, ["sample_id", "feature_hash", "label", "attack_cat"]].copy()
                pred["raw_score"], pred["alarm"] = z, z >= tau
                pred.to_parquet(folder / "validation-predictions.parquet", index=False)
            write_json(root / "auxiliary-evaluations.json", auxiliary)
            if sha256_file(root / "FROZEN_SELECTION.json") != frozen_hash:
                raise ValueError("frozen selections changed during validation")
            summary = {"runs": records, "skipped": skipped, "qualification": diagnostics,
                "teacher_fits": teacher_fits, "warmup_fits": warmup_fits, "student_fits": len(records),
                "status": "development_only_no_winner_promoted", "independent_confirmation": False,
                "note": "All planned comparisons retained, including negative teacher headroom; only an exact alias is skipped."}
            write_json(root / "summary.json", summary)
            write_json(root / "paired-comparisons.json", _paired_evidence(root, records))
            _write_report(root, summary, auxiliary)
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
    done = {"schema_version": 1, "status": "development_complete", "created_at": utc_now(),
        "binding_sha256": canonical_hash(binding), "reserved_partitions_read": [],
        "n_teacher_fits": teacher_fits, "n_warmup_fits": warmup_fits, "n_student_fits": len(records),
        "files": {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}}
    write_json(root / COMPLETE, done)
    return verify_transfer(root)
