"""One bounded F trajectory, selected by V, delivered without a final refit."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from .preprocessing import POLICIES, Preprocessor
from .sampling import GROUPS, IDENTITY, GroupSampler, canonical_frame, derived_seed, group_indices
from .selection import Selector, selection_metrics


FORMAT = "model-training-redesign-v1"


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rng_state(device):
    import torch
    result = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state(device).clone()
    return result


def _seed_torch(seed, device):
    import torch
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _parameters(design):
    from ids.recent_baselines import _parameters as resolve
    model, training, ple = design["model"], design["training"], design["input"]["ple"]
    return resolve({"epochs": 1, "patience": design["selection"]["patience_validation_events"],
                    "batch_size": training["sampler"]["batch_size"], "k": model["k"],
                    "n_blocks": model["n_blocks"], "d_block": model["d_block"],
                    "embedding_dim": ple["embedding_dim"], "n_bins": ple["max_bins"],
                    "learning_rate": training["optimizer"]["lr"],
                    "weight_decay": training["optimizer"]["weight_decay"],
                    "dropout": model["dropout"], "num_embeddings": "piecewise_linear",
                    "arch_type": model["arch_type"]})


def _validate(frame, validation, design):
    for name, data in (("F", frame), ("V", validation)):
        if not len(data) or not data.label.isin([0, 1]).all() or not data.profile.isin(["A", "B"]).all():
            raise ValueError(f"{name} requires nonempty A/B binary-label data")
        if not all(str(source).startswith("TQH-" + profile + "_")
                   for source, profile in zip(data.source_id, data.profile)):
            raise ValueError(f"{name} profile does not match source_id")
        if len(data) != design["roles"][name]["rows"]:
            raise ValueError(f"{name} row count differs from the registered design")
    if set(map(tuple, frame[IDENTITY].to_numpy())) & set(map(tuple, validation[IDENTITY].to_numpy())):
        raise ValueError("F and V identities overlap")
    groups = group_indices(frame)
    expected = {item["id"]: item["rows"] for item in design["training"]["groups"]}
    if {g: len(indices) for g, indices in groups.items()} != expected:
        raise ValueError("F group counts differ from the registered design")
    for panel in design["selection"]["panels"]:
        part = validation.loc[validation.profile.eq(panel["profile"])
                              & validation.ip_protocol.eq(panel["ip_protocol"])]
        if ((int(part.label.eq(0).sum()), int(part.label.eq(1).sum()))
                != (panel["normal"], panel["positive"])):
            raise ValueError("Validation panel counts differ from the registered design")
    if design["training"].get("sample_weight") is not None or design["training"].get("pos_weight") is not None:
        raise ValueError("Four-group batches must not receive a second loss weight")
    if design["training"]["final_refits"] != 0:
        raise ValueError("Final refitting is not supported")


def four_group_loss(logits, labels, group_batch_size):
    """Each contiguous group contributes one quarter, then each member equally."""
    import torch
    if logits.ndim != 3 or logits.shape[2] != 1 or len(logits) != 4 * group_batch_size:
        raise ValueError("Expected four equally sized groups and (B,K,1) logits")
    if labels.shape != (len(logits),):
        raise ValueError("Labels must align with the four-group batch")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), labels[:, None].expand(-1, logits.shape[1]), reduction="none")
    groups = losses.reshape(4, group_batch_size, logits.shape[1]).mean(dim=(1, 2))
    return groups.mean(), groups


def _endpoint(network, numeric, categorical, labels, device, batch_size, *, verify_rng=False):
    import torch
    from ids.route_batch.tabm_training import stable_scores
    network.eval()
    scores, head_losses = [], []
    before = _state_sha256(_rng_state(device)) if verify_rng else None
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            logits = network(torch.from_numpy(numeric[start:start + batch_size]).to(device),
                             torch.from_numpy(categorical[start:start + batch_size]).to(device))
            score = stable_scores(logits)
            if not torch.isfinite(score).all():
                raise FloatingPointError("Nonfinite endpoint score")
            scores.append(score.cpu().numpy())
            if labels is not None:
                targets = torch.from_numpy(labels[start:start + batch_size]).to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), targets[:, None].expand(-1, logits.shape[1]), reduction="none")
                head_losses.append(loss.mean(dim=1).cpu().numpy())
    if verify_rng and _state_sha256(_rng_state(device)) != before:
        raise RuntimeError("Validation or eval-mode scoring consumed training RNG")
    result = {"scores": np.concatenate(scores).astype(np.float64) if scores else np.empty(0)}
    if labels is not None:
        result["head_bce_by_row"] = np.concatenate(head_losses).astype(np.float64)
        result["ensemble_bce_by_row"] = np.logaddexp(
            0.0, np.where(labels == 1, -result["scores"], result["scores"]))
    return result


def _group_report(endpoint, frame):
    return {g: {"rows": len(idx), "head_bce": float(endpoint["head_bce_by_row"][idx].mean()),
                "ensemble_bce": float(endpoint["ensemble_bce_by_row"][idx].mean())}
            for g, idx in group_indices(frame).items()}


def _profile_separation(scores, frame):
    return {profile: bool(scores[(frame.profile == profile) & (frame.label == 1)].min()
                          > scores[(frame.profile == profile) & (frame.label == 0)].max())
            for profile in ("A", "B")}


@dataclass
class Predictor:
    network: object
    preprocessor: Preprocessor
    device: object
    batch_size: int

    def score(self, frame):
        return self.score_transformed(*self.preprocessor.transform(frame))

    def score_transformed(self, numeric, categorical):
        return _endpoint(self.network, numeric, categorical, None, self.device, self.batch_size)["scores"]


def fit_trajectory(frame, validation, output_dir, *, arm, seed, threads, device, design):
    """Never reads calibration/evaluation data or performs a second fit."""
    if arm not in POLICIES or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Expected a registered protocol arm and uint32 seed")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Training output directory must be empty; incomplete fits are not retried")
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "attempt.json", {"status": "started", "arm": arm, "seed": int(seed),
                                   "fit_rows": len(frame), "validation_rows": len(validation),
                                   "final_refits": 0, "design": design})
    try:
        result = _fit(frame, validation, root, arm=arm, seed=int(seed), threads=threads,
                      device=device, design=design)
    except Exception as error:
        result = {"status": "failed", "arm": arm, "seed": int(seed), "fit_calls": 1,
                  "error_type": type(error).__name__, "error": str(error), "retry_allowed": False}
        _write(root / "failure.json", {**result, "traceback": traceback.format_exc()})
    _write(root / "summary.json", result)
    return result


def _fit(frame, validation, root, *, arm, seed, threads, device, design):
    import torch
    from ids.flow_history.model import _device, _sync
    from ids.recent_baselines import _numeric_bins, make_tabm_network

    started = time.perf_counter()
    frame, validation = canonical_frame(frame), canonical_frame(validation)
    _validate(frame, validation, design)
    parameters = _parameters(design)
    preprocessor = Preprocessor.fit(frame, arm=arm, design=design)
    numeric, categorical = preprocessor.transform(frame)
    v_num, v_cat = preprocessor.transform(validation)
    y, vy = frame.label.to_numpy(dtype=np.float32), validation.label.to_numpy(dtype=np.float32)
    bins, constants = _numeric_bins(numeric, parameters)
    _write(root / "preprocessing.json", preprocessor.to_dict())
    _write(root / "numeric_bins.json", {"bins": bins, "constant_columns": constants})
    sampler = GroupSampler(frame, seed=seed,
                           group_batch_size=design["training"]["sampler"]["draws_per_group_per_step"])
    if 4 * sampler.batch_size != parameters["batch_size"]:
        raise ValueError("Batch size differs from four equal groups")
    cap = int(design["training"]["max_optimizer_steps"])
    validation_steps = design["selection"]["validation_steps"]
    if cap < 1 or validation_steps != sorted(set(validation_steps)) or not validation_steps or validation_steps[-1] != cap:
        raise ValueError("Validation schedule must be increasing and end at the optimizer cap")
    actual = _device(device, threads)
    if actual.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic)
    history, fit_seconds, selected_result = [], 0.0, None
    selector = Selector(int(design["selection"]["patience_validation_events"]))
    pairing = {"preprocessing_sha256": _hash_json(preprocessor.base.to_dict()),
               "numeric_bins_sha256": _hash_json(bins),
               "F_identity_sha256": _hash_json(frame[IDENTITY].values.tolist()),
               "initialization_seed": seed, "dropout_seed": derived_seed(seed, "dropout"),
               "sampler_seeds": sampler.seeds,
               "draw_chain_by_validation_step": {}, "dropout_rng_by_validation_step": {}}

    def save(path, network, step):
        state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
        torch.save({"format": FORMAT, "arm": arm, "seed": seed, "selected_step": step,
                    "preprocessing": preprocessor.to_dict(), "numeric_bins": bins,
                    "constant_bin_columns": constants, "parameters": parameters,
                    "state_dict": state, "state_sha256": _state_sha256(state),
                    "dropout_rng_state": _rng_state(actual), "rng_step": step,
                    "sampling_state": sampler.state(),
                    "score_batch_size": parameters["batch_size"], "fit_rows": len(frame),
                    "score_contract": "FP32 network; FP64 stable log-odds of mean member probabilities",
                    "preprocessing_sha256": pairing["preprocessing_sha256"],
                    "numeric_bins_sha256": pairing["numeric_bins_sha256"], "final_refits": 0}, path)

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=[] if actual.type == "cpu" else [actual.index]):
            _seed_torch(seed, actual)
            network = make_tabm_network(5, preprocessor.cat_cardinalities, parameters, numeric_bins=bins).to(actual)
            pairing["initial_state_sha256"] = _state_sha256(network.state_dict())
            optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"],
                                          weight_decay=parameters["weight_decay"])
            _seed_torch(pairing["dropout_seed"], actual)
            pairing["initial_dropout_rng_sha256"] = _state_sha256(_rng_state(actual))
            pairing["initial_sampler_rng_sha256"] = sampler.rng_sha256()
            _write(root / "pairing.json", pairing)
            x, cat, targets = (torch.from_numpy(value).to(actual) for value in (numeric, categorical, y))
            clip = float(design["training"]["optimizer"]["gradient_clip_norm"])
            path_groups, clip_count, norm_sum, norm_max, window_steps = np.zeros(4), 0, 0.0, 0.0, 0
            _sync(actual)
            interval_started = time.perf_counter()
            stop_reason = "optimizer_step_cap"
            for step in range(1, cap + 1):
                network.train()
                indices = torch.from_numpy(sampler.next()).to(actual)
                optimizer.zero_grad(set_to_none=True)
                loss, groups = four_group_loss(network(x[indices], cat[indices]), targets[indices], sampler.batch_size)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), clip, error_if_nonfinite=True))
                optimizer.step()
                path_groups += groups.detach().cpu().numpy()
                clip_count += norm > clip
                norm_sum, norm_max = norm_sum + norm, max(norm_max, norm)
                window_steps += 1
                if step not in validation_steps:
                    continue
                _sync(actual)
                interval_seconds = time.perf_counter() - interval_started
                fit_seconds += interval_seconds
                tr = _endpoint(network, numeric, categorical, y, actual, parameters["batch_size"], verify_rng=True)
                vr = _endpoint(network, v_num, v_cat, vy, actual, parameters["batch_size"], verify_rng=True)
                metrics = selection_metrics(vr["scores"], validation)
                decision = selector.update(metrics, step)
                sample_audit = sampler.audit(reset_window=True)
                pairing["draw_chain_by_validation_step"][str(step)] = sample_audit["draw_sha256"]
                pairing["dropout_rng_by_validation_step"][str(step)] = _state_sha256(_rng_state(actual))
                record = {"step": step, "optimizer_steps": step, "fit_seconds": interval_seconds,
                          "path_group_head_bce": dict(zip(GROUPS, (path_groups / window_steps).tolist())),
                          "path_loss_is_endpoint_loss": False, "training_groups": _group_report(tr, frame),
                          "validation_groups": _group_report(vr, validation),
                          "F_profile_strict_separation": _profile_separation(tr["scores"], frame),
                          "gradient": {"preclip_mean_norm": norm_sum / window_steps,
                                       "preclip_max_norm": norm_max, "clipped_steps": int(clip_count),
                                       "window_steps": window_steps},
                          "sampling": sample_audit, "validation": metrics, "selection": decision}
                history.append(record)
                if decision["improved"]:
                    save(root / "best.pt", network, step)
                    selected_result = {"train_scores": tr["scores"], "validation_scores": vr["scores"]}
                _write(root / "history.json", history)
                _write(root / "pairing.json", pairing)
                _write(root / "sampling.json", sampler.state())
                torch.save({"step": step, "dropout_seed": pairing["dropout_seed"],
                            "dropout_rng_state": _rng_state(actual)}, root / "rng_state.pt")
                if decision["stop"]:
                    stop_reason = decision["stop_reason"]
                    break
                path_groups, clip_count, norm_sum, norm_max, window_steps = np.zeros(4), 0, 0.0, 0.0, 0
                _sync(actual)
                interval_started = time.perf_counter()
            if selected_result is None:
                raise RuntimeError("No checkpoint was selected")
            shutil.copyfile(root / "best.pt", root / "model.pt")
            np.savez_compressed(root / "selected_scores.npz", **selected_result)
            _write(root / "score_identities.json", {"F": frame[IDENTITY].values.tolist(),
                                                     "V": validation[IDENTITY].values.tolist()})
            parameters_count = sum(p.numel() for p in network.parameters())
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[1:]
    return {"status": "complete", "arm": arm, "seed": seed, "fit_calls": 1, "final_refits": 0,
            "fit_rows": len(frame), "selected_step": selector.best_step, "optimizer_steps": step,
            "completed_validation_events": len(history), "stop_reason": stop_reason,
            "artifact_path": str(root / "model.pt"), "best_path": str(root / "best.pt"),
            "history_path": str(root / "history.json"), "pairing_path": str(root / "pairing.json"),
            "rng_audit_path": str(root / "rng_state.pt"),
            "train_scores_path": str(root / "selected_scores.npz"), "parameters": parameters_count,
            "selected_preprocessing_sha256": pairing["preprocessing_sha256"],
            "selected_numeric_bins_sha256": pairing["numeric_bins_sha256"],
            "selected_artifact_sha256": hashlib.sha256((root / "model.pt").read_bytes()).hexdigest(),
            "model_artifact_bytes": (root / "model.pt").stat().st_size,
            "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(actual) if actual.type == "cuda" else None}


def load_predictor(artifact, threads=1, device="cpu"):
    import torch
    from ids.flow_history.model import _device
    from ids.recent_baselines import make_tabm_network
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT or state.get("arm") not in POLICIES or state.get("final_refits") != 0:
        raise ValueError("Unsupported redesign artifact")
    preprocessor = Preprocessor.from_dict(state["preprocessing"])
    if preprocessor.protocol_policy != POLICIES[state["arm"]]:
        raise ValueError("Artifact arm and protocol policy differ")
    if (_state_sha256(state["state_dict"]) != state["state_sha256"]
            or _hash_json(preprocessor.base.to_dict()) != state["preprocessing_sha256"]
            or _hash_json(state["numeric_bins"]) != state["numeric_bins_sha256"]):
        raise ValueError("Model or fitted preprocessing binding changed")
    actual = _device(device, threads)
    with torch.random.fork_rng(devices=[]):
        network = make_tabm_network(5, preprocessor.cat_cardinalities, state["parameters"],
                                   numeric_bins=state["numeric_bins"])
    network.load_state_dict(state["state_dict"])
    return Predictor(network.to(actual).eval(), preprocessor, actual, state["score_batch_size"])
