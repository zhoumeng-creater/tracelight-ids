"""Fixed old HGB/MLP architectures with explicit training-support weights.

The eight Zeek names do not match the legacy NF categorical-name list.  Thus
the legacy MLP's *actual* transform applies signed log1p to all eight fields,
including the ports and protocol.  Reusing that implementation is intentional.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from torch import nn

from ids.common import read_json, sha256_file, write_json
from ids.flow_history import model as old_mlp
from ids.flow_history import trees as old_hgb
from ids.threshold_validation.observations import RAW_FEATURES


EPOCHS = old_mlp.EPOCHS
TRAIN_BATCH = old_mlp.TRAIN_BATCH
GROUPS = {"hgb": None, "mlp17": 17, "mlp29": 29, "mlp43": 43}
VIEWS = {"raw8", "ports_zero8"}
META_KEY = "support_diagnostic"


@dataclass
class Model:
    backend: object
    view: str
    group: str


def _view(features):
    if isinstance(features, str):
        if features not in VIEWS:
            raise ValueError(f"unknown diagnostic view: {features}")
        return features
    if tuple(features) != RAW_FEATURES:
        raise ValueError("diagnostic features must be the registered eight-field order")
    return "raw8"


def _group(group):
    if group not in GROUPS:
        raise ValueError(f"unknown diagnostic model group: {group}")
    return group


def view_frame(frame, features):
    """Keep eight input slots and change only the two ports in the masked view."""
    view = _view(features)
    if view == "raw8":
        return frame
    masked = frame.copy()
    masked.loc[:, ["orig_port", "resp_port"]] = 0
    return masked


def _weights(weights, labels):
    values = np.ones(len(labels), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape != (len(labels),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("weights must be one finite nonnegative value per training row")
    mean = float(values.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("training weights must have positive global mean")
    # Normalize globally once.  Never renormalize within a minibatch.
    normalized = np.ascontiguousarray(values / mean, dtype=np.float64)
    if any(float(normalized[labels == label].sum()) <= 0 for label in (0, 1)):
        raise ValueError("both binary classes must have positive training weight")
    info = {
        "provided": weights is not None,
        "normalization": "divide once by global training-weight mean; no minibatch normalization",
        "input_mean": mean,
        "normalized_mean": float(normalized.mean()),
        "min": float(normalized.min()),
        "max": float(normalized.max()),
        "sum": float(normalized.sum()),
        "class_sums": {str(label): float(normalized[labels == label].sum()) for label in (0, 1)},
        "float64_le_sha256": hashlib.sha256(normalized.astype("<f8", copy=False).tobytes()).hexdigest(),
    }
    return normalized, info


def weighted_bce(logits, targets, weights):
    """Global-normalized row weights multiply unreduced BCE, then row mean."""
    if logits.shape != targets.shape or weights.shape != targets.shape:
        raise ValueError("logits, targets and row weights must have the same shape")
    losses = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (losses * weights).mean()


def _fit_mlp(frame, path, *, group, view, weights, weight_info, threads, device):
    actual = old_mlp._device(device, threads)
    values = old_mlp._transform(frame, RAW_FEATURES)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1.0
    if actual.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual)
    x = old_mlp._tensor(values, mean, scale, actual)
    labels = frame["label"].to_numpy()
    targets = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(actual)
    row_weights = torch.from_numpy(np.asarray(weights, dtype=np.float32)).to(actual)
    if not torch.isfinite(row_weights).all().item():
        raise ValueError("weights cannot be represented as finite FP32 values")
    seed = GROUPS[group]
    network = old_mlp._network(8, seed=seed, base_width=8).to(actual)
    probe = x[:TRAIN_BATCH]
    with torch.inference_mode():
        network(probe)
        old_mlp._sync(actual)
        started = time.perf_counter()
        prediction = network(probe)
        old_mlp._sync(actual)
        probe_seconds = time.perf_counter() - started
    if prediction.device != actual or not torch.isfinite(prediction).all().item():
        raise RuntimeError("model-device probe failed")
    optimizer = torch.optim.AdamW(network.parameters(), lr=old_mlp.LR, weight_decay=old_mlp.WEIGHT_DECAY)
    sampler = torch.Generator(device="cpu").manual_seed(seed)
    training_loss = []
    old_mlp._sync(actual)
    started = time.perf_counter()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=sampler).to(actual)
        epoch_loss = torch.zeros((), device=actual)
        for indices in order.split(TRAIN_BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = network(x[indices]).flatten()
            loss = weighted_bce(logits, targets[indices], row_weights[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(indices)
        value = float(epoch_loss / len(x))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite training loss")
        training_loss.append(value)
    old_mlp._sync(actual)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(actual) if actual.type == "cuda" else None
    batch_probe = old_mlp._select_score_batch(network.eval(), x)
    state = {
        "format": old_mlp.FORMAT,
        "features": list(RAW_FEATURES),
        "hidden": list(old_mlp.HIDDEN),
        "score_batch_probe": batch_probe,
        "mean": torch.from_numpy(mean.copy()),
        "scale": torch.from_numpy(scale.copy()),
        "weights": {name: value.detach().cpu() for name, value in network.state_dict().items()},
        META_KEY: {"group": group, "view": view},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".fit.json"), {
        "format": old_mlp.FORMAT,
        "study": "support-diagnostic-v1",
        "group": group,
        "view": view,
        "features": list(RAW_FEATURES),
        "rows": len(frame),
        "positive_rows": int(np.sum(labels)),
        "seed": seed,
        "base_width": 8,
        "epochs": EPOCHS,
        "training_batch_size": TRAIN_BATCH,
        "scoring_batch_size": batch_probe["selected"],
        "cpu_scoring_batch_size": old_mlp.SCORE_BATCH,
        "score_batch_probe": batch_probe,
        "loss": "mean(global_mean_one_weight * unreduced_binary_cross_entropy_with_logits)",
        "weighting": weight_info,
        "preprocessing": {
            "function": "ids.flow_history.model._transform",
            "actual_transform": "signed log1p on all eight Zeek fields, including protocol and ports",
            "fit_scope": "only the frame passed to this training arm; unweighted training mean/std",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        "optimizer": "AdamW",
        "learning_rate": old_mlp.LR,
        "weight_decay": old_mlp.WEIGHT_DECAY,
        "hidden": list(old_mlp.HIDDEN),
        "training_loss_per_epoch": training_loss,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "threads": torch.get_num_threads(),
        "device": str(prediction.device),
        "device_name": torch.cuda.get_device_name(actual) if actual.type == "cuda" else "cpu",
        "device_probe_rows": len(probe),
        "device_probe_seconds": probe_seconds,
        "device_probe_rows_per_second": len(probe) / probe_seconds,
        "train_seconds": elapsed,
        "train_rows_per_second": len(frame) * EPOCHS / elapsed,
        "cuda_peak_allocated_bytes": peak,
        "model_sha256": sha256_file(path),
        "model_artifact_bytes": path.stat().st_size,
    })
    return path


def fit(frame, features, path, *, group, weights, threads, device="cuda:0"):
    """Train one explicitly registered arm; no validation or future rows used."""
    group, view = _group(group), _view(features)
    labels = frame["label"].to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("training frame must contain both binary labels")
    normalized, weight_info = _weights(weights, labels)
    prepared = view_frame(frame, view)
    path = Path(path)
    if group != "hgb":
        return _fit_mlp(prepared, path, group=group, view=view, weights=normalized,
                        weight_info=weight_info, threads=threads, device=device)
    # Same estimator and raw finite numeric preprocessing as the historical arm.
    old_hgb.fit_arm(prepared, RAW_FEATURES, path, threads=threads, sample_weight=normalized)
    state = joblib.load(path)
    state[META_KEY] = {"group": group, "view": view}
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(state, temporary)
    temporary.replace(path)
    metadata = read_json(path.with_suffix(".fit.json"))
    metadata.update({
        "study": "support-diagnostic-v1", "group": group, "view": view,
        "weighting": weight_info,
        "preprocessing": {"actual_transform": "raw eight numeric fields",
                          "fit_scope": "only this arm's training frame; HGB bins fit by historical fit_arm"},
        "model_sha256": sha256_file(path), "model_artifact_bytes": path.stat().st_size,
    })
    write_json(path.with_suffix(".fit.json"), metadata)
    return path


def load(path, *, group, threads, device="cuda:0"):
    """Load new weighted arms or historical raw arms without re-fitting."""
    group = _group(group)
    if group == "hgb":
        state = joblib.load(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    metadata = state.get(META_KEY, {})
    view = _view(metadata.get("view", "raw8"))
    if metadata.get("group", group) != group:
        raise ValueError("requested model group differs from the saved arm")
    if tuple(state.get("features", ())) != RAW_FEATURES:
        raise ValueError("saved model is not the registered eight-field arm")
    backend = (old_hgb.load_arm(path, threads=threads) if group == "hgb"
               else old_mlp.load_arm(path, threads=threads, device=device))
    return Model(backend, view, group)


def score(model, frame, features, group):
    """Score using the saved view; a raw model cannot stand in for masked fit."""
    view, group = _view(features), _group(group)
    if model.view != view or model.group != group:
        raise ValueError("requested view or group differs from the fitted model")
    prepared = view_frame(frame, view)
    scorer = old_hgb.score_arm if group == "hgb" else old_mlp.score_arm
    return scorer(model.backend, prepared, RAW_FEATURES)
