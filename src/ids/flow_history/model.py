"""Frozen small MLP for the paired NF-v3 historical-observation experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from ids.common import sha256_file, write_json
from ids.context_cache.lab_data import CATEGORICAL, FEATURES
from .arm_cost import benchmark_arm, checked_features


FORMAT = "flow-history-mlp-v1"
SEED = 20261111
EPOCHS = 80
TRAIN_BATCH = 512
SCORE_BATCH = 4096
HIDDEN = (128, 64)
LR = 0.001
WEIGHT_DECAY = 0.01


@dataclass
class Arm:
    network: nn.Module
    features: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    device: torch.device
    score_batch: int


def _device(device, threads):
    if not isinstance(threads, (int, np.integer)) or isinstance(threads, bool) or threads < 1:
        raise ValueError("threads must be a positive integer")
    actual = torch.device(device)
    if actual.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if actual.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        actual = torch.device("cuda", torch.cuda.current_device() if actual.index is None else actual.index)
        torch.cuda.get_device_properties(actual)
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(int(threads))
    return actual


def _transform(frame, names):
    values = frame.loc[:, list(names)].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("input contains nonfinite values")
    # Keep port and protocol identities on the uncompressed numeric scale.
    numerical = [i for i, name in enumerate(names) if name not in CATEGORICAL]
    values[:, numerical] = np.sign(values[:, numerical]) * np.log1p(np.abs(values[:, numerical]))
    return values


def _tensor(values, mean, scale, device):
    standardized = np.ascontiguousarray((values - mean) / scale, dtype=np.float32)
    if not np.isfinite(standardized).all():
        raise ValueError("nonfinite standardized FP32 model input")
    return torch.from_numpy(standardized).to(device)


def _network(width):
    # Draw common layers once in the same order for every width. Historical
    # inputs start with zero coefficients, so every arm starts at the raw model.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        base = nn.Linear(len(FEATURES), HIDDEN[0])
        hidden = nn.Linear(HIDDEN[0], HIDDEN[1])
        output = nn.Linear(HIDDEN[1], 1)
        first = nn.Linear(width, HIDDEN[0])
        common = min(width, len(FEATURES))
        with torch.no_grad():
            first.weight.zero_()
            first.weight[:, :common].copy_(base.weight[:, :common])
            first.bias.copy_(base.bias)
    return nn.Sequential(first, nn.GELU(), hidden, nn.GELU(), output)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _network_scores(network, x, batch_size):
    logits = [network(block).reshape(-1) for block in x.split(batch_size)]
    values = torch.cat(logits).cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite scoring output")
    return values


def _select_score_batch(network, x):
    """Compare equal P inputs on the fitted GPU network, independent of C/E."""
    if x.device.type != "cuda":
        return {"status": "fixed_cpu", "selected": SCORE_BATCH}
    probe = x[:4096]
    candidates = sorted({min(len(probe), n) for n in (512, 2048, 4096)})
    results = {size: {"batch": size, "seconds": [], "cuda_peak_allocated_bytes": 0} for size in candidates}
    reference = _network_scores(network, probe, candidates[0])
    for size in candidates:
        warm = _network_scores(network, probe, size)
        diff = np.abs(warm - reference)
        results[size]["max_abs_score_delta_vs_smallest"] = float(diff.max())
        results[size]["within_score_tolerance"] = bool((diff <= 3e-5 + 3e-5*np.abs(reference)).all())
    # Rotate order to avoid systematically favouring the last candidate as clocks warm up.
    for repeat in range(5):
        start_at = repeat % len(candidates)
        for size in candidates[start_at:] + candidates[:start_at]:
            _sync(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
            start = time.perf_counter()
            _network_scores(network, probe, size)
            _sync(x.device)
            elapsed = time.perf_counter() - start
            row = results[size]
            row["seconds"].append(elapsed)
            row["cuda_peak_allocated_bytes"] = max(row["cuda_peak_allocated_bytes"],
                                                   torch.cuda.max_memory_allocated(x.device))
    for row in results.values():
        row["median_rows_per_second"] = float(len(probe) / np.median(row["seconds"]))
    eligible = [size for size in candidates if results[size]["within_score_tolerance"]]
    best = max(eligible, key=lambda size: results[size]["median_rows_per_second"])
    return {"status": "measured_gpu", "selected": best, "source": "first P rows, no labels",
            "rows_per_trial": len(probe), "repeats_per_candidate": 5, "candidates": list(results.values()),
            "scope": "same standardized inputs already on GPU; fitted FP32 forward and logits to CPU; "
                     "peak PyTorch allocated memory includes retained training tensors",
            "selection": "highest median throughput among numerically matched candidates; smaller size breaks ties"}


def fit_arm(p, features, path, *, threads, device="cuda:0"):
    """Fit once on P. Persist portable weights, preprocessing and training evidence."""
    names = checked_features(features)
    actual = _device(device, threads)
    y = p["label"].to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("training P must contain both binary labels")
    values = _transform(p, names)
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale == 0] = 1.0
    if actual.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual)
    x = _tensor(values, mean, scale, actual)
    targets = torch.from_numpy(np.array(y, dtype=np.float32)).to(actual)
    network = _network(len(names)).to(actual)
    # Representative forward on the requested device, before committing to the full fit.
    probe = x[:TRAIN_BATCH]
    with torch.inference_mode():
        network(probe)
        _sync(actual)
        start = time.perf_counter()
        prediction = network(probe)
        _sync(actual)
        probe_seconds = time.perf_counter() - start
    if prediction.device != actual or not torch.isfinite(prediction).all().item():
        raise RuntimeError("model-device probe failed")
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sampler = torch.Generator(device="cpu").manual_seed(SEED)
    training_loss = []
    start = time.perf_counter()
    for _ in range(EPOCHS):
        order = torch.randperm(len(x), generator=sampler).to(actual)
        epoch_loss = torch.zeros((), device=actual)
        for indices in order.split(TRAIN_BATCH):
            optimizer.zero_grad(set_to_none=True)
            logits = network(x[indices]).flatten()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[indices])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(indices)
        value = float(epoch_loss / len(x))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite training loss")
        training_loss.append(value)
    elapsed = time.perf_counter() - start
    training_peak = torch.cuda.max_memory_allocated(actual) if actual.type == "cuda" else None
    batch_probe = _select_score_batch(network.eval(), x)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": FORMAT, "features": list(names), "hidden": list(HIDDEN), "score_batch_probe": batch_probe,
             "mean": torch.from_numpy(mean.copy()), "scale": torch.from_numpy(scale.copy()),
             "weights": {name: value.detach().cpu() for name, value in network.state_dict().items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    write_json(path.with_suffix(".fit.json"), {
        "format": FORMAT, "features": list(names), "rows": len(p), "positive_rows": int(np.sum(y)),
        "seed": SEED, "epochs": EPOCHS, "training_batch_size": TRAIN_BATCH, "scoring_batch_size": batch_probe["selected"],
        "score_batch_probe": batch_probe, "cpu_scoring_batch_size": SCORE_BATCH,
        "loss": "binary_cross_entropy_with_logits", "optimizer": "AdamW", "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY, "hidden": list(HIDDEN), "training_loss_per_epoch": training_loss,
        "parameters": sum(param.numel() for param in network.parameters()), "threads": torch.get_num_threads(),
        "device": str(prediction.device), "device_name": torch.cuda.get_device_name(actual) if actual.type == "cuda" else "cpu",
        "device_probe_rows": len(probe), "device_probe_seconds": probe_seconds,
        "device_probe_rows_per_second": len(probe) / probe_seconds,
        "train_seconds": elapsed, "train_rows_per_second": len(p)*EPOCHS/elapsed,
        "cuda_peak_allocated_bytes": training_peak,
        "model_sha256": sha256_file(path), "model_artifact_bytes": path.stat().st_size,
    })
    return path


def load_arm(path, *, device, threads):
    actual = _device(device, threads)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != FORMAT or state["hidden"] != list(HIDDEN):
        raise ValueError("unrecognized flow-history model")
    names = checked_features(state["features"])
    mean, scale = state["mean"].numpy(), state["scale"].numpy()
    if mean.shape != (len(names),) or scale.shape != mean.shape or not np.isfinite(mean).all() or not (
        np.isfinite(scale).all() and (scale > 0).all()
    ):
        raise ValueError("invalid saved normalization")
    network = _network(len(names))
    network.load_state_dict(state["weights"])
    saved_batch = state.get("score_batch_probe", {}).get("selected", SCORE_BATCH)
    if type(saved_batch) is not int or saved_batch < 1:
        raise ValueError("invalid saved scoring batch size")
    batch = saved_batch if actual.type == "cuda" else SCORE_BATCH
    return Arm(network.to(actual).eval(), names, mean, scale, actual, batch)


@torch.inference_mode()
def score_arm(model, frame, features):
    if tuple(features) != model.features:
        raise ValueError("features or feature order differ from the fitted arm")
    x = _tensor(_transform(frame, model.features), model.mean, model.scale, model.device)
    return _network_scores(model.network, x, model.score_batch)


def bench_arm(path, C, E, features, threads, *, reference_c, reference_e, output_dir):
    return benchmark_arm("mlp", path, C, E, features, threads, reference_c=reference_c,
                         reference_e=reference_e, output_dir=output_dir)
