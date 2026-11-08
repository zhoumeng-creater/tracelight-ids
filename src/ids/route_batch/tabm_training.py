"""Bounded TabM trajectories on prepared numeric arrays, without calibration I/O."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..recent_baselines import _parameters, make_tabm_network, tabm_head_loss


ARTIFACT_VERSION = 1
SCORE_CONTRACT = "fp32-network; fp64-log-mean-sigmoid-odds; no-probability-clipping"


def _matrix(value, *, width=None, allow_empty=False):
    result = np.asarray(value)
    if (result.dtype != np.float32 or result.ndim != 2 or result.shape[1] < 1
            or (not allow_empty and not len(result))
            or (width is not None and result.shape[1] != width)
            or not np.isfinite(result).all()):
        raise ValueError("Expected a finite float32 feature matrix with matching width")
    return np.ascontiguousarray(result)


def _labels(value, rows):
    result = np.asarray(value)
    if result.ndim != 1 or len(result) != rows or not np.isin(result, [0, 1]).all():
        raise ValueError("Expected aligned binary labels")
    return np.asarray(result, dtype=np.float32)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _log_probabilities(logits):
    import torch

    if logits.ndim != 3 or logits.shape[2] != 1 or logits.shape[1] < 1:
        raise ValueError("Expected logits with shape (rows, heads, 1)")
    values = logits.squeeze(-1).to(torch.float64)
    normalizer = math.log(values.shape[1])
    return (torch.logsumexp(torch.nn.functional.logsigmoid(values), dim=1) - normalizer,
            torch.logsumexp(torch.nn.functional.logsigmoid(-values), dim=1) - normalizer)


def stable_scores(logits):
    """Log odds of the mean head probability, including saturated FP32 logits."""
    positive, negative = _log_probabilities(logits)
    return positive - negative


def _evaluate(network, x, y, *, device, batch_size):
    import torch

    network.eval()
    head_loss, ensemble_loss, scores = 0.0, 0.0, []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            features = torch.from_numpy(x[start:start + batch_size]).to(device)
            targets = torch.from_numpy(y[start:start + batch_size]).to(device)
            logits = network(features, None)
            positive, negative = _log_probabilities(logits)
            score = positive - negative
            batch_head = tabm_head_loss(logits, targets)
            batch_ensemble = -(targets.to(torch.float64) * positive
                               + (1.0 - targets.to(torch.float64)) * negative).sum()
            if not (torch.isfinite(score).all() and torch.isfinite(batch_head)
                    and torch.isfinite(batch_ensemble)):
                raise FloatingPointError("Non-finite endpoint score or loss")
            head_loss += float(batch_head) * len(targets)
            ensemble_loss += float(batch_ensemble)
            scores.append(score.cpu().numpy())
    return {"head_bce": head_loss / len(x), "ensemble_bce": ensemble_loss / len(x),
            "scores": np.concatenate(scores)}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _save_checkpoint(path, network, parameters, config, *, epoch, updates, complete_epoch):
    import torch

    torch.save({"artifact_version": ARTIFACT_VERSION, "parameters": parameters,
                "config": config, "epoch": int(epoch), "updates": int(updates),
                "complete_epoch": bool(complete_epoch), "score_contract": SCORE_CONTRACT,
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in network.state_dict().items()}}, path)


def _save_endpoint(root, name, network, parameters, config, epoch, updates, train_result,
                   validation_result=None):
    path, score_path = root / f"{name}.pt", root / f"{name}.scores.npz"
    _save_checkpoint(path, network, parameters, config, epoch=epoch, updates=updates,
                     complete_epoch=True)
    arrays = {"train_scores": train_result["scores"]}
    if validation_result is not None:
        arrays["validation_scores"] = validation_result["scores"]
    np.savez_compressed(score_path, **arrays)
    result = {"epoch": epoch, "updates": updates, "model_path": str(path),
              "scores_path": str(score_path), "train_head_bce": train_result["head_bce"],
              "train_ensemble_bce": train_result["ensemble_bce"]}
    if validation_result is not None:
        result.update({"validation_head_bce": validation_result["head_bce"],
                       "validation_ensemble_bce": validation_result["ensemble_bce"]})
    return result


def _train(x_train, y_train, x_val, y_val, *, width, learning_rate, seed, output_dir,
           device, epochs, checkpoints, batch_size, k, n_blocks, weight_decay,
           dropout, max_seconds, fixed):
    import torch

    x_train = _matrix(x_train)
    y_train = _labels(y_train, len(x_train))
    if np.unique(y_train).size != 2:
        raise ValueError("Training requires both binary classes")
    if not fixed:
        x_val = _matrix(x_val, width=x_train.shape[1])
        y_val = _labels(y_val, len(x_val))
    epochs = _positive_integer(epochs, "epochs")
    batch_size = _positive_integer(batch_size, "batch_size")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    if max_seconds is not None and (isinstance(max_seconds, bool)
                                   or not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    checkpoint_epochs = sorted({epochs, *(_positive_integer(e, "checkpoint") for e in checkpoints)})
    checkpoint_epochs = [epoch for epoch in checkpoint_epochs if epoch <= epochs]
    parameters = _parameters({"d_block": width, "learning_rate": learning_rate,
                              "epochs": epochs, "batch_size": batch_size, "k": k,
                              "n_blocks": n_blocks, "weight_decay": weight_decay,
                              "dropout": dropout, "num_embeddings": "none", "arch_type": "tabm"})
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Training output directory must be empty")
    config = {"n_features": x_train.shape[1], "training_rows": len(x_train),
              "validation_rows": 0 if fixed else len(x_val), "seed": int(seed),
              "requested_epochs": epochs, "device": str(device),
              "max_seconds": None if max_seconds is None else float(max_seconds),
              "gradient_clip_norm": 1.0, "positive_weight": 1.0,
              "objective": "mean_independent_head_unweighted_bce",
              "selection": "fixed_epoch_no_validation" if fixed else "validation_ensemble_bce",
              "score_contract": SCORE_CONTRACT}
    start = time.perf_counter()
    summary = {"status": "running", "best_epoch": None, "best_checkpoint": None,
               "checkpoints": {}, "epochs_completed": 0, "updates": 0,
               "config": config, "parameters": parameters, "output_dir": str(root)}
    history, updates, completed = [], 0, 0
    best_loss = float("inf")
    selected_device = torch.device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cuda_devices = [] if selected_device.type != "cuda" else [selected_device.index or 0]
    # The caller runs one trajectory at a time; restore its process-wide flags.
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_benchmark, cudnn_deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            network = make_tabm_network(x_train.shape[1], [], parameters).to(selected_device)
            summary["parameter_count"] = sum(item.numel() for item in network.parameters())
            optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"],
                                          weight_decay=parameters["weight_decay"])
            generator = torch.Generator().manual_seed(int(seed))
            try:
                for epoch in range(1, epochs + 1):
                    if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                        summary["status"] = "timeout"
                        break
                    epoch_start = time.perf_counter()
                    path_loss, observed, norms, clipped = 0.0, 0, [], 0
                    network.train()
                    for index in torch.randperm(len(x_train), generator=generator).split(batch_size):
                        if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                            summary["status"] = "timeout"
                            break
                        rows = index.numpy()
                        features = torch.from_numpy(x_train[rows]).to(selected_device)
                        targets = torch.from_numpy(y_train[rows]).to(selected_device)
                        optimizer.zero_grad(set_to_none=True)
                        logits = network(features, None)
                        loss = tabm_head_loss(logits, targets)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite training loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0,
                                                             error_if_nonfinite=False)
                        if not torch.isfinite(norm):
                            raise FloatingPointError("Non-finite gradient norm")
                        norm_value = float(norm)
                        norms.append(norm_value)
                        clipped += int(norm_value > 1.0)
                        optimizer.step()
                        updates += 1
                        path_loss += float(loss.detach()) * len(rows)
                        observed += len(rows)
                    if summary["status"] == "timeout":
                        break
                    train_result = _evaluate(network, x_train, y_train, device=selected_device,
                                             batch_size=batch_size)
                    validation_result = None if fixed else _evaluate(
                        network, x_val, y_val, device=selected_device, batch_size=batch_size)
                    completed = epoch
                    record = {"epoch": epoch, "updates": updates,
                              "path_mean_head_bce": path_loss / observed,
                              "path_loss_is_endpoint_loss": False,
                              "train_head_bce": train_result["head_bce"],
                              "train_ensemble_bce": train_result["ensemble_bce"],
                              "gradient_norm_mean_before_clip": float(np.mean(norms)),
                              "gradient_norm_max_before_clip": max(norms),
                              "clipped_updates": clipped,
                              "epoch_seconds": time.perf_counter() - epoch_start,
                              "elapsed_seconds": time.perf_counter() - start}
                    if validation_result is not None:
                        record.update({"validation_head_bce": validation_result["head_bce"],
                                       "validation_ensemble_bce": validation_result["ensemble_bce"]})
                    history.append(record)
                    if validation_result is not None and validation_result["ensemble_bce"] < best_loss:
                        best_loss = validation_result["ensemble_bce"]
                        summary["best_epoch"] = epoch
                        summary["best_checkpoint"] = _save_endpoint(
                            root, "best", network, parameters, config, epoch, updates,
                            train_result, validation_result)
                    if epoch in checkpoint_epochs:
                        name = "model" if fixed and epoch == epochs else f"epoch{epoch:04d}"
                        summary["checkpoints"][str(epoch)] = _save_endpoint(
                            root, name, network, parameters, config, epoch, updates,
                            train_result, validation_result)
                    _write_json(root / "history.json", history)
                if completed == epochs:
                    summary["status"] = "complete"
                    if fixed:
                        summary["model_path"] = summary["checkpoints"][str(epochs)]["model_path"]
                elif summary["status"] == "running":
                    summary["status"] = "incomplete"
            except FloatingPointError as error:
                summary.update(status="nonfinite", error=str(error))
            if summary["status"] != "complete":
                interrupted = root / "interrupted.pt"
                _save_checkpoint(interrupted, network, parameters, config, epoch=completed,
                                 updates=updates, complete_epoch=False)
                summary["interrupted_model_path"] = str(interrupted)
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
    summary.update(epochs_completed=completed, updates=updates,
                   elapsed_seconds=time.perf_counter() - start,
                   history_path=str(root / "history.json"), score_contract=SCORE_CONTRACT)
    _write_json(root / "history.json", history)
    _write_json(root / "summary.json", summary)
    return summary


def train_trajectory(x_train, y_train, x_val, y_val, *, width, learning_rate, seed,
                     output_dir, device="cuda:0", epochs=200, checkpoints=(50, 200),
                     batch_size=256, k=32, n_blocks=3, weight_decay=3e-4, dropout=.1,
                     max_seconds=None):
    """Run the whole budget; validation selects a saved epoch without early stopping."""
    return _train(x_train, y_train, x_val, y_val, width=width, learning_rate=learning_rate,
                  seed=seed, output_dir=output_dir, device=device, epochs=epochs,
                  checkpoints=checkpoints, batch_size=batch_size, k=k, n_blocks=n_blocks,
                  weight_decay=weight_decay, dropout=dropout, max_seconds=max_seconds, fixed=False)


def train_fixed(x, y, *, width, learning_rate, seed, output_dir, device="cuda:0", epochs,
                batch_size=256, k=32, n_blocks=3, weight_decay=3e-4, dropout=.1,
                max_seconds=None):
    """Refit from the seed for an externally selected epoch count, without validation."""
    return _train(x, y, None, None, width=width, learning_rate=learning_rate, seed=seed,
                  output_dir=output_dir, device=device, epochs=epochs, checkpoints=(epochs,),
                  batch_size=batch_size, k=k, n_blocks=n_blocks, weight_decay=weight_decay,
                  dropout=dropout, max_seconds=max_seconds, fixed=True)


@dataclass
class TabMPredictor:
    network: object
    n_features: int
    device: object
    metadata: dict

    def predict_scores(self, matrix, batch_size=128):
        import torch

        values = _matrix(matrix, width=self.n_features, allow_empty=True)
        batch_size = _positive_integer(batch_size, "batch_size")
        if not len(values):
            return np.empty(0, dtype=np.float64)
        self.network.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                logits = self.network(torch.from_numpy(values[start:start + batch_size]).to(self.device), None)
                result.append(stable_scores(logits).cpu().numpy())
        scores = np.concatenate(result)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite prediction score")
        return scores


def load_predictor(path, device="cpu"):
    import torch

    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (artifact.get("artifact_version") != ARTIFACT_VERSION
            or artifact.get("score_contract") != SCORE_CONTRACT
            or not artifact.get("complete_epoch")):
        raise ValueError("Unsupported or incomplete TabM checkpoint")
    parameters = _parameters(artifact["parameters"])
    n_features = _positive_integer(artifact["config"]["n_features"], "n_features")
    selected_device = torch.device(device)
    network = make_tabm_network(n_features, [], parameters)
    network.load_state_dict(artifact["state_dict"], strict=True)
    network.to(selected_device).eval()
    return TabMPredictor(network, n_features, selected_device,
                         {key: value for key, value in artifact.items() if key != "state_dict"})
