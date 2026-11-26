"""Two matched small-MLP arms and cached CPU inference for the frozen comparators."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch

from scripts.comprehensive_batch.training import load_predictor
from scripts.coverage_batch.materialize import RAW8_FIELDS


SEEDS = (17, 29, 43)
FIT_SPEC = {"hidden_width": 32, "epochs": 16, "batch_size": 256, "learning_rate": .002}


def network(width):
    return torch.nn.Sequential(
        torch.nn.Linear(width, FIT_SPEC["hidden_width"]),
        torch.nn.ReLU(),
        torch.nn.Linear(FIT_SPEC["hidden_width"], 1),
        torch.nn.Flatten(start_dim=0),
    )


def raw_fields(rows):
    values = np.asarray([[row["raw8"][f] for f in RAW8_FIELDS] for row in rows], dtype=np.float64)
    return values[:, 0], np.log1p(values[:, 3:])


def preprocessing(rows):
    protocol, numeric = raw_fields(rows)
    median = np.asarray([
        np.nanmedian(values) if not np.isnan(values).all() else 0.0 for values in numeric.T
    ])
    filled = np.where(np.isnan(numeric), median, numeric)
    scale = filled.std(axis=0)
    return {
        "protocols": sorted(np.unique(protocol[np.isfinite(protocol)]).tolist()),
        "median": median.tolist(), "mean": filled.mean(axis=0).tolist(),
        "scale": np.where(scale > 0, scale, 1.).tolist(),
        "port_policy": "zero_ports_excluded_from_input",
    }


def transform(rows, fitted):
    protocol, numeric = raw_fields(rows)
    protocols = np.asarray(fitted["protocols"])
    known = protocol[:, None] == protocols
    categories = np.column_stack((known, ~known.any(axis=1)))
    missing = np.isnan(numeric)
    filled = np.where(missing, fitted["median"], numeric)
    normalized = (filled - fitted["mean"]) / fitted["scale"]
    return np.column_stack((categories, normalized, missing)).astype(np.float32)


class MLPPredictor:
    def __init__(self, fitted, width, state):
        self.fitted = fitted
        with torch.random.fork_rng(devices=[]):
            self.model = network(width)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.inference_mode()
    def score(self, rows):
        return self.model(torch.from_numpy(transform(rows, self.fitted))).numpy().astype(np.float64)


def fit_mlp_pair(rows, incumbent_alerts, seed, directory):
    """Use identical raw rows, optimizer, order and weights except the fixed residual mask."""
    fitted = preprocessing(rows)
    inputs = torch.from_numpy(transform(rows, fitted))
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.float32)
    class_counts = torch.bincount(labels.to(torch.int64), minlength=2)
    class_weight = len(rows) / (2 * class_counts)
    class_weight = class_weight[labels.to(torch.int64)]
    mask = ~torch.tensor([incumbent_alerts[row["flow_id"]] for row in rows], dtype=torch.bool)
    predictors, histories = {}, {}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for kind, weights in (
        ("ordinary", class_weight), ("specialist", class_weight * mask),
    ):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = network(inputs.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=FIT_SPEC["learning_rate"])
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(FIT_SPEC["epochs"]):
            order = rng.permutation(len(rows))
            epoch_loss = 0.0
            batch_size = FIT_SPEC["batch_size"]
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                optimizer.zero_grad()
                losses = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(inputs[batch]), labels[batch], reduction="none",
                )
                # Keep both objectives on the ordinary arm's identical batch scale.
                loss = (losses * weights[batch]).sum() / class_weight[batch].sum()
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite fitting loss for {kind}, seed {seed}")
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
            history.append(epoch_loss / ((len(order) + batch_size - 1) // batch_size))
        path = directory / f"{kind}_s{seed}.pt"
        torch.save({
            "format": "tracelight-residual-mlp-v1", "spec": FIT_SPEC, "seed": seed,
            "kind": kind, "input_width": inputs.shape[1], "preprocessing": fitted,
            "state_dict": model.state_dict(),
        }, path)
        predictors[kind] = load_mlp(path)
        histories[kind] = {"training_loss_by_epoch": history, "model_file": path.name}
    return predictors, histories


def load_mlp(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != "tracelight-residual-mlp-v1" or state["spec"] != FIT_SPEC:
        raise ValueError("Unexpected residual model definition")
    return MLPPredictor(state["preprocessing"], state["input_width"], state["state_dict"])


def load_tabm(comparator):
    return load_predictor(io.BytesIO(comparator.model_bytes), device="cpu", threads=4)


class ResidualPredictor:
    def __init__(self, incumbent, branch, rule):
        self.incumbent, self.branch, self.rule = incumbent, branch, rule

    def score(self, rows):
        """CPU work from canonical rows through to the final cascade decision."""
        result = self.incumbent.score(rows) > self.rule["incumbent_threshold"]
        if self.rule["enabled"] and not result.all():
            subset = [row for row, alert in zip(rows, result, strict=True) if not alert]
            result[~result] = self.branch.score(subset) > self.rule["threshold"]
        return result
