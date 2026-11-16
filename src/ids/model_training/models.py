"""Bounded author-TabM/HGB trajectories and one selected-stage P refit.

Only F/V are accepted by the selection API. Calibration and evaluation are
owned by the runner. Existing nonempty directories are never fitted again.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .preprocessing import TypedPreprocessor


FORMAT = "model-training-c2-v1"
ARMS = {"hgb", "tabm_ple", "tabm_no_ple"}


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _labels(frame):
    labels = frame["label"].to_numpy()
    if not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("F, V and P must each contain both binary classes")
    return np.asarray(labels, dtype=np.float32)


def _nll(scores, labels):
    values = np.logaddexp(0.0, np.where(labels == 1, -scores, scores))
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite endpoint NLL")
    return float(values.mean())


def selection_metrics(scores, labels, *, ranking_values=None):
    scores = np.asarray(scores, dtype=np.float64)
    ranking = scores if ranking_values is None else np.asarray(ranking_values, dtype=np.float64)
    if scores.shape != labels.shape or ranking.shape != labels.shape or not (
            np.isfinite(scores).all() and np.isfinite(ranking).all()):
        raise ValueError("Expected finite row-aligned validation scores")
    return {"pauc": float(roc_auc_score(labels, ranking, max_fpr=0.01)), "nll": _nll(scores, labels)}


def selection_key(metrics, stage):
    return metrics["pauc"], -metrics["nll"], -int(stage)


def _prepare(frame, root, design):
    preprocessor = TypedPreprocessor.fit(frame, design=design)
    _write(root / "preprocessing.json", preprocessor.to_dict())
    return preprocessor


def _scores_file(root, name, train, validation=None):
    values = {"train_scores": np.asarray(train, dtype=np.float64)}
    if validation is not None:
        values["validation_scores"] = np.asarray(validation, dtype=np.float64)
    path = root / (name + ".npz")
    np.savez_compressed(path, **values)
    return str(path)


@dataclass
class HGBPredictor:
    estimator: object
    preprocessor: TypedPreprocessor
    stage: int
    threads: int

    def score(self, frame):
        if not len(frame):
            return np.empty(0, dtype=np.float64)
        matrix = self.preprocessor.hgb_matrix(frame)
        with threadpool_limits(limits=self.threads):
            if self.stage == self.estimator.n_iter_:
                scores = self.estimator.decision_function(matrix)
            else:
                for stage, scores in enumerate(self.estimator.staged_decision_function(matrix), 1):
                    if stage == self.stage:
                        break
        result = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(result) != len(frame) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid HGB scores")
        return result


@dataclass
class TabMPredictor:
    network: object
    preprocessor: TypedPreprocessor
    device: object
    batch_size: int

    def score(self, frame):
        if not len(frame):
            return np.empty(0, dtype=np.float64)
        numeric, categorical = self.preprocessor.transform(frame)
        return _tabm_endpoint(self.network, numeric, categorical, None,
                              self.device, self.batch_size)["scores"]


def _tabm_endpoint(network, numeric, categorical, labels, device, batch_size):
    import torch
    from ids.recent_baselines import tabm_head_loss
    from ids.route_batch.tabm_training import stable_scores

    network.eval()
    head_loss, chunks = 0.0, []
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            logits = network(torch.from_numpy(numeric[start:start + batch_size]).to(device),
                             torch.from_numpy(categorical[start:start + batch_size]).to(device))
            score = stable_scores(logits)
            if not torch.isfinite(score).all().item():
                raise FloatingPointError("Nonfinite TabM endpoint score")
            chunks.append(score.cpu().numpy())
            if labels is not None:
                targets = torch.from_numpy(labels[start:start + batch_size]).to(device)
                head_loss += float(tabm_head_loss(logits, targets)) * len(targets)
    scores = np.concatenate(chunks).astype(np.float64)
    result = {"scores": scores}
    if labels is not None:
        result.update(head_bce=head_loss / len(labels), ensemble_bce=_nll(scores, labels))
    return result


def _embedding_audit(network, numeric, categorical, device, batch_size):
    import torch

    if network.num_module is None:
        return {"numeric_embedding": "none", "unique_inputs_after_embedding":
                int(len(np.unique(np.column_stack([numeric, categorical]), axis=0)))}
    chunks = []
    network.eval()
    with torch.inference_mode():
        for block in np.array_split(numeric, max(1, (len(numeric) + batch_size - 1) // batch_size)):
            output = network.num_module(torch.from_numpy(block).to(device)).flatten(1)
            if not torch.isfinite(output).all().item():
                raise FloatingPointError("Nonfinite numerical embedding")
            chunks.append(output.cpu().numpy())
    values = np.column_stack([np.concatenate(chunks), categorical])
    return {"numeric_embedding": "piecewise_linear_B", "unique_inputs_after_embedding":
            int(len(np.unique(values, axis=0))), "scope": "selected fitted numerical module plus categorical identities"}


def _hgb(frame, validation, root, *, seed, threads, design, stage):
    started = time.perf_counter()
    preprocessor = _prepare(frame, root, design)
    x, y = preprocessor.hgb_matrix(frame), _labels(frame)
    spec = design["training"]["hgb"]
    budget = _integer(spec["max_iter"] if stage is None else stage, "HGB stages")
    estimator = HistGradientBoostingClassifier(max_iter=budget, max_leaf_nodes=spec["max_leaf_nodes"],
        learning_rate=spec["learning_rate"], random_state=int(seed), early_stopping=False)
    fit_started = time.perf_counter()
    with threadpool_limits(limits=threads):
        estimator.fit(x, y)  # Natural row weights; no internal validation or automatic refit.
    fit_seconds = time.perf_counter() - fit_started
    history, best_key, selected, selected_scores = [], None, budget, None
    validation_audit = None
    with threadpool_limits(limits=threads):
        train_stages = estimator.staged_decision_function(x)
        if validation is not None:
            vx, vy = preprocessor.hgb_matrix(validation), _labels(validation)
            validation_audit = preprocessor.audit(validation)
            streams = zip(train_stages, estimator.staged_decision_function(vx), estimator.staged_predict_proba(vx))
        else:
            streams = ((train, None, None) for train in train_stages)
        for number, (train, val, probabilities) in enumerate(streams, 1):
            record = {"stage": number, "updates": number, "update_unit": "boosting_stage",
                      "path_mean_head_bce": None, "path_loss_applicable": False,
                      "path_loss_is_endpoint_loss": False, "train_ensemble_bce": _nll(train, y)}
            if validation is not None:
                metric = selection_metrics(val, vy, ranking_values=probabilities[:, 1])
                record.update(validation_pauc=metric["pauc"], validation_ensemble_bce=metric["nll"])
                key = selection_key(metric, number)
                if best_key is None or key > best_key:
                    best_key, selected = key, number
                    selected_scores = (np.asarray(train).copy(), np.asarray(val).copy())
            else:
                selected_scores = (np.asarray(train).copy(), None)
            history.append(record)
    _write(root / "history.json", history)
    artifact = root / "model.joblib"
    joblib.dump({"format": FORMAT, "arm": "hgb", "preprocessing": preprocessor.to_dict(),
                 "estimator": estimator, "selected_stage": selected,
                 "score_contract": "HGB decision_function at selected stage; float64",
                 "fit_rows": len(frame), "seed": int(seed)}, artifact)
    scores_path = _scores_file(root, "selected_scores" if validation is not None else "train_scores", *selected_scores)
    _write(root / "input_audit.json", {"train": preprocessor.audit(frame), "validation": validation_audit})
    return {"status": "complete", "selected_stage": selected, "completed_stages": budget,
            "artifact_path": str(artifact), "history_path": str(root / "history.json"),
            "train_scores_path": scores_path, "fit_rows": len(frame), "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - started, "model_artifact_bytes": artifact.stat().st_size,
            "cuda_peak_allocated_bytes": None, "fit_calls": 1,
            "trajectory_artifact_keeps_full_estimator": validation is not None,
            "selection": None if best_key is None else {"pauc": best_key[0], "nll": -best_key[1]}}


def _tabm(frame, validation, root, *, arm, seed, threads, device, design, stage):
    import torch
    from ids.flow_history.model import _device, _sync
    from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network, tabm_head_loss

    started = time.perf_counter()
    preprocessor = _prepare(frame, root, design)
    numeric, categorical = preprocessor.transform(frame)
    labels = _labels(frame)
    spec = design["training"]["tabm"]
    budget = _integer(spec["max_epochs"] if stage is None else stage, "TabM epochs")
    parameters = _parameters({"epochs": budget, "patience": spec["patience"], "batch_size": spec["batch"],
        "k": spec["k"], "n_blocks": spec["n_blocks"], "d_block": spec["d_block"],
        "embedding_dim": spec["embedding_dim"], "n_bins": spec["max_bins"],
        "learning_rate": spec["lr"], "weight_decay": spec["weight_decay"], "dropout": spec["dropout"],
        "num_embeddings": "piecewise_linear" if arm == "tabm_ple" else "none", "arch_type": spec["architecture"]})
    bins, constant_columns = _numeric_bins(numeric, parameters)
    actual = _device(device, threads)
    if actual.type == "cuda":
        torch.cuda.reset_peak_memory_stats(actual)
    validation_data = None if validation is None else (*preprocessor.transform(validation), _labels(validation))
    cuda_devices = [] if actual.type != "cuda" else [actual.index]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic)
    network = None
    history, updates, best_key, selected, best_state, selected_scores = [], 0, None, budget, None, None
    stale, fit_seconds, completed = 0, 0.0, 0
    artifact = root / "model.pt"

    def save(path, state, selected_stage):
        torch.save({"format": FORMAT, "arm": arm, "preprocessing": preprocessor.to_dict(),
                    "parameters": parameters, "numeric_bins": bins, "constant_bin_columns": constant_columns,
                    "state_dict": state, "selected_stage": selected_stage, "fit_rows": len(frame),
                    "seed": int(seed), "score_batch_size": parameters["batch_size"],
                    "score_contract": "FP32 member logits; FP64 log odds of mean probabilities; no clipping"}, path)

    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(seed))
            network = make_tabm_network(5, preprocessor.cat_cardinalities, parameters, numeric_bins=bins).to(actual)
            optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"],
                                          weight_decay=parameters["weight_decay"])
            sampler = torch.Generator().manual_seed(int(seed))
            x = torch.from_numpy(numeric).to(actual)
            cat = torch.from_numpy(categorical).to(actual)
            y = torch.from_numpy(labels).to(actual)
            for epoch in range(1, budget + 1):
                network.train()
                _sync(actual)
                epoch_started = time.perf_counter()
                path_loss, observed = 0.0, 0
                for indices in torch.randperm(len(frame), generator=sampler).split(parameters["batch_size"]):
                    indices = indices.to(actual)
                    optimizer.zero_grad(set_to_none=True)
                    loss = tabm_head_loss(network(x[indices], cat[indices]), y[indices])
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), spec["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    path_loss += float(loss.detach()) * len(indices)
                    observed += len(indices)
                _sync(actual)
                seconds = time.perf_counter() - epoch_started
                fit_seconds += seconds
                train_result = _tabm_endpoint(network, numeric, categorical, labels, actual, parameters["batch_size"])
                val_result = None if validation_data is None else _tabm_endpoint(
                    network, *validation_data, actual, parameters["batch_size"])
                record = {"stage": epoch, "updates": updates, "update_unit": "optimizer_step",
                    "path_mean_head_bce": path_loss / observed, "path_loss_applicable": True,
                    "path_loss_is_endpoint_loss": False, "train_head_bce": train_result["head_bce"],
                    "train_ensemble_bce": train_result["ensemble_bce"], "fit_seconds": seconds}
                if val_result is not None:
                    metric = selection_metrics(val_result["scores"], validation_data[2])
                    record.update(validation_pauc=metric["pauc"], validation_ensemble_bce=metric["nll"],
                                  validation_head_bce=val_result["head_bce"])
                    key = selection_key(metric, epoch)
                    if best_key is None or key > best_key:
                        best_key, selected, stale = key, epoch, 0
                        best_state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
                        selected_scores = (train_result["scores"], val_result["scores"])
                        save(root / "best.pt", best_state, epoch)
                    else:
                        stale += 1
                else:
                    selected_scores = (train_result["scores"], None)
                completed = epoch
                history.append(record)
                _write(root / "history.json", history)
                if validation_data is not None and stale >= parameters["patience"]:
                    break
            if best_state is not None:
                network.load_state_dict(best_state)
            state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            save(artifact, state, selected)
            train_audit = preprocessor.audit(frame)
            train_audit.update(_embedding_audit(network, numeric, categorical, actual, parameters["batch_size"]))
            validation_audit = None if validation is None else preprocessor.audit(validation)
            if validation_audit is not None:
                validation_audit.update(_embedding_audit(network, *validation_data[:2], actual, parameters["batch_size"]))
            _write(root / "input_audit.json", {"train": train_audit, "validation": validation_audit,
                                              "numeric_bins": bins, "constant_bin_columns": constant_columns})
    except Exception:
        if network is not None:
            save(root / "interrupted.pt", {name: value.detach().cpu().clone()
                 for name, value in network.state_dict().items()}, completed)
        raise
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[1:]
    scores_path = _scores_file(root, "selected_scores" if validation is not None else "train_scores", *selected_scores)
    return {"status": "complete", "selected_stage": selected, "completed_stages": completed,
        "stopped_early": completed < budget, "artifact_path": str(artifact),
        "history_path": str(root / "history.json"), "train_scores_path": scores_path, "fit_rows": len(frame),
        "fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
        "parameters": sum(parameter.numel() for parameter in network.parameters()),
        "model_artifact_bytes": artifact.stat().st_size,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(actual) if actual.type == "cuda" else None,
        "fit_calls": 1, "selection": None if best_key is None else {"pauc": best_key[0], "nll": -best_key[1]}}


def _fit(frame, validation, output_dir, *, arm, seed, threads, device, design, stage):
    if arm not in ARMS:
        raise ValueError(f"Unknown model arm: {arm}")
    threads = _integer(threads, "threads")
    if seed is None and arm == "hgb":
        seed = design["training"]["hgb"]["random_state"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a uint32 integer")
    training = design["training"]
    if training["row_weight"] != 1 or training["source_mixing"] or training["normal_subsampling"]:
        raise ValueError("This trainer only implements natural row weights without source mixing or subsampling")
    budget = training["hgb"]["max_iter"] if arm == "hgb" else training["tabm"]["max_epochs"]
    if stage is not None and _integer(stage, "selected stage") > budget:
        raise ValueError("Selected stage exceeds the registered budget")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Training output directory must be empty; incomplete fits are not automatically retried")
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "attempt.json", {"status": "started", "arm": arm, "seed": int(seed),
        "phase": "trajectory" if validation is not None else "final_refit", "requested_stage": stage,
        "fit_rows": len(frame), "validation_rows": 0 if validation is None else len(validation),
        "natural_row_weights": True, "design": copy.deepcopy(design)})
    try:
        result = (_hgb(frame, validation, root, seed=seed, threads=threads, design=design, stage=stage)
                  if arm == "hgb" else _tabm(frame, validation, root, arm=arm, seed=seed, threads=threads,
                                             device=device, design=design, stage=stage))
    except Exception as error:
        result = {"status": "failed", "arm": arm, "selected_stage": None, "artifact_path": None,
                  "error_type": type(error).__name__, "error": str(error), "retry_allowed": False}
        _write(root / "failure.json", {**result, "traceback": traceback.format_exc()})
    _write(root / "summary.json", result)
    return result


def fit_trajectory(frameF, frameV, output_dir, *, arm, seed, threads, device, design):
    if frameV is None:
        raise ValueError("A development partition is required to select a stage")
    return _fit(frameF, frameV, output_dir, arm=arm, seed=seed, threads=threads, device=device,
                design=design, stage=None)


def fit_final(frameP, output_dir, *, arm, seed, stage, threads, device, design):
    _integer(stage, "selected stage")
    return _fit(frameP, None, output_dir, arm=arm, seed=seed, threads=threads, device=device,
                design=design, stage=stage)


def load_predictor(path, device="cpu", threads=1):
    path = Path(path)
    threads = _integer(threads, "threads")
    if path.suffix == ".joblib":
        state = joblib.load(path)
        if state.get("format") != FORMAT or state.get("arm") != "hgb":
            raise ValueError("Unsupported HGB artifact")
        estimator = state["estimator"]
        stage = _integer(state["selected_stage"], "stored stage")
        if stage > estimator.n_iter_ or not np.array_equal(estimator.classes_, [0, 1]):
            raise ValueError("Invalid HGB stage or class order")
        return HGBPredictor(estimator, TypedPreprocessor.from_dict(state["preprocessing"]), stage, threads)
    import torch
    from ids.flow_history.model import _device
    from ids.recent_baselines import make_tabm_network

    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT or state.get("arm") not in {"tabm_ple", "tabm_no_ple"}:
        raise ValueError("Unsupported TabM artifact")
    preprocessor = TypedPreprocessor.from_dict(state["preprocessing"])
    actual = _device(device, threads)
    network = make_tabm_network(5, preprocessor.cat_cardinalities, state["parameters"],
                                numeric_bins=state["numeric_bins"])
    network.load_state_dict(state["state_dict"])
    return TabMPredictor(network.to(actual).eval(), preprocessor, actual, state["score_batch_size"])
