"""Fixed terminal TabM trajectories using only F; no selection or held-out I/O."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ids.flow_history.model import _device, _sync
from ids.recent_baselines import _numeric_bins, _parameters, make_tabm_network, tabm_head_loss
from scripts.calibration_repair.data import Preprocessor, build_groups, file_digest, json_digest
from scripts.calibration_repair.objective import backward_objective, score_full, score_gradient


FORMAT = "calibration-reference-terminal-v2"
ARMS = ("A_BCE", "G_group_tail", "R_reference_envelope")
FIXED_TRAINING = {
    "optimizer_steps": 15600, "batch_size": 256, "draws_per_class": 128,
    "score_batch_size": 1024, "report_every": 100, "learning_rate": .002,
    "weight_decay": .0003, "gradient_clip_norm": 1., "transfer_weight": 1.,
}
FIXED_MODEL = {
    "epochs": 1, "patience": 1, "batch_size": 256, "k": 32,
    "n_blocks": 2, "d_block": 512, "embedding_dim": 16, "n_bins": 48,
    "learning_rate": .002, "weight_decay": .0003, "dropout": .1,
    "num_embeddings": "piecewise_linear", "arch_type": "tabm",
}


@contextmanager
def scoring_arithmetic():
    """Match the arithmetic used by the exact transfer objective after reload."""
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[1:3]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[3:5]


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf8")


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _derived_seed(seed, stream):
    return int.from_bytes(hashlib.sha256(f"calibration-transfer:{seed}:{stream}".encode()).digest()[:4], "little")


def _seed(seed, device):
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _rng_digest(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state_digest(state)


class ClassSampler:
    """Independent class-shuffled cyclic queues; no source weights or resampling grid."""

    def __init__(self, labels, *, seed, per_class):
        labels = np.asarray(labels)
        if labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
            raise ValueError("Sampler requires both binary F classes")
        if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
            raise ValueError("per_class must be a positive integer")
        self.per_class = per_class
        self.groups = {label: np.flatnonzero(labels == label) for label in (0, 1)}
        self.seeds = {label: _derived_seed(seed, f"sampler-class-{label}") for label in (0, 1)}
        self.rngs = {label: np.random.Generator(np.random.PCG64(self.seeds[label])) for label in (0, 1)}
        self.queues = {label: self.rngs[label].permutation(self.groups[label]) for label in (0, 1)}
        self.cursors = {label: 0 for label in (0, 1)}
        self.cycles = {label: 0 for label in (0, 1)}
        self.seen = np.zeros(len(labels), dtype=bool)
        self.draws = 0
        self.chain = hashlib.sha256()

    def next(self):
        blocks = []
        for label in (0, 1):
            needed = self.per_class
            pieces = []
            while needed:
                cursor, queue = self.cursors[label], self.queues[label]
                if cursor == len(queue):
                    self.cycles[label] += 1
                    queue = self.rngs[label].permutation(self.groups[label])
                    self.queues[label], cursor = queue, 0
                take = min(needed, len(queue) - cursor)
                pieces.append(queue[cursor:cursor + take])
                self.cursors[label] = cursor + take
                needed -= take
            blocks.append(np.concatenate(pieces))
        result = np.concatenate(blocks).astype(np.int64, copy=False)
        self.seen[result] = True
        self.draws += len(result)
        self.chain.update(result.astype("<i8", copy=False).tobytes())
        return result

    def audit(self):
        return {"draw_sha256": self.chain.hexdigest(), "total_draws": self.draws,
                "seeds": {str(k): v for k, v in self.seeds.items()},
                "classes": {str(label): {"rows": len(self.groups[label]),
                                         "draws": self.draws // 2,
                                         "unique_drawn": int(self.seen[self.groups[label]].sum()),
                                         "completed_cycles": self.cycles[label] +
                                         int(self.cursors[label] == len(self.queues[label])),
                                         "cursor": self.cursors[label]}
                            for label in (0, 1)}}


def train_loop(network, numeric, categorical, labels, groups, *, arm, seed,
               training_spec, device, output_dir=None):
    """Core used by production fit and tiny synthetic tests; never reads a file.

    Production ``fit`` validates the immutable 15,600-step specification. Tests
    call this core with fresh tiny networks and synthetic arrays only.
    """
    if arm not in ARMS:
        raise ValueError("Unknown calibration-transfer arm")
    spec = training_spec
    if spec["batch_size"] != 2 * spec["draws_per_class"]:
        raise ValueError("BCE batch requires equal normal/attack draws")
    if (spec["optimizer_steps"] < 1 or spec["report_every"] < 1
            or spec["transfer_weight"] != 1.):
        raise ValueError("Expected positive fixed budget and unit transfer weight")
    device = torch.device(device)
    labels = np.asarray(labels, dtype=np.float32)
    sampler = ClassSampler(labels, seed=seed, per_class=spec["draws_per_class"])
    dropout_seed = _derived_seed(seed, "dropout")
    _seed(dropout_seed, device)
    initial_rng = _rng_digest(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=spec["learning_rate"],
                                  weight_decay=spec["weight_decay"])
    x, cat, target = (torch.from_numpy(value).to(device) for value in (numeric, categorical, labels))
    costs = {"bce_forward_rows": 0, "bce_backward_rows": 0,
             "transfer_discovery_forward_rows": 0, "transfer_replay_forward_rows": 0,
             "transfer_replay_backward_rows": 0, "transfer_replay_blocks": 0,
             "bce_seconds": 0., "transfer_seconds": 0., "optimizer_seconds": 0.}
    history = []
    window = {"bce": 0., "transfer": 0., "gradient_norm": 0., "max_gradient_norm": 0.,
              "clipped_steps": 0, "steps": 0}
    last_transfer = None
    _sync(device)
    started = heartbeat = time.perf_counter()
    for step in range(1, spec["optimizer_steps"] + 1):
        network.train()
        indices = torch.from_numpy(sampler.next()).to(device)
        optimizer.zero_grad(set_to_none=True)
        segment = time.perf_counter()
        logits = network(x[indices], cat[indices])
        if logits.dtype != torch.float32:
            raise ValueError("Training requires FP32 member logits")
        bce = tabm_head_loss(logits, target[indices])
        if not bool(torch.isfinite(bce)):
            raise FloatingPointError("Nonfinite BCE training loss")
        bce.backward()
        _sync(device)
        costs["bce_seconds"] += time.perf_counter() - segment
        costs["bce_forward_rows"] += len(indices)
        costs["bce_backward_rows"] += len(indices)
        if arm != "A_BCE":
            segment = time.perf_counter()
            last_transfer = backward_objective(network, numeric, categorical, groups, arm=arm,
                                              device=device, batch_size=spec["score_batch_size"],
                                              weight=spec["transfer_weight"])
            _sync(device)
            costs["transfer_seconds"] += time.perf_counter() - segment
            extra = last_transfer["cost"]
            costs["transfer_discovery_forward_rows"] += extra["discovery_forward_rows_this_call"]
            costs["transfer_replay_forward_rows"] += extra["replay_forward_rows"]
            costs["transfer_replay_backward_rows"] += extra["replay_backward_rows"]
            costs["transfer_replay_blocks"] += extra["replay_blocks"]
        segment = time.perf_counter()
        norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), spec["gradient_clip_norm"],
                                                   error_if_nonfinite=True))
        optimizer.step()
        _sync(device)
        costs["optimizer_seconds"] += time.perf_counter() - segment
        window["bce"] += float(bce.detach())
        window["transfer"] += 0. if last_transfer is None else last_transfer["penalty"]
        window["gradient_norm"] += norm
        window["max_gradient_norm"] = max(window["max_gradient_norm"], norm)
        window["clipped_steps"] += int(norm > spec["gradient_clip_norm"])
        window["steps"] += 1
        now = time.perf_counter()
        report_due = step % spec["report_every"] == 0 or step == spec["optimizer_steps"]
        if report_due:
            record = {"step": step, "elapsed_seconds": now - started,
                      "window_steps": window["steps"],
                      "mean_path_head_bce": window["bce"] / window["steps"],
                      "mean_path_transfer_penalty": window["transfer"] / window["steps"],
                      "preclip_mean_gradient_norm": window["gradient_norm"] / window["steps"],
                      "preclip_max_gradient_norm": window["max_gradient_norm"],
                      "clipped_steps": window["clipped_steps"],
                      "last_transfer_at_parameter_step": step - 1 if last_transfer else None,
                      "last_pre_update_transfer": last_transfer,
                      "sampling": sampler.audit(), "dropout_rng_sha256": _rng_digest(device)}
            history.append(record)
            if output_dir is not None:
                _write(Path(output_dir) / "history.json", history)
            window = {key: 0 if key in ("clipped_steps", "steps") else 0. for key in window}
        if report_due or step == 1 or now - heartbeat >= 30.:
            print(json.dumps({"arm": arm, "seed": seed, "step": step,
                              "optimizer_steps": spec["optimizer_steps"],
                              "elapsed_seconds": round(now - started, 3)}, separators=(",", ":")), flush=True)
            heartbeat = now
    _sync(device)
    costs["optimizer_loop_seconds"] = time.perf_counter() - started
    return {"history": history, "sampling": sampler.audit(), "cost": costs,
            "optimizer_steps": step, "dropout_seed": dropout_seed,
            "initial_dropout_rng_sha256": initial_rng, "final_dropout_rng_sha256": _rng_digest(device)}


def fit(rows, preprocessor, output_dir, *, arm, seed, training_spec, device="cuda:0", threads=4):
    """One immutable production fit; failures propagate and are never retried."""
    if training_spec != FIXED_TRAINING or arm not in ARMS or seed not in (17, 29, 43):
        raise ValueError("Production arm, seed or training budget differs from registration")
    if (len(rows) != 13252 or [r["flow_id"] for r in rows] != sorted(r["flow_id"] for r in rows)
            or any(r["dataset_role"] not in ("F_base", "F_expansion") for r in rows)):
        raise ValueError("Production fitting requires canonical registered F only")
    labels = np.asarray([r["label"] for r in rows], dtype=np.float32)
    if (int((labels == 0).sum()), int((labels == 1).sum())) != (12747, 505):
        raise ValueError("Production F class counts differ")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Training output must be empty; partial trajectories are preserved")
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "attempt.json", {"arm": arm, "seed": seed, "status": "started",
                                   "training_spec": training_spec, "fit_calls": 1,
                                   "checkpoint_rule": "terminal_step_only"})
    started = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark,
                torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    actual = _device(device, threads)
    if actual.type != "cuda":
        raise ValueError("Production fit is laboratory CUDA only; synthetic tests use train_loop")
    numeric, categorical = preprocessor.transform(rows)
    if numeric.shape != (13252, 5) or categorical.shape != (13252, 8):
        raise ValueError("Production encoding differs from the fixed raw8/missingness contract")
    parameters = _parameters(FIXED_MODEL)
    bins, constants = _numeric_bins(numeric, parameters)
    groups = build_groups(rows, Path(__file__).resolve().parents[2])
    preprocessing = preprocessor.to_dict()
    pairing = {"preprocessing_sha256": json_digest(preprocessing),
               "numeric_bins_sha256": json_digest(bins),
               "F_identity_sha256": json_digest([r["flow_id"] for r in rows])}
    _write(root / "preprocessing.json", preprocessing)
    _write(root / "numeric_bins.json", {"bins": bins, "constant_columns": constants})
    _write(root / "groups.json", {"groups": groups, "optimized": arm})
    torch.cuda.reset_peak_memory_stats(actual)
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.random.fork_rng(devices=[actual.index]):
            _seed(seed, actual)
            network = make_tabm_network(5, preprocessor.cat_cardinalities, parameters,
                                       numeric_bins=bins).to(actual)
            initial = state_digest(network.state_dict())
            result = train_loop(network, numeric, categorical, labels, groups, arm=arm, seed=seed,
                                training_spec=training_spec, device=actual, output_dir=root)
            # One final F-only score pass diagnoses both objectives for every arm.
            _sync(actual)
            final_started = time.perf_counter()
            final = score_full(network, numeric, categorical, device=actual,
                               batch_size=training_spec["score_batch_size"])
            reports = {name: score_gradient(final.scores, groups, name)[1]
                       for name in ("G_group_tail", "R_reference_envelope")}
            _sync(actual)
            result["cost"].update(final_F_forward_rows=len(rows),
                                   final_F_diagnostic_seconds=time.perf_counter() - final_started,
                                   cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(actual),
                                   cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(actual))
            state = {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}
            checkpoint = {"format": FORMAT, "arm": arm, "seed": seed,
                          "selected_step": result["optimizer_steps"], "final_refits": 0,
                          "checkpoint_rule": "terminal_step_only", "training_spec": training_spec,
                          "parameters": parameters, "preprocessing": preprocessing,
                          "numeric_bins": bins, "constant_bin_columns": constants,
                          "state_dict": state, "state_sha256": state_digest(state),
                          "score_batch_size": training_spec["score_batch_size"],
                          "initial_state_sha256": initial, **pairing,
                          "score_contract": "FP32 members; FP64 stable log-odds of mean sigmoid probabilities"}
            torch.save(checkpoint, root / "model.pt")
            np.savez_compressed(root / "F_terminal_scores.npz", scores=final.scores.numpy())
            _write(root / "F_score_identities.json", [r["flow_id"] for r in rows])
            _write(root / "F_reference_diagnostics.json", reports)
            parameters_count = sum(p.numel() for p in network.parameters())
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[1:3]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[3:5]
    result.pop("history")
    summary = {"status": "complete", "arm": arm, "seed": seed, "fit_calls": 1,
               "model_constructions": 1, "final_refits": 0, "fit_rows": len(rows),
               "checkpoint_rule": "terminal_step_only", "selected_step": result["optimizer_steps"],
               "artifact_path": str(root / "model.pt"), "artifact_sha256": file_digest(root / "model.pt"),
               "artifact_bytes": (root / "model.pt").stat().st_size, "parameters": parameters_count,
               "initial_state_sha256": initial, **pairing, **result,
               "sampler_draw_sha256": result["sampling"]["draw_sha256"],
               "F_terminal_reference_diagnostics": reports,
               "total_seconds": time.perf_counter() - started}
    _write(root / "summary.json", summary)
    return summary


@dataclass
class Predictor:
    network: object
    preprocessor: Preprocessor
    device: object
    batch_size: int

    def score(self, rows):
        numeric, categorical = self.preprocessor.transform(rows)
        with scoring_arithmetic():
            return score_full(self.network, numeric, categorical, device=self.device,
                              batch_size=self.batch_size).scores.numpy()


def load_predictor(artifact, *, device="cuda:0", threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if (state.get("format") != FORMAT or state.get("arm") not in ARMS
            or state.get("training_spec") != FIXED_TRAINING
            or state.get("parameters") != _parameters(FIXED_MODEL)
            or state.get("selected_step") != FIXED_TRAINING["optimizer_steps"]
            or state.get("checkpoint_rule") != "terminal_step_only" or state.get("final_refits") != 0):
        raise ValueError("Unsupported or nonterminal calibration-transfer artifact")
    preprocessor = Preprocessor.from_dict(state["preprocessing"])
    if (state_digest(state["state_dict"]) != state["state_sha256"]
            or json_digest(state["preprocessing"]) != state["preprocessing_sha256"]
            or json_digest(state["numeric_bins"]) != state["numeric_bins_sha256"]
            or state.get("score_batch_size") != FIXED_TRAINING["score_batch_size"]):
        raise ValueError("Artifact state or fitted preprocessing binding changed")
    actual = _device(device, threads)
    with torch.random.fork_rng(devices=[]):
        network = make_tabm_network(5, preprocessor.cat_cardinalities, state["parameters"],
                                   numeric_bins=state["numeric_bins"])
    network.load_state_dict(state["state_dict"])
    return Predictor(network.to(actual).eval(), preprocessor, actual, state["score_batch_size"])
