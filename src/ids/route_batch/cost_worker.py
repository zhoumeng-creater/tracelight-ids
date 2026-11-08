"""Inference-only cost entry; callers launch this in a new serial CPU process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ids.common import read_json, sha256_file, write_json
from ids.context_cache.preprocessing import CSVSchema, labels, read_csv
from .benchmark import benchmark_predictor
from .preprocessing import Representation, raw_numeric


def _local_file(directory, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("deployment artifacts must use relative file names")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"deployment artifact is missing or outside its directory: {name}")
    return path


def _verified_file(directory, descriptor, field, hash_field):
    path = _local_file(directory, descriptor[field])
    if sha256_file(path) != descriptor[hash_field]:
        raise ValueError(f"deployment SHA-256 mismatch: {field}")
    return path


def _file_sizes(paths):
    sizes = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required deployment file is missing: {path}")
        sizes[str(path)] = path.stat().st_size
    return sizes


def _load_deployment(directory, descriptor, schema, query_batch_size):
    """Load hash-bound artifacts, returning raw-frame scoring and actual files."""
    kind = descriptor["kind"]
    common = [directory / "threshold.json", directory / "deployment.json"]
    if kind == "tabm":
        from .tabm_training import load_predictor
        model_path = _verified_file(directory, descriptor, "model_file", "model_sha256")
        preprocessing_path = _verified_file(directory, descriptor, "preprocessing_file", "preprocessing_sha256")
        representation = Representation.load(preprocessing_path)
        if representation.schema.to_dict() != schema.to_dict():
            raise ValueError("deployment preprocessing schema differs from registered input")
        model = load_predictor(model_path, device="cpu")

        def predict(frame):
            return model.predict_scores(representation.transform(frame), batch_size=query_batch_size)

        sizes = _file_sizes([*common, model_path, preprocessing_path])
        storage = "model + fitted preprocessing + threshold + deployment descriptor"
    elif kind == "foundation":
        from .foundation import load_foundation
        model_path = _verified_file(directory, descriptor, "model_file", "model_sha256")
        model = load_foundation(model_path, device="cpu")
        checkpoint_path = Path(model.metadata["checkpoint_path"])
        # The loader checks its hash; count the required file itself, not a cached size.
        sizes = _file_sizes([*common, model_path, checkpoint_path])

        def predict(frame):
            return model.predict_scores(raw_numeric(frame, schema), batch_size=query_batch_size)

        storage = "support/rebuild recipe + required external checkpoint + threshold + deployment descriptor"
    elif kind == "reference":
        from ids.context_cache.csv_workflow import CSVDetector
        reference = Path(descriptor["directory"])
        if not reference.is_absolute():
            raise ValueError("reference deployment directory must be absolute")
        model = CSVDetector.load(reference)

        def predict(frame):
            return model.predict_frame(frame)["score"].to_numpy(dtype=np.float64)

        sizes = _file_sizes(path for path in reference.iterdir() if path.is_file())
        storage = "all original full/QCOC deployment files, including its own preprocessing and threshold"
    else:
        raise ValueError(f"unsupported cost deployment kind: {kind}")

    def checked(frame):
        result = np.asarray(predict(frame), dtype=np.float64)
        if result.shape != (len(frame),) or not np.isfinite(result).all():
            raise ValueError("deployment returned nonfinite or misaligned scores")
        return result

    return checked, sizes, storage


def run_cost(config_path, deployment_dir) -> dict:
    """Measure one saved deployment without importing a training runner or P/C.

    ``ids.route_batch _cost-worker`` must invoke this in a newly started process.
    External model load/rebuild includes artifact verification and fitted
    preprocessing; interpreter startup, shared imports, and E parsing are
    excluded from that clock. Hot RSS includes this worker and the same E
    frame for every method, and contains no prior gradient-training state.
    """
    import torch

    config_path, directory = Path(config_path).resolve(), Path(deployment_dir).resolve()
    spec = json.loads(config_path.read_text(encoding="utf-8-sig"))
    settings = spec["settings"]
    if settings["threads"] != 1:
        raise ValueError("cost workers require CPU single-thread settings")
    schema = CSVSchema.from_dict(spec["schema"])
    evaluation = spec["files"]["evaluation"]
    evaluation_path = Path(evaluation["path"])
    if not evaluation_path.is_absolute():
        evaluation_path = config_path.parent / evaluation_path
    if sha256_file(evaluation_path) != evaluation["sha256"]:
        raise ValueError("registered E SHA-256 mismatch")
    frame = read_csv(evaluation_path, schema)
    if len(frame) != evaluation["rows"]:
        raise ValueError("registered E row count differs")
    if schema.id_column is None:
        raise ValueError("cost replay requires explicit sample IDs")
    replay_rows = min(128, len(frame))
    saved = pd.read_csv(directory / "evaluation.csv", nrows=replay_rows,
                        dtype={"row_id": str}, float_precision="round_trip")
    if (saved["row_id"].astype(str).tolist() != frame[schema.id_column].astype(str).iloc[:replay_rows].tolist()
            or not np.array_equal(saved["label"].to_numpy(), labels(frame.iloc[:replay_rows], schema))):
        raise ValueError("saved evaluation replay IDs or labels differ")
    expected_scores = saved["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(expected_scores).all():
        raise ValueError("saved evaluation replay scores must be finite")
    descriptor = read_json(directory / "deployment.json")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            started = time.perf_counter()
            predict, sizes, storage = _load_deployment(directory, descriptor, schema,
                                                       settings["query_batch_size"])
            load_seconds = time.perf_counter() - started
            if torch.get_num_threads() != 1:
                raise ValueError("loaded deployment changed the single-thread cost contract")
            replay = predict(frame.iloc[:replay_rows].loc[:, list(schema.features)])
            if not np.allclose(replay, expected_scores, atol=3e-5, rtol=3e-5):
                raise ValueError("cost deployment does not reproduce saved evaluation scores")
            selected = np.linspace(0, len(frame) - 1, min(512, len(frame)), dtype=int)
            feature_sample = frame.iloc[selected].loc[:, list(schema.features)].copy()
            cost = benchmark_predictor(predict, feature_sample, batch_size=settings["query_batch_size"],
                                       **settings["benchmark"])
            csv_path = directory / "benchmark_features.csv"
            feature_sample.to_csv(csv_path, index=False)
            durations = []
            for _ in range(settings["benchmark"]["repeats"]):
                start = time.perf_counter()
                parsed = read_csv(csv_path, schema, require_label=False)
                predict(parsed)
                durations.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(previous_threads)
    cost.update({
        "worker": {"pid": os.getpid(), "entry": "ids.route_batch._cost-worker",
                   "contract": "new inference-only subprocess; no gradient training in this worker",
                   "gradient_training_performed": False, "loaded_data_roles": ["evaluation"],
                   "registered_evaluation_rows_in_memory": len(frame)},
        "model_load_or_rebuild_seconds": load_seconds,
        "model_load_scope": "artifact verification + complete model/preprocessing load or cache rebuild; "
                            "excludes interpreter startup, shared imports and E CSV parsing",
        "cold_model_load_measured": True,
        "required_deployment_file_bytes": sum(sizes.values()),
        "deployment_files": sizes, "deployment_storage": storage,
        "replay": {"rows": replay_rows, "ids_and_labels_match": True, "atol": 3e-5, "rtol": 3e-5,
                   "max_absolute_score_error": float(np.max(np.abs(replay - expected_scores)))},
        "benchmark_evaluation_row_indices": selected.tolist(),
        "csv_to_scores": {"rows": len(feature_sample), "seconds": durations,
                          "scope": "disk CSV parse + preprocessing + CPU scores; warm model; excludes UI/export",
                          "file_sha256": sha256_file(csv_path), "os_file_cache": "not cleared"},
    })
    cost["rss"]["scope"] = "fresh inference-only worker, whole-process RSS during timed hot inference; " \
                           "includes the common E frame and process baseline; excludes model-load peak"
    write_json(directory / "cost.json", cost)
    return cost
