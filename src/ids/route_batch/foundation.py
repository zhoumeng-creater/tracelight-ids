"""Offline, pinned foundation-model system references for the route batch.

Inputs are raw numeric features, not the protected-ID adapter's prepared tensor.
TabICL's own scaling, clipping, permutations and optional power transform remain
active. These arms therefore compare complete pipelines, not just ensemble size.
Saved joblib files are trusted rebuild recipes: they contain support data but no
weights or fitted cache. Loading fits again and reports the rebuild time.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import time
from unittest.mock import patch

import joblib
import numpy as np
from threadpoolctl import threadpool_limits


TABPFN_VERSION = "9.2.0"
TABPFN_CHECKPOINT = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
TABPFN_CHECKPOINT_SHA256 = "5d7170e2d3af01f9c501bb09ec3bd12e9944f8604de18002c647873c6ec04a12"
SCORE_EPSILON = float(np.finfo(np.float64).eps)
QUERY_BATCH_SIZE = 128
_FORMAT = "tracelight-foundation-rebuild-v1"
_SKLEARN_HASHES = {
    "classifier.py": "64f5ef7ef88a920a3f7a5f6e2d9d8dc08769ec343e931f910acc2d9c4eaa5534",
    "base.py": "7f8f284ca91bf1f5a016179834e508890b3f5368c4a48a7dcb6cb4b83a30051a",
}
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    "TABPFN_DISABLE_TELEMETRY": "1", "TABPFN_NO_BROWSER": "1",
    # CPU is the registered deployment device even when P has over 1000 rows.
    "TABPFN_ALLOW_CPU_LARGE_DATASET": "true",
}


class FoundationUnavailable(RuntimeError):
    """An optional arm lacks its dependency or local checkpoint; safe to skip."""

    def __init__(self, kind: str, reason: str):
        self.kind, self.reason = kind, reason
        super().__init__(f"{kind} unavailable: {reason}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric(x, *, width=None, allow_empty=False):
    values = np.asarray(x)
    if values.ndim != 2 or values.shape[1] == 0 or (not allow_empty and len(values) == 0):
        raise ValueError("x must be a nonempty two-dimensional numeric feature matrix")
    if values.dtype.kind not in "biuf":
        raise ValueError("x must already contain raw numeric values, not strings or category objects")
    if width is not None and values.shape[1] != width:
        raise ValueError(f"feature width changed: expected {width}, got {values.shape[1]}")
    values = np.asarray(values, dtype=np.float64, order="C")
    if np.isinf(values).any():
        raise ValueError("raw numeric input contains infinity")
    return values


@contextmanager
def _runtime():
    """Restore caller settings after a serial offline, one-thread operation."""
    import torch
    old_threads = torch.get_num_threads()
    with patch.dict(os.environ, _OFFLINE_ENV), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        try:
            yield torch
        finally:
            torch.set_num_threads(old_threads)


def _tabicl_class():
    from ids.context_cache.preprocessing import require_pinned_preprocessing
    _, provenance = require_pinned_preprocessing()
    module = importlib.import_module("tabicl._sklearn.classifier")
    root = Path(module.__file__).resolve().parent
    observed = {}
    for name, expected in _SKLEARN_HASHES.items():
        digest = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if digest != expected:
            raise RuntimeError(f"pinned TabICL sklearn source mismatch: {name}")
        observed[name] = digest
    return module.TabICLClassifier, {**provenance, "sklearn_source_sha256": observed}


def _tabpfn_class():
    try:
        installed = version("tabpfn")
    except PackageNotFoundError as exc:
        raise FoundationUnavailable("tabpfn25", f"install optional tabpfn=={TABPFN_VERSION}") from exc
    if installed != TABPFN_VERSION:
        raise FoundationUnavailable("tabpfn25", f"tabpfn=={TABPFN_VERSION} required; found {installed}")
    module = importlib.import_module("tabpfn")
    constants = importlib.import_module("tabpfn.constants")
    loading = importlib.import_module("tabpfn.model_loading")
    # Settings may have been imported before the environment was set.
    settings = importlib.import_module("tabpfn.settings").settings
    return module.TabPFNClassifier, constants.ModelVersion.V2_5, loading, settings


def _no_download(*args, **kwargs):
    raise RuntimeError("automatic model download is disabled for this registered batch")


@dataclass
class FoundationModel:
    kind: str
    estimator: object
    metadata: dict
    _support_x: np.ndarray
    _support_y: np.ndarray

    def predict_scores(self, x, batch_size=QUERY_BATCH_SIZE):
        """Return float64 log odds of the API's class-1 probability.

        API probability rounding is retained. In particular, saturated 0/1
        probabilities are tied after clipping; this is not a native-logit score.
        """
        if type(batch_size) is not int or batch_size != QUERY_BATCH_SIZE:
            raise ValueError("registered foundation inference requires query batch_size=128")
        values = _numeric(x, width=self._support_x.shape[1], allow_empty=True)
        output = np.empty(len(values), dtype=np.float64)
        with _runtime():
            for start in range(0, len(values), batch_size):
                stop = min(start + batch_size, len(values))
                probabilities = np.asarray(self.estimator.predict_proba(values[start:stop]), dtype=np.float64)
                if probabilities.shape != (stop - start, 2) or not np.isfinite(probabilities).all():
                    raise ValueError("foundation predict_proba returned invalid binary probabilities")
                if (probabilities < 0).any() or (probabilities > 1).any():
                    raise ValueError("foundation probabilities must lie in [0,1]")
                if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=2e-6):
                    raise ValueError("foundation probability rows do not sum to one")
                p = np.clip(probabilities[:, 1], SCORE_EPSILON, 1.0 - SCORE_EPSILON)
                output[start:stop] = np.log(p) - np.log1p(-p)
        return output

    def save(self, path):
        """Save trusted support/recipe data; loading rebuilds the fitted model/cache."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": _FORMAT, "kind": self.kind, "metadata": self.metadata,
            "support_x": self._support_x, "support_y": self._support_y,
        }
        joblib.dump(payload, target, compress=0)
        return {"path": str(target.resolve()), "sha256": _sha256(target),
                "bytes": target.stat().st_size, "contains_weights": False,
                "contains_fitted_cache": False, "requires_rebuild_on_load": True}


def fit_foundation(kind, x, y, *, checkpoint_path, seed, device, n_estimators):
    """Build ``tabicl`` (1/8 views) or optional ``tabpfn25`` (8 views).

    ``device`` applies to initial fitting/cache construction. Scoring always
    uses CPU FP32 with one thread and registered batches of at most 128 rows.
    No checkpoint is downloaded. TabPFN accepts only the registered default-2.5
    checkpoint digest. TabICL's supplied checkpoint is hashed for persistence;
    the enclosing experiment registration owns its pretrained-weight identity.
    """
    if kind not in {"tabicl", "tabpfn25"}:
        raise ValueError("kind must be 'tabicl' or 'tabpfn25'")
    if type(n_estimators) is not int or n_estimators not in ({1, 8} if kind == "tabicl" else {8}):
        raise ValueError("TabICL supports registered 1/8 views; TabPFN-2.5 requires 8 views")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    values = _numeric(x)
    truth = np.asarray(y)
    if truth.shape != (len(values),) or not np.array_equal(np.unique(truth), [0, 1]):
        raise ValueError("support labels must contain both 0=normal and 1=attack")
    truth = truth.astype(np.int64, copy=True)
    if checkpoint_path is None or not Path(checkpoint_path).is_file():
        raise FoundationUnavailable(kind, f"local checkpoint is missing: {checkpoint_path}; no download attempted")
    checkpoint = Path(checkpoint_path).resolve()
    digest = _sha256(checkpoint)
    if kind == "tabpfn25" and digest != TABPFN_CHECKPOINT_SHA256:
        raise ValueError("TabPFN-2.5 checkpoint SHA-256 differs from the registered default checkpoint")
    if kind == "tabpfn25" and checkpoint.name != TABPFN_CHECKPOINT:
        raise ValueError(f"retain TabPFN's version-bearing checkpoint filename: {TABPFN_CHECKPOINT}")
    started = time.perf_counter()
    with _runtime() as torch:
        try:
            if kind == "tabicl":
                classifier, source = _tabicl_class()
                parameters = {
                    "n_estimators": n_estimators, "norm_methods": ["none", "power"],
                    "feat_shuffle_method": "latin", "class_shuffle_method": "shift",
                    "outlier_threshold": 4.0, "softmax_temperature": 0.9,
                    "average_logits": True, "batch_size": 1, "kv_cache": True,
                    "model_path": str(checkpoint), "allow_auto_download": False,
                    "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
                    "device": device, "use_amp": False, "use_fa3": False,
                    "offload_mode": "auto", "random_state": seed, "n_jobs": 1,
                }
                estimator = classifier(**parameters).fit(values, truth)
                # Pinned upstream's own device-transfer path, also used by load().
                estimator.device = "cpu"
                estimator._resolve_device()
                estimator.model_.to(estimator.device_)
                estimator._build_inference_config()
                estimator._move_cache_to_device()
                views = sum(len(v) for v in estimator.ensemble_generator_.class_shuffles_.values())
            else:
                classifier, model_version, loading, settings = _tabpfn_class()
                parameters = {
                    "model_path": str(checkpoint), "n_estimators": 8,
                    "device": device, "random_state": seed,
                    "inference_precision": torch.float32, "fit_mode": "fit_with_cache",
                    # In 9.2, None may choose architecture-default int8; "auto"
                    # explicitly retains the computed dtype (forced FP32 here).
                    "kv_cache_precision": "auto", "n_preprocessing_jobs": 1,
                    "softmax_temperature": 0.9, "balance_probabilities": False,
                    "average_before_softmax": False, "ignore_pretraining_limits": False,
                    "categorical_features_indices": [],
                }
                with patch.object(loading, "download_model", _no_download), \
                     patch.object(settings.tabpfn, "allow_cpu_large_dataset", True):
                    estimator = classifier.create_default_for_version(model_version, **parameters)
                    estimator.fit(values, truth)
                    estimator.to("cpu")
                views = int(estimator.n_estimators_)
                source = {"repository": "https://github.com/PriorLabs/TabPFN", "version": TABPFN_VERSION,
                          "commit": "f02b3b5941717472eafd078c15ae2043676b3fcf",
                          "checkpoint_revision": "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b",
                          "real_data_finetuned": True}
        except ImportError as exc:
            raise FoundationUnavailable(kind, f"optional dependency import failed: {exc}") from exc
    if not np.array_equal(estimator.classes_, [0, 1]):
        raise ValueError("fitted foundation class order must be [normal=0, attack=1]")
    metadata = {
        "kind": kind, "source": source, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": digest, "checkpoint_bytes": checkpoint.stat().st_size,
        "seed": seed, "build_device": str(device), "score_device": "cpu", "threads": 1,
        "requested_estimators": n_estimators, "actual_estimators": views,
        "parameters": {k: str(v) if k == "inference_precision" else v for k, v in parameters.items()},
        "support_rows": len(values), "input_features": values.shape[1],
        "input": "raw-numeric; all columns supplied as numeric ndarray; own support-fitted preprocessing",
        "comparison": "complete-pipeline reference; not an ensemble-only ablation of protected-ID adapter",
        "score": "float64(log(clip(p_attack))-log1p(-clip(p_attack))) from official predict_proba",
        "score_clip_epsilon": SCORE_EPSILON, "score_saturation": "API probability 0/1 ties are retained",
        "query_batch_size": QUERY_BATCH_SIZE, "build_and_cpu_transfer_seconds": time.perf_counter() - started,
        "persistence": "trusted support-data rebuild recipe; external checkpoint; rebuild cost required on load",
        "network": "local checkpoint only; HF offline and telemetry disabled; no automatic model download",
    }
    return FoundationModel(kind, estimator, metadata, values.copy(), truth)


def load_foundation(path, *, checkpoint_path=None, device="cpu"):
    """Load a trusted recipe, verify its external checkpoint, and rebuild.

    The caller must verify the registered SHA of this joblib artifact before
    loading it; joblib may execute code. The return metadata records reload cost.
    """
    started = time.perf_counter()
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError("unsupported foundation recipe format")
    metadata = payload["metadata"]
    checkpoint = Path(checkpoint_path or metadata["checkpoint_path"])
    if not checkpoint.is_file():
        raise FoundationUnavailable(payload["kind"], f"recipe requires external checkpoint: {checkpoint}")
    if _sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("foundation recipe checkpoint SHA-256 mismatch")
    result = fit_foundation(payload["kind"], payload["support_x"], payload["support_y"],
                            checkpoint_path=checkpoint, seed=metadata["seed"], device=device,
                            n_estimators=metadata["requested_estimators"])
    result.metadata["load_including_rebuild_seconds"] = time.perf_counter() - started
    return result
