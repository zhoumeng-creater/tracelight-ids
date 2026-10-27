"""Train-only preprocessing and portable, trusted-local model artifacts.

Only call :func:`load_bundle` on artifacts you created or otherwise trust:
joblib uses pickle and loading an untrusted file can execute arbitrary code.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from .data import CATEGORICAL, FEATURES


# The runner may collect CUDA metadata before calling train_model. Configure
# cuBLAS when this module is imported, without eagerly importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


_EXCLUDED = frozenset({"id", "sample_id", "feature_hash", "label", "attack_cat"})
_MODELS = frozenset({"lr", "rf", "hgb", "mlp"})
_ARTIFACT_VERSION = 1


def _feature_names() -> list[str]:
    names = list(FEATURES)
    if not names or len(set(names)) != len(names) or _EXCLUDED.intersection(names):
        raise ValueError("FEATURES must contain unique predictors and no identifiers or labels")
    return names


def _check_frame(frame: pd.DataFrame, features: list[str], *, allow_empty: bool) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if not allow_empty and frame.empty:
        raise ValueError("Training and validation frames must not be empty")


def _labels(frame: pd.DataFrame, *, require_both: bool) -> np.ndarray:
    if "label" not in frame:
        raise ValueError("A binary 'label' column is required")
    try:
        labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be binary 0/1 values") from exc
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be finite binary 0/1 values")
    if require_both and len(np.unique(labels)) != 2:
        raise ValueError("Training requires both normal (0) and attack (1) records")
    return labels.astype(np.int64)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=np.float64)
    try:
        values = frame.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric features contain invalid non-numeric values") from exc
    if np.isinf(values).any():
        raise ValueError("Numeric features contain infinity; clean inputs explicitly")
    return values


def _categorical(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(frame), 0), dtype=object)
    return frame.loc[:, columns].astype("string").fillna("<MISSING>").to_numpy(dtype=object)


@dataclass
class FeaturePreprocessor:
    """A fitted transform; metadata columns never enter its encoded matrix."""

    selected_features: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    imputer: Any = None
    scaler: Any = None
    encoder: Any = None
    encoded_dim: int = 0

    @classmethod
    def fit(cls, frame: pd.DataFrame, selected_features: list[str]) -> "FeaturePreprocessor":
        numeric = [name for name in selected_features if name not in CATEGORICAL]
        categorical = [name for name in selected_features if name in CATEGORICAL]
        result = cls(list(selected_features), numeric, categorical)
        if numeric:
            result.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            values = result.imputer.fit_transform(_numeric(frame, numeric))
            result.scaler = StandardScaler().fit(values)
        if categorical:
            result.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            result.encoder.fit(_categorical(frame, categorical))
        result.encoded_dim = result.transform(frame).shape[1]
        return result

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        _check_frame(frame, self.selected_features, allow_empty=True)
        if frame.empty:
            return np.empty((0, self.encoded_dim), dtype=np.float32)
        arrays = []
        if self.numeric_columns:
            values = self.imputer.transform(_numeric(frame, self.numeric_columns))
            arrays.append(self.scaler.transform(values))
        if self.categorical_columns:
            arrays.append(self.encoder.transform(_categorical(frame, self.categorical_columns)))
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(np.concatenate(arrays, axis=1), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Encoded features are not finite float32 values")
        if self.encoded_dim and result.shape[1] != self.encoded_dim:
            raise ValueError("Encoded feature width differs from the fitted schema")
        return np.ascontiguousarray(result)


def _select_features(frame: pd.DataFrame, labels: np.ndarray, settings: dict, seed: int):
    names = _feature_names()
    method = settings.get("method", "all")
    k = settings.get("k")
    if method == "all":
        if k is not None:
            raise ValueError("features.k must be null when features.method is 'all'")
        return names, {}
    if method != "mutual_info":
        raise ValueError("features.method must be 'all' or 'mutual_info'")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(names):
        raise ValueError(f"features.k must be an integer between 1 and {len(names)}")
    categorical = [name for name in names if name in CATEGORICAL]
    numeric = [name for name in names if name not in CATEGORICAL]
    columns: dict[str, np.ndarray] = {}
    if numeric:
        numeric_values = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(
            _numeric(frame, numeric)
        )
        columns.update({name: numeric_values[:, index] for index, name in enumerate(numeric)})
    if categorical:
        category_values = OrdinalEncoder().fit_transform(_categorical(frame, categorical))
        columns.update({name: category_values[:, index] for index, name in enumerate(categorical)})
    matrix = np.column_stack([columns[name] for name in names])
    scores = mutual_info_classif(
        matrix,
        labels,
        discrete_features=[name in CATEGORICAL for name in names],
        random_state=seed,
        n_neighbors=min(3, len(frame) - 1),
    )
    # Stable ties preserve the official field order. k counts original fields,
    # never the one-hot columns produced later.
    ranked = np.lexsort((np.arange(len(names)), -scores))[:k]
    chosen = set(ranked.tolist())
    return [name for index, name in enumerate(names) if index in chosen], dict(zip(names, scores.tolist()))


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("The mlp model requires PyTorch; install the project's torch requirements") from exc
    return torch


def _resolve_device(device: str) -> str:
    torch = _torch()
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid torch device: {device!r}") from exc
    if parsed.type not in {"cpu", "cuda"}:
        raise ValueError("Supported MLP devices are cpu, cuda, and cuda:<index>")
    if parsed.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if parsed.index is not None and parsed.index >= torch.cuda.device_count():
            raise ValueError("Requested CUDA device index does not exist")
    return str(parsed)


def _network(encoded_dim: int, hidden_dims: tuple[int, ...]):
    torch = _torch()
    layers = []
    previous = encoded_dim
    for width in hidden_dims:
        layers.extend([torch.nn.Linear(previous, width), torch.nn.ReLU()])
        previous = width
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


@dataclass
class ModelBundle:
    model_name: str
    preprocessor: FeaturePreprocessor
    selected_features: list[str]
    encoded_dim: int
    history: list[dict[str, Any]]
    estimator: Any = None
    network_state: Any = None
    hidden_dims: tuple[int, ...] = (64, 32)
    device: str = "cpu"
    batch_size: int = 1024
    num_threads: int = 1
    feature_scores: dict[str, float] = field(default_factory=dict)
    best_epoch: int | None = None
    seed: int = 0
    artifact_version: int = _ARTIFACT_VERSION
    _cached_network: Any = field(default=None, repr=False, compare=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_network"] = None
        state["device"] = "cpu"
        if self.network_state is not None:
            state["network_state"] = {key: value.detach().cpu().clone() for key, value in self.network_state.items()}
        return state

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Return attack scores in input row order; never use a label column.

        The public inference contract requires all official original fields,
        even for a selected-feature model. Extra metadata columns are ignored.
        The caller controls inference thread pools outside repeated calls;
        threadpool discovery must not contaminate single-record latency.
        """
        _check_frame(frame, _feature_names(), allow_empty=True)
        matrix = self.preprocessor.transform(frame)
        if not len(matrix):
            return np.empty(0, dtype=np.float32)
        if self.model_name == "mlp":
            torch = _torch()
            if self._cached_network is None:
                self.device = _resolve_device(self.device)
                self._cached_network = _network(self.encoded_dim, self.hidden_dims)
                self._cached_network.load_state_dict(self.network_state)
                self._cached_network.to(self.device).eval()
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(matrix), self.batch_size):
                    batch = torch.from_numpy(matrix[start : start + self.batch_size]).to(self.device)
                    chunks.append(torch.sigmoid(self._cached_network(batch).squeeze(1)).cpu().numpy())
            scores = np.concatenate(chunks)
        else:
            classes = self.estimator.classes_.tolist()
            scores = self.estimator.predict_proba(matrix)[:, classes.index(1)]
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(frame),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Model returned invalid attack scores")
        return scores


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _val_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    return {
        "val_loss": float(log_loss(labels, np.clip(scores.astype(np.float64), 1e-7, 1 - 1e-7), labels=[0, 1])),
        "val_ap": float(average_precision_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
    }


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict,
    seed: int,
    device: str = "cpu",
) -> ModelBundle:
    """Fit on train only; validation is used for reporting and MLP stopping.

    Configuration: ``model: lr|rf|hgb|mlp``, ``features: {method, k}``,
    and ``hyperparameters: {...}``. ``name`` is accepted as a model-name alias.
    Classical estimators execute on CPU regardless of the requested MLP device.
    """
    names = _feature_names()
    _check_frame(train_df, names, allow_empty=False)
    _check_frame(val_df, names, allow_empty=False)
    labels = _labels(train_df, require_both=True)
    val_labels = _labels(val_df, require_both=False)
    model_name = config.get("model", config.get("name"))
    if model_name not in _MODELS:
        raise ValueError(f"model must be one of {sorted(_MODELS)}")
    selected, feature_scores = _select_features(train_df, labels, config.get("features") or {}, seed)
    preprocessor = FeaturePreprocessor.fit(train_df, selected)
    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    hyperparameters = copy.deepcopy(config.get("hyperparameters") or {})
    num_threads = _positive_integer(hyperparameters.pop("num_threads", 1), "num_threads")
    bundle = ModelBundle(model_name, preprocessor, selected, x_train.shape[1], [], num_threads=num_threads,
                         feature_scores=feature_scores, seed=seed)
    if model_name == "mlp":
        return _train_mlp(bundle, x_train, labels, x_val, val_labels, hyperparameters, seed, device)
    if "random_state" in hyperparameters and hyperparameters["random_state"] != seed:
        raise ValueError("Set the experiment seed via seed, not a conflicting random_state")
    hyperparameters["random_state"] = seed
    if model_name in {"lr", "rf"}:
        if hyperparameters.get("class_weight", "balanced") != "balanced":
            raise ValueError("This baseline requires class_weight='balanced'")
        hyperparameters["class_weight"] = "balanced"
    if model_name == "lr":
        parameters = {"max_iter": 1000, "solver": "lbfgs", **hyperparameters}
        estimator = LogisticRegression(**parameters)
    elif model_name == "rf":
        parameters = {"n_estimators": 200, "min_samples_leaf": 2, "n_jobs": num_threads, **hyperparameters}
        estimator = RandomForestClassifier(**parameters)
    else:
        # Do not let sklearn make an implicit random validation split: the
        # project's explicit partitions keep duplicate feature groups intact.
        parameters = {"max_iter": 200, "learning_rate": 0.1, "early_stopping": False, **hyperparameters}
        estimator = HistGradientBoostingClassifier(**parameters)
    with threadpool_limits(limits=num_threads):
        if model_name == "hgb":
            estimator.fit(x_train, labels, sample_weight=compute_sample_weight("balanced", labels))
        else:
            estimator.fit(x_train, labels)
    bundle.estimator = estimator
    bundle.history = [{"epoch": 1, **_val_metrics(val_labels, bundle.predict_scores(val_df))}]
    return bundle


def _train_mlp(bundle, x_train, labels, x_val, val_labels, parameters, seed, device):
    # Set before the first CUDA linear operation for deterministic cuBLAS.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch = _torch()
    device = _resolve_device(device)
    epochs = _positive_integer(parameters.pop("epochs", 60), "epochs")
    batch_size = _positive_integer(parameters.pop("batch_size", 1024), "batch_size")
    patience = _positive_integer(parameters.pop("patience", 10), "patience")
    if "hidden_dims" in parameters and "hidden_sizes" in parameters:
        raise ValueError("Set only one of hidden_dims or hidden_sizes")
    hidden_dims = tuple(parameters.pop("hidden_sizes", parameters.pop("hidden_dims", (64, 32))))
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer width")
    for width in hidden_dims:
        _positive_integer(width, "hidden_dims entries")
    learning_rate = float(parameters.pop("learning_rate", parameters.pop("lr", 1e-3)))
    weight_decay = float(parameters.pop("weight_decay", 1e-4))
    min_delta = float(parameters.pop("min_delta", 1e-5))
    if not np.isfinite([learning_rate, weight_decay, min_delta]).all() or learning_rate <= 0 or weight_decay < 0 or min_delta < 0:
        raise ValueError("learning_rate must be positive; weight_decay and min_delta must be nonnegative")
    if parameters:
        raise ValueError(f"Unknown MLP hyperparameters: {sorted(parameters)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(bundle.num_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        network = _network(bundle.encoded_dim, hidden_dims).to(device)
        positive_weight = float((labels == 0).sum() / (labels == 1).sum())
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
        optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(labels.astype(np.float32))),
            batch_size=batch_size, shuffle=True, num_workers=0, generator=generator,
        )
        best_loss, stale_epochs, best_state = float("inf"), 0, None
        for epoch in range(1, epochs + 1):
            network.train()
            running_loss = 0.0
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(network(inputs).squeeze(1), targets)
                if not torch.isfinite(loss):
                    raise ValueError("MLP training loss became non-finite")
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(inputs)
            network.eval()
            validation_loss, score_chunks = 0.0, []
            with torch.inference_mode():
                for start in range(0, len(x_val), batch_size):
                    inputs = torch.from_numpy(x_val[start : start + batch_size]).to(device)
                    targets = torch.from_numpy(val_labels[start : start + batch_size].astype(np.float32)).to(device)
                    logits = network(inputs).squeeze(1)
                    validation_loss += criterion(logits, targets).item() * len(inputs)
                    score_chunks.append(torch.sigmoid(logits).cpu().numpy())
            validation_loss /= len(x_val)
            if not np.isfinite(validation_loss):
                raise ValueError("MLP validation loss became non-finite")
            scores = np.concatenate(score_chunks)
            bundle.history.append({"epoch": epoch, "train_loss": running_loss / len(x_train),
                                   "val_loss": validation_loss,
                                   "val_ap": float(average_precision_score(val_labels, scores)) if len(np.unique(val_labels)) == 2 else None})
            if validation_loss < best_loss - min_delta:
                best_loss, stale_epochs = validation_loss, 0
                best_state = {key: value.detach().cpu().clone() for key, value in network.state_dict().items()}
                bundle.best_epoch = epoch
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break
        bundle.network_state = best_state
        bundle.hidden_dims, bundle.batch_size, bundle.device = hidden_dims, batch_size, device
        return bundle
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic


def save_bundle(bundle: ModelBundle, path: str | Path) -> None:
    """Atomically save a trusted-local artifact, with all torch weights on CPU."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
        joblib.dump(bundle, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_bundle(path: str | Path, device: str = "cpu") -> ModelBundle:
    """Load only trusted artifacts: joblib/pickle can execute arbitrary code."""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle) or bundle.artifact_version != _ARTIFACT_VERSION:
        raise ValueError("Unsupported model bundle or artifact version")
    if bundle.model_name not in _MODELS or bundle.encoded_dim != bundle.preprocessor.encoded_dim:
        raise ValueError("Invalid model bundle metadata")
    if bundle.selected_features != bundle.preprocessor.selected_features or _EXCLUDED.intersection(bundle.selected_features):
        raise ValueError("Invalid model feature schema")
    if not set(bundle.selected_features).issubset(_feature_names()):
        raise ValueError("Artifact features do not match the current data schema")
    bundle._cached_network = None
    bundle.device = _resolve_device(device) if bundle.model_name == "mlp" else "cpu"
    return bundle
