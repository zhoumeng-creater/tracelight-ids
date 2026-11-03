"""Optional official TabM baseline, with train-only original-field preprocessing.

Importing this module never installs packages or imports PyTorch/TabM. The
optional dependencies must be installed explicitly in a separate environment.
Only load model artifacts from a trusted source: joblib uses pickle.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .models import _check_frame, _labels, _numeric, _resolve_device
from .selection import normalize_feature_schema

TABM_VERSION = "0.0.3"
NUM_EMBEDDINGS_VERSION = "0.0.12"
_ARTIFACT_VERSION = 1
_TRAINING_PROTOCOL_VERSION = 2


def recent_dependency_versions() -> dict[str, str]:
    """Check the isolated optional environment; never install or upgrade it."""
    versions = {}
    for package, required in (("tabm", TABM_VERSION),
                              ("rtdl_num_embeddings", NUM_EMBEDDINGS_VERSION)):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(
                "TabM requires explicit optional installation in an isolated environment: "
                f"tabm=={TABM_VERSION}, rtdl_num_embeddings=={NUM_EMBEDDINGS_VERSION}; "
                "see docs/RECENT_METHODS.md"
            ) from exc
        if actual != required:
            raise ImportError(f"Reproducible TabM adapter requires {package}=={required}; found {actual}")
        versions[package] = actual
    for package in ("torch", "numpy", "pandas", "scikit-learn"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _parameters(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("TabM config must be a dictionary")
    values = copy.deepcopy(config.get("hyperparameters", config))
    if not isinstance(values, dict):
        raise ValueError("TabM hyperparameters must be a dictionary")
    if "lr" in values and "learning_rate" in values:
        raise ValueError("Set only one of lr and learning_rate")
    result = {name: _positive_int(values.pop(name, default), name) for name, default in
              (("epochs", 60), ("patience", 8), ("batch_size", 1024), ("k", 32),
               ("n_blocks", 2), ("d_block", 512), ("embedding_dim", 16), ("n_bins", 48))}
    for name, default in (("learning_rate", 1e-3), ("weight_decay", 1e-4),
                          ("min_delta", 1e-5), ("dropout", 0.1)):
        value = values.pop(name, values.pop("lr", default) if name == "learning_rate" else default)
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        result[name] = float(value)
    if (not np.isfinite([result[x] for x in ("learning_rate", "weight_decay", "min_delta", "dropout")]).all()
            or result["learning_rate"] <= 0 or result["weight_decay"] < 0
            or result["min_delta"] < 0 or not 0 <= result["dropout"] < 1):
        raise ValueError("Invalid TabM learning_rate, weight_decay, min_delta or dropout")
    result["num_embeddings"] = values.pop("num_embeddings", "piecewise_linear")
    result["arch_type"] = values.pop("arch_type", "tabm")
    if result["num_embeddings"] not in {"none", "piecewise_linear"}:
        raise ValueError("num_embeddings must be none or piecewise_linear")
    if result["arch_type"] not in {"tabm", "tabm-mini"}:
        raise ValueError("arch_type must be tabm or tabm-mini")
    if result["n_bins"] < 2:
        raise ValueError("n_bins must be at least 2")
    if values:
        raise ValueError(f"Unknown TabM hyperparameters: {sorted(values)}")
    return result


@dataclass
class TabMPreprocessor:
    """Numeric imputation/scaling and categorical vocabularies fitted on train.

    Categorical index 0 is missing, 1 is unseen, and 2 onward are train-only
    categories. The official TabM module performs the one-hot expansion.
    """

    selected_features: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    vocabularies: dict[str, dict[str, int]] = field(default_factory=dict)
    imputer: Any = None
    scaler: Any = None

    @classmethod
    def fit(cls, train_df, selected_features, feature_schema=None):
        schema = normalize_feature_schema(feature_schema)
        _check_frame(train_df, schema["features"], allow_empty=False)
        if (not selected_features or len(set(selected_features)) != len(selected_features)
                or not set(selected_features).issubset(schema["features"])):
            raise ValueError("Selected features do not match the feature schema")
        numeric = [x for x in selected_features if x not in schema["categorical"]]
        categorical = [x for x in selected_features if x in schema["categorical"]]
        result = cls(list(selected_features), numeric, categorical)
        if numeric:
            result.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            matrix = result.imputer.fit_transform(_numeric(train_df, numeric))
            result.scaler = StandardScaler().fit(matrix)
        for name in categorical:
            categories = sorted(train_df[name].astype("string").dropna().unique().tolist())
            result.vocabularies[name] = {category: index + 2 for index, category in enumerate(categories)}
        return result

    @property
    def cat_cardinalities(self):
        return [len(self.vocabularies[name]) + 2 for name in self.categorical_columns]

    @property
    def encoded_dim(self):
        """Input width before optional numerical embeddings, including one-hot."""
        return len(self.numeric_columns) + sum(self.cat_cardinalities)

    def transform(self, frame):
        _check_frame(frame, self.selected_features, allow_empty=True)
        numeric = np.empty((len(frame), len(self.numeric_columns)), dtype=np.float32)
        if self.numeric_columns and len(frame):
            values = self.imputer.transform(_numeric(frame, self.numeric_columns))
            with np.errstate(over="ignore", invalid="ignore"):
                numeric = np.asarray(self.scaler.transform(values), dtype=np.float32)
            if not np.isfinite(numeric).all():
                raise ValueError("TabM numeric input is not finite float32")
        categorical = np.empty((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for index, name in enumerate(self.categorical_columns):
            values = frame[name].astype("string")
            encoded = values.map(self.vocabularies[name]).fillna(1).to_numpy(dtype=np.int64)
            encoded[values.isna().to_numpy()] = 0
            categorical[:, index] = encoded
        return np.ascontiguousarray(numeric), np.ascontiguousarray(categorical)


def _numeric_bins(matrix, parameters):
    if parameters["num_embeddings"] == "none" or matrix.shape[1] == 0:
        return [], []
    import torch
    from rtdl_num_embeddings import compute_bins

    bins, constant_columns = [], []
    for index in range(matrix.shape[1]):
        column = torch.from_numpy(matrix[:, index:index + 1])
        if torch.all(column == column[0]):
            # A train-constant column has no empirical quantiles. StandardScaler
            # centers it to zero; this fixed interval avoids fitting on val.
            bins.append([-1.0, 1.0])
            constant_columns.append(index)
        elif len(matrix) == 2:
            bins.append([float(column.min()), float(column.max())])
        else:
            boundaries = compute_bins(column, n_bins=min(parameters["n_bins"], len(matrix) - 1))[0]
            bins.append(boundaries.tolist())
    return bins, constant_columns


def make_tabm_network(n_num_features, cat_cardinalities, parameters, *, numeric_bins=None):
    """Build the pinned official model, including its original ensemble layers."""
    recent_dependency_versions()
    import torch
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings
    from tabm import TabM

    embeddings = None
    if parameters["num_embeddings"] == "piecewise_linear" and n_num_features:
        if numeric_bins is None or len(numeric_bins) != n_num_features:
            raise ValueError("Train-fitted numeric bins are required for all numeric fields")
        embeddings = PiecewiseLinearEmbeddings(
            [torch.tensor(x, dtype=torch.float32) for x in numeric_bins],
            parameters["embedding_dim"], activation=True, version="B",
        )
    return TabM.make(
        n_num_features=n_num_features, cat_cardinalities=cat_cardinalities or None,
        d_out=1, num_embeddings=embeddings, k=parameters["k"],
        n_blocks=parameters["n_blocks"], d_block=parameters["d_block"],
        dropout=parameters["dropout"], arch_type=parameters["arch_type"],
    )


def tabm_head_loss(logits, labels, positive_weight=None):
    """Mean of independent per-head BCE losses, never BCE of averaged logits."""
    import torch

    if logits.ndim != 3 or logits.shape[2] != 1 or labels.ndim != 1 or len(labels) != len(logits):
        raise ValueError("TabM expects logits (B,k,1) and labels (B,)")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(-1), labels[:, None].expand(-1, logits.shape[1]), pos_weight=positive_weight,
    )


def tabm_probabilities(logits):
    """Inference is the mean probability, rather than sigmoid(mean logits)."""
    import torch

    if logits.ndim != 3 or logits.shape[2] != 1:
        raise ValueError("TabM expects logits (B,k,1)")
    return torch.sigmoid(logits.squeeze(-1)).mean(dim=1)


def tabm_ensemble_loss(logits, labels, positive_weight=None):
    """Weighted BCE of the deployed mean probability, for validation only.

    Training still optimizes ``tabm_head_loss``. The ensemble probability is
    clamped to [eps, 1-eps], where eps is its floating dtype's machine epsilon,
    before log/log1p. This keeps saturated probabilities finite and explicitly
    bounds their validation penalty; it is not sigmoid(mean(logits)). Positive
    weights come from train, without normalization by total sample weight.
    """
    import torch

    probabilities = tabm_probabilities(logits)
    if labels.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("TabM ensemble loss expects labels (B,)")
    epsilon = torch.finfo(probabilities.dtype).eps
    safe = probabilities.clamp(min=epsilon, max=1 - epsilon)
    weight = 1.0 if positive_weight is None else positive_weight
    return -(weight * labels * safe.log() + (1 - labels) * torch.log1p(-safe)).mean()


def _forward(network, numeric, categorical, device):
    import torch

    return network(torch.from_numpy(numeric).to(device) if numeric.shape[1] else None,
                   torch.from_numpy(categorical).to(device) if categorical.shape[1] else None)


@dataclass
class TabMModelBundle:
    preprocessor: TabMPreprocessor
    feature_schema: dict
    parameters: dict
    numeric_bins: list[list[float]]
    network_state: dict
    selected_features: list[str]
    history: list[dict] = field(default_factory=list)
    best_epoch: int | None = None
    training_statistics: dict = field(default_factory=dict)
    seed: int = 0
    device: str = "cpu"
    num_threads: int = 1
    model_name: str = "tabm"
    artifact_version: int = _ARTIFACT_VERSION
    _cached_network: Any = field(default=None, repr=False, compare=False)

    @property
    def encoded_dim(self):
        return self.preprocessor.encoded_dim

    @property
    def batch_size(self):
        return self.parameters["batch_size"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["device"], state["_cached_network"] = "cpu", None
        state["network_state"] = {key: value.detach().cpu().clone()
                                  for key, value in self.network_state.items()}
        return state

    def predict_scores(self, frame):
        _check_frame(frame, self.feature_schema["features"], allow_empty=True)
        numeric, categorical = self.preprocessor.transform(frame)
        if frame.empty:
            return np.empty(0, dtype=np.float32)
        import torch

        if self._cached_network is None:
            self.device = _resolve_device(self.device)
            self._cached_network = make_tabm_network(
                len(self.preprocessor.numeric_columns), self.preprocessor.cat_cardinalities,
                self.parameters, numeric_bins=self.numeric_bins,
            )
            self._cached_network.load_state_dict(self.network_state, strict=True)
            self._cached_network = self._cached_network.to(self.device).eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(frame), self.batch_size):
                logits = _forward(self._cached_network, numeric[start:start + self.batch_size],
                                  categorical[start:start + self.batch_size], self.device)
                chunks.append(tabm_probabilities(logits).cpu().numpy())
        scores = np.concatenate(chunks)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("TabM returned invalid attack scores")
        return scores


def train_tabm(train_df, val_df, config, seed, *, device="cpu", num_threads=1,
               feature_schema=None, selected_features=None):
    """Fit one optional baseline; inputs are only train and development val.

    Callers own data partitions, selection provenance and experiment budgets.
    Validation controls early stopping only; this function never calibrates a
    threshold or loads calibration/test files.
    """
    parameters = _parameters(config)
    num_threads = _positive_int(num_threads, "num_threads")
    schema = normalize_feature_schema(feature_schema)
    for frame in (train_df, val_df):
        _check_frame(frame, schema["features"], allow_empty=False)
    labels, val_labels = _labels(train_df, require_both=True), _labels(val_df, require_both=False)
    selected = list(schema["features"] if selected_features is None else selected_features)
    versions = recent_dependency_versions()
    import torch

    resolved_device = _resolve_device(device)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    preprocessor = TabMPreprocessor.fit(train_df, selected, feature_schema=schema)
    x_train, c_train = preprocessor.transform(train_df)
    x_val, c_val = preprocessor.transform(val_df)
    bins, constants = _numeric_bins(x_train, parameters)
    bundle = TabMModelBundle(preprocessor, schema, parameters, bins, {}, selected,
                            seed=int(seed), num_threads=num_threads)
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cuda_devices = [] if resolved_device == "cpu" else [torch.device(resolved_device).index
                                                       or torch.cuda.current_device()]
    try:
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=cuda_devices), threadpool_limits(limits=num_threads):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(seed)
            network = make_tabm_network(len(preprocessor.numeric_columns), preprocessor.cat_cardinalities,
                                        parameters, numeric_bins=bins).to(resolved_device)
            optimizer = torch.optim.AdamW(network.parameters(), lr=parameters["learning_rate"],
                                           weight_decay=parameters["weight_decay"])
            positive_weight = float((labels == 0).sum() / (labels == 1).sum())
            weight = torch.tensor(positive_weight, dtype=torch.float32, device=resolved_device)
            generator = torch.Generator().manual_seed(seed)
            best_loss, stale = float("inf"), 0
            for epoch in range(1, parameters["epochs"] + 1):
                network.train()
                running_loss = 0.0
                for indices in torch.randperm(len(labels), generator=generator).split(parameters["batch_size"]):
                    index = indices.numpy()
                    targets = torch.tensor(labels[index], dtype=torch.float32, device=resolved_device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = _forward(network, x_train[index], c_train[index], resolved_device)
                    loss = tabm_head_loss(logits, targets, weight)
                    if not torch.isfinite(loss):
                        raise ValueError("TabM training loss became non-finite")
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * len(index)
                network.eval()
                validation_loss, validation_head_loss, chunks = 0.0, 0.0, []
                with torch.inference_mode():
                    for start in range(0, len(val_df), parameters["batch_size"]):
                        stop = start + parameters["batch_size"]
                        logits = _forward(network, x_val[start:stop], c_val[start:stop], resolved_device)
                        targets = torch.tensor(val_labels[start:stop], dtype=torch.float32, device=resolved_device)
                        validation_head_loss += tabm_head_loss(logits, targets, weight).item() * len(targets)
                        validation_loss += tabm_ensemble_loss(logits, targets, weight).item() * len(targets)
                        chunks.append(tabm_probabilities(logits).cpu().numpy())
                validation_loss /= len(val_df)
                validation_head_loss /= len(val_df)
                if not np.isfinite([validation_loss, validation_head_loss]).all():
                    raise ValueError("TabM validation loss became non-finite")
                scores = np.concatenate(chunks)
                bundle.history.append({"epoch": epoch, "train_loss": running_loss / len(labels),
                                       "val_loss": validation_loss,
                                       "val_ensemble_loss": validation_loss,
                                       "val_head_loss": validation_head_loss,
                                       "val_ap": float(average_precision_score(val_labels, scores))
                                       if len(np.unique(val_labels)) == 2 else None})
                if validation_loss < best_loss - parameters["min_delta"]:
                    best_loss, stale, bundle.best_epoch = validation_loss, 0, epoch
                    bundle.network_state = {key: value.detach().cpu().clone()
                                            for key, value in network.state_dict().items()}
                else:
                    stale += 1
                    if stale >= parameters["patience"]:
                        break
            bundle.training_statistics = {
                "parameter_count": sum(value.numel() for value in network.parameters()),
                "trainable_parameter_count": sum(value.numel() for value in network.parameters() if value.requires_grad),
                "best_epoch": bundle.best_epoch, "epochs_completed": len(bundle.history),
                "best_validation_loss": best_loss, "positive_weight": positive_weight,
                "training_objective": "mean_independent_head_weighted_bce",
                "training_protocol_version": _TRAINING_PROTOCOL_VERSION,
                "validation_objective": "weighted_bce_of_mean_sigmoid_probability",
                "early_stopping_metric": "val_ensemble_loss",
                "validation_loss_clamp": {"policy": "floating_dtype_machine_epsilon",
                    "epsilon": float(torch.finfo(weight.dtype).eps), "dtype": str(weight.dtype)},
                "inference_aggregation": "mean_sigmoid_probability", "actual_device": resolved_device,
                "dependencies": versions, "n_numeric_fields": len(preprocessor.numeric_columns),
                "categorical_cardinalities": preprocessor.cat_cardinalities,
                "numeric_bins_source": "train_only_quantiles" if bins else "not_used",
                "constant_numeric_fields": [preprocessor.numeric_columns[x] for x in constants],
                "numeric_bin_counts": [len(x) - 1 for x in bins],
                "encoded_dim_before_numeric_embeddings": preprocessor.encoded_dim,
                "backbone_input_dim": (len(preprocessor.numeric_columns) * parameters["embedding_dim"]
                                       if bins else len(preprocessor.numeric_columns))
                                      + sum(preprocessor.cat_cardinalities),
                "research_status": "project_budget_adaptation_of_official_TabM_not_paper_metric_reproduction",
            }
        return bundle
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[2:]


def save_tabm_bundle(bundle, path):
    """Atomically save the trusted bundle with CPU weights and no cached net."""
    if not isinstance(bundle, TabMModelBundle):
        raise ValueError("Expected a TabMModelBundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        joblib.dump(bundle, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_tabm_bundle(path, device="cpu"):
    """Load only trusted artifacts: joblib deserialization can execute code."""
    recent_dependency_versions()
    bundle = joblib.load(path)
    if not isinstance(bundle, TabMModelBundle) or bundle.artifact_version != _ARTIFACT_VERSION:
        raise ValueError("Unsupported TabM artifact")
    schema = normalize_feature_schema(bundle.feature_schema)
    if (not bundle.selected_features or len(set(bundle.selected_features)) != len(bundle.selected_features)
            or bundle.selected_features != bundle.preprocessor.selected_features
            or not set(bundle.selected_features).issubset(schema["features"])
            or bundle.preprocessor.numeric_columns != [x for x in bundle.selected_features
                                                        if x not in schema["categorical"]]
            or bundle.preprocessor.categorical_columns != [x for x in bundle.selected_features
                                                            if x in schema["categorical"]]):
        raise ValueError("Invalid TabM artifact feature schema")
    if bundle.parameters != _parameters(bundle.parameters):
        raise ValueError("Invalid TabM artifact parameters")
    bundle.device, bundle._cached_network = _resolve_device(device), None
    return bundle
