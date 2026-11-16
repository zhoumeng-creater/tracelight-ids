"""Fit-only typed Zeek inputs shared by the new HGB and TabM systems."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONTINUOUS = ("duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
PROTOCOLS = (1, 6, 17, 58, 132)
CAT_CARDINALITIES = (6, 2, 2, 2, 2, 2)
FORMAT = "typed-zeek-c2-v1"


def _raw(frame):
    if not frame.columns.is_unique:
        raise ValueError("Input column names must be unique")
    values = frame.loc[:, list(CONTINUOUS)].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(values).any() or ((values < 0) & (values != -1)).any():
        raise ValueError("Continuous fields require nonnegative numbers, -1, or NaN")
    missing = np.isnan(values) | (values == -1)
    protocol = frame.loc[:, "ip_protocol"].to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(protocol)
    if np.isinf(protocol).any() or np.any(finite & ((protocol < -1) | (protocol > 255)
                                                  | (protocol != np.floor(protocol)))):
        raise ValueError("Protocol must be an integer IP code, -1, or NaN")
    encoded = np.full(len(frame), len(PROTOCOLS), dtype=np.int64)
    for index, code in enumerate(PROTOCOLS):
        encoded[protocol == code] = index
    categorical = np.column_stack((encoded, missing.astype(np.int64)))
    return values, missing, np.ascontiguousarray(categorical, dtype=np.int64)


def _unique(matrix):
    # Numeric equality, including +0/-0, defines an identical model input.
    return int(len(np.unique(matrix, axis=0)))


@dataclass
class TypedPreprocessor:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    all_missing: np.ndarray
    constant: np.ndarray
    fit_rows: int

    @classmethod
    def fit(cls, frame, *, design=None):
        if not len(frame):
            raise ValueError("Cannot fit preprocessing on an empty frame")
        if design is not None:
            spec = design["input"]
            if tuple(spec["continuous"]) != CONTINUOUS:
                raise ValueError("Continuous feature order differs from the typed contract")
            if spec.get("protocol_categories", [*PROTOCOLS, "UNKNOWN"]) != [*PROTOCOLS, "UNKNOWN"]:
                raise ValueError("Protocol categories differ from the fixed semantic mapping")
            if (spec.get("all_missing_fill", 0) != 0 or spec.get("zero_variance_scale", 1) != 1
                    or spec.get("missing_masks_enter_PLE", False)):
                raise ValueError("Missingness or scaling policy differs from the registered contract")
        values, missing, _ = _raw(frame)
        all_missing = missing.all(axis=0)
        median = np.array([0.0 if all_missing[i] else np.median(values[~missing[:, i], i])
                           for i in range(len(CONTINUOUS))])
        filled = np.where(missing, median, values)
        logged = np.log1p(filled)
        mean, scale = logged.mean(axis=0), logged.std(axis=0)
        constant = np.ptp(logged, axis=0) == 0
        mean[constant] = logged[0, constant]
        scale[constant] = 1.0
        minimum = np.array([0.0 if all_missing[i] else values[~missing[:, i], i].min()
                            for i in range(len(CONTINUOUS))])
        maximum = np.array([0.0 if all_missing[i] else values[~missing[:, i], i].max()
                            for i in range(len(CONTINUOUS))])
        return cls(median, mean, scale, minimum, maximum, all_missing, constant, len(frame))

    @property
    def cat_cardinalities(self):
        return list(CAT_CARDINALITIES)

    def _transform64(self, frame):
        values, missing, categorical = _raw(frame)
        numeric = (np.log1p(np.where(missing, self.median, values)) - self.mean) / self.scale
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite transformed input")
        return np.ascontiguousarray(numeric), categorical

    def transform(self, frame):
        numeric, categorical = self._transform64(frame)
        numeric = np.ascontiguousarray(numeric, dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite FP32 input")
        return numeric, categorical

    def hgb_matrix(self, frame):
        numeric, categorical = self._transform64(frame)
        one_hot = [np.eye(width, dtype=np.float64)[categorical[:, i]]
                   for i, width in enumerate(CAT_CARDINALITIES)]
        return np.ascontiguousarray(np.column_stack([numeric, *one_hot]))

    def audit(self, frame):
        values, missing, categorical = _raw(frame)
        numeric64, _ = self._transform64(frame)
        numeric32, _ = self.transform(frame)
        typed = np.column_stack([np.where(missing, 0.0, values), categorical])
        return {
            "rows": len(frame), "fit_rows": self.fit_rows,
            "missing_by_feature": dict(zip(CONTINUOUS, missing.sum(axis=0).astype(int).tolist())),
            "outside_fit_range_by_feature": dict(zip(CONTINUOUS, (
                ~missing & ((values < self.minimum) | (values > self.maximum))).sum(axis=0).astype(int).tolist())),
            "unknown_protocol_rows": int(np.sum(categorical[:, 0] == len(PROTOCOLS))),
            "unique_typed_inputs": _unique(typed),
            "unique_standardized_float64_inputs": _unique(np.column_stack([numeric64, categorical])),
            "unique_standardized_float32_inputs": _unique(np.column_stack([numeric32, categorical])),
            "categorical_cardinalities": self.cat_cardinalities,
            "hgb_numeric_dtype": "float64", "tabm_numeric_dtype": "float32",
            "port_and_identity_columns_used": False,
        }

    def to_dict(self):
        return {"format": FORMAT, "continuous": list(CONTINUOUS),
                "protocol_categories": [*PROTOCOLS, "UNKNOWN"],
                "categorical_cardinalities": self.cat_cardinalities, "fit_rows": self.fit_rows,
                **{name: getattr(self, name).tolist() for name in
                   ("median", "mean", "scale", "minimum", "maximum", "all_missing", "constant")}}

    @classmethod
    def from_dict(cls, state):
        if (state.get("format") != FORMAT or state.get("continuous") != list(CONTINUOUS)
                or state.get("protocol_categories") != [*PROTOCOLS, "UNKNOWN"]
                or state.get("categorical_cardinalities") != list(CAT_CARDINALITIES)):
            raise ValueError("Unsupported typed preprocessing artifact")
        arrays = {name: np.asarray(state[name], dtype=bool if name in ("all_missing", "constant") else np.float64)
                  for name in ("median", "mean", "scale", "minimum", "maximum", "all_missing", "constant")}
        if any(array.shape != (len(CONTINUOUS),) for array in arrays.values()):
            raise ValueError("Preprocessing arrays have the wrong shape")
        if (any(not np.isfinite(array).all() for array in arrays.values()) or (arrays["scale"] <= 0).any()
                or (arrays["median"] < 0).any() or int(state["fit_rows"]) < 1):
            raise ValueError("Invalid stored preprocessing parameters")
        return cls(**arrays, fit_rows=int(state["fit_rows"]))
