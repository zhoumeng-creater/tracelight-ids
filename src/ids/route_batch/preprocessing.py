"""Train-only representations; identifiers never use an unknown-value bucket."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ids.context_cache.preprocessing import CSVPreprocessor, PROTECTED_IDS_PROFILE, frame_features


def raw_numeric(frame, schema):
    values = frame_features(frame, schema).apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("NF-v3 route comparison requires finite numeric raw fields")
    return values


class Representation:
    """Protected TabICL view or signed-log standardization, fitted on T or P only."""

    def __init__(self, kind, schema):
        if kind not in {"protected_v2", "log1p_standard"}:
            raise ValueError(f"unknown representation: {kind}")
        self.kind, self.schema = kind, schema

    def fit(self, frame):
        if self.kind == "protected_v2":
            self.processor = CSVPreprocessor(self.schema, profile=PROTECTED_IDS_PROFILE).fit(frame)
        else:
            raw = raw_numeric(frame, self.schema)
            self.ids = np.array([name in self.schema.categorical for name in self.schema.features])
            # Reuse the exact identifier domain validation and FP32 injectivity check.
            self.id_processor = CSVPreprocessor(self.schema, profile=PROTECTED_IDS_PROFILE)
            identifiers = self.id_processor._protected_ids(frame_features(frame, self.schema))
            self.id_processor.protected_mean_ = identifiers.mean(axis=0)
            scale = identifiers.std(axis=0)
            self.id_processor.protected_scale_ = np.where(scale > 0, scale, 1.)
            self.id_processor._verify_protected_domains()
            raw[:, ~self.ids] = np.sign(raw[:, ~self.ids]) * np.log1p(np.abs(raw[:, ~self.ids]))
            self.mean, self.scale = raw.mean(axis=0), raw.std(axis=0)
            self.scale[self.scale == 0] = 1.
        self.fit_rows = len(frame)
        return self

    def transform(self, frame):
        if self.kind == "protected_v2":
            return self.processor.transform(frame)
        raw = raw_numeric(frame, self.schema)
        self.id_processor._protected_ids(frame_features(frame, self.schema))
        raw[:, ~self.ids] = np.sign(raw[:, ~self.ids]) * np.log1p(np.abs(raw[:, ~self.ids]))
        result = np.asarray((raw - self.mean) / self.scale, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("representation produced nonfinite FP32 input")
        return np.ascontiguousarray(result)

    def report(self):
        if self.kind == "protected_v2":
            details = self.processor.report()
        else:
            details = {"identifier_domains": self.id_processor.protected_verification_,
                       "numeric_operation": "sign(x)*log1p(abs(x)), then train mean/std; no clipping",
                       "features": list(self.schema.features)}
        return {"kind": self.kind, "fit_rows": self.fit_rows, "details": details}

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        result = joblib.load(path)
        if not isinstance(result, Representation):
            raise ValueError("not a route-batch representation")
        return result


def collision_report(matrix, truth):
    x = np.asarray(matrix, dtype="<f4").copy()
    x[x == 0] = 0
    y = np.asarray(truth)
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    counts = np.zeros((int(inverse.max()) + 1, 2), dtype=np.int64)
    np.add.at(counts, (inverse, y.astype(int)), 1)
    conflict = np.all(counts > 0, axis=1)
    return {"rows": len(x), "width": x.shape[1], "unique_inputs": len(counts),
            "mixed_label_groups": int(conflict.sum()),
            "mixed_label_rows": int(counts[conflict].sum()),
            "empirical_total_error_lower_bound": int(counts.min(axis=1).sum())}
