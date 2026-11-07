"""Support-fitted CSV preprocessing around pinned TabICLv2 (2026).

The transformation chain is the official TransformToNumerical, constant-feature
filter, and PreprocessingPipeline(normalization_method='none'). No ensemble,
feature permutation or class permutation is created. This 'none' option still
uses the official standard scaling and outlier handling. Pickled state files
are trusted project artifacts; verify their registered SHA before loading.
The opt-in NF-v3 profile preserves four integer identifiers without a vocabulary,
constant-column removal or clipping. It is an input-contract repair, not a claim
that numerical identifier distances express application-protocol semantics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import TABICL_COMMIT
from .upstream import require_pinned_tabicl


_PREPROCESSING_FILES = {
    "preprocessing.py": "2163a3bf6754127365a7aedac854d99a6f565545cc1ea7e63c68a596081b2976",
    "sklearn_utils.py": "e5639073a9c80bcaf13c1b59de424be76f17d4e66418f74ff0c41840920d54b2",
}
_FORMAT = "tracelight-tabicl-csv-preprocessor-v1"
_PROTECTED_FORMAT = "tracelight-tabicl-csv-preprocessor-v2"
OFFICIAL_PROFILE = "official_single_view_v1"
PROTECTED_IDS_PROFILE = "nf_v3_protected_ids_v2"
PROTECTED_ID_DOMAINS = {
    "L4_SRC_PORT": (0, 65535),
    "L4_DST_PORT": (0, 65535),
    "PROTOCOL": (0, 255),
    # An explicit project acceptance domain, not a claimed publisher upper bound.
    "L7_PROTO": (0, 65535),
}


def require_pinned_preprocessing():
    source = require_pinned_tabicl()
    module = importlib.import_module("tabicl._sklearn.preprocessing")
    observed = {}
    root = Path(module.__file__).resolve().parent
    for name, expected in _PREPROCESSING_FILES.items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"TabICLv2 preprocessing source mismatch in {name}; expected {TABICL_COMMIT}")
        observed[name] = actual
    return module, {**source, "preprocessing_source_sha256": observed}


def _name(value, role, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{role} must be a nonempty column name without surrounding whitespace")
    return value


def _names(values, role):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{role} must be an ordered list of column names")
    result = tuple(_name(v, role) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{role} contains duplicate column names")
    return result


@dataclass(frozen=True)
class CSVSchema:
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    label_column: str = "label"
    id_column: str | None = "sample_id"
    group_column: str | None = None
    family_column: str | None = None
    label_mapping: dict[str, int] | None = None

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("CSV schema must be a dictionary")
        known = {"features", "numeric", "categorical", "label_column", "id_column",
                 "group_column", "family_column", "label_mapping"}
        if set(value) - known:
            raise ValueError(f"unknown CSV schema fields: {sorted(set(value) - known)}")
        features = _names(value.get("features"), "features")
        if not features:
            raise ValueError("schema must contain at least one feature")
        numeric = _names(value["numeric"], "numeric") if "numeric" in value else None
        categorical = _names(value["categorical"], "categorical") if "categorical" in value else None
        if numeric is None and categorical is None:
            numeric, categorical = features, ()
        elif numeric is None:
            numeric = tuple(v for v in features if v not in categorical)
        elif categorical is None:
            categorical = tuple(v for v in features if v not in numeric)
        if set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features):
            raise ValueError("numeric and categorical must partition the registered features")
        metadata = {
            "label_column": _name(value.get("label_column", "label"), "label_column"),
            "id_column": _name(value.get("id_column", "sample_id"), "id_column", optional=True),
            "group_column": _name(value.get("group_column"), "group_column", optional=True),
            "family_column": _name(value.get("family_column"), "family_column", optional=True),
        }
        used_metadata = [v for v in metadata.values() if v is not None]
        if len(set(used_metadata)) != len(used_metadata):
            raise ValueError("metadata roles must use different columns")
        if set(features) & set(used_metadata):
            raise ValueError("label, ID, group and family metadata cannot be model features")
        mapping = value.get("label_mapping")
        if mapping is not None:
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("label_mapping must be a nonempty raw-string to 0/1 dictionary")
            if any(not isinstance(k, str) or not k.strip() or type(v) is not int or v not in (0, 1)
                   for k, v in mapping.items()):
                raise ValueError("label_mapping must map nonempty raw strings to integer 0 or 1")
            mapping = dict(mapping)
        # All subsequent order derives from the single registered feature list.
        return cls(features, tuple(v for v in features if v in numeric),
                   tuple(v for v in features if v in categorical), **metadata, label_mapping=mapping)

    def to_dict(self):
        return {"features": list(self.features), "numeric": list(self.numeric),
                "categorical": list(self.categorical), "label_column": self.label_column,
                "id_column": self.id_column, "group_column": self.group_column,
                "family_column": self.family_column,
                "label_mapping": None if self.label_mapping is None else dict(self.label_mapping)}


def _columns(frame):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("CSV data must be a nonempty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("duplicate column headers are not allowed")


def frame_features(frame, schema):
    """Select registered features and enforce types without examining labels."""
    _columns(frame)
    missing = set(schema.features) - set(frame.columns)
    if missing:
        raise ValueError(f"missing registered feature columns: {sorted(missing)}")
    result = frame.loc[:, list(schema.features)].copy()
    for name in schema.numeric:
        try:
            values = pd.to_numeric(result[name], errors="raise").astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric value in feature {name}") from exc
        if np.isinf(values.to_numpy()).any():
            raise ValueError(f"infinite numeric value in feature {name}")
        result[name] = values
    for name in schema.categorical:
        # Preserve lexical category identity (e.g. 001 versus 1); np.nan is the
        # missing marker understood by the pinned sklearn OrdinalEncoder.
        result[name] = result[name].map(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
    return result


def labels(frame, schema):
    """Return canonical binary labels, mapping raw strings only when declared.

    CSV parsing retains raw labels as strings. Once read_csv assigns the mapped
    integer column, repeated calls see a numeric dtype and simply validate it.
    Thus a mapping for raw codes '0'/'1' can also safely reverse their meaning.
    """
    if schema.label_column not in frame.columns:
        raise ValueError(f"missing label column {schema.label_column}")
    column = frame[schema.label_column]
    if schema.label_mapping is not None and not pd.api.types.is_numeric_dtype(column.dtype):
        if column.isna().any() or column.astype(str).str.strip().eq("").any():
            raise ValueError("raw labels must be nonempty registered label_mapping keys")
        raw = column.astype(str)
        unknown = sorted(set(raw) - set(schema.label_mapping))
        if unknown:
            raise ValueError(f"raw labels are absent from label_mapping: {unknown[:5]}")
        return raw.map(schema.label_mapping).to_numpy(dtype=np.int64)
    try:
        result = pd.to_numeric(column, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be registered binary values normal=0 and attack=1") from exc
    if not np.isin(result, [0, 1]).all():
        raise ValueError("labels must be registered binary values normal=0 and attack=1")
    return result.astype(np.int64)


def read_csv(path, schema, require_label=True):
    """Read UTF-8 CSV without dtype inference or automatic category coercion.

    Empty fields are missing; strings such as NA and 001 remain literal values.
    The parser checks original headers before pandas can rename duplicates.
    Declared ID/group metadata must be present and nonempty for labelled
    experiment roles; IDs are unique within a file. Unlabelled prediction only
    requires features, and validates ID/group values when supplied. The caller
    supplies stable row keys when IDs are absent.
    """
    if not isinstance(schema, CSVSchema):
        schema = CSVSchema.from_dict(schema)
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
        if not header or any(not name or name != name.strip() for name in header):
            raise ValueError("CSV headers must be nonempty and have no surrounding whitespace")
        if len(set(header)) != len(header):
            raise ValueError("duplicate CSV headers are not allowed")
        rows = []
        for record in reader:
            if len(record) != len(header):
                raise ValueError(f"CSV row ending at line {reader.line_num} has the wrong number of fields")
            rows.append([np.nan if value == "" else value for value in record])
    frame = pd.DataFrame(rows, columns=header, dtype=object)
    typed = frame_features(frame, schema)
    for name in schema.features:
        frame[name] = typed[name]
    for role, name in (("ID", schema.id_column), ("group", schema.group_column)):
        if name is None:
            continue
        if name not in frame.columns:
            if not require_label:
                continue
            raise ValueError(f"missing registered {role} column {name}")
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"registered {role} values must be nonempty")
        frame[name] = frame[name].astype(str)
    if schema.id_column is not None and schema.id_column in frame and frame[schema.id_column].duplicated().any():
        raise ValueError("sample IDs must be unique within each CSV")
    if require_label or schema.label_column in frame.columns:
        frame[schema.label_column] = labels(frame, schema)
    if require_label and schema.family_column is not None and schema.family_column not in frame.columns:
        raise ValueError(f"missing registered family column {schema.family_column}")
    return frame


class CSVPreprocessor:
    """Fit official transform state on support rows; transform every later role."""

    def __init__(self, schema, profile=OFFICIAL_PROFILE):
        self.schema = schema if isinstance(schema, CSVSchema) else CSVSchema.from_dict(schema)
        if profile not in (OFFICIAL_PROFILE, PROTECTED_IDS_PROFILE):
            raise ValueError(f"unknown preprocessing profile: {profile}")
        self.profile = profile
        if profile == PROTECTED_IDS_PROFILE and set(self.schema.categorical) != set(PROTECTED_ID_DOMAINS):
            raise ValueError("protected-ID profile requires exactly the four NF-v3 identifier categoricals")

    def _protected_ids(self, features):
        """Canonical integer identity; Decimal prevents tiny fractions rounding in."""
        output = np.empty((len(features), len(self.schema.categorical)), dtype=np.float64)
        for index, name in enumerate(self.schema.categorical):
            lower, upper = PROTECTED_ID_DOMAINS[name]
            for row_index, value in enumerate(features[name]):
                try:
                    number = Decimal(str(value))
                    valid = (number.is_finite() and number == number.to_integral_value()
                             and lower <= number <= upper)
                except (InvalidOperation, ValueError, TypeError):
                    valid = False
                if not valid:
                    raise ValueError(f"protected ID {name} must be a nonmissing integer in [{lower}, {upper}]")
                output[row_index, index] = int(number)
        return output

    def _verify_protected_domains(self):
        """Check the actual fitted affine transform over every accepted integer."""
        evidence = {}
        for index, name in enumerate(self.schema.categorical):
            lower, upper = PROTECTED_ID_DOMAINS[name]
            domain = np.arange(lower, upper + 1, dtype=np.float64)
            encoded = ((domain - self.protected_mean_[index]) / self.protected_scale_[index]).astype(np.float32)
            reconstructed = encoded.astype(np.float64) * self.protected_scale_[index] + self.protected_mean_[index]
            unique = np.isfinite(encoded).all() and np.unique(encoded).size == domain.size
            recoverable = np.array_equal(np.rint(reconstructed), domain)
            if not unique or not recoverable:
                raise ValueError(f"protected ID {name} loses identity after FP32 conversion")
            evidence[name] = {"domain": [lower, upper], "domain_size": int(domain.size),
                              "fp32_unique": True, "rounded_inverse_exact": True,
                              "max_inverse_absolute_error": float(np.max(np.abs(reconstructed - domain)))}
        self.protected_verification_ = evidence

    def _fit_protected(self, features, official):
        identifiers = self._protected_ids(features)
        self.protected_mean_ = identifiers.mean(axis=0)
        standard_deviation = identifiers.std(axis=0)
        self.protected_scale_ = np.where(standard_deviation > 0, standard_deviation, 1.0)
        self.protected_support_constant_ = standard_deviation == 0
        self._verify_protected_domains()
        self.encoder_ = self.feature_filter_ = self.pipeline_ = None
        numeric_output = ()
        if self.schema.numeric:
            numeric = features.loc[:, list(self.schema.numeric)]
            self.encoder_ = official.TransformToNumerical(verbose=False).fit(numeric)
            encoded = self.encoder_.transform(numeric)
            self.feature_filter_ = official.UniqueFeatureFilter().fit(encoded)
            # Unlike the legacy all-constant fallback, the four protected ID
            # channels already guarantee nonzero input width.
            if all(np.unique(encoded[:, i]).size <= 1 for i in range(encoded.shape[1])):
                self.feature_filter_.features_to_keep_ = np.zeros(encoded.shape[1], dtype=bool)
                self.feature_filter_.n_features_out_ = 0
            mask = self.feature_filter_.features_to_keep_
            numeric_output = tuple(name for name, keep in zip(self.schema.numeric, mask) if keep)
            if numeric_output:
                filtered = self.feature_filter_.transform(encoded)
                self.pipeline_ = official.PreprocessingPipeline(
                    normalization_method="none", outlier_threshold=4.0, random_state=0,
                ).fit(filtered)
                del self.pipeline_.X_transformed_
        self.output_features_ = (*self.schema.categorical, *numeric_output)

    def fit(self, support_frame):
        official, source = require_pinned_preprocessing()
        features = frame_features(support_frame, self.schema)
        if self.profile == PROTECTED_IDS_PROFILE:
            self._fit_protected(features, official)
            self.source_ = source
            self.support_rows_ = len(features)
            self.encoded_features_ = (*self.schema.categorical, *self.schema.numeric)
            self.n_features_out_ = len(self.output_features_)
            return self
        encoder = official.TransformToNumerical(verbose=False).fit(features)
        encoded = encoder.transform(features)
        feature_filter = official.UniqueFeatureFilter().fit(encoded)
        filtered = feature_filter.transform(encoded)
        pipeline = official.PreprocessingPipeline(
            normalization_method="none", outlier_threshold=4.0, random_state=0,
        ).fit(filtered)
        # transform() needs fitted statistics, not this saved support matrix.
        del pipeline.X_transformed_
        self.encoder_ = encoder
        self.feature_filter_ = feature_filter
        self.pipeline_ = pipeline
        self.source_ = source
        self.support_rows_ = len(features)
        # Official ColumnTransformer emits categoricals, then numeric columns.
        self.encoded_features_ = (*self.schema.categorical, *self.schema.numeric)
        self.output_features_ = tuple(name for name, keep in zip(
            self.encoded_features_, feature_filter.features_to_keep_) if keep)
        self.n_features_out_ = len(self.output_features_)
        return self

    def _fitted(self):
        if not hasattr(self, "pipeline_"):
            raise RuntimeError("fit the CSV preprocessor on support rows first")

    def transform(self, frame):
        self._fitted()
        features = frame_features(frame, self.schema)
        if self.profile == PROTECTED_IDS_PROFILE:
            identifiers = self._protected_ids(features)
            output = ((identifiers - self.protected_mean_) / self.protected_scale_).astype(np.float32)
            if self.pipeline_ is not None:
                encoded = self.encoder_.transform(features.loc[:, list(self.schema.numeric)])
                filtered = self.feature_filter_.transform(encoded)
                numeric = self.pipeline_.transform(filtered).astype(np.float32)
                output = np.concatenate((output, numeric), axis=1)
        else:
            encoded = self.encoder_.transform(features)
            filtered = self.feature_filter_.transform(encoded)
            output = self.pipeline_.transform(filtered).astype(np.float32)
        if output.ndim != 2 or output.shape[1] != self.n_features_out_ or not np.isfinite(output).all():
            raise ValueError("official preprocessing did not produce finite registered float32 features")
        return np.ascontiguousarray(output)

    def report(self):
        self._fitted()
        result = {"method": "TabICLv2-2026-official-single-view",
                "profile": self.profile,
                "normalization_method": "none", "feature_shuffle": "none", "class_shuffle": "none",
                "support_rows": self.support_rows_, "input_features": list(self.schema.features),
                "encoded_feature_order": list(self.encoded_features_),
                "output_features": list(self.output_features_),
                "removed_constant_features": [v for v in self.encoded_features_ if v not in self.output_features_],
                "n_features_in": len(self.schema.features), "n_features_out": self.n_features_out_,
                "stored_support_rows": False, "source": self.source_}
        if self.profile == PROTECTED_IDS_PROFILE:
            result["method"] = "TabICLv2-2026-nf-v3-protected-ids"
            result["protected_ids"] = {
                "columns": list(self.schema.categorical),
                "transform": "P-fitted mean/std affine; constant std uses 1; no clipping or removal",
                "mean": self.protected_mean_.tolist(), "scale": self.protected_scale_.tolist(),
                "support_constant": self.protected_support_constant_.tolist(),
                "domain_verification": self.protected_verification_,
                "l7_domain_scope": "project acceptance domain; verify full source at laboratory preparation",
                "scope": "identifier identity preservation only; not a detection or categorical-distance guarantee",
            }
        return result

    def save(self, path):
        self._fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"format": _FORMAT, "schema": self.schema.to_dict(), "encoder": self.encoder_,
                 "feature_filter": self.feature_filter_, "pipeline": self.pipeline_, "report": self.report()}
        if self.profile == PROTECTED_IDS_PROFILE:
            state.update({"format": _PROTECTED_FORMAT, "profile": self.profile,
                          "protected_mean": self.protected_mean_, "protected_scale": self.protected_scale_,
                          "protected_support_constant": self.protected_support_constant_})
        joblib.dump(state, path, compress=3)
        return path

    @classmethod
    def load(cls, path, *, expected_sha256=None):
        _, source = require_pinned_preprocessing()
        path = Path(path)
        if expected_sha256 is not None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_sha256.lower():
                raise ValueError("CSV preprocessor SHA-256 does not match the registered state")
        state = joblib.load(path)
        if state.get("format") not in (_FORMAT, _PROTECTED_FORMAT) or state["report"]["source"] != source:
            raise ValueError("CSV preprocessor format or pinned source does not match")
        profile = state.get("profile", OFFICIAL_PROFILE)
        if (state["format"] == _PROTECTED_FORMAT) != (profile == PROTECTED_IDS_PROFILE):
            raise ValueError("CSV preprocessor format/profile mismatch")
        if state["report"].get("profile", OFFICIAL_PROFILE) != profile:
            raise ValueError("CSV preprocessor report/profile mismatch")
        result = cls(CSVSchema.from_dict(state["schema"]), profile=profile)
        result.encoder_ = state["encoder"]
        result.feature_filter_ = state["feature_filter"]
        result.pipeline_ = state["pipeline"]
        if hasattr(result.pipeline_, "X_transformed_"):
            raise ValueError("deployed preprocessing state must not contain support rows")
        result.source_ = source
        result.support_rows_ = state["report"]["support_rows"]
        result.encoded_features_ = tuple(state["report"]["encoded_feature_order"])
        result.output_features_ = tuple(state["report"]["output_features"])
        result.n_features_out_ = len(result.output_features_)
        if profile == PROTECTED_IDS_PROFILE:
            result.protected_mean_ = np.asarray(state["protected_mean"], dtype=np.float64)
            result.protected_scale_ = np.asarray(state["protected_scale"], dtype=np.float64)
            result.protected_support_constant_ = np.asarray(state["protected_support_constant"], dtype=bool)
            if (result.protected_mean_.shape != (4,) or result.protected_scale_.shape != (4,)
                    or result.protected_support_constant_.shape != (4,)
                    or not np.isfinite(result.protected_mean_).all()
                    or not np.isfinite(result.protected_scale_).all() or not (result.protected_scale_ > 0).all()):
                raise ValueError("invalid protected-ID affine state")
            if result.output_features_[:4] != result.schema.categorical:
                raise ValueError("protected identifier columns must remain first and present")
            result._verify_protected_domains()
        return result
