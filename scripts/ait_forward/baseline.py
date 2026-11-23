"""The frozen HGB and opened archive evidence used by the AIT forward protocol."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from hashlib import sha256
from importlib.metadata import version
import io
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import numpy as np


MODEL_ID = "HGB_narrow_ports_zero"
MODEL_MEMBER = f"fits/{MODEL_ID}/model.joblib"
MODEL_SHA256 = "584e39ae1af4491abea9415233b2e6c35c63b10594cc0672ac344038b501e278"
ARCHIVE_PREFIX = "comprehensive-20261122-results/"
ARCHIVE_SHA256 = "16301653952d4bce619002456044e5a994701779db9083cf9bb94c9295d7c5f5"
SKLEARN_VERSION = "1.5.1"
ROWS_PATH = Path("docs/coverage_basis_20261119/batch/rows.jsonl.gz")
ROWS_SHA256 = "8822c9ff41b5172d4024a15c84a0d96ebfa9d61fe6dbb3845b1888851ee6561c"
REPO = Path(__file__).resolve().parents[2]

_IDENTITIES = "evaluation/common_identities.jsonl.gz"
_UNIT_DEFINITIONS = "evaluation/normal_unit_definitions.jsonl.gz"
_SCORES = f"scores/{MODEL_ID}.npz"
_SUMMARY = f"evaluation/models/{MODEL_ID}/summary.json"
_MEMBERS = {
    MODEL_MEMBER: (218848, MODEL_SHA256),
    _SCORES: (76086, "81a3098ea51af4dee9718e7695cc47330bea35126d9fb9fb802b9739fd6bf508"),
    "score_identities.json": (2692961, "673122a85ff3cf2f1b25c7569c16c47bf6fb73ec8b5693ace1cde9b1e75508ee"),
    _IDENTITIES: (1579382, "fa7e5707af88e9b6fcb7336388355a144f23d948ccf18ce63fda1de4e04151db"),
    _UNIT_DEFINITIONS: (939381, "c300f61aa608a8c3d6fdb7c2eaeea713a89b990a2dd34a1c7631737efe460087"),
    _SUMMARY: (3309, "d9372e3ba382e4d2cd751ef51e93d085966564cee00daee39eaa4fc16b88f4dd"),
}


def _verified_members(archive):
    with ZipFile(archive) as bundle:
        counts = {}
        for name in bundle.namelist():
            counts[name] = counts.get(name, 0) + 1
        result = {}
        for member, (size, digest) in _MEMBERS.items():
            name = ARCHIVE_PREFIX + member
            if counts.get(name) != 1 or bundle.getinfo(name).file_size != size:
                raise ValueError(f"Missing, ambiguous or incorrectly sized archive member: {member}")
            data = bundle.read(name)
            if sha256(data).hexdigest() != digest:
                raise ValueError(f"Frozen archive member hash differs: {member}")
            result[member] = data
    return result


def _retained_rows(data, key, roles):
    with gzip.open(io.BytesIO(data), "rb") as stream:
        pattern = _role_pattern(key, roles)
        return [json.loads(line) for line in stream if pattern.search(line)]


def _role_pattern(key, roles):
    alternatives = b"|".join(re.escape(role.encode("ascii")) for role in roles)
    return re.compile(rb'"' + key.encode("ascii") + rb'"\s*:\s*"(?:' + alternatives + rb')"')


def _historical_fox(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != ROWS_SHA256:
            raise ValueError("Opened fox feature source hash differs")
        stream.seek(0)
        pattern = _role_pattern("dataset_role", ("V",))
        with gzip.GzipFile(fileobj=stream) as archive:
            rows = [json.loads(line) for line in archive if pattern.search(line)]
    if any(row["dataset_role"] != "V" or row["allocation_group"] != "ait:fox:complete_run" for row in rows):
        raise ValueError("Historical feature source includes a row outside opened fox")
    return rows


def _counts(rows, alerts, units):
    labels = {row["flow_id"]: row["label"] for row in rows}
    normal = {identity for identity, label in labels.items() if label == 0}
    controls = {row["flow_id"] for row in rows if row["label"] == 1 and row["target_kind"] == "control"}
    downloads = {row["flow_id"] for row in rows if row["label"] == 1 and row["target_kind"] == "download"}
    return {
        "normal": len(normal),
        "normal_fp": len(normal & alerts),
        "control": len(controls),
        "control_tp": len(controls & alerts),
        "download": len(downloads),
        "download_tp": len(downloads & alerts),
        "unknown": sum(label is None for label in labels.values()),
        "normal_units": {
            unit: {"normal": len(ids), "fp": len(set(ids) & alerts), "budget": len(ids) // 100}
            for unit, ids in units.items()
        },
    }


@dataclass(frozen=True)
class FrozenBaseline:
    model: Any
    model_bytes: bytes
    pooled_threshold: float
    calibration_scores: dict[str, float]
    calibration_units: dict[str, tuple[str, ...]]
    known_flow_ids: frozenset[str]
    known_stable_flow_ids: frozenset[str]
    last_calibration_time: float
    archived_fox_scores: dict[str, float]
    archived_fox_identities: tuple[dict, ...]
    archived_fox_units: dict[str, tuple[str, ...]]
    archived_fox_summary: dict

    @property
    def model_sha256(self):
        return MODEL_SHA256

    @property
    def archive_member_sha256(self):
        return {name: digest for name, (_, digest) in _MEMBERS.items()}

    def score(self, rows):
        """Use the exact coverage-batch feature order and finite-port masking."""
        from scripts.coverage_batch.materialize import RAW8_FIELDS
        from scripts.coverage_batch.run import features
        from threadpoolctl import threadpool_limits

        if not rows:
            raise ValueError("Scoring requires at least one input row")
        feature_rows = [
            row if "features8" in row else {"features8": [row["raw8"][field] for field in RAW8_FIELDS]}
            for row in rows
        ]
        with threadpool_limits(limits=4):
            scores = np.asarray(self.model.decision_function(features(feature_rows, True)), dtype=np.float64)
        if scores.shape != (len(rows),) or not np.isfinite(scores).all():
            raise ValueError("Frozen HGB did not produce one finite scalar score per row")
        return scores

    def historical_fox_regression(self, rules, repo=REPO):
        """Validate exact CPU scoring and the old and new rules on opened fox."""
        thresholds = {name: float(rules[name]["threshold"]) for name in ("pooled", "business_max")}
        if thresholds["pooled"] != self.pooled_threshold or not np.isfinite(list(thresholds.values())).all():
            raise ValueError("Regression rules require the original pooled threshold and finite scores")
        by_id = {row["flow_id"]: row for row in _historical_fox(Path(repo) / ROWS_PATH)}
        if set(by_id) != self.archived_fox_scores.keys() or len(by_id) != len(self.archived_fox_identities):
            raise ValueError("Opened fox feature identities differ from the archive")
        ordered = [by_id[row["flow_id"]] for row in self.archived_fox_identities]
        identity_fields = ("stable_flow_id", "label", "target_kind", "replay_export_time")
        if any(left[field] != right[field] for left, right in zip(ordered, self.archived_fox_identities)
               for field in identity_fields):
            raise ValueError("Opened fox feature labels, times or stable identities differ from the archive")
        saved = np.array(list(self.archived_fox_scores.values()), dtype=np.float64)
        actual = self.score(ordered)
        if not np.array_equal(actual, saved):
            raise ValueError("Loaded CPU model does not reproduce archived fox scores exactly")
        results = {}
        for name, threshold in thresholds.items():
            alarms = {row["flow_id"] for row, value in zip(ordered, actual) if value > threshold}
            results[name] = _counts(ordered, alarms, self.archived_fox_units) | {"threshold": threshold}
        old = results["pooled"]
        if (old["normal_fp"], old["control_tp"] + old["download_tp"], old["unknown"]) != (
            self.archived_fox_summary["FP"], self.archived_fox_summary["TP"],
            self.archived_fox_summary["unknown_label_rows"],
        ):
            raise ValueError("Original pooled decisions differ from the archived fox report")
        return {"role": "opened_fox_development_regression", "rows": len(ordered),
                "cpu_scores_exactly_equal": True, "rules": results}


def load_baseline(archive):
    """Read pinned members and load the verified HGB without performing a forward pass."""
    from joblib import load
    from sklearn.ensemble import HistGradientBoostingClassifier

    if version("scikit-learn") != SKLEARN_VERSION:
        raise RuntimeError(f"The frozen HGB requires scikit-learn {SKLEARN_VERSION}")
    members = _verified_members(archive)
    ids = json.loads(members["score_identities.json"])
    with np.load(io.BytesIO(members[_SCORES]), allow_pickle=False) as data:
        values = data["scores"]
    if len(set(ids)) != len(ids) or values.shape != (len(ids),) or not np.isfinite(values).all():
        raise ValueError("Archived score identities and scores do not form a finite one-to-one mapping")
    scored = dict(zip(ids, map(float, values)))
    identities = [json.loads(line) for line in gzip.decompress(members[_IDENTITIES]).splitlines()]
    stable = frozenset(row["stable_flow_id"] for row in identities)
    if [row["flow_id"] for row in identities] != ids or None in stable or len(stable) != len(ids):
        raise ValueError("Archived stable identities differ from the scored order")
    c_rows = [row for row in identities if row["dataset_role"] == "C"]
    if len(c_rows) != 3475 or any(row["label"] != 0 for row in c_rows):
        raise ValueError("Original C must contain the registered normal-only cohort")
    c_scores = {row["flow_id"]: scored[row["flow_id"]] for row in c_rows}
    c_time = np.array([row["replay_export_time"] for row in c_rows], dtype=np.float64)
    if not np.isfinite(c_time).all():
        raise ValueError("Original calibration rows need finite observation times")
    units = _retained_rows(members[_UNIT_DEFINITIONS], "role", ("C", "V"))
    business = {unit["unit_id"]: tuple(unit["flow_ids"]) for unit in units
                if unit["role"] == "C" and unit.get("view") == "business"}
    members_of_business = [identity for ids_here in business.values() for identity in ids_here]
    if len(business) != 3 or len(members_of_business) != len(c_scores) or set(members_of_business) != c_scores.keys():
        raise ValueError("Original C business units must partition the entire calibration cohort")

    summary = json.loads(members[_SUMMARY])
    model_metadata, rule = summary["model"], summary["threshold"]
    if model_metadata["sha256"] != MODEL_SHA256 or model_metadata["snapshot_id"] != MODEL_ID:
        raise ValueError("Archived model summary does not bind the required HGB")
    pooled = sorted(c_scores.values())[-(len(c_scores) // 100 + 1)]
    if rule["comparison"] != ">" or rule["threshold"] != pooled:
        raise ValueError("Archived C rule differs from its frozen score order")
    fox = tuple(row for row in identities if row["dataset_role"] == "V")
    fox_scores = {row["flow_id"]: scored[row["flow_id"]] for row in fox}
    fox_units = {unit["unit_id"]: tuple(unit["flow_ids"]) for unit in units if unit["role"] == "V"}
    fox_summary = next(row for row in summary["by_role"] if row["role"] == "V")

    model = load(io.BytesIO(members[MODEL_MEMBER]))
    if type(model) is not HistGradientBoostingClassifier or model.n_features_in_ != 8:
        raise ValueError("Verified artifact does not contain the registered eight-field HGB")
    return FrozenBaseline(model, members[MODEL_MEMBER], pooled, c_scores, business, frozenset(ids),
                          stable, float(c_time.max()),
                          fox_scores, fox, fox_units, fox_summary)
