"""F-only inputs and fixed calibration-transfer episodes; no model imports.

The frozen files are bound by bytes. Input inspection deserializes F records
only; calibration and development records are not materialized by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from ids.model_training.preprocessing import TypedPreprocessor


BATCH = Path("docs/coverage_basis_20261119/batch")
BOUND_INPUTS = {
    "rows.jsonl.gz": "8822c9ff41b5172d4024a15c84a0d96ebfa9d61fe6dbb3845b1888851ee6561c",
    "cohorts.json": "358f8ecc0da25d9be8fc7a553a880838aa8ce39a22f70981a17ab8e346b33277",
    "events.jsonl.gz": "35f374ac82107456a3495523f4140ae6b86b53327f0aec7f7b03c2a675b0186d",
    "normal_units.jsonl.gz": "e1d7f3a71626b9586ae66d0b5628f8f111a7ce186028278e2dd5ec31b02b1551",
}
F_ROLE = re.compile(r'"(?:dataset_role|role)"\s*:\s*"F_(?:base|expansion)"')
EXPECTED_EPISODES = [
    ("ait:santos:complete_run", "2022-01-15", 119, 64),
    ("ait:santos:complete_run", "2022-01-16", 183, 113),
    ("ait:santos:complete_run", "2022-01-17", 296, 143),
    ("unraveled:apt_2021_single_campaign", "2021-06-22", 10110, 2198),
]
SEEDS = (17, 29, 43)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_inputs(repo):
    result = {}
    for name, expected in BOUND_INPUTS.items():
        path = Path(repo) / BATCH / name
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f"Frozen input changed: {name}")
        result[name] = {"path": (BATCH / name).as_posix(), "sha256": actual,
                        "bytes": path.stat().st_size}
    return result


def read_f_records(path):
    """Filter the serialized role before JSON parsing; hash binding is separate."""
    result = []
    with gzip.open(path, "rt", encoding="utf8") as stream:
        for line in stream:
            if not F_ROLE.search(line):
                continue
            row = json.loads(line)
            if row.get("dataset_role", row.get("role")) not in ("F_base", "F_expansion"):
                raise ValueError("Role filter selected a non-F record")
            result.append(row)
    return result


def load_fit(repo):
    """Return exactly the fixed 13,252 F members in canonical identity order."""
    bind_inputs(repo)
    all_f = read_f_records(Path(repo) / BATCH / "rows.jsonl.gz")
    rows = sorted((r for r in all_f if r["fit_expanded_member"]), key=lambda r: r["flow_id"])
    if len(all_f) != 13386 or len(rows) != 13252:
        raise ValueError("Frozen F counts differ")
    if Counter(r["target_kind"] for r in rows) != {"normal": 12747, "control": 504, "download": 1}:
        raise ValueError("Frozen F target counts differ")
    if any(r["label"] != r["target_label"] or r["label"] not in (0, 1) for r in rows):
        raise ValueError("F requires binary target labels; unknown is not normal")
    if len({r["flow_id"] for r in rows}) != len(rows) or len({r["stable_flow_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate physical F identity")
    if any(r["actual_available_at"] is not None for r in rows):
        raise ValueError("Unexpected label/detector arrival contract change")
    episodes = build_episodes(rows)
    actual = [(e["source"], e["E_day"], len(e["Q"]), len(e["E"])) for e in episodes]
    if actual != EXPECTED_EPISODES:
        raise ValueError("F calendar episodes differ from the fixed design")
    return rows


def _day(row):
    value = row["replay_export_time"]
    if value is None or not np.isfinite(value):
        raise ValueError("Every F row requires its offline export time")
    return datetime.fromtimestamp(value, timezone.utc).date().isoformat()


def identity_permutation(rows, seed):
    """Position -> physical identity, one independent bijection per source.

    Integer seeds are derived by SHA-256, then NumPy PCG64. All appearances of a
    position share this mapping. This is not independent reshuffling per episode.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("Permutation seed must be a uint32 integer")
    source_normals = defaultdict(list)
    for index, row in enumerate(rows):
        if row["label"] == 0:
            source_normals[row["source_group"]].append(index)
    mapping = np.arange(len(rows), dtype=np.int64)
    details = []
    for source, positions in sorted(source_normals.items()):
        derived = int.from_bytes(hashlib.sha256(f"calibration-transfer:{seed}:{source}".encode()).digest()[:8], "little")
        permutation = np.random.Generator(np.random.PCG64(derived)).permutation(positions).tolist()
        mapping[positions] = permutation
        details.append({"source": source, "derived_seed": derived,
                        "positions": positions, "identities": permutation})
    return mapping, details


def build_episodes(rows, seed=None):
    """Q contains all earlier same-source normals; E is one UTC day of F."""
    ids = [r["flow_id"] for r in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Episode indices require unique rows sorted by flow_id")
    grouped = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        if row["label"] == 0:
            grouped[row["source_group"]][_day(row)].append(index)
    mapping = np.arange(len(rows)) if seed is None else identity_permutation(rows, seed)[0]
    episodes = []
    for source, days in sorted(grouped.items()):
        prior = []
        for day, current in sorted(days.items()):
            if prior:
                episodes.append({"id": source + "|" + day, "source": source, "E_day": day,
                                 "Q": mapping[prior].tolist(), "E": mapping[current].tolist(),
                                 "E_budget": len(current) // 100})
            prior.extend(current)
    return episodes


def frame_from_rows(rows):
    """Only raw measurements enter the encoder; identities and time stay metadata."""
    return pd.DataFrame([r["raw8"] for r in rows])


def _port_missing(rows):
    values = np.asarray([[r["raw8"].get(name) for name in ("orig_port", "resp_port")]
                         for r in rows], dtype=np.float64).reshape(-1, 2)
    missing = np.isnan(values) | (values == -1)
    valid = missing | (np.isfinite(values) & (values >= 0) & (values <= 65535) & (values == np.floor(values)))
    if not valid.all():
        raise ValueError("Ports require integers 0..65535, -1 or missing")
    return missing.astype(np.int64)


@dataclass
class Preprocessor:
    """Existing typed PLE input plus two retained port-missing indicators.

    Finite port values are discarded. Numeric F medians and log1p means/scales
    are shared by every arm and seed; missing indicators never enter PLE.
    """
    base: TypedPreprocessor

    @classmethod
    def fit(cls, rows):
        if not rows or any(r.get("dataset_role") not in ("F_base", "F_expansion") for r in rows):
            raise ValueError("Preprocessing may be fitted on F rows only")
        _port_missing(rows)
        return cls(TypedPreprocessor.fit(frame_from_rows(rows)))

    @property
    def cat_cardinalities(self):
        return [*self.base.cat_cardinalities, 2, 2]

    def transform(self, rows):
        numeric, categorical = self.base.transform(frame_from_rows(rows))
        return numeric, np.ascontiguousarray(np.column_stack((categorical, _port_missing(rows))), dtype=np.int64)

    def to_dict(self):
        return {"format": "calibration-transfer-ports-zero-v1", "base": self.base.to_dict(),
                "port_policy": "finite_values_discarded_missingness_retained",
                "categorical_cardinalities": self.cat_cardinalities}

    @classmethod
    def from_dict(cls, value):
        if (value.get("format") != "calibration-transfer-ports-zero-v1"
                or value.get("port_policy") != "finite_values_discarded_missingness_retained"):
            raise ValueError("Unsupported calibration-transfer preprocessing")
        result = cls(TypedPreprocessor.from_dict(value["base"]))
        if value.get("categorical_cardinalities") != result.cat_cardinalities:
            raise ValueError("Categorical schema differs")
        return result


def _component_permutation_audit(rows, mapping, components):
    index = {r["flow_id"]: i for i, r in enumerate(rows)}
    inverse = np.argsort(mapping)
    details = []
    for component in components:
        positions = [int(inverse[index[fid]]) for fid in component["normal_flow_ids"]]
        days = sorted({_day(rows[i]) for i in positions})
        details.append({"component_id": component["component_id"], "normal_count": len(positions),
                        "assigned_days": days, "split_across_assigned_days": len(days) > 1})
    return {"known_components": len(details), "split_across_assigned_days": sum(x["split_across_assigned_days"] for x in details),
            "details": details, "unraveled_action_partition_available": False}


def prepare_inputs(repo):
    """Materialize F encodings/episodes without model or held-out-data access."""
    rows = load_fit(repo)
    preprocessor = Preprocessor.fit(rows)
    numeric, categorical = preprocessor.transform(rows)
    evidence_path = Path(repo) / "docs/normal_tail_mechanism_20261120/episodes_evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf8"))
    components = evidence["normal_action_components"]
    actual = build_episodes(rows)
    permutations = {}
    for seed in SEEDS:
        mapping, details = identity_permutation(rows, seed)
        permutations[str(seed)] = {"episodes": build_episodes(rows, seed), "source_bijections": details,
                                  "known_action_components": _component_permutation_audit(rows, mapping, components)}
    manifest = {
        "format": "calibration-transfer-inputs-v1", "bindings": bind_inputs(repo),
        "F_evidence_sha256": file_digest(evidence_path),
        "fit_flow_ids": [r["flow_id"] for r in rows], "fit_rows": len(rows),
        "target_counts": dict(Counter(r["target_kind"] for r in rows)),
        "excluded_F_rows": {"unknown": 123, "other_attack": 11},
        "feature_clock": "offline complete-flow replay_export_time",
        "online_label_availability_established": False,
        "actual_episodes": actual, "permuted_episodes_by_seed": permutations,
        "preprocessor": preprocessor.to_dict(),
        "numeric_shape": list(numeric.shape), "categorical_shape": list(categorical.shape),
        "numeric_sha256": hashlib.sha256(numeric.tobytes()).hexdigest(),
        "categorical_sha256": hashlib.sha256(categorical.tobytes()).hexdigest(),
        "parsed_data_roles": ["F_base", "F_expansion"],
        "model_constructions": 0, "fit_calls": 0, "score_calls": 0, "threshold_calculations": 0,
        "permutation_interpretation": "Preserves sizes and reuse-count multiset, not each physical identity's reuse count. Known action components can split; C-B measures temporal/business correspondence, not a pure causal time arrow.",
    }
    manifest["fit_flow_ids_sha256"] = json_digest(manifest["fit_flow_ids"])
    return rows, preprocessor, numeric, categorical, manifest
