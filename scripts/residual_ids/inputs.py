"""Read the bounded, already-open development comparison input."""
from __future__ import annotations

import gzip
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path

from scripts.ait_forward.decisions import TARGETS
from scripts.coverage_batch.materialize import RAW8_FIELDS


SCHEMA = "tracelight-residual-rescue-input-v1"
SCOPE = "opened_development_only"
REQUIRED_ARTIFACTS = {"fit_rows.jsonl.gz", "calibration_rows.jsonl.gz", "evaluation_rows.jsonl.gz", "memberships.json"}
BENCHMARK_ARTIFACT = "benchmark_rows.jsonl.gz"


class Bundle:
    def __init__(self, directory):
        self.directory = Path(directory)
        manifest = (self.directory / "manifest.json").read_bytes()
        self.digest = sha256(manifest).hexdigest()
        self.manifest = json.loads(manifest)
        if self.manifest["schema"] != SCHEMA or self.manifest["scope"] != SCOPE:
            raise ValueError("Only the opened development input contract is supported")
        if not REQUIRED_ARTIFACTS <= self.manifest["artifacts"].keys():
            raise ValueError("Bundle manifest omits a required artifact")

    def content(self, name):
        artifact = self.manifest["artifacts"][name]
        raw = (self.directory / name).read_bytes()
        if sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Changed input artifact: {name}")
        if "bytes" in artifact and len(raw) != artifact["bytes"]:
            raise ValueError(f"Changed input size: {name}")
        return raw

    def rows(self, name, roles):
        rows = [json.loads(line) for line in gzip.decompress(self.content(name)).splitlines()]
        if "rows" in self.manifest["artifacts"][name] and len(rows) != self.manifest["artifacts"][name]["rows"]:
            raise ValueError(f"Changed row count: {name}")
        validate_rows(rows, roles)
        return rows


def validate_rows(rows, roles):
    if not rows:
        raise ValueError("Expected nonempty canonical row source")
    for row in rows:
        if row["dataset_role"] not in roles or "harrison" in row["source_run"].casefold():
            raise ValueError("Role or capture source outside the opened protocol")
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ("flow_id", "stable_flow_id", "source_run", "allocation_group")):
            raise ValueError("Canonical row requires nonempty source and identity fields")
        if row["label"] is not None and type(row["label"]) is not int:
            raise ValueError("Row labels must be integers or null")
        kind = row["target_kind"]
        if (
            row["label"] not in (0, 1, None)
            or (row["label"] == 1 and kind not in TARGETS)
            or (row["label"] == 0 and kind != "normal")
            or (row["label"] is None and kind != "unknown")
        ):
            raise ValueError("Invalid known/unknown row label or target")
        raw = row["raw8"]
        if set(raw) != set(RAW8_FIELDS) or len(row["features8"]) != len(RAW8_FIELDS):
            raise ValueError("Both canonical scorer input representations are required")
        for values in (raw.values(), row["features8"]):
            if any(value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                          or not isfinite(value) or value < 0) for value in values):
                raise ValueError("Feature values must be finite nonnegative scalars or null")
    for key in ("flow_id", "stable_flow_id"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"Repeated canonical identity: {key}")


def ensure_disjoint(*sources):
    for key in ("flow_id", "stable_flow_id", "allocation_group", "source_run"):
        sets = [{row[key] for row in rows} for rows in sources]
        if sum(map(len, sets)) != len(set().union(*sets)):
            raise ValueError(f"Cross-role overlap in {key}")


def validate_evaluations(rows, evaluations):
    index = {row["flow_id"]: row for row in rows}
    assigned = set()
    for name, cohort in evaluations.items():
        ids = cohort["flow_ids"]
        members = set(ids)
        if not members or len(members) != len(ids) or members - index.keys() or members & assigned:
            raise ValueError(f"Invalid or overlapping evaluation identities: {name}")
        assigned.update(members)
        normals = [fid for unit in cohort["normal_units"].values() for fid in unit]
        if len(set(normals)) != len(normals) or set(normals) != {
            fid for fid in members if index[fid]["label"] == 0
        }:
            raise ValueError(f"Normal units must partition all positively identified normal rows: {name}")
        attacks = cohort["attack_ids"]
        if set(attacks) != set(TARGETS):
            raise ValueError("Evaluation target registry must identify both target dispositions")
        for target in TARGETS:
            registered = attacks[target] or []
            if len(set(registered)) != len(registered) or set(registered) != {
                fid for fid in members if index[fid]["label"] == 1 and index[fid]["target_kind"] == target
            }:
                raise ValueError(f"Invalid confirmed {target} membership: {name}")
        events = cohort["events"]
        if len({event["event_id"] for event in events}) != len(events):
            raise ValueError(f"Duplicate event identity: {name}")
        for event in events:
            if (event["label"] != 1 or event["kind"] not in TARGETS
                    or set(event["candidate_flow_ids"] + event.get("confirmed_flow_ids", [])) - members):
                raise ValueError(f"Invalid event scope: {name}")
    if assigned != index.keys():
        raise ValueError("Evaluation suites must partition all retained opened rows")
