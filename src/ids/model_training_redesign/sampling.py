"""Four independent cyclic queues; no replacement inside a queue cycle."""
from __future__ import annotations

import hashlib
import json

import numpy as np


GROUPS = ("A0", "A1", "B0", "B1")
IDENTITY = ["source_id", "row_id"]


def derived_seed(seed, stream):
    return int.from_bytes(hashlib.sha256(f"redesign-v1:{int(seed)}:{stream}".encode()).digest()[:8],
                          "little") % (2**63 - 1)


def canonical_frame(frame):
    if frame[IDENTITY].isna().any().any() or frame.duplicated(IDENTITY).any():
        raise ValueError("Expected unique nonmissing (source_id,row_id) identities")
    ids = frame.row_id.to_numpy()
    if not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any():
        raise ValueError("row_id must be a nonnegative integer for numeric ordering")
    return frame.sort_values(IDENTITY, kind="stable").reset_index(drop=True)


def group_indices(frame):
    if not frame.profile.isin(["A", "B"]).all() or not frame.label.isin([0, 1]).all():
        raise ValueError("Training requires A/B profiles and binary labels")
    result = {}
    for group in GROUPS:
        positions = np.flatnonzero((frame.profile.to_numpy() == group[0])
                                   & (frame.label.to_numpy() == int(group[1])))
        if not len(positions):
            raise ValueError(f"Empty training group: {group}")
        result[group] = positions
    return result


class GroupSampler:
    def __init__(self, frame, *, seed, group_batch_size=64):
        if isinstance(group_batch_size, bool) or int(group_batch_size) != group_batch_size or group_batch_size < 1:
            raise ValueError("group_batch_size must be a positive integer")
        self.batch_size = int(group_batch_size)
        self.groups = group_indices(frame)
        self.seeds = {g: derived_seed(seed, "sampler-" + g) for g in GROUPS}
        self.rng = {g: np.random.default_rng(self.seeds[g]) for g in GROUPS}
        self.order = {g: self.rng[g].permutation(self.groups[g]) for g in GROUPS}
        self.cursor = dict.fromkeys(GROUPS, 0)
        self.cycles = dict.fromkeys(GROUPS, 0)
        self.draws = dict.fromkeys(GROUPS, 0)
        self.seen = {g: set() for g in GROUPS}
        self.window_seen = {g: set() for g in GROUPS}
        self.window_draws = dict.fromkeys(GROUPS, 0)
        self.digest = hashlib.sha256()
        self.steps = 0

    def next(self):
        chunks = []
        for g in GROUPS:
            remaining, parts = self.batch_size, []
            while remaining:
                if self.cursor[g] == len(self.order[g]):
                    self.order[g] = self.rng[g].permutation(self.groups[g])
                    self.cursor[g] = 0
                count = min(remaining, len(self.order[g]) - self.cursor[g])
                part = self.order[g][self.cursor[g]:self.cursor[g] + count]
                parts.append(part)
                self.cursor[g] += count
                if self.cursor[g] == len(self.order[g]):
                    self.cycles[g] += 1
                remaining -= count
            chosen = np.concatenate(parts).astype(np.int64)
            self.draws[g] += len(chosen)
            self.window_draws[g] += len(chosen)
            self.seen[g].update(chosen.tolist())
            self.window_seen[g].update(chosen.tolist())
            chunks.append(chosen)
        result = np.concatenate(chunks)
        self.steps += 1
        self.digest.update(result.astype("<i8").tobytes())
        return result

    def audit(self, *, reset_window=False):
        result = {"steps": self.steps, "draw_sha256": self.digest.hexdigest(), "groups": {
            g: {"rows": len(self.groups[g]), "draws": self.draws[g], "unique": len(self.seen[g]),
                "completed_cycles": self.cycles[g], "cursor": self.cursor[g],
                "window_draws": self.window_draws[g], "window_unique": len(self.window_seen[g])}
            for g in GROUPS}}
        if reset_window:
            self.window_seen = {g: set() for g in GROUPS}
            self.window_draws = dict.fromkeys(GROUPS, 0)
        return result

    def state(self):
        """Audit state, not permission to resume an interrupted fit."""
        return {"seeds": self.seeds, "orders": {g: self.order[g].tolist() for g in GROUPS},
                "cursors": self.cursor, "rng": {g: self.rng[g].bit_generator.state for g in GROUPS},
                "audit": self.audit()}

    def rng_sha256(self):
        return hashlib.sha256(json.dumps({g: self.rng[g].bit_generator.state for g in GROUPS},
                                       sort_keys=True).encode()).hexdigest()
