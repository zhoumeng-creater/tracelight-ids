"""CNS2022 Improved CICIDS2017: bounded-memory, group-isolated preparation.

IP/port/time/label-derived columns are audit metadata, never predictors. The
session definition is an explicit heuristic, not proof of host/time independence.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import sha256_file

CIC_SCHEMA_VERSION = "cicids2017-cns2022-group-v1"
CIC_ARCHIVE_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"
CIC_ARCHIVE_NAME = "CICIDS2017_improved.zip"
# Observed complete official-host download, 2026-11-01; not a publisher checksum.
CIC_ARCHIVE_SHA256 = "97fdb91d339e2d8cf5627f981b831e5e7e400b981c58181c451a38fd03c48883"
CIC_ARCHIVE_BYTES = 343549013
CIC_LABEL_POLICY_URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Dataset_Download.html"
CIC_FILES = ("monday.csv", "tuesday.csv", "wednesday.csv", "thursday.csv", "friday.csv")
CIC_SOURCE_SHA256 = {
    "monday.csv": "51fe5dc962626efb4ae70dce0303072fb780da0932822b651202ee9c2fbc1aff",
    "tuesday.csv": "e2a0a5b631dfc6b455cc9f9a88b944110637d70a7f74171473925f76f38b6b0c",
    "wednesday.csv": "bf46c5f3c792e8817381f724511229569606918eaf07ac986d7a2592b6341bc2",
    "thursday.csv": "78a4d11eaf473d099e30e71ddb01e0f38218e844c0a9cdd36602145d674af482",
    "friday.csv": "ebd499e6f23bd59f9cb81bec28178491b02b925fa5640a24215c9437d79482d0",
}
CIC_SOURCE_ROWS = dict(zip(CIC_FILES, (371624, 322078, 496641, 362076, 547557)))
CIC_COLUMNS = (
    "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
    "Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
    "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd RST Flags", "Bwd RST Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min", "Packet Length Max",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std",
    "Idle Max", "Idle Min", "ICMP Code", "ICMP Type", "Total TCP Flow Time", "Label",
    "Attempted Category",
)
CIC_EXCLUDED = ("id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp",
                "Label", "Attempted Category")
CIC_FEATURES = tuple(c for c in CIC_COLUMNS if c not in CIC_EXCLUDED)
CIC_CATEGORICAL = ("Protocol", "ICMP Code", "ICMP Type")
CIC_DISCRETE = tuple(c for c in CIC_FEATURES if c in CIC_CATEGORICAL or c in {
    "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags", "Bwd RST Flags",
    "Fwd Header Length", "Bwd Header Length", "Packet Length Min", "Packet Length Max",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWR Flag Count", "ECE Flag Count", "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
})
CIC_SPLITS = ("train", "validation", "calibration", "test")
CIC_FRACTIONS = (.60, .15, .10, .15)
_MISSING = {"", "nan", "na", "n/a", "null", "none"}


def _json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read_header(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f), [])
    if tuple(header) != CIC_COLUMNS:
        raise ValueError(f"{path.name}: expected the exact CNS2022 91-column schema")


def _numeric(values: pd.Series, column: str) -> pd.Series:
    text = values.astype(str).str.strip()
    text = text.mask(text.str.lower().isin(_MISSING), np.nan)
    try:
        return pd.to_numeric(text, errors="raise").astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Non-numeric value in {column}") from exc


def _integer(values: pd.Series, column: str, low: int, high: int) -> np.ndarray:
    arr = _numeric(values, column).to_numpy()
    if not np.isfinite(arr).all() or (arr % 1 != 0).any() or ((arr < low) | (arr > high)).any():
        raise ValueError(f"Invalid finite integer metadata/category: {column}")
    return arr.astype(np.int64)


def iter_cicids_csv(path: str | Path, *, chunksize: int = 50000,
                    source_sha256: str | None = None) -> Iterator[tuple[pd.DataFrame, dict]]:
    """Yield strict cleaned chunks with raw labels and parser audit counts.

    Infinity is explicitly converted to missing. No imputation/scaling is fit.
    `_session_key`, `_start_ns`, `_end_ns` are private preparation fields only.
    """
    path = Path(path)
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    _read_header(path)
    source_sha256 = source_sha256 or sha256_file(path)
    offset = 0
    for raw in pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig",
                           chunksize=chunksize, on_bad_lines="error"):
        if not raw.index.equals(pd.RangeIndex(offset, offset + len(raw))):
            raise ValueError("Malformed CSV row width/index; refusing implicit CSV index")
        raw = raw.reset_index(drop=True)
        audit = {"infinity_to_missing": {}, "missing_values": {}, "negative_values": {},
                 "raw_label_counts": dict(Counter(raw["Label"].str.strip())), "attempted_rows": 0}
        values: dict[str, pd.Series] = {}
        for name in CIC_FEATURES:
            value = _numeric(raw[name], name)
            inf = np.isinf(value.to_numpy())
            if inf.any():
                audit["infinity_to_missing"][name] = int(inf.sum())
                value = value.mask(inf, np.nan)
            if value.isna().any():
                audit["missing_values"][name] = int(value.isna().sum())
            if (value < 0).any():
                audit["negative_values"][name] = int((value < 0).sum())
            values[name] = value
        frame = pd.DataFrame(values)
        for name in CIC_CATEGORICAL:
            arr = frame[name].dropna()
            if (arr % 1 != 0).any() or ((arr < (-1 if name != "Protocol" else 0)) | (arr > 255)).any():
                raise ValueError(f"Invalid categorical code: {name}")
        # Stable IEEE754 little-endian numeric representation, including one
        # canonical missing representation; all 82 fields have numeric source codes.
        matrix = np.ascontiguousarray(frame.loc[:, list(CIC_FEATURES)].to_numpy(), dtype="<f8")
        matrix[matrix == 0] = 0.0
        matrix[np.isnan(matrix)] = np.nan
        frame["feature_hash"] = [hashlib.sha256(row.tobytes()).hexdigest() for row in matrix]
        for name in CIC_CATEGORICAL:
            frame[name] = frame[name].map(lambda x: None if pd.isna(x) else str(int(x)))
        attempted = _integer(raw["Attempted Category"], "Attempted Category", -1, 6)
        labels = raw["Label"].str.strip()
        if labels.eq("").any() or labels.str.lower().isin(_MISSING).any():
            raise ValueError("Missing Label; malformed CSV or unspecified ground truth")
        # The category column is the authoritative author's mapping instruction.
        if ((labels.str.contains("Attempted", case=False, regex=False)) & (attempted == -1)).any():
            raise ValueError("Attempted label disagrees with Attempted Category")
        normal = (attempted != -1) | labels.str.casefold().eq("benign")
        frame["raw_label"] = labels
        frame["attempted_category"] = attempted.astype(np.int8)
        frame["label"] = (~normal).astype(np.int8)
        frame["attack_cat"] = labels.where(~normal, "Normal")
        audit["attempted_rows"] = int((attempted != -1).sum())
        # Naive timestamps in this fixed release are UTC, per author documentation.
        try:
            stamps = pd.to_datetime(raw["Timestamp"].str.strip(), utc=True, errors="raise", format="mixed")
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid UTC Timestamp") from exc
        if stamps.isna().any():
            raise ValueError("Missing UTC Timestamp")
        duration = frame["Flow Duration"].to_numpy()
        if not np.isfinite(duration).all() or (duration < 0).any() or (duration > 7 * 86400 * 1e6).any():
            raise ValueError("Invalid Flow Duration for session reconstruction")
        ports = [_integer(raw[n], n, 0, 65535) for n in ("Src Port", "Dst Port")]
        protocols = _integer(raw["Protocol"], "Protocol", 0, 255)
        session_keys = []
        for sip, dip, sp, dp, proto in zip(raw["Src IP"], raw["Dst IP"], *ports, protocols):
            try:
                ends = sorted(((str(ipaddress.IPv4Address(sip.strip())), int(sp)),
                               (str(ipaddress.IPv4Address(dip.strip())), int(dp))))
            except ipaddress.AddressValueError as exc:
                raise ValueError("Invalid IPv4 address in session metadata") from exc
            session_keys.append(_canonical_hash([ends, int(proto)]))
        _integer(raw["id"], "id", 0, 2**53 - 1)
        frame["sample_id"] = [f"{source_sha256}:{i}" for i in range(offset, offset + len(raw))]
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(offset, offset + len(raw), dtype=np.int64)
        frame["timestamp_utc"] = stamps
        frame["_session_key"] = session_keys
        frame["_start_ns"] = stamps.astype("int64")
        frame["_end_ns"] = stamps.astype("int64") + np.rint(duration * 1000).astype(np.int64)
        offset += len(raw)
        yield frame, audit
    if offset == 0:
        raise ValueError("CSV contains no records")


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def join(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def roots(self) -> np.ndarray:
        while True:
            nxt = self.parent[self.parent]
            if np.array_equal(nxt, self.parent):
                return nxt
            self.parent = nxt


def _joint_groups(db: sqlite3.Connection, n: int, gap: int, conflicts_path: Path):
    uf = _UnionFind(n)
    stats = {"n_feature_groups": 0, "duplicate_groups": 0, "duplicate_extra_rows": 0,
             "conflicting_binary_label_groups": 0, "conflicting_family_groups": 0,
             "rows_removed": 0, "conflict_examples": []}
    db.execute("CREATE INDEX by_feature ON rows(feature_hash, row_id)")
    previous = None
    first = -1
    counts: Counter = Counter()
    binary: Counter = Counter()

    def finish(stream):
        if previous is None:
            return
        size = sum(counts.values())
        stats["n_feature_groups"] += 1
        stats["duplicate_groups"] += int(size > 1)
        stats["duplicate_extra_rows"] += size - 1
        if len(counts) > 1:
            stats["conflicting_family_groups"] += 1
            stats["conflicting_binary_label_groups"] += int(len(binary) > 1)
            record = {"feature_hash": previous, "n_rows": size,
                      "attack_cat_counts": dict(counts), "label_counts": dict(binary)}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(stats["conflict_examples"]) < 20:
                stats["conflict_examples"].append(record)

    with conflicts_path.open("w", encoding="utf-8") as stream:
        for index, feature, label, family in db.execute(
                "SELECT row_id,feature_hash,label,family FROM rows ORDER BY feature_hash,row_id"):
            if feature != previous:
                finish(stream)
                previous, first, counts, binary = feature, index, Counter(), Counter()
            else:
                uf.join(first, index)
            counts[family] += 1
            binary[str(label)] += 1
        finish(stream)
    db.execute("CREATE INDEX by_session ON rows(session_key,start_ns,row_id)")
    sessions = np.empty(n, dtype=np.int64)
    previous, first, end = None, -1, -1
    n_sessions = 0
    for index, key, start, stop in db.execute(
            "SELECT row_id,session_key,start_ns,end_ns FROM rows ORDER BY session_key,start_ns,row_id"):
        if key != previous or start > end + gap * 1_000_000_000:
            previous, first, end = key, index, stop
            n_sessions += 1
        else:
            uf.join(first, index)
            end = max(end, stop)
        sessions[index] = first
    roots = uf.roots()
    _, sizes = np.unique(roots, return_counts=True)
    stats.update(n_sessions=n_sessions, n_joint_groups=int(len(sizes)),
                 largest_joint_group_rows=int(sizes.max()),
                 session_rule="bidirectional 5-tuple; interval overlap or inactivity gap <= configured seconds")
    return roots, sessions, stats


def _allocate(roots: np.ndarray, families: np.ndarray, seed: int) -> np.ndarray:
    unique, inverse = np.unique(roots, return_inverse=True)
    classes, codes = np.unique(families, return_inverse=True)
    if len(unique) < 4:
        raise ValueError("Insufficient independent joint groups for four partitions")
    counts = np.zeros((len(unique), len(classes)), dtype=np.int32)
    np.add.at(counts, (inverse, codes), 1)
    sizes = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(unique)), -sizes))
    fractions = np.asarray(CIC_FRACTIONS)
    targets = fractions[:, None] * counts.sum(axis=0)
    row_targets = fractions * len(roots)
    actual, actual_rows = np.zeros_like(targets), np.zeros(4)
    allocations = np.empty(len(unique), dtype=np.int8)
    for i in order:
        vector, size = counts[i], sizes[i]
        delta = ((2 * (actual - targets) * vector + vector.astype(float)**2) / np.maximum(targets, 1)).sum(axis=1)
        delta += (2 * (actual_rows - row_targets) * size + float(size)**2) / row_targets
        chosen = int(np.argmin(delta))
        allocations[i] = chosen
        actual[chosen] += vector
        actual_rows[chosen] += size
    return allocations[inverse]


def _source_records(raw: Path, data_kind: str) -> tuple[dict, dict]:
    if data_kind not in {"official_repaired", "synthetic"}:
        raise ValueError("data_kind must be official_repaired or synthetic")
    marker_path = raw / ("synthetic_source.json" if data_kind == "synthetic" else "provenance.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("data_kind") != data_kind:
        raise ValueError("Source provenance/data_kind mismatch")
    if data_kind == "official_repaired":
        archive = marker.get("archive", {})
        if archive.get("url") != CIC_ARCHIVE_URL or archive.get("sha256") != CIC_ARCHIVE_SHA256:
            raise ValueError("Official repaired release is not the pinned archive")
        if sha256_file(raw / CIC_ARCHIVE_NAME) != CIC_ARCHIVE_SHA256:
            raise ValueError("Archive checksum mismatch")
    sources = {}
    for name in CIC_FILES:
        path = raw / name
        _read_header(path)
        sha = sha256_file(path)
        if data_kind == "official_repaired" and sha != CIC_SOURCE_SHA256[name]:
            raise ValueError(f"CSV does not match pinned official archive member: {name}")
        expected = marker.get("files", {}).get(name, {})
        if expected.get("sha256") != sha:
            raise ValueError(f"Source checksum mismatch: {name}")
        sources[name] = {"name": name, "sha256": sha, "n_bytes": path.stat().st_size}
    return sources, marker


def prepare_cicids_dataset(raw_dir: str | Path, output_dir: str | Path, seed: int = 20261102,
                          *, data_kind: str = "official_repaired", chunksize: int = 50000,
                          session_gap_seconds: int = 120, reuse: bool = False,
                          min_class_rows: int = 20) -> dict:
    """Prepare once; reuse requires identical sources/settings and intact hashes.

    Approximate 60/15/10/15 splits are jointly isolated by session and exact
    feature groups. Rare-family absence is disclosed, never repaired by splitting
    a group. Every split must support both binary labels with min_class_rows.
    """
    for value, name, minimum in ((seed, "seed", 0), (chunksize, "chunksize", 1),
                                  (session_gap_seconds, "session_gap_seconds", 0),
                                  (min_class_rows, "min_class_rows", 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    raw, output = Path(raw_dir).resolve(), Path(output_dir).resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Raw and prepared directories must be separate siblings/independent paths")
    sources, provenance = _source_records(raw, data_kind)
    settings = {"split_seed": seed, "session_gap_seconds": session_gap_seconds,
                "min_class_rows": min_class_rows, "target_fractions": dict(zip(CIC_SPLITS, CIC_FRACTIONS))}
    identity = {"schema_version": CIC_SCHEMA_VERSION, "data_kind": data_kind,
                "source_files": sources, "preparation": settings, "features": list(CIC_FEATURES)}
    identity_hash = _canonical_hash(identity)
    if output.exists() and any(output.iterdir()):
        if not reuse:
            raise FileExistsError("Prepared directory already exists; use verified reuse or a new directory")
        manifest = load_cic_prepared_verified(output, splits=CIC_SPLITS)
        if manifest.get("preparation_identity_sha256") != identity_hash:
            raise ValueError("Prepared reuse does not match source files/settings")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cic-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary)
        parts = stage / "chunks"
        parts.mkdir()
        db = sqlite3.connect(stage / "groups.sqlite")
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("CREATE TABLE rows(row_id INTEGER PRIMARY KEY, feature_hash TEXT, session_key TEXT, "
                   "start_ns INTEGER,end_ns INTEGER,label INTEGER,family TEXT)")
        audit = {"schema_version": CIC_SCHEMA_VERSION, "data_kind": data_kind,
                 "label_policy": {"attempted_category_not_minus_one": "Normal", "url": CIC_LABEL_POLICY_URL},
                 "source_files": {}, "partitions": {}, "warnings": [], "rows_removed": 0}
        count = 0
        chunk_files = []
        family_chunks = []
        label_chunks = []
        source_ranges = []
        try:
            for name in CIC_FILES:
                path, sha = raw / name, sources[name]["sha256"]
                file_count = 0
                source_ranges.append((count, sha))
                combined = {k: Counter() for k in ("infinity_to_missing", "missing_values", "negative_values", "raw_label_counts")}
                attempted_count = 0
                for frame, counts in iter_cicids_csv(path, chunksize=chunksize, source_sha256=sha):
                    size = len(frame)
                    for key in combined:
                        combined[key].update(counts[key])
                    attempted_count += counts["attempted_rows"]
                    indexes = np.arange(count, count + size, dtype=np.int64)
                    db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?)", (
                        (int(i), fh, sk, int(st), int(en), int(y), family)
                        for i, fh, sk, st, en, y, family in zip(indexes, frame["feature_hash"],
                            frame["_session_key"], frame["_start_ns"], frame["_end_ns"], frame["label"], frame["attack_cat"])))
                    db.commit()
                    family_chunks.append(frame["attack_cat"].to_numpy())
                    label_chunks.append(frame["label"].to_numpy())
                    frame = frame.drop(columns=["_session_key", "_start_ns", "_end_ns"])
                    frame["_row_id"] = indexes
                    chunk = parts / f"{len(chunk_files):06d}.parquet"
                    frame.to_parquet(chunk, index=False)
                    chunk_files.append(chunk)
                    count += size
                    file_count += size
                if sha256_file(path) != sha:
                    raise ValueError(f"Source changed during preparation: {name}")
                expected_rows = provenance["files"][name].get("n_rows")
                if expected_rows is not None and expected_rows != file_count:
                    raise ValueError(f"Source row count mismatch: {name}")
                sources[name]["n_rows"] = file_count
                audit["source_files"][name] = {**{k: dict(v) for k, v in combined.items()},
                                               "n_rows": file_count, "attempted_rows": attempted_count}
            roots, sessions, duplicate_audit = _joint_groups(db, count, session_gap_seconds, stage / "conflicts.jsonl")
        finally:
            db.close()
        families = np.concatenate(family_chunks)
        labels = np.concatenate(label_chunks)
        del family_chunks, label_chunks
        assignment = _allocate(roots, families, seed)
        all_families = sorted(set(families))
        for i, split in enumerate(CIC_SPLITS):
            mask = assignment == i
            binary = Counter(map(int, labels[mask]))
            if any(binary.get(c, 0) < min_class_rows for c in (0, 1)):
                raise ValueError(f"Insufficient binary class support in {split}: {dict(binary)}; joint groups remain indivisible")
            cats = Counter(families[mask])
            absent = sorted(set(all_families) - set(cats))
            audit["partitions"][split] = {"n_rows": int(mask.sum()), "fraction_of_sources": float(mask.mean()),
                "binary_label_counts": {str(k): int(v) for k, v in binary.items()},
                "attack_cat_counts": {str(k): int(cats.get(k, 0)) for k in all_families}, "absent_families": absent,
                "n_joint_groups": int(len(np.unique(roots[mask]))), "n_sessions": int(len(np.unique(sessions[mask])))}
            if absent:
                audit["warnings"].append(f"{split} lacks families {absent}; no group was split to manufacture class support.")
        audit["groups"] = duplicate_audit
        audit["isolation"] = {"sample_ids": True, "complete_feature_groups": True, "session_groups": True,
                              "joint_groups": True, "verified_by": "one assignment per union-find component"}
        audit["warnings"].extend([
            "Grouped within-capture split, not chronological, held-out-host or cross-domain evaluation.",
            "Session reconstruction is a bidirectional 5-tuple inactivity heuristic, not a packet-level session oracle.",
            "Identifier removal does not eliminate all collection-environment or attack-schedule shortcuts.",
            "Negative finite flow values are retained and audited; infinity is missing and imputation remains train-only.",
        ])
        starts = np.array([x[0] for x in source_ranges])

        def row_identity(index):
            slot = int(np.searchsorted(starts, index, side="right") - 1)
            return f"{source_ranges[slot][1]}:{int(index - starts[slot])}"

        joint_hashes = {int(r): hashlib.sha256(("cic-joint-v1:" + row_identity(r)).encode()).hexdigest()
                        for r in np.unique(roots)}
        session_hashes = {int(r): hashlib.sha256(("cic-session-v1:" + row_identity(r)).encode()).hexdigest()
                          for r in np.unique(sessions)}
        writers: dict[str, pq.ParquetWriter] = {}
        try:
            for chunk in chunk_files:
                frame = pd.read_parquet(chunk)
                indexes = frame.pop("_row_id").to_numpy()
                frame["group_id"] = [joint_hashes[int(r)] for r in roots[indexes]]
                frame["group_hash"] = frame["group_id"]
                frame["session_hash"] = [session_hashes[int(r)] for r in sessions[indexes]]
                for i, split in enumerate(CIC_SPLITS):
                    subset = frame.loc[assignment[indexes] == i].reset_index(drop=True)
                    if subset.empty:
                        continue
                    table = pa.Table.from_pandas(subset, preserve_index=False)
                    if split not in writers:
                        writers[split] = pq.ParquetWriter(stage / f"{split}.parquet", table.schema)
                    writers[split].write_table(table)
                chunk.unlink()
        finally:
            for writer in writers.values():
                writer.close()
        parts.rmdir()
        (stage / "groups.sqlite").unlink()
        _json(stage / "audit.json", audit)
        # Bind only public provenance metadata, never the caller's absolute paths.
        _json(stage / "provenance.json", provenance)
        manifest = {"schema_version": CIC_SCHEMA_VERSION,
            "dataset_id": f"cicids2017-cns2022-{data_kind}-{identity_hash[:16]}",
            "data_kind": data_kind, "source_files": sources,
            "preparation_identity_sha256": identity_hash, "preparation": settings,
            "split_seed": seed, "features": list(CIC_FEATURES), "categorical_features": list(CIC_CATEGORICAL),
            "discrete_features": list(CIC_DISCRETE), "excluded_source_columns": list(CIC_EXCLUDED),
            "feature_schema": {"name": CIC_SCHEMA_VERSION, "features": list(CIC_FEATURES),
                               "categorical": list(CIC_CATEGORICAL), "discrete": list(CIC_DISCRETE)},
            "group_column": "group_id", "split_method": "greedy_stratified_union_of_feature_and_session_groups",
            "target_fractions": dict(zip(CIC_SPLITS, CIC_FRACTIONS)), "exact_proportions_guaranteed": False,
            "provenance_status": "synthetic_fixture" if data_kind == "synthetic" else "author_repaired_release_pinned_observed_archive_sha256",
            "partitions": {}, "audit": {"file": "audit.json", "sha256": sha256_file(stage / "audit.json")},
            "provenance": {"file": "provenance.json", "sha256": sha256_file(stage / "provenance.json")},
            "conflicts": {"file": "conflicts.jsonl", "sha256": sha256_file(stage / "conflicts.jsonl")}}
        for split in CIC_SPLITS:
            manifest["partitions"][split] = {"file": f"{split}.parquet", "sha256": sha256_file(stage / f"{split}.parquet"),
                                               "n_rows": audit["partitions"][split]["n_rows"]}
        _json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256_file(stage / "manifest.json") + "\n", encoding="ascii")
        if output.exists():
            output.rmdir()  # Only the pre-checked empty destination; never recursive.
        os.replace(stage, output)
    return manifest


def load_cic_prepared_verified(prepared_dir: str | Path, *, splits=()) -> dict:
    """Verify manifest/audit and only explicitly requested partition files."""
    root = Path(prepared_dir)
    if sha256_file(root / "manifest.json") != (root / "manifest.sha256").read_text(encoding="ascii").strip():
        raise ValueError("CIC manifest checksum mismatch")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != CIC_SCHEMA_VERSION or manifest.get("features") != list(CIC_FEATURES)
            or manifest.get("categorical_features") != list(CIC_CATEGORICAL)
            or manifest.get("discrete_features") != list(CIC_DISCRETE) or manifest.get("group_column") != "group_id"):
        raise ValueError("CIC manifest schema mismatch")
    for key, filename in (("audit", "audit.json"), ("provenance", "provenance.json"), ("conflicts", "conflicts.jsonl")):
        item = manifest.get(key, {})
        if item.get("file") != filename or sha256_file(root / filename) != item.get("sha256"):
            raise ValueError(f"CIC {key} checksum mismatch")
    for split in splits:
        if split not in CIC_SPLITS:
            raise ValueError(f"Unknown CIC partition: {split}")
        item = manifest["partitions"][split]
        if item.get("file") != f"{split}.parquet" or sha256_file(root / item["file"]) != item.get("sha256"):
            raise ValueError(f"CIC partition checksum mismatch: {split}")
    return manifest


def load_cic_partition(prepared_dir: str | Path, split: str) -> pd.DataFrame:
    manifest = load_cic_prepared_verified(prepared_dir, splits=(split,))
    frame = pd.read_parquet(Path(prepared_dir) / f"{split}.parquet")
    needed = {*CIC_FEATURES, "label", "attack_cat", "sample_id", "feature_hash", "group_id", "group_hash", "session_hash"}
    if len(frame) != manifest["partitions"][split]["n_rows"] or not needed.issubset(frame.columns):
        raise ValueError("CIC partition schema/row count mismatch")
    if not frame["sample_id"].is_unique or not frame["group_id"].equals(frame["group_hash"]):
        raise ValueError("CIC partition identity mismatch")
    frame.attrs.update(dataset_id=manifest["dataset_id"], data_kind=manifest["data_kind"], split=split,
                       features=list(CIC_FEATURES), categorical_features=list(CIC_CATEGORICAL),
                       discrete_features=list(CIC_DISCRETE), group_column="group_id",
                       feature_schema=manifest["feature_schema"])
    return frame


def make_synthetic_cicids_dataset(output_dir: str | Path, n_rows: int = 600, seed: int = 42) -> Path:
    """Five marked, schema-exact CI source files; never research performance data."""
    if n_rows < 400:
        raise ValueError("Synthetic CIC fixture needs at least 400 rows for binary split support")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / n).exists() for n in (*CIC_FILES, "synthetic_source.json")):
        raise FileExistsError("Synthetic CIC source already exists")
    rng = np.random.default_rng(seed)
    sizes = [n_rows // 5 + int(i < n_rows % 5) for i in range(5)]
    for day, (name, size) in enumerate(zip(CIC_FILES, sizes)):
        raw = pd.DataFrame({f: rng.integers(1, 10000, size).astype(float) for f in CIC_FEATURES})
        for f in CIC_CATEGORICAL:
            raw[f] = 6 if f == "Protocol" else -1
        raw["id"] = np.arange(size)
        raw["Flow ID"] = [f"fixture-{day}-{i}" for i in range(size)]
        raw["Src IP"] = "192.0.2.1"
        raw["Dst IP"] = "198.51.100.2"
        raw["Src Port"] = np.arange(size) % 50000 + 1000
        raw["Dst Port"] = 80
        raw["Timestamp"] = pd.date_range(f"2017-07-{3+day:02d}T09:00:00Z", periods=size, freq="10min").strftime("%Y-%m-%d %H:%M:%S.%f")
        raw["Label"] = np.where(np.arange(size) % 2, "DoS Hulk", "Benign")
        raw["Attempted Category"] = -1
        raw.loc[:, list(CIC_COLUMNS)].to_csv(root / name, index=False)
    _json(root / "synthetic_source.json", {"data_kind": "synthetic", "seed": seed,
        "purpose": "CI fixture only; never research evidence", "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {n: {"sha256": sha256_file(root / n), "n_rows": size} for n, size in zip(CIC_FILES, sizes)}})
    return root
