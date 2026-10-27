"""Strict UNSW-NB15 ingestion and auditable, duplicate-group-safe partitions.

No fitted preprocessing lives here. Numeric missing values are retained so the
model pipeline can fit its imputer on the training partition only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = (
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
)
CATEGORICAL = ("proto", "service", "state")
NUMERIC = tuple(x for x in FEATURES if x not in CATEGORICAL)
ATTACK_CATEGORIES = (
    "Normal", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
    "Reconnaissance", "Shellcode", "Worms",
)
OFFICIAL_FILES = {
    "train": ("UNSW_NB15_training-set.csv", 175341),
    "test": ("UNSW_NB15_testing-set.csv", 82332),
}
SPLITS = ("train", "validation", "calibration", "test")
SCHEMA_VERSION = "unsw-nb15-split-v1"
_MISSING = {"", "nan", "na", "n/a", "null", "none"}


def sha256_file(path: str | Path) -> str:
    """Hash complete file bytes without loading them all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _feature_hashes(frame: pd.DataFrame) -> list[str]:
    # Canonical semantic values: column order, IDs, labels and CSV formatting do
    # not affect this hash. Missing values share a single representation.
    result = []
    categories = set(CATEGORICAL)
    for row in frame.loc[:, list(FEATURES)].itertuples(index=False, name=None):
        values = []
        for name, value in zip(FEATURES, row):
            if pd.isna(value):
                values.append(None)
            elif name in categories:
                values.append(str(value))
            else:
                # hex is stable and avoids dependence on display precision.
                number = float(value)
                values.append((0.0 if number == 0 else number).hex())
        encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        result.append(hashlib.sha256(encoded).hexdigest())
    return result


def load_unsw_csv(path: str | Path, require_labels: bool = True) -> pd.DataFrame:
    """Read the 42-feature split schema, rejecting extras and invalid values.

    `id` is optional metadata. Labeled input must include both `label` and
    `attack_cat`; inference may omit both. If labels are present they are still
    validated. sample_id uses the source-byte SHA prefix and zero-based CSV data
    row index; feature_hash deliberately excludes IDs and labels.
    """
    path = Path(path)
    source_sha = sha256_file(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column names are not allowed")
    required = set(FEATURES) | ({"label", "attack_cat"} if require_labels else set())
    missing = sorted(required - set(header))
    extra = sorted(set(header) - set(FEATURES) - {"id", "label", "attack_cat"})
    if missing or extra:
        raise ValueError(f"Invalid UNSW schema: missing={missing}; unexpected={extra}")
    if ("label" in header) != ("attack_cat" in header):
        raise ValueError("Provide both label and attack_cat, or omit both for inference")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("CSV contains no data rows")
    for column in NUMERIC:
        text = frame[column].str.strip()
        text = text.mask(text.str.lower().isin(_MISSING), np.nan)
        try:
            values = pd.to_numeric(text, errors="raise").astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Non-numeric value in feature {column}") from exc
        if np.isinf(values.to_numpy()).any():
            raise ValueError(f"Infinite value in feature {column}; use missing only for genuinely unknown values")
        frame[column] = values
    for column in CATEGORICAL:
        values = frame[column].str.strip()
        frame[column] = values.mask(values == "", np.nan)
    if "id" in frame:
        try:
            ids = pd.to_numeric(frame["id"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError("id must contain numeric integer identifiers") from exc
        if not np.isfinite(ids).all() or (ids % 1 != 0).any() or ids.duplicated().any():
            raise ValueError("id must contain finite unique integer identifiers")
        frame["id"] = ids.astype(np.int64)
    if "label" in frame:
        labels = pd.to_numeric(frame["label"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError("label must contain only binary integers 0 or 1")
        frame["label"] = labels.astype(np.int8)
        attack = frame["attack_cat"].str.strip().replace({"Backdoors": "Backdoor"})
        unknown = sorted(set(attack) - set(ATTACK_CATEGORIES))
        if unknown:
            raise ValueError(f"Unknown attack_cat values: {unknown}")
        if not ((frame["label"] == 0) == (attack == "Normal")).all():
            raise ValueError("label and attack_cat disagree: only Normal may have label 0")
        frame["attack_cat"] = attack
    frame["sample_id"] = [f"{source_sha[:12]}:{i}" for i in range(len(frame))]
    frame["feature_hash"] = _feature_hashes(frame)
    if sha256_file(path) != source_sha:
        raise ValueError("Source CSV changed while it was being read")
    frame.attrs.update(source_sha256=source_sha, source_name=path.name)
    return frame


def _group_partitions(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Greedy stratified group allocation; proportions are approximate.

    All rows with an identical feature_hash form an indivisible group, including
    conflicting-label rows. Minimize incremental normalized squared deviations
    from desired class and row totals. Largest groups are placed first; seeded
    random ties avoid dependence on source ordering.
    """
    counts = pd.crosstab(frame["feature_hash"], frame["attack_cat"])
    if len(counts) < 3:
        raise ValueError("At least three distinct feature groups are required for three independent partitions")
    vectors = counts.to_numpy(dtype=float)
    sizes = vectors.sum(axis=1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(counts)), -sizes))
    proportions = np.array([0.70, 0.15, 0.15])
    target = proportions[:, None] * vectors.sum(axis=0)
    target_size = proportions * len(frame)
    totals = np.zeros_like(target)
    row_totals = np.zeros(3)
    assignments = np.full(len(counts), -1, dtype=int)
    group_totals = np.zeros(3, dtype=int)

    def objective(class_totals: np.ndarray, rows: np.ndarray) -> float:
        return float(np.square(class_totals - target).sum(axis=1).dot(np.ones(3)) / max(len(frame), 1)
                     + np.sum(np.square(rows - target_size) / target_size))

    for index in order:
        vector, size = vectors[index], sizes[index]
        # Each stratum receives equal relative consideration, so a large class
        # cannot suppress all validation/calibration examples of rare classes.
        delta = ((2 * (totals - target) * vector + vector ** 2) / np.maximum(target, 1)).sum(axis=1)
        delta += (2 * (row_totals - target_size) * size + size ** 2) / target_size
        chosen = int(np.argmin(delta))
        assignments[index] = chosen
        totals[chosen] += vector
        row_totals[chosen] += size
        group_totals[chosen] += 1
    # Tiny datasets or very large indivisible groups can otherwise leave an
    # empty partition. Move an entire group; never split duplicate rows.
    for empty in np.flatnonzero(group_totals == 0):
        candidates = []
        for index in range(len(counts)):
            donor = assignments[index]
            if group_totals[donor] <= 1:
                continue
            new_totals, new_rows = totals.copy(), row_totals.copy()
            new_totals[donor] -= vectors[index]
            new_totals[empty] += vectors[index]
            new_rows[donor] -= sizes[index]
            new_rows[empty] += sizes[index]
            candidates.append((objective(new_totals, new_rows), index))
        _, index = min(candidates)
        donor = assignments[index]
        totals[donor] -= vectors[index]
        totals[empty] += vectors[index]
        row_totals[donor] -= sizes[index]
        row_totals[empty] += sizes[index]
        group_totals[donor] -= 1
        group_totals[empty] += 1
        assignments[index] = empty
    mapping = dict(zip(counts.index, assignments))
    row_split = frame["feature_hash"].map(mapping)
    return {name: frame.loc[row_split == number].copy().reset_index(drop=True)
            for number, name in enumerate(SPLITS[:3])}


def _duplicate_audit(frame: pd.DataFrame) -> dict[str, Any]:
    groups = frame.groupby("feature_hash", sort=True)
    sizes = groups.size()
    conflicts = []
    for key in groups["attack_cat"].nunique().loc[lambda values: values > 1].index:
        group = groups.get_group(key)
        conflicts.append({"feature_hash": key, "n_rows": len(group),
                          "label_counts": {str(k): int(v) for k, v in group["label"].value_counts().items()},
                          "attack_cat_counts": {str(k): int(v) for k, v in group["attack_cat"].value_counts().items()}})
    return {"n_feature_groups": int(len(sizes)), "duplicate_groups": int((sizes > 1).sum()),
            "rows_in_duplicate_groups": int(sizes[sizes > 1].sum()),
            "duplicate_extra_rows": int((sizes - 1).sum()),
            "conflicting_label_groups": len(conflicts), "conflicts": conflicts,
            "rows_removed": 0}


def _synthetic_marked(path: Path, source_sha: str) -> bool:
    marker = path.parent / "synthetic_source.json"
    if not marker.exists():
        return False
    data = json.loads(marker.read_text(encoding="utf-8"))
    return data.get("files", {}).get(path.name, {}).get("sha256") == source_sha


def prepare_dataset(
    train_csv: str | Path, test_csv: str | Path, output_dir: str | Path,
    seed: int = 20261027, *, data_kind: str = "official",
) -> dict[str, Any]:
    """Prepare train/validation/calibration/test and hash-bound audit manifests.

    official/mirror modes enforce the published filenames and row counts.
    `official` describes user-declared provenance, not independently verified
    source bytes; no official cryptographic reference hash is supplied here.
    Synthetic fixtures must explicitly use data_kind='synthetic'.
    """
    if data_kind not in {"official", "mirror", "synthetic"}:
        raise ValueError("data_kind must be official, mirror, or synthetic")
    paths = {"train": Path(train_csv), "test": Path(test_csv)}
    frames = {role: load_unsw_csv(path) for role, path in paths.items()}
    sources = {}
    for role, frame in frames.items():
        path = paths[role]
        sha = frame.attrs["source_sha256"]
        if _synthetic_marked(path, sha) and data_kind != "synthetic":
            raise ValueError("Synthetic fixture cannot be declared as official or mirrored research data")
        if data_kind != "synthetic":
            expected_name, expected_rows = OFFICIAL_FILES[role]
            if path.name != expected_name or len(frame) != expected_rows:
                raise ValueError(f"{role} requires {expected_name} with {expected_rows} rows; got {path.name} with {len(frame)} rows. Do not swap the official split.")
            if "id" not in frame:
                raise ValueError("Official/mirror split files require the original id metadata column")
        sources[role] = {"name": path.name, "sha256": sha, "n_rows": len(frame)}
        provenance_path = path.parent / "provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            record = provenance.get("files", {}).get(path.name)
            if not record or record.get("sha256") != sha:
                raise ValueError(f"Download provenance checksum does not match {path.name}")
            if provenance.get("data_kind") == "mirror" and data_kind != "mirror":
                raise ValueError("Downloaded mirror must use data_kind='mirror'")
            sources[role]["provenance"] = {
                "url": record.get("url"),
                "mirror_revision": provenance.get("mirror_revision"),
                "official_hash_verified": provenance.get("official_hash_verified", False),
            }
    if sources["train"]["sha256"] == sources["test"]["sha256"]:
        raise ValueError("Training and test source files are identical")
    combined_ids = pd.concat([frame["sample_id"] for frame in frames.values()], ignore_index=True)
    if combined_ids.duplicated().any():
        raise ValueError("Source-hash prefix collision: sample IDs are not unique")
    partitions = _group_partitions(frames["train"], seed)
    partitions["test"] = frames["test"].copy()
    train_hashes = set(frames["train"]["feature_hash"])
    test_hashes = set(frames["test"]["feature_hash"])
    overlap = train_hashes & test_hashes
    overlap_rows = pd.concat([frame.loc[frame["feature_hash"].isin(overlap)]
                              for frame in frames.values()], ignore_index=True)
    overlap_conflicts = _duplicate_audit(overlap_rows)["conflicts"] if overlap else []
    audit = {
        "train_source": _duplicate_audit(frames["train"]),
        "test_source": _duplicate_audit(frames["test"]),
        "train_test_overlap": {
            "n_feature_groups": len(overlap),
            "train_rows": int(frames["train"]["feature_hash"].isin(overlap).sum()),
            "test_rows": int(frames["test"]["feature_hash"].isin(overlap).sum()),
            "feature_hash_examples": sorted(overlap)[:20], "rows_removed": 0,
            "conflicting_label_groups": len(overlap_conflicts), "conflicts": overlap_conflicts,
        },
        "missing_values": {role: {k: int(v) for k, v in frame.loc[:, list(FEATURES)].isna().sum().items() if v}
                           for role, frame in frames.items()},
        "partitions": {name: {"n_rows": len(frame),
                              "fraction_of_training_source": len(frame) / len(frames["train"]) if name != "test" else None,
                              "attack_cat_counts": {str(k): int(v) for k, v in frame["attack_cat"].value_counts().items()},
                              "binary_label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().items()}}
                       for name, frame in partitions.items()},
        "warnings": [],
    }
    if overlap:
        audit["warnings"].append("Training/test sources share exact feature rows; all rows retained. Report this limitation.")
    for role in ("train_source", "test_source"):
        if audit[role]["conflicting_label_groups"]:
            audit["warnings"].append(f"{role} has conflicting labels for identical features; rows retained and conflicts listed.")
    source_classes = set(frames["train"]["attack_cat"])
    for name, frame in partitions.items():
        absent = sorted(source_classes - set(frame["attack_cat"]))
        if absent:
            audit["warnings"].append(f"{name} lacks training-source classes: {absent}; group-safe splitting cannot guarantee class support.")
    output = Path(output_dir)
    targets = [output / f"{split}.parquet" for split in SPLITS]
    targets += [output / "manifest.json", output / "manifest.sha256", output / "audit.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Prepared output already contains artifacts; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    identity = {"schema_version": SCHEMA_VERSION, "source_files": sources,
                "split_seed": seed, "data_kind": data_kind, "features": list(FEATURES)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = {**identity, "dataset_id": f"unsw-nb15-{data_kind}-{fingerprint[:16]}",
                "categorical_features": list(CATEGORICAL),
                "provenance_status": "synthetic_fixture" if data_kind == "synthetic" else
                    ("mirror_not_verified_against_official_hash" if data_kind == "mirror" else "user_declared_official_source_no_reference_hash"),
                "split_method": "seeded_greedy_stratified_feature_groups",
                "target_fractions": {"train": .70, "validation": .15, "calibration": .15},
                "exact_proportions_guaranteed": False, "partitions": {}}
    for name, frame in partitions.items():
        filename = f"{name}.parquet"
        frame.attrs = {}
        frame.to_parquet(output / filename, index=False)
        manifest["partitions"][name] = {"file": filename, "sha256": sha256_file(output / filename), "n_rows": len(frame)}
    _write_json(output / "audit.json", audit)
    manifest["audit"] = {"file": "audit.json", "sha256": sha256_file(output / "audit.json")}
    _write_json(output / "manifest.json", manifest)
    (output / "manifest.sha256").write_text(sha256_file(output / "manifest.json") + "\n", encoding="ascii")
    return manifest


def load_partition(prepared_dir: str | Path, split: str) -> pd.DataFrame:
    """Load a partition only after verifying manifest and partition checksums."""
    if split not in SPLITS:
        raise ValueError(f"Unknown partition {split!r}; expected one of {SPLITS}")
    prepared = Path(prepared_dir)
    manifest_path = prepared / "manifest.json"
    expected_manifest = (prepared / "manifest.sha256").read_text(encoding="ascii").strip()
    if sha256_file(manifest_path) != expected_manifest:
        raise ValueError("Manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("features") != list(FEATURES):
        raise ValueError("Manifest schema does not match this UNSW feature contract")
    record = manifest["partitions"][split]
    if record["file"] != f"{split}.parquet":
        raise ValueError("Invalid partition filename in manifest")
    path = prepared / record["file"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Partition checksum mismatch: {split}")
    audit_record = manifest.get("audit", {})
    if audit_record.get("file") != "audit.json" or sha256_file(prepared / "audit.json") != audit_record.get("sha256"):
        raise ValueError("Audit checksum mismatch")
    frame = pd.read_parquet(path)
    if len(frame) != record["n_rows"] or not set(FEATURES + ("label", "attack_cat", "sample_id", "feature_hash")).issubset(frame.columns):
        raise ValueError(f"Partition schema or row count mismatch: {split}")
    frame.attrs.update(dataset_id=manifest["dataset_id"], data_kind=manifest["data_kind"], split=split)
    return frame


def make_synthetic_dataset(
    output_dir: str | Path, n_train: int = 600, n_test: int = 120, seed: int = 42,
) -> tuple[Path, Path]:
    """Create explicitly marked schema-compatible CI fixtures, not IDS evidence."""
    if n_train < 30 or n_test < 10:
        raise ValueError("Synthetic fixtures need n_train >= 30 and n_test >= 10 for class support")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = tuple(output / OFFICIAL_FILES[role][0] for role in ("train", "test"))
    if any(path.exists() for path in paths) or (output / "synthetic_source.json").exists():
        raise FileExistsError("Synthetic source files already exist; use a new directory")
    rng = np.random.default_rng(seed)
    for path, size in zip(paths, (n_train, n_test)):
        classes = np.arange(size) % len(ATTACK_CATEGORIES)
        rng.shuffle(classes)
        frame = pd.DataFrame({column: rng.lognormal(0.2 * classes, 0.6, size) for column in NUMERIC})
        frame["proto"] = rng.choice(["tcp", "udp", "icmp"], size=size)
        frame["service"] = rng.choice(["-", "http", "dns"], size=size)
        frame["state"] = rng.choice(["FIN", "CON", "INT"], size=size)
        # A weak, noisy fixture signal is solely for pipeline exercise.
        frame["id"] = np.arange(1, size + 1)
        frame["attack_cat"] = np.array(ATTACK_CATEGORIES)[classes]
        frame["label"] = (classes != 0).astype(int)
        frame.loc[:, ["id", *FEATURES, "attack_cat", "label"]].to_csv(path, index=False)
    _write_json(output / "synthetic_source.json", {
        "data_kind": "synthetic", "purpose": "CI fixture only; never report as research performance",
        "seed": seed, "files": {path.name: {"sha256": sha256_file(path)} for path in paths},
    })
    return paths
