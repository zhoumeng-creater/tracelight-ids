"""Recover only registered R26 training inputs from retained archives, without model execution."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ids.response_evidence import tqh
from ids.threshold_validation.observations import (
    COUNTS, MISSING, PORTS, PROTOCOL, RAW_FEATURES, REQUIRED, _family, _numbers,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def raw_features(raw):
    """Use the original scalar conversions; intentionally omit the cumulative export clock."""
    if not REQUIRED.issubset(raw.columns):
        raise ValueError(f"Missing registered Zeek fields: {sorted(REQUIRED - set(raw.columns))}")
    raw = raw.fillna("-")
    _numbers(raw.ts, "ts", required=True)
    proto = raw.proto.str.lower()
    if (raw["id.orig_h"].isin(MISSING).any() or raw["id.resp_h"].isin(MISSING).any()
            or proto.isin(MISSING).any()):
        raise ValueError("Zeek connection lacks a source, destination or protocol")
    values = {
        "ip_protocol": proto.map(PROTOCOL).fillna(-1).to_numpy(dtype=np.int16),
        "duration_seconds": _numbers(raw.duration, "duration"),
        "family": [_family(str(a), str(b)) for a, b in zip(raw.label, raw["detailed-label"], strict=True)],
    }
    for original, target in [*PORTS.items(), *((name, name) for name in COUNTS)]:
        numeric = _numbers(raw[original], original, integer=True)
        if original in PORTS and (numeric > 65535).any():
            raise ValueError(f"Out-of-range port in {original}")
        values[target] = numeric
    return pd.DataFrame(values)[["family", *RAW_FEATURES]]


def selected_gzip(path, spec, row_ids):
    """Hash every original byte, but parse only selected non-comment physical records."""
    wanted = {int(value) for value in row_ids}
    if not wanted or min(wanted) < 1 or len(wanted) != len(row_ids):
        raise ValueError("Selected row identities must be positive and unique")
    digest, size, physical = hashlib.sha256(), 0, 0
    fields, selected, identities = None, [], []
    with gzip.open(path, "rb") as stream:
        for original in stream:
            digest.update(original)
            size += len(original)
            line = original.decode("utf-8").strip()
            if line.startswith("#fields"):
                fields = ["detailed-label" if name == "det_label" else name for name in line.split()[1:]]
                if len(fields) != len(set(fields)) or not REQUIRED.issubset(fields):
                    raise ValueError("Missing or ambiguous IoT-23 connection columns")
                continue
            if not line or line.startswith("#"):
                continue
            if fields is None:
                raise ValueError("Missing Zeek #fields header")
            physical += 1
            if physical in wanted:
                pieces = line.split()
                if len(pieces) != len(fields):
                    raise ValueError(f"Selected physical row {physical} differs from its field header")
                identities.append(physical)
                selected.append(pieces)
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Official uncompressed log differs from protocol: {path}")
    if physical != spec["rows"] or set(identities) != wanted:
        raise ValueError("Registered physical row count or selected identities differ")
    frame = raw_features(pd.DataFrame(selected, columns=fields))
    frame.insert(0, "row_id", identities)
    frame.insert(0, "source_id", spec["name"])
    return frame, {"path": str(Path(path).resolve()), "uncompressed_bytes": size,
                   "uncompressed_sha256": digest.hexdigest(), "physical_rows": physical,
                   "selected_rows": len(frame), "compressed_bytes": Path(path).stat().st_size,
                   "compressed_sha256": sha256(path)}


def registered_identities(identities):
    required = {"source_id", "row_id", "family", "domain", "label", "weight", "support"}
    if not required.issubset(identities.columns):
        raise ValueError("Training identity table lacks registered columns")
    if set(identities.support.unique()) != {"source", "target", "mixed"}:
        raise ValueError("Expected source, target and mixed identity tables")
    keys = ["source_id", "row_id"]
    mixed = identities.loc[identities.support.eq("mixed")].drop(columns="support").copy()
    singles = identities.loc[identities.support.ne("mixed")].drop(columns="support").copy()
    if mixed.duplicated(keys).any() or singles.duplicated(keys).any():
        raise ValueError("Duplicated physical training identity")
    compare = ["family", "domain", "label"]
    if not mixed.set_index(keys)[compare].sort_index().equals(singles.set_index(keys)[compare].sort_index()):
        raise ValueError("Mixed training identities or labels differ from source plus target")
    if mixed.family.eq("Unresolved").any() or not np.array_equal(
            mixed.label.to_numpy(), mixed.family.ne("Benign").astype("int8").to_numpy()):
        raise ValueError("Training binary label differs from registered family")
    if not np.isfinite(mixed.weight).all() or (mixed.weight <= 0).any():
        raise ValueError("Invalid registered mixed weights")
    return mixed.reset_index(drop=True)


def align_inputs(values, identities):
    keys = ["source_id", "row_id"]
    if values.duplicated(keys).any():
        raise ValueError("Recovered inputs duplicate a physical identity")
    expected = pd.MultiIndex.from_frame(identities[keys])
    actual = pd.MultiIndex.from_frame(values[keys])
    if len(actual) != len(expected) or not set(actual) == set(expected):
        raise ValueError("Recovered physical identities differ from registered training rows")
    ordered = values.set_index(keys).loc[expected].reset_index()
    if not np.array_equal(ordered.family.to_numpy(), identities.family.to_numpy()):
        raise ValueError("Recovered author family differs from registered training label")
    result = identities.copy()
    for feature in RAW_FEATURES:
        result[feature] = ordered[feature].to_numpy()
    return result[["source_id", "row_id", "domain", "family", "label", "weight", *RAW_FEATURES]]


def recover(r26_run, raw_dir, output):
    r26_run, raw_dir, output = Path(r26_run), Path(raw_dir), Path(output)
    protocol_path = r26_run / "protocol.json"
    identity_path = r26_run / "evidence/replay/training-identities.parquet"
    manifest = json.loads((r26_run / "archive_manifest.json").read_text(encoding="utf-8-sig"))
    for path in (protocol_path, identity_path):
        relative = path.relative_to(r26_run).as_posix()
        if sha256(path) != manifest[relative]["sha256"]:
            raise ValueError(f"R26 return artifact differs from its manifest: {relative}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    identities = registered_identities(pd.read_parquet(identity_path))
    specs = protocol["source_specs"]
    registration = protocol["registration"]
    allowed = registration["source_train"] + registration["roles"]["target_train"]
    if set(identities.source_id) != set(allowed):
        raise ValueError("Training identity sources differ from protocol roles")
    frames, sources, verified_archives = [], [], {}
    for name in allowed:
        spec = specs[name]
        selected = identities.loc[identities.source_id.eq(name)]
        if spec.get("format") != "tqh":
            path = raw_dir / (name + ".conn.log.labeled.gz")
            if not path.is_file():
                raise FileNotFoundError(f"Retained raw input is missing; no download attempted: {path}")
            frame, receipt = selected_gzip(path, spec, selected.row_id.tolist())
            old = protocol["input_binding"]["raw_inputs"][path.name]
            receipt["compressed_container_matches_lab"] = (
                receipt["compressed_bytes"] == old["bytes"] and receipt["compressed_sha256"] == old["sha256"])
            frames.append(frame)
            sources.append(receipt)
            continue
        path = raw_dir / spec["archive"]
        if not path.is_file():
            raise FileNotFoundError(f"Retained raw archive is missing; no download attempted: {path}")
        if path.name not in verified_archives:
            actual = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}
            old = protocol["input_binding"]["raw_inputs"][path.name]
            if (actual["bytes"], actual["sha256"]) != (old["bytes"], old["sha256"]):
                raise ValueError(f"TQH archive differs from R26 protocol: {path}")
            verified_archives[path.name] = actual
        if (spec["archive_bytes"], spec["archive_sha256"]) != (
                verified_archives[path.name]["bytes"], verified_archives[path.name]["sha256"]):
            raise ValueError("TQH archive differs from source registration")
        with ZipFile(path) as archive:
            obs, labels, receipt = tqh._cell(archive, spec["profile"] + "/" + spec["unit"])
        for key, value in receipt.items():
            if key in spec and value != spec[key]:
                raise ValueError(f"TQH cell {name} differs from registered {key}")
        frame = obs.select(["row_id", *RAW_FEATURES]).to_pandas()
        family = labels.select(["row_id", "family"]).to_pandas()
        if not np.array_equal(frame.row_id, family.row_id):
            raise ValueError("TQH observation and truth identities differ")
        frame["family"] = family.family.to_numpy()
        frame["source_id"] = name
        frame = frame.loc[frame.row_id.isin(selected.row_id)].copy()
        frames.append(frame)
        sources.append({"source_id": name, "archive": path.name, **receipt, "selected_rows": len(frame)})
    result = align_inputs(pd.concat(frames, ignore_index=True), identities)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False, compression="zstd")
    receipt = {"status": "registered_training_inputs_recovered", "rows": len(result),
               "rows_by_domain": result.domain.value_counts().to_dict(),
               "protocol_sha256": sha256(protocol_path), "training_identity_sha256": sha256(identity_path),
               "output_sha256": sha256(output), "output": str(output.resolve()),
               "sources": sources, "archives": verified_archives,
               "weight_semantics": "original mixed-support weights from the R26 identity artifact",
               "original_parquet_bytes_reconstructed": False,
               "clock_policy": "No flow_end_ms: sampled source rows do not reconstruct the full export clock.",
               "gzip_binding": "Uncompressed original bytes are checked against protocol.source_specs; "
                               "compression-container differences from the laboratory are recorded separately.",
               "new_fits": 0, "model_forward_calls": 0, "downloads": 0}
    output.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r26-run", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = recover(args.r26_run, args.raw_dir, args.output)
    print(json.dumps({key: result[key] for key in ("status", "rows", "rows_by_domain", "output")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
