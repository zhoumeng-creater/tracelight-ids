"""Pinned, resumable DEDALE members and daily causal observations with separate truth."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import psutil

from ids.common import read_json, sha256_file, write_json
from ids.contact_evidence.data import BLIND
from ids.contact_evidence.download import ssl_context
from ids.response_evidence.observations import (
    GAP_COLUMNS, QUALITY_COLUMNS, classify_response, counter_gap_frame,
)
from ids.threshold_validation.observations import REQUIRED, _prepare_chunk

REGISTRY = Path(__file__).resolve().parents[3] / "configs/behavior-transfer-20261114/dedale.json"
BLOCK = 8 << 20
AUTHOR_COLUMNS = ["uid", "ts", "ip_src", "port_src", "ip_dst", "port_dst", "proto",
                  "label", "step", "attack_step", "tactic", "technique", "comments"]
AUTHOR_NAMES = {"id.orig_h": "ip_src", "id.resp_h": "ip_dst",
                "id.orig_p": "port_src", "id.resp_p": "port_dst"}
D15_HTTP = "CI7cJo10RfMA3xxRC5"
D15_FILE = "CnOKRT2GLSyyTe1kRf"
D15_CONTROL = ("C83quq4MsJ2zNbCSq5", "C6YQ5KYHHprCyFZkg", "CVF4Qb1wXhZYjd591g")


def read_registration():
    return read_json(REGISTRY)


def _pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _range(archive, start, size):
    request = Request(archive["url"], headers={
        "Range": f"bytes={start}-{start + size - 1}", "Accept-Encoding": "identity",
    })
    expected = f"bytes {start}-{start + size - 1}/{archive['bytes']}"
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60, context=ssl_context()) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Publisher did not honor the pinned byte range")
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise ValueError("Incomplete DEDALE byte range")
                return payload
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"DEDALE download failed at {start}, length {size}") from error
            time.sleep(2 ** attempt)


def _verify_member(path, member):
    digest, crc, size = hashlib.sha256(), 0, 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    if size != member["bytes"] or f"{crc:08x}" != member["crc32"]:
        raise ValueError(f"DEDALE member length/CRC differs: {member['path']}")
    result = digest.hexdigest()
    if member.get("sha256") and member["sha256"] != result:
        raise ValueError("DEDALE registered uncompressed SHA-256 differs")
    return result


def resolve_member(cache_dir, member, registry):
    """Cache a verified gzip projection. Interrupted deflate transfers resume by byte."""
    cache = Path(cache_dir)
    folder = cache / member["archive"].removesuffix(".zip")
    path = folder / (member["path"] + ".gz")
    receipt = path.with_suffix(".sha256.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = _verify_member(path, member)
        if receipt.exists() and read_json(receipt).get("sha256") != digest:
            raise ValueError("Cached DEDALE member changed after verification")
        write_json(receipt, {"pin": _pin(member), "sha256": digest, "bytes": member["bytes"]})
        return path

    archive = registry["archives"][member["archive"]]
    local_archive = cache / member["archive"]
    pending = path.with_suffix(".pending")
    if local_archive.is_file():
        if local_archive.stat().st_size != archive["bytes"]:
            raise ValueError("Offline DEDALE ZIP size differs from the registered source")
        with ZipFile(local_archive) as bundle:
            info = bundle.getinfo(member["path"])
            if (info.CRC != int(member["crc32"], 16) or info.file_size != member["bytes"]
                    or info.compress_size != member["compressed_bytes"]):
                raise ValueError("Offline DEDALE member metadata differs")
            with bundle.open(info) as source, gzip.open(pending, "wb", compresslevel=3) as target:
                shutil.copyfileobj(source, target, length=BLOCK)
    else:
        directory = _range(archive, archive["central_directory_start"], archive["central_directory_bytes"])
        if hashlib.sha256(directory).hexdigest() != archive["central_directory_sha256"]:
            raise ValueError("Publisher DEDALE directory differs from the frozen registration")
        header = _range(archive, member["offset"], 30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Missing ZIP local header")
        method = struct.unpack_from("<H", header, 8)[0]
        flags = struct.unpack_from("<H", header, 6)[0]
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        name = _range(archive, member["offset"] + 30, name_length)
        if name.decode("utf8") != member["path"] or method != 8 or flags & 1:
            raise ValueError("Unexpected DEDALE ZIP member name, encryption or compression")
        start = member["offset"] + 30 + name_length + extra_length
        compressed = path.with_suffix(".deflate.part")
        position = compressed.stat().st_size if compressed.exists() else 0
        if position > member["compressed_bytes"]:
            raise ValueError("Partial DEDALE member is longer than registered")
        with compressed.open("ab") as target:
            while position < member["compressed_bytes"]:
                size = min(BLOCK, member["compressed_bytes"] - position)
                target.write(_range(archive, start + position, size))
                target.flush()
                position += size
        decoder = zlib.decompressobj(-15)
        try:
            with compressed.open("rb") as source, gzip.open(pending, "wb", compresslevel=3) as target:
                for block in iter(lambda: source.read(BLOCK), b""):
                    target.write(decoder.decompress(block))
                target.write(decoder.flush())
        except zlib.error:
            compressed.unlink()
            pending.unlink(missing_ok=True)
            raise
        if not decoder.eof or decoder.unused_data:
            compressed.unlink()
            pending.unlink(missing_ok=True)
            raise ValueError("Incomplete compressed DEDALE member")
    try:
        digest = _verify_member(pending, member)
    except (OSError, ValueError):
        pending.unlink(missing_ok=True)
        path.with_suffix(".deflate.part").unlink(missing_ok=True)
        raise
    pending.replace(path)
    write_json(receipt, {"pin": _pin(member), "sha256": digest, "bytes": member["bytes"]})
    path.with_suffix(".deflate.part").unlink(missing_ok=True)
    return path


def _zeek_chunks(path, *, batch_rows=131072):
    with gzip.open(path, "rt", encoding="utf8", newline="") as stream:
        for line in stream:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\r\n").split("\t")[1:]
                break
            if not line.startswith("#"):
                raise ValueError("Zeek records precede the fields header")
        else:
            raise ValueError("Zeek fields header missing")
        if len(set(fields)) != len(fields):
            raise ValueError("Duplicate Zeek field name")
        with pd.read_csv(stream, sep="\t", names=fields, comment="#", dtype=str,
                         na_filter=False, chunksize=batch_rows, quoting=3) as reader:
            yield from reader


def _truth_frame(author, row_ids, clock, spec, mapping, levels):
    original = author.label
    if not original.isin(["0", "1", "2"]).all():
        raise ValueError("Unknown DEDALE native label")
    family = np.full(len(author), "OtherAttack", dtype=object)
    native_type = np.full(len(author), "other_author_attack", dtype=object)
    for comment, action in mapping.items():
        matches = author.comments.eq(comment).to_numpy()
        family[matches], native_type[matches] = action["family"], action["native_type"]
    family[original.eq("0")], native_type[original.eq("0")] = "Benign", "normal"
    family[original.eq("2")], native_type[original.eq("2")] = "Unresolved", "attack_related"
    positive = original.isin(["1", "2"])
    event = pd.Series("", index=author.index)
    event.loc[positive] = spec["source_id"] + ":step:" + author.loc[positive, "step"]
    evidence = original.map({"0": "author_benign", "1": "author_action_rule", "2": "author_related_rule"})
    for uid, level in levels.items():
        if (author.uid.eq(uid) & original.ne("1")).any():
            raise ValueError("Confirmed event UID has a contradictory native label")
        evidence.loc[author.uid.eq(uid)] = level
    return pd.DataFrame({
        "row_id": row_ids, "uid": author.uid.to_numpy(), "family": family,
        "native_label": original.to_numpy(), "original_label": original.to_numpy(),
        "native_detail": author.comments.to_numpy(), "native_type": native_type,
        "author_step": author.step.to_numpy(), "attack_step": author.attack_step.to_numpy(),
        "technique": author.technique.to_numpy(), "event_id": event.to_numpy(),
        "evidence_level": evidence.to_numpy(),
        "window": (clock // 86400000).astype("datetime64[D]").astype(str),
    })


def convert_day(conn, labels, features, truth, spec, *, batch_rows=131072, registry=None, levels=None):
    """UID-join sorted author CSV to physical conn.log, preserving all 0/1/2 records."""
    registry = read_registration() if registry is None else registry
    author = pd.read_csv(labels, compression="gzip", sep=";", dtype=str, na_filter=False,
                         usecols=AUTHOR_COLUMNS).reset_index(drop=True)
    if author.empty or author.uid.eq("").any() or not author.uid.is_unique:
        raise ValueError("Missing or repeated author UID")
    index = pd.Index(author.uid)
    used = np.zeros(len(author), dtype=bool)
    features, truth = Path(features), Path(truth)
    features.parent.mkdir(parents=True, exist_ok=True)
    truth.parent.mkdir(parents=True, exist_ok=True)
    rows, inversions, ties, lag_max = 0, 0, 0, 0
    watermark, previous_end, first = -1, -1, None
    actual_min, actual_max = np.iinfo(np.int64).max, -1
    start_min, start_max = float("inf"), -1
    uid_digest = hashlib.sha256()
    counts, native, missing = Counter(), Counter(), Counter()
    with TemporaryDirectory(dir=features.parent, prefix="dedale-") as temp, ExitStack() as stack:
        temp = Path(temp)
        for raw in _zeek_chunks(conn, batch_rows=batch_rows):
            if not (REQUIRED - {"label", "detailed-label"} | {"uid"}).issubset(raw.columns):
                raise ValueError("Missing DEDALE Zeek connection fields")
            order = index.get_indexer(raw.uid)
            if (order < 0).any() or len(np.unique(order)) != len(order) or used[order].any():
                raise ValueError("Missing or repeated raw UID in label alignment")
            matched = author.iloc[order].reset_index(drop=True)
            if not raw.proto.to_numpy().tolist() == matched.proto.tolist():
                raise ValueError("UID protocol mismatch")
            for raw_name, author_name in AUTHOR_NAMES.items():
                if not np.array_equal(raw[raw_name].to_numpy(), matched[author_name].to_numpy()):
                    raise ValueError(f"UID endpoint mismatch in {raw_name}")
            starts = pd.to_numeric(raw.ts).to_numpy(dtype=float)
            if not np.allclose(starts, pd.to_numeric(matched.ts), rtol=0, atol=0.000001):
                raise ValueError("UID start timestamp mismatch")
            used[order] = True
            start_min, start_max = min(start_min, starts.min()), max(start_max, starts.max())
            raw["label"], raw["detailed-label"] = "Benign", "-"
            records, clock, end, lag, decreases = _prepare_chunk(
                raw, rows, watermark, previous_end, counts, native, missing
            )
            observation = records.select(BLIND).append_column("uid", pa.array(raw.uid))
            if "ip_proto" in raw:
                protocol = pd.to_numeric(raw.ip_proto, errors="raise").to_numpy(dtype=np.int16)
                if ((protocol < 0) | (protocol > 255)).any():
                    raise ValueError("Invalid native IP protocol number")
                observation = observation.set_column(
                    observation.schema.get_field_index("ip_protocol"), "ip_protocol", pa.array(protocol)
                )
            code = classify_response(raw)
            observation = observation.append_column("response_evidence_code", pa.array(code))
            for value, column in zip((-1, 0, 1), QUALITY_COLUMNS, strict=True):
                observation = observation.append_column(column, pa.array((code == value).astype(np.int8)))
            gaps = counter_gap_frame(raw)
            for column in GAP_COLUMNS:
                observation = observation.append_column(column, pa.array(gaps[column]))
            target = pa.Table.from_pandas(_truth_frame(
                matched, records["row_id"].to_numpy(), clock, spec,
                registry["action_mapping"], levels or {},
            ), preserve_index=False)
            if first is None:
                first = int(clock[0])
                fw = stack.enter_context(pq.ParquetWriter(temp / "features", observation.schema, compression="zstd"))
                tw = stack.enter_context(pq.ParquetWriter(temp / "truth", target.schema, compression="zstd"))
            fw.write_table(observation)
            tw.write_table(target)
            uid_digest.update(("\n".join(raw.uid) + "\n").encode())
            rows += len(raw)
            inversions += decreases
            ties += int((np.diff(clock, prepend=watermark) == 0).sum())
            lag_max = max(lag_max, int(lag.max()))
            watermark, previous_end = int(clock[-1]), int(end[-1])
            actual_min, actual_max = min(actual_min, int(end.min())), max(actual_max, int(end.max()))
        stack.close()
        if not rows or not used.all():
            raise ValueError("Raw and label files do not have identical UID sets")
        (temp / "features").replace(features)
        (temp / "truth").replace(truth)
    return {
        "rows": rows, "first_flow_end_ms": first, "last_flow_end_ms": watermark,
        "min_original_start_ms": int(start_min * 1000), "max_original_start_ms": int(start_max * 1000),
        "min_original_end_ms": actual_min, "max_original_end_ms": actual_max,
        "adjacent_end_time_decreases": inversions, "logical_time_ties": ties,
        "max_clock_lag_ms": lag_max, "uid_order_sha256": uid_digest.hexdigest(),
        "identity": "Same unique UID sets; timestamps/endpoints/protocol match; native physical row order",
        "source_file_history_reset": True, "registered_file_date": spec["date"],
        "role": spec["role"], "source_id": spec["source_id"],
    }


def _d15_evidence(paths, host):
    """Confirm the previously opened example from separately recorded host and network events."""
    targets = {D15_HTTP, D15_FILE, *D15_CONTROL}
    matches = {}
    for key in ("http.log", "files.log", "ssl.log"):
        records = []
        for chunk in _zeek_chunks(paths[key]):
            records.extend(chunk[chunk.uid.isin(targets)].to_dict("records"))
        matches[key] = records
    http = [r for r in matches["http.log"] if r["uid"] == D15_HTTP
            and r["uri"] == "/Bank-of-Nuthington.zip" and r["status_code"] == "200"
            and r["response_body_len"] == "1105"]
    files = [r for r in matches["files.log"] if r["uid"] == D15_HTTP and r["source"] == "HTTP"
             and r["is_orig"] == "F" and r["mime_type"] == "application/zip"
             and r["seen_bytes"] == r["total_bytes"] == "1105"]
    tls = {r["uid"] for r in matches["ssl.log"] if r["established"] == "T"}
    linked = {f for r in http for f in r["resp_fuids"].split(",")}
    network = bool(http and files and any(r["fuid"] in linked for r in files))
    evidence = Counter()
    with gzip.open(host, "rt", encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            timestamp = event.get("@timestamp", "")
            if not timestamp.startswith("2025-01-06T") or event.get("host", {}).get("name") != "CLIENT2.breach.local":
                continue
            code = str(event.get("event", {}).get("code", ""))
            content = event.get("message", "")
            if code == "600" and "http://172.19.1.1/Bank-of-Nuthington.zip" in content:
                evidence["download_command"] += 1
            if code == "11" and timestamp == "2025-01-06T10:01:13.986Z" and "tmp.zip" in content:
                evidence["zip_created"] += 1
            if code == "1" and timestamp == "2025-01-06T10:03:44.504Z" and "scvhost.exe" in content:
                evidence["first_program_started"] += 1
            if code == "11" and timestamp == "2025-01-06T10:04:04.449Z" and "svcmon.exe" in content:
                evidence["second_program_created"] += 1
    if not network or not set((D15_FILE, *D15_CONTROL)).issubset(tls) or len(evidence) != 4:
        raise ValueError("The registered D15 host/network evidence could not be reproduced")
    levels = {D15_HTTP: "host_and_network_confirmed", D15_FILE: "host_correlated_tls"}
    levels.update({uid: "author_action_rule_tls" for uid in D15_CONTROL})
    return levels, {"uid_evidence": levels, "host_event_counts": dict(evidence),
                    "http_uid": D15_HTTP, "http_file_uid": sorted(linked),
                    "http_complete_file_bytes": 1105,
                    "tls_note": "Handshake and host sequence corroborate a transfer; no byte-for-byte file hash join"}


def prepare_one(source_id, cache_dir, output_dir, *, batch_rows=131072):
    registry = read_registration()
    spec = next((s for s in registry["sources"] if s["source_id"] == source_id), None)
    if spec is None:
        raise ValueError(f"Unregistered DEDALE source {source_id}")
    output = Path(output_dir)
    result = {key: spec[key] for key in ("name", "source_id", "role")}
    result.update(features=output / "features" / f"{source_id}.parquet",
                  truth=output / "truth" / f"{source_id}.parquet",
                  audit=output / "evidence" / f"{source_id}.json")
    pin = _pin(registry)
    if result["audit"].is_file():
        audit = read_json(result["audit"])
        if audit.get("registration_sha256") == pin and all(
            result[key].is_file() and sha256_file(result[key]) == audit["outputs"][key]["sha256"]
            for key in ("features", "truth")
        ):
            return result
    paths = {key: resolve_member(cache_dir, member, registry) for key, member in spec["members"].items()}
    levels, event_evidence = {}, {}
    if spec["day"] == 15:
        host = resolve_member(cache_dir, registry["host_evidence"], registry)
        levels, event_evidence = _d15_evidence(paths, host)
    info = convert_day(paths["conn.log"], paths["conn_labeled.csv"],
                       result["features"], result["truth"], spec, batch_rows=batch_rows,
                       registry=registry, levels=levels)
    info["registration_sha256"] = pin
    info["event_evidence"] = event_evidence
    info["member_sha256"] = {
        key: read_json(path.with_suffix(".sha256.json"))["sha256"] for key, path in paths.items()
    }
    info["outputs"] = {
        key: {"sha256": sha256_file(result[key]), "bytes": result[key].stat().st_size}
        for key in ("features", "truth")
    }
    write_json(result["audit"], info)
    return result


def _prepare_result(spec, cache_dir, output_dir):
    try:
        result = prepare_one(spec["source_id"], cache_dir, output_dir)
        result["status"] = "complete"
        return result
    except Exception as error:
        return {key: spec[key] for key in ("name", "source_id", "role")} | {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
        }


def prepare_sources(cache_dir, output_dir, workers=4):
    if workers < 1:
        raise ValueError("DEDALE workers must be positive")
    sources = read_registration()["sources"]
    # Author CSV joins are held one daily file per worker. Budget 1.5 GiB each.
    slots = max(1, min(workers, 8, int(psutil.virtual_memory().available // (1536 << 20))))
    prepare = partial(_prepare_result, cache_dir=cache_dir, output_dir=output_dir)
    if slots == 1:
        return [prepare(source) for source in sources]
    with ProcessPoolExecutor(max_workers=slots) as pool:
        return list(pool.map(prepare, sources))
