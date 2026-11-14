"""One IoT-SFC sensor, exact backup deduplication, and separate publisher truth.

Native Zeek files are retrieved by frozen ZIP ranges; v7 truth is read only by
``prepare_sources``. ``identity_probe`` never opens the labelled CSV.
"""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import os
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ids.common import canonical_hash, read_json, sha256_file, write_json
from ids.contact_evidence.data import BLIND
from ids.contact_evidence.download import ssl_context
from ids.response_evidence.data import _sfc_inputs
from ids.response_evidence.observations import (
    GAP_COLUMNS,
    QUALITY_COLUMNS,
    classify_response,
    counter_gap_frame,
)
from ids.response_evidence.sfc import RAW, LABELS, IDENTITY, _identity
from ids.response_evidence.sources import archive_file
from ids.threshold_validation.observations import _prepare_chunk


REGISTRY = Path(__file__).resolve().parents[3] / "configs/behavior-transfer-20261114/sfc.json"
VERSION = 1


def _range(archive, start, size):
    end = start + size - 1
    for attempt in range(5):
        try:
            request = Request(
                archive["url"],
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
            )
            with urlopen(request, timeout=120, context=ssl_context()) as response:
                if response.status != 206 or response.headers.get("Content-Range") != (
                    f"bytes {start}-{end}/{archive['bytes']}"
                ):
                    raise ValueError("Unexpected publisher byte range")
                payload = response.read(size + 1)
            if len(payload) != size:
                raise ValueError("Truncated publisher byte range")
            return payload
        except (OSError, EOFError, ValueError) as error:
            if attempt == 4:
                raise RuntimeError(f"Cannot retrieve {start}:{end}") from error
            time.sleep(2**attempt)


def _raw_digest(path):
    digest, crc, count = hashlib.sha256(), 0, 0
    with gzip.open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            count += len(block)
    return dict(sha256=digest.hexdigest(), crc32=crc, size=count)


def fetch_member(cache_dir, member, archive, *, bootstrap=False):
    """Resume independently verified raw-deflate ranges; keep one gzip native copy."""
    cache = Path(cache_dir) / "members"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (member["name"].split("/")[-2] + ".conn.log.gz")
    expected = {k: member[k] for k in ("size", "crc32")}
    if not bootstrap:
        expected["sha256"] = member["sha256"]
    if target.exists():
        actual = _raw_digest(target)
        if all(actual[k] == v for k, v in expected.items()):
            return target
        raise ValueError(f"Cached native member differs: {target}")
    header = _range(archive, member["offset"], 30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Missing ZIP local header")
    flags, method = struct.unpack_from("<HH", header, 6)
    name_size, extra_size = struct.unpack_from("<HH", header, 26)
    name = _range(archive, member["offset"] + 30, name_size)
    if flags & 1 or method != 8 or name.decode() != member["name"]:
        raise ValueError("Unexpected or encrypted ZIP member")
    start = member["offset"] + 30 + name_size + extra_size
    pending = target.with_suffix(".deflate.part")
    offset = pending.stat().st_size if pending.exists() else 0
    if offset > member["compressed"]:
        raise ValueError("Partial native member is larger than registered compressed content")
    with pending.open("ab") as stream:
        while offset < member["compressed"]:
            size = min(4 << 20, member["compressed"] - offset)
            stream.write(_range(archive, start + offset, size))
            stream.flush()
            offset += size
    staged = target.with_suffix(".verifying")
    with staged.open("wb") as out, pending.open("rb") as stream:
        out.write(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff")
        for block in iter(lambda: stream.read(4 << 20), b""):
            out.write(block)
        out.write(struct.pack("<II", member["crc32"], member["size"] & 0xFFFFFFFF))
    actual = _raw_digest(staged)
    if not all(actual[k] == v for k, v in expected.items()):
        raise ValueError("Downloaded native member differs from frozen bytes")
    staged.replace(target)
    pending.unlink()
    return target


def _decode_member(args):
    path, member, scratch = args
    dest = Path(scratch) / (member["name"].split("/")[-2] + ".parquet")
    receipt = dest.with_suffix(".json")
    if receipt.exists() and dest.exists():
        info = read_json(receipt)
        if (
            info.get("decoder_version") == 2
            and info.get("raw_sha256") == member["sha256"]
            and info.get("parquet_sha256") == sha256_file(dest)
        ):
            return dest
    started, cpu_started = time.perf_counter(), time.process_time()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream, ExitStack() as stack:
        for line in stream:
            if line.startswith("#fields\t"):
                fields = [name.removeprefix("id.") for name in line.rstrip().split("\t")[1:]]
                break
        else:
            raise ValueError(f"Missing native Zeek fields: {path}")
        if not set(RAW).issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"Incomplete native Zeek fields: {path}")
        writer, rows = None, 0
        pending = dest.with_suffix(".pending")
        for frame in pd.read_csv(
            stream,
            sep="\t",
            names=fields,
            comment="#",
            quoting=3,
            dtype=str,
            na_filter=False,
            usecols=list(RAW),
            chunksize=131072,
        ):
            table = pa.Table.from_pandas(
                frame.loc[:, list(RAW)],
                preserve_index=False,
                schema=pa.schema([(name, pa.string()) for name in RAW]),
            )
            if writer is None:
                writer = stack.enter_context(pq.ParquetWriter(pending, table.schema, compression="zstd"))
            writer.write_table(table)
            rows += len(frame)
        if writer is None:
            pq.write_table(pa.table({name: pa.array([], type=pa.string()) for name in RAW}), pending)
    pending.replace(dest)
    write_json(
        receipt,
        dict(
            rows=rows,
            raw_sha256=member["sha256"],
            parquet_sha256=sha256_file(dest),
            decoder_version=2,
            decoder_pid=os.getpid(),
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu_started,
        ),
    )
    return dest


def deduplicate(tables, members, opened_members):
    """Keep physical first appearances; reject differing copies instead of merging counters."""
    table = pa.concat_tables(tables)
    if not len(table):
        raise ValueError("No native connection records in registered sensor backups")
    ids = table["uid"].combine_chunks()
    codes = pc.dictionary_encode(ids).indices.to_numpy()
    first = np.full(int(codes.max()) + 1, len(codes), dtype=np.int64)
    np.minimum.at(first, codes, np.arange(len(codes)))
    original = first[codes]
    repeated = np.flatnonzero(original != np.arange(len(codes)))
    if len(repeated):
        for name in RAW:
            equal = pc.equal(
                table[name].take(pa.array(repeated)), table[name].take(pa.array(original[repeated]))
            )
            if not pc.all(equal).as_py():
                raise ValueError(f"Rolling backup UID has different native {name}")
    keep = np.sort(first)
    member_index = np.repeat(np.arange(len(members), dtype=np.int16), [len(t) for t in tables])
    known = np.zeros(len(first), dtype=bool)
    for i, member in enumerate(members):
        if member["name"] in opened_members:
            known[codes[member_index == i]] = True
    unique = table.take(pa.array(keep))
    return (
        unique,
        member_index[keep],
        known[codes[keep]],
        dict(
            physical_rows=len(table),
            unique_rows=len(unique),
            repeated_rows=len(repeated),
            previously_opened_rows=int(known.sum()),
            deduplication="exact UID and equality of all 15 native fields; earliest backup / physical row retained",
        ),
    )


def _clock(raw):
    ts = pc.cast(raw["ts"], pa.float64()).to_numpy()
    duration = pd.to_numeric(raw["duration"].to_pandas().replace("-", "0"), errors="raise").to_numpy()
    start_us = np.rint(ts * 1e6).astype(np.int64)
    end = (start_us + np.rint(np.maximum(duration, 0) * 1e6).astype(np.int64) + 999) // 1000
    return start_us // 1000, end, np.maximum.accumulate(end)


def collect_raw(cache_dir, workers=4, *, config=None, bootstrap=False):
    config = read_json(REGISTRY) if config is None else config
    members = config["members"]
    cache = Path(cache_dir)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        paths = list(
            pool.map(lambda m: fetch_member(cache, m, config["raw_archive"], bootstrap=bootstrap), members)
        )
    if bootstrap:
        for member, path in zip(members, paths):
            member.update(_raw_digest(path))
    scratch = cache / "decoded"
    scratch.mkdir(parents=True, exist_ok=True)
    # Each native backup is decoded by an independent CPU process; no history can cross days.
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        decoded = list(pool.map(_decode_member, [(p, m, scratch) for p, m in zip(paths, members)]))
    tables = [pq.read_table(path) for path in decoded]
    raw, backup, opened, info = deduplicate(tables, members, set(config["opened_members"]))
    return raw, backup, opened, info, config


def identity_probe(cache_dir, workers=4, *, config=None, bootstrap=False):
    """Data identity only: no publisher truth, model, score or class statistics."""
    raw, backup, opened, info, config = collect_raw(cache_dir, workers, config=config, bootstrap=bootstrap)
    start, end, clock = _clock(raw)
    day = clock // 86400000
    info.update(
        first_start_ms=int(start.min()),
        last_end_ms=int(end.max()),
        clock_max_lag_ms=int((clock - end).max()),
        end_time_inversions=int((np.diff(end) < 0).sum()),
        uid_order_sha256=hashlib.sha256(("\n".join(raw["uid"].to_pylist()) + "\n").encode()).hexdigest(),
        days=[
            dict(
                day=str(np.datetime64(int(d), "D")),
                rows=int((day == d).sum()),
                previously_opened=int(opened[day == d].sum()),
                first_clock_ms=int(clock[day == d][0]),
                last_clock_ms=int(clock[day == d][-1]),
            )
            for d in np.unique(day)
        ],
    )
    info["schedule_time_intersection"] = []
    for event in config.get("events", {}).get("scanning_schedule", []):
        low = pd.Timestamp(event["start_time_utc"]).value // 1_000_000
        high = pd.Timestamp(event["end_time_utc"]).value // 1_000_000
        selected = (start >= low) & (start <= high)
        info["schedule_time_intersection"].append(
            dict(
                event_id=event["event_id"],
                start_time_utc=event["start_time_utc"],
                end_time_utc=event["end_time_utc"],
                raw_timestamp_rows=int(selected.sum()),
                previously_opened_rows=int(opened[selected].sum()),
                interpretation="Timestamp intersection only; no flow attack labels or completeness claim",
            )
        )
    return info, config


def attach_truth(raw, archive, config):
    """Exact sensor/UID/time/tuple join; labels cannot enter feature or clock construction."""
    uids = pd.Index(raw["uid"].to_numpy())
    labels = []
    columns = [*IDENTITY, *LABELS, "sensor_ip", "log_type"]
    with ZipFile(archive) as z, z.open(config["labels_archive"]["csv_member"]) as stream:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=16 << 20),
            parse_options=pacsv.ParseOptions(newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                include_columns=columns, column_types=dict.fromkeys(columns, pa.string())
            ),
        )
        for batch in reader:
            subset = batch.filter(
                pc.and_(pc.equal(batch["sensor_ip"], config["sensor"]), pc.equal(batch["log_type"], "conn"))
            )
            if not len(subset):
                continue
            locations = uids.get_indexer(subset["uid"].to_numpy(zero_copy_only=False))
            found = np.flatnonzero(locations >= 0)
            if not len(found):
                continue
            matched = subset.take(pa.array(found)).to_pandas()
            reference = raw.select(IDENTITY).take(pa.array(locations[found])).to_pandas()
            if not _identity(reference).equals(_identity(matched)):
                raise ValueError("Publisher UID conflicts with native timestamp or five-tuple")
            item = matched.loc[:, list(LABELS)].copy()
            item["position"] = locations[found]
            labels.append(item)
    if not labels:
        raise ValueError("No publisher truth matches the sensor's native flows")
    truth = pd.concat(labels, ignore_index=True).drop_duplicates()
    if truth.position.duplicated().any():
        raise ValueError("Conflicting publisher truth for one native flow")
    truth = truth.set_index("position").reindex(range(len(raw)))
    truth.loc[truth.attack.isna(), "label_source"] = "unmatched_v7"
    return truth.fillna("")


def _labels_archive(cache_dir, spec):
    cache = Path(cache_dir)
    previous = cache.parent.parent / "response-evidence-20261112" / spec["filename"]
    if (
        previous.is_file()
        and previous.stat().st_size == spec["bytes"]
        and sha256_file(previous) == spec["sha256"]
    ):
        return previous
    return archive_file(cache, spec)


def _prepare_day(args):
    raw, truth, clocks, opened, backups, members, day, output, source_id = args
    name = "IoT-SFC-106-" + day
    output = Path(output)
    features, target, audit = (
        output / "features" / (name + ".parquet"),
        output / "truth" / (name + ".parquet"),
        output / "audit" / (name + ".json"),
    )
    role = (
        "earlier_transfer"
        if day < "2025-04-10"
        else (
            "replay"
            if day <= "2025-04-11" or opened.all()
            else "forward_mixed"
            if opened.any()
            else "forward_unopened"
        )
    )
    spec = dict(
        name=name,
        features=features,
        truth=target,
        audit=audit,
        role=role,
        source_id=source_id,
        status="complete",
    )
    try:
        if audit.exists():
            previous = read_json(audit)
            if (
                previous.get("source_id") == source_id
                and previous.get("version") == VERSION
                and all(
                    p.exists() and sha256_file(p) == previous[k]
                    for p, k in [(features, "features_sha256"), (target, "truth_sha256")]
                )
            ):
                return spec
        for path in (features, target, audit):
            path.parent.mkdir(parents=True, exist_ok=True)
        with ExitStack() as stack:
            bw = tw = None
            for lo in range(0, len(raw), 131072):
                hi = min(lo + 131072, len(raw))
                chunk = raw.slice(lo, hi - lo).to_pandas()
                labels = truth.iloc[lo:hi].reset_index(drop=True)
                merged = _sfc_inputs(pd.concat([chunk, labels], axis=1))
                record, _, _, _, _ = _prepare_chunk(merged, lo, -1, -1, Counter(), Counter(), Counter())
                blind = record.select(BLIND)
                blind = blind.set_column(
                    blind.schema.get_field_index("flow_end_ms"), "flow_end_ms", pa.array(clocks[lo:hi])
                )
                blind = blind.append_column("uid", pa.array(chunk.uid))
                codes = classify_response(merged)
                blind = blind.append_column("response_evidence_code", pa.array(codes))
                for code, field in zip((-1, 0, 1), QUALITY_COLUMNS):
                    blind = blind.append_column(field, pa.array((codes == code).astype(np.int8)))
                gaps = counter_gap_frame(merged)
                for field in GAP_COLUMNS:
                    blind = blind.append_column(field, pa.array(gaps[field]))
                starts, ends, _ = _clock(raw.slice(lo, hi - lo))
                targets = pa.table(
                    dict(
                        row_id=np.arange(lo + 1, hi + 1, dtype=np.int64),
                        uid=chunk.uid,
                        window=[day] * (hi - lo),
                        family=record["family"],
                        native_label=labels.attack,
                        native_detail=labels.type_of_attack,
                        event_id=labels.window_id,
                        evidence_level=np.where(
                            labels.label_source.eq("window"),
                            "author_attack_window",
                            np.where(
                                labels.label_source.eq("background"), "author_background", labels.label_source
                            ),
                        ),
                        label_source=labels.label_source,
                        attacker_role=labels.attacker_role,
                        original_start_ms=starts,
                        original_end_ms=ends,
                        previously_opened=opened[lo:hi],
                        exposure=np.where(
                            opened[lo:hi],
                            "previously_opened",
                            "earlier_transfer_unopened"
                            if day < "2025-04-10"
                            else "development_same_day_unopened"
                            if day <= "2025-04-11"
                            else "forward_unopened",
                        ),
                        source_backup=[members[i]["name"] for i in backups[lo:hi]],
                    )
                )
                if bw is None:
                    bw = stack.enter_context(
                        pq.ParquetWriter(features.with_suffix(".pending"), blind.schema, compression="zstd")
                    )
                    tw = stack.enter_context(
                        pq.ParquetWriter(target.with_suffix(".pending"), targets.schema, compression="zstd")
                    )
                bw.write_table(blind)
                tw.write_table(targets)
        features.with_suffix(".pending").replace(features)
        target.with_suffix(".pending").replace(target)
        write_json(
            audit,
            dict(
                version=VERSION,
                source_id=source_id,
                role=role,
                rows=len(raw),
                previously_opened_rows=int(opened.sum()),
                unopened_rows=int((~opened).sum()),
                first_flow_end_ms=int(clocks[0]),
                last_flow_end_ms=int(clocks[-1]),
                uid_order_sha256=hashlib.sha256(
                    ("\n".join(raw["uid"].to_pylist()) + "\n").encode()
                ).hexdigest(),
                features_sha256=sha256_file(features),
                truth_sha256=sha256_file(target),
                clock="physical first appearance across chronological backups, cumulative flow end, UTC day; cold start per day",
                coverage="all available sensor-106 native conn records; missing collection periods remain missing",
                native_label_counts={
                    str(k): int(v)
                    for k, v in truth.groupby(["attack", "type_of_attack", "label_source"]).size().items()
                },
            ),
        )
        return spec
    except Exception as error:
        spec.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(audit, {k: str(v) if isinstance(v, Path) else v for k, v in spec.items()})
        return spec


def prepare_sources(cache_dir, output_dir, workers=4):
    """Lab entry: completed daily receipts survive failures and reruns independently."""
    config = read_json(REGISTRY)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_id = "IoT-SFC:23371775:sensor106:" + canonical_hash(config)
    receipt = output / "sources.json"
    if receipt.exists():
        prior = read_json(receipt)
        complete = bool(prior)
        for spec in prior:
            if spec.get("source_id") != source_id or spec.get("status") != "complete":
                complete = False
                break
            audit = Path(spec["audit"])
            if not audit.exists():
                complete = False
                break
            evidence = read_json(audit)
            if evidence.get("version") != VERSION or any(
                not Path(spec[k]).exists() or sha256_file(spec[k]) != evidence.get(k + "_sha256")
                for k in ("features", "truth")
            ):
                complete = False
                break
        if complete:
            return [
                {k: Path(v) if k in ("features", "truth", "audit") else v for k, v in s.items()}
                for s in prior
            ]
    raw, backup, opened, info, _ = collect_raw(cache_dir, workers, config=config)
    write_json(output / "audit" / "native_identity.json", dict(source_id=source_id, **info))
    archive = _labels_archive(cache_dir, config["labels_archive"])
    events = archive_file(Path(cache_dir), dict(config["events"], filename="all_attacks_logs.json"))
    import json

    event_rows = json.loads(
        "\n".join(
            line for line in events.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
        )
    )
    write_json(output / "audit" / "author_event_schedule.json", event_rows)
    projected = Path(cache_dir) / ("labels-" + source_id.rsplit(":", 1)[-1] + ".parquet")
    label_receipt = projected.with_suffix(".json")
    if (
        projected.exists()
        and label_receipt.exists()
        and read_json(label_receipt).get("sha256") == sha256_file(projected)
    ):
        labels = pd.read_parquet(projected)
        if len(labels) != len(raw):
            raise ValueError("Cached publisher projection has a different row count")
    else:
        labels = attach_truth(raw, archive, config)
        labels.to_parquet(projected.with_suffix(".pending"), index=False, compression="zstd")
        projected.with_suffix(".pending").replace(projected)
        write_json(label_receipt, dict(source_id=source_id, rows=len(labels), sha256=sha256_file(projected)))
    _, _, clock = _clock(raw)
    day_codes = clock // 86400000
    jobs = []
    for code in np.unique(day_codes):
        indices = np.flatnonzero(day_codes == code)
        day = str(np.datetime64(int(code), "D"))
        jobs.append(
            (
                raw.take(pa.array(indices)),
                labels.iloc[indices].reset_index(drop=True),
                clock[indices],
                opened[indices],
                backup[indices],
                config["members"],
                day,
                output,
                source_id,
            )
        )
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 4))) as pool:
        specs = list(pool.map(_prepare_day, jobs))
    write_json(
        output / "sources.json",
        [{k: str(v) if isinstance(v, Path) else v for k, v in s.items()} for s in specs],
    )
    return specs


def probe_opened_day(cache_dir, opened_csv, output_dir, workers=4):
    """Prepare only the previously opened 2025-04-10 day, without reading new truth."""
    from ids.threshold_validation.sources import verify_log

    verify_log(
        Path(opened_csv),
        dict(bytes=41354928, sha256="6a958525d8a526590d8c8054e3fc676a82958435f32c47907e06072ec5679778"),
    )
    raw, backup, opened, _, config = collect_raw(cache_dir, workers)
    _, _, clock = _clock(raw)
    day = "2025-04-10"
    selected = np.flatnonzero(clock // 86400000 == np.datetime64(day, "D").astype(int))
    current = raw.take(pa.array(selected))
    old = pd.read_csv(opened_csv, dtype=str, na_filter=False)
    _, _, old_clock = _clock(pa.Table.from_pandas(old.loc[:, list(RAW)], preserve_index=False))
    previous = old.loc[old_clock // 86400000 == np.datetime64(day, "D").astype(int)].reset_index(drop=True)
    if not current.to_pandas().equals(previous.loc[:, list(RAW)]):
        raise ValueError("Old-day native fields or physical UID order changed")
    if not np.array_equal(
        clock[selected], old_clock[old_clock // 86400000 == np.datetime64(day, "D").astype(int)]
    ):
        raise ValueError("Old-day registration clock changed")
    args = (
        current,
        previous.loc[:, list(LABELS)],
        clock[selected],
        opened[selected],
        backup[selected],
        config["members"],
        day,
        output_dir,
        "old-day-probe:" + canonical_hash(config),
    )
    spec = _prepare_day(args)
    if spec["status"] != "complete":
        raise ValueError(spec["error"])
    features, truth = pq.read_table(spec["features"]), pq.read_table(spec["truth"])
    counts = Counter(truth["family"].to_pylist())
    if len(features) != 274871 or counts["Scanning"] != 83936 or counts["Benign"] != 38189:
        raise ValueError("Opened 4/10 reference counts changed")
    return dict(
        source="IoT-SFC-106-2025-04-10",
        rows=len(features),
        family_counts=dict(counts),
        native_fields_exact=True,
        uid_order_exact=True,
        registration_clock_exact=True,
        new_truth_opened=False,
        model_run=False,
        features_schema=str(features.schema),
        truth_schema=str(truth.schema),
        features_sha256=sha256_file(spec["features"]),
        truth_sha256=sha256_file(spec["truth"]),
        source_registry_canonical_sha256=canonical_hash(config),
    )


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch/audit native SFC identity without new truth or models"
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--opened-csv", type=Path, help="Only the already-opened 4/10 reference projection")
    args = parser.parse_args(argv)
    if args.opened_csv is None:
        report, _ = identity_probe(args.cache, args.workers)
    else:
        report = probe_opened_day(args.cache, args.opened_csv, args.cache / "probes/2025-04-10", args.workers)
    write_json(args.report, report)
    print(args.report)


if __name__ == "__main__":
    main()
