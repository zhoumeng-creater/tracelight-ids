"""Keep complete physical Zeek backups and attach v7 labels by sensor and flow identity."""

import gzip
import hashlib
from http.client import HTTPException
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

from ids.contact_evidence.download import ssl_context
from ids.threshold_validation.sources import verify_log

RAW = (
    "uid",
    "ts",
    "orig_h",
    "orig_p",
    "resp_h",
    "resp_p",
    "proto",
    "duration",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "history",
    "conn_state",
    "missed_bytes",
)
LABELS = ("attack", "type_of_attack", "label_source", "window_id", "attacker_role")
IDENTITY = ("uid", "ts", "orig_h", "resp_h", "orig_p", "resp_p", "proto")


def fetch_raw(spec, archive):
    start = spec["raw_member_offset"]
    end = start + spec["raw_deflated_bytes"] + 512
    request = Request(
        archive["raw_archive"], headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    )
    for attempt in range(5):
        try:
            with urlopen(request, timeout=120, context=ssl_context()) as response:
                if (
                    response.status != 206
                    or response.headers.get("Content-Range")
                    != f"bytes {start}-{end}/{archive['raw_archive_bytes']}"
                ):
                    raise ValueError("Unexpected raw member byte range")
                data = response.read(end - start + 2)
            if len(data) != end - start + 1 or data[:4] != b"PK\x03\x04":
                raise ValueError("Missing ZIP local header")
            name_size, extra = struct.unpack_from("<HH", data, 26)
            if data[30 : 30 + name_size].decode() != spec["raw_member"]:
                raise ValueError("Unexpected ZIP member")
            lo = 30 + name_size + extra
            raw = zlib.decompress(data[lo : lo + spec["raw_deflated_bytes"]], -15)
            if hashlib.sha256(raw).hexdigest() != spec["raw_member_sha256"]:
                raise ValueError("Raw conn.log digest differs")
            return raw
        except (OSError, EOFError, ValueError, zlib.error, HTTPException) as error:
            if attempt == 4:
                raise RuntimeError(f"Could not verify raw snapshot {spec['name']}") from error
            time.sleep(2**attempt)


def raw_frame(payload, spec):
    stream = io.BytesIO(payload)
    for line in stream:
        if line.startswith(b"#fields"):
            fields = [s.decode().removeprefix("id.") for s in line.rstrip().split(b"\t")[1:]]
            break
    else:
        raise ValueError("Missing Zeek header")
    frame = pd.read_csv(
        stream, sep="\t", names=fields, comment="#", quoting=3, usecols=list(RAW), na_filter=False, dtype=str
    ).loc[:, list(RAW)]
    digest = hashlib.sha256(("\n".join(frame.uid) + "\n").encode()).hexdigest()
    if len(frame) != spec["rows"] or digest != spec["uid_order_sha256"] or frame.uid.duplicated().any():
        raise ValueError("Raw flow UID count, uniqueness or sequence differs")
    return frame


def _identity(frame):
    frame = frame.loc[:, list(IDENTITY)].copy()
    frame["ts"] = np.rint(pd.to_numeric(frame.ts, errors="raise") * 1e6).astype("int64")
    return pd.MultiIndex.from_frame(frame)


def project(archive, cache, specs, archive_spec, *, raw_reader=fetch_raw, verify=True):
    """v7 deduplicates backups. Match labels across backup folders of the same sensor."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    frames = [raw_frame(raw_reader(s, archive_spec), s) for s in specs]
    raw = pd.concat(frames, ignore_index=True)
    sensors = {s["sensor"] for s in specs}
    if len(sensors) != 1 or raw.uid.duplicated().any():
        raise ValueError("Register nonoverlapping snapshots from one sensor")
    identities = _identity(raw)
    uids = pd.Index(raw.uid)
    labels = []
    cols = [*IDENTITY, *LABELS, "sensor_ip", "log_type"]
    with ZipFile(archive) as z, z.open(archive_spec["csv_member"]) as stream:
        reader = csv.open_csv(
            stream,
            read_options=csv.ReadOptions(block_size=16 << 20),
            parse_options=csv.ParseOptions(newlines_in_values=True),
            convert_options=csv.ConvertOptions(
                include_columns=cols, column_types=dict.fromkeys(cols, pa.string())
            ),
        )
        for batch in reader:
            matched = batch.filter(
                pc.and_(pc.equal(batch["log_type"], "conn"), pc.equal(batch["sensor_ip"], specs[0]["sensor"]))
            )
            if not matched.num_rows:
                continue
            data = matched.to_pandas()
            selected = data.loc[uids.get_indexer(data.uid) >= 0]
            if selected.empty:
                continue
            loc = identities.get_indexer(_identity(selected))
            if (loc < 0).any():
                raise ValueError("Same sensor UID has inconsistent timestamp or five-tuple")
            rows = selected.loc[:, list(LABELS)].copy()
            rows["position"] = loc
            labels.append(rows)
    if not labels:
        raise ValueError("No matching v7 sensor truth rows")
    truth = pd.concat(labels, ignore_index=True).drop_duplicates()
    if truth.position.duplicated().any():
        raise ValueError("Conflicting v7 truth for the same physical flow")
    truth = truth.set_index("position").reindex(range(len(raw)))
    truth.loc[truth.attack.isna(), "label_source"] = "unmatched_v7"
    joined = pd.concat([raw, truth.fillna("")], axis=1)
    with TemporaryDirectory(dir=cache, prefix="sfc-") as temp:
        start = 0
        for spec in specs:
            dest = Path(temp) / spec["filename"]
            with gzip.open(dest, "wt", encoding="utf-8", newline="", compresslevel=3) as out:
                joined.iloc[start : start + spec["rows"]].to_csv(out, index=False, lineterminator="\n")
            start += spec["rows"]
            if verify:
                verify_log(dest, spec)
        for spec in specs:
            (Path(temp) / spec["filename"]).replace(cache / spec["filename"])
