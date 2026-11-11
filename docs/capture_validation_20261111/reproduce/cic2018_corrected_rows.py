"""Stream the two selected deflate members; validate each member against the author's ZIP64 CRC."""

from collections import Counter
import json
import io
import zlib
import hashlib
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from audit_paths import RESULTS as ROOT, CORRECTED as RAW
from cic2018_corrected_range import NAMES

PROBE = ROOT / "cic2018_nf_probe.parquet"
COLUMNS = [
    "id",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Label",
    "Attempted Category",
]
BLOCK = 8 << 20


class CheckedReader(io.RawIOBase):
    def __init__(self, path, e):
        super().__init__()
        self.needed = e["compressed_bytes"]
        journal = RAW / (e["name"] + ".received")
        received = set(map(int, journal.read_text(encoding="ascii").splitlines()))
        if received != set(range(0, self.needed, BLOCK)) or path.stat().st_size != self.needed:
            raise ValueError(f"Incomplete byte ranges for {path.name}. Run cic2018_corrected_range.py first.")
        self.fd = path.open("rb")
        self.offset = 0
        self.inflate = zlib.decompressobj(-15)
        self.overflow = b""
        self.crc = 0
        self.size = 0
        self.check = hashlib.sha256()
        self.compressed_hash = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buf):
        while not self.overflow and self.offset < self.needed:
            data = self.fd.read(min(1 << 20, self.needed - self.offset, BLOCK - self.offset % BLOCK))
            if not data:
                raise IOError("Missing received compressed data")
            self.offset += len(data)
            self.compressed_hash.update(data)
            self.overflow = self.inflate.decompress(data)
            self.crc = zlib.crc32(self.overflow, self.crc)
            self.size += len(self.overflow)
            self.check.update(self.overflow)
        n = min(len(buf), len(self.overflow))
        buf[:n] = self.overflow[:n]
        self.overflow = self.overflow[n:]
        return n

    def close(self):
        self.fd.close()
        super().close()


def extract(name):
    date = "2018-02-28" if "28-02" in name else "2018-03-01"
    source = "172.31.69.24" if "28-02" in name else "172.31.69.13"
    descriptor = json.loads((RAW / (name + ".descriptor.json")).read_bytes())
    probes = pd.read_parquet(PROBE)
    times = probes.loc[probes.day == date, "FLOW_START_MILLISECONDS"].to_numpy()
    candidate_time = np.unique(np.concatenate([times + i for i in range(-1, 2)]))
    source_path = RAW / (name + ".deflate")
    result = RAW / (date + ".corrected_endpoints.parquet")
    part = result.with_suffix(".parquet.part")
    all_labels = Counter()
    ranges = {}
    rows = 0
    selected = 0
    checkpoint = 0
    r = CheckedReader(source_path, descriptor)
    writer = None
    try:
        inp = io.BufferedReader(r, 4 << 20)
        arrow = pcsv.open_csv(
            inp,
            read_options=pcsv.ReadOptions(block_size=8 << 20),
            convert_options=pcsv.ConvertOptions(
                include_columns=COLUMNS,
                column_types={
                    "Timestamp": pa.timestamp("us"),
                    "Src IP": pa.string(),
                    "Dst IP": pa.string(),
                    "Src Port": pa.int64(),
                    "Dst Port": pa.int64(),
                    "Protocol": pa.int64(),
                    "Label": pa.string(),
                    "Attempted Category": pa.string(),
                },
            ),
        )
        for batch in arrow:
            f = batch.to_pandas()
            us = f.Timestamp.to_numpy().astype("datetime64[us]").astype("int64")
            f["corr_start_us"] = us
            f["corr_end_us"] = us + f["Flow Duration"].to_numpy(dtype="int64")
            f["corr_start_ms"] = us // 1000
            f["corr_end_ms"] = f.corr_end_us // 1000
            f["day"] = date
            all_labels.update(f.groupby(["Label", "Attempted Category"], dropna=False).size().to_dict())
            for key, g in f.groupby("Label"):
                bounds = [int(g.corr_start_us.min()), int(g.corr_end_us.max())]
                prev = ranges.setdefault(key, bounds)
                prev[0] = min(prev[0], bounds[0])
                prev[1] = max(prev[1], bounds[1])
            mask = (f["Src IP"] == source) | (f["Dst IP"] == source) | f["corr_start_ms"].isin(candidate_time)
            if mask.any():
                g = f.loc[mask].drop(columns="Timestamp")
                t = pa.Table.from_pandas(g, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(part, t.schema, compression="zstd")
                writer.write_table(t)
                selected += len(g)
            rows += len(f)
            if rows // 1000000 > checkpoint:
                checkpoint = rows // 1000000
                print(date, rows, dict(all_labels), flush=True)
    finally:
        if writer:
            writer.close()
        inp.close()
    assert r.inflate.eof and not r.inflate.unused_data
    assert r.size == descriptor["uncompressed_bytes"] and r.crc == descriptor["crc32"]
    part.replace(result)
    rec = {
        **descriptor,
        "full_read_rows": rows,
        "selected_for_join": selected,
        "selected_file": result.name,
        "selection": "Every corrected flow touching the specified scanner host in either direction plus every flow starting in a +/-1ms bin of a deterministic UQ metadata probe, on the date.",
        "crc_verified": True,
        "uncompressed_sha256": r.check.hexdigest(),
        "compressed_sha256": r.compressed_hash.hexdigest(),
        "counts_by_original_CNS_label": [
            {"label": label, "attempted_category": str(cat), "rows": int(n)}
            for (label, cat), n in sorted(all_labels.items())
        ],
        "per_label_min_start_us_max_end_us": ranges,
    }
    (RAW / (date + ".corrected_receipt.json")).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf8")
    print("complete", date, rows, selected, flush=True)


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as pool:
        list(pool.map(extract, NAMES))
