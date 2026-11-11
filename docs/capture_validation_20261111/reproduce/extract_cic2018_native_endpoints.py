"""Extract identities of native NF rows involving the two known portscan sources on the campaign days. No scores."""

from zipfile import ZipFile
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

from audit_paths import UQ_WORK
from audit_cic2018_nf import FILE, RULES, DAY_MS, PROBE


def main():
    tmp = UQ_WORK / "cic2018_native_endpoints.parquet.tmp"
    dest = UQ_WORK / "cic2018_native_endpoints.parquet"
    cols = {col: pa.int64() for col in PROBE}
    for col in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"):
        cols[col] = pa.string()
    offsets = 0
    picked = 0
    writer = None
    with (
        ZipFile(FILE) as z,
        z.open(next(m for m in z.namelist() if m.endswith("/NF-CICIDS2018-v3.csv"))) as raw,
    ):
        reader = csv.open_csv(
            raw,
            read_options=csv.ReadOptions(block_size=1 << 24),
            convert_options=csv.ConvertOptions(include_columns=PROBE, column_types=cols),
        )
        for batch in reader:
            df = batch.to_pandas()
            for date, (host, targets, lo, hi) in RULES.items():
                start = pd.Timestamp(date, tz="UTC").value // 1000000
                is_day = df.FLOW_END_MILLISECONDS.ge(start) & df.FLOW_END_MILLISECONDS.lt(start + DAY_MS)
                if not is_day.any():
                    continue
                relevant = is_day & (df.IPV4_SRC_ADDR.eq(host) | df.IPV4_DST_ADDR.eq(host))
                if not relevant.any():
                    continue
                q = df.loc[relevant].copy()
                listed = {f"172.31.69.{i}" for i in targets}
                q["day"] = date
                q["row_position_zero_based"] = q.index + offsets
                window = q.FLOW_START_MILLISECONDS.ge(-(-lo // 1000000)) & (
                    (q.FLOW_START_MILLISECONDS + 1) * 1000000 - 1
                ).le(hi)
                q["match_forward_rule"] = (
                    window & q.IPV4_SRC_ADDR.eq(host) & q.IPV4_DST_ADDR.isin(listed) & q.L4_SRC_PORT.ne(68)
                )
                q["match_reversed_rule"] = (
                    window & q.IPV4_DST_ADDR.eq(host) & q.IPV4_SRC_ADDR.isin(listed) & q.L4_DST_PORT.ne(68)
                )
                out = pa.Table.from_pandas(q, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, out.schema, compression="zstd")
                writer.write_table(out)
                picked += len(out)
            offsets += len(batch)
    if writer is None:
        raise ValueError("No candidate endpoints")
    writer.close()
    assert offsets == 20115529
    tmp.replace(dest)
    print("Coverage identity rows:", picked, "bytes:", dest.stat().st_size)


if __name__ == "__main__":
    main()
