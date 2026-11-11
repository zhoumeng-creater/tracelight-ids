"""Project all two-day native one-packet SYN rows for direct packet/flow correspondence."""

from zipfile import ZipFile
import numpy as np
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
from audit_paths import UQ_WORK
from audit_cic2018_nf import FILE, OUT, PROBE, RULES

import pandas as pd


def main():
    dest = UQ_WORK / "cic2018_native_packet_probes.parquet"
    fields = [*PROBE, "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"]
    types = {k: pa.int64() for k in fields}
    for k in ("Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
        types[k] = pa.string()
    exact = np.load(OUT / "cic2018_syn47_rows.npy")
    writer = None
    count = 0
    rows = 0
    with (
        ZipFile(FILE) as z,
        z.open(next(m for m in z.namelist() if m.endswith("/NF-CICIDS2018-v3.csv"))) as raw,
    ):
        reader = csv.open_csv(
            raw,
            read_options=csv.ReadOptions(block_size=1 << 24),
            convert_options=csv.ConvertOptions(include_columns=fields, column_types=types),
        )
        for batch in reader:
            df = batch.to_pandas()
            syn = (
                df.PROTOCOL.eq(6)
                & df.IN_PKTS.eq(1)
                & df.OUT_PKTS.eq(0)
                & df.TCP_FLAGS.eq(2)
                & df.CLIENT_TCP_FLAGS.eq(2)
                & df.SERVER_TCP_FLAGS.eq(0)
            )
            for day, (scanner, targets, lo, hi) in RULES.items():
                start = pd.Timestamp(day, tz="UTC").value // 1000000
                sel = syn & df.FLOW_END_MILLISECONDS.between(start, start + 86400000 - 1)
                if not sel.any():
                    continue
                f = df.loc[sel].copy()
                f["day"] = day
                f["one_based_id"] = f.index + rows + 1
                f["row_position_zero_based"] = f.one_based_id - 1
                f["exact47_verified_direction_candidate"] = f.one_based_id.isin(exact)
                f["involves_scanner"] = f.IPV4_SRC_ADDR.eq(scanner) | f.IPV4_DST_ADDR.eq(scanner)
                listed = {f"172.31.69.{i}" for i in targets}
                t = f.FLOW_START_MILLISECONDS
                window = t.ge(-(-lo // 1000000)) & ((t + 1) * 1000000 - 1).le(hi)
                f["match_forward_rule"] = (
                    window & f.IPV4_SRC_ADDR.eq(scanner) & f.IPV4_DST_ADDR.isin(listed) & f.L4_SRC_PORT.ne(68)
                )
                f["match_reversed_rule"] = (
                    window & f.IPV4_DST_ADDR.eq(scanner) & f.IPV4_SRC_ADDR.isin(listed) & f.L4_DST_PORT.ne(68)
                )
                f["raw_uq_normal_outside_rule"] = f.Label.eq(0) & ~(
                    f.match_forward_rule | f.match_reversed_rule
                )
                table = pa.Table.from_pandas(f, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        dest.with_suffix(".parquet.part"), table.schema, compression="zstd"
                    )
                writer.write_table(table)
                count += len(f)
            rows += len(df)
    writer.close()
    assert rows == 20115529
    dest.with_suffix(".parquet.part").replace(dest)
    print(count, dest.stat().st_size)


if __name__ == "__main__":
    main()
