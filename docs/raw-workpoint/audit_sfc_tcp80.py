"""Check a previously proposed download signature against two sealed SFC captures."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ids.response_evidence.data import prepare_log
from ids.threshold_validation.sources import verify_log


def inspect_capture(z, spec, root, temp, raw_bit):
    name = spec["name"]
    src = verify_log(root / spec["filename"], spec)
    blind, truth = temp / "blind", temp / "truth"
    prepare_log(src, blind, truth, sfc=True)
    df = pd.read_parquet(blind, columns=["row_id", "ip_protocol", "resp_port", "resp_ip_bytes"])
    target = pd.read_parquet(truth, columns=["row_id", "family", "window"])
    sealed = pd.read_parquet(BytesIO(z.read(f"evidence/blind/{name}/hgb.parquet")))
    ids = np.arange(1, 1 + len(df))
    if not np.array_equal(df.row_id, ids) or not np.array_equal(target.row_id, ids):
        raise ValueError("SFC source order mismatch")
    if len(sealed) != len(ids):
        raise ValueError("Sealed row count differs")
    sample = pd.read_parquet(BytesIO(z.read(f"evidence/blind/{name}/hgb-sample.parquet")))
    if not np.array_equal(sealed["flags"].to_numpy()[sample.row_id - 1], sample["flags"]):
        raise ValueError("Sampled physical row IDs differ from sealed flags")
    normal = target.family.eq("Benign")
    tcp80 = df.ip_protocol.eq(6) & df.resp_port.eq(80)
    pattern = tcp80 & df.resp_ip_bytes.ge(65536)
    raw = ((sealed["flags"].to_numpy() >> raw_bit) & 1).astype(bool)
    counts = (
        pd.DataFrame(
            {
                "window": target.window,
                "normal": normal,
                "normal_tcp_80": normal & tcp80,
                "normal_tcp_80_with_at_least_64k": normal & pattern,
                "baseline_fp": normal & raw,
                "added_fp": normal & pattern & ~raw,
                "combined_fp": normal & (raw | pattern),
            }
        )
        .groupby("window")
        .sum()
        .astype(int)
    )
    counts["normal_budget"] = counts.normal // 100
    counts["combined_within_budget"] = counts.combined_fp <= counts.normal_budget
    return {
        "raw_bytes_sha256": spec["sha256"],
        "verified_rows": len(ids),
        "utc_days": counts.reset_index().to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r23-zip", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with ZipFile(args.r23_zip) as z, TemporaryDirectory(dir=args.output.parent) as temp:
        protocol = json.loads(z.read("protocol.json"))
        bit = protocol["bit_order"].index("raw")
        specs = [s for s in protocol["data"]["sources"] if s.get("format") == "iot-sfc"]
        if len(specs) != 2:
            raise ValueError("Expected the two registered SFC snapshots")
        result = {s["name"]: inspect_capture(z, s, args.source_dir, Path(temp), bit) for s in specs}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
