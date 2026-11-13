"""Audit the earliest complete daily ISP QUIC member without downloading the monthly ZIP."""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

import numpy as np
import pyarrow.parquet as pq

from ids.contact_evidence.download import ssl_context

URL = "https://zenodo.org/records/17249078/files/2024-06.zip"
LENGTH = 1747398591
MD5 = "4b1fd8bcf5ddd143f7b350aa7c0d4814"
EARLIEST_DAY = "20240601.parquet"
MEMBER_SHA = "0391bfbc2c489e944bb0f1093c483f9d6b4f92c16295f96fcd47991378c7e28a"
FIELDS = ["DST_PORT", "PROTOCOL", "DURATION", "QUIC_SNI"]


def span(start, stop):
    headers = {"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity"}
    req = Request(URL + f"?span={start}-{stop - 1}", headers=headers)
    for attempt in range(3):
        try:
            with urlopen(req, timeout=60, context=ssl_context()) as resp:
                if (
                    resp.status != 206
                    or resp.headers["Content-Range"] != f"bytes {start}-{stop - 1}/{LENGTH}"
                ):
                    raise ValueError("Server returned wrong archive byte interval")
                part = resp.read(stop - start + 1)
                if len(part) != stop - start:
                    raise ValueError("Incomplete archive byte interval")
                return part
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def get_first_day():
    # End-of-central-directory and the complete small central directory are in the last 64 KiB.
    start = LENGTH - 65536
    tail = span(start, LENGTH)
    with ZipFile(BytesIO(tail)) as index:
        entries = sorted(index.infolist(), key=lambda e: e.filename)
        chosen = entries[0]
        assert chosen.filename == EARLIEST_DAY and len(entries) == 30
        offset = start + chosen.header_offset
        next_offset = min(start + i.header_offset for i in entries if i.header_offset > chosen.header_offset)
    segment = span(offset, next_offset)
    assert segment[:4] == b"PK\x03\x04"
    name_len, extra_len = struct.unpack_from("<HH", segment, 26)
    assert segment[30 : 30 + name_len] == EARLIEST_DAY.encode()
    data_start = 30 + name_len + extra_len
    assert chosen.compress_type == 8
    raw = zlib.decompress(segment[data_start : data_start + chosen.compress_size], -15)
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    assert len(raw) == chosen.file_size and crc == chosen.CRC
    if hashlib.sha256(raw).hexdigest() != MEMBER_SHA:
        raise ValueError("Daily Parquet digest differs from original audit")
    return raw, {
        "official_data_record": "https://zenodo.org/records/17249078",
        "archive_url": URL,
        "archive_total_bytes": LENGTH,
        "publisher_monthly_md5": MD5,
        "selection": "Earliest complete day in first published month; all rows of that Parquet member.",
        "central_directory_tail_sha256": hashlib.sha256(tail).hexdigest(),
        "members_in_month": len(entries),
        "member": chosen.filename,
        "compressed_bytes": chosen.compress_size,
        "extracted_bytes": chosen.file_size,
        "data_byte_range_inclusive": [offset + data_start, offset + data_start + chosen.compress_size - 1],
        "member_crc32": f"{crc:08x}",
        "member_sha256": hashlib.sha256(raw).hexdigest(),
    }


def counts(mask, vals):
    x = vals[mask]
    return {
        "support": len(x),
        **{f"at_least_{t}s": int((x >= t).sum()) for t in (10, 30, 60)},
        "duration_quantiles_0_50_90_99_100": list(map(float, np.quantile(x, [0, 0.5, 0.9, 0.99, 1])))
        if len(x)
        else [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw, report = get_first_day()
    frame = pq.read_table(BytesIO(raw), columns=FIELDS).to_pandas()
    protocol = frame.PROTOCOL.astype(str)
    if not protocol.isin(["udp", "17", "UDP"]).all():
        raise ValueError(f"Unexpected protocol values: {protocol.unique()}")
    duration = frame.DURATION.to_numpy(dtype=float)
    assert np.isfinite(duration).all() and (duration >= 0).all()
    port = frame.DST_PORT.to_numpy() == 443
    sni = frame.QUIC_SNI.fillna("").str.lower().str.rstrip(".")
    found = port & (duration >= 30)
    named = found & sni.ne("").to_numpy()
    # Fixed well-known DNS suffixes with exact domain boundaries; SNI is a service hint, not an IDS label.
    report.update(
        {
            "sampling": "CESNET3, uniform 1:100; bidirectional UDP flows with QUIC metadata.",
            "label_scope": "Passive application traffic, no author benign/malicious IDS truth. Source port omitted. Destination port 443 is sufficient for the either-end rule; non-443 destinations cannot be fully tested.",
            "protocol_counts": dict(Counter(protocol)),
            "all_rows": len(frame),
            "udp_destination_443": counts(port, duration),
            "udp_other_destination": counts(~port, duration),
            "hits_with_nonempty_sni": int(named.sum()),
            "distinct_nonempty_sni_on_hits": int(sni[named].nunique()),
            "hits_by_dns_suffix": {
                key: counts(port & ((sni.eq(key) | sni.str.endswith("." + key)).to_numpy()), duration)
                for key in [
                    "googlevideo.com",
                    "google.com",
                    "youtube.com",
                    "gstatic.com",
                    "facebook.com",
                    "fbcdn.net",
                    "cloudflare.com",
                    "apple.com",
                    "icloud.com",
                    "microsoft.com",
                    "instagram.com",
                    "whatsapp.net",
                ]
            },
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(
        report["member_sha256"],
        report["all_rows"],
        report["udp_destination_443"],
        report["hits_with_nonempty_sni"],
    )


if __name__ == "__main__":
    main()
