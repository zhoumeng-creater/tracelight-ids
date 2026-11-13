"""Compare port and bidirectional packet support for the missed downloads without refitting."""

from concurrent.futures import ThreadPoolExecutor
import gzip
import json

import pandas as pd

from RH_raw8 import HERE, INPUT


def summarize(group):
    cap, rows = group
    pairs = set(zip(rows.orig_pkts, rows.resp_pkts, strict=True))
    all_benign = all_service = pair_hits = 0
    with gzip.open(INPUT / (cap + ".conn.log.labeled.gz"), "rt", encoding="utf8") as stream:
        for line in stream:
            if line.startswith("#fields"):
                names = line.split()[1:]
                break
        reader = pd.read_csv(
            stream,
            names=names,
            sep=r"\s+",
            usecols=["label", "proto", "id.resp_p", "orig_pkts", "resp_pkts"],
            comment="#",
            na_filter=False,
            chunksize=260000,
            quoting=3,
        )
        with reader:
            for frame in reader:
                normal = frame.label.str.lower().eq("benign")
                service = normal & frame.proto.eq("tcp") & frame["id.resp_p"].eq(80)
                all_benign += int(normal.sum())
                all_service += int(service.sum())
                ix = frame.loc[service, ["orig_pkts", "resp_pkts"]]
                if len(ix):
                    ix = ix.apply(pd.to_numeric, errors="coerce")
                    pair_hits += int(pd.MultiIndex.from_frame(ix).isin(pairs).sum())
    fields = ("orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
    return dict(
        capture=cap,
        raw_missed_downloads=len(rows),
        normal=all_benign,
        same_tcp80_normal=all_service,
        same_tcp80_and_both_packet_counts_normal=pair_hits,
        native=rows.native.value_counts().to_dict(),
        all_positive_responder_packets=bool((rows.resp_pkts > 0).all()),
        field_ranges={f: [int(rows[f].min()), int(rows[f].max())] for f in fields},
    )


if __name__ == "__main__":
    m = pd.read_csv(HERE / "RH_raw8_missed.csv")
    d = m[m.family == "Download"]
    assert ((d.ip_protocol == 6) & (d.resp_port == 80)).all()
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(summarize, d.groupby("capture")))
    (HERE / "RH_raw8_download_services.json").write_text(json.dumps(results, indent=2), encoding="utf8")
    print(results)
