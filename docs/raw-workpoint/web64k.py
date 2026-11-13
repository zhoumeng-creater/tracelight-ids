"""Count support of an exploratory eight-field download rule on previously opened data."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from RH_raw8 import HERE, REPO, ROOT, INPUT, PACKAGE


def count(args):
    source, spec, pkg, tag, calibration = args
    name = spec["name"]
    with ZipFile(pkg) as z:
        if tag == "R21":
            frozen = pd.read_parquet(
                BytesIO(z.read(f"evidence/{name}/row-scores-and-alerts.parquet")), columns=["raw_fixed"]
            )
            raw = frozen.raw_fixed.to_numpy()
        elif spec["role"] in ("R", "H"):
            flags = pd.read_parquet(BytesIO(z.read(f"evidence/blind/{name}/hgb.parquet")), columns=["flags"])
            raw = (flags["flags"].to_numpy() & 1) != 0
        elif spec["role"] == "C":
            frozen = calibration[calibration.capture == name]
            assert np.array_equal(frozen.row_id, np.arange(1, len(frozen) + 1))
            raw = (frozen["flags"].to_numpy() & 1) != 0
        else:
            raw = None
    by_day = Counter()
    off = 0
    prev = -1
    with gzip.open(source, "rt", encoding="utf8") as stream:
        for line in stream:
            if line.startswith("#fields"):
                names = ["detailed-label" if f == "det_label" else f for f in line.split()[1:]]
                break
        needed = ["ts", "duration", "label", "detailed-label", "proto", "id.resp_p", "resp_ip_bytes"]
        with pd.read_csv(
            stream,
            names=names,
            usecols=needed,
            sep=r"\s+",
            comment="#",
            na_filter=False,
            dtype=str,
            chunksize=250000,
            quoting=3,
        ) as reader:
            for f in reader:
                ts = np.rint(pd.to_numeric(f.ts).to_numpy() * 1e6).astype(np.int64)
                duration = np.rint(
                    pd.to_numeric(f.duration.replace({"-": "0", "": "0"})).to_numpy() * 1e6
                ).astype(np.int64)
                clock = np.maximum.accumulate(np.maximum((ts + duration + 999) // 1000, prev))
                prev = clock[-1]
                days = clock // 86400000
                b = f.label.str.lower().eq("benign").to_numpy()
                native = f["detailed-label"].str.lower()
                fam = np.select(
                    [
                        b,
                        native.eq("c&c"),
                        native.str.contains("filedownload", regex=False),
                        native.str.contains("heartbeat", regex=False),
                        native.eq("partofahorizontalportscan"),
                    ],
                    ["Benign", "PureCC", "Download", "Heartbeat", "Scanning"],
                    default="OtherAttack",
                )
                svc = (f.proto.eq("tcp") & f["id.resp_p"].eq("80")).to_numpy()
                web = svc & (
                    pd.to_numeric(f.resp_ip_bytes.replace({"-": "-1", "": "-1"})).to_numpy() >= 65536
                )
                base = raw[off : off + len(f)] if raw is not None else np.zeros(len(f), bool)
                for (d, k), ix in (
                    pd.DataFrame({"day": days, "kind": fam})
                    .groupby(["day", "kind"], sort=False)
                    .indices.items()
                ):
                    day = np.datetime_as_string(
                        np.datetime64("1970-01-01") + np.timedelta64(int(d), "D"), unit="D"
                    )
                    for key, val in (
                        ("rows", len(ix)),
                        ("tcp80", svc[ix].sum()),
                        ("web64k", web[ix].sum()),
                        ("raw", base[ix].sum()),
                        ("gained", (web[ix] & ~base[ix]).sum()),
                    ):
                        by_day[day, k, key] += int(val)
                off += len(f)
    assert off == spec["rows"] and (raw is None or off == len(raw))
    out = []
    for d, k in sorted({(d, k) for d, k, f in by_day}):
        vals = {f: by_day[d, k, f] for f in ("rows", "tcp80", "web64k", "raw", "gained")}
        if raw is None:
            vals["raw"] = vals["gained"] = None
        out.append(dict(round=tag, role=spec["role"], capture=name, utc_day=d, endpoint=k, **vals))
    return out


def main():
    old = next((ROOT.parent / "20261111_第二十一轮前向阈值").glob("*.zip"))
    with ZipFile(PACKAGE) as z:
        reg = json.loads(z.read("protocol.json"))["data"]
        c = pd.read_parquet(BytesIO(z.read("evidence/calibration/hgb.parquet")))
    with ZipFile(old) as z:
        r21 = json.loads(z.read("protocol.json"))["registration"]
    todo = [
        (INPUT / (s["name"] + ".conn.log.labeled.gz"), s, PACKAGE, "R23", c)
        for s in reg["sources"]
        if s["role"] in ("P", "C", "R", "H")
    ]
    for spec in r21["sources"]:
        if spec["name"].endswith(("34-1", "49-1")):
            source = REPO / "data/raw/iot23-forward-20261111" / (spec["name"] + ".conn.log.labeled.gz")
            todo.append((source, spec, old, "R21", c))
    # Small controls are reported first. Large sources are streamed once.
    with ThreadPoolExecutor(4) as pool:
        rows = []
        for capture in pool.map(count, sorted(todo, key=lambda a: a[1]["rows"])):
            rows.extend(capture)
            print(
                capture[0]["capture"],
                [
                    (s["utc_day"], s["rows"], s["tcp80"], s["web64k"], s["gained"])
                    for s in capture
                    if s["endpoint"] == "Benign"
                ],
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "web64k_逐日.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
