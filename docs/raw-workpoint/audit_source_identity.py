"""Check complete log time, published label counts, and overlap with captures 42-1/44-1.

No model or alert files are opened. Old logs retain only their compact flow keys in memory.
"""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import gzip
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

FIELDS = [f"f{i}" for i in range(21)]
SUBSET = [FIELDS[i] for i in (0, 2, 3, 4, 5, 6, 8, 20)]
DTYPES = {k: pa.float64() if k in {"f0", "f8"} else pa.string() for k in SUBSET}
READ = csv.ReadOptions(column_names=FIELDS, use_threads=False, block_size=4 << 20)
CONVERT = csv.ConvertOptions(include_columns=SUBSET, column_types=DTYPES, null_values=["-"])


def blocks(path, size=4 << 20, *, digest=None):
    with gzip.open(path, "rb") as stream:
        while line := stream.readline():
            if not line.startswith(b"#"):
                break
            if digest is not None:
                digest.update(line)
        while line:
            body = line + stream.read(size)
            if not body.endswith(b"\n"):
                body += stream.readline()
            if digest is not None:
                digest.update(body)
            yield body
            line = stream.readline()


def parse(body):
    settings = csv.ParseOptions(
        delimiter="\t",
        quote_char=False,
        invalid_row_handler=lambda r: "skip" if r.text.startswith("#") else "error",
    )
    return csv.read_csv(BytesIO(body), read_options=READ, parse_options=settings, convert_options=CONVERT)


def micros(t):
    return np.rint(t.to_numpy() * 1e6).astype(np.int64)


def records(table):
    return zip(
        micros(table["f0"]), *(table[k].to_pylist() for k in ["f2", "f3", "f4", "f5", "f6"]), strict=True
    )


def compact_keyset(path):
    result = Counter()
    for body in blocks(path):
        result.update(records(parse(body)))
    return result


def reduce_block(body, oldkeys, oldtime):
    table = parse(body)
    n = len(table)
    if n == 0:
        return None
    start = micros(table["f0"])
    end = (start + micros(pc.fill_null(table["f8"], 0)) + 999) // 1000
    # Exact integer microseconds; first narrow to old times, then check both addresses, ports and protocol.
    ix = np.flatnonzero(pc.is_in(pa.array(start), value_set=oldtime).to_numpy(zero_copy_only=False))
    exact = [Counter() for _ in oldkeys]
    for row in records(table.take(pa.array(ix))):
        for k, earlier in enumerate(oldkeys):
            if row in earlier:
                exact[k][row] += 1
    labels = Counter()
    for item in pc.value_counts(table["f20"]).to_pylist():
        label = item["values"].split()
        if len(label) != 3:
            raise ValueError("Unexpected physical label suffix")
        labels[(label[1], label[2])] += item["counts"]
    return n, int(end[0]), int(end.max()), exact, labels


def check(path, earlier, workers=6, *, digest=None):
    keys = [compact_keyset(p) for p in earlier]
    oldtime = pa.array(sorted({key[0] for items in keys for key in items}), type=pa.int64())
    first, last, total, labels = None, -1, 0, Counter()
    matched = [Counter() for _ in earlier]

    def combine(result):
        nonlocal first, last, total
        if result is None:
            return
        n, beg, end, exact, counts = result
        if first is None:
            first = beg
        last = max(last, end)
        total += n
        for all_hits, hits in zip(matched, exact, strict=True):
            all_hits.update(hits)
        labels.update(counts)

    with ThreadPoolExecutor(workers) as pool:
        queue = deque()
        for b in blocks(path, digest=digest):
            queue.append(pool.submit(reduce_block, b, keys, oldtime))
            if len(queue) > workers + 2:
                combine(queue.popleft().result())
        for fut in queue:
            combine(fut.result())
    return {
        "rows": total,
        "first_flow_end_ms": first,
        "last_flow_end_ms": last,
        "native_label_detail_counts": [
            {"label": key[0], "detail": key[1], "rows": count} for key, count in sorted(labels.items())
        ],
        "key": ["ts_microseconds", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto"],
        "overlap": [
            {
                "source": p.name.split(".")[0],
                "old_rows": sum(k.values()),
                "unique_old_keys": len(k),
                "shared_unique_keys": len(m),
                "old_rows_sharing_key": sum(k[x] for x in m),
                "new_rows_sharing_key": sum(m.values()),
            }
            for p, k, m in zip(earlier, keys, matched, strict=True)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = [args.old_dir / f"CTU-IoT-Malware-Capture-{k}-1.conn.log.labeled.gz" for k in [42, 44]]
    digest = hashlib.sha256()
    result = check(args.log, old, digest=digest)
    result["sha256_uncompressed"] = digest.hexdigest()
    for field in ["first_flow_end_ms", "last_flow_end_ms"]:
        result[field + "_utc"] = datetime.fromtimestamp(result[field] / 1000, timezone.utc).isoformat(
            timespec="milliseconds"
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(result, flush=True)
