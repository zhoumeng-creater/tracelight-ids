"""Retrieve only five scanner-host members, from the public capture owners' S3 bucket."""

from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
import requests
import io
import re
import struct
import json
import time

from audit_paths import PCAP as RAW

BASE = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
DAYS = [
    (
        "2018-02-28",
        "Wednesday-28-02-2018",
        "172.31.69.24",
        53251694487,
        '"b688b1c7c529c8754fe11aec1a963270-3175"',
    ),
    (
        "2018-03-01",
        "Thursday-01-03-2018",
        "172.31.69.13",
        52358069106,
        '"652d07144e22ca627d2362fa9e77c75e-3121"',
    ),
]


def fetch(url, total, etag, a, b):
    for retry in range(5):
        try:
            r = requests.get(
                url,
                headers={"Range": f"bytes={a}-{b}", "If-Match": etag, "Accept-Encoding": "identity"},
                timeout=(20, 60),
            )
            assert (
                r.status_code == 206
                and r.headers["Content-Range"] == f"bytes {a}-{b}/{total}"
                and r.headers["ETag"] == etag
                and len(r.content) == b - a + 1
            )
            return r.content
        except (requests.RequestException, AssertionError):
            if retry == 4:
                raise
            time.sleep(2 * (retry + 1))


class Tail(io.BytesIO):
    def __init__(self, t, n):
        super().__init__(t)
        self.begin = n - len(t)

    def seek(self, a, w=0):
        return super().seek(a - self.begin if w == 0 else a, w) + self.begin

    def tell(self):
        return super().tell() + self.begin


def discover():
    desc = []
    for day, folder, host, total, etag in DAYS:
        url = BASE + quote("Original Network Traffic and Log data/" + folder + "/pcap.zip")
        tail = fetch(url, total, etag, total - (256 << 10), total - 1)
        with ZipFile(Tail(tail, total)) as z:
            for i in z.infolist():
                if not re.search(re.escape(host) + r"(?:\D|$)", i.filename):
                    continue
                h = fetch(url, total, etag, i.header_offset, i.header_offset + 511)
                assert h[:4] == b"PK\x03\x04" and i.compress_type == 8
                n, x = struct.unpack_from("<HH", h, 26)
                assert h[30 : 30 + n].decode() == i.filename
                desc.append(
                    {
                        "day": day,
                        "host": host,
                        "member": i.filename,
                        "label": day + "-" + str(len(desc)),
                        "input_url": url,
                        "zip_bytes": total,
                        "etag": etag,
                        "offset": i.header_offset + 30 + n + x,
                        "compressed_bytes": i.compress_size,
                        "bytes": i.file_size,
                        "crc32": i.CRC,
                    }
                )
    assert len(desc) == 5
    return desc


def run():
    desc = discover()
    (RAW / "pcap-members.json").write_text(json.dumps(desc, indent=2) + "\n", encoding="utf8")
    filemap = {}
    tasks = []
    for e in desc:
        d = RAW / (e["label"] + ".deflate")
        filemap[e["label"]] = d.open("wb")
        for start in range(0, e["compressed_bytes"], 8 << 20):
            tasks.append((e, start, min(8 << 20, e["compressed_bytes"] - start)))

    def get(t):
        e, a, n = t
        b = fetch(e["input_url"], e["zip_bytes"], e["etag"], e["offset"] + a, e["offset"] + a + n - 1)
        return e["label"], a, b

    jobs = iter(tasks)
    done = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(8) as pool:
        pending = {pool.submit(get, t) for t in [next(jobs) for _ in range(8)]}
        while pending:
            future = next(as_completed(pending))
            pending.remove(future)
            label, a, b = future.result()
            f = filemap[label]
            f.seek(a)
            f.write(b)
            done += len(b)
            t = next(jobs, None)
            if t:
                pending.add(pool.submit(get, t))
            if done % (128 << 20) < (8 << 20):
                print(done, round(time.monotonic() - t0, 1), flush=True)
    for f in filemap.values():
        f.close()
    print("five verified byte ranges complete", done, flush=True)


if __name__ == "__main__":
    run()
