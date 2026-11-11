"""Fetch only two corrected flow CSV members by verified ZIP64 offsets. No model activity."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from zipfile import ZipFile
import struct
import time
import io
import json
import requests

from audit_paths import CORRECTED as RAW

URL = "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip"
SIZE = 10426851729
ETAG = '"26d7d2191-5f86fb8b66b7b"'
NAMES = ["Wednesday-28-02-2018.csv", "Thursday-01-03-2018.csv"]


def fetch(a, b):
    for retry in range(5):
        try:
            r = requests.get(
                URL,
                headers={"Range": f"bytes={a}-{b}", "If-Match": ETAG, "Accept-Encoding": "identity"},
                timeout=(15, 75),
            )
            assert (
                r.status_code == 206
                and r.headers["Content-Range"] == f"bytes {a}-{b}/{SIZE}"
                and r.headers["ETag"] == ETAG
                and len(r.content) == b - a + 1
            )
            return r.content
        except (ValueError, AssertionError, requests.RequestException):
            if retry == 4:
                raise
            time.sleep((retry + 1) * 2)


class RemoteTail(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.first = SIZE - len(data)

    def seek(self, offset, whence=0):
        return super().seek(offset - self.first if whence == 0 else offset, whence) + self.first

    def tell(self):
        return super().tell() + self.first


def member_info():
    with ZipFile(RemoteTail(fetch(SIZE - 65536, SIZE - 1))) as z:
        entries = []
        for name in NAMES:
            i = z.getinfo(name)
            h = fetch(i.header_offset, i.header_offset + 511)
            assert h[:4] == b"PK\x03\x04" and i.compress_type == 8
            n, x = struct.unpack_from("<HH", h, 26)
            assert h[30 : 30 + n].decode() == name
            entries.append(
                {
                    "name": name,
                    "method": "deflate",
                    "offset": i.header_offset + 30 + n + x,
                    "compressed_bytes": i.compress_size,
                    "uncompressed_bytes": i.file_size,
                    "crc32": i.CRC,
                }
            )
        return entries


def chunk(task):
    e, start, n = task
    block = fetch(e["offset"] + start, e["offset"] + start + n - 1)
    return e["name"], start, block


def run():
    entries = member_info()
    print(entries, flush=True)
    size = 8 << 20
    tasks = []
    outputs = {}
    indices = {}
    for e in entries:
        name = e["name"]
        (RAW / f"{name}.descriptor.json").write_text(json.dumps(e, indent=2) + "\n", encoding="utf-8")
        f = RAW / (name + ".deflate")
        journal = RAW / (name + ".received")
        if not f.exists() or not journal.exists():
            f.open("wb").close()
            journal.write_text("", encoding="ascii")
        known = set(journal.read_text(encoding="ascii").splitlines())
        outputs[name] = f.open("r+b")
        indices[name] = journal.open("a", encoding="ascii")
        for a in range(0, e["compressed_bytes"], size):
            if str(a) not in known:
                tasks.append((e, a, min(size, e["compressed_bytes"] - a)))
    tasks.sort(key=lambda t: t[1])
    start_time = time.monotonic()
    done = 0
    with ThreadPoolExecutor(8) as pool:
        todo = list(tasks)
        pending = set()
        while todo or pending:
            while len(pending) < 8 and todo:
                pending.add(pool.submit(chunk, todo.pop(0)))
            if not pending:
                break
            fut = next(as_completed(pending))
            pending.remove(fut)
            name, at, data = fut.result()
            fout = outputs[name]
            fout.seek(at)
            fout.write(data)
            fout.flush()
            inds = indices[name]
            inds.write(str(at) + "\n")
            inds.flush()
            done += len(data)
            if done % (128 << 20) < size:
                print(f"{done:,} bytes in {time.monotonic() - start_time:.0f}s", flush=True)
    for f in list(outputs.values()) + list(indices.values()):
        f.close()
    print("retrieved", done, "in", time.monotonic() - start_time, flush=True)


if __name__ == "__main__":
    run()
