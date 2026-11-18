"""Fetch only the selected AIT cloud PCAP member using verified HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import requests

ROOT = Path(__file__).resolve().parent
MEMBER = "gather/cloud_share/logs/suricata/log.pcap.1642684647"
OFFSET, COMPRESSED, SIZE, CRC = 6345484912, 464709023, 525862169, 3074706299
ARCHIVE_SIZE = 7132670599
URL = "https://zenodo.org/api/records/19483937/files/russellmitchell.zip/content"
CHUNK, WORKERS = 16 * 1024 * 1024, 4
CACHE = ROOT / "source" / "cloud_member_chunks"
DEST = ROOT / "source" / "cloud-share-log.pcap.1642684647"


def digest(path):
    h, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return {"sha256": h.hexdigest(), "crc32": crc, "bytes": size}


def get_range(start, end, dest):
    last = None
    for attempt in range(1, 4):
        base = URL if attempt != 2 else "https://zenodo.org/records/19483937/files/russellmitchell.zip"
        url = f"{base}?download=1&ait_cloud_offset={start}"
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}, stream=True, timeout=(15, 45)) as response:
                assert response.status_code == 206, response.status_code
                assert response.headers.get("Content-Range") == f"bytes {start}-{end}/{ARCHIVE_SIZE}"
                written = 0
                with dest.open("wb") as out:
                    for block in response.iter_content(256 * 1024):
                        written += len(block)
                        assert written <= end - start + 1
                        out.write(block)
                assert written == end - start + 1, written
            return url
        except (requests.RequestException, AssertionError, OSError) as exc:
            last = exc
            print(json.dumps({"start": start, "attempt": attempt, "error": str(exc)}), flush=True)
    raise RuntimeError(f"Range {start}-{end} failed") from last


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        actual = digest(DEST)
        assert actual["bytes"] == SIZE and actual["crc32"] == CRC
        print(json.dumps({"reused_verified_pcap": str(DEST), **actual}), flush=True)
        return
    header = CACHE / "local_header.bin"
    if not header.exists():
        get_range(OFFSET, OFFSET + 255, header)
    raw = header.read_bytes()
    fields = struct.unpack("<IHHHHHIIIHH", raw[:30])
    assert fields[0] == 0x04034b50 and fields[3] == 8
    assert raw[30:30 + fields[-2]].decode() == MEMBER
    data_offset = OFFSET + 30 + fields[-2] + fields[-1]
    jobs = [(i, data_offset + i * CHUNK, min(data_offset + (i + 1) * CHUNK, data_offset + COMPRESSED) - 1) for i in range((COMPRESSED + CHUNK - 1) // CHUNK)]

    def one(job):
        index, start, end = job
        path = CACHE / f"{index:04d}.deflate"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            old = json.loads(receipt.read_text(encoding="utf-8"))
            if old["start"] == start and old["end"] == end and old["sha256"] == digest(path)["sha256"] and path.stat().st_size == end - start + 1:
                return path
        temporary = path.with_suffix(".partial")
        source_url = get_range(start, end, temporary)
        temporary.replace(path)
        receipt.write_text(json.dumps({"start": start, "end": end, "source_url": source_url, **digest(path)}, indent=2), encoding="utf-8")
        return path

    began, complete = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(one, job): job for job in jobs}
        for future in as_completed(futures):
            path = future.result()
            complete += path.stat().st_size
            print(json.dumps({"completed_compressed_bytes": complete, "total_compressed_bytes": COMPRESSED, "chunk": futures[future][0], "elapsed_seconds": round(time.perf_counter() - began, 2)}), flush=True)
    decoder = zlib.decompressobj(-15)
    partial = DEST.with_suffix(DEST.suffix + ".partial")
    with partial.open("wb") as target:
        for index, _, _ in jobs:
            with (CACHE / f"{index:04d}.deflate").open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(decoder.decompress(block))
        target.write(decoder.flush())
    assert decoder.eof and not decoder.unused_data
    actual = digest(partial)
    assert actual["bytes"] == SIZE and actual["crc32"] == CRC, actual
    partial.replace(DEST)
    receipt = {"member": MEMBER, "archive_url": URL, "archive_bytes": ARCHIVE_SIZE, "header_offset": OFFSET, "data_offset": data_offset, "compressed_bytes": COMPRESSED, "pcap": str(DEST), **actual, "range_workers": WORKERS, "chunk_bytes": CHUNK, "elapsed_seconds": time.perf_counter() - began}
    (ROOT / "source" / "cloud-source-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
