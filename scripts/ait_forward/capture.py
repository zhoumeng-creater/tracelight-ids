"""Fetch the two pinned AIT Harrison target-service capture members."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
import zlib

from scripts.ait_forward.qualification import API_URL, ARCHIVES, LANDING_URL, RangeReader, official_metadata


_ARCHIVE = "harrison.zip"
_CHUNK = 8 * 1024 * 1024
_MEMBERS = {
    "cloud_share": (
        "gather/cloud_share/logs/suricata/log.pcap.1643880216",
        826834598, 753684692, 0xfb5342ac,
    ),
    "intranet_server": (
        "gather/intranet_server/logs/suricata/log.pcap.1643880208",
        996130302, 878334915, 0x6690de48,
    ),
}


def _member(archive, host):
    name, size, compressed_size, crc = _MEMBERS[host]
    matches = [entry for entry in archive.infolist() if entry.filename == name]
    if len(matches) != 1:
        raise ValueError(f"Archive must contain exactly one {name}")
    entry = matches[0]
    if (entry.file_size, entry.compress_size, entry.CRC, entry.compress_type) != (
        size, compressed_size, crc, 8,
    ):
        raise ValueError(f"Archive member metadata differs for {name}")
    return entry


def _remote_metadata():
    size, _ = ARCHIVES[_ARCHIVE]
    with RangeReader(_ARCHIVE, size) as source, ZipFile(source) as archive:
        result = {}
        for host in _MEMBERS:
            entry = _member(archive, host)
            source.seek(entry.header_offset)
            header = source.read(30)
            if header[:4] != b"PK\x03\x04" or struct.unpack_from("<H", header, 8)[0] != 8:
                raise ValueError(f"Unexpected local ZIP header for {entry.filename}")
            name_size, extra_size = struct.unpack_from("<HH", header, 26)
            if source.read(name_size) != entry.filename.encode("ascii"):
                raise ValueError(f"Local ZIP member differs for {entry.filename}")
            result[host] = entry, source.tell() + extra_size
    return result


def _range(bounds):
    start, end = bounds
    length = end - start + 1
    archive_size = ARCHIVES[_ARCHIVE][0]
    expected = f"bytes {start}-{end}/{archive_size}"
    for attempt in range(4):
        base = f"{API_URL}/files/{_ARCHIVE}/content" if attempt % 2 else f"{LANDING_URL}/files/{_ARCHIVE}"
        request = Request(
            f"{base}?download=1&qualification_range={start}-{end}",
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                     "User-Agent": "TraceLight-IDS (pinned AIT capture intake)"},
        )
        try:
            with urlopen(request, timeout=24) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Official archive did not return the requested range: {expected}")
                data = response.read(length + 1)
                if len(data) != length:
                    raise OSError(f"Short official archive range: {expected}")
                return data
        except OSError as error:
            if attempt == 3:
                raise OSError(f"Could not retrieve the official archive range {expected}") from error
            time.sleep(attempt + 1)


def _cached_range(bounds, cache):
    start, end = bounds
    path = cache / f"{start}-{end}.chunk"
    if path.exists():
        packed = path.read_bytes()
        data, digest = packed[:-32], packed[-32:]
        if len(data) == end - start + 1 and hashlib.sha256(data).digest() == digest:
            return data
    data = _range(bounds)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as stream:
        temp = Path(stream.name)
        stream.write(data)
        stream.write(hashlib.sha256(data).digest())
    os.replace(temp, path)
    return data


def _remote_chunks(entry, data_start, cache):
    cache.mkdir(parents=True, exist_ok=True)
    decompressor = zlib.decompressobj(wbits=-15)
    finish = data_start + entry.compress_size
    starts = range(data_start, finish, _CHUNK)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for block in range(0, len(starts), 4):
            bounds = [(start, min(start + _CHUNK, finish) - 1) for start in starts[block:block + 4]]
            for data in workers.map(lambda item: _cached_range(item, cache), bounds):
                yield decompressor.decompress(data)
    yield decompressor.flush()
    if not decompressor.eof or decompressor.unused_data:
        raise ValueError("Official archive member DEFLATE stream has the wrong extent")


def _chunks(stream):
    return iter(lambda: stream.read(_CHUNK), b"")


def _copy_and_measure(chunks, limit, target=None):
    digest = hashlib.sha256()
    crc = size = 0
    for data in chunks:
        size += len(data)
        if size > limit:
            raise ValueError("Uncompressed capture exceeds its official size")
        digest.update(data)
        crc = zlib.crc32(data, crc)
        if target is not None:
            target.write(data)
    return size, crc, digest.hexdigest()


def _write_receipt(destination, result):
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != result:
            raise ValueError(f"Existing capture receipt differs: {destination}")
        return
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def _store(output, host, entry, chunks):
    path = output / f"{host}.pcap"
    temporary = None
    try:
        if path.exists():
            with path.open("rb") as source:
                size, crc, digest = _copy_and_measure(_chunks(source), entry.file_size)
        else:
            with tempfile.NamedTemporaryFile(dir=output, delete=False) as target:
                temporary = Path(target.name)
                size, crc, digest = _copy_and_measure(chunks, entry.file_size, target)
        if (size, crc) != (entry.file_size, entry.CRC):
            raise ValueError(f"Capture length or CRC differs for {entry.filename}")
        if temporary is not None:
            os.replace(temporary, path)
        result = {
            "source": LANDING_URL, "archive": _ARCHIVE,
            "published_archive_checksum": ARCHIVES[_ARCHIVE][1],
            "archive_member": entry.filename, "compressed_bytes": entry.compress_size,
            "bytes": size, "zip_crc32": f"{crc:08x}", "sha256": digest,
            "capture_host": host, "file": path.name,
        }
        _write_receipt(output / f"{host}.json", result)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(output, archive_path=None, host=None):
    output.mkdir(parents=True, exist_ok=True)
    hosts = (host,) if host else _MEMBERS
    if archive_path is not None:
        size, digest = ARCHIVES[_ARCHIVE]
        with archive_path.open("rb") as stream:
            if archive_path.stat().st_size != size or hashlib.file_digest(stream, "md5").hexdigest() != digest[4:]:
                raise ValueError("Local archive must match the official release size and MD5")
            with ZipFile(stream) as archive:
                for member_host in hosts:
                    entry = _member(archive, member_host)
                    with archive.open(entry) as source:
                        yield _store(output, member_host, entry, _chunks(source))
    else:
        official_metadata()
        metadata = _remote_metadata()
        for member_host in hosts:
            entry, data_start = metadata[member_host]
            cache = output / f".{member_host}-ranges"
            result = _store(output, member_host, entry, _remote_chunks(entry, data_start, cache))
            if cache.exists():
                for start in range(data_start, data_start + entry.compress_size, _CHUNK):
                    end = min(start + _CHUNK, data_start + entry.compress_size) - 1
                    (cache / f"{start}-{end}.chunk").unlink(missing_ok=True)
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-archive", type=Path)
    parser.add_argument("--host", choices=_MEMBERS)
    args = parser.parse_args()
    for result in fetch(args.output, args.local_archive, args.host):
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
