"""Inspect official AIT Harrison archive metadata without opening flow evidence."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

RECORD = 19483937
LANDING_URL = f"https://zenodo.org/records/{RECORD}"
API_URL = f"https://zenodo.org/api/records/{RECORD}"
ARCHIVES = {
    "harrison.zip": (16_773_705_987, "md5:4fbe3dd17a194776b2851116f8b51380"),
    "harrison_no-pcaps.zip": (667_985_140, "md5:4e837053ce4c5496227e0a3b65048c1c"),
}
MAX_RECORD_BYTES = 1024 * 1024
MAX_RANGE_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_METADATA_MEMBER_BYTES = 4096
USER_AGENT = "TraceLight-IDS (dataset qualification metadata)"
HOSTS = ("intranet_server", "cloud_share")
METADATA_PATHS = ("dataset.yaml", "processing/config/servers.yaml")


class RangeReader(io.RawIOBase):
    """Adapt a pinned HTTP object to the standard library's seekable ZIP reader."""

    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length
        self.position = 0
        self.receipts = []
        self.requested_bytes = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_CUR:
            offset += self.position
        elif whence == os.SEEK_END:
            offset += self.length
        elif whence != os.SEEK_SET:
            raise ValueError("Unsupported seek origin")
        if offset < 0:
            raise OSError("Seek before start of archive")
        self.position = offset
        return offset

    def read(self, size=-1):
        remaining = max(0, self.length - self.position)
        size = remaining if size < 0 else min(remaining, size)
        if size == 0:
            return b""
        if size > MAX_RANGE_BYTES:
            raise ValueError("Requested range exceeds the qualification limit")
        start = self.position
        end = start + size - 1
        expected = f"bytes {start}-{end}/{self.length}"
        url = f"{LANDING_URL}/files/{self.name}?download=1&qualification_range={start}-{end}"
        request = Request(url, headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
            "User-Agent": USER_AGENT,
        })
        for attempt in range(2):
            if self.requested_bytes + size > MAX_ARCHIVE_BYTES:
                raise ValueError("Requested ranges exceed the archive qualification limit")
            self.requested_bytes += size
            try:
                with urlopen(request, timeout=20) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Official archive did not return the requested range: {expected}")
                    payload = response.read(size + 1)
                break
            except HTTPError as error:
                if attempt or error.code not in (429, 500, 502, 503, 504):
                    raise
            except (URLError, TimeoutError):
                if attempt:
                    raise
        if len(payload) != size:
            raise ValueError(f"Official archive returned an incomplete range: {expected}")
        self.position += size
        self.receipts.append({
            "url": url, "content_range": expected, "bytes": size,
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        return payload


def official_metadata():
    request = Request(API_URL, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        data = response.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("Official record is larger than the metadata limit")
    record = json.loads(data)
    if record["id"] != RECORD:
        raise ValueError("Official release identifier differs")
    files = {row["key"]: (row["size"], row["checksum"]) for row in record["files"]}
    if any(files.get(name) != spec for name, spec in ARCHIVES.items()):
        raise ValueError("Official archive sizes or checksum declarations differ")
    return {"record": RECORD, "url": API_URL, "sha256": hashlib.sha256(data).hexdigest()}


def summarize_index(archive):
    members = [entry for entry in archive.infolist() if not entry.is_dir()]
    names = [entry.filename.lower() for entry in members]
    actor_logs = [entry for entry in members if re.fullmatch(
        r"gather/(?:internal_employee|external_employee|remote_employee)_\d+/logs/sm\.log",
        entry.filename.lower(),
    )]
    access_logs = {
        host: [entry for entry in members if re.fullmatch(
            rf"gather/{host}/logs/apache2/[^/]*access\.log(?:\.\d+)?(?:\.gz)?",
            entry.filename.lower(),
        )]
        for host in HOSTS
    }
    neutral_paths = set(METADATA_PATHS)
    neutral_paths.update(entry.filename for entries in access_logs.values() for entry in entries)
    return {
        "file_member_count": len(members),
        "dataset_manifest_present": "dataset.yaml" in names,
        "service_inventory_present": "processing/config/servers.yaml" in names,
        "normal_actor_log_member_count": len(actor_logs),
        "nonempty_normal_actor_log_member_count": sum(entry.file_size > 0 for entry in actor_logs),
        "neutral_member_headers": [
            {
                "path": entry.filename, "bytes": entry.file_size,
                "compressed_bytes": entry.compress_size, "zip_crc32": f"{entry.CRC:08x}",
            }
            for entry in members if entry.filename in neutral_paths
        ],
        "service_members": {
            host: {
                "network_capture_member_count": sum(
                    name.startswith(f"gather/{host}/") and re.search(r"\.pcap(?:ng)?(?:\.|$)", name)
                    is not None for name in names
                ),
                "access_log_member_count": len(access_logs[host]),
                "nonempty_access_log_member_count": sum(
                    entry.file_size > 0 for entry in access_logs[host]
                ),
                "label_member_count": sum(name.startswith(f"labels/{host}/") for name in names),
            }
            for host in HOSTS
        },
    }


def read_time_manifest(archive):
    entries = [entry for entry in archive.infolist() if entry.filename == "dataset.yaml"]
    if len(entries) != 1:
        raise ValueError("Expected one exact dataset.yaml member")
    entry = entries[0]
    if max(entry.file_size, entry.compress_size) > MAX_METADATA_MEMBER_BYTES:
        raise ValueError("Dataset manifest is larger than the qualification limit")
    raw = archive.read(entry)
    # Retain only the public observation bounds. Do not retain raw manifest contents.
    dates = {}
    text = raw.decode("utf-8")
    iso = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?"
    for key in ("start", "end"):
        matches = re.findall(rf"^{key}:[ \t]*['\"]?({iso})['\"]?[ \t]*\r?$",
                             text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"Unexpected dataset manifest field: {key}")
        dates[key] = datetime.fromisoformat(matches[0]).isoformat()
    start, end = (datetime.fromisoformat(dates[key]) for key in ("start", "end"))
    if (start.tzinfo is None) != (end.tzinfo is None) or start >= end:
        raise ValueError("Dataset manifest has an invalid observation interval")
    return {
        "path": entry.filename, "size": len(raw), "zip_crc32": f"{entry.CRC:08x}",
        "sha256": hashlib.sha256(raw).hexdigest(), **dates,
        "timezone_reported": start.tzinfo is not None,
    }


def qualify(*, time_metadata=False):
    report = {
        "official_metadata": official_metadata(),
        "inspection": "archive_index_and_time_manifest" if time_metadata else "archive_index_only",
        "eligibility_established": False,
        "admission_status": "not_admitted",
        "unverified": ["event_identity", "normal_business_denominators", "network_observation_clock"],
        "archives": {},
    }
    for name, (size, checkpoint) in ARCHIVES.items():
        with RangeReader(name, size) as reader, zipfile.ZipFile(reader) as archive:
            item = summarize_index(archive)
            item.update(archive_size=size, published_checksum=checkpoint, range_receipts=reader.receipts)
            if time_metadata and name == "harrison_no-pcaps.zip":
                item["observation_interval"] = read_time_manifest(archive)
            report["archives"][name] = item
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON evidence file outside the worktree")
    parser.add_argument("--time-metadata", action="store_true", help="Read only dataset.yaml start/end from the no-PCAP ZIP")
    args = parser.parse_args()
    output = args.output.resolve()
    worktree = Path(__file__).resolve().parents[2]
    if output.is_relative_to(worktree):
        parser.error("Evidence output must be outside the source worktree")
    if output.exists():
        parser.error("Evidence output already exists")
    report = qualify(time_metadata=args.time_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(output)


if __name__ == "__main__":
    main()
