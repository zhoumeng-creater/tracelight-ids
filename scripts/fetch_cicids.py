#!/usr/bin/env python3
"""Fetch the pinned author-repaired CICIDS2017 release, not the original CSVs.

Raw data remains local. The pin is a complete official-host download observed
on 2026-11-01, not a checksum published independently by the authors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.cicids import (CIC_ARCHIVE_BYTES, CIC_ARCHIVE_NAME, CIC_ARCHIVE_SHA256,  # noqa: E402
                        CIC_ARCHIVE_URL, CIC_COLUMNS, CIC_FILES, CIC_LABEL_POLICY_URL,
                        CIC_SOURCE_SHA256, CIC_SOURCE_ROWS)
from ids.data import sha256_file  # noqa: E402

MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


def validate_archive_members(archive: zipfile.ZipFile) -> None:
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(CIC_FILES):
        raise ValueError("ZIP must contain exactly the five known daily CSV files")
    total = 0
    for item in archive.infolist():
        path = PurePosixPath(item.filename)
        mode = item.external_attr >> 16
        if (path.is_absolute() or len(path.parts) != 1 or ".." in path.parts
                or "\\" in item.filename or ":" in item.filename
                or stat.S_ISLNK(mode) or item.is_dir() or item.flag_bits & 1):
            raise ValueError("Unsafe ZIP path/type/encryption")
        total += item.file_size
        if item.file_size < 1 or total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Unexpected uncompressed archive size")


def _download(destination: Path) -> dict:
    digest, count = hashlib.sha256(), 0
    request = Request(CIC_ARCHIVE_URL, headers={"User-Agent": "TraceLight-IDS-reproducible-data/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("xb") as stream:
        if urlparse(response.geturl()).hostname != "intrusion-detection.distrinet-research.be":
            raise ValueError("Unexpected archive host redirect")
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Archive URL returned HTML")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != CIC_ARCHIVE_BYTES:
            raise ValueError("Official archive changed size; refusing an unreviewed release")
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > CIC_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds pinned size")
            digest.update(chunk)
            stream.write(chunk)
        metadata = {"last_modified": response.headers.get("Last-Modified")}
    if count != CIC_ARCHIVE_BYTES or digest.hexdigest() != CIC_ARCHIVE_SHA256:
        raise ValueError("Archive checksum/size differs from the pinned release")
    return metadata


def _inspect_csv(path: Path) -> dict:
    labels: Counter = Counter()
    attempted: Counter = Counter()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, [])) != CIC_COLUMNS:
            raise ValueError(f"{path.name}: expected exact 91-column CNS2022 schema")
        for count, row in enumerate(reader, 1):
            if len(row) != len(CIC_COLUMNS):
                raise ValueError(f"{path.name}: row {count} has invalid field count")
            label, category = row[-2].strip(), row[-1].strip()
            if not label:
                raise ValueError(f"{path.name}: missing label")
            try:
                number = int(category)
            except ValueError as exc:
                raise ValueError(f"{path.name}: invalid Attempted Category") from exc
            if not -1 <= number <= 6:
                raise ValueError(f"{path.name}: invalid Attempted Category range")
            labels[label] += 1
            attempted[str(number)] += 1
    if not count:
        raise ValueError(f"{path.name}: empty source")
    sha = sha256_file(path)
    if sha != CIC_SOURCE_SHA256[path.name] or count != CIC_SOURCE_ROWS[path.name]:
        raise ValueError(f"{path.name}: bytes/row count do not match pinned archive member")
    return {"sha256": sha, "n_bytes": path.stat().st_size, "n_rows": count,
            "n_columns": len(CIC_COLUMNS), "raw_label_counts": dict(labels),
            "attempted_category_counts": dict(attempted)}


def fetch_cicids(output_dir: str | Path) -> Path:
    """Return raw directory after download/extraction and full-byte validation.

    Existing identical files can be reused. Any changed existing archive/CSV or
    provenance fails closed and is never overwritten. Downloads and extraction
    are staged in a private temporary child directory.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path, provenance_path = output / CIC_ARCHIVE_NAME, output / "provenance.json"
    if archive_path.exists():
        if archive_path.stat().st_size != CIC_ARCHIVE_BYTES or sha256_file(archive_path) != CIC_ARCHIVE_SHA256:
            raise ValueError("Existing archive differs from pinned release; refusing overwrite")
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (old.get("data_kind") != "official_repaired" or old.get("archive", {}).get("sha256") != CIC_ARCHIVE_SHA256
                or old.get("archive", {}).get("url") != CIC_ARCHIVE_URL):
            raise ValueError("Existing provenance differs from pinned release")
        if not archive_path.exists():
            raise ValueError("Existing provenance requires its original archive")
        for name in CIC_FILES:
            expected = old.get("files", {}).get(name, {})
            if (expected.get("sha256") != CIC_SOURCE_SHA256[name] or expected.get("n_rows") != CIC_SOURCE_ROWS[name]
                    or not (output / name).is_file() or sha256_file(output / name) != expected.get("sha256")):
                raise ValueError(f"Existing CSV checksum differs: {name}")
        return output
    with tempfile.TemporaryDirectory(prefix=".cic-fetch-", dir=output) as temp:
        stage = Path(temp)
        downloaded = not archive_path.exists()
        headers = _download(stage / CIC_ARCHIVE_NAME) if downloaded else {}
        actual_archive = stage / CIC_ARCHIVE_NAME if downloaded else archive_path
        files = {}
        with zipfile.ZipFile(actual_archive) as archive:
            validate_archive_members(archive)
            for name in CIC_FILES:
                target = stage / name
                with archive.open(name) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
                files[name] = _inspect_csv(target)
                existing = output / name
                if existing.exists() and sha256_file(existing) != files[name]["sha256"]:
                    raise ValueError(f"Existing CSV differs from the pinned archive: {name}")
                print(f"Verified {name}: rows={files[name]['n_rows']}, sha256={files[name]['sha256']}", flush=True)
        provenance = {"dataset": "Improved CICIDS2017, DistriNet CNS2022 release",
            "data_kind": "official_repaired", "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive": {"name": CIC_ARCHIVE_NAME, "url": CIC_ARCHIVE_URL,
                        "sha256": CIC_ARCHIVE_SHA256, "n_bytes": CIC_ARCHIVE_BYTES, **headers},
            "hash_status": "pinned complete official-host download observed 2026-11-01; not a publisher-supplied checksum",
            "label_policy_url": CIC_LABEL_POLICY_URL,
            "source_citation": "Liu et al., Error Prevalence in NIDS Datasets, IEEE CNS 2022",
            "source_page": "https://intrusion-detection.distrinet-research.be/CNS2022/",
            "license_note": "Public research release; retain source citations. No new license is assigned to raw data.",
            "files": files}
        if downloaded:
            os.rename(actual_archive, archive_path)
        for name in CIC_FILES:
            if not (output / name).exists():
                os.rename(stage / name, output / name)
        p = stage / "provenance.json"
        p.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(p, provenance_path)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = fetch_cicids(args.output)
    except Exception as exc:
        print(f"CIC download/verification failed: {exc}. No mirror was substituted.", file=sys.stderr)
        return 1
    print(f"Verified raw directory: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
