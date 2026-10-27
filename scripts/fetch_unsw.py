#!/usr/bin/env python3
"""Download only the two small UNSW split CSVs, with explicit provenance.

Standard library only. The optional mirror is pinned to a Git commit and is
never silently substituted for the official source.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen

OFFICIAL_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
MIRROR_REVISION = "44bbd02b9e70f74dd915218d5eb702f125341ffc"
MIRROR_BASE = f"https://raw.githubusercontent.com/Nir-J/ML-Projects/{MIRROR_REVISION}/UNSW-Network_Packet_Classification"
FILES = {"UNSW_NB15_training-set.csv": 175341, "UNSW_NB15_testing-set.csv": 82332}
HEADER = (
    "id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,"
    "sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,"
    "smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,"
    "ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,"
    "ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,attack_cat,label"
).split(",")
MAX_FILE_BYTES = 64 * 1024 * 1024


def download(url: str, destination: Path) -> dict:
    if urlparse(url).scheme != "https":
        raise ValueError("Download URLs must use HTTPS")
    digest = hashlib.sha256()
    total = 0
    request = Request(url, headers={"User-Agent": "TraceLight-IDS-dataset-preparation/1.0"})
    with urlopen(request, timeout=45) as response, destination.open("wb") as stream:
        final_host = (urlparse(response.geturl()).hostname or "").lower()
        if "login" in final_host or "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Source returned a login/HTML page, not a CSV; download manually from the official page")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_FILE_BYTES:
            raise ValueError("Refusing a large archive; only the two split CSV files are supported")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise ValueError("CSV exceeds 64 MiB; refusing a raw capture/archive download")
            digest.update(chunk)
            stream.write(chunk)
    return {"url": url, "sha256": digest.hexdigest(), "n_bytes": total}


def validate_csv(path: Path, expected_rows: int) -> dict:
    labels = {"0": 0, "1": 0}
    categories: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if header != HEADER:
            raise ValueError(f"{path.name}: expected the original 45-column split schema")
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(HEADER):
                raise ValueError(f"{path.name}: malformed CSV row {row_count}")
            category, label = row[-2].strip(), row[-1].strip()
            if label not in labels or ((label == "0") != (category == "Normal")):
                raise ValueError(f"{path.name}: inconsistent labels at row {row_count}")
            labels[label] += 1
            categories[category] = categories.get(category, 0) + 1
    if row_count != expected_rows:
        raise ValueError(f"{path.name}: expected {expected_rows} rows, received {row_count}; split may be swapped or altered")
    return {"n_rows": row_count, "n_columns": len(HEADER), "label_counts": labels, "attack_cat_counts": categories}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination for raw CSVs and provenance.json")
    parser.add_argument("--source", choices=("official", "mirror"), default="official")
    parser.add_argument("--accept-unverified-mirror", action="store_true",
                        help="Explicitly accept a pinned third-party copy without official checksum verification")
    parser.add_argument("--train-url", help="Direct official HTTPS training CSV URL, if available")
    parser.add_argument("--test-url", help="Direct official HTTPS testing CSV URL, if available")
    args = parser.parse_args(argv)
    if args.source == "mirror":
        if not args.accept_unverified_mirror:
            parser.error("Mirror downloads require --accept-unverified-mirror; no official hash comparison is available")
        if args.train_url or args.test_url:
            parser.error("Custom URLs cannot be combined with the pinned mirror")
        urls = {name: f"{MIRROR_BASE}/{name}" for name in FILES}
        print("Using explicitly selected third-party mirror. It is NOT verified against official source hashes.")
    else:
        if args.accept_unverified_mirror:
            parser.error("--accept-unverified-mirror requires --source mirror")
        if not args.train_url or not args.test_url:
            parser.error(f"Obtain the two split CSVs from {OFFICIAL_PAGE} and import local paths, or supply --train-url and --test-url. The official SharePoint folder required login during the 2026-10-27 check. No mirror was substituted.")
        urls = dict(zip(FILES, (args.train_url, args.test_url)))
        for url in urls.values():
            host = (urlparse(url).hostname or "").lower()
            if not (host == "unsw.edu.au" or host.endswith(".unsw.edu.au") or host == "unsw-my.sharepoint.com"):
                parser.error("Official URLs must be hosted by UNSW or its SharePoint domain")
    output = args.output
    if any((output / name).exists() for name in (*FILES, "provenance.json")):
        parser.error("Destination already contains source artifacts; use a new directory to avoid overwriting evidence")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": "UNSW-NB15 official split format", "data_kind": args.source,
        "official_reference": OFFICIAL_PAGE, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_hash_verified": False,
        "verification_scope": "Downloaded bytes hashed locally; schema and published row counts checked. No official reference checksum was available.",
        "mirror_revision": MIRROR_REVISION if args.source == "mirror" else None,
        "files": {},
    }
    try:
        with tempfile.TemporaryDirectory(prefix=".unsw-download-", dir=output) as staging:
            for name, expected_rows in FILES.items():
                path = Path(staging) / name
                record = download(urls[name], path)
                record.update(validate_csv(path, expected_rows))
                provenance["files"][name] = record
                print(f"Verified structure: {name}, rows={record['n_rows']}, bytes={record['n_bytes']}, sha256={record['sha256']}")
            for name in FILES:
                (Path(staging) / name).replace(output / name)
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"Download failed: {exc}. No alternate source was selected.")
        return 1
    print(f"Download complete. Use data_kind='{args.source}' when preparing these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
