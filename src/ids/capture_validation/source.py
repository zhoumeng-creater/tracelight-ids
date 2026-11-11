"""Fetch the UQ release and bind the CSV and row-label index to their published identities."""

from http.client import IncompleteRead
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np

from ids.common import sha256_file

API = "https://api.rdm.uq.edu.au/datasets/4ac221b1-6bd6-42b1-bdf7-03f4fc7efb22/download"
ZIP_BYTES = 780113687
ZIP_SHA = "7399bdef73d29875ab388a53e2ae7e592d1952c4ef07ae6de195ba133823d97f"
CSV_MEMBER = "f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv"
CSV = {
    "sha256": "242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74",
    "bytes": 4222783755,
    "records": 20115529,
}
ASSETS = Path(__file__).resolve().parents[3] / "configs" / "capture-validation-2018"


def checked(path, size, digest):
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise ValueError(f"Size or SHA-256 mismatch: {path}")
    return path


def download(target):
    if target.is_file():
        return checked(target, ZIP_BYTES, ZIP_SHA)
    part = target.with_suffix(".zip.part")
    for attempt in range(4):
        offset = part.stat().st_size if part.is_file() else 0
        if offset >= ZIP_BYTES:
            if offset == ZIP_BYTES and sha256_file(part) == ZIP_SHA:
                part.replace(target)
                return target
            part.unlink()  # This downloader owns the incomplete file.
            offset = 0
        try:
            with urlopen(API, timeout=40) as response:
                meta = json.load(response)
            request = Request(meta["url"], headers={"Range": f"bytes={offset}-"} if offset else {})
            with urlopen(request, timeout=90) as response:
                if offset and response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise OSError("Incorrect resume range")
                    mode = "ab"
                elif response.status == 200:
                    mode = "wb"
                else:
                    raise OSError("Unexpected UQ download response")
                with part.open(mode) as out:
                    shutil.copyfileobj(response, out, 4 << 20)
            checked(part, ZIP_BYTES, ZIP_SHA).replace(target)
            return target
        except (OSError, EOFError, IncompleteRead, ValueError) as exc:
            if attempt == 3:
                raise RuntimeError(
                    "UQ archive download failed; resume this run or specify the verified source"
                ) from exc
            time.sleep(1 + attempt)
    raise RuntimeError("UQ archive unavailable")


def resolve(work, supplied):
    work.mkdir(parents=True, exist_ok=True)
    if supplied is not None and supplied.suffix.lower() != ".zip":
        return checked(supplied, CSV["bytes"], CSV["sha256"])
    csv_path = work / "NF-CICIDS2018-v3.csv"
    if csv_path.is_file():
        return checked(csv_path, CSV["bytes"], CSV["sha256"])
    archive_path = (
        checked(supplied, ZIP_BYTES, ZIP_SHA)
        if supplied is not None
        else download(work / "NF-CSE-CIC-IDS2018-v3-official.zip")
    )
    pending = csv_path.with_suffix(".csv.part")
    with ZipFile(archive_path) as archive:
        if archive.getinfo(CSV_MEMBER).file_size != CSV["bytes"]:
            raise ValueError("Invalid official CSV member")
        with archive.open(CSV_MEMBER) as data, pending.open("wb") as out:
            shutil.copyfileobj(data, out, 4 << 20)
    checked(pending, CSV["bytes"], CSV["sha256"]).replace(csv_path)
    return csv_path


def verified_index(entry):
    path = ASSETS / entry["file"]
    checked(path, entry["bytes"], entry["sha256"])
    rows = np.load(path, allow_pickle=False)
    if (
        rows.ndim != 1
        or rows.dtype.str != "<u4"
        or len(rows) != entry["rows"]
        or not len(rows)
        or (rows[1:] <= rows[:-1]).any()
        or rows[0] < 1
        or rows[-1] > CSV["records"]
    ):
        raise ValueError("Invalid certified one-based UQ row index")
    return path


def label_index():
    spec = json.loads((ASSETS / "registration.json").read_bytes())
    strict = verified_index(spec["labels"])
    packets = verified_index(spec["packet_labels"]) if spec.get("packet_labels") else None
    return strict, packets, spec
