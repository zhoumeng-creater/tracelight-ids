"""Locate an offline IoT-23 labeled Zeek log or stream the exact official file to gzip."""

import gzip
import hashlib
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

REGISTRY = Path(__file__).resolve().parents[3] / "configs/threshold-validation-20261111/registration.json"


def verify_log(path, spec):
    digest = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        while chunk := stream.read(4 << 20):
            size += len(chunk)
            digest.update(chunk)
    if size != spec["bytes"] or digest.hexdigest() != spec["sha256"]:
        raise ValueError(f"Official uncompressed log differs from registration: {path}")
    return path


def resolve_log(source_dir, cache, spec):
    name = spec["name"] + ".conn.log.labeled.gz"
    for folder in (source_dir, cache):
        if folder is not None and (folder / name).is_file():
            return verify_log(folder / name, spec)
    if source_dir is not None:
        raise FileNotFoundError(f"Offline dataset directory lacks {name}: {source_dir}")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    temp = cache / (name + ".part")
    for attempt in range(3):
        request = Request(spec["url"], headers={"Accept-Encoding": "identity", "User-Agent": "TraceLight/1"})
        try:
            with urlopen(request, timeout=120) as response, gzip.open(temp, "wb", compresslevel=3) as out:
                if response.headers.get("Content-Encoding", "").lower() == "gzip":
                    with gzip.GzipFile(fileobj=response) as stream:
                        shutil.copyfileobj(stream, out, 1 << 20)
                else:
                    shutil.copyfileobj(response, out, 1 << 20)
            verify_log(temp, spec)
            temp.replace(path)
            return path
        except (OSError, EOFError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed for {name}; supply the offline source folder") from error
            time.sleep(2**attempt)
