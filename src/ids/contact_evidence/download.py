"""Resolve each pinned IoT-23 log from disk or the publisher, resuming verified byte ranges."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import partial
import gzip
import shutil
import time
from pathlib import Path

from http.client import HTTPException
from urllib.request import Request, urlopen

from ids.flow_history.execution import run_lock
from ids.threshold_validation.sources import verify_log


BLOCK_BYTES = 4 << 20
SOURCE_DIR = Path(__file__).resolve().parents[3] / "data/raw/iot23-contact-20261112"
PREVIOUS_DIR = "iot23-forward-20261111"


def fetch_part(folder, spec, span):
    start, end = span
    part = folder / f"{start:012d}.gz"
    expected = end - start + 1
    if part.is_file():
        try:
            with gzip.open(part, "rb") as f:
                if len(f.read(expected + 1)) == expected:
                    return part
        except (OSError, EOFError):
            pass
    headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
    expected_range = f"bytes {start}-{end}/{spec['bytes']}"
    for attempt in range(4):
        try:
            with urlopen(Request(spec["url"], headers=headers), timeout=120) as resp:
                single = start == 0 and expected == spec["bytes"]
                if resp.status == 206 and resp.headers.get("Content-Range") != expected_range:
                    raise ValueError("Official server returned the wrong byte range")
                if resp.status != 206 and not (single and resp.status == 200):
                    raise ValueError("Official server did not honor the range")
                payload = resp.read(expected + 1)
                if len(payload) != expected:
                    raise ValueError("Truncated source response")
            pending = part.with_suffix(".part")
            pending.write_bytes(gzip.compress(payload, compresslevel=3, mtime=0))
            pending.replace(part)
            return part
        except (HTTPException, OSError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Download failed for {spec['name']} bytes {start}-{end}") from error
            time.sleep(2**attempt)


def resolve_source(source_dir, spec, *, cache=None, workers=6):
    """Explicit source directory means fully offline; also search for the four R21 inputs."""
    cache = SOURCE_DIR if cache is None else Path(cache)
    name = spec["name"] + ".conn.log.labeled.gz"
    folders = [cache, cache.parent / PREVIOUS_DIR]
    if source_dir is not None:
        source_dir = Path(source_dir)
        folders = [source_dir, source_dir.parent / PREVIOUS_DIR, *folders]
    for folder in dict.fromkeys(folders):
        path = folder / name
        if path.is_file():
            try:
                return verify_log(path, spec)
            except (OSError, EOFError, ValueError):
                if source_dir is not None:
                    raise
                print(f"[source] {name} failed verification; restoring from the pinned source", flush=True)
    if source_dir is not None:
        raise FileNotFoundError(f"Missing offline source {name}, searched {folders}")
    if workers < 1:
        raise ValueError("download workers must be positive")
    cache.mkdir(parents=True, exist_ok=True)
    lock = cache / ".locks" / spec["name"]
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(run_lock(lock))
                break
            except RuntimeError as error:
                if not isinstance(error.__cause__, OSError):
                    raise
                time.sleep(2)
        dest = cache / name
        if dest.is_file():
            try:
                return verify_log(dest, spec)
            except (OSError, EOFError, ValueError):
                pass
        return _download(cache, spec, workers, dest)


def _download(cache, spec, workers, dest):
    parts = cache / ".partial" / spec["name"]
    parts.mkdir(parents=True, exist_ok=True)
    spans = [(i, min(i + BLOCK_BYTES, spec["bytes"]) - 1) for i in range(0, spec["bytes"], BLOCK_BYTES)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fragments = []
        for index, path in enumerate(pool.map(partial(fetch_part, parts, spec), spans), start=1):
            fragments.append(path)
            if index % 64 == 0 or index == len(spans):
                print(f"[source] {spec['name']} {index}/{len(spans)} parts", flush=True)
    pending = dest.with_suffix(".joining")
    try:
        with pending.open("wb") as out:
            for fragment in fragments:
                with fragment.open("rb") as f:
                    shutil.copyfileobj(f, out, length=1 << 20)
        verify_log(pending, spec)
        pending.replace(dest)
    except ValueError:
        # A valid gzip segment with an incorrect payload must not poison subsequent retries.
        for fragment in fragments:
            fragment.unlink(missing_ok=True)
        raise
    finally:
        pending.unlink(missing_ok=True)
    for fragment in fragments:
        fragment.unlink()
    parts.rmdir()
    return dest
