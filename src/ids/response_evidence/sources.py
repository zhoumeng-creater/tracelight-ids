"""Locate pinned inputs in the current or previous checkout and build portable offline inputs."""

import argparse
from contextlib import ExitStack
from http.client import HTTPException
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from ids.common import sha256_file
from ids.contact_evidence import download as r22
from ids.flow_history.execution import run_lock
from ids.threshold_validation.sources import verify_log
from .sfc import project

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / "data/raw/response-evidence-20261112"
PREVIOUS = ("iot23-contact-20261112", "iot23-forward-20261111")


def filename(spec):
    return spec.get("filename", spec["name"] + ".conn.log.labeled.gz")


def candidates(cache, source_dir):
    places = [cache, *(cache.parent / folder for folder in PREVIOUS)]
    for folder in PREVIOUS:
        places.append(REPO.parent / "tracelight-contact-r22" / "data/raw" / folder)
    if source_dir is not None:
        offline = Path(source_dir)
        places = [offline, *(offline.parent / folder for folder in PREVIOUS), *places]
    return list(dict.fromkeys(places))


def archive_file(cache, spec):
    """Append confirmed byte ranges to a resumable ZIP and verify the entire publisher artifact."""
    target = cache / spec["filename"]
    pending = target.with_suffix(".part")
    if target.is_file() and target.stat().st_size == spec["bytes"] and sha256_file(target) == spec["sha256"]:
        return target
    for _ in range(2):
        start = pending.stat().st_size if pending.exists() else 0
        if start > spec["bytes"]:
            pending.unlink()
            start = 0
        with pending.open("ab") as out:
            while start < spec["bytes"]:
                end = min(start + (8 << 20), spec["bytes"]) - 1
                for attempt in range(5):
                    try:
                        request = Request(
                            spec["url"],
                            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                        )
                        with urlopen(request, timeout=120, context=r22.ssl_context()) as response:
                            if (
                                response.status != 206
                                or response.headers.get("Content-Range")
                                != f"bytes {start}-{end}/{spec['bytes']}"
                            ):
                                raise ValueError("Publisher did not serve the requested ZIP byte range")
                            raw = response.read(end - start + 2)
                            if len(raw) != end - start + 1:
                                raise ValueError("Truncated ZIP range")
                        break
                    except (OSError, ValueError, EOFError, HTTPException) as error:
                        if attempt == 4:
                            raise RuntimeError(
                                f"Could not retrieve official archive bytes {start}-{end}"
                            ) from error
                        time.sleep(2**attempt)
                out.write(raw)
                start = end + 1
        if sha256_file(pending) == spec["sha256"]:
            pending.replace(target)
            return target
        pending.unlink()
    raise ValueError("Publisher ZIP SHA-256 differs")


def resolve(source_dir, spec, reg, *, cache=None):
    cache = CACHE if cache is None else Path(cache)
    for folder in candidates(cache, source_dir):
        path = folder / filename(spec)
        if path.is_file():
            try:
                return verify_log(path, spec)
            except (ValueError, OSError, EOFError):
                if source_dir is not None:
                    raise
    if source_dir is not None:
        raise FileNotFoundError(f"Missing offline input {filename(spec)}")
    if spec.get("format") != "iot-sfc":
        return r22.resolve_source(None, spec, cache=cache)
    cache.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        while True:
            try:
                stack.enter_context(run_lock(cache / ".locks" / "IoT-SFC-v2"))
                break
            except RuntimeError as exc:
                if not isinstance(exc.__cause__, OSError):
                    raise
                time.sleep(2)
        local = cache / filename(spec)
        if local.is_file():
            try:
                return verify_log(local, spec)
            except (ValueError, OSError, EOFError):
                pass
        archive = archive_file(cache, reg["sfc_archive"])
        project(
            archive, cache, [s for s in reg["sources"] if s.get("format") == "iot-sfc"], reg["sfc_archive"]
        )
        archive.unlink()
        return local


def main(argv=None):
    from .data import read_registration

    parser = argparse.ArgumentParser(description="Fetch and verify every R23 source for offline transfer")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    reg = read_registration()
    for spec in reg["sources"]:
        if spec.get("format") == "tqh":
            from . import tqh

            tqh.copy_offline(spec, args.output)
            continue
        dest = args.output / filename(spec)
        if dest.is_file():
            verify_log(dest, spec)
        else:
            found = resolve(None, spec, reg)
            if found.resolve() != dest.resolve():
                pending = dest.with_suffix(".copying")
                shutil.copyfile(found, pending)
                verify_log(pending, spec)
                pending.replace(dest)
        print(dest, flush=True)


if __name__ == "__main__":
    main()
