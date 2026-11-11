"""Compare reproduced row ID arrays to the release and the archived evidence receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_paths import REPO, RESULTS


def verify(results):
    evidence = REPO / "docs/capture_validation_20261111"
    cfg = REPO / "configs/capture-validation-2018"
    flow = json.loads((evidence / "matched_scan_evidence.json").read_bytes())["verified_NF_row_list"]
    packet = json.loads((evidence / "packet_scan_evidence.json").read_bytes())["ids"]["nmap"]
    arrays = []
    for filename, n, sha in [
        (flow["name"], flow["rows"], flow["sha256"]),
        (packet["file"], packet["rows"], packet["sha256"]),
    ]:
        for directory in (cfg, results):
            path = directory / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Unexpected SHA-256 for {path}")
        a = np.load(results / filename, allow_pickle=False)
        if a.dtype != np.dtype("<u4") or a.shape != (n,) or a[0] < 1 or np.any(a[1:] <= a[:-1]):
            raise ValueError(f"Invalid one-based positions in {filename}")
        arrays.append(a)
        print(f"{filename}: {n:,} rows, SHA-256 {sha}")
    if len(np.intersect1d(*arrays, assume_unique=True)):
        raise ValueError("Cross-source scan row sets overlap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS, help="Folder containing both NPY row lists")
    verify(parser.parse_args().results.resolve())
