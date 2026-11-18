"""Fetch the four explicitly selected 2019-09-19 OpTC sources (110.3 MB).
No source discovery, model training or scoring. Run with --offline to verify cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests

BASE = Path(__file__).resolve().parent
SOURCES = {
    "ecarbro": {"name": "20190919_benign_AIA501-525_ecarbro.json.gz", "id": "19SMwYBPqzTJbBKpIMxdfm5NqoanNXRMD", "bytes": 13429353},
    "conn": {"name": "20190919_conn_17-18.log.gz", "id": "1fJvUt8nEhfROmzP3AjsofKo4GTHSvhd-", "bytes": 60278906},
    "ssl": {"name": "20190919_ssl_17-18.log.gz", "id": "1IgiMYewELJLusoU5yUR81VDL_chSYsgM", "bytes": 35942521},
    "x509": {"name": "20190919_x509_17-18.log.gz", "id": "14TY0kdYS23gM0KVFUVPw7CgnQRLUcuIK", "bytes": 681522},
}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def fetch(item, offline):
    key, spec = item
    dest = BASE / "sources" / spec["name"]
    url = f"https://drive.usercontent.google.com/download?id={spec['id']}&export=download&confirm=t"
    if not dest.exists() or dest.stat().st_size != spec["bytes"]:
        if offline:
            raise FileNotFoundError(dest)
        print(f"FETCH {key} {spec['bytes']} bytes", flush=True)
        temp = dest.with_suffix(dest.suffix + ".partial")
        total = 0
        with requests.get(url, stream=True, timeout=(30, 120), headers={"Cache-Control": "no-cache"}) as r:
            r.raise_for_status()
            with temp.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > spec["bytes"]:
                        raise ValueError(f"Unexpected source size: {key}")
                    f.write(chunk)
        if total != spec["bytes"]:
            raise ValueError(f"Incomplete source: {key}: {total}")
        temp.replace(dest)
    row = {**spec, "path": str(dest), "url": url, "view_url": f"https://drive.google.com/file/d/{spec['id']}/view", "sha256": sha256(dest)}
    print(f"READY {key} {row['sha256']}", flush=True)
    return key, row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    (BASE / "sources").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        sources = dict(pool.map(lambda item: fetch(item, args.offline), SOURCES.items()))
    prior = BASE.parents[1] / "独立来源与算法主线_20261118" / "evidence" / "optc"
    prior_receipt = json.loads((prior / "benign_file_business.json").read_text(encoding="utf-8-sig"))
    for key, row in zip(("http", "files"), prior_receipt["sources"]):
        path = prior / row["file"]
        actual = sha256(path)
        if actual != row["sha256"]:
            raise ValueError(f"Prior source changed: {path}")
        sources[key] = {**row, "path": str(path), "reused_existing": True}
    receipt = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "additional_data_bytes": sum(s["bytes"] for s in SOURCES.values()),
        "sources": sources,
        "official_directory_evidence": {
            "ecar_bro_benign": "https://drive.google.com/drive/folders/1XOGWT8R8ovw6J2DgNsui09ZwvrfOXIVa",
            "19Sep19": "https://drive.google.com/drive/folders/1ALKFkSXaWLnuC4oa51DNyOhN1Jz2XnSq",
            "AIA-501-525": "https://drive.google.com/drive/folders/1l0T20hYOwfBnUxPBBT89XPn9QrWHsVaV",
            "observed_benign_child_folders": ["17-18Sep19", "18-19Sep19", "19Sep19", "20-23Sep19"],
            "collection_semantics": "Official benign directory membership; not merely a date-before-attack inference. It does not override object-level contradictory evidence.",
            "research_paper": "https://arxiv.org/abs/2103.03080",
            "paper_section": "3.1, page 2: ecar/benign normal activity September 19-23; ecar-bro contains linking bro IDs.",
        },
        "known_unknown": "Six 2019-09-19 news.com PE downloads remain unknown; not assigned a normal label.",
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (BASE / "source_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
