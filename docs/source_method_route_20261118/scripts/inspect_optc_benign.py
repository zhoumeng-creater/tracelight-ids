"""Inspect a specified OpTC pre-evaluation hour; do not infer benign labels."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1] / "evidence" / "optc"
SOURCES = [
    ("benign_20190919_http_17.gz", "1rFfW740CFN4EKbj_szEiPE52gYyMzgqW", 16636170),
    ("benign_20190919_files_17.gz", "110oZbclIrKkQ3OPWOUSxr7y4XYwNI-Vh", 12629303),
]


def fetch(item):
    name, fid, expected = item
    path = ROOT / name
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    if not path.exists():
        with requests.get(url, headers={"Range": f"bytes=0-{expected-1}"}, stream=True, timeout=(20, 60)) as r:
            r.raise_for_status()
            part = path.with_suffix(".part")
            count = 0
            with part.open("wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    count += len(chunk)
                    if count > expected:
                        raise ValueError(f"Unexpected size: {name}")
                    f.write(chunk)
            if count != expected:
                raise ValueError(f"Incomplete file: {name}: {count}/{expected}")
            part.replace(path)
    blob = path.read_bytes()
    if len(blob) != expected or blob[:2] != b"\x1f\x8b":
        raise ValueError(f"Not expected gzip: {name}")
    return {"file": name, "url": url, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def rows(path):
    fields = None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#fields\t"):
                fields = line.rstrip("\n").split("\t")[1:]
            elif not line.startswith("#") and line.strip():
                if fields is None:
                    raise ValueError("No Zeek field header")
                values = line.rstrip("\n").split("\t")
                if len(fields) != len(values):
                    raise ValueError("Malformed Zeek row")
                yield dict(zip(fields, values))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        provenance = list(pool.map(fetch, SOURCES))
    mime_counts = Counter()
    candidates = []
    all_files = 0
    interesting = ("pdf", "zip", "executable", "msword", "officedocument", "octet-stream", "x-dosexec", "x-msdownload", "archive")
    for row in rows(ROOT / SOURCES[1][0]):
        all_files += 1
        mime = row.get("mime_type", "-")
        mime_counts[mime] += 1
        if any(x in mime.lower() for x in interesting):
            candidates.append(row)
    fuids = {r["fuid"] for r in candidates}
    http_matches = []
    all_http = 0
    for row in rows(ROOT / SOURCES[0][0]):
        all_http += 1
        if fuids.intersection(row.get("resp_fuids", "-").split(",")):
            http_matches.append(row)
    report = {"role": "source_qualification_opened_pre_evaluation_hour_not_random_sample", "sources": provenance,
              "date": "2019-09-19", "hour_utc": "17:00:00-18:00:00", "file_rows": all_files,
              "http_rows": all_http, "mime_counts": dict(mime_counts),
              "candidate_file_rows": candidates, "http_matches": http_matches,
              "candidate_labels": [{"uid": r["uid"], "label": "unknown"} for r in http_matches],
              "benign_labels_assigned": 0,
              "label_rule": "Candidate document/archive/executable MIME only; inspect business meaning and completeness before labeling normal comparable downloads."}
    out = ROOT / "benign_file_business.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"report": str(out), "file_rows": all_files, "http_rows": all_http, "mime_counts": dict(mime_counts),
                      "candidates": len(candidates), "http_matches": len(http_matches)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
