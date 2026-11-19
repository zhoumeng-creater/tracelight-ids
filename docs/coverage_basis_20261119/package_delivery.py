"""Copy the reviewable coverage/batch artifacts, preserving exact bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


SECTIONS = ("normal_calibration", "normal_management", "history_mechanism", "attack_coverage", "unified", "batch", "batch_checks")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def include(path, root):
    rel = path.relative_to(root)
    name = rel.as_posix()
    # The lab consumes batch/, not duplicate raw extractor output or packet ledgers.
    # These remain in the original workspace and can be rebuilt from retained sources.
    local_only_files = {
        "unified/observations.jsonl.gz",
        "attack_coverage/full_capture/packet_membership.jsonl.gz",
        "attack_coverage/full_capture/packet_stats.json.gz",
        "attack_coverage/full_capture/base_connections.jsonl.gz",
        "attack_coverage/source/unraveled/flow_20210622_net1013x.csv",
    }
    local_only_prefixes = (
        "attack_coverage/full_capture/zeek/",
        "attack_coverage/full_capture/zeek_original_truncated_tail/",
        "attack_coverage/source/unraveled/repo/data/host-logs/windows/",
        "attack_coverage/source/unraveled/repo/",
        "history_mechanism/sources/",
    )
    if name in local_only_files or name.startswith(local_only_prefixes):
        return False
    if name.startswith("attack_coverage/source/") and path.suffix not in {".json", ".jsonl"}:
        return False
    if any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in rel.parts):
        return False
    if path.suffix.lower() in {".pcap", ".pcapng", ".bin", ".zip", ".whl", ".deb", ".pdf", ".pyc"}:
        return False
    if path.name.endswith((".tar.gz", ".tar.xz")):
        return False
    if rel.parts[0] == "batch_checks":
        latest = json.loads((root / "batch_checks/latest.json").read_text(encoding="utf-8"))
        if rel.name != "latest.json" and Path(latest["artifacts"]).name not in rel.parts:
            return False
    return path.stat().st_size < 95 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    dest = args.repository / "docs/coverage_basis_20261119"
    dest.mkdir(parents=True, exist_ok=True)
    prior = dest / "delivery_receipt.json"
    if prior.exists():
        for item in json.loads(prior.read_text(encoding="utf-8"))["files"]:
            source = root / item["path"]
            target = (dest / item["path"]).resolve()
            assert target.is_relative_to(dest.resolve())
            if target.is_file() and not include(source, root):
                assert sha(target) == item["sha256"], "Do not remove an edited delivery copy"
                target.unlink()
    copied, omitted = [], []
    for section in SECTIONS:
        for path in sorted((root / section).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if not include(path, root):
                if "__pycache__" not in rel.parts:
                    omitted.append({"path": str(rel), "bytes": path.stat().st_size})
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hash = sha(path)
            assert sha(target) == source_hash
            copied.append({"path": rel.as_posix(), "sha256": source_hash, "bytes": path.stat().st_size})
    for name in ("package_delivery.py", "README.md"):
        path = root / name
        if path.exists():
            shutil.copyfile(path, dest / name)
            copied.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size})
    (dest / ".gitattributes").write_text("* -text -whitespace\n", encoding="utf-8")
    result = {"files": copied, "file_count": len(copied), "total_bytes": sum(r["bytes"] for r in copied),
              "omitted_workspace_only_files": omitted,
              "raw_capture_access": "Original PCAPs remain in the workspace; exact URLs, ETags, SHA256 and download/extraction scripts are retained.",
              "artifact_bytes_preserved": True}
    (dest / "delivery_receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"files", "omitted_workspace_only_files"}}))


if __name__ == "__main__":
    main()
