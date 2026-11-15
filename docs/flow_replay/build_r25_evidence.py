"""Export verified R25 readiness results for the existing offline replay page.

Usage: python docs/flow_replay/build_r25_evidence.py ARCHIVE_ROOT
Reads returned aggregates only; does not load models, labels for fitting, or raw flows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
POLICIES = ("raw_old", "add_quality_old", "ports_local", "quality_counts_local")
GROUPS = ("hgb", "hgb_family", "mlp17", "mlp29", "mlp43")
COMPARISONS = ("raw_old", "add_quality_old", "same_policy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    opening = gzip.open if path.suffix == ".gz" else open
    with opening(path, "rt", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build(root):
    result = root / "results"
    audit = root / "复核"
    evidence = result / "evidence"
    verification = read_json(audit / "效果/效果独立复核.json")
    manifest = read_json(result / "archive_manifest.json")
    files = {}

    def verified(path, expected=None):
        actual = digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Independent audit hash differs: {path.name}")
        if path.is_relative_to(result):
            registered = manifest[path.relative_to(result).as_posix()]
            if actual != registered["sha256"] or path.stat().st_size != registered["bytes"]:
                raise ValueError(f"Returned manifest differs: {path.name}")
        files[path.relative_to(root).as_posix()] = actual
        return path

    budget_path = verified(evidence / "all-budget.csv.gz", verification["inputs"]["budget_sha256"])
    pair_path = verified(evidence / "all-paired.csv.gz", verification["inputs"]["paired_sha256"])
    readiness_path = verified(audit / "效果/42来源就绪边界.csv")
    resource_path = verified(audit / "资源/resource_audit.json")
    verified(audit / "效果/效果独立复核.json")
    sources = {row["source"]: {
        "name": row["source"], "provider": "DEDALE" if row["source"].startswith("DEDALE") else "SFC",
        "ready": row["status"], "reason": row["reason"],
        "referenceRows": int(row["prefix_rows"]) if row["prefix_rows"] else None,
        "readyAt": row["cutoff_utc"] or None, "rows": int(row["rows"]),
        "waitingRows": int(row["waiting_for_readiness_rows"]),
        "eventCounts": {}, "eventLevels": [], "budgets": {}, "pairs": {}, "unopenedPairs": {},
    } for row in read_csv(readiness_path)}
    for row in read_csv(budget_path):
        if row["policy"] not in POLICIES or row["phase"] not in {"all", "startup", "after"}:
            continue
        if row["version"] == "original" and row["phase"] != "all":
            continue
        source = sources[row["source"]]
        source["window"], source["role"] = row["window"], row["source_role"]
        normal, fp, cap = (int(row[key]) for key in ("normal", "fp", "cap"))
        if cap != normal // 100 or not 0 <= fp <= normal:
            raise ValueError("Invalid false-positive budget arithmetic")
        key = "|".join(row[k] for k in ("group", "policy", "version", "phase"))
        if key in source["budgets"]:
            raise ValueError("Duplicated source/window budget")
        source["budgets"][key] = [normal, fp, cap, int(row["unresolved_rows"]), int(row["unresolved_alerts"])]

    sums = defaultdict(lambda: [0] * 5)
    unopened = defaultdict(lambda: [0] * 5)
    for row in read_csv(pair_path):
        if row["phase"] != "all" or row["policy"] not in POLICIES or row["comparison"] not in COMPARISONS:
            continue
        key = tuple(row[k] for k in ("source", "group", "policy", "comparison", "family"))
        values = [int(row[k]) for k in ("rows", "before", "after", "gained", "lost")]
        if values[2] - values[1] != values[3] - values[4] or not 0 <= values[2] <= values[0]:
            raise ValueError("Paired detection counts do not balance")
        sums[key] = [a + b for a, b in zip(sums[key], values, strict=True)]
        if row["exposure"] in {"forward_unopened", "earlier_transfer_unopened", "forward_same_testbed"}:
            unopened[key] = [a + b for a, b in zip(unopened[key], values, strict=True)]
    for (source, group, policy, comparison, family), values in sums.items():
        key = "|".join((group, policy, comparison))
        sources[source]["pairs"].setdefault(key, {})[family] = values
    for (source, group, policy, comparison, family), values in unopened.items():
        key = "|".join((group, policy, comparison))
        sources[source]["unopenedPairs"].setdefault(key, {})[family] = values

    for source in sources.values():
        events_path = verified(evidence / "evaluation" / source["name"] / "hgb/events.csv.gz")
        events, levels = defaultdict(set), set()
        for row in read_csv(events_path):
            if row["phase"] == "all" and row["policy"] == "raw_old" and row["comparison"] == "same_policy":
                if row["event_id"]:
                    events[row["family"]].add(row["event_id"])
                    levels.add(row["evidence_level"])
        source["eventCounts"] = {family: len(ids) for family, ids in events.items()}
        source["eventLevels"] = sorted(levels)
        for group in GROUPS:
            for policy in POLICIES:
                budget = source["budgets"][f"{group}|{policy}|repaired|all"]
                for comparison in COMPARISONS:
                    families = source["pairs"][f"{group}|{policy}|{comparison}"]
                    normal = families.get("Benign", [0] * 5)
                    if normal[0] != budget[0] or normal[2] != budget[1]:
                        raise ValueError("Paired normal counts and budgets differ")
                    base_policy = policy if comparison == "same_policy" else comparison
                    base_version = "original" if comparison == "same_policy" else "repaired"
                    baseline = source["budgets"][f"{group}|{base_policy}|{base_version}|all"]
                    if normal[1] != baseline[1] or sum(v[0] for v in families.values()) != source["rows"]:
                        raise ValueError("Baseline counts or complete-source denominators differ")

    resources = read_json(resource_path)
    runtime, completion = resources["runtime"], resources["completion"]
    if completion["sources"] != len(sources) or completion["group_sources"] != len(sources) * len(GROUPS):
        raise ValueError("Incomplete source/model coverage")
    samples = runtime["parallel_resource_samples"]
    return {
        "version": 1, "release": "readiness-replay-lab-v1", "commit": resources["inputs"]["git_commit"],
        "groups": GROUPS, "policies": POLICIES, "comparisons": COMPARISONS,
        "sources": sorted(sources.values(), key=lambda s: (s["provider"] != "SFC", s["window"], s["name"])),
        "resource": {"workers": runtime["dispatch"]["workers"],
                     "meanCores": samples["tree_observed_cpu_cores"]["mean"],
                     "peakCores": samples["tree_observed_cpu_cores"]["max"],
                     "peakRssGiB": runtime["tree_rss_peak_GiB"], "seconds": runtime["invocation_seconds"],
                     "fits": completion["fits"], "forwards": completion["model_forward_calls"],
                     "sourceModels": completion["group_sources"]},
        "inputHashes": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    payload = build(args.archive_root.resolve())
    target = HERE / "r25-data.js"
    target.write_text("window.readinessEvidence = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n", encoding="utf-8")
    print(f"Exported {len(payload['sources'])} sources, {payload['resource']['sourceModels']} source/model pairs; {target.stat().st_size} bytes")
