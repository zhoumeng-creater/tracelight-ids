"""Open truth after predictions are sealed; reduce exact rowwise paired decisions by family."""

from collections import defaultdict
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from . import data
from .experiment import GROUPS, ROWS
from .policy import COMPARISONS, POLICIES

BINS = np.arange(256, dtype=np.uint8)


def unpack(flags, policy):
    return ((flags >> POLICIES.index(policy)) & 1).astype(bool)


def counts(hist, baseline, candidate):
    a, b = unpack(BINS, baseline), unpack(BINS, candidate)
    return {
        "rows": int(hist.sum()),
        "before": int(hist[a].sum()),
        "after": int(hist[b].sum()),
        "gained": int(hist[~a & b].sum()),
        "lost": int(hist[a & ~b].sum()),
    }


def gate_breakdown(hist, cut_active):
    """Partition the bits sealed at scoring time; include inside-anchor gate hits in total coverage."""
    gate = (BINS & 128) != 0
    anchor = unpack(BINS, "anchor")
    final = unpack(BINS, "contact_add")
    outside = ~anchor
    eligible = outside & gate
    if hist[(anchor & ~final) | (outside & final & ~gate)].any() or (
        not cut_active and hist[eligible & final].any()
    ):
        raise ValueError("Sealed decisions contradict the registered contact addition")
    no_add = int(hist[eligible & ~final].sum())
    return {
        "rows": int(hist.sum()),
        "gate_covered": int(hist[gate].sum()),
        "anchor_alerts": int(hist[anchor].sum()),
        "outside_anchor": int(hist[outside].sum()),
        "outside_anchor_rejected_by_gate": int(hist[outside & ~gate].sum()),
        "outside_anchor_eligible": int(hist[eligible].sum()),
        "added_by_contact_rule": int(hist[eligible & final].sum()),
        "eligible_not_passing_score_rule": no_add if cut_active else 0,
        "eligible_no_registered_cut": 0 if cut_active else no_add,
        "final_alerts": int(hist[final].sum()),
    }


def criteria(groups):
    normal = groups.get("Benign")
    scan = groups.get("Scanning")
    cc = groups.get("C&C")
    ok_budget = bool(normal and normal["rows"] and normal["after"] <= normal["rows"] // 100)
    both_budget = bool(ok_budget and normal["before"] <= normal["rows"] // 100)
    gain = bool(scan and scan["rows"] and scan["after"] > scan["before"])
    retained = bool(cc and cc["before"] > 0 and cc["lost"] == 0)
    other = [v for k, v in groups.items() if k not in {"Scanning", "C&C", "Benign", "Unresolved"}]
    joint_evaluable = bool(normal and normal["rows"] and scan and scan["rows"] and cc and cc["before"])
    passed = both_budget and gain and retained and not any(v["lost"] for v in other)
    return {
        "normal_budget": normal["rows"] // 100 if normal else None,
        "candidate_within_budget": ok_budget,
        "both_within_budget": both_budget,
        "scan_gain": gain,
        "cc_support": cc["rows"] if cc else 0,
        "cc_retained_nonvacuous": retained,
        "other_attack_gained": sum(v["gained"] for v in other),
        "other_attack_lost": sum(v["lost"] for v in other),
        "joint_pass": passed if joint_evaluable else None,
        "joint_status": ("pass" if passed else "fail") if joint_evaluable else "not_evaluable",
    }


class _FlagStream:
    """Match sealed byte flags to truth rows even when Parquet row groups are cut differently."""

    def __init__(self, parquet, expected, column="flags"):
        if parquet.metadata.num_rows != expected:
            raise ValueError("Incomplete capture: sealed flag count differs from truth")
        self.batches = parquet.iter_batches(batch_size=ROWS, columns=[column], use_threads=False)
        self.pending = np.empty(0, dtype=np.uint8)

    def take(self, size):
        parts = []
        while size:
            if not len(self.pending):
                batch = next(self.batches, None)
                if batch is None:
                    raise ValueError("Short blind stream")
                column = batch.column(0)
                if column.null_count:
                    raise ValueError("Missing sealed flags")
                self.pending = column.to_numpy()
            count = min(len(self.pending), size)
            parts.append(self.pending[:count])
            self.pending = self.pending[count:]
            size -= count
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


def _add_allocation_counts(hist, before, after, families):
    for policy in hist:
        bit = POLICIES.index(policy)
        cells = ((before >> bit) & 1) * 2 + ((after >> bit) & 1)
        for prefix, labels, codes in families:
            bins = np.bincount(codes * 4 + cells, minlength=len(labels) * 4).reshape(-1, 4)
            for label, counts4 in zip(labels, bins):
                hist[policy][prefix + str(label)] += counts4


def _allocation_report(hist):
    policies = {}
    for policy, tables in hist.items():
        counts_by_label = {}
        for label, cells in tables.items():
            neither, gained, lost, both = map(int, cells)
            counts_by_label[label] = {
                "rows": neither + gained + lost + both,
                "before": lost + both,
                "after": gained + both,
                "gained": gained,
                "lost": lost,
            }
        policies[policy] = {"counts": counts_by_label}
    return {"baseline": "hgb", "candidate": "hgb_family", "policies": policies}


def judge(root, name, groups=GROUPS):
    directory = root / "evidence/blind" / name
    hist = {g: defaultdict(lambda: np.zeros(256, dtype=np.int64)) for g in groups}
    allocation = {}
    if {"hgb", "hgb_family"}.issubset(groups):
        allocation = {
            p: defaultdict(lambda: np.zeros(4, dtype=np.int64))
            for p in ("raw", "history", "service", "contact", "contact_add")
        }
    examples = defaultdict(list)
    rare_cc = []
    offset = 0
    with ExitStack() as opened:
        truth = opened.enter_context(pq.ParquetFile(data.truth(root, name)))
        streams = {
            group: _FlagStream(
                opened.enter_context(pq.ParquetFile(directory / (group + ".parquet"))),
                truth.metadata.num_rows,
            )
            for group in groups
        }
        for batch in truth.iter_batches(batch_size=ROWS, columns=["row_id", "family", "native_detail"]):
            df = batch.to_pandas()
            if not np.array_equal(df.row_id, np.arange(offset + 1, offset + len(df) + 1)):
                raise ValueError("Truth stream does not match registered blind row identities")
            native_codes, native = pd.factorize(df.native_detail, sort=False)
            family_codes, families = pd.factorize(df.family, sort=False)
            if (family_codes < 0).any() or (native_codes < 0).any():
                raise ValueError("Missing truth labels")
            labels_by_family = (("family/", families, family_codes), ("native/", native, native_codes))
            block_flags = {group: stream.take(len(df)) for group, stream in streams.items()}
            for i in np.flatnonzero(df.family.eq("C&C").to_numpy()):
                rare_cc.append(
                    {
                        "row_id": int(df.row_id.iloc[i]),
                        "alerts": {
                            group: [p for bit, p in enumerate(POLICIES) if int(flags[i]) & (1 << bit)]
                            for group, flags in block_flags.items()
                        },
                    }
                )
            if allocation:
                _add_allocation_counts(
                    allocation, block_flags["hgb"], block_flags["hgb_family"], labels_by_family
                )
            for group, flags in block_flags.items():
                for prefix, labels, codes in labels_by_family:
                    block = np.bincount(codes * 256 + flags, minlength=len(labels) * 256).reshape(-1, 256)
                    for key, vector in zip(labels, block):
                        hist[group][prefix + str(key)] += vector
                if group == "hgb":
                    raw, candidate = unpack(flags, "raw"), unpack(flags, "contact_add")
                    for code, family in enumerate(families):
                        for event, mask in (("gained", candidate & ~raw), ("lost", ~candidate & raw)):
                            bucket = examples[str(family) + "/" + event]
                            if len(bucket) < 16:
                                take = np.flatnonzero((family_codes == code) & mask)[: 16 - len(bucket)]
                                bucket.extend(df.row_id.to_numpy()[take].tolist())
            offset += len(df)
        if not offset or offset != truth.metadata.num_rows:
            raise ValueError("Incomplete capture")
    report = {
        "capture": name,
        "rows": offset,
        "groups": {},
        "hgb_raw_to_contact_add_example_row_ids": dict(examples),
        "pure_cc_decisions": rare_cc,
    }
    for group, tables in hist.items():
        comparisons = {}
        for a, b in COMPARISONS:
            summary = {name: counts(h, a, b) for name, h in tables.items()}
            families = {k[7:]: v for k, v in summary.items() if k.startswith("family/")}
            comparisons[a + "_vs_" + b] = {"counts": summary, **criteria(families)}
        registered = read_json(root / "evidence/calibration" / (group + ".json"))
        cut_active = registered["contact_add"]["threshold"] is not None
        report["groups"][group] = {
            "pairs": comparisons,
            "gate_breakdown": {key: gate_breakdown(h, cut_active) for key, h in tables.items()},
            "flag_histograms": {k: v.tolist() for k, v in tables.items()},
        }
    if allocation:
        report["training_allocation"] = _allocation_report(allocation)
    output = root / "evidence/outcomes" / (name + ".json")
    write_json(output, report)
    return [output]


def overview(root, names):
    out = {"primary_group": "hgb", "captures": {}}
    for name in names:
        report = read_json(root / "evidence/outcomes" / (name + ".json"))
        pairs = report["groups"]["hgb"]["pairs"]
        out["captures"][name] = {
            k: {c: v for c, v in record.items() if c != "counts"}
            | {"families": {c[7:]: v for c, v in record["counts"].items() if c.startswith("family/")}}
            for k, record in pairs.items()
        }
        if name.endswith("-52-1"):
            required = ["unrestricted_add_vs_contact_add", "raw_vs_contact_add"]
            statuses = [pairs[k]["joint_status"] for k in required]
            out["primary_scan_and_cc"] = {
                "capture": name,
                "comparisons": required,
                "status": "fail"
                if "fail" in statuses
                else ("pass" if all(s == "pass" for s in statuses) else "not_evaluable"),
                "individual_status": dict(zip(required, statuses)),
            }
        if name.endswith("-60-1"):
            record = pairs["raw_vs_contact_add"]
            native = {k[7:]: v for k, v in record["counts"].items() if k.startswith("native/")}
            details = {k: v for k, v in native.items() if k.casefold() in {"c&c-heartbeat", "ddos"}}
            identifiable = len(details) == 2 and all(v["before"] > 0 for v in details.values())
            lost = any(v["lost"] > 0 for v in details.values())
            status = (
                "fail"
                if lost or not record["candidate_within_budget"]
                else ("pass" if identifiable else "not_evaluable")
            )
            out["other_traffic_stress"] = {
                "capture": name,
                "comparison": "raw_vs_contact_add",
                "status": status,
                "native_endpoints": details,
                "candidate_within_budget": record["candidate_within_budget"],
            }
    target = root / "evidence/outcome.json"
    write_json(target, out)
    return [target]
