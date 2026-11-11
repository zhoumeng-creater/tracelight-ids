"""Compare native NF rows with directly observed labels from corrected CICFlowMeter rows, without training."""

import json
import hashlib
import numpy as np
import pandas as pd
from audit_paths import RESULTS as ROOT, CORRECTED as RAW, UQ_WORK as UQ


DAYS = ("2018-02-28", "2018-03-01")
NF_INPUT = (
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "FLOW_START_MILLISECONDS",
)
CIC_INPUT = ("Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "corr_start_ms")
NMAP = "Infiltration - NMAP Portscan"
ID = "row_position_zero_based"


def join_day(day, native, probes):
    native = native[native.day == day].copy()
    probes = probes[probes.day == day].copy()
    subjects = pd.concat([native, probes], ignore_index=True, sort=False).drop_duplicates(ID).copy()
    assert subjects[ID].is_unique
    subjects["nf_key_count"] = subjects.groupby(list(NF_INPUT))[ID].transform("size")
    truth = pd.read_parquet(RAW / (day + ".corrected_endpoints.parquet")).rename(
        columns={"Label": "corrected_label"}
    )
    if not truth.id.is_unique:
        raise ValueError("Author CSV identifiers repeated")
    truth = truth.rename(columns=dict(zip(CIC_INPUT, NF_INPUT)))
    truth["corrected_key_count"] = truth.groupby(list(NF_INPUT)).id.transform("size")
    j = subjects.merge(truth, on=list(NF_INPUT), how="left")
    # Keep the strict, unique match; report both rejection reasons as distinct cases.
    strict = j[j.nf_key_count.eq(1) & j.corrected_key_count.eq(1)].copy()
    strict["packets_equal"] = (strict.IN_PKTS == strict["Total Fwd Packet"]) & (
        strict.OUT_PKTS == strict["Total Bwd packets"]
    )
    strict["end_ms_equal"] = strict.FLOW_END_MILLISECONDS == strict.corr_end_ms
    strict["full_match"] = strict.packets_equal & strict.end_ms_equal
    jidx = strict.set_index(ID)
    unknown = j.loc[j.corrected_key_count.isna(), ID].unique()
    source_labels = sorted(v for v in truth.corrected_label.unique() if "nmap" in v.lower())
    if len(source_labels) != 1:
        raise ValueError(f"Unexpected NMAP labels {source_labels}")
    nmap = source_labels[0]

    def audit(group):
        ids = group[ID].unique()
        x = jidx.loc[jidx.index.intersection(ids)]
        statuses = x.groupby(
            ["Attack", "corrected_label", "end_ms_equal", "packets_equal"], dropna=False
        ).size()
        hn = x[x.corrected_label.eq(nmap) & x.full_match]
        return {
            "nf_rows": len(ids),
            "no_exact_start_and_directional_tuple": int(np.isin(ids, unknown).sum()),
            "ambiguous_key_in_either_observed_table": len(ids) - len(x) - int(np.isin(ids, unknown).sum()),
            "unique_both": len(x),
            "same_end_ms": int(x.end_ms_equal.sum()),
            "same_two_packet_counts": int(x.packets_equal.sum()),
            "same_start_and_end_ms_direction_and_packet_counts": int(x.full_match.sum()),
            "corrected_NMAP_at_unique_key": int(x.corrected_label.eq(nmap).sum()),
            "corrected_NMAP_strict_full_match": len(hn),
            "NF_vs_corrected_unique": [
                {
                    "NF": str(a),
                    "corrected": str(b),
                    "end_ms_equal": bool(c),
                    "packets_equal": bool(d),
                    "n": int(v),
                }
                for (a, b, c, d), v in statuses.items()
            ],
            "examples": [
                {
                    "NF_csv_row_zero_based": int(i),
                    "corrected_author_id": int(r.id),
                    "native": str(r.Attack),
                    "corrected": str(r.corrected_label),
                    "start_ms": int(r.FLOW_START_MILLISECONDS),
                    "corrected_start_us": int(r.corr_start_us),
                    "src": r.IPV4_SRC_ADDR,
                    "sport": int(r.L4_SRC_PORT),
                    "dst": r.IPV4_DST_ADDR,
                    "dport": int(r.L4_DST_PORT),
                    "proto": int(r.PROTOCOL),
                    "end_and_pkts_match": bool(r.full_match),
                }
                for i, r in x.head(3).iterrows()
            ],
        }

    checks = {
        "all_nf_endpoint_forward": audit(native[native.match_forward_rule]),
        "all_nf_endpoint_other": audit(native[~native.match_forward_rule]),
    }
    for label, g in native[native.match_forward_rule].groupby("Attack"):
        checks["all_forward_" + label] = audit(g)
    for label, g in probes.groupby("stratum"):
        checks["probe_" + label] = audit(g)
    # Do not rely on NF coarse labels to define subtype truth.
    ids = set(native.loc[native.match_forward_rule, ID])
    x = strict[strict[ID].isin(ids)]
    strong = x[x.full_match & x.corrected_label.eq(nmap)].copy()
    row_ids = np.sort((strong[ID].to_numpy(dtype="int64") + 1).astype("<u4"))
    assert len(np.unique(row_ids)) == len(row_ids) and len(row_ids) > 0
    np.save(RAW / (day + ".verified_nmap.npy"), row_ids, allow_pickle=False)
    mask_broad = (
        strong.PROTOCOL.eq(6)
        & strong.IN_PKTS.eq(1)
        & strong.OUT_PKTS.eq(0)
        & strong.TCP_FLAGS.eq(2)
        & strong.CLIENT_TCP_FLAGS.eq(2)
    )
    syn47 = np.load(ROOT / "cic2018_syn47_rows.npy", allow_pickle=False)
    assert syn47.dtype == np.dtype("<u4") and syn47.ndim == 1
    strong["from_47"] = np.isin(strong[ID] + 1, syn47)
    strong["single_syn"] = mask_broad
    per_origin = []
    for fam, g in strong.groupby("Attack"):
        per_origin.append(
            {
                "original_NF_attack": fam,
                "rows": len(g),
                "single_unanswered_SYN": int(g.single_syn.sum()),
                "round19_syn47": int(g.from_47.sum()),
            }
        )
    nmap_author_record = truth[truth.corrected_label.eq(nmap)]
    corrected_outside_endpoint = int(
        (
            ~(nmap_author_record["IPV4_SRC_ADDR"] == "172.31.69." + ("24" if day.endswith("28") else "13"))
        ).sum()
    )
    all_nmap = len(truth[truth.corrected_label.eq(nmap)])
    vals = {
        "day": day,
        "corrected": json.loads((RAW / (day + ".corrected_receipt.json")).read_text("utf8")),
        "nmap_exact_author_label": nmap,
        "strict_by_original_NF_class": per_origin,
        "retained_corrected_NMAP_other_orientation": corrected_outside_endpoint,
        "total_author_NMAP_rows": all_nmap,
        "NF_forward_strict_unique_NMAP": len(strong),
        "NMAP_ground_truth_coverage_by_matched_forward_NF": len(strong) / all_nmap,
        "checks": checks,
    }
    # Compact ID pairs for joined flow provenance; no feature matrices or new attack scores.
    x = strict[
        [
            ID,
            "id",
            "Attack",
            "corrected_label",
            "full_match",
            "end_ms_equal",
            "packets_equal",
            "match_forward_rule",
        ]
    ].copy()
    x.to_parquet(RAW / (day + ".nf_corrected_matches.parquet"), compression="zstd", index=False)
    print(
        day,
        len(strong),
        all_nmap,
        [
            (
                a,
                v["nf_rows"],
                v["unique_both"],
                v["corrected_NMAP_at_unique_key"],
                v["corrected_NMAP_strict_full_match"],
            )
            for a, v in checks.items()
        ],
        flush=True,
    )
    return vals


if __name__ == "__main__":
    nf = pd.read_parquet(UQ / "cic2018_native_endpoints.parquet")
    probes = pd.read_parquet(ROOT / "cic2018_nf_probe.parquet")
    results = {day: join_day(day, nf, probes) for day in DAYS}
    combined = np.sort(
        np.concatenate([np.load(RAW / (day + ".verified_nmap.npy")) for day in DAYS]).astype("<u4")
    )
    assert len(np.unique(combined)) == len(combined)
    verified_path = ROOT / "cic2018_verified_nmap_rows.npy"
    np.save(verified_path, combined, allow_pickle=False)
    report = {
        "verified_NF_row_list": {
            "name": verified_path.name,
            "rows": len(combined),
            "format": "1-D little-endian uint32, strictly increasing, one-based original UQ CSV data row number (header excluded).",
            "sha256": hashlib.sha256(verified_path.read_bytes()).hexdigest(),
        },
        "purpose": "Metadata and direct corrected-label agreement on two independent-collection dates. No models, thresholds or score selection.",
        "ground_truth_author": "https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html#infiltration",
        "ground_truth_archive": "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CSECICIDS2018_improved.zip",
        "matching": "Exact directional (source IP,port,destination IP,port,protocol) plus floor(author first UTC packet microsecond/1000)=NF first packet ms. Require uniqueness on both sides of the retained comparison, then separately check exact last ms, NF IN/OUT packet counts. For scanner endpoint flows, both input tables contain all endpoint rows. For non-endpoint UQ controls NF-side uniqueness is among the selected sample. No fuzzy timestamp alignment or assumption of equality for unmatched records.",
        "results": results,
    }
    (ROOT / "cic2018_cross_exporter_labels.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
