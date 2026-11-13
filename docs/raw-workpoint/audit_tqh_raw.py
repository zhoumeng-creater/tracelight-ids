"""Audit the frozen R23 eight-field HGB against original TQH/P/C observations. No fitting."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from ids.response_evidence import tqh
from ids.threshold_validation.observations import RAW_FEATURES

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data/raw/response-evidence-20261112"

PCTS = [0, 0.1, 0.5, 0.9, 1]
MASKS = {
    "all_eight": list(RAW_FEATURES),
    "without_origin_port": [k for k in RAW_FEATURES if k != "orig_port"],
    "without_origin_port_duration": [k for k in RAW_FEATURES if k not in ("orig_port", "duration_seconds")],
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pct(a):
    if not len(a):
        return []
    return [round(float(v), 6) for v in np.quantile(a, PCTS)]


def rule_masks(proto, sp, dp, duration, orig, resp):
    https = (sp == 443) | (dp == 443)
    quic = (proto == 17) & https
    long = duration >= 30
    return {
        "udp443": quic,
        "udp443_ge_10s": quic & (duration >= 10),
        "udp443_ge_30s": quic & long,
        "udp443_ge_60s": quic & (duration >= 60),
        "any_ge_30s": long,
        "udp_ge_30s": (proto == 17) & long,
        "tcp8443": (proto == 6) & ((sp == 8443) | (dp == 8443)),
        "tcp80": (proto == 6) & ((sp == 80) | (dp == 80)),
        "udp443_lt_10pkts_each": quic & (orig > 0) & (orig < 10) & (resp > 0) & (resp < 10),
    }


def key_overlap(normal, missed, cols):
    return int(pd.MultiIndex.from_frame(missed[cols]).isin(pd.MultiIndex.from_frame(normal[cols])).sum())


def collect_pc(sources, selected, source_dir):
    rows = {}
    for s in sources:
        if s["role"] not in {"P", "C"}:
            continue
        result = defaultdict(Counter)
        stream = hashlib.sha256()
        select = set(selected.loc[selected.capture.eq(s["name"]), "row_id"].astype(int))
        descriptors = sorted(rule_masks(17, 0, 443, 32, 1, 1))
        n = 0
        with gzip.open(source_dir / (s["name"] + ".conn.log.labeled.gz"), "rb") as source:
            for line in source:
                stream.update(line)
                if line.startswith(b"#") or not line.strip():
                    continue
                n += 1
                f = line.split()
                duration = -1 if f[8] == b"-" else float(f[8])
                code = {b"tcp": 6, b"udp": 17}.get(f[6], -1)
                flags = rule_masks(code, int(f[3]), int(f[5]), duration, int(f[16]), int(f[18]))
                fam = f[-1].decode() if f[-2] == b"Malicious" else "Benign"
                for tag, keep in [("all", True), ("selected", n in select)]:
                    if keep:
                        dst = result[tag + "/" + fam]
                        dst["rows"] += 1
                        for k in descriptors:
                            dst[k] += int(flags[k])
        assert stream.hexdigest() == s["sha256"] and n == s["rows"]
        rows[s["name"]] = dict(result)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-zip", required=True, type=Path)
    parser.add_argument("--source-dir", default=DATA, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    path = args.result_zip
    args.out.mkdir(parents=True, exist_ok=True)
    assert sha(path.read_bytes()) == "1fd561df5374546d2abef25ea162bfa072534d990d554bb537fd1de322dae9b3"
    details = []
    all_forms = []
    profiles = defaultdict(Counter)
    rule_counts = defaultdict(Counter)
    support_dists = []
    with threadpool_limits(limits=4), ExitStack() as stack:
        z = stack.enter_context(ZipFile(path))
        manifest = json.loads(z.read("archive_manifest.json"))

        def member(k):
            b = z.read(k)
            assert sha(b) == manifest[k]["sha256"]
            return b

        protocol = json.loads(member("protocol.json"))
        m = joblib.load(BytesIO(member("models/frozen/hgb-raw.joblib")))
        cfg = json.loads(member("evidence/calibration/hgb.json"))["raw"]
        fields = list(m["features"])
        assert fields == list(RAW_FEATURES)
        tree = m["estimator"]
        nodes = np.concatenate([t[0].nodes for t in tree._predictors])
        split_stats = {}
        for j, k in enumerate(fields):
            cut = nodes["num_threshold"][(nodes["feature_idx"] == j) & (nodes["is_leaf"] == 0)]
            split_stats[k] = {
                "nodes": len(cut),
                "unique": len(np.unique(cut)),
                "range": list(map(float, [cut.min(), cut.max()])) if len(cut) else [],
            }
        duration_ceiling = split_stats["duration_seconds"]["range"][1]
        used = {
            i for t in tree._predictors for d in t for i in d.nodes["feature_idx"][d.nodes["is_leaf"] == 0]
        }
        assert all(i < len(fields) for i in used)
        src = protocol["data"]["sources"]
        archives = {}
        for s in src:
            if s["role"] != "X":
                continue
            archive = s["archive"]
            pr = s["profile"]
            u = s["unit"]
            if archive not in archives:
                ap = args.source_dir / archive
                assert sha(ap.read_bytes()) == s["archive_sha256"]
                archives[archive] = stack.enter_context(ZipFile(ap))
            obs, truth, receipt = tqh._cell(archives[archive], pr + "/" + u)
            for k in ("sha256", "labels_sha256", "uid_order_sha256", "rows"):
                assert receipt[k] == s[k]
            f = obs.to_pandas()
            f["family"] = truth["family"].to_numpy()
            x = np.ascontiguousarray(f[fields], dtype=np.float64)
            f["score"] = tree.decision_function(x)
            f["alert"] = f.score > cfg["threshold"]
            flags = pq.read_table(BytesIO(member("evidence/blind/" + s["name"] + "/hgb.parquet")))[
                "flags"
            ].to_numpy()
            assert np.array_equal(f.alert, (flags & (1 << protocol["bit_order"].index("raw"))) != 0)
            for arm in ["contact", "prior", "quality"]:
                f[arm] = (flags & (1 << protocol["bit_order"].index(arm))) != 0
            f["profile"] = pr
            f["unit"] = u
            bins = tree._bin_mapper.transform(x)[:, sorted(used)]
            for j in range(bins.shape[1]):
                f[f"bin{j}"] = bins[:, j]
            proto, sp, dp, dur, o, r, ob, rb = (f[k] for k in fields)
            for k, mask in rule_masks(proto, sp, dp, dur, o, r).items():
                f["rule_" + k] = mask
            n = f.loc[f.family == "Benign"]
            c = f.loc[f.family == "ExternalC2"]
            lost = c.loc[~c.alert]
            rec = f.loc[f.family == "ExternalRecon"]
            if f.family.nunique() > 3:
                raise ValueError("Unexpected TQH family")
            # Which target scores are at or below even the *best* scalar score threshold at local FP<=1%?
            cap = len(n) // 100
            oracle = float(np.sort(n.score.to_numpy())[-cap - 1])
            counts = dict(
                unit=u,
                profile=pr,
                rows=len(f),
                normal=len(n),
                c2=len(c),
                recon=len(rec),
                raw_fp=int(n.alert.sum()),
                fp_cap=cap,
                within_budget=int(n.alert.sum()) <= cap,
                c2_detected=int(c.alert.sum()),
                c2_missed=len(lost),
                recon_detected=int(rec.alert.sum()),
                recon_missed=int((~rec.alert).sum()),
                oracle_c2=int((c.score > oracle).sum()),
                baseline_misses_oracle_recovered=int((lost.score > oracle).sum()),
                misses_same_bin_vector_as_normal=key_overlap(
                    n, lost, [f"bin{j}" for j in range(bins.shape[1])]
                ),
                misses_exact_score_as_normal=int(lost.score.isin(n.score).sum()),
                misses_strictly_below_max_normal=int((lost.score < n.score.max()).sum()),
            )
            for mask, cols in MASKS.items():
                counts["misses_match_normal_" + mask] = key_overlap(n, lost, cols)
            if pr == "B":
                above = f.loc[(f.family == "ExternalC2") & (f.duration_seconds > duration_ceiling)]
                alt = np.ascontiguousarray(above[fields], dtype=np.float64)
                alt[:, fields.index("duration_seconds")] = 3.0
                assert np.array_equal(tree.decision_function(alt), above.score.to_numpy())
                counts["B_c2_score_unchanged_when_all_duration_over_final_split_set_to_3s"] = len(above)
            for k in rule_masks(proto, sp, dp, dur, o, r):
                key = "rule_" + k
                for fam, sub in [("normal", n), ("c2", c), ("c2_misses", lost), ("recon", rec)]:
                    rule_counts[pr + "/" + k][fam] += int(sub[key].sum())
                fp = int((n.alert | n[key]).sum())
                pos = int((c.alert | c[key]).sum())
                b = rule_counts[pr + "/" + k]
                b["union_raw_fp"] += fp
                b["union_raw_c2"] += pos
                b["union_cells_budget_ok"] += int(fp <= cap)
                b["rule_only_cells_budget_ok"] += int(n[key].sum() <= cap)
                if k in ("udp443_ge_30s", "tcp8443", "tcp80"):
                    for label, val in [
                        ("new_c2", lost[key].sum()),
                        ("union_fp", fp),
                        ("union_ok", fp <= cap),
                    ]:
                        counts[k + "_" + label] = int(val)
            details.append(counts)
            for group, slice_ in [
                ("normal", n),
                ("c2_all", c),
                ("c2_misses", lost),
                ("c2_detected", c.loc[c.alert]),
            ]:
                ix = np.minimum(slice_.orig_port, slice_.resp_port)
                out_table = profiles[pr + "/" + group]
                out_table.update([(int(proto_n), int(port)) for proto_n, port in zip(slice_.ip_protocol, ix)])
            all_forms.append(
                f[["profile", "unit", "family", "alert", "score", "contact", "prior", "quality", *fields]]
            )
        selected = pq.read_table(BytesIO(member("evidence/train-selected.parquet"))).to_pandas()
    df = pd.DataFrame(details).sort_values(["profile", "unit"])
    df.to_csv(args.out / "tqh-raw-48-units.csv", index=False, encoding="utf-8-sig")
    full = pd.concat(all_forms, ignore_index=True)
    for pr, g in full.groupby("profile"):
        if pr not in ["B", "D"]:
            continue
        same = (g.ip_protocol == 17) & ((g.orig_port == 443) | (g.resp_port == 443))
        for fam, sub in g.loc[same].groupby("family"):
            support_dists.append(
                dict(
                    profile=pr,
                    family=fam,
                    rows=len(sub),
                    **{
                        k: pct(sub[k])
                        for k in [
                            "duration_seconds",
                            "orig_pkts",
                            "resp_pkts",
                            "orig_ip_bytes",
                            "resp_ip_bytes",
                        ]
                    },
                )
            )
    tally = {
        k: {col: int(g[col].sum()) for col in df.columns if col not in ("profile", "unit")}
        for k, g in df.groupby("profile")
    }
    out = {
        "actual_numeric_split_thresholds": split_stats,
        "package_sha256": sha(path.read_bytes()),
        "frozen_cut": cfg,
        "features": fields,
        "used_fields_in_tree_splits": [fields[i] for i in sorted(used)],
        "scope": "All physical X rows; same-cell exact source floats before HGB binning. Score ties use float64 equality. Rule checks exploratory after opening; no change to registered scores. Port group display uses lower endpoint port (other tests use explicit both endpoint membership).",
        "profile_sums": tally,
        "bounded_port_pair_proxy_top12": {
            k: [(list(t), v) for t, v in c.most_common(12)] for k, c in profiles.items()
        },
        "local_rule_checks_by_profile": rule_counts,
        "udp443_durations_sizes_quantiles_0_10_50_90_100": support_dists,
        "verified_earlier_training_calibration_support": collect_pc(src, selected, args.source_dir),
        "sealed_history_comparisons": {
            str(pr): {
                k: int(g.loc[g.family.eq("ExternalC2"), k].sum())
                for k in ["alert", "contact", "prior", "quality"]
            }
            for pr, g in full.groupby("profile")
        },
    }
    (args.out / "tqh-raw-diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        df.groupby("profile")[
            [
                "normal",
                "raw_fp",
                "c2",
                "c2_detected",
                "c2_missed",
                "misses_match_normal_all_eight",
                "misses_match_normal_without_origin_port_duration",
                "misses_exact_score_as_normal",
                "misses_same_bin_vector_as_normal",
                "udp443_ge_30s_new_c2",
            ]
        ]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
