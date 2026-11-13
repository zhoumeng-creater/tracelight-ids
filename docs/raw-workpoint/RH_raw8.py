"""Read frozen score bits, replay relevant history scores and audit raw misses against normal signatures."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import psutil

SCRIPT = Path(__file__).resolve()
# Allow both a repository docs checkout and the project archive layout.
if (SCRIPT.parents[2] / "src/ids").is_dir():
    REPO = SCRIPT.parents[2]
    default_archive = REPO.parents[1] / "实验归档/20261113_第二十三轮回包状态"
else:
    default_archive = SCRIPT.parents[2]
    REPO = default_archive.parents[1] / "代码/tracelight-ids"
REPO = Path(os.environ.get("TRACELIGHT_REPO", REPO))
ROOT = Path(os.environ.get("TRACELIGHT_R23_ARCHIVE", default_archive))
HERE = Path(os.environ.get("TRACELIGHT_AUDIT_OUTPUT", ROOT / "复核/八字段基线"))
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from ids.flow_history.trees import load_arm, score_arm  # noqa: E402
from ids.response_evidence import data  # noqa: E402
from ids.response_evidence.history import enrich_prepared  # noqa: E402
from ids.threshold_validation.sources import verify_log  # noqa: E402

PACKAGE = next(ROOT.glob("response-results-*.zip"))
INPUT = REPO / "data/raw/response-evidence-20261112"
TARGET = ("Scanning", "PureCC", "Download", "Heartbeat")
FIELDS = (
    "ip_protocol",
    "orig_port",
    "resp_port",
    "duration_seconds",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
)


def endpoints(frame):
    family = frame.family
    detail = frame.native_detail.str.lower()
    return np.select(
        [
            family.eq("Benign"),
            family.eq("Scanning"),
            family.eq("C&C"),
            detail.str.contains("filedownload", regex=False),
            detail.str.contains("heartbeat", regex=False),
            family.eq("Unresolved"),
        ],
        ["Benign", "Scanning", "PureCC", "Download", "Heartbeat", "Unresolved"],
        default="OtherAttack",
    )


def strict_cut(a, allow):
    if allow < 0:
        return None
    return float(np.sort(a)[-(allow + 1)]) if len(a) > allow else -np.inf


def read_protocol(z):
    proto = json.loads(z.read("protocol.json"))
    rules = json.loads(z.read("evidence/calibration/hgb.json"))
    c = pd.read_parquet(BytesIO(z.read("evidence/calibration/hgb.parquet")))
    neg = c.raw.to_numpy() <= rules["raw"]["threshold"]
    budget = len(c) // 100
    left = budget - int((~neg).sum())
    additions = {}
    for arm in ("contact", "quality"):
        scores = c.loc[neg, arm].to_numpy()
        t = strict_cut(scores, left)
        additions[arm] = {
            "cut": t,
            "original_cut": rules[arm]["threshold"],
            "spare_cap": left,
            "original_or_fp": int(
                (scores > rules[arm]["threshold"]).sum() + (~neg).sum()
            ),
            "residual_or_fp": int((scores > t).sum() + (~neg).sum()),
        }
    return proto, rules, additions


def audit_source(spec):
    name = spec["name"]
    role = spec["role"]
    previous = HERE / (name + ".partial.json")
    if previous.is_file():
        return json.loads(previous.read_text(encoding="utf8"))
    with ZipFile(PACKAGE) as z, TemporaryDirectory(dir=HERE, prefix=name + "-") as temp:
        work = Path(temp)
        proto, rules, new = read_protocol(z)
        p = f"evidence/blind/{name}/hgb"
        sealed = pd.read_parquet(BytesIO(z.read(p + ".parquet")))["flags"].to_numpy()
        sample = pd.read_parquet(BytesIO(z.read(p + "-sample.parquet"))).set_index(
            "row_id"
        )
        decisions = {
            arm: ((sealed >> proto["bit_order"].index(arm)) & 1).astype(bool)
            for arm in ("raw", "contact", "quality")
        }
        src = verify_log(INPUT / (name + ".conn.log.labeled.gz"), spec)
        data.prepare_log(src, work / "blind", work / "truth")
        # History is built on EVERY logged row; only candidate decisions on raw negatives need recomputation.
        enrich_prepared(work / "blind", work / "enriched", workers=2)
        models = {}
        for arm in ("contact", "quality"):
            path = work / (arm + ".joblib")
            path.write_bytes(z.read("models/frozen/hgb-" + arm + ".joblib"))
            models[arm] = load_arm(path, threads=2)
        check_truth = pd.read_parquet(work / "truth", columns=["window", "family"])
        benign = check_truth.family.eq("Benign")
        day_counts = check_truth.loc[benign, "window"].value_counts()
        bad = check_truth.loc[benign & decisions["raw"], "window"].value_counts()
        full_ok = int((benign & decisions["raw"]).sum()) <= int(benign.sum()) // 100
        eligible_dates = set(
            day_counts.index[
                (bad.reindex(day_counts.index, fill_value=0) <= day_counts // 100)
            ]
        )
        del check_truth, benign
        n = pq.read_metadata(work / "truth").num_rows
        assert n == len(sealed) == spec["rows"]
        summary = Counter()
        miss = []
        normal_parts = []
        score_parts = []
        offset = 0
        errors = {arm: 0.0 for arm in models}
        with (
            pq.ParquetFile(work / "enriched") as enriched,
            pq.ParquetFile(work / "truth") as tr,
        ):
            use = sorted(set(FIELDS) | set(proto["arms"]["quality"]) | {"row_id"})
            for fb, tb in zip(
                enriched.iter_batches(batch_size=131072, columns=use),
                tr.iter_batches(
                    batch_size=131072,
                    columns=["row_id", "window", "family", "native_detail"],
                ),
                strict=True,
            ):
                frame, truth = fb.to_pandas(), tb.to_pandas()
                ids = truth.row_id.to_numpy()
                assert np.array_equal(ids, frame.row_id) and ids[0] == offset + 1
                ix = slice(offset, offset + len(ids))
                bits = {k: v[ix] for k, v in decisions.items()}
                mean = bits["raw"]
                cat = endpoints(truth)
                take = ~mean & np.isin(cat, ("Benign", *TARGET))
                sampled = (ids - 1) % 4096 == 0
                # Both stricter C cuts exceed the legacy cuts. A sealed history non-alert cannot become a residual alert.
                need = sampled | (take & (bits["contact"] | bits["quality"]))
                rawneg = ~mean
                sc = {}
                for arm in models:
                    values = np.full(len(frame), -np.inf)
                    if need.any():
                        values[need] = score_arm(
                            models[arm], frame.loc[need], proto["arms"][arm]
                        )
                        assert np.array_equal(
                            values[need] > rules[arm]["threshold"], bits[arm][need]
                        )
                        if sampled.any():
                            err = np.max(
                                np.abs(values[sampled] - sample.loc[ids[sampled], arm])
                            )
                            errors[arm] = max(float(err), errors[arm])
                            assert err == 0
                    sc[arm] = values
                days = truth.window.to_numpy()
                for (date, k), j in (
                    pd.DataFrame({"d": days, "k": cat})
                    .groupby(["d", "k"], sort=False)
                    .indices.items()
                ):
                    flags = mean[j]
                    entry = (str(date), str(k))
                    summary[entry + ("rows",)] += len(j)
                    summary[entry + ("raw",)] += int(flags.sum())
                    for arm, s in sc.items():
                        history = bits[arm][j]
                        summary[entry + (arm + "_gained",)] += int(
                            (~flags & history).sum()
                        )
                        summary[entry + (arm + "_lost",)] += int(
                            (flags & ~history).sum()
                        )
                        if k in ("Benign", *TARGET):
                            summary[entry + (arm + "_residual_gain",)] += int(
                                (~flags & (s[j] > new[arm]["cut"])).sum()
                            )
                m = rawneg & np.isin(cat, TARGET)
                if m.any():
                    tmp = frame.loc[m, list(FIELDS)].copy()
                    tmp["utc_day"] = days[m]
                    tmp["family"] = cat[m]
                    tmp["native"] = truth.native_detail.to_numpy()[m]
                    tmp["row_id"] = ids[m]
                    for arm, s in sc.items():
                        tmp[arm + "_old"] = bits[arm][m]
                        tmp[arm + "_new"] = s[m] > new[arm]["cut"]
                    miss.append(tmp)
                normal = cat == "Benign"
                if normal.any():
                    part = frame.loc[normal, list(FIELDS)].copy()
                    if n >= 10_000_000 and name.endswith("35-1"):
                        group_path = work / f"normal-{offset:09d}.parquet"
                        part.to_parquet(group_path, index=False)
                        normal_parts.append(group_path)
                    else:
                        normal_parts.append(part)
                # Keep exact continuous scores for ceiling diagnostics later, even below the C-registered cut.
                relevant = (
                    rawneg
                    & np.isin(cat, ("Benign", *TARGET))
                    & (full_ok | np.isin(days, list(eligible_dates)))
                )
                if relevant.any():
                    rx = {"utc_day": days[relevant], "endpoint": cat[relevant]}
                    for arm, s in sc.items():
                        low = relevant & ~need
                        if low.any():
                            s[low] = score_arm(
                                models[arm], frame.loc[low], proto["arms"][arm]
                            )
                            assert np.array_equal(
                                s[low] > rules[arm]["threshold"], bits[arm][low]
                            )
                        rx[arm] = s[relevant]
                    score_parts.append(pd.DataFrame(rx))
                offset += len(frame)
        assert offset == n and len(miss) > 0
        missed = pd.concat(miss, ignore_index=True)
        all_scores = (
            pd.concat(score_parts, ignore_index=True)
            if score_parts
            else pd.DataFrame(columns=["utc_day", "endpoint", "contact", "quality"])
        )
        # Only keep known normal rows. Signature checks are within the SAME capture, independent of alert status.
        # For the largest background use bounded chunks of signatures; don't retain a set of 8M Python tuples.
        specs = {
            "all8": list(FIELDS),
            "except_source_port": [s for s in FIELDS if s != "orig_port"],
            "except_source_port_and_duration": [
                s for s in FIELDS if s not in ("orig_port", "duration_seconds")
            ],
            "proto_dport_packets": [
                "ip_protocol",
                "resp_port",
                "orig_pkts",
                "resp_pkts",
            ],
        }
        for label, cols in specs.items():
            query = pd.MultiIndex.from_frame(missed[cols])
            distinct = query.unique()
            number = np.zeros(len(distinct), np.int64)
            wanted = distinct.get_indexer(query)
            for group in normal_parts:
                if isinstance(group, Path):
                    group = pd.read_parquet(group, columns=cols)
                positions = distinct.get_indexer(pd.MultiIndex.from_frame(group[cols]))
                hits = positions[positions >= 0]
                number += np.bincount(hits, minlength=len(distinct))
            missed[label] = number[wanted]
        cohort = []
        combinations = list(dict.fromkeys((d, k) for d, k, c in summary))
        labels = sorted(set(d for d, k in combinations))
        for period in ["ALL", *labels]:
            for k in sorted(set(k for d, k in combinations)):
                fields = [
                    "rows",
                    "raw",
                    "contact_gained",
                    "contact_lost",
                    "quality_gained",
                    "quality_lost",
                ]
                if k in ("Benign", *TARGET):
                    fields += ["contact_residual_gain", "quality_residual_gain"]
                item = {
                    field: sum(
                        summary[(d, k, field)]
                        for d in labels
                        if d == period or period == "ALL"
                    )
                    for field in fields
                }
                if item["rows"] == 0:
                    continue
                cohort.append(
                    dict(capture=name, role=role, utc_day=period, endpoint=k, **item)
                )
        bounds = []
        for period in ["ALL", *labels]:
            daily = [r for r in cohort if r["utc_day"] == period]
            normal = next(r for r in daily if r["endpoint"] == "Benign")
            rem = normal["rows"] // 100 - normal["raw"]
            scores = (
                all_scores
                if period == "ALL"
                else all_scores[all_scores.utc_day == period]
            )
            for arm in models:
                pivot = strict_cut(scores.loc[scores.endpoint == "Benign", arm], rem)
                for row in daily:
                    k = row["endpoint"]
                    if k not in ("Benign", *TARGET):
                        continue
                    vals = scores.loc[scores.endpoint == k, arm]
                    bounds.append(
                        dict(
                            capture=name,
                            role=role,
                            utc_day=period,
                            endpoint=k,
                            ranking=arm,
                            normal_rows=normal["rows"],
                            raw_fp=normal["raw"],
                            spare=rem,
                            oracle_cut=None
                            if pivot is None or not np.isfinite(pivot)
                            else float(pivot),
                            feasible=pivot is not None,
                            possible_add=int((vals > pivot).sum())
                            if pivot is not None
                            else None,
                        )
                    )
        missed.insert(0, "capture", name)
        missed.insert(1, "role", role)
        result = dict(
            cohort=cohort,
            missed=missed.to_dict("records"),
            bounds=bounds,
            errors=errors,
        )
        # Commit small results on each capture so an interrupted long replay resumes at capture boundaries.
        out = HERE / (name + ".partial.json")
        out.write_text(json.dumps(result), encoding="utf8")
        return result


def main():
    with ZipFile(PACKAGE) as z:
        proto, rules, new = read_protocol(z)
    specs = sorted(
        (s for s in proto["data"]["sources"] if s["role"] in ("H", "R")),
        key=lambda s: -s["rows"],
    )
    workers = min(2, max(1, psutil.virtual_memory().available // (5 << 30)))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = []
        for spec, r in zip(specs, pool.map(audit_source, specs)):
            results.append(r)
            print(spec["name"], len(r["missed"]), flush=True)
    frames = {}
    for key in ("cohort", "missed", "bounds"):
        frames[key] = pd.DataFrame([row for r in results for row in r[key]])
        frames[key].to_csv(
            HERE / ("RH_raw8_" + key + ".csv"), index=False, encoding="utf-8-sig"
        )
    meta = {
        "package_sha256": sha256(PACKAGE.read_bytes()).hexdigest(),
        "calibration": new,
        "score_max_abs_sample_difference": [r["errors"] for r in results],
        "fields": FIELDS,
        "raw_C_rule": rules["raw"],
        "unregistered_sensitivity": "same fixed raw cut; C residual history cut uses only normal raw negatives, evaluated after outcomes were known",
        "upper_bound": "Uses true same-day/same-capture normal labels. Deterministic single strict history cut outside fixed raw alerts. Infeasible if raw already exceeds the local cap.",
        "similarity": "Exact eight fields; nested drop-source-port, then drop duration, then protocol/resp_port/two packet counts. Coarse matches do not imply impossibility.",
        "cohort": "full counts include all endpoints; residual and oracle results concern normal, scan, pure control, download, heartbeat only",
    }
    (HERE / "RH_raw8_checks.json").write_text(
        json.dumps(meta, indent=2), encoding="utf8"
    )
    for spec in specs:
        (HERE / (spec["name"] + ".partial.json")).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
