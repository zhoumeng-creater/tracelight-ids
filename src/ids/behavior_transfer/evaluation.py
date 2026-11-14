"""Summarize sealed decisions against independently loaded native event labels."""
from pathlib import Path

import numpy as np
import pandas as pd

from ids.common import write_json
from ids.response_evidence.experiment import ARMS

BASELINES = ("raw_old", "add_quality_old")
META = {"row_id", "flow_end_ms", "phase"}


def counts(frame, columns, baselines=BASELINES):
    out = []
    for name in columns:
        hit = frame[name].to_numpy(dtype=bool)
        for base in baselines:
            old = frame[base].to_numpy(dtype=bool)
            out.append(dict(policy=name, baseline=base, rows=len(frame),
                            before=int(old.sum()), after=int(hit.sum()),
                            gained=int((~old & hit).sum()), lost=int((old & ~hit).sum())))
    return out


def judge(predictions, truth_file, features_file, output, *, source, group, scores_file=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pred = pd.read_parquet(predictions)
    truth = pd.read_parquet(truth_file)
    feature = pd.read_parquet(features_file, columns=["row_id", "ip_protocol", "resp_port"])
    numeric = pd.read_parquet(features_file)
    sample_columns = [c for c in ARMS["quality"] if c in numeric]
    sample_values = numeric[sample_columns]
    if scores_file is not None:
        score_values = pd.read_parquet(scores_file, columns=["row_id", "score_raw", "score_contact", "score_quality"])
        if not np.array_equal(score_values.row_id, pred.row_id):
            raise ValueError("Sample scores and predictions differ in row identity")
        sample_values = pd.concat([sample_values, score_values.drop(columns="row_id")], axis=1)
    del numeric
    expected = np.arange(1, len(pred) + 1)
    for frame in (pred, truth, feature):
        if not np.array_equal(frame.row_id, expected):
            raise ValueError("Prediction, native truth or feature identity changed")
    if truth.empty:
        raise ValueError("Cannot evaluate an empty source")
    columns = [c for c in pred if c not in META]
    if any(pred[c].dtype != bool for c in columns):
        raise ValueError("Only sealed Boolean decisions may enter this evaluator")
    # Full comments remain in local truth. Group by compact native classes/steps.
    if "native_type" in truth:
        truth["native_group"] = truth.native_type
    else:
        truth["native_group"] = truth.native_detail
    truth["native_group"] = truth.native_group.fillna("").astype(str)
    for col, fallback in (("event_id", ""), ("evidence_level", "author_label"),
                          ("exposure", source["role"])):
        if col not in truth:
            truth[col] = fallback
        truth[col] = truth[col].fillna("").astype(str)
    merged = pd.concat([truth.reset_index(drop=True), pred.drop(columns="row_id"),
                        feature.drop(columns="row_id")], axis=1)
    if merged.columns.duplicated().any():
        raise ValueError("Ambiguous columns between sealed decisions and truth")
    event_starts = merged.loc[
        merged.event_id.ne("") & ~merged.family.isin(["Benign", "Unresolved"])
    ].groupby("event_id").flow_end_ms.min()
    rows, budget, services, events, examples, normal_services = [], [], [], [], [], []
    for window, daily in merged.groupby("window", sort=True):
        scopes = [("all", daily)] + [(str(phase), part) for phase, part in daily.groupby("phase")]
        for phase, part in scopes:
            common = dict(source=source["name"], source_role=source["role"],
                          group=group, window=str(window), phase=phase)
            normal = part.loc[part.family.eq("Benign")]
            cap = len(normal) // 100
            for policy in columns:
                fp = int(normal[policy].sum())
                budget.append(dict(**common, policy=policy, normal=len(normal), fp=fp, cap=cap,
                                   fpr=fp / len(normal) if len(normal) else None,
                                   within_budget=bool(fp <= cap) if len(normal) else None,
                                   unresolved_rows=int(part.family.eq("Unresolved").sum()),
                                   unresolved_alerts=int(part.loc[part.family.eq("Unresolved"), policy].sum())))
            for keys, family in part.groupby(["family", "native_group", "evidence_level", "exposure"],
                                            dropna=False, sort=True):
                extra = dict(zip(("family", "native_group", "evidence_level", "exposure"), map(str, keys)))
                rows.extend({**common, **extra, **record} for record in counts(family, columns))
            for (proto, port), service in part.groupby(["ip_protocol", "resp_port"], sort=True):
                normals = service.loc[service.family.eq("Benign")]
                for policy in columns:
                    hits = int(normals[policy].sum())
                    if hits:
                        normal_services.append(dict(**common, protocol=int(proto), port=int(port),
                                                    policy=policy, normal=len(normals), fp=hits,
                                                    fpr=hits / len(normals)))
                attack = service.loc[~service.family.isin(["Benign", "Unresolved"])]
                # Store all supported attack services, including zero-normal ones.
                if attack.empty:
                    continue
                for family_name, family in attack.groupby("family"):
                    for record in counts(family, columns):
                        normal_fp = int(normals[record["policy"]].sum())
                        services.append(dict(**common, protocol=int(proto), port=int(port),
                                             family=family_name, normal=len(normals), normal_fp=normal_fp,
                                             normal_fpr=normal_fp/len(normals) if len(normals) else None,
                                             **record))
            for (event, family_name, level), event_rows in part.loc[
                part.event_id.ne("") & ~part.family.isin(["Benign", "Unresolved"])
            ].groupby(["event_id", "family", "evidence_level"]):
                for record in counts(event_rows, columns):
                    detected = event_rows.loc[event_rows[record["policy"]], "flow_end_ms"]
                    events.append(dict(**common, event_id=event, family=family_name, evidence_level=level,
                                       first_record_ms=int(event_starts.loc[event]),
                                       first_alert_ms=int(detected.min()) if len(detected) else None,
                                       observed_delay_ms=int(detected.min()-event_starts.loc[event])
                                       if len(detected) else None, **record))
        # Small deterministic examples retain UID and scores locally through the original row_id.
        for (family, native), part in daily.groupby(["family", "native_group"]):
            for policy in columns:
                changed = part[policy] != part["add_quality_old"]
                for label, case in (("gained", part.loc[changed & part[policy]]),
                                    ("lost", part.loc[changed & ~part[policy]]),
                                    ("both_negative", part.loc[~changed & ~part[policy]]),
                                    ("both_positive", part.loc[~changed & part[policy]])):
                    for row in case.head(3).itertuples():
                        examples.append(dict(source=source["name"], group=group, window=str(window),
                                             family=family, native_group=native, policy=policy, change=label,
                                             row_id=int(row.row_id), uid=getattr(row, "uid", ""),
                                             event_id=row.event_id, flow_end_ms=int(row.flow_end_ms),
                                             **sample_values.iloc[int(row.row_id) - 1].to_dict()))
    outputs = []
    for name, records in (("paired", rows), ("budget", budget), ("services", services),
                          ("events", events), ("examples", examples),
                          ("normal_service_fp", normal_services)):
        path = output / (name + ".csv.gz")
        pd.DataFrame(records).to_csv(path, index=False, compression="gzip")
        outputs.append(path)
    report = output / "summary.json"
    write_json(report, {"source": source["name"], "role": source["role"], "group": group,
                        "rows": len(pred), "policies": columns,
                        "evaluated_phases": sorted(pred.phase.unique().tolist()),
                        "family_rows": truth.family.value_counts().to_dict(),
                        "events_with_known_attack_labels": int(truth.loc[
                            truth.event_id.ne("") & ~truth.family.isin(["Benign", "Unresolved"]),
                            "event_id"].nunique()),
                        "interpretation": "Source day-file/block cold start. Warmup uses old calibration; suffix uses frozen local cuts. "
                        "Event delay starts at first observed record, not the author's actual attack onset."})
    return outputs + [report]
