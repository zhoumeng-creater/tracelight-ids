"""Recompute R26 split counts, flow observation spans and a source-support bound.

Reads archived data only; no model loading, model scoring, fitting or downloading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURES = ["ip_protocol", "orig_port", "resp_port", "duration_seconds", "orig_pkts",
            "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]
SIX = [name for name in FEATURES if name not in {"orig_port", "resp_port"}]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(4 << 20):
            value.update(block)
    return value.hexdigest()


def same_service(frame, protocol, port):
    return frame.ip_protocol.eq(protocol) & (frame.orig_port.eq(port) | frame.resp_port.eq(port))


def counts(frame):
    normal = frame.family.eq("Benign")
    return {"rows": len(frame), "normal": int(normal.sum()),
            "families": {str(k): int(v) for k, v in frame.family.value_counts().items()},
            "C2": int(frame.family.isin(["ExternalC2", "C&C"]).sum()),
            "normal_UDP443": int((normal & same_service(frame, 17, 443)).sum()),
            "normal_TCP8443": int((normal & same_service(frame, 6, 8443)).sum())}


def duration_stats(frame):
    values = frame.duration_seconds.to_numpy(dtype=float)
    return {"rows": len(frame), "min_seconds": float(values.min()),
            "median_seconds": float(np.median(values)), "max_seconds": float(values.max()),
            "over_seconds": {str(n): int((values > n).sum()) for n in (30, 60, 300, 1800, 3600)}}


def audit(r26, training_path, conflicts_path, tqh_archive=None):
    r26, training_path, conflicts_path = Path(r26), Path(training_path), Path(conflicts_path)
    manifest, protocol = read_json(r26 / "archive_manifest.json"), read_json(r26 / "protocol.json")
    provenance = []

    def record(path, registered=False):
        path = Path(path)
        sha = digest(path)
        if registered and sha != manifest[path.relative_to(r26).as_posix()]["sha256"]:
            raise ValueError(f"R26 artifact differs from its manifest: {path}")
        provenance.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha})

    record(r26 / "protocol.json", True)
    identity_path = r26 / "evidence/replay/training-identities.parquet"
    record(identity_path, True)
    record(training_path)
    record(conflicts_path)
    training = pd.read_parquet(training_path)
    identity = pd.read_parquet(identity_path)
    columns = ["source_id", "row_id", "domain", "family", "label", "weight"]
    pd.testing.assert_frame_equal(training[columns].reset_index(drop=True),
                                  identity.loc[identity.support.eq("mixed"), columns].reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    recovery = read_json(training_path.with_suffix(".json"))
    if recovery["output_sha256"] != digest(training_path):
        raise ValueError("Recovered training input differs from its recovery receipt")
    conflicts = read_json(conflicts_path)
    if not any(item["sha256"] == digest(training_path) for item in conflicts["inputs"]):
        raise ValueError("Conflict report is not bound to this recovered training frame")
    reg, specs = protocol["registration"], protocol["source_specs"]
    roles = reg["roles"]
    frames = {}
    for name in roles["forward_development"] + roles["transport_external"]:
        path = r26 / "evidence/replay" / (name + ".parquet")
        record(path, True)
        frames[name] = pd.read_parquet(path)
    calibration_path = r26 / "evidence/replay/calibration.parquet"
    record(calibration_path, True)
    calibration = pd.read_parquet(calibration_path)
    target = training.loc[training.domain.eq("target")]
    split_names = {"F_i30": [n for n in roles["target_train"] if "_i30_" in n],
                   "V_i300": [n for n in roles["target_train"] if "_i300_" in n],
                   "C_i1800": roles["normal_calibration"],
                   "E_i3600": roles["forward_development"], "D_transport": roles["transport_external"]}
    if set(split_names["F_i30"] + split_names["V_i300"]) != set(roles["target_train"]):
        raise ValueError("Proposed fit/development split does not cover original target training")
    split_data = {"F_i30": target.loc[target.source_id.isin(split_names["F_i30"])],
                  "V_i300": target.loc[target.source_id.isin(split_names["V_i300"])],
                  "C_i1800": calibration,
                  "E_i3600": pd.concat([frames[n] for n in split_names["E_i3600"]]),
                  "D_transport": pd.concat([frames[n] for n in split_names["D_transport"]])}
    splits = {}
    for role, names in split_names.items():
        per_cell = []
        for name in names:
            subset = split_data[role].loc[split_data[role].source_id.eq(name)]
            per_cell.append({"source_id": name, **counts(subset),
                             "registered_source_rows": specs[name]["rows"],
                             "first_flow_end_ms": specs[name]["first_flow_end_ms"],
                             "last_flow_end_ms": specs[name]["last_flow_end_ms"],
                             "manifest_start_utc": specs[name]["manifest_start_utc"],
                             "manifest_end_utc": specs[name]["manifest_end_utc"]})
        splits[role] = {**counts(split_data[role]), "cells": per_cell}
    splits["C_i1800"]["excluded_source_rows"] = sum(specs[n]["rows"] for n in split_names["C_i1800"]) - len(calibration)
    splits["C_i1800"]["C2_used_for_calibration"] = 0
    splits["C_i1800"]["excluded_labels_verified"] = False
    if tqh_archive:
        archive_path = Path(tqh_archive)
        expected = protocol["input_binding"]["raw_inputs"][archive_path.name]
        if digest(archive_path) != expected["sha256"]:
            raise ValueError("Optional TQH archive differs from R26 protocol")
        record(archive_path)
        with ZipFile(archive_path) as archive:
            label_counts = {}
            for name in split_names["C_i1800"]:
                spec = specs[name]
                raw = archive.read(spec["profile"] + "/" + spec["unit"] + "/labeled.jsonl")
                if hashlib.sha256(raw).hexdigest() != spec["labels_sha256"]:
                    raise ValueError(f"C labels differ from registration: {name}")
                labels = pd.Series([json.loads(line)["label"] for line in raw.splitlines()])
                label_counts[name] = {str(k): int(v) for k, v in labels.value_counts().items()}
        splits["C_i1800"].update(excluded_labels_verified=True, full_source_author_labels=label_counts,
                                 full_source_C2=sum(v.get("malicious_c2", 0) for v in label_counts.values()))
    boundaries = [(max(specs[n]["last_flow_end_ms"] for n in split_names[left]),
                   min(specs[n]["first_flow_end_ms"] for n in split_names[right]))
                  for left, right in zip(split_names, list(split_names)[1:])]
    if not all(a < b for a, b in boundaries):
        raise ValueError("Proposed role chronology overlaps")
    b = split_data["E_i3600"]
    attacks = b.loc[b.family.eq("ExternalC2")].sort_values("duration_seconds").reset_index(drop=True)
    lower, upper = attacks.iloc[:len(attacks) // 2], attacks.iloc[len(attacks) // 2:]
    normal = b.loc[b.family.eq("Benign") & same_service(b, 17, 443)]
    representative = attacks.iloc[[0, len(lower) - 1, len(lower), len(attacks) - 1]]
    b_spans = {"all_C2": duration_stats(attacks), "lower_half": duration_stats(lower),
               "upper_half": duration_stats(upper), "same_service_normal": duration_stats(normal),
               "representative_rows": representative[["source_id", "row_id", "duration_seconds", "flow_end_ms"]].to_dict("records"),
               "exact_target_training_input_matches": {
                   name: int(pd.MultiIndex.from_frame(attacks[cols]).isin(pd.MultiIndex.from_frame(target[cols])).sum())
                   for name, cols in (("raw8", FEATURES), ("ports_zero6", SIX))},
               "meaning": "Published full-flow observation span, not measured log emission, ingestion or alert delay. "
                          "The large median lies between two separated clusters. flow_end_ms is the replay's logical clock."}
    source = training.loc[training.domain.eq("source")]
    group = max(conflicts["mixed_ports_zero8_groups_sorted_by_floor_contribution"], key=lambda item: item["rows"])
    matching = source.loc[(source[SIX] == pd.Series(group["input"])[SIX]).all(axis=1)]
    source_normal = int(source.label.eq(0).sum())
    budget = source_normal // 100
    group_normal = int(matching.label.eq(0).sum())
    group_c2 = int(matching.family.eq("C&C").sum())
    source_c2 = int(source.family.eq("C&C").sum())
    if len(matching) != group["rows"] or group_normal != group["negative_rows"] or group_normal <= budget:
        raise ValueError("Selected exact group does not establish the claimed FP-budget obstruction")
    bound = {"group_id": group["group_id"], "input": group["input"], "source_normal": source_normal,
             "pooled_source_fp_budget_at_1percent": budget, "group_normal": group_normal,
             "group_C2": group_c2, "source_C2": source_c2,
             "C2_detected_at_most": source_c2 - group_c2,
             "C2_recall_at_most": (source_c2 - group_c2) / source_c2,
             "FPR_if_this_group_alerts": group_normal / source_normal,
             "group_composition": matching.groupby(["source_id", "family"]).size().reset_index(name="rows").to_dict("records"),
             "conditions": "Deterministic per-flow scoring from the same six non-port inputs, one common threshold, "
                           "and a 1% budget on the pooled 4096 source-normal training rows. "
                           "Does not cover capture-specific thresholds, identity routing, stochastic decisions or additional observations.",
             "interpretation": "Upper bound only; attainability for the remaining controls is not established. "
                               "This is not a D-transfer bound and does not assert the old model detected these controls."}
    return {"status": "recomputed", "model_loads": 0, "model_forward_calls": 0, "new_fits": 0,
            "downloads": 0, "provenance": provenance, "splits": splits,
            "target_protocol_by_label": target.groupby(["label", "ip_protocol"]).size().reset_index(name="rows").to_dict("records"),
            "strict_role_chronology": True,
            "split_interpretation": "Already-opened development roles; time and beacon-interval changes are coupled. "
                                    "The split does not create a new blind or independent-implementation validation.",
            "B_flow_observation_spans": b_spans, "D_C2_duration": duration_stats(split_data["D_transport"].loc[
                split_data["D_transport"].family.eq("ExternalC2")]), "source_control_bound": bound}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r26-run", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--conflicts", required=True, type=Path)
    parser.add_argument("--tqh-archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evidence.json"))
    args = parser.parse_args()
    result = audit(args.r26_run, args.training, args.conflicts, args.tqh_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "C2_bound": result["source_control_bound"]["C2_detected_at_most"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
