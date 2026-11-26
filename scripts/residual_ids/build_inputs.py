"""Build the portable, provenance-bound inputs for the retrospective rescue experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from scripts.ait_forward import baseline, comparators
from scripts.comprehensive_batch.matrix import PRIMARY_STEP, SEEDS, job_id, snapshot_id

SCHEMA = "tracelight-residual-rescue-input-v1"
EVALUATION_ROLES = {"fox": "V", "optc": "external_development"}
FIT_SOURCE = "unraveled:apt_2021_single_campaign"
EVENT_TARGETS = {
    "webshell_command": "control",
    "reverse_shell_channel": "control",
    "known_control_transport": "control",
    "attack_download": "download",
    "malicious_tool_download": "download",
}
EVENT_MEMBER = "evaluation/event_definitions.jsonl.gz"
UNIT_MEMBER = "evaluation/normal_unit_definitions.jsonl.gz"
ROW_FIELDS = ("flow_id", "stable_flow_id", "allocation_group", "source_group",
              "source_run", "dataset_role", "label", "target_kind", "raw8", "features8")
EVENT_FIELDS = ("event_id", "kind", "label", "chain_id", "candidate_flow_ids",
                "confirmed_flow_ids", "association_mode", "candidate_set_complete")
IDENTITY_FIELDS = ("stable_flow_id", "allocation_group", "source_group", "dataset_role",
                   "label", "target_kind", "replay_export_time", "clock_basis")


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_jsonl(compressed: bytes) -> list[dict]:
    return [json.loads(line) for line in gzip.decompress(compressed).splitlines()]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as target, gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                          + "\n").encode("utf-8"))


def archive_inputs(archive: Path) -> tuple[dict[str, bytes], bytes]:
    if sha_file(archive) != baseline.ARCHIVE_SHA256:
        raise ValueError("The comprehensive source ZIP does not match its frozen SHA-256.")
    receipt_name, receipt_pin = "output_receipt.json", comparators._RECEIPT_PIN
    receipt_bytes = baseline._verified_members(archive, {receipt_name: receipt_pin})[receipt_name]
    receipt = json.loads(receipt_bytes)
    model_ids = [snapshot_id(job_id("A_BCE", seed), PRIMARY_STEP) for seed in SEEDS]
    model_paths = comparators._fixed_model_paths()
    names = {name for model_id in model_ids
             for name in comparators._selected_members(model_id, model_paths[model_id])}
    pins = {name: (receipt[name]["bytes"], receipt[name]["sha256"])
            for name in names | {EVENT_MEMBER}}
    pins.update(baseline._MEMBERS)
    members = baseline._verified_members(archive, pins)
    events = members.pop(EVENT_MEMBER)
    return members | {receipt_name: receipt_bytes}, events


def audit_gate(rows, original_scores, thresholds, events):
    sections = {}
    for name, selected in rows.items():
        sections[name] = {}
        for rule, threshold in thresholds.items():
            counts = Counter((row["target_kind"], original_scores[row["flow_id"]] > threshold)
                             for row in selected)
            sections[name][rule] = {
                kind: {"total": counts[kind, False] + counts[kind, True],
                       "hgb_nonalert": counts[kind, False], "hgb_alert": counts[kind, True]}
                for kind in ("normal", "control", "download", "unknown", "other_attack")
            }
    fit = sections["fit_unraveled"]["business_max"]
    fit_ids = {row["flow_id"] for row in rows["fit_unraveled"] if row["target_kind"] == "control"}
    events_touching_fit = [event for event in events
                           if event["role"] == "F_expansion"
                           and fit_ids.intersection(event.get("confirmed_flow_ids", []))]
    if len(events_touching_fit) != 1 or set(events_touching_fit[0]["confirmed_flow_ids"]) != fit_ids:
        raise ValueError("The archived control-event support differs from the registered single observation set.")
    if (fit["normal"]["total"], fit["control"]["total"], fit["normal"]["hgb_nonalert"],
        fit["control"]["hgb_nonalert"], fit["download"]["total"]) != (12308, 473, 12249, 473, 0):
        raise ValueError("The registered residual-training counts differ from the frozen evidence.")
    missing = sum(any(value is None for value in row["raw8"].values()) for row in rows["fit_unraveled"])
    return {
        "source_archive_sha256": baseline.ARCHIVE_SHA256,
        "feature_rows_sha256": baseline.ROWS_SHA256,
        "hgb_model_sha256": baseline.MODEL_SHA256,
        "role_gate_counts": sections,
        "hgb_frozen_thresholds": thresholds,
        "fit_control_complete_observation_sets": len(events_touching_fit),
        "fit_control_campaigns": 1,
        "fit_control_independent_commands_established": 0,
        "fit_rows_with_missing_raw8_fields": missing,
        "positive_fit_membership_differs_between_mlp_arms": False,
        "removed_normal_rows_specialist_business_max": fit["normal"]["hgb_alert"],
        "admission": "retrospective_development_fit_only",
        "independent_confirmation_admitted": False,
    }


def build(archive: Path, input_rows: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if sha_file(input_rows) != baseline.ROWS_SHA256:
        raise ValueError("The feature input does not match the frozen canonical rows.")
    members, event_bytes = archive_inputs(archive)
    events = read_jsonl(event_bytes)
    identities = read_jsonl(members["evaluation/common_identities.jsonl.gz"])
    order = json.loads(members["score_identities.json"])
    if [row["flow_id"] for row in identities] != order or len(set(order)) != len(order):
        raise ValueError("Archived score order and identities do not agree.")
    with np.load(io.BytesIO(members[f"scores/{baseline.MODEL_ID}.npz"]), allow_pickle=False) as saved:
        scores = saved["scores"]
    if scores.shape != (len(order),) or not np.isfinite(scores).all():
        raise ValueError("Invalid frozen HGB score vector.")
    original_scores = dict(zip(order, scores.tolist(), strict=True))

    rows_by_id = {}
    for row in read_jsonl(input_rows.read_bytes()):
        fid = row["flow_id"]
        if fid in rows_by_id:
            raise ValueError("Duplicate feature row.")
        rows_by_id[fid] = row
    if rows_by_id.keys() != original_scores.keys():
        raise ValueError("Feature and scored populations differ.")
    for identity in identities:
        row = rows_by_id[identity["flow_id"]]
        if any(row[key] != identity[key] for key in IDENTITY_FIELDS):
            raise ValueError("Feature labels, identity, role or observation clock differ from the archive.")

    by_role = {}
    for fid in order:
        row = rows_by_id[fid]
        by_role.setdefault(row["dataset_role"], []).append(row)
    fit = [row for row in by_role["F_expansion"] if row["target_kind"] in {"normal", "control"}]
    if ({row["allocation_group"] for row in fit} != {FIT_SOURCE}
            or any(row["fit_base_member"] or not row["fit_expanded_member"] for row in fit)):
        raise ValueError("Rescue fit must use only independently held-out HGB development supervision.")
    calibration = by_role["C"]
    if len(calibration) != 3475 or any(row["label"] != 0 for row in calibration):
        raise ValueError("Expected precisely the normal-only Russell calibration rows.")
    evaluations = {name: by_role[role] for name, role in EVALUATION_ROLES.items()}

    units = read_jsonl(members[UNIT_MEMBER])
    calibration_units = {unit["unit_id"]: unit["flow_ids"] for unit in units
                         if unit["role"] == "C" and unit.get("view") == "business"}
    if sorted(map(len, calibration_units.values())) != [188, 910, 2377]:
        raise ValueError("The three registered calibration normal groups differ.")
    calibration_ids = {row["flow_id"] for row in calibration}
    if sum(map(len, calibration_units.values())) != len(calibration_ids) or (
            set().union(*map(set, calibration_units.values())) != calibration_ids):
        raise ValueError("Calibration groups must partition all registered C rows.")
    thresholds = {
        "pooled": sorted((original_scores[fid] for fid in calibration_ids), reverse=True)[len(calibration_ids) // 100],
        "business_max": max(sorted((original_scores[fid] for fid in ids), reverse=True)[len(ids) // 100]
                            for ids in calibration_units.values()),
    }
    memberships = {"calibration_units": calibration_units, "evaluations": {}}
    for name, role in EVALUATION_ROLES.items():
        current = evaluations[name]
        current_ids = {row["flow_id"] for row in current}
        positive_events = [{key: event[key] for key in EVENT_FIELDS if key in event}
                           | {"kind": EVENT_TARGETS[event["kind"]], "source_kind": event["kind"]}
                           for event in events if event.get("role") == role and event.get("label") == 1]
        if any(not set(event.get("candidate_flow_ids", [])) <= current_ids for event in positive_events):
            raise ValueError("An archived positive event references rows outside its run.")
        memberships["evaluations"][name] = {
            "flow_ids": [row["flow_id"] for row in current],
            "normal_units": {unit["unit_id"]: unit["flow_ids"] for unit in units if unit["role"] == role},
            "attack_ids": {kind: [row["flow_id"] for row in current if row["label"] == 1
                                  and row["target_kind"] == kind] for kind in ("control", "download")},
            "events": positive_events,
        }

    audit = audit_gate({"fit_unraveled": fit, "excluded_hgb_training_santos": by_role["F_base"],
                        "calibration_russell": calibration, **evaluations},
                       original_scores, thresholds, events)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".rescue-build-", dir=output.parent) as root:
        stage = Path(root) / "data"
        stage.mkdir()
        inputs = {"fit_rows.jsonl.gz": fit, "calibration_rows.jsonl.gz": calibration,
                  "evaluation_rows.jsonl.gz": [row for group in evaluations.values() for row in group]}
        for name, original in inputs.items():
            exported = [{field: row[field] for field in ROW_FIELDS if field in row} for row in original]
            if name == "fit_rows.jsonl.gz":
                for row in exported:
                    row["dataset_role"] = "fit"
            write_jsonl(stage / name, exported)
        write_json(stage / "memberships.json", memberships)
        write_json(stage / "data_audit.json", audit)
        with ZipFile(stage / "frozen-models.zip", "w") as out:
            for name in sorted(members):
                member = ZipInfo(baseline.ARCHIVE_PREFIX + name)
                out.writestr(member, members[name], compress_type=ZIP_DEFLATED, compresslevel=6)
        manifest = {
            "schema": SCHEMA,
            "scope": "opened_development_only",
            "lineage": {"source_archive_sha256": baseline.ARCHIVE_SHA256,
                        "frozen_feature_rows_sha256": baseline.ROWS_SHA256,
                        "rescue_fit_allocation_group": FIT_SOURCE,
                        "confirmation_used": False},
            "artifacts": {path.name: {"sha256": sha_file(path), "bytes": path.stat().st_size}
                          | ({"rows": len(inputs[path.name])} if path.name in inputs else {})
                          for path in sorted(stage.iterdir())},
        }
        write_json(stage / "manifest.json", manifest)
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--input-rows", type=Path, default=baseline.ROWS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_archive, args.input_rows, args.output), indent=2))


if __name__ == "__main__":
    main()
