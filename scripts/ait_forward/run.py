"""Run the fixed, zero-fit AIT comparison using one HGB and two C-only rules."""
from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from scripts.ait_forward.baseline import load_baseline
from scripts.ait_forward.decisions import calibrate_rules, evaluate_comparison
from scripts.comprehensive_batch.execution import exclusive_output
from scripts.comprehensive_batch.run import package, read, write
from scripts.coverage_batch.materialize import RAW8_FIELDS

SCHEMA = "ait-forward-calibration-v1"
CONFIRMATION_SCHEMA = "ait-harrison-qualification-v1"
GROUP = "ait:harrison:complete_run"
CLOCK = "offline_Zeek_export_network_time_not_detector_receipt"
TARGETS = ("control", "download")
INPUTS = ("rows.jsonl.gz", "normal_units.jsonl.gz", "events.jsonl.gz")
SHA256 = re.compile(r"[a-f0-9]{64}")


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _confirmation_window(spec):
    window = spec.get("confirmation_window", {})
    values = [window.get(name) for name in ("start_utc", "end_utc")]
    if not all(isinstance(value, str) and value.endswith("Z") for value in values):
        raise ValueError("Provide the qualified confirmation window with explicit UTC timestamps")
    start, end = (datetime.fromisoformat(value).timestamp() for value in values)
    if start >= end:
        raise ValueError("Confirmation window start must precede its end")
    return start, end


def load_qualification(path):
    """Read the evidence receipt alone, without opening any confirmation rows."""
    payload = Path(path).read_bytes()
    spec = json.loads(payload.decode("utf-8-sig"))
    if (
        spec.get("schema") != CONFIRMATION_SCHEMA
        or spec.get("status") != "qualified"
        or spec.get("source_group") != GROUP
        or spec.get("clock_basis") != CLOCK
    ):
        raise ValueError("Confirmation requires a qualified, offline Harrison receipt")
    _confirmation_window(spec)
    files = spec.get("files", {})
    if set(files) != set(INPUTS) or any(
        not isinstance(value, str) or not SHA256.fullmatch(value) for value in files.values()
    ):
        raise ValueError("Qualification must bind the three input files by SHA-256")
    if not _nonempty_string(spec.get("normal_evidence")) or not _nonempty_string(spec.get("clock_evidence")):
        raise ValueError("Qualification must identify normal denominator and clock evidence")
    targets = spec.get("targets", {})
    if set(targets) != set(TARGETS) or any(
        item.get("status") not in {"qualified", "unavailable"}
        or not _nonempty_string(item.get("evidence"))
        for item in targets.values()
    ):
        raise ValueError("Qualification must record separate control and download evidence")
    if not any(item["status"] == "qualified" for item in targets.values()):
        raise ValueError("At least one confirmation target must be qualified")
    return spec, hashlib.sha256(payload).hexdigest()


def _minimums(qualification, control, download):
    policy = dict(control=control, download=download)
    for kind, minimum in policy.items():
        if qualification["targets"][kind]["status"] == "qualified":
            if type(minimum) is not int or minimum < 1:
                raise ValueError(f"Set the predeclared minimum TP count for {kind}")
        elif minimum is not None:
            raise ValueError(f"Unavailable target {kind} cannot have a minimum TP count")
    return policy


def _freeze_record(baseline, qualification, qualification_digest, minimum_tp):
    """Build the immutable rule and input binding without reading the holdout."""
    rules = calibrate_rules(baseline.calibration_scores, baseline.calibration_units)
    if rules["pooled"]["threshold"] != baseline.pooled_threshold:
        raise ValueError("Pooled rule does not reproduce the original frozen threshold")
    start, _ = _confirmation_window(qualification)
    if not math.isfinite(baseline.last_calibration_time) or baseline.last_calibration_time >= start:
        raise ValueError("Frozen calibration timestamps must precede the Harrison window")
    record = {
        "schema": SCHEMA, "model_sha256": baseline.model_sha256,
        "archive_member_sha256": baseline.archive_member_sha256,
        "scoring_clock": CLOCK, "last_calibration_time": baseline.last_calibration_time,
        "qualification_sha256": qualification_digest,
        "qualification": qualification, "minimum_tp": minimum_tp, "rules": rules,
        "code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("baseline.py", "decisions.py", "run.py")
        },
        "fit_count": 0,
    }
    encoded = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return {**record, "registration_digest": hashlib.sha256(encoded).hexdigest()}


def freeze(archive, qualification_path, output, control=None, download=None):
    qualification, digest = load_qualification(qualification_path)
    minimum_tp = _minimums(qualification, control, download)
    baseline = load_baseline(archive)
    record = _freeze_record(baseline, qualification, digest, minimum_tp)
    output = Path(output)
    receipt = output / "registration.json"
    with exclusive_output(output):
        if receipt.exists():
            if read(receipt) != record:
                raise ValueError("Output already has a different frozen registration")
        else:
            if any(path.name != ".execution.lock" for path in output.iterdir()):
                raise ValueError("Use an empty output directory for registration")
            write(receipt, record)
    return record


def _verified_rows(directory, name, qualification):
    payload = (Path(directory) / name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != qualification["files"][name]:
        raise ValueError(f"Confirmation file differs from its qualified hash: {name}")
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _confirmation_input(directory, qualification, baseline, minimum_tp):
    start, limit = _confirmation_window(qualification)
    rows = _verified_rows(directory, INPUTS[0], qualification)
    units = _verified_rows(directory, INPUTS[1], qualification)
    events = _verified_rows(directory, INPUTS[2], qualification)
    if not rows or not units:
        raise ValueError("Confirmation requires observations and true normal business units")
    flows = set()
    stable_flows = set()
    normal = set()
    positive = {kind: [] for kind in TARGETS}
    for row in rows:
        flow, stable = row.get("flow_id"), row.get("stable_flow_id")
        if not _nonempty_string(flow) or not _nonempty_string(stable):
            raise ValueError("Each row needs flow_id and stable_flow_id")
        if (
            flow in flows or stable in stable_flows or flow in baseline.known_flow_ids
            or stable in baseline.known_stable_flow_ids
        ):
            raise ValueError(f"Duplicated identity or previous-role overlap: {flow}")
        flows.add(flow)
        stable_flows.add(stable)
        if (
            row.get("allocation_group") != GROUP or row.get("source_run") != "harrison"
            or row.get("dataset_role") != "confirmation" or row.get("clock_basis") != CLOCK
        ):
            raise ValueError(f"Row is outside the registered Harrison whole-run role: {flow}")
        end, when = row.get("observed_end"), row.get("replay_export_time")
        if (
            type(end) not in (int, float) or type(when) not in (int, float)
            or not start <= end <= when < limit or when <= baseline.last_calibration_time
        ):
            raise ValueError(f"Invalid complete-flow export time: {flow}")
        raw8 = row.get("raw8")
        if (
            not isinstance(raw8, dict) or set(raw8) != set(RAW8_FIELDS) or "features8" in row
            or any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
                   for value in raw8.values())
        ):
            raise ValueError(f"Use the canonical raw8 fields with null for missing measurements: {flow}")
        label = row.get("label")
        if "label" not in row or (label is not None and (type(label) is not int or label not in (0, 1))):
            raise ValueError(f"Specify a normal, attack, or unknown label: {flow}")
        if label is not None and not _nonempty_string(row.get("label_evidence")):
            raise ValueError(f"Known-label flow lacks provenance: {flow}")
        if label == 0:
            normal.add(flow)
        elif label == 1:
            kind = row.get("target_kind")
            if kind not in positive:
                raise ValueError(f"Specify control or download for attack flow: {flow}")
            positive[kind].append(flow)

    normal_units = {}
    for unit in units:
        uid, members = unit.get("unit_id"), unit.get("flow_ids")
        if not _nonempty_string(uid) or uid in normal_units or not isinstance(members, list) or not members:
            raise ValueError("Normal units need unique IDs and nonempty flow membership")
        if not all(_nonempty_string(member) for member in members) or not set(members) <= normal:
            raise ValueError(f"Normal unit contains non-normal or absent flows: {uid}")
        if len(members) != len(set(members)):
            raise ValueError(f"Repeated membership within normal unit: {uid}")
        normal_units[uid] = members
    if set().union(*map(set, normal_units.values())) != normal:
        raise ValueError("Each known normal flow must be in a registered business unit")

    attack_ids = {}
    for kind, ids in positive.items():
        minimum = minimum_tp[kind]
        if minimum is None and ids:
            raise ValueError(f"Found qualified positives for an unavailable target: {kind}")
        if minimum is not None and len(ids) < minimum:
            raise ValueError(f"The qualified denominator cannot meet the predeclared floor for {kind}")
        attack_ids[kind] = ids if minimum is not None else None

    event_ids = set()
    event_targets = set()
    for event in events:
        identity, kind = event.get("event_id"), event.get("kind")
        candidates = event.get("candidate_flow_ids")
        if not _nonempty_string(identity) or identity in event_ids or kind not in TARGETS:
            raise ValueError("Event requires a unique identity and a registered target kind")
        if (
            not isinstance(candidates, list)
            or not all(_nonempty_string(candidate) for candidate in candidates)
            or not set(candidates) <= flows - normal
            or len(set(candidates)) != len(candidates)
            or event.get("association_mode") not in {"all_confirmed", "one_of"}
            or type(event.get("candidate_set_complete")) is not bool
            or not _nonempty_string(event.get("label_evidence"))
        ):
            raise ValueError(f"Invalid or unsubstantiated event-to-flow mapping: {identity}")
        if set(candidates).intersection(positive["download" if kind == "control" else "control"]):
            raise ValueError(f"Event candidates contain a flow of the opposite known target: {identity}")
        if minimum_tp[kind] is None:
            raise ValueError(f"Unavailable endpoint contains a registered positive event: {identity}")
        if set(candidates).intersection(positive[kind]):
            event_targets.add(kind)
        event_ids.add(identity)
    if event_targets != {kind for kind, ids in attack_ids.items() if ids is not None}:
        raise ValueError("Each qualified target needs a positive event linked to its known flows")

    return rows, normal_units, attack_ids, events


def _score_pass(output, baseline, rows):
    path = output / "score_pass.json"
    binding = hashlib.sha256((output / "registration.json").read_bytes()).hexdigest()
    identities = [row["flow_id"] for row in rows]
    if path.exists():
        record = read(path)
        if record["registration_sha256"] != binding or record["flow_ids"] != identities:
            raise ValueError("Saved score pass differs from the registration or input identities")
    else:
        start = time.perf_counter()
        scores = np.asarray(baseline.score(rows), dtype=np.float64)
        seconds = time.perf_counter() - start
        if scores.shape != (len(rows),) or not np.isfinite(scores).all():
            raise ValueError("Confirmation requires one finite score per registered observation")
        record = {
            "registration_sha256": binding, "flow_ids": identities, "scores": scores.tolist(),
            "seconds": seconds, "score_passes": 1,
            "scores_sha256": hashlib.sha256(scores.astype("<f8", copy=False).tobytes()).hexdigest(),

            "timing_scope": "single CPU scoring call including preparation and any model load",
        }
        write(path, record)
    scores = np.asarray(record["scores"], dtype=np.float64)
    if (
        scores.shape != (len(rows),) or not np.isfinite(scores).all() or record["score_passes"] != 1
        or hashlib.sha256(scores.astype("<f8", copy=False).tobytes()).hexdigest() != record["scores_sha256"]
    ):
        raise ValueError("Invalid cached score pass")
    return record, dict(zip(identities, scores.tolist(), strict=True))


def _result_with_status(result, attack_ids):
    """Report the evaluated scope; diagnostics never substitute an oracle cutoff."""
    outcome = {}
    for name, rule in result["rules"].items():
        targets = rule["targets"]
        floors_met = all(targets[kind]["minimum_TP_met"] for kind in TARGETS if attack_ids[kind] is not None)
        ranking = result["score_ranking_upper_bound"]
        if rule["all_normal_units_within_budget"] and floors_met:
            status = "criteria_met" if all(attack_ids.values()) else "partial_scope_criteria_met"
        elif ranking and all(
            ranking[kind]["minimum_TP_attainable"] for kind in TARGETS if attack_ids[kind] is not None
        ):
            status = "fixed_rule_failed_score_ordering_has_room"
        else:
            status = "fixed_score_ordering_insufficient"
        outcome[name] = status
    return outcome


def run(archive, qualification_path, output):
    output = Path(output)
    if not (output / "registration.json").is_file():
        raise ValueError("Freeze the model, C rules, targets and input hashes before reading confirmation")
    qualification, digest = load_qualification(qualification_path)
    baseline = load_baseline(archive)
    with exclusive_output(output):
        registration = read(output / "registration.json")
        minimum = _minimums(qualification, **registration["minimum_tp"])
        if registration != _freeze_record(baseline, qualification, digest, minimum):
            raise ValueError("Frozen registration no longer matches model, code, C rules or qualification")
        try:
            rows, units, attacks, events = _confirmation_input(
                Path(qualification_path).parent, qualification, baseline, minimum
            )
            pass_record, scores = _score_pass(output, baseline, rows)
            result = evaluate_comparison(
                scores, normal_units=units, attack_ids=attacks, events=events,
                rules=registration["rules"], minimum_tp=minimum,
            )
            model = output / "model.joblib"
            model_target = model.with_suffix(".tmp")
            model_target.write_bytes(baseline.model_bytes)
            os.replace(model_target, model)
            write(output / "decisions.json", [
                {
                    **{key: row.get(key) for key in (
                        "flow_id", "stable_flow_id", "label", "target_kind", "label_evidence",
                        "replay_export_time", "actual_available_at",
                    )},
                    "score": scores[row["flow_id"]],
                    "alerts": {
                        name: scores[row["flow_id"]] > rule["threshold"]
                        for name, rule in registration["rules"].items()
                    },
                }
                for row in rows
            ])
            write(output / "evaluation_membership.json", {"normal_units": units, "events": events})
            write(output / "summary.json", {
                "status": "complete", "fit_count": 0, "score_passes": pass_record["score_passes"],
                "confirmation_scope": "later AIT run; shared attack implementation; offline export",
                "rows": len(rows), "unknown_labels": sum(row["label"] is None for row in rows),
                "qualification_targets": qualification["targets"],
                "outcomes": _result_with_status(result, attacks), "result": result,
            })
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, KeyboardInterrupt) as error:
            write(output / "interruption.json", {"type": type(error).__name__, "message": str(error)})
            package(output)
            raise
        return package(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("verify", "freeze", "run"):
        stage = stages.add_parser(name)
        stage.add_argument("--archive", type=Path, required=True, help="Original verified 11-22 result ZIP")
        if name != "verify":
            stage.add_argument("--qualification", type=Path, required=True)
            stage.add_argument("--output", type=Path, required=True)
        if name == "freeze":
            stage.add_argument("--control", type=int, help="Frozen control TP floor; omit if unavailable")
            stage.add_argument("--download", type=int, help="Frozen download TP floor; omit if unavailable")
    args = parser.parse_args()
    if args.stage == "verify":
        baseline = load_baseline(args.archive)
        rules = calibrate_rules(baseline.calibration_scores, baseline.calibration_units)
        report = {"calibration": rules, "fox": baseline.historical_fox_regression(rules)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.stage == "freeze":
        freeze(args.archive, args.qualification, args.output, args.control, args.download)
        print(f"Frozen without opening confirmation rows: {args.output / 'registration.json'}")
    else:
        result = run(args.archive, args.qualification, args.output)
        print(f"Return this ZIP and SHA-256 receipt: {result}")


if __name__ == "__main__":
    main()
