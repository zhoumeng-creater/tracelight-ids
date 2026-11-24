"""Run one preregistered forward batch of the 22 historical main scorers."""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import platform
from pathlib import Path
import time

import numpy as np

from scripts.ait_forward.baseline import MODEL_ID, load_baseline, load_opened_fox_rows
from scripts.ait_forward.decisions import calibrate_rules, evaluate_candidate, validate_required_event_kinds
from scripts.ait_forward.run import (
    _confirmation_input, _freeze_record, _minimums, _validate_event_protocol, load_qualification,
)
from scripts.comprehensive_batch.execution import exclusive_output
from scripts.comprehensive_batch.matrix import PRIMARY_STEP, SEEDS, job_id, snapshot_id
from scripts.comprehensive_batch.run import package, read, write


SCHEMA = "ait-forward-main-suite-v1"
OBJECTIVES = (
    "A_BCE", "H_reference_attack", "D_reference_alignment",
    "G_group_tail", "R_reference_envelope", "S_alignment_group",
)
HGB_IDS = (
    "HGB_narrow_raw8", "HGB_narrow_ports_zero", "HGB_mixed_raw8", "HGB_mixed_ports_zero",
)


def _tabm(objective, seed):
    return snapshot_id(job_id(objective, seed), PRIMARY_STEP)


def planned_ids():
    return set(HGB_IDS) | {_tabm(objective, seed) for objective in OBJECTIVES for seed in SEEDS}


def planned_pairs():
    narrow_raw, narrow_zero, mixed_raw, mixed_zero = HGB_IDS
    pairs = [
        ("source_bundle", narrow_raw, mixed_raw),
        ("source_bundle", narrow_zero, mixed_zero),
        ("port_removal", narrow_raw, narrow_zero),
        ("port_removal", mixed_raw, mixed_zero),
    ]
    a, h, d, g, r, s = OBJECTIVES
    for seed in SEEDS:
        pairs.extend([
            ("model_recipe", mixed_zero, _tabm(a, seed)),
            *[("auxiliary_vs_BCE", _tabm(a, seed), _tabm(arm, seed)) for arm in (h, d, g, r, s)],
            ("alignment_addition", _tabm(h, seed), _tabm(r, seed)),
            ("alignment_addition", _tabm(g, seed), _tabm(s, seed)),
            ("attack_boundary", _tabm(r, seed), _tabm(s, seed)),
        ])
    return pairs



def runtime_environment(device):
    import torch

    result = {
        "python": platform.python_version(), "platform": platform.platform(),
        "scikit_learn": version("scikit-learn"), "threadpoolctl": version("threadpoolctl"),
        "numpy": np.__version__, "torch": torch.__version__, "tabm": version("tabm"),
        "rtdl_num_embeddings": version("rtdl-num-embeddings"),
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Registered TabM CUDA scoring requires an available CUDA device")
        result["cuda"] = {"torch_cuda": torch.version.cuda,
                          "name": torch.cuda.get_device_name(),
                          "capability": list(torch.cuda.get_device_capability())}
    return result


def _registration(baseline, comparators, qualification, digest, minimums, protocol, device):
    if set(comparators) != planned_ids():
        raise ValueError("Frozen archive does not supply the complete original main comparator set")
    primary = _freeze_record(baseline, qualification, digest, minimums, protocol)
    candidates = {}
    for model_id, model in comparators.items():
        rules = calibrate_rules(model.calibration_scores, baseline.calibration_units)
        if (set(model.calibration_scores) != set(baseline.calibration_scores)
                or rules["pooled"]["threshold"] != model.pooled_threshold):
            raise ValueError(f"Frozen C identity or pooled cutoff differs for {model_id}")
        candidates[model_id] = {
            "family": model.family, "sha256": model.model_sha256,
            "archive_member_sha256": model.archive_member_sha256,
            "rules": rules, "metadata": model.metadata,
        }
    if (candidates[MODEL_ID]["sha256"] != primary["model_sha256"]
            or candidates[MODEL_ID]["rules"] != primary["rules"]):
        raise ValueError("Registered primary scorer differs from its independent frozen binding")
    binding = {
        "schema": SCHEMA, "primary_id": MODEL_ID, "primary": primary, "candidates": candidates,
        "contrasts": planned_pairs(), "tabm_device": device, "threads": 4, "fit_count": 0,
        "runtime": runtime_environment(device),
        "code_sha256": {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("suite.py", "comparators.py")},
    }
    canonical = json.loads(json.dumps(binding, sort_keys=True, allow_nan=False))
    return canonical


def _score_digest(scores):
    return sha256(scores.astype("<f8", copy=False).tobytes()).hexdigest()


def _score(model, rows, output, binding, device):
    path = output / "scores.json"
    if path.exists():
        record = read(path)
        values = np.asarray(record["scores"], dtype=np.float64)
    else:
        started = time.perf_counter()
        values = np.asarray(model.score(rows, device=device, threads=4), dtype=np.float64)
        elapsed = time.perf_counter() - started
        record = {
            "registration_sha256": binding, "scorer_sha256": model.model_sha256, "device": device,
            "score_passes": 1, "score_sha256": _score_digest(values), "scores": values.tolist(),
            "seconds_including_model_load_and_preprocessing": elapsed,
        }
    if (values.shape != (len(rows),) or not np.isfinite(values).all() or record["score_passes"] != 1
            or record["registration_sha256"] != binding or record["scorer_sha256"] != model.model_sha256
            or record["device"] != device or record["score_sha256"] != _score_digest(values)):
        raise ValueError(f"Invalid scoring result for {model.model_id}")
    if not path.exists():
        write(path, record)
    return values


def _changes(before, after, identities, units, attacks, event_before, event_after):
    gained = {identity for identity, left, right in zip(identities, before, after, strict=True)
              if not left and right}
    lost = {identity for identity, left, right in zip(identities, before, after, strict=True)
            if left and not right}
    return {
        "normal_units": [
            {"unit_id": unit, "FP_added": len(gained.intersection(ids)), "FP_removed": len(lost.intersection(ids))}
            for unit, ids in units.items()
        ],
        "targets": {kind: None if ids is None else {"gained": sorted(gained.intersection(ids)),
                                                   "lost": sorted(lost.intersection(ids))}
                    for kind, ids in attacks.items()},
        "event_changes": [
            {"event_id": left["event_id"], "before": [left["lower_hit"], left["upper_hit"]],
             "after": [right["lower_hit"], right["upper_hit"]]}
            for left, right in zip(event_before, event_after, strict=True)
            if (left["lower_hit"], left["upper_hit"]) != (right["lower_hit"], right["upper_hit"])
        ],
    }


def _paired(predictions, reports, identities, units, attacks):
    items = []
    for question, before, after in planned_pairs():
        pair = {"question": question, "before": before, "after": after}
        if before not in reports or after not in reports:
            items.append({**pair, "status": "score_unavailable"})
            continue
        changes = {}
        for name in ("pooled", "business_max"):
            changes[name] = _changes(
                predictions[before][name], predictions[after][name], identities, units, attacks,
                reports[before]["result"]["rules"][name]["events"],
                reports[after]["result"]["rules"][name]["events"],
            )
        items.append({**pair, "status": "evaluated", "changes": changes})
    return items


def _opened_fox_preflight(output, binding, baseline, comparators, registration, device):
    path = output / "opened_fox_regression.json"
    if path.exists():
        saved = read(path)
        reports = saved["reports"]
        encoded = json.dumps(reports, sort_keys=True, allow_nan=False).encode()
        if (saved["registration_sha256"] != binding or set(reports) != set(comparators)
                or saved["reports_sha256"] != sha256(encoded).hexdigest()):
            raise ValueError("Saved opened Fox regression differs from the registered suite")
    else:
        fox = load_opened_fox_rows(baseline)
        reports = {}
        for model_id, candidate in comparators.items():
            candidate_device = device if candidate.family == "tabm" else "cpu"
            reports[model_id] = candidate.historical_fox_regression(
                fox, registration["candidates"][model_id]["rules"], device=candidate_device, threads=4,
            )
        encoded = json.dumps(reports, sort_keys=True, allow_nan=False).encode()
        write(path, {"registration_sha256": binding,
                     "reports_sha256": sha256(encoded).hexdigest(), "reports": reports})
    if any(rule["flipped_decisions"] for report in reports.values() for rule in report["rules"].values()):
        raise ValueError("Opened Fox decision regression failed. Check the fixed TabM device and environment.")



def _load_event_protocol(protocol_path, qualification):
    content = Path(protocol_path).read_bytes()
    record = json.loads(content.decode("utf-8-sig"))
    if record is None:
        raise ValueError("The forward suite requires a prespecified event protocol")
    bindings = qualification.get("lineage", {}).get("evidence_files", [])
    registered = [binding["sha256"] for binding in bindings
                  if Path(binding["path"]).name == "event-protocol.json"]
    if registered and registered != [sha256(content).hexdigest()]:
        raise ValueError("Event protocol differs from the source-qualified event registry")
    return _validate_event_protocol(record, qualification)



def execute(archive, qualification_path, protocol_path, output, device, *, control=None, download=None):
    from scripts.ait_forward.comparators import load_comparators

    qualification, digest = load_qualification(qualification_path)
    protocol = _load_event_protocol(protocol_path, qualification)
    minimums = _minimums(qualification, control, download)
    baseline = load_baseline(archive)
    comparators = load_comparators(archive, baseline)
    registration = _registration(baseline, comparators, qualification, digest, minimums, protocol, device)

    output = Path(output).resolve()
    path = output / "registration.json"
    with exclusive_output(output):
        if path.exists():
            if read(path) != registration:
                raise ValueError("Archived models, rules, inputs, protocol, device or code changed")
        else:
            if any(path.name != ".execution.lock" for path in output.iterdir()):
                raise ValueError("Use a new output directory for the initial suite")
            write(path, registration)
        binding = sha256(path.read_bytes()).hexdigest()
        try:
            _opened_fox_preflight(output, binding, baseline, comparators, registration, device)
            write(output / "archived_training_and_cost.json", {
                model_id: {"original_summary": model.original_summary, "cpu_benchmark": model.benchmark,
                           "training_history_summary": model.training_history_summary,
                           "primary_training_diagnostics": model.primary_training_diagnostics}
                for model_id, model in comparators.items()
            })
            rows, units, attacks, events = _confirmation_input(
                Path(qualification_path).parent, qualification, baseline, minimums,
            )
            required_events = protocol["required_events"]
            validate_required_event_kinds(events, required_events)
            identities = [row["flow_id"] for row in rows]
            write(output / "evaluation_identities.json", [
                {key: row.get(key) for key in (
                    "flow_id", "stable_flow_id", "label", "target_kind", "label_evidence",
                    "observed_end", "replay_export_time", "actual_available_at",
                )} for row in rows
            ])
            write(output / "evaluation_membership.json", {"normal_units": units, "events": events})
            reports, predictions, failed = {}, {}, {}
            for model_id, candidate in comparators.items():
                candidate_output = output / "scorers" / model_id
                candidate_output.mkdir(parents=True, exist_ok=True)
                rules = registration["candidates"][model_id]["rules"]
                candidate_device = device if candidate.family == "tabm" else "cpu"
                try:
                    values = _score(candidate, rows, candidate_output, binding, candidate_device)
                    scores = dict(zip(identities, values.tolist(), strict=True))
                    comparison = evaluate_candidate(
                        scores, normal_units=units, attack_ids=attacks, events=events, rules=rules,
                        minimum_tp=minimums, required_events=required_events,
                    )
                    suffix = ".pt" if candidate.family == "tabm" else ".joblib"
                    model_path = candidate_output / f"model{suffix}"
                    if not model_path.exists() or sha256(model_path.read_bytes()).hexdigest() != candidate.model_sha256:
                        temporary = model_path.with_suffix(".tmp")
                        temporary.write_bytes(candidate.model_bytes)
                        temporary.replace(model_path)
                    write(candidate_output / "evaluation.json", comparison)
                    reports[model_id] = comparison
                    predictions[model_id] = {name: values > rule["threshold"] for name, rule in rules.items()}
                except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
                    failed[model_id] = {"type": type(error).__name__, "message": str(error)}
            contrasts = _paired(predictions, reports, identities, units, attacks)
            write(output / "paired_comparisons.json", contrasts)
            write(output / "summary.json", {
                "status": "complete" if not failed else "incomplete",
                "primary_id": MODEL_ID, "fit_count": 0, "planned_scorers": len(comparators),
                "successful_score_passes": len(reports), "failed_scorers": failed,
                "attempted_qualified_targets": qualification["targets"], "rows": len(rows),
                "unknown": sum(row["label"] is None for row in rows),
                "forward_scope": "one later AIT run, offline full-flow features",
                "outcomes": {key: {
                    "flow": report["outcomes"], "confirmation": report["confirmation_outcomes"],
                    "contrast": report["result"]["rule_contrast"],
                } for key, report in reports.items()},
            })
            if failed:
                raise RuntimeError(f"{len(failed)} registered scorer(s) failed; resume the same output")
            (output / "interruption.json").unlink(missing_ok=True)
            return package(output)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, KeyboardInterrupt) as error:
            write(output / "interruption.json", {"type": type(error).__name__, "message": str(error)})
            package(output)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--event-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tabm-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--control", type=int)
    parser.add_argument("--download", type=int)
    args = parser.parse_args()
    output = execute(
        args.archive, args.qualification, args.event_protocol, args.output, args.tabm_device,
        control=args.control, download=args.download,
    )
    print(output)


if __name__ == "__main__":
    main()
