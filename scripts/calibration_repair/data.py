"""F-only source references and intact business groups for calibration repair.

The fixed input bytes and encoder are inherited unchanged. No calibration or
development record is deserialized here. Existing source/business memberships
are used whole; no scored row is used to define, split or select a group.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.calibration_transfer.data import (
    BATCH, Preprocessor, bind_inputs, build_episodes, file_digest,
    json_digest, load_fit, read_f_records,
)


EXPECTED_TARGETS = {
    "F:intranet_server:qualified_normal_business": 125,
    "F:cloud_share:qualified_normal_business": 314,
    "Unraveled:normal:dns": 4388,
    "Unraveled:normal:http_application": 1645,
    "Unraveled:normal:ntp": 884,
    "Unraveled:normal:other_tcp": 27,
    "Unraveled:normal:other_udp": 263,
    "Unraveled:normal:ssh": 20,
    "Unraveled:normal:tls_application": 5081,
}
EXPECTED_REFERENCES = {
    "ait:santos:complete_run": 439,
    "unraveled:apt_2021_single_campaign": 12308,
}
EVIDENCE = Path("docs/normal_tail_mechanism_20261120/episodes_evidence.json")


def assemble_groups(rows, units, components):
    """Validate intact memberships; also usable with synthetic fixtures."""
    ids = [row["flow_id"] for row in rows]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("Unique canonical F identity order is required")
    if any(row["dataset_role"] not in {"F_base", "F_expansion"} or row["label"] not in {0, 1}
           for row in rows):
        raise ValueError("Groups require explicit binary F rows")
    index = {fid: position for position, fid in enumerate(ids)}
    normal = {i for i, row in enumerate(rows) if row["label"] == 0}
    by_source = defaultdict(list)
    for i in sorted(normal):
        by_source[rows[i]["source_group"]].append(i)
    references = [{"id": "reference:" + source, "source": source, "indices": members,
                   "normal_count": len(members), "integer_budget": len(members) // 100}
                  for source, members in sorted(by_source.items())]
    targets, membership = [], Counter()
    if len({unit["unit_id"] for unit in units}) != len(units):
        raise ValueError("Duplicate business unit identity")
    for unit in sorted(units, key=lambda x: x["unit_id"]):
        members = unit["flow_ids"]
        if not members or len(set(members)) != len(members) or any(fid not in index for fid in members):
            raise ValueError("Every business unit needs unique nonempty F memberships")
        positions = sorted(index[fid] for fid in members)
        sources = {rows[i]["source_group"] for i in positions}
        if not set(positions) <= normal or len(sources) != 1:
            raise ValueError("Business unit must contain same-source F normal rows only")
        membership.update(positions)
        targets.append({"id": unit["unit_id"], "source": sources.pop(), "indices": positions,
                        "normal_count": len(positions), "integer_budget": len(positions) // 100,
                        "original_role": unit.get("role"),
                        "label_scope": unit.get("source_grade", "qualified normal actor/business evidence")})
    if set(membership) != normal or any(n != 1 for n in membership.values()):
        raise ValueError("Fixed business units must partition every F normal exactly once")
    targets_by_position = {i: group["id"] for group in targets for i in group["indices"]}
    component_report = []
    for component in components:
        known = component["normal_flow_ids"]
        if not known or any(fid not in index or index[fid] not in normal for fid in known):
            raise ValueError("Known normal component is outside F normal memberships")
        groups = sorted({targets_by_position[index[fid]] for fid in known})
        if len(groups) != 1:
            raise ValueError("Business groups split a known normal action component")
        component_report.append({"component_id": component["component_id"],
                                 "normal_count": len(known), "target_id": groups[0]})
    budgets = []
    for reference in references:
        own = [target for target in targets if target["source"] == reference["source"]]
        target_budget_sum = sum(target["integer_budget"] for target in own)
        gap = reference["integer_budget"] - target_budget_sum
        budgets.append({"source": reference["source"], "normal_count": reference["normal_count"],
                        "whole_source_integer_budget": reference["integer_budget"],
                        "business_integer_budget_sum": target_budget_sum, "floor_gap": gap,
                        "minimum_boundary_tie_size_if_all_own_business_units_pass": gap + 1,
                        "meaning": "At the whole-source boundary, business compliance permits at most the sum of business budgets above it. The pooled boundary rank then requires at least floor_gap+1 tied scores at that boundary; this is not evidence such scores were learned."})
    return {"references": references, "targets": targets,
            "attacks": [i for i, row in enumerate(rows) if row["label"] == 1],
            "normal_partition": {"normal_rows": len(normal), "target_memberships": sum(membership.values()),
                                 "uncovered_normal_rows": 0, "overlapping_memberships": 0},
            "known_action_components": component_report,
            "source_budget_arithmetic": budgets}


def build_groups(rows, repo):
    """Build two full-source reference pools and nine existing F business targets."""
    repo = Path(repo)
    units = read_f_records(repo / BATCH / "normal_units.jsonl.gz")
    evidence = json.loads((repo / EVIDENCE).read_text(encoding="utf8"))
    result = assemble_groups(rows, units, evidence["normal_action_components"])
    if {x["id"]: x["normal_count"] for x in result["targets"]} != EXPECTED_TARGETS:
        raise ValueError("Fixed business group identities or cardinalities changed")
    if {x["source"]: x["normal_count"] for x in result["references"]} != EXPECTED_REFERENCES:
        raise ValueError("Fixed whole-source normal reference pools changed")
    if len(result["attacks"]) != 505 or len(result["known_action_components"]) != 73:
        raise ValueError("Frozen attack or known normal-action support changed")
    return result


def _signature_audit(rows, numeric, categorical):
    encoded = [tuple(a.tolist() + b.tolist()) for a, b in zip(numeric, categorical)]
    raw_fields = ("ip_protocol", "duration_seconds", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes")
    raw = [tuple(row["raw8"].get(k) for k in raw_fields)
           + tuple(row["raw8"].get(k) in (None, -1) for k in ("orig_port", "resp_port"))
           for row in rows]

    def collisions(signatures):
        groups = defaultdict(list)
        for i, signature in enumerate(signatures):
            groups[signature].append(i)
        mixed = [members for members in groups.values() if len({rows[i]["label"] for i in members}) > 1]
        return {"unique_inputs": len(groups), "positive_negative_conflicting_inputs": len(mixed),
                "positive_negative_conflicting_rows": sum(map(len, mixed))}

    normal_signatures, source_counts = defaultdict(list), defaultdict(Counter)
    for i, row in enumerate(rows):
        if row["label"] == 0:
            normal_signatures[encoded[i]].append(i)
            source_counts[row["source_group"]][encoded[i]] += 1
    cross_source = [members for members in normal_signatures.values()
                    if len({rows[i]["source_group"] for i in members}) > 1]
    source_tie_support = {}
    for source, counts in sorted(source_counts.items()):
        # This is input multiplicity, not a score-based boundary selection. A
        # deterministic model ties identical inputs, but those inputs need not
        # occur at its 1% boundary. Distinct inputs can also learn equal scores.
        source_tie_support[source] = {
            "normal_rows": sum(counts.values()), "unique_encoded_inputs": len(counts),
            "multiplicity_histogram": {str(k): v for k, v in sorted(Counter(counts.values()).items())},
            "maximum_multiplicity": max(counts.values()),
            "encoded_input_sets_with_at_least_five_rows": sum(n >= 5 for n in counts.values()),
            "normal_rows_in_input_sets_with_at_least_five_rows": sum(n for n in counts.values() if n >= 5),
        }
    return {"raw_ports_zero": collisions(raw), "actual_FP32_numeric_and_categories": collisions(encoded),
            "normal_encoded_signatures_shared_across_sources": len(cross_source),
            "normal_rows_in_cross_source_signatures": sum(map(len, cross_source)),
            "normal_multiplicity_by_source": source_tie_support,
            "scope": "F inputs only. No label collision removes one deterministic impossibility; it proves neither model optimization nor cross-source generalization. Input multiplicity does not establish learned scores or their boundary rank. Exact cross-source overlap zero is not a lack-of-support theorem."}


def prepare_inputs(repo):
    repo = Path(repo)
    rows = load_fit(repo)
    processor = Preprocessor.fit(rows)
    numeric, categorical = processor.transform(rows)
    groups = build_groups(rows, repo)
    manifest = {
        "format": "calibration-repair-F-groups-v1", "bindings": bind_inputs(repo),
        "F_evidence_sha256": file_digest(repo / EVIDENCE),
        "fit_flow_ids": [row["flow_id"] for row in rows], "fit_rows": len(rows),
        "fit_flow_ids_sha256": json_digest([row["flow_id"] for row in rows]),
        "target_counts": dict(Counter(row["target_kind"] for row in rows)),
        "excluded_F_rows": {"unknown": 123, "other_attack": 11},
        "feature_clock": "offline complete-flow replay_export_time",
        "groups": groups, "signatures": _signature_audit(rows, numeric, categorical),
        "chronological_pairs_diagnostic_only": build_episodes(rows),
        "preprocessor": processor.to_dict(),
        "numeric_shape": list(numeric.shape), "categorical_shape": list(categorical.shape),
        "numeric_sha256": hashlib.sha256(numeric.tobytes()).hexdigest(),
        "categorical_sha256": hashlib.sha256(categorical.tobytes()).hexdigest(),
        "parsed_data_roles": ["F_base", "F_expansion"],
        "normal_business_partition_is_disjoint_in_current_materialized_F": True,
        "known_action_components_preserved": True, "unraveled_action_partition_available": False,
        "source_groups": 2, "independent_evaluation_units_in_this_manifest": 0,
        "online_label_availability_established": False,
        "group_role": "offline F-only training reference and normal-business target; not held-out or chronological validation",
        "model_constructions": 0, "fit_calls": 0, "score_calls": 0, "threshold_calculations": 0,
    }
    return rows, processor, numeric, categorical, manifest


def materialize(repo, output_dir):
    """Write the small receipt and deterministic F-only manifest, without models."""
    _, _, _, _, manifest = prepare_inputs(repo)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False) + "\n").encode()
    path = output_dir / "inputs.json.gz"
    if path.exists() or (output_dir / "inputs_receipt.json").exists():
        raise FileExistsError("Preserve existing input materialization")
    path.write_bytes(gzip.compress(encoded, mtime=0))
    receipt = {"format": "calibration-repair-F-input-receipt-v1", "manifest_path": path.name,
               "manifest_file_sha256": file_digest(path), "manifest_canonical_sha256": json_digest(manifest),
               "manifest_bytes": path.stat().st_size,
               "fit_flow_ids_sha256": manifest["fit_flow_ids_sha256"], "fit_rows": manifest["fit_rows"],
               "target_counts": manifest["target_counts"], "references": len(manifest["groups"]["references"]),
               "targets": len(manifest["groups"]["targets"]),
               "normal_partition": manifest["groups"]["normal_partition"],
               "source_budget_arithmetic": manifest["groups"]["source_budget_arithmetic"],
               "known_normal_action_components_preserved": len(manifest["groups"]["known_action_components"]),
               "input_signature_audit": manifest["signatures"],
               "parsed_data_roles": manifest["parsed_data_roles"],
               "model_constructions": 0, "fit_calls": 0, "score_calls": 0, "threshold_calculations": 0}
    (output_dir / "inputs_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8", newline="\n")
    return receipt
