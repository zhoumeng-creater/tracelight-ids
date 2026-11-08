"""Prepare unseen communication-group confirmation data; never load a model.

The target population is the same official source conditioned on excluding all
communication groups already sampled in rounds 13 and 14. It is not a temporal,
host-disjoint, or population-FPR validation set.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import heapq
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from ids.common import sha256_file
from ids.context_cache import lab_data as lab
from ids.context_cache.csv_workflow import _feature_signatures
from ids.context_cache.preprocessing import CSVSchema, frame_features, read_csv


CONFIRMATION_SEED = 20261109
FORMAT = "tracelight-unseen-communication-group-confirmation-v1"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _digest(value, name, length=64):
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"Invalid {name} digest")
    return value


def _source_receipt(receipt, expected_format):
    source = receipt.get("source", {})
    if (receipt.get("format") != expected_format or receipt.get("data_kind") != "development"
            or receipt.get("seed") != lab.SEED
            or source.get("dataset") != lab.DATASET
            or source.get("official_release_verified") is not True
            or source.get("sha1") != lab.SOURCE_SHA1 or source.get("bytes") != lab.SOURCE_BYTES
            or CSVSchema.from_dict(receipt.get("schema", {})).to_dict() != CSVSchema.from_dict(lab.SCHEMA).to_dict()):
        raise ValueError("Preparation receipt does not identify the frozen official source and schema")
    _digest(source.get("sha256"), "source SHA-256")
    return source


def _read_seen(inputs, previous_prepared_dir):
    schema = inputs["schema"]
    if schema.to_dict() != CSVSchema.from_dict(lab.SCHEMA).to_dict():
        raise ValueError("Confirmation requires the registered NF-v3 48-field schema")
    current_paths = inputs["paths"]
    current_receipt_path = Path(current_paths.get(
        "preparation", Path(current_paths["support"]).parent / "preparation.json"))
    previous_root = Path(previous_prepared_dir).resolve()
    previous_receipt_path = previous_root / "preparation.json"
    receipts = {"round13": _json(previous_receipt_path), "round14": _json(current_receipt_path)}
    previous_source = _source_receipt(receipts["round13"], "tracelight-nf-ton-v3-preparation-v1")
    current_source = _source_receipt(receipts["round14"], "tracelight-nf-ton-v3-preparation-v2")
    if any(previous_source[key] != current_source[key] for key in ("sha1", "sha256", "bytes")):
        raise ValueError("Rounds 13 and 14 identify different source files")
    seen_ids, seen_groups, seen_patterns, evidence = {}, set(), set(), []
    for round_name, receipt in receipts.items():
        for role in lab.ROLES:
            path = (previous_root / lab.FILENAMES[role] if round_name == "round13"
                    else Path(current_paths[role])).resolve()
            item = receipt.get("files", {}).get(role, {})
            expected_hash = _digest(item.get("sha256"), f"{round_name}/{role} SHA-256")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"Seen CSV hash mismatch: {round_name}/{role}")
            if round_name == "round14":
                registered = inputs["spec"]["files"][role]
                if registered.get("sha256") != actual_hash or registered.get("rows") != item.get("rows"):
                    raise ValueError(f"Current registered CSV differs from preparation: {role}")
            frame = read_csv(path, schema)
            if len(frame) != item.get("rows"):
                raise ValueError(f"Seen CSV row count mismatch: {round_name}/{role}")
            ids = frame[schema.id_column].astype(str).tolist()
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate sample IDs within {round_name}/{role}")
            groups = frame[schema.group_column].astype(str).tolist()
            for sample_id, group, label in zip(ids, groups, frame[schema.label_column]):
                if re.fullmatch(re.escape(current_source["sha256"]) + r":[1-9][0-9]*", sample_id) is None:
                    raise ValueError("Seen sample ID does not belong to the registered source")
                _digest(group, "communication group")
                identity = (group, int(label))
                if sample_id in seen_ids and seen_ids[sample_id] != identity:
                    raise ValueError("Conflicting seen row identity across rounds")
                seen_ids[sample_id] = identity
                seen_groups.add(group)
            seen_patterns.update(_feature_signatures(frame_features(frame, schema), schema))
            evidence.append({"round": round_name, "role": role, "path": str(path),
                             "sha256": actual_hash, "rows": len(frame)})
    receipt_evidence = [{"round": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
                        for name, path in (("round13", previous_receipt_path),
                                           ("round14", current_receipt_path))]
    return current_source, seen_ids, seen_groups, seen_patterns, evidence, receipt_evidence


def _confirmation_role(group, seed):
    bucket = int(lab._digest(["route-confirmation-group-v1", seed, group])[:16], 16) % 100
    return "calibration" if bucket < 20 else "evaluation"


def _priority(row_number, source_sha256, seed):
    return int(lab._digest(["route-confirmation-row-v1", seed, source_sha256, row_number]), 16)


class _HashingReader(io.RawIOBase):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.sha1, self.sha256, self.byte_count = hashlib.sha1(), hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        block = self.stream.read(len(buffer))
        size = len(block)
        buffer[:size] = block
        self.sha1.update(block)
        self.sha256.update(block)
        self.byte_count += size
        return size


def _increment(counter, label):
    counter["rows"] += 1
    counter["normal" if label == 0 else "attack"] += 1


def _counts(counter):
    return {key: int(counter[key]) for key in ("rows", "normal", "attack")}


def _scan_rows(reader, *, source_sha256, seen_ids, seen_groups, seed, quotas):
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise ValueError("Official source CSV is empty") from exc
    header = [name.strip() for name in raw_header]
    if len(set(header)) != len(header):
        raise ValueError("Duplicate source CSV headers")
    required = {*lab.RAW_FEATURES, "Label", "Attack"}
    if not required.issubset(header):
        raise ValueError(f"Official CSV header mismatch: {sorted(required - set(header))}")
    indices = {name: index for index, name in enumerate(header)}
    counts = {key: Counter() for key in (
        "source", "excluded_union", "excluded_by_seen_id", "excluded_by_seen_group",
        "remaining", "calibration_pool", "evaluation_pool")}
    excluded_group_counts, found_ids = Counter(), set()
    heaps = {role: [] for role in quotas}
    family_counts = {role: Counter() for role in quotas}
    calibration_attacks = 0
    for row_number, row in enumerate(reader, 1):
        if len(row) != len(header):
            raise ValueError(f"Wrong source field count at row {row_number}")
        identifiers = [lab._source_identifier(row[indices[name]], maximum)
                       for name, maximum in lab.IDENTIFIER_DOMAINS]
        label = lab._integer(row[indices["Label"]], "binary label", 1)
        family = lab.FAMILIES.get(row[indices["Attack"]].strip().casefold())
        if family is None or (family == "Benign") != (label == 0):
            raise ValueError(f"Binary/family label disagreement at source row {row_number}")
        group = lab._communication_group_from_ids(
            row[indices["IPV4_SRC_ADDR"]], row[indices["IPV4_DST_ADDR"]], *identifiers[:3])
        sample_id = f"{source_sha256}:{row_number}"
        _increment(counts["source"], label)
        seen_id, seen_group = sample_id in seen_ids, group in seen_groups
        if seen_id:
            if seen_ids[sample_id] != (group, label):
                raise ValueError(f"Seen row identity disagrees with official source at row {row_number}")
            found_ids.add(sample_id)
            _increment(counts["excluded_by_seen_id"], label)
        if seen_group:
            _increment(counts["excluded_by_seen_group"], label)
            excluded_group_counts[group] += 1
        if seen_id or seen_group:
            _increment(counts["excluded_union"], label)
            continue
        _increment(counts["remaining"], label)
        role = _confirmation_role(group, seed)
        _increment(counts[f"{role}_pool"], label)
        if role == "calibration" and label != 0:
            calibration_attacks += 1
            continue
        family_counts[role][family] += 1
        priority = _priority(row_number, source_sha256, seed)
        heap, quota = heaps[role], quotas[role]
        if len(heap) == quota and (priority, row_number) >= (-heap[0][0], -heap[0][1]):
            continue
        values, _ = lab._features(row, indices)
        output_row = values + [str(label), sample_id, group, family]
        item = (-priority, -row_number, output_row)
        if len(heap) < quota:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)
    if found_ids != set(seen_ids):
        raise ValueError(f"Source scan did not recover all seen row IDs: {len(seen_ids) - len(found_ids)} missing")
    selected = {role: [item[2] for item in sorted(heap, key=lambda item: (-item[0], -item[1]))]
                for role, heap in heaps.items()}
    largest_excluded = max(excluded_group_counts.values(), default=0)
    report = {key: _counts(counter) for key, counter in counts.items()}
    report.update({"seen_ids_recovered": len(found_ids),
                   "calibration_attack_rows_not_sampled": calibration_attacks,
                   "eligible_family_counts": {role: dict(sorted(values.items()))
                                               for role, values in family_counts.items()},
                   "excluded_largest_communication_group_rows": largest_excluded,
                   "excluded_largest_group_fraction_of_source":
                       largest_excluded / counts["source"]["rows"] if counts["source"]["rows"] else None,
                   "source_header": header})
    return selected, report


def prepare_confirmation(source_zip, inputs, output_dir, *, previous_prepared_dir,
                         seed=CONFIRMATION_SEED, calibration_rows=10000, evaluation_rows=50000):
    """Stream the verified source once and write C'/E' after excluding both rounds.

    The caller freezes candidates before invoking this function. This function
    has no model, threshold, candidate-selection, or prediction capability.
    ``previous_prepared_dir`` is the round-13 directory containing its receipt
    and all three original prepared CSV files.
    """
    if type(seed) is not int or seed != CONFIRMATION_SEED:
        raise ValueError("Confirmation seed is frozen at 20261109")
    if any(type(value) is not int or value <= 0 for value in (calibration_rows, evaluation_rows)):
        raise ValueError("Confirmation row budgets must be positive integers")
    root, archive_path = Path(output_dir).resolve(), Path(source_zip).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Confirmation output directory must be empty")
    source, seen_ids, seen_groups, seen_patterns, evidence, receipts = _read_seen(inputs, previous_prepared_dir)
    archive_sha = sha256_file(archive_path)
    registered_archive = inputs["spec"].get("source_zip")
    if registered_archive is None:
        registered_archive = inputs["spec"].get("artifacts", {}).get("source_zip")
    if registered_archive is not None:
        expected = _digest(registered_archive.get("sha256"), "registered source ZIP SHA-256")
        if archive_sha != expected:
            raise ValueError("Source ZIP differs from registered SHA-256")
    quotas = {"calibration": calibration_rows, "evaluation": evaluation_rows}
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist()
                   if not item.is_dir() and PurePosixPath(item.filename).name == "NF-ToN-IoT-v3.csv"]
        if len(members) != 1:
            raise ValueError("Source ZIP must contain exactly one NF-ToN-IoT-v3.csv")
        member = members[0]
        if member.file_size != lab.SOURCE_BYTES or member.file_size != source["bytes"]:
            raise ValueError("Source CSV member byte count differs from official receipt")
        with archive.open(member) as binary:
            hashing = _HashingReader(binary)
            with io.TextIOWrapper(io.BufferedReader(hashing), encoding="utf-8-sig", newline="") as text:
                selected, scan = _scan_rows(csv.reader(text, strict=True), source_sha256=source["sha256"],
                                            seen_ids=seen_ids, seen_groups=seen_groups, seed=seed, quotas=quotas)
        observed = {"sha1": hashing.sha1.hexdigest(), "sha256": hashing.sha256.hexdigest(),
                    "bytes": hashing.byte_count}
        if (observed["sha1"] != lab.SOURCE_SHA1
                or any(observed[key] != source[key] for key in ("sha1", "sha256", "bytes"))):
            raise ValueError("Complete source CSV digest differs from official/preparation receipt")
    for role, quota in quotas.items():
        if len(selected[role]) != quota:
            raise ValueError(f"Insufficient unseen-group {role} rows: {len(selected[role])} < {quota}; seed unchanged")
    root.mkdir(parents=True, exist_ok=True)
    schema = inputs["schema"]
    files, frames, signatures = {}, {}, {}
    for role, rows in selected.items():
        name = "normal_calibration.csv" if role == "calibration" else "confirmation_evaluation.csv"
        path = root / name
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow([*lab.FEATURES, "label", "sample_id", "communication_group", "attack_family"])
            writer.writerows(rows)
        frames[role] = read_csv(path, schema)
        signatures[role] = _feature_signatures(frame_features(frames[role], schema), schema)
        files[role] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
    cal_groups = set(frames["calibration"][schema.group_column])
    eval_groups = set(frames["evaluation"][schema.group_column])
    if cal_groups & eval_groups or (cal_groups | eval_groups) & seen_groups:
        raise AssertionError("Confirmation communication-group separation failed")
    overlaps = {}
    for role, values in signatures.items():
        patterns = set(values)
        overlaps[role] = {"unique_input_patterns": len(patterns),
                          "within_role_duplicate_rows": len(values) - len(patterns),
                          "unique_patterns_shared_with_seen": len(patterns & seen_patterns),
                          "rows_with_input_pattern_seen_before": sum(value in seen_patterns for value in values),
                          "rows_with_new_input_pattern": sum(value not in seen_patterns for value in values)}
    report = {"format": FORMAT, "status": "prepared", "seed": seed,
              "scope": "same-source flow-weighted confirmation conditional on excluding all round13/14 sampled communication groups",
              "generalization_scope": "unseen communication groups; not time-disjoint or host-disjoint",
              "risk_unit": "flow", "population_fpr_guarantee": False,
              "model_runs": 0, "preprocessor_fits": 0, "threshold_registrations": 0,
              "source": {**observed, "dataset": lab.DATASET, "official_release_verified": True,
                         "archive_path": str(archive_path), "archive_sha256": archive_sha,
                         "registered_archive_sha256_verified": registered_archive is not None,
                         "member": member.filename, "member_crc32": f"{member.CRC:08x}",
                         "member_complete_read_and_crc_verified": True},
              "schema": schema.to_dict(), "files": files, "budgets": quotas,
              "seen_evidence": evidence, "seen_preparation_receipts": receipts,
              "seen_unique_ids": len(seen_ids), "seen_unique_communication_groups": len(seen_groups),
              "seen_unique_input_patterns": len(seen_patterns), "scan": scan,
              "sampling": {"group_pool": "sha256 compact JSON [route-confirmation-group-v1,seed,group], first64bit mod100: C<20,E>=20",
                           "row_selection": "independent bottom-k sha256 compact JSON [route-confirmation-row-v1,seed,source_sha256,1based_csv_record]",
                           "calibration_label_filter": "normal only", "evaluation_label_or_family_stratification": False,
                           "feature_pattern_exclusion": False, "role_priority": None},
              "selected_concentration": {role: lab._selected_concentration(frame)
                                         for role, frame in frames.items()},
              "selected_family_counts": {role: dict(sorted(Counter(frame[schema.family_column]).items()))
                                         for role, frame in frames.items()},
              "input_pattern_overlap": {**overlaps, "calibration_evaluation_unique_patterns":
                                         len(set(signatures["calibration"]) & set(signatures["evaluation"]))},
              "cross_role_group_overlap": 0}
    (root / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return report
