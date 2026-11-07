"""Bounded-memory preparation of the official 2025 NF-ToN-IoT-v3 release.

This is a development comparison split, not a temporal/generalization test.
The legacy profile gives earlier roles priority over exact-input collisions.
The v2 profile samples each fixed group pool independently and reports shared
inputs without discarding their flow frequency. No model is loaded here.
"""

from __future__ import annotations

from collections import Counter
import csv
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import heapq
import ipaddress
import json
import math
from pathlib import Path

from ids.common import sha256_file, write_json
from .csv_workflow import audit_roles
from .preprocessing import CSVSchema, read_csv


DATASET = "NF-ToN-IoT-v3"
SOURCE_SHA1 = "24854ca3072ab7c2ade9ebfb202345a666a94cfe"
SOURCE_BYTES = 5302886266
SEED = 20261107
LEGACY_SAMPLING = "legacy_sequential_v1"
FLOW_SAMPLING = "group_disjoint_flow_v2"
# L7's bound is this project's accepted input domain, not a claim about the
# complete set of identifiers defined by the dataset publisher.
IDENTIFIER_DOMAINS = (
    ("L4_SRC_PORT", 65535), ("L4_DST_PORT", 65535), ("PROTOCOL", 255), ("L7_PROTO", 65535),
)
RAW_FEATURES = (
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE",
    "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
)
EXCLUDED_FEATURES = (
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "DNS_QUERY_ID",
)
FEATURES = tuple(name for name in RAW_FEATURES if name not in EXCLUDED_FEATURES)
CATEGORICAL = ("L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO")
SCHEMA = {
    "features": list(FEATURES),
    "numeric": [name for name in FEATURES if name not in CATEGORICAL],
    "categorical": list(CATEGORICAL),
    "label_column": "label", "id_column": "sample_id",
    "group_column": "communication_group", "family_column": "attack_family", "label_mapping": None,
}
FAMILIES = {v.casefold(): v for v in (
    "Benign", "Backdoor", "DoS", "DDoS", "Injection", "MITM", "Password", "Ransomware", "Scanning", "XSS",
)}
ROLES = ("support", "calibration", "evaluation")
FILENAMES = dict(zip(ROLES, ("support.csv", "normal_calibration.csv", "development_evaluation.csv")))


def _digest(parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=4096)
def _ipv4(value):
    return str(ipaddress.IPv4Address(value.strip()))


def _integer(value, name, maximum):
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or not 0 <= number <= maximum:
        raise ValueError(f"invalid {name}")
    return int(number)


@lru_cache(maxsize=32768)
def _communication_group(src_ip, dst_ip, src_port, dst_port, protocol):
    """Unordered endpoint pair plus protocol, with no invented session timeout."""
    endpoints = sorted(((_ipv4(src_ip), _integer(src_port, "source port", 65535)),
                        (_ipv4(dst_ip), _integer(dst_port, "destination port", 65535))))
    return _digest([endpoints, _integer(protocol, "IP protocol", 255)])


@lru_cache(maxsize=32768)
def _communication_group_from_ids(src_ip, dst_ip, src_port, dst_port, protocol):
    """Use already validated identifiers; keep the legacy group digest."""
    endpoints = sorted(((_ipv4(src_ip), src_port), (_ipv4(dst_ip), dst_port)))
    return _digest([endpoints, protocol])


@lru_cache(maxsize=65536)
def _source_identifier(raw, maximum):
    """Exact integer parsing, bounded cache for repeated source spellings."""
    value = raw.strip()
    if not value:
        raise ValueError("missing identifier")
    try:
        if value.isascii() and value.isdecimal():
            number = int(value)
        else:
            number = Decimal(value)
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("identifier is not a finite integer")
        if not 0 <= number <= maximum:
            raise ValueError("identifier outside accepted domain")
        return int(number)
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("identifier is not a finite integer") from exc


@lru_cache(maxsize=32768)
def _group_role(group, seed):
    bucket = int(_digest(["communication-role-v1", seed, group])[:16], 16) % 100
    return "support" if bucket < 50 else "calibration" if bucket < 70 else "evaluation"


def _priority(row_number, source_sha256, seed):
    return int(_digest(["row-priority-v1", seed, source_sha256, row_number]), 16)


def _source_hashes(path):
    sha1, sha256, size = hashlib.sha1(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha1.update(block)
            sha256.update(block)
            size += len(block)
    return {"sha1": sha1.hexdigest(), "sha256": sha256.hexdigest(), "bytes": size}


def _features(row, indices):
    values, signature_values = [], []
    for name in FEATURES:
        raw = row[indices[name]].strip()
        if not raw:
            values.append("")
            signature_values.append(None)
            continue
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric feature {name}")
        number = 0.0 if number == 0 else number
        rendered = format(number, ".17g")
        values.append(rendered)
        signature_values.append(rendered if name in CATEGORICAL else number.hex())
    return values, _digest(signature_values)


def _scan(path, role, quotas, forbidden, source_sha256, seed, *, validate_identifiers=False):
    heaps = {key: [] for key in quotas}
    counters = Counter()
    eligible_families = Counter()
    identifier_ranges = {name: [None, None] for name, _ in IDENTIFIER_DOMAINS}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError("source CSV is empty") from exc
        header = [name.strip() for name in raw_header]
        if len(header) != len(set(header)):
            raise ValueError("duplicate source CSV headers after stripping whitespace")
        required = {*RAW_FEATURES, "Label", "Attack"}
        if not required <= set(header):
            raise ValueError(f"official CSV header mismatch; missing {sorted(required - set(header))}")
        indices = {name: i for i, name in enumerate(header)}
        identifier_columns = [(name, indices[name], maximum) for name, maximum in IDENTIFIER_DOMAINS]
        for row_number, row in enumerate(reader, 1):
            counters["source_rows_scanned"] += 1
            if len(row) != len(header):
                raise ValueError(f"wrong source field count at row {row_number}")
            if validate_identifiers:
                identifiers = []
                for name, index, maximum in identifier_columns:
                    try:
                        value = _source_identifier(row[index], maximum)
                    except ValueError as exc:
                        raise ValueError(f"invalid identifier {name} at source row {row_number}: {exc}") from exc
                    identifiers.append(value)
                    bounds = identifier_ranges[name]
                    if bounds[0] is None or value < bounds[0]:
                        bounds[0] = value
                    if bounds[1] is None or value > bounds[1]:
                        bounds[1] = value
            label = _integer(row[indices["Label"]], "binary label", 1)
            family = FAMILIES.get(row[indices["Attack"]].strip().casefold())
            if family is None or (family == "Benign") != (label == 0):
                raise ValueError(f"binary/family label disagreement or unknown family at row {row_number}")
            if validate_identifiers:
                group = _communication_group_from_ids(
                    row[indices["IPV4_SRC_ADDR"]], row[indices["IPV4_DST_ADDR"]], *identifiers[:3],
                )
            else:
                group = _communication_group(*(row[indices[name]] for name in (
                    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL")))
            if _group_role(group, seed) != role:
                continue
            counters["assigned_group_rows"] += 1
            if role == "calibration" and label != 0:
                counters["non_normal_rows_excluded_by_calibration_contract"] += 1
                continue
            counters["label_eligible_rows"] += 1
            eligible_families[family] += 1
            key = label if role == "support" else "all"
            heap, quota = heaps[key], quotas[key]
            priority = _priority(row_number, source_sha256, seed)
            # Rows outside the current bottom-k cannot enter the final sample.
            # Avoid materializing 48 numeric values for those rows.
            if len(heap) == quota and (priority, row_number) >= (-heap[0][0], -heap[0][1]):
                counters["noncompetitive_priority_rows"] += 1
                continue
            values, signature = _features(row, indices)
            if signature in forbidden:
                counters["cross_role_exact_feature_candidates_excluded"] += 1
                continue
            output_row = values + [str(label), f"{source_sha256}:{row_number}", group, family]
            item = (-priority, -row_number, signature, output_row)
            if len(heap) < quota:
                heapq.heappush(heap, item)
            else:
                heapq.heapreplace(heap, item)
        for key, quota in quotas.items():
            if len(heaps[key]) != quota:
                raise ValueError(f"insufficient eligible {role} rows for {key}: {len(heaps[key])} < {quota}; "
                                 "the frozen seed and quotas were not changed")
    chosen = sorted((item for heap in heaps.values() for item in heap), key=lambda item: (-item[0], -item[1]))
    report = {
        **{name: int(counters[name]) for name in (
            "source_rows_scanned", "assigned_group_rows", "label_eligible_rows",
            "non_normal_rows_excluded_by_calibration_contract", "noncompetitive_priority_rows",
            "cross_role_exact_feature_candidates_excluded",
        )},
        "eligible_family_counts_before_exact_feature_exclusion": dict(sorted(eligible_families.items())),
        "selected_family_counts": dict(sorted(Counter(item[3][-1] for item in chosen).items())),
        "source_header": raw_header, "ignored_source_columns": sorted(set(header) - required),
    }
    if validate_identifiers:
        report["identifier_validation"] = {
            "scope": "all source rows before role, label-quota or bottom-k filtering",
            "rows_validated": int(counters["source_rows_scanned"]),
            "fields": {name: {"minimum": identifier_ranges[name][0], "maximum": identifier_ranges[name][1],
                              "accepted_minimum": 0, "accepted_maximum": maximum,
                              "domain_basis": "project input contract" if name == "L7_PROTO" else "wire field width"}
                       for name, maximum in IDENTIFIER_DOMAINS},
        }
    return chosen, report


def _selected_concentration(frame):
    """Describe selected flow frequency; do not reinterpret it as sample ESS."""
    group_sizes = Counter(frame["communication_group"].astype(str))
    sizes = sorted(group_sizes.values(), reverse=True)
    return {"rows": len(frame), "unique_communication_groups": len(group_sizes),
            "largest_group_rows": sizes[0], "largest_group_fraction": sizes[0] / len(frame),
            "top_five_group_rows": sum(sizes[:5]), "top_five_group_fraction": sum(sizes[:5]) / len(frame),
            "scope": "selected flow counts; not independent sample count or inferential effective sample size"}


def prepare_lab_data(raw_csv, output, *, data_kind="development", support_per_class=4096,
                     calibration_rows=5000, evaluation_rows=20000, seed=SEED,
                     sampling_profile=LEGACY_SAMPLING):
    """Write the fixed source's P/C/E CSVs and an auditable preparation report.

    Development uses only the frozen official source and default budgets.
    Smaller budgets and nonofficial input are available only as synthetic tests.
    Evaluation bottom-k sampling never conditions its quota on a label/family.
    """
    if data_kind not in {"development", "synthetic"}:
        raise ValueError("data_kind must be development or synthetic; final test is not accepted")
    if sampling_profile not in {LEGACY_SAMPLING, FLOW_SAMPLING}:
        raise ValueError("unknown sampling profile")
    budgets = (support_per_class, calibration_rows, evaluation_rows)
    if any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError("sample budgets must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit nonnegative integer")
    if data_kind == "development" and (budgets != (4096, 5000, 20000) or seed != SEED):
        raise ValueError("development preparation requires the frozen first-run budgets and seed")
    raw_csv, output = Path(raw_csv).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = _source_hashes(raw_csv)
    if data_kind == "development" and (source["sha1"] != SOURCE_SHA1 or source["bytes"] != SOURCE_BYTES):
        raise ValueError("source differs from the official NF-ToN-IoT-v3 BagIt SHA-1/byte count")
    source.update({"path": str(raw_csv), "dataset": DATASET,
                   "expected_official_sha1": SOURCE_SHA1, "official_release_verified": data_kind == "development"})
    forbidden, selected, scans = set(), {}, {}
    quotas = {"support": {0: support_per_class, 1: support_per_class},
              "calibration": {"all": calibration_rows}, "evaluation": {"all": evaluation_rows}}
    for role in ROLES:
        selected[role], scans[role] = _scan(
            raw_csv, role, quotas[role], forbidden, source["sha256"], seed,
            validate_identifiers=sampling_profile == FLOW_SAMPLING,
        )
        if sampling_profile == LEGACY_SAMPLING:
            forbidden.update(item[2] for item in selected[role])
    # Only the bounded selected rows enter pandas/the standard role auditor.
    output.mkdir(parents=True)
    files, frames = {}, {}
    columns = [*FEATURES, "label", "sample_id", "communication_group", "attack_family"]
    schema = CSVSchema.from_dict(SCHEMA)
    for role in ROLES:
        path = output / FILENAMES[role]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(item[3] for item in selected[role])
        files[role] = {"path": str(path), "rows": len(selected[role]), "sha256": sha256_file(path)}
        frames[role] = read_csv(path, schema)
    audit = (audit_roles(frames, schema, exact_feature_policy="report")
             if sampling_profile == FLOW_SAMPLING else audit_roles(frames, schema))
    report = {
        "format": "tracelight-nf-ton-v3-preparation-v1", "data_kind": data_kind,
        "source": source, "seed": seed, "schema": schema.to_dict(), "files": files,
        "split": {"group": "unordered (IPv4,port) endpoint pair plus IP protocol; no time/session claim",
                  "group_role": "sha256 JSON [communication-role-v1,seed,group], first64bit mod100: P<50,C<70,E<100",
                  "sampling": "lowest sha256 JSON [row-priority-v1,seed,source_sha256,1based_csv_record] per quota",
                  "role_priority": list(ROLES),
                  "evaluation": "no label/family stratification; actual selected composition is reported",
                  "cross_role_duplicate_policy": "exclude candidate exact inputs already selected by earlier roles; refill",
                  "duplicate_count_scope": "competitive bottom-k candidates only, not all source duplicates",
                  "within_role_duplicates": "retained", "data_scans": 3,
                  "metadata_only_fields": list(EXCLUDED_FEATURES)},
        "budgets": {"support_per_class": support_per_class, "calibration_rows": calibration_rows,
                    "evaluation_rows": evaluation_rows,
                    "slots": 512 if data_kind == "development" else None, "alpha": .01},
        "scans": scans, "audit": audit,
        "scope": "new-source development comparison; no temporal or cross-source generalization claim",
    }
    if sampling_profile == FLOW_SAMPLING:
        report["format"] = "tracelight-nf-ton-v3-preparation-v2"
        report.update({"sampling_profile": FLOW_SAMPLING, "risk_unit": "flow", "exact_feature_policy": "report"})
        report["split"].update({
            "sampling_profile": FLOW_SAMPLING, "risk_unit": "flow", "exact_feature_policy": "report",
            "role_priority": None,
            "cross_role_duplicate_policy": "retain equal inputs across distinct groups; reject shared IDs or groups",
            "duplicate_count_scope": "selected rows; no feature-based candidate exclusion",
            "within_role_duplicates": "retained at their sampled flow frequency; no per-group cap",
            "role_selection": "independent bottom-k within each fixed group-role pool",
        })
        report["selected_concentration"] = {role: _selected_concentration(frames[role]) for role in ROLES}
        report["scope"] = ("flow-weighted development comparison on fixed communication-group pools; "
                           "no IID, exchangeability, temporal or population FPR guarantee")
    write_json(output / "schema.json", schema.to_dict())
    write_json(output / "preparation.json", report)
    return report
