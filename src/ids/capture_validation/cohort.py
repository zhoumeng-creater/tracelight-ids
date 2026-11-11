"""Register the CSE-CIC-IDS2018-v3 temporal cohort from native rows and confirmed portscans."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
from ipaddress import IPv4Address
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv

from ids.context_cache import lab_data as lab
from ids.flow_history.evaluation import syn_cohort

SOURCE_SHA = "242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74"
SOURCE_ROWS = 20_115_529
SEED = 20261111
P_END, C_END, E_END = 1519844400000, 1519912800000, 1520035200000
WINDOW = 60000
QUOTAS = {"P_normal": 4096, "P_attack": 4096, "C_time": 10000, "E_time": 50000}
ROLES = ("P_time", "C_time", "E_time")
NATIVE = {
    "Benign": "Benign",
    "Bot": "Bot",
    "FTP-BruteForce": "BruteForce",
    "SSH-Bruteforce": "BruteForce",
    "DDOS_attack-HOIC": "DDoS",
    "DDOS_attack-LOIC-UDP": "DDoS",
    "DDoS_attacks-LOIC-HTTP": "DDoS",
    "DoS_attacks-Hulk": "DoS",
    "DoS_attacks-GoldenEye": "DoS",
    "DoS_attacks-SlowHTTPTest": "DoS",
    "DoS_attacks-Slowloris": "DoS",
    "Brute_Force_-Web": "Web Attacks",
    "Brute_Force_-XSS": "Web Attacks",
    "SQL_Injection": "Web Attacks",
    "Infilteration": "Infiltration",
}
SUBNET = int(IPv4Address("172.31.69.0"))
TARGETS = {1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
# Native millisecond origin. Check the author's predicate on the first packet, not the completion.
RULES = ((24, 1519829182000000000, 1519843140746247000), (13, 1519913388354333000, 1519933092182726000))
SRC, DST = "IPV4_SRC_ADDR", "IPV4_DST_ADDR"
START, END = "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
META = [
    "row_id",
    "role",
    "source_label",
    "source_family",
    "label",
    "attack_family",
    "truth_source",
    "flow_end_ms",
    "communication_group",
    "host_key",
    "end_day_utc",
]


def _mix(numbers):
    v = numbers.astype(np.uint64, copy=False) + np.uint64(0x9E3779B97F4A7C15)
    v = (v ^ (v >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    v = (v ^ (v >> 27)) * np.uint64(0x94D049BB133111EB)
    return v ^ (v >> 31)


def _ips(column):
    encoded = column.combine_chunks().dictionary_encode()
    addresses = np.array([int(IPv4Address(x)) for x in encoded.dictionary.to_pylist()], dtype=np.uint64)
    return addresses[encoded.indices.to_numpy()]


def _bucket(src, dst, sport, dport, proto):
    a = (src << 16) | sport.astype(np.uint64)
    b = (dst << 16) | dport.astype(np.uint64)
    return _mix(_mix(np.minimum(a, b)) ^ np.maximum(a, b) ^ _mix(proto.astype(np.uint64) + SEED)) % 100


def _scan_candidates(start, src, dst, sport, dport):
    strict, possible = np.zeros(len(start), dtype=bool), np.zeros(len(start), dtype=bool)
    for scanner, lower, upper in RULES:
        peers = [SUBNET + i for i in TARGETS - {scanner}]
        forward = (src == SUBNET + scanner) & np.isin(dst, peers) & (sport != 68)
        reverse = (dst == SUBNET + scanner) & np.isin(src, peers) & (dport != 68)
        # All nanoseconds in this reported millisecond satisfy the original label predicate.
        strict |= forward & (start >= -(-lower // 1000000)) & (start <= (upper - 999999) // 1000000)
        # Quarantine both boundary bins and the opposite direction, even for UQ-native Benign.
        possible |= (forward | reverse) & (start >= lower // 1000000) & (start <= upper // 1000000)
    return strict, possible


@contextmanager
def _source(path):
    path = Path(path)
    if path.suffix.casefold() == ".zip":
        with ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if name.split("/")[-1] == "NF-CICIDS2018-v3.csv"]
            if len(matches) != 1:
                raise ValueError("Expected one native NF-CICIDS2018-v3.csv in archive")
            with archive.open(matches[0]) as input_file:
                yield input_file
    else:
        with path.open("rb") as input_file:
            yield input_file


def _verified_ids(ids):
    a = np.load(ids, allow_pickle=False) if isinstance(ids, (Path, str)) else np.asarray(ids)
    if a.ndim != 1 or a.dtype != np.dtype("<u4") or (len(a) and (a[0] == 0 or (a[1:] <= a[:-1]).any())):
        raise ValueError(
            "Verified identities must be a strictly ascending, one-based, little-endian uint32 vector"
        )
    return a


def _retain(rows_by_stratum, name, mask, quota, priorities, data, record_ids, verified):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return
    previous = rows_by_stratum.get(name)
    if previous is not None and len(previous) == quota:
        indices = indices[priorities[indices] < previous["_priority"].iloc[-1]]
    if not len(indices):
        return
    if len(indices) > quota:
        indices = indices[np.argpartition(priorities[indices], quota - 1)[:quota]]
    frame = data.take(pa.array(indices)).to_pandas()
    frame["_source_number"] = record_ids[indices]
    frame["_priority"] = priorities[indices]
    frame["_verified"] = verified[indices]
    if previous is not None:
        frame = pd.concat([previous, frame], ignore_index=True)
    rows_by_stratum[name] = frame.nsmallest(quota, "_priority")


def _finalize(sketches, sha):
    parts = []
    for stratum, frame in sketches.items():
        f = frame.copy()
        f["role"] = "P_time" if stratum.startswith("P_") else stratum
        f["source_label"] = f.Label.astype(int)
        f["source_family"] = f.Attack
        f["label"] = np.where(f._verified, 1, f.source_label)
        f["attack_family"] = np.where(f._verified, "Scanning", f.source_family.map(NATIVE))
        f["truth_source"] = np.array(
            ["UQ_native", "DistriNet_unique_NMAP", "published_Nmap_event_packet_match"]
        )[f._verified.to_numpy(dtype=int)]
        f["row_id"] = sha + ":" + f._source_number.astype(str)
        f["flow_end_ms"] = f[END]
        f["end_day_utc"] = pd.to_datetime(f[END], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        hosts = {v: hashlib.sha256(v.encode()).hexdigest()[:16] for v in f[SRC].unique()}
        f["host_key"] = f[SRC].map(hosts)
        f["communication_group"] = [
            lab._communication_group_from_ids(s, d, int(sp), int(dp), int(pr))
            for s, d, sp, dp, pr in zip(f[SRC], f[DST], f.L4_SRC_PORT, f.L4_DST_PORT, f.PROTOCOL)
        ]
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=[*META, *lab.FEATURES])
    f = pd.concat(parts).sort_values(["role", "_source_number"], kind="stable")
    return f[[*META, *lab.FEATURES]].reset_index(drop=True)


SIGNATURE_FIELDS = (
    "PROTOCOL",
    "L4_DST_PORT",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
)


def signature_overlap(evaluation):
    """Count co-observed shapes among fixed E rows; this never selects targets or a threshold."""
    f = evaluation.loc[
        evaluation.attack_family.isin(["Scanning", "Benign"]),
        ["end_day_utc", "attack_family", *SIGNATURE_FIELDS],
    ]
    out = {"fields": list(SIGNATURE_FIELDS)}
    for name, keys in (
        ("all_days", list(SIGNATURE_FIELDS)),
        ("within_day", ["end_day_utc", *SIGNATURE_FIELDS]),
    ):
        by = f.groupby([*keys, "attack_family"], dropna=False).size().unstack("attack_family", fill_value=0)
        by = by.reindex(columns=["Scanning", "Benign"], fill_value=0)
        both = by.loc[(by.Scanning > 0) & (by.Benign > 0)]
        out[name] = {
            "matched_scan_rows": int(both.Scanning.sum()),
            "matched_normal_rows": int(both.Benign.sum()),
            "matched_keys": len(both),
            "distinct_signatures": len(
                both.index.to_frame(index=False)
                .drop(columns="end_day_utc", errors="ignore")
                .drop_duplicates()
            ),
            "n_scan": int(by.Scanning.sum()),
            "n_normal": int(by.Benign.sum()),
        }
    return out


def role_identity_sha256(targets):
    """SHA-256 of one-based native row number, final family, truth source; UTF-8 TSV with final LF."""
    identities = {}
    for role in ROLES:
        f = targets.loc[targets.role.eq(role), ["row_id", "attack_family", "truth_source"]].copy()
        f["number"] = f.row_id.str.rsplit(":", n=1).str[-1].astype("uint64")
        digest = hashlib.sha256()
        for number, family, origin in f.sort_values("number")[
            ["number", "attack_family", "truth_source"]
        ].itertuples(index=False, name=None):
            digest.update(f"{number}\t{family}\t{origin}\n".encode("utf-8"))
        identities[role] = digest.hexdigest()
    return identities


def _coverage(targets, quotas):
    groups, reports = {}, {}
    masks = {role: targets[targets.role.eq(role)] for role in ROLES}
    train_hosts = set(masks["P_time"].host_key)
    train_scanners = set(masks["P_time"].loc[masks["P_time"].attack_family.eq("Scanning"), "host_key"])
    for role, f in masks.items():
        groups[role] = set(f.communication_group)
        syn = (
            f.PROTOCOL.eq(6)
            & f.IN_PKTS.eq(1)
            & f.OUT_PKTS.eq(0)
            & f.TCP_FLAGS.eq(2)
            & f.CLIENT_TCP_FLAGS.eq(2)
            & f.SERVER_TCP_FLAGS.eq(0)
        )
        scan, normal = f.attack_family.eq("Scanning"), f.label.eq(0)
        exact = syn_cohort(f) if len(f) else np.zeros(0, dtype=bool)
        dates = {
            date: {str(k): int(n) for k, n in part.attack_family.value_counts().items()}
            for date, part in f.groupby("end_day_utc")
        }
        reports[role] = {
            "rows": len(f),
            "groups": len(groups[role]),
            "hosts": f.host_key.nunique(),
            "unseen_train_hosts": len(set(f.host_key) - train_hosts),
            "families": dict(f.attack_family.value_counts().items()),
            "native_families": dict(f.source_family.value_counts().items()),
            "daily_families": dates,
            "scan_source_hosts": dict(f.loc[scan, "host_key"].value_counts().items()),
            "scan_on_unseen_sources": int((scan & ~f.host_key.isin(train_hosts)).sum()),
            "scan_on_source_unseen_as_scanner": int((scan & ~f.host_key.isin(train_scanners)).sum()),
            "confirmed_scans_by_evidence": [
                {
                    "truth_source": origin,
                    "end_day_utc": day,
                    "host_key": host,
                    "native_family": family,
                    "rows": int(n),
                }
                for (origin, day, host, family), n in f.loc[scan]
                .groupby(["truth_source", "end_day_utc", "host_key", "source_family"])
                .size()
                .items()
            ],
            "scan_syn": int((scan & syn).sum()),
            "normal_syn": int((normal & syn).sum()),
            "scan_syn47": int((scan & exact).sum()),
            "normal_syn47": int((normal & exact).sum()),
            "min_end_ms": int(f.flow_end_ms.min()) if len(f) else None,
            "max_end_ms": int(f.flow_end_ms.max()) if len(f) else None,
        }
    p, c, e = (masks[key] for key in ROLES)
    if (
        targets.row_id.duplicated().any()
        or groups["P_time"] & groups["C_time"]
        or groups["E_time"] & (groups["P_time"] | groups["C_time"])
    ):
        raise ValueError("Target identity or communication group crosses roles")
    if not (
        p.flow_end_ms.lt(P_END).all()
        and c.flow_end_ms.between(P_END + WINDOW, C_END - 1).all()
        and e.flow_end_ms.between(C_END + WINDOW, E_END - 1).all()
        and c.label.eq(0).all()
    ):
        raise ValueError("Chronology or normal-only calibration violation")
    reports["E_time"]["seven_field_overlap"] = signature_overlap(e)
    checks = {
        "target_quotas": (
            int(p.label.eq(1).sum()) == quotas["P_attack"]
            and int(p.label.eq(0).sum()) == quotas["P_normal"]
            and len(c) == quotas["C_time"]
            and len(e) == quotas["E_time"]
        ),
        "train_scan_and_other_attack": "Scanning" in set(p.attack_family)
        and bool(set(p.attack_family) - {"Benign", "Scanning"}),
        "calibration_can_register_one_percent": len(c) >= 99,
        "evaluation_scan_other_attack_and_normal": {"Scanning", "Benign"} <= set(e.attack_family)
        and bool(set(e.attack_family) - {"Scanning", "Benign"}),
        "evaluation_has_similar_normal_syn": reports["E_time"]["normal_syn"] > 0,
    }
    return reports, checks


def prepare_targets(
    csv_path,
    confirmed_rows,
    *,
    packet_rows=None,
    source_sha256=SOURCE_SHA,
    expected_rows=SOURCE_ROWS,
    quotas=None,
    block_size=16 << 20,
    threads=None,
):
    """Return archived row identities/48 inputs and a metadata-only qualifying report before any model fit."""
    ids = _verified_ids(confirmed_rows)
    packets = np.empty(0, dtype="<u4") if packet_rows is None else _verified_ids(packet_rows)
    if np.intersect1d(ids, packets, assume_unique=True).size:
        raise ValueError("Overlapping corrected-flow and original-packet evidence identities")
    limits = dict(QUOTAS if quotas is None else quotas)
    if set(limits) != set(QUOTAS) or any(type(n) is not int or n < 1 for n in limits.values()):
        raise ValueError("Four positive integer quotas are required")
    types = {key: pa.float64() for key in lab.FEATURES}
    types.update(
        {
            SRC: pa.string(),
            DST: pa.string(),
            START: pa.int64(),
            END: pa.int64(),
            "Label": pa.int8(),
            "Attack": pa.string(),
        }
    )
    for k in ("L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"):
        types[k] = pa.int64()
    opts = arrow_csv.ConvertOptions(include_columns=list(types), column_types=types)
    sha = hashlib.sha256()
    rows = 0
    checked = [0, 0]
    original_truth_counts = Counter()
    excluded = Counter()
    eligible = Counter()
    sketches = {}
    prior = pa.cpu_count()
    try:
        if threads is not None:
            if type(threads) is not int or threads < 1:
                raise ValueError("Expected positive number of Arrow parser threads")
            pa.set_cpu_count(threads)
        with _source(csv_path) as source:
            header = source.readline()
            sha.update(header)
            names = next(csv.reader([header.decode("utf-8-sig")]))
            if len(names) != len(set(names)) or not set(types) <= set(names):
                raise ValueError("Native NF-CSE-CIC-IDS2018-v3 source header mismatch")
            read_opts = arrow_csv.ReadOptions(column_names=names, use_threads=True)
            while chunk := source.read(block_size):
                if not chunk.endswith(b"\n"):
                    chunk += source.readline()
                sha.update(chunk)
                data = arrow_csv.read_csv(
                    pa.BufferReader(chunk), read_options=read_opts, convert_options=opts
                )
                n = len(data)
                rec = np.arange(rows + 1, rows + n + 1, dtype=np.uint64)

                def get(key):
                    return data[key].to_numpy()

                s, d = _ips(data[SRC]), _ips(data[DST])
                sp, dp, proto = (get(k) for k in ("L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"))
                start, end, labels = get(START), get(END), get("Label")
                if (
                    np.any(start > end)
                    or np.any(sp < 0)
                    or np.any(sp > 65535)
                    or np.any(dp < 0)
                    or np.any(dp > 65535)
                    or np.any(proto < 0)
                    or np.any(proto > 255)
                    or not np.isin(labels, [0, 1]).all()
                ):
                    raise ValueError("Invalid native timestamp, endpoint or binary label")
                names_col = data["Attack"].combine_chunks().dictionary_encode()
                categories = names_col.dictionary.to_pylist()
                codes = names_col.indices.to_numpy()
                if names_col.null_count or set(categories) - set(NATIVE):
                    raise ValueError("Unknown native attack family")
                normal = labels == 0
                if not np.array_equal(normal, np.array([v == "Benign" for v in categories])[codes]):
                    raise ValueError("Native binary and family labels disagree")
                infiltration = np.array([v == "Infilteration" for v in categories])[codes]
                strict, possible = _scan_candidates(start, s, d, sp, dp)
                origin = np.zeros(n, dtype=np.uint8)
                for index, manifest_ids in enumerate((ids, packets)):
                    lo, hi = np.searchsorted(manifest_ids, [rows + 1, rows + n + 1])
                    offsets = manifest_ids[lo:hi].astype(np.int64) - rows - 1
                    if not strict[offsets].all():
                        raise ValueError(
                            "Verified identity does not satisfy the directed reference rule in the bound source"
                        )
                    if index == 1 and len(offsets):
                        syn = (
                            (proto[offsets] == 6)
                            & (get("IN_PKTS")[offsets] == 1)
                            & (get("OUT_PKTS")[offsets] == 0)
                        )
                        syn &= (
                            (get("TCP_FLAGS")[offsets] == 2)
                            & (get("CLIENT_TCP_FLAGS")[offsets] == 2)
                            & (get("SERVER_TCP_FLAGS")[offsets] == 0)
                        )
                        if not syn.all():
                            raise ValueError(
                                "Packet reference identity is not a native single unanswered SYN"
                            )
                    if len(offsets):
                        dates = pd.to_datetime(end[offsets], unit="ms", utc=True).strftime("%Y-%m-%d")
                        unique_host = {
                            ip: hashlib.sha256(str(IPv4Address(int(ip))).encode()).hexdigest()[:16]
                            for ip in np.unique(s[offsets])
                        }
                        source_name = (
                            "DistriNet_unique_NMAP" if index == 0 else "published_Nmap_event_packet_match"
                        )
                        original_truth_counts.update(
                            (source_name, day, unique_host[srcip], categories[family_code])
                            for day, srcip, family_code in zip(dates, s[offsets], codes[offsets])
                        )
                    origin[offsets] = index + 1
                    checked[index] += int(hi - lo)
                verified = origin > 0
                valid = verified | (~possible & ~infiltration)
                excluded.update(
                    {
                        "uncertain_scan_rows": int((possible & ~verified).sum()),
                        "remaining_native_infiltration": int((~possible & infiltration).sum()),
                        "verified_original_native_benign": int((verified & normal).sum()),
                    }
                )
                is_normal = normal & ~verified
                bucket = _bucket(s, d, sp, dp, proto)
                priority = _mix(rec ^ np.uint64(SEED))
                p = valid & (end < P_END) & (bucket < 50)
                masks = {
                    "P_normal": p & is_normal,
                    "P_attack": p & ~is_normal,
                    "C_time": valid
                    & is_normal
                    & (end >= P_END + WINDOW)
                    & (end < C_END)
                    & (bucket >= 50)
                    & (bucket < 70),
                    "E_time": valid & (end >= C_END + WINDOW) & (end < E_END) & (bucket >= 70),
                }
                for key, mask in masks.items():
                    eligible[key] += int(mask.sum())
                    _retain(sketches, key, mask, limits[key], priority, data, rec, origin)
                rows += n
    finally:
        pa.set_cpu_count(prior)
    if sha.hexdigest() != source_sha256 or rows != expected_rows or checked != [len(ids), len(packets)]:
        raise ValueError("Native source hash, row total or confirmed identity coverage mismatch")
    targets = _finalize(sketches, source_sha256)
    reports, checks = _coverage(targets, limits)
    return targets, {
        "source_sha256": source_sha256,
        "source_rows": rows,
        "confirmed_rows": checked[0],
        "packet_confirmed_rows": checked[1],
        "all_confirmed_rows": sum(checked),
        "packet_uint32_payload_sha256": hashlib.sha256(packets.tobytes()).hexdigest(),
        "source_truth_counts_before_sampling": [
            {
                "truth_source": origin,
                "end_day_utc": day,
                "host_key": host,
                "native_family": family,
                "rows": int(count),
            }
            for (origin, day, host, family), count in sorted(original_truth_counts.items())
        ],
        "confirmed_uint32_payload_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "excluded": dict(excluded),
        "eligible_before_priority": dict(eligible),
        "quotas": limits,
        "seed": SEED,
        "p_end": P_END,
        "c_end": C_END,
        "e_end": E_END,
        "opening_warmup_ms": WINDOW,
        "cross_role_group_overlap": 0,
        "rows_by_role": reports,
        "role_identity_sha256": role_identity_sha256(targets),
        "identity_encoding": "One-based native record number, final attack_family, truth_source; numerically row-sorted, UTF-8 TSV, LF after every row, no header.",
        "eligible_for_training": all(checks.values()),
        "coverage_checks": checks,
        "subgroup_evaluable": {
            "one_unanswered_syn": bool(reports["E_time"]["scan_syn"] and reports["E_time"]["normal_syn"]),
            "original_47_field_signature": bool(
                reports["E_time"]["scan_syn47"] and reports["E_time"]["normal_syn47"]
            ),
        },
        "evaluation_selection": "Within the adjudicable pool, E sampled by source row priority without label stratification.",
        "truth_scope": "Scanning from uniquely matched corrected NMAP flows or verified native SYN packets inside published Nmap events; non-scan positives and eligible normal from native UQ. Others quarantined.",
    }
