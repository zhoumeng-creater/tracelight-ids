"""Coverage only: independently dated NF-CSE-CIC-IDS2018-v3 and corrected portscan rules."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv

from audit_paths import RESULTS as OUT, UQ_ZIP as FILE, EXPECTED_UQ_ZIP_SHA256
from ids.context_cache.lab_data import FEATURES
from ids.flow_history.evaluation import syn_cohort


DAY_MS = 86400000
RULES = {
    "2018-02-28": (
        "172.31.69.24",
        [1, 10, 11, 12, 13, 14, 16, 17, 19, 20, 23, 4, 5, 6, 8, 9, 7, 22, 15, 21, 18],
        1519829182000000000,
        1519843140746247000,
    ),
    "2018-03-01": (
        "172.31.69.13",
        [1, 11, 12, 16, 8, 9, 10, 14, 4, 5, 6, 17, 20, 23, 24, 19, 7, 15, 18, 22, 21],
        1519913388354333000,
        1519933092182726000,
    ),
}
COUNT = Counter()
HOST = defaultdict(set)
SAMPLES = {}
CAMPAIGN_EXACT47 = []
PROBE = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_PKTS",
    "OUT_PKTS",
    "IN_BYTES",
    "OUT_BYTES",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "Label",
    "Attack",
]
N_PER_STRATUM = 512


class DigestReader:
    def __init__(self, stream):
        self.stream = stream
        self.h = hashlib.sha256()
        self.s = hashlib.sha1()
        self.closed = False

    def read(self, n):
        b = self.stream.read(n)
        self.h.update(b)
        self.s.update(b)
        return b


def sample(group, mask, name, day):
    if not mask.any():
        return
    sub = group.loc[mask, PROBE].copy()
    sub["row_position_zero_based"] = sub.index
    sub["day"] = day
    sub["stratum"] = name
    # Smallest pseudorandom keys by unique original CSV row; independent of scores.
    key = pd.util.hash_pandas_object(sub["row_position_zero_based"], index=False).to_numpy()
    if len(key) > N_PER_STRATUM:
        pick = np.argpartition(key, N_PER_STRATUM - 1)[:N_PER_STRATUM]
        sub = sub.iloc[pick].copy()
        key = key[pick]
    sub["selection_key"] = key
    prev = SAMPLES.get((day, name))
    SAMPLES[(day, name)] = (sub if prev is None else pd.concat([prev, sub])).nsmallest(
        N_PER_STRATUM, "selection_key"
    )


def observe(group, day):
    coarse = (
        group.PROTOCOL.eq(6)
        & group.IN_PKTS.eq(1)
        & group.OUT_PKTS.eq(0)
        & group.CLIENT_TCP_FLAGS.eq(2)
        & group.TCP_FLAGS.eq(2)
    )
    exact = pd.Series(False, index=group.index)
    pre = coarse & group.IN_BYTES.eq(44) & group.L4_DST_PORT.eq(443) & group.L7_PROTO.eq(91)
    if pre.any():
        exact.loc[pre] = syn_cohort(group.loc[pre])
    for fam, n in group.Attack.value_counts().items():
        COUNT[day, "all", str(fam)] += int(n)
    for name, mask in (("single_unanswered_syn", coarse), ("exact_r19_syn47", exact)):
        for fam, n in group.loc[mask, "Attack"].value_counts().items():
            COUNT[day, name, str(fam)] += int(n)
    for fam, g in group.groupby("Attack"):
        HOST[day, str(fam)].update(g.IPV4_SRC_ADDR.unique())
    if day not in RULES:
        return
    source, suffix, lower, upper = RULES[day]
    dst = {f"172.31.69.{i}" for i in suffix}
    # Full millisecond bin lies within author's flow-start window; end boundary measured separately.
    t = group.FLOW_START_MILLISECONDS
    window = t.ge(-(-lower // 1000000)) & (t * 1000000 + 999999).le(upper)
    source_side = group.IPV4_SRC_ADDR.eq(source)
    directed = source_side & group.IPV4_DST_ADDR.isin(dst)
    fwd = directed & group.L4_SRC_PORT.ne(68) & window
    reverse = (
        group.IPV4_DST_ADDR.eq(source) & group.IPV4_SRC_ADDR.isin(dst) & group.L4_DST_PORT.ne(68) & window
    )
    rough = group.Attack.str.lower().str.startswith("infil")
    benign = group.Label.eq(0)
    ambiguous_time = (t * 1000000).le(upper) & (t * 1000000 + 999999).ge(lower) & ~window
    masks = {
        "source_window_targets_excl_dhcp": fwd,
        "reverse_window_targets": reverse,
        "straddles_end_with_start_match": fwd & (group.FLOW_END_MILLISECONDS * 1000000).gt(upper),
        "ambiguous_truncated_start_boundary": ambiguous_time & directed & group.L4_SRC_PORT.ne(68),
        "source_window_dhcp": directed & window & group.L4_SRC_PORT.eq(68),
        "source_other_dest_window": source_side & ~directed & window,
        "source_target_outside_window": directed & ~window,
        "uq_infil_outside_candidate": rough & ~fwd,
        "candidate_exact_syn47": fwd & exact,
        "candidate_single_unanswered_syn": fwd & coarse,
    }
    CAMPAIGN_EXACT47.extend(int(i) + 1 for i in group.index[exact & fwd])
    for name, mask in masks.items():
        for fam, n in group.loc[mask, "Attack"].value_counts().items():
            COUNT[day, name, str(fam)] += int(n)
    for name, mask in (
        ("rule_uq_benign", fwd & benign),
        ("rule_uq_attack", fwd & ~benign),
        ("reverse_direction", reverse),
        ("same_source_normal_outside_rule", source_side & ~fwd & benign),
        ("other_benign", ~source_side & benign),
        ("normal_single_syn", coarse & benign & ~fwd),
        ("other_uq_infiltration", rough & ~fwd),
        ("same_source_normal_syn_outside_rule", source_side & ~fwd & benign & coarse),
    ):
        sample(group, mask, name, day)


def main():
    rows = 0
    with FILE.open("rb") as fs:
        zip_sha = hashlib.file_digest(fs, "sha256").hexdigest()
    if zip_sha != EXPECTED_UQ_ZIP_SHA256:
        raise ValueError("Official UQ ZIP SHA-256 differs from the registered release")
    with ZipFile(FILE) as z:
        m = next(i for i in z.infolist() if i.filename.endswith("/NF-CICIDS2018-v3.csv"))
        published = z.read(next(n for n in z.namelist() if n.endswith("/manifest-sha1.txt"))).decode()
        with z.open(m) as raw:
            rd = DigestReader(raw)
            types = {k: pa.float64() for k in FEATURES}
            types.update(
                {
                    "FLOW_START_MILLISECONDS": pa.int64(),
                    "FLOW_END_MILLISECONDS": pa.int64(),
                    "IPV4_SRC_ADDR": pa.string(),
                    "IPV4_DST_ADDR": pa.string(),
                    "Attack": pa.string(),
                    "Label": pa.int8(),
                }
            )
            reader = pcsv.open_csv(
                rd,
                read_options=pcsv.ReadOptions(block_size=1 << 24),
                convert_options=pcsv.ConvertOptions(column_types=types),
            )
            head = reader.schema.names
            for batch in reader:
                df = batch.to_pandas()
                df.index = pd.RangeIndex(rows, rows + len(df))
                rows += len(df)
                assert (df.FLOW_START_MILLISECONDS <= df.FLOW_END_MILLISECONDS).all()
                assert df.Attack.notna().all() and df.Label.isin([0, 1]).all()
                for days, group in df.groupby(df.FLOW_END_MILLISECONDS // DAY_MS, sort=False):
                    day = (date(1970, 1, 1) + timedelta(days=int(days))).isoformat()
                    observe(group, day)
            assert rd.s.hexdigest() in published
        assert rows == 20115529
        probes = pd.concat(SAMPLES.values(), ignore_index=True).sort_values(
            ["day", "stratum", "selection_key"]
        )
        for k in (
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "IN_PKTS",
            "OUT_PKTS",
            "IN_BYTES",
            "OUT_BYTES",
            "TCP_FLAGS",
            "CLIENT_TCP_FLAGS",
            "SERVER_TCP_FLAGS",
        ):
            probes[k] = probes[k].astype("int64")
        probes.to_parquet(OUT / "cic2018_nf_probe.parquet", index=False)
        result = {
            "scope": "Full UQ public file, coverage only. Classifier fit and scores never read. No new deployment labels assigned.",
            "zip_bytes": FILE.stat().st_size,
            "zip_sha256": zip_sha,
            "csv_bytes": m.file_size,
            "csv_sha256": rd.h.hexdigest(),
            "csv_sha1_verified_against_bagit": rd.s.hexdigest(),
            "zip_member": m.filename,
            "rows": rows,
            "columns": head,
            "round19_features_present": len(set(FEATURES) & set(head)),
            "publisher_manifest": published,
            "rules": RULES,
            "rule_semantics": "UTC completion-date grouping; strict first-packet millisecond-bin inclusion in author windows, not end-of-flow filtering; source port 68 excluded.",
            "counts": {},
            "distinct_src_hosts": {},
            "probe_rows": len(probes),
            "probe_sampling": "up to 512 NF rows per day/stratum by bottom hash of zero-based immutable row position; strata can overlap; no selection via predictions.",
        }
        for (day, k, fam), n in sorted(COUNT.items()):
            result["counts"].setdefault(day, {}).setdefault(k, {})[fam] = n
        for (day, fam), v in sorted(HOST.items()):
            result["distinct_src_hosts"].setdefault(day, {})[fam] = len(v)
        (OUT / "cic2018_nf_coverage.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        np.save(
            OUT / "cic2018_syn47_rows.npy",
            np.array(sorted(CAMPAIGN_EXACT47), dtype="<u4"),
            allow_pickle=False,
        )
        print(rows, len(probes), zip_sha)
        print({day: result["counts"][day].get("source_window_targets_excl_dhcp") for day in RULES})


if __name__ == "__main__":
    main()
