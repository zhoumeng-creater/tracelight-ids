"""Bind the opened development data and sample new forward-time targets."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv

from ids.common import read_json, sha256_file, write_json
from ids.context_cache import lab_data as lab


SOURCE_SHA = "53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78"
ARCHIVE_SHA = "b3b6256e970a8986d87716edea4bfd436d68f04058bf5c26cf67e6d839c83698"
PROTOCOL_SHA = "3b2706b764df1f7b1b0c3ab5229b4173cd5e92db4f0b9c2ea029bd9d10bbbdbf"
PREPARATION_SHA = "dd0e87e36104eb8851ff78cd4c548cb12a1654a621a1985a775ab25b6d616d44"
SOURCE_ROWS = 27_520_260
SEED = 20261111
WINDOW_MS = 60_000
QUOTAS = {"P_normal": 4096, "P_attack": 4096, "C_normal": 10000, "E_all": 50000}
STRATA = (("P_time", "P_normal", 0), ("P_time", "P_attack", 1),
          ("C_time", "C_normal", 0), ("E_time", "E_all", None))
OLD_NAMES = {"support": "P", "calibration": "C", "evaluation": "E"}
FIELDS = ["row_id", *lab.FEATURES, "label", "attack_family", "communication_group"]
META = ["FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT",
        "PROTOCOL", "Label"]


def _utc(day, hour=0):
    return int(datetime(2019, 4, day, hour, tzinfo=timezone.utc).timestamp()) * 1000


INTERVALS = {"P_time": (_utc(23), _utc(23, 16)), "C_time": (_utc(23, 16), _utc(24)),
             "E_time": (_utc(24), _utc(30))}


def _checked(path, expected):
    path = Path(path)
    if sha256_file(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return path


def _inputs(original_run):
    run = Path(original_run)
    protocol = read_json(_checked(run / "protocol.json", PROTOCOL_SHA))
    prepared = read_json(_checked(run / "confirmation" / "prepared" / "preparation.json", PREPARATION_SHA))
    return protocol, prepared


def resolve_source(original_run, source=None, *, work=None):
    """Return the official plain CSV. Atomically expand the frozen ZIP when needed."""
    protocol, prepared = _inputs(original_run)
    if source is None:
        item = protocol["artifacts"]["preparation"]
        csv_path = Path(read_json(_checked(item["path"], item["sha256"]))["source"]["path"])
        source = csv_path if csv_path.is_file() else prepared["source"]["archive_path"]
    source = Path(source).resolve()
    if source.suffix.lower() != ".zip":
        return _checked(source, SOURCE_SHA)
    _checked(source, ARCHIVE_SHA)
    cache = Path(work).resolve() if work is not None else source.parent
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / "NF-ToN-IoT-v3.csv"
    if destination.exists():
        return _checked(destination, SOURCE_SHA)
    with zipfile.ZipFile(source) as archive:
        member = archive.getinfo(prepared["source"]["member"])
        if member.file_size != prepared["source"]["bytes"]:
            raise ValueError("Source ZIP member has unexpected size")
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            digest = hashlib.sha256()
            with temporary.open("wb") as output, archive.open(member) as reader:
                for block in iter(lambda: reader.read(8 << 20), b""):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != SOURCE_SHA:
                raise ValueError("Source ZIP member SHA-256 mismatch")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _read_opened(original_run):
    protocol, receipt = _inputs(original_run)
    seen, frames = set(), {}
    records = [(r["round"], r["role"], r) for r in receipt["seen_evidence"]]
    records.extend(("round17", role, item) for role, item in receipt["files"].items())
    if len(records) != 8:
        raise ValueError("Eight previously opened target CSVs are required")
    for previous_round, role, item in records:
        if previous_round == "round14" and item["sha256"] != protocol["files"][role]["sha256"]:
            raise ValueError(f"Development protocol differs from prior confirmation receipt: {role}")
        path = _checked(item["path"], item["sha256"])
        development = previous_round == "round14"
        frame = pd.read_csv(path, usecols=None if development else ["communication_group"],
                            float_precision="round_trip")
        if len(frame) != item["rows"] or frame["communication_group"].isna().any():
            raise ValueError(f"Invalid previously opened rows: {path}")
        seen.update(frame.communication_group)
        if development:
            frame = frame.rename(columns={"sample_id": "row_id"})[FIELDS]
            if not frame.row_id.is_unique or not frame.row_id.str.startswith(SOURCE_SHA + ":").all():
                raise ValueError(f"Invalid registered row identity: {path}")
            frames[OLD_NAMES[role]] = frame
    return frames, seen


def _metadata(source, threads):
    # Arrow's whole-file CSV reader preserves file order with parallel decoding.
    previous_threads = pa.cpu_count()
    try:
        pa.set_cpu_count(threads)
        strings = {key: pa.string() for key in META}
        strings["FLOW_END_MILLISECONDS"] = pa.int64()
        # Published preparations accepted "0"/"1" and equivalent numeric CSV
        # spellings. Validate binary values before sampling.
        strings["Label"] = pa.float64()
        table = arrow_csv.read_csv(source, read_options=arrow_csv.ReadOptions(
            use_threads=True, block_size=8 << 20), convert_options=arrow_csv.ConvertOptions(
                include_columns=META, column_types=strings))
    finally:
        pa.set_cpu_count(previous_threads)
    if table.num_rows != SOURCE_ROWS or any(table[key].null_count for key in META):
        raise ValueError("Invalid source row count or missing time, label or group identifier")
    return table


def _group_role(group, seed):
    bucket = int(lab._digest(["forward-history-group-v1", seed, group])[:16], 16) % 100
    return "P_time" if bucket < 50 else "C_time" if bucket < 70 else "E_time"


def _sample_indices(meta, seen_groups, *, intervals, quotas, seed):
    """Permute eligible row numbers once, filter groups, keep the first fixed quota."""
    end = meta["FLOW_END_MILLISECONDS"].to_numpy()
    labels = meta["Label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid source binary label")
    selected, stats = {}, {}
    group_columns = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]
    # Random Arrow takes on a multi-chunk column otherwise concatenate all
    # chunks repeatedly. Combine this narrow set once before drawing targets.
    group_data = meta.select(group_columns).combine_chunks()
    for index, (role, stratum, label) in enumerate(STRATA):
        lower, upper = intervals[role]
        eligible = (end >= lower + WINDOW_MS) & (end < upper)
        if label is not None:
            eligible &= labels == label
        indices = np.flatnonzero(eligible)
        rng = np.random.Generator(np.random.PCG64([seed, index]))
        rng.shuffle(indices)
        rows, target_sources, target_destinations = [], set(), set()
        inspected, old_groups, other_role = 0, 0, 0
        for offset in range(0, len(indices), 8192):
            chunk = indices[offset:offset + 8192]
            values = group_data.take(pa.array(chunk)).to_pydict()
            for position, src_ip, dst_ip, sport, dport, proto in zip(chunk, *(values[k] for k in group_columns)):
                inspected += 1
                group = lab._communication_group_from_ids(src_ip, dst_ip, lab._source_identifier(sport, 65535),
                    lab._source_identifier(dport, 65535), lab._source_identifier(proto, 255))
                if group in seen_groups:
                    old_groups += 1
                    continue
                if _group_role(group, seed) != role:
                    other_role += 1
                    continue
                rows.append((int(position), role, group, int(end[position])))
                target_sources.add(src_ip)
                target_destinations.add(dst_ip)
                if len(rows) == quotas[stratum]:
                    break
            if len(rows) == quotas[stratum]:
                break
        selected[stratum] = rows
        stats[stratum] = {"time_and_label_eligible_before_group_filter": len(indices),
                          "inspected_random_order": inspected, "seen_group_excluded_among_inspected": old_groups,
                          "other_role_excluded_among_inspected": other_role, "selected": len(rows),
                          "selected_source_hosts": len(target_sources), "selected_destination_hosts": len(target_destinations),
                          "quota": quotas[stratum], "quota_met": len(rows) == quotas[stratum]}
    return selected, stats


def _read_selected(source, selected):
    # For this SHA the prior metadata audit verified one physical line per CSV
    # record. Check the count, selected identity and SHA again during extraction.
    wanted = {position: (role, group, end) for rows in selected.values() for position, role, group, end in rows}
    ordered = sorted(wanted)
    output = {role: [] for role in INTERVALS}
    cursor, line_start = 0, 0
    digest = hashlib.sha256()
    with Path(source).open("rb") as raw:
        header = raw.readline()
        digest.update(header)
        columns = next(csv.reader([header.decode("utf-8-sig")], strict=True))
        indices = {name: i for i, name in enumerate(columns)}
        while block := raw.read(8 << 20):
            if not block.endswith(b"\n"):
                block += raw.readline()
            digest.update(block)
            lines = block.splitlines()
            stop = line_start + len(lines)
            while cursor < len(ordered) and ordered[cursor] < stop:
                pos = ordered[cursor]
                role, group, end = wanted[pos]
                fields = next(csv.reader([lines[pos - line_start].decode("utf-8")], strict=True))
                values, _ = lab._features(fields, indices)
                label = lab._integer(fields[indices["Label"]], "source label", 1)
                family = lab.FAMILIES[fields[indices["Attack"]].strip().casefold()]
                source_group = lab._communication_group(fields[indices["IPV4_SRC_ADDR"]],
                    fields[indices["IPV4_DST_ADDR"]], fields[indices["L4_SRC_PORT"]],
                    fields[indices["L4_DST_PORT"]], fields[indices["PROTOCOL"]])
                if (source_group != group or int(fields[indices["FLOW_END_MILLISECONDS"]]) != end
                        or (family == "Benign") != (label == 0)):
                    raise ValueError(f"Selected row identity mismatch at raw record {pos + 1}")
                row = [f"{SOURCE_SHA}:{pos + 1}", *[float(v) if v else np.nan for v in values],
                       label, family, group, end]
                output[role].append(row)
                cursor += 1
            line_start = stop
    if cursor != len(ordered) or line_start != SOURCE_ROWS or digest.hexdigest() != SOURCE_SHA:
        raise ValueError("Source row count, selected coverage or SHA-256 differs in extraction pass")
    return {role: pd.DataFrame(rows, columns=FIELDS + ["flow_end_ms"]) for role, rows in output.items()}


def _forward_summary(frames, seen, stats):
    groups, rows = {}, {}
    for role, frame in frames.items():
        groups[role] = set(frame.communication_group)
        lower, upper = INTERVALS[role]
        if (groups[role] & seen or not frame.row_id.is_unique or
                not frame.flow_end_ms.between(lower + WINDOW_MS, upper - 1).all()):
            raise ValueError(f"Forward target isolation failed for {role}")
        day = pd.to_datetime(frame.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        daily = frame.groupby([day, "attack_family"]).size()
        by_date = {}
        for (date, family), count in daily.items():
            by_date.setdefault(date, {})[family] = int(count)
        rows[role] = {"rows": len(frame), "groups": len(groups[role]), "utc_end_day_family_counts": by_date,
                      "families": {str(k): int(v) for k, v in frame.attack_family.value_counts().items()},
                      "min_end_ms": int(frame.flow_end_ms.min()) if len(frame) else None,
                      "max_end_ms": int(frame.flow_end_ms.max()) if len(frame) else None}
    if groups["P_time"] & groups["C_time"] or (groups["P_time"] | groups["C_time"]) & groups["E_time"]:
        raise ValueError("A communication group appears in multiple time roles")
    p, c, e = (frames[k] for k in INTERVALS)
    if c.label.any():
        raise ValueError("Forward calibration includes an attack")
    binary = p.label.nunique() == 2 and len(c) >= 99 and e.label.nunique() == 2
    return {"binary_evaluable": bool(binary), "scanning_evaluable": bool(binary and
            "Scanning" in rows["P_time"]["families"] and "Scanning" in rows["E_time"]["families"]),
            "rows": rows, "sampling": stats, "excluded_old_groups": len(seen),
            "overlap_with_old_target_groups": 0, "cross_time_role_group_overlap": 0}


def prepare_cohorts(original_run, source, work, threads):
    """Prepare/load six role DataFrames. The caller freezes arms and owns the run lock."""
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    root = Path(work).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frames, seen = _read_opened(original_run)
    source = _checked(source, SOURCE_SHA)
    recipe = {"format": "flow-history-forward-cohort-v1", "source_sha256": SOURCE_SHA,
              "opened_protocol_sha256": PROTOCOL_SHA, "opened_confirmation_receipt_sha256": PREPARATION_SHA,
              "seed": SEED, "intervals_utc_ms": {k: list(v) for k, v in INTERVALS.items()},
              "warmup_ms": WINDOW_MS, "quotas": QUOTAS,
              "group": "old group digest, then hash partition P<50,C<70,E>=70; exclude 8 opened tables",
              "priority": "PCG64([seed,stratum_index]) Fisher-Yates over ascending eligible raw row numbers",
              "time_assignment": "completed flow by FLOW_END_MILLISECONDS; a long flow may begin in a prior interval",
              "scope": "same-source forward time + unseen target groups, not unseen hosts; E target selection label-blind"}
    receipt = root / "cohort_manifest.json"
    freeze = root / "cohort_recipe.json"
    if freeze.exists() and read_json(freeze) != recipe:
        raise ValueError("Cannot resume with a different frozen cohort recipe")
    if receipt.is_file():
        old = read_json(receipt)
        if old["recipe"] != recipe:
            raise ValueError("Cannot resume a different cohort recipe")
        for role in INTERVALS:
            item = old["files"][role]
            frames[role] = pd.read_parquet(_checked(root / f"{role}.parquet", item["sha256"]))
        return frames
    write_json(freeze, recipe)
    metadata = _metadata(source, threads)
    selected, stats = _sample_indices(metadata, seen, intervals=INTERVALS, quotas=QUOTAS, seed=SEED)
    del metadata
    forward = _read_selected(source, selected)
    report = _forward_summary(forward, seen, stats)
    files = {}
    for role, frame in forward.items():
        path = root / f"{role}.parquet"
        temp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(path)
        files[role] = {"name": path.name, "sha256": sha256_file(path), "rows": len(frame)}
    write_json(receipt, {"recipe": recipe, "files": files, "forward": report, "source_rows_verified": SOURCE_ROWS,
                         "previously_opened_target_files_sha_verified": 8,
                         "arrow_parse_threads": threads, "model_fits": 0, "predictions": 0})
    return {**frames, **forward}
