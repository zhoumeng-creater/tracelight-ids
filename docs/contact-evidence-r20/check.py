"""Recompute R20's gate evidence on opened calibration rows, without fitting, scoring or choosing a rule."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

NATIVE_CSV = "242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74"
R20_ARCHIVE = "c19bdf20f745801f3b860e0ef849025e73a748ada0da04659b7acffcb7aed63e"
SOURCE = "23.225.163.149"
WINDOW = 60000
ENC = "ascii"


def file_sha(path):
    digest = sha256()
    with path.open("rb") as source:
        while data := source.read(4 << 20):
            digest.update(data)
    return digest.hexdigest()


def calibration(path):
    if file_sha(path) != R20_ARCHIVE:
        raise ValueError("Round-20 evidence SHA differs")
    with ZipFile(path) as z:

        def read_frame(member):
            return pd.read_parquet(BytesIO(z.read(member))).set_index("row_id", verify_integrity=True)

        rows = read_frame("evidence/inputs/registered_rows.parquet")
        rows = rows[rows.role.eq("C_time")].copy()
        scored = read_frame("evidence/forward/hgb52_history/calibration.parquet")
        cut = json.loads(z.read("evidence/forward/hgb52_history/metrics.json"))["threshold"]["threshold"]
        if not rows.index.equals(scored.index) or len(rows) != 10000 or not rows.label.eq(0).all():
            raise ValueError("Expected 10,000 aligned normal calibration rows")
        rows["score"] = scored.score
        prefix = NATIVE_CSV + ":"
        if not rows.index.str.startswith(prefix).all():
            raise ValueError("Calibration bound to another native CSV")
        rows.index = rows.index.str.removeprefix(prefix).astype(int)
        return rows, cut


def service_events(path, low_ms, high_ms):
    digest = sha256()
    events = defaultdict(list)
    pos = 0
    with ZipFile(path) as archive:
        [member] = [m for m in archive.namelist() if m.endswith("/NF-CICIDS2018-v3.csv")]
        with archive.open(member) as source:
            header = source.readline()
            digest.update(header)
            expected = (
                "FLOW_START_MILLISECONDS,FLOW_END_MILLISECONDS,IPV4_SRC_ADDR,L4_SRC_PORT,"
                "IPV4_DST_ADDR,L4_DST_PORT,PROTOCOL,L7_PROTO,IN_BYTES,IN_PKTS,OUT_BYTES,OUT_PKTS"
            )
            if header.split(b",", 12)[:12] != expected.encode(ENC).split(b","):
                raise ValueError("Unexpected native column order")
            carry = b""
            while chunk := source.read(8 << 20):
                digest.update(chunk)
                lines = (carry + chunk).split(b"\n")
                carry = lines.pop()
                for i, line in enumerate(lines, pos + 1):
                    if b",135,6," not in line:
                        continue
                    cols = line.split(b",", 12)
                    if cols[5:7] != [b"135", b"6"]:
                        continue
                    time = int(cols[1])
                    if low_ms <= time <= high_ms:
                        resp = int(cols[11])
                        if resp < 0:
                            raise ValueError("Unexpected unknown NetFlow responder packet count")
                        events[cols[2]].append((time, cols[4], resp, i))
                pos += len(lines)
            if carry:
                raise ValueError("Expected the pinned official newline-terminated CSV")
    if digest.hexdigest() != NATIVE_CSV:
        raise ValueError("Native CSV digest differs")
    return events, pos


def history(times, responses, at):
    lo, hi = times.searchsorted([at - WINDOW, at])
    prev = responses[lo:hi]
    return int(hi - lo), int(np.sum(prev > 0)), int(np.sum(prev == 0))


def observations(events, rows, cut):
    selected = []
    for address, entries in events.items():
        targets = [entry for entry in entries if entry[3] in rows.index]
        if not targets:
            continue
        known_service = sorted(entries)
        times = np.array([e[0] for e in known_service], dtype=np.int64)
        replies = np.array([e[2] for e in known_service], dtype=np.int64)
        dests = {e[1] for e in targets}
        by_dest = defaultdict(list)
        for time, dst, response, _ in known_service:
            if dst in dests:
                by_dest[dst].append((time, response))
        contexts = {d: np.array(values, dtype=np.int64).T for d, values in by_dest.items()}
        source_id = sha256(address).hexdigest()[:16]
        for time, dst, _, row_id in targets:
            archived = rows.loc[row_id]
            if source_id != archived.host_key or time != archived.flow_end_ms:
                raise ValueError("Raw row differs from archive source/time")
            flows, positive, zero = history(times, replies, time)
            n, reply, _ = history(*contexts[dst], time)
            selected.append(
                {
                    "row_id": row_id,
                    "time_ms": time,
                    "source": address.decode(ENC),
                    "dst": dst.decode(ENC),
                    "score_vs_cut": "above"
                    if archived.score > cut
                    else "tied"
                    if archived.score == cut
                    else "below",
                    "service_prior": flows,
                    "service_reply": positive,
                    "service_zero": zero,
                    "contact_prior": n,
                    "contact_reply": reply,
                }
            )
    return pd.DataFrame(selected).sort_values("row_id")


def summarize(frame):
    has_zero = frame.service_zero.gt(0)
    pair_reply = frame.contact_reply.gt(0)
    pass_gate = has_zero & ~pair_reply
    columns = [
        "row_id",
        "time_ms",
        "service_prior",
        "service_reply",
        "service_zero",
        "contact_prior",
        "contact_reply",
    ]
    byte_table = np.ascontiguousarray(frame[columns].to_numpy(dtype="<i8")).tobytes()
    return {
        "rows": len(frame),
        "sources": frame.source.nunique(),
        "destinations": frame.dst.nunique(),
        "archived_above": int(frame.score_vs_cut.eq("above").sum()),
        "archived_tied": int(frame.score_vs_cut.eq("tied").sum()),
        "service_known_zero_and_pair_no_reply__gate_pass": int(pass_gate.sum()),
        "service_known_zero_and_pair_had_reply": int((has_zero & pair_reply).sum()),
        "service_no_known_zero_and_pair_no_reply": int((~has_zero & ~pair_reply).sum()),
        "service_no_known_zero_and_pair_had_reply": int((~has_zero & pair_reply).sum()),
        "contact_no_prior": int(frame.contact_prior.eq(0).sum()),
        "contact_prior_only_zero_response": int((frame.contact_prior.gt(0) & ~pair_reply).sum()),
        "contact_any_prior_response": int(pair_reply.sum()),
        "rowwise_values_sha256": sha256(byte_table).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--round20-zip", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, cut = calibration(args.round20_zip)
    events, csv_rows = service_events(
        args.source_zip, rows.flow_end_ms.min() - WINDOW, rows.flow_end_ms.max()
    )
    frame = observations(events, rows, cut)
    key = sha256(SOURCE.encode(ENC)).hexdigest()[:16]
    host = frame[frame.source.eq(SOURCE)]
    above, tied = host[host.score_vs_cut.eq("above")], host[host.score_vs_cut.eq("tied")]
    if len(above) != 88 or len(tied) != 15 or rows.host_key.eq(key).sum() != 135:
        raise ValueError("Failed to recover archived source and fixed-threshold rows")
    groups = {
        "leading_host_above": above,
        "leading_host_tied": tied,
        "leading_host_below": host[host.score_vs_cut.eq("below")],
        "other_hosts_same_service": frame[frame.source.ne(SOURCE)],
        "all_c_normal_same_service": frame,
    }
    record = {
        "purpose": "development diagnosis on already opened CIC2018 normal calibration; no scoring or fitting",
        "native_csv_sha256": NATIVE_CSV,
        "round20_zip_sha256": R20_ARCHIVE,
        "native_csv_rows": csv_rows,
        "source_zip_bytes": args.source_zip.stat().st_size,
        "normal_calibration_rows": len(rows),
        "raw_selected_host": SOURCE,
        "all_calibration_rows_for_host": int(rows.host_key.eq(key).sum()),
        "protocol": 6,
        "destination_port": 135,
        "frozen_hist_threshold": cut,
        "history": "[t-60000,t) by completion time across all native flows, with exact source/protocol/port; "
        "contact further matches destination IP. OUT_PKTS>0 means a return packet, not service success.",
        "native_past_read_rows_for_service_in_calibration_time_span": sum(map(len, events.values())),
        "rowwise_hash_schema": "Sort by one-based original row ID, contiguous little-endian int64 columns "
        "row_id,time_ms,service_prior,service_reply,service_zero,contact_prior,contact_reply",
        "groups": {name: summarize(g) for name, g in groups.items()},
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: [len(v), record["groups"][k]["service_known_zero_and_pair_no_reply__gate_pass"]]
                for k, v in groups.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
