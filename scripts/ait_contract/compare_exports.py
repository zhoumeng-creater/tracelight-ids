"""Compare two Zeek exports by exact connection keys, without selecting an arm."""
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path


KEY_FIELDS = ("ts", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto")
FIELDS = ("duration", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
          "conn_state", "missed_bytes", "contract_export_network_time",
          "contract_after_input_eof", "contract_during_shutdown")


def _read(directory):
    directory = Path(directory)
    log, receipt = directory / "conn.log", directory / "export-receipt.json"
    if not log.is_file():
        saved = json.loads(receipt.read_text(encoding="utf-8-sig")) if receipt.is_file() else {}
        if saved.get("status") == "complete" and saved.get("logs", {}).get("conn", {}).get("exists") is False:
            return [], {"conn_log_present": False, "empty_export_confirmed_by_receipt": True,
                        "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
        raise FileNotFoundError(f"Missing conn.log without a successful explicit absent-log receipt: {directory}")
    rows, sha = [], hashlib.sha256()
    with log.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            sha.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw.decode("utf-8-sig"), parse_float=Decimal)
            if not isinstance(row, dict) or any(name not in row or row[name] is None for name in KEY_FIELDS):
                raise ValueError(f"Connection key is incomplete at {log}:{line_number}")
            rows.append((line_number, row))
    return rows, {"conn_log_present": True, "conn_log_sha256": sha.hexdigest(), "rows": len(rows)}


def _key(row):
    return tuple(row[name] for name in KEY_FIELDS)


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _single(key, side, numbered_row):
    number, row = numbered_row
    return {**dict(zip(KEY_FIELDS, key)), "side": side, "line_number": number,
            "uid": row.get("uid", ""), **{name: row.get(name, "") for name in FIELDS},
            "present_fields": _json([name for name in FIELDS if name in row])}


def compare_exports(strict_dir, ignore_dir, output_dir) -> dict:
    """Write five CSV partitions and a summary; UID never participates in pairing."""
    strict, strict_receipt = _read(strict_dir)
    ignore, ignore_receipt = _read(ignore_dir)
    left, right = defaultdict(list), defaultdict(list)
    for rows, groups in ((strict, left), (ignore, right)):
        for number, row in rows:
            groups[_key(row)].append((number, row))
    partitions = {name: [] for name in ("strict_only", "ignore_checksums_only", "common_identical",
                                        "common_changed", "ambiguous")}
    changes, ambiguous_keys = Counter(), 0
    for key in sorted(left.keys() | right.keys(), key=lambda value: tuple(str(x) for x in value)):
        lhs, rhs = left[key], right[key]
        if len(lhs) > 1 or len(rhs) > 1:
            ambiguous_keys += 1
            for side, rows in (("strict", lhs), ("ignore_checksums", rhs)):
                partitions["ambiguous"].extend(_single(key, side, row) for row in rows)
            continue
        if not lhs or not rhs:
            side = "strict" if lhs else "ignore_checksums"
            partitions[side + "_only"].append(_single(key, side, (lhs or rhs)[0]))
            continue
        lnum, lrow = lhs[0]
        rnum, rrow = rhs[0]
        changed = [name for name in FIELDS if (name in lrow, lrow.get(name)) != (name in rrow, rrow.get(name))]
        changes.update(changed)
        common = {**dict(zip(KEY_FIELDS, key)), "strict_line": lnum, "ignore_checksums_line": rnum,
                  "strict_uid": lrow.get("uid", ""), "ignore_checksums_uid": rrow.get("uid", ""),
                  "changed_fields": _json(changed),
                  "strict_present_fields": _json([name for name in FIELDS if name in lrow]),
                  "ignore_checksums_present_fields": _json([name for name in FIELDS if name in rrow])}
        for name in FIELDS:
            common["strict_" + name] = lrow.get(name, "")
            common["ignore_checksums_" + name] = rrow.get(name, "")
        partitions["common_changed" if changed else "common_identical"].append(common)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    single_columns = [*KEY_FIELDS, "side", "line_number", "uid", *FIELDS, "present_fields"]
    common_columns = [*KEY_FIELDS, "strict_line", "ignore_checksums_line", "strict_uid", "ignore_checksums_uid",
                      "changed_fields", "strict_present_fields", "ignore_checksums_present_fields"]
    common_columns += [prefix + name for name in FIELDS for prefix in ("strict_", "ignore_checksums_")]
    for name, rows in partitions.items():
        _csv(output / (name + ".csv"), rows, common_columns if name.startswith("common_") else single_columns)
    summary = {"status": "complete", "inputs": {"strict": strict_receipt, "ignore_checksums": ignore_receipt},
               "connection_rows": {"strict": len(strict), "ignore_checksums": len(ignore)},
               "exact_key_fields": list(KEY_FIELDS), "compared_fields": list(FIELDS),
               "counts": {name: len(rows) for name, rows in partitions.items()},
               "ambiguous_keys": ambiguous_keys,
               "field_change_counts": dict(changes),
               "comparison_scope": "Common identical means equal only on the listed fields; UID and other fields are not compared.",
               "key_policy": "Exact numeric JSON timestamps and endpoint/protocol keys; no UID or nearest-time pairing.",
               "ambiguity_policy": "A repeated key in either arm leaves every row of that key unpaired in ambiguous.csv.",
               "variant_selection": "none", "models_scored": 0, "causal_inference": "none"}
    (output / "export-comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
