"""Evaluate sealed pairs without selecting thresholds, captures, models or seeds."""

from collections import defaultdict
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ids.common import read_json, write_json
from ids.contact_evidence.evaluation import _FlagStream
from ids.contact_evidence.experiment import GROUPS, ROWS, enriched
from . import data
from .experiment import ARM_NAMES, COMPARISONS
from .observations import GAP_COLUMNS

BINS = np.arange(4 << len(ARM_NAMES), dtype=np.uint16)
TARGETS = ("Scanning", "PureCC", "Download")
STATE_NAMES = ("unknown", "outbound_only", "reply_counted")


def endpoint(family, detail):
    """Mutually exclusive endpoints. Pure C&C and heartbeat exclude download rows."""
    native = str(detail).casefold().split("-")
    if family in {"Benign", "Unresolved", "Scanning", "ExternalC2", "ExternalRecon"}:
        return family
    if family == "C&C":
        return "PureCC"
    if "filedownload" in native:
        return "Download"
    if "heartbeat" in native:
        return "Heartbeat"
    return "OtherAttack"


def pair(hist, old, new):
    left = (BINS & (1 << ARM_NAMES.index(old))) != 0
    right = (BINS & (1 << ARM_NAMES.index(new))) != 0
    n = int(hist.sum())
    before, after = int(hist[left].sum()), int(hist[right].sum())
    return {
        "rows": n,
        "before": before,
        "after": after,
        "gained": int(hist[~left & right].sum()),
        "lost": int(hist[left & ~right].sum()),
        "net": after - before,
        "before_rate": before / n if n else None,
        "after_rate": after / n if n else None,
    }


def criteria(counts):
    normal = counts.get("Benign", {})
    n = normal.get("rows", 0)
    cap = n // 100 if n else None
    within = normal["after"] <= cap if n else None
    base_ok = normal["before"] <= cap if n else None
    target = {}
    for k in TARGETS:
        item = counts.get(k, {})
        support = item.get("rows", 0)
        target[k] = {
            "support": support,
            "recoverable_misses": support - item.get("before", 0),
            "net_more_detected": item["net"] > 0 if support else None,
        }
    available = all(t["support"] for t in target.values()) and bool(n)
    all_gain = all(t["net_more_detected"] for t in target.values())
    return {
        "normal_count": n,
        "allowed_fp": cap,
        "baseline_fp": normal.get("before"),
        "candidate_fp": normal.get("after"),
        "baseline_within_budget": base_ok,
        "candidate_within_budget": within,
        "targets": target,
        "joint": ("pass" if base_ok and within and all_gain else "fail") if available else "not_evaluable",
    }


def _add(tables, names, codes, flags):
    if np.any(codes < 0):
        raise ValueError("Missing group identity")
    width = len(BINS)
    cells = np.bincount(codes.astype(np.int64) * width + flags, minlength=len(names) * width).reshape(
        -1, width
    )
    for name, row in zip(names, cells):
        tables[str(name)] += row


def judge(root, spec, groups=GROUPS):
    name = spec["name"]
    table = {
        g: {k: defaultdict(lambda: np.zeros(len(BINS), dtype=np.int64)) for k in ("native", "endpoint")}
        for g in groups
    }
    windows = defaultdict(lambda: np.zeros(len(BINS), dtype=np.int64))
    author_windows = defaultdict(lambda: np.zeros(len(BINS), dtype=np.int64))
    provenance = defaultdict(lambda: np.zeros(len(BINS), dtype=np.int64))
    direction = defaultdict(lambda: np.zeros(len(BINS), dtype=np.int64))
    seen = 0
    folder = root / "evidence/blind" / name
    with ExitStack() as handles:
        truth = handles.enter_context(pq.ParquetFile(data.truth(root, name)))
        blind = handles.enter_context(pq.ParquetFile(enriched(root, name)))
        gap_columns = [_FlagStream(blind, truth.metadata.num_rows, column=c) for c in GAP_COLUMNS]
        streams = {
            g: _FlagStream(
                handles.enter_context(pq.ParquetFile(folder / (g + ".parquet"))), truth.metadata.num_rows
            )
            for g in groups
        }
        fields = ["row_id", "family", "native_detail"]
        if "window" in truth.schema_arrow.names:
            fields.append("window")
        if "author_window" in truth.schema_arrow.names:
            fields.append("author_window")
        if "label_source" in truth.schema_arrow.names:
            fields.append("label_source")
        for batch in truth.iter_batches(batch_size=ROWS, columns=fields):
            frame = batch.to_pandas()
            if not np.array_equal(frame.row_id, np.arange(seen + 1, seen + 1 + len(frame))):
                raise ValueError("Truth differs from the sealed capture row order")
            # A native detail may occur with different author binary labels.
            composite = pd.MultiIndex.from_frame(frame[["family", "native_detail"]])
            pair_codes, labels = composite.factorize(sort=False)
            native = np.array([str(f) + "/" + str(d) for f, d in labels])
            ends = [endpoint(f, d) for f, d in labels]
            codes, names = pd.factorize(np.asarray(ends), sort=False)
            first_state = None
            for g, stream in streams.items():
                flags = stream.take(len(frame))
                states = flags >> len(ARM_NAMES)
                if (states > 2).any() or (
                    first_state is not None and not np.array_equal(first_state, states)
                ):
                    raise ValueError("Inconsistent sealed response states")
                first_state = states
                _add(table[g]["native"], native, pair_codes, flags)
                _add(table[g]["endpoint"], names, codes[pair_codes], flags)
                if g == "hgb":
                    orig, resp = (col.take(len(frame)) for col in gap_columns)
                    if not np.isin(orig, [0, 1]).all() or not np.isin(resp, [0, 1]).all():
                        raise ValueError("Invalid directional checksum flag")
                    for i, label in enumerate(("no_mark", "originator", "responder", "both")):
                        keep = (orig + 2 * resp) == i
                        if keep.any():
                            _add(
                                direction,
                                [label + "/" + k for k in names],
                                codes[pair_codes[keep]],
                                flags[keep],
                            )
                if g == "hgb" and "window" in fields:
                    if frame.window.isna().any():
                        raise ValueError("Missing evaluation window")
                    for window, indices in frame.groupby("window", sort=False).indices.items():
                        _add(
                            windows,
                            [str(window) + "/" + k for k in names],
                            codes[pair_codes[indices]],
                            flags[indices],
                        )
                if g == "hgb" and "author_window" in fields:
                    for window, indices in frame.groupby("author_window", sort=False).indices.items():
                        if str(window) not in {"", "-", "(empty)"}:
                            _add(
                                author_windows,
                                [str(window) + "/" + k for k in names],
                                codes[pair_codes[indices]],
                                flags[indices],
                            )
                if g == "hgb" and "label_source" in fields:
                    for label_source, indices in frame.groupby("label_source", sort=False).indices.items():
                        _add(
                            provenance,
                            [str(label_source) + "/" + k for k in native],
                            pair_codes[indices],
                            flags[indices],
                        )
            seen += len(frame)
    if not seen or seen != truth.metadata.num_rows:
        raise ValueError("Incomplete truth stream")
    comparisons = {}
    for g, tables in table.items():
        report = {}
        for old, new in COMPARISONS:
            key = old + "_to_" + new
            counts = {k: pair(v, old, new) for k, v in tables["endpoint"].items()}
            report[key] = {
                "endpoints": counts,
                "decision": criteria(counts),
                "native": {k: pair(v, old, new) for k, v in tables["native"].items()},
            }
            if (old, new) == COMPARISONS[0]:
                report[key]["by_evidence_state"] = {
                    k: {
                        s: pair(v * ((BINS >> len(ARM_NAMES)) == i), old, new)
                        for i, s in enumerate(STATE_NAMES)
                    }
                    for k, v in tables["endpoint"].items()
                }
        comparisons[g] = report
    path = root / "evidence/evaluation" / (name + ".json")
    write_json(
        path,
        {
            "capture": name,
            "role": spec["role"],
            "rows": seen,
            "groups": comparisons,
            "hgb_by_checksum_direction_and_endpoint": {
                old + "_to_" + new: {k: pair(v, old, new) for k, v in direction.items()}
                for old, new in (COMPARISONS[0], ("quality", "counter_masked"))
            },
            "primary_by_window": _windows(windows),
            "publisher_named_windows": _windows(author_windows),
            "primary_by_label_source_and_native": {
                k: pair(v, *COMPARISONS[0]) for k, v in provenance.items()
            },
        },
    )
    return [path]


def _windows(hist):
    by_window = defaultdict(dict)
    for key, bins in hist.items():
        window, endpoint_name = key.rsplit("/", 1)
        by_window[window][endpoint_name] = pair(bins, *COMPARISONS[0])
    return {k: {"endpoints": v, "decision": criteria(v)} for k, v in by_window.items()}


def overview(root, specs):
    captures = {}
    for s in specs:
        item = read_json(root / "evidence/evaluation" / (s["name"] + ".json"))
        primary = item["groups"]["hgb"]["contact_to_quality"]
        captures[s["name"]] = {
            "role": s["role"],
            "decision": primary["decision"],
            "endpoints": primary["endpoints"],
            "windows": item["primary_by_window"],
        }
    new = [k for k, v in captures.items() if v["role"] == "E"]
    forward = forward_targets([captures[k] for k in new])
    full_status = summarize_joint([captures[k]["decision"]["joint"] for k in new])
    if full_status == "pass":
        full_status = summarize_joint([v["status"] for v in forward.values()])
    result = root / "outcome.json"
    write_json(
        result,
        {
            "primary": "hgb/contact_to_quality",
            "captures": captures,
            "new_forward_captures": new,
            "new_forward_joint": full_status,
            "forward_by_target": forward,
            "unscored_historical_by_target": forward_targets(
                [v for v in captures.values() if v["role"] == "H"]
            ),
            "external_c2_by_unit": forward_targets(
                [v for v in captures.values() if v["role"] == "X"], targets=("ExternalC2", "ExternalRecon")
            ),
            "joint_scope": "Both frozen arms within 1% in each capture; strict net gain separately for scanning, pure C&C and file download.",
        },
    )
    return [result]


def summarize_joint(statuses):
    if "fail" in statuses:
        return "fail"
    return "pass" if statuses and all(s == "pass" for s in statuses) else "not_evaluable"


def forward_targets(captures, targets=TARGETS):
    result = {}
    budgets = [
        d for cap in captures for d in [cap["decision"], *(w["decision"] for w in cap["windows"].values())]
    ]
    for k in targets:
        counts = [cap["endpoints"].get(k, {}) for cap in captures]
        totals = {
            field: sum(c.get(field, 0) for c in counts)
            for field in ("rows", "before", "after", "gained", "lost", "net")
        }
        budget = [v[key] for v in budgets for key in ("baseline_within_budget", "candidate_within_budget")]
        if not totals["rows"]:
            status = "not_evaluable"
        elif False in budget or totals["net"] <= 0:
            status = "fail"
        else:
            status = "pass" if budget and None not in budget else "not_evaluable"
        result[k] = {**totals, "status": status}
    return result
