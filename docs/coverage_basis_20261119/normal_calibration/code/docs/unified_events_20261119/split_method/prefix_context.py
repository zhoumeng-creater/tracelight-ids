"""Deterministic, non-training completed-flow context export.

Every supplied observation is processed or explicitly blocked with its whole
group, including unlabelled background. The
caller must supply the complete declared capture membership; this module cannot
prove that upstream extraction did not omit observations. ``host`` is the
monitored host, not necessarily the source IP. Flow IDs must be namespaced and
unique across the supplied groups.

The batch API sorts inputs to make results independent of input order. Sorting
and returned rows take O(N) memory; only the context state is constant per host.
This is an offline exporter, not a claim of bounded-memory online deployment.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import gzip
from itertools import groupby
import json
import math
from numbers import Real
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "prefix-context-v1"
RAW8_FIELDS = (
    "ip_protocol", "orig_port", "resp_port", "duration_seconds",
    "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
)
TIME_MODES = ("available_at", "offline_flow_end_replay")
UNAVAILABLE_POLICIES = ("error", "block-group")


def _timestamp(value: object, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite epoch-second number or timezone-aware ISO timestamp")
    if isinstance(value, Real):
        result = float(value)
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid {name}: {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{name} requires an explicit timezone")
        result = parsed.timestamp()
    else:
        raise ValueError(f"Unsupported {name} type")
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity(row: Mapping, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _prepare(row: Mapping, time_mode: str, *, allow_unavailable: bool = False) -> dict:
    if not isinstance(row, Mapping):
        raise ValueError("Each observation must be a mapping")
    result = {name: _identity(row, name) for name in ("group", "host", "peer", "flow_id")}
    result["physical_flow_id"] = _identity(row, "physical_flow_id") if "physical_flow_id" in row else result["flow_id"]
    clock_field = "flow_end_at" if time_mode == "offline_flow_end_replay" else "available_at"
    # Offline mode uses flow_end_at for every record, never a mixed-clock fallback.
    value = row.get(clock_field)
    result["time"] = None if value is None and allow_unavailable else _timestamp(value, clock_field)
    raw = row.get("raw8")
    if not isinstance(raw, (list, tuple)) or len(raw) != 8:
        raise ValueError(f"raw8 must contain exactly 8 values: {result['flow_id']}")
    values = []
    for value in raw:
        if value is None:
            values.append(None)
        elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"raw8 values must be finite nonnegative numbers or null: {result['flow_id']}")
        else:
            values.append(math.log1p(value))
    result["values"] = values
    return result


@dataclass
class _HostState:
    count: int = 0
    means: list[float] = field(default_factory=lambda: [0.0] * 8)
    valid_counts: list[int] = field(default_factory=lambda: [0] * 8)
    previous: dict | None = None

    def query(self, row: dict, time_mode: str) -> dict:
        empty = self.previous is None
        if empty:
            previous_values, means = [0.0] * 8, [0.0] * 8
            elapsed, same_peer, previous_id, max_time = 0.0, 0, None, None
        else:
            previous_values = list(self.previous["values"])
            means = [value if count else None for value, count in zip(self.means, self.valid_counts)]
            max_time = self.previous["time"]
            if max_time >= row["time"]:
                raise ValueError("History must be strictly earlier than the query record")
            elapsed = math.log1p(row["time"] - max_time)
            same_peer = int(self.previous["peer"] == row["peer"])
            previous_id = self.previous["flow_id"]
        history = previous_values + means + [elapsed, same_peer, int(empty)]
        return {
            "group": row["group"], "host": row["host"], "flow_id": row["flow_id"],
            "physical_flow_id": row["physical_flow_id"],
            "time_mode": time_mode, "effective_available_at": row["time"],
            "max_history_available_at": max_time, "previous_flow_id": previous_id,
            "history_flow_count": self.count, "mean_valid_counts": list(self.valid_counts),
            "previous_raw8_log1p": previous_values, "prefix_mean_log1p": means,
            "log1p_elapsed_since_previous_available": elapsed,
            "same_peer_as_previous": same_peer, "empty_history": int(empty),
            "current_raw8_log1p": list(row["values"]),
            "history_features": history, "model_features": list(row["values"]) + history,
        }

    def update(self, row: dict) -> None:
        self.count += 1
        for i, value in enumerate(row["values"]):
            if value is not None:
                self.valid_counts[i] += 1
                self.means[i] += (value - self.means[i]) / self.valid_counts[i]
        self.previous = row


def build_prefix_context(observations: Iterable[Mapping], *, time_mode: str = "available_at",
                         unavailable_policy: str = "error") -> dict:
    """Return all context rows plus an audit; labels never affect membership.

    Required input fields: group, host, peer, flow_id, available_at, raw8.
    Explicit ``offline_flow_end_replay`` additionally requires flow_end_at and
    uses it for *all* observations, regardless of available_at. It does not
    represent actual online availability. Other fields, including labels, are
    ignored. Duplicate view flow IDs are rejected instead of silently deduplicated.
    Optional physical_flow_id defaults to flow_id. Distinct hosts may observe
    one physical flow inside the same group, but a host may update from that
    physical flow only once. Views do not increase the independent-flow count.
    Missing clock values raise by default. Explicit ``block-group`` excludes
    every observation in an affected group and records its membership and the
    original missing-clock rows in the audit. Invalid non-null timestamps still
    raise; the policy does not coerce or repair timestamps.
    """
    if time_mode not in TIME_MODES:
        raise ValueError(f"time_mode must be one of {TIME_MODES}")
    if unavailable_policy not in UNAVAILABLE_POLICIES:
        raise ValueError(f"unavailable_policy must be one of {UNAVAILABLE_POLICIES}")
    prepared, seen_ids, seen_physical_hosts, physical_groups = [], set(), set(), {}
    unavailable_rows = []
    clock_field = "flow_end_at" if time_mode == "offline_flow_end_replay" else "available_at"
    for raw in observations:
        row = _prepare(raw, time_mode, allow_unavailable=unavailable_policy == "block-group")
        if row["flow_id"] in seen_ids:
            raise ValueError(f"Duplicate flow_id, including across groups: {row['flow_id']}")
        physical_id = row["physical_flow_id"]
        if physical_id in physical_groups and physical_groups[physical_id] != row["group"]:
            raise ValueError(f"One physical_flow_id cannot cross capture groups: {physical_id}")
        pair = (physical_id, row["host"])
        if pair in seen_physical_hosts:
            raise ValueError(f"Repeated physical flow for the same monitored host: {physical_id}")
        physical_groups[physical_id] = row["group"]
        seen_physical_hosts.add(pair)
        seen_ids.add(row["flow_id"])
        prepared.append(row)
        if row["time"] is None:
            unavailable_rows.append({
                "group": row["group"], "flow_id": row["flow_id"],
                "physical_flow_id": physical_id, "missing_clock_field": clock_field,
                "reason": "Required clock is null or absent; no start-time substitution or row-level dropping.",
                "original_observation": dict(raw),
            })
    unavailable_rows.sort(key=lambda r: (r["group"], r["flow_id"]))
    blocked_names = {r["group"] for r in unavailable_rows}
    blocked_groups = []
    for group_name in sorted(blocked_names):
        flow_ids = sorted(r["flow_id"] for r in prepared if r["group"] == group_name)
        blocked_groups.append({
            "group": group_name, "reason": "missing_required_clock_blocks_complete_group",
            "clock_field": clock_field, "blocked_observation_count": len(flow_ids),
            "flow_ids": flow_ids,
            "unavailable_flow_ids": [r["flow_id"] for r in unavailable_rows if r["group"] == group_name],
        })
    eligible = [r for r in prepared if r["group"] not in blocked_names]
    eligible.sort(key=lambda r: (r["group"], r["time"], r["flow_id"]))
    output, group_counts = [], Counter()
    max_hosts = 0
    for group_name, group_rows in groupby(eligible, key=lambda r: r["group"]):
        # A fresh state dictionary enforces the complete-capture boundary.
        states: dict[str, _HostState] = {}
        for _, equal_time_rows in groupby(group_rows, key=lambda r: r["time"]):
            batch = list(equal_time_rows)
            for row in batch:
                state = states.setdefault(row["host"], _HostState())
                output.append(state.query(row, time_mode))
                group_counts[group_name] += 1
            # The sorted flow_id breaks ties only for state seen by later times.
            for row in batch:
                states[row["host"]].update(row)
        max_hosts = max(max_hosts, len(states))
    offline = time_mode == "offline_flow_end_replay"
    return {
        "schema": SCHEMA, "time_mode": time_mode, "model_calls": 0,
        "unavailable_policy": unavailable_policy,
        "raw8_fields": list(RAW8_FIELDS),
        "clock_semantics": (
            "Offline flow-end replay only; flow_end_at is not actual detector receipt/export time."
            if offline else "Caller-supplied available_at; its receipt/export/replay provenance must be retained upstream."
        ),
        "actual_online_availability_claimed": False,
        "rows": output,
        "audit": {
            "input_observations": len(prepared), "output_observations": len(output),
            "unique_physical_flows": len(physical_groups), "host_views": len(prepared),
            "output_unique_physical_flows": len({r["physical_flow_id"] for r in eligible}),
            "input_group_counts": dict(sorted(Counter(r["group"] for r in prepared).items())),
            "blocked_observations": len(prepared) - len(eligible),
            "unavailable_observation_count": len(unavailable_rows),
            "blocked_groups": blocked_groups, "unavailable_observations": unavailable_rows,
            "all_supplied_rows_accounted_for": len(prepared) == len(output) + sum(g["blocked_observation_count"] for g in blocked_groups),
            "history_updates": sum(group_counts.values()), "group_counts": dict(group_counts),
            "max_hosts_in_one_group": max_hosts, "label_filter_applied": False,
            "strictly_past_history": all(r["max_history_available_at"] is None or
                r["max_history_available_at"] < r["effective_available_at"] for r in output),
            "group_membership_completeness": "Every supplied row processed or explicitly whole-group blocked; verify supplied groups against the extraction manifest.",
            "memory_scope": "Constant context state per host; batch sorting and returned rows require O(N) memory.",
        },
    }


def _text_open(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Complete declared observation JSONL or JSONL.gz")
    parser.add_argument("output", type=Path, help="Context JSONL or JSONL.gz")
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--time-mode", choices=TIME_MODES, default="available_at")
    parser.add_argument("--unavailable-policy", choices=UNAVAILABLE_POLICIES, default="error")
    args = parser.parse_args()
    with _text_open(args.input, "rt") as source:
        result = build_prefix_context((json.loads(line) for line in source if line.strip()),
            time_mode=args.time_mode, unavailable_policy=args.unavailable_policy)
    with _text_open(args.output, "wt") as destination:
        for row in result.pop("rows"):
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    args.audit.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
