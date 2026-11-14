"""Connection-log proxy for the historical Zeek 6 TCP port/address scan defaults."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd

WINDOW_MS = 300_000
PORT_THRESHOLD = 15
ADDRESS_THRESHOLD = 25
COUNTS = ("scan_failed_ports_5m", "scan_failed_targets_5m")
ALERTS = ("scan_port_alert", "scan_address_alert", "scan_proxy")
REQUIRED = ("source", "destination", "native_protocol", "resp_port", "flow_end_ms", "response_evidence_code")
SOURCE = "https://raw.githubusercontent.com/zeek/zeek/v6.0.9/scripts/policy/misc/scan.zeek"


def _prior_distinct(clock, keys, values, eligible):
    """Count value lifetimes within independent key groups using compiled NumPy sorting."""
    order = np.lexsort(tuple(reversed(keys)))
    change = np.zeros(max(0, len(order) - 1), dtype=bool)
    for key in keys:
        ordered = key[order]
        change |= ordered[1:] != ordered[:-1]
    delta = np.diff(clock[order])
    delta[change] = WINDOW_MS + 1
    axis = np.empty(len(order), dtype=np.int64)
    axis[0] = 0
    np.cumsum(delta, out=axis[1:])
    keep = eligible[order]
    result = np.zeros(len(order), dtype=np.int32)
    if not keep.any():
        return result
    t, val = axis[keep], values[order][keep]
    by_value = np.lexsort((t, val))
    t, val = t[by_value], val[by_value]
    breaks = (val[1:] != val[:-1]) | (np.diff(t) > WINDOW_MS)
    starts = np.sort(t[np.r_[True, breaks]])
    ends = np.sort(t[np.r_[breaks, True]] + WINDOW_MS)
    result[order] = starts.searchsorted(axis, side="left") - ends.searchsorted(axis, side="left")
    return result


def build_scan_history(frame, workers=1):
    """Return two prior-five-minute counts and their per-record default alert mapping.

    Only observed outbound-only TCP records enter this proxy's histories. The log
    clock is monotonic, the current record and every equal-clock record are excluded.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    source = frame.loc[:, REQUIRED]
    if source.isna().any().any():
        raise ValueError("Missing scan metadata")
    clock = np.asarray(source.flow_end_ms, dtype=np.int64)
    if not np.array_equal(clock, source.flow_end_ms) or (clock < 0).any() or (np.diff(clock) < 0).any():
        raise ValueError("Expected a nondecreasing integer-millisecond log clock")
    response = source.response_evidence_code.to_numpy()
    if not np.isin(response, [-1, 0, 1]).all():
        raise ValueError("Unknown response evidence code")
    port = np.asarray(source.resp_port, dtype=np.int64)
    if not np.array_equal(port, source.resp_port) or not ((port >= -1) & (port <= 65535)).all():
        raise ValueError("Expected a known port or -1")
    start = perf_counter()
    src = pd.factorize(source.source, sort=False)[0]
    dst = pd.factorize(source.destination, sort=False)[0]
    native = source.native_protocol.astype(str).str.lower()
    proto = pd.factorize(native, sort=False)[0]
    eligible = (response == 0) & native.eq("tcp").to_numpy() & (port >= 0)
    jobs = ((clock, (src, proto, dst), port, eligible), (clock, (src, proto, port), dst, eligible))
    if len(frame) and workers > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(lambda args: _prior_distinct(*args), jobs))
    elif len(frame):
        values = [_prior_distinct(*args) for args in jobs]
    else:
        values = [np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)]
    result = pd.DataFrame(dict(zip(COUNTS, values, strict=True)), index=frame.index)
    result["scan_port_alert"] = eligible & (values[0] >= PORT_THRESHOLD)
    result["scan_address_alert"] = eligible & (values[1] >= ADDRESS_THRESHOLD)
    result["scan_proxy"] = result.scan_port_alert | result.scan_address_alert
    result.attrs["evidence"] = {
        "rows": len(frame),
        "seconds": perf_counter() - start,
        "workers": min(workers, 2),
        "window_ms": WINDOW_MS,
        "window": "[clock-300000,clock); complete same-clock group excluded",
        "port_threshold": PORT_THRESHOLD,
        "address_threshold": ADDRESS_THRESHOLD,
        "comparison": ">=",
        "source": SOURCE,
        "source_status": "Historical script deprecated in Zeek 6.0 and removed in 6.1.",
        "eligible_records": int(eligible.sum()),
        "unknown_response_rows": int((response == -1).sum()),
        "count_input": "TCP response_evidence_code==0, observed outbound with no observed response; valid destination port",
        "flow_mapping": "Current eligible flow plus prior matching-source/victim port count or source/service victim count at threshold. Repeated flows remain flagged while the rolling criterion holds.",
        "differences_from_zeek6": [
            "No packet-level connection_attempt/rejected/reset/pending events or reverse-origin reconstruction.",
            "Known response packets are excluded even when native TCP events would identify a rejected connection.",
            "Rolling prior window replaces SumStats epochs and event-time accounting.",
            "All equal-clock records and current record excluded; native event increments include the triggering connection.",
            "Flow flags replace source-level notices and suppression.",
        ],
    }
    return result
