"""Evidence available when one IoT-23 conn.log record is emitted."""

import numpy as np
import pandas as pd

from ids.threshold_validation.observations import _numbers

QUALITY_INPUTS = (
    "history", "conn_state", "missed_bytes", "proto", "orig_pkts", "resp_pkts",
    "orig_ip_bytes", "resp_ip_bytes",
)
QUALITY_COLUMNS = ("reply_unknown", "reply_unanswered_observed", "reply_counted")
GAP_COLUMNS = ("orig_checksum_uncounted", "resp_checksum_uncounted")


def classify_response(raw: pd.DataFrame) -> np.ndarray:
    """Return -1 unknown, 0 outbound activity only, 1 counted responder packet.

    Class 0 is deliberately narrow. A TCP SYN or UDP payload must be recorded on
    the origin side, with no responder or anomaly event. Zeek's gap counter is
    an additional veto; zero is never interpreted as proof of lossless capture.
    """
    history = raw["history"].fillna("")
    protocol = raw["proto"]
    state = raw["conn_state"].eq("S0")
    # Full matches exclude flips, checksum errors, gaps and opposite-side events.
    tcp_syn_only = protocol.eq("tcp") & history.str.fullmatch("S+", na=False)
    udp_data_only = protocol.eq("udp") & history.str.fullmatch("D+", na=False)
    outbound = _numbers(raw["orig_pkts"], "orig_pkts", integer=True)
    inbound = _numbers(raw["resp_pkts"], "resp_pkts", integer=True)
    sent_bytes = _numbers(raw["orig_ip_bytes"], "orig_ip_bytes", integer=True)
    recv_bytes = _numbers(raw["resp_ip_bytes"], "resp_ip_bytes", integer=True)
    missed = _numbers(raw["missed_bytes"], "missed_bytes", integer=True)
    recorded_out = (outbound > 0) & (sent_bytes > 0) & (inbound == 0) & (recv_bytes == 0)
    compatible = (state & (tcp_syn_only | udp_data_only)).fillna(False).to_numpy(dtype=bool)
    code = np.full(len(raw), -1, dtype=np.int8)
    code[recorded_out & compatible & (missed == 0)] = 0
    code[inbound > 0] = 1
    return code


def quality_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """One indicator per state; keep the physical log row's index."""
    code = classify_response(raw)
    return pd.DataFrame(
        {name: (code == value).astype(np.int8) for value, name in zip((-1, 0, 1), QUALITY_COLUMNS)},
        index=raw.index,
    )


def counter_gap_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flag a directional checksum event despite an explicitly zero packet counter."""
    history = raw["history"].fillna("")
    flags = {}
    for column, count_name, symbol in zip(GAP_COLUMNS, ("orig_pkts", "resp_pkts"), ("C", "c")):
        count = _numbers(raw[count_name], count_name, integer=True)
        marked = history.str.contains(symbol, regex=False, na=False).to_numpy(dtype=bool)
        flags[column] = (marked & (count == 0)).astype(np.int8)
    return pd.DataFrame(flags, index=raw.index)
