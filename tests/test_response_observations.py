import numpy as np
import pandas as pd
import pytest

from ids.response_evidence.observations import QUALITY_COLUMNS, classify_response, counter_gap_frame, quality_frame


def record(**changes):
    return dict(dict(history="S", conn_state="S0", proto="tcp", orig_pkts="2", resp_pkts="0",
                     orig_ip_bytes="80", resp_ip_bytes="0", missed_bytes="0"), **changes)


@pytest.mark.parametrize(("changes", "expected"), [
    ({}, 0),
    ({"history": "D", "proto": "udp"}, 0),
    ({"history": "SS"}, 0),
    ({"history": "ShA", "resp_pkts": "3", "resp_ip_bytes": "156", "conn_state": "SF"}, 1),
    ({"history": "ShADadcgttfF", "missed_bytes": "1448", "resp_pkts": "78"}, 1),
    ({"history": "^c", "conn_state": "OTH", "orig_pkts": "0", "orig_ip_bytes": "0"}, -1),
    ({"history": "C", "conn_state": "OTH", "orig_pkts": "0", "orig_ip_bytes": "0"}, -1),
    ({"missed_bytes": "-"}, -1),
    ({"history": None}, -1),
    ({"missed_bytes": "1"}, -1),
    ({"orig_pkts": "0"}, -1),
    ({"resp_pkts": "-"}, -1),
    ({"resp_ip_bytes": "40"}, -1),
    ({"history": "Sh"}, -1),
    ({"history": "Sg"}, -1),
    ({"history": "Sd"}, -1),
    ({"history": "^S"}, -1),
    ({"conn_state": "OTH"}, -1),
    ({"proto": "icmp"}, -1),
])
def test_completed_record_state(changes, expected):
    raw = pd.DataFrame([record(**changes)], index=[782])
    state = classify_response(raw)
    assert state.dtype == np.int8
    assert state.tolist() == [expected]
    indicators = quality_frame(raw)
    assert list(indicators.columns) == list(QUALITY_COLUMNS)
    assert indicators.index.tolist() == [782]
    assert indicators.iloc[0].tolist() == [int(expected == code) for code in (-1, 0, 1)]


def test_labels_and_addresses_cannot_change_state():
    raw = pd.DataFrame([record(), record()])
    raw["label"] = ["Malicious", "Benign"]
    raw["id.resp_h"] = ["192.0.2.1", "192.0.2.2"]
    assert classify_response(raw).tolist() == [0, 0]


def test_directional_checksum_events_require_an_explicit_zero_counter():
    raw = pd.DataFrame([
        record(history="^c", orig_pkts="0"), record(history="C", orig_pkts="0"),
        record(history="Cc", orig_pkts="-", resp_pkts="1"), record(history="-", orig_pkts="0"),
    ])
    assert counter_gap_frame(raw).values.tolist() == [[0, 1], [1, 0], [0, 0], [0, 0]]
