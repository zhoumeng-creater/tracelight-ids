"""Prefix-causal rolling port/target proxy checked against a slow independent oracle."""

import numpy as np
import pandas as pd
import pytest

from ids.behavior_transfer.scan_history import build_scan_history, COUNTS


def frame(n=36):
    f = pd.DataFrame(
        {
            "source": ["s"] * n,
            "destination": ["d"] * n,
            "native_protocol": ["tcp"] * n,
            "resp_port": np.arange(n) + 1,
            "flow_end_ms": np.arange(n) * 1000,
            "response_evidence_code": [0] * n,
        }
    )
    return f


def brute(f):
    rows = []
    for _, r in f.iterrows():
        old = f.loc[
            (f.flow_end_ms < r.flow_end_ms)
            & (f.flow_end_ms >= r.flow_end_ms - 300000)
            & f.source.eq(r.source)
            & f.native_protocol.eq(r.native_protocol)
            & f.native_protocol.eq("tcp")
            & f.response_evidence_code.eq(0)
            & f.resp_port.ge(0)
        ]
        rows.append(
            (
                old.loc[old.destination.eq(r.destination), "resp_port"].nunique(),
                old.loc[old.resp_port.eq(r.resp_port), "destination"].nunique(),
            )
        )
    return np.asarray(rows, dtype=np.int32)


def test_default_mapping_and_censoring():
    f = frame()
    f.loc[16, "response_evidence_code"] = -1
    f.loc[17, "response_evidence_code"] = 1
    f.loc[18, "native_protocol"] = "udp"
    f.loc[19, "resp_port"] = -1
    b = build_scan_history(f, 2)
    np.testing.assert_array_equal(b[list(COUNTS)], brute(f))
    assert not b.scan_port_alert.iloc[:15].any() and b.scan_port_alert.iloc[15]
    assert not b.scan_proxy.iloc[16:20].any() and b.scan_proxy.iloc[20]
    assert b.attrs["evidence"]["unknown_response_rows"] == 1
    assert b.attrs["evidence"]["eligible_records"] == 32
    target = frame(27).assign(resp_port=80, destination=[f"d{i}" for i in range(27)])
    c = build_scan_history(target)
    assert not c.scan_address_alert.iloc[:25].any() and c.scan_address_alert.iloc[25:].all()
    assert not c.scan_port_alert.any()


def test_same_time_future_prefix_and_window_edges_against_independent_loop():
    f = frame(90)
    f["source"] = np.tile(["a", "b", "a"], 30)
    f["destination"] = np.tile(["one", "one", "two", "two", "one"], 18)
    f["resp_port"] = np.arange(90) % 9
    f["flow_end_ms"] = np.repeat(np.arange(30) * 60000, 3)
    f.loc[[0, 4, 8], "response_evidence_code"] = -1
    f.loc[[10, 20], "response_evidence_code"] = 1
    b = build_scan_history(f, 2)
    np.testing.assert_array_equal(b[list(COUNTS)], brute(f))
    for n in (1, 2, 17, 31, 88):
        # Includes a prefix terminating inside an equal-clock group.
        c = build_scan_history(f.iloc[:n], 1)
        np.testing.assert_array_equal(b.iloc[:n], c)
    edge = frame(4)
    edge["flow_end_ms"] = [0, 0, 300000, 300001]
    e = build_scan_history(edge)
    assert e.scan_failed_ports_5m.tolist() == [0, 0, 2, 1]


def test_empty_input_and_invalid_metadata():
    f = frame()
    assert build_scan_history(f.iloc[:0]).empty
    with pytest.raises(ValueError, match="nondecreasing"):
        build_scan_history(f.iloc[::-1])
    f.loc[0, "source"] = None
    with pytest.raises(ValueError, match="Missing"):
        build_scan_history(f)
