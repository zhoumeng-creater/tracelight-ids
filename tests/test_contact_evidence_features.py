import numpy as np
import pandas as pd
import pytest

from ids.contact_evidence.features import ARMS, NEW_FEATURES, enrich_prepared
from ids.contact_evidence.online import OnlineHistory, benchmark_online
from ids.threshold_validation.observations import RAW_FEATURES, HISTORY_FEATURES


def _reference(frame, i):
    record = frame.iloc[i]
    previous = frame.iloc[:i]
    base = previous.loc[
        previous.flow_end_ms.between(record.flow_end_ms - 60_000, record.flow_end_ms, inclusive="left")
        & previous.source.eq(record.source)
        & previous.native_protocol.eq(record.native_protocol)
    ]
    valid = base.resp_pkts.ge(0)
    missing = -1.0 if len(base) else 0.0
    values = [
        len(base),
        base.destination.nunique(),
        base.loc[base.resp_port.ge(0), "resp_port"].nunique(),
        base.loc[valid, "resp_pkts"].eq(0).mean() if valid.any() else missing,
    ]
    service = base.loc[base.resp_port.eq(record.resp_port) & base.resp_port.ge(0)]
    for part in (service, service.loc[service.destination.eq(record.destination)]):
        known = part.loc[part.resp_pkts.ge(0), "resp_pkts"]
        values.extend([len(part), known.gt(0).mean() if len(known) else -1.0])
    return values


def _trace():
    rng = np.random.default_rng(547)
    # The long tie straddles many chunks, followed by an exact lower endpoint then one ms beyond.
    times = [0] * 12 + [1, 1, 2, 100, 59999, 60000, 60001, 60001, 61000] + [200000] * 5
    n = len(times)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(1, n + 1),
            "flow_end_ms": times,
            "source": rng.choice(["a", "b"], n),
            "destination": rng.choice(["x", "y"], n),
            "native_protocol": rng.choice(["tcp", "udp", "unknown-protocol"], n),
            **{col: np.zeros(n, dtype=float) for col in RAW_FEATURES},
        }
    )
    frame["resp_port"] = rng.choice([-1, 22, 80], n)
    frame["resp_pkts"] = rng.choice([-1, 0, 1, 7], n)
    # One stable service; shared and new targets; all outcomes with missing/zero/positive counts.
    for i, (dst, recv) in enumerate((("x", -1), ("x", 0), ("y", 4), ("y", 4))):
        frame.loc[[i, i + 12], ["source", "native_protocol", "destination", "resp_port", "resp_pkts"]] = [
            "a",
            "tcp",
            dst,
            22,
            recv,
        ]
    expected = np.array([_reference(frame, i) for i in range(n)])
    for j, col in enumerate(HISTORY_FEATURES):
        frame[col] = expected[:, j]
    return frame, expected


@pytest.mark.parametrize("chunk", [1, 4, 99])
def test_disk_history_compares_to_literal_past_at_every_row(tmp_path, chunk):
    frame, expected = _trace()
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    frame.to_parquet(inp, index=False, row_group_size=3)
    assert enrich_prepared(inp, out, chunk=chunk)["rows"] == len(frame)
    recovered = pd.read_parquet(out)
    pd.testing.assert_frame_equal(recovered[list(frame)], frame)
    np.testing.assert_array_equal(recovered[list(NEW_FEATURES)], expected[:, 4:])


@pytest.mark.parametrize("arm", ARMS)
def test_streaming_service_uses_only_prior_completed_records_and_verifies_model(tmp_path, arm):
    frame, _ = _trace()
    inp, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    frame.to_parquet(inp, index=False, row_group_size=3)
    enrich_prepared(inp, out, chunk=5)
    measured = benchmark_online(out, np.sum, arm, rows=10, threshold=23)
    assert measured["rows"] == len(measured["latencies_ns"]) == 10
    assert measured["score_max_abs_error"] == 0
    assert measured["checked_alerts"]
    enriched = pd.read_parquet(out)
    expected_alerts = []
    for block in measured["blocks"]:
        start = block["first_row_id"] - 1
        matrix = enriched.iloc[start : start + block["rows"]][list(ARMS[arm])].to_numpy()
        expected_alerts.extend((matrix.sum(axis=1) > 23).tolist())
    assert measured["alerts"] == expected_alerts
    assert "alerts" not in benchmark_online(out, np.sum, arm, rows=1)
    assert len(measured["blocks"]) == 3
    if arm != "raw":
        assert sum(block["warm_rows"] for block in measured["blocks"]) > 0


@pytest.mark.parametrize("issue", ["label", "family", "row_id", "clock"])
def test_rejected_input_preserves_earlier_output(tmp_path, issue):
    frame, _ = _trace()
    if issue in {"label", "family"}:
        frame[issue] = "unused"
    elif issue == "clock":
        frame.loc[18, "flow_end_ms"] = 0
    else:
        frame.loc[3, "row_id"] = 9
    inp, out = tmp_path / "bad.parquet", tmp_path / "existing"
    out.write_bytes(b"previous")
    frame.to_parquet(inp, index=False)
    with pytest.raises(ValueError):
        enrich_prepared(inp, out, chunk=4)
    assert out.read_bytes() == b"previous"
    assert sorted(file.name for file in tmp_path.iterdir()) == ["bad.parquet", "existing"]


def test_tie_never_sees_earlier_peers_in_the_same_millisecond_or_unknown_port():
    st = OnlineHistory("contact")
    for _ in range(50):
        assert st.observe(10, "x", "tcp", "s", 80, 1)[-4:] == (0, -1, 0, -1)
    assert len(st.pending) == 1
    assert st.observe(11, "x", "tcp", "other", 80, 0)[-4:] == (50, 1, 0, -1)
    assert st.observe(11, "x", "tcp", "s", 80, 0)[-4:] == (50, 1, 50, 1)
    assert st.observe(11, "x", "tcp", "s", -1, 0)[-4:] == (0, -1, 0, -1)
    assert st.observe(60010, "x", "tcp", "s", 80, 0)[-4:] == (52, 50 / 52, 51, 50 / 51)
    assert st.observe(60011, "x", "tcp", "s", 80, 0)[-4:] == (3, 0, 2, 0)
    with pytest.raises(ValueError, match="backwards"):
        st.observe(7, "x", "tcp", "s", 80, 0)
