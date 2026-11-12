import numpy as np
import pandas as pd
import pytest

from ids.response_evidence.history import ADDED, CODE, QUALITY_HISTORY_COLUMNS, enrich_prepared
from ids.response_evidence.history_benchmark import benchmark_history


def _flow(n=298):
    rng = np.random.default_rng(234)
    t = [0] * 32 + [1, 59999, 60000, 60001, 60002] + [120002] * 9
    t += list(np.sort(rng.integers(120002, 500000, n - len(t))))
    return pd.DataFrame(
        {
            "row_id": np.arange(n) + 1,
            "flow_end_ms": t,
            "source": rng.choice(["a", "a", "b"], n),
            "destination": rng.choice(["x", "y", "z"], n),
            "native_protocol": rng.choice(["icmp", "tcp", "udp"], n),
            "resp_port": rng.choice([80, 80, -1, 22], n),
            "resp_pkts": rng.choice([-1, 0, 2, 9], n),
            CODE: rng.choice([-1, 0, 1], n).astype("int8"),
        }
    )


def _quality_expected(df, index):
    row = df.iloc[index]
    mask = (
        df.flow_end_ms.between(row.flow_end_ms - 60000, row.flow_end_ms, inclusive="left")
        & df.source.eq(row.source)
        & df.native_protocol.eq(row.native_protocol)
    )
    expected = []
    for scope in ("src", "svc", "contact"):
        if scope == "svc":
            mask &= df.resp_port.eq(row.resp_port) & (df.resp_port >= 0)
        if scope == "contact":
            mask &= df.destination.eq(row.destination)
        code = df.loc[mask, CODE]
        known = code.loc[code >= 0]
        expected.extend([known.eq(0).mean() if len(known) else -1, code.lt(0).mean() if len(code) else -1])
    return expected


@pytest.mark.parametrize("workers,block_rows", [(1, 9), (4, 13), (4, 4096)])
def test_quality_and_legacy_history_on_every_row(tmp_path, workers, block_rows):
    df = _flow()
    path, output = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(path, index=False, row_group_size=5)
    audit = enrich_prepared(path, output, workers=workers, block_rows=block_rows)
    assert audit["rows"] == len(df)
    result = pd.read_parquet(output)
    pd.testing.assert_frame_equal(result[list(df)], df)
    np.testing.assert_array_equal(
        result[list(QUALITY_HISTORY_COLUMNS)], [_quality_expected(df, i) for i in range(len(df))]
    )
    bench = benchmark_history(path, rows=len(df), workers=workers, block_rows=block_rows)
    assert len(bench["all_rows_legacy_column_sha256"]) == 9
    assert len(bench["all_rows_serial_column_sha256"]) == 1 + len(ADDED)


def test_hot_host_all_tied_no_history_expiry_boundary(tmp_path):
    df = _flow(32768)
    df["flow_end_ms"] = [18] * 32765 + [60018, 60019, 60019]
    df["source"], df["native_protocol"], df["resp_port"], df["destination"] = "a", "tcp", 5, "b"
    df[CODE] = -1
    path, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(path, index=False, row_group_size=417)
    audit = enrich_prepared(path, out, workers=4, block_rows=253)
    result = pd.read_parquet(out)
    assert audit["blocks"] <= 3
    assert not result.hist_flows_60s.iloc[:32765].any()
    assert result.hist_flows_60s.iloc[-3:].tolist() == [32765, 1, 1]
    assert result.contact_prior_unknown_fraction_60s.iloc[-3:].tolist() == [1, 1, 1]
    assert result.src_prior_observed_no_reply_fraction_60s.eq(-1).all()


@pytest.mark.parametrize("fault", ["time", "ids", "evidence", "label", "null"])
def test_invalid_stream_cannot_replace_completed_output(tmp_path, fault):
    df = _flow()
    if fault == "time":
        df.loc[35, "flow_end_ms"] = 1
    if fault == "ids":
        df.loc[4, "row_id"] = 7
    if fault == "evidence":
        df.loc[4, CODE] = 2
    if fault == "label":
        df["family"] = "Scanning"
    if fault == "null":
        df.loc[9, "source"] = None
    path, out = tmp_path / "in.parquet", tmp_path / "out.parquet"
    df.to_parquet(path, index=False)
    out.write_bytes(b"previous")
    with pytest.raises(ValueError):
        enrich_prepared(path, out)
    assert out.read_bytes() == b"previous"
