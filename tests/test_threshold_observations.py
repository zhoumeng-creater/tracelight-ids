import gzip

import numpy as np
import pandas as pd
import pytest

from ids.threshold_validation import observations as obs


def write_log(path, rows):
    fields = [
        "ts",
        "id.orig_h",
        "id.resp_h",
        "id.orig_p",
        "id.resp_p",
        "proto",
        "duration",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "tunnel_parents",
    ]
    with gzip.open(path, "wt", encoding="utf8") as f:
        f.write("#separator \\x09\n#fields\t" + "\t".join(fields) + "   label   detailed-label\n")
        f.write("#types\t" + ("\t".join(["string"] * len(fields))) + "   string   string\n")
        for t, src, dst, port, proto, dur, response, label, detail in rows:
            values = [t, src, dst, "1024", port, proto, dur, "1", response, "60", "0", "-"]
            f.write("\t".join(map(str, values)) + f"   {label}   {detail}\n")
        f.write("#close\tend\n")


def test_physical_clock_strict_prefix_across_chunks_and_missing_reply(tmp_path, monkeypatch):
    rows = [
        ("1000.000000", "s", "a", 80, "tcp", "-", "-", "benign", "-"),
        ("999.998123", "s", "b", 80, "tcp", "0", "-", "Malicious", "PartOfAHorizontalPortScan"),
        ("1000.000001", "s", "c", 81, "tcp", "0", 0, "Malicious", "C&C-PartOfAHorizontalPortScan"),
        ("1000.001001", "s", "a", "-", "tcp", "-", 1, "Malicious", "C&C"),
        ("1000.001001", "s", "b", 82, "udp", "0", 0, "Malicious", "DDoS"),
        ("1000.000013", "t", "b", 82, "tcp", "0.001989", 0, "Other", "-"),
        ("1060", "s", "d", 83, "tcp", "-", 1, "Malicious", "-"),
        ("1060.001", "s", "a", 84, "tcp", "-", 0, "Benign", "-"),
    ]
    monkeypatch.setattr(obs, "CHUNK_ROWS", 2)
    p, out = tmp_path / "conn.gz", tmp_path / "output.pq"
    write_log(p, rows)
    audit = obs.prepare(p, out)
    df = pd.read_parquet(out)
    assert df.row_id.tolist() == list(range(1, 9))
    assert df.flow_end_ms.tolist() == [1000000, 1000000, 1000001, 1000002, 1000002, 1000003, 1060000, 1060001]
    assert df.family.tolist() == [
        "Benign",
        "Scanning",
        "Mixed",
        "C&C",
        "OtherAttack",
        "Unresolved",
        "Unresolved",
        "Benign",
    ]
    assert audit["missing_or_unknown"]["resp_pkts"] == 2
    assert (
        audit["physical_end_adjacent_decreases"] == audit["physical_end_clamped_to_previous_watermark"] == 1
    )
    np.testing.assert_allclose(
        df.loc[:, obs.HISTORY_FEATURES].to_numpy(),
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 2, 1, -1],
            [3, 3, 2, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 3, 2, 0.5],
            [3, 3, 2, 1 / 3],
        ],
    )
    assert df.orig_port.iloc[0] == 1024 and df.resp_port.iloc[3] == -1
    assert df.duration_seconds.iloc[0] == -1 and df.resp_pkts.iloc[0] == -1
    assert not {"source", "destination", "uid", "id.orig_h", "id.resp_h"}.intersection(df.columns)
    assert not set(obs.RAW_FEATURES + obs.HISTORY_FEATURES).intersection(
        ["native_label", "native_detail", "family", "row_id", "flow_end_ms"]
    )
    assert df.iloc[2].native_detail == "C&C-PartOfAHorizontalPortScan"


def test_future_and_class_names_do_not_change_observations(tmp_path):
    p, out = tmp_path / "conn.gz", tmp_path / "output.parquet"
    rows = [
        ("1666666666.999015", "::1", "abc", 80, "tcp", "0.000985", 1, "Benign", "-"),
        ("1666666667.000005", "::1", "def", 80, "tcp", "0", 0, "Malicious", "PartOfAHorizontalPortScan"),
    ]
    write_log(p, rows)
    obs.prepare(p, out)
    first = pd.read_parquet(out)
    assert first.flow_end_ms.iloc[0] == 1666666667000
    rows[0] = (*rows[0][:-2], "Malicious", "C&C-Torii")
    rows.append(("1666666668", "::1", "xyz", 81, "tcp", "0", 0, "Malicious", "C&C"))
    write_log(p, rows)
    obs.prepare(p, out)
    second = pd.read_parquet(out).iloc[:2]
    cols = ["flow_end_ms", *obs.RAW_FEATURES, *obs.HISTORY_FEATURES]
    pd.testing.assert_frame_equal(first[cols], second[cols])


def test_random_histories_match_direct_prior_row_reference():
    rng = np.random.default_rng(997)
    clock = np.cumsum(rng.choice([0, 1, 333, 30000, 60000], size=180))
    keys = rng.integers(0, 4, len(clock))
    dst = rng.integers(1, 8, len(clock))
    port = rng.choice([-1, 0, 22, 80], len(clock))
    replied = rng.choice([-1, 0, 1, 2], len(clock))
    hist = np.column_stack(obs._history(clock, keys, dst, port, replied))
    for i, t in enumerate(clock):
        mask = (keys == keys[i]) & (clock >= t - 60000) & (clock < t)
        known = mask & (replied >= 0)
        ratio = (replied[known] == 0).mean() if known.any() else (-1 if mask.any() else 0)
        np.testing.assert_allclose(
            hist[i], (mask.sum(), len(set(dst[mask])), len(set(port[mask & (port >= 0)])), ratio)
        )


def test_failed_read_preserves_previous_destination(tmp_path):
    p, out = tmp_path / "conn.gz", tmp_path / "out.pq"
    write_log(p, [("bad-time", "s", "t", 80, "tcp", 0, 0, "Benign", "-")])
    out.write_bytes(b"previous-complete")
    with pytest.raises(ValueError):
        obs.prepare(p, out)
    assert out.read_bytes() == b"previous-complete"


@pytest.mark.parametrize(
    ("coarse", "fine", "expected"),
    [
        ("benign", "-", "Benign"),
        ("Malicious", "C&C", "C&C"),
        ("Malicious", "PartOfAHorizontalPortScan", "Scanning"),
        ("Malicious", "C&C-FileDownload", "Mixed"),
        ("Malicious", "C&C-HeartBeat", "Mixed"),
        ("Malicious", "PartOfAHorizontalPortScan-Attack", "Mixed"),
        ("Malicious", "C&C-PartOfAHorizontalPortScan", "Mixed"),
        ("Malicious", "Okiru-Attack", "OtherAttack"),
        ("Benign", "C&C", "Unresolved"),
    ],
)
def test_exact_native_endpoints(coarse, fine, expected):
    assert obs._family(coarse, fine) == expected
