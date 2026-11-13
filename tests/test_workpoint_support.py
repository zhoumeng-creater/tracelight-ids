import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ids.raw_workpoint.evaluate import ARMS
from ids.raw_workpoint.support import summarize


def inputs(root, corrupt=False):
    ids = np.arange(1, 11)
    raw = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0], dtype=bool)
    proto = [6, 6, 6, 6, 6, 17, 17, 6, 6, 17]
    port = [80, 80, 80, 80, 80, 443, 443, 443, 80, 53]
    size = [70000, 70000, 10, 70000, 90, 80000, 0, 2, 70000, 0]
    web = (np.array(proto) == 6) & (np.array(port) == 80) & (np.array(size) >= 65536)
    decisions = {a: raw.copy() for a in ARMS}
    decisions["add_contact"][[0, 3, 5]] = True
    decisions["add_web64k"] = raw | web
    if corrupt:
        decisions["add_web64k"][0] = False
    blind = pa.table(
        {
            "row_id": ids,
            "flow_end_ms": np.r_[np.arange(8), 86400010, 172800010],
            "ip_protocol": proto,
            "resp_port": port,
            "resp_ip_bytes": size,
        }
    )
    pred = pa.table({"row_id": ids, **decisions})
    truth = pa.table(
        {
            "row_id": ids,
            "family": [
                "Benign",
                "Benign",
                "Benign",
                "Mixed",
                "OtherAttack",
                "Scanning",
                "Benign",
                "Benign",
                "C&C",
                "OtherAttack",
            ],
            "native_detail": [
                "-",
                "-",
                "-",
                "C&C-FileDownload",
                "FileDownload",
                "PartOfAHorizontalPortScan",
                "-",
                "-",
                "C&C",
                "DDoS",
            ],
        }
    )
    paths = [root / (n + ".parquet") for n in ("pred", "truth", "blind")]
    for path, table, group in zip(paths, (pred, truth, blind), (4, 2, 3), strict=True):
        pq.write_table(table, path, row_group_size=group)
    return paths


def test_support_matches_same_service_and_independent_capture_control(tmp_path):
    paths = inputs(tmp_path)
    dest = tmp_path / "support.csv.gz"
    assert summarize(*paths, dest, batch_rows=3) == [dest]
    frame = pd.read_csv(dest)
    again = tmp_path / "again.csv.gz"
    summarize(*paths, again, batch_rows=1)
    pd.testing.assert_frame_equal(frame, pd.read_csv(again))
    day = frame[
        (frame.scope == "utc_day") & (frame.utc_day == "1970-01-01") & (frame.record_type == "service")
    ]
    download = day[day.endpoint.eq("Download")].sort_values("native_detail")
    assert download.rows_normal.tolist() == [3, 3]
    assert download.raw_negative_normal.tolist() == [2, 2]
    assert download.web64k_trigger_normal.tolist() == [2, 2]
    assert download.add_web64k_gain_normal.tolist() == [1, 1]
    assert download.port80_gain_normal.tolist() == [2, 2]
    assert download.add_web64k_gain.tolist() == [1, 0]
    assert download.add_contact_gain.tolist() == [1, 0]
    assert download.port80_gain.tolist() == [1, 1]
    assert set(download.native_detail) == {"C&C-FileDownload", "FileDownload"}
    scan = day[day.endpoint.eq("Scanning")].iloc[0]
    assert (scan.ip_protocol, scan.resp_port, scan.rows_normal, scan.raw_negative_normal) == (17, 443, 1, 0)
    cc = frame[(frame.utc_day == "1970-01-02") & (frame.record_type == "service")].iloc[0]
    assert cc.rows_normal == 0 and cc.raw_negative == 1
    capture = frame[
        (frame.scope == "capture") & (frame.record_type == "port80_control") & (frame.endpoint == "Benign")
    ].iloc[0]
    assert (
        capture.rows_normal,
        capture.raw_fp_normal,
        capture.raw_or_port80_fp_normal,
        capture.raw_or_web64k_fp_normal,
    ) == (5, 2, 4, 3)
    unsupported = frame[(frame.record_type == "port80_control") & (frame.utc_day == "1970-01-03")]
    assert len(unsupported) == 1 and unsupported.iloc[0].rows_normal == 0


def test_refuses_corrupt_sealed_rule_or_shifted_id(tmp_path):
    paths = inputs(tmp_path, corrupt=True)
    dest = tmp_path / "bad.csv.gz"
    with pytest.raises(ValueError, match="web64k"):
        summarize(*paths, dest, batch_rows=4)
    paths = inputs(tmp_path)
    t = pq.read_table(paths[1])
    t = t.set_column(0, "row_id", pa.array([2, 1, *range(3, 11)]))
    pq.write_table(t, paths[1])
    with pytest.raises(ValueError, match="physical IDs"):
        summarize(*paths, dest, batch_rows=3)
    assert not dest.exists()


def test_no_target_capture_has_full_normal_control(tmp_path):
    paths = inputs(tmp_path)
    t = pq.read_table(paths[1])
    t = t.set_column(1, "family", pa.array(["Benign"] * 10))
    pq.write_table(t, paths[1])
    dest = tmp_path / "normals.csv.gz"
    summarize(*paths, dest, batch_rows=4)
    frame = pd.read_csv(dest)
    assert set(frame.record_type) == {"port80_control"}
    assert frame.loc[frame.scope.eq("capture"), "rows_normal"].item() == 10
