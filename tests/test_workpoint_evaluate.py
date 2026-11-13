import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ids.common import read_json
from ids.raw_workpoint.evaluate import ARMS, evaluate


def sources(root, *, bad_id=False, bad_pred=False, only_attacks=False):
    t = pa.table(
        dict(
            row_id=[1, 2, 3, 4, 5],
            family=["Mixed", "OtherAttack", "C&C", "Benign", "Benign"],
            native_label=["Malicious"] * 3 + ["Benign"] * 2,
            native_detail=["C&C-FileDownload", "FileDownload", "C&C", "-", "-"],
            window=["author_cell"] * 5,
        )
    )
    if only_attacks:
        t = t.set_column(1, "family", pa.array(["OtherAttack"] * 5))
    raw = np.array([True, False, False, True, False])
    mask = np.array([True, True, False, True, False])
    decisions = {arm: mask.copy() for arm in ARMS}
    decisions["raw"] = raw
    decisions["contact"] = ~raw
    if bad_pred:
        decisions["add_contact"][0] = False
    p = pa.table({"row_id": [1, 2, 3, 4, 5], **decisions})
    b = pa.table(
        {"row_id": [1, 2, 3, 5, 4] if bad_id else [1, 2, 3, 4, 5], "flow_end_ms": [0, 12, 23, 1445, 86400001]}
    )
    paths = [root / (name + ".parquet") for name in ("pred", "truth", "blind")]
    for file, table, size in zip(paths, (p, t, b), (3, 1, 2), strict=True):
        pq.write_table(table, file, row_group_size=size)
    return paths


def test_native_and_utc_costs_are_stable_across_batch_sizes(tmp_path):
    paths = sources(tmp_path)
    a = evaluate(*paths, tmp_path / "capture.json", batch_rows=3)
    b = evaluate(*paths, tmp_path / "small.json", batch_rows=1)
    assert a == b == read_json(tmp_path / "capture.json")
    assert a["rows"] == 5 == sum(day["rows"] for day in a["utc_days"].values())
    capture = a["capture"]["comparisons"]
    assert capture["contact"]["endpoints"]["Download"] == dict(
        rows=2, before=1, after=1, gained=1, lost=1, net=0
    )
    assert capture["add_quality"]["endpoints"]["Download"]["after"] == 2
    assert capture["add_web64k"]["endpoints"]["Download"]["gained"] == 1
    assert a["utc_days"]["1970-01-01"]["comparisons"]["raw"]["raw_within_budget"] is False
    assert a["utc_days"]["1970-01-02"]["comparisons"]["raw"]["raw_within_budget"] is True
    assert {row["native_detail"] for row in a["capture"]["native"]} == {
        "C&C-FileDownload",
        "FileDownload",
        "C&C",
        "-",
    }
    assert capture["add_contact"]["endpoints"]["Benign"]["after"] == 1


@pytest.mark.parametrize("arg,error", [("bad_id", "row IDs"), ("bad_pred", "removed")])
def test_reject_misaligned_rows_or_erased_raw_alert(tmp_path, arg, error):
    paths = sources(tmp_path, **{arg: True})
    with pytest.raises(ValueError, match=error):
        evaluate(*paths, tmp_path / "bad.json", batch_rows=4)
    assert not (tmp_path / "bad.json").exists()


def test_absent_normal_support_has_no_pass_status(tmp_path):
    paths = sources(tmp_path, only_attacks=True)
    result = evaluate(*paths, tmp_path / "no-normal.json")
    assert result["capture"]["comparisons"]["quality"]["both_within_budget"] is None
    assert result["capture"]["comparisons"]["raw"]["allowed_fp"] is None
