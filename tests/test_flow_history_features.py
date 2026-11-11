import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from ids.context_cache import lab_data as lab
from ids.flow_history.features import HISTORY_COLUMNS, _extract, extract_history


def flow(end, dst, port=443, out=0, protocol=6, src="1.1.1.1"):
    row = dict.fromkeys(lab.RAW_FEATURES, 0)
    row.update(
        FLOW_START_MILLISECONDS=end - 1,
        FLOW_END_MILLISECONDS=end,
        IPV4_SRC_ADDR=src,
        IPV4_DST_ADDR=dst,
        L4_SRC_PORT=16711,
        L4_DST_PORT=port,
        PROTOCOL=protocol,
        OUT_PKTS=out,
        Label=0,
        Attack="Benign",
    )
    return row


def fixture(tmp_path, rows, target_indices, name="fixture.csv"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    selected = []
    for index in target_indices:
        row = rows[index]
        group = lab._communication_group_from_ids(
            row["IPV4_SRC_ADDR"],
            row["IPV4_DST_ADDR"],
            row["L4_SRC_PORT"],
            row["L4_DST_PORT"],
            row["PROTOCOL"],
        )
        selected.append(
            {
                **{k: row[k] for k in lab.FEATURES},
                "row_id": f"{digest}:{index + 1}",
                "label": row["Label"],
                "attack_family": row["Attack"],
                "communication_group": group,
                "flow_end_ms": row["FLOW_END_MILLISECONDS"],
            }
        )
    return path, pd.DataFrame(selected), {"sha256": digest, "bytes": len(data), "records": len(rows)}


def test_window_same_ms_expiry_group_and_physical_order(tmp_path):
    rows = [
        flow(61000, "3.3.3.3", 81, 1),
        flow(1000, "2.2.2.2"),
        flow(61000, "2.2.2.2", 443),
        flow(61001, "3.3.3.3"),
        flow(120999, "2.2.2.2"),
        flow(181000, "3.3.3.3"),
        flow(60999, "8.8.8.8", protocol=17),
        flow(60999, "9.9.9.9", src="1.1.1.2"),
    ]
    source, targets, expected = fixture(tmp_path, rows, [5, 3, 1, 2, 4, 0, 6])
    with ZipFile(tmp_path / "official.zip", "w") as z:
        z.write(source, "source/data/NF-ToN-IoT-v3.csv")
    out = _extract(tmp_path / "official.zip", targets, tmp_path / "two", 2, expected)
    assert out.row_id.tolist() == targets.row_id.tolist()
    assert out[list(HISTORY_COLUMNS)].values.tolist() == [
        [0, 0, 0, 0],
        [2, 2, 2, 0.5],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [3, 2, 2, 2 / 3],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
    ]
    one = _extract(source, targets, tmp_path / "one", 1, expected)
    pd.testing.assert_frame_equal(out, one)
    audit = json.loads((tmp_path / "two" / "history_audit.json").read_text(encoding="utf-8"))
    assert audit["checked_target_rows"] == 7 and audit["source"]["records"] == 8
    assert audit["source_end_time_inversions"] > 0 and len(audit["shards"]) == 2
    meta = pd.read_parquet(tmp_path / "two" / "target_metadata.parquet")
    assert list(meta.row_id) == list(targets.row_id)
    assert list(meta.end_ms) == list(targets.flow_end_ms)


def test_future_append_labels_address_renaming_and_tie_order(tmp_path):
    a = [flow(100, "8.8.8.8"), flow(100, "7.7.7.7", out=1), flow(200, "7.7.7.7")]
    b = [dict(r) for r in (a[1], a[0], a[2], flow(30000, "7.7.7.7"))]
    for r in b:
        r.update(
            IPV4_SRC_ADDR="3.3.3.3",
            IPV4_DST_ADDR={"8.8.8.8": "2.2.2.2", "7.7.7.7": "5.5.5.5"}[r["IPV4_DST_ADDR"]],
            Label=1,
            Attack="Scanning",
        )

    def run(rows, order, key):
        src, selection, expected = fixture(tmp_path, rows, order, key + ".csv")
        return _extract(src, selection, tmp_path / key, 1, expected)[list(HISTORY_COLUMNS)]

    pd.testing.assert_frame_equal(run(a, [0, 1, 2], "a"), run(b, [1, 0, 2], "b"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("L4_DST_PORT", 80),
        ("label", 1),
        ("attack_family", "Scanning"),
        ("communication_group", "wrong"),
        ("flow_end_ms", 123),
    ],
)
def test_misaligned_target_is_rejected(tmp_path, field, value):
    source, targets, expected = fixture(tmp_path, [flow(100, "2.2.2.2")], [0])
    targets.loc[0, field] = value
    with pytest.raises(ValueError, match="mismatch"):
        _extract(source, targets, tmp_path / "out", 1, expected)
    assert not (tmp_path / "out" / "history_audit.json").exists()


def test_official_identity_and_invalid_times(tmp_path):
    source, targets, expected = fixture(tmp_path, [flow(100, "2.2.2.2")], [0])
    with pytest.raises(ValueError, match="row_id"):
        extract_history(Path(source), targets, tmp_path / "out", 1)
    with pytest.raises(ValueError, match="source identity"):
        _extract(source, targets, tmp_path / "out", 1, {**expected, "records": 2})
    bad = flow(200, "3.3.3.3")
    bad["FLOW_START_MILLISECONDS"] = 201
    source, targets, expected = fixture(tmp_path, [bad], [0], "bad.csv")
    with pytest.raises(ValueError, match="timestamp"):
        _extract(source, targets, tmp_path / "out", 1, expected)


def test_full_warmup_tie_closed_benchmark_and_shard_resume(tmp_path):
    rows = [flow(1000 + i * 125, "8.8.8.8" if i % 3 else "9.9.9.9", 80 + i % 4, i % 5) for i in range(1100)]
    rows[992]["FLOW_END_MILLISECONDS"] = rows[991]["FLOW_END_MILLISECONDS"]
    rows[992]["FLOW_START_MILLISECONDS"] = rows[991]["FLOW_START_MILLISECONDS"]
    source, targets, expected = fixture(tmp_path, rows, range(len(rows)))
    work = tmp_path / "run"
    result = _extract(source, targets, work, 2, expected).set_index("row_id")
    warm = pd.read_parquet(work / "benchmark_warmup.parquet")
    timed = pd.read_parquet(work / "benchmark_stream.parquet")
    recipe = json.loads((work / "benchmark_segment.json").read_text(encoding="utf-8"))
    assert recipe["status"] == "ready" and len(warm) == 480 and len(timed) == 513
    assert warm.flow_end_ms.min() == timed.flow_end_ms.min() - 60000
    assert warm.flow_end_ms.max() < timed.flow_end_ms.min()
    assert timed.source_record.tolist() == list(range(481, 994))
    for _, row in timed.iterrows():
        actual = result.loc[f"{expected['sha256']}:{int(row.source_record)}"]
        assert actual[list(HISTORY_COLUMNS)].tolist() == row[list(HISTORY_COLUMNS)].tolist()
    _extract(source, targets, work, 2, expected)
    audit = json.loads((work / "history_audit.json").read_text(encoding="utf-8"))
    assert audit["shards_reused_after_revalidating_source"] == 2
