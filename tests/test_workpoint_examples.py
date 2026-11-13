import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json
from ids.raw_workpoint.evaluate import ARMS
from ids.raw_workpoint.examples import collect
from ids.response_evidence.experiment import FULL


def test_examples_preserve_all_rare_flows_and_bound_common_groups(tmp_path):
    ids = np.arange(1, 501)
    rare = ids == 103
    features = pa.table({"row_id": ids, "flow_end_ms": ids, **{col: ids for col in FULL}})
    pred = pa.table(
        {
            "row_id": ids,
            **{col: rare for col in ARMS},
            **{"score_" + col: ids.astype(float) for col in ("raw", "contact", "quality")},
        }
    )
    truth = pa.table(
        {
            "row_id": ids,
            "family": np.where(rare, "Mixed", "Benign"),
            "native_detail": np.where(rare, "C&C-FileDownload", "-"),
        }
    )
    paths = []
    for name, table in [("pred", pred), ("truth", truth), ("features", features)]:
        path = tmp_path / (name + ".parquet")
        pq.write_table(table, path, row_group_size=19)
        paths.append(path)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    collect(*paths, a, batch_rows=27)
    collect(*paths, b, batch_rows=83)
    sample = pq.read_table(a)
    assert sample.equals(pq.read_table(b))
    assert sample.num_rows == 33
    assert set(sample["stratum_rows"].to_pylist()) == {1, 499}
    assert 103 in sample["row_id"].to_pylist()
    assert read_json(a.with_suffix(".json"))["total_scored"] == 500
