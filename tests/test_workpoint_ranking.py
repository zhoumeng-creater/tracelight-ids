import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ids.common import read_json
from ids.raw_workpoint.evaluate import ARMS, evaluate
from ids.raw_workpoint.ranking import diagnose


def test_oracle_ties_and_overbudget_day(tmp_path):
    normal = 100
    size = 2 * (normal + 2)
    seq = np.arange(size)
    day = seq // (normal + 2)
    offset = seq % (normal + 2)
    raw = (offset == 0) | ((day == 1) & (offset == 1))
    attack = offset >= normal
    ids = seq + 1
    paths = [tmp_path / (s + ".parquet") for s in ("pred", "truth", "blind")]
    cols = {arm: raw for arm in ARMS}
    score = np.where(attack, 3.0, 1.0)
    score[offset == normal + 1] = 1.0
    pq.write_table(
        pa.table({"row_id": ids, **cols, **{"score_" + v: score for v in ("raw", "contact", "quality")}}),
        paths[0],
    )
    pq.write_table(
        pa.table(
            {
                "row_id": ids,
                "family": np.where(attack, "C&C", "Benign"),
                "native_label": np.where(attack, "Malicious", "Benign"),
                "native_detail": np.where(attack, "C&C", "-"),
            }
        ),
        paths[1],
    )
    pq.write_table(pa.table({"row_id": ids, "flow_end_ms": day * 86400000}), paths[2])
    summary, ranks = tmp_path / "paired.json", tmp_path / "ranks.json"
    evaluate(*paths, summary, batch_rows=7)
    diagnose(*paths, summary, ranks, 5)
    days = read_json(ranks)["utc_days"]
    assert not days["1970-01-02"]["contact"]["feasible"]
    control = days["1970-01-01"]["quality"]
    assert control["strict_cut"] == 1
    assert next(r for r in control["rows_by_label"] if r["family"] == "C&C")["gained"] == 1
