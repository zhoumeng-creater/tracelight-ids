import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from ids.flow_history.features import HISTORY_COLUMNS

@pytest.mark.parametrize("historical,kind", [(False, "mlp"), (True, "mlp"), (True, "hgb")])
def test_stream_cpu_cost_ties_expiry_warmup_and_replay(tmp_path, monkeypatch, historical, kind):
    from ids.flow_history.streaming import META, bench_stream_arm
    key = (9 << 8) | 6
    warm = pd.DataFrame([[1,key,88,443,1,1000], [2,key,99,80,0,60999]], columns=META)
    warm_path = tmp_path / "benchmark_warmup.parquet"
    warm.to_parquet(warm_path, index=False)
    segment = pd.DataFrame([[3,key,77,443,1,61000], [4,key,66,444,0,61000], [5,key,55,22,1,61001]],
                           columns=META)
    for col in FEATURES:
        segment[col] = 0.
    segment["L4_DST_PORT"], segment["PROTOCOL"] = segment.history_dst_port, 6
    segment["OUT_PKTS"] = 1-segment.history_no_reply
    independent = np.array([[2,2,2,.5], [2,2,2,.5], [3,3,3,1/3]])
    segment[list(HISTORY_COLUMNS)] = independent
    segment_path = tmp_path/"benchmark_stream.parquet"
    segment.to_parquet(segment_path, index=False)
    write_json(tmp_path/"benchmark_segment.json", {"status":"ready", "benchmark_warmup_sha256":sha256_file(warm_path),
                                                   "benchmark_stream_sha256":sha256_file(segment_path)})
    names = list(FEATURES) + (list(HISTORY_COLUMNS) if historical else [])
    rng = np.random.default_rng(114)
    p = pd.DataFrame(rng.uniform(0, 20, (72, len(names))), columns=names)
    p["label"] = (p["IN_BYTES"] > 10).astype(int)
    if kind == "mlp":
        pytest.importorskip("torch")
        from ids.flow_history import model
        monkeypatch.setattr(model, "EPOCHS", 2)
        path = model.fit_arm(p, names, tmp_path/"test.pt", threads=1, device="cpu")
    else:
        from ids.flow_history import trees
        monkeypatch.setitem(trees.PARAMETERS, "max_iter", 12)
        path = trees.fit_arm(p, names, tmp_path/"test.joblib", threads=2)
    out = tmp_path/"results"
    summary = bench_stream_arm(path, segment_path, names, 1, threshold_value=.5, output_dir=out)
    assert summary["status"] == "complete" and summary["measured_rows"] == 3
    assert summary["kind"] == kind
    assert summary["torch_imported_in_worker"] == (kind == "mlp")
    assert summary["max_records_sharing_completion_ms"] == 2 and summary["warm_prior_key_rows"] == 2
    assert summary["history_matches_offline"] is (True if historical else None)
    assert summary["service_p95_ms"] > 0 and summary["throughput_rows_per_second"] > 0
    evidence = pd.read_csv(out/"stream-rows.csv.gz")
    assert evidence.source_record.tolist() == [3, 4, 5]
    if historical:
        np.testing.assert_allclose(evidence[["observed_"+name for name in HISTORY_COLUMNS]], independent)
