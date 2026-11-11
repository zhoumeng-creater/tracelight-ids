import gzip
import hashlib

import numpy as np
import pandas as pd
import pytest

from ids.common import read_json
from ids.threshold_validation import experiment as ex
from ids.threshold_validation.runner import parallel
from ids.threshold_validation.rules import replay_thresholds
from ids.threshold_validation.sources import resolve_log


def test_train_freeze_score_and_pair(tmp_path):
    src = ["early", "normal", "future"]
    flags = ["P", "C", "E"]
    reg = {
        "seed": 12,
        "train_rows_per_class": 50,
        "sources": [{"name": n, "role": r} for n, r in zip(src, flags)],
    }
    (tmp_path / "prepared").mkdir()
    (tmp_path / "evidence").mkdir()
    rng = np.random.default_rng(31)
    for i, name in enumerate(src):
        size = 130 if i == 1 else 240
        row = np.arange(1, size + 1, dtype=np.int64)
        family = np.full(size, "Benign", dtype=object)
        if i != 1:
            family[0::3] = "Scanning"
            family[1::3] = "C&C"
        frame = pd.DataFrame({"row_id": row, "flow_end_ms": row * 3000 + i * 100000000, "family": family})
        for col in ex.ARMS["history"]:
            frame[col] = rng.random(size)
        frame.to_parquet(ex.prepared(tmp_path, name), index=False)
    ex.draw_train(tmp_path, reg)
    jobs = [
        (tmp_path, "synthetic", arm + "-fit", ex.train_arm, (tmp_path, arm, 2, reg), 2) for arm in ex.ARMS
    ]
    parallel(jobs, 2)
    for arm in ex.ARMS:
        ex.score_arm(tmp_path, "future", arm, 2)
    artifacts = ex.judge(tmp_path, "future")
    assert len(artifacts) == 6
    compare = read_json(tmp_path / "evidence/future/comparison.json")["comparisons"]
    assert len(compare) == 7
    for group in compare.values():
        assert set(group["groups"]) >= {"C&C", "Scanning"}
        assert "Bot" not in group["groups"]
        assert group["normal_false_alert_budget"] == 0
    before = pd.read_parquet(tmp_path / "evidence/future/row-scores-and-alerts.parquet")
    # Changing only future attack identities never changes the scores or primary unlabeled policy.
    frame["family"] = np.where(family == "Scanning", "C&C", family)
    frame.to_parquet(ex.prepared(tmp_path, "future"), index=False)
    for arm in ex.ARMS:
        ex.score_arm(tmp_path, "future", arm, 2)
    ex.judge(tmp_path, "future")
    after = pd.read_parquet(tmp_path / "evidence/future/row-scores-and-alerts.parquet")
    for arm in ex.ARMS:
        for suffix in ("_score", "_fixed", "_unlabeled"):
            assert before[arm + suffix].equals(after[arm + suffix])


def test_delayed_feedback_can_use_previous_day_only_after_confirmation():
    times = np.r_[np.arange(1100), 4 * 3_600_000]
    frame = pd.DataFrame(
        {"row_id": np.arange(len(times)), "flow_end_ms": times, "score": np.arange(len(times))}
    )
    release = pd.Series(times + 900_000)
    _, audit = replay_thresholds(
        frame,
        10000.0,
        start_ms=int(times[-1]),
        end_ms=int(times[-1] + 1),
        trusted_normal_available_ms=release,
        trusted_lag_ms=900_000,
        trusted_window_ms=86_400_000,
    )
    assert audit.set_index("policy").loc["recent_normal", "n_reference"] == 1100
    assert audit.set_index("policy").loc["unlabeled", "n_reference"] == 0


def test_offline_dataset_checks_uncompressed_identity(tmp_path):
    data = b"official source bytes"
    spec = {"name": "case", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    path = tmp_path / "case.conn.log.labeled.gz"
    path.write_bytes(gzip.compress(data))
    assert resolve_log(tmp_path, tmp_path / "cache", spec) == path
    path.write_bytes(gzip.compress(data + b"x"))
    with pytest.raises(ValueError):
        resolve_log(tmp_path, tmp_path / "cache", spec)
