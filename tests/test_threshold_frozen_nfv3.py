from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.context_cache.lab_data import FEATURES
from ids.flow_history.features import HISTORY_COLUMNS, WINDOW_MS
from ids.threshold_validation import frozen_nfv3 as frozen


def observations(offsets):
    timestamps = frozen.LAST_R20_EXCLUSIVE + np.array(offsets, dtype="int64")
    flows = pd.DataFrame(np.zeros((len(timestamps), len(FEATURES))), columns=FEATURES)
    flows["row_id"] = [f"row{n}" for n in range(len(flows))]
    flows["observed_ms"] = flows["FLOW_END_MILLISECONDS"] = timestamps
    flows["IPV4_SRC_ADDR"] = "192.0.2.10"
    flows["IPV4_DST_ADDR"] = "192.0.2.11"
    flows["PROTOCOL"] = 6
    flows["L4_DST_PORT"] = 443
    flows["OUT_PKTS"] = 0
    return flows


def test_exact_window_same_millisecond_and_batch_tie():
    events = observations([0, 1, 1, 60000, 60001, 60002])
    events.loc[2, "IPV4_DST_ADDR"] = "192.0.2.12"
    events.loc[2, "L4_DST_PORT"] = 80
    events.loc[2, "OUT_PKTS"] = 1
    replay = frozen.CompletionHistory()
    actual = np.vstack([replay.observe(events.iloc[:2]), replay.observe(events.iloc[2:])])
    assert actual.tolist()[:3] == [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    np.testing.assert_allclose(actual[3:], [[3, 2, 2, 2 / 3], [3, 2, 2, 2 / 3], [2, 1, 1, 1]])
    oracle = frozen.CompletionHistory().observe(events)
    np.testing.assert_array_equal(actual, oracle)


def test_delayed_order_and_invalid_inputs_refused():
    flow = observations([0, 10])
    with pytest.raises(ValueError, match="actual observation"):
        frozen.CompletionHistory().observe(flow.assign(observed_ms=flow.observed_ms + 1))
    with pytest.raises(ValueError, match="ordered"):
        frozen.CompletionHistory().observe(flow.iloc[::-1])
    replay = frozen.CompletionHistory()
    replay.observe(flow.iloc[-1:])
    with pytest.raises(ValueError, match="ordered"):
        replay.observe(flow.iloc[:1])
    with pytest.raises(ValueError, match="Invalid protocol"):
        frozen.CompletionHistory().observe(flow.assign(PROTOCOL=300))


def test_separate_score_and_evaluate_on_full_synthetic_capture(tmp_path, monkeypatch):
    obs = observations([0, 5000, 59999, 60000, 60100, 60105, 60105, 60106])
    start = int(obs.observed_ms.iloc[4])
    end = int(obs.observed_ms.iloc[-1]) + 1
    original = tmp_path / "nf.parquet"
    obs.to_parquet(original, index=False)
    archive = tmp_path / "original.zip"
    archive.write_bytes(b"synthetic archive")
    monkeypatch.setattr(frozen, "ARCHIVE_SHA", frozen.sha256_file(archive))
    models = {
        a: SimpleNamespace(features=list(FEATURES) + (list(HISTORY_COLUMNS) if a.endswith("history") else []))
        for a in frozen.MODELS
    }
    monkeypatch.setattr(frozen, "load_models", lambda *_: (models, {name: 1.5 for name in models}))

    def score(model, flows, features):
        return flows[HISTORY_COLUMNS[0]].to_numpy() if len(features) == 52 else flows.OUT_PKTS.to_numpy()

    monkeypatch.setattr(frozen.trees, "score_arm", score)
    output = tmp_path / "sealed"
    rec = frozen.predict(archive, original, start, end, output, 2)
    assert rec["evaluation_rows"] == 4 and not rec["truth_opened_by_scoring"]
    assert rec["history_only_warmup_rows"] == 3 and rec["prior_scored_reference_rows"] == 1
    scored = pd.read_parquet(output / frozen.PREDICTIONS)
    assert "attack_family" not in scored and "label" not in scored
    assert scored.hgb52_history.tolist() == [3, 4, 4, 6]
    assert (output / frozen.RECEIPT).exists() and not (output / "frozen_evaluation.json").exists()
    assert frozen.predict(archive, original, start, end, output, 2) == rec
    with pytest.raises(ValueError, match="different input, interval"):
        frozen.predict(archive, original, start - 1, end, output, 2)
    truth = tmp_path / "truth.parquet"
    rows = pd.DataFrame(
        {
            "row_id": scored.row_id,
            "label": [0, 1, 0, 1],
            "attack_family": ["Benign", "Bot", "Benign", "Scanning"],
        }
    )
    rows.iloc[:-1].to_parquet(truth, index=False)
    with pytest.raises(ValueError, match="precisely"):
        frozen.evaluate(truth, output)
    rows.to_parquet(truth, index=False)
    result = frozen.evaluate(truth, output)
    assert result["has_scan_bot_and_normal"]
    assert result["same_policy_between_models"]["unlabeled"]["groups"]["Bot"]["added"] == 1
    assert frozen.sha256_file(output / frozen.PREDICTIONS) == rec["predictions_sha256"]
    with pytest.raises(ValueError, match="span"):
        frozen.predict(archive, original, start, end + 100, tmp_path / "too-late", 2)
    with pytest.raises(ValueError, match="60 seconds"):
        frozen.predict(archive, original, start - WINDOW_MS, end, tmp_path / "too-early", 2)
