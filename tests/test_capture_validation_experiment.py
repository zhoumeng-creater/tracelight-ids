"""Synthetic external-role validation, two-HGB execution, scoring and resume."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ids.capture_validation import experiment
from ids.common import read_json, sha256_file, write_json
from ids.context_cache.lab_data import FEATURES
from ids.flow_history.evaluation import syn_cohort
from ids.flow_history.features import HISTORY_COLUMNS
from ids.flow_history.pipeline import save_scores
from ids.flow_history.streaming import META


def make_roles(root):
    rng = np.random.default_rng(1920)
    data = {}
    (root / "inputs").mkdir()
    for ix, (role, n) in enumerate((("P", 180), ("C", 120), ("E", 180))):
        frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
        for feature in HISTORY_COLUMNS:
            frame[feature] = rng.random(n)
        fam = np.array(["Scanning", "DoS", "Benign"] * (n // 3))
        if role == "C":
            fam[:] = "Benign"
        frame["attack_family"] = fam
        frame["truth_source"] = np.where(
            fam == "Scanning",
            np.where(np.arange(n) % 2 == 0, "corrected_unique", "pcap_event"),
            "source_native",
        )
        frame["source_family"] = np.where(fam == "Scanning", "Infilteration", fam)
        frame["source_label"] = (fam != "Benign").astype(int)
        frame["label"] = (fam != "Benign").astype(int)
        frame["row_id"] = [f"{role}-{i}" for i in range(n)]
        frame["communication_group"] = frame.row_id
        frame["host_key"] = ["overlap" if i % 3 == 0 else role + "host" for i in range(n)]
        frame["flow_end_ms"] = 1546300800000 + ix * 86400000 + np.arange(n)
        frame["end_day_utc"] = pd.to_datetime(frame.flow_end_ms, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        frame["IN_BYTES"] += 2 * frame.label
        frame.loc[:11, ["PROTOCOL", "TCP_FLAGS", "IN_PKTS", "OUT_PKTS"]] = [6, 2, 1, 0]
        frame.to_parquet(root / "inputs" / f"{role}_time.parquet", index=False)
        data[role] = frame
    write_json(
        root / "history-work/benchmark_segment.json",
        {"status": "unavailable", "reason": "synthetic no stream segment"},
    )
    return data


def test_registered_scores_preserve_family_tradeoff_and_shape_coverage(tmp_path):
    roles = make_roles(tmp_path)
    e = roles["E"]
    c = roles["C"]
    old = e.attack_family.eq("DoS").astype(float) + 0.5
    new = e.attack_family.eq("Scanning").astype(float) + 0.5
    cal = np.arange(len(c), dtype=float) / len(c)
    for arm, scores in zip(experiment.ARMS, (old, new)):
        save_scores(tmp_path, arm, "forward", c, e, cal, scores.to_numpy(), timing={})
    checked, _ = experiment.coverage(tmp_path)
    result = experiment.compare(tmp_path, checked)
    assert result["delta_tp"] == 0 and result["delta_scan_tp"] == 60
    assert result["passes_detection_comparison"]
    assert result["delta_other_attack_tp"] == -60 and not result["non_scanning_attack_tp_not_lower"]
    assert result["subsets"]["tcp_single_syn_no_return"]["scan_and_normal_observed"]
    labels = pd.read_parquet(tmp_path / result["row_label_trace"])
    assert labels.row_id.equals(e.row_id) and labels.broad_syn.sum() == 12
    assert labels.truth_source.equals(e.truth_source)
    assert result["scanning_by_truth_source"]["corrected_unique"]["hgb52_history_scan_tp"] == 30
    assert result["scanning_by_truth_source"]["pcap_event"]["hgb52_history_scan_tp"] == 30
    assert not syn_cohort(e).any()
    assert not result["subsets"]["exact_syn47"]["scan_and_normal_observed"]
    strata = pd.read_parquet(tmp_path / result["strata"])
    seen = strata.loc[strata.by.eq("source_seen_in_train")].set_index("value")
    assert seen.loc["seen", "hgb52_history_scan_tp"] == 60 and seen.loc["unseen", "lost_attack"] == 60
    assert set(strata.by) == {
        "source_seen_in_train",
        "source_host",
        "day",
        "family",
        "source_label",
        "source_family",
        "truth_source",
    }
    assert (
        strata[strata.by.eq("source_family")].set_index("value").loc["Infilteration", "hgb52_history_scan_tp"]
        == 60
    )


def test_no_scores_or_fits_when_forward_target_has_no_scans(tmp_path):
    roles = make_roles(tmp_path)
    frame = roles["E"].replace({"attack_family": {"Scanning": "DoS"}})
    frame.to_parquet(tmp_path / "inputs/E_time.parquet", index=False)
    report = experiment.run_pair(tmp_path, "no-scan-binding", 2, SimpleNamespace(activity=""))
    assert report["status"] == "not_evaluable" and report["coverage"]["not_evaluable_reasons"]
    assert not (tmp_path / "models").exists()


def test_chronology_and_cross_role_ids(tmp_path):
    roles = make_roles(tmp_path)
    c = roles["C"].copy()
    c.loc[0, "flow_end_ms"] = roles["P"].flow_end_ms.max()
    c.to_parquet(tmp_path / "inputs/C_time.parquet", index=False)
    with pytest.raises(ValueError, match="Nonchronological"):
        experiment.coverage(tmp_path)
    c = roles["C"].copy()
    c.loc[0, "row_id"] = roles["E"].row_id.iloc[0]
    c.to_parquet(tmp_path / "inputs/C_time.parquet", index=False)
    with pytest.raises(ValueError, match="Row overlap"):
        experiment.coverage(tmp_path)


def test_duplicate_communication_group_rejected_even_with_distinct_row_ids(tmp_path):
    roles = make_roles(tmp_path)
    c = roles["C"].copy()
    c.loc[0, "communication_group"] = roles["P"].communication_group.iloc[0]
    c.to_parquet(tmp_path / "inputs/C_time.parquet", index=False)
    with pytest.raises(ValueError, match="Communication group overlap"):
        experiment.coverage(tmp_path)


def test_spawned_two_hgb_and_isolated_cost_resume(tmp_path):
    make_roles(tmp_path)
    store = tmp_path / "history-work"
    key = (1 << 8) | 6
    warm = pd.DataFrame([[1, key, 8, 53, 1, 1000]], columns=META)
    stream = pd.DataFrame(
        [[2, key, 9, 443, 1, 61000], [3, key, 10, 80, 0, 61000], [4, key, 11, 53, 1, 61001]], columns=META
    )
    for name in FEATURES:
        stream[name] = 0.0
    stream["PROTOCOL"] = 6
    stream["L4_DST_PORT"] = stream.history_dst_port
    stream["OUT_PKTS"] = 1 - stream.history_no_reply
    stream[list(HISTORY_COLUMNS)] = [[1, 1, 1, 1.0], [1, 1, 1, 1.0], [2, 2, 2, 0.5]]
    warm.to_parquet(store / "benchmark_warmup.parquet", index=False)
    stream.to_parquet(store / "benchmark_stream.parquet", index=False)
    write_json(
        store / "benchmark_segment.json",
        {
            "status": "ready",
            "benchmark_warmup_sha256": sha256_file(store / "benchmark_warmup.parquet"),
            "benchmark_stream_sha256": sha256_file(store / "benchmark_stream.parquet"),
        },
    )
    mon = SimpleNamespace(activity="")
    first = experiment.run_pair(tmp_path, "synthetic-external", 2, mon)
    assert first["status"] == "complete" and first["training_threads_per_arm"] == 1
    assert len(first["cost"]) == 4
    assert all(
        v["cpu_replay"]["equivalent_to_reference"]
        and v["stream"]["status"] == "complete"
        and v["stream"]["service_p95_ms"] > 0
        for v in first["cost"].values()
    )
    assert first["full_cost_replay_available_and_equivalent"]
    assert first["cost"]["hgb52_history/cpu-2"]["stream"]["history_matches_offline"]
    for arm in experiment.ARMS:
        fit = read_json(tmp_path / f"evidence/forward/{arm}/fit.json")
        assert fit["iterations_fitted"] == 200 and fit["threads"] == 1
    saved = {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in (tmp_path / "receipts").glob("*.json")}
    repeated = experiment.run_pair(tmp_path, "synthetic-external", 2, mon)
    assert repeated == first and len(saved) == 12
    assert saved == {p: (p.stat().st_mtime_ns, sha256_file(p)) for p in saved}
