import json

import numpy as np
import pandas as pd
import pytest

from ids.readiness_replay.ranking import MODEL_SCORES, RANKING_SCORES, rank_diagnostics


SOURCE = {"name": "synthetic-source", "role": "opened_replay"}


def inputs(normal_scores, target_scores, target_families=None):
    values = np.asarray([*normal_scores, *target_scores], dtype=float)
    n, total = len(normal_scores), len(values)
    ids = np.arange(11, total + 11, dtype=np.int64)
    features = pd.DataFrame(dict(
        row_id=ids, flow_end_ms=np.arange(total, dtype=np.int64),
        ip_protocol=6, resp_port=443, hist_dst_ports_60s=values, hist_dst_ips_60s=values,
        orig_pkts=np.arange(total) + 1, resp_pkts=2, orig_ip_bytes=64, resp_ip_bytes=128,
    ))
    scores = pd.DataFrame({"row_id": ids, **{key: values for key in MODEL_SCORES}})
    families = target_families or ["ExternalC2"] * len(target_scores)
    truth = pd.DataFrame(dict(
        row_id=ids, window="2025-01-08", family=["Benign"] * n + families,
        uid=[f"uid-{i}" for i in ids], event_id=[""] * n + ["step-8"] * len(target_scores),
        evidence_level="host_and_network_confirmed", exposure="forward_unopened",
        native_label=[0] * n + [1] * len(target_scores), native_type="native-step",
        native_detail="Complete author comment", author_step="8",
    ))
    return features, scores, truth


def run(tmp_path, feature, scores, truth, scopes=None, **kwargs):
    paths = rank_diagnostics(feature, scores, truth,
                             scopes or {"all": np.ones(len(truth), dtype=bool)},
                             kwargs.get("source", SOURCE), kwargs.get("group", "mlp29"), tmp_path)
    result = {p.name.removesuffix(".csv.gz"): pd.read_csv(p) for p in paths if p.suffix == ".gz"}
    result["summary"] = json.loads(paths[-1].read_text())
    return result


def test_strict_ties_do_not_split_normal_block(tmp_path):
    f, s, t = inputs([0] * 98 + [1, 1], [1, 2, 0], ["ExternalC2", "Download", "ExternalC2"])
    out = run(tmp_path, f, s, t)
    cuts = out["ranking_thresholds"]
    assert set(cuts.ranking_score) == set(RANKING_SCORES)
    assert (cuts.normal == 100).all() and (cuts.cap == 1).all()
    assert (cuts.oracle_normal_threshold == 1).all() and (cuts.fp == 0).all()
    assert (cuts.boundary_normal_ties == 2).all() and (cuts.unused_budget == 1).all()
    targets = out["ranking_targets"].query("ranking_score == 'score_raw'").set_index("target_score")
    assert not targets.loc[1, "oracle_detected"]
    assert targets.loc[1, ["global_normal_gt", "global_normal_ge", "global_normal_eq"]].tolist() == [0, 2, 2]
    assert targets.loc[1, "minimum_fp_to_include"] == 2 and targets.loc[1, "budget_excess"] == 1
    assert not targets.loc[1, "can_include_within_budget"]
    assert targets.loc[2, "can_include_within_budget"] and targets.loc[2, "oracle_detected"]
    assert targets.loc[0, "global_normal_ge"] == 100 and targets.loc[0, "budget_excess"] == 99
    family = out["ranking_families"].query("ranking_score == 'score_raw'").set_index("family")
    assert family.loc["Download", "tp"] == 1 and family.loc["ExternalC2", "tp"] == 0
    assert family.loc["Benign", "fp"] == 0 and pd.isna(family.loc["Benign", "tp"])


def test_zero_cap_and_service_without_normals_are_explicit(tmp_path):
    f, s, t = inputs(list(range(99)), [98, 99], ["C&C", "Download"])
    f.loc[len(f) - 1, "resp_port"] = 80
    out = run(tmp_path, f, s, t)
    target = out["ranking_targets"].query("ranking_score == 'score_quality'").set_index("target_score")
    assert target.loc[98, "cap"] == 0 and not target.loc[98, "can_include_within_budget"]
    assert target.loc[99, "can_include_within_budget"]
    assert not target.loc[99, "service_supported"] and target.loc[99, "service_normal"] == 0
    assert pd.isna(target.loc[99, "service_can_include_within_budget"])
    assert pd.isna(target.loc[99, "service_cap"])
    assert target.loc[98, "service_normal_ge"] == 1


def test_no_normals_produces_no_fpr_or_tp_claim(tmp_path):
    f, s, t = inputs([], [0, 1], ["Download", "Unresolved"])
    out = run(tmp_path, f, s, t)
    cuts = out["ranking_thresholds"]
    assert (cuts.status == "no_normals").all()
    assert cuts[["cap", "fp", "fpr", "within_budget", "oracle_normal_threshold"]].isna().all().all()
    target = out["ranking_targets"]
    assert len(target) == 5 and (target.global_normal_ge == 0).all()
    assert target[["oracle_detected", "can_include_within_budget", "budget_excess"]].isna().all().all()
    families = out["ranking_families"]
    assert set(families.family) == {"Download", "Unresolved"}
    assert families[["oracle_alerts", "tp"]].isna().all().all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("column", ["score_raw", "score_contact", "hist_dst_ports_60s"])
def test_nonfinite_values_fail_without_silently_dropping_rows(tmp_path, bad, column):
    f, s, t = inputs([0, 1], [2])
    frame = s if column in s else f
    frame.loc[0, column] = bad
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="finite"):
        run(destination, f, s, t)
    assert not destination.exists()


@pytest.mark.parametrize("frame_name", ["features", "scores", "truth"])
def test_duplicate_row_identity_rejected(tmp_path, frame_name):
    f, s, t = inputs([0, 1], [2])
    dict(features=f, scores=s, truth=t)[frame_name].loc[1, "row_id"] = 11
    with pytest.raises(ValueError, match="row_id"):
        run(tmp_path, f, s, t)


def test_row_order_mismatch_rejected_even_when_sets_match(tmp_path):
    f, s, t = inputs([0, 1], [2])
    with pytest.raises(ValueError, match="row_id differs"):
        run(tmp_path, f, s.iloc[::-1], t)


@pytest.mark.parametrize("mask", [np.ones(3, dtype=int), np.array([True, False]),
                                   np.array([True, None, False], dtype=object)])
def test_bad_scope_rejected(tmp_path, mask):
    f, s, t = inputs([0, 1], [2])
    with pytest.raises(ValueError, match="Scope"):
        run(tmp_path, f, s, t, {"all": mask})


def test_windows_scopes_and_scores_remain_separate(tmp_path):
    f, s, t = inputs([0] * 50 + [10] * 50, [1, 11], ["Download", "Download"])
    t.loc[50:99, "window"] = "2025-01-09"
    t.loc[101, "window"] = "2025-01-09"
    s["score_quality"] = -s.score_raw
    after = np.arange(len(t)) >= 50
    scopes = {"all": np.ones(len(t), bool), "startup": ~after, "after": after,
              "original_startup": ~after, "original_after": after}
    out = run(tmp_path, f, s, t, scopes)
    cuts = out["ranking_thresholds"]
    assert len(cuts) == 2 * 5 * 5
    all_raw = cuts.query("scope == 'all' and ranking_score == 'score_raw'")
    assert all_raw.normal.tolist() == [50, 50] and all_raw.cap.tolist() == [0, 0]
    assert all_raw.oracle_normal_threshold.tolist() == [0, 10]
    rare = out["ranking_targets"].query("scope == 'all'")
    assert rare.query("ranking_score == 'score_raw'").oracle_detected.all()
    assert not rare.query("ranking_score == 'score_quality'").oracle_detected.any()
    assert (cuts.query("scope == 'startup' and window == '2025-01-09'").status == "no_normals").all()
    assert (cuts.source == SOURCE["name"]).all() and (cuts.group == "mlp29").all()
    assert out["summary"]["diagnostic_only"]


def test_service_counts_are_directional_and_do_not_replace_global_budget(tmp_path):
    f, s, t = inputs([0] * 98 + [2, 2], [1, 2])
    # Only one equal-score normal shares the target's protocol and destination port.
    f.loc[99, "ip_protocol"] = 17
    f.loc[:97, "resp_port"] = 53
    out = run(tmp_path, f, s, t)
    rare = out["ranking_targets"].query("ranking_score == 'score_raw'")
    assert rare.global_normal_ge.tolist() == [2, 2]
    assert rare.service_normal.tolist() == [1, 1]
    assert rare.service_normal_ge.tolist() == [1, 1]
    assert rare.service_normal_gt.tolist() == [1, 0]
    assert rare.service_normal_eq.tolist() == [0, 1]
    assert not rare.can_include_within_budget.any()


def test_all_rare_rows_metadata_and_numeric_features_retained(tmp_path):
    f, s, t = inputs([0] * 100, list(range(12)), ["Download"] * 6 + ["ExternalC2"] * 6)
    for i in range(27):
        f[f"numeric_{i}"] = np.arange(len(f)) + i
    original_f, original_t = f.copy(deep=True), t.copy(deep=True)
    f.index = np.arange(len(f)) + 900
    t.index = np.arange(len(t)) + 1000
    out = run(tmp_path, f, s, t)
    rare = out["ranking_targets"]
    assert len(rare) == 12 * 5
    assert rare.uid.nunique() == 12 and rare.event_id.eq("step-8").all()
    assert rare.evidence_level.eq("host_and_network_confirmed").all()
    assert rare.exposure.eq("forward_unopened").all()
    assert rare.native_detail.eq("Complete author comment").all()
    assert all(key in rare for key in [*MODEL_SCORES, *[f"numeric_{i}" for i in range(27)]])
    assert out["summary"]["unique_rare_rows"] == 12
    pd.testing.assert_frame_equal(f.reset_index(drop=True), original_f)
    pd.testing.assert_frame_equal(t.reset_index(drop=True), original_t)


def test_zero_rare_targets_still_emits_readable_target_schema(tmp_path):
    f, s, t = inputs([0] * 100, [2, 3], ["Scanning", "OtherAttack"])
    out = run(tmp_path, f, s, t)
    assert out["ranking_targets"].empty
    assert {"uid", "score_raw", "global_normal_ge", "service_supported"} <= set(out["ranking_targets"])
    families = out["ranking_families"].query("ranking_score == 'score_raw'").set_index("family")
    assert families.loc["Scanning", "tp"] == families.loc["OtherAttack", "tp"] == 1


@pytest.mark.parametrize("source,group", [({"name": "", "role": "replay"}, "hgb"),
                                           ({"name": "s"}, "hgb"), (SOURCE, "")])
def test_source_and_group_are_required(tmp_path, source, group):
    f, s, t = inputs([0], [1])
    with pytest.raises(ValueError, match="source|group"):
        run(tmp_path, f, s, t, source=source, group=group)


def test_reported_cut_attains_bruteforce_monotone_ceiling(tmp_path):
    rng = np.random.default_rng(19)
    normal = rng.integers(0, 5, size=200)
    target = np.arange(-1, 7, dtype=float)
    f, s, t = inputs(normal, target, ["Scanning"] * len(target))
    out = run(tmp_path, f, s, t)
    # Include threshold locations immediately below every distinct score, not only ties.
    possible = np.unique(np.r_[normal, target])
    cuts = np.r_[possible, np.nextafter(possible, -np.inf)]
    feasible = [cut for cut in cuts if np.count_nonzero(normal > cut) <= len(normal) // 100]
    best = max(np.count_nonzero(target > cut) for cut in feasible)
    reported = out["ranking_families"].query("ranking_score == 'score_raw' and family == 'Scanning'")
    assert reported.tp.item() == best
