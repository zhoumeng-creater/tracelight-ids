"""Synthetic inputs and tiny official CPU models only; no historical model fit."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids.model_training_redesign.preprocessing import CONTINUOUS, Preprocessor
from ids.model_training_redesign.sampling import GROUPS, GroupSampler, canonical_frame
from ids.model_training_redesign.selection import Selector, selection_key, selection_metrics
from ids.model_training_redesign import training


def synthetic(role="F"):
    rng = np.random.default_rng(18 if role == "F" else 43)
    counts = (12, 8, 9, 3) if role == "F" else (9, 6, 8, 4)
    frames = []
    interval = 30 if role == "F" else 300
    for group, count in zip(GROUPS, counts):
        f = pd.DataFrame(rng.uniform(0.1, 12, (count, 5)), columns=CONTINUOUS)
        f["source_id"] = f"TQH-{group[0]}_i{interval}_j0"
        f["row_id"] = np.arange(count) + (100 if group[1] == "1" else 0)
        f["profile"], f["label"] = group[0], int(group[1])
        f["ip_protocol"] = 6 if group[0] == "A" else 17
        frames.append(f)
    result = pd.concat(frames, ignore_index=True)
    result.loc[0, "duration_seconds"] = -1
    return result


@pytest.fixture
def design():
    root = Path(__file__).resolve().parents[1]
    d = json.loads((root / "configs/model-training-redesign-20261117/design.json").read_text(encoding="utf-8"))
    d["model"].update(k=2, n_blocks=1, d_block=8, dropout=0.1)
    d["input"]["ple"].update(embedding_dim=4, max_bins=4)
    d["training"]["sampler"].update(batch_size=16, draws_per_group_per_step=4)
    d["training"]["max_optimizer_steps"] = 3
    d["selection"].update(validation_steps=[1, 2, 3], patience_validation_events=2)
    for role in ("F", "V"):
        d["roles"][role]["rows"] = len(synthetic(role))
    for item, n in zip(d["training"]["groups"], (12, 8, 9, 3)):
        item["rows"] = n
    for item, n0, n1 in zip(d["selection"]["panels"], (9, 8), (6, 4)):
        item.update(normal=n0, positive=n1)
    return d


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_protocol_intervention_only_changes_explicit_slot_and_survives_reload(design):
    f = synthetic()
    blind = Preprocessor.fit(f, arm="protocol_blind", design=design)
    aware = Preprocessor.fit(f, arm="protocol_aware", design=design)
    bn, bc = blind.transform(f)
    an, ac = aware.transform(f)
    np.testing.assert_array_equal(bn, an)
    np.testing.assert_array_equal(bc[:, 1:], ac[:, 1:])
    assert (bc[:, 0] == 5).all() and set(ac[:, 0]) == {1, 2}
    changed = f.assign(ip_protocol=np.resize([1, 6, 17, 58, 132, -1], len(f)))
    restored = Preprocessor.from_dict(blind.to_dict())
    np.testing.assert_array_equal(restored.transform(changed)[0], bn)
    np.testing.assert_array_equal(restored.transform(changed)[1], bc)
    assert blind.base.to_dict() == aware.base.to_dict()


def test_preprocessing_is_F_only_handles_missing_constant_and_outside_range(design):
    f = synthetic().assign(orig_pkts=-1, resp_pkts=3)
    pre = Preprocessor.fit(f, arm="protocol_blind", design=design)
    before = copy.deepcopy(pre.to_dict())
    v = synthetic("V").assign(orig_pkts=1000, resp_pkts=4, duration_seconds=1e12)
    n, c = pre.transform(v)
    assert pre.to_dict() == before
    assert pre.base.median[1] == 0 and pre.base.scale[1] == pre.base.scale[2] == 1
    assert np.isfinite(n).all() and c.shape == (len(v), 6)
    assert n[:, 0].max() > 10  # No evaluation-range clipping.


def test_sampler_exact_group_mass_cycles_reproducibility_and_prefixes():
    f = canonical_frame(synthetic())
    a, b = GroupSampler(f, seed=17, group_batch_size=4), GroupSampler(f, seed=17, group_batch_size=4)
    seen_b = []
    for step in range(4):
        ia, ib = a.next(), b.next()
        np.testing.assert_array_equal(ia, ib)
        for offset, group in enumerate(GROUPS):
            part = f.iloc[ia[offset * 4:(offset + 1) * 4]]
            assert part.profile.eq(group[0]).all() and part.label.eq(int(group[1])).all()
        seen_b.extend(ia[-4:].tolist())
        assert a.audit()["draw_sha256"] == b.audit()["draw_sha256"]
    # Three B positives: every completed queue cycle visits each exactly once.
    for start in range(0, 15, 3):
        assert len(set(seen_b[start:start + 3])) == 3
    audit = a.audit(reset_window=True)
    assert audit["groups"]["B1"]["draws"] == 16
    assert audit["groups"]["B1"]["completed_cycles"] == 5
    assert audit["groups"]["B1"]["unique"] == 3
    assert a.audit()["groups"]["B1"]["window_draws"] == 0
    assert a.state()["rng"] == b.state()["rng"]


def test_canonical_identity_uses_numeric_row_order_and_rejects_duplicates():
    f = synthetic().sample(frac=1, random_state=9)
    sorted_f = canonical_frame(f)
    assert sorted_f.iloc[:3].row_id.tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="unique"):
        canonical_frame(pd.concat([f, f.iloc[:1]]))


def test_four_group_loss_matches_independent_group_means_and_gradients():
    import torch
    values = torch.tensor(np.linspace(-3, 3, 24).reshape(8, 3, 1), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.float64)
    loss, groups = training.four_group_loss(values, labels, 2)
    manual = [torch.nn.functional.binary_cross_entropy_with_logits(
        values[start:start + 2, :, 0], labels[start:start + 2, None].expand(-1, 3)) for start in range(0, 8, 2)]
    torch.testing.assert_close(groups, torch.stack(manual))
    torch.testing.assert_close(loss, sum(manual) / 4)
    loss.backward()
    torch.testing.assert_close(values.grad[:, :, 0], (values.detach()[:, :, 0].sigmoid() - labels[:, None]) / 24)


def metrics(a=.8, b=.9, pooled=.85, separated=False, nll=1):
    return {"panels": {"A_TCP": {"pauc": a}, "B_UDP": {"pauc": b}},
            "pooled": {"pauc": pooled, "nll": nll}, "strictly_separated": separated}


def test_selection_ignores_NLL_ties_and_patience():
    s = Selector(patience=2)
    assert s.update(metrics(nll=10), 139)["improved"]
    assert not s.update(metrics(nll=.001), 278)["improved"]
    final = s.update(metrics(nll=.000001), 417)
    assert final["stop"] and final["stop_reason"] == "ranking_patience"
    assert s.best_step == 139
    assert selection_key(metrics(a=.81, b=.81, pooled=.5), 1) > selection_key(metrics(), 1)
    assert selection_key(metrics(pooled=.9), 1) > selection_key(metrics(pooled=.8), 1)


def test_panel_separation_does_not_hide_cross_profile_offsets():
    f = pd.DataFrame({"profile": ["A", "A", "B", "B"], "ip_protocol": [6, 6, 17, 17],
                      "label": [0, 1, 0, 1], "source_id": ["a", "a", "b", "b"]})
    m = selection_metrics(np.array([10., 11., 0., 1.]), f)
    assert all(x["strictly_separated"] for x in m["panels"].values())
    assert not m["strictly_separated"]
    s = Selector()
    assert not s.update(m, 139)["stop"]
    sep = selection_metrics(np.array([0., 2., -1., 3.]), f)
    result = s.update(sep, 278)
    assert result["stop"] and s.best_step == 278


def test_first_exact_separation_replaces_a_rounded_metric_tie():
    s = Selector()
    s.update(metrics(a=1, b=1, pooled=1, separated=False), 139)
    strict = s.update(metrics(a=1, b=1, pooled=1, separated=True), 278)
    assert strict["improved"] and strict["stop"] and s.best_step == 278
    assert strict["replacement_reason"] == "first_strict_full_separation"
    later = s.update(metrics(a=1, b=1, pooled=1, separated=True), 417)
    assert not later["improved"] and s.best_step == 278


def test_equal_scores_do_not_count_as_strict_separation():
    f = synthetic("V")
    m = selection_metrics(np.zeros(len(f)), f)
    assert not m["strictly_separated"]
    with pytest.raises(ValueError, match="finite"):
        selection_metrics(np.full(len(f), np.nan), f)


def test_tiny_official_pair_and_cold_load_deliver_same_selected_checkpoint(tmp_path, design):
    pytest.importorskip("tabm")
    import torch
    f, v = synthetic(), synthetic("V")
    results = []
    for arm in ("protocol_blind", "protocol_aware"):
        result = training.fit_trajectory(f, v, tmp_path / arm, arm=arm, seed=17, threads=1,
                                         device="cpu", design=design)
        assert result["status"] == "complete", result
        assert result["fit_calls"] == 1 and result["final_refits"] == 0
        assert Path(result["artifact_path"]).read_bytes() == Path(result["best_path"]).read_bytes()
        history = read(result["history_path"])
        assert result["selected_step"] == max(history, key=lambda h: tuple(h["selection"]["key"]))["step"]
        assert result["optimizer_steps"] <= 3
        checkpoint = torch.load(result["artifact_path"], weights_only=True)
        rng_audit = torch.load(result["rng_audit_path"], weights_only=True)
        pair = read(result["pairing_path"])
        assert checkpoint["rng_step"] == result["selected_step"]
        assert training._state_sha256(checkpoint["dropout_rng_state"]) == pair[
            "dropout_rng_by_validation_step"][str(result["selected_step"])]
        assert rng_audit["step"] == result["optimizer_steps"]
        assert training._state_sha256(rng_audit["dropout_rng_state"]) == pair[
            "dropout_rng_by_validation_step"][str(result["optimizer_steps"])]
        assert checkpoint["sampling_state"]["audit"]["steps"] == result["selected_step"]
        predictor = training.load_predictor(result["artifact_path"])
        stored = np.load(result["train_scores_path"])
        np.testing.assert_array_equal(predictor.score(canonical_frame(v)), stored["validation_scores"])
        n, c = predictor.preprocessor.transform(v)
        np.testing.assert_array_equal(predictor.score(v), predictor.score_transformed(n, c))
        if arm == "protocol_blind":
            np.testing.assert_array_equal(predictor.score(v), predictor.score(v.assign(ip_protocol=132)))
        results.append(result)
    a, b = (read(r["pairing_path"]) for r in results)
    for key in ["preprocessing_sha256", "numeric_bins_sha256", "initial_state_sha256", "F_identity_sha256",
                "initialization_seed", "dropout_seed", "sampler_seeds", "initial_dropout_rng_sha256",
                "initial_sampler_rng_sha256"]:
        assert a[key] == b[key], key
    for name in ("draw_chain_by_validation_step", "dropout_rng_by_validation_step"):
        for step in a[name].keys() & b[name].keys():
            assert a[name][step] == b[name][step]
    assert results[0]["parameters"] == results[1]["parameters"]


def test_fit_rejects_overlap_and_never_retries_nonempty_directory(tmp_path, design):
    bad = copy.deepcopy(design)
    bad["roles"]["V"]["rows"] = len(synthetic())
    result = training.fit_trajectory(synthetic(), synthetic(), tmp_path / "bad", arm="protocol_blind",
                                     seed=17, threads=1, device="cpu", design=bad)
    assert result["status"] == "failed" and "overlap" in result["error"]
    assert not (tmp_path / "bad/model.pt").exists()
    with pytest.raises(ValueError, match="empty"):
        training.fit_trajectory(synthetic(), synthetic("V"), tmp_path / "bad", arm="protocol_blind",
                                seed=17, threads=1, device="cpu", design=design)


def test_fit_rejects_second_weighting_and_wrong_panel_counts(tmp_path, design):
    for name, mutate, message in [("weight", lambda d: d["training"].update(pos_weight=2), "second loss weight"),
                                   ("panel", lambda d: d["selection"]["panels"][0].update(normal=500), "panel counts")]:
        bad = copy.deepcopy(design)
        mutate(bad)
        result = training.fit_trajectory(synthetic(), synthetic("V"), tmp_path / name,
                                         arm="protocol_blind", seed=17, threads=1, device="cpu", design=bad)
        assert result["status"] == "failed" and message in result["error"]


def test_protocol_policy_tampering_is_rejected_on_load(tmp_path, design):
    pytest.importorskip("tabm")
    import torch
    result = training.fit_trajectory(synthetic(), synthetic("V"), tmp_path / "fit", arm="protocol_blind",
                                     seed=29, threads=1, device="cpu", design=design)
    assert result["status"] == "complete", result
    state = torch.load(result["artifact_path"], weights_only=True)
    state["preprocessing"]["protocol_policy"] = "original_mapping"
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="protocol policy"):
        training.load_predictor(tmp_path / "bad.pt")
