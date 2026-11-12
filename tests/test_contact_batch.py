"""Synthetic regression cases for the round-22 information boundary and paired budgets."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ids.common import read_json
from ids.contact_evidence import data, evaluation, experiment
from ids.contact_evidence.features import ARMS
from ids.contact_evidence.policy import POLICIES, SINGLE, apply, contact_gate, register


def test_budget_ties_and_protection_on_a_future_stream():
    rng = np.random.default_rng(5)
    n = 1826
    cal = {k: rng.integers(0, 518, n).astype(float) for k in SINGLE}
    gate = rng.random(n) < 0.25
    rules = register(cal, gate)
    cal_flags = apply(rules, cal, gate)
    assert all(evaluation.unpack(cal_flags, key).sum() <= n // 100 for key in POLICIES)
    future = {k: rng.normal(450, 100, 2000) for k in SINGLE}
    future_gate = rng.random(2000) < 0.4
    flags = apply(rules, future, future_gate)
    anchor = evaluation.unpack(flags, "anchor")
    protected = evaluation.unpack(flags, "contact_add")
    assert protected[anchor].all()
    assert not protected[~anchor & ~future_gate].any()
    assert np.array_equal(flags >> 7, future_gate)
    assert np.array_equal(evaluation.unpack(flags, "raw"), future["raw"] > rules["raw"]["threshold"])


def test_empty_gate_and_missing_responses():
    gate = contact_gate(
        pd.DataFrame(
            {
                "svc_prior_reply_fraction_60s": [-1, 0, 0.8, 1, 0.1],
                "contact_prior_reply_fraction_60s": [-1, -1, 0, 0, 0.3],
            }
        )
    )
    assert gate.tolist() == [False, True, True, False, False]
    scores = {k: np.arange(250, dtype=float) for k in SINGLE}
    false = np.zeros(250, dtype=bool)
    reg = register(scores, false)
    assert reg["contact_add"]["threshold"] is None
    flags = apply(reg, scores, ~false)
    assert np.array_equal(evaluation.unpack(flags, "anchor"), evaluation.unpack(flags, "contact_add"))


def test_zero_cc_or_scan_support_never_passes():
    normal = {"rows": 100, "before": 1, "after": 1, "gained": 0, "lost": 0}
    gained = {"rows": 6, "before": 3, "after": 4, "gained": 1, "lost": 0}
    positive = {"Benign": normal, "Scanning": gained, "C&C": gained}
    assert evaluation.criteria(positive)["joint_pass"]
    assert not evaluation.criteria({k: v for k, v in positive.items() if k != "Scanning"})["joint_pass"]
    assert not evaluation.criteria({**positive, "C&C": {**gained, "before": 0}})["joint_pass"]
    assert not evaluation.criteria({**positive, "Benign": {**normal, "after": 2}})["joint_pass"]


def synthetic_input(root, name, n, phase):
    ids = np.arange(1, n + 1)
    rng = np.random.default_rng(phase)
    f = pd.DataFrame({k: rng.integers(0, 20, n).astype(float) for k in ARMS["contact"]})
    f["row_id"] = ids
    f["flow_end_ms"] = ids * 5
    for k in ("svc_prior_reply_fraction_60s", "contact_prior_reply_fraction_60s"):
        f[k] = rng.choice([-1.0, 0.0, 0.3, 1.0], n)
    labels = np.where(ids % 2, "Benign", np.where(ids % 7, "Scanning", "C&C"))
    if phase == 2:
        labels[:] = "Benign"
    obs = experiment.enriched(root, name)
    obs.parent.mkdir(exist_ok=True, parents=True)
    f.to_parquet(obs, index=False)
    truth = data.truth(root, name)
    truth.parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame({"row_id": ids, "family": labels, "native_detail": labels}).to_parquet(truth, index=False)
    return obs


def test_hgb_fit_seal_then_reveal_truth(tmp_path, monkeypatch):
    from ids.flow_history import trees

    root = tmp_path
    (root / "evidence").mkdir()
    reg = {"sources": [{"name": k, "role": k} for k in "PCE"]}
    for tag, n, ph in [("P", 8300, 1), ("C", 802, 2), ("E", 992, 3)]:
        synthetic_input(root, tag, n, ph)
    monkeypatch.setitem(trees.PARAMETERS, "max_iter", 2)
    experiment.training(root, reg)
    assert pd.read_parquet(root / "evidence/train-selected.parquet").shape[0] == 8192
    for arm in SINGLE:
        experiment.fit(root, "hgb", arm, 1)
    experiment.calibrate(root, reg, "hgb", 1)
    future_truth = data.truth(root, "E")
    hidden = future_truth.with_suffix(".hidden")
    future_truth.rename(hidden)
    try:
        files = experiment.blind_score(root, "E", "hgb", 1)
        assert all(Path(p).is_file() for p in files)
    finally:
        hidden.rename(future_truth)
    evidence = read_json(evaluation.judge(root, "E", ("hgb",))[0])
    flags = pq.read_table(files[0])["flags"].to_numpy()
    labels = pd.read_parquet(future_truth).family
    raw = evaluation.unpack(flags, "raw")
    revised = evaluation.unpack(flags, "contact_add")
    counts = evidence["groups"]["hgb"]["pairs"]["raw_vs_contact_add"]["counts"]
    for family in labels.unique():
        mask = labels.eq(family).to_numpy()
        assert counts["family/" + family]["gained"] == int((~raw & revised & mask).sum())
        assert counts["family/" + family]["lost"] == int((raw & ~revised & mask).sum())
    pq.write_table(pa.table({"flags": flags[:-1]}), files[0])
    with pytest.raises(ValueError):
        evaluation.judge(root, "E", ("hgb",))


def test_seeded_widths_share_initial_raw_function_and_change_by_seed():
    import torch
    from ids.flow_history.model import _network

    x = torch.ones((13, 16))
    raw = _network(8, seed=17, base_width=8)
    contact = _network(16, seed=17, base_width=8)
    changed = _network(8, seed=29, base_width=8)
    torch.testing.assert_close(raw(x[:, :8]), contact(x))
    assert not torch.allclose(raw(x[:, :8]), changed(x[:, :8]))


def test_family_allocation_changes_only_attack_mass_within_training():
    family = pd.Series(["Benign"] * 4096 + ["Scanning"] * 4040 + ["C&C"] * 56)
    w, weights = experiment.family_weights(family)
    assert weights["Benign"] == 1
    for name in ("Scanning", "C&C"):
        assert w[family.eq(name)].sum() == pytest.approx(2048)
    assert w.sum() == pytest.approx(len(w))
