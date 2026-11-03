"""Whole-group allocation tests use in-memory synthetic frames only."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from ids.risk_data import validate_risk_roles
from ids.risk_roles import make_risk_roles


def _mixed_groups(n, *, prefix="train"):
    size = 2 * n
    frame = pd.DataFrame({"sample_id": [f"{prefix}-{i:04d}" for i in range(size)],
                          "feature_hash": [f"{prefix}-feature-{i:04d}" for i in range(size)],
                          "label": [0, 1] * n, "attack_cat": ["Normal", "Attack"] * n,
                          "group_id": [f"{prefix}-joint-{i // 2:04d}" for i in range(size)],
                          "value": np.arange(size, dtype=float)})
    frame.attrs = {"group_column": "group_id", "feature_schema": {"name": "cicids2017-synthetic"}}
    return frame


@pytest.fixture
def frames():
    return _mixed_groups(30), _mixed_groups(4, prefix="validation")


def test_complete_deterministic_roles_match_validator_and_preserve_inputs(frames):
    train, val = frames
    old_train, old_val = train.copy(deep=True), val.copy(deep=True)
    roles, manifest = make_risk_roles(train, val, seed=17)
    assert list(roles) == ["fit", "reference", "constraint"]
    assert {name: len(frame) for name, frame in roles.items()} == {"fit": 42, "reference": 6, "constraint": 12}
    accepted = validate_risk_roles(train, **roles, validation=val)
    assert manifest["role_assignment_sha256"] == accepted["role_assignment_sha256"]
    assert manifest["allocation"]["rows_removed"] == manifest["allocation"]["groups_split"] == 0
    assert manifest["allocation"]["retries"] == 0
    assert manifest["constraint_scope"] == "training_development_not_certificate"
    assert sum(manifest["allocation"]["actual_rows"].values()) == len(train)
    assert manifest["allocation"]["family_coverage_guaranteed"] is False
    json.dumps(manifest, allow_nan=False)
    again, again_manifest = make_risk_roles(train, val, seed=17)
    assert manifest == again_manifest
    for name in roles:
        pd.testing.assert_frame_equal(roles[name], again[name])
        assert roles[name].attrs == train.attrs
        expected = train[train.sample_id.isin(roles[name].sample_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(roles[name], expected)
    pd.testing.assert_frame_equal(train, old_train)
    pd.testing.assert_frame_equal(val, old_val)


def test_assignment_is_independent_of_input_row_order(frames):
    train, val = frames
    _, first = make_risk_roles(train, val, seed=17)
    _, second = make_risk_roles(train.iloc[::-1], val, seed=17)
    assert first["role_assignment_sha256"] == second["role_assignment_sha256"]
    assert first["allocation"] == second["allocation"]
    assert first["roles"]["fit"]["identity_order_sha256"] != second["roles"]["fit"]["identity_order_sha256"]


def test_seed_changes_group_assignment_but_never_retries(frames):
    train, val = frames
    _, first = make_risk_roles(train, val, seed=17)
    _, second = make_risk_roles(train, val, seed=29)
    assert first["allocation"]["group_assignment_sha256"] != second["allocation"]["group_assignment_sha256"]
    assert first["allocation"]["retries"] == second["allocation"]["retries"] == 0


def test_validation_values_and_labels_cannot_select_allocation(frames):
    train, val = frames
    _, first = make_risk_roles(train, val, seed=17)
    changed = val.assign(value=1000000, label=1-val.label, attack_cat="different_validation_family")
    _, second = make_risk_roles(train, changed, seed=17)
    assert first["allocation"] == second["allocation"]
    assert first["role_assignment_sha256"] == second["role_assignment_sha256"]


def test_reference_can_be_normal_only_and_minimum_support_overrides_soft_fractions():
    train = pd.DataFrame({"sample_id": [f"s{i}" for i in range(5)],
                          "feature_hash": [f"f{i}" for i in range(5)],
                          "label": [0, 0, 0, 1, 1], "attack_cat": ["Normal"] * 3 + ["A", "B"]})
    val = train.iloc[:2].assign(sample_id=["v0", "v1"], feature_hash=["vf0", "vf1"])
    roles, manifest = make_risk_roles(train, val, seed=17)
    assert roles["reference"].label.tolist() == [0]
    assert set(roles["fit"].label) == set(roles["constraint"].label) == {0, 1}
    assert len(pd.concat(roles.values())) == 5
    assert manifest["allocation"]["fraction_targets_are_soft"]
    assert manifest["roles"]["reference"]["missing_attack_families"] == ["A", "B"]


def test_mixed_group_cannot_be_double_booked_for_different_roles():
    rows = []
    for group, labels in (("huge_attack", [1] * 1000), ("two_normals", [0, 0]),
                          ("mixed", [0, 1]), ("one_normal", [0])):
        for index, label in enumerate(labels):
            rows.append({"sample_id": f"{group}-{index}", "feature_hash": f"{group}-feature-{index}",
                         "group_id": group, "label": label, "attack_cat": "Normal" if label == 0 else "A"})
    train = pd.DataFrame(rows)
    val = pd.DataFrame([{"sample_id": "v", "feature_hash": "vf", "group_id": "vg",
                         "label": 0, "attack_cat": "Normal"}])
    # A mere remaining-normal/attack group count permits an early assignment
    # which needs this one mixed group in two different roles. Do not change
    # the seed or backtrack: rule out that assignment before taking the step.
    roles, manifest = make_risk_roles(train, val, seed=4)
    assert set(roles["fit"].label) == set(roles["constraint"].label) == {0, 1}
    assert 0 in set(roles["reference"].label)
    assert manifest["allocation"]["actual_rows"] == {"fit": 1001, "reference": 2, "constraint": 2}
    assert manifest["allocation"]["retries"] == 0
    assert manifest["allocation"]["binary_support_feasibility_preserved"]


def test_giant_attack_group_is_kept_whole_and_missing_family_is_reported(frames):
    train, val = frames
    giant = train.iloc[[1] * 100].copy().reset_index(drop=True)
    giant["sample_id"] = [f"giant-{i}" for i in range(len(giant))]
    giant["feature_hash"] = "giant-pattern"
    giant["group_id"] = "giant-union"
    giant["attack_cat"] = "Portscan"
    original = pd.concat([train, giant], ignore_index=True)
    original.attrs = copy.deepcopy(train.attrs)
    roles, manifest = make_risk_roles(original, val, seed=17)
    located = [name for name, frame in roles.items() if frame.group_id.eq("giant-union").any()]
    assert len(located) == 1
    assert roles[located[0]].group_id.eq("giant-union").sum() == 100
    assert sum(len(frame) for frame in roles.values()) == 160
    assert manifest["allocation"]["largest_groups"][0]["n_rows"] == 100
    for name in set(roles) - set(located):
        assert "Portscan" in manifest["roles"][name]["missing_attack_families"]
        assert manifest["roles"][name]["families"]["Portscan"] == 0


@pytest.mark.parametrize("kind", ["one_group", "one_attack_group", "two_normal_groups"])
def test_indivisible_support_failure_is_rejected_without_splitting(frames, kind):
    train, val = frames
    if kind == "one_group":
        train = train.assign(group_id="one-huge-group")
    elif kind == "one_attack_group":
        train.loc[train.label == 1, "group_id"] = "one-attack-group"
    else:
        train.loc[train.label == 0, "group_id"] = [f"normal-{i % 2}" for i in range(30)]
    before = train.copy(deep=True)
    with pytest.raises(ValueError, match="Whole-group role support is impossible"):
        make_risk_roles(train, val, seed=17)
    pd.testing.assert_frame_equal(train, before)


def test_cic_transitive_union_and_exact_feature_relations_do_not_cross_roles(frames):
    train, val = frames
    # A-B share a feature, B-C share a session; prepared union binds all four
    # rows of the two formerly separate pairs, without reconstructing raw time.
    train.loc[:3, "group_id"] = "transitive-union"
    train.loc[0:1, "feature_hash"] = "same-feature"
    train["session_hash"] = [f"session-{i}" for i in range(len(train))]
    train.loc[1:2, "session_hash"] = "same-session"
    val = val.assign(session_hash=[f"val-session-{i}" for i in range(len(val))])
    roles, manifest = make_risk_roles(train, val, seed=17)
    assert manifest["group_column"] == "group_id"
    assert sum(frame.group_id.eq("transitive-union").any() for frame in roles.values()) == 1
    validate_risk_roles(train, **roles, validation=val)


def test_feature_hash_cannot_claim_two_cic_union_groups(frames):
    train, val = frames
    train.loc[2, "feature_hash"] = train.loc[0, "feature_hash"]
    with pytest.raises(ValueError, match="feature_hash maps to multiple joint groups"):
        make_risk_roles(train, val, seed=17)


@pytest.mark.parametrize("fallback", ["group_hash", "feature_hash"])
def test_supported_group_fallbacks(frames, fallback):
    train, val = frames
    if fallback == "group_hash":
        train = train.rename(columns={"group_id": "group_hash"})
        val = val.rename(columns={"group_id": "group_hash"})
    else:
        train, val = train.drop(columns="group_id"), val.drop(columns="group_id")
        train.attrs, val.attrs = {}, {}
    _, manifest = make_risk_roles(train, val, seed=17)
    assert manifest["group_column"] == fallback


def test_marked_cic_missing_joint_groups_is_rejected(frames):
    train, val = frames
    with pytest.raises(ValueError, match="requires its prepared joint"):
        make_risk_roles(train.drop(columns="group_id"), val.drop(columns="group_id"), seed=17)


@pytest.mark.parametrize("column", ["sample_id", "feature_hash", "group_id"])
def test_validation_overlap_is_rejected(frames, column):
    train, val = frames
    val.loc[0, column] = train.loc[0, column]
    with pytest.raises(ValueError, match=f"overlapping {column}"):
        make_risk_roles(train, val, seed=17)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "17", None])
def test_invalid_seed_is_rejected(frames, seed):
    with pytest.raises(ValueError, match="seed"):
        make_risk_roles(*frames, seed=seed)


@pytest.mark.parametrize("ref,con", [(0, .2), (-.1, .2), (.1, 0), (.1, -.1), (True, .2),
                                     (float("nan"), .2), (.1, float("inf")), (".1", .2),
                                     (.5, .5), (.8, .3), (1, .2)])
def test_invalid_fractions_are_rejected(frames, ref, con):
    with pytest.raises(ValueError, match="fraction"):
        make_risk_roles(*frames, seed=17, reference_fraction=ref, constraint_fraction=con)


def test_no_io_download_or_rng_is_used(frames, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("allocation accessed files or random state")

    with monkeypatch.context() as context:
        context.setattr("builtins.open", forbidden)
        context.setattr(pd, "read_csv", forbidden)
        context.setattr(pd, "read_parquet", forbidden)
        context.setattr(np.random, "default_rng", forbidden)
        _, manifest = make_risk_roles(*frames, seed=17)
    assert manifest["allocation"]["retries"] == 0
