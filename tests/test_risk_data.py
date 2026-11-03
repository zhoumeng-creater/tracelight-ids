"""Synthetic role validation: no file access, splitting or model training."""
import copy
import json

import pandas as pd
import pytest

from ids.risk_data import role_identity, validate_risk_roles


@pytest.fixture
def roles():
    original = pd.DataFrame({
        "sample_id": [f"row-{i}" for i in range(8)],
        "feature_hash": ["fa", "fa", "fb", "fc", "fd", "fe", "ff", "fg"],
        "label": [0, 0, 1, 1, 0, 0, 0, 1],
        "attack_cat": ["Normal", "Normal", "A", "B", "Normal", "Normal", "Normal", "A"],
        "value": range(8),
    })
    validation = pd.DataFrame({"sample_id": ["val-n", "val-a"], "feature_hash": ["vn", "va"],
                               "label": [0, 1], "attack_cat": ["Normal", "Unseen"], "value": [90, 91]})
    return {"original_train": original, "fit": original.iloc[:4].copy(),
            "reference": original.iloc[4:6].copy(), "constraint": original.iloc[6:].copy(),
            "validation": validation}


def test_explicit_partition_allows_normal_only_reference_and_reports_missing_families(roles):
    before = {name: frame.copy(deep=True) for name, frame in roles.items()}
    result = validate_risk_roles(**roles)
    assert result["checks"] == dict.fromkeys(
        ["exhaustive", "mutually_disjoint", "original_identity_preserved", "validation_isolated"], True)
    assert result["roles"]["reference"]["labels"] == {"0": 2, "1": 0}
    assert result["roles"]["reference"]["missing_attack_families"] == ["A", "B"]
    assert result["roles"]["reference"]["families"]["A"] == 0
    assert result["roles"]["validation"]["families_not_in_original_train"] == ["Unseen"]
    assert result["constraint_scope"] == "training_development_not_certificate"
    assert result["statistical_independence_verified"] is False
    json.dumps(result, allow_nan=False)
    for name, frame in roles.items():
        pd.testing.assert_frame_equal(frame, before[name])


def test_content_hash_is_order_independent_and_order_hash_binds_exact_alignment(roles):
    first = validate_risk_roles(**roles)
    roles["fit"] = roles["fit"].iloc[::-1]
    second = validate_risk_roles(**roles)
    assert first["role_assignment_sha256"] == second["role_assignment_sha256"]
    assert first["roles"]["fit"]["identity_sha256"] == second["roles"]["fit"]["identity_sha256"]
    assert first["roles"]["fit"]["identity_order_sha256"] != second["roles"]["fit"]["identity_order_sha256"]


@pytest.mark.parametrize("key,value", [("label", 0), ("attack_cat", "Changed"), ("feature_hash", "changed")])
def test_original_label_family_and_identity_mutations_are_rejected(roles, key, value):
    roles["fit"].loc[roles["fit"].index[2], key] = value
    with pytest.raises(ValueError, match="were changed"):
        validate_risk_roles(**roles)


def test_foreign_and_missing_rows_cannot_fake_exhaustiveness(roles):
    roles["constraint"].loc[roles["constraint"].index[-1], "sample_id"] = "foreign"
    with pytest.raises(ValueError, match="exhaust"):
        validate_risk_roles(**roles)


def test_dropped_original_row_is_rejected(roles):
    roles["fit"] = roles["fit"].iloc[1:]
    with pytest.raises(ValueError, match="exhaust"):
        validate_risk_roles(**roles)


def test_duplicate_sample_id_across_roles_is_rejected(roles):
    roles["reference"] = pd.concat([roles["reference"], roles["fit"].iloc[[2]]])
    with pytest.raises(ValueError, match="overlapping sample_id"):
        validate_risk_roles(**roles)


def test_exact_feature_group_cannot_be_split_even_with_distinct_ids(roles):
    roles["reference"] = pd.concat([roles["reference"], roles["fit"].iloc[[0]]])
    roles["fit"] = roles["fit"].iloc[1:]
    with pytest.raises(ValueError, match="overlapping feature_hash"):
        validate_risk_roles(**roles)


@pytest.mark.parametrize("key", ["sample_id", "feature_hash"])
def test_validation_is_isolated_by_each_relation(roles, key):
    roles["validation"].loc[0, key] = roles["original_train"].iloc[0][key]
    with pytest.raises(ValueError, match=f"original_train/validation: overlapping {key}"):
        validate_risk_roles(**roles)


def _cic(roles):
    result = copy.deepcopy(roles)
    for name, frame in result.items():
        frame["group_id"] = frame.feature_hash.map(lambda value: "joint-" + value)
        frame["group_hash"] = frame.group_id
        frame["session_hash"] = frame.feature_hash.map(lambda value: "session-" + value)
        frame.attrs["feature_schema"] = {"name": "cicids2017-example"}
    return result


@pytest.mark.parametrize("drop_id", [False, True])
def test_cic_prefers_union_id_and_allows_group_hash_fallback(roles, drop_id):
    roles = _cic(roles)
    if drop_id:
        roles = {name: frame.drop(columns="group_id") for name, frame in roles.items()}
    expected = "group_hash" if drop_id else "group_id"
    assert validate_risk_roles(**roles)["group_column"] == expected
    with pytest.raises(ValueError, match="cannot weaken"):
        validate_risk_roles(**roles, group_column="feature_hash")


def test_transitive_cic_union_group_cannot_cross_roles(roles):
    roles = _cic(roles)
    # A/B share a feature; B/C share a session. The prepared union binds all.
    for name in ("original_train", "fit", "reference", "constraint"):
        frame = roles[name]
        frame.loc[frame.sample_id.isin(["row-0", "row-1", "row-4"]), ["group_id", "group_hash"]] = "joined"
        frame.loc[frame.sample_id.isin(["row-1", "row-4"]), "session_hash"] = "bridge"
    with pytest.raises(ValueError, match="overlapping group_id"):
        validate_risk_roles(**roles)


@pytest.mark.parametrize("drop_id", [False, True])
def test_cic_validation_cannot_reuse_union_group(roles, drop_id):
    roles = _cic(roles)
    roles["validation"].loc[0, ["group_id", "group_hash"]] = roles["fit"].iloc[0].group_id
    group = "group_hash" if drop_id else "group_id"
    if drop_id:
        roles = {name: frame.drop(columns="group_id") for name, frame in roles.items()}
    with pytest.raises(ValueError, match=f"original_train/validation: overlapping {group}"):
        validate_risk_roles(**roles)


@pytest.mark.parametrize("key", ["group_id", "group_hash", "session_hash"])
def test_union_or_session_identity_tampering_is_rejected(roles, key):
    roles = _cic(roles)
    roles["fit"].loc[roles["fit"].index[2], key] = "changed"
    with pytest.raises(ValueError, match="disagree|were changed"):
        validate_risk_roles(**roles)


def test_marked_cic_cannot_silently_fall_back_to_feature_only_groups(roles):
    roles["original_train"].attrs["group_column"] = "group_id"
    with pytest.raises(ValueError, match="requires its prepared joint"):
        validate_risk_roles(**roles)


def test_inconsistent_prepared_union_mapping_is_rejected(roles):
    roles = _cic(roles)
    roles["original_train"].loc[1, ["group_id", "group_hash"]] = "split-duplicate"
    with pytest.raises(ValueError, match="feature_hash maps to multiple joint groups"):
        validate_risk_roles(**roles)


@pytest.mark.parametrize("change", ["missing", "null", "duplicate", "empty", "label", "bool", "columns"])
def test_malformed_or_missing_identity_is_rejected(roles, change):
    frame = roles["reference"]
    if change == "missing":
        roles["reference"] = frame.drop(columns="feature_hash")
    elif change == "null":
        frame.loc[frame.index[0], "sample_id"] = None
    elif change == "duplicate":
        frame.loc[frame.index[1], "sample_id"] = frame.iloc[0].sample_id
    elif change == "empty":
        roles["reference"] = frame.iloc[:0]
    elif change == "label":
        frame.loc[frame.index[0], "label"] = 2
    elif change == "bool":
        roles["reference"] = frame.assign(label=False)
    else:
        roles["reference"] = pd.concat([frame, frame[["sample_id"]]], axis=1)
    with pytest.raises(ValueError):
        validate_risk_roles(**roles)


def test_large_indivisible_family_is_reported_without_automatic_repair():
    large = pd.DataFrame({"sample_id": [f"portscan-{i}" for i in range(200)],
                          "feature_hash": ["large-pattern"] * 200, "group_id": ["large-joint"] * 200,
                          "label": [1] * 200, "attack_cat": ["Portscan"] * 200})
    normals = pd.DataFrame({"sample_id": ["n0", "n1"], "feature_hash": ["n0", "n1"],
                            "group_id": ["n0", "n1"], "label": [0, 0], "attack_cat": ["Normal"] * 2})
    validation = normals.iloc[[0]].assign(sample_id="val", feature_hash="val", group_id="val")
    result = validate_risk_roles(pd.concat([large, normals]), large, normals.iloc[[0]],
                                 normals.iloc[[1]], validation)
    assert result["roles"]["fit"]["largest_group_rows"] == 200
    assert result["roles"]["constraint"]["families"]["Portscan"] == 0
    assert result["roles"]["constraint"]["missing_attack_families"] == ["Portscan"]
    assert result["roles"]["original_train"]["n_rows"] == 202


def test_role_identity_is_a_detached_copy_and_excludes_model_features(roles):
    identity = role_identity(roles["fit"])
    assert "value" not in identity
    identity.loc[0, "label"] = 1
    assert roles["fit"].iloc[0].label == 0


def test_missing_optional_cic_identity_column_is_rejected(roles):
    roles = _cic(roles)
    roles["constraint"] = roles["constraint"].drop(columns="session_hash")
    with pytest.raises(ValueError, match="identity columns differ"):
        validate_risk_roles(**roles)


def test_assignment_hash_changes_when_whole_group_changes_role(roles):
    first = validate_risk_roles(**roles)
    roles["constraint"] = pd.concat([roles["constraint"], roles["reference"].iloc[[0]]])
    roles["reference"] = roles["reference"].iloc[1:]
    second = validate_risk_roles(**roles)
    assert first["role_assignment_sha256"] != second["role_assignment_sha256"]
    assert first["roles"]["original_train"] == second["roles"]["original_train"]


def test_validation_uses_no_file_access(roles, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("role validation attempted file access")

    with monkeypatch.context() as context:
        context.setattr("builtins.open", forbidden)
        context.setattr(pd, "read_csv", forbidden)
        context.setattr(pd, "read_parquet", forbidden)
        result = validate_risk_roles(**roles)
    assert result["roles"]["original_train"]["n_rows"] == 8
