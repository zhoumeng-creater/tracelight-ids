"""Input identity, role separation and read-only historical binding contracts."""
from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from ids.model_training_redesign import data


def frame(source, start, labels=(0, 1)):
    n = len(labels)
    return pd.DataFrame({
        "source_id": [source] * n, "row_id": np.arange(1, n + 1),
        "flow_end_ms": np.arange(start, start + n), "label": labels,
        "family": ["Benign" if y == 0 else "ExternalC2" for y in labels],
        "ip_protocol": [6 if "TQH-A" in source or "TQH-D" in source else 17] * n,
        "orig_port": [12345] * n, "resp_port": [443] * n,
        "duration_seconds": [0.25] * n, "orig_pkts": [2] * n, "resp_pkts": [3] * n,
        "orig_ip_bytes": [100] * n, "resp_ip_bytes": [200] * n,
    })


@pytest.fixture
def roles_and_design():
    roles = {
        "F": pd.concat([frame("TQH-A_i30_j0", 10), frame("TQH-B_i30_j0", 40)], ignore_index=True),
        "V": pd.concat([frame("TQH-A_i300_j0", 20), frame("TQH-B_i300_j0", 50)], ignore_index=True),
        "C": frame("TQH-B_i1800_j0", 60, (0,)),
        "E_B": frame("TQH-B_i3600_j0", 70),
        "E_D": frame("TQH-D_i30_j0", 80),
    }
    design = {"roles": {}}
    for name, value in roles.items():
        cells = []
        for source, group in value.groupby("source_id"):
            cells.append({"source_id": source, "profile": data._profile(source),
                          "normal": int(group.label.eq(0).sum()), "attack": int(group.label.eq(1).sum()),
                          "first_flow_end_ms": int(group.flow_end_ms.min()),
                          "last_flow_end_ms": int(group.flow_end_ms.max())})
        design["roles"][name] = {"cells": cells, "rows": len(value),
                                "normal": int(value.label.eq(0).sum()), "attack": int(value.label.eq(1).sum())}
    return roles, design


def test_chronology_is_within_profile_not_global(roles_and_design):
    roles, design = roles_and_design
    result, bounds = data.validate_roles(roles, design)
    assert bounds["F"]["last"] > bounds["V"]["first"]
    assert result["F"].profile.tolist() == ["A", "A", "B", "B"]


def test_truth_must_not_coerce_unknown_to_normal(roles_and_design):
    roles, design = roles_and_design
    roles["F"].loc[0, "family"] = "Unresolved"
    with pytest.raises(ValueError, match="Unresolved"):
        data.validate_roles(roles, design)


def test_family_label_disagreement_is_rejected():
    value = frame("TQH-B_i30_j0", 10)
    value.loc[1, "label"] = 0
    with pytest.raises(ValueError, match="Family and binary"):
        data._frame(value)


def test_profile_cannot_be_relabeled():
    value = frame("TQH-B_i30_j0", 10)
    value["profile"] = "A"
    with pytest.raises(ValueError, match="Profile metadata"):
        data._frame(value)


def test_complete_cell_counts_not_only_total(roles_and_design):
    roles, design = roles_and_design
    # Preserve role totals, move one positive between source cells.
    roles["F"].loc[1, "label"] = 0
    roles["F"].loc[1, "family"] = "Benign"
    roles["F"].loc[2, "label"] = 1
    roles["F"].loc[2, "family"] = "ExternalC2"
    with pytest.raises(ValueError, match="Complete cell counts"):
        data.validate_roles(roles, design)


def test_within_profile_reversal_is_rejected(roles_and_design):
    roles, design = roles_and_design
    roles["V"].loc[roles["V"].source_id.eq("TQH-A_i300_j0"), "flow_end_ms"] = [8, 9]
    design["roles"]["V"]["cells"][0].update(first_flow_end_ms=8, last_flow_end_ms=9)
    with pytest.raises(ValueError, match="Within-profile"):
        data.validate_roles(roles, design)


def test_validation_identity_leak_is_rejected(roles_and_design):
    roles, design = roles_and_design
    roles["V"] = roles["F"].copy()
    design["roles"]["V"] = copy.deepcopy(design["roles"]["F"])
    with pytest.raises(ValueError, match="identities overlap"):
        data.validate_roles(roles, design)


def test_frozen_alignment_accepts_reorder_but_rejects_numeric_change():
    original = frame("TQH-B_i30_j0", 10)
    reordered = original.iloc[::-1].reset_index(drop=True)
    data.align_frozen(reordered, original, name="B")
    reordered.loc[0, "duration_seconds"] += 1e-12
    with pytest.raises(ValueError, match="differ from frozen"):
        data.align_frozen(reordered, original, name="B")


def test_offline_binding_detects_file_change_and_path_escape(tmp_path):
    path = tmp_path / "inputs.parquet"
    path.write_bytes(b"frozen content")
    required = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    data.verify_files(tmp_path, required)
    path.write_bytes(b"edited content")
    with pytest.raises(ValueError, match="missing or changed"):
        data.verify_files(tmp_path, required)
    with pytest.raises(ValueError, match="escapes"):
        data.verify_files(tmp_path, {"../elsewhere": "0" * 64})


def test_prediction_projection_excludes_truth_ports_and_identity():
    value = frame("TQH-B_i30_j0", 10)
    value["profile"], value["weight"] = "B", 99.0
    projected = data.blind(value)
    assert projected.columns.tolist() == data.PREDICTORS
    assert not {"profile", "label", "family", "source_id", "row_id", "orig_port", "resp_port", "weight"} & set(projected)


def test_historical_decisions_are_read_without_reconstruction(tmp_path):
    truth = frame("TQH-D_i30_j0", 80)
    saved = truth.assign(alert=[True, False], ready=True, model_id="old", threshold=None)
    saved.to_parquet(tmp_path / "old.parquet", index=False)
    result = data.load_reference_decisions(tmp_path, {"r29_tabm_ple_s17": "old.parquet"}, truth)
    assert result.alert.tolist() == [True, False]
    assert result.model_id.unique().tolist() == ["r29_tabm_ple_s17"]
    assert result.score.isna().all()
    saved.loc[0, "row_id"] = 99
    saved.to_parquet(tmp_path / "old.parquet", index=False)
    with pytest.raises(ValueError, match="differ from frozen"):
        data.load_reference_decisions(tmp_path, {"old": "old.parquet"}, truth)


def test_native_uid_map_and_author_label_count_are_retained(tmp_path, monkeypatch):
    source = "TQH-A_i30_j0"
    raw = frame(source, 10)
    obs = pa.Table.from_pandas(raw[["row_id", "flow_end_ms", *data.RAW_FEATURES]], preserve_index=False)
    labels = pa.Table.from_pydict({"row_id": [1, 2], "family": ["Benign", "ExternalC2"],
                                  "native_label": ["benign", "malicious_c2"]})
    receipt = {"rows": 2, "sha256": "registered"}
    monkeypatch.setattr(data.tqh, "_cell", lambda archive, prefix: (obs, labels, receipt))
    path = tmp_path / "source.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("A/A_i30_j0/conn.log", '\n'.join(json.dumps({"uid": uid}) for uid in ["native-2", "native-1"]))
        archive.writestr("A/A_i30_j0/manifest.json", json.dumps({"tooling": {"zeek_version": "7.0.11"}}))
    spec = {"name": source, "profile": "A", "unit": "A_i30_j0", "archive": path.name, **receipt}
    expected = {"source_binding": {"sha256": "registered"}, "author_label_counts": {"benign": 1, "malicious_c2": 1}}
    with ZipFile(path) as archive:
        values, uid, audit = data._read_raw_cell(archive, spec, expected)
    assert uid.uid.tolist() == ["native-2", "native-1"]
    assert uid.row_id.tolist() == [1, 2]
    assert values.label.tolist() == [0, 1]
    assert audit["tooling"]["zeek_version"] == "7.0.11"
    expected["author_label_counts"]["benign"] = 2
    with ZipFile(path) as archive, pytest.raises(ValueError, match="Author label counts"):
        data._read_raw_cell(archive, spec, expected)


def test_persisted_manifest_rejects_modified_inputs(tmp_path, roles_and_design):
    roles, design = roles_and_design
    roles, bounds = data.validate_roles(roles, design)
    roles["ED"] = pd.concat([roles["E_B"], roles["E_D"]], ignore_index=True)
    roles["uid_mapping"] = roles["F"][data.IDENTITY].assign(uid=["a", "b", "c", "d"])
    roles["reference_decisions"] = roles["ED"].assign(model_id="legacy", alert=False, ready=True)
    paths = data.persist_inputs(tmp_path, roles, {"chronology": bounds}, {"legacy_hgb": {"threshold": 0.9}})
    loaded, binding, refs = data.load_persisted_inputs(tmp_path)
    assert len(paths) == 11
    pd.testing.assert_frame_equal(loaded["F"], roles["F"])
    assert refs["legacy_hgb"]["threshold"] == 0.9
    assert binding["chronology"] == bounds
    target = tmp_path / "inputs/F.parquet"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="missing or changed"):
        data.load_persisted_inputs(tmp_path)
