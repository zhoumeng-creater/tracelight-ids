"""Synthetic contracts for old balanced rows and new temporal roles."""
import copy

import numpy as np
import pandas as pd
import pytest

from ids.common import sha256_file
from ids.model_training import data
from ids.model_training.protocol import specifications, tasks
from ids.threshold_validation.observations import RAW_FEATURES


def fixture():
    registration, design = specifications()
    design = copy.deepcopy(design)
    bounds, parts = {}, {}
    for group_no, role in enumerate(("F", "V", "C", "E", "D")):
        cells = (registration["evaluation_cells"][3:] if role == "D" else design["roles"][role]["cells"])
        rows = []
        for cell_no, cell in enumerate(cells):
            start = 1_000_000 * (group_no + 1) + cell_no * 100
            bounds[cell] = {"first_flow_end_ms": start, "last_flow_end_ms": start + 90}
            for i in range(2 if role == "C" else 3):
                row = {name: float(i) for name in RAW_FEATURES}
                row.update(source_id=cell, row_id=i, flow_end_ms=start+i, ip_protocol=17,
                           family="ExternalC2" if i == 2 else "Benign", label=int(i == 2),
                           domain="target", weight=26.2 if i == 2 else 0.45)
                rows.append(row)
        parts[role] = pd.DataFrame(rows)
        design["roles"][role]["normal"] = len(cells) * 2
        if role != "C":
            design["roles"][role]["attack"] = len(cells)
    design["roles"]["P"].update(normal=12, attack=6)
    train = pd.concat([parts["F"], parts["V"]], ignore_index=True)
    source = train.iloc[:1].assign(source_id="historical-source", domain="source", family="Scanning", label=1)
    train = pd.concat([train, source], ignore_index=True)
    ed = pd.concat([parts["E"], parts["D"]], ignore_index=True)
    return train, parts["C"], ed, design, registration, bounds


def test_old_balanced_weights_are_audited_new_weights_are_natural():
    roles, bounds = data.split_roles(*fixture())
    assert len(roles["P"]) == 18
    assert roles["P"].weight.eq(1).all()
    assert set(roles["P"].historical_weight) == {0.45, 26.2}
    assert roles["P"].domain.eq("target").all()
    assert bounds["F"]["last"] < bounds["V"]["first"]
    assert roles["C"].label.eq(0).all()


@pytest.mark.parametrize("mutation,match", [
    ("calibration_attack", "C must contain"),
    ("missing_cell", "exactly the registered"),
    ("chronology", "chronology"),
    ("clock", "clocks"),
    ("unresolved", "Unexpected target family"),
    ("duplicate", "physical row identity"),
])
def test_broken_roles_fail_before_any_model_use(mutation, match):
    train, cal, ed, design, registration, bounds = fixture()
    if mutation == "calibration_attack":
        cal.loc[0, ["family", "label"]] = ["ExternalC2", 1]
    elif mutation == "missing_cell":
        ed = ed.loc[ed.source_id.ne(registration["evaluation_cells"][-1])]
    elif mutation == "chronology":
        bounds[design["roles"]["F"]["cells"][0]]["last_flow_end_ms"] = 99_000_000
    elif mutation == "clock":
        ed.loc[0, "flow_end_ms"] = -100
    elif mutation == "unresolved":
        ed.loc[0, "family"] = "Unresolved"
    elif mutation == "duplicate":
        train = pd.concat([train, train.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match=match):
        data.split_roles(train, cal, ed, design, registration, bounds)


def test_prediction_interface_drops_labels_identifiers_and_historical_scores():
    train, *_ = fixture()
    actual = data.blind(train.assign(old_score=123))
    assert list(actual) == list(RAW_FEATURES)
    np.testing.assert_array_equal(actual.to_numpy(), train[list(RAW_FEATURES)].to_numpy())


def test_frozen_input_digest_is_checked(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text("fixed", encoding="utf8")
    registration = {"required_r26_artifacts": {"protocol.json": sha256_file(path)}}
    data.verify_r26(tmp_path, registration)
    path.write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="missing or changed"):
        data.verify_r26(tmp_path, registration)


def test_fixed_task_budget_has_all_seeds_and_matched_ablation():
    registration, design = specifications()
    planned = tasks(design)
    assert [t["id"] for t in planned] == ["hgb", "tabm_ple_s17", "tabm_ple_s29", "tabm_ple_s43",
                                          "tabm_no_ple_s17", "tabm_no_ple_s29", "tabm_no_ple_s43"]
    assert len(planned) * 2 == registration["max_fits"] == 14
