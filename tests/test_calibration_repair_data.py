"""Synthetic membership and floor arithmetic checks; no model construction."""
from copy import deepcopy

import numpy as np
import pytest

from scripts.calibration_repair.data import assemble_groups, _signature_audit


def fixture():
    rows = []
    for source in ("a", "b"):
        for i in range(201):
            rows.append({"flow_id": f"{source}:{i:03}", "source_group": source,
                         "dataset_role": "F_base", "label": int(i == 200)})
    units = [{"unit_id": f"{source}:{part}", "role": "F_base",
              "flow_ids": [f"{source}:{i:03}" for i in range(start, stop)]}
             for source in ("a", "b") for part, start, stop in (("low", 0, 99), ("high", 99, 200))]
    components = [{"component_id": "action1", "normal_flow_ids": ["a:000", "a:001"]}]
    return rows, units, components


def test_intact_groups_cover_normal_rows_and_keep_attacks_separate():
    rows, units, components = fixture()
    groups = assemble_groups(rows, units, components)
    assert [len(x["indices"]) for x in groups["references"]] == [200, 200]
    assert sorted(len(x["indices"]) for x in groups["targets"]) == [99, 99, 101, 101]
    assert groups["attacks"] == [200, 401]
    assert groups["normal_partition"] == {"normal_rows": 400, "target_memberships": 400,
                                           "uncovered_normal_rows": 0, "overlapping_memberships": 0}
    assert groups["known_action_components"][0]["target_id"] == "a:low"


def test_floor_gap_requires_additional_boundary_ties():
    rows, units, components = fixture()
    groups = assemble_groups(rows, units, components)
    for report in groups["source_budget_arithmetic"]:
        assert report["whole_source_integer_budget"] == 2
        assert report["business_integer_budget_sum"] == 1
        assert report["floor_gap"] == 1
        assert report["minimum_boundary_tie_size_if_all_own_business_units_pass"] == 2
    # For N=200 the fixed boundary is ascending rank198. Passing total budget1
    # means at most one value above it, forcing equality at ranks198 and199.
    distinct = np.arange(200, dtype=float)
    threshold = np.sort(distinct)[-(200 // 100 + 1)]
    assert int((distinct > threshold).sum()) == 2
    distinct[198] = distinct[197]
    threshold = np.sort(distinct)[-(200 // 100 + 1)]
    assert int((distinct > threshold).sum()) == 1
    assert int((distinct == threshold).sum()) == 2


@pytest.mark.parametrize("mutation", ["overlap", "missing", "attack", "cross_source", "component"])
def test_bad_membership_or_broken_component_is_rejected(mutation):
    rows, units, components = fixture()
    if mutation == "overlap":
        units[0]["flow_ids"].append(units[1]["flow_ids"][0])
    elif mutation == "missing":
        units[0]["flow_ids"].pop()
    elif mutation == "attack":
        units[0]["flow_ids"].append("a:200")
    elif mutation == "cross_source":
        units[0]["flow_ids"].append("b:000")
    else:
        components[0]["normal_flow_ids"].append("a:199")
    with pytest.raises(ValueError):
        assemble_groups(rows, units, components)


def test_signature_audit_distinguishes_encoding_collision_from_raw_collision():
    raw = {"ip_protocol": 6, "duration_seconds": 1., "orig_pkts": 1, "resp_pkts": 1,
           "orig_ip_bytes": 100, "resp_ip_bytes": 200, "orig_port": 1234, "resp_port": 443}
    rows = [{"label": 0, "source_group": "a", "raw8": deepcopy(raw)},
            {"label": 1, "source_group": "b", "raw8": deepcopy(raw)},
            {"label": 0, "source_group": "b", "raw8": deepcopy(raw)}]
    rows[1]["raw8"]["duration_seconds"] = 1.0000000001
    values = np.array([[1.], [1.0000000001], [2.]], dtype=np.float32)
    categories = np.zeros((3, 1), dtype=np.int64)
    report = _signature_audit(rows, values, categories)
    assert report["raw_ports_zero"]["positive_negative_conflicting_inputs"] == 0
    assert report["actual_FP32_numeric_and_categories"]["positive_negative_conflicting_inputs"] == 1
    assert report["normal_encoded_signatures_shared_across_sources"] == 0
