import json

import pandas as pd
import pytest

from ids.context_cache.lab_data import FEATURES
from ids.flow_history.features import HISTORY_COLUMNS
from ids.flow_history.signal import diagnose


def row(port, family, hist, dst=443):
    values = dict.fromkeys(FEATURES, 0)
    values.update(
        L4_SRC_PORT=port,
        L4_DST_PORT=dst,
        PROTOCOL=6,
        L7_PROTO=91,
        IN_BYTES=44,
        IN_PKTS=1,
        FLOW_DURATION_MILLISECONDS=1,
        TCP_FLAGS=2,
        CLIENT_TCP_FLAGS=2,
        LONGEST_FLOW_PKT=44,
        SHORTEST_FLOW_PKT=44,
        MAX_IP_PKT_LEN=44,
        SRC_TO_DST_AVG_THROUGHPUT=352000,
        NUM_PKTS_UP_TO_128_BYTES=1,
        TCP_WIN_MAX_IN=1024,
    )
    return {
        **values,
        **dict(zip(HISTORY_COLUMNS, hist)),
        "label": int(family != "Benign"),
        "attack_family": family,
    }


def test_split_mixed_inputs_and_positive_direction_with_ties():
    rows = [
        row(1, "Scanning", (2, 1, 1, 0.5)),
        row(1, "Benign", (2, 1, 1, 0.5)),
        row(2, "Scanning", (1, 1, 1, 0)),
        row(2, "Benign", (0, 0, 0, 0)),
        row(1, "DDoS", (2, 1, 1, 0.5)),
        row(1, "Scanning", (4, 1, 1, 0), dst=80),
    ]
    result = diagnose(pd.DataFrame(rows))
    assert result["syn47_scans"] == result["syn47_normals"] == 2
    assert result["syn47_other_attack_rows_excluded_from_pairing"] == 1
    assert (
        result["syn47_scans_with_nonempty_history"] == 2
        and result["syn47_normals_with_nonempty_history"] == 1
    )
    variable = result["fixed_direction_variables"]["hist_flows_60s"]
    assert variable["scan_above"] == 2 and variable["tied"] == 1 and variable["scan_below"] == 1
    assert variable["auc_higher_means_scanning"] == 0.625
    collision = result["identical_input_scan_normal_collisions"]
    assert (
        collision["48_fields"]["mixed_inputs"]
        == collision["48_fields"]["minimum_errors_for_deterministic_per_row_decision"]
        == 2
    )
    assert (
        collision["52_fields"]["mixed_inputs"]
        == collision["52_fields"]["minimum_errors_for_deterministic_per_row_decision"]
        == 1
    )
    assert collision["formerly_mixed_48_patterns_with_no_mixed_52_child"] == 1
    assert result == diagnose(pd.DataFrame(rows[::-1]))


@pytest.mark.parametrize("empty", [False, True])
def test_single_class_and_empty_role_report_no_pairwise_auc(empty):
    frame = pd.DataFrame([row(1, "Benign", (0, 0, 0, 0))])
    result = diagnose(frame.iloc[:0] if empty else frame)
    json.dumps(result, allow_nan=False)
    for summary in result["fixed_direction_variables"].values():
        assert summary["auc_higher_means_scanning"] is None and summary["pairs"] == 0
    assert (
        result["identical_input_scan_normal_collisions"]["52_fields"][
            "minimum_errors_for_deterministic_per_row_decision"
        ]
        == 0
    )
