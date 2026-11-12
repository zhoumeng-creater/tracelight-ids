import numpy as np
import pandas as pd
import pytest

from ids.common import read_json, write_json
from ids.contact_evidence import data, evaluation
from ids.contact_evidence.policy import POLICIES


def test_gate_breakdown_per_backend_and_native_label_without_rescoring(tmp_path):
    def bit(policy):
        return 1 << POLICIES.index(policy)

    anchor, contact, gate = bit("anchor"), bit("contact_add"), 128
    truth = data.truth(tmp_path, "held")
    truth.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "row_id": np.arange(1, 8),
            "family": ["Benign"] * 2 + ["Scanning"] * 5,
            "native_detail": ["a", "b", "x", "x", "y", "y", "y"],
        }
    ).to_parquet(truth)
    setting = tmp_path / "evidence" / "calibration"
    setting.mkdir(parents=True)
    blind = tmp_path / "evidence" / "blind" / "held"
    blind.mkdir(parents=True)
    original = [anchor | contact | gate, gate | contact, anchor | contact, gate, 0, gate | contact, gate]
    for learner, cutoff in (("hgb", 2.5), ("mlp17", None)):
        write_json(setting / (learner + ".json"), {"contact_add": {"threshold": cutoff}})
        flags = original if cutoff is not None else [v if v & anchor else v & ~contact for v in original]
        pd.DataFrame({"flags": np.array(flags, dtype=np.uint8)}).to_parquet(blind / (learner + ".parquet"))
    report = read_json(evaluation.judge(tmp_path, "held", groups=("hgb", "mlp17"))[0])
    for learner, learner_result in report["groups"].items():
        by_label = learner_result["gate_breakdown"]
        for row in by_label.values():
            assert row["rows"] == row["anchor_alerts"] + row["outside_anchor"]
            assert (
                row["outside_anchor"]
                == row["outside_anchor_rejected_by_gate"] + row["outside_anchor_eligible"]
            )
            assert row["outside_anchor_eligible"] == sum(
                row[k]
                for k in ("added_by_contact_rule", "eligible_not_passing_score_rule", "eligible_no_registered_cut")
            )
            assert row["final_alerts"] == row["anchor_alerts"] + row["added_by_contact_rule"]
        for key in by_label["family/Scanning"]:
            assert by_label["native/x"][key] + by_label["native/y"][key] == by_label["family/Scanning"][key]
        scan, normal = by_label["family/Scanning"], by_label["family/Benign"]
        assert normal["gate_covered"] == 2 and normal["outside_anchor_eligible"] == 1
        assert (
            scan["gate_covered"] == 3 and scan["anchor_alerts"] == 1 and scan["outside_anchor_eligible"] == 3
        )
        assert scan["added_by_contact_rule"] == (1 if learner == "hgb" else 0)
        assert scan["eligible_not_passing_score_rule"] == (2 if learner == "hgb" else 0)
        assert scan["eligible_no_registered_cut"] == (0 if learner == "hgb" else 3)


def test_gate_breakdown_rejects_non_anchor_additions_without_gate():
    hist = np.zeros(256, dtype=np.int64)
    hist[1 << POLICIES.index("contact_add")] = 1
    with pytest.raises(ValueError, match="contradict"):
        evaluation.gate_breakdown(hist, True)
