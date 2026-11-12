import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ids.common import read_json, write_json
from ids.contact_evidence import data, evaluation
from ids.contact_evidence.policy import POLICIES


FAMILIES = ["Benign", "Scanning", "Scanning", "C&C", "Benign", "Mixed", "Unresolved", "Scanning", "C&C"]
DETAILS = ["-", "scan-a", "scan-b", "C&C", "-", "C&C-HeartBeat", "-", "scan-b", "C&C"]
PHASES = ("raw", "history", "service", "contact", "contact_add")


def _sealed(root):
    truth = data.truth(root, "capture")
    truth.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"row_id": np.arange(1, len(FAMILIES) + 1), "family": FAMILIES, "native_detail": DETAILS}),
        truth,
        row_group_size=2,
    )
    directory = root / "evidence/blind/capture"
    directory.mkdir(parents=True)
    masks = {}
    for index, group in enumerate(("hgb", "hgb_family")):
        flags = np.full(len(FAMILIES), 128, dtype=np.uint8)
        masks[group] = {}
        for bit, policy in enumerate(PHASES):
            selected = (np.arange(len(FAMILIES)) + index + bit) % 3 == 1
            masks[group][policy] = selected
            flags |= selected.astype(np.uint8) << POLICIES.index(policy)
        pq.write_table(pa.table({"flags": flags}), directory / (group + ".parquet"), row_group_size=1 + index)
        write_json(root / "evidence/calibration" / (group + ".json"), {"contact_add": {"threshold": 0.25}})
    return masks


def test_equal_policy_cross_training_allocation_compares_each_physical_row(tmp_path, monkeypatch):
    masks = _sealed(tmp_path)
    monkeypatch.setattr(evaluation, "ROWS", 3)
    (output,) = evaluation.judge(tmp_path, "capture", ("hgb", "hgb_family"))
    report = read_json(output)
    result = report["training_allocation"]
    assert report["rows"] == len(FAMILIES)
    assert (result["baseline"], result["candidate"]) == ("hgb", "hgb_family")
    assert set(result["policies"]) == set(PHASES)
    for policy, pair in result["policies"].items():
        before, after = masks["hgb"][policy], masks["hgb_family"][policy]
        for prefix, values in (("native/", DETAILS), ("family/", FAMILIES)):
            totals = 0
            for key in set(values):
                which = np.array(values) == key
                n = int(which.sum())
                totals += pair["counts"][prefix + key]["rows"]
                assert pair["counts"][prefix + key] == {
                    "rows": n,
                    "before": int((which & before).sum()),
                    "after": int((which & after).sum()),
                    "gained": int((which & ~before & after).sum()),
                    "lost": int((which & before & ~after).sum()),
                }
            assert totals == len(values)


def test_allocation_requires_both_sealed_groups_and_preserves_all_rows(tmp_path):
    _sealed(tmp_path)
    (path,) = evaluation.judge(tmp_path, "capture", groups=("hgb",))
    assert "training_allocation" not in read_json(path)
    pq.write_table(
        pa.table({"flags": np.zeros(7, np.uint8)}), tmp_path / "evidence/blind/capture/hgb_family.parquet"
    )
    with pytest.raises(ValueError, match="Incomplete capture"):
        evaluation.judge(tmp_path, "capture", groups=("hgb", "hgb_family"))
